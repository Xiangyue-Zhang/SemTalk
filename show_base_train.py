#!/usr/bin/env python3
"""Strict train-only entry point for the SHOW SemTalk Base reproduction.

This deliberately bypasses SemTalk's stock test-during-training loop.  It keeps
the model, trainer, optimizer, scheduler, and epoch update semantics intact,
while adding resumable checkpoints and finite-value audits.  Launch it with
``torchrun`` even for a one-GPU run so rank handling is unambiguous.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from numbers import Real
import os
from pathlib import Path
import random
import stat
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

# Formal source receipts require the checkout to stay byte-for-byte clean.
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from utils import config, logger_tools, other_tools
from utils.show_official_transfer import load_official_model_state
from utils.lower_target_cache import (
    RECEIPT_KEY as LOWER_TARGET_CACHE_RECEIPT_KEY,
    attach_lower_target_cache_receipt,
    canonical_json_sha256 as lower_target_cache_receipt_sha256,
    validate_activation_args as validate_lower_target_cache_activation,
    verify_lower_target_cache_resume_receipt,
)
from utils.rvq_distributed import (
    REPRESENTATION_STAGES,
    RVQ_STAGES,
    assert_rvq_rank_state,
    initialize_loaded_rvq_ema,
    representation_ddp_receipt,
    validate_representation_ddp_receipt,
)
from utils.smplx_training import (
    clip_aligned_spans,
    parse_smplx_helper_devices,
)


FORMAL_SMPLX_FILENAME = "SMPLX_NEUTRAL_2020.npz"
FORMAL_SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
FORMAL_SMPLX_STAGES = frozenset({"face", "hands", "upper", "lower"})
SMPLX_TRAINING_POOL_GATE_FORMAT = (
    "semtalk_show_smplx_sharded_local_loss_formal_gate_v1"
)
SMPLX_TRAINING_POOL_TARGET_OFFLOAD_GATE_FORMAT = (
    "semtalk_show_smplx_target_offload_formal_gate_v1"
)
SMPLX_TRAINING_POOL_GATE_HARNESS_SHA256 = (
    "01c77aa005039f9936f667b8b97052084be3e3364f64be402b53df383ed7ae2d"
)
SMPLX_TRAINING_POOL_GATE_ADAPTER_SHA256 = (
    "5b3ac7fdeb4ca680036998159f64db7369430d4cc0e5852a49436d97f04aeeba"
)
SMPLX_TRAINING_POOL_TARGET_OFFLOAD_GATE_HARNESS_SHA256 = (
    "5423ed9a5a8076e23e72ffaad3104f94c81ea5b2cc7d48aec32d50c10e6772de"
)
SMPLX_TRAINING_POOL_TARGET_OFFLOAD_GATE_ADAPTER_SHA256 = (
    "2fd00b05eec9f68267ab74d36d9a074a2692465f55e422ed2c6477c25821a269"
)
SMPLX_TRAINING_POOL_IMPLEMENTATION_FILES = frozenset(
    {
        "show_base_train.py",
        "utils/smplx_training.py",
        "utils/config.py",
        "aeface_trainer.py",
        "ae_trainer.py",
        "aelower_trainer.py",
    }
)
SMPLX_TRAINING_POOL_GATE_MIN_SPEEDUP = 1.50
SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES = {
    "model": {"atol": 5e-6, "rtol": 2e-5},
    "grads": {"atol": 2e-6, "rtol": 2e-5},
    "optimizer": {"atol": 5e-6, "rtol": 2e-5},
    "rvq_ema": {"atol": 5e-6, "rtol": 2e-5},
    "tracker": {"atol": 5e-6, "rtol": 2e-5},
}
SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY = "smplx_training_pool_gate"
SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY = (
    "smplx_training_pool_runtime_evidence"
)
SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_FORMAT = (
    "semtalk_show_smplx_sharded_local_loss_runtime_evidence_v1"
)
SMPLX_TRAINING_POOL_TARGET_OFFLOAD_RUNTIME_EVIDENCE_FORMAT = (
    "semtalk_show_smplx_target_offload_runtime_evidence_v1"
)
SMPLX_TRAINING_POOL_MODE_SPECS = {
    "target_offload": {
        "gate_format": SMPLX_TRAINING_POOL_TARGET_OFFLOAD_GATE_FORMAT,
        "harness_sha256": (
            SMPLX_TRAINING_POOL_TARGET_OFFLOAD_GATE_HARNESS_SHA256
        ),
        "adapter_sha256": (
            SMPLX_TRAINING_POOL_TARGET_OFFLOAD_GATE_ADAPTER_SHA256
        ),
        "runtime_evidence_format": (
            SMPLX_TRAINING_POOL_TARGET_OFFLOAD_RUNTIME_EVIDENCE_FORMAT
        ),
        "partition": (
            "primary_stock_full_batch_reconstruction_and_"
            "helper_full_batch_target"
        ),
        "transfer_to_primary": "detached_target_full_outputs_only",
        "minimum_speedup": 1.03,
        "cross_mode_comparison": "cross_mode_byte_exact",
        "cross_mode_final": "byte_exact",
        "receipt_cross_mode": "byte_exact",
    },
    "sharded_local_loss": {
        "gate_format": SMPLX_TRAINING_POOL_GATE_FORMAT,
        "harness_sha256": SMPLX_TRAINING_POOL_GATE_HARNESS_SHA256,
        "adapter_sha256": SMPLX_TRAINING_POOL_GATE_ADAPTER_SHA256,
        "runtime_evidence_format": (
            SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_FORMAT
        ),
        "partition": "balanced_contiguous_whole_clips",
        "transfer_to_primary": (
            "differentiable_scalar_numerators_only"
        ),
        "minimum_speedup": SMPLX_TRAINING_POOL_GATE_MIN_SPEEDUP,
        "cross_mode_comparison": "cross_mode_bounded",
        "cross_mode_final": (
            "bounded_float_exact_discrete_rng_and_smplx"
        ),
        "receipt_cross_mode": (
            "bounded_float_exact_discrete_rng_and_smplx"
        ),
    },
}
LOWER_TARGET_CACHE_GATE_FORMAT = (
    "semtalk_show_lower_target_cache_formal_gate_v1"
)
LOWER_TARGET_CACHE_GATE_MIN_SPEEDUP = 1.05
LOWER_TARGET_BACKEND_RECEIPT_KEY = "lower_target_backend"
LOWER_TARGET_BACKEND_FORMAT = "semtalk_show_lower_target_backend_v1"
LOWER_TARGET_BACKEND_RECEIPT_KEYS = {
    "format",
    "backend",
    "formal_stage",
    "cache_enabled",
    "target_forward",
    "target_forward_sha256",
    "target_batch_contract",
    "torch_no_grad",
    "return_shaped",
    "source_binding",
    "smplx_asset_sha256",
    "receipt_sha256",
}
BASE_CANDIDATE_INTERVAL_EPOCHS = 10
REPRESENTATION_CANDIDATE_INTERVAL_EPOCHS = 20
BASE_CANDIDATE_TRANSACTION_FILENAME = "base_candidate_transaction.json"
BASE_CANDIDATE_STAGING_FILENAME = ".base_candidate_checkpoint.staging"


def _require_exact_audit_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(
            f"{label} must be an exact integer, got {value!r}"
        )
    return value


def _require_exact_audit_int_mapping(
    value: Any,
    expected: dict[str, int],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimeError(f"{label} must have exactly {sorted(expected)}")
    for key, expected_value in expected.items():
        if (
            _require_exact_audit_int(value[key], f"{label}.{key}")
            != expected_value
        ):
            raise RuntimeError(
                f"{label}.{key} must equal {expected_value}"
            )
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_torch_save(
    path: Path,
    payload: dict[str, Any],
    *,
    staging_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = (
        staging_path
        if staging_path is not None
        else path.with_name(f".{path.name}.tmp.{os.getpid()}")
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.is_symlink() or (tmp.exists() and not tmp.is_file()):
        raise RuntimeError(
            f"atomic torch staging path must be a regular non-symlink file: {tmp}"
        )
    torch.save(payload, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    for directory in {tmp.parent, path.parent}:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load_candidate_snapshot(
    snapshot: bytes,
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Deserialize exactly the immutable bytes that were hashed."""
    try:
        return torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
        )


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _smplx_training_pool_mode_spec(mode: Any) -> dict[str, Any]:
    spec = SMPLX_TRAINING_POOL_MODE_SPECS.get(mode)
    if not isinstance(spec, dict):
        raise RuntimeError(
            "formal SMPL-X pooling mode must be target_offload or "
            "sharded_local_loss"
        )
    return spec


def _validate_smplx_training_pool_helpers(
    mode: str,
    helpers: tuple[int, ...],
) -> None:
    if mode == "target_offload":
        if helpers != (1,):
            raise RuntimeError(
                "target_offload requires exactly logical helper 1"
            )
        return
    if (
        mode != "sharded_local_loss"
        or len(helpers) < 2
        or helpers != tuple(range(1, len(helpers) + 1))
    ):
        raise RuntimeError(
            "sharded_local_loss requires at least two contiguous logical "
            "helpers starting at 1"
        )


def _smplx_training_pool_topology(
    mode: str,
    helpers: tuple[int, ...],
) -> dict[str, Any]:
    spec = _smplx_training_pool_mode_spec(mode)
    _validate_smplx_training_pool_helpers(mode, helpers)
    visible_device_count = len(helpers) + 1
    return {
        "format": "semtalk_smplx_training_pool_topology_v1",
        "pool_runtime_format": "semtalk_smplx_training_pool_v2",
        "mode": mode,
        "primary_device": 0,
        "helper_devices": list(helpers),
        "replica_devices": list(range(visible_device_count)),
        "visible_device_count": visible_device_count,
        "partition": spec["partition"],
        "transfer_to_primary": spec["transfer_to_primary"],
    }


def _validate_smplx_training_pool_last_forward(
    last_forward: Any,
    *,
    formal_stage: str,
    topology: dict[str, Any],
) -> None:
    mode = topology.get("mode")
    if not isinstance(last_forward, dict):
        raise RuntimeError("SMPL-X pool last-forward evidence is invalid")
    if mode == "target_offload":
        required_keys = {
            "stage",
            "total_rows",
            "clip_length",
            "total_clips",
            "helpers",
            "reconstruction_device",
            "reconstruction_execution",
            "target_device",
            "target_execution",
            "transfer_to_primary",
            "output_keys",
        }
        expected_output_keys = (
            ["joints"] if formal_stage == "lower" else ["vertices"]
        )
        if (
            set(last_forward) != required_keys
            or last_forward.get("stage") != formal_stage
            or last_forward.get("helpers") != [1]
            or topology.get("helper_devices") != [1]
            or last_forward.get("reconstruction_device") != 0
            or last_forward.get("reconstruction_execution")
            != "stock_full_batch_primary_current_stream"
            or last_forward.get("target_device") != 1
            or last_forward.get("target_execution")
            != "detached_full_batch_helper"
            or last_forward.get("transfer_to_primary")
            != "detached_target_full_outputs_only"
            or last_forward.get("output_keys") != expected_output_keys
        ):
            raise RuntimeError(
                "SMPL-X target_offload last-forward path is invalid"
            )
        total_rows = _require_exact_audit_int(
            last_forward.get("total_rows"),
            "SMPL-X pool last_forward total_rows",
        )
        clip_length = _require_exact_audit_int(
            last_forward.get("clip_length"),
            "SMPL-X pool last_forward clip_length",
        )
        total_clips = _require_exact_audit_int(
            last_forward.get("total_clips"),
            "SMPL-X pool last_forward total_clips",
        )
        if (
            total_rows <= 0
            or clip_length <= 0
            or total_rows % clip_length
            or total_clips != total_rows // clip_length
        ):
            raise RuntimeError(
                "SMPL-X target_offload last-forward dimensions are invalid"
            )
        return

    if mode != "sharded_local_loss":
        raise RuntimeError("unsupported SMPL-X pool runtime evidence mode")
    required_keys = {
        "stage",
        "total_rows",
        "clip_length",
        "total_clips",
        "helpers",
        "spans",
        "component_counts",
        "aggregation",
    }
    if (
        set(last_forward) != required_keys
        or last_forward.get("stage") != formal_stage
        or last_forward.get("helpers") != topology.get("helper_devices")
        or last_forward.get("aggregation")
        != "ordered_numerator_sum_over_exact_global_count"
    ):
        raise RuntimeError("SMPL-X pool last-forward evidence is invalid")
    total_rows = _require_exact_audit_int(
        last_forward.get("total_rows"),
        "SMPL-X pool last_forward total_rows",
    )
    clip_length = _require_exact_audit_int(
        last_forward.get("clip_length"),
        "SMPL-X pool last_forward clip_length",
    )
    total_clips = _require_exact_audit_int(
        last_forward.get("total_clips"),
        "SMPL-X pool last_forward total_clips",
    )
    helpers = topology.get("helper_devices")
    if (
        total_rows <= 0
        or clip_length <= 0
        or total_rows % clip_length
        or total_clips != total_rows // clip_length
        or not isinstance(helpers, list)
    ):
        raise RuntimeError("SMPL-X pool last-forward dimensions are invalid")
    expected_spans = [
        {
            "device": device,
            "row_start": row_start,
            "row_end": row_end,
            "clip_start": clip_start,
            "clip_end": clip_end,
        }
        for device, (
            row_start,
            row_end,
            clip_start,
            clip_end,
        ) in zip(
            helpers,
            clip_aligned_spans(
                total_rows,
                clip_length,
                len(helpers),
            ),
        )
    ]
    if last_forward.get("spans") != expected_spans:
        raise RuntimeError("SMPL-X pool last-forward spans are not exact")

    component_counts = last_forward.get("component_counts")
    expected_components = (
        {"ver", "foot"}
        if formal_stage == "lower"
        else {"ver", "ver_vel", "ver_acc"}
    )
    if (
        not isinstance(component_counts, dict)
        or set(component_counts) != expected_components
        or any(
            not isinstance(counts, list)
            or len(counts) != len(helpers)
            or any(type(count) is not int or count <= 0 for count in counts)
            for counts in component_counts.values()
        )
    ):
        raise RuntimeError(
            "SMPL-X pool last-forward component counts are invalid"
        )
    for shard_index, span in enumerate(expected_spans):
        local_rows = span["row_end"] - span["row_start"]
        ver_count = component_counts["ver"][shard_index]
        if (
            local_rows <= 0
            or ver_count % local_rows
            or (ver_count // local_rows) % 3
        ):
            raise RuntimeError(
                "SMPL-X pool vertex/joint count is not row aligned"
            )
        per_row = ver_count // local_rows
        if formal_stage == "lower":
            if component_counts["foot"][shard_index] != local_rows * 12:
                raise RuntimeError(
                    "SMPL-X pool lower foot count is not exact"
                )
        elif (
            per_row <= 6
            or component_counts["ver_vel"][shard_index]
            != local_rows * (per_row - 3)
            or component_counts["ver_acc"][shard_index]
            != local_rows * (per_row - 6)
        ):
            raise RuntimeError(
                "SMPL-X pool vertex derivative counts are not exact"
            )


def _json_document_sha256(payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _formal_smplx_asset_receipt(args: Any) -> dict[str, Any] | None:
    if args.formal_stage not in FORMAL_SMPLX_STAGES:
        if args.expected_smplx_asset_sha256:
            raise RuntimeError(
                "--expected_smplx_asset_sha256 is restricted to the four "
                "SMPL-X-backed RVQ stages"
            )
        return None

    expected = str(args.expected_smplx_asset_sha256 or "").strip().lower()
    if expected != FORMAL_SMPLX_SHA256:
        raise RuntimeError(
            "formal face/hands/upper/lower training requires the frozen "
            f"SMPL-X SHA-256 {FORMAL_SMPLX_SHA256}"
        )
    asset_input = (
        Path(args.data_path_1)
        / "smplx_models"
        / "smplx"
        / FORMAL_SMPLX_FILENAME
    )
    try:
        asset_stat = asset_input.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(asset_input) from None
    if stat.S_ISLNK(asset_stat.st_mode) or not stat.S_ISREG(asset_stat.st_mode):
        raise RuntimeError(
            f"formal SMPL-X asset must be a regular non-symlink file: {asset_input}"
        )
    if asset_input.name != FORMAL_SMPLX_FILENAME:
        raise RuntimeError("formal SMPL-X asset basename mismatch")
    asset_path = asset_input.resolve(strict=True)
    observed = _sha256(asset_path)
    if observed != expected:
        raise RuntimeError(
            f"formal SMPL-X asset SHA mismatch: {observed} != {expected}"
        )
    return {
        "format": "semtalk_show_smplx_asset_v1",
        "filename": FORMAL_SMPLX_FILENAME,
        "path": str(asset_path),
        "sha256": observed,
        "bytes": int(asset_stat.st_size),
        "regular_file": True,
        "symlink": False,
    }


def _source_receipt() -> dict[str, str]:
    root = Path(__file__).resolve().parent

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "formal training source must be a clean checkout; first change: "
            f"{status.splitlines()[0]}"
        )
    origin = git("remote", "get-url", "origin")
    expected_origin = "git@github.com:Xiangyue-Zhang/SemTalk.git"
    if origin != expected_origin:
        raise RuntimeError(
            f"formal training origin {origin!r} != {expected_origin!r}"
        )
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "origin": origin,
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": _sha256(Path(__file__).resolve()),
    }


def _validate_lower_target_backend_receipt(
    receipt: dict[str, Any] | None,
    *,
    formal_stage: str,
    current_source: dict[str, str],
    smplx_asset_receipt: dict[str, Any] | None,
) -> None:
    if formal_stage != "lower":
        if receipt is not None:
            raise RuntimeError(
                "lower target backend receipt is forbidden outside lower"
            )
        return
    if type(receipt) is not dict:
        raise RuntimeError("formal lower requires one live SMPL-X backend receipt")
    receipt_without_sha = dict(receipt)
    receipt_sha = receipt_without_sha.pop("receipt_sha256", None)
    source_binding = receipt.get("source_binding")
    target_forward_source = (
        Path(__file__).resolve().parent / "utils" / "smplx_training.py"
    )
    target_forward_sha = (
        _sha256(target_forward_source)
        if (
            not target_forward_source.is_symlink()
            and target_forward_source.is_file()
        )
        else None
    )
    if (
        set(receipt) != LOWER_TARGET_BACKEND_RECEIPT_KEYS
        or receipt.get("format") != LOWER_TARGET_BACKEND_FORMAT
        or receipt.get("backend") != "live_smplx"
        or receipt.get("formal_stage") != "lower"
        or receipt.get("cache_enabled") is not False
        or receipt.get("target_forward")
        != "utils.smplx_training.smplx_target_forward"
        or type(receipt.get("target_forward_sha256")) is not str
        or len(receipt["target_forward_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt["target_forward_sha256"]
        )
        or receipt.get("target_forward_sha256") != target_forward_sha
        or receipt.get("target_batch_contract")
        != "current_mixed_dataloader_batch"
        or receipt.get("torch_no_grad") is not True
        or receipt.get("return_shaped") is not False
        or type(source_binding) is not dict
        or set(source_binding) != {"origin", "commit", "tree"}
        or source_binding
        != {
            key: current_source[key]
            for key in ("origin", "commit", "tree")
        }
        or type(smplx_asset_receipt) is not dict
        or smplx_asset_receipt.get("format")
        != "semtalk_show_smplx_asset_v1"
        or smplx_asset_receipt.get("filename") != FORMAL_SMPLX_FILENAME
        or smplx_asset_receipt.get("sha256") != FORMAL_SMPLX_SHA256
        or receipt.get("smplx_asset_sha256")
        != FORMAL_SMPLX_SHA256
        or type(receipt_sha) is not str
        or receipt_sha != _payload_sha256(receipt_without_sha)
    ):
        raise RuntimeError("invalid formal lower live SMPL-X backend receipt")


def _formal_lower_target_backend_receipt(
    args: Any,
    *,
    current_source: dict[str, str],
    smplx_asset_receipt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    cache_enabled = validate_lower_target_cache_activation(args)
    if args.formal_stage != "lower":
        return None
    if cache_enabled:
        raise RuntimeError(
            "formal lower uses live SMPL-X targets; cache activation is forbidden"
        )
    target_forward_input = (
        Path(__file__).resolve().parent / "utils" / "smplx_training.py"
    )
    if (
        target_forward_input.is_symlink()
        or not target_forward_input.is_file()
    ):
        raise RuntimeError(
            "live lower target implementation must be a regular non-symlink "
            f"file: {target_forward_input}"
        )
    receipt: dict[str, Any] = {
        "format": LOWER_TARGET_BACKEND_FORMAT,
        "backend": "live_smplx",
        "formal_stage": "lower",
        "cache_enabled": False,
        "target_forward": "utils.smplx_training.smplx_target_forward",
        "target_forward_sha256": _sha256(target_forward_input),
        "target_batch_contract": "current_mixed_dataloader_batch",
        "torch_no_grad": True,
        "return_shaped": False,
        "source_binding": {
            key: current_source[key]
            for key in ("origin", "commit", "tree")
        },
        "smplx_asset_sha256": (
            smplx_asset_receipt.get("sha256")
            if isinstance(smplx_asset_receipt, dict)
            else None
        ),
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    _validate_lower_target_backend_receipt(
        receipt,
        formal_stage=args.formal_stage,
        current_source=current_source,
        smplx_asset_receipt=smplx_asset_receipt,
    )
    return receipt


def _attach_lower_target_backend_receipt(
    payload: dict[str, Any],
    receipt: dict[str, Any] | None,
) -> None:
    if LOWER_TARGET_BACKEND_RECEIPT_KEY in payload:
        raise RuntimeError("duplicate lower target backend receipt")
    if receipt is not None:
        payload[LOWER_TARGET_BACKEND_RECEIPT_KEY] = copy.deepcopy(receipt)


def _lower_target_backend_overlay(
    receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    if receipt is None:
        return {}
    return {LOWER_TARGET_BACKEND_RECEIPT_KEY: copy.deepcopy(receipt)}


def _verify_lower_target_backend_resume_receipt(
    payload: dict[str, Any],
    expected_receipt: dict[str, Any] | None,
) -> None:
    observed_present = LOWER_TARGET_BACKEND_RECEIPT_KEY in payload
    expected_present = expected_receipt is not None
    if (
        observed_present != expected_present
        or (
            expected_present
            and payload.get(LOWER_TARGET_BACKEND_RECEIPT_KEY)
            != expected_receipt
        )
    ):
        raise RuntimeError("resume lower target backend receipt does not match")


def _config_fingerprint(args: Any) -> str:
    snapshot = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume_state", "local_rank"}
    }
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_lowercase_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            f"{label} must be exactly 64 lowercase hexadecimal digits"
        )
    return value


def _validate_target_offload_preliminary_numerical_gate(
    receipt: Any,
    *,
    smplx_asset_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reopen and revalidate the target gate's preliminary numeric proof."""
    required_receipt_keys = {
        "classification",
        "authorization",
        "files",
        "format",
        "batch",
        "clips",
        "frames",
        "vertices_loss_six_input_gradients_byte_exact",
        "runner_status",
    }
    file_labels = {
        "numerical report",
        "numerical script",
        "numerical runner status",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != required_receipt_keys
        or receipt.get("classification")
        != "preliminary_not_source_bound"
        or receipt.get("authorization") is not False
        or not isinstance(receipt.get("files"), dict)
        or set(receipt["files"]) != file_labels
        or not isinstance(smplx_asset_receipt, dict)
        or smplx_asset_receipt.get("sha256") != FORMAL_SMPLX_SHA256
    ):
        raise RuntimeError(
            "target_offload preliminary numerical receipt is invalid"
        )

    files: dict[str, dict[str, str]] = {}
    for label in sorted(file_labels):
        record = receipt["files"].get(label)
        path_value = record.get("path") if isinstance(record, dict) else None
        expected_sha = (
            record.get("sha256") if isinstance(record, dict) else None
        )
        path = Path(path_value) if type(path_value) is str else Path()
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or not path.is_absolute()
            or path != path.resolve()
            or path.is_symlink()
            or not path.is_file()
            or type(expected_sha) is not str
            or _sha256(path)
            != _require_lowercase_sha256(
                expected_sha,
                f"target_offload {label} SHA",
            )
        ):
            raise RuntimeError(
                f"target_offload preliminary {label} differs"
            )
        files[label] = {
            "path": str(path),
            "sha256": expected_sha,
        }

    with Path(files["numerical report"]["path"]).open(
        encoding="utf-8"
    ) as handle:
        numerical = json.load(handle)
    comparisons = (
        numerical.get("comparisons")
        if isinstance(numerical, dict)
        else None
    )
    gradient_names = {
        "body_pose",
        "expression",
        "global_orient",
        "jaw_pose",
        "left_hand_pose",
        "right_hand_pose",
    }

    def exact_numeric(record: Any, label: str) -> None:
        if (
            not isinstance(record, dict)
            or record.get("dtype") != "torch.float32"
            or record.get("equal") is not True
            or record.get("max_abs") != 0.0
            or record.get("mismatch_count") != 0
        ):
            raise RuntimeError(
                "target_offload preliminary numerical result differs: "
                f"{label}"
            )

    if (
        not isinstance(numerical, dict)
        or numerical.get("format")
        != "semtalk_smplx_full_batch_grad_quick_gate_v1"
        or numerical.get("status") != "pass"
        or numerical.get("asset_sha256") != FORMAL_SMPLX_SHA256
        or numerical.get("batch") != 64 * 64
        or numerical.get("clips") != 64
        or numerical.get("frames") != 64
        or numerical.get("device_names") != ["NVIDIA H200"] * 8
        or not isinstance(comparisons, dict)
        or not isinstance(comparisons.get("input_gradients"), dict)
        or set(comparisons["input_gradients"]) != gradient_names
    ):
        raise RuntimeError(
            "target_offload preliminary numerical schema differs"
        )
    exact_numeric(comparisons.get("vertices"), "vertices")
    exact_numeric(comparisons.get("loss"), "loss")
    for name in sorted(gradient_names):
        exact_numeric(comparisons["input_gradients"][name], name)

    asset_value = smplx_asset_receipt.get("path")
    asset_path = Path(asset_value) if type(asset_value) is str else Path()
    if (
        not asset_path.is_absolute()
        or asset_path != asset_path.resolve()
        or asset_path.is_symlink()
        or not asset_path.is_file()
        or asset_path.name != FORMAL_SMPLX_FILENAME
        or _sha256(asset_path) != FORMAL_SMPLX_SHA256
        or len(asset_path.parents) < 3
    ):
        raise RuntimeError(
            "target_offload preliminary SMPL-X asset binding differs"
        )
    asset_root = asset_path.parents[2]
    if (
        asset_path
        != asset_root
        / "smplx_models"
        / "smplx"
        / FORMAL_SMPLX_FILENAME
    ):
        raise RuntimeError(
            "target_offload preliminary SMPL-X asset layout differs"
        )
    with Path(files["numerical runner status"]["path"]).open(
        encoding="utf-8"
    ) as handle:
        runner_status = json.load(handle)
    expected_command = [
        "/usr/bin/python3.12",
        files["numerical script"]["path"],
        "--asset-root",
        str(asset_root),
        "--report",
        files["numerical report"]["path"],
        "--clips",
        "64",
        "--frames",
        "64",
    ]
    restored = (
        runner_status.get("restored_guards")
        if isinstance(runner_status, dict)
        else None
    )
    if (
        not isinstance(runner_status, dict)
        or runner_status.get("state") != "finished"
        or runner_status.get("return_code") != 0
        or runner_status.get("error") is not None
        or runner_status.get("cleanup_error") is not None
        or runner_status.get("restore_error") is not None
        or runner_status.get("received_signal") is not None
        or runner_status.get("command") != expected_command
        or type(runner_status.get("wrapper_pid")) is not int
        or runner_status["wrapper_pid"] <= 1
        or type(runner_status.get("child_pid")) is not int
        or runner_status["child_pid"] <= 1
        or not isinstance(restored, dict)
        or set(restored) != {str(index) for index in range(8)}
        or len(set(restored.values())) != 8
        or any(
            type(pid) is not int or pid <= 1
            for pid in restored.values()
        )
    ):
        raise RuntimeError(
            "target_offload preliminary guarded-runner receipt differs"
        )
    derived_receipt = {
        "classification": "preliminary_not_source_bound",
        "authorization": False,
        "files": files,
        "format": numerical["format"],
        "batch": numerical["batch"],
        "clips": numerical["clips"],
        "frames": numerical["frames"],
        "vertices_loss_six_input_gradients_byte_exact": True,
        "runner_status": {
            "wrapper_pid": runner_status["wrapper_pid"],
            "child_pid": runner_status["child_pid"],
            "return_code": runner_status["return_code"],
            "restored_guards": restored,
        },
    }
    if receipt != derived_receipt:
        raise RuntimeError(
            "target_offload preliminary receipt does not match its files"
        )
    return {
        "asset_root": str(asset_root),
        "files": files,
        "receipt": copy.deepcopy(derived_receipt),
    }


def _validate_smplx_training_pool_device_matrix(
    args: Any,
    *,
    world_size: int,
    cuda_available: bool,
    visible_device_count: int,
) -> None:
    """Fail closed before any model or process-group initialization."""

    if type(world_size) is not int:
        raise RuntimeError("world_size must be an exact integer")
    if not cuda_available:
        raise RuntimeError("formal SHOW training requires CUDA")
    if type(visible_device_count) is not int:
        raise RuntimeError("visible CUDA device count must be an exact integer")

    stage = getattr(args, "formal_stage", None)
    mode = getattr(args, "smplx_training_pool_mode", None)
    try:
        helpers = parse_smplx_helper_devices(
            getattr(args, "smplx_training_helper_devices", "")
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "invalid --smplx_training_helper_devices"
        ) from error
    gate_values = (
        getattr(args, "smplx_training_pool_gate_report", None),
        getattr(
            args,
            "expected_smplx_training_pool_gate_sha256",
            None,
        ),
    )
    gate_supplied = any(value not in {None, ""} for value in gate_values)

    if stage in REPRESENTATION_STAGES:
        expected_world_size = 1 if stage == "global" else None
        if stage in RVQ_STAGES and world_size not in {2, 4}:
            raise RuntimeError(
                "formal SHOW RVQ training requires world_size 2 or 4"
            )
        if expected_world_size is not None and world_size != expected_world_size:
            raise RuntimeError("formal Global training requires world_size 1")
        if visible_device_count != world_size:
            raise RuntimeError(
                "formal representation training requires one visible CUDA "
                "device per local rank"
            )
        if mode != "disabled" or helpers or gate_supplied:
            raise RuntimeError(
                "DDP representation training forbids detached SMPL-X pools"
            )
        return

    if stage in FORMAL_SMPLX_STAGES:
        if mode == "disabled":
            if helpers:
                raise RuntimeError(
                    "formal stock SMPL-X training forbids helper devices"
                )
            if gate_supplied:
                raise RuntimeError(
                    "formal stock SMPL-X training forbids pool gate arguments"
                )
            if visible_device_count != 1:
                raise RuntimeError(
                    "formal stock SMPL-X training requires exactly one "
                    "visible CUDA device"
                )
            return
        _smplx_training_pool_mode_spec(mode)
        _validate_smplx_training_pool_helpers(mode, helpers)
        if visible_device_count != len(helpers) + 1:
            raise RuntimeError(
                f"{mode} requires exactly primary+helpers visible"
            )
        if not all(type(value) is str and bool(value) for value in gate_values):
            raise RuntimeError(
                f"{mode} requires an explicit topology-specific "
                "formal gate report and expected SHA-256"
            )
        return

    if stage != "base":
        raise RuntimeError(
            "disabled SMPL-X pooling is restricted to formal global/Base "
            "training"
        )
    if mode != "disabled":
        raise RuntimeError(
            "SMPL-X pooling is restricted to face/hands/upper/lower "
            "target_offload or sharded_local_loss training"
        )
    if helpers:
        raise RuntimeError("disabled SMPL-X pooling forbids helper devices")
    if gate_supplied:
        raise RuntimeError(
            "disabled or non-VQ training forbids SMPL-X pool gate arguments"
        )
    if world_size != 1 or visible_device_count != 1:
        raise RuntimeError(
            "disabled and global/Base formal tasks require exactly one "
            "visible CUDA device"
        )


def _formal_module_cuda_devices(module: Any) -> set[int]:
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError("SMPL-X pool replica must be a torch module")
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        raise RuntimeError("SMPL-X pool replica has no auditable tensors")
    if any(
        tensor.device.type != "cuda" or tensor.device.index is None
        for tensor in tensors
    ):
        raise RuntimeError("SMPL-X pool replica contains non-CUDA tensors")
    return {int(tensor.device.index) for tensor in tensors}


def _validate_smplx_training_pool_trainer_runtime(
    args: Any,
    trainer: Any,
    *,
    gate_receipt: dict[str, Any] | None,
) -> None:
    """Prove that the trainer instantiated the gate-authorized pool."""

    requested_mode = getattr(
        args,
        "smplx_training_pool_mode",
        None,
    )
    trainer_mode = getattr(
        trainer,
        "smplx_parallel_mode",
        "disabled",
    )
    trainer_pool = getattr(trainer, "smplx_parallel_pool", None)
    pool_alias = getattr(trainer, "smplx_pool", None)
    stage = getattr(args, "formal_stage", None)
    if requested_mode == "disabled":
        if stage not in {*FORMAL_SMPLX_STAGES, "global", "base"}:
            raise RuntimeError("invalid formal stock SMPL-X stage")
        if (
            gate_receipt is not None
            or trainer_mode != "disabled"
            or trainer_pool is not None
            or pool_alias is not None
        ):
            raise RuntimeError(
                "disabled SMPL-X pooling produced a trainer pool or receipt"
            )
        return

    requested_helpers = parse_smplx_helper_devices(
        getattr(args, "smplx_training_helper_devices", "")
    )
    _smplx_training_pool_mode_spec(requested_mode)
    _validate_smplx_training_pool_helpers(
        requested_mode,
        requested_helpers,
    )
    visible_device_count = len(requested_helpers) + 1
    expected_devices = {0, *requested_helpers}
    if (
        stage not in FORMAL_SMPLX_STAGES
        or not isinstance(gate_receipt, dict)
        or gate_receipt.get("formal_stage") != stage
        or gate_receipt.get("mode") != requested_mode
        or gate_receipt.get("helper_devices")
        != list(requested_helpers)
        or trainer_mode != requested_mode
        or trainer_pool is None
        or trainer_pool is not pool_alias
    ):
        raise RuntimeError(
            "trainer SMPL-X pool does not match the authorized gate"
        )
    pool = trainer_pool
    replicas = getattr(pool, "replicas", None)
    models = getattr(pool, "models", None)
    streams = getattr(pool, "streams", None)
    gate_modules_method = getattr(pool, "gate_replica_modules", None)
    runtime_receipt_method = getattr(pool, "runtime_receipt", None)
    if (
        getattr(pool, "mode", None) != requested_mode
        or type(getattr(pool, "primary_device", None)) is not int
        or pool.primary_device != 0
        or getattr(pool, "helper_devices", None) != requested_helpers
        or not isinstance(replicas, dict)
        or set(replicas) != expected_devices
        or models is not replicas
        or not isinstance(streams, dict)
        or set(streams) != expected_devices
        or any(stream is None for stream in streams.values())
        or not callable(gate_modules_method)
        or not callable(runtime_receipt_method)
    ):
        raise RuntimeError("trainer SMPL-X pool topology is invalid")
    helper_modules = gate_modules_method()
    if (
        type(helper_modules) is not tuple
        or len(helper_modules) != len(requested_helpers)
        or any(
            module is not replicas[device]
            for module, device in zip(helper_modules, requested_helpers)
        )
        or getattr(trainer, "smplx", None) is not replicas[0]
        or len({id(module) for module in replicas.values()})
        != visible_device_count
    ):
        raise RuntimeError(
            "trainer SMPL-X helper replica exposure is inconsistent"
        )

    replica_parameter_ids: set[int] = set()
    for device, module in replicas.items():
        if _formal_module_cuda_devices(module) != {device}:
            raise RuntimeError(
                f"trainer SMPL-X replica is not isolated on logical{device}"
            )
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("trainer SMPL-X replicas must remain frozen")
        current_parameter_ids = {
            id(parameter) for parameter in module.parameters()
        }
        if current_parameter_ids & replica_parameter_ids:
            raise RuntimeError("trainer SMPL-X replicas share parameters")
        replica_parameter_ids.update(current_parameter_ids)

    training_parameter_ids = {
        id(parameter) for parameter in trainer.model.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.opt.param_groups
        for parameter in group["params"]
    }
    if (
        replica_parameter_ids & training_parameter_ids
        or replica_parameter_ids & optimizer_parameter_ids
    ):
        raise RuntimeError(
            "trainer SMPL-X replica leaked into the trainable model/optimizer"
        )

    runtime = gate_receipt.get("runtime")
    actual_names = [
        torch.cuda.get_device_name(device)
        for device in range(visible_device_count)
    ]
    actual_runtime = {
        "visible_device_count": torch.cuda.device_count(),
        "device_names": actual_names,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
    }
    topology = gate_receipt.get("topology")
    pool_runtime = runtime_receipt_method()
    expected_topology = _smplx_training_pool_topology(
        requested_mode,
        requested_helpers,
    )
    if (
        torch.cuda.current_device() != 0
        or actual_runtime["visible_device_count"] != visible_device_count
        or actual_names != ["NVIDIA H200"] * visible_device_count
        or runtime != actual_runtime
        or topology != expected_topology
        or gate_receipt.get("topology_sha256")
        != _payload_sha256(expected_topology)
        or not isinstance(pool_runtime, dict)
        or pool_runtime.get("format")
        != expected_topology["pool_runtime_format"]
        or pool_runtime.get("mode") != expected_topology["mode"]
        or pool_runtime.get("primary_device") != 0
        or pool_runtime.get("helper_devices")
        != expected_topology["helper_devices"]
        or pool_runtime.get("replica_devices")
        != expected_topology["replica_devices"]
        or pool_runtime.get("partition")
        != expected_topology["partition"]
        or pool_runtime.get("transfer_to_primary")
        != expected_topology["transfer_to_primary"]
        or pool_runtime.get("last_forward") is not None
        or (
            requested_mode == "target_offload"
            and pool_runtime.get("completed_forward_pairs") != 0
        )
        or (
            requested_mode != "target_offload"
            and "completed_forward_pairs" in pool_runtime
        )
    ):
        raise RuntimeError(
            "trainer SMPL-X runtime differs from the formal gate runtime"
        )


def _attach_smplx_training_pool_gate_receipt(
    payload: dict[str, Any],
    receipt: dict[str, Any] | None,
) -> None:
    if SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY in payload:
        raise RuntimeError("duplicate SMPL-X training pool gate receipt")
    if receipt is not None:
        payload[SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY] = copy.deepcopy(
            receipt
        )


def _verify_smplx_training_pool_gate_resume_receipt(
    payload: dict[str, Any],
    expected_receipt: dict[str, Any] | None,
) -> None:
    observed_present = SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY in payload
    expected_present = expected_receipt is not None
    if (
        observed_present != expected_present
        or (
            expected_present
            and payload.get(SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY)
            != expected_receipt
        )
    ):
        raise RuntimeError(
            "resume SMPL-X training pool gate receipt does not match"
        )


def _smplx_training_pool_gate_overlay(
    receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    if receipt is None:
        return {}
    return {
        SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY: copy.deepcopy(receipt)
    }


def _validate_smplx_training_pool_runtime_evidence(
    receipt: Any,
    *,
    formal_stage: str,
    gate_receipt: dict[str, Any] | None,
    expected_optimizer_updates: int | None = None,
) -> dict[str, Any] | None:
    """Validate proof that production executed the authorized local-loss path."""
    if formal_stage not in FORMAL_SMPLX_STAGES:
        if receipt is not None:
            raise RuntimeError(
                "global/Base must not carry SMPL-X pool runtime evidence"
            )
        return None
    if gate_receipt is None and receipt is None:
        return None
    if not isinstance(gate_receipt, dict):
        raise RuntimeError(
            "SMPL-X pool runtime evidence requires its formal gate receipt"
        )
    required_keys = {
        "format",
        "formal_stage",
        "mode",
        "gate_report_sha256",
        "gate_topology_sha256",
        "source_binding",
        "pool_runtime",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != required_keys:
        raise RuntimeError("SMPL-X pool runtime evidence schema is invalid")
    receipt_body = dict(receipt)
    receipt_sha256 = receipt_body.pop("receipt_sha256")
    mode = gate_receipt.get("mode")
    spec = _smplx_training_pool_mode_spec(mode)
    if (
        receipt.get("format")
        != spec["runtime_evidence_format"]
        or receipt.get("formal_stage") != formal_stage
        or receipt.get("mode") != mode
        or receipt.get("gate_report_sha256") != gate_receipt.get("sha256")
        or receipt.get("gate_topology_sha256")
        != gate_receipt.get("topology_sha256")
        or receipt.get("source_binding")
        != gate_receipt.get("source_binding")
        or receipt_sha256 != _payload_sha256(receipt_body)
    ):
        raise RuntimeError(
            "SMPL-X pool runtime evidence is not gate/source bound"
        )

    topology = gate_receipt.get("topology")
    runtime = receipt.get("pool_runtime")
    runtime_keys = {
        "format",
        "mode",
        "primary_device",
        "helper_devices",
        "replica_devices",
        "partition",
        "transfer_to_primary",
        "last_forward",
    }
    if mode == "target_offload":
        runtime_keys.add("completed_forward_pairs")
    if (
        not isinstance(topology, dict)
        or not isinstance(runtime, dict)
        or set(runtime) != runtime_keys
        or runtime.get("format") != topology.get("pool_runtime_format")
        or runtime.get("mode") != topology.get("mode")
        or runtime.get("primary_device") != topology.get("primary_device")
        or runtime.get("helper_devices") != topology.get("helper_devices")
        or runtime.get("replica_devices") != topology.get("replica_devices")
        or runtime.get("partition") != topology.get("partition")
        or runtime.get("transfer_to_primary")
        != topology.get("transfer_to_primary")
    ):
        raise RuntimeError(
            "SMPL-X pool runtime evidence topology differs from its gate"
        )
    if mode == "target_offload":
        completed_forward_pairs = _require_exact_audit_int(
            runtime.get("completed_forward_pairs"),
            "SMPL-X target_offload completed_forward_pairs",
        )
        if completed_forward_pairs < 0:
            raise RuntimeError(
                "SMPL-X target_offload completed_forward_pairs is negative"
            )
        if (
            expected_optimizer_updates is not None
            and completed_forward_pairs
            != _require_exact_audit_int(
                expected_optimizer_updates,
                "SMPL-X target_offload expected optimizer updates",
            )
        ):
            raise RuntimeError(
                "SMPL-X target_offload forward count differs from optimizer "
                "updates"
            )

    last_forward = runtime.get("last_forward")
    _validate_smplx_training_pool_last_forward(
        last_forward,
        formal_stage=formal_stage,
        topology=topology,
    )
    return copy.deepcopy(receipt)


def _current_smplx_training_pool_runtime_evidence(
    trainer: Any,
    *,
    formal_stage: str,
    gate_receipt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if (
        formal_stage not in FORMAL_SMPLX_STAGES
        or gate_receipt is None
    ):
        return _validate_smplx_training_pool_runtime_evidence(
            None,
            formal_stage=formal_stage,
            gate_receipt=gate_receipt,
        )
    pool = getattr(trainer, "smplx_pool", None)
    runtime_method = getattr(pool, "runtime_receipt", None)
    if not callable(runtime_method):
        raise RuntimeError("SMPL-X pool runtime evidence is unavailable")
    mode = gate_receipt.get("mode")
    spec = _smplx_training_pool_mode_spec(mode)
    pool_runtime = runtime_method()
    if not isinstance(pool_runtime, dict):
        raise RuntimeError("SMPL-X pool runtime receipt is not an object")
    if mode == "target_offload":
        last_forward = pool_runtime.get("last_forward")
        if (
            not isinstance(last_forward, dict)
            or last_forward.get("stage") != formal_stage
        ):
            raise RuntimeError(
                "target_offload pool last-forward evidence is unavailable"
            )
    body = {
        "format": spec["runtime_evidence_format"],
        "formal_stage": formal_stage,
        "mode": mode,
        "gate_report_sha256": gate_receipt.get("sha256"),
        "gate_topology_sha256": gate_receipt.get("topology_sha256"),
        "source_binding": copy.deepcopy(gate_receipt.get("source_binding")),
        "pool_runtime": pool_runtime,
    }
    receipt = {**body, "receipt_sha256": _payload_sha256(body)}
    return _validate_smplx_training_pool_runtime_evidence(
        receipt,
        formal_stage=formal_stage,
        gate_receipt=gate_receipt,
    )


def _restore_smplx_training_pool_runtime_evidence(
    trainer: Any,
    payload: dict[str, Any],
    *,
    gate_receipt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY not in payload:
        raise RuntimeError(
            "resume is missing SMPL-X pool runtime evidence"
        )
    receipt = _validate_smplx_training_pool_runtime_evidence(
        payload.get(SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY),
        formal_stage=trainer.args.formal_stage,
        gate_receipt=gate_receipt,
        expected_optimizer_updates=payload.get("optimizer_updates"),
    )
    if (
        receipt is not None
        and receipt.get("mode") == "target_offload"
    ):
        pool = getattr(trainer, "smplx_pool", None)
        restore_method = getattr(
            pool,
            "restore_completed_forward_pairs",
            None,
        )
        if not callable(restore_method):
            raise RuntimeError(
                "target_offload pool cannot restore completed forward pairs"
            )
        restore_method(
            receipt["pool_runtime"]["completed_forward_pairs"]
        )
    trainer.smplx_training_pool_runtime_evidence = copy.deepcopy(receipt)
    return receipt


def _smplx_training_pool_runtime_evidence_overlay(
    receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY: copy.deepcopy(receipt)
    }


def _formal_smplx_training_pool_gate_receipt(
    args: Any,
    *,
    current_source: dict[str, str],
    representation: dict[str, Any],
    smplx_asset_receipt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the source-bound, self-contained H200 pool gate."""

    mode = getattr(args, "smplx_training_pool_mode", None)
    report_value = getattr(
        args,
        "smplx_training_pool_gate_report",
        None,
    )
    expected_value = getattr(
        args,
        "expected_smplx_training_pool_gate_sha256",
        None,
    )
    stage = getattr(args, "formal_stage", None)
    if mode == "disabled":
        if stage not in {*FORMAL_SMPLX_STAGES, "global", "base"}:
            raise RuntimeError("invalid formal stock SMPL-X stage")
        if report_value not in {None, ""} or expected_value not in {None, ""}:
            raise RuntimeError(
                "disabled SMPL-X pooling forbids formal gate arguments"
            )
        return None
    try:
        helpers = parse_smplx_helper_devices(
            getattr(args, "smplx_training_helper_devices", "")
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "invalid topology-specific SMPL-X helper devices"
        ) from error
    spec = _smplx_training_pool_mode_spec(mode)
    _validate_smplx_training_pool_helpers(mode, helpers)
    if stage not in FORMAL_SMPLX_STAGES:
        raise RuntimeError(
            f"{mode} gate requires a formal RVQ stage"
        )
    if type(report_value) is not str or not report_value:
        raise RuntimeError(
            f"{mode} requires --smplx_training_pool_gate_report"
        )
    gate_format = spec["gate_format"]
    helper_list = list(helpers)
    helper_csv = ",".join(str(device) for device in helpers)
    visible_device_count = len(helpers) + 1
    expected_sha = _require_lowercase_sha256(
        expected_value,
        "expected SMPL-X training pool gate SHA-256",
    )
    report_input = Path(report_value)
    if report_input.is_symlink() or not report_input.is_file():
        raise RuntimeError(
            "SMPL-X training pool gate report must be a regular file"
        )
    report_path = report_input.resolve()
    report_sha = _sha256(report_path)
    if report_sha != expected_sha:
        raise RuntimeError("SMPL-X training pool gate report SHA mismatch")
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise RuntimeError("SMPL-X training pool gate report must be an object")

    scope = report.get("scope")
    protocol = report.get("protocol")
    preflight = report.get("preflight")
    semantic_common = report.get("semantic_common")
    equivalence = report.get("equivalence")
    performance = report.get("performance")
    children = report.get("children")
    guarded_ancestry = report.get("guarded_ancestry")
    report_topology = report.get("topology")
    if (
        set(report)
        != {
            "format",
            "status",
            "authorization",
            "scope",
            "topology",
            "protocol",
            "preflight",
            "semantic_common",
            "equivalence",
            "performance",
            "guarded_ancestry",
            "children",
            "started_unix",
            "completed_unix",
        }
        or report.get("format") != gate_format
        or report.get("status") != "pass"
        or report.get("authorization") is not True
        or not isinstance(scope, dict)
        or scope.get("stage") != args.formal_stage
        or scope.get("dataset") != "show_base"
        or scope.get("speaker_scope") != "All"
        or scope.get("speaker_ids") != [0, 1, 2, 3]
        or scope.get("forbidden")
        != ["speaker2-only", "SemGate", "sparse motion"]
        or not isinstance(protocol, dict)
        or protocol.get("candidate_mode") != mode
        or protocol.get("helper_devices") != helper_list
        or protocol.get("visible_device_count") != visible_device_count
        or (
            mode == "sharded_local_loss"
            and protocol.get("bounded_tolerances")
            != SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES
        )
        or (
            mode == "target_offload"
            and protocol.get("bounded_tolerances") is not None
        )
        or report_topology
        != {
            "primary_device": 0,
            "helper_devices": helper_list,
            "visible_device_count": visible_device_count,
            "device_name": "NVIDIA H200",
        }
        or _require_exact_audit_int(
            protocol.get("equivalence_updates"),
            "SMPL-X pool gate equivalence_updates",
        )
        != 2
        or protocol.get("abba")
        != ["stock", "candidate", "candidate", "stock"]
        or _require_exact_audit_int(
            protocol.get("warmup"),
            "SMPL-X pool gate warmup",
        )
        != 5
        or _require_exact_audit_int(
            protocol.get("measured"),
            "SMPL-X pool gate measured",
        )
        != 25
        or not isinstance(preflight, dict)
        or not isinstance(semantic_common, dict)
        or not isinstance(equivalence, dict)
        or not isinstance(performance, dict)
        or not isinstance(children, list)
        or len(children) != 8
        or not isinstance(guarded_ancestry, list)
        or len(guarded_ancestry) < 3
    ):
        raise RuntimeError("SMPL-X training pool gate schema is incomplete")

    minimum = protocol.get("minimum_speedup")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, Real)
        or not np.isfinite(float(minimum))
        or float(minimum) != spec["minimum_speedup"]
    ):
        raise RuntimeError(
            "SMPL-X pool gate minimum speedup differs from its mode contract"
        )
    minimum = float(minimum)

    gate_source = preflight.get("source")
    required_source_keys = {
        "origin",
        "commit",
        "tree",
        "implementation_files",
        "reference_gate",
        "reference_gate_sha256",
        "adapter",
        "adapter_sha256",
        "harness_sha256",
        "coordinator",
    }
    if (
        preflight.get("format") != gate_format
        or preflight.get("stage") != args.formal_stage
        or preflight.get("candidate_mode") != mode
        or preflight.get("helper_devices") != helper_list
        or preflight.get("real_show_all") is not True
        or preflight.get("preliminary_numerical_gate", {}).get(
            "authorization"
        )
        is not False
        or not isinstance(gate_source, dict)
        or set(gate_source) != required_source_keys
        or gate_source.get("harness_sha256")
        != spec["harness_sha256"]
        or gate_source.get("adapter_sha256")
        != spec["adapter_sha256"]
        or {
            key: gate_source.get(key)
            for key in ("origin", "commit", "tree")
        }
        != {
            key: current_source.get(key)
            for key in ("origin", "commit", "tree")
        }
    ):
        raise RuntimeError("SMPL-X pool gate source/stage binding differs")
    preliminary_binding: dict[str, Any] | None = None
    if mode == "target_offload":
        preliminary_binding = (
            _validate_target_offload_preliminary_numerical_gate(
                preflight.get("preliminary_numerical_gate"),
                smplx_asset_receipt=smplx_asset_receipt,
            )
        )
    for label in (
        "reference_gate_sha256",
        "adapter_sha256",
        "harness_sha256",
    ):
        _require_lowercase_sha256(
            gate_source.get(label),
            f"SMPL-X pool gate {label}",
        )
    implementation_files = gate_source.get("implementation_files")
    if (
        not isinstance(implementation_files, dict)
        or set(implementation_files)
        != SMPLX_TRAINING_POOL_IMPLEMENTATION_FILES
    ):
        raise RuntimeError(
            "SMPL-X pool gate implementation file set differs"
        )
    repository = Path(__file__).resolve().parent
    for relative_value, observed_sha in implementation_files.items():
        if type(relative_value) is not str:
            raise RuntimeError("invalid SMPL-X pool implementation path")
        relative = Path(relative_value)
        current_path = repository / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or current_path.is_symlink()
            or not current_path.is_file()
            or _sha256(current_path)
            != _require_lowercase_sha256(
                observed_sha,
                f"SMPL-X pool implementation SHA {relative_value}",
            )
        ):
            raise RuntimeError(
                f"SMPL-X pool implementation binding differs: {relative_value}"
            )
    coordinator = gate_source.get("coordinator")
    coordinator_path_value = (
        coordinator.get("path")
        if isinstance(coordinator, dict)
        else None
    )
    coordinator_sha = (
        coordinator.get("sha256")
        if isinstance(coordinator, dict)
        else None
    )
    source_coordinator_identity = (
        coordinator.get("identity")
        if isinstance(coordinator, dict)
        else None
    )
    coordinator_path = (
        Path(coordinator_path_value)
        if type(coordinator_path_value) is str
        else Path()
    )
    if (
        not isinstance(coordinator, dict)
        or set(coordinator) != {"path", "sha256", "identity"}
        or not coordinator_path.is_absolute()
        or coordinator_path != coordinator_path.resolve()
        or coordinator_path.is_symlink()
        or not coordinator_path.is_file()
        or _sha256(coordinator_path)
        != _require_lowercase_sha256(
            coordinator_sha,
            "SMPL-X pool coordinator SHA",
        )
        or not isinstance(source_coordinator_identity, dict)
    ):
        raise RuntimeError("SMPL-X pool coordinator source binding differs")

    expected_representation = {
        "lmdb": str(Path(representation["lmdb"]).resolve()),
        "data_sha256": representation["data_sha256"],
        "summary": str(Path(representation["summary"]).resolve()),
        "summary_sha256": representation["summary_sha256"],
        "lineage": str(Path(representation["lineage"]).resolve()),
        "lineage_sha256": representation["lineage_sha256"],
        "entry_aggregate_sha256": representation[
            "entry_aggregate_sha256"
        ],
    }
    for key in (
        "data_sha256",
        "summary_sha256",
        "lineage_sha256",
        "entry_aggregate_sha256",
    ):
        _require_lowercase_sha256(
            expected_representation[key],
            f"current representation {key}",
        )
    runtime = semantic_common.get("runtime")
    if (
        semantic_common.get("format") != gate_format
        or semantic_common.get("source") != gate_source
        or semantic_common.get("stage") != args.formal_stage
        or semantic_common.get("dataset") != "show_base"
        or semantic_common.get("speaker_scope") != "All"
        or semantic_common.get("speaker_ids") != [0, 1, 2, 3]
        or _require_exact_audit_int(
            semantic_common.get("batch_size"),
            "SMPL-X pool gate batch_size",
        )
        != 64
        or _require_exact_audit_int(
            semantic_common.get("frames"),
            "SMPL-X pool gate frames",
        )
        != 64
        or semantic_common.get("representation")
        != expected_representation
        or smplx_asset_receipt is None
        or semantic_common.get("smplx_asset_sha256")
        != smplx_asset_receipt.get("sha256")
        or semantic_common.get("preliminary_full_batch_numerical_gate")
        != preflight.get("preliminary_numerical_gate")
        or not isinstance(runtime, dict)
        or runtime.get("device_names")
        != ["NVIDIA H200"] * visible_device_count
        or type(runtime.get("torch")) is not str
        or not runtime["torch"]
        or type(runtime.get("cuda")) is not str
        or not runtime["cuda"]
        or type(runtime.get("cudnn")) is not int
        or runtime["cudnn"] <= 0
        or type(runtime.get("visible")) is not str
    ):
        raise RuntimeError(
            "SMPL-X pool gate semantic/runtime/representation binding differs"
        )
    visible_devices = runtime["visible"].split(",")
    if (
        len(visible_devices) != visible_device_count
        or any(not value for value in visible_devices)
        or len(set(visible_devices)) != visible_device_count
    ):
        raise RuntimeError(
            "SMPL-X pool gate runtime does not bind the exact requested H200 "
            "topology"
        )

    public_identity = guarded_ancestry[0]
    if (
        not isinstance(public_identity, dict)
        or type(public_identity.get("pid")) is not int
        or public_identity["pid"] <= 1
    ):
        raise RuntimeError("SMPL-X pool gate public ancestry is invalid")
    public_pid = public_identity["pid"]
    coordinator_identity = guarded_ancestry[1]
    runner_identity = guarded_ancestry[2]
    if (
        not isinstance(coordinator_identity, dict)
        or not isinstance(runner_identity, dict)
        or not isinstance(coordinator_identity.get("argv"), list)
        or not isinstance(runner_identity.get("argv"), list)
    ):
        raise RuntimeError(
            "SMPL-X pool coordinator/guarded-runner ancestry is invalid"
        )
    if (
        type(coordinator_identity.get("pid")) is not int
        or coordinator_identity["pid"] <= 1
        or coordinator_identity.get("ppid") != runner_identity.get("pid")
        or str(coordinator_path)
        not in coordinator_identity.get("argv", [])
        or public_identity.get("ppid") != coordinator_identity["pid"]
        or "/tmp/globaldiff_guarded_runner.py"
        not in runner_identity["argv"]
        or source_coordinator_identity != coordinator_identity
    ):
        raise RuntimeError(
            "SMPL-X pool coordinator/guarded-runner ancestry differs"
        )

    def cli_values(argv: list[str], flag: str) -> list[str]:
        values: list[str] = []
        for index, item in enumerate(argv):
            if item == flag:
                if index + 1 >= len(argv):
                    raise RuntimeError(
                        f"SMPL-X pool gate child has dangling {flag}"
                    )
                values.append(argv[index + 1])
            elif item.startswith(flag + "="):
                values.append(item.split("=", 1)[1])
        return values

    def cli_value(argv: list[str], flag: str) -> str:
        values = cli_values(argv, flag)
        if len(values) != 1:
            raise RuntimeError(
                f"SMPL-X pool gate child requires exactly one {flag}"
            )
        return values[0]

    runner_gpus = cli_value(runner_identity["argv"], "--gpus")
    runner_gpu_values = runner_gpus.split(",")
    if (
        any(not value or not value.isdecimal() for value in runner_gpu_values)
        or len(set(runner_gpu_values)) != len(runner_gpu_values)
        or (
            mode == "target_offload"
            and runner_gpu_values
            != [str(index) for index in range(8)]
        )
    ):
        raise RuntimeError(
            "SMPL-X pool guarded-runner GPU argv is invalid"
        )

    expected_children = [
        ("equivalence-stock-0", "equivalence", "stock"),
        ("equivalence-candidate-0", "equivalence", "candidate"),
        ("equivalence-stock-1", "equivalence", "stock"),
        ("equivalence-candidate-1", "equivalence", "candidate"),
        ("benchmark-0-stock", "benchmark", "stock"),
        ("benchmark-1-candidate", "benchmark", "candidate"),
        ("benchmark-2-candidate", "benchmark", "candidate"),
        ("benchmark-3-stock", "benchmark", "stock"),
    ]
    child_results: list[dict[str, Any]] = []
    process_identities: set[tuple[int, str]] = set()
    child_pids: set[int] = set()
    result_shas: list[str] = []
    harness_path: Path | None = None
    for sequence, (child, expected_child) in enumerate(
        zip(children, expected_children)
    ):
        expected_name, expected_kind, expected_mode = expected_child
        if not isinstance(child, dict):
            raise RuntimeError("SMPL-X pool gate child receipt is not an object")
        argv = child.get("argv")
        pid = child.get("pid")
        result_value = child.get("result_path")
        result_expected_sha = _require_lowercase_sha256(
            child.get("result_sha256"),
            f"SMPL-X pool gate child {sequence} result SHA",
        )
        if (
            child.get("sequence") != sequence
            or child.get("name") != expected_name
            or child.get("kind") != expected_kind
            or child.get("mode") != expected_mode
            or type(pid) is not int
            or pid <= 1
            or pid in child_pids
            or child.get("return_code") != 0
            or not isinstance(argv, list)
            or len(argv) < 2
            or any(type(item) is not str for item in argv)
            or child.get("argv_sha256")
            != hashlib.sha256(
                b"\0".join(item.encode() for item in argv) + b"\0"
            ).hexdigest()
            or type(result_value) is not str
            or not result_value
        ):
            raise RuntimeError(
                f"SMPL-X pool gate child {sequence} command is invalid"
            )
        child_pids.add(pid)
        candidate_harness = Path(argv[1])
        if harness_path is None:
            harness_path = candidate_harness
        if (
            candidate_harness != harness_path
            or candidate_harness.is_symlink()
            or not candidate_harness.is_file()
            or _sha256(candidate_harness.resolve())
            != gate_source["harness_sha256"]
        ):
            raise RuntimeError("SMPL-X pool gate harness binding differs")
        if (
            cli_value(argv, "--stage") != args.formal_stage
            or cli_value(argv, "--candidate-mode") != mode
            or cli_value(argv, "--helper-devices") != helper_csv
            or Path(cli_value(argv, "--representation-lmdb")).resolve()
            != Path(expected_representation["lmdb"])
            or Path(cli_value(argv, "--representation-summary")).resolve()
            != Path(expected_representation["summary"])
            or Path(cli_value(argv, "--representation-lineage")).resolve()
            != Path(expected_representation["lineage"])
            or cli_value(argv, "--expected-representation-data-sha256")
            != expected_representation["data_sha256"]
            or cli_value(argv, "--expected-representation-summary-sha256")
            != expected_representation["summary_sha256"]
            or cli_value(argv, "--expected-representation-lineage-sha256")
            != expected_representation["lineage_sha256"]
            or cli_value(
                argv,
                "--expected-representation-entry-aggregate-sha256",
            )
            != expected_representation["entry_aggregate_sha256"]
            or cli_value(argv, "--expected-smplx-asset-sha256")
            != smplx_asset_receipt["sha256"]
            or cli_value(argv, "--expected-source-commit")
            != current_source["commit"]
            or cli_value(argv, "--expected-source-tree")
            != current_source["tree"]
            or cli_value(argv, "--expected-device-name") != "NVIDIA H200"
            or cli_value(argv, "--expected-visible-device-count")
            != str(visible_device_count)
            or cli_value(argv, "--runner-gpus") != runner_gpus
            or (
                mode == "target_offload"
                and (
                    preliminary_binding is None
                    or Path(cli_value(argv, "--asset-root")).resolve()
                    != Path(preliminary_binding["asset_root"])
                    or Path(
                        cli_value(argv, "--numerical-gate-report")
                    ).resolve()
                    != Path(
                        preliminary_binding["files"][
                            "numerical report"
                        ]["path"]
                    )
                    or cli_value(
                        argv,
                        "--expected-numerical-gate-report-sha256",
                    )
                    != preliminary_binding["files"][
                        "numerical report"
                    ]["sha256"]
                    or Path(
                        cli_value(argv, "--numerical-gate-script")
                    ).resolve()
                    != Path(
                        preliminary_binding["files"][
                            "numerical script"
                        ]["path"]
                    )
                    or cli_value(
                        argv,
                        "--expected-numerical-gate-script-sha256",
                    )
                    != preliminary_binding["files"][
                        "numerical script"
                    ]["sha256"]
                    or Path(
                        cli_value(
                            argv,
                            "--numerical-gate-runner-status",
                        )
                    ).resolve()
                    != Path(
                        preliminary_binding["files"][
                            "numerical runner status"
                        ]["path"]
                    )
                    or cli_value(
                        argv,
                        "--expected-numerical-gate-runner-status-sha256",
                    )
                    != preliminary_binding["files"][
                        "numerical runner status"
                    ]["sha256"]
                )
            )
            or cli_value(argv, "--expected-runner-pid")
            != str(runner_identity["pid"])
            or cli_value(argv, "--expected-runner-starttime")
            != str(runner_identity["starttime"])
            or cli_value(argv, "--expected-runner-argv-sha256")
            != runner_identity["argv_sha256"]
            or cli_value(argv, "--coordinator-pid")
            != str(coordinator_identity["pid"])
            or Path(cli_value(argv, "--coordinator-script")).resolve()
            != coordinator_path
            or cli_value(
                argv,
                "--expected-coordinator-script-sha256",
            )
            != coordinator_sha
            or cli_value(
                argv,
                "--expected-coordinator-starttime",
            )
            != str(coordinator_identity["starttime"])
            or cli_value(
                argv,
                "--expected-coordinator-argv-sha256",
            )
            != coordinator_identity["argv_sha256"]
            or float(cli_value(argv, "--minimum-speedup")) != minimum
            or Path(cli_value(argv, "--output-root")).resolve()
            != report_path.parent
            or cli_value(argv, "--internal-mode") != expected_name
            or cli_value(argv, "--internal-parent-pid") != str(public_pid)
            or set(cli_values(argv, "--implementation-file"))
            != set(implementation_files)
        ):
            raise RuntimeError(
                f"SMPL-X pool gate child {sequence} input binding differs"
            )
        adapter_path = Path(cli_value(argv, "--adapter"))
        if (
            adapter_path.resolve()
            != Path(str(gate_source["adapter"])).resolve()
            or adapter_path.is_symlink()
            or not adapter_path.is_file()
            or _sha256(adapter_path.resolve()) != gate_source["adapter_sha256"]
        ):
            raise RuntimeError("SMPL-X pool gate adapter binding differs")

        result_input = Path(result_value)
        if result_input.is_symlink() or not result_input.is_file():
            raise RuntimeError(
                f"SMPL-X pool gate child {sequence} result is unavailable"
            )
        result_path = result_input.resolve()
        if (
            result_path.name != "result.json"
            or result_path.parent
            != Path(cli_value(argv, "--snapshot-dir")).resolve()
            or _sha256(result_path) != result_expected_sha
        ):
            raise RuntimeError(
                f"SMPL-X pool gate child {sequence} result SHA/path differs"
            )
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        ancestry = result.get("ancestry") if isinstance(result, dict) else None
        semantic = result.get("semantic") if isinstance(result, dict) else None
        child_identity = ancestry[0] if isinstance(ancestry, list) and ancestry else None
        if (
            not isinstance(result, dict)
            or result.get("format") != gate_format
            or result.get("status") != "pass"
            or result.get("kind") != expected_kind
            or result.get("mode") != expected_mode
            or not isinstance(semantic, dict)
            or semantic.get("common") != semantic_common
            or semantic.get("mode", {}).get("label") != expected_mode
            or semantic.get("mode", {}).get("smplx_parallel_mode")
            != (
                "disabled"
                if expected_mode == "stock"
                else mode
            )
            or semantic.get("mode", {}).get("helper_devices")
            != ([] if expected_mode == "stock" else helper_list)
            or not isinstance(ancestry, list)
            or ancestry[1:] != guarded_ancestry
            or not isinstance(child_identity, dict)
            or child_identity.get("pid") != pid
            or child_identity.get("ppid") != public_pid
            or child_identity.get("argv") != argv
            or child_identity.get("argv_sha256")
            != child["argv_sha256"]
            or type(child_identity.get("starttime")) is not str
            or not child_identity["starttime"]
        ):
            raise RuntimeError(
                f"SMPL-X pool gate child {sequence} result binding differs"
            )
        identity = (pid, child_identity["starttime"])
        if identity in process_identities:
            raise RuntimeError("SMPL-X pool gate children are not fresh")
        process_identities.add(identity)
        result_shas.append(result_expected_sha)
        child_results.append(result)

    reference_gate_path = Path(str(gate_source["reference_gate"]))
    if (
        reference_gate_path.is_symlink()
        or not reference_gate_path.is_file()
        or _sha256(reference_gate_path.resolve())
        != gate_source["reference_gate_sha256"]
    ):
        raise RuntimeError("SMPL-X pool reference gate binding differs")

    def validate_equivalence_proof(proof: Any, label: str) -> None:
        repeat = label in {"stock_repeat", "candidate_repeat"}
        expected_comparison = (
            "repeat_byte_exact"
            if repeat
            else spec["cross_mode_comparison"]
        )
        expected_final = (
            "byte_exact"
            if repeat
            else spec["cross_mode_final"]
        )
        if (
            not isinstance(proof, dict)
            or set(proof)
            != {
                "status",
                "real_batches_exact",
                "comparison",
                "states",
                "numeric_scope",
                "initial_byte_exact",
                "two_complete_updates_and_final",
            }
            or proof.get("status") != "pass"
            or proof.get("real_batches_exact") is not True
            or proof.get("comparison") != expected_comparison
            or proof.get("numeric_scope")
            != (
                "loss_tracker_gradients_model_optimizer_scheduler_"
                "rvq_ema_rng"
            )
            or proof.get("initial_byte_exact") is not True
            or proof.get("two_complete_updates_and_final")
            != expected_final
            or set(proof.get("states", {}))
            != {"initial", "step_1", "step_2", "final"}
        ):
            raise RuntimeError(
                f"SMPL-X pool equivalence proof {label} is incomplete"
            )
        for state_name, state in proof["states"].items():
            exact = (
                repeat
                or state_name == "initial"
                or mode == "target_offload"
            )
            if not isinstance(state, dict) or state.get("pass") is not True:
                raise RuntimeError(
                    f"SMPL-X pool equivalence {label}.{state_name} differs"
                )
            if exact:
                if (
                    set(state) != {"comparison", "pass", "canonical_sha256"}
                    or state.get("comparison") != "byte_exact"
                ):
                    raise RuntimeError(
                        f"SMPL-X pool exact proof {label}.{state_name} differs"
                    )
                _require_lowercase_sha256(
                    state.get("canonical_sha256"),
                    f"SMPL-X pool equivalence {label}.{state_name} SHA",
                )
                continue
            if (
                set(state)
                != {
                    "comparison",
                    "pass",
                    "tolerances",
                    "exact_nodes",
                    "all_floating_values_within_tolerance",
                    "rng_indices_scheduler_smplx_and_nonfloating_exact",
                }
                or state.get("comparison")
                != "bounded_float_exact_discrete_rng_and_smplx"
                or state.get("all_floating_values_within_tolerance")
                is not True
                or state.get(
                    "rng_indices_scheduler_smplx_and_nonfloating_exact"
                )
                is not True
                or type(state.get("exact_nodes")) is not int
                or state["exact_nodes"] < 0
                or set(state.get("tolerances", {}))
                != set(SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES)
            ):
                raise RuntimeError(
                    f"SMPL-X pool bounded proof {label}.{state_name} differs"
                )
            for category, expected_tolerance in (
                SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES.items()
            ):
                observed = state["tolerances"][category]
                if (
                    not isinstance(observed, dict)
                    or set(observed)
                    != {
                        "atol",
                        "rtol",
                        "floating_nodes",
                        "elements",
                        "bitwise_mismatch_elements",
                        "max_abs",
                        "max_normalized_error",
                    }
                    or observed.get("atol") != expected_tolerance["atol"]
                    or observed.get("rtol") != expected_tolerance["rtol"]
                    or type(observed.get("floating_nodes")) is not int
                    or observed["floating_nodes"] < 0
                    or type(observed.get("elements")) is not int
                    or observed["elements"] < 0
                    or type(observed.get("bitwise_mismatch_elements")) is not int
                    or observed["bitwise_mismatch_elements"] < 0
                    or observed["bitwise_mismatch_elements"]
                    > observed["elements"]
                    or isinstance(observed.get("max_abs"), bool)
                    or not isinstance(observed.get("max_abs"), Real)
                    or not np.isfinite(float(observed["max_abs"]))
                    or float(observed["max_abs"]) < 0.0
                    or isinstance(
                        observed.get("max_normalized_error"),
                        bool,
                    )
                    or not isinstance(
                        observed.get("max_normalized_error"),
                        Real,
                    )
                    or not np.isfinite(
                        float(observed["max_normalized_error"])
                    )
                    or float(observed["max_normalized_error"]) < 0.0
                    or float(observed["max_normalized_error"])
                    > 1.0 + 1e-12
                ):
                    raise RuntimeError(
                        "SMPL-X pool bounded tolerance evidence differs: "
                        f"{label}.{state_name}.{category}"
                    )

    expected_proofs = {
        "stock_repeat",
        "candidate_repeat",
        "stock_candidate_0",
        "stock_candidate_1",
    }
    if set(equivalence) != expected_proofs:
        raise RuntimeError("SMPL-X pool equivalence proof set differs")
    for proof_name in sorted(expected_proofs):
        validate_equivalence_proof(equivalence[proof_name], proof_name)
    for result in child_results[:4]:
        if (
            result.get("optimizer_steps") != 2
            or result.get("formal_updates") != 2
            or result.get("scheduler_steps") != 1
            or len(result.get("batches", [])) != 2
            or set(result.get("states", {}))
            != {"initial", "step_1", "step_2", "final"}
        ):
            raise RuntimeError(
                "SMPL-X pool equivalence child update evidence differs"
            )

    benchmark_results = child_results[4:]
    expected_order = ["stock", "candidate", "candidate", "stock"]
    if (
        performance.get("order") != expected_order
        or performance.get("fresh_processes") != 4
        or [result.get("mode") for result in benchmark_results]
        != expected_order
    ):
        raise RuntimeError("SMPL-X pool ABBA benchmark order differs")
    for result in benchmark_results:
        timing = result.get("timing")
        if (
            result.get("optimizer_steps") != 30
            or result.get("formal_updates") != 30
            or result.get("scheduler_steps") != 1
            or not isinstance(timing, dict)
            or timing.get("warmup") != 5
            or timing.get("measured") != 25
            or len(timing.get("wall_seconds", [])) != 25
            or len(timing.get("cuda_seconds", [])) != 25
            or len(timing.get("batches", [])) != 30
        ):
            raise RuntimeError(
                "SMPL-X pool benchmark child timing evidence differs"
            )
        for key in ("wall_seconds", "cuda_seconds"):
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or float(value) <= 0.0
                for value in timing[key]
            ):
                raise RuntimeError(
                    f"SMPL-X pool benchmark {key} is invalid"
                )

    performance_receipt: dict[str, Any] = {}
    for report_key, timing_key in (
        ("wall", "wall_seconds"),
        ("cuda", "cuda_seconds"),
    ):
        metric = performance.get(report_key)
        stock = [
            float(value)
            for result in benchmark_results
            if result["mode"] == "stock"
            for value in result["timing"][timing_key]
        ]
        candidate = [
            float(value)
            for result in benchmark_results
            if result["mode"] == "candidate"
            for value in result["timing"][timing_key]
        ]
        stock_median = statistics.median(stock)
        candidate_median = statistics.median(candidate)
        pooled_speedup = stock_median / candidate_median
        paired_speedups = [
            statistics.median(
                benchmark_results[0]["timing"][timing_key]
            )
            / statistics.median(
                benchmark_results[1]["timing"][timing_key]
            ),
            statistics.median(
                benchmark_results[3]["timing"][timing_key]
            )
            / statistics.median(
                benchmark_results[2]["timing"][timing_key]
            ),
        ]
        if (
            not isinstance(metric, dict)
            or metric.get("pass") is not True
            or float(metric.get("minimum_required", float("nan")))
            != minimum
            or float(metric.get("stock_median_seconds", float("nan")))
            != stock_median
            or float(metric.get("candidate_median_seconds", float("nan")))
            != candidate_median
            or float(metric.get("pooled_speedup", float("nan")))
            != pooled_speedup
            or metric.get("paired_speedups") != paired_speedups
            or pooled_speedup < minimum
            or min(paired_speedups) < minimum
        ):
            raise RuntimeError(
                f"SMPL-X pool {report_key} speedup evidence differs"
            )
        performance_receipt[report_key] = {
            "pooled_speedup": pooled_speedup,
            "paired_speedups": paired_speedups,
        }

    started = report.get("started_unix")
    completed = report.get("completed_unix")
    if (
        isinstance(started, bool)
        or not isinstance(started, Real)
        or isinstance(completed, bool)
        or not isinstance(completed, Real)
        or not np.isfinite(float(started))
        or not np.isfinite(float(completed))
        or float(completed) < float(started)
    ):
        raise RuntimeError("SMPL-X pool gate timestamps are invalid")

    topology = _smplx_training_pool_topology(mode, helpers)
    return {
        "format": gate_format,
        "status": "pass",
        "authorization": True,
        "path": str(report_path),
        "sha256": report_sha,
        "formal_stage": args.formal_stage,
        "mode": mode,
        "helper_devices": helper_list,
        "topology": topology,
        "topology_sha256": _payload_sha256(topology),
        "source_binding": {
            key: current_source[key] for key in ("origin", "commit", "tree")
        },
        "gate_artifacts": {
            "harness_sha256": gate_source["harness_sha256"],
            "adapter_sha256": gate_source["adapter_sha256"],
            "coordinator_sha256": coordinator_sha,
        },
        "representation": {
            key: expected_representation[key]
            for key in (
                "data_sha256",
                "summary_sha256",
                "lineage_sha256",
                "entry_aggregate_sha256",
            )
        },
        "smplx_asset_sha256": smplx_asset_receipt["sha256"],
        "semantic_common_sha256": _payload_sha256(semantic_common),
        "runtime": {
            "visible_device_count": visible_device_count,
            "device_names": runtime["device_names"],
            "torch": runtime["torch"],
            "cuda": runtime["cuda"],
            "cudnn": runtime["cudnn"],
        },
        "equivalence": {
            "proofs": sorted(expected_proofs),
            "repeat_byte_exact": True,
            "cross_mode": spec["receipt_cross_mode"],
            **(
                {
                    "bounded_tolerances": copy.deepcopy(
                        SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES
                    )
                }
                if mode == "sharded_local_loss"
                else {}
            ),
        },
        "performance": {
            "minimum_speedup": minimum,
            **performance_receipt,
        },
        "children": {
            "count": 8,
            "fresh_processes": 8,
            "result_sha256": result_shas,
        },
    }


def _formal_lower_target_cache_gate_receipt(
    args: Any,
    *,
    lower_target_cache_receipt: dict[str, Any] | None,
    current_source: dict[str, str],
) -> dict[str, Any] | None:
    """Require and bind the lossless/performance gate before lower training."""

    report_value = getattr(args, "lower_target_cache_gate_report", None)
    expected_value = getattr(
        args,
        "expected_lower_target_cache_gate_sha256",
        None,
    )
    builder_process_value = getattr(
        args,
        "lower_target_cache_builder_process_receipt",
        None,
    )
    expected_builder_process_value = getattr(
        args,
        "expected_lower_target_cache_builder_process_receipt_sha256",
        None,
    )
    cache_enabled = lower_target_cache_receipt is not None
    if not cache_enabled:
        if any(
            value not in {None, ""}
            for value in (
                report_value,
                expected_value,
                builder_process_value,
                expected_builder_process_value,
            )
        ):
            raise RuntimeError(
                "lower target cache gate arguments are forbidden when the "
                "cache is disabled"
            )
        return None
    if args.formal_stage != "lower":
        raise RuntimeError("lower target cache gate is restricted to lower")
    if (
        type(report_value) is not str
        or not report_value
        or type(expected_value) is not str
        or len(expected_value) != 64
        or any(character not in "0123456789abcdef" for character in expected_value)
        or type(builder_process_value) is not str
        or not builder_process_value
        or type(expected_builder_process_value) is not str
        or len(expected_builder_process_value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_builder_process_value
        )
    ):
        raise RuntimeError(
            "formal lower training requires an explicit gate report and SHA-256"
        )
    builder_process_input = Path(builder_process_value)
    if (
        builder_process_input.is_symlink()
        or not builder_process_input.is_file()
    ):
        raise RuntimeError(
            "lower target cache builder process receipt must be a regular file"
        )
    builder_process_path = builder_process_input.resolve()
    builder_process_sha = _sha256(builder_process_path)
    if builder_process_sha != expected_builder_process_value:
        raise RuntimeError("lower target cache builder process SHA mismatch")
    with builder_process_path.open(encoding="utf-8") as handle:
        builder_process_payload = json.load(handle)
    if not isinstance(builder_process_payload, dict):
        raise RuntimeError(
            "lower target cache builder process receipt must be an object"
        )
    report_input = Path(report_value)
    if report_input.is_symlink() or not report_input.is_file():
        raise RuntimeError(
            f"lower target cache gate must be a regular file: {report_input}"
        )
    report_path = report_input.resolve()
    report_sha = _sha256(report_path)
    if report_sha != expected_value:
        raise RuntimeError("lower target cache gate report SHA mismatch")
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise RuntimeError("lower target cache gate report must be an object")
    expected_speakers = {
        "oliver": 0,
        "chemistry": 1,
        "seth": 2,
        "conan": 3,
    }
    input_source_binding = None
    scope = report.get("scope")
    protocol = report.get("protocol")
    equivalence = report.get("equivalence")
    performance = report.get("performance")
    preflight = report.get("preflight")
    child_commands = report.get("child_commands")
    if (
        report.get("format") != LOWER_TARGET_CACHE_GATE_FORMAT
        or report.get("status") != "pass"
        or not isinstance(scope, dict)
        or scope.get("dataset") != "show_base"
        or scope.get("formal_stage") != "lower"
        or scope.get("speaker_scope") != "All"
        or scope.get("speaker_ids") != [0, 1, 2, 3]
        or _require_exact_audit_int_mapping(
            scope.get("speaker_map"),
            expected_speakers,
            "lower cache gate speaker_map",
        )
        != expected_speakers
        or not isinstance(protocol, dict)
        or protocol.get("speaker_scope") != "All"
        or _require_exact_audit_int_mapping(
            protocol.get("speakers"),
            expected_speakers,
            "lower cache gate protocol speakers",
        )
        != expected_speakers
        or _require_exact_audit_int(
            protocol.get("equivalence_updates"),
            "lower cache gate equivalence_updates",
        )
        != 2
        or _require_exact_audit_int(
            protocol.get("equivalence_fresh_processes"),
            "lower cache gate equivalence fresh processes",
        )
        != 4
        or _require_exact_audit_int(
            protocol.get("equivalence_repeats_per_mode"),
            "lower cache gate equivalence repeats",
        )
        != 2
        or protocol.get("abba_order")
        != ["legacy", "cache", "cache", "legacy"]
        or _require_exact_audit_int(
            protocol.get("benchmark_fresh_processes"),
            "lower cache gate benchmark fresh processes",
        )
        != 4
        or _require_exact_audit_int(
            protocol.get("warmup_updates_per_block"),
            "lower cache gate warmup updates",
        )
        != 5
        or _require_exact_audit_int(
            protocol.get("measured_updates_per_block"),
            "lower cache gate measured updates",
        )
        != 25
        or float(protocol.get("minimum_speedup", float("nan")))
        != LOWER_TARGET_CACHE_GATE_MIN_SPEEDUP
        or _require_exact_audit_int(
            protocol.get("training_updates"),
            "lower cache gate training updates",
        )
        != 600 * 1_988
        or not isinstance(equivalence, dict)
        or equivalence.get("status") != "pass"
        or equivalence.get(
            "initial_and_two_complete_updates_and_final_byte_exact"
        )
        is not True
        or equivalence.get("repeat_controls_exact") is not True
        or equivalence.get("semantic_receipts_exact") is not True
        or _require_exact_audit_int(
            equivalence.get("fresh_processes"),
            "lower cache gate equivalence fresh processes report",
        )
        != 4
        or set(equivalence.get("states", {}))
        != {"initial", "step_1", "step_2", "final"}
        or len(equivalence.get("target_joints", [])) != 2
        or any(
            item.get("byte_exact") is not True
            for item in equivalence.get("target_joints", [])
        )
        or not isinstance(performance, dict)
        or performance.get("status") != "pass"
        or _require_exact_audit_int(
            performance.get("fresh_processes"),
            "lower cache gate benchmark fresh processes report",
        )
        != 4
        or performance.get("semantic_receipts_exact") is not True
        or performance.get("real_batch_receipts_exact") is not True
        or performance.get("initial_whole_state_exact") is not True
        or not isinstance(performance.get("initial_state_sha256"), str)
        or len(performance["initial_state_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in performance["initial_state_sha256"]
        )
        or not isinstance(preflight, dict)
        or not isinstance(child_commands, list)
        or len(child_commands) != 8
    ):
        raise RuntimeError("lower target cache gate protocol is incomplete")
    repeat_controls = equivalence.get("repeat_controls")
    cross_mode_pairs = equivalence.get("cross_mode_pairs")
    if (
        not isinstance(repeat_controls, dict)
        or set(repeat_controls) != {"legacy_a_a", "cache_b_b"}
        or not isinstance(cross_mode_pairs, list)
        or len(cross_mode_pairs) != 2
    ):
        raise RuntimeError("lower target cache repeat controls are incomplete")
    for label, proof in [
        *repeat_controls.items(),
        *[
            (f"cross_mode_{index}", proof)
            for index, proof in enumerate(cross_mode_pairs)
        ],
    ]:
        if (
            not isinstance(proof, dict)
            or proof.get("status") != "pass"
            or proof.get("semantic_receipts_exact") is not True
            or proof.get(
                "initial_and_two_complete_updates_and_final_byte_exact"
            )
            is not True
            or set(proof.get("states", {}))
            != {"initial", "step_1", "step_2", "final"}
            or len(proof.get("target_joints", [])) != 2
            or any(
                item.get("byte_exact") is not True
                for item in proof.get("target_joints", [])
            )
        ):
            raise RuntimeError(
                f"lower target cache equivalence proof {label} is incomplete"
            )
    expected_child_modes = [
        "equivalence-legacy-0",
        "equivalence-legacy-1",
        "equivalence-cache-0",
        "equivalence-cache-1",
        "benchmark-0-legacy",
        "benchmark-1-cache",
        "benchmark-2-cache",
        "benchmark-3-legacy",
    ]
    for index, (command, expected_mode) in enumerate(
        zip(child_commands, expected_child_modes)
    ):
        argv = command.get("argv") if isinstance(command, dict) else None
        if (
            not isinstance(command, dict)
            or command.get("sequence_index") != index
            or command.get("mode") != expected_mode
            or command.get("return_code") != 0
            or command.get("exact_argv_and_rc") is not True
            or command.get("cuda_visible_devices") != "0"
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) for item in argv)
            or command.get("argv_sha256")
            != hashlib.sha256(
                b"\0".join(os.fsencode(item) for item in argv) + b"\0"
            ).hexdigest()
            or command.get("reported_argv_sha256")
            != command.get("argv_sha256")
        ):
            raise RuntimeError(
                f"lower target cache child command {index} is invalid"
            )

    def require_speedup(item: Any, label: str) -> float:
        if (
            not isinstance(item, dict)
            or item.get("pass") is not True
            or float(item.get("threshold", float("nan")))
            != LOWER_TARGET_CACHE_GATE_MIN_SPEEDUP
        ):
            raise RuntimeError(f"{label} gate receipt is incomplete")
        value = float(item.get("speedup", float("nan")))
        if (
            not np.isfinite(value)
            or value < LOWER_TARGET_CACHE_GATE_MIN_SPEEDUP
        ):
            raise RuntimeError(f"{label} speedup is below 1.05")
        return value

    pooled = performance.get("pooled")
    pairs = performance.get("block_pairs")
    amortization = performance.get("amortization")
    if (
        not isinstance(pooled, dict)
        or not isinstance(pairs, list)
        or len(pairs) != 2
        or not isinstance(amortization, dict)
    ):
        raise RuntimeError("lower target cache benchmark evidence is incomplete")
    pooled_wall = require_speedup(pooled.get("wall"), "pooled wall")
    pooled_cuda = require_speedup(
        pooled.get("cuda_event"),
        "pooled CUDA-event",
    )
    for pair_index, pair in enumerate(pairs):
        if (
            not isinstance(pair, dict)
            or _require_exact_audit_int(
                pair.get("pair_index"),
                f"lower cache gate pair {pair_index} index",
            )
            != pair_index
        ):
            raise RuntimeError("lower target cache block pair is invalid")
        require_speedup(pair.get("wall"), f"pair {pair_index} wall")
        require_speedup(
            pair.get("cuda_event"),
            f"pair {pair_index} CUDA-event",
        )
    amortized = require_speedup(amortization, "builder-amortized wall")
    if (
        _require_exact_audit_int(
            amortization.get("epochs"),
            "lower cache gate amortization epochs",
        )
        != 600
        or _require_exact_audit_int(
            amortization.get("updates_per_epoch"),
            "lower cache gate amortization updates_per_epoch",
        )
        != 1_988
        or _require_exact_audit_int(
            amortization.get("training_updates"),
            "lower cache gate amortization training_updates",
        )
        != 600 * 1_988
        or not np.isfinite(
            float(amortization.get("full_builder_seconds", float("nan")))
        )
        or float(amortization["full_builder_seconds"]) <= 0.0
    ):
        raise RuntimeError("lower target cache amortization receipt is invalid")

    cache_manifest = preflight.get("cache_manifest")
    checker = preflight.get("checker_receipt")
    representation = preflight.get("representation")
    smplx = preflight.get("smplx")
    gate_source = preflight.get("source")
    builder_process = preflight.get("builder_process_receipt")
    builder_timing = preflight.get("builder_timing")
    current_inputs = lower_target_cache_receipt.get("current_inputs")
    embedded_builder_payload = (
        builder_process.get("payload")
        if isinstance(builder_process, dict)
        else None
    )
    builder_validation = (
        builder_process.get("validation")
        if isinstance(builder_process, dict)
        else None
    )
    if (
        not all(
            isinstance(item, dict)
            for item in (
                cache_manifest,
                checker,
                representation,
                smplx,
                gate_source,
                builder_process,
                builder_timing,
                current_inputs,
            )
        )
        or cache_manifest.get("sha256")
        != lower_target_cache_receipt.get("manifest_sha256")
        or checker.get("sha256")
        != lower_target_cache_receipt.get("checker_receipt_sha256")
        or builder_process.get("path") != str(builder_process_path)
        or builder_process.get("sha256") != builder_process_sha
        or not isinstance(embedded_builder_payload, dict)
        or embedded_builder_payload != builder_process_payload
        or embedded_builder_payload.get("format")
        != "semtalk_show_lower_target_cache_builder_process_v1"
        or embedded_builder_payload.get("status") != "complete"
        or not isinstance(builder_validation, dict)
        or builder_validation.get("return_code") != 0
        or builder_timing.get("source")
        != "external_full_child_process_receipt"
        or builder_timing.get("receipt_sha256") != builder_process_sha
        or performance.get("builder_process_receipt_sha256")
        != builder_process_sha
        or float(builder_timing.get("full_builder_seconds", float("nan")))
        != float(amortization["full_builder_seconds"])
        or representation.get("data_mdb_sha256")
        != current_inputs.get("data_mdb_sha256")
        or representation.get("summary_sha256")
        != current_inputs.get("representation_summary_sha256")
        or representation.get("lineage_sha256")
        != current_inputs.get("representation_lineage_sha256")
        or smplx.get("asset_sha256")
        != current_inputs.get("smplx_asset_sha256")
        or {
            key: gate_source.get(key)
            for key in ("origin", "commit", "tree")
        }
        != {
            key: current_source.get(key)
            for key in ("origin", "commit", "tree")
        }
    ):
        raise RuntimeError(
            "lower target cache gate is not bound to current cache/data/source"
        )
    gate_entrypoint = Path(str(gate_source.get("entrypoint", "")))
    expected_entrypoint = (
        Path(__file__).resolve().parent
        / "scripts"
        / "show_base"
        / "run_lower_target_cache_formal_gate.py"
    )
    if (
        gate_entrypoint.is_symlink()
        or not gate_entrypoint.is_file()
        or gate_entrypoint.resolve() != expected_entrypoint
        or _sha256(gate_entrypoint.resolve())
        != gate_source.get("entrypoint_sha256")
    ):
        raise RuntimeError("lower target cache gate source entrypoint changed")
    return {
        "format": LOWER_TARGET_CACHE_GATE_FORMAT,
        "status": "pass",
        "path": str(report_path),
        "sha256": report_sha,
        "pooled_wall_speedup": pooled_wall,
        "pooled_cuda_event_speedup": pooled_cuda,
        "builder_amortized_wall_speedup": amortized,
        "builder_process_receipt_path": str(builder_process_path),
        "builder_process_receipt_sha256": builder_process_sha,
        "source_binding": {
            key: current_source[key] for key in ("origin", "commit", "tree")
        },
        "cache_manifest_sha256": lower_target_cache_receipt[
            "manifest_sha256"
        ],
        "cache_checker_sha256": lower_target_cache_receipt[
            "checker_receipt_sha256"
        ],
    }


def _dataset_receipt(
    args: Any,
    *,
    train_samples: int,
    current_source: dict[str, str],
    lower_target_cache_receipt: dict[str, Any] | None = None,
    lower_target_backend_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    smplx_asset_receipt = _formal_smplx_asset_receipt(args)
    if not args.dataset_summary:
        raise RuntimeError("formal training requires --dataset_summary")
    summary_input = Path(args.dataset_summary)
    lineage_input = Path(args.lineage_manifest)
    if summary_input.is_symlink() or lineage_input.is_symlink():
        raise RuntimeError("dataset summary and lineage must be regular files")
    summary_path = summary_input.resolve()
    lineage_path = lineage_input.resolve()
    if not summary_path.is_file() or not lineage_path.is_file():
        raise FileNotFoundError(
            f"missing dataset receipt: {summary_path} / {lineage_path}"
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with lineage_path.open(encoding="utf-8") as handle:
        lineage = json.load(handle)
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise RuntimeError(f"{summary_path}: dataset summary is not complete")
    if not isinstance(lineage, dict) or lineage.get("status") != "complete":
        raise RuntimeError(f"{lineage_path}: dataset lineage is not complete")

    lmdb_path = Path(args.train_path).resolve()
    data_path = lmdb_path / "data.mdb"
    if not lmdb_path.is_dir() or data_path.is_symlink() or not data_path.is_file():
        raise RuntimeError(f"invalid formal training LMDB: {lmdb_path}")
    data_sha = _sha256(data_path)
    if (
        Path(summary.get("lmdb", "")).resolve() != lmdb_path
        or summary.get("data_mdb_sha256") != data_sha
        or _require_exact_audit_int(
            summary.get("entries"),
            "dataset summary entries",
        )
        != train_samples
        or train_samples != 127_286
        or _require_exact_audit_int(
            summary.get("train_clips"),
            "dataset summary train_clips",
        )
        != 13687
    ):
        raise RuntimeError(
            f"{summary_path}: LMDB/sample/frozen-train receipt mismatch"
        )

    expected_speakers = {
        "oliver": 0,
        "chemistry": 1,
        "seth": 2,
        "conan": 3,
    }
    if args.formal_stage == "base":
        protocol = lineage.get("protocol")
        expected_forbidden = {
            "ASR",
            "TextGrid",
            "vocabulary",
            "CLIP",
            "emotion",
            "semantic",
            "SemGate",
            "Sparse",
        }
        checkpoints = lineage.get("formal_checkpoints")
        if (
            summary.get("format") != "semtalk_show_base_lmdb_summary_v1"
            or lineage.get("format")
            != "semtalk_show_base_feature_lineage_v1"
            or Path(summary.get("lineage_json", "")).resolve() != lineage_path
            or summary.get("lineage_json_sha256") != _sha256(lineage_path)
            or _require_exact_audit_int(
                lineage.get("entries"),
                "Base lineage entries",
            )
            != train_samples
            or _require_exact_audit_int(
                lineage.get("train_clips"),
                "Base lineage train_clips",
            )
            != 13687
            or not isinstance(protocol, dict)
            or protocol.get("scope") != "SemTalk Base only"
            or protocol.get("split") != "train"
            or _require_exact_audit_int_mapping(
                protocol.get("speakers"),
                expected_speakers,
                "Base protocol speakers",
            )
            != expected_speakers
            or _require_exact_audit_int(
                protocol.get("window_length"),
                "Base protocol window_length",
            )
            != 64
            or _require_exact_audit_int(
                protocol.get("stride"),
                "Base protocol stride",
            )
            != 20
            or protocol.get("in_word")
            != "int64_all_zero_unused_placeholder"
            or set(protocol.get("forbidden_components", []))
            != expected_forbidden
            or not isinstance(checkpoints, dict)
            or set(checkpoints) != {
                "face",
                "hands",
                "upper",
                "lower",
                "global",
            }
            or any(
                not isinstance(record, dict)
                or record.get("formal_stage") != stage
                for stage, record in checkpoints.items()
            )
            or {
                key: lineage.get("source_receipt", {}).get(key)
                for key in ("origin", "commit", "tree")
            }
            != {
                key: current_source.get(key)
                for key in ("origin", "commit", "tree")
            }
        ):
            raise RuntimeError(
                "Base dataset summary/lineage are not mutually bound"
            )
    else:
        representation_format = summary.get("format")
        representation_protocol = summary.get("protocol")
        representation_source = summary.get("source_receipt")
        if (
            representation_format
            != "semtalk_show_representation_lmdb_v2_global_foot"
        ):
            raise RuntimeError(
                "formal prerequisites require the Global-foot v2 receipt"
            )
        if (
            not isinstance(representation_protocol, dict)
            or representation_protocol.get("split") != "train"
            or _require_exact_audit_int(
                representation_protocol.get("window_length"),
                "representation protocol window_length",
            )
            != 64
            or _require_exact_audit_int(
                representation_protocol.get("stride"),
                "representation protocol stride",
            )
            != 20
            or _require_exact_audit_int_mapping(
                representation_protocol.get("speaker_map"),
                expected_speakers,
                "representation protocol speaker_map",
            )
            != expected_speakers
        ):
            raise RuntimeError(
                "invalid four-speaker representation training protocol"
            )
        if summary.get("protocol", {}).get(
            "global_foot_fastpath"
        ) != {
            "enabled": True,
            "contract": "semtalk_show_global_foot_fastpath_v1",
            "field": "lower_foot_local",
            "shape": [64, 4, 3],
            "dtype": "float32",
            "activation_env": (
                "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH=1"
            ),
        }:
            raise RuntimeError(
                "invalid Global-foot fastpath representation receipt"
            )
        if (
            not isinstance(representation_source, dict)
            or representation_source.get("origin")
            != "git@github.com:Xiangyue-Zhang/SemTalk.git"
            or any(
                not isinstance(representation_source.get(key), str)
                or len(representation_source[key]) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in representation_source[key]
                )
                for key in ("commit", "tree")
            )
        ):
            raise RuntimeError(
                "invalid representation producer source receipt"
            )
        input_source_binding = {
            key: representation_source[key]
            for key in ("origin", "commit", "tree")
        }
        expected_fastpath = (
            representation_format.endswith("_v2_global_foot")
            and args.formal_stage == "global"
        )
        observed_fastpath = os.environ.get(
            "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH",
            "0",
        )
        if observed_fastpath != ("1" if expected_fastpath else "0"):
            raise RuntimeError(
                "Global-foot fastpath environment does not match the "
                "representation receipt and formal stage"
            )
        if (
            summary_path != lineage_path
            or lineage != summary
        ):
            raise RuntimeError(
                "representation training must bind the exact LMDB summary "
                "as its lineage manifest"
            )
    parity_receipt = None
    if args.formal_stage == "global":
        if (
            not args.global_fastpath_parity_bundle
            or not args.expected_global_fastpath_parity_sha256
        ):
            raise RuntimeError("Global training requires a parity bundle")
        parity_input = Path(args.global_fastpath_parity_bundle)
        if parity_input.is_symlink() or not parity_input.is_file():
            raise FileNotFoundError(parity_input)
        parity_path = parity_input.resolve()
        parity_sha = _sha256(parity_path)
        expected_parity_sha = (
            args.expected_global_fastpath_parity_sha256.strip().lower()
        )
        if (
            len(expected_parity_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_parity_sha
            )
        ):
            raise RuntimeError("Global parity expected SHA is not lowercase SHA-256")
        if parity_sha != expected_parity_sha:
            raise RuntimeError("Global parity bundle SHA mismatch")
        with parity_path.open(encoding="utf-8") as handle:
            parity = json.load(handle)
        if (
            parity.get("format")
            != "semtalk_show_global_foot_parity_suite_v1"
            or parity.get("status") != "pass"
            or parity.get("contract")
            != "semtalk_show_global_foot_fastpath_v1"
            or _require_exact_audit_int_mapping(
                parity.get("speakers"),
                expected_speakers,
                "Global parity speakers",
            )
            != expected_speakers
            or parity.get("canonical_receipt")
            != summary.get("canonical_receipt")
            or {
                key: parity.get("source_receipt", {}).get(key)
                for key in ("origin", "commit", "tree")
            }
            != input_source_binding
        ):
            raise RuntimeError("invalid Global parity bundle")
        checker_path = (
            Path(__file__).resolve().parent
            / "scripts"
            / "show_base"
            / "check_global_foot_fastpath_parity.py"
        )
        if (
            checker_path.is_symlink()
            or not checker_path.is_file()
            or parity.get("checker_sha256") != _sha256(checker_path)
        ):
            raise RuntimeError("Global parity checker source mismatch")
        smplx_asset = (
            Path(args.data_path_1).resolve()
            / "smplx_models"
            / "smplx"
            / "SMPLX_NEUTRAL_2020.npz"
        )
        if smplx_asset.is_symlink() or not smplx_asset.is_file():
            raise FileNotFoundError(smplx_asset)
        smplx_sha = _sha256(smplx_asset)
        if parity.get("smplx_asset_sha256") != smplx_sha:
            raise RuntimeError("Global parity/SMPL-X asset SHA mismatch")
        reports = parity.get("reports")
        report_speakers = set()
        if isinstance(reports, list):
            for record in reports:
                if not isinstance(record, dict):
                    break
                speaker = record.get("speaker")
                speaker_id = _require_exact_audit_int(
                    record.get("speaker_id"),
                    "Global parity report speaker_id",
                )
                report_speakers.add((speaker, speaker_id))
        if (
            not isinstance(reports, list)
            or len(reports) != 4
            or report_speakers != set(expected_speakers.items())
        ):
            raise RuntimeError(
                "Global parity suite does not cover all four SHOW speakers"
            )
        canonical_receipt = summary.get("canonical_receipt")
        if not isinstance(canonical_receipt, dict):
            raise RuntimeError("Global parity lacks a canonical receipt")
        canonical_input = Path(str(canonical_receipt.get("manifest", "")))
        if canonical_input.is_symlink() or not canonical_input.is_file():
            raise FileNotFoundError(canonical_input)
        canonical_manifest = canonical_input.resolve()
        if _sha256(canonical_manifest) != canonical_receipt.get(
            "manifest_sha256"
        ):
            raise RuntimeError("Global parity canonical manifest SHA mismatch")
        selected_rows: dict[str, dict[str, Any]] = {}
        with canonical_manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"{canonical_manifest}:{line_number}: non-object row"
                    )
                speaker = str(row.get("speaker"))
                speaker_id = _require_exact_audit_int(
                    row.get("speaker_id"),
                    f"{canonical_manifest}:{line_number}: speaker_id",
                )
                frames = _require_exact_audit_int(
                    row.get("frames"),
                    f"{canonical_manifest}:{line_number}: frames",
                )
                if (
                    row.get("split") == "train"
                    and speaker in expected_speakers
                    and speaker_id == expected_speakers[speaker]
                    and frames >= 64
                    and speaker not in selected_rows
                ):
                    selected_rows[speaker] = row
        if set(selected_rows) != set(expected_speakers):
            raise RuntimeError(
                "Global parity canonical manifest lacks four speaker windows"
            )
        expected_thresholds = {
            "contact": 0.95,
            "loss_atol": 2e-6,
            "loss_rtol": 1e-5,
            "gradient_atol": 2e-6,
            "gradient_rtol": 1e-5,
            "cache_atol": 1e-6,
            "cache_rtol": 1e-6,
            "wrong_axis_zero_atol": 2e-7,
        }
        expected_optimizer = {
            "name": "Adam",
            "lr": 1.5e-4,
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "steps": 1,
        }
        expected_loss_checks = {
            "contact",
            "vertex",
            "vertex_velocity_wrong_axis",
            "vertex_acceleration_wrong_axis",
            "foot",
            "total",
        }
        expected_check_keys = {
            "cache",
            "losses",
            "gradient",
            "adam_parameter",
            "adam_exp_avg",
            "adam_exp_avg_sq",
        }

        def validate_close(
            value: Any,
            *,
            atol: float,
            rtol: float,
            label: str,
        ) -> None:
            if not isinstance(value, dict) or set(value) != {
                "max_abs",
                "max_rel",
            }:
                raise RuntimeError(f"{label}: invalid close-check payload")
            max_abs = float(value["max_abs"])
            max_rel = float(value["max_rel"])
            if (
                not np.isfinite(max_abs)
                or not np.isfinite(max_rel)
                or max_abs < 0
                or max_rel < 0
                or (max_abs > atol and max_rel > rtol)
            ):
                raise RuntimeError(
                    f"{label}: parity delta exceeds its recorded tolerance"
                )

        report_receipts = []
        observed_report_paths: set[Path] = set()
        for record in reports:
            assert isinstance(record, dict)
            report_input = Path(str(record.get("report", "")))
            if report_input.is_symlink() or not report_input.is_file():
                raise FileNotFoundError(report_input)
            report_path = report_input.resolve()
            report_sha = _sha256(report_path)
            with report_path.open(encoding="utf-8") as handle:
                report_payload = json.load(handle)
            speaker = str(record["speaker"])
            selected_row = selected_rows[speaker]
            checks = report_payload.get("checks")
            if (
                report_path in observed_report_paths
                or report_sha != record.get("report_sha256")
                or report_payload != record.get("payload")
                or record.get("clip_id") != selected_row.get("clip_id")
                or report_payload.get("status") != "pass"
                or report_payload.get("contract")
                != "semtalk_show_global_foot_fastpath_v1"
                or Path(
                    str(report_payload.get("canonical_npz", ""))
                ).resolve()
                != Path(str(selected_row.get("canonical_npz", ""))).resolve()
                or report_payload.get("canonical_npz_sha256")
                != selected_row.get("canonical_npz_sha256")
                or Path(
                    str(report_payload.get("lower_foot_local", ""))
                ).resolve()
                != Path(
                    str(selected_row.get("lower_foot_local", ""))
                ).resolve()
                or report_payload.get("lower_foot_local_sha256")
                != selected_row.get("lower_foot_local_sha256")
                or report_payload.get("smplx_asset_sha256") != smplx_sha
                or report_payload.get("window")
                != {"start_frame": 0, "frames": 64}
                or report_payload.get("thresholds") != expected_thresholds
                or report_payload.get("optimizer") != expected_optimizer
                or not isinstance(checks, dict)
                or set(checks) != expected_check_keys
                or not isinstance(checks.get("losses"), dict)
                or set(checks["losses"]) != expected_loss_checks
            ):
                raise RuntimeError(
                    f"invalid Global parity report: {report_path}"
                )
            observed_report_paths.add(report_path)
            wrong_axis = float(
                report_payload.get("legacy_wrong_axis_max_loss", float("nan"))
            )
            if (
                not np.isfinite(wrong_axis)
                or wrong_axis < 0
                or wrong_axis > expected_thresholds["wrong_axis_zero_atol"]
            ):
                raise RuntimeError(
                    f"{report_path}: invalid wrong-axis parity loss"
                )
            validate_close(
                checks["cache"],
                atol=expected_thresholds["cache_atol"],
                rtol=expected_thresholds["cache_rtol"],
                label=f"{report_path}:cache",
            )
            for name, value in checks["losses"].items():
                validate_close(
                    value,
                    atol=expected_thresholds["loss_atol"],
                    rtol=expected_thresholds["loss_rtol"],
                    label=f"{report_path}:losses.{name}",
                )
            for name in (
                "gradient",
                "adam_parameter",
                "adam_exp_avg",
                "adam_exp_avg_sq",
            ):
                validate_close(
                    checks[name],
                    atol=expected_thresholds["gradient_atol"],
                    rtol=expected_thresholds["gradient_rtol"],
                    label=f"{report_path}:{name}",
                )
            report_receipts.append(
                {
                    "speaker": record["speaker"],
                    "speaker_id": record["speaker_id"],
                    "path": str(report_path),
                    "sha256": report_sha,
                }
            )
        parity_receipt = {
            "path": str(parity_path),
            "sha256": parity_sha,
            "format": parity["format"],
            "status": parity["status"],
            "speakers": parity["speakers"],
            "checker": str(checker_path),
            "checker_sha256": parity["checker_sha256"],
            "smplx_asset": str(smplx_asset),
            "smplx_asset_sha256": smplx_sha,
            "reports": report_receipts,
        }
    elif (
        args.global_fastpath_parity_bundle is not None
        or args.expected_global_fastpath_parity_sha256 is not None
    ):
        raise RuntimeError(
            "Global parity arguments are forbidden outside the global stage"
        )
    cache_enabled = validate_lower_target_cache_activation(args)
    if cache_enabled != (lower_target_cache_receipt is not None):
        raise RuntimeError(
            "formal lower target cache activation and trainer receipt disagree"
        )
    if (
        lower_target_cache_receipt is not None
        and lower_target_backend_receipt is not None
    ):
        raise RuntimeError(
            "lower target cache and live backend receipts are mutually exclusive"
        )
    if cache_enabled:
        assert lower_target_cache_receipt is not None
        cache_source = lower_target_cache_receipt.get("source_receipt")
        current_inputs = lower_target_cache_receipt.get("current_inputs")
        if (
            args.formal_stage != "lower"
            or not isinstance(cache_source, dict)
            or not isinstance(current_inputs, dict)
            or {
                key: cache_source.get(key)
                for key in ("origin", "commit", "tree")
            }
            != {
                key: current_source.get(key)
                for key in ("origin", "commit", "tree")
            }
            or current_inputs.get("representation_lmdb_path")
            != str(lmdb_path)
            or current_inputs.get("data_mdb_sha256") != data_sha
            or current_inputs.get("representation_summary_path")
            != str(summary_path)
            or current_inputs.get("representation_lineage_path")
            != str(lineage_path)
            or current_inputs.get("representation_summary_sha256")
            != _sha256(summary_path)
            or current_inputs.get("representation_lineage_sha256")
            != _sha256(lineage_path)
            or current_inputs.get("smplx_asset_sha256")
            != smplx_asset_receipt["sha256"]
        ):
            raise RuntimeError(
                "formal lower target cache receipt is not bound to the "
                "current dataset/source/SMPL-X inputs"
            )
    _validate_lower_target_backend_receipt(
        lower_target_backend_receipt,
        formal_stage=args.formal_stage,
        current_source=current_source,
        smplx_asset_receipt=smplx_asset_receipt,
    )
    smplx_training_pool_gate_receipt = (
        _formal_smplx_training_pool_gate_receipt(
            args,
            current_source=current_source,
            representation={
                "lmdb": str(lmdb_path),
                "data_sha256": data_sha,
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "lineage": str(lineage_path),
                "lineage_sha256": _sha256(lineage_path),
                "entry_aggregate_sha256": summary.get(
                    "entry_aggregate_sha256"
                ),
            },
            smplx_asset_receipt=smplx_asset_receipt,
        )
    )
    receipt = {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "lineage": str(lineage_path),
        "lineage_sha256": _sha256(lineage_path),
        "lmdb": str(lmdb_path),
        "data_mdb_sha256": data_sha,
        "entries": train_samples,
        "train_clips": 13687,
        "split_label": "SHOW available frozen subset",
        "source_binding": {
            key: current_source[key] for key in ("origin", "commit", "tree")
        },
        "smplx_asset": smplx_asset_receipt,
        "global_fastpath_parity": parity_receipt,
    }
    if input_source_binding is not None:
        receipt["input_source_binding"] = input_source_binding
    _attach_smplx_training_pool_gate_receipt(
        receipt,
        smplx_training_pool_gate_receipt,
    )
    attach_lower_target_cache_receipt(
        receipt,
        lower_target_cache_receipt,
    )
    _attach_lower_target_backend_receipt(
        receipt,
        lower_target_backend_receipt,
    )
    return receipt


def _finite_tree(value: Any, prefix: str) -> list[str]:
    bad: list[str] = []
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            bad.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            bad.extend(_finite_tree(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            bad.extend(_finite_tree(child, f"{prefix}[{index}]"))
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        bad.append(prefix)
    return bad


def _local_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def _all_rng_states(world_size: int) -> list[dict[str, Any]] | None:
    local = _local_rng_state()
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [state for state in gathered if state is not None]


def _tracker_snapshot(trainer: Any) -> dict[str, dict[str, float | int]]:
    trainer._flush_train_metrics()
    snapshot: dict[str, dict[str, float | int]] = {}
    for name, states in trainer.tracker.loss_meters.items():
        meter = states["train"]
        if meter.count:
            snapshot[name] = {
                "avg": float(meter.avg),
                "count": int(meter.count),
            }
    return snapshot


def _trainer_distributed_receipt(
    trainer: Any,
    *,
    world_size: int,
) -> dict[str, Any]:
    return representation_ddp_receipt(
        formal_stage=trainer.args.formal_stage,
        world_size=world_size,
        local_batch_size=int(trainer.args.batch_size),
        train_samples=len(trainer.train_data),
        updates_per_epoch=trainer.train_length,
        seed=int(trainer.args.random_seed),
    )


def _rvq_ema_state(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    """Capture legacy EMA state omitted by QuantizeEMAReset.state_dict()."""
    state: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "QuantizeEMAReset":
            continue
        initialized = bool(module.init)
        item: dict[str, Any] = {"init": initialized}
        if initialized:
            if module.code_sum is None or module.code_count is None:
                raise RuntimeError(f"initialized RVQ EMA module {name} has empty state")
            item["code_sum"] = module.code_sum.detach().cpu()
            item["code_count"] = module.code_count.detach().cpu()
        state[name] = item
    return state


def _restore_rvq_ema_state(
    model: torch.nn.Module,
    state: dict[str, dict[str, Any]],
) -> None:
    modules = {
        name: module
        for name, module in model.named_modules()
        if module.__class__.__name__ == "QuantizeEMAReset"
    }
    if set(modules) != set(state):
        raise RuntimeError(
            "resume RVQ EMA module set mismatch: "
            f"model={sorted(modules)} checkpoint={sorted(state)}"
        )
    for name, module in modules.items():
        item = state[name]
        if not isinstance(item, dict) or type(item.get("init")) is not bool:
            raise RuntimeError(
                f"resume RVQ EMA module {name} init must be an exact boolean"
            )
        initialized = item["init"]
        expected_keys = (
            {"init", "code_sum", "code_count"}
            if initialized
            else {"init"}
        )
        if set(item) != expected_keys:
            raise RuntimeError(
                f"resume RVQ EMA module {name} has invalid state keys"
            )
        module.init = initialized
        if initialized:
            module.code_sum = item["code_sum"].to(
                device=module.codebook.device,
                dtype=module.codebook.dtype,
            )
            module.code_count = item["code_count"].to(
                device=module.codebook.device,
                dtype=module.codebook.dtype,
            )
        else:
            module.code_sum = None
            module.code_count = None


def _rvq_ema_invariant_errors(model: torch.nn.Module) -> list[str]:
    errors: list[str] = []
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "QuantizeEMAReset"
    ]
    if modules and len(modules) != 6:
        errors.append(f"expected 6 RVQ EMA layers, found {len(modules)}")
    for name, module in modules:
        if not bool(module.init):
            errors.append(f"{name}.init is false")
            continue
        if module.code_sum is None or module.code_count is None:
            errors.append(f"{name} has empty EMA accumulators")
            continue
        if module.code_sum.shape != module.codebook.shape:
            errors.append(f"{name}.code_sum shape mismatch")
        if module.code_count.shape != (module.codebook.shape[0],):
            errors.append(f"{name}.code_count shape mismatch")
        if bool((module.code_count < 0).any().item()):
            errors.append(f"{name}.code_count contains negative values")
        if float(module.code_count.sum().item()) <= 0:
            errors.append(f"{name}.code_count has non-positive total")
    return errors


def _validate_formal_stage(args: Any, *, world_size: int = 1) -> None:
    representation_stage = args.formal_stage in REPRESENTATION_STAGES
    global_batch_size = 256 if representation_stage else 64
    local_batch_size = (
        global_batch_size // world_size
        if args.formal_stage in RVQ_STAGES
        else global_batch_size
    )
    common = {
        "dataset": "show_base",
        "training_speakers": [0, 1, 2, 3],
        "ori_joints": "beat_smplx_joints",
        "batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "pose_length": 64,
        "pre_frames": 4,
        "stride": 20,
        "opt_betas": [0.5, 0.999],
        "opt": "adam",
        "weight_decay": 0.0,
        "lr_policy": "step",
        "decay_rate": 0.3,
        "warmup_epochs": 0,
        "amsgrad": False,
        "pose_fps": 30,
        "vae_codebook_size": 256,
        "vae_grow": [1, 1, 2, 1],
        "vae_quantizer_lambda": 1.0,
        "variational": False,
        "rot6d": True,
        "dropout_prob": 0.3,
        "pretrain": False,
        "sparse": 0,
        "data_path": "",
        "cache_path": "",
        "e_path": "",
        "e_name": None,
        "test_path": "",
        "word_cache": False,
        "word_rep": "disabled_zero_placeholder",
        "t_pre_encoder": "disabled",
        "word_index_num": 0,
        "word_dims": 0,
        "word_f": 0,
        "freeze_wordembed": True,
        "hubert_mean_path": "",
        "hubert_std_path": "",
        "audio_infer_path": "",
        "base_ckpt": "",
        "test_ckpt": "",
        "deterministic": True,
        "benchmark": True,
        "cudnn_enabled": True,
        "log_period": 497 if representation_stage else 1_988,
        "save_every": 5,
        "use_lower_target_joints_cache": False,
    }
    stages: dict[str, dict[str, Any]] = {
        "face": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "aeface",
            "train_rvq": True,
            "tar_joints": "beat_smplx_face",
            "vae_test_dim": 106,
            "vae_layer": 2,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 200,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "show_ft_face_200.bin",
        },
        "hands": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "ae",
            "train_rvq": True,
            "tar_joints": "beat_smplx_hands",
            "vae_test_dim": 180,
            "vae_layer": 2,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 200,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "show_ft_hands_200.bin",
        },
        "upper": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "ae",
            "train_rvq": True,
            "tar_joints": "beat_smplx_upper",
            "vae_test_dim": 78,
            "vae_layer": 2,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 200,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 9999,
            "final_ckpt_name": "show_ft_upper_200.bin",
        },
        "lower": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "aelower",
            "train_rvq": True,
            "tar_joints": "beat_smplx_lower",
            "vae_test_dim": 61,
            "vae_layer": 4,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 200,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "show_ft_lower_200.bin",
            "use_lower_target_joints_cache": False,
        },
        "global": {
            "model": "motion_representation",
            "g_name": "VAEConvZero",
            "trainer": "aelowerfoot",
            "train_rvq": True,
            "tar_joints": "beat_smplx_lower",
            "vae_test_dim": 61,
            "vae_layer": 4,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 200,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "show_ft_global_200.bin",
        },
        "base": {
            "model": "semtalk",
            "g_name": "semtalk_base",
            "trainer": "semtalk_base",
            "train_rvq": False,
            "tar_joints": "beat_smplx_full",
            "pose_dims": 330,
            "audio_f": 256,
            "motion_f": 256,
            "hidden_size": 768,
            "lf": 3,
            "ll": 3,
            "lu": 3,
            "lh": 3,
            "cf": 1,
            "cl": 1,
            "cu": 1,
            "ch": 1,
            "vae_test_dim": 330,
            "vae_layer": 4,
            "vae_length": 240,
            "rec_weight": 1.0,
            "rec_pos_weight": 0.0,
            "rec_ver_weight": 0.0,
            "grad_norm": 0.99,
            "epochs": 400,
            "random_seed": 43,
            "lr_base": 1e-4,
            "decay_epochs": 999,
            "final_ckpt_name": "semtalk_base_epoch_400.bin",
        },
    }
    if args.formal_stage not in stages:
        raise RuntimeError(
            f"--formal_stage must be one of {sorted(stages)}, got {args.formal_stage!r}"
        )
    expected = {**common, **stages[args.formal_stage]}
    mismatches = []
    for name, value in expected.items():
        actual = getattr(args, name)
        if actual != value:
            mismatches.append(f"{name}={actual!r} expected {value!r}")
    if mismatches:
        raise RuntimeError(
            f"formal stage {args.formal_stage} config mismatch: "
            + "; ".join(mismatches)
        )
    if args.load_ckpt not in {None, ""}:
        raise RuntimeError(
            "formal SHOW training forbids unaudited --load_ckpt; use the "
            "strict --initial-model-checkpoint path"
        )
    if args.d_name is not None:
        raise RuntimeError("formal Base-only training forbids a discriminator")


def _load_resume(
    trainer: Any,
    resume_path: Path,
    rank: int,
    world_size: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_summary_sha256: str,
    data_mdb_sha256: str,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    source_receipt_sha256: str,
    candidate_manifest_path: Path,
) -> tuple[
    int,
    dict[str, dict[str, float | int]],
    float,
    str | None,
    int,
    dict[str, Any] | None,
]:
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "semtalk_show_train_resume_v5":
        raise RuntimeError("unsupported or unsafe resume checkpoint format")
    completed_epochs = _require_exact_audit_int(
        payload.get("completed_epochs"),
        "resume completed_epochs",
    )
    optimizer_updates = _require_exact_audit_int(
        payload.get("optimizer_updates"),
        "resume optimizer_updates",
    )
    if (
        _require_exact_audit_int(
            payload.get("world_size"),
            "resume world_size",
        )
        != world_size
    ):
        raise RuntimeError(
            f"resume world_size={payload['world_size']} does not match "
            f"{world_size}"
        )
    if (
        _require_exact_audit_int(
            payload.get("train_samples"),
            "resume train_samples",
        )
        != len(trainer.train_data)
    ):
        raise RuntimeError("resume train sample count does not match current dataset")
    if (
        _require_exact_audit_int(
            payload.get("updates_per_epoch"),
            "resume updates_per_epoch",
        )
        != trainer.train_length
    ):
        raise RuntimeError("resume updates/epoch does not match current dataloader")
    if (
        _require_exact_audit_int(
            payload.get("batch_size"),
            "resume batch_size",
        )
        != trainer.args.batch_size
    ):
        raise RuntimeError("resume batch size does not match current config")
    validate_representation_ddp_receipt(
        payload.get("distributed_training_receipt"),
        formal_stage=trainer.args.formal_stage,
        world_size=world_size,
        local_batch_size=int(trainer.args.batch_size),
        train_samples=len(trainer.train_data),
        updates_per_epoch=trainer.train_length,
        seed=int(trainer.args.random_seed),
    )
    if payload["config_sha256"] != config_sha256:
        raise RuntimeError("resume training config fingerprint does not match")
    if payload.get("lineage_manifest_sha256") != lineage_sha256:
        raise RuntimeError("resume lineage manifest fingerprint does not match")
    if payload.get("dataset_summary_sha256") != dataset_summary_sha256:
        raise RuntimeError("resume dataset summary fingerprint does not match")
    if payload.get("data_mdb_sha256") != data_mdb_sha256:
        raise RuntimeError("resume LMDB fingerprint does not match")
    if payload.get("dataset_receipt_sha256") != _payload_sha256(dataset_receipt):
        raise RuntimeError("resume dataset receipt fingerprint does not match")
    if payload.get("smplx_asset_receipt") != dataset_receipt.get("smplx_asset"):
        raise RuntimeError("resume SMPL-X asset receipt does not match")
    if payload.get("source_receipt_sha256") != source_receipt_sha256:
        raise RuntimeError("resume source checkout fingerprint does not match")
    if payload.get("initialization_receipt") != getattr(
        trainer,
        "initialization_receipt",
        None,
    ):
        raise RuntimeError("resume official initialization receipt mismatch")
    if payload.get("rvq_ema_prior_receipt") != getattr(
        trainer,
        "rvq_ema_prior_receipt",
        None,
    ):
        raise RuntimeError("resume RVQ EMA-prior receipt mismatch")
    trainer.latest_representation_candidate = (
        _validate_representation_candidate_receipt(
            trainer,
            payload.get("latest_representation_candidate"),
            completed_epochs=completed_epochs,
        )
    )
    verify_lower_target_cache_resume_receipt(
        payload,
        dataset_receipt.get(LOWER_TARGET_CACHE_RECEIPT_KEY),
    )
    _verify_smplx_training_pool_gate_resume_receipt(
        payload,
        dataset_receipt.get(SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY),
    )
    _restore_smplx_training_pool_runtime_evidence(
        trainer,
        payload,
        gate_receipt=dataset_receipt.get(
            SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
        ),
    )
    _verify_lower_target_backend_resume_receipt(
        payload,
        dataset_receipt.get(LOWER_TARGET_BACKEND_RECEIPT_KEY),
    )
    expected_optimizer_updates = completed_epochs * trainer.train_length
    if optimizer_updates != expected_optimizer_updates:
        raise RuntimeError(
            "resume actual optimizer update count does not match completed epochs"
        )
    committed_candidate_count = _require_exact_audit_int(
        payload.get("candidate_manifest_entry_count"),
        "resume candidate_manifest_entry_count",
    )
    expected_committed_count = (
        completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS
        if trainer.args.formal_stage == "base"
        else 0
    )
    if committed_candidate_count != expected_committed_count:
        raise RuntimeError("resume Base candidate committed entry count mismatch")
    transaction = _inspect_base_candidate_transaction(
        candidate_manifest_path,
        formal_stage=trainer.args.formal_stage,
        updates_per_epoch=trainer.train_length,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    current_candidate_entries = _base_candidate_entries(
        candidate_manifest_path
    )
    current_candidate_count = len(current_candidate_entries)
    allowed_candidate_counts = {committed_candidate_count}
    if (
        trainer.args.formal_stage == "base"
        and completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS
        == BASE_CANDIDATE_INTERVAL_EPOCHS // 2
    ):
        allowed_candidate_counts.add(committed_candidate_count + 1)
    if current_candidate_count not in allowed_candidate_counts:
        raise RuntimeError(
            "resume Base candidate manifest has more than one uncommitted "
            "future candidate"
        )
    if transaction is None:
        if current_candidate_count != committed_candidate_count:
            raise RuntimeError(
                "future Base candidate manifest is missing its transaction"
            )
        candidate_manifest_sha = _validate_base_candidate_manifest(
            candidate_manifest_path,
            formal_stage=trainer.args.formal_stage,
            completed_epochs=(
                current_candidate_count * BASE_CANDIDATE_INTERVAL_EPOCHS
            ),
            updates_per_epoch=trainer.train_length,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt_sha256=source_receipt_sha256,
        )
    else:
        candidate_manifest_sha = transaction["manifest_sha256"]
        target_epoch = _require_exact_audit_int(
            transaction.get("target_epoch"),
            "candidate transaction target_epoch",
        )
        stale_committed = (
            target_epoch == completed_epochs
            and transaction["manifested"]
            and transaction["state"] == "manifest_committed"
            and current_candidate_count == committed_candidate_count
        )
        future_transaction = (
            target_epoch
            == completed_epochs + BASE_CANDIDATE_INTERVAL_EPOCHS // 2
            and completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS
            == BASE_CANDIDATE_INTERVAL_EPOCHS // 2
            and _require_exact_audit_int(
                transaction.get("previous_entry_count"),
                "candidate transaction previous_entry_count",
            )
            == committed_candidate_count
        )
        if not stale_committed and not future_transaction:
            raise RuntimeError(
                "Base candidate transaction is not aligned to the exact "
                "resume boundary"
            )
        if (
            current_candidate_count == committed_candidate_count + 1
            and transaction["core"]["previous_manifest_sha256"]
            != payload.get("candidate_manifest_sha256")
        ):
            raise RuntimeError(
                "future Base candidate transaction previous manifest "
                "fingerprint mismatch"
            )
    committed_entries = current_candidate_entries[:committed_candidate_count]
    if payload.get(
        "candidate_manifest_entries_sha256"
    ) != _base_candidate_entries_sha256(committed_entries):
        raise RuntimeError("resume Base candidate committed prefix mismatch")
    if (
        current_candidate_count == committed_candidate_count
        and payload.get("candidate_manifest_sha256")
        != candidate_manifest_sha
    ):
        raise RuntimeError(
            "resume Base candidate manifest fingerprint does not match"
        )
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    ema_modules = _rvq_ema_state(trainer.model)
    if ema_modules:
        if "rvq_ema_state" not in payload:
            raise RuntimeError("RVQ resume checkpoint is missing EMA state")
        _restore_rvq_ema_state(trainer.model, payload["rvq_ema_state"])
    elif payload.get("rvq_ema_state"):
        raise RuntimeError("non-RVQ model received unexpected RVQ EMA state")
    if trainer.args.formal_stage in RVQ_STAGES:
        state_receipt = assert_rvq_rank_state(trainer.model)
        if payload.get("rvq_rank_state_receipt") != state_receipt:
            raise RuntimeError(
                "resume RVQ rank-state receipt does not match restored state"
            )
    trainer.opt.load_state_dict(payload["optimizer_state"])
    trainer.opt_s.load_state_dict(payload["scheduler_state"])
    trainer._restore_formal_optimizer_updates(optimizer_updates)
    rng_states = payload["rng_states"]
    if len(rng_states) != world_size:
        raise RuntimeError("resume RNG state count does not match world size")
    _restore_rng_state(rng_states[rank])
    bad = _finite_tree(trainer.model.state_dict(), "model")
    bad.extend(_finite_tree(_rvq_ema_state(trainer.model), "rvq_ema"))
    bad.extend(_finite_tree(trainer.opt.state_dict(), "optimizer"))
    bad.extend(_finite_tree(trainer.opt_s.state_dict(), "scheduler"))
    bad.extend(_rvq_ema_invariant_errors(trainer.model))
    if bad:
        raise FloatingPointError(
            "invalid state immediately after resume: " + ", ".join(bad[:20])
        )
    last_metrics = payload.get("last_metrics")
    if not isinstance(last_metrics, dict):
        raise RuntimeError("resume checkpoint is missing last_metrics")
    for metric_name, metric in last_metrics.items():
        average = metric.get("avg") if isinstance(metric, dict) else None
        if (
            not isinstance(metric_name, str)
            or not isinstance(metric, dict)
            or set(metric) != {"avg", "count"}
            or isinstance(average, bool)
            or not isinstance(average, Real)
            or not bool(np.isfinite(float(average)))
            or _require_exact_audit_int(
                metric.get("count"),
                f"resume last_metrics[{metric_name!r}].count",
            )
            < 1
        ):
            raise RuntimeError(
                f"resume checkpoint has invalid metric receipt {metric_name!r}"
            )
    started_unix = float(payload.get("started_unix", 0.0))
    if not np.isfinite(started_unix) or started_unix <= 0:
        raise RuntimeError("resume checkpoint has invalid started_unix")
    return (
        completed_epochs,
        last_metrics,
        started_unix,
        candidate_manifest_sha,
        current_candidate_count,
        transaction,
    )


def _save_resume(
    trainer: Any,
    path: Path,
    completed_epochs: int,
    rng_states: list[dict[str, Any]],
    world_size: int,
    train_samples: int,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_summary_sha256: str,
    data_mdb_sha256: str,
    dataset_receipt: dict[str, Any],
    source_receipt_sha256: str,
    optimizer_updates: int,
    candidate_manifest_sha256: str | None,
    candidate_manifest_entry_count: int,
    candidate_manifest_entries_sha256: str,
    last_metrics: dict[str, dict[str, float | int]],
    started_unix: float,
    distributed_training_receipt: dict[str, Any],
    rvq_rank_state_receipt: dict[str, Any] | None,
) -> None:
    completed_epochs = _require_exact_audit_int(
        completed_epochs,
        "resume save completed_epochs",
    )
    world_size = _require_exact_audit_int(
        world_size,
        "resume save world_size",
    )
    train_samples = _require_exact_audit_int(
        train_samples,
        "resume save train_samples",
    )
    updates_per_epoch = _require_exact_audit_int(
        updates_per_epoch,
        "resume save updates_per_epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "resume save optimizer_updates",
    )
    candidate_manifest_entry_count = _require_exact_audit_int(
        candidate_manifest_entry_count,
        "resume save candidate_manifest_entry_count",
    )
    batch_size = _require_exact_audit_int(
        trainer.args.batch_size,
        "resume save batch_size",
    )
    expected_optimizer_updates = completed_epochs * updates_per_epoch
    if (
        optimizer_updates != expected_optimizer_updates
        or trainer.formal_optimizer_updates != expected_optimizer_updates
    ):
        raise RuntimeError(
            "refusing to save resume with a derived optimizer update count"
        )
    expected_candidate_count = (
        completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS
        if trainer.args.formal_stage == "base"
        else 0
    )
    if candidate_manifest_entry_count != expected_candidate_count:
        raise RuntimeError(
            "refusing to save resume with an uncommitted candidate count"
        )
    if (
        not isinstance(candidate_manifest_entries_sha256, str)
        or len(candidate_manifest_entries_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate_manifest_entries_sha256
        )
    ):
        raise RuntimeError("invalid candidate entry-prefix SHA-256")
    if (
        expected_candidate_count == 0
        and candidate_manifest_sha256 is not None
    ) or (
        expected_candidate_count > 0
        and (
            not isinstance(candidate_manifest_sha256, str)
            or len(candidate_manifest_sha256) != 64
        )
    ):
        raise RuntimeError("invalid Base candidate manifest SHA binding")
    runtime_evidence = _validate_smplx_training_pool_runtime_evidence(
        getattr(
            trainer,
            "smplx_training_pool_runtime_evidence",
            None,
        ),
        formal_stage=trainer.args.formal_stage,
        gate_receipt=dataset_receipt.get(
            SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
        ),
        expected_optimizer_updates=optimizer_updates,
    )
    payload = {
        "format": "semtalk_show_train_resume_v5",
        "completed_epochs": completed_epochs,
        "world_size": world_size,
        "train_samples": train_samples,
        "updates_per_epoch": updates_per_epoch,
        "batch_size": batch_size,
        "distributed_training_receipt": distributed_training_receipt,
        "rvq_rank_state_receipt": rvq_rank_state_receipt,
        "initialization_receipt": copy.deepcopy(
            getattr(trainer, "initialization_receipt", None)
        ),
        "rvq_ema_prior_receipt": copy.deepcopy(
            getattr(trainer, "rvq_ema_prior_receipt", None)
        ),
        "latest_representation_candidate": copy.deepcopy(
            getattr(trainer, "latest_representation_candidate", None)
        ),
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_summary_sha256": dataset_summary_sha256,
        "data_mdb_sha256": data_mdb_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt_sha256": source_receipt_sha256,
        "optimizer_updates": optimizer_updates,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_manifest_entry_count": candidate_manifest_entry_count,
        "candidate_manifest_entries_sha256": (
            candidate_manifest_entries_sha256
        ),
        "last_metrics": last_metrics,
        "started_unix": started_unix,
        "model_state": trainer.model.state_dict(),
        "rvq_ema_state": _rvq_ema_state(trainer.model),
        "optimizer_state": trainer.opt.state_dict(),
        "scheduler_state": trainer.opt_s.state_dict(),
        "rng_states": rng_states,
        SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY: runtime_evidence,
    }
    attach_lower_target_cache_receipt(
        payload,
        dataset_receipt.get(LOWER_TARGET_CACHE_RECEIPT_KEY),
    )
    _attach_smplx_training_pool_gate_receipt(
        payload,
        dataset_receipt.get(SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY),
    )
    _attach_lower_target_backend_receipt(
        payload,
        dataset_receipt.get(LOWER_TARGET_BACKEND_RECEIPT_KEY),
    )
    _atomic_torch_save(
        path,
        payload,
    )


def _model_payload(
    trainer: Any,
    *,
    formal_stage: str,
    config_sha256: str,
    lineage_sha256: str,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    optimizer_updates: int,
    candidate_manifest_receipt: dict[str, Any] | None,
    distributed_training_receipt: dict[str, Any],
    rvq_rank_state_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "model audit optimizer_updates",
    )
    if candidate_manifest_receipt is not None:
        for key in ("entries", "last_epoch", "last_optimizer_updates"):
            _require_exact_audit_int(
                candidate_manifest_receipt.get(key),
                f"model audit Base candidate manifest {key}",
            )
    runtime_evidence = _validate_smplx_training_pool_runtime_evidence(
        getattr(
            trainer,
            "smplx_training_pool_runtime_evidence",
            None,
        ),
        formal_stage=formal_stage,
        gate_receipt=dataset_receipt.get(
            SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
        ),
        expected_optimizer_updates=optimizer_updates,
    )
    pool_gate_receipt = dataset_receipt.get(
        SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
    )
    pool_mode = (
        pool_gate_receipt.get("mode")
        if isinstance(pool_gate_receipt, dict)
        else "disabled"
    )
    audit = {
        "format": "semtalk_show_model_v2",
        "formal_stage": formal_stage,
        "smplx_training_pool_mode": pool_mode,
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_summary_sha256": dataset_receipt["summary_sha256"],
        "data_mdb_sha256": dataset_receipt["data_mdb_sha256"],
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
        "optimizer_updates": optimizer_updates,
        "distributed_training_receipt": distributed_training_receipt,
        "rvq_rank_state_receipt": rvq_rank_state_receipt,
        "initialization_receipt": copy.deepcopy(
            getattr(trainer, "initialization_receipt", None)
        ),
        "rvq_ema_prior_receipt": copy.deepcopy(
            getattr(trainer, "rvq_ema_prior_receipt", None)
        ),
        "latest_representation_candidate": copy.deepcopy(
            getattr(trainer, "latest_representation_candidate", None)
        ),
        "base_candidate_manifest": candidate_manifest_receipt,
        SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY: runtime_evidence,
    }
    attach_lower_target_cache_receipt(
        audit,
        dataset_receipt.get(LOWER_TARGET_CACHE_RECEIPT_KEY),
    )
    _attach_smplx_training_pool_gate_receipt(
        audit,
        dataset_receipt.get(SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY),
    )
    _attach_lower_target_backend_receipt(
        audit,
        dataset_receipt.get(LOWER_TARGET_BACKEND_RECEIPT_KEY),
    )
    return {
        "model_state": trainer.model.state_dict(),
        "audit": audit,
    }


def _base_candidate_audit(
    *,
    epoch: int,
    optimizer_updates: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    epoch = _require_exact_audit_int(
        epoch,
        "Base candidate audit epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "Base candidate audit optimizer_updates",
    )
    return {
        "format": "semtalk_show_base_candidate_model_v1",
        "formal_stage": "base",
        "smplx_training_pool_mode": "disabled",
        "candidate_epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
    }


def _candidate_filename(epoch: int, optimizer_updates: int) -> str:
    epoch = _require_exact_audit_int(epoch, "Base candidate filename epoch")
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "Base candidate filename optimizer_updates",
    )
    return (
        f"semtalk_base_candidate_epoch_{epoch:04d}"
        f"_step_{optimizer_updates:09d}.bin"
    )


def _candidate_relative_path(epoch: int, optimizer_updates: int) -> str:
    return (
        f"base_candidates/"
        f"{_candidate_filename(epoch, optimizer_updates)}"
    )


def _candidate_transaction_path(manifest_path: Path) -> Path:
    return manifest_path.parent / BASE_CANDIDATE_TRANSACTION_FILENAME


def _candidate_staging_path(manifest_path: Path) -> Path:
    return manifest_path.parent / BASE_CANDIDATE_STAGING_FILENAME


def _load_regular_json(path: Path, label: str) -> dict[str, Any]:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _base_candidate_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if manifest_path.is_symlink():
        raise RuntimeError(
            f"Base candidate manifest must not be a symlink: {manifest_path}"
        )
    if not manifest_path.exists():
        return []
    manifest = _load_regular_json(
        manifest_path,
        "Base candidate manifest",
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("Base candidate manifest entries must be a list")
    return entries


def _base_candidate_entries_sha256(
    entries: list[dict[str, Any]],
) -> str:
    return _payload_sha256(entries)


def _base_candidate_manifest_payload(
    *,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "semtalk_show_base_candidate_manifest_v2",
        "formal_stage": "base",
        "interval_epochs": BASE_CANDIDATE_INTERVAL_EPOCHS,
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
        "entries": entries,
    }


def _base_candidate_transaction_core(
    *,
    epoch: int,
    optimizer_updates: int,
    previous_manifest_sha256: str | None,
    previous_entries: list[dict[str, Any]],
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    epoch = _require_exact_audit_int(
        epoch,
        "Base candidate transaction epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "Base candidate transaction optimizer_updates",
    )
    candidate_audit = _base_candidate_audit(
        epoch=epoch,
        optimizer_updates=optimizer_updates,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    return {
        "format": "semtalk_show_base_candidate_transaction_core_v1",
        "formal_stage": "base",
        "candidate_epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "checkpoint": _candidate_relative_path(epoch, optimizer_updates),
        "staging_checkpoint": BASE_CANDIDATE_STAGING_FILENAME,
        "previous_manifest_sha256": previous_manifest_sha256,
        "previous_entry_count": len(previous_entries),
        "previous_entries_sha256": _base_candidate_entries_sha256(
            previous_entries
        ),
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
        "model_audit_sha256": _payload_sha256(candidate_audit),
    }


def _base_candidate_transaction_payload(
    core: dict[str, Any],
    *,
    state: str,
    checkpoint_sha256: str | None,
    manifest_sha256: str | None,
) -> dict[str, Any]:
    if state not in {
        "prepared",
        "checkpoint_committed",
        "manifest_committed",
    }:
        raise RuntimeError(f"invalid Base candidate transaction state: {state}")
    return {
        "format": "semtalk_show_base_candidate_transaction_v1",
        "state": state,
        "core": core,
        "core_sha256": _payload_sha256(core),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
    }


def _write_base_candidate_transaction(
    transaction_path: Path,
    core: dict[str, Any],
    *,
    state: str,
    checkpoint_sha256: str | None,
    manifest_sha256: str | None,
) -> dict[str, Any]:
    payload = _base_candidate_transaction_payload(
        core,
        state=state,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
    )
    _atomic_json(transaction_path, payload)
    return payload


def _validate_base_candidate_snapshot(
    manifest_path: Path,
    *,
    expected_sha256: str | None,
    expected_entry_count: int,
    allowed_pending_checkpoint: Path | None = None,
    allow_empty_candidate_dir: bool = False,
) -> list[dict[str, Any]]:
    candidate_dir = manifest_path.parent / "base_candidates"
    if expected_entry_count == 0:
        if expected_sha256 is not None or manifest_path.exists() or (
            manifest_path.is_symlink()
        ):
            raise RuntimeError("unexpected Base candidate snapshot before epoch 10")
        if allowed_pending_checkpoint is None:
            if allow_empty_candidate_dir:
                if candidate_dir.is_symlink() or (
                    candidate_dir.exists()
                    and (
                        not candidate_dir.is_dir()
                        or any(candidate_dir.iterdir())
                    )
                ):
                    raise RuntimeError(
                        "empty transaction candidate directory changed"
                    )
                return []
            if candidate_dir.exists() or candidate_dir.is_symlink():
                raise RuntimeError(
                    "unexpected Base candidate snapshot before epoch 10"
                )
            return []
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            raise RuntimeError("pending Base candidate directory is invalid")
        actual_children = list(candidate_dir.iterdir())
        if len(actual_children) != 1:
            raise RuntimeError("pending Base candidate file set changed")
        child = actual_children[0]
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(
            child_stat.st_mode
        ):
            raise RuntimeError(
                f"pending Base candidate is not a regular file: {child}"
            )
        if child.resolve() != allowed_pending_checkpoint.resolve():
            raise RuntimeError("pending Base candidate path changed")
        return []
    entries = _base_candidate_entries(manifest_path)
    if (
        len(entries) != expected_entry_count
        or expected_sha256 is None
        or _sha256(manifest_path) != expected_sha256
        or candidate_dir.is_symlink()
        or not candidate_dir.is_dir()
    ):
        raise RuntimeError("Base candidate manifest snapshot changed")
    expected_paths = {
        (manifest_path.parent / str(record.get("checkpoint", ""))).resolve()
        for record in entries
        if isinstance(record, dict)
    }
    if allowed_pending_checkpoint is not None:
        expected_paths.add(allowed_pending_checkpoint.resolve())
    actual_children = list(candidate_dir.iterdir())
    for child in actual_children:
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(
            child_stat.st_mode
        ):
            raise RuntimeError(
                f"Base candidate is no longer a regular file: {child}"
            )
    actual_paths = {child.resolve() for child in actual_children}
    expected_file_count = expected_entry_count + int(
        allowed_pending_checkpoint is not None
    )
    if len(expected_paths) != expected_file_count or actual_paths != expected_paths:
        raise RuntimeError("Base candidate file set changed")
    return entries


def _inspect_new_prepared_base_candidate_transaction(
    manifest_path: Path,
    *,
    core: dict[str, Any],
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Strictly verify a transaction just created by this process.

    The recovery inspector intentionally performs a full checkpoint audit,
    because it must distrust artifacts left by an earlier process.  A newly
    prepared transaction has no checkpoint yet, so reloading every historical
    candidate would add quadratic I/O without strengthening this boundary.
    This inspector instead reopens the atomic intent, verifies it byte-for-byte
    at the payload level, and rechecks the immutable manifest/file snapshot.
    """

    transaction_path = _candidate_transaction_path(manifest_path)
    staging_path = _candidate_staging_path(manifest_path)
    expected_payload = _base_candidate_transaction_payload(
        core,
        state="prepared",
        checkpoint_sha256=None,
        manifest_sha256=None,
    )
    transaction = _load_regular_json(
        transaction_path,
        "Base candidate transaction",
    )
    if transaction != expected_payload:
        raise RuntimeError("new Base candidate transaction payload mismatch")

    previous_count = _require_exact_audit_int(
        core.get("previous_entry_count"),
        "new candidate transaction previous_entry_count",
    )
    previous_entries = _validate_base_candidate_snapshot(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_entry_count=previous_count,
    )
    if (
        len(previous_entries) != previous_count
        or _base_candidate_entries_sha256(previous_entries)
        != core["previous_entries_sha256"]
    ):
        raise RuntimeError("new Base candidate transaction prefix mismatch")

    candidate_path = (
        manifest_path.parent / str(core["checkpoint"])
    ).resolve()
    if candidate_path.exists() or candidate_path.is_symlink():
        raise RuntimeError(
            "new Base candidate transaction unexpectedly has a checkpoint"
        )
    if staging_path.exists() or staging_path.is_symlink():
        raise RuntimeError(
            "new Base candidate transaction unexpectedly has a staging file"
        )
    return {
        "path": transaction_path,
        "payload": transaction,
        "core": core,
        "state": "prepared",
        "target_epoch": _require_exact_audit_int(
            core.get("candidate_epoch"),
            "new candidate transaction candidate_epoch",
        ),
        "previous_entry_count": previous_count,
        "current_entry_count": previous_count,
        "candidate_path": candidate_path,
        "candidate_exists": False,
        "checkpoint_sha256": None,
        "manifested": False,
        "manifest_sha256": expected_manifest_sha256,
        "staging_exists": False,
    }


def _validate_base_candidate_manifest(
    manifest_path: Path,
    *,
    formal_stage: str,
    completed_epochs: int,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt_sha256: str,
    allowed_pending_checkpoint: Path | None = None,
    allow_empty_candidate_dir: bool = False,
) -> str | None:
    candidate_dir = manifest_path.parent / "base_candidates"
    expected_epochs = list(
        range(
            BASE_CANDIDATE_INTERVAL_EPOCHS,
            int(completed_epochs) + 1,
            BASE_CANDIDATE_INTERVAL_EPOCHS,
        )
    )
    if formal_stage != "base":
        if (
            manifest_path.exists()
            or manifest_path.is_symlink()
            or candidate_dir.exists()
            or candidate_dir.is_symlink()
        ):
            raise RuntimeError(
                "Base candidate artifacts are forbidden outside formal_stage=base"
            )
        return None
    if not expected_epochs:
        if allowed_pending_checkpoint is not None:
            _validate_base_candidate_snapshot(
                manifest_path,
                expected_sha256=None,
                expected_entry_count=0,
                allowed_pending_checkpoint=allowed_pending_checkpoint,
            )
            return None
        if allow_empty_candidate_dir:
            _validate_base_candidate_snapshot(
                manifest_path,
                expected_sha256=None,
                expected_entry_count=0,
                allow_empty_candidate_dir=True,
            )
            return None
        if (
            manifest_path.exists()
            or manifest_path.is_symlink()
            or candidate_dir.exists()
            or candidate_dir.is_symlink()
        ):
            raise RuntimeError(
                "unexpected Base candidate artifacts before epoch 10"
            )
        return None
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise RuntimeError(
            f"Base candidate directory must be a regular directory: {candidate_dir}"
        )

    manifest = _load_regular_json(
        manifest_path,
        "Base candidate manifest",
    )
    expected_manifest_keys = {
        "format",
        "formal_stage",
        "interval_epochs",
        "config_sha256",
        "lineage_manifest_sha256",
        "dataset_receipt_sha256",
        "smplx_asset_receipt",
        "source_receipt",
        "source_receipt_sha256",
        "entries",
    }
    source_receipt = manifest.get("source_receipt")
    entries = manifest.get("entries")
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("format")
        != "semtalk_show_base_candidate_manifest_v2"
        or manifest.get("formal_stage") != "base"
        or _require_exact_audit_int(
            manifest.get("interval_epochs"),
            "candidate manifest interval_epochs",
        )
        != BASE_CANDIDATE_INTERVAL_EPOCHS
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("lineage_manifest_sha256") != lineage_sha256
        or manifest.get("dataset_receipt_sha256")
        != _payload_sha256(dataset_receipt)
        or manifest.get("smplx_asset_receipt")
        != dataset_receipt.get("smplx_asset")
        or not isinstance(source_receipt, dict)
        or _payload_sha256(source_receipt) != source_receipt_sha256
        or manifest.get("source_receipt_sha256") != source_receipt_sha256
        or not isinstance(entries, list)
        or len(entries) != len(expected_epochs)
    ):
        raise RuntimeError("Base candidate manifest binding mismatch")

    expected_files: set[Path] = set()
    for entry_index, (expected_epoch, record) in enumerate(
        zip(expected_epochs, entries, strict=True)
    ):
        expected_updates = expected_epoch * int(updates_per_epoch)
        expected_filename = _candidate_filename(
            expected_epoch,
            expected_updates,
        )
        expected_relative_path = _candidate_relative_path(
            expected_epoch,
            expected_updates,
        )
        previous_entries = entries[:entry_index]
        if previous_entries:
            previous_manifest = {
                **manifest,
                "entries": previous_entries,
            }
            previous_manifest_sha256 = _json_document_sha256(
                previous_manifest
            )
        else:
            previous_manifest_sha256 = None
        expected_transaction_core = _base_candidate_transaction_core(
            epoch=expected_epoch,
            optimizer_updates=expected_updates,
            previous_manifest_sha256=previous_manifest_sha256,
            previous_entries=previous_entries,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "epoch",
                "optimizer_updates",
                "checkpoint",
                "checkpoint_sha256",
                "model_audit_sha256",
                "transaction_core",
                "transaction_core_sha256",
            }
            or _require_exact_audit_int(
                record.get("epoch"),
                "candidate record epoch",
            )
            != expected_epoch
            or _require_exact_audit_int(
                record.get("optimizer_updates"),
                "candidate record optimizer_updates",
            )
            != expected_updates
            or record.get("checkpoint") != expected_relative_path
            or not isinstance(record.get("transaction_core"), dict)
            or _require_exact_audit_int(
                record["transaction_core"].get("candidate_epoch"),
                "candidate transaction core candidate_epoch",
            )
            != expected_epoch
            or _require_exact_audit_int(
                record["transaction_core"].get("optimizer_updates"),
                "candidate transaction core optimizer_updates",
            )
            != expected_updates
            or _require_exact_audit_int(
                record["transaction_core"].get("previous_entry_count"),
                "candidate transaction core previous_entry_count",
            )
            != entry_index
            or record.get("transaction_core") != expected_transaction_core
            or record.get("transaction_core_sha256")
            != _payload_sha256(expected_transaction_core)
        ):
            raise RuntimeError(
                f"invalid Base candidate record for epoch {expected_epoch}"
            )
        checkpoint_input = candidate_dir / expected_filename
        try:
            checkpoint_stat = checkpoint_input.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(checkpoint_input) from None
        if (
            stat.S_ISLNK(checkpoint_stat.st_mode)
            or not stat.S_ISREG(checkpoint_stat.st_mode)
        ):
            raise RuntimeError(
                "Base candidate checkpoint must be a regular non-symlink "
                f"file: {checkpoint_input}"
            )
        checkpoint_snapshot = checkpoint_input.read_bytes()
        checkpoint_sha = hashlib.sha256(checkpoint_snapshot).hexdigest()
        if checkpoint_sha != record.get("checkpoint_sha256"):
            raise RuntimeError(
                f"Base candidate checkpoint SHA mismatch: {checkpoint_input}"
            )
        checkpoint = _torch_load_candidate_snapshot(
            checkpoint_snapshot,
            checkpoint_path=checkpoint_input,
        )
        expected_audit = _base_candidate_audit(
            epoch=expected_epoch,
            optimizer_updates=expected_updates,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint) != {"model_state", "audit"}
            or checkpoint.get("audit") != expected_audit
            or record.get("model_audit_sha256")
            != _payload_sha256(expected_audit)
            or not isinstance(checkpoint.get("model_state"), dict)
            or not checkpoint["model_state"]
            or _finite_tree(
                checkpoint["model_state"],
                f"candidate_epoch_{expected_epoch}",
            )
        ):
            raise RuntimeError(
                f"invalid Base candidate checkpoint: {checkpoint_input}"
            )
        expected_files.add(checkpoint_input.resolve())

    if allowed_pending_checkpoint is not None:
        pending_input = allowed_pending_checkpoint
        if pending_input.parent.resolve() != candidate_dir.resolve():
            raise RuntimeError("pending Base candidate is outside candidate_dir")
        try:
            pending_stat = pending_input.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(pending_input) from None
        if (
            stat.S_ISLNK(pending_stat.st_mode)
            or not stat.S_ISREG(pending_stat.st_mode)
        ):
            raise RuntimeError(
                "pending Base candidate checkpoint must be a regular "
                f"non-symlink file: {pending_input}"
            )
        expected_files.add(pending_input.resolve())

    actual_children = list(candidate_dir.iterdir())
    for child in actual_children:
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(
            child_stat.st_mode
        ):
            raise RuntimeError(
                f"Base candidate is not a regular non-symlink file: {child}"
            )
    actual_files = {child.resolve() for child in actual_children}
    if actual_files != expected_files:
        raise RuntimeError("Base candidate directory contains missing or extra files")
    return _sha256(manifest_path)


def _load_and_validate_base_candidate_checkpoint(
    candidate_path: Path,
    *,
    expected_audit: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        candidate_stat = candidate_path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(candidate_path) from None
    if (
        stat.S_ISLNK(candidate_stat.st_mode)
        or not stat.S_ISREG(candidate_stat.st_mode)
    ):
        raise RuntimeError(
            "Base candidate checkpoint must be a regular non-symlink file: "
            f"{candidate_path}"
        )
    checkpoint_snapshot = candidate_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_snapshot).hexdigest()
    checkpoint = _torch_load_candidate_snapshot(
        checkpoint_snapshot,
        checkpoint_path=candidate_path,
    )
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"model_state", "audit"}
        or checkpoint.get("audit") != expected_audit
        or not isinstance(checkpoint.get("model_state"), dict)
        or not checkpoint["model_state"]
        or _finite_tree(checkpoint["model_state"], "pending_candidate")
    ):
        raise RuntimeError(
            f"invalid Base candidate checkpoint: {candidate_path}"
        )
    return checkpoint, checkpoint_sha256


def _valid_optional_sha256(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _inspect_base_candidate_transaction(
    manifest_path: Path,
    *,
    formal_stage: str,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
) -> dict[str, Any] | None:
    transaction_path = _candidate_transaction_path(manifest_path)
    staging_path = _candidate_staging_path(manifest_path)
    if transaction_path.is_symlink():
        raise RuntimeError("Base candidate transaction must not be a symlink")
    if not transaction_path.exists():
        if staging_path.exists() or staging_path.is_symlink():
            raise RuntimeError(
                "Base candidate staging file exists without a transaction"
            )
        return None
    if formal_stage != "base":
        raise RuntimeError(
            "Base candidate transaction is forbidden outside formal_stage=base"
        )

    transaction = _load_regular_json(
        transaction_path,
        "Base candidate transaction",
    )
    if (
        set(transaction)
        != {
            "format",
            "state",
            "core",
            "core_sha256",
            "checkpoint_sha256",
            "manifest_sha256",
        }
        or transaction.get("format")
        != "semtalk_show_base_candidate_transaction_v1"
        or transaction.get("state")
        not in {
            "prepared",
            "checkpoint_committed",
            "manifest_committed",
        }
        or not isinstance(transaction.get("core"), dict)
        or transaction.get("core_sha256")
        != _payload_sha256(transaction["core"])
        or not _valid_optional_sha256(
            transaction.get("checkpoint_sha256")
        )
        or not _valid_optional_sha256(transaction.get("manifest_sha256"))
    ):
        raise RuntimeError("invalid Base candidate transaction envelope")

    core = transaction["core"]
    target_epoch = _require_exact_audit_int(
        core.get("candidate_epoch"),
        "candidate transaction candidate_epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        core.get("optimizer_updates"),
        "candidate transaction optimizer_updates",
    )
    previous_count = _require_exact_audit_int(
        core.get("previous_entry_count"),
        "candidate transaction previous_entry_count",
    )
    if (
        target_epoch <= 0
        or target_epoch % BASE_CANDIDATE_INTERVAL_EPOCHS != 0
        or target_epoch
        != (previous_count + 1) * BASE_CANDIDATE_INTERVAL_EPOCHS
        or optimizer_updates != target_epoch * int(updates_per_epoch)
    ):
        raise RuntimeError("invalid Base candidate transaction target")

    entries = _base_candidate_entries(manifest_path)
    current_count = len(entries)
    if current_count not in {previous_count, previous_count + 1}:
        raise RuntimeError(
            "Base candidate transaction/manifest entry count mismatch"
        )
    previous_entries = entries[:previous_count]
    if len(previous_entries) != previous_count:
        raise RuntimeError("Base candidate transaction prefix is incomplete")
    if previous_entries:
        manifest = _load_regular_json(
            manifest_path,
            "Base candidate manifest",
        )
        previous_manifest_sha256 = _json_document_sha256(
            {**manifest, "entries": previous_entries}
        )
    else:
        previous_manifest_sha256 = None
    expected_core = _base_candidate_transaction_core(
        epoch=target_epoch,
        optimizer_updates=optimizer_updates,
        previous_manifest_sha256=previous_manifest_sha256,
        previous_entries=previous_entries,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    if core != expected_core:
        raise RuntimeError("Base candidate transaction binding mismatch")

    candidate_path = (
        manifest_path.parent / str(core["checkpoint"])
    ).resolve()
    expected_candidate_path = (
        manifest_path.parent
        / _candidate_relative_path(target_epoch, optimizer_updates)
    ).resolve()
    if candidate_path != expected_candidate_path:
        raise RuntimeError("Base candidate transaction path mismatch")
    candidate_exists = candidate_path.exists() or candidate_path.is_symlink()
    if candidate_exists and (staging_path.exists() or staging_path.is_symlink()):
        raise RuntimeError(
            "Base candidate final checkpoint and staging file coexist"
        )
    if staging_path.exists() or staging_path.is_symlink():
        staging_stat = staging_path.lstat()
        if stat.S_ISLNK(staging_stat.st_mode) or not stat.S_ISREG(
            staging_stat.st_mode
        ):
            raise RuntimeError(
                "Base candidate staging path must be a regular non-symlink file"
            )

    expected_audit = _base_candidate_audit(
        epoch=target_epoch,
        optimizer_updates=optimizer_updates,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    checkpoint_sha256 = None
    if candidate_exists:
        _, checkpoint_sha256 = _load_and_validate_base_candidate_checkpoint(
            candidate_path,
            expected_audit=expected_audit,
        )

    manifested = current_count == previous_count + 1
    manifest_sha256 = _validate_base_candidate_manifest(
        manifest_path,
        formal_stage="base",
        completed_epochs=(
            current_count * BASE_CANDIDATE_INTERVAL_EPOCHS
        ),
        updates_per_epoch=updates_per_epoch,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt_sha256=_payload_sha256(source_receipt),
        allowed_pending_checkpoint=(
            candidate_path if candidate_exists and not manifested else None
        ),
        allow_empty_candidate_dir=(
            not candidate_exists and not manifested
        ),
    )
    if not manifested and manifest_sha256 != core["previous_manifest_sha256"]:
        raise RuntimeError(
            "Base candidate transaction previous manifest SHA mismatch"
        )
    if manifested and not candidate_exists:
        raise RuntimeError(
            "Base candidate manifest committed without its checkpoint"
        )

    state = transaction["state"]
    recorded_checkpoint_sha256 = transaction["checkpoint_sha256"]
    recorded_manifest_sha256 = transaction["manifest_sha256"]
    if state == "prepared":
        if (
            recorded_checkpoint_sha256 is not None
            or recorded_manifest_sha256 is not None
        ):
            raise RuntimeError("prepared candidate transaction has commit SHAs")
    elif state == "checkpoint_committed":
        if (
            not candidate_exists
            or recorded_checkpoint_sha256 != checkpoint_sha256
            or recorded_manifest_sha256 is not None
        ):
            raise RuntimeError(
                "checkpoint_committed candidate transaction mismatch"
            )
    elif (
        not manifested
        or not candidate_exists
        or recorded_checkpoint_sha256 != checkpoint_sha256
        or recorded_manifest_sha256 != manifest_sha256
    ):
        raise RuntimeError("manifest_committed candidate transaction mismatch")

    return {
        "path": transaction_path,
        "payload": transaction,
        "core": core,
        "state": state,
        "target_epoch": target_epoch,
        "previous_entry_count": previous_count,
        "current_entry_count": current_count,
        "candidate_path": candidate_path,
        "candidate_exists": candidate_exists,
        "checkpoint_sha256": checkpoint_sha256,
        "manifested": manifested,
        "manifest_sha256": manifest_sha256,
        "staging_exists": staging_path.exists(),
    }


def _assert_candidate_matches_model(
    trainer: Any,
    checkpoint: dict[str, Any],
) -> None:
    current_state = trainer.model.state_dict()
    existing_state = checkpoint.get("model_state")
    if (
        not isinstance(existing_state, dict)
        or set(existing_state) != set(current_state)
    ):
        raise RuntimeError(
            "replayed Base candidate model-state keys do not match"
        )
    for name, current_value in current_state.items():
        existing_value = existing_state[name]
        if (
            not torch.is_tensor(existing_value)
            or not torch.is_tensor(current_value)
            or existing_value.dtype != current_value.dtype
            or tuple(existing_value.shape) != tuple(current_value.shape)
            or not torch.equal(
                existing_value,
                current_value.detach().cpu(),
            )
        ):
            raise RuntimeError(
                "replayed Base candidate differs from immutable "
                f"checkpoint at {name}"
            )


def _clear_committed_base_candidate_transaction(
    manifest_path: Path,
    *,
    completed_epochs: int,
    manifest_sha256: str | None,
) -> None:
    transaction_path = _candidate_transaction_path(manifest_path)
    if not transaction_path.exists() and not transaction_path.is_symlink():
        return
    transaction = _load_regular_json(
        transaction_path,
        "Base candidate transaction",
    )
    core = transaction.get("core")
    if (
        transaction.get("state") != "manifest_committed"
        or not isinstance(core, dict)
        or _require_exact_audit_int(
            core.get("candidate_epoch"),
            "committed candidate transaction epoch",
        )
        != completed_epochs
        or transaction.get("manifest_sha256") != manifest_sha256
        or _candidate_staging_path(manifest_path).exists()
        or _candidate_staging_path(manifest_path).is_symlink()
    ):
        raise RuntimeError(
            "refusing to clear an uncommitted Base candidate transaction"
        )
    transaction_path.unlink()
    directory_fd = os.open(transaction_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _save_base_candidate(
    trainer: Any,
    manifest_path: Path,
    *,
    completed_epochs: int,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if trainer.args.formal_stage != "base":
        raise RuntimeError("Base candidate save requested for a non-Base stage")
    if (
        completed_epochs <= 0
        or completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS != 0
    ):
        raise RuntimeError("Base candidates may only be saved every 10 epochs")
    optimizer_updates = trainer.formal_optimizer_updates
    expected_updates = completed_epochs * int(updates_per_epoch)
    if optimizer_updates != expected_updates:
        raise RuntimeError("Base candidate actual optimizer update count mismatch")

    transaction_path = _candidate_transaction_path(manifest_path)
    staging_path = _candidate_staging_path(manifest_path)
    candidate_dir = manifest_path.parent / "base_candidates"
    candidate_path = candidate_dir / _candidate_filename(
        completed_epochs,
        optimizer_updates,
    )
    transaction = _inspect_base_candidate_transaction(
        manifest_path,
        formal_stage="base",
        updates_per_epoch=updates_per_epoch,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    if transaction is None:
        previous_count = (
            completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS - 1
        )
        previous_entries = _validate_base_candidate_snapshot(
            manifest_path,
            expected_sha256=expected_manifest_sha256,
            expected_entry_count=previous_count,
        )
        core = _base_candidate_transaction_core(
            epoch=completed_epochs,
            optimizer_updates=optimizer_updates,
            previous_manifest_sha256=expected_manifest_sha256,
            previous_entries=previous_entries,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        _write_base_candidate_transaction(
            transaction_path,
            core,
            state="prepared",
            checkpoint_sha256=None,
            manifest_sha256=None,
        )
        transaction = _inspect_new_prepared_base_candidate_transaction(
            manifest_path,
            core=core,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        assert transaction is not None
    else:
        if (
            transaction["target_epoch"] != completed_epochs
            or transaction["core"]["optimizer_updates"] != optimizer_updates
            or transaction["manifest_sha256"] != expected_manifest_sha256
        ):
            raise RuntimeError(
                "active Base candidate transaction does not match this epoch "
                "or trusted manifest fingerprint"
            )
    core = transaction["core"]
    previous_count = _require_exact_audit_int(
        core.get("previous_entry_count"),
        "candidate save previous_entry_count",
    )
    previous_entries = _base_candidate_entries(manifest_path)[:previous_count]
    if candidate_dir.exists() and (
        candidate_dir.is_symlink() or not candidate_dir.is_dir()
    ):
        raise RuntimeError(
            f"invalid Base candidate directory: {candidate_dir}"
        )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_audit = _base_candidate_audit(
        epoch=completed_epochs,
        optimizer_updates=optimizer_updates,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    if not candidate_path.exists() and not candidate_path.is_symlink():
        _atomic_torch_save(
            candidate_path,
            {
                "model_state": trainer.model.state_dict(),
                "audit": candidate_audit,
            },
            staging_path=staging_path,
        )
    checkpoint, checkpoint_sha256 = (
        _load_and_validate_base_candidate_checkpoint(
            candidate_path,
            expected_audit=candidate_audit,
        )
    )
    _assert_candidate_matches_model(trainer, checkpoint)
    if transaction["state"] != "manifest_committed":
        _write_base_candidate_transaction(
            transaction_path,
            core,
            state="checkpoint_committed",
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=None,
        )

    record = {
        "epoch": completed_epochs,
        "optimizer_updates": optimizer_updates,
        "checkpoint": _candidate_relative_path(
            completed_epochs,
            optimizer_updates,
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "model_audit_sha256": _payload_sha256(candidate_audit),
        "transaction_core": core,
        "transaction_core_sha256": _payload_sha256(core),
    }
    current_entries = _base_candidate_entries(manifest_path)
    if len(current_entries) == previous_count:
        manifest_payload = _base_candidate_manifest_payload(
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
            entries=[*previous_entries, record],
        )
        _atomic_json(manifest_path, manifest_payload)
    elif (
        len(current_entries) != previous_count + 1
        or current_entries[-1] != record
    ):
        raise RuntimeError(
            "Base candidate manifest is not at the recoverable transaction edge"
        )

    manifest_sha = _sha256(manifest_path)
    if manifest_sha != _json_document_sha256(
        _load_regular_json(
            manifest_path,
            "Base candidate manifest",
        )
    ):
        raise RuntimeError("Base candidate manifest serialization mismatch")
    _validate_base_candidate_snapshot(
        manifest_path,
        expected_sha256=manifest_sha,
        expected_entry_count=previous_count + 1,
    )
    _write_base_candidate_transaction(
        transaction_path,
        core,
        state="manifest_committed",
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha,
    )
    return {
        "path": manifest_path.name,
        "sha256": manifest_sha,
        "entries": len(previous_entries) + 1,
        "last_epoch": completed_epochs,
        "last_optimizer_updates": optimizer_updates,
    }


def _base_candidate_manifest_receipt(
    manifest_path: Path,
    manifest_sha256: str | None,
    *,
    completed_epochs: int,
    updates_per_epoch: int,
) -> dict[str, Any] | None:
    if manifest_sha256 is None:
        return None
    completed_epochs = _require_exact_audit_int(
        completed_epochs,
        "Base candidate receipt completed_epochs",
    )
    updates_per_epoch = _require_exact_audit_int(
        updates_per_epoch,
        "Base candidate receipt updates_per_epoch",
    )
    last_epoch = (
        completed_epochs
        // BASE_CANDIDATE_INTERVAL_EPOCHS
        * BASE_CANDIDATE_INTERVAL_EPOCHS
    )
    return {
        "path": manifest_path.name,
        "sha256": manifest_sha256,
        "entries": completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS,
        "last_epoch": last_epoch,
        "last_optimizer_updates": last_epoch * updates_per_epoch,
    }


def _require_trusted_candidate_manifest_sha(
    observed_sha256: str | None,
    trusted_sha256: str | None,
) -> str | None:
    if observed_sha256 != trusted_sha256:
        raise RuntimeError(
            "final Base candidate manifest SHA differs from the trusted "
            "training snapshot"
        )
    return observed_sha256


def _verify_or_write_final(
    path: Path,
    expected_payload: dict[str, Any],
) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"existing final checkpoint is not a regular file: {path}"
            )
        try:
            actual = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            actual = torch.load(path, map_location="cpu")
        if (
            not isinstance(actual, dict)
            or set(actual) != {"model_state", "audit"}
            or actual.get("audit") != (
                expected_payload["audit"]
            )
        ):
            raise RuntimeError(
                "existing final checkpoint audit does not match this run"
            )
        actual_state = actual.get("model_state")
        expected_state = expected_payload["model_state"]
        if (
            not isinstance(actual_state, dict)
            or set(actual_state) != set(expected_state)
        ):
            raise RuntimeError(
                "existing final checkpoint model-state keys do not match"
            )
        for name, expected in expected_state.items():
            actual_value = actual_state[name]
            if (
                not torch.is_tensor(actual_value)
                or not torch.is_tensor(expected)
                or actual_value.dtype != expected.dtype
                or tuple(actual_value.shape) != tuple(expected.shape)
                or not torch.equal(
                    actual_value.detach().cpu(),
                    expected.detach().cpu(),
                )
            ):
                raise RuntimeError(
                    "existing final checkpoint differs from resumed model at "
                    f"{name}"
                )
    else:
        _atomic_torch_save(path, expected_payload)
    return _sha256(path)


def _save_representation_candidate(
    trainer: Any,
    checkpoint_dir: Path,
    *,
    completed_epochs: int,
    config_sha256: str,
    lineage_sha256: str,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    distributed_training_receipt: dict[str, Any],
    rvq_rank_state_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    if trainer.args.formal_stage not in REPRESENTATION_STAGES:
        raise RuntimeError(
            "representation candidate is forbidden outside representation stages"
        )
    if (
        completed_epochs <= 0
        or completed_epochs % REPRESENTATION_CANDIDATE_INTERVAL_EPOCHS
    ):
        raise RuntimeError("representation candidate epoch is not registered")
    optimizer_updates = _require_exact_audit_int(
        trainer.formal_optimizer_updates,
        "representation candidate optimizer updates",
    )
    expected_updates = completed_epochs * trainer.train_length
    if optimizer_updates != expected_updates:
        raise RuntimeError("representation candidate update count mismatch")
    filename = (
        f"{trainer.args.formal_stage}_epoch_{completed_epochs:04d}"
        f"_step_{optimizer_updates:09d}.bin"
    )
    path = checkpoint_dir / "representation_candidates" / filename
    audit = {
        "format": "semtalk_show_representation_candidate_v1",
        "formal_stage": trainer.args.formal_stage,
        "completed_epochs": completed_epochs,
        "optimizer_updates": optimizer_updates,
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
        "initialization_receipt": copy.deepcopy(
            trainer.initialization_receipt
        ),
        "rvq_ema_prior_receipt": copy.deepcopy(
            trainer.rvq_ema_prior_receipt
        ),
        "distributed_training_receipt": distributed_training_receipt,
        "rvq_rank_state_receipt": rvq_rank_state_receipt,
        "selection_status": "offline_validation_pending",
    }
    checkpoint_sha256 = _verify_or_write_final(
        path,
        {
            "model_state": trainer.model.state_dict(),
            "audit": audit,
        },
    )
    return {
        "path": str(path),
        "sha256": checkpoint_sha256,
        "completed_epochs": completed_epochs,
        "optimizer_updates": optimizer_updates,
        "selection_status": "offline_validation_pending",
    }


def _validate_representation_candidate_receipt(
    trainer: Any,
    receipt: Any,
    *,
    completed_epochs: int,
) -> dict[str, Any] | None:
    if trainer.args.formal_stage not in REPRESENTATION_STAGES:
        if receipt is not None:
            raise RuntimeError(
                "non-representation resume has a representation candidate"
            )
        return None
    expected_epoch = (
        completed_epochs // REPRESENTATION_CANDIDATE_INTERVAL_EPOCHS
    ) * REPRESENTATION_CANDIDATE_INTERVAL_EPOCHS
    if expected_epoch == 0:
        if receipt is not None:
            raise RuntimeError(
                "representation candidate exists before its first boundary"
            )
        return None
    if not isinstance(receipt, dict):
        raise RuntimeError(
            "resume is missing its latest representation candidate receipt"
        )
    expected_updates = expected_epoch * trainer.train_length
    expected_path = (
        Path(trainer.checkpoint_path)
        / "representation_candidates"
        / (
            f"{trainer.args.formal_stage}_epoch_{expected_epoch:04d}"
            f"_step_{expected_updates:09d}.bin"
        )
    )
    if (
        set(receipt)
        != {
            "path",
            "sha256",
            "completed_epochs",
            "optimizer_updates",
            "selection_status",
        }
        or Path(receipt["path"]).resolve() != expected_path.resolve()
        or _require_exact_audit_int(
            receipt["completed_epochs"],
            "representation candidate completed_epochs",
        )
        != expected_epoch
        or _require_exact_audit_int(
            receipt["optimizer_updates"],
            "representation candidate optimizer_updates",
        )
        != expected_updates
        or receipt["selection_status"] != "offline_validation_pending"
    ):
        raise RuntimeError("invalid latest representation candidate receipt")
    if expected_path.is_symlink() or not expected_path.is_file():
        raise RuntimeError(
            "latest representation candidate checkpoint is unavailable"
        )
    observed_sha256 = _sha256(expected_path)
    if receipt["sha256"] != observed_sha256:
        raise RuntimeError(
            "latest representation candidate checkpoint SHA-256 mismatch"
        )
    return copy.deepcopy(receipt)


def main() -> None:
    args = config.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.local_rank = local_rank
    args.ddp = world_size > 1
    args.gpus = list(range(world_size)) if args.ddp else [0]
    args.skip_test_init = True
    expected_global_batch_size = (
        256 if args.formal_stage in REPRESENTATION_STAGES else 64
    )
    expected_updates_per_epoch = (
        497 if args.formal_stage in REPRESENTATION_STAGES else 1_988
    )
    if args.global_batch_size not in {0, expected_global_batch_size}:
        raise RuntimeError(
            "formal SHOW training has an invalid stage-specific global batch"
        )
    args.global_batch_size = expected_global_batch_size

    if not args.train_only:
        raise RuntimeError("show_base_train.py requires --train_only true")
    if not args.strict_finite:
        raise RuntimeError("formal training requires --strict_finite true")
    if (
        args.expected_train_samples != 127_286
        or args.expected_updates_per_epoch != expected_updates_per_epoch
    ):
        raise RuntimeError(
            "formal training requires exactly 127286 samples and "
            f"{expected_updates_per_epoch} optimizer updates per epoch"
        )
    if not args.lineage_manifest:
        raise RuntimeError("formal training requires --lineage_manifest")
    if not args.dataset_summary:
        raise RuntimeError("formal training requires --dataset_summary")
    if args.save_every <= 0:
        raise RuntimeError("formal training requires positive --save_every")
    if args.debug or args.inference or args.test_state:
        raise RuntimeError("debug, inference, and test_state are forbidden in formal training")
    if not args.run_name:
        raise RuntimeError("--run_name is required for an auditable run")
    if "/" in args.run_name or args.run_name in {".", ".."}:
        raise RuntimeError("--run_name must be a single path component")
    if not args.final_ckpt_name:
        raise RuntimeError("--final_ckpt_name is required")
    if Path(args.final_ckpt_name).name != args.final_ckpt_name:
        raise RuntimeError("--final_ckpt_name must be a basename")
    if args.final_ckpt_name in {
        "latest_resume.pt",
        "formal_training_status.json",
        "base_candidate_manifest.json",
        BASE_CANDIDATE_TRANSACTION_FILENAME,
        BASE_CANDIDATE_STAGING_FILENAME,
    }:
        raise RuntimeError("--final_ckpt_name collides with a reserved artifact")
    _validate_smplx_training_pool_device_matrix(
        args,
        world_size=world_size,
        cuda_available=torch.cuda.is_available(),
        visible_device_count=torch.cuda.device_count(),
    )

    stage_key = (args.model, args.g_name, args.trainer, bool(args.train_rvq))
    allowed_stages = {
        ("rvq", "RVQVAE", "aeface", True),
        ("rvq", "RVQVAE", "ae", True),
        ("rvq", "RVQVAE", "aelower", True),
        ("motion_representation", "VAEConvZero", "aelowerfoot", True),
        ("semtalk", "semtalk_base", "semtalk_base", False),
    }
    if stage_key not in allowed_stages:
        raise RuntimeError(
            "formal SHOW training only permits the five representation models "
            f"and semtalk_base; received {stage_key!r}"
        )
    if args.formal_stage in REPRESENTATION_STAGES:
        if not args.initial_model_checkpoint:
            raise RuntimeError(
                "formal representation fine-tuning requires "
                "--initial-model-checkpoint"
            )
    elif args.initial_model_checkpoint:
        raise RuntimeError(
            "formal Base training forbids --initial-model-checkpoint"
        )
    _validate_formal_stage(args, world_size=world_size)
    lower_target_cache_enabled = validate_lower_target_cache_activation(args)
    if os.environ.get("PYTHONHASHSEED") != str(args.random_seed):
        raise RuntimeError(
            "PYTHONHASHSEED must be exported before launch and match random_seed"
        )

    source_receipt = _source_receipt()
    source_receipt_sha = _payload_sha256(source_receipt)
    initial_smplx_asset_receipt = _formal_smplx_asset_receipt(args)
    lower_target_backend_receipt = (
        _formal_lower_target_backend_receipt(
            args,
            current_source=source_receipt,
            smplx_asset_receipt=initial_smplx_asset_receipt,
        )
    )
    initial_dataset_receipt = _dataset_receipt(
        args,
        train_samples=args.expected_train_samples,
        current_source=source_receipt,
        lower_target_cache_receipt=None,
        lower_target_backend_receipt=lower_target_backend_receipt,
    )
    initial_pool_gate_receipt = initial_dataset_receipt.get(
        SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    if world_size > 1:
        distributed_seed = int(args.random_seed) + rank
        random.seed(distributed_seed)
        np.random.seed(distributed_seed)
        torch.manual_seed(distributed_seed)
        torch.cuda.manual_seed_all(distributed_seed)

    trainer = __import__(
        f"{args.trainer}_trainer", fromlist=["something"]
    ).CustomTrainer(args)
    initialization_receipt = None
    rvq_ema_prior_receipt = None
    if args.formal_stage in REPRESENTATION_STAGES:
        trainable_model = getattr(trainer.model, "module", trainer.model)
        initialization_receipt = load_official_model_state(
            torch,
            stage=args.formal_stage,
            checkpoint=Path(args.initial_model_checkpoint),
            model=trainable_model,
        )
        if args.formal_stage in RVQ_STAGES:
            rvq_ema_prior_receipt = initialize_loaded_rvq_ema(
                trainer.model
            )
        gathered_initialization = [None] * world_size
        dist.all_gather_object(
            gathered_initialization,
            {
                "checkpoint": initialization_receipt,
                "rvq_ema_prior": rvq_ema_prior_receipt,
            },
        )
        if any(
            item != gathered_initialization[0]
            for item in gathered_initialization
        ):
            raise RuntimeError(
                "official initialization receipt differs across ranks"
            )
    trainer.initialization_receipt = copy.deepcopy(
        initialization_receipt
    )
    trainer.rvq_ema_prior_receipt = copy.deepcopy(
        rvq_ema_prior_receipt
    )
    trainer.latest_representation_candidate = None
    _validate_smplx_training_pool_trainer_runtime(
        args,
        trainer,
        gate_receipt=initial_pool_gate_receipt,
    )
    lower_target_cache_receipt = getattr(
        trainer,
        "lower_target_cache_receipt",
        None,
    )
    if lower_target_cache_enabled != (
        lower_target_cache_receipt is not None
    ):
        raise RuntimeError(
            "formal lower target cache activation did not produce exactly "
            "one trainer receipt"
        )
    if (
        args.formal_stage == "lower"
        and getattr(trainer, "lower_target_joints_cache", None) is not None
    ):
        raise RuntimeError(
            "formal lower initialized a cache despite its live backend contract"
        )
    trainer.lower_target_backend_receipt = lower_target_backend_receipt
    lower_target_cache_gate_receipt = (
        _formal_lower_target_cache_gate_receipt(
            args,
            lower_target_cache_receipt=lower_target_cache_receipt,
            current_source=source_receipt,
        )
    )
    if lower_target_cache_receipt is not None:
        if lower_target_cache_gate_receipt is None:
            raise RuntimeError(
                "enabled lower target cache lacks its formal gate receipt"
            )
        lower_target_cache_receipt = copy.deepcopy(
            lower_target_cache_receipt
        )
        previous_receipt_sha = lower_target_cache_receipt.pop(
            "receipt_sha256",
            None,
        )
        if (
            type(previous_receipt_sha) is not str
            or previous_receipt_sha
            != lower_target_cache_receipt_sha256(
                lower_target_cache_receipt
            )
        ):
            raise RuntimeError(
                "lower target cache receipt changed before gate attachment"
            )
        lower_target_cache_receipt["formal_gate"] = (
            lower_target_cache_gate_receipt
        )
        lower_target_cache_receipt["receipt_sha256"] = (
            lower_target_cache_receipt_sha256(
                lower_target_cache_receipt
            )
        )
        trainer.lower_target_cache_receipt = lower_target_cache_receipt
    train_samples = len(trainer.train_data)
    if args.expected_train_samples and train_samples != args.expected_train_samples:
        raise RuntimeError(
            f"train sample count {train_samples} != expected {args.expected_train_samples}"
        )
    updates_per_epoch = trainer.train_length
    if train_samples <= 0 or updates_per_epoch <= 0:
        raise RuntimeError("train dataset and dataloader must both be non-empty")
    if (
        args.expected_updates_per_epoch
        and updates_per_epoch != args.expected_updates_per_epoch
    ):
        raise RuntimeError(
            f"updates/epoch {updates_per_epoch} != expected "
            f"{args.expected_updates_per_epoch}"
        )
    distributed_training_receipt = _trainer_distributed_receipt(
        trainer,
        world_size=world_size,
    )
    rvq_rank_state_receipt = (
        assert_rvq_rank_state(trainer.model)
        if args.formal_stage in RVQ_STAGES
        else None
    )
    dataset_receipt = _dataset_receipt(
        args,
        train_samples=train_samples,
        current_source=source_receipt,
        lower_target_cache_receipt=lower_target_cache_receipt,
        lower_target_backend_receipt=lower_target_backend_receipt,
    )
    if dataset_receipt != initial_dataset_receipt:
        raise RuntimeError(
            "formal dataset/gate receipt changed while initializing trainer"
        )

    checkpoint_dir = Path(trainer.checkpoint_path)
    status_path = checkpoint_dir / "formal_training_status.json"
    resume_path = checkpoint_dir / "latest_resume.pt"
    final_path = checkpoint_dir / args.final_ckpt_name
    candidate_manifest_path = checkpoint_dir / "base_candidate_manifest.json"
    lineage_sha = None
    if args.lineage_manifest:
        lineage_path = Path(args.lineage_manifest)
        if not lineage_path.is_file():
            raise FileNotFoundError(lineage_path)
        lineage_sha = _sha256(lineage_path)
    config_sha = _config_fingerprint(args)

    start_epoch = 0
    started_at = time.time()
    last_metrics: dict[str, dict[str, float | int]] = {}
    candidate_manifest_sha: str | None = None
    current_candidate_count = 0
    active_candidate_transaction: dict[str, Any] | None = None
    requested_resume = Path(args.resume_state) if args.resume_state else None
    if requested_resume:
        if requested_resume.is_symlink() or not requested_resume.is_file():
            raise FileNotFoundError(requested_resume)
        (
            start_epoch,
            last_metrics,
            started_at,
            candidate_manifest_sha,
            current_candidate_count,
            active_candidate_transaction,
        ) = _load_resume(
                trainer,
                requested_resume,
                rank=rank,
                world_size=world_size,
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_summary_sha256=dataset_receipt["summary_sha256"],
                data_mdb_sha256=dataset_receipt["data_mdb_sha256"],
                dataset_receipt=dataset_receipt,
                source_receipt=source_receipt,
                source_receipt_sha256=source_receipt_sha,
                candidate_manifest_path=candidate_manifest_path,
        )
    else:
        active_candidate_transaction = _inspect_base_candidate_transaction(
            candidate_manifest_path,
            formal_stage=args.formal_stage,
            updates_per_epoch=updates_per_epoch,
            config_sha256=config_sha,
            lineage_sha256=lineage_sha,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        if active_candidate_transaction is not None:
            raise RuntimeError(
                "an uncommitted Base candidate transaction requires "
                "--resume_state"
            )
        current_candidate_count = len(
            _base_candidate_entries(candidate_manifest_path)
        )
        candidate_manifest_sha = _validate_base_candidate_manifest(
            candidate_manifest_path,
            formal_stage=args.formal_stage,
            completed_epochs=(
                current_candidate_count * BASE_CANDIDATE_INTERVAL_EPOCHS
            ),
            updates_per_epoch=updates_per_epoch,
            config_sha256=config_sha,
            lineage_sha256=lineage_sha,
            dataset_receipt=dataset_receipt,
            source_receipt_sha256=source_receipt_sha,
        )
    smplx_runtime_evidence = copy.deepcopy(
        getattr(
            trainer,
            "smplx_training_pool_runtime_evidence",
            None,
        )
    )
    if requested_resume is None:
        trainer.smplx_training_pool_runtime_evidence = None
    if start_epoch < 0 or start_epoch > args.epochs:
        raise RuntimeError(f"invalid resume epoch {start_epoch}")
    expected_start_updates = start_epoch * updates_per_epoch
    if trainer.formal_optimizer_updates != expected_start_updates:
        raise RuntimeError("initial actual optimizer update count mismatch")
    committed_candidate_count = (
        start_epoch // BASE_CANDIDATE_INTERVAL_EPOCHS
        if args.formal_stage == "base"
        else 0
    )
    if (
        active_candidate_transaction is not None
        and _require_exact_audit_int(
            active_candidate_transaction["target_epoch"],
            "active candidate transaction target_epoch",
        )
        == start_epoch
    ):
        _clear_committed_base_candidate_transaction(
            candidate_manifest_path,
            completed_epochs=start_epoch,
            manifest_sha256=candidate_manifest_sha,
        )
        active_candidate_transaction = None
    replay_candidate_epoch = None
    if active_candidate_transaction is not None:
        replay_candidate_epoch = _require_exact_audit_int(
            active_candidate_transaction["target_epoch"],
            "active candidate transaction target_epoch",
        )
    elif current_candidate_count != committed_candidate_count:
        raise RuntimeError("initial Base candidate entry count mismatch")
    if args.formal_stage in RVQ_STAGES:
        rvq_rank_state_receipt = assert_rvq_rank_state(trainer.model)
    candidate_manifest_receipt = _base_candidate_manifest_receipt(
        candidate_manifest_path,
        candidate_manifest_sha,
        completed_epochs=(
            current_candidate_count * BASE_CANDIDATE_INTERVAL_EPOCHS
        ),
        updates_per_epoch=updates_per_epoch,
    )

    if rank == 0:
        _atomic_json(
            status_path,
            {
                "status": "running",
                "run_name": args.run_name,
                "model": args.g_name,
                "trainer": args.trainer,
                "formal_stage": args.formal_stage,
                "smplx_training_pool_mode": args.smplx_training_pool_mode,
                "epochs": args.epochs,
                "start_epoch": start_epoch,
                "world_size": world_size,
                "train_samples": train_samples,
                "updates_per_epoch": updates_per_epoch,
                "optimizer_updates": trainer.formal_optimizer_updates,
                "batch_size": args.batch_size,
                "distributed_training_receipt": (
                    distributed_training_receipt
                ),
                "rvq_rank_state_receipt": rvq_rank_state_receipt,
                "initialization_receipt": initialization_receipt,
                "rvq_ema_prior_receipt": rvq_ema_prior_receipt,
                "latest_representation_candidate": (
                    trainer.latest_representation_candidate
                ),
                "lineage_manifest_sha256": lineage_sha,
                "dataset_receipt": dataset_receipt,
                "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
                **_smplx_training_pool_gate_overlay(
                    dataset_receipt.get(
                        SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                    )
                ),
                **_smplx_training_pool_runtime_evidence_overlay(
                    smplx_runtime_evidence
                ),
                **(
                    {
                        LOWER_TARGET_CACHE_RECEIPT_KEY:
                        lower_target_cache_receipt
                    }
                    if lower_target_cache_receipt is not None
                    else {}
                ),
                **_lower_target_backend_overlay(
                    lower_target_backend_receipt
                ),
                "base_candidate_manifest": candidate_manifest_receipt,
                "source_receipt": source_receipt,
                "source_receipt_sha256": source_receipt_sha,
                "config_sha256": config_sha,
                "argv": sys.argv,
                "started_unix": started_at,
                "resumed_unix": (
                    time.time() if requested_resume is not None else None
                ),
            },
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            if args.ddp:
                trainer.train_loader.sampler.set_epoch(epoch)
            trainer.tracker.reset()
            updates_before_epoch = trainer.formal_optimizer_updates
            trainer.train(epoch)
            updates_after_epoch = trainer.formal_optimizer_updates
            if updates_after_epoch - updates_before_epoch != updates_per_epoch:
                raise RuntimeError(
                    "formal epoch actual optimizer update delta mismatch: "
                    f"epoch={epoch} before={updates_before_epoch} "
                    f"after={updates_after_epoch} expected_delta={updates_per_epoch}"
                )
            observed_runtime_evidence = (
                _current_smplx_training_pool_runtime_evidence(
                    trainer,
                    formal_stage=args.formal_stage,
                    gate_receipt=dataset_receipt.get(
                        SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                    ),
                )
            )
            pool_gate_receipt = dataset_receipt.get(
                SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
            )
            if (
                isinstance(pool_gate_receipt, dict)
                and pool_gate_receipt.get("mode") == "target_offload"
            ):
                if not isinstance(observed_runtime_evidence, dict):
                    raise RuntimeError(
                        "target_offload runtime evidence is unavailable"
                    )
                observed_pairs = _require_exact_audit_int(
                    observed_runtime_evidence["pool_runtime"].get(
                        "completed_forward_pairs"
                    ),
                    "target_offload completed forward pairs after epoch",
                )
                previous_pairs = 0
                if smplx_runtime_evidence is not None:
                    previous_pairs = _require_exact_audit_int(
                        smplx_runtime_evidence["pool_runtime"].get(
                            "completed_forward_pairs"
                        ),
                        "target_offload completed forward pairs before epoch",
                    )
                    previous_stable = copy.deepcopy(
                        smplx_runtime_evidence
                    )
                    observed_stable = copy.deepcopy(
                        observed_runtime_evidence
                    )
                    previous_stable.pop("receipt_sha256")
                    observed_stable.pop("receipt_sha256")
                    previous_stable["pool_runtime"].pop(
                        "completed_forward_pairs"
                    )
                    observed_stable["pool_runtime"].pop(
                        "completed_forward_pairs"
                    )
                    if previous_stable != observed_stable:
                        raise RuntimeError(
                            "production target_offload runtime evidence "
                            "changed outside its forward counter"
                        )
                if (
                    previous_pairs != updates_before_epoch
                    or observed_pairs != updates_after_epoch
                    or observed_pairs - previous_pairs
                    != updates_per_epoch
                ):
                    raise RuntimeError(
                        "target_offload forward-count delta differs from "
                        "optimizer-update delta"
                    )
                _validate_smplx_training_pool_runtime_evidence(
                    observed_runtime_evidence,
                    formal_stage=args.formal_stage,
                    gate_receipt=pool_gate_receipt,
                    expected_optimizer_updates=updates_after_epoch,
                )
            elif (
                smplx_runtime_evidence is not None
                and observed_runtime_evidence != smplx_runtime_evidence
            ):
                raise RuntimeError(
                    "production SMPL-X pool runtime evidence changed"
                )
            smplx_runtime_evidence = observed_runtime_evidence
            trainer.smplx_training_pool_runtime_evidence = copy.deepcopy(
                smplx_runtime_evidence
            )
            last_metrics = _tracker_snapshot(trainer)

            if args.strict_finite:
                bad = _finite_tree(trainer.model.state_dict(), "model")
                bad.extend(_finite_tree(_rvq_ema_state(trainer.model), "rvq_ema"))
                bad.extend(_finite_tree(trainer.opt.state_dict(), "optimizer"))
                bad.extend(_finite_tree(trainer.opt_s.state_dict(), "scheduler"))
                bad.extend(_finite_tree(last_metrics, "metrics"))
                bad.extend(_rvq_ema_invariant_errors(trainer.model))
                if bad:
                    raise FloatingPointError(
                        "non-finite training state: " + ", ".join(bad[:20])
                    )
            if args.formal_stage in RVQ_STAGES:
                rvq_rank_state_receipt = assert_rvq_rank_state(
                    trainer.model
                )

            completed_epochs = epoch + 1
            expected_total_updates = completed_epochs * updates_per_epoch
            if updates_after_epoch != expected_total_updates:
                raise RuntimeError(
                    "formal cumulative actual optimizer update count mismatch"
                )
            if (
                args.formal_stage in REPRESENTATION_STAGES
                and completed_epochs
                % REPRESENTATION_CANDIDATE_INTERVAL_EPOCHS
                == 0
            ):
                candidate_result: list[dict[str, Any] | None] = [None]
                if rank == 0:
                    try:
                        candidate_result[0] = {
                            "ok": True,
                            "receipt": _save_representation_candidate(
                                trainer,
                                checkpoint_dir,
                                completed_epochs=completed_epochs,
                                config_sha256=config_sha,
                                lineage_sha256=lineage_sha,
                                dataset_receipt=dataset_receipt,
                                source_receipt=source_receipt,
                                distributed_training_receipt=(
                                    distributed_training_receipt
                                ),
                                rvq_rank_state_receipt=(
                                    rvq_rank_state_receipt
                                ),
                            ),
                        }
                    except Exception as candidate_error:
                        candidate_result[0] = {
                            "ok": False,
                            "error_type": type(candidate_error).__name__,
                            "error": str(candidate_error),
                        }
                if args.ddp:
                    dist.broadcast_object_list(candidate_result, src=0)
                candidate_message = candidate_result[0]
                if (
                    not isinstance(candidate_message, dict)
                    or candidate_message.get("ok") is not True
                ):
                    raise RuntimeError(
                        "rank-zero representation candidate save failed: "
                        f"{candidate_message!r}"
                    )
                trainer.latest_representation_candidate = copy.deepcopy(
                    candidate_message["receipt"]
                )
            if (
                args.formal_stage == "base"
                and completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS == 0
            ):
                candidate_manifest_receipt = _save_base_candidate(
                    trainer,
                    candidate_manifest_path,
                    completed_epochs=completed_epochs,
                    updates_per_epoch=updates_per_epoch,
                    config_sha256=config_sha,
                    lineage_sha256=lineage_sha,
                    dataset_receipt=dataset_receipt,
                    source_receipt=source_receipt,
                    expected_manifest_sha256=candidate_manifest_sha,
                )
                candidate_manifest_sha = candidate_manifest_receipt["sha256"]
                current_candidate_count = _require_exact_audit_int(
                    candidate_manifest_receipt["entries"],
                    "Base candidate manifest receipt entries",
                )
            else:
                expected_candidate_count = (
                    completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS
                    if args.formal_stage == "base"
                    else 0
                )
                allowed_pending_checkpoint = None
                allow_empty_candidate_dir = False
                if (
                    replay_candidate_epoch is not None
                    and completed_epochs < replay_candidate_epoch
                ):
                    assert active_candidate_transaction is not None
                    expected_candidate_count = _require_exact_audit_int(
                        active_candidate_transaction["current_entry_count"],
                        "active candidate transaction current_entry_count",
                    )
                    if (
                        active_candidate_transaction["candidate_exists"]
                        and not active_candidate_transaction["manifested"]
                    ):
                        allowed_pending_checkpoint = (
                            active_candidate_transaction["candidate_path"]
                        )
                    allow_empty_candidate_dir = (
                        not active_candidate_transaction["candidate_exists"]
                        and not active_candidate_transaction["manifested"]
                    )
                current_candidate_entries = _validate_base_candidate_snapshot(
                    candidate_manifest_path,
                    expected_sha256=candidate_manifest_sha,
                    expected_entry_count=expected_candidate_count,
                    allowed_pending_checkpoint=allowed_pending_checkpoint,
                    allow_empty_candidate_dir=allow_empty_candidate_dir,
                )
                candidate_manifest_receipt = _base_candidate_manifest_receipt(
                    candidate_manifest_path,
                    candidate_manifest_sha,
                    completed_epochs=(
                        len(current_candidate_entries)
                        * BASE_CANDIDATE_INTERVAL_EPOCHS
                    ),
                    updates_per_epoch=updates_per_epoch,
                )
            should_save = (
                completed_epochs == args.epochs
                or (args.save_every > 0 and completed_epochs % args.save_every == 0)
            )
            if should_save:
                rng_states = _all_rng_states(world_size)
                if rank == 0:
                    assert rng_states is not None
                    committed_candidate_entries = _base_candidate_entries(
                        candidate_manifest_path
                    )
                    _save_resume(
                        trainer,
                        resume_path,
                        completed_epochs=completed_epochs,
                        rng_states=rng_states,
                        world_size=world_size,
                        train_samples=train_samples,
                        updates_per_epoch=updates_per_epoch,
                        config_sha256=config_sha,
                        lineage_sha256=lineage_sha,
                        dataset_summary_sha256=dataset_receipt[
                            "summary_sha256"
                        ],
                        data_mdb_sha256=dataset_receipt[
                            "data_mdb_sha256"
                        ],
                        dataset_receipt=dataset_receipt,
                        source_receipt_sha256=source_receipt_sha,
                        optimizer_updates=trainer.formal_optimizer_updates,
                        candidate_manifest_sha256=candidate_manifest_sha,
                        candidate_manifest_entry_count=len(
                            committed_candidate_entries
                        ),
                        candidate_manifest_entries_sha256=(
                            _base_candidate_entries_sha256(
                                committed_candidate_entries
                            )
                        ),
                        last_metrics=last_metrics,
                        started_unix=started_at,
                        distributed_training_receipt=(
                            distributed_training_receipt
                        ),
                        rvq_rank_state_receipt=rvq_rank_state_receipt,
                    )
                    if (
                        args.formal_stage == "base"
                        and completed_epochs
                        % BASE_CANDIDATE_INTERVAL_EPOCHS
                        == 0
                    ):
                        _clear_committed_base_candidate_transaction(
                            candidate_manifest_path,
                            completed_epochs=completed_epochs,
                            manifest_sha256=candidate_manifest_sha,
                        )
                        active_candidate_transaction = None
                        replay_candidate_epoch = None
                    _atomic_json(
                        status_path,
                        {
                            "status": "running",
                            "run_name": args.run_name,
                            "model": args.g_name,
                            "trainer": args.trainer,
                            "formal_stage": args.formal_stage,
                            "smplx_training_pool_mode": (
                                args.smplx_training_pool_mode
                            ),
                            "epochs": args.epochs,
                            "completed_epochs": completed_epochs,
                            "world_size": world_size,
                            "train_samples": train_samples,
                            "updates_per_epoch": updates_per_epoch,
                            "batch_size": args.batch_size,
                            "distributed_training_receipt": (
                                distributed_training_receipt
                            ),
                            "rvq_rank_state_receipt": (
                                rvq_rank_state_receipt
                            ),
                            "initialization_receipt": (
                                initialization_receipt
                            ),
                            "rvq_ema_prior_receipt": (
                                rvq_ema_prior_receipt
                            ),
                            "latest_representation_candidate": (
                                trainer.latest_representation_candidate
                            ),
                            "optimizer_updates": trainer.formal_optimizer_updates,
                            "lineage_manifest_sha256": lineage_sha,
                            "dataset_receipt": dataset_receipt,
                            "smplx_asset_receipt": dataset_receipt.get(
                                "smplx_asset"
                            ),
                            **_smplx_training_pool_gate_overlay(
                                dataset_receipt.get(
                                    SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                                )
                            ),
                            **_smplx_training_pool_runtime_evidence_overlay(
                                smplx_runtime_evidence
                            ),
                            **(
                                {
                                    LOWER_TARGET_CACHE_RECEIPT_KEY:
                                    lower_target_cache_receipt
                                }
                                if lower_target_cache_receipt is not None
                                else {}
                            ),
                            **_lower_target_backend_overlay(
                                lower_target_backend_receipt
                            ),
                            "base_candidate_manifest": (
                                candidate_manifest_receipt
                            ),
                            "source_receipt": source_receipt,
                            "source_receipt_sha256": source_receipt_sha,
                            "config_sha256": config_sha,
                            "latest_resume_sha256": _sha256(resume_path),
                            "last_metrics": last_metrics,
                            "argv": sys.argv,
                            "started_unix": started_at,
                            "updated_unix": time.time(),
                        },
                    )
            if args.ddp:
                dist.barrier()

        if rank == 0:
            bad = _finite_tree(trainer.model.state_dict(), "model")
            bad.extend(_finite_tree(_rvq_ema_state(trainer.model), "rvq_ema"))
            bad.extend(_rvq_ema_invariant_errors(trainer.model))
            if bad:
                raise FloatingPointError(
                    "non-finite final model state: " + ", ".join(bad[:20])
                )
            final_source_receipt = _source_receipt()
            if final_source_receipt != source_receipt:
                raise RuntimeError("formal source changed during training")
            final_dataset_receipt = _dataset_receipt(
                args,
                train_samples=train_samples,
                current_source=final_source_receipt,
                lower_target_cache_receipt=lower_target_cache_receipt,
                lower_target_backend_receipt=(
                    lower_target_backend_receipt
                ),
            )
            if final_dataset_receipt != dataset_receipt:
                raise RuntimeError(
                    "formal dataset/parity assets changed during training"
                )
            expected_final_updates = args.epochs * updates_per_epoch
            if trainer.formal_optimizer_updates != expected_final_updates:
                raise RuntimeError(
                    "formal final actual optimizer update count mismatch"
                )
            if (
                _candidate_transaction_path(candidate_manifest_path).exists()
                or _candidate_transaction_path(
                    candidate_manifest_path
                ).is_symlink()
                or _candidate_staging_path(candidate_manifest_path).exists()
                or _candidate_staging_path(
                    candidate_manifest_path
                ).is_symlink()
            ):
                raise RuntimeError(
                    "Base candidate transaction/staging remains at final audit"
                )
            observed_candidate_manifest_sha = (
                _validate_base_candidate_manifest(
                candidate_manifest_path,
                formal_stage=args.formal_stage,
                completed_epochs=args.epochs,
                updates_per_epoch=updates_per_epoch,
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_receipt=dataset_receipt,
                source_receipt_sha256=source_receipt_sha,
                )
            )
            _require_trusted_candidate_manifest_sha(
                observed_candidate_manifest_sha,
                candidate_manifest_sha,
            )
            candidate_manifest_receipt = _base_candidate_manifest_receipt(
                candidate_manifest_path,
                observed_candidate_manifest_sha,
                completed_epochs=args.epochs,
                updates_per_epoch=updates_per_epoch,
            )
            assert lineage_sha is not None
            final_sha = _verify_or_write_final(
                final_path,
                _model_payload(
                    trainer,
                    formal_stage=args.formal_stage,
                    config_sha256=config_sha,
                    lineage_sha256=lineage_sha,
                    dataset_receipt=dataset_receipt,
                    source_receipt=source_receipt,
                    optimizer_updates=trainer.formal_optimizer_updates,
                    candidate_manifest_receipt=(
                        candidate_manifest_receipt
                    ),
                    distributed_training_receipt=(
                        distributed_training_receipt
                    ),
                    rvq_rank_state_receipt=rvq_rank_state_receipt,
                ),
            )
            _atomic_json(
                status_path,
                {
                    "status": "complete",
                    "run_name": args.run_name,
                    "model": args.g_name,
                    "trainer": args.trainer,
                    "formal_stage": args.formal_stage,
                    "smplx_training_pool_mode": (
                        args.smplx_training_pool_mode
                    ),
                    "epochs": args.epochs,
                    "completed_epochs": args.epochs,
                    "world_size": world_size,
                    "train_samples": train_samples,
                    "updates_per_epoch": updates_per_epoch,
                    "batch_size": args.batch_size,
                    "distributed_training_receipt": (
                        distributed_training_receipt
                    ),
                    "rvq_rank_state_receipt": rvq_rank_state_receipt,
                    "initialization_receipt": initialization_receipt,
                    "rvq_ema_prior_receipt": rvq_ema_prior_receipt,
                    "latest_representation_candidate": (
                        trainer.latest_representation_candidate
                    ),
                    "optimizer_updates": trainer.formal_optimizer_updates,
                    "lineage_manifest_sha256": lineage_sha,
                    "dataset_receipt": dataset_receipt,
                    "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
                    **_smplx_training_pool_gate_overlay(
                        dataset_receipt.get(
                            SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                        )
                    ),
                    **_smplx_training_pool_runtime_evidence_overlay(
                        smplx_runtime_evidence
                    ),
                    **(
                        {
                            LOWER_TARGET_CACHE_RECEIPT_KEY:
                            lower_target_cache_receipt
                        }
                        if lower_target_cache_receipt is not None
                        else {}
                    ),
                    **_lower_target_backend_overlay(
                        lower_target_backend_receipt
                    ),
                    "base_candidate_manifest": candidate_manifest_receipt,
                    "source_receipt": source_receipt,
                    "source_receipt_sha256": source_receipt_sha,
                    "config_sha256": config_sha,
                    "final_checkpoint": str(final_path),
                    "final_checkpoint_sha256": final_sha,
                    "latest_resume_sha256": _sha256(resume_path),
                    "last_metrics": last_metrics,
                    "argv": sys.argv,
                    "started_unix": started_at,
                    "completed_unix": time.time(),
                },
            )
            logger.info(f"strict train-only finalize PASS: {final_path} {final_sha}")
        dist.barrier()
    except BaseException as exc:
        if rank == 0:
            _atomic_json(
                status_path,
                {
                    "status": "failed",
                    "run_name": args.run_name,
                    "model": args.g_name,
                    "trainer": args.trainer,
                    "formal_stage": args.formal_stage,
                    "smplx_training_pool_mode": (
                        args.smplx_training_pool_mode
                    ),
                    "epochs": args.epochs,
                    "world_size": world_size,
                    "train_samples": train_samples,
                    "updates_per_epoch": updates_per_epoch,
                    "optimizer_updates": trainer.formal_optimizer_updates,
                    "batch_size": args.batch_size,
                    "distributed_training_receipt": (
                        distributed_training_receipt
                    ),
                    "rvq_rank_state_receipt": rvq_rank_state_receipt,
                    "initialization_receipt": initialization_receipt,
                    "rvq_ema_prior_receipt": rvq_ema_prior_receipt,
                    "latest_representation_candidate": (
                        trainer.latest_representation_candidate
                    ),
                    "lineage_manifest_sha256": lineage_sha,
                    "dataset_receipt": dataset_receipt,
                    "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
                    **_smplx_training_pool_gate_overlay(
                        dataset_receipt.get(
                            SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                        )
                    ),
                    **_smplx_training_pool_runtime_evidence_overlay(
                        smplx_runtime_evidence
                    ),
                    **(
                        {
                            LOWER_TARGET_CACHE_RECEIPT_KEY:
                            lower_target_cache_receipt
                        }
                        if lower_target_cache_receipt is not None
                        else {}
                    ),
                    **_lower_target_backend_overlay(
                        lower_target_backend_receipt
                    ),
                    "base_candidate_manifest": candidate_manifest_receipt,
                    "source_receipt": source_receipt,
                    "source_receipt_sha256": source_receipt_sha,
                    "config_sha256": config_sha,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "argv": sys.argv,
                    "started_unix": started_at,
                    "failed_unix": time.time(),
                },
            )
        raise
    finally:
        if rank == 0 and getattr(trainer, "writer", None) is not None:
            trainer.writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
