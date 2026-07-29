#!/usr/bin/env python3
"""Strict train-only entry point for the SHOW SemTalk Base reproduction.

This deliberately bypasses SemTalk's stock test-during-training loop.  It keeps
the model, trainer, optimizer, scheduler, and epoch update semantics intact,
while adding resumable checkpoints and finite-value audits.  Launch it with
``torchrun`` even for a one-GPU run so rank handling is unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
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


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
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


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _dataset_receipt(
    args: Any,
    *,
    train_samples: int,
    current_source: dict[str, str],
) -> dict[str, Any]:
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
        or int(summary.get("entries", -1)) != train_samples
        or train_samples != 127_309
        or int(summary.get("train_clips", -1)) != 13687
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
            or int(lineage.get("entries", -1)) != train_samples
            or int(lineage.get("train_clips", -1)) != 13687
            or not isinstance(protocol, dict)
            or protocol.get("scope") != "SemTalk Base only"
            or protocol.get("split") != "train"
            or protocol.get("speakers") != expected_speakers
            or int(protocol.get("window_length", -1)) != 64
            or int(protocol.get("stride", -1)) != 20
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
            or int(representation_protocol.get("window_length", -1)) != 64
            or int(representation_protocol.get("stride", -1)) != 20
            or representation_protocol.get("speaker_map")
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
        if {
            key: summary.get("source_receipt", {}).get(key)
            for key in ("origin", "commit", "tree")
        } != {
            key: current_source.get(key)
            for key in ("origin", "commit", "tree")
        }:
            raise RuntimeError(
                "representation/training source receipt mismatch"
            )
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
            or parity.get("speakers") != expected_speakers
            or parity.get("canonical_receipt")
            != summary.get("canonical_receipt")
            or {
                key: parity.get("source_receipt", {}).get(key)
                for key in ("origin", "commit", "tree")
            }
            != {
                key: current_source.get(key)
                for key in ("origin", "commit", "tree")
            }
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
            or Path(str(parity.get("checker", ""))).resolve()
            != checker_path
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
        if (
            not isinstance(reports, list)
            or len(reports) != 4
            or {
                (record.get("speaker"), record.get("speaker_id"))
                for record in reports
                if isinstance(record, dict)
            }
            != set(expected_speakers.items())
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
                if (
                    row.get("split") == "train"
                    and speaker in expected_speakers
                    and int(row.get("speaker_id", -1))
                    == expected_speakers[speaker]
                    and int(row.get("frames", -1)) >= 64
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
    return {
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
        "global_fastpath_parity": parity_receipt,
    }


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


def _rvq_ema_state(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    """Capture rank-local EMA state omitted by QuantizeEMAReset.state_dict()."""
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
        initialized = bool(item["init"])
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


def _validate_formal_stage(args: Any) -> None:
    common = {
        "dataset": "show_base",
        "training_speakers": [0, 1, 2, 3],
        "ori_joints": "beat_smplx_joints",
        "batch_size": 64,
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
        "log_period": 1_989,
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
            "epochs": 600,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "rvq_face_600.bin",
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
            "epochs": 500,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "rvq_hands_500.bin",
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
            "epochs": 500,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 9999,
            "final_ckpt_name": "rvq_upper_500.bin",
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
            "epochs": 600,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "rvq_lower_600.bin",
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
            "epochs": 1700,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "last_1700_foot.bin",
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
            "formal SHOW training must start from scratch; --load_ckpt is forbidden"
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
    source_receipt_sha256: str,
) -> tuple[int, dict[str, dict[str, float | int]], float]:
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "semtalk_show_train_resume_v4":
        raise RuntimeError("unsupported or unsafe resume checkpoint format")
    if int(payload["world_size"]) != world_size:
        raise RuntimeError(
            f"resume world_size={payload['world_size']} does not match {world_size}"
        )
    if int(payload["train_samples"]) != len(trainer.train_data):
        raise RuntimeError("resume train sample count does not match current dataset")
    if int(payload["updates_per_epoch"]) != trainer.train_length:
        raise RuntimeError("resume updates/epoch does not match current dataloader")
    if int(payload["batch_size"]) != trainer.args.batch_size:
        raise RuntimeError("resume batch size does not match current config")
    if payload["config_sha256"] != config_sha256:
        raise RuntimeError("resume training config fingerprint does not match")
    if payload.get("lineage_manifest_sha256") != lineage_sha256:
        raise RuntimeError("resume lineage manifest fingerprint does not match")
    if payload.get("dataset_summary_sha256") != dataset_summary_sha256:
        raise RuntimeError("resume dataset summary fingerprint does not match")
    if payload.get("data_mdb_sha256") != data_mdb_sha256:
        raise RuntimeError("resume LMDB fingerprint does not match")
    if payload.get("source_receipt_sha256") != source_receipt_sha256:
        raise RuntimeError("resume source checkout fingerprint does not match")
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    ema_modules = _rvq_ema_state(trainer.model)
    if ema_modules:
        if "rvq_ema_state" not in payload:
            raise RuntimeError("RVQ resume checkpoint is missing EMA state")
        _restore_rvq_ema_state(trainer.model, payload["rvq_ema_state"])
    elif payload.get("rvq_ema_state"):
        raise RuntimeError("non-RVQ model received unexpected RVQ EMA state")
    trainer.opt.load_state_dict(payload["optimizer_state"])
    trainer.opt_s.load_state_dict(payload["scheduler_state"])
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
    started_unix = float(payload.get("started_unix", 0.0))
    if not np.isfinite(started_unix) or started_unix <= 0:
        raise RuntimeError("resume checkpoint has invalid started_unix")
    return int(payload["completed_epochs"]), last_metrics, started_unix


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
    source_receipt_sha256: str,
    last_metrics: dict[str, dict[str, float | int]],
    started_unix: float,
) -> None:
    _atomic_torch_save(
        path,
        {
            "format": "semtalk_show_train_resume_v4",
            "completed_epochs": completed_epochs,
            "world_size": world_size,
            "train_samples": train_samples,
            "updates_per_epoch": updates_per_epoch,
            "batch_size": trainer.args.batch_size,
            "config_sha256": config_sha256,
            "lineage_manifest_sha256": lineage_sha256,
            "dataset_summary_sha256": dataset_summary_sha256,
            "data_mdb_sha256": data_mdb_sha256,
            "source_receipt_sha256": source_receipt_sha256,
            "last_metrics": last_metrics,
            "started_unix": started_unix,
            "model_state": trainer.model.state_dict(),
            "rvq_ema_state": _rvq_ema_state(trainer.model),
            "optimizer_state": trainer.opt.state_dict(),
            "scheduler_state": trainer.opt_s.state_dict(),
            "rng_states": rng_states,
        },
    )


def _model_payload(
    trainer: Any,
    *,
    formal_stage: str,
    config_sha256: str,
    lineage_sha256: str,
    dataset_summary_sha256: str,
    data_mdb_sha256: str,
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    return {
        "model_state": trainer.model.state_dict(),
        "audit": {
            "format": "semtalk_show_model_v1",
            "formal_stage": formal_stage,
            "config_sha256": config_sha256,
            "lineage_manifest_sha256": lineage_sha256,
            "dataset_summary_sha256": dataset_summary_sha256,
            "data_mdb_sha256": data_mdb_sha256,
            "source_receipt": source_receipt,
            "source_receipt_sha256": _payload_sha256(source_receipt),
        },
    }


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


def main() -> None:
    args = config.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.local_rank = local_rank
    args.ddp = world_size > 1
    args.gpus = list(range(world_size)) if args.ddp else [0]
    args.skip_test_init = True

    if not args.train_only:
        raise RuntimeError("show_base_train.py requires --train_only true")
    if not args.strict_finite:
        raise RuntimeError("formal training requires --strict_finite true")
    if (
        args.expected_train_samples != 127_309
        or args.expected_updates_per_epoch != 1_989
    ):
        raise RuntimeError(
            "formal training requires exactly 127309 samples and "
            "1989 optimizer updates per epoch"
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
    if args.final_ckpt_name in {"latest_resume.pt", "formal_training_status.json"}:
        raise RuntimeError("--final_ckpt_name collides with a reserved artifact")
    if world_size != 1:
        raise RuntimeError(
            "formal SHOW reproduction training is single-GPU per model: RVQ EMA "
            "is rank-local and Base has unused branches plus multiple forwards"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each formal model task must receive exactly one isolated visible GPU"
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
    _validate_formal_stage(args)
    if os.environ.get("PYTHONHASHSEED") != str(args.random_seed):
        raise RuntimeError(
            "PYTHONHASHSEED must be exported before launch and match random_seed"
        )

    source_receipt = _source_receipt()
    source_receipt_sha = _payload_sha256(source_receipt)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)

    trainer = __import__(
        f"{args.trainer}_trainer", fromlist=["something"]
    ).CustomTrainer(args)
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
    dataset_receipt = _dataset_receipt(
        args,
        train_samples=train_samples,
        current_source=source_receipt,
    )

    checkpoint_dir = Path(trainer.checkpoint_path)
    status_path = checkpoint_dir / "formal_training_status.json"
    resume_path = checkpoint_dir / "latest_resume.pt"
    final_path = checkpoint_dir / args.final_ckpt_name
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
    requested_resume = Path(args.resume_state) if args.resume_state else None
    if requested_resume:
        if requested_resume.is_symlink() or not requested_resume.is_file():
            raise FileNotFoundError(requested_resume)
        start_epoch, last_metrics, started_at = _load_resume(
            trainer,
            requested_resume,
            rank=rank,
            world_size=world_size,
            config_sha256=config_sha,
            lineage_sha256=lineage_sha,
            dataset_summary_sha256=dataset_receipt["summary_sha256"],
            data_mdb_sha256=dataset_receipt["data_mdb_sha256"],
            source_receipt_sha256=source_receipt_sha,
        )
    if start_epoch < 0 or start_epoch > args.epochs:
        raise RuntimeError(f"invalid resume epoch {start_epoch}")

    if rank == 0:
        _atomic_json(
            status_path,
            {
                "status": "running",
                "run_name": args.run_name,
                "model": args.g_name,
                "trainer": args.trainer,
                "formal_stage": args.formal_stage,
                "epochs": args.epochs,
                "start_epoch": start_epoch,
                "world_size": world_size,
                "train_samples": train_samples,
                "updates_per_epoch": updates_per_epoch,
                "batch_size": args.batch_size,
                "lineage_manifest_sha256": lineage_sha,
                "dataset_receipt": dataset_receipt,
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
            trainer.train(epoch)
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

            completed_epochs = epoch + 1
            should_save = (
                completed_epochs == args.epochs
                or (args.save_every > 0 and completed_epochs % args.save_every == 0)
            )
            if should_save:
                rng_states = _all_rng_states(world_size)
                if rank == 0:
                    assert rng_states is not None
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
                        source_receipt_sha256=source_receipt_sha,
                        last_metrics=last_metrics,
                        started_unix=started_at,
                    )
                    _atomic_json(
                        status_path,
                        {
                            "status": "running",
                            "run_name": args.run_name,
                            "model": args.g_name,
                            "trainer": args.trainer,
                            "formal_stage": args.formal_stage,
                            "epochs": args.epochs,
                            "completed_epochs": completed_epochs,
                            "world_size": world_size,
                            "train_samples": train_samples,
                            "updates_per_epoch": updates_per_epoch,
                            "batch_size": args.batch_size,
                            "optimizer_updates": completed_epochs * updates_per_epoch,
                            "lineage_manifest_sha256": lineage_sha,
                            "dataset_receipt": dataset_receipt,
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
            )
            if final_dataset_receipt != dataset_receipt:
                raise RuntimeError(
                    "formal dataset/parity assets changed during training"
                )
            assert lineage_sha is not None
            final_sha = _verify_or_write_final(
                final_path,
                _model_payload(
                    trainer,
                    formal_stage=args.formal_stage,
                    config_sha256=config_sha,
                    lineage_sha256=lineage_sha,
                    dataset_summary_sha256=dataset_receipt[
                        "summary_sha256"
                    ],
                    data_mdb_sha256=dataset_receipt["data_mdb_sha256"],
                    source_receipt=source_receipt,
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
                    "epochs": args.epochs,
                    "completed_epochs": args.epochs,
                    "world_size": world_size,
                    "train_samples": train_samples,
                    "updates_per_epoch": updates_per_epoch,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.epochs * updates_per_epoch,
                    "lineage_manifest_sha256": lineage_sha,
                    "dataset_receipt": dataset_receipt,
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
                    "epochs": args.epochs,
                    "world_size": world_size,
                    "train_samples": train_samples,
                    "updates_per_epoch": updates_per_epoch,
                    "batch_size": args.batch_size,
                    "lineage_manifest_sha256": lineage_sha,
                    "dataset_receipt": dataset_receipt,
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
