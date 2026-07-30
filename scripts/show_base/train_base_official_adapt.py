#!/usr/bin/env python3
"""Audited SHOW adaptation of the official All-Speakers SemTalk Base.

This is intentionally separate from ``show_base_train.py``.  The latter is the
from-scratch reproduction contract and must not accept a warm start.  This
entry point accepts exactly one warm start: the hash-pinned official
``best_semtalk_base.bin`` release.  It performs one audio-conditioned Base
forward and one optimizer update per batch.  Frozen RVQ targets are consumed
from the already-audited Base LMDB; no VQ model is instantiated here.

The executable modes are:

``throughput_gate``
    Run exactly 20 warm-up and 50 measured updates, then write a gate receipt.

``train``
    Require the matching throughput receipt and train for 40 epochs, emitting
    immutable candidates after epochs 1/2/4/8/16/32/40.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
OFFICIAL_BASE_SOURCE = "released_all_speakers_v1"
OFFICIAL_BASE_CLASSIFICATION = (
    "official_BEAT2_All-Speakers_released_weights_not_SHOW-trained"
)
OFFICIAL_BASE_SPEC = {
    "filename": "best_semtalk_base.bin",
    "sha256": (
        "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
    ),
    "checkpoint_container_schema": [
        "epoch",
        "lrs",
        "model_state",
        "opt_state",
    ],
    "model_class": "semtalk_base",
    "training_dataset": "BEAT2",
    "speaker_scope": "All-Speakers",
}
OFFICIAL_BASE_EPOCH = 401
OFFICIAL_BASE_OPTIMIZER_STATE_ENTRIES = 1_655
OFFICIAL_BASE_OPTIMIZER_PARAMETERS = 1_783
OFFICIAL_BASE_LRS = {
    "param_group_field": "lr",
    "_initial_param_group_field": "initial_lr",
    "base_values": [0.00030000000000000003],
    "metric": None,
    "noise_range_t": None,
    "noise_pct": 0.67,
    "noise_type": "normal",
    "noise_std": 1.0,
    "noise_seed": 42,
    "decay_t": 999,
    "decay_rate": 0.3,
    "warmup_t": 0,
    "warmup_lr_init": 0.0005,
    "t_in_epochs": True,
    "warmup_steps": [1],
}
OFFICIAL_BASE_OPTIMIZER_GROUP = {
    "lr": 0.00030000000000000003,
    "betas": (0.5, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "decoupled_weight_decay": False,
    "initial_lr": 0.00030000000000000003,
}
OFFICIAL_PREREQUISITE_SPECS = {
    "face": {
        "filename": "rvq_face_600.bin",
        "sha256": (
            "31b04c88456a25f4d57841c0cb507b4c856daccb3875878d06545110a6152127"
        ),
    },
    "hands": {
        "filename": "rvq_hands_500.bin",
        "sha256": (
            "08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a539b03e436"
        ),
    },
    "upper": {
        "filename": "rvq_upper_500.bin",
        "sha256": (
            "05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d88d08ac17"
        ),
    },
    "lower": {
        "filename": "rvq_lower_600.bin",
        "sha256": (
            "2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c171ae4efce8"
        ),
    },
    "global": {
        "filename": "last_1700_foot.bin",
        "sha256": (
            "6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e127e19e0144ee12"
        ),
    },
}
SHOW_SPEAKERS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
SPEAKER_EMBEDDING_KEYS = (
    "spearker_encoder_face.weight",
    "spearker_encoder_body.weight",
)
FORBIDDEN_SOURCE_MARKERS = ("e30", "speaker2")
CANDIDATE_EPOCHS = (1, 2, 4, 8, 16, 32, 40)
LOCAL_BATCH_SIZE = 64
WORLD_SIZE = 8
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * WORLD_SIZE
POSE_LENGTH = 64
PRE_FRAMES = 4
CODEBOOK_SIZE = 256
RVQ_LEVELS = 6
EXPECTED_TRAIN_SAMPLES = 127_286
EXPECTED_TRAIN_CLIPS = 13_687
EXPECTED_UPDATES_PER_EPOCH = 248
THROUGHPUT_WARMUP_UPDATES = 20
THROUGHPUT_TIMED_UPDATES = 50
CHECKPOINT_FORMAT = "semtalk_show_base_official_adapt_checkpoint_v1"
MANIFEST_FORMAT = "semtalk_show_base_official_adapt_manifest_v1"
STATUS_FORMAT = "semtalk_show_base_official_adapt_status_v1"
GATE_FORMAT = "semtalk_show_base_official_adapt_throughput_gate_v1"
PROTOCOL_FORMAT = "semtalk_show_base_official_adapt_protocol_v1"
LOSS_COMPONENTS = (
    "zq_face",
    "zq_upper",
    "zq_hands",
    "zq_lower",
    "ce_face",
    "ce_upper",
    "ce_hands",
    "ce_lower",
    "hubert_consistency",
    "beat_consistency",
)


class AdaptationContractError(RuntimeError):
    """Raised before work begins when an immutable contract is violated."""


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise AdaptationContractError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise AdaptationContractError(f"unsafe {label}: {path}")
    return resolved


def reject_forbidden_source_labels(*values: object) -> None:
    """Hard reject every e30 or Speaker2 source label, case-insensitively."""

    for value in values:
        normalized = str(value).casefold().replace("-", "").replace("_", "")
        for marker in FORBIDDEN_SOURCE_MARKERS:
            compact_marker = marker.casefold().replace("-", "").replace("_", "")
            if compact_marker in normalized:
                raise AdaptationContractError(
                    f"forbidden Base adaptation source marker {marker!r}: {value}"
                )


def _state_schema_sha256(state: Mapping[str, Any]) -> str:
    schema = []
    for key, value in sorted(state.items()):
        schema.append(
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    return canonical_json_sha256(schema)


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def read_official_base_checkpoint(
    checkpoint_path: Path,
    *,
    torch_module: Any | None = None,
    specification: Mapping[str, Any] = OFFICIAL_BASE_SPEC,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the exact official Base checkpoint and normalize ``module.`` keys."""

    reject_forbidden_source_labels(checkpoint_path)
    expected_filename = str(specification["filename"])
    expected_sha = str(specification["sha256"])
    if checkpoint_path.name != expected_filename:
        raise AdaptationContractError(
            f"official Base filename {checkpoint_path.name!r} != "
            f"{expected_filename!r}"
        )
    resolved = _regular_file(checkpoint_path, "official Base checkpoint")
    if resolved.name != expected_filename:
        raise AdaptationContractError("resolved official Base basename changed")
    payload_bytes = resolved.read_bytes()
    observed_sha = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha != expected_sha:
        raise AdaptationContractError(
            f"official Base SHA-256 {observed_sha} != {expected_sha}"
        )

    if torch_module is None:
        import torch as torch_module

    try:
        envelope = torch_module.load(
            io.BytesIO(payload_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - supported runtime has it
        raise AdaptationContractError(
            "official Base import requires torch.load(weights_only=True)"
        ) from error
    expected_envelope = set(specification["checkpoint_container_schema"])
    if type(envelope) is not dict or set(envelope) != expected_envelope:
        raise AdaptationContractError(
            "official Base envelope must contain exactly "
            "epoch/lrs/model_state/opt_state"
        )
    if type(envelope.get("epoch")) is not int or envelope["epoch"] != OFFICIAL_BASE_EPOCH:
        raise AdaptationContractError(
            f"official Base epoch must be exactly {OFFICIAL_BASE_EPOCH}"
        )
    if type(envelope.get("lrs")) is not dict or envelope["lrs"] != OFFICIAL_BASE_LRS:
        raise AdaptationContractError("official Base scheduler envelope changed")
    optimizer_state = envelope.get("opt_state")
    if (
        type(optimizer_state) is not dict
        or set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(optimizer_state["state"], Mapping)
        or len(optimizer_state["state"])
        != OFFICIAL_BASE_OPTIMIZER_STATE_ENTRIES
        or type(optimizer_state["param_groups"]) is not list
        or len(optimizer_state["param_groups"]) != 1
    ):
        raise AdaptationContractError("official Base optimizer envelope changed")
    optimizer_group = optimizer_state["param_groups"][0]
    if (
        type(optimizer_group) is not dict
        or set(optimizer_group)
        != {*OFFICIAL_BASE_OPTIMIZER_GROUP, "params"}
        or {
            key: optimizer_group[key]
            for key in OFFICIAL_BASE_OPTIMIZER_GROUP
        }
        != OFFICIAL_BASE_OPTIMIZER_GROUP
        or optimizer_group.get("params")
        != list(range(OFFICIAL_BASE_OPTIMIZER_PARAMETERS))
    ):
        raise AdaptationContractError(
            "official Base optimizer param-group envelope changed"
        )
    for parameter_index, parameter_state in optimizer_state["state"].items():
        if (
            type(parameter_index) is not int
            or type(parameter_state) is not dict
            or set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}
            or any(
                not torch_module.is_tensor(value)
                for value in parameter_state.values()
            )
        ):
            raise AdaptationContractError(
                "official Base optimizer tensor-state envelope changed"
            )
    raw_state = envelope.get("model_state")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise AdaptationContractError("official Base model_state is empty")
    normalized: dict[str, Any] = {}
    for raw_key, value in raw_state.items():
        if type(raw_key) is not str or not raw_key:
            raise AdaptationContractError("invalid official Base state key")
        if not torch_module.is_tensor(value):
            raise AdaptationContractError(
                f"official Base state value {raw_key!r} is not a tensor"
            )
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key or key in normalized:
            raise AdaptationContractError(
                f"official Base normalized state key collision: {key!r}"
            )
        if (
            value.is_floating_point() or value.is_complex()
        ) and not bool(value.isfinite().all().item()):
            raise AdaptationContractError(
                f"official Base state tensor {raw_key!r} contains NaN/Inf"
            )
        normalized[key] = value
    receipt = {
        "source": OFFICIAL_BASE_SOURCE,
        "classification": OFFICIAL_BASE_CLASSIFICATION,
        "path": str(resolved),
        "filename": expected_filename,
        "sha256": observed_sha,
        "bytes": len(payload_bytes),
        "checkpoint_container_schema": list(
            specification["checkpoint_container_schema"]
        ),
        "official_training_epoch": OFFICIAL_BASE_EPOCH,
        "official_optimizer_state_entries": (
            OFFICIAL_BASE_OPTIMIZER_STATE_ENTRIES
        ),
        "official_optimizer_parameters": OFFICIAL_BASE_OPTIMIZER_PARAMETERS,
        "official_optimizer_state_used_for_adaptation": False,
        "model_class": specification["model_class"],
        "training_dataset": specification["training_dataset"],
        "speaker_scope": specification["speaker_scope"],
        "model_state_tensors": len(normalized),
        "model_state_schema_sha256": _state_schema_sha256(normalized),
        "all_model_state_tensors_finite": True,
        "strict_state_dict_load": True,
    }
    return normalized, receipt


def strict_load_and_initialize_show_speakers(
    model: Any,
    official_state: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Strict-load official state, then mean-initialize only SHOW rows 0..3."""

    if torch_module is None:
        import torch as torch_module

    expected = set(model.state_dict())
    observed = set(official_state)
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise AdaptationContractError(
            "official Base state schema mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    model.load_state_dict(official_state, strict=True)
    # ``load_state_dict`` copies into module storage, so the deserialized
    # official tensors remain an immutable comparison oracle without a second
    # full-model clone.
    before = official_state
    speaker_receipts: dict[str, Any] = {}
    with torch_module.no_grad():
        current = model.state_dict()
        for key in SPEAKER_EMBEDDING_KEYS:
            weight = current.get(key)
            if weight is None or tuple(weight.shape) != (25, 768):
                raise AdaptationContractError(
                    f"official Base speaker embedding {key} has shape "
                    f"{None if weight is None else tuple(weight.shape)}"
                )
            mean = before[key].mean(dim=0)
            weight[:4].copy_(mean.unsqueeze(0).expand(4, -1))
            speaker_receipts[key] = {
                "official_25_rows_sha256": _tensor_sha256(before[key]),
                "official_25_row_mean_sha256": _tensor_sha256(mean),
                "initialized_show_rows": [0, 1, 2, 3],
                "initialized_show_rows_sha256": _tensor_sha256(weight[:4]),
                "unchanged_official_rows": [4, 24],
                "unchanged_official_rows_sha256": _tensor_sha256(weight[4:]),
            }

    after = model.state_dict()
    for key in expected:
        if key not in SPEAKER_EMBEDDING_KEYS:
            if not torch_module.equal(after[key], before[key]):
                raise AdaptationContractError(
                    f"non-speaker parameter changed during initialization: {key}"
                )
            continue
        source = before[key]
        target = after[key]
        mean = source.mean(dim=0)
        if (
            not torch_module.equal(target[4:], source[4:])
            or not torch_module.equal(
                target[:4], mean.unsqueeze(0).expand(4, -1)
            )
        ):
            raise AdaptationContractError(
                f"speaker mean initialization contract failed for {key}"
            )
    return {
        "format": "semtalk_show_official_speaker_mean_init_v1",
        "source_rows": 25,
        "target_show_rows": [0, 1, 2, 3],
        "target_show_speakers": SHOW_SPEAKERS,
        "reduction": "exact_arithmetic_mean_across_all_25_official_rows",
        "other_state_unchanged": True,
        "embeddings": speaker_receipts,
    }


def _load_json_receipt(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], Path, str]:
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise AdaptationContractError(f"{label} expected SHA-256 is invalid")
    resolved = _regular_file(path, label)
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise AdaptationContractError(
            f"{label} SHA-256 {observed} != {expected_sha256}"
        )
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AdaptationContractError(f"{label} must be a JSON object")
    return payload, resolved, observed


def validate_dataset_receipts(args: argparse.Namespace) -> dict[str, Any]:
    """Bind the train LMDB to official All-Speakers frozen RVQ targets."""

    summary, summary_path, summary_sha = _load_json_receipt(
        Path(args.dataset_summary),
        args.expected_dataset_summary_sha256,
        "Base dataset summary",
    )
    lineage, lineage_path, lineage_sha = _load_json_receipt(
        Path(args.lineage_manifest),
        args.expected_lineage_sha256,
        "Base feature lineage",
    )
    lmdb_path = Path(args.train_lmdb).resolve()
    data_path = _regular_file(lmdb_path / "data.mdb", "Base LMDB data.mdb")
    lock_path = _regular_file(lmdb_path / "lock.mdb", "Base LMDB lock.mdb")
    observed_data_sha = sha256_file(data_path)
    observed_lock_sha = sha256_file(lock_path)
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
    protocol = lineage.get("protocol")
    records = lineage.get("formal_checkpoints")
    if (
        summary.get("format") != "semtalk_show_base_lmdb_summary_v1"
        or summary.get("status") != "complete"
        or summary.get("scope") != "SemTalk Base only"
        or summary.get("entries") != EXPECTED_TRAIN_SAMPLES
        or summary.get("train_clips") != EXPECTED_TRAIN_CLIPS
        or Path(str(summary.get("lmdb", ""))).resolve() != lmdb_path
        or summary.get("data_mdb_sha256") != observed_data_sha
        or summary.get("lock_mdb_sha256") != observed_lock_sha
        or Path(str(summary.get("lineage_json", ""))).resolve() != lineage_path
        or summary.get("lineage_json_sha256") != lineage_sha
        or lineage.get("format") != "semtalk_show_base_feature_lineage_v1"
        or lineage.get("status") != "complete"
        or lineage.get("entries") != EXPECTED_TRAIN_SAMPLES
        or lineage.get("train_clips") != EXPECTED_TRAIN_CLIPS
        or not isinstance(protocol, dict)
        or protocol.get("scope") != "SemTalk Base only"
        or protocol.get("split") != "train"
        or protocol.get("speakers") != SHOW_SPEAKERS
        or protocol.get("window_length") != POSE_LENGTH
        or protocol.get("stride") != 20
        or protocol.get("in_word")
        != "int64_all_zero_unused_placeholder"
        or set(protocol.get("forbidden_components", []))
        != expected_forbidden
        or protocol.get("prerequisite_source") != OFFICIAL_BASE_SOURCE
        or not isinstance(records, dict)
        or set(records) != set(OFFICIAL_PREREQUISITE_SPECS)
    ):
        raise AdaptationContractError(
            "Base summary/lineage does not match the frozen All-Speakers "
            "SHOW Base-only feature contract"
        )
    for stage, specification in OFFICIAL_PREREQUISITE_SPECS.items():
        record = records[stage]
        if not isinstance(record, dict):
            raise AdaptationContractError(
                f"Base feature lineage {stage} record is not an object"
            )
        reject_forbidden_source_labels(record.get("path", ""), record)
        if (
            record.get("formal_stage") != stage
            or record.get("filename") != specification["filename"]
            or record.get("sha256") != specification["sha256"]
            or record.get("prerequisite_source") != OFFICIAL_BASE_SOURCE
            or record.get("classification") != OFFICIAL_BASE_CLASSIFICATION
            or record.get("training_dataset") != "BEAT2"
            or record.get("speaker_scope") != "All-Speakers"
            or record.get("show_trained") is not False
            or record.get("checkpoint_container_schema") != ["model_state"]
            or record.get("strict_state_dict_load") is not True
            or record.get("all_model_state_tensors_finite") is not True
            or record.get("frozen_eval") is not True
        ):
            raise AdaptationContractError(
                f"Base feature lineage does not use exact official {stage} weight"
            )
    return {
        "format": "semtalk_show_base_official_feature_dataset_receipt_v1",
        "lmdb": str(lmdb_path),
        "entries": EXPECTED_TRAIN_SAMPLES,
        "train_clips": EXPECTED_TRAIN_CLIPS,
        "data_mdb_sha256": observed_data_sha,
        "lock_mdb_sha256": observed_lock_sha,
        "summary": str(summary_path),
        "summary_sha256": summary_sha,
        "lineage": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "prerequisite_source": OFFICIAL_BASE_SOURCE,
        "formal_checkpoints": {
            stage: {
                "filename": records[stage]["filename"],
                "sha256": records[stage]["sha256"],
                "formal_stage": stage,
                "frozen_eval": True,
            }
            for stage in sorted(records)
        },
        "vq_models_in_training_graph": False,
        "vq_targets": "precomputed_frozen_lmdb_tensors",
    }


def source_receipt() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    if git("remote", "get-url", "origin") != EXPECTED_ORIGIN:
        raise AdaptationContractError("SemTalk origin is not the unique trust root")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise AdaptationContractError(
            "official Base adaptation requires a clean source checkout; "
            f"first change: {status.splitlines()[0]}"
        )
    script = Path(__file__).resolve()
    branch_result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if branch_result.returncode not in {0, 1}:
        raise AdaptationContractError("could not inspect source branch state")
    return {
        "origin": EXPECTED_ORIGIN,
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": (
            branch_result.stdout.strip()
            if branch_result.returncode == 0
            else None
        ),
        "clean": True,
        "entrypoint": str(script),
        "entrypoint_sha256": sha256_file(script),
    }


def protocol_receipt(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": PROTOCOL_FORMAT,
        "scope": "SemTalk Base only",
        "initialization": {
            "source": OFFICIAL_BASE_SOURCE,
            "classification": OFFICIAL_BASE_CLASSIFICATION,
            "filename": OFFICIAL_BASE_SPEC["filename"],
            "sha256": OFFICIAL_BASE_SPEC["sha256"],
            "speaker_scope": "All-Speakers",
            "forbidden_sources": ["e30", "Speaker2"],
        },
        "target_dataset": "SHOW",
        "target_speaker_scope": "All",
        "target_speakers": SHOW_SPEAKERS,
        "speaker_initialization": (
            "rows_0_to_3_equal_mean_of_all_25_official_rows"
        ),
        "pose_length": POSE_LENGTH,
        "pre_frames": PRE_FRAMES,
        "stride": 20,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "expected_train_samples": EXPECTED_TRAIN_SAMPLES,
        "expected_updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "epochs": 40,
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.99,
            "scheduler": "constant",
        },
        "precision": args.precision,
        "forward_contract": {
            "forwards_per_optimizer_step": 1,
            "audio_conditioned_main_forward": True,
            "masked_self_forward": False,
            "word_auxiliary_forward": False,
            "use_attentions": True,
            "use_word_in_main_forward": True,
        },
        "loss": {
            "components": list(LOSS_COMPONENTS),
            "zq_stage_weights": {
                "face": 3.0,
                "upper": 3.0,
                "hands": 3.0,
                "lower": 3.0,
            },
            "zq_aggregate_divisor": 6.0,
            "code_ce_stage_weights": {
                "face": 1.0,
                "upper": 1.0,
                "hands": 1.0,
                "lower": 1.0,
            },
            "code_ce_rvq_level_weights": [
                1.0 / float(level + 1) for level in range(RVQ_LEVELS)
            ],
            "hubert_consistency_weight": 1.0,
            "beat_consistency_weight": 1.0,
        },
        "vq_models_in_training_graph": False,
        "throughput_gate": {
            "warmup_updates": THROUGHPUT_WARMUP_UPDATES,
            "timed_updates": THROUGHPUT_TIMED_UPDATES,
        },
    }


def _coerce_target_zq(target: Any, expected_shape: Sequence[int], name: str) -> Any:
    if tuple(target.shape) != tuple(expected_shape):
        raise AdaptationContractError(
            f"{name} shape {tuple(target.shape)} != {tuple(expected_shape)}"
        )
    return target


def audio_conditioned_objective(
    model: Any,
    batch: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute exactly one official main forward and compute its stock loss."""

    if torch_module is None:
        import torch as torch_module

    batch_size = int(batch["latent_all"].shape[0])
    mask = torch_module.ones_like(batch["latent_all"])
    mask[:, :PRE_FRAMES, :] = 0.0
    output = model(
        batch["beat"],
        batch["in_word"],
        mask=mask,
        in_id=batch["tar_id"],
        in_motion=batch["latent_all"],
        use_attentions=True,
        use_word=True,
        hubert=batch["hubert"],
        is_train=True,
    )

    expected_zq_shape = (batch_size, RVQ_LEVELS, 1, 16, 256)
    latent_losses: dict[str, Any] = {}
    ce_losses: dict[str, Any] = {}
    for stage in ("face", "upper", "hands", "lower"):
        reconstruction = output[f"rec_{stage}"]
        target = _coerce_target_zq(
            batch[f"zq_{stage}"], expected_zq_shape, f"zq_{stage}"
        )
        if tuple(reconstruction.shape) != expected_zq_shape:
            raise AdaptationContractError(
                f"rec_{stage} shape {tuple(reconstruction.shape)} "
                f"!= {expected_zq_shape}"
            )
        latent_losses[stage] = torch_module.nn.functional.mse_loss(
            reconstruction, target
        )
        logits = output[f"cls_{stage}"]
        target_indices = batch[f"tar_index_value_{stage}_top"]
        if tuple(logits.shape) != (
            batch_size,
            16,
            CODEBOOK_SIZE,
            RVQ_LEVELS,
        ):
            raise AdaptationContractError(
                f"cls_{stage} has invalid shape {tuple(logits.shape)}"
            )
        if tuple(target_indices.shape) != (batch_size, 16, RVQ_LEVELS):
            raise AdaptationContractError(
                f"tar_index_value_{stage}_top has invalid shape "
                f"{tuple(target_indices.shape)}"
            )
        stage_ce = logits.new_zeros(())
        for level in range(RVQ_LEVELS):
            stage_ce = stage_ce + (
                torch_module.nn.functional.cross_entropy(
                    logits[:, :, :, level].reshape(-1, CODEBOOK_SIZE),
                    target_indices[:, :, level].reshape(-1),
                )
                / float(level + 1)
            )
        ce_losses[stage] = stage_ce

    latent_total = (
        3.0
        * sum(latent_losses.values(), next(iter(latent_losses.values())).new_zeros(()))
        / 6.0
    )
    ce_total = sum(
        ce_losses.values(), next(iter(ce_losses.values())).new_zeros(())
    )
    hubert_consistency = output["hubert_cons_loss"]
    beat_consistency = output["beat_cons_loss"]
    total = latent_total + ce_total + hubert_consistency + beat_consistency
    metrics = {
        **{f"zq_{key}": value for key, value in latent_losses.items()},
        **{f"ce_{key}": value for key, value in ce_losses.items()},
        "zq_total": latent_total,
        "ce_total": ce_total,
        "hubert_consistency": hubert_consistency,
        "beat_consistency": beat_consistency,
        "total": total,
    }
    return total, metrics


def _all_finite(tensors: Iterable[Any]) -> bool:
    import torch

    for tensor in tensors:
        if torch.is_tensor(tensor) and (
            tensor.is_floating_point() or tensor.is_complex()
        ):
            if not bool(tensor.isfinite().all().item()):
                return False
    return True


def _assert_distributed_finite(local_finite: bool, device: Any) -> None:
    import torch
    import torch.distributed as dist

    flag = torch.tensor(
        [1 if local_finite else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if int(flag.item()) != 1:
        raise FloatingPointError("a distributed rank produced NaN or Inf")


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            raise AdaptationContractError(f"batch field {key!r} is not a tensor")
        result[key] = value.to(device=device, non_blocking=True)
    return result


def one_optimizer_update(
    model: Any,
    optimizer: Any,
    batch: Mapping[str, Any],
    *,
    device: Any,
    precision: str,
) -> dict[str, float]:
    """One forward, one backward, one optimizer step; never accumulates."""

    import torch

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    ):
        loss, metric_tensors = audio_conditioned_objective(model, batch)
    _assert_distributed_finite(
        _all_finite(metric_tensors.values()), device
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    _assert_distributed_finite(_all_finite(gradients), device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.99)
    optimizer.step()
    return {
        key: float(value.detach().float().item())
        for key, value in metric_tensors.items()
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _finite_model_and_optimizer(model: Any, optimizer: Any) -> None:
    import torch

    state = _unwrap_model(model).state_dict()
    for key, value in state.items():
        if (
            value.is_floating_point() or value.is_complex()
        ) and not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"non-finite model state: {key}")
    for parameter_index, parameter_state in optimizer.state.items():
        for key, value in parameter_state.items():
            if torch.is_tensor(value) and (
                value.is_floating_point() or value.is_complex()
            ) and not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(
                    f"non-finite optimizer state: {parameter_index}/{key}"
                )


def _save_candidate(
    *,
    model: Any,
    optimizer: Any,
    run_dir: Path,
    epoch: int,
    optimizer_updates: int,
    frozen_receipt: Mapping[str, Any],
    manifest: dict[str, Any],
) -> None:
    if epoch not in CANDIDATE_EPOCHS:
        raise AdaptationContractError(f"epoch {epoch} is not a candidate")
    _finite_model_and_optimizer(model, optimizer)
    filename = f"base_official_adapt_epoch_{epoch:02d}.bin"
    checkpoint_path = run_dir / "candidates" / filename
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise AdaptationContractError(f"candidate already exists: {checkpoint_path}")
    model_state = {
        key: value.detach().cpu()
        for key, value in _unwrap_model(model).state_dict().items()
    }
    audit = {
        "format": CHECKPOINT_FORMAT,
        "completed_epochs": epoch,
        "optimizer_updates": optimizer_updates,
        "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        "official_base_checkpoint_sha256": OFFICIAL_BASE_SPEC["sha256"],
        "speaker_scope": "SHOW_All",
        "speaker_rows": [0, 1, 2, 3],
        "vq_models_in_training_graph": False,
        "all_model_state_tensors_finite": True,
    }
    _atomic_torch_save(
        checkpoint_path,
        {
            "model_state": model_state,
            "audit": audit,
        },
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest["entries"].append(
        {
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "checkpoint": f"candidates/{filename}",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_container_schema": ["audit", "model_state"],
            "model_state_tensors": len(model_state),
            "model_state_schema_sha256": _state_schema_sha256(model_state),
            "all_model_state_tensors_finite": True,
            "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        }
    )
    manifest["entries_sha256"] = canonical_json_sha256(manifest["entries"])
    _atomic_json(run_dir / "candidate_manifest.json", manifest)


def validate_throughput_gate(
    args: argparse.Namespace,
    *,
    frozen_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not args.throughput_gate_report or not args.expected_throughput_gate_sha256:
        raise AdaptationContractError(
            "train mode requires a hash-pinned throughput gate report"
        )
    report, path, observed_sha = _load_json_receipt(
        Path(args.throughput_gate_report),
        args.expected_throughput_gate_sha256,
        "Base throughput gate",
    )
    if (
        report.get("format") != GATE_FORMAT
        or report.get("status") != "pass"
        or report.get("frozen_receipt_sha256")
        != frozen_receipt["receipt_sha256"]
        or report.get("world_size") != WORLD_SIZE
        or report.get("local_batch_size") != LOCAL_BATCH_SIZE
        or report.get("global_batch_size") != GLOBAL_BATCH_SIZE
        or report.get("warmup_updates") != THROUGHPUT_WARMUP_UPDATES
        or report.get("timed_updates") != THROUGHPUT_TIMED_UPDATES
        or report.get("precision") != args.precision
        or float(report.get("learning_rate", math.nan)) != args.learning_rate
        or report.get("all_losses_finite") is not True
        or not isinstance(report.get("samples_per_second"), (int, float))
        or float(report["samples_per_second"]) <= 0.0
    ):
        raise AdaptationContractError(
            "throughput gate does not bind the exact training protocol"
        )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "samples_per_second": float(report["samples_per_second"]),
        "seconds_per_update": float(report["seconds_per_update"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official All-Speakers SemTalk Base adaptation on SHOW"
    )
    parser.allow_abbrev = False
    parser.add_argument(
        "--mode",
        choices=("throughput_gate", "train"),
        required=True,
    )
    parser.add_argument("--official-base-checkpoint", required=True)
    parser.add_argument("--train-lmdb", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--expected-dataset-summary-sha256", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--throughput-gate-report")
    parser.add_argument("--expected-throughput-gate-sha256")
    parser.add_argument("--local-batch-size", type=int, default=LOCAL_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=43)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    reject_forbidden_source_labels(
        args.run_name,
        args.output_root,
        args.official_base_checkpoint,
        args.train_lmdb,
        args.dataset_summary,
        args.lineage_manifest,
        args.throughput_gate_report or "",
    )
    if (
        not args.run_name
        or "/" in args.run_name
        or args.run_name in {".", ".."}
    ):
        raise AdaptationContractError("--run-name must be one path component")
    if args.local_batch_size != LOCAL_BATCH_SIZE:
        raise AdaptationContractError("local batch size must be exactly 64")
    if args.epochs != 40:
        raise AdaptationContractError("official adaptation must expose e40")
    if not (0.0 < args.learning_rate <= 3e-4):
        raise AdaptationContractError("learning rate must be in (0, 3e-4]")
    if args.loader_workers < 0 or args.loader_workers > 16:
        raise AdaptationContractError("loader workers must be in [0,16]")
    if args.mode == "throughput_gate" and (
        args.throughput_gate_report is not None
        or args.expected_throughput_gate_sha256 is not None
    ):
        raise AdaptationContractError(
            "throughput_gate mode cannot consume a previous gate"
        )


def _model_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=768,
        audio_f=256,
        motion_f=256,
        pose_dims=330,
        pose_length=POSE_LENGTH,
        vae_codebook_size=CODEBOOK_SIZE,
        vae_layer=4,
        vae_length=240,
    )


def _distributed_context() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise AdaptationContractError(
            "launch with torchrun --nproc_per_node=8"
        ) from error
    if world_size != WORLD_SIZE or not (0 <= local_rank < WORLD_SIZE):
        raise AdaptationContractError(
            "official adaptation requires exactly one node with 8 ranks"
        )
    return rank, local_rank, world_size


def _create_dataloader(args: argparse.Namespace, rank: int, world_size: int) -> Any:
    import torch
    from dataloaders.show_base import LMDBNPZDataset

    dataset_args = SimpleNamespace(
        train_path=args.train_lmdb,
        pose_length=POSE_LENGTH,
    )
    dataset = LMDBNPZDataset(dataset_args, "train")
    if len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise AdaptationContractError(
            f"Base LMDB entries {len(dataset)} != {EXPECTED_TRAIN_SAMPLES}"
        )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=LOCAL_BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=args.loader_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.loader_workers > 0,
    )
    if len(loader) != EXPECTED_UPDATES_PER_EPOCH:
        raise AdaptationContractError(
            f"updates/epoch {len(loader)} != {EXPECTED_UPDATES_PER_EPOCH}"
        )
    return loader, sampler


def _frozen_receipt(
    *,
    source: Mapping[str, Any],
    official_base: Mapping[str, Any],
    speaker_initialization: Mapping[str, Any],
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "format": "semtalk_show_base_official_adapt_frozen_inputs_v1",
        "source": dict(source),
        "official_base": dict(official_base),
        "speaker_initialization": dict(speaker_initialization),
        "dataset": dict(dataset),
        "protocol": dict(protocol),
    }
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    return payload


def _run_throughput_gate(
    *,
    model: Any,
    optimizer: Any,
    loader: Any,
    sampler: Any,
    args: argparse.Namespace,
    rank: int,
    device: Any,
    run_dir: Path,
    frozen_receipt: Mapping[str, Any],
) -> None:
    import torch
    import torch.distributed as dist

    model.train()
    sampler.set_epoch(0)
    iterator = iter(loader)
    last_metrics: dict[str, float] = {}
    for _ in range(THROUGHPUT_WARMUP_UPDATES):
        last_metrics = one_optimizer_update(
            model,
            optimizer,
            _move_batch(next(iterator), device),
            device=device,
            precision=args.precision,
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(THROUGHPUT_TIMED_UPDATES):
        last_metrics = one_optimizer_update(
            model,
            optimizer,
            _move_batch(next(iterator), device),
            device=device,
            precision=args.precision,
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - started
    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed = float(elapsed_tensor.item())
    _assert_distributed_finite(_all_finite(model.parameters()), device)
    if rank == 0:
        report = {
            "format": GATE_FORMAT,
            "status": "pass",
            "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "warmup_updates": THROUGHPUT_WARMUP_UPDATES,
            "timed_updates": THROUGHPUT_TIMED_UPDATES,
            "precision": args.precision,
            "learning_rate": args.learning_rate,
            "elapsed_seconds": elapsed,
            "seconds_per_update": elapsed / THROUGHPUT_TIMED_UPDATES,
            "samples_per_second": (
                GLOBAL_BATCH_SIZE * THROUGHPUT_TIMED_UPDATES / elapsed
            ),
            "estimated_40_epoch_training_seconds": (
                elapsed
                / THROUGHPUT_TIMED_UPDATES
                * EXPECTED_UPDATES_PER_EPOCH
                * 40
            ),
            "last_metrics": last_metrics,
            "all_losses_finite": True,
            "optimizer_updates": (
                THROUGHPUT_WARMUP_UPDATES + THROUGHPUT_TIMED_UPDATES
            ),
            "peak_cuda_memory_bytes_rank0": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "completed_unix": time.time(),
        }
        report["receipt_sha256"] = canonical_json_sha256(report)
        _atomic_json(run_dir / "throughput_gate.json", report)


def _run_training(
    *,
    model: Any,
    optimizer: Any,
    loader: Any,
    sampler: Any,
    args: argparse.Namespace,
    rank: int,
    device: Any,
    run_dir: Path,
    frozen_receipt: Mapping[str, Any],
    throughput_receipt: Mapping[str, Any],
) -> None:
    import torch
    import torch.distributed as dist

    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "status": "running",
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        "throughput_gate": dict(throughput_receipt),
        "entries": [],
        "entries_sha256": canonical_json_sha256([]),
    }
    if rank == 0:
        _atomic_json(run_dir / "candidate_manifest.json", manifest)
    optimizer_updates = 0
    started_unix = time.time()
    model.train()
    for epoch_index in range(40):
        sampler.set_epoch(epoch_index)
        epoch_sums = {name: 0.0 for name in (*LOSS_COMPONENTS, "total")}
        for batch in loader:
            metrics = one_optimizer_update(
                model,
                optimizer,
                _move_batch(batch, device),
                device=device,
                precision=args.precision,
            )
            optimizer_updates += 1
            for key in epoch_sums:
                metric_key = key
                if key == "hubert_consistency":
                    metric_key = "hubert_consistency"
                elif key == "beat_consistency":
                    metric_key = "beat_consistency"
                epoch_sums[key] += metrics[metric_key]
        if optimizer_updates != (epoch_index + 1) * EXPECTED_UPDATES_PER_EPOCH:
            raise AdaptationContractError("optimizer update accounting mismatch")
        values = torch.tensor(
            [epoch_sums[key] for key in epoch_sums],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        epoch_metrics = {
            key: float(values[index].item())
            / float(WORLD_SIZE * EXPECTED_UPDATES_PER_EPOCH)
            for index, key in enumerate(epoch_sums)
        }
        completed_epoch = epoch_index + 1
        if completed_epoch in CANDIDATE_EPOCHS:
            dist.barrier()
            if rank == 0:
                _save_candidate(
                    model=model,
                    optimizer=optimizer,
                    run_dir=run_dir,
                    epoch=completed_epoch,
                    optimizer_updates=optimizer_updates,
                    frozen_receipt=frozen_receipt,
                    manifest=manifest,
                )
            dist.barrier()
        if rank == 0:
            _atomic_json(
                run_dir / "status.json",
                {
                    "format": STATUS_FORMAT,
                    "status": "running",
                    "completed_epochs": completed_epoch,
                    "optimizer_updates": optimizer_updates,
                    "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
                    "last_epoch_metrics": epoch_metrics,
                    "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
                    "candidate_manifest_sha256": sha256_file(
                        run_dir / "candidate_manifest.json"
                    ),
                    "started_unix": started_unix,
                    "updated_unix": time.time(),
                },
            )
    _assert_distributed_finite(_all_finite(model.parameters()), device)
    if rank == 0:
        if [entry["epoch"] for entry in manifest["entries"]] != list(
            CANDIDATE_EPOCHS
        ):
            raise AdaptationContractError("candidate set is incomplete")
        manifest["status"] = "complete"
        manifest["completed_epochs"] = 40
        manifest["optimizer_updates"] = optimizer_updates
        manifest["entries_sha256"] = canonical_json_sha256(manifest["entries"])
        _atomic_json(run_dir / "candidate_manifest.json", manifest)
        _atomic_json(
            run_dir / "status.json",
            {
                "format": STATUS_FORMAT,
                "status": "complete",
                "completed_epochs": 40,
                "optimizer_updates": optimizer_updates,
                "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
                "candidate_manifest_sha256": sha256_file(
                    run_dir / "candidate_manifest.json"
                ),
                "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
                "throughput_gate": dict(throughput_receipt),
                "world_size": WORLD_SIZE,
                "local_batch_size": LOCAL_BATCH_SIZE,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "all_training_state_finite": True,
                "started_unix": started_unix,
                "completed_unix": time.time(),
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    rank, local_rank, world_size = _distributed_context()

    import torch
    import torch.distributed as dist
    from models.semtalk import semtalk_base

    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise AdaptationContractError(
            "official Base adaptation requires exactly eight visible CUDA GPUs"
        )
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise AdaptationContractError("bf16 is not supported by this CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(args.output_root).resolve() / args.run_name
    try:
        creation_result: list[Any] = [None]
        if rank == 0:
            try:
                if run_dir.exists() or run_dir.is_symlink():
                    raise AdaptationContractError(
                        f"refusing to reuse output directory: {run_dir}"
                    )
                run_dir.mkdir(parents=True)
                creation_result[0] = {"status": "created"}
            except BaseException as error:
                creation_result[0] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(creation_result, src=0)
        if (
            not isinstance(creation_result[0], dict)
            or creation_result[0].get("status") != "created"
        ):
            failure = (
                creation_result[0]
                if isinstance(creation_result[0], dict)
                else {}
            )
            raise AdaptationContractError(
                "output preflight failed: "
                f"{failure.get('error_type')}: "
                f"{failure.get('error')}"
            )

        shared_receipts: list[Any] = [None]
        if rank == 0:
            try:
                shared_receipts[0] = {
                    "status": "complete",
                    "source": source_receipt(),
                    "dataset": validate_dataset_receipts(args),
                }
            except BaseException as error:
                shared_receipts[0] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(shared_receipts, src=0)
        if (
            not isinstance(shared_receipts[0], dict)
            or shared_receipts[0].get("status") != "complete"
        ):
            failure = (
                shared_receipts[0]
                if isinstance(shared_receipts[0], dict)
                else {}
            )
            raise AdaptationContractError(
                "source/dataset preflight failed: "
                f"{failure.get('error_type')}: "
                f"{failure.get('error')}"
            )
        current_source = shared_receipts[0]["source"]
        dataset_receipt = shared_receipts[0]["dataset"]
        official_state, official_receipt = read_official_base_checkpoint(
            Path(args.official_base_checkpoint),
            torch_module=torch,
        )
        model = semtalk_base(_model_args())
        speaker_initialization = strict_load_and_initialize_show_speakers(
            model,
            official_state,
            torch_module=torch,
        )
        del official_state
        protocol = protocol_receipt(args)
        frozen_receipt = _frozen_receipt(
            source=current_source,
            official_base=official_receipt,
            speaker_initialization=speaker_initialization,
            dataset=dataset_receipt,
            protocol=protocol,
        )
        frozen_hashes: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(
            frozen_hashes, frozen_receipt["receipt_sha256"]
        )
        if set(frozen_hashes) != {frozen_receipt["receipt_sha256"]}:
            raise AdaptationContractError(
                "distributed ranks disagree on frozen input receipt"
            )
        if rank == 0:
            _atomic_json(run_dir / "frozen_inputs.json", frozen_receipt)
        dist.barrier()

        throughput_receipt: dict[str, Any] = {}
        if args.mode == "train":
            throughput_receipt = validate_throughput_gate(
                args, frozen_receipt=frozen_receipt
            )
        loader, sampler = _create_dataloader(args, rank, world_size)
        model = model.to(device)
        process_group = dist.new_group()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model, process_group
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )
        if args.mode == "throughput_gate":
            _run_throughput_gate(
                model=model,
                optimizer=optimizer,
                loader=loader,
                sampler=sampler,
                args=args,
                rank=rank,
                device=device,
                run_dir=run_dir,
                frozen_receipt=frozen_receipt,
            )
        else:
            _run_training(
                model=model,
                optimizer=optimizer,
                loader=loader,
                sampler=sampler,
                args=args,
                rank=rank,
                device=device,
                run_dir=run_dir,
                frozen_receipt=frozen_receipt,
                throughput_receipt=throughput_receipt,
            )
        dist.barrier()
        return 0
    except BaseException as error:
        if rank == 0 and run_dir.is_dir():
            _atomic_json(
                run_dir / "failure.json",
                {
                    "format": STATUS_FORMAT,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_unix": time.time(),
                },
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
