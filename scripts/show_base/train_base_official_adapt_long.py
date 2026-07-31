#!/usr/bin/env python3
"""Audited long SHOW adaptation of the official All-Speakers SemTalk Base.

This is intentionally separate from ``show_base_train.py``.  The latter is the
from-scratch reproduction contract and must not accept a warm start.  This
entry point accepts exactly one warm start: the hash-pinned official
``best_semtalk_base.bin`` release.  Each optimizer update executes the complete
official Base objective from ``semtalk_base_trainer.py``: the seeded main
forward, masked self-motion forward, and masked word/audio forward, with one
latent and one RVQ-code classification loss family for every forward.  Frozen
RVQ targets are consumed from the already-audited Base LMDB; no VQ model is
instantiated here.

The executable modes are:

``throughput_gate``
    Run exactly 20 warm-up and 50 measured updates, then write a gate receipt.
    For fresh SHOW prerequisite lineages, the receipt also contains a
    byte-exact semantic hash of the resulting model state.

``train``
    Require the matching throughput receipt and train one uninterrupted
    400-epoch trajectory on a sparse, pre-registered candidate grid.  A fresh
    lineage repeats the first 70 updates and must reproduce the gate's model
    state before any candidate may be published.

``short_quality``
    Require the matching throughput receipt, but deliberately do not consume
    a topology-selection receipt.  Restart from the official Base checkpoint
    and publish only the provisional e1/e2/e4/e8 validation candidates needed
    to decide that topology.  These artifacts use a separate namespace and
    cannot be consumed as a final 400-epoch training bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import selected_prerequisites as selected_contract

EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
OFFICIAL_BASE_SOURCE = "released_all_speakers_v1"
SHOW_VAL_SELECTED_SOURCE = "show_val_selected_v1"
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
OFFICIAL_BASE_TRAINER_SHA256 = (
    "c85765847918a3d41779eaf78472a9b1b351c4bfbe82c1141dd6f0cdb4ef6a6c"
)
OFFICIAL_BASE_TRAINER_GIT_BLOB_SHA1 = (
    "4c4797e6a12fa0fc2e52ee5074b9e3930a97de73"
)
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
TOTAL_EPOCHS = 400
CANDIDATE_EPOCHS = (
    1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80, 100, 120, 140, 160,
    180, 200, 240, 280, 320, 360, 400,
)
SHORT_QUALITY_TOTAL_EPOCHS = 8
SHORT_QUALITY_EPOCHS = (1, 2, 4, 8)
SHORT_QUALITY_MODE = "short_quality"
RUN_PURPOSE_THROUGHPUT = "topology_throughput_gate"
RUN_PURPOSE_SHORT_QUALITY = "topology_short_quality"
RUN_PURPOSE_FORMAL_TRAINING = "formal_training"
TRAJECTORY_ANCHOR_EPOCHS = (1, 2, 4, 8, 16, 32, 40)
RESUME_EPOCHS = (40, 80, 120, 160, 200, 240, 280, 320, 360, 400)
OFFICIAL_W1_REFERENCE_MODE = "official_w1_b64_reference"
W8_GLOBAL64_MODE = "official_objective_w8_l8_g64_ddp_adaptation"
W16_GLOBAL64_MODE = "official_objective_w16_l4_g64_ddp_adaptation"
W8_GLOBAL512_MODE = "validation_gated_w8_l64_g512_empirical_acceleration"
W16_GLOBAL512_MODE = "validation_gated_w16_l32_g512_empirical_acceleration"
TOPOLOGY_SPECS = {
    OFFICIAL_W1_REFERENCE_MODE: {
        "classification": "exact_official_runtime_topology_reference",
        "node_count": 1,
        "local_world_size": 1,
        "world_size": 1,
        "local_batch_size": 64,
        "global_batch_size": 64,
        "updates_per_epoch": 1_988,
        "unique_samples_per_epoch": 127_232,
        "learning_rate": 5e-5,
        "precision": "fp32",
        "formal_training_eligible": True,
    },
    W8_GLOBAL64_MODE: {
        "classification": (
            "official_objective_ddp_adaptation_not_trajectory_equivalent"
        ),
        "node_count": 1,
        "local_world_size": 8,
        "world_size": 8,
        "local_batch_size": 8,
        "global_batch_size": 64,
        "updates_per_epoch": 1_988,
        "unique_samples_per_epoch": 127_232,
        "learning_rate": 5e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    W16_GLOBAL64_MODE: {
        "classification": (
            "official_objective_ddp_adaptation_not_trajectory_equivalent"
        ),
        "node_count": 2,
        "local_world_size": 8,
        "world_size": 16,
        "local_batch_size": 4,
        "global_batch_size": 64,
        "updates_per_epoch": 1_988,
        "unique_samples_per_epoch": 127_232,
        "learning_rate": 5e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    W8_GLOBAL512_MODE: {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 1,
        "local_world_size": 8,
        "world_size": 8,
        "local_batch_size": 64,
        "global_batch_size": 512,
        "updates_per_epoch": 248,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 3e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    W16_GLOBAL512_MODE: {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 2,
        "local_world_size": 8,
        "world_size": 16,
        "local_batch_size": 32,
        "global_batch_size": 512,
        "updates_per_epoch": 248,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 3e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
}
# Default aliases describe the intended formal two-node run.  Runtime code
# always rebinds them to an explicitly selected topology receipt.
NODE_COUNT = 2
LOCAL_WORLD_SIZE = 8
WORLD_SIZE = 16
LOCAL_BATCH_SIZE = 4
GLOBAL_BATCH_SIZE = 64
FORMAL_HOST_BY_SLOT = {
    0: (
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0"
    ),
    1: (
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0"
    ),
}
# Backward-compatible public name for callers that only describe the fixed
# two-node topology.  Runtime code below always resolves a physical host slot
# explicitly so single-node torch node rank zero may run on either audited
# machine without weakening hostname evidence.
FORMAL_HOST_BY_NODE_RANK = FORMAL_HOST_BY_SLOT
POSE_LENGTH = 64
PRE_FRAMES = 4
CODEBOOK_SIZE = 256
RVQ_LEVELS = 6
EXPECTED_TRAIN_SAMPLES = 127_286
EXPECTED_TRAIN_CLIPS = 13_687
EXPECTED_SPLIT_COUNTS = {"train": 13_687, "val": 1_715, "test": 1_708}
EXPECTED_CANONICAL_CLIPS = sum(EXPECTED_SPLIT_COUNTS.values())
EXPECTED_UPDATES_PER_EPOCH = 1_988
EXPECTED_UNIQUE_SAMPLES_PER_EPOCH = 127_232
THROUGHPUT_WARMUP_UPDATES = 20
THROUGHPUT_TIMED_UPDATES = 50
# Keep the candidate envelope compatible with the already-frozen SemTalk
# inference loader.  The long-run distinction belongs to the surrounding
# manifest/status protocol; the model checkpoint payload has the same schema
# and trust semantics as the original official-adaptation candidates.
CHECKPOINT_FORMAT = "semtalk_show_base_official_adapt_checkpoint_v1"
MANIFEST_FORMAT = "semtalk_show_base_official_adapt_long_manifest_v1"
STATUS_FORMAT = "semtalk_show_base_official_adapt_long_status_v1"
SHORT_QUALITY_CHECKPOINT_FORMAT = (
    "semtalk_show_base_official_adapt_short_quality_checkpoint_v1"
)
SHORT_QUALITY_MANIFEST_FORMAT = (
    "semtalk_show_base_official_adapt_short_quality_manifest_v1"
)
SHORT_QUALITY_STATUS_FORMAT = (
    "semtalk_show_base_official_adapt_short_quality_status_v1"
)
SHORT_QUALITY_READY_RECEIPT_FORMAT = (
    "semtalk_show_base_official_adapt_short_quality_candidate_ready_v1"
)
SHORT_QUALITY_EPOCH_METRIC_FORMAT = (
    "semtalk_show_base_official_adapt_short_quality_epoch_metric_v1"
)
GATE_FORMAT = "semtalk_show_base_official_adapt_long_throughput_gate_v1"
TOPOLOGY_GATE_SPEC_FORMAT = "semtalk_show_base_topology_gate_spec_v1"
TOPOLOGY_SELECTION_FORMAT = "semtalk_show_base_topology_selection_v1"
PROTOCOL_FORMAT = "semtalk_show_base_official_adapt_long_protocol_v1"
READY_RECEIPT_FORMAT = (
    "semtalk_show_base_official_adapt_long_candidate_ready_v1"
)
SCHEDULE_FORMAT = "semtalk_show_base_long_schedule_v1"
FRESH_SCHEDULE_FORMAT = "semtalk_show_base_fresh_lineage_schedule_v1"
ANCHOR_FORMAT = "semtalk_show_base_long_trajectory_anchor_v1"
LEGACY_TRAJECTORY_MODE = "legacy_external_anchor_v1"
FRESH_TRAJECTORY_MODE = "fresh_lineage_gate_v1"
FRESH_TRAJECTORY_FORMAT = (
    "semtalk_show_base_fresh_lineage_trajectory_contract_v1"
)
TRAJECTORY_PROBE_FORMAT = "semtalk_show_base_trajectory_probe_v2"
TRAJECTORY_PROBE_UPDATES = (
    THROUGHPUT_WARMUP_UPDATES + THROUGHPUT_TIMED_UPDATES
)
OFFICIAL_FORWARD_MODES = ("main", "masked_self", "word_audio")
LOSS_STAGES = ("face", "upper", "hands", "lower")
LOSS_COMPONENTS = tuple(
    f"{kind}_{mode}_{stage}"
    for mode in OFFICIAL_FORWARD_MODES
    for kind in ("zq", "ce")
    for stage in LOSS_STAGES
) + ("hubert_consistency", "beat_consistency")


class AdaptationContractError(RuntimeError):
    """Raised before work begins when an immutable contract is violated."""


def _run_purpose(args: argparse.Namespace) -> str:
    if args.mode == "throughput_gate":
        return RUN_PURPOSE_THROUGHPUT
    if args.mode == SHORT_QUALITY_MODE:
        return RUN_PURPOSE_SHORT_QUALITY
    if args.mode == "train":
        return RUN_PURPOSE_FORMAL_TRAINING
    raise AdaptationContractError("unknown Base run purpose")


def _target_epochs(args: argparse.Namespace) -> list[int]:
    if args.mode == "throughput_gate":
        return []
    if args.mode == SHORT_QUALITY_MODE:
        return list(SHORT_QUALITY_EPOCHS)
    if args.mode == "train":
        return list(CANDIDATE_EPOCHS)
    raise AdaptationContractError("unknown Base target epoch contract")


def _activate_topology(args: argparse.Namespace) -> dict[str, Any]:
    """Select one explicit runtime topology before any receipt is replayed."""

    mode = getattr(args, "topology_mode", None)
    if mode not in TOPOLOGY_SPECS:
        raise AdaptationContractError(
            "topology mode is not in the immutable Base gate matrix"
        )
    specification = dict(TOPOLOGY_SPECS[mode])
    global NODE_COUNT, LOCAL_WORLD_SIZE, WORLD_SIZE
    global LOCAL_BATCH_SIZE, GLOBAL_BATCH_SIZE
    global EXPECTED_UPDATES_PER_EPOCH
    global EXPECTED_UNIQUE_SAMPLES_PER_EPOCH
    NODE_COUNT = int(specification["node_count"])
    LOCAL_WORLD_SIZE = int(specification["local_world_size"])
    WORLD_SIZE = int(specification["world_size"])
    LOCAL_BATCH_SIZE = int(specification["local_batch_size"])
    GLOBAL_BATCH_SIZE = int(specification["global_batch_size"])
    EXPECTED_UPDATES_PER_EPOCH = int(specification["updates_per_epoch"])
    EXPECTED_UNIQUE_SAMPLES_PER_EPOCH = int(
        specification["unique_samples_per_epoch"]
    )
    return {"mode": mode, **specification}


def _active_host_slots(args: argparse.Namespace) -> tuple[int, ...]:
    """Map torch node ranks to the exact audited physical host slots."""

    topology = _activate_topology(args)
    host_slot = getattr(args, "formal_host_slot", None)
    if host_slot not in FORMAL_HOST_BY_SLOT:
        raise AdaptationContractError(
            "formal physical host slot is not in the audited host inventory"
        )
    if topology["node_count"] == 1:
        return (int(host_slot),)
    if topology["node_count"] == 2:
        return (0, 1)
    raise AdaptationContractError("unsupported formal physical host inventory")


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


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _read_regular_file_bytes(
    path: Path,
    label: str,
) -> tuple[Path, bytes, dict[str, int]]:
    """Read, identify, hash, and later parse one immutable file descriptor."""

    candidate = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise AdaptationContractError(
            f"could not safely open {label}: {candidate}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdaptationContractError(
                f"{label} must be a regular non-symlink file: {candidate}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise AdaptationContractError(
                f"{label} changed while its verified descriptor was read"
            )
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise AdaptationContractError(
                f"{label} size changed while its verified descriptor was read"
            )
        resolved = candidate.resolve(strict=True)
        current = os.stat(candidate, follow_symlinks=False)
        if _stat_identity(current) != _stat_identity(before):
            raise AdaptationContractError(
                f"{label} pathname no longer names its verified descriptor"
            )
    finally:
        os.close(descriptor)
    return resolved, payload, _stat_identity(before)


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdaptationContractError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AdaptationContractError(
                    f"non-finite JSON token in {label}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdaptationContractError(f"invalid strict JSON: {label}") from error
    if not isinstance(value, dict):
        raise AdaptationContractError(f"expected JSON object: {label}")
    return value


def strict_json(path: Path) -> dict[str, Any]:
    resolved, payload, _ = _read_regular_file_bytes(path, "JSON receipt")
    return _strict_json_bytes(payload, str(resolved))


def _strict_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        rows.append(_strict_json_bytes(line, f"{label}:{line_number}"))
    return rows


def _canonical_file_payload_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    """Reject exact e30/Speaker2 labels without falsely rejecting e300."""

    patterns = {
        "e30": re.compile(r"(?<![a-z0-9])e[-_ ]?30(?![0-9])", re.IGNORECASE),
        "speaker2": re.compile(
            r"(?<![a-z0-9])speaker[-_ ]?2(?![0-9])",
            re.IGNORECASE,
        ),
    }
    for value in values:
        for marker, pattern in patterns.items():
            if pattern.search(str(value)):
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


def _model_state_semantic_sha256(state: Mapping[str, Any]) -> str:
    rows = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for key, value in sorted(state.items())
    ]
    return canonical_json_sha256(rows)


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
    resolved, payload_bytes, checkpoint_identity = _read_regular_file_bytes(
        checkpoint_path,
        "official Base checkpoint",
    )
    if resolved.name != expected_filename:
        raise AdaptationContractError("resolved official Base basename changed")
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
        "file_identity": checkpoint_identity,
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
    resolved, payload_bytes, identity = _read_regular_file_bytes(path, label)
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if observed != expected_sha256:
        raise AdaptationContractError(
            f"{label} SHA-256 {observed} != {expected_sha256}"
        )
    payload = _strict_json_bytes(payload_bytes, str(resolved))
    # The same bytes are hashed and parsed above.  Reopening by pathname here
    # would reintroduce a hash/parse replacement window.
    if identity["size"] != len(payload_bytes):
        raise AdaptationContractError(f"{label} descriptor size mismatch")
    return payload, resolved, observed


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdaptationContractError(f"{label} is not a canonical SHA-256")
    return value


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdaptationContractError(f"{label} must be an exact integer")
    return value


def _hash_regular_descriptor(
    descriptor: int,
    *,
    label: str,
) -> tuple[str, dict[str, int]]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise AdaptationContractError(f"{label} is not a regular file")
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise AdaptationContractError(f"{label} changed while being hashed")
    return digest.hexdigest(), _stat_identity(before)


def _verified_lmdb_receipt(
    path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Bind LMDB hashes to the exact directory and file inodes consumed."""

    candidate = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(candidate, flags)
    except OSError as error:
        raise AdaptationContractError(
            f"could not safely open Base LMDB directory {candidate}: {error}"
        ) from error
    try:
        directory_stat = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise AdaptationContractError(
                f"Base LMDB path is not a directory: {candidate}"
            )
        files: dict[str, Any] = {}
        for filename in ("data.mdb", "lock.mdb"):
            file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            file_flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(
                    filename,
                    file_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise AdaptationContractError(
                    f"could not safely open Base LMDB {filename}: {error}"
                ) from error
            try:
                observed_sha, identity = _hash_regular_descriptor(
                    descriptor,
                    label=f"Base LMDB {filename}",
                )
            finally:
                os.close(descriptor)
            files[filename] = {
                "sha256": observed_sha,
                "identity": identity,
            }
        final_directory_stat = os.fstat(directory_descriptor)
        if _stat_identity(directory_stat) != _stat_identity(
            final_directory_stat
        ):
            raise AdaptationContractError(
                "Base LMDB directory changed during immutable preflight"
            )
        resolved = candidate.resolve(strict=True)
        current_directory = os.stat(candidate, follow_symlinks=False)
        if _stat_identity(current_directory) != _stat_identity(directory_stat):
            raise AdaptationContractError(
                "Base LMDB pathname no longer names its verified directory"
            )
    finally:
        os.close(directory_descriptor)
    return resolved, {
        "format": "semtalk_show_base_lmdb_inode_binding_v1",
        "directory_identity": _stat_identity(directory_stat),
        "files": files,
    }


def _validate_canonical_dataset_evidence(
    summary: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay full canonical split and clip-ledger evidence for Base train."""

    canonical = lineage.get("canonical_receipt")
    manifest_bindings = lineage.get("canonical_manifest_sha256")
    if not isinstance(canonical, dict) or set(canonical) != {
        "manifest",
        "manifest_sha256",
        "summary",
        "summary_sha256",
        "lineage",
        "lineage_sha256",
        "lineage_contract_sha256",
        "source_receipt",
    }:
        raise AdaptationContractError(
            "Base feature lineage lacks the exact full canonical receipt"
        )
    manifest_sha = _require_sha256(
        canonical["manifest_sha256"], "canonical manifest SHA-256"
    )
    canonical_summary_sha = _require_sha256(
        canonical["summary_sha256"], "canonical summary SHA-256"
    )
    canonical_lineage_sha = _require_sha256(
        canonical["lineage_sha256"], "canonical lineage SHA-256"
    )
    lineage_contract_sha = _require_sha256(
        canonical["lineage_contract_sha256"],
        "canonical lineage contract SHA-256",
    )
    manifest_path, manifest_bytes, _ = _read_regular_file_bytes(
        Path(str(canonical["manifest"])), "canonical SHOW manifest"
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha:
        raise AdaptationContractError("canonical SHOW manifest SHA mismatch")
    canonical_summary, canonical_summary_path, _ = _load_json_receipt(
        Path(str(canonical["summary"])),
        canonical_summary_sha,
        "canonical SHOW summary",
    )
    canonical_lineage, canonical_lineage_path, _ = _load_json_receipt(
        Path(str(canonical["lineage"])),
        canonical_lineage_sha,
        "canonical SHOW lineage",
    )
    if (
        manifest_path != Path(str(canonical["manifest"])).resolve(strict=True)
        or canonical_summary_path
        != Path(str(canonical["summary"])).resolve(strict=True)
        or canonical_lineage_path
        != Path(str(canonical["lineage"])).resolve(strict=True)
        or not isinstance(manifest_bindings, dict)
        or len(manifest_bindings) != 1
        or list(manifest_bindings.values()) != [manifest_sha]
        or Path(next(iter(manifest_bindings))).resolve(strict=True)
        != manifest_path
    ):
        raise AdaptationContractError(
            "Base feature lineage canonical artifact paths/hashes disagree"
        )
    source_receipt = canonical_lineage.get("lineage_contract", {}).get(
        "source_receipt"
    )
    if (
        canonical_summary.get("status") != "complete"
        or canonical_summary.get("schema_name")
        != "semtalk-show-canonical-motion"
        or _exact_int(
            canonical_summary.get("schema_version"),
            "canonical summary schema_version",
        )
        != 1
        or canonical_summary.get("split_counts") != EXPECTED_SPLIT_COUNTS
        or _exact_int(
            canonical_summary.get("clip_count"),
            "canonical summary clip_count",
        )
        != EXPECTED_CANONICAL_CLIPS
        or canonical_summary.get("manifest_sha256") != manifest_sha
        or canonical_summary.get("exact_once") is not True
        or canonical_summary.get("finite") is not True
        or canonical_summary.get("split_disjoint") is not True
        or canonical_lineage.get("final_manifest_sha256") != manifest_sha
        or canonical_lineage.get("lineage_contract_sha256")
        != lineage_contract_sha
        or canonical_summary.get("lineage_contract_sha256")
        != lineage_contract_sha
        or _canonical_file_payload_sha256(canonical_lineage)
        != canonical_summary.get("lineage_sha256")
        or _canonical_file_payload_sha256(
            canonical_lineage.get("lineage_contract")
        )
        != lineage_contract_sha
        or not isinstance(source_receipt, dict)
        or canonical.get("source_receipt") != source_receipt
        or source_receipt.get("origin") != EXPECTED_ORIGIN
        or canonical_summary.get("source_receipt_sha256")
        != _canonical_file_payload_sha256(source_receipt)
    ):
        raise AdaptationContractError(
            "canonical SHOW summary/lineage/source proof is incomplete"
        )

    rows = _strict_jsonl_bytes(manifest_bytes, str(manifest_path))
    if len(rows) != EXPECTED_CANONICAL_CLIPS:
        raise AdaptationContractError("canonical SHOW manifest row count changed")
    split_ids: dict[str, list[str]] = {
        split: [] for split in EXPECTED_SPLIT_COUNTS
    }
    global_indices: set[int] = set()
    all_ids: set[str] = set()
    train_rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, 1):
        split = row.get("split")
        clip_id = row.get("clip_id")
        global_index = _exact_int(
            row.get("global_index"),
            f"canonical manifest row {line_number} global_index",
        )
        speaker = row.get("speaker")
        speaker_id = _exact_int(
            row.get("speaker_id"),
            f"canonical manifest row {line_number} speaker_id",
        )
        if (
            split not in EXPECTED_SPLIT_COUNTS
            or not isinstance(clip_id, str)
            or not clip_id
            or clip_id in all_ids
            or global_index < 0
            or global_index >= EXPECTED_CANONICAL_CLIPS
            or global_index in global_indices
            or speaker not in SHOW_SPEAKERS
            or speaker_id != SHOW_SPEAKERS[speaker]
            or _exact_int(
                row.get("frames"),
                f"canonical manifest row {line_number} frames",
            )
            <= 0
            or row.get("lineage_contract_sha256") != lineage_contract_sha
        ):
            raise AdaptationContractError(
                f"canonical SHOW manifest row {line_number} is invalid"
            )
        _require_sha256(
            row.get("canonical_npz_sha256"),
            f"canonical manifest row {line_number} NPZ SHA-256",
        )
        all_ids.add(clip_id)
        global_indices.add(global_index)
        split_ids[split].append(clip_id)
        if split == "train":
            train_rows.append(row)
    if (
        {split: len(ids) for split, ids in split_ids.items()}
        != EXPECTED_SPLIT_COUNTS
        or global_indices != set(range(EXPECTED_CANONICAL_CLIPS))
        or sum(len(set(ids)) for ids in split_ids.values()) != len(all_ids)
    ):
        raise AdaptationContractError(
            "canonical SHOW manifest is not exact-once and split-disjoint"
        )
    train_rows.sort(
        key=lambda row: (str(row["clip_id"]), str(row.get("canonical_npz", "")))
    )
    expected_train_ids = [str(row["clip_id"]) for row in train_rows]

    per_clip = summary.get("per_clip")
    skipped = summary.get("skipped_short_clip_ids")
    if (
        not isinstance(per_clip, list)
        or len(per_clip) != EXPECTED_TRAIN_CLIPS
        or not isinstance(skipped, list)
        or any(not isinstance(value, str) for value in skipped)
        or len(set(skipped)) != len(skipped)
        or summary.get("entry_aggregate_sha256")
        != lineage.get("entry_aggregate_sha256")
    ):
        raise AdaptationContractError("Base per-clip ledger schema mismatch")
    _require_sha256(
        summary.get("entry_aggregate_sha256"),
        "Base entry aggregate SHA-256",
    )
    ledger_ids: list[str] = []
    zero_window_ids: set[str] = set()
    windows_total = 0
    raw_frames_total = 0
    usable_frames_total = 0
    dropped_frames_total = 0
    for index, (ledger, canonical_row) in enumerate(zip(per_clip, train_rows)):
        if not isinstance(ledger, dict):
            raise AdaptationContractError(
                f"Base per-clip ledger row {index} is not an object"
            )
        clip_id = ledger.get("clip_id")
        windows = _exact_int(ledger.get("windows"), f"ledger {index} windows")
        raw_frames = _exact_int(
            ledger.get("raw_frames"), f"ledger {index} raw_frames"
        )
        usable_frames = _exact_int(
            ledger.get("usable_frames"), f"ledger {index} usable_frames"
        )
        dropped_frames = _exact_int(
            ledger.get("dropped_tail_frames"),
            f"ledger {index} dropped_tail_frames",
        )
        if (
            clip_id != canonical_row["clip_id"]
            or ledger.get("canonical_npz_sha256")
            != canonical_row.get("canonical_npz_sha256")
            or raw_frames != canonical_row["frames"]
            or usable_frames != (raw_frames // 30) * 30
            or dropped_frames != raw_frames - usable_frames
            or windows != max(0, (usable_frames - POSE_LENGTH) // 20 + 1)
            or ledger.get("speaker_id") != canonical_row["speaker_id"]
        ):
            raise AdaptationContractError(
                f"Base per-clip ledger row {index} disagrees with canonical train"
            )
        ledger_ids.append(clip_id)
        windows_total += windows
        raw_frames_total += raw_frames
        usable_frames_total += usable_frames
        dropped_frames_total += dropped_frames
        if windows == 0:
            zero_window_ids.add(clip_id)
    if (
        ledger_ids != expected_train_ids
        or len(set(ledger_ids)) != EXPECTED_TRAIN_CLIPS
        or set(skipped) != zero_window_ids
        or windows_total != EXPECTED_TRAIN_SAMPLES
        or summary.get("raw_frames") != raw_frames_total
        or summary.get("usable_frames") != usable_frames_total
        or summary.get("dropped_tail_frames") != dropped_frames_total
    ):
        raise AdaptationContractError(
            "Base train clip ledger is not exact-once or count-complete"
        )
    return {
        "format": "semtalk_show_base_canonical_split_proof_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "summary": str(canonical_summary_path),
        "summary_sha256": canonical_summary_sha,
        "lineage": str(canonical_lineage_path),
        "lineage_sha256": canonical_lineage_sha,
        "lineage_contract_sha256": lineage_contract_sha,
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "split_clip_ids_sha256": {
            split: canonical_json_sha256(sorted(ids))
            for split, ids in split_ids.items()
        },
        "global_clip_ids_sha256": canonical_json_sha256(sorted(all_ids)),
        "split_disjoint": True,
        "exact_once": True,
        "train_per_clip_ledger_exact": True,
        "train_windows": windows_total,
        "test_rows_used_as_training_samples": False,
        "source_receipt": source_receipt,
    }


def validate_dataset_receipts(args: argparse.Namespace) -> dict[str, Any]:
    """Bind the train LMDB to one immutable five-prerequisite transaction."""

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
    lmdb_path, lmdb_binding = _verified_lmdb_receipt(
        Path(args.train_lmdb)
    )
    observed_data_sha = lmdb_binding["files"]["data.mdb"]["sha256"]
    observed_lock_sha = lmdb_binding["files"]["lock.mdb"]["sha256"]
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
        or not isinstance(records, dict)
        or set(records) != set(OFFICIAL_PREREQUISITE_SPECS)
    ):
        raise AdaptationContractError(
            "Base summary/lineage does not match the frozen All-Speakers "
            "SHOW Base-only feature contract"
        )
    canonical_evidence = _validate_canonical_dataset_evidence(
        summary,
        lineage,
    )
    selected_mode = (
        getattr(args, "prerequisite_selection_json", None) is not None
    )
    selected_bridge: dict[str, Any] | None = None
    if selected_mode:
        try:
            selected_bridge = selected_contract.load_selected_prerequisites(
                args.prerequisite_selection_json,
                getattr(
                    args,
                    "expected_prerequisite_selection_sha256",
                    None,
                ),
            )
        except selected_contract.SelectedPrerequisiteError as error:
            raise AdaptationContractError(str(error)) from error
        if (
            protocol.get("prerequisite_source") != SHOW_VAL_SELECTED_SOURCE
            or lineage.get("prerequisite_source_receipt")
            != selected_bridge
            or selected_bridge.get("global_verified_not_consumed") is not True
        ):
            raise AdaptationContractError(
                "Base feature lineage is not bound to the externally "
                "hash-pinned five-stage validation selection"
            )
        for stage in selected_contract.STAGES:
            record = records[stage]
            selected = selected_bridge["selected"][stage]
            checkpoint = selected["candidate_checkpoint"]
            if not isinstance(record, dict):
                raise AdaptationContractError(
                    f"Base feature lineage {stage} record is not an object"
                )
            reject_forbidden_source_labels(record.get("path", ""), record)
            if (
                record.get("formal_stage") != stage
                or record.get("path") != checkpoint["path"]
                or record.get("sha256") != checkpoint["sha256"]
                or record.get("bytes") != checkpoint["bytes"]
                or record.get("prerequisite_source")
                != SHOW_VAL_SELECTED_SOURCE
                or record.get("training_dataset") != "SHOW"
                or record.get("speaker_scope") != "All"
                or record.get("show_trained") is not True
                or record.get("selection_split") != "val"
                or record.get("test_visible") is not False
                or record.get("selected_epoch") != selected["epoch"]
                or record.get("selected_optimizer_updates")
                != selected["optimizer_updates"]
                or record.get("selection_metric")
                != selected["selection_metric"]
                or record.get("selection_score")
                != selected["selection_score"]
                or record.get("candidate_audit_sha256")
                != selected["candidate_audit_sha256"]
                or record.get("measurement_receipt")
                != selected["measurement_receipt"]
                or record.get("checkpoint_container_schema")
                != ["audit", "model_state"]
                or record.get("strict_state_dict_load") is not True
                or record.get("all_model_state_tensors_finite") is not True
                or record.get("frozen_eval") is not True
            ):
                raise AdaptationContractError(
                    f"Base feature lineage does not bind selected {stage}"
                )
    else:
        if protocol.get("prerequisite_source") != OFFICIAL_BASE_SOURCE:
            raise AdaptationContractError(
                "legacy Base feature lineage does not use official "
                "All-Speakers prerequisites"
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
                or record.get("classification")
                != OFFICIAL_BASE_CLASSIFICATION
                or record.get("training_dataset") != "BEAT2"
                or record.get("speaker_scope") != "All-Speakers"
                or record.get("show_trained") is not False
                or record.get("checkpoint_container_schema")
                != ["model_state"]
                or record.get("strict_state_dict_load") is not True
                or record.get("all_model_state_tensors_finite") is not True
                or record.get("frozen_eval") is not True
            ):
                raise AdaptationContractError(
                    "Base feature lineage does not use exact official "
                    f"{stage} weight"
                )
    receipt = {
        "format": (
            "semtalk_show_base_selected_feature_dataset_receipt_v1"
            if selected_mode
            else "semtalk_show_base_official_feature_dataset_receipt_v1"
        ),
        "lmdb": str(lmdb_path),
        "entries": EXPECTED_TRAIN_SAMPLES,
        "train_clips": EXPECTED_TRAIN_CLIPS,
        "split": "train",
        "test_visible": False,
        "data_mdb_sha256": observed_data_sha,
        "lock_mdb_sha256": observed_lock_sha,
        "lmdb_inode_binding": lmdb_binding,
        "summary": str(summary_path),
        "summary_sha256": summary_sha,
        "lineage": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "canonical_dataset_evidence": canonical_evidence,
        "prerequisite_source": (
            SHOW_VAL_SELECTED_SOURCE if selected_mode else OFFICIAL_BASE_SOURCE
        ),
        "formal_checkpoints": {
            stage: {
                "sha256": records[stage]["sha256"],
                "formal_stage": stage,
                "frozen_eval": True,
            }
            for stage in sorted(records)
        },
        "vq_models_in_training_graph": False,
        "vq_targets": "precomputed_frozen_lmdb_tensors",
    }
    if selected_bridge is not None:
        receipt["prerequisite_selection"] = selected_bridge["selection"]
        receipt["selected_prerequisite_sha256"] = {
            stage: selected_bridge["selected"][stage][
                "candidate_checkpoint"
            ]["sha256"]
            for stage in selected_contract.STAGES
        }
        receipt["global_verified_not_consumed"] = True
    else:
        for stage in OFFICIAL_PREREQUISITE_SPECS:
            receipt["formal_checkpoints"][stage]["filename"] = records[stage][
                "filename"
            ]
    return receipt


def _portable_source_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return only Git semantics that must agree between local-SSD clones."""

    portable = {
        key: receipt[key]
        for key in (
            "origin",
            "commit",
            "tree",
            "clean",
            "entrypoint_sha256",
        )
    }
    if (
        portable["origin"] != EXPECTED_ORIGIN
        or portable["clean"] is not True
    ):
        raise AdaptationContractError("portable source receipt is invalid")
    return portable


def _portable_dataset_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Strip only the node-local inode binding for semantic comparison."""

    portable = dict(receipt)
    binding = portable.pop("lmdb_inode_binding", None)
    if (
        not isinstance(binding, dict)
        or binding.get("format")
        != "semtalk_show_base_lmdb_inode_binding_v1"
    ):
        raise AdaptationContractError("node-local LMDB binding is missing")
    return portable


def _global_dataset_receipt(
    node_receipts: Sequence[Mapping[str, Any]],
    *,
    host_slots: Sequence[int],
) -> dict[str, Any]:
    if (
        len(host_slots) != len(node_receipts)
        or len(set(host_slots)) != len(host_slots)
        or any(slot not in FORMAL_HOST_BY_SLOT for slot in host_slots)
    ):
        raise AdaptationContractError(
            "dataset receipt host-slot inventory is invalid"
        )
    semantic = _portable_dataset_receipt(node_receipts[0]["dataset"])
    for receipt in node_receipts[1:]:
        if _portable_dataset_receipt(receipt["dataset"]) != semantic:
            raise AdaptationContractError(
                "nodes disagree on SHOW feature content/lineage semantics"
            )
    global_receipt = dict(semantic)
    global_receipt["lmdb_binding_scope"] = (
        "ordered_node_local_inode_bindings_with_global_content_sha256"
    )
    global_receipt["node_lmdb_inode_bindings"] = [
        {
            "node_rank": node_rank,
            "host_slot": host_slots[node_rank],
            "hostname": FORMAL_HOST_BY_SLOT[host_slots[node_rank]],
            "binding": dict(receipt["dataset"]["lmdb_inode_binding"]),
        }
        for node_rank, receipt in enumerate(node_receipts)
    ]
    return global_receipt


def _global_official_base_receipt(
    rank_receipts: Sequence[Mapping[str, Any]],
    *,
    host_slots: Sequence[int],
) -> dict[str, Any]:
    """Bind equal checkpoint content plus ordered node-local open identities."""

    semantic_receipts: list[dict[str, Any]] = []
    for receipt in rank_receipts:
        semantic = dict(receipt)
        identity = semantic.pop("file_identity", None)
        path = semantic.pop("path", None)
        if not isinstance(identity, dict) or not isinstance(path, str):
            raise AdaptationContractError(
                "official Base node-local file identity is missing"
            )
        semantic_receipts.append(semantic)
    if any(
        receipt != semantic_receipts[0]
        for receipt in semantic_receipts[1:]
    ):
        raise AdaptationContractError(
            "ranks disagree on official Base checkpoint semantics"
        )
    if (
        len(host_slots) != NODE_COUNT
        or len(set(host_slots)) != len(host_slots)
        or any(slot not in FORMAL_HOST_BY_SLOT for slot in host_slots)
    ):
        raise AdaptationContractError(
            "official Base receipt host-slot inventory is invalid"
        )
    result = dict(semantic_receipts[0])
    result["file_binding_scope"] = (
        "ordered_node_local_open_identity_with_global_content_sha256"
    )
    result["node_local_files"] = [
        {
            "node_rank": node_rank,
            "host_slot": host_slots[node_rank],
            "hostname": FORMAL_HOST_BY_SLOT[host_slots[node_rank]],
            "path": rank_receipts[node_rank * LOCAL_WORLD_SIZE]["path"],
            "file_identity": dict(
                rank_receipts[node_rank * LOCAL_WORLD_SIZE]["file_identity"]
            ),
        }
        for node_rank in range(NODE_COUNT)
    ]
    return result


def validate_long_contract_receipts(
    args: argparse.Namespace,
    *,
    dataset_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    topology = {"mode": args.topology_mode, **TOPOLOGY_SPECS[args.topology_mode]}
    expected_learning_rate = float(topology["learning_rate"])
    schedule, schedule_path, schedule_sha = _load_json_receipt(
        Path(args.schedule_json),
        args.expected_schedule_sha256,
        "long-training schedule",
    )
    training = schedule.get("training")
    selection = schedule.get("selection")
    initialization = schedule.get("initialization")
    if (
        schedule.get("scope") != "SemTalk Base only"
        or schedule.get("target_dataset") != "SHOW"
        or schedule.get("target_speaker_scope") != "All"
        or not isinstance(initialization, dict)
        or initialization.get("checkpoint_sha256")
        != OFFICIAL_BASE_SPEC["sha256"]
        or initialization.get("forbidden_epoch") != 30
        or set(initialization.get("forbidden_sources", []))
        != {"Speaker2", "SemGate", "Sparse"}
        or not isinstance(training, dict)
        or training.get("total_epochs") != TOTAL_EPOCHS
        or training.get("topology_source")
        != "sealed_five_mode_topology_gate_v1"
        or training.get("topology_matrix") != TOPOLOGY_SPECS
        or training.get("precision_source")
        != "selected_topology_matrix_entry"
        or args.precision != topology["precision"]
        or training.get("optimizer") != "Adam"
        or training.get("learning_rate_source")
        != "selected_topology_matrix_entry"
        or args.learning_rate != expected_learning_rate
        or training.get("betas") != [0.5, 0.999]
        or training.get("weight_decay") != 0.0
        or training.get("gradient_clip_norm") != 0.99
        or training.get("scheduler") != "constant"
        or training.get("seed") != args.seed
        or training.get("vq_models_in_training_graph") is not False
        or schedule.get("candidate_epochs") != list(CANDIDATE_EPOCHS)
        or not isinstance(selection, dict)
        or selection.get("split") != "val"
        or selection.get("test_visible") is not False
        or selection.get("ordering") != ["FGD", "epoch"]
        or selection.get("direction") != ["min", "min"]
        or selection.get("test_runs") != 1
    ):
        raise AdaptationContractError(
            "long-training schedule does not match the frozen e400 contract"
        )

    schedule_receipt = {
        "path": str(schedule_path),
        "sha256": schedule_sha,
        "payload_sha256": canonical_json_sha256(schedule),
    }
    if args.trajectory_mode == FRESH_TRAJECTORY_MODE:
        expected_trajectory_contract = {
            "mode": FRESH_TRAJECTORY_MODE,
            "external_anchor": False,
            "probe_optimizer_updates": TRAJECTORY_PROBE_UPDATES,
            "probe_source": "matching_frozen_receipt_throughput_gate",
            "comparison": "byte_exact_model_state_semantic_sha256",
            "restart_from_official_initialization": True,
        }
        prerequisite_selection = dataset_receipt.get(
            "prerequisite_selection"
        )
        selected_sha256 = dataset_receipt.get(
            "selected_prerequisite_sha256"
        )
        canonical_evidence = dataset_receipt.get(
            "canonical_dataset_evidence"
        )
        lmdb_inode_binding = dataset_receipt.get("lmdb_inode_binding")
        if (
            schedule.get("format") != FRESH_SCHEDULE_FORMAT
            or schedule.get("trajectory_contract")
            != expected_trajectory_contract
            or training.get("loader_workers") != args.loader_workers
            or schedule.get("trajectory_anchor_epochs") is not None
            or dataset_receipt.get("format")
            != "semtalk_show_base_selected_feature_dataset_receipt_v1"
            or dataset_receipt.get("prerequisite_source")
            != SHOW_VAL_SELECTED_SOURCE
            or dataset_receipt.get("split") != "train"
            or dataset_receipt.get("test_visible") is not False
            or dataset_receipt.get("summary_sha256")
            != args.expected_dataset_summary_sha256
            or dataset_receipt.get("lineage_sha256")
            != args.expected_lineage_sha256
            or not isinstance(prerequisite_selection, dict)
            or prerequisite_selection.get("sha256")
            != args.expected_prerequisite_selection_sha256
            or not isinstance(selected_sha256, dict)
            or set(selected_sha256) != set(selected_contract.STAGES)
            or dataset_receipt.get("global_verified_not_consumed") is not True
            or not isinstance(canonical_evidence, dict)
            or canonical_evidence.get("split_counts")
            != EXPECTED_SPLIT_COUNTS
            or canonical_evidence.get("split_disjoint") is not True
            or canonical_evidence.get("exact_once") is not True
            or canonical_evidence.get("train_per_clip_ledger_exact") is not True
            or canonical_evidence.get("test_rows_used_as_training_samples")
            is not False
            or not isinstance(lmdb_inode_binding, dict)
            or lmdb_inode_binding.get("format")
            != "semtalk_show_base_lmdb_inode_binding_v1"
        ):
            raise AdaptationContractError(
                "fresh Base trajectory is not bound to the exact selected "
                "SHOW prerequisite/feature lineage"
            )
        binding = {
            "format": FRESH_TRAJECTORY_FORMAT,
            "mode": FRESH_TRAJECTORY_MODE,
            "schedule_sha256": schedule_sha,
            "official_base_checkpoint_sha256": OFFICIAL_BASE_SPEC["sha256"],
            "dataset_receipt_payload_sha256": canonical_json_sha256(
                _portable_dataset_receipt(dataset_receipt)
            ),
            "dataset_split": "train",
            "test_visible": False,
            "dataset_summary_sha256": dataset_receipt["summary_sha256"],
            "feature_lineage_sha256": dataset_receipt["lineage_sha256"],
            "data_mdb_sha256": dataset_receipt["data_mdb_sha256"],
            "lock_mdb_sha256": dataset_receipt["lock_mdb_sha256"],
            # Device/inode identities are node-local (EFS presents different
            # st_dev values in the two pods).  They remain mandatory local
            # DataLoader evidence, but the cross-node semantic lineage binds
            # the content hashes and canonical ledger instead.
            "lmdb_binding_scope": "node_local_inode_content_global_sha256",
            "canonical_dataset_evidence": canonical_evidence,
            "prerequisite_selection_sha256": prerequisite_selection[
                "sha256"
            ],
            "selected_prerequisite_sha256": {
                stage: selected_sha256[stage]
                for stage in sorted(selected_sha256)
            },
            "probe_optimizer_updates": TRAJECTORY_PROBE_UPDATES,
            "probe_source": "matching_frozen_receipt_throughput_gate",
            "comparison": (
                "byte_exact_all_rank_model_adam_rng_and_sample_order_v2"
            ),
            "seed": args.seed,
            "precision": args.precision,
            "learning_rate": args.learning_rate,
            "topology_mode": args.topology_mode,
            "topology_classification": topology["classification"],
            "world_size": WORLD_SIZE,
            "node_count": NODE_COUNT,
            "local_world_size": LOCAL_WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
            "unique_samples_per_epoch": EXPECTED_UNIQUE_SAMPLES_PER_EPOCH,
            "loader_workers": args.loader_workers,
        }
        binding_sha = canonical_json_sha256(binding)
        return {
            "format": "semtalk_show_base_long_contract_receipts_v1",
            "schedule": schedule_receipt,
            # Keep the outer key stable for checkpoint/receipt consumers.  In
            # fresh mode this is a lineage binding, never an old checkpoint
            # manifest masquerading as an applicable trajectory anchor.
            "trajectory_anchor": {
                **binding,
                "path": None,
                "sha256": binding_sha,
                "payload_sha256": binding_sha,
                "entries": {},
            },
        }

    if (
        args.trajectory_mode != LEGACY_TRAJECTORY_MODE
        or schedule.get("format") != SCHEDULE_FORMAT
        or schedule.get("trajectory_anchor_epochs")
        != list(TRAJECTORY_ANCHOR_EPOCHS)
        or dataset_receipt.get("format")
        == "semtalk_show_base_selected_feature_dataset_receipt_v1"
    ):
        raise AdaptationContractError(
            "legacy trajectory anchors cannot be used with freshly selected "
            "SHOW prerequisites"
        )

    anchor, anchor_path, anchor_sha = _load_json_receipt(
        Path(args.trajectory_anchor_json),
        args.expected_trajectory_anchor_sha256,
        "trajectory anchor",
    )
    entries = anchor.get("entries")
    source_manifest_path = Path(
        str(anchor.get("source_candidate_manifest", ""))
    )
    source_manifest_sha = str(
        anchor.get("source_candidate_manifest_sha256", "")
    )
    if (
        anchor.get("format") != ANCHOR_FORMAT
        or not isinstance(entries, dict)
        or set(entries) != {str(epoch) for epoch in TRAJECTORY_ANCHOR_EPOCHS}
    ):
        raise AdaptationContractError("invalid long-training trajectory anchor")
    manifest, source_manifest, _ = _load_json_receipt(
        source_manifest_path,
        source_manifest_sha,
        "trajectory source candidate manifest",
    )
    manifest_entries = manifest.get("entries")
    if (
        manifest.get("format")
        != "semtalk_show_base_official_adapt_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("candidate_epochs")
        != list(TRAJECTORY_ANCHOR_EPOCHS)
        or not isinstance(manifest_entries, list)
        or len(manifest_entries) != len(TRAJECTORY_ANCHOR_EPOCHS)
    ):
        raise AdaptationContractError(
            "trajectory source manifest contract mismatch"
        )
    by_epoch = {
        int(entry["epoch"]): entry
        for entry in manifest_entries
        if isinstance(entry, dict) and type(entry.get("epoch")) is int
    }
    for epoch in TRAJECTORY_ANCHOR_EPOCHS:
        entry = entries[str(epoch)]
        source_entry = by_epoch.get(epoch)
        if (
            not isinstance(entry, dict)
            or not isinstance(source_entry, dict)
            or entry.get("checkpoint_sha256")
            != source_entry.get("checkpoint_sha256")
            or entry.get("tensor_count") != 1790
            or not isinstance(
                entry.get("model_state_semantic_sha256"), str
            )
            or len(entry["model_state_semantic_sha256"]) != 64
        ):
            raise AdaptationContractError(
                f"trajectory anchor mismatch at epoch {epoch}"
            )
    return {
        "format": "semtalk_show_base_long_contract_receipts_v1",
        "schedule": schedule_receipt,
        "trajectory_anchor": {
            "mode": LEGACY_TRAJECTORY_MODE,
            "path": str(anchor_path),
            "sha256": anchor_sha,
            "payload_sha256": canonical_json_sha256(anchor),
            "source_candidate_manifest": str(source_manifest),
            "source_candidate_manifest_sha256": source_manifest_sha,
            "entries": entries,
        },
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


def protocol_receipt(
    args: argparse.Namespace,
    *,
    contract_receipts: Mapping[str, Any],
    topology_gate_spec: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory = contract_receipts["trajectory_anchor"]
    fresh_trajectory = trajectory.get("mode") == FRESH_TRAJECTORY_MODE
    topology = _activate_topology(args)
    official_trainer = Path(__file__).resolve().parents[2] / (
        "semtalk_base_trainer.py"
    )
    if (
        sha256_file(official_trainer) != OFFICIAL_BASE_TRAINER_SHA256
        or subprocess.run(
            [
                "git",
                "-C",
                str(official_trainer.parent),
                "hash-object",
                str(official_trainer),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        != OFFICIAL_BASE_TRAINER_GIT_BLOB_SHA1
    ):
        raise AdaptationContractError(
            "published official Base trainer semantics changed"
        )
    host_slots = _active_host_slots(args)
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
        "node_count": NODE_COUNT,
        "local_world_size": LOCAL_WORLD_SIZE,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "distributed_topology": {
            "mode": topology["mode"],
            "classification": topology["classification"],
            "backend": "nccl",
            "node_count": NODE_COUNT,
            "local_world_size": LOCAL_WORLD_SIZE,
            "world_size": WORLD_SIZE,
            "nodes": [
                {
                    "node_rank": rank,
                    "host_slot": host_slots[rank],
                    "hostname": FORMAL_HOST_BY_SLOT[host_slots[rank]],
                    "rank_range": list(
                        range(
                            rank * LOCAL_WORLD_SIZE,
                            (rank + 1) * LOCAL_WORLD_SIZE,
                        )
                    ),
                }
                for rank in range(NODE_COUNT)
            ],
            "master_addr": args.formal_master_addr,
            "master_port": args.formal_master_port,
            "formal_run_id": args.formal_run_id,
            "official_reference": {
                "mode": OFFICIAL_W1_REFERENCE_MODE,
                **TOPOLOGY_SPECS[OFFICIAL_W1_REFERENCE_MODE],
                "optimizer_updates_400_epochs": (
                    TOTAL_EPOCHS
                    * int(
                        TOPOLOGY_SPECS[OFFICIAL_W1_REFERENCE_MODE][
                            "updates_per_epoch"
                        ]
                    )
                ),
                "adam_learning_rate": 5e-5,
            },
            "trajectory_equivalence_to_official_w1": (
                topology["mode"] == OFFICIAL_W1_REFERENCE_MODE
            ),
            "fp32_numeric_equivalence_to_official_w1": (
                topology["mode"] == OFFICIAL_W1_REFERENCE_MODE
            ),
            "ddp_differences": (
                []
                if topology["mode"] == OFFICIAL_W1_REFERENCE_MODE
                else [
                    "per_rank_batchnorm_statistics",
                    "dropout_and_rng_streams",
                    "distributed_sampler_order",
                    "floating_point_gradient_reduction",
                ]
            ),
        },
        "topology_gate_spec": dict(topology_gate_spec),
        "loader_workers": args.loader_workers,
        "expected_train_samples": EXPECTED_TRAIN_SAMPLES,
        "expected_updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "expected_unique_samples_per_epoch": (
            EXPECTED_UNIQUE_SAMPLES_PER_EPOCH
        ),
        "epochs": TOTAL_EPOCHS,
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "trajectory_anchor_epochs": (
            [] if fresh_trajectory else list(TRAJECTORY_ANCHOR_EPOCHS)
        ),
        "schedule": {
            "path": str(Path(args.schedule_json).resolve()),
            "sha256": args.expected_schedule_sha256,
        },
        "trajectory_anchor": {
            "mode": trajectory["mode"],
            "path": trajectory["path"],
            "sha256": trajectory["sha256"],
            "external": not fresh_trajectory,
        },
        "trajectory_gate": {
            "required": fresh_trajectory,
            "probe_optimizer_updates": (
                TRAJECTORY_PROBE_UPDATES if fresh_trajectory else None
            ),
            "probe_source": (
                "matching_frozen_receipt_throughput_gate"
                if fresh_trajectory
                else None
            ),
            "comparison": (
                "byte_exact_all_rank_model_adam_rng_and_sample_order_v2"
                if fresh_trajectory
                else None
            ),
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.99,
            "scheduler": "constant",
        },
        "precision": args.precision,
        "determinism": {
            "python_hash_seed": str(args.seed),
            "python_hash_seed_required_at_interpreter_start": True,
            "python_random_seed": args.seed,
            "numpy_random_seed": args.seed,
            "torch_cpu_seed": args.seed,
            "torch_cuda_seed_all": args.seed,
            "model_initialization_seed": args.seed,
            "rank_training_seed": "seed_plus_global_rank_after_strict_load",
            "dropout_rng_correlated_across_ranks": False,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "float32_matmul_precision": "highest",
            "loader_generator_seed": args.seed,
            "loader_worker_seed": "torch_initial_seed_mod_2pow32",
            "multiprocessing_context": "fork",
            "lmdb_worker_binding": (
                "fork_inherited_pinned_data_fd_reverified_at_worker_init_"
                "and_before_and_after_open"
            ),
        },
        "forward_contract": {
            "reference_entrypoint": "semtalk_base_trainer.py:_g_training",
            "reference_sha256": OFFICIAL_BASE_TRAINER_SHA256,
            "reference_git_blob_sha1": OFFICIAL_BASE_TRAINER_GIT_BLOB_SHA1,
            "forwards_per_optimizer_step": 3,
            "audio_conditioned_main_forward": True,
            "masked_self_forward": True,
            "word_auxiliary_forward": True,
            "use_attentions": True,
            "use_word_in_main_forward": True,
            "use_word_in_masked_self_forward": False,
            "use_word_in_word_audio_forward": True,
            "shared_random_mask_for_auxiliary_forwards": True,
            "auxiliary_mask_rng": (
                "torch_cpu_default_generator_then_float_cuda_transfer"
            ),
            "mask_ratio_schedule": {
                "epochs_0_to_129": "epoch / 400 * 0.95 + 0.05",
                "epochs_130_plus": 0.35875,
            },
            "optimizer_steps_per_batch": 1,
            "masked_self_ce_published_source_semantics": (
                "repeat_rvq_level_5_six_times_divided_by_6_v1"
            ),
            "classification_operator_path": (
                "log_softmax_dim_2_class_axis_then_reshape_then_nll_loss"
            ),
        },
        "loss": {
            "components": list(LOSS_COMPONENTS),
            "forward_families": list(OFFICIAL_FORWARD_MODES),
            "latent_and_code_ce_per_forward": True,
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
            "consistency_losses_from": "main_forward_only",
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


def _official_forward_loss_family(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    mode: str,
    masked_self_published_ce: bool,
    torch_module: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return one official latent/CE family without another model call.

    ``semtalk_base_trainer.py`` accidentally retains ``i == 5`` inside its
    masked-self ``for j in range(6)`` loop.  Released All-Speakers Base was
    trained from that published source, so the formal adaptation preserves
    the observable behavior: level five is repeated six times with divisor
    six.  Main and word/audio families use levels 0..5 with 1/(level+1).
    """

    if mode not in OFFICIAL_FORWARD_MODES:
        raise AdaptationContractError(
            f"unknown official Base forward family {mode!r}"
        )
    batch_size = int(batch["latent_all"].shape[0])
    expected_zq_shape = (batch_size, RVQ_LEVELS, 1, 16, 256)
    stage_latent: dict[str, Any] = {}
    stage_ce: dict[str, Any] = {}
    for stage in LOSS_STAGES:
        reconstruction = output[f"rec_{stage}"]
        target = _coerce_target_zq(
            batch[f"zq_{stage}"],
            expected_zq_shape,
            f"zq_{stage}",
        )
        if tuple(reconstruction.shape) != expected_zq_shape:
            raise AdaptationContractError(
                f"{mode} rec_{stage} shape {tuple(reconstruction.shape)} "
                f"!= {expected_zq_shape}"
            )
        stage_latent[stage] = torch_module.nn.functional.mse_loss(
            reconstruction,
            target,
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
                f"{mode} cls_{stage} has invalid shape "
                f"{tuple(logits.shape)}"
            )
        if tuple(target_indices.shape) != (
            batch_size,
            16,
            RVQ_LEVELS,
        ):
            raise AdaptationContractError(
                f"tar_index_value_{stage}_top has invalid shape "
                f"{tuple(target_indices.shape)}"
            )
        ce = logits.new_zeros(())
        for loop_level in range(RVQ_LEVELS):
            source_level = RVQ_LEVELS - 1 if masked_self_published_ce else loop_level
            divisor = float(RVQ_LEVELS if masked_self_published_ce else loop_level + 1)
            ce = ce + (
                torch_module.nn.functional.nll_loss(
                    torch_module.nn.functional.log_softmax(
                        logits[:, :, :, source_level],
                        # Official ``semtalk_base_trainer.py`` constructs
                        # ``nn.LogSoftmax(dim=2)`` and applies it to the
                        # sliced [B,T,C] classifier tensor.  Dimension two
                        # is therefore the codebook/class axis; dimension
                        # one would incorrectly normalize over time.
                        dim=2,
                    ).reshape(-1, CODEBOOK_SIZE),
                    target_indices[:, :, source_level].reshape(-1),
                )
                / divisor
            )
        stage_ce[stage] = ce

    zero = next(iter(stage_latent.values())).new_zeros(())
    latent_total = 3.0 * sum(stage_latent.values(), zero) / 6.0
    ce_total = sum(stage_ce.values(), zero)
    metrics = {
        **{
            f"zq_{mode}_{stage}": value
            for stage, value in stage_latent.items()
        },
        **{
            f"ce_{mode}_{stage}": value
            for stage, value in stage_ce.items()
        },
        f"zq_{mode}_total": latent_total,
        f"ce_{mode}_total": ce_total,
    }
    return latent_total, ce_total, metrics


def audio_conditioned_objective(
    model: Any,
    batch: Mapping[str, Any],
    *,
    epoch: int,
    torch_module: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute the published official three-forward Base train objective."""

    if torch_module is None:
        import torch as torch_module
    if type(epoch) is not int or epoch < 0 or epoch >= TOTAL_EPOCHS:
        raise AdaptationContractError(
            f"official Base objective epoch must be in [0,{TOTAL_EPOCHS})"
        )

    seed_mask = torch_module.ones_like(batch["latent_all"])
    seed_mask[:, :PRE_FRAMES, :] = 0.0
    main_output = model(
        batch["beat"],
        batch["in_word"],
        mask=seed_mask,
        in_id=batch["tar_id"],
        in_motion=batch["latent_all"],
        use_attentions=True,
        use_word=True,
        hubert=batch["hubert"],
        is_train=True,
    )
    main_latent, main_ce, metrics = _official_forward_loss_family(
        main_output,
        batch,
        mode="main",
        masked_self_published_ce=False,
        torch_module=torch_module,
    )
    mask_ratio = (
        (float(epoch) / 400.0) * 0.95 + 0.05
        if epoch < 130
        else 0.35875
    )
    # Preserve the published CPU RNG path exactly: torch.rand(bs,n,dims) is
    # created on CPU, thresholded there, converted to float, then transferred
    # to the active CUDA device.  rand_like(latent_all) would consume CUDA RNG
    # and change the official W1 trajectory.
    auxiliary_mask = (
        torch_module.rand(tuple(batch["latent_all"].shape), device="cpu")
        < mask_ratio
    ).to(
        device=batch["latent_all"].device,
        dtype=batch["latent_all"].dtype,
    )
    masked_self_output = model(
        batch["beat"],
        batch["in_word"],
        mask=auxiliary_mask,
        in_id=batch["tar_id"],
        in_motion=batch["latent_all"],
        use_attentions=True,
        use_word=False,
        hubert=batch["hubert"],
        is_train=True,
    )
    self_latent, self_ce, self_metrics = _official_forward_loss_family(
        masked_self_output,
        batch,
        mode="masked_self",
        masked_self_published_ce=True,
        torch_module=torch_module,
    )
    word_output = model(
        batch["beat"],
        batch["in_word"],
        mask=auxiliary_mask,
        in_id=batch["tar_id"],
        in_motion=batch["latent_all"],
        use_attentions=True,
        use_word=True,
        hubert=batch["hubert"],
        is_train=True,
    )
    word_latent, word_ce, word_metrics = _official_forward_loss_family(
        word_output,
        batch,
        mode="word_audio",
        masked_self_published_ce=False,
        torch_module=torch_module,
    )
    hubert_consistency = main_output["hubert_cons_loss"]
    beat_consistency = main_output["beat_cons_loss"]
    total = (
        main_latent
        + main_ce
        + self_latent
        + self_ce
        + word_latent
        + word_ce
        + hubert_consistency
        + beat_consistency
    )
    metrics.update(self_metrics)
    metrics.update(word_metrics)
    metrics.update({
        "hubert_consistency": hubert_consistency,
        "beat_consistency": beat_consistency,
        "total": total,
    })
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
    epoch: int,
) -> dict[str, float]:
    """Three official forwards, one backward and one optimizer step."""

    import torch

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    ):
        loss, metric_tensors = audio_conditioned_objective(
            model,
            batch,
            epoch=epoch,
        )
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
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        0.99,
    )
    _assert_distributed_finite(
        bool(torch.isfinite(gradient_norm).item()),
        device,
    )
    optimizer.step()
    result = {
        key: float(value.detach().float().item())
        for key, value in metric_tensors.items()
    }
    result["gradient_norm_preclip"] = float(
        gradient_norm.detach().float().item()
    )
    return result


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


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite immutable receipt: {path}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is an atomic no-replace publish: readers can observe either
        # no receipt or the complete, fsynced receipt, never a partial file.
        os.link(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _append_epoch_metric(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _state_tree_semantic_sha256(value: Any) -> str:
    import torch

    def normalize(item: Any) -> Any:
        if torch.is_tensor(item):
            return {
                "type": "tensor",
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "sha256": _tensor_sha256(item),
            }
        if isinstance(item, Mapping):
            normalized_items = [
                [normalize(key), normalize(child)]
                for key, child in item.items()
            ]
            normalized_items.sort(
                key=lambda row: json.dumps(
                    row[0],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return {"type": "mapping", "items": normalized_items}
        if isinstance(item, list):
            return {"type": "list", "items": [normalize(child) for child in item]}
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [normalize(child) for child in item]}
        if item is None or type(item) in {bool, int, str}:
            return {"type": type(item).__name__, "value": item}
        if type(item) is float and math.isfinite(item):
            return {"type": "float", "value": item}
        raise AdaptationContractError(
            f"unsupported or non-finite trajectory state value: {type(item)}"
        )

    return canonical_json_sha256(normalize(value))


def _record_sample_order(
    digest: Any,
    batch: Mapping[str, Any],
    *,
    rank: int,
    optimizer_update: int,
) -> int:
    import torch

    indices = batch.get("sample_index")
    if (
        not torch.is_tensor(indices)
        or indices.ndim != 1
        or int(indices.numel()) != LOCAL_BATCH_SIZE
    ):
        raise AdaptationContractError(
            "trajectory batches must carry exactly the selected local batch "
            "of immutable LMDB indices"
        )
    values = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    digest.update(int(rank).to_bytes(4, "little", signed=False))
    digest.update(int(optimizer_update).to_bytes(8, "little", signed=False))
    digest.update(int(values.numel()).to_bytes(4, "little", signed=False))
    digest.update(values.numpy().astype("<i8", copy=False).tobytes(order="C"))
    return int(values.numel())


def _batch_sample_indices(batch: Mapping[str, Any]) -> list[int]:
    import torch

    indices = batch.get("sample_index")
    if (
        not torch.is_tensor(indices)
        or indices.ndim != 1
        or int(indices.numel()) != LOCAL_BATCH_SIZE
    ):
        raise AdaptationContractError("invalid gate sample-index inventory")
    return [
        int(value)
        for value in indices.detach().to(device="cpu", dtype=torch.int64).tolist()
    ]


def _rng_state_hashes(device: Any) -> dict[str, str]:
    import random

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    normalized_numpy = {
        "algorithm": str(numpy_state[0]),
        "keys_sha256": hashlib.sha256(
            np.asarray(numpy_state[1], dtype="<u4").tobytes(order="C")
        ).hexdigest(),
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    return {
        "python_random_state_sha256": _state_tree_semantic_sha256(
            random.getstate()
        ),
        "numpy_random_state_sha256": canonical_json_sha256(normalized_numpy),
        "torch_cpu_rng_state_sha256": _tensor_sha256(torch.get_rng_state()),
        "torch_cuda_rng_state_sha256": _tensor_sha256(
            torch.cuda.get_rng_state(device)
        ),
    }


def _trajectory_rank_probe(
    model: Any,
    optimizer: Any,
    optimizer_updates: int,
    *,
    rank: int,
    device: Any,
    sample_order_sha256: str,
    sample_count: int,
) -> dict[str, Any]:
    state = _unwrap_model(model).state_dict()
    return {
        "rank": rank,
        "optimizer_updates": optimizer_updates,
        "model_state_tensors": len(state),
        "model_state_schema_sha256": _state_schema_sha256(state),
        "model_state_semantic_sha256": _model_state_semantic_sha256(state),
        "optimizer_state_semantic_sha256": _state_tree_semantic_sha256(
            optimizer.state_dict()
        ),
        **_rng_state_hashes(device),
        "sample_order_sha256": _require_sha256(
            sample_order_sha256,
            f"rank {rank} sample order SHA-256",
        ),
        "sample_count": sample_count,
    }


def _assemble_trajectory_probe(
    rank_probes: Sequence[Any],
    *,
    optimizer_updates: int,
) -> dict[str, Any]:
    expected_rank_keys = {
        "rank",
        "optimizer_updates",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
        "optimizer_state_semantic_sha256",
        "python_random_state_sha256",
        "numpy_random_state_sha256",
        "torch_cpu_rng_state_sha256",
        "torch_cuda_rng_state_sha256",
        "sample_order_sha256",
        "sample_count",
    }
    if (
        len(rank_probes) != WORLD_SIZE
        or any(
            not isinstance(probe, dict) or set(probe) != expected_rank_keys
            for probe in rank_probes
        )
        or [probe["rank"] for probe in rank_probes] != list(range(WORLD_SIZE))
    ):
        raise AdaptationContractError(
            "trajectory probe does not contain exact ordered all-rank state"
        )
    for probe in rank_probes:
        if (
            probe["optimizer_updates"] != optimizer_updates
            or probe["sample_count"]
            != optimizer_updates * LOCAL_BATCH_SIZE
            or probe["model_state_tensors"] <= 0
        ):
            raise AdaptationContractError(
                f"trajectory rank {probe['rank']} metadata mismatch"
            )
        for key, value in probe.items():
            if key.endswith("_sha256"):
                _require_sha256(value, f"trajectory rank {probe['rank']} {key}")
    model_consensus = {
        (
            probe["model_state_tensors"],
            probe["model_state_schema_sha256"],
            probe["model_state_semantic_sha256"],
        )
        for probe in rank_probes
    }
    optimizer_consensus = {
        probe["optimizer_state_semantic_sha256"] for probe in rank_probes
    }
    if len(model_consensus) != 1 or len(optimizer_consensus) != 1:
        raise AdaptationContractError(
            "DDP ranks disagree on model or Adam state at trajectory probe"
        )
    return {
        "format": TRAJECTORY_PROBE_FORMAT,
        "optimizer_updates": optimizer_updates,
        "world_size": WORLD_SIZE,
        "rank_order": list(range(WORLD_SIZE)),
        "all_rank_model_state_identical": True,
        "all_rank_optimizer_state_identical": True,
        "ranks": [dict(probe) for probe in rank_probes],
    }


def _distributed_trajectory_probe(
    model: Any,
    optimizer: Any,
    optimizer_updates: int,
    *,
    rank: int,
    device: Any,
    sample_order_sha256: str,
    sample_count: int,
) -> dict[str, Any]:
    import torch.distributed as dist

    try:
        local_probe: Any = {
            "status": "complete",
            "probe": _trajectory_rank_probe(
                model,
                optimizer,
                optimizer_updates,
                rank=rank,
                device=device,
                sample_order_sha256=sample_order_sha256,
                sample_count=sample_count,
            ),
        }
    except BaseException as error:
        local_probe = {
            "status": "failed",
            "rank": rank,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, local_probe)
    if any(
        not isinstance(item, dict) or item.get("status") != "complete"
        for item in gathered
    ):
        raise AdaptationContractError(
            f"distributed trajectory rank probe failed: {gathered}"
        )
    rank_probes = [item["probe"] for item in gathered]
    return _assemble_trajectory_probe(
        rank_probes,
        optimizer_updates=optimizer_updates,
    )


def _validate_trajectory_probe(
    probe: Any,
    *,
    label: str,
) -> dict[str, Any]:
    expected_keys = {
        "format",
        "optimizer_updates",
        "world_size",
        "rank_order",
        "all_rank_model_state_identical",
        "all_rank_optimizer_state_identical",
        "ranks",
    }
    if not isinstance(probe, dict) or set(probe) != expected_keys:
        raise AdaptationContractError(f"{label} schema mismatch")
    if (
        probe.get("format") != TRAJECTORY_PROBE_FORMAT
        or type(probe.get("optimizer_updates")) is not int
        or probe["optimizer_updates"] != TRAJECTORY_PROBE_UPDATES
        or probe.get("world_size") != WORLD_SIZE
        or probe.get("rank_order") != list(range(WORLD_SIZE))
        or probe.get("all_rank_model_state_identical") is not True
        or probe.get("all_rank_optimizer_state_identical") is not True
        or not isinstance(probe.get("ranks"), list)
    ):
        raise AdaptationContractError(f"{label} metadata mismatch")
    assembled = _assemble_trajectory_probe(
        probe["ranks"],
        optimizer_updates=TRAJECTORY_PROBE_UPDATES,
    )
    if assembled != probe:
        raise AdaptationContractError(f"{label} canonical aggregate mismatch")
    return assembled


def _require_matching_trajectory_probe(
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    expected_probe = _validate_trajectory_probe(
        expected,
        label="throughput trajectory probe",
    )
    observed_probe = _validate_trajectory_probe(
        observed,
        label="training trajectory probe",
    )
    if observed_probe != expected_probe:
        raise AdaptationContractError(
            "fresh Base trajectory probe does not reproduce the exact "
            "matching-lineage throughput gate"
        )
    return observed_probe


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
    contract_receipts: Mapping[str, Any],
    manifest: dict[str, Any],
    provisional: bool,
) -> None:
    candidate_epochs = (
        SHORT_QUALITY_EPOCHS if provisional else CANDIDATE_EPOCHS
    )
    if epoch not in candidate_epochs:
        raise AdaptationContractError(f"epoch {epoch} is not a candidate")
    _finite_model_and_optimizer(model, optimizer)
    if provisional:
        filename = f"base_official_adapt_short_quality_epoch_{epoch:02d}.bin"
        checkpoint_relative = f"provisional_candidates/{filename}"
    else:
        filename = f"base_official_adapt_epoch_{epoch:02d}.bin"
        checkpoint_relative = f"candidates/{filename}"
    checkpoint_path = run_dir / checkpoint_relative
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise AdaptationContractError(f"candidate already exists: {checkpoint_path}")
    model_state = {
        key: value.detach().cpu()
        for key, value in _unwrap_model(model).state_dict().items()
    }
    semantic_sha = _model_state_semantic_sha256(model_state)
    trajectory_contract = contract_receipts["trajectory_anchor"]
    fresh_trajectory = (
        trajectory_contract.get("mode") == FRESH_TRAJECTORY_MODE
    )
    if fresh_trajectory and manifest.get("trajectory_probe_verified") is not True:
        raise AdaptationContractError(
            "fresh Base candidate publication precedes trajectory-gate "
            "verification"
        )
    expected_anchor = trajectory_contract["entries"].get(
        str(epoch)
    )
    anchor_match: bool | None = None
    if expected_anchor is not None:
        anchor_match = (
            len(model_state) == expected_anchor["tensor_count"]
            and semantic_sha
            == expected_anchor["model_state_semantic_sha256"]
        )
        if not anchor_match:
            raise AdaptationContractError(
                f"trajectory anchor failed at epoch {epoch}: "
                f"{semantic_sha} != "
                f"{expected_anchor['model_state_semantic_sha256']}"
            )
    audit = {
        "format": (
            SHORT_QUALITY_CHECKPOINT_FORMAT
            if provisional
            else CHECKPOINT_FORMAT
        ),
        "completed_epochs": epoch,
        "optimizer_updates": optimizer_updates,
        "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        "official_base_checkpoint_sha256": OFFICIAL_BASE_SPEC["sha256"],
        "speaker_scope": "SHOW_All",
        "speaker_rows": [0, 1, 2, 3],
        "vq_models_in_training_graph": False,
        "all_model_state_tensors_finite": True,
        "model_state_semantic_sha256": semantic_sha,
        "trajectory_anchor_match": anchor_match,
        "trajectory_probe_verified": (
            True if fresh_trajectory else None
        ),
    }
    if provisional:
        audit["run_purpose"] = RUN_PURPOSE_SHORT_QUALITY
        audit["target_epochs"] = list(candidate_epochs)
    _atomic_torch_save(
        checkpoint_path,
        {
            "model_state": model_state,
            "audit": audit,
        },
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_entry = {
        "epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "checkpoint": checkpoint_relative,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_container_schema": ["audit", "model_state"],
        "model_state_tensors": len(model_state),
        "model_state_schema_sha256": _state_schema_sha256(model_state),
        "model_state_semantic_sha256": semantic_sha,
        "all_model_state_tensors_finite": True,
        "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        "trajectory_anchor_match": anchor_match,
        "trajectory_probe_verified": (
            True if fresh_trajectory else None
        ),
    }
    if provisional:
        manifest_entry["run_purpose"] = RUN_PURPOSE_SHORT_QUALITY
    manifest["entries"].append(manifest_entry)
    manifest["entries_sha256"] = canonical_json_sha256(manifest["entries"])
    manifest_name = (
        "short_quality_candidate_manifest.json"
        if provisional
        else "candidate_manifest.json"
    )
    snapshot_directory = (
        "short_quality_candidate_manifest_snapshots"
        if provisional
        else "candidate_manifest_snapshots"
    )
    receipt_directory = (
        "short_quality_candidate_receipts"
        if provisional
        else "candidate_receipts"
    )
    _atomic_json(run_dir / manifest_name, manifest)
    live_manifest_path = run_dir / manifest_name
    manifest_snapshot_path = (
        run_dir
        / snapshot_directory
        / f"epoch-{epoch:04d}.json"
    )
    _write_new_json(manifest_snapshot_path, manifest)
    ready_body = {
        "format": (
            SHORT_QUALITY_READY_RECEIPT_FORMAT
            if provisional
            else READY_RECEIPT_FORMAT
        ),
        "status": "ready",
        "selection_eligible": False,
        "test_visible": False,
        "epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "candidate_checkpoint": {
            "path": str(checkpoint_path.resolve(strict=True)),
            "relative_path": checkpoint_relative,
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
            "model_state_tensors": len(model_state),
            "model_state_schema_sha256": _state_schema_sha256(model_state),
            "model_state_semantic_sha256": semantic_sha,
        },
        "candidate_manifest": {
            "path": str(manifest_snapshot_path.resolve(strict=True)),
            "sha256_at_ready": sha256_file(manifest_snapshot_path),
            "entries_sha256_at_ready": manifest["entries_sha256"],
            "immutable_snapshot": True,
            "live_path": str(live_manifest_path.resolve(strict=True)),
        },
        "frozen_inputs": {
            "path": str(
                (run_dir / "frozen_inputs.json").resolve(strict=True)
            ),
            "sha256": sha256_file(run_dir / "frozen_inputs.json"),
            "receipt_payload_sha256": frozen_receipt["receipt_sha256"],
        },
        "protocol": {
            "format": frozen_receipt["protocol"]["format"],
            "payload_sha256": canonical_json_sha256(
                frozen_receipt["protocol"]
            ),
        },
        "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        "schedule_sha256": contract_receipts["schedule"]["sha256"],
        "trajectory_anchor_sha256": contract_receipts[
            "trajectory_anchor"
        ]["sha256"],
        "trajectory_anchor_match": anchor_match,
        "published_unix": time.time(),
    }
    if provisional:
        ready_body["run_purpose"] = RUN_PURPOSE_SHORT_QUALITY
        ready_body["target_epochs"] = list(candidate_epochs)
    _write_new_json(
        run_dir / receipt_directory / f"epoch-{epoch:04d}.json",
        {
            **ready_body,
            "receipt_payload_sha256": canonical_json_sha256(ready_body),
        },
    )


def _cpu_tree(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def validate_topology_gate_spec(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Replay the immutable five-mode gate before any GPU probe/train work."""

    payload, path, observed_sha = _load_json_receipt(
        Path(args.topology_gate_spec),
        args.expected_topology_gate_spec_sha256,
        "Base topology gate specification",
    )
    expected_candidates = list(TOPOLOGY_SPECS)
    if (
        payload.get("format") != TOPOLOGY_GATE_SPEC_FORMAT
        or payload.get("scope") != "SemTalk Base on fresh SHOW five-stage features"
        or payload.get("reference_mode") != OFFICIAL_W1_REFERENCE_MODE
        or payload.get("candidate_modes") != expected_candidates
        or payload.get("topology_matrix") != TOPOLOGY_SPECS
        or payload.get("warmup_updates") != THROUGHPUT_WARMUP_UPDATES
        or payload.get("timed_updates") != THROUGHPUT_TIMED_UPDATES
        or payload.get("optimizer_objective")
        != "published_three_forward_base_objective_v1"
        or payload.get("sampler")
        != "distributed_sampler_drop_last_true_no_padding_duplicates"
        or payload.get("required_measurements")
        != [
            "median_seconds",
            "p90_seconds",
            "p99_seconds",
            "peak_cuda_memory_bytes_all_ranks",
            "oom",
            "all_losses_finite",
            "all_gradients_finite",
            "data_wait_seconds",
            "collective_seconds",
            "batchnorm_inventory",
            "rng_inventory",
            "sample_inventory",
        ]
        or args.topology_mode not in TOPOLOGY_SPECS
    ):
        raise AdaptationContractError(
            "immutable Base topology gate specification changed"
        )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "payload_sha256": canonical_json_sha256(payload),
        "selected_probe_mode": args.topology_mode,
    }


def _save_latest_resume(
    *,
    optimizer: Any,
    run_dir: Path,
    epoch: int,
    optimizer_updates: int,
    frozen_receipt: Mapping[str, Any],
    contract_receipts: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rank_rng_states: Sequence[Mapping[str, Any]],
) -> None:
    if epoch not in RESUME_EPOCHS:
        raise AdaptationContractError(f"epoch {epoch} is not resumable")
    entry = manifest["entries"][-1]
    if entry["epoch"] != epoch:
        raise AdaptationContractError("resume does not follow its candidate")
    checkpoint_path = run_dir / entry["checkpoint"]
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or sha256_file(checkpoint_path) != entry["checkpoint_sha256"]
    ):
        raise AdaptationContractError("resume candidate checkpoint changed")
    resume_path = run_dir / "resume" / "latest_resume.bin"
    payload = {
        "format": "semtalk_show_base_official_adapt_long_resume_v1",
        "completed_epochs": epoch,
        "optimizer_updates": optimizer_updates,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "scheduler": "constant",
        "ema": "absent",
        "dataloader_worker_rng": (
            "not_consumed_by_deterministic_read_only_lmdb_dataset"
        ),
        "candidate_checkpoint": {
            "path": str(checkpoint_path.resolve(strict=True)),
            "sha256": entry["checkpoint_sha256"],
            "model_state_semantic_sha256": entry[
                "model_state_semantic_sha256"
            ],
        },
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "rank_rng_states": _cpu_tree(list(rank_rng_states)),
        "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        "schedule_sha256": contract_receipts["schedule"]["sha256"],
        "trajectory_anchor_sha256": contract_receipts[
            "trajectory_anchor"
        ]["sha256"],
        "trajectory_mode": contract_receipts["trajectory_anchor"]["mode"],
        "trajectory_probe_verified": manifest.get(
            "trajectory_probe_verified"
        ),
        "trajectory_probe": manifest.get("trajectory_probe"),
    }
    _atomic_torch_save(resume_path, payload)
    _atomic_json(
        run_dir / "resume" / "latest_resume.json",
        {
            "format": "semtalk_show_base_official_adapt_long_resume_receipt_v1",
            "status": "complete",
            "completed_epochs": epoch,
            "optimizer_updates": optimizer_updates,
            "path": str(resume_path.resolve(strict=True)),
            "sha256": sha256_file(resume_path),
            "bytes": resume_path.stat().st_size,
            "candidate_checkpoint": payload["candidate_checkpoint"],
            "rank_rng_states": len(rank_rng_states),
            "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
            "schedule_sha256": contract_receipts["schedule"]["sha256"],
            "trajectory_anchor_sha256": contract_receipts[
                "trajectory_anchor"
            ]["sha256"],
            "trajectory_mode": payload["trajectory_mode"],
            "trajectory_probe_verified": payload[
                "trajectory_probe_verified"
            ],
            "completed_unix": time.time(),
        },
    )


def validate_throughput_gate(
    args: argparse.Namespace,
    *,
    frozen_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not args.throughput_gate_report or not args.expected_throughput_gate_sha256:
        raise AdaptationContractError(
            "training mode requires a hash-pinned throughput gate report"
        )
    report, path, observed_sha = _load_json_receipt(
        Path(args.throughput_gate_report),
        args.expected_throughput_gate_sha256,
        "Base throughput gate",
    )
    trajectory_mode = frozen_receipt["long_contract"][
        "trajectory_anchor"
    ]["mode"]
    frozen_compatibility_sha256 = _frozen_gate_compatibility_sha256(
        frozen_receipt
    )
    if (
        report.get("format") != GATE_FORMAT
        or report.get("status") != "pass"
        or report.get("topology_mode") != args.topology_mode
        or report.get("topology_classification")
        != TOPOLOGY_SPECS[args.topology_mode]["classification"]
        or report.get("topology_gate_spec_sha256")
        != args.expected_topology_gate_spec_sha256
        or not isinstance(report.get("topology_independent_input_sha256"), str)
        or len(report["topology_independent_input_sha256"]) != 64
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(report.get("frozen_receipt_sha256")),
        )
        is None
        or report.get("frozen_gate_compatibility_sha256")
        != frozen_compatibility_sha256
        or report.get("topology_independent_input_sha256")
        != _topology_independent_gate_semantic_sha256(frozen_receipt)
        or report.get("topology_receipt_sha256")
        != frozen_receipt["topology"]["receipt_sha256"]
        or report.get("node_count") != NODE_COUNT
        or report.get("local_world_size") != LOCAL_WORLD_SIZE
        or report.get("world_size") != WORLD_SIZE
        or report.get("local_batch_size") != LOCAL_BATCH_SIZE
        or report.get("global_batch_size") != GLOBAL_BATCH_SIZE
        or report.get("updates_per_epoch") != EXPECTED_UPDATES_PER_EPOCH
        or report.get("unique_samples_per_epoch")
        != EXPECTED_UNIQUE_SAMPLES_PER_EPOCH
        or report.get("warmup_updates") != THROUGHPUT_WARMUP_UPDATES
        or report.get("timed_updates") != THROUGHPUT_TIMED_UPDATES
        or report.get("optimizer_updates") != TRAJECTORY_PROBE_UPDATES
        or report.get("trajectory_mode") != trajectory_mode
        or report.get("precision") != args.precision
        or float(report.get("learning_rate", math.nan)) != args.learning_rate
        or report.get("all_losses_finite") is not True
        or report.get("all_gradients_finite") is not True
        or report.get("oom") is not False
        or not isinstance(report.get("samples_per_second"), (int, float))
        or float(report["samples_per_second"]) <= 0.0
        or any(
            not isinstance(report.get(key), (int, float))
            or not math.isfinite(float(report[key]))
            or float(report[key]) <= 0.0
            for key in ("median_seconds", "p90_seconds", "p99_seconds")
        )
        or not (
            float(report["median_seconds"])
            <= float(report["p90_seconds"])
            <= float(report["p99_seconds"])
        )
        or not isinstance(report.get("peak_cuda_memory_bytes_all_ranks"), list)
        or len(report["peak_cuda_memory_bytes_all_ranks"]) != WORLD_SIZE
        or any(
            type(value) is not int or value <= 0
            for value in report["peak_cuda_memory_bytes_all_ranks"]
        )
        or not isinstance(report.get("data_wait_seconds"), dict)
        or not isinstance(report.get("collective_seconds"), dict)
        or not isinstance(report.get("batchnorm_inventory"), list)
        or len(report["batchnorm_inventory"]) != WORLD_SIZE
        or not isinstance(report.get("rng_inventory"), list)
        or len(report["rng_inventory"]) != WORLD_SIZE
        or report.get("sample_inventory", {}).get("sampler_drop_last") is not True
        or report.get("sample_inventory", {}).get("padding_duplicates") != 0
        or report.get("sample_inventory", {}).get("full_epoch_samples")
        != EXPECTED_UNIQUE_SAMPLES_PER_EPOCH
        or report.get("sample_inventory", {}).get("full_epoch_unique_samples")
        != EXPECTED_UNIQUE_SAMPLES_PER_EPOCH
        or report.get("receipt_sha256")
        != canonical_json_sha256(
            {key: value for key, value in report.items() if key != "receipt_sha256"}
        )
    ):
        raise AdaptationContractError(
            "throughput gate does not bind the exact training protocol"
        )
    trajectory_probe = report.get("trajectory_probe")
    if trajectory_mode == FRESH_TRAJECTORY_MODE:
        trajectory_probe = _validate_trajectory_probe(
            trajectory_probe,
            label="throughput trajectory probe",
        )
    elif trajectory_probe is not None:
        raise AdaptationContractError(
            "legacy throughput gate must not claim a fresh trajectory probe"
        )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "topology_mode": args.topology_mode,
        "samples_per_second": float(report["samples_per_second"]),
        "seconds_per_update": float(report["seconds_per_update"]),
        "median_seconds": float(report["median_seconds"]),
        "p90_seconds": float(report["p90_seconds"]),
        "p99_seconds": float(report["p99_seconds"]),
        "estimated_training_seconds": float(
            report["estimated_training_seconds"]
        ),
        "trajectory_mode": trajectory_mode,
        "trajectory_probe": trajectory_probe,
        "gate_frozen_receipt_sha256": report[
            "frozen_receipt_sha256"
        ],
        "frozen_gate_compatibility_sha256": (
            frozen_compatibility_sha256
        ),
    }


def validate_topology_selection(
    args: argparse.Namespace,
    *,
    throughput_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind formal training to the sealed result of all five real probes."""

    report, path, observed_sha = _load_json_receipt(
        Path(args.topology_selection_report),
        args.expected_topology_selection_sha256,
        "Base topology selection",
    )
    selected = report.get("selected")
    probes = report.get("probes")
    quality_reports = report.get("quality_reports")
    quality_decisions = report.get("quality_decisions")
    if (
        report.get("format") != TOPOLOGY_SELECTION_FORMAT
        or report.get("status") != "pass"
        or report.get("topology_gate_spec_sha256")
        != args.expected_topology_gate_spec_sha256
        or report.get("reference_mode") != OFFICIAL_W1_REFERENCE_MODE
        or report.get("candidate_modes") != list(TOPOLOGY_SPECS)
        or not isinstance(probes, list)
        or [probe.get("mode") for probe in probes]
        != list(TOPOLOGY_SPECS)
        or any(
            not isinstance(probe, dict)
            or probe.get("status") != "pass"
            or not isinstance(probe.get("report_sha256"), str)
            or len(probe["report_sha256"]) != 64
            for probe in probes
        )
        or not isinstance(quality_reports, list)
        or [quality.get("mode") for quality in quality_reports]
        != list(TOPOLOGY_SPECS)
        or any(
            not isinstance(quality, dict)
            or quality.get("report_sha256") is None
            or len(str(quality.get("report_sha256"))) != 64
            or quality.get("candidate_fgd") is None
            or set(quality["candidate_fgd"]) != {"1", "2", "4", "8"}
            for quality in quality_reports
        )
        or not isinstance(quality_decisions, dict)
        or set(quality_decisions) != set(TOPOLOGY_SPECS)
        or not isinstance(selected, dict)
        or selected.get("mode") != args.topology_mode
        or selected.get("report_sha256") != throughput_gate["sha256"]
        or selected.get("classification")
        != TOPOLOGY_SPECS[args.topology_mode]["classification"]
        or selected.get("precision")
        != TOPOLOGY_SPECS[args.topology_mode]["precision"]
        or selected.get("formal_training_eligible") is not True
        or selected
        != next(
            (
                probe
                for probe in probes
                if probe.get("mode") == args.topology_mode
            ),
            None,
        )
        or report.get("w1_trajectory_equivalence_claimed_for_selected")
        is not False
        or report.get("selection_policy")
        != "fastest_quality_safe_finite_under_24h_all_measured_topologies_v2"
        or report.get("selection_decision_branch")
        != "fastest_quality_safe_finite_under_24h"
        or report.get("quality_gate_policy", {}).get(
            "raw_prediction_replay_required"
        )
        is not True
        or report.get("quality_gate_policy", {}).get(
            "maximum_training_seconds"
        )
        != 86_400
        or report.get("receipt_sha256")
        != canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "receipt_sha256"
            }
        )
    ):
        raise AdaptationContractError(
            "formal Base topology selection is missing, forged, or stale"
        )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "selected": dict(selected),
        "probe_report_sha256": {
            probe["mode"]: probe["report_sha256"] for probe in probes
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official All-Speakers SemTalk Base adaptation on SHOW"
    )
    parser.allow_abbrev = False
    parser.add_argument(
        "--mode",
        choices=("throughput_gate", SHORT_QUALITY_MODE, "train"),
        required=True,
    )
    parser.add_argument("--official-base-checkpoint", required=True)
    parser.add_argument("--train-lmdb", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--expected-dataset-summary-sha256", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--prerequisite-selection-json")
    parser.add_argument("--expected-prerequisite-selection-sha256")
    parser.add_argument("--schedule-json", required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--topology-gate-spec", required=True)
    parser.add_argument(
        "--expected-topology-gate-spec-sha256", required=True
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=(LEGACY_TRAJECTORY_MODE, FRESH_TRAJECTORY_MODE),
        required=True,
    )
    parser.add_argument("--trajectory-anchor-json")
    parser.add_argument(
        "--expected-trajectory-anchor-sha256",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--throughput-gate-report")
    parser.add_argument("--expected-throughput-gate-sha256")
    parser.add_argument("--topology-selection-report")
    parser.add_argument("--expected-topology-selection-sha256")
    parser.add_argument("--formal-node-rank", type=int, required=True)
    parser.add_argument(
        "--formal-host-slot", type=int, choices=tuple(FORMAL_HOST_BY_SLOT),
        required=True,
    )
    parser.add_argument("--formal-master-addr", required=True)
    parser.add_argument("--formal-master-port", type=int, required=True)
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument(
        "--topology-mode", choices=tuple(TOPOLOGY_SPECS), required=True
    )
    parser.add_argument("--local-batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=43)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    topology = _activate_topology(args)
    host_slots = _active_host_slots(args)
    reject_forbidden_source_labels(
        args.run_name,
        args.output_root,
        args.official_base_checkpoint,
        args.train_lmdb,
        args.dataset_summary,
        args.lineage_manifest,
        args.prerequisite_selection_json or "",
        args.schedule_json,
        args.topology_gate_spec,
        args.trajectory_anchor_json,
        args.throughput_gate_report or "",
        args.topology_selection_report or "",
        args.formal_master_addr,
        args.formal_run_id,
    )
    if (
        not args.run_name
        or "/" in args.run_name
        or args.run_name in {".", ".."}
    ):
        raise AdaptationContractError("--run-name must be one path component")
    if (args.prerequisite_selection_json is None) != (
        args.expected_prerequisite_selection_sha256 is None
    ):
        raise AdaptationContractError(
            "prerequisite selection JSON and external SHA-256 must be "
            "supplied together"
        )
    if args.expected_prerequisite_selection_sha256 is not None:
        selected_contract.require_sha256(
            args.expected_prerequisite_selection_sha256,
            "prerequisite selection expected SHA-256",
        )
    anchor_pair_supplied = (
        args.trajectory_anchor_json is not None
        and args.expected_trajectory_anchor_sha256 is not None
    )
    if (args.trajectory_anchor_json is None) != (
        args.expected_trajectory_anchor_sha256 is None
    ):
        raise AdaptationContractError(
            "trajectory anchor JSON and external SHA-256 must be supplied "
            "together"
        )
    if (
        args.trajectory_mode != FRESH_TRAJECTORY_MODE
        or anchor_pair_supplied
        or args.prerequisite_selection_json is None
    ):
        raise AdaptationContractError(
            "formal Base training requires the hash-pinned five-stage SHOW "
            "selection and forbids legacy released-VQ feature authority"
        )
    if (
        args.formal_node_rank not in range(topology["node_count"])
        or args.formal_host_slot not in FORMAL_HOST_BY_SLOT
        or host_slots[args.formal_node_rank] != args.formal_host_slot
        or (
            topology["node_count"] == 2
            and args.formal_host_slot != args.formal_node_rank
        )
        or os.uname().nodename != FORMAL_HOST_BY_SLOT[args.formal_host_slot]
        or not re.fullmatch(r"[A-Za-z0-9.-]+", args.formal_master_addr)
        or not (1024 <= args.formal_master_port <= 65535)
        or re.fullmatch(r"[A-Za-z0-9._-]{8,128}", args.formal_run_id)
        is None
    ):
        raise AdaptationContractError(
            "formal selected-topology node/host/static-rendezvous identity "
            "changed"
        )
    if args.local_batch_size != topology["local_batch_size"]:
        raise AdaptationContractError(
            "local batch size differs from the selected topology"
        )
    expected_epochs = (
        SHORT_QUALITY_TOTAL_EPOCHS
        if args.mode == SHORT_QUALITY_MODE
        else TOTAL_EPOCHS
    )
    if args.epochs != expected_epochs:
        if args.mode == SHORT_QUALITY_MODE:
            raise AdaptationContractError(
                "provisional short-quality adaptation must run exactly "
                f"{SHORT_QUALITY_TOTAL_EPOCHS} epochs"
            )
        raise AdaptationContractError(
            f"long official adaptation must run exactly {TOTAL_EPOCHS} epochs"
        )
    if args.learning_rate != float(topology["learning_rate"]):
        raise AdaptationContractError(
            "Base Adam learning rate differs from the immutable topology "
            "matrix entry"
        )
    if args.precision != topology["precision"]:
        raise AdaptationContractError(
            "Base precision differs from the immutable topology matrix entry"
        )
    if args.loader_workers < 0 or args.loader_workers > 16:
        raise AdaptationContractError("loader workers must be in [0,16]")
    if args.mode == "throughput_gate" and (
        args.throughput_gate_report is not None
        or args.expected_throughput_gate_sha256 is not None
        or args.topology_selection_report is not None
        or args.expected_topology_selection_sha256 is not None
    ):
        raise AdaptationContractError(
            "throughput_gate mode cannot consume a previous gate"
        )
    if args.mode == SHORT_QUALITY_MODE and (
        args.throughput_gate_report is None
        or args.expected_throughput_gate_sha256 is None
        or args.topology_selection_report is not None
        or args.expected_topology_selection_sha256 is not None
    ):
        raise AdaptationContractError(
            "short_quality requires one hash-pinned topology throughput gate "
            "and forbids topology selection"
        )
    if args.mode == "train" and (
        args.throughput_gate_report is None
        or args.expected_throughput_gate_sha256 is None
        or args.topology_selection_report is None
        or args.expected_topology_selection_sha256 is None
    ):
        raise AdaptationContractError(
            "formal training requires one topology selected by the sealed "
            "five-mode gate"
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


def _prepare_deterministic_environment(*, seed: int) -> None:
    expected_hash_seed = str(seed)
    observed_hash_seed = os.environ.get("PYTHONHASHSEED")
    if observed_hash_seed != expected_hash_seed:
        raise AdaptationContractError(
            "PYTHONHASHSEED must be set before interpreter startup to "
            f"{expected_hash_seed}, got {observed_hash_seed!r}"
        )
    expected = ":4096:8"
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed not in {None, expected}:
        raise AdaptationContractError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or exactly :4096:8"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected


def _configure_deterministic_runtime(
    torch_module: Any,
    *,
    seed: int,
) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.set_float32_matmul_precision("highest")


def _configure_rank_training_rng(
    torch_module: Any,
    *,
    seed: int,
    rank: int,
) -> None:
    """Give dropout/masking independent deterministic streams per rank."""

    import random

    import numpy as np

    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch_module.manual_seed(rank_seed)
    torch_module.cuda.manual_seed_all(rank_seed)


def _distributed_context(
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        group_rank = int(os.environ["GROUP_RANK"])
        role_rank = int(os.environ["ROLE_RANK"])
        role_world_size = int(os.environ["ROLE_WORLD_SIZE"])
        master_port = int(os.environ["MASTER_PORT"])
    except (KeyError, ValueError) as error:
        raise AdaptationContractError(
            "launch through the exact static topology-gate torchrun contract"
        ) from error
    expected_rank = args.formal_node_rank * LOCAL_WORLD_SIZE + local_rank
    if (
        world_size != WORLD_SIZE
        or local_world_size != LOCAL_WORLD_SIZE
        or group_rank != args.formal_node_rank
        or role_rank != rank
        or role_world_size != WORLD_SIZE
        or rank != expected_rank
        or not (0 <= local_rank < LOCAL_WORLD_SIZE)
        or os.environ.get("MASTER_ADDR") != args.formal_master_addr
        or master_port != args.formal_master_port
        or os.uname().nodename != FORMAL_HOST_BY_SLOT[args.formal_host_slot]
    ):
        raise AdaptationContractError(
            "official adaptation requires the exact selected rank/node/host topology"
        )
    return rank, local_rank, world_size


def _distributed_topology_receipt(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
) -> dict[str, Any]:
    import torch.distributed as dist

    local = {
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": args.formal_node_rank,
        "host_slot": args.formal_host_slot,
        "hostname": os.uname().nodename,
        "master_addr": os.environ["MASTER_ADDR"],
        "master_port": int(os.environ["MASTER_PORT"]),
        "formal_run_id": args.formal_run_id,
    }
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, local)
    host_slots = _active_host_slots(args)
    expected = [
        {
            "rank": node_rank * LOCAL_WORLD_SIZE + local_rank_value,
            "local_rank": local_rank_value,
            "node_rank": node_rank,
            "host_slot": host_slots[node_rank],
            "hostname": FORMAL_HOST_BY_SLOT[host_slots[node_rank]],
            "master_addr": args.formal_master_addr,
            "master_port": args.formal_master_port,
            "formal_run_id": args.formal_run_id,
        }
        for node_rank in range(NODE_COUNT)
        for local_rank_value in range(LOCAL_WORLD_SIZE)
    ]
    if gathered != expected:
        raise AdaptationContractError(
            "distributed selected rank/node/host evidence is not exact"
        )
    receipt = {
        "format": "semtalk_show_base_topology_receipt_v1",
        "topology_mode": args.topology_mode,
        "classification": TOPOLOGY_SPECS[args.topology_mode][
            "classification"
        ],
        "backend": "nccl",
        "node_count": NODE_COUNT,
        "local_world_size": LOCAL_WORLD_SIZE,
        "world_size": WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "unique_samples_per_epoch": EXPECTED_UNIQUE_SAMPLES_PER_EPOCH,
        "master_addr": args.formal_master_addr,
        "master_port": args.formal_master_port,
        "formal_run_id": args.formal_run_id,
        "ranks": expected,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def _seed_loader_worker(worker_id: int) -> None:
    import random

    import numpy as np
    import torch

    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        raise AdaptationContractError("loader worker context is unavailable")
    worker_info.dataset.assert_source_unchanged(full_hash=False)


def _create_dataloader(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    dataset_receipt: Mapping[str, Any],
) -> Any:
    import torch
    from dataloaders.show_base import LMDBNPZDataset

    dataset_args = SimpleNamespace(
        # Never reopen the caller's original alias after rank-0 preflight.
        train_path=dataset_receipt["lmdb"],
        pose_length=POSE_LENGTH,
        lmdb_inode_binding=dataset_receipt["lmdb_inode_binding"],
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
        # Do not let DistributedSampler pad by duplicating samples.  With the
        # frozen 127286-entry SHOW train LMDB this yields exactly 127232 unique
        # samples for global batch 64, or 126976 for global batch 512.
        drop_last=True,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {}
    if args.loader_workers > 0:
        loader_kwargs["multiprocessing_context"] = "fork"
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=LOCAL_BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=args.loader_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.loader_workers > 0,
        worker_init_fn=_seed_loader_worker,
        generator=generator,
        **loader_kwargs,
    )
    if len(loader) != EXPECTED_UPDATES_PER_EPOCH:
        raise AdaptationContractError(
            f"updates/epoch {len(loader)} != {EXPECTED_UPDATES_PER_EPOCH}"
        )
    return loader, sampler


def _distributed_verify_loader_source(
    loader: Any,
    *,
    rank: int,
    full_hash_on_rank0: bool,
) -> None:
    import torch.distributed as dist

    local: dict[str, Any]
    try:
        loader.dataset.assert_source_unchanged(
            full_hash=full_hash_on_rank0 and rank == 0
        )
        local = {"rank": rank, "status": "verified"}
    except BaseException as error:
        local = {
            "rank": rank,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, local)
    if [item.get("rank") for item in gathered if isinstance(item, dict)] != list(
        range(WORLD_SIZE)
    ) or any(
        not isinstance(item, dict) or item.get("status") != "verified"
        for item in gathered
    ):
        raise AdaptationContractError(
            f"distributed immutable Base LMDB verification failed: {gathered}"
        )


def _frozen_receipt(
    *,
    source: Mapping[str, Any],
    official_base: Mapping[str, Any],
    speaker_initialization: Mapping[str, Any],
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    long_contract: Mapping[str, Any],
    topology: Mapping[str, Any],
    run_purpose: str,
    target_epochs: Sequence[int],
) -> dict[str, Any]:
    payload = {
        "format": "semtalk_show_base_official_adapt_frozen_inputs_v1",
        "run_purpose": run_purpose,
        "target_epochs": list(target_epochs),
        "source": dict(source),
        "official_base": dict(official_base),
        "speaker_initialization": dict(speaker_initialization),
        "dataset": dict(dataset),
        "protocol": dict(protocol),
        "long_contract": dict(long_contract),
        "topology": dict(topology),
    }
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    return payload


def _frozen_gate_compatibility_sha256(
    frozen_receipt: Mapping[str, Any],
) -> str:
    """Bind a gate to training semantics while excluding its run purpose.

    Throughput measurement, provisional quality training, and the eventual
    formal trajectory are three separate immutable runs.  Their full frozen
    receipts must therefore state different purposes and target epochs.  This
    projection removes exactly those run-local declarations (plus the
    self-hash), while retaining every source, dataset, objective, topology,
    and rendezvous binding needed to replay the gate safely.
    """

    required = {"run_purpose", "target_epochs", "receipt_sha256"}
    if not isinstance(frozen_receipt, Mapping) or not required.issubset(
        frozen_receipt
    ):
        raise AdaptationContractError(
            "frozen Base receipt lacks explicit run purpose/target epochs"
        )
    payload = {
        key: value
        for key, value in frozen_receipt.items()
        if key not in required
    }
    return canonical_json_sha256(payload)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise AdaptationContractError("invalid topology-gate percentile")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _batchnorm_inventory(model: Any) -> list[dict[str, Any]]:
    import torch

    result: list[dict[str, Any]] = []
    for name, module in _unwrap_model(model).named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            result.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "num_features": int(module.num_features),
                    "training": bool(module.training),
                    "track_running_stats": bool(module.track_running_stats),
                    "running_mean_sha256": (
                        _tensor_sha256(module.running_mean)
                        if module.running_mean is not None
                        else None
                    ),
                    "running_var_sha256": (
                        _tensor_sha256(module.running_var)
                        if module.running_var is not None
                        else None
                    ),
                    "num_batches_tracked": (
                        int(module.num_batches_tracked.item())
                        if module.num_batches_tracked is not None
                        else None
                    ),
                }
            )
    return result


def _topology_independent_gate_semantic_sha256(
    frozen_receipt: Mapping[str, Any],
) -> str:
    official = dict(frozen_receipt["official_base"])
    official.pop("file_binding_scope", None)
    official.pop("node_local_files", None)
    dataset = dict(frozen_receipt["dataset"])
    dataset.pop("lmdb_binding_scope", None)
    dataset.pop("node_lmdb_inode_bindings", None)
    protocol = frozen_receipt["protocol"]
    long_contract = frozen_receipt["long_contract"]
    return canonical_json_sha256(
        {
            "source": {
                key: frozen_receipt["source"][key]
                for key in (
                    "origin",
                    "commit",
                    "tree",
                    "clean",
                    "entrypoint_sha256",
                )
            },
            "official_base": official,
            "speaker_initialization": frozen_receipt[
                "speaker_initialization"
            ],
            "dataset": dataset,
            "schedule": long_contract["schedule"],
            "forward_contract": protocol["forward_contract"],
            "loss": protocol["loss"],
            "precision": protocol["precision"],
            "target_dataset": protocol["target_dataset"],
            "target_speaker_scope": protocol["target_speaker_scope"],
            "vq_models_in_training_graph": protocol[
                "vq_models_in_training_graph"
            ],
        }
    )


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
    torch.cuda.reset_peak_memory_stats(device)
    sampler.set_epoch(0)
    iterator = iter(loader)
    last_metrics: dict[str, float] = {}
    sample_order = hashlib.sha256()
    sample_count = 0
    observed_indices: list[int] = []
    optimizer_update = 0
    batchnorm_before = _batchnorm_inventory(model)
    for _ in range(THROUGHPUT_WARMUP_UPDATES):
        batch = next(iterator)
        observed_indices.extend(_batch_sample_indices(batch))
        optimizer_update += 1
        sample_count += _record_sample_order(
            sample_order,
            batch,
            rank=rank,
            optimizer_update=optimizer_update,
        )
        last_metrics = one_optimizer_update(
            model,
            optimizer,
            _move_batch(batch, device),
            device=device,
            precision=args.precision,
            epoch=0,
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    update_seconds: list[float] = []
    data_wait_seconds: list[float] = []
    for _ in range(THROUGHPUT_TIMED_UPDATES):
        wait_started = time.perf_counter()
        batch = next(iterator)
        data_wait_seconds.append(time.perf_counter() - wait_started)
        observed_indices.extend(_batch_sample_indices(batch))
        optimizer_update += 1
        sample_count += _record_sample_order(
            sample_order,
            batch,
            rank=rank,
            optimizer_update=optimizer_update,
        )
        update_started = time.perf_counter()
        last_metrics = one_optimizer_update(
            model,
            optimizer,
            _move_batch(batch, device),
            device=device,
            precision=args.precision,
            epoch=0,
        )
        torch.cuda.synchronize(device)
        update_seconds.append(time.perf_counter() - update_started)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - started
    _assert_distributed_finite(_all_finite(model.parameters()), device)
    _distributed_verify_loader_source(
        loader,
        rank=rank,
        full_hash_on_rank0=True,
    )
    trajectory_mode = frozen_receipt["long_contract"][
        "trajectory_anchor"
    ]["mode"]
    trajectory_probe = (
        _distributed_trajectory_probe(
            model,
            optimizer,
            TRAJECTORY_PROBE_UPDATES,
            rank=rank,
            device=device,
            sample_order_sha256=sample_order.hexdigest(),
            sample_count=sample_count,
        )
        if trajectory_mode == FRESH_TRAJECTORY_MODE
        else None
    )
    # Measure a small fixed collective separately from the training update so
    # the gate can diagnose cross-node fabric without inflating the ETA.
    collective_seconds: list[float] = []
    collective_tensor = torch.ones(1, dtype=torch.float64, device=device)
    for _ in range(10):
        torch.cuda.synchronize(device)
        collective_started = time.perf_counter()
        dist.all_reduce(collective_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        collective_seconds.append(time.perf_counter() - collective_started)

    consumed_per_rank = EXPECTED_UPDATES_PER_EPOCH * LOCAL_BATCH_SIZE
    full_epoch_indices = list(iter(sampler))[:consumed_per_rank]
    local_observation = {
        "rank": rank,
        "elapsed_seconds": elapsed,
        "update_seconds": update_seconds,
        "data_wait_seconds": data_wait_seconds,
        "collective_seconds": collective_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "all_losses_finite": all(
            math.isfinite(float(value)) for value in last_metrics.values()
        ),
        "all_gradients_finite": _all_finite(
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
        "oom": False,
        "batchnorm_before": batchnorm_before,
        "batchnorm_after": _batchnorm_inventory(model),
        "rng_after": _rng_state_hashes(device),
        "observed_indices": observed_indices,
        "full_epoch_indices": full_epoch_indices,
    }
    gathered_observations: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered_observations, local_observation)
    if [item.get("rank") for item in gathered_observations] != list(
        range(WORLD_SIZE)
    ):
        raise AdaptationContractError("topology-gate rank inventory changed")
    critical_update_seconds = [
        max(
            float(observation["update_seconds"][update_index])
            + float(observation["data_wait_seconds"][update_index])
            for observation in gathered_observations
        )
        for update_index in range(THROUGHPUT_TIMED_UPDATES)
    ]
    all_observed_indices = [
        index
        for observation in gathered_observations
        for index in observation["observed_indices"]
    ]
    all_epoch_indices = [
        index
        for observation in gathered_observations
        for index in observation["full_epoch_indices"]
    ]
    expected_probe_samples = TRAJECTORY_PROBE_UPDATES * GLOBAL_BATCH_SIZE
    if (
        len(all_observed_indices) != expected_probe_samples
        or len(set(all_observed_indices)) != expected_probe_samples
        or len(all_epoch_indices) != EXPECTED_UNIQUE_SAMPLES_PER_EPOCH
        or len(set(all_epoch_indices)) != EXPECTED_UNIQUE_SAMPLES_PER_EPOCH
        or any(
            index < 0 or index >= EXPECTED_TRAIN_SAMPLES
            for index in all_epoch_indices
        )
    ):
        raise AdaptationContractError(
            "DistributedSampler sample inventory is padded or duplicated"
        )
    median_seconds = _percentile(critical_update_seconds, 0.5)
    p90_seconds = _percentile(critical_update_seconds, 0.9)
    p99_seconds = _percentile(critical_update_seconds, 0.99)
    elapsed = max(
        float(observation["elapsed_seconds"])
        for observation in gathered_observations
    )
    if rank == 0:
        report = {
            "format": GATE_FORMAT,
            "status": "pass",
            "topology_mode": args.topology_mode,
            "topology_classification": TOPOLOGY_SPECS[
                args.topology_mode
            ]["classification"],
            "topology_gate_spec_sha256": (
                args.expected_topology_gate_spec_sha256
            ),
            "topology_independent_input_sha256": (
                _topology_independent_gate_semantic_sha256(frozen_receipt)
            ),
            "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
            "frozen_gate_compatibility_sha256": (
                _frozen_gate_compatibility_sha256(frozen_receipt)
            ),
            "topology_receipt_sha256": frozen_receipt["topology"][
                "receipt_sha256"
            ],
            "node_count": NODE_COUNT,
            "local_world_size": LOCAL_WORLD_SIZE,
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
            "unique_samples_per_epoch": EXPECTED_UNIQUE_SAMPLES_PER_EPOCH,
            "warmup_updates": THROUGHPUT_WARMUP_UPDATES,
            "timed_updates": THROUGHPUT_TIMED_UPDATES,
            "precision": args.precision,
            "learning_rate": args.learning_rate,
            "elapsed_seconds": elapsed,
            "seconds_per_update": median_seconds,
            "median_seconds": median_seconds,
            "p90_seconds": p90_seconds,
            "p99_seconds": p99_seconds,
            "samples_per_second": (
                GLOBAL_BATCH_SIZE / median_seconds
            ),
            "estimated_training_seconds": (
                median_seconds * EXPECTED_UPDATES_PER_EPOCH * TOTAL_EPOCHS
            ),
            "estimated_epochs": TOTAL_EPOCHS,
            "last_metrics": last_metrics,
            "all_losses_finite": True,
            "all_gradients_finite": all(
                observation["all_gradients_finite"]
                for observation in gathered_observations
            ),
            "oom": any(
                observation["oom"] for observation in gathered_observations
            ),
            "optimizer_updates": (
                THROUGHPUT_WARMUP_UPDATES + THROUGHPUT_TIMED_UPDATES
            ),
            "trajectory_mode": trajectory_mode,
            "trajectory_probe": trajectory_probe,
            "peak_cuda_memory_bytes_all_ranks": [
                int(observation["peak_cuda_memory_bytes"])
                for observation in gathered_observations
            ],
            "data_wait_seconds": {
                "median": _percentile(
                    [
                        value
                        for observation in gathered_observations
                        for value in observation["data_wait_seconds"]
                    ],
                    0.5,
                ),
                "p99": _percentile(
                    [
                        value
                        for observation in gathered_observations
                        for value in observation["data_wait_seconds"]
                    ],
                    0.99,
                ),
            },
            "collective_seconds": {
                "probe": "ten_scalar_nccl_all_reduce_calls",
                "median": _percentile(
                    [
                        value
                        for observation in gathered_observations
                        for value in observation["collective_seconds"]
                    ],
                    0.5,
                ),
                "p99": _percentile(
                    [
                        value
                        for observation in gathered_observations
                        for value in observation["collective_seconds"]
                    ],
                    0.99,
                ),
            },
            "batchnorm_inventory": [
                {
                    "rank": observation["rank"],
                    "before": observation["batchnorm_before"],
                    "after": observation["batchnorm_after"],
                }
                for observation in gathered_observations
            ],
            "rng_inventory": [
                {
                    "rank": observation["rank"],
                    **observation["rng_after"],
                }
                for observation in gathered_observations
            ],
            "sample_inventory": {
                "sampler_drop_last": True,
                "padding_duplicates": 0,
                "probe_samples": expected_probe_samples,
                "probe_unique_samples": len(set(all_observed_indices)),
                "full_epoch_samples": len(all_epoch_indices),
                "full_epoch_unique_samples": len(set(all_epoch_indices)),
                "dataset_samples": EXPECTED_TRAIN_SAMPLES,
                "dropped_tail_samples": (
                    EXPECTED_TRAIN_SAMPLES - len(all_epoch_indices)
                ),
                "full_epoch_sorted_indices_sha256": hashlib.sha256(
                    b"".join(
                        int(index).to_bytes(8, "little", signed=False)
                        for index in sorted(all_epoch_indices)
                    )
                ).hexdigest(),
            },
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
    contract_receipts: Mapping[str, Any],
    throughput_receipt: Mapping[str, Any],
) -> None:
    import random

    import numpy as np
    import torch
    import torch.distributed as dist

    provisional = args.mode == SHORT_QUALITY_MODE
    if args.mode not in {SHORT_QUALITY_MODE, "train"}:
        raise AdaptationContractError(
            "candidate training requires short_quality or train mode"
        )
    target_total_epochs = (
        SHORT_QUALITY_TOTAL_EPOCHS if provisional else TOTAL_EPOCHS
    )
    target_candidate_epochs = (
        SHORT_QUALITY_EPOCHS if provisional else CANDIDATE_EPOCHS
    )
    run_purpose = (
        RUN_PURPOSE_SHORT_QUALITY
        if provisional
        else RUN_PURPOSE_FORMAL_TRAINING
    )
    manifest_name = (
        "short_quality_candidate_manifest.json"
        if provisional
        else "candidate_manifest.json"
    )
    status_name = (
        "short_quality_status.json" if provisional else "status.json"
    )
    trajectory_mode = contract_receipts["trajectory_anchor"]["mode"]
    fresh_trajectory = trajectory_mode == FRESH_TRAJECTORY_MODE
    expected_trajectory_probe = throughput_receipt.get(
        "trajectory_probe"
    )
    if fresh_trajectory:
        expected_trajectory_probe = _validate_trajectory_probe(
            expected_trajectory_probe,
            label="throughput trajectory probe",
        )
    manifest: dict[str, Any] = {
        "format": (
            SHORT_QUALITY_MANIFEST_FORMAT
            if provisional
            else MANIFEST_FORMAT
        ),
        "status": "running",
        "candidate_epochs": list(target_candidate_epochs),
        "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
        "schedule_sha256": contract_receipts["schedule"]["sha256"],
        "trajectory_anchor_sha256": contract_receipts[
            "trajectory_anchor"
        ]["sha256"],
        "throughput_gate": dict(throughput_receipt),
        "trajectory_mode": trajectory_mode,
        "trajectory_probe_verified": False if fresh_trajectory else None,
        "trajectory_probe": None,
        "entries": [],
        "entries_sha256": canonical_json_sha256([]),
    }
    if provisional:
        manifest["run_purpose"] = run_purpose
        manifest["target_epochs"] = list(target_candidate_epochs)
    if rank == 0:
        _atomic_json(run_dir / manifest_name, manifest)
    optimizer_updates = 0
    probe_sample_order = hashlib.sha256()
    probe_sample_count = 0
    started_unix = time.time()
    model.train()
    for epoch_index in range(target_total_epochs):
        sampler.set_epoch(epoch_index)
        epoch_sums = {
            name: 0.0
            for name in (
                *LOSS_COMPONENTS,
                "total",
                "gradient_norm_preclip",
            )
        }
        for batch in loader:
            if fresh_trajectory and optimizer_updates < TRAJECTORY_PROBE_UPDATES:
                probe_sample_count += _record_sample_order(
                    probe_sample_order,
                    batch,
                    rank=rank,
                    optimizer_update=optimizer_updates + 1,
                )
            metrics = one_optimizer_update(
                model,
                optimizer,
                _move_batch(batch, device),
                device=device,
                precision=args.precision,
                epoch=epoch_index,
            )
            optimizer_updates += 1
            if fresh_trajectory and optimizer_updates == TRAJECTORY_PROBE_UPDATES:
                _distributed_verify_loader_source(
                    loader,
                    rank=rank,
                    # Rank-0 already content-hashed the exact pinned inode at
                    # this process preflight.  Recheck inode/metadata here;
                    # the mandatory second content hash is at strict finalize.
                    full_hash_on_rank0=False,
                )
                observed_probe = _distributed_trajectory_probe(
                    model,
                    optimizer,
                    optimizer_updates,
                    rank=rank,
                    device=device,
                    sample_order_sha256=probe_sample_order.hexdigest(),
                    sample_count=probe_sample_count,
                )
                matched_probe = _require_matching_trajectory_probe(
                    expected_trajectory_probe,
                    observed_probe,
                )
                if rank == 0:
                    manifest["trajectory_probe_verified"] = True
                    manifest["trajectory_probe"] = matched_probe
                    _atomic_json(
                        run_dir / manifest_name,
                        manifest,
                    )
                dist.barrier()
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
        if completed_epoch in target_candidate_epochs:
            _distributed_verify_loader_source(
                loader,
                rank=rank,
                full_hash_on_rank0=False,
            )
            if rank == 0:
                _save_candidate(
                    model=model,
                    optimizer=optimizer,
                    run_dir=run_dir,
                    epoch=completed_epoch,
                    optimizer_updates=optimizer_updates,
                    frozen_receipt=frozen_receipt,
                    contract_receipts=contract_receipts,
                    manifest=manifest,
                    provisional=provisional,
                )
            dist.barrier()
        if not provisional and completed_epoch in RESUME_EPOCHS:
            local_rng_state = {
                "rank": rank,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state().cpu(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state(
                    device
                ).cpu(),
            }
            rank_rng_states: list[Any] = [
                None for _ in range(WORLD_SIZE)
            ]
            dist.all_gather_object(rank_rng_states, local_rng_state)
            if [state["rank"] for state in rank_rng_states] != list(
                range(WORLD_SIZE)
            ):
                raise AdaptationContractError(
                    "rank RNG state ordering changed"
                )
            if rank == 0:
                _save_latest_resume(
                    optimizer=optimizer,
                    run_dir=run_dir,
                    epoch=completed_epoch,
                    optimizer_updates=optimizer_updates,
                    frozen_receipt=frozen_receipt,
                    contract_receipts=contract_receipts,
                    manifest=manifest,
                    rank_rng_states=rank_rng_states,
                )
            dist.barrier()
        if rank == 0:
            metric_record = {
                "format": (
                    SHORT_QUALITY_EPOCH_METRIC_FORMAT
                    if provisional
                    else "semtalk_show_base_long_epoch_metric_v1"
                ),
                "epoch": completed_epoch,
                "optimizer_updates": optimizer_updates,
                "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "metrics": epoch_metrics,
                "all_finite": all(
                    math.isfinite(float(value))
                    for value in epoch_metrics.values()
                ),
                "completed_unix": time.time(),
            }
            if provisional:
                metric_record["run_purpose"] = run_purpose
                metric_record["target_epochs"] = list(
                    target_candidate_epochs
                )
            _append_epoch_metric(
                run_dir / "epoch_metrics.jsonl",
                metric_record,
            )
            running_status = {
                "format": (
                    SHORT_QUALITY_STATUS_FORMAT
                    if provisional
                    else STATUS_FORMAT
                ),
                "status": "running",
                "completed_epochs": completed_epoch,
                "optimizer_updates": optimizer_updates,
                "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
                "last_epoch_metrics": epoch_metrics,
                "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
                "candidate_manifest_sha256": sha256_file(
                    run_dir / manifest_name
                ),
                "started_unix": started_unix,
                "updated_unix": time.time(),
            }
            if provisional:
                running_status["run_purpose"] = run_purpose
                running_status["target_epochs"] = list(
                    target_candidate_epochs
                )
            _atomic_json(run_dir / status_name, running_status)
    _assert_distributed_finite(_all_finite(model.parameters()), device)
    _distributed_verify_loader_source(
        loader,
        rank=rank,
        full_hash_on_rank0=True,
    )
    if rank == 0:
        if fresh_trajectory and manifest["trajectory_probe_verified"] is not True:
            raise AdaptationContractError(
                "fresh Base trajectory gate was not verified"
            )
        if [entry["epoch"] for entry in manifest["entries"]] != list(
            target_candidate_epochs
        ):
            raise AdaptationContractError("candidate set is incomplete")
        manifest["status"] = "complete"
        manifest["completed_epochs"] = target_total_epochs
        manifest["optimizer_updates"] = optimizer_updates
        manifest["entries_sha256"] = canonical_json_sha256(manifest["entries"])
        _atomic_json(run_dir / manifest_name, manifest)
        status_payload: dict[str, Any] = {
            "format": (
                SHORT_QUALITY_STATUS_FORMAT
                if provisional
                else STATUS_FORMAT
            ),
            "status": "complete",
            "completed_epochs": target_total_epochs,
            "optimizer_updates": optimizer_updates,
            "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
            "candidate_manifest_sha256": sha256_file(
                run_dir / manifest_name
            ),
            "frozen_receipt_sha256": frozen_receipt["receipt_sha256"],
            "throughput_gate": dict(throughput_receipt),
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "all_training_state_finite": True,
            "epoch_metrics_jsonl": str(
                (run_dir / "epoch_metrics.jsonl").resolve(strict=True)
            ),
            "epoch_metrics_sha256": sha256_file(
                run_dir / "epoch_metrics.jsonl"
            ),
            "epoch_metrics_records": target_total_epochs,
            "schedule_sha256": contract_receipts["schedule"]["sha256"],
            "trajectory_anchor_sha256": contract_receipts[
                "trajectory_anchor"
            ]["sha256"],
            "trajectory_mode": trajectory_mode,
            "trajectory_probe_verified": manifest[
                "trajectory_probe_verified"
            ],
            "trajectory_probe": manifest["trajectory_probe"],
            "started_unix": started_unix,
            "completed_unix": time.time(),
        }
        if provisional:
            status_payload["run_purpose"] = run_purpose
            status_payload["target_epochs"] = list(
                target_candidate_epochs
            )
        if not provisional:
            status_payload["resume_receipt"] = str(
                (
                    run_dir / "resume" / "latest_resume.json"
                ).resolve(strict=True)
            )
            status_payload["resume_receipt_sha256"] = sha256_file(
                run_dir / "resume" / "latest_resume.json"
            )
        else:
            final_manifest_path = (run_dir / manifest_name).resolve(
                strict=True
            )
            metrics_path = (run_dir / "epoch_metrics.jsonl").resolve(
                strict=True
            )
            ready_receipts: list[dict[str, Any]] = []
            for epoch in SHORT_QUALITY_EPOCHS:
                path = (
                    run_dir
                    / "short_quality_candidate_receipts"
                    / f"epoch-{epoch:04d}.json"
                ).resolve(strict=True)
                ready_payload = _strict_json_bytes(
                    path.read_bytes(),
                    f"short-quality e{epoch} candidate receipt",
                )
                ready_receipts.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "receipt_payload_sha256": ready_payload[
                            "receipt_payload_sha256"
                        ],
                    }
                )
            status_payload["candidate_manifest"] = {
                "path": str(final_manifest_path),
                "sha256": sha256_file(final_manifest_path),
                "bytes": final_manifest_path.stat().st_size,
            }
            status_payload["epoch_metrics"] = {
                "path": str(metrics_path),
                "sha256": sha256_file(metrics_path),
                "bytes": metrics_path.stat().st_size,
                "records": SHORT_QUALITY_TOTAL_EPOCHS,
            }
            status_payload["candidate_ready_receipts"] = ready_receipts
            status_payload["receipt_payload_sha256"] = (
                canonical_json_sha256(status_payload)
            )
        _atomic_json(run_dir / status_name, status_payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    topology_gate_spec_receipt = validate_topology_gate_spec(args)
    _prepare_deterministic_environment(seed=args.seed)
    rank, local_rank, world_size = _distributed_context(args)

    import torch
    import torch.distributed as dist
    from models.semtalk import semtalk_base

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < LOCAL_WORLD_SIZE
    ):
        raise AdaptationContractError(
            "each formal Base node exposes fewer CUDA GPUs than the selected "
            "local world size"
        )
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise AdaptationContractError("bf16 is not supported by this CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://")
    _configure_deterministic_runtime(torch, seed=args.seed)
    topology_receipt = _distributed_topology_receipt(
        args,
        rank=rank,
        local_rank=local_rank,
    )

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

        # Each physical node independently replays source/dataset/five-stage
        # closure on its local rank zero.  Content/lineage semantics must be
        # identical.  Device/inode identities are node-local and are frozen
        # as an ordered set instead of compared across pods.
        local_node_receipt: Any = None
        if local_rank == 0:
            try:
                current_source = source_receipt()
                current_dataset = validate_dataset_receipts(args)
                local_node_receipt = {
                    "status": "complete",
                    "source": current_source,
                    "dataset": current_dataset,
                    "long_contract": validate_long_contract_receipts(
                        args,
                        dataset_receipt=current_dataset,
                    ),
                }
            except BaseException as error:
                local_node_receipt = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        gathered_node_receipts: list[Any] = [
            None for _ in range(WORLD_SIZE)
        ]
        dist.all_gather_object(gathered_node_receipts, local_node_receipt)
        node_receipts = [
            gathered_node_receipts[node_rank * LOCAL_WORLD_SIZE]
            for node_rank in range(NODE_COUNT)
        ]
        if any(
            not isinstance(receipt, dict)
            or receipt.get("status") != "complete"
            for receipt in node_receipts
        ):
            failure = (
                next(
                    (
                        receipt
                        for receipt in node_receipts
                        if not isinstance(receipt, dict)
                        or receipt.get("status") != "complete"
                    ),
                    {
                        "error_type": "NodeReceiptFailure",
                        "error": "a formal node preflight failed",
                    },
                )
                if node_receipts
                else {}
            )
            raise AdaptationContractError(
                "dual-node source/dataset/five-stage preflight failed: "
                f"{failure.get('error_type')}: "
                f"{failure.get('error')}"
            )
        source_semantics = [
            _portable_source_receipt(receipt["source"])
            for receipt in node_receipts
        ]
        contract_semantics = [
            receipt["long_contract"] for receipt in node_receipts
        ]
        if (
            any(value != source_semantics[0] for value in source_semantics[1:])
            or any(
                value != contract_semantics[0]
                for value in contract_semantics[1:]
            )
        ):
            raise AdaptationContractError(
                "formal nodes disagree on Git or training-contract semantics"
            )
        local_dataset_receipt = node_receipts[
            args.formal_node_rank
        ]["dataset"]
        host_slots = _active_host_slots(args)
        dataset_receipt = _global_dataset_receipt(
            node_receipts, host_slots=host_slots
        )
        current_source = {
            **source_semantics[0],
            "node_local_clones": [
                {
                    "node_rank": node_rank,
                    "host_slot": host_slots[node_rank],
                    "hostname": FORMAL_HOST_BY_SLOT[host_slots[node_rank]],
                    "entrypoint": receipt["source"]["entrypoint"],
                    "branch": receipt["source"]["branch"],
                }
                for node_rank, receipt in enumerate(node_receipts)
            ],
        }
        contract_receipts = contract_semantics[0]
        official_state, local_official_receipt = read_official_base_checkpoint(
            Path(args.official_base_checkpoint),
            torch_module=torch,
        )
        gathered_official_receipts: list[Any] = [
            None for _ in range(WORLD_SIZE)
        ]
        dist.all_gather_object(
            gathered_official_receipts, local_official_receipt
        )
        official_receipt = _global_official_base_receipt(
            gathered_official_receipts,
            host_slots=host_slots,
        )
        model = semtalk_base(_model_args())
        speaker_initialization = strict_load_and_initialize_show_speakers(
            model,
            official_state,
            torch_module=torch,
        )
        del official_state
        protocol = protocol_receipt(
            args,
            contract_receipts=contract_receipts,
            topology_gate_spec=topology_gate_spec_receipt,
        )
        frozen_receipt = _frozen_receipt(
            source=current_source,
            official_base=official_receipt,
            speaker_initialization=speaker_initialization,
            dataset=dataset_receipt,
            protocol=protocol,
            long_contract=contract_receipts,
            topology=topology_receipt,
            run_purpose=_run_purpose(args),
            target_epochs=_target_epochs(args),
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
        if args.mode in {SHORT_QUALITY_MODE, "train"}:
            throughput_receipt = validate_throughput_gate(
                args, frozen_receipt=frozen_receipt
            )
            if args.mode == "train":
                throughput_receipt["topology_selection"] = (
                    validate_topology_selection(
                        args,
                        throughput_gate=throughput_receipt,
                    )
                )
        loader, sampler = _create_dataloader(
            args,
            rank,
            world_size,
            local_dataset_receipt,
        )
        model = model.to(device)
        # Preserve the published W1 BatchNorm semantics for the reference
        # probe.  DDP adaptations intentionally retain per-rank BatchNorm;
        # they are therefore objective-preserving adaptations, never claimed
        # to be trajectory-equivalent to W1.
        if WORLD_SIZE > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        _configure_rank_training_rng(
            torch,
            seed=args.seed,
            rank=rank,
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
                contract_receipts=contract_receipts,
                throughput_receipt=throughput_receipt,
            )
        dist.barrier()
        return 0
    except BaseException as error:
        if rank == 0 and run_dir.is_dir():
            failure_payload = {
                "format": (
                    SHORT_QUALITY_STATUS_FORMAT
                    if args.mode == SHORT_QUALITY_MODE
                    else STATUS_FORMAT
                ),
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_unix": time.time(),
            }
            if args.mode == SHORT_QUALITY_MODE:
                failure_payload["run_purpose"] = _run_purpose(args)
                failure_payload["target_epochs"] = _target_epochs(args)
            _atomic_json(run_dir / "failure.json", failure_payload)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
