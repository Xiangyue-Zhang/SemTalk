#!/usr/bin/env python3
"""Run frozen SemTalk Base-only inference on the canonical SHOW test split.

This entry point deliberately does not instantiate a trainer.  It loads exactly
the formal Base checkpoint, the four formal RVQ checkpoints, and the formal
global/root VAE checkpoint.  SemGate, Sparse, Speaker2 remapping, CLIP,
emotion, semantic, ASR, TextGrid, and vocabulary components are forbidden.
The default checkpoint source remains the fully audited SHOW-trained contract.
An explicit ``released_all_speakers_v1`` mode instead accepts only the exact
hash-pinned official BEAT2 All-Speakers release files, forbids SHOW-training
status receipts for those files, and labels them as not SHOW-trained.  Base and
the five representation prerequisites have independent source selections so a
future SHOW-trained Base can consume the frozen official representations while
the official All-Speakers Base remains an independently auditable option.

The autoregressive contract is the released SemTalk Base ``_g_test`` contract,
with its future-ground-truth dependency removed:

* fixed 64-frame windows, four seed frames, and a 60-frame stride;
* the first window receives only the first four GT pose/translation frames;
* every later window receives only the previous generated last four frames;
* all remaining motion payload values are zero before the learned mask is
  applied; and
* RVQ indices are stitched (16 tokens, then 15 tokens per later window) before
  one whole-stream decode.

Every invocation owns one deterministic modulo shard.  After all eight shards
complete, ``--finalize`` verifies SHA/lineage/runtime closure, proves exact-once
coverage, and materializes DiffSHEG canonical local-axis-angle ``res_*.npz`` /
``gt_*.npz`` pairs under ``<output-root>/final/npz/test``.  Ground truth is generated
deterministically from the frozen canonical SHOW test cache and never from the
incompatible GlobalDiff global-axis-angle export.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import stat
import subprocess
import sys
from types import MethodType, SimpleNamespace
from typing import Any
import uuid
import zipfile

import numpy as np

from scripts.show_base import prerequisite_val_contract as prerequisite_contract


# Formal source cleanliness is rechecked at completion; local model imports must
# therefore never create untracked bytecode inside the immutable source tree.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_TEST_CLIPS = 1708
EXPECTED_NUM_SHARDS = 8
DIFFSHEG_WINDOW = 88
FINAL_BUNDLE_NAME = "final"
POSE_FPS = 30
POSE_DIM = 165
EXPRESSION_DIM = 100
BETA_DIM = 300
WINDOW = 64
PRE_FRAMES = 4
STRIDE = WINDOW - PRE_FRAMES
CODEBOOK_SIZE = 256
RVQ_LEVELS = 6
RVQ_TOKEN_FRAMES = 4
RVQ_TOKENS_PER_WINDOW = WINDOW // RVQ_TOKEN_FRAMES
CANONICAL_FIELDS = (
    "pose",
    "contact",
    "facial",
    "beta",
    "trans",
    "speaker_id",
)
AUDIO_FIELDS = ("beat", "hubert")
AUDIO_ALIGNMENT_PROTOCOL = {
    "feature_extraction": "full_source_waveform_before_alignment",
    "native_30fps_alignment": "linear_align_corners_true",
    "canonical_alignment": "leading_prefix",
    "long_audio": "discard_source_feature_tail_after_canonical_frames",
    "short_audio": (
        "edge_pad_only_within_one_frame_and_without_losing_a_whole_second"
    ),
    "max_shortfall_frames": 1,
    "reference": "public_loader_shortest_whole_second_leading_prefix",
}
OUTPUT_FIELDS = (
    "betas",
    "poses",
    "expressions",
    "trans",
    "model",
    "gender",
    "mocap_frame_rate",
)
RVQ_DIMS = {
    "face": 106,
    "upper": 78,
    "hands": 180,
    "lower": 61,
}
CHECKPOINT_STAGES = ("base", "face", "upper", "hands", "lower", "global")
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SHOW_TRAINED_CHECKPOINT_SOURCE = "show_trained_v1"
RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE = "released_all_speakers_v1"
OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE = "official_show_adapt_v1"
CHECKPOINT_SOURCES = (
    SHOW_TRAINED_CHECKPOINT_SOURCE,
    RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
    OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
)
OFFICIAL_SHOW_ADAPT_MODE = "official_show_adapt_v1"
OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT = (
    "semtalk_show_base_official_adapt_checkpoint_v1"
)
OFFICIAL_SHOW_ADAPT_BASE_MANIFEST_FORMAT = (
    "semtalk_show_base_official_adapt_manifest_v1"
)
OFFICIAL_SHOW_ADAPT_BASE_STATUS_FORMAT = (
    "semtalk_show_base_official_adapt_status_v1"
)
OFFICIAL_SHOW_ADAPT_BASE_FROZEN_FORMAT = (
    "semtalk_show_base_official_adapt_frozen_inputs_v1"
)
OFFICIAL_SHOW_ADAPT_TRANSFER_FORMAT = "semtalk_show_official_transfer_v1"
OFFICIAL_SHOW_ADAPT_BASE_CANDIDATE_EPOCHS = (1, 2, 4, 8, 16, 32, 40)
OFFICIAL_SHOW_ADAPT_BASE_UPDATES_PER_EPOCH = 248
INFERENCE_AUXILIARY_LOSS_BYPASS_FORMAT = (
    "semtalk_inference_auxiliary_loss_bypass_v1"
)
INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS = (
    "hubert_face_cons_loss",
    "beat_cons_loss",
)
PINNED_SEMTALK_MODEL_SOURCE = {
    "relative_path": "models/semtalk.py",
    "sha256": (
        "ddcc622c9778413b73c2b51b90354cd82dcedce62fea25006f8219de17f7ba2e"
    ),
    "git_blob_sha1": "6e786a5f5d29c070145c49f2a277cc7667c78ace",
}
PINNED_SEMTALK_BASE_FORWARD = {
    "source_sha256": (
        "032ee956dde6297cb9e3211c713f9eb1183b4937663f42a76586f1a42c0dbbcb"
    ),
    "signature": (
        "(self, in_audio=None, in_word=None, mask=None, is_train=False, "
        "in_motion=None, use_attentions=True, use_word=True, in_id=None, "
        "hubert=None)"
    ),
}
PINNED_RHYTHMIC_LOSS_FORWARD = {
    "source_sha256": (
        "eae08dca9b9f7605dc4ff35576c9630380a365e4ab2b61429755aa490fef659e"
    ),
    "signature": "(self, facial_features, audio_features)",
}
PINNED_RVQ_INDICES_SOURCE_SHA256 = (
    "467e0115387738b2ccc9aa46a0174d4a7b93aa1199d8e8bae9070e90f1150ffc"
)
PINNED_INFER_CLIP = {
    "source_sha256": (
        "40e4b6982c85ded6f6fef2dafb2f9f5f4bc876150ecf40fac59de9c7c6e8242c"
    ),
    "signature": (
        "(*, pose: 'np.ndarray', trans: 'np.ndarray', beat: 'np.ndarray', "
        "hubert: 'np.ndarray', speaker_id: 'int', "
        "models: 'Mapping[str, Any]', masks: 'Mapping[str, Any]', "
        "device: 'str') -> 'dict[str, np.ndarray]'"
    ),
}
RELEASED_ALL_SPEAKERS_CLASSIFICATION = (
    "official_BEAT2_All-Speakers_released_weights_not_SHOW-trained"
)
RELEASED_ALL_SPEAKERS_BASE_CLASSIFICATION = (
    "official_BEAT2_All-Speakers_released_Base_not_SHOW-trained"
)
RELEASED_ALL_SPEAKERS_RELEASE_TRUST_ROOT = {
    "origin": EXPECTED_ORIGIN,
    "commit": "806b008c97bf51fce203e54109e4c22325253618",
    "tree": "029deb438330fcaa36377195bad79ffe0f06335c",
    "archive_sha256": (
        "3cbe7a3a923075ad39bcdd41bb88e828fd6161be4c5299c4"
        "e62eedcf20ea5664"
    ),
    "readme_sha256": (
        "27f846e150e8101c1124c3a8bfd026617508554f59ee359c5"
        "5eae825b2edeb1e"
    ),
    "sha256s_sha256": (
        "f7c08cb621f884c7deb0c0cba757fcd08c8b1ad471bab939e"
        "b0d1bd02d4abc1b"
    ),
    "best_run_sha256": (
        "6edaae9f21b7a7164f7457ca93240989fcfa3cb8ea02aa94f"
        "7602c17f9491c9a"
    ),
    "all_speaker_metrics_sha256": (
        "3ecd9586b6eb34eb4a05cdd57f29ed51de50399e02ff476d"
        "f8c4d4c7fb4cc317"
    ),
}
FULLY_RELEASED_ZERO_SHOT_MODE = "fully_released_zero_shot_v1"
RELEASED_CROSS_DOMAIN_GATE_FORMAT = (
    "semtalk_released_all_speakers_show_cross_domain_gate_v1"
)
RELEASED_CROSS_DOMAIN_GATE_SCRIPT_RELATIVE = (
    "scripts/show_base/gate_released_all_speakers_on_show.py"
)
RELEASED_CROSS_DOMAIN_GATE_KEYS = {
    "format",
    "status",
    "authorization",
    "release_trust_root",
    "official_weights",
    "canonical_receipt",
    "source_roles",
    "protocol",
    "gate_script",
    "measurement_receipt",
    "threshold_receipt",
    "measurements",
    "thresholds",
    "decisions",
    "receipt_sha256",
}
RELEASED_CROSS_DOMAIN_ARTIFACT_KEYS = {"path", "sha256"}
SOURCE_RECEIPT_KEYS = {
    "source_root",
    "origin",
    "commit",
    "tree",
    "clean",
    "script",
    "script_relative",
    "script_sha256",
}
RELEASED_ALL_SPEAKERS_MODELS = {
    "base": {
        "filename": "best_semtalk_base.bin",
        "sha256": (
            "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc"
            "1d89603"
        ),
        "model_class": "semtalk_base",
    },
    "face": {
        "filename": "rvq_face_600.bin",
        "sha256": (
            "31b04c88456a25f4d57841c0cb507b4c856daccb3875878d06545110"
            "a6152127"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 106,
        "vae_layer": 2,
    },
    "hands": {
        "filename": "rvq_hands_500.bin",
        "sha256": (
            "08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a5"
            "39b03e436"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 180,
        "vae_layer": 2,
    },
    "upper": {
        "filename": "rvq_upper_500.bin",
        "sha256": (
            "05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d8"
            "8d08ac17"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 78,
        "vae_layer": 2,
    },
    "lower": {
        "filename": "rvq_lower_600.bin",
        "sha256": (
            "2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c17"
            "1ae4efce8"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 61,
        "vae_layer": 4,
    },
    "global": {
        "filename": "last_1700_foot.bin",
        "sha256": (
            "6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e127e19e"
            "0144ee12"
        ),
        "model_class": "VAEConvZero",
        "vae_test_dim": 61,
        "vae_layer": 4,
    },
}
RELEASED_PREREQUISITE_RECORD_KEYS = {
    "path",
    "filename",
    "sha256",
    "bytes",
    "formal_stage",
    "prerequisite_source",
    "classification",
    "training_dataset",
    "speaker_scope",
    "show_trained",
    "checkpoint_container_schema",
    "model_class",
    "model_state_tensors",
    "model_state_schema_sha256",
    "all_model_state_tensors_finite",
    "strict_state_dict_load",
    "frozen_eval",
    "source_receipt",
    "source_receipt_sha256",
}
RELEASED_IMPORT_RECEIPT_KEYS = {
    "format",
    "status",
    "prerequisite_source",
    "classification",
    "training_dataset",
    "speaker_scope",
    "show_trained",
    "source_receipt",
    "source_receipt_sha256",
    "files",
    "strict_state_dict_load",
    "all_model_state_tensors_finite",
    "frozen_eval",
    "receipt_sha256",
}
RELEASED_BASE_CONTAINER_SCHEMA = [
    "epoch",
    "lrs",
    "model_state",
    "opt_state",
]
RELEASED_BASE_LRS = {
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
FORMAL_SMPLX_FILENAME = "SMPLX_NEUTRAL_2020.npz"
FORMAL_SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
MODEL_V2_AUDIT_KEYS = {
    "format",
    "formal_stage",
    "config_sha256",
    "lineage_manifest_sha256",
    "dataset_summary_sha256",
    "data_mdb_sha256",
    "dataset_receipt_sha256",
    "smplx_asset_receipt",
    "source_receipt",
    "source_receipt_sha256",
    "optimizer_updates",
    "smplx_training_pool_mode",
    "distributed_training_receipt",
    "optimizer_runtime_receipt",
    "rvq_rank_state_receipt",
    "initialization_receipt",
    "rvq_ema_prior_receipt",
    "latest_representation_candidate",
    "smplx_training_pool_runtime_evidence",
    "base_candidate_manifest",
}
MODEL_V2_OPTIONAL_AUDIT_KEYS = {
    "continuation_wave_receipt",
    "smplx_training_pool_gate",
    "lower_target_joints_cache",
}
LOWER_TARGET_CACHE_RECEIPT_KEY = "lower_target_joints_cache"
LOWER_TARGET_CACHE_RECEIPT_KEYS = {
    "format",
    "cache_version",
    "cache_path",
    "manifest_path",
    "manifest_sha256",
    "checker_receipt_path",
    "checker_receipt_sha256",
    "data_mdb_sha256",
    "lock_mdb_sha256",
    "entry_aggregate_sha256",
    "entries",
    "entry_shape",
    "dtype",
    "speaker_scope",
    "speaker_ids",
    "source_receipt",
    "current_inputs",
    "exact_once",
    "finite",
    "torch_equal_checked",
    "target_requires_grad",
    "target_optimizer_excluded",
    "formal_gate",
    "receipt_sha256",
}
LOWER_TARGET_CACHE_FORMAL_GATE_KEYS = {
    "format",
    "status",
    "path",
    "sha256",
    "pooled_wall_speedup",
    "pooled_cuda_event_speedup",
    "builder_amortized_wall_speedup",
    "builder_process_receipt_path",
    "builder_process_receipt_sha256",
    "source_binding",
    "cache_manifest_sha256",
    "cache_checker_sha256",
}
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
BASE_CANDIDATE_AUDIT_KEYS = {
    "format",
    "formal_stage",
    "smplx_training_pool_mode",
    "candidate_epoch",
    "optimizer_updates",
    "config_sha256",
    "lineage_manifest_sha256",
    "dataset_receipt_sha256",
    "smplx_asset_receipt",
    "source_receipt",
    "source_receipt_sha256",
}
BASE_CANDIDATE_MANIFEST_KEYS = {
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
BASE_CANDIDATE_RECORD_KEYS = {
    "epoch",
    "optimizer_updates",
    "checkpoint",
    "checkpoint_sha256",
    "model_audit_sha256",
    "transaction_core",
    "transaction_core_sha256",
}
BASE_CANDIDATE_CORE_KEYS = {
    "format",
    "formal_stage",
    "candidate_epoch",
    "optimizer_updates",
    "checkpoint",
    "staging_checkpoint",
    "previous_manifest_sha256",
    "previous_entry_count",
    "previous_entries_sha256",
    "config_sha256",
    "lineage_manifest_sha256",
    "dataset_receipt_sha256",
    "smplx_asset_receipt",
    "source_receipt",
    "source_receipt_sha256",
    "model_audit_sha256",
}
BASE_CANDIDATE_INTERVAL_EPOCHS = 10
BASE_CANDIDATE_COUNT = 40
BASE_EPOCHS = 400
FORMAL_UPDATES_PER_EPOCH = 1_988
SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
FORBIDDEN_COMPONENTS = {
    "ASR",
    "TextGrid",
    "vocabulary",
    "CLIP",
    "emotion",
    "semantic",
    "SemGate",
    "Sparse",
}
CANONICAL_SOURCE_AUDIO_RATE = 22_000
CANONICAL_SOURCE_AUDIO_SAMPLE_WIDTH = 2
CANONICAL_HUBERT_TARGET_RATE = 16_000
CANONICAL_WAV_MONO_POLICY = (
    "librosa.load(sr=None,mono=True):arithmetic_channel_mean"
)
CANONICAL_AUDIO_CHANNEL_PROTOCOL = {
    "accepted_source_channels": [1, 2],
    "source_sample_width_bytes": CANONICAL_SOURCE_AUDIO_SAMPLE_WIDTH,
    "source_compression": "NONE",
    "decode": "librosa.load(BytesIO(wav_payload),sr=None,mono=True)",
    "multichannel_mix": "arithmetic_mean_across_channels",
    "manifest_policy": CANONICAL_WAV_MONO_POLICY,
}


class InferenceContractError(RuntimeError):
    """Raised when frozen inputs or generated outputs violate the contract."""


_STABLE_FILE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(metadata, field)) for field in _STABLE_FILE_FIELDS
    )


def _safe_file_snapshot(
    path: str | Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, bytes, str, os.stat_result]:
    """Read and hash one stable regular file through one descriptor."""
    expected = (
        _require_sha256(expected_sha256, f"{label} expected SHA")
        if expected_sha256 is not None
        else None
    )
    resolved = _resolved_regular_file(path, label)
    try:
        path_before = os.lstat(resolved)
    except OSError as exc:
        raise InferenceContractError(
            f"cannot safely inspect {label}: {resolved}"
        ) from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
        path_before.st_mode
    ):
        raise InferenceContractError(
            f"{label} must be a regular non-symlink file: {resolved}"
        )
    parts = resolved.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise InferenceContractError(
            f"{label} must be below the filesystem root: {resolved}"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    file_flags = os.O_RDONLY | nofollow
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(os.sep, directory_flags)
        for component in parts[1:-1]:
            next_fd = os.open(
                component,
                directory_flags,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or _file_identity(path_before) != _file_identity(before)
        ):
            raise InferenceContractError(
                f"{label} changed before it was opened: {resolved}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if _file_identity(before) != _file_identity(after):
            raise InferenceContractError(
                f"{label} changed while it was read: {resolved}"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise InferenceContractError(
                f"{label} size changed while it was read: {resolved}"
            )
        leaf_after = os.stat(
            parts[-1],
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        path_after = os.lstat(resolved)
        if (
            _file_identity(after) != _file_identity(leaf_after)
            or _file_identity(after) != _file_identity(path_after)
        ):
            raise InferenceContractError(
                f"{label} path changed while it was read: {resolved}"
            )
        observed = hashlib.sha256(payload).hexdigest()
        if expected is not None and observed != expected:
            raise InferenceContractError(
                f"{label} SHA mismatch: {observed} != {expected} "
                f"({resolved})"
            )
        return resolved, payload, observed, after
    except InferenceContractError:
        raise
    except OSError as exc:
        raise InferenceContractError(
            f"cannot safely read {label}: {resolved}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def sha256_file(path: str | Path) -> str:
    _resolved, _payload, observed, _metadata = _safe_file_snapshot(
        path,
        "file",
    )
    return observed


def _require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _require_git_oid(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if (
        len(normalized) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a 40- or 64-character Git object ID")
    return normalized


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise InferenceContractError(f"{label} must be an exact integer, got {value!r}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compact_json_sha256(value: Any) -> str:
    """Match show_base_train._payload_sha256 for checkpoint source receipts."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def official_base_adapt_json_sha256(value: Any) -> str:
    """Match train_base_official_adapt.canonical_json_sha256 exactly."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def official_transfer_state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Match show_official_transfer.state_dict_sha256 exactly."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not hasattr(value, "detach"):
            raise InferenceContractError(
                f"official transfer state entry {key!r} is not a tensor"
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_lower_target_cache_binding(
    *,
    formal_stage: str,
    audit: Mapping[str, Any],
    dataset_receipt: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    path: Path,
) -> None:
    """Require the frozen lower-target receipt on lower and forbid it elsewhere."""

    key = LOWER_TARGET_CACHE_RECEIPT_KEY
    if formal_stage != "lower":
        if (
            key in audit
            or key in dataset_receipt
            or (status is not None and key in status)
        ):
            raise InferenceContractError(
                f"{path}: non-lower stage carries a lower target cache receipt"
            )
        return

    audit_receipt = audit.get(key)
    dataset_cache_receipt = dataset_receipt.get(key)
    status_receipt = status.get(key) if status is not None else audit_receipt
    if (
        type(audit_receipt) is not dict
        or audit_receipt != dataset_cache_receipt
        or audit_receipt != status_receipt
        or set(audit_receipt) != LOWER_TARGET_CACHE_RECEIPT_KEYS
    ):
        raise InferenceContractError(
            f"{path}: lower target cache receipt is missing or inconsistent"
        )
    receipt_without_sha = dict(audit_receipt)
    receipt_sha = receipt_without_sha.pop("receipt_sha256", None)
    current_inputs = audit_receipt.get("current_inputs")
    source = audit_receipt.get("source_receipt")
    audit_source = audit.get("source_receipt")
    smplx_asset = dataset_receipt.get("smplx_asset")
    formal_gate = audit_receipt.get("formal_gate")
    digest_fields = (
        "manifest_sha256",
        "checker_receipt_sha256",
        "data_mdb_sha256",
        "lock_mdb_sha256",
        "entry_aggregate_sha256",
    )
    if (
        audit_receipt.get("format")
        != "semtalk_show_lower_target_joints_raw_lmdb_v1"
        or type(audit_receipt.get("cache_version")) is not int
        or audit_receipt.get("cache_version") != 1
        or type(audit_receipt.get("entries")) is not int
        or audit_receipt.get("entries") != 127_286
        or audit_receipt.get("entry_shape") != [64, 127, 3]
        or audit_receipt.get("dtype") != "<f4"
        or audit_receipt.get("speaker_scope") != "All"
        or audit_receipt.get("speaker_ids") != [0, 1, 2, 3]
        or audit_receipt.get("exact_once") is not True
        or audit_receipt.get("finite") is not True
        or audit_receipt.get("torch_equal_checked") is not True
        or audit_receipt.get("target_requires_grad") is not False
        or audit_receipt.get("target_optimizer_excluded") is not True
        or type(formal_gate) is not dict
        or set(formal_gate) != LOWER_TARGET_CACHE_FORMAL_GATE_KEYS
        or formal_gate.get("format")
        != "semtalk_show_lower_target_cache_formal_gate_v1"
        or formal_gate.get("status") != "pass"
        or type(formal_gate.get("path")) is not str
        or not formal_gate["path"]
        or type(formal_gate.get("builder_process_receipt_path")) is not str
        or not formal_gate["builder_process_receipt_path"]
        or formal_gate.get("cache_manifest_sha256")
        != audit_receipt.get("manifest_sha256")
        or formal_gate.get("cache_checker_sha256")
        != audit_receipt.get("checker_receipt_sha256")
        or formal_gate.get("source_binding") != source
        or any(
            type(formal_gate.get(field)) is not str
            or len(formal_gate[field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in formal_gate[field]
            )
            for field in ("sha256", "builder_process_receipt_sha256")
        )
        or any(
            type(formal_gate.get(field)) not in (int, float)
            or type(formal_gate[field]) is bool
            or not math.isfinite(float(formal_gate[field]))
            or float(formal_gate[field]) < 1.05
            for field in (
                "pooled_wall_speedup",
                "pooled_cuda_event_speedup",
                "builder_amortized_wall_speedup",
            )
        )
        or any(
            type(audit_receipt.get(field)) is not str
            or len(audit_receipt[field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in audit_receipt[field]
            )
            for field in digest_fields
        )
        or type(receipt_sha) is not str
        or receipt_sha != canonical_json_sha256(receipt_without_sha)
        or type(source) is not dict
        or set(source) != {"origin", "commit", "tree"}
        or source.get("origin") != EXPECTED_ORIGIN
        or any(
            type(source.get(field)) is not str
            or len(source[field]) != 40
            or any(
                character not in "0123456789abcdef"
                for character in source[field]
            )
            for field in ("commit", "tree")
        )
        or type(audit_source) is not dict
        or {
            name: source.get(name)
            for name in ("origin", "commit", "tree")
        }
        != {
            name: audit_source.get(name)
            for name in ("origin", "commit", "tree")
        }
        or type(current_inputs) is not dict
        or current_inputs.get("representation_summary_sha256")
        != dataset_receipt.get("summary_sha256")
        or current_inputs.get("representation_lineage_sha256")
        != dataset_receipt.get("lineage_sha256")
        or current_inputs.get("data_mdb_sha256")
        != dataset_receipt.get("data_mdb_sha256")
        or type(smplx_asset) is not dict
        or current_inputs.get("smplx_asset_sha256")
        != smplx_asset.get("sha256")
    ):
        raise InferenceContractError(
            f"{path}: invalid lower target cache receipt contract"
        )


def _validate_lower_target_backend_binding(
    *,
    formal_stage: str,
    audit: Mapping[str, Any],
    dataset_receipt: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    path: Path,
) -> None:
    """Require formal lower's live SMPL-X receipt and reject cache/live mixing."""

    backend_key = LOWER_TARGET_BACKEND_RECEIPT_KEY
    cache_key = LOWER_TARGET_CACHE_RECEIPT_KEY
    payloads = (
        (audit, dataset_receipt)
        if status is None
        else (audit, dataset_receipt, status)
    )
    if formal_stage != "lower":
        if any(
            key in payload
            for key in (backend_key, cache_key)
            for payload in payloads
        ):
            raise InferenceContractError(
                f"{path}: non-lower stage carries a lower target receipt"
            )
        return

    if any(cache_key in payload for payload in payloads):
        raise InferenceContractError(
            f"{path}: formal lower cannot mix cache and live backend receipts"
        )
    audit_receipt = audit.get(backend_key)
    dataset_backend_receipt = dataset_receipt.get(backend_key)
    status_receipt = (
        status.get(backend_key) if status is not None else audit_receipt
    )
    if (
        type(audit_receipt) is not dict
        or audit_receipt != dataset_backend_receipt
        or audit_receipt != status_receipt
        or set(audit_receipt) != LOWER_TARGET_BACKEND_RECEIPT_KEYS
    ):
        raise InferenceContractError(
            f"{path}: lower live backend receipt is missing or inconsistent"
        )

    receipt_without_sha = dict(audit_receipt)
    receipt_sha = receipt_without_sha.pop("receipt_sha256", None)
    source_binding = audit_receipt.get("source_binding")
    audit_source = audit.get("source_receipt")
    smplx_asset = dataset_receipt.get("smplx_asset")
    target_forward_source = (
        Path(__file__).resolve().parents[2] / "utils" / "smplx_training.py"
    )
    if (
        audit_receipt.get("format") != LOWER_TARGET_BACKEND_FORMAT
        or audit_receipt.get("backend") != "live_smplx"
        or audit_receipt.get("formal_stage") != "lower"
        or audit_receipt.get("cache_enabled") is not False
        or audit_receipt.get("target_forward")
        != "utils.smplx_training.smplx_target_forward"
        or type(audit_receipt.get("target_forward_sha256")) is not str
        or len(audit_receipt["target_forward_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in audit_receipt["target_forward_sha256"]
        )
        or target_forward_source.is_symlink()
        or not target_forward_source.is_file()
        or audit_receipt.get("target_forward_sha256")
        != sha256_file(target_forward_source)
        or audit_receipt.get("target_batch_contract")
        != "current_mixed_dataloader_batch"
        or audit_receipt.get("torch_no_grad") is not True
        or audit_receipt.get("return_shaped") is not False
        or type(source_binding) is not dict
        or set(source_binding) != {"origin", "commit", "tree"}
        or type(audit_source) is not dict
        or source_binding
        != {
            key: audit_source.get(key)
            for key in ("origin", "commit", "tree")
        }
        or source_binding.get("origin") != EXPECTED_ORIGIN
        or any(
            type(source_binding.get(field)) is not str
            or len(source_binding[field]) != 40
            or any(
                character not in "0123456789abcdef"
                for character in source_binding[field]
            )
            for field in ("commit", "tree")
        )
        or type(smplx_asset) is not dict
        or smplx_asset.get("format") != "semtalk_show_smplx_asset_v1"
        or smplx_asset.get("filename") != FORMAL_SMPLX_FILENAME
        or smplx_asset.get("sha256") != FORMAL_SMPLX_SHA256
        or audit_receipt.get("smplx_asset_sha256")
        != FORMAL_SMPLX_SHA256
        or type(receipt_sha) is not str
        or receipt_sha != compact_json_sha256(receipt_without_sha)
    ):
        raise InferenceContractError(
            f"{path}: invalid lower live SMPL-X backend receipt contract"
        )


def training_json_document_sha256(value: Any) -> str:
    """Match show_base_train._json_document_sha256 exactly."""
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InferenceContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                InferenceContractError(
                    f"{source}: non-finite JSON constant {token!r}"
                )
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceContractError(f"cannot parse JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise InferenceContractError(f"{source}: expected a JSON object")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_object_without_duplicates,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            InferenceContractError(
                                f"{source}:{line_number}: non-finite JSON "
                                f"constant {token!r}"
                            )
                        ),
                    )
                except json.JSONDecodeError as exc:
                    raise InferenceContractError(
                        f"cannot parse {source}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise InferenceContractError(
                        f"{source}:{line_number}: expected a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError) as exc:
        raise InferenceContractError(f"cannot read JSONL {source}: {exc}") from exc
    return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_new(path: str | Path, payload: bytes) -> None:
    """Atomically create *path* and refuse to replace any existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        if os.path.lexists(destination):
            raise FileExistsError(
                f"refusing to overwrite existing file: {destination}"
            )
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(destination):
            raise FileExistsError(
                f"refusing to overwrite existing file: {destination}"
            )
        os.rename(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_new(path: str | Path, value: Any) -> None:
    atomic_write_new(path, canonical_json_bytes(value))


def atomic_jsonl_new(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    atomic_write_new(path, payload)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    canonical_array = np.asarray(array)
    if canonical_array.ndim:
        canonical_array = np.ascontiguousarray(canonical_array)
    np.lib.format.write_array(
        buffer,
        canonical_array,
        allow_pickle=False,
    )
    return buffer.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != OUTPUT_FIELDS:
        raise ValueError(
            f"output NPZ fields must be exactly {OUTPUT_FIELDS}, "
            f"observed {tuple(arrays)}"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in OUTPUT_FIELDS:
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    return buffer.getvalue()


def _write_bytes_fsync_new(path: Path, payload: bytes) -> None:
    """Create one file inside an unpublished generation and durably write it."""

    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _copy_file_fsync_new(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    """Copy *source* into an independent, fsynced inode and verify copied bytes."""

    expected_sha256 = _require_sha256(expected_sha256, "copied artifact SHA")
    if source.is_symlink() or not source.is_file():
        raise InferenceContractError(f"unsafe copy source: {source}")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as input_handle, destination.open("xb") as output_handle:
            for block in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                output_handle.write(block)
                digest.update(block)
                copied += len(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    observed_sha256 = digest.hexdigest()
    if copied != expected_bytes or observed_sha256 != expected_sha256:
        destination.unlink(missing_ok=True)
        raise InferenceContractError(
            f"copied artifact mismatch for {source}: "
            f"bytes={copied}/{expected_bytes}, "
            f"sha256={observed_sha256}/{expected_sha256}"
        )
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (
        source_stat.st_dev == destination_stat.st_dev
        and source_stat.st_ino == destination_stat.st_ino
    ):
        destination.unlink(missing_ok=True)
        raise InferenceContractError("final artifact unexpectedly aliases shard inode")
    return {
        "bytes": copied,
        "sha256": observed_sha256,
        "source_device": int(source_stat.st_dev),
        "source_inode": int(source_stat.st_ino),
        "destination_device": int(destination_stat.st_dev),
        "destination_inode": int(destination_stat.st_ino),
    }


def _load_and_validate_output_npz(
    path: Path,
    *,
    frames: int,
    prediction: bool,
) -> dict[str, np.ndarray]:
    """Strictly reopen one canonical DiffSHEG NPZ and validate its full schema."""

    if path.is_symlink() or not path.is_file():
        raise InferenceContractError(f"unsafe output NPZ: {path}")
    try:
        with zipfile.ZipFile(path, mode="r") as raw_archive:
            members = tuple(info.filename for info in raw_archive.infolist())
        expected_members = tuple(f"{name}.npy" for name in OUTPUT_FIELDS)
        if members != expected_members or len(set(members)) != len(members):
            raise InferenceContractError(
                f"{path}: ZIP members {members} != {expected_members}"
            )
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != OUTPUT_FIELDS:
                raise InferenceContractError(
                    f"{path}: fields {tuple(archive.files)} != {OUTPUT_FIELDS}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in OUTPUT_FIELDS
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise InferenceContractError(f"cannot read output NPZ {path}: {exc}") from exc

    expected = {
        "betas": ((BETA_DIM,), np.dtype(np.float32)),
        "poses": ((frames, POSE_DIM), np.dtype(np.float32)),
        "expressions": ((frames, EXPRESSION_DIM), np.dtype(np.float32)),
        "trans": ((frames, 3), np.dtype(np.float32)),
        "model": ((), np.asarray("smplx2020").dtype),
        "gender": ((), np.asarray("neutral").dtype),
        "mocap_frame_rate": ((), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise InferenceContractError(
                f"{path}: {name} {array.shape}/{array.dtype} != {shape}/{dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise InferenceContractError(f"{path}: non-finite {name}")
    if arrays["model"].item() != "smplx2020":
        raise InferenceContractError(f"{path}: model is not smplx2020")
    if arrays["gender"].item() != "neutral":
        raise InferenceContractError(f"{path}: gender is not neutral")
    if int(arrays["mocap_frame_rate"].item()) != POSE_FPS:
        raise InferenceContractError(f"{path}: mocap frame rate is not {POSE_FPS}")
    if prediction and not np.array_equal(
        arrays["poses"][:, 69:75],
        np.zeros((frames, 6), dtype=np.float32),
    ):
        raise InferenceContractError(f"{path}: prediction eye pose is not zero")
    return arrays


def _git_output(source_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InferenceContractError(
            f"cannot inspect formal source Git tree {source_root}: {exc}"
        ) from exc
    return completed.stdout.strip()


def _source_receipt(args: argparse.Namespace) -> dict[str, Any]:
    script = Path(__file__).resolve()
    source_root = script.parents[2]
    observed_commit = _git_output(source_root, "rev-parse", "HEAD^{commit}")
    observed_tree = _git_output(source_root, "rev-parse", "HEAD^{tree}")
    observed_origin = _git_output(source_root, "remote", "get-url", "origin")
    observed_script_sha = sha256_file(script)
    if observed_commit != args.expected_source_commit:
        raise InferenceContractError(
            f"source commit {observed_commit} != {args.expected_source_commit}"
        )
    if observed_tree != args.expected_source_tree:
        raise InferenceContractError(
            f"source tree {observed_tree} != {args.expected_source_tree}"
        )
    expected_origin = "git@github.com:Xiangyue-Zhang/SemTalk.git"
    if observed_origin != expected_origin:
        raise InferenceContractError(
            f"source origin {observed_origin!r} != {expected_origin!r}"
        )
    if observed_script_sha != args.expected_inference_script_sha256:
        raise InferenceContractError(
            "inference script SHA mismatch: "
            f"{observed_script_sha} != {args.expected_inference_script_sha256}"
        )
    dirty = _git_output(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if dirty:
        raise InferenceContractError(
            "formal inference source tree is not clean: "
            + dirty.splitlines()[0]
        )
    try:
        script_relative = str(script.relative_to(source_root))
    except ValueError as exc:
        raise InferenceContractError("inference script escapes source root") from exc
    tracked = _git_output(source_root, "ls-files", "--error-unmatch", script_relative)
    if tracked != script_relative:
        raise InferenceContractError("inference script is not tracked by source tree")
    return {
        "source_root": str(source_root),
        "origin": observed_origin,
        "commit": observed_commit,
        "tree": observed_tree,
        "clean": True,
        "script": str(script),
        "script_relative": script_relative,
        "script_sha256": observed_script_sha,
    }


def _expected_source_role_receipt(
    *,
    role: str,
    commit: str,
    tree: str,
) -> dict[str, str]:
    formats = {
        "training": "semtalk_show_training_source_expectation_v1",
        "base_training": (
            "semtalk_show_base_training_source_expectation_v1"
        ),
        "transfer_training": (
            "semtalk_show_transfer_training_source_expectation_v1"
        ),
        "input_artifact": "semtalk_show_input_artifact_source_v1",
    }
    if role not in formats:
        raise ValueError(f"unsupported source role: {role!r}")
    return {
        "format": formats[role],
        "origin": EXPECTED_ORIGIN,
        "commit": commit,
        "tree": tree,
    }


def _official_adapt_producer_source_receipts(
    args: argparse.Namespace,
) -> dict[str, dict[str, str]]:
    """Return the two independently pinned official-adapt producer trees."""

    specifications = {
        "base_training": (
            "expected_base_training_source_commit",
            "expected_base_training_source_tree",
        ),
        "transfer_training": (
            "expected_transfer_training_source_commit",
            "expected_transfer_training_source_tree",
        ),
    }
    receipts: dict[str, dict[str, str]] = {}
    for role, (commit_name, tree_name) in specifications.items():
        commit = getattr(args, commit_name, None)
        tree = getattr(args, tree_name, None)
        if type(commit) is not str or type(tree) is not str:
            raise InferenceContractError(
                f"official_show_adapt_v1 lacks the {role} source receipt"
            )
        try:
            normalized_commit = _require_git_oid(
                commit,
                f"--expected-{role.replace('_', '-')}-source-commit",
            )
            normalized_tree = _require_git_oid(
                tree,
                f"--expected-{role.replace('_', '-')}-source-tree",
            )
        except ValueError as exc:
            raise InferenceContractError(str(exc)) from exc
        receipts[role] = _expected_source_role_receipt(
            role=role,
            commit=normalized_commit,
            tree=normalized_tree,
        )
    return receipts


def _official_adapt_source_roles(
    *,
    inference_source: Mapping[str, Any],
    producer_sources: Mapping[str, Mapping[str, Any]],
    input_source: Mapping[str, Any],
    canonical_source: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if set(producer_sources) != {"base_training", "transfer_training"}:
        raise InferenceContractError(
            "official_show_adapt_v1 requires exactly the Base-training and "
            "transfer-training producer source receipts"
        )
    return {
        "inference": dict(inference_source),
        "base_training": dict(producer_sources["base_training"]),
        "transfer_training": dict(producer_sources["transfer_training"]),
        "input_artifact": dict(input_source),
        "canonical": dict(canonical_source),
    }


@contextmanager
def _finalize_lock(output_root: Path) -> Iterable[None]:
    lock_path = output_root / ".finalize.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InferenceContractError("another finalizer holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonical_clip_id(source_clip_id: str) -> str:
    pieces = source_clip_id.split("/")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        raise InferenceContractError(
            f"invalid canonical SHOW source clip ID: {source_clip_id!r}"
        )
    speaker, _video, sequence = pieces
    if speaker not in SHOW_SPEAKER_IDS:
        raise InferenceContractError(f"unknown SHOW speaker {speaker!r}")
    if (
        "__" in speaker
        or "/" in sequence
        or sequence in {"", ".", ".."}
        or "\x00" in sequence
    ):
        raise InferenceContractError(
            f"unsafe canonical SHOW source clip ID: {source_clip_id!r}"
        )
    return f"{speaker}__{sequence}"


def _resolved_regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise InferenceContractError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _verify_file_sha(path: Path, expected: str, label: str) -> str:
    expected = _require_sha256(expected, f"{label} expected SHA")
    observed = sha256_file(path)
    if observed != expected:
        raise InferenceContractError(
            f"{label} SHA mismatch: {observed} != {expected} ({path})"
        )
    return observed


def _read_verified_checkpoint_snapshot(
    path: str | Path,
    expected: str,
    label: str,
) -> tuple[Path, bytes, str]:
    """Read, hash, and later deserialize one immutable byte snapshot."""
    resolved, payload, observed, _metadata = _safe_file_snapshot(
        path,
        label,
        expected_sha256=expected,
    )
    return resolved, payload, observed


def _validate_canonical_root_receipts(
    manifest: Path,
    summary_path: Path,
    lineage_path: Path,
    expected_manifest_sha: str,
    expected_canonical_source_commit: str,
    expected_canonical_source_tree: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    manifest_sha = _verify_file_sha(
        manifest,
        expected_manifest_sha,
        "canonical manifest",
    )
    summary = load_json(summary_path)
    lineage = load_json(lineage_path)
    if (
        summary.get("status") != "complete"
        or summary.get("schema_name") != "semtalk-show-canonical-motion"
        or _require_exact_int(
            summary.get("schema_version"),
            "canonical summary schema_version",
        )
        != 1
        or summary.get("manifest_sha256") != manifest_sha
        or summary.get("finite") is not True
        or summary.get("exact_once") is not True
        or summary.get("split_disjoint") is not True
    ):
        raise InferenceContractError("canonical cache summary is not strictly complete")
    split_counts = summary.get("split_counts")
    if (
        not isinstance(split_counts, dict)
        or _require_exact_int(
            split_counts.get("test"),
            "canonical summary test split count",
        )
        != EXPECTED_TEST_CLIPS
        or _require_exact_int(
            summary.get("num_shards"),
            "canonical summary num_shards",
        )
        != EXPECTED_NUM_SHARDS
        or _require_exact_int(
            summary.get("clip_count"),
            "canonical summary clip_count",
        )
        != sum(
            _require_exact_int(
                value,
                f"canonical summary split_counts[{key!r}]",
            )
            for key, value in split_counts.items()
        )
        or _require_exact_int(
            summary.get("frame_count"),
            "canonical summary frame_count",
        )
        <= 0
    ):
        raise InferenceContractError(
            "canonical cache root counts/shards are inconsistent"
        )
    contract = lineage.get("lineage_contract")
    contract_sha = lineage.get("lineage_contract_sha256")
    if (
        not isinstance(contract, dict)
        or canonical_json_sha256(contract) != contract_sha
        or summary.get("lineage_contract_sha256") != contract_sha
    ):
        raise InferenceContractError("canonical lineage contract hash is invalid")
    canonical_contract_expectations = {
        "schema_name": "semtalk-show-canonical-motion",
        "schema_version": 1,
        "split_counts": {
            key: _require_exact_int(
                value,
                f"canonical summary split_counts[{key!r}]",
            )
            for key, value in split_counts.items()
        },
        "speaker_mapping": SHOW_SPEAKER_IDS,
        "pose_fps": POSE_FPS,
        "source_audio_sample_rate": CANONICAL_SOURCE_AUDIO_RATE,
        "hubert_target_sample_rate": CANONICAL_HUBERT_TARGET_RATE,
        "audio_channel_protocol": CANONICAL_AUDIO_CHANNEL_PROTOCOL,
        "npz_fields": {
            "pose": ["frames", POSE_DIM],
            "contact": ["frames", 4],
            "facial": ["frames", EXPRESSION_DIM],
            "beta": ["frames", BETA_DIM],
            "trans": ["frames", 3],
            "speaker_id": ["frames", 1],
        },
    }
    for key, expected in canonical_contract_expectations.items():
        if contract.get(key) != expected:
            raise InferenceContractError(
                f"canonical lineage contract {key} mismatch"
            )
    canonical_source = contract.get("source_receipt")
    if (
        not isinstance(canonical_source, dict)
        or canonical_source.get("origin")
        != "git@github.com:Xiangyue-Zhang/SemTalk.git"
        or canonical_source.get("commit")
        != expected_canonical_source_commit
        or canonical_source.get("tree")
        != expected_canonical_source_tree
        or summary.get("source_receipt_sha256")
        != canonical_json_sha256(canonical_source)
    ):
        raise InferenceContractError("canonical source receipt mismatch")
    if lineage.get("final_manifest_sha256") != manifest_sha:
        raise InferenceContractError("canonical lineage does not bind the manifest")
    build_runtime = lineage.get("build_runtime")
    build_runtime_sha = lineage.get("build_runtime_sha256")
    if (
        not isinstance(build_runtime, dict)
        or canonical_json_sha256(build_runtime) != build_runtime_sha
        or summary.get("build_runtime_sha256") != build_runtime_sha
    ):
        raise InferenceContractError("canonical runtime receipt mismatch")
    if summary.get("lineage_sha256") != canonical_json_sha256(lineage):
        raise InferenceContractError("canonical summary does not bind lineage JSON")
    return summary, lineage, manifest_sha, str(contract_sha)


def _validate_canonical_audio_metadata(
    row: Mapping[str, Any],
    context: str,
) -> None:
    channels = row.get("wav_channels")
    sample_width = row.get("wav_sample_width")
    sample_rate = row.get("wav_sample_rate")
    frames = row.get("wav_frames")
    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels not in {1, 2}
        or isinstance(sample_width, bool)
        or not isinstance(sample_width, int)
        or sample_width != CANONICAL_SOURCE_AUDIO_SAMPLE_WIDTH
        or isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate != CANONICAL_SOURCE_AUDIO_RATE
        or isinstance(frames, bool)
        or not isinstance(frames, int)
        or frames < 1
        or row.get("wav_mono_policy") != CANONICAL_WAV_MONO_POLICY
    ):
        raise InferenceContractError(
            f"{context}: invalid canonical WAV metadata"
        )


def _canonical_test_rows(
    manifest: Path,
    *,
    expected_lineage_contract_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    expected_lineage_contract_sha256 = _require_sha256(
        expected_lineage_contract_sha256,
        "canonical root lineage contract SHA",
    )
    required = {
        "global_index",
        "clip_id",
        "split",
        "speaker",
        "speaker_id",
        "sequence",
        "source_wav",
        "canonical_npz",
        "frames",
        "pose_fps",
        "source_wav_sha256",
        "canonical_npz_sha256",
        "lineage_contract_sha256",
        "wav_channels",
        "wav_sample_width",
        "wav_sample_rate",
        "wav_frames",
        "wav_mono_policy",
    }
    test_rows: list[dict[str, Any]] = []
    all_indices: set[int] = set()
    for row in load_jsonl(manifest):
        index = row.get("global_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise InferenceContractError(f"invalid canonical global_index {index!r}")
        if index in all_indices:
            raise InferenceContractError(f"duplicate canonical global_index {index}")
        all_indices.add(index)
        if row.get("split") != "test":
            continue
        missing = sorted(required - set(row))
        if missing:
            raise InferenceContractError(
                f"canonical row {row.get('clip_id')!r} misses {missing}"
            )
        clip_id = str(row["clip_id"])
        _validate_canonical_audio_metadata(row, clip_id)
        speaker = str(row["speaker"])
        if clip_id.split("/", 1)[0] != speaker:
            raise InferenceContractError(f"{clip_id}: speaker field mismatch")
        expected_speaker_id = SHOW_SPEAKER_IDS.get(speaker)
        if (
            expected_speaker_id is None
            or _require_exact_int(
                row.get("speaker_id"),
                f"{clip_id} speaker_id",
            )
            != expected_speaker_id
        ):
            raise InferenceContractError(f"{clip_id}: invalid SHOW speaker ID")
        if _require_exact_int(row.get("pose_fps"), f"{clip_id} pose_fps") != POSE_FPS:
            raise InferenceContractError(f"{clip_id}: pose_fps must be 30")
        if (
            _require_exact_int(row.get("frames"), f"{clip_id} frames")
            < DIFFSHEG_WINDOW
        ):
            raise InferenceContractError(
                f"{clip_id}: fewer than {DIFFSHEG_WINDOW} DiffSHEG frames"
            )
        if (
            _require_sha256(
                str(row["lineage_contract_sha256"]),
                f"{clip_id} canonical lineage contract SHA",
            )
            != expected_lineage_contract_sha256
        ):
            raise InferenceContractError(
                f"{clip_id}: canonical row/root lineage contract mismatch"
            )
        canonical_clip_id(clip_id)
        test_rows.append(row)
    test_rows.sort(
        key=lambda row: _require_exact_int(
            row.get("global_index"),
            "canonical row global_index",
        )
    )
    if len(test_rows) != EXPECTED_TEST_CLIPS:
        raise InferenceContractError(
            f"canonical test rows {len(test_rows)} != {EXPECTED_TEST_CLIPS}"
        )
    observed_speakers = {str(row["speaker"]) for row in test_rows}
    if observed_speakers != set(SHOW_SPEAKER_IDS):
        raise InferenceContractError(
            "SHOW test must contain all four canonical speakers and no speaker2"
        )
    clip_ids = [str(row["clip_id"]) for row in test_rows]
    output_ids = [canonical_clip_id(clip_id) for clip_id in clip_ids]
    if len(set(clip_ids)) != len(clip_ids):
        raise InferenceContractError("duplicate canonical test clip_id")
    if len(set(output_ids)) != len(output_ids):
        raise InferenceContractError("SHOW clip IDs collide after canonicalization")
    return test_rows, {
        str(row["clip_id"]): row
        for row in test_rows
    }


def _audio_feature_rows(
    manifests: Sequence[Path],
    canonical_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, list[dict[str, Any]]],
]:
    required = {
        "format",
        "split",
        "clip_id",
        "canonical_npz",
        "canonical_npz_sha256",
        "source_wav",
        "source_wav_sha256",
        "audio_feature_npz",
        "audio_feature_npz_sha256",
        "frames",
        "lineage_contract_sha256",
        "audio_lineage_contract_sha256",
        "shard_id",
        "num_shards",
        "source_audio_field",
        "audio_samples_16k",
        "native_30fps_frames",
        "canonical_usable_frames",
        "discarded_source_30fps_frames",
        "edge_padded_tail_frames",
        "source_sample_rate",
        "hubert_native_frames",
        "beat_shape",
        "hubert_shape",
    }
    result: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    rows_by_manifest: dict[str, list[dict[str, Any]]] = {}
    for path in manifests:
        resolved = _resolved_regular_file(path, "audio feature manifest")
        manifest_key = str(resolved)
        hashes[manifest_key] = sha256_file(resolved)
        manifest_rows: list[dict[str, Any]] = []
        for row in load_jsonl(resolved):
            if row.get("split") != "test":
                raise InferenceContractError(
                    f"{resolved}: inference feature manifests must contain test only"
                )
            missing = sorted(required - set(row))
            if missing:
                raise InferenceContractError(
                    f"audio row {row.get('clip_id')!r} misses {missing}"
                )
            if row.get("format") != "semtalk_show_audio_clip_v1":
                raise InferenceContractError("unexpected audio feature row format")
            clip_id = str(row["clip_id"])
            if clip_id in result:
                raise InferenceContractError(
                    f"duplicate audio feature clip_id {clip_id!r}"
                )
            canonical = canonical_by_id.get(clip_id)
            if canonical is None:
                raise InferenceContractError(
                    f"audio feature row is not a canonical test clip: {clip_id}"
                )
            expected_values = {
                "canonical_npz": str(
                    Path(str(canonical["canonical_npz"])).expanduser().resolve()
                ),
                "canonical_npz_sha256": canonical["canonical_npz_sha256"],
                "source_wav_sha256": canonical["source_wav_sha256"],
                "source_wav": str(
                    Path(str(canonical["source_wav"])).expanduser().resolve()
                ),
                "frames": _require_exact_int(
                    canonical.get("frames"),
                    f"{clip_id} canonical frames",
                ),
                "lineage_contract_sha256": canonical[
                    "lineage_contract_sha256"
                ],
                "num_shards": EXPECTED_NUM_SHARDS,
            }
            observed_values = {
                "canonical_npz": str(
                    Path(str(row["canonical_npz"])).expanduser().resolve()
                ),
                "canonical_npz_sha256": row["canonical_npz_sha256"],
                "source_wav_sha256": row["source_wav_sha256"],
                "source_wav": str(
                    Path(str(row["source_wav"])).expanduser().resolve()
                ),
                "frames": _require_exact_int(
                    row.get("frames"),
                    f"{clip_id} audio frames",
                ),
                "lineage_contract_sha256": row["lineage_contract_sha256"],
                "num_shards": _require_exact_int(
                    row.get("num_shards"),
                    f"{clip_id} audio num_shards",
                ),
            }
            if observed_values != expected_values:
                raise InferenceContractError(
                    f"{clip_id}: audio/canonical lineage mismatch"
                )
            if row["source_audio_field"] != "source_wav":
                raise InferenceContractError(
                    f"{clip_id}: audio feature did not use canonical source_wav"
                )
            frames = observed_values["frames"]
            native_frames = _require_exact_int(
                row.get("native_30fps_frames"),
                f"{clip_id} native audio frames",
            )
            canonical_usable = _require_exact_int(
                row.get("canonical_usable_frames"),
                f"{clip_id} canonical usable frames",
            )
            discarded = _require_exact_int(
                row.get("discarded_source_30fps_frames"),
                f"{clip_id} discarded audio frames",
            )
            edge_padded = _require_exact_int(
                row.get("edge_padded_tail_frames"),
                f"{clip_id} edge-padded audio frames",
            )
            audio_samples = _require_exact_int(
                row.get("audio_samples_16k"),
                f"{clip_id} 16 kHz audio samples",
            )
            source_sample_rate = _require_exact_int(
                row.get("source_sample_rate"),
                f"{clip_id} source sample rate",
            )
            hubert_native_frames = _require_exact_int(
                row.get("hubert_native_frames"),
                f"{clip_id} native HuBERT frames",
            )
            if (
                frames <= 0
                or native_frames <= 0
                or audio_samples <= 0
                or source_sample_rate <= 0
                or hubert_native_frames <= 0
                or native_frames != (audio_samples * POSE_FPS) // 16000
                or row.get("beat_shape") != [frames, 3]
                or row.get("hubert_shape") != [frames, 1024]
                or canonical_usable != (frames // POSE_FPS) * POSE_FPS
                or discarded != max(native_frames - frames, 0)
                or edge_padded != max(frames - native_frames, 0)
                or edge_padded > 1
                or (
                    native_frames < frames
                    and native_frames < canonical_usable
                )
            ):
                raise InferenceContractError(
                    f"{clip_id}: invalid public-prefix timing receipt"
                )
            result[clip_id] = row
            manifest_rows.append(row)
        rows_by_manifest[manifest_key] = manifest_rows
    if set(result) != set(canonical_by_id):
        missing = sorted(set(canonical_by_id) - set(result))
        extra = sorted(set(result) - set(canonical_by_id))
        raise InferenceContractError(
            "audio features do not cover canonical test exactly once: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    return result, hashes, rows_by_manifest


def _validate_audio_receipts(
    *,
    manifests: Sequence[Path],
    manifest_hashes: Mapping[str, str],
    rows_by_manifest: Mapping[str, Sequence[Mapping[str, Any]]],
    summary_paths: Sequence[Path],
    lineage_paths: Sequence[Path],
    canonical_manifest: Path,
    canonical_manifest_sha: str,
    canonical_lineage_contract_sha256: str,
    canonical_receipt: Mapping[str, Any],
    expected_source_commit: str,
    expected_source_tree: str,
    expected_hubert_tree_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    canonical_hash_map = {str(canonical_manifest): canonical_manifest_sha}
    manifest_set = set(manifest_hashes)
    summaries_by_manifest: dict[str, dict[str, Any]] = {}
    summary_hashes: dict[str, str] = {}
    for path in summary_paths:
        resolved = _resolved_regular_file(path, "audio feature summary")
        summary = load_json(resolved)
        if (
            summary.get("format") != "semtalk_show_audio_summary_v1"
            or summary.get("status") != "complete"
        ):
            raise InferenceContractError(f"invalid audio summary: {resolved}")
        manifest = str(Path(str(summary["output_manifest"])).resolve())
        if manifest not in manifest_set or manifest in summaries_by_manifest:
            raise InferenceContractError(
                f"audio summary does not bind one requested manifest: {resolved}"
            )
        if summary.get("output_manifest_sha256") != manifest_hashes[manifest]:
            raise InferenceContractError(f"{resolved}: manifest SHA mismatch")
        if (
            _require_exact_int(
                summary.get("num_shards"),
                "audio summary num_shards",
            )
            != EXPECTED_NUM_SHARDS
        ):
            raise InferenceContractError(f"{resolved}: expected eight audio shards")
        summary_hashes[str(resolved)] = sha256_file(resolved)
        summaries_by_manifest[manifest] = summary
    if set(summaries_by_manifest) != manifest_set:
        raise InferenceContractError(
            "every audio manifest needs exactly one complete summary"
        )

    lineages_by_manifest: dict[str, dict[str, Any]] = {}
    lineage_hashes: dict[str, str] = {}
    for path in lineage_paths:
        resolved = _resolved_regular_file(path, "audio feature lineage")
        lineage = load_json(resolved)
        if (
            lineage.get("format") != "semtalk_show_audio_lineage_v1"
            or lineage.get("status") != "complete"
        ):
            raise InferenceContractError(f"invalid audio lineage: {resolved}")
        manifest = str(Path(str(lineage["output_manifest"])).resolve())
        if manifest not in manifest_set or manifest in lineages_by_manifest:
            raise InferenceContractError(
                f"audio lineage does not bind one requested manifest: {resolved}"
            )
        if lineage.get("output_manifest_sha256") != manifest_hashes[manifest]:
            raise InferenceContractError(f"{resolved}: manifest SHA mismatch")
        if lineage.get("canonical_manifest_sha256") != canonical_hash_map:
            raise InferenceContractError(f"{resolved}: canonical SHA mismatch")
        if lineage.get("canonical_receipt") != canonical_receipt:
            raise InferenceContractError(
                f"{resolved}: canonical receipt mismatch"
            )
        lineage_source = lineage.get("source_receipt")
        if (
            not isinstance(lineage_source, dict)
            or lineage_source.get("origin")
            != "git@github.com:Xiangyue-Zhang/SemTalk.git"
            or lineage_source.get("commit") != expected_source_commit
            or lineage_source.get("tree") != expected_source_tree
        ):
            raise InferenceContractError(
                f"{resolved}: feature source receipt mismatch"
            )
        if (
            lineage.get("hubert_model_tree_sha256")
            != expected_hubert_tree_sha256
        ):
            raise InferenceContractError(
                f"{resolved}: pinned HuBERT tree mismatch"
            )
        if (
            lineage.get("lineage_contract_sha256")
            != canonical_lineage_contract_sha256
        ):
            raise InferenceContractError(
                f"{resolved}: canonical motion contract mismatch"
            )
        contract = lineage.get("audio_lineage_contract")
        if (
            not isinstance(contract, dict)
            or canonical_json_sha256(contract)
            != lineage.get("audio_lineage_contract_sha256")
        ):
            raise InferenceContractError(
                f"{resolved}: invalid audio lineage contract"
            )
        contract_expectations = {
            "format": "semtalk_show_audio_lineage_contract_v1",
            "protocol": lineage.get("protocol"),
            "canonical_manifest_sha256": canonical_hash_map,
            "canonical_receipt": canonical_receipt,
            "source_receipt": lineage_source,
            "hubert_model": lineage.get("hubert_model"),
            "hubert_model_tree_sha256": lineage.get(
                "hubert_model_tree_sha256"
            ),
            "shard_id": lineage.get("shard_id"),
            "num_shards": lineage.get("num_shards"),
            "full_split_clips": EXPECTED_TEST_CLIPS,
            "runtime": lineage.get("runtime"),
        }
        for key, expected in contract_expectations.items():
            if contract.get(key) != expected:
                raise InferenceContractError(
                    f"{resolved}: audio contract {key} mismatch"
                )
        _require_sha256(
            str(lineage.get("hubert_model_tree_sha256")),
            f"{resolved} HuBERT tree SHA",
        )
        protocol = lineage.get("protocol")
        hubert_preprocessing = (
            protocol.get("hubert_preprocessing")
            if isinstance(protocol, dict)
            else None
        )
        expected_hubert_preprocessing = {
            "label": "corrected_true_16khz",
            "source_decode": "librosa.load(sr=None,mono=True)",
            "channel_mix": "librosa_to_mono_arithmetic_mean",
            "resample": "native_to_true_16000hz_before_feature_extractor",
            "feature_extractor_sampling_rate": 16000,
            "released_code_difference": (
                "The public loader decodes with librosa's 22050 Hz default "
                "and passes that waveform to a feature extractor declared as "
                "16000 Hz. This run corrects that sample-rate mismatch."
            ),
            "claim": "adapted_reconstruction_not_official_input_exact",
        }
        if (
            not isinstance(protocol, dict)
            or protocol.get("split") != "test"
            or _require_exact_int(
                protocol.get("sample_rate"),
                "audio protocol sample_rate",
            )
            != 16000
            or _require_exact_int(
                protocol.get("fps"),
                "audio protocol fps",
            )
            != POSE_FPS
            or hubert_preprocessing != expected_hubert_preprocessing
            or protocol.get("alignment") != AUDIO_ALIGNMENT_PROTOCOL
        ):
            raise InferenceContractError(
                f"{resolved}: audio protocol is not corrected SHOW test audio"
            )
        forbidden = set(
            protocol.get("forbidden_components", [])
        )
        if not FORBIDDEN_COMPONENTS.issubset(forbidden):
            raise InferenceContractError(
                f"{resolved}: forbidden-component gate is incomplete"
            )
        summary = summaries_by_manifest[manifest]
        if (
            str(Path(str(summary.get("lineage_json"))).resolve())
            != str(resolved)
            or summary.get("lineage_json_sha256") != sha256_file(resolved)
        ):
            raise InferenceContractError(f"{resolved}: summary lineage SHA mismatch")
        if (
            _require_exact_int(
                lineage.get("num_shards"),
                "audio lineage num_shards",
            )
            != EXPECTED_NUM_SHARDS
            or _require_exact_int(
                lineage.get("full_split_clips"),
                "audio lineage full_split_clips",
            )
            != EXPECTED_TEST_CLIPS
        ):
            raise InferenceContractError(f"{resolved}: expected eight audio shards")
        summary_expectations = {
            "shard_id": lineage.get("shard_id"),
            "num_shards": lineage.get("num_shards"),
            "full_split_clips": EXPECTED_TEST_CLIPS,
            "shard_clips": lineage.get("shard_clips"),
            "artifact_aggregate_sha256": lineage.get(
                "artifact_aggregate_sha256"
            ),
        }
        for key, expected in summary_expectations.items():
            if summary.get(key) != expected:
                raise InferenceContractError(
                    f"{resolved}: audio summary/lineage {key} mismatch"
                )
        manifest_rows = rows_by_manifest[manifest]
        if (
            _require_exact_int(
                lineage.get("shard_clips"),
                "audio lineage shard_clips",
            )
            != len(manifest_rows)
        ):
            raise InferenceContractError(
                f"{resolved}: audio shard row count mismatch"
            )
        output_directories = {
            str(
                Path(str(row["audio_feature_npz"]))
                .expanduser()
                .resolve()
                .parent
            )
            for row in manifest_rows
        }
        expected_output_directory = (
            next(iter(output_directories))
            if output_directories
            else str(Path(str(summary.get("output_dir"))).expanduser().resolve())
        )
        if (
            len(output_directories) > 1
            or str(
                Path(str(summary.get("output_dir"))).expanduser().resolve()
            )
            != expected_output_directory
        ):
            raise InferenceContractError(
                f"{resolved}: audio output directory receipt mismatch"
            )
        artifact_aggregate = hashlib.sha256()
        for row in manifest_rows:
            if (
                _require_exact_int(
                    row.get("shard_id"),
                    f"{row.get('clip_id')} audio row shard_id",
                )
                != _require_exact_int(
                    lineage.get("shard_id"),
                    "audio lineage shard_id",
                )
                or _require_exact_int(
                    row.get("num_shards"),
                    f"{row.get('clip_id')} audio row num_shards",
                )
                != EXPECTED_NUM_SHARDS
                or row["audio_lineage_contract_sha256"]
                != lineage["audio_lineage_contract_sha256"]
                or row["lineage_contract_sha256"]
                != canonical_lineage_contract_sha256
            ):
                raise InferenceContractError(
                    f"{resolved}: audio row is owned by another lineage"
                )
            artifact_sha = _require_sha256(
                str(row["audio_feature_npz_sha256"]),
                f"{row['clip_id']} audio feature SHA",
            )
            artifact_aggregate.update(str(row["clip_id"]).encode("utf-8"))
            artifact_aggregate.update(bytes.fromhex(artifact_sha))
        if (
            artifact_aggregate.hexdigest()
            != lineage.get("artifact_aggregate_sha256")
        ):
            raise InferenceContractError(
                f"{resolved}: audio artifact aggregate mismatch"
            )
        lineage_hashes[str(resolved)] = sha256_file(resolved)
        lineages_by_manifest[manifest] = lineage
    if set(lineages_by_manifest) != manifest_set:
        raise InferenceContractError(
            "every audio manifest needs exactly one complete lineage"
        )
    if len(manifest_set) != EXPECTED_NUM_SHARDS:
        raise InferenceContractError(
            f"expected {EXPECTED_NUM_SHARDS} audio manifests, got {len(manifest_set)}"
        )
    shard_ids = sorted(
        _require_exact_int(
            lineage.get("shard_id"),
            "audio lineage shard_id",
        )
        for lineage in lineages_by_manifest.values()
    )
    if shard_ids != list(range(EXPECTED_NUM_SHARDS)):
        raise InferenceContractError(
            f"audio lineage shard IDs are not 0..7: {shard_ids}"
        )
    if sum(
        _require_exact_int(
            lineage.get("shard_clips"),
            "audio lineage shard_clips",
        )
        for lineage in lineages_by_manifest.values()
    ) != EXPECTED_TEST_CLIPS:
        raise InferenceContractError("audio shard counts do not sum to 1708")
    model_hashes = {
        str(lineage["hubert_model_tree_sha256"])
        for lineage in lineages_by_manifest.values()
    }
    protocols = {
        canonical_json_sha256(lineage["protocol"])
        for lineage in lineages_by_manifest.values()
    }
    runtimes = {
        canonical_json_sha256(lineage["runtime"])
        for lineage in lineages_by_manifest.values()
    }
    contracts = {
        str(lineage["audio_lineage_contract_sha256"])
        for lineage in lineages_by_manifest.values()
    }
    if (
        any(len(group) != 1 for group in (model_hashes, protocols, runtimes))
        or len(contracts) != EXPECTED_NUM_SHARDS
    ):
        raise InferenceContractError(
            "audio shards disagree on model/protocol/runtime or do not have "
            "one lineage contract per shard"
        )
    return (
        [lineages_by_manifest[key] for key in sorted(lineages_by_manifest)],
        summary_hashes,
        lineage_hashes,
    )


def _load_canonical_clip(
    row: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], int]:
    path = _resolved_regular_file(row["canonical_npz"], "canonical NPZ")
    _verify_file_sha(
        path,
        str(row["canonical_npz_sha256"]),
        "canonical NPZ",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != CANONICAL_FIELDS:
                raise InferenceContractError(
                    f"{path}: fields {tuple(archive.files)} != {CANONICAL_FIELDS}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in CANONICAL_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise InferenceContractError(f"cannot read canonical NPZ {path}: {exc}") from exc
    frames = _require_exact_int(row.get("frames"), "canonical row frames")
    expected = {
        "pose": ((frames, POSE_DIM), np.float32),
        "contact": ((frames, 4), np.float32),
        "facial": ((frames, EXPRESSION_DIM), np.float32),
        "beta": ((frames, BETA_DIM), np.float32),
        "trans": ((frames, 3), np.float32),
        "speaker_id": ((frames, 1), np.int64),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise InferenceContractError(
                f"{path}: {name} {array.shape}/{array.dtype} != {shape}/{dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise InferenceContractError(f"{path}: non-finite {name}")
    contact = arrays["contact"]
    if not np.logical_or(contact == 0.0, contact == 1.0).all():
        raise InferenceContractError(f"{path}: contact is not binary")
    speaker_id = SHOW_SPEAKER_IDS[str(row["speaker"])]
    if not np.array_equal(
        arrays["speaker_id"],
        np.full((frames, 1), speaker_id, dtype=np.int64),
    ):
        raise InferenceContractError(f"{path}: speaker_id changes or is remapped")
    if not np.array_equal(
        arrays["beta"],
        np.broadcast_to(arrays["beta"][0], arrays["beta"].shape),
    ):
        raise InferenceContractError(f"{path}: betas vary within one SHOW clip")
    return arrays, frames


def _load_audio_features(
    row: Mapping[str, Any],
    *,
    expected_frames: int,
) -> dict[str, np.ndarray]:
    path = _resolved_regular_file(row["audio_feature_npz"], "audio feature NPZ")
    _verify_file_sha(
        path,
        str(row["audio_feature_npz_sha256"]),
        "audio feature NPZ",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != AUDIO_FIELDS:
                raise InferenceContractError(
                    f"{path}: fields {tuple(archive.files)} != {AUDIO_FIELDS}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in AUDIO_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise InferenceContractError(
            f"cannot read audio feature NPZ {path}: {exc}"
        ) from exc
    expected = {
        "beat": (expected_frames, 3),
        "hubert": (expected_frames, 1024),
    }
    for name, shape in expected.items():
        array = arrays[name]
        if (
            array.shape != shape
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise InferenceContractError(
                f"{path}: invalid {name} {array.shape}/{array.dtype}"
            )
    if (
        _require_exact_int(row.get("frames"), "audio feature row frames")
        != expected_frames
    ):
        raise InferenceContractError(f"{path}: feature frame count mismatch")
    return arrays


def _torch_load_checkpoint(
    payload_bytes: bytes,
    path: Path,
) -> dict[str, Any]:
    import torch

    buffer = io.BytesIO(payload_bytes)
    try:
        payload = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        payload = torch.load(buffer, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(
        payload.get("model_state"), dict
    ):
        raise InferenceContractError(f"{path}: checkpoint lacks model_state")
    return payload


def _finite_state_dict(state: Mapping[str, Any], path: Path) -> None:
    import torch

    for name, value in state.items():
        if torch.is_tensor(value) and (
            value.is_floating_point() or value.is_complex()
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise InferenceContractError(
                    f"{path}: non-finite checkpoint tensor {name}"
                )


def _normalize_data_parallel_state(
    state: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    import torch

    keys = list(state)
    if not keys:
        raise InferenceContractError(f"{path}: empty model_state")
    if any(not isinstance(key, str) for key in keys):
        raise InferenceContractError(f"{path}: model_state keys must be strings")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise InferenceContractError(
            f"{path}: every model_state value must be a tensor"
        )
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise InferenceContractError(
            f"{path}: mixed DataParallel and unprefixed state keys"
        )
    normalized = {
        (key[7:] if prefixed[0] else key): value
        for key, value in state.items()
    }
    if len(normalized) != len(state):
        raise InferenceContractError(f"{path}: state-key collision after normalization")
    forbidden_tokens = ("sparse", "semgate", "speaker2", "clip", "emotion")
    for key in normalized:
        lowered = key.lower()
        if any(token in lowered for token in forbidden_tokens):
            raise InferenceContractError(
                f"{path}: forbidden checkpoint component in key {key!r}"
            )
    return normalized


def _state_dict_schema_sha256(state: Mapping[str, Any]) -> str:
    schema = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(state.items())
    ]
    return canonical_json_sha256(schema)


def _released_weight_source_receipt(
    *,
    classification: str = RELEASED_ALL_SPEAKERS_CLASSIFICATION,
    checkpoint_container_schema: Sequence[str] = ("model_state",),
) -> dict[str, Any]:
    return {
        "format": "semtalk_released_all_speakers_weight_source_v1",
        "prerequisite_source": RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
        "classification": classification,
        "release_trust_root": dict(
            RELEASED_ALL_SPEAKERS_RELEASE_TRUST_ROOT
        ),
        "official_release": True,
        "training_dataset": "BEAT2",
        "speaker_scope": "All-Speakers",
        "show_trained": False,
        "checkpoint_container_schema": list(checkpoint_container_schema),
    }


def _official_released_weights_receipt() -> dict[str, dict[str, str]]:
    return {
        stage: {
            "filename": str(RELEASED_ALL_SPEAKERS_MODELS[stage]["filename"]),
            "sha256": str(RELEASED_ALL_SPEAKERS_MODELS[stage]["sha256"]),
        }
        for stage in CHECKPOINT_STAGES
    }


def _parse_verified_json_snapshot(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    resolved, snapshot, observed_sha = _read_verified_checkpoint_snapshot(
        path,
        expected_sha256,
        label,
    )
    try:
        payload = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                InferenceContractError(
                    f"{resolved}: non-finite JSON constant {token!r}"
                )
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise InferenceContractError(
            f"cannot parse verified JSON snapshot {resolved}: {exc}"
        ) from exc
    if type(payload) is not dict:
        raise InferenceContractError(
            f"{resolved}: expected a JSON object"
        )
    return resolved, payload, observed_sha


def _validate_finite_measurement_tree(value: Any, label: str) -> None:
    if type(value) is dict:
        if not value or any(type(key) is not str or not key for key in value):
            raise InferenceContractError(
                f"{label} must be a non-empty string-keyed object"
            )
        for key, item in value.items():
            _validate_finite_measurement_tree(item, f"{label}.{key}")
        return
    if type(value) is list:
        if not value:
            raise InferenceContractError(f"{label} must not be empty")
        for index, item in enumerate(value):
            _validate_finite_measurement_tree(
                item,
                f"{label}[{index}]",
            )
        return
    if type(value) is int:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise InferenceContractError(
        f"{label} must contain only finite numeric leaves"
    )


def _validate_gate_artifact_receipt(
    value: Any,
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != (
        RELEASED_CROSS_DOMAIN_ARTIFACT_KEYS
    ):
        raise InferenceContractError(
            f"{label} must have the exact path/SHA receipt schema"
        )
    path_value = value.get("path")
    sha_value = value.get("sha256")
    if type(path_value) is not str or not Path(path_value).is_absolute():
        raise InferenceContractError(
            f"{label} path must be absolute and canonical"
        )
    resolved = _resolved_regular_file(Path(path_value), label)
    if str(resolved) != path_value:
        raise InferenceContractError(
            f"{label} path is not canonical: {path_value}"
        )
    if type(sha_value) is not str:
        raise InferenceContractError(f"{label} SHA must be a string")
    observed_sha = _verify_file_sha(resolved, sha_value, label)
    return {"path": str(resolved), "sha256": observed_sha}


def _validate_released_cross_domain_gate(
    *,
    args: argparse.Namespace,
    source_receipt: Mapping[str, Any],
    input_source_receipt: Mapping[str, Any],
    canonical_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        args.released_cross_domain_gate_json is None
        or args.expected_released_cross_domain_gate_sha256 is None
    ):
        raise InferenceContractError(
            "fully released zero-shot inference lacks its cross-domain gate"
        )
    gate_path, gate, gate_file_sha = _parse_verified_json_snapshot(
        args.released_cross_domain_gate_json,
        args.expected_released_cross_domain_gate_sha256,
        "released cross-domain gate",
    )
    artifact_receipts = {
        name: _validate_gate_artifact_receipt(
            gate.get(name),
            f"released cross-domain {name.replace('_', ' ')}",
        )
        for name in (
            "gate_script",
            "measurement_receipt",
            "threshold_receipt",
        )
    }
    artifact_paths = {
        receipt["path"] for receipt in artifact_receipts.values()
    }
    if len(artifact_paths) != len(artifact_receipts) or str(gate_path) in (
        artifact_paths
    ):
        raise InferenceContractError(
            f"{gate_path}: cross-domain gate artifacts must be distinct"
        )
    if (
        set(source_receipt) != SOURCE_RECEIPT_KEYS
        or source_receipt.get("clean") is not True
    ):
        raise InferenceContractError(
            "current inference source receipt has an invalid schema"
        )
    source_root_value = source_receipt.get("source_root")
    if (
        type(source_root_value) is not str
        or not Path(source_root_value).is_absolute()
    ):
        raise InferenceContractError(
            "current inference source root is not canonical"
        )
    source_root = Path(source_root_value).resolve()
    if str(source_root) != source_root_value:
        raise InferenceContractError(
            "current inference source root is not canonical"
        )
    gate_script_path = Path(artifact_receipts["gate_script"]["path"])
    try:
        gate_script_relative = str(gate_script_path.relative_to(source_root))
    except ValueError as exc:
        raise InferenceContractError(
            "cross-domain gate script escapes the inference source root"
        ) from exc
    if gate_script_relative != RELEASED_CROSS_DOMAIN_GATE_SCRIPT_RELATIVE:
        raise InferenceContractError(
            "cross-domain gate script is not the frozen formal entrypoint"
        )
    tracked_gate_script = _git_output(
        source_root,
        "ls-files",
        "--error-unmatch",
        gate_script_relative,
    )
    if tracked_gate_script != gate_script_relative:
        raise InferenceContractError(
            "cross-domain gate script is not tracked by the source tree"
        )
    expected_gate_source = {
        "source_root": str(source_root),
        "origin": source_receipt["origin"],
        "commit": source_receipt["commit"],
        "tree": source_receipt["tree"],
        "clean": True,
        "script": str(gate_script_path),
        "script_relative": gate_script_relative,
        "script_sha256": artifact_receipts["gate_script"]["sha256"],
    }
    expected_canonical_receipt = {
        key: canonical_receipt[key]
        for key in (
            "manifest",
            "manifest_sha256",
            "summary",
            "summary_sha256",
            "lineage",
            "lineage_sha256",
            "lineage_contract_sha256",
        )
    }
    expected_source_roles = {
        "current": dict(source_receipt),
        "gate": expected_gate_source,
        "canonical": dict(canonical_receipt["source_receipt"]),
        "input_artifact": dict(input_source_receipt),
    }
    expected_protocol = {
        "mode": FULLY_RELEASED_ZERO_SHOT_MODE,
        "split": "test",
        "show_speakers": [0, 1, 2, 3],
        "exact_once": True,
        "all_tensors_finite": True,
        "deterministic": True,
        "evaluated_components": [
            "face",
            "upper",
            "hands",
            "lower",
            "global_sanity",
        ],
        "bound_not_evaluated": ["base"],
        "forbidden_components": sorted(
            FORBIDDEN_COMPONENTS | {"Speaker2"}
        ),
    }
    if (
        set(gate) != RELEASED_CROSS_DOMAIN_GATE_KEYS
        or gate.get("format") != RELEASED_CROSS_DOMAIN_GATE_FORMAT
        or gate.get("status") != "pass"
        or gate.get("authorization") is not True
        or gate.get("release_trust_root")
        != RELEASED_ALL_SPEAKERS_RELEASE_TRUST_ROOT
        or gate.get("official_weights")
        != _official_released_weights_receipt()
        or gate.get("canonical_receipt") != expected_canonical_receipt
        or gate.get("source_roles") != expected_source_roles
        or gate.get("protocol") != expected_protocol
    ):
        raise InferenceContractError(
            f"{gate_path}: invalid or mismatched released cross-domain gate"
        )
    receipt_sha = gate.get("receipt_sha256")
    if type(receipt_sha) is not str:
        raise InferenceContractError(
            f"{gate_path}: cross-domain receipt SHA is not a string"
        )
    receipt_sha = _require_sha256(
        receipt_sha,
        "released cross-domain gate receipt",
    )
    payload_without_sha = dict(gate)
    del payload_without_sha["receipt_sha256"]
    if canonical_json_sha256(payload_without_sha) != receipt_sha:
        raise InferenceContractError(
            f"{gate_path}: cross-domain receipt SHA mismatch"
        )

    measurements = gate.get("measurements")
    thresholds = gate.get("thresholds")
    _validate_finite_measurement_tree(
        measurements,
        "released cross-domain measurements",
    )
    _validate_finite_measurement_tree(
        thresholds,
        "released cross-domain thresholds",
    )
    decisions = gate.get("decisions")
    if (
        type(decisions) is not dict
        or not decisions
        or any(type(key) is not str or not key for key in decisions)
        or any(value is not True for value in decisions.values())
    ):
        raise InferenceContractError(
            f"{gate_path}: every cross-domain decision must be exactly true"
        )
    return {
        "path": str(gate_path),
        "sha256": gate_file_sha,
        "receipt_sha256": receipt_sha,
        "payload": gate,
        "artifacts": artifact_receipts,
    }


def _load_released_model_state_only(
    path: Path,
    *,
    expected_filename: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], Path, bytes, str]:
    """Read one exact official release checkpoint without pickle fallback."""
    import torch

    candidate = Path(path).expanduser()
    if candidate.name != expected_filename:
        raise InferenceContractError(
            f"released checkpoint filename {candidate.name!r} != "
            f"{expected_filename!r}"
        )
    resolved, snapshot, observed_sha = _read_verified_checkpoint_snapshot(
        candidate,
        expected_sha256,
        "released checkpoint",
    )
    if resolved.name != expected_filename:
        raise InferenceContractError(
            f"unsafe released checkpoint path: {candidate}"
        )
    try:
        payload = torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:  # pragma: no cover - supported formal torch has it
        raise InferenceContractError(
            "released_all_speakers_v1 requires "
            "torch.load(weights_only=True)"
        ) from exc
    except Exception as exc:
        raise InferenceContractError(
            f"{resolved}: cannot deserialize released checkpoint with "
            "weights_only=True"
        ) from exc
    if type(payload) is not dict or set(payload) != {"model_state"}:
        raise InferenceContractError(
            f"{resolved}: released checkpoint must contain only model_state"
        )
    raw_state = payload["model_state"]
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise InferenceContractError(
            f"{resolved}: released model_state must be a non-empty mapping"
        )
    normalized = _normalize_data_parallel_state(raw_state, resolved)
    _finite_state_dict(normalized, resolved)
    return normalized, resolved, snapshot, observed_sha


def _load_released_base_state(
    path: Path,
    *,
    expected_filename: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], Path, bytes, str, dict[str, Any]]:
    """Read the exact official Base envelope and validate all auxiliary state."""
    import torch

    candidate = Path(path).expanduser()
    if candidate.name != expected_filename:
        raise InferenceContractError(
            f"released Base filename {candidate.name!r} != "
            f"{expected_filename!r}"
        )
    resolved, snapshot, observed_sha = _read_verified_checkpoint_snapshot(
        candidate,
        expected_sha256,
        "released Base",
    )
    if resolved.name != expected_filename:
        raise InferenceContractError(
            f"unsafe released Base path: {candidate}"
        )
    try:
        payload = torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:  # pragma: no cover - supported formal torch has it
        raise InferenceContractError(
            "released official Base requires torch.load(weights_only=True)"
        ) from exc
    except Exception as exc:
        raise InferenceContractError(
            f"{resolved}: cannot deserialize released Base with "
            "weights_only=True"
        ) from exc
    if (
        type(payload) is not dict
        or set(payload) != set(RELEASED_BASE_CONTAINER_SCHEMA)
        or type(payload.get("epoch")) is not int
        or payload["epoch"] != 401
        or payload.get("lrs") != RELEASED_BASE_LRS
    ):
        raise InferenceContractError(
            f"{resolved}: invalid exact official Base checkpoint envelope"
        )
    raw_state = payload.get("model_state")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise InferenceContractError(
            f"{resolved}: official Base model_state is invalid"
        )
    state = _normalize_data_parallel_state(raw_state, resolved)
    _finite_state_dict(state, resolved)
    optimizer = payload.get("opt_state")
    if (
        type(optimizer) is not dict
        or set(optimizer) != {"state", "param_groups"}
        or type(optimizer["state"]) is not dict
        or len(optimizer["state"]) != 1_655
        or type(optimizer["param_groups"]) is not list
        or len(optimizer["param_groups"]) != 1
    ):
        raise InferenceContractError(
            f"{resolved}: invalid official Base optimizer envelope"
        )
    group = optimizer["param_groups"][0]
    expected_group_without_params = {
        "lr": 0.00030000000000000003,
        "betas": (0.5, 0.999),
        "eps": 1e-08,
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
    if (
        type(group) is not dict
        or set(group) != set(expected_group_without_params) | {"params"}
        or {
            key: group[key] for key in expected_group_without_params
        }
        != expected_group_without_params
        or type(group["params"]) is not list
        or group["params"] != list(range(1_783))
    ):
        raise InferenceContractError(
            f"{resolved}: invalid official Base optimizer parameter group"
        )
    for parameter_index, parameter_state in optimizer["state"].items():
        if (
            type(parameter_index) is not int
            or not 0 <= parameter_index < 1_783
            or type(parameter_state) is not dict
            or set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}
            or any(
                not torch.is_tensor(parameter_state[key])
                for key in ("step", "exp_avg", "exp_avg_sq")
            )
            or any(
                not bool(torch.isfinite(parameter_state[key]).all().item())
                for key in ("step", "exp_avg", "exp_avg_sq")
            )
        ):
            raise InferenceContractError(
                f"{resolved}: invalid/non-finite official Base optimizer "
                f"state at parameter {parameter_index!r}"
            )
    auxiliary_receipt = {
        "epoch_counter": payload["epoch"],
        "lr_scheduler_sha256": canonical_json_sha256(payload["lrs"]),
        "optimizer_state_entries": len(optimizer["state"]),
        "optimizer_parameter_count": len(group["params"]),
        "optimizer_all_tensors_finite": True,
    }
    return state, resolved, snapshot, observed_sha, auxiliary_receipt


_RELEASED_REPRESENTATION_SCHEMAS: (
    dict[str, dict[str, tuple[Any, tuple[int, ...]]]] | None
) = None


def _expected_released_representation_schemas(
) -> dict[str, dict[str, tuple[Any, tuple[int, ...]]]]:
    global _RELEASED_REPRESENTATION_SCHEMAS
    if _RELEASED_REPRESENTATION_SCHEMAS is None:
        import torch
        from models.motion_representation import VAEConvZero
        from models.rvq import RVQVAE

        schemas: dict[str, dict[str, tuple[Any, tuple[int, ...]]]] = {}
        for name in (*RVQ_DIMS, "global"):
            specification = RELEASED_ALL_SPEAKERS_MODELS[name]
            model_args = SimpleNamespace(
                vae_test_dim=specification["vae_test_dim"],
                vae_layer=specification["vae_layer"],
                vae_length=256,
            )
            try:
                with torch.device("meta"):
                    model = (
                        VAEConvZero(model_args)
                        if name == "global"
                        else RVQVAE(model_args)
                    )
            except Exception as exc:
                raise InferenceContractError(
                    f"cannot construct strict released {name} schema on "
                    "the PyTorch meta device"
                ) from exc
            schemas[name] = {
                key: (value.dtype, tuple(value.shape))
                for key, value in model.state_dict().items()
            }
            del model
        _RELEASED_REPRESENTATION_SCHEMAS = schemas
    return _RELEASED_REPRESENTATION_SCHEMAS


def _validate_released_model_state_schema(
    state: Mapping[str, Any],
    *,
    formal_stage: str,
    path: Path,
) -> None:
    if formal_stage == "base":
        _validate_base_model_state_schema(state, path)
        return
    expected = _expected_released_representation_schemas()[formal_stage]
    if set(state) != set(expected):
        raise InferenceContractError(
            f"{path}: model_state keys are not the exact released "
            f"{formal_stage} schema"
        )
    for key, value in state.items():
        expected_dtype, expected_shape = expected[key]
        if value.dtype != expected_dtype or tuple(value.shape) != expected_shape:
            raise InferenceContractError(
                f"{path}: released model_state tensor schema mismatch "
                f"for {key!r}"
            )


def _strict_load_freeze_eval(
    model: Any,
    state: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise InferenceContractError(
            f"{path}: strict released state load was not exact"
        )
    model.eval()
    model.requires_grad_(False)
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise InferenceContractError(
            f"{path}: released model did not freeze in eval mode"
        )


def _validate_released_prerequisite_lineage_binding(
    *,
    lineage: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the exact builder receipt for the five official weights."""
    stages = set(CHECKPOINT_STAGES) - {"base"}
    records = lineage.get("formal_checkpoints")
    receipt = lineage.get("prerequisite_source_receipt")
    source_receipt = _released_weight_source_receipt()
    source_receipt_sha = canonical_json_sha256(source_receipt)
    if (
        not isinstance(records, dict)
        or set(records) != stages
        or not isinstance(receipt, dict)
        or set(receipt) != RELEASED_IMPORT_RECEIPT_KEYS
        or receipt.get("format")
        != "semtalk_released_all_speakers_import_receipt_v1"
        or receipt.get("status") != "complete"
        or receipt.get("prerequisite_source")
        != RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
        or receipt.get("classification")
        != RELEASED_ALL_SPEAKERS_CLASSIFICATION
        or receipt.get("training_dataset") != "BEAT2"
        or receipt.get("speaker_scope") != "All-Speakers"
        or receipt.get("show_trained") is not False
        or receipt.get("source_receipt") != source_receipt
        or receipt.get("source_receipt_sha256") != source_receipt_sha
        or receipt.get("strict_state_dict_load") is not True
        or receipt.get("all_model_state_tensors_finite") is not True
        or receipt.get("frozen_eval") is not True
    ):
        raise InferenceContractError(
            "Base feature lineage lacks the exact official All-Speakers "
            "prerequisite import receipt"
        )
    receipt_without_sha = dict(receipt)
    receipt_sha = receipt_without_sha.pop("receipt_sha256", None)
    if (
        type(receipt_sha) is not str
        or receipt_sha != canonical_json_sha256(receipt_without_sha)
    ):
        raise InferenceContractError(
            "official All-Speakers import receipt SHA is invalid"
        )
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != stages:
        raise InferenceContractError(
            "official All-Speakers import receipt file cover is invalid"
        )
    normalized_records: dict[str, dict[str, Any]] = {}
    for stage in sorted(stages):
        specification = RELEASED_ALL_SPEAKERS_MODELS[stage]
        expected_path = _resolved_regular_file(
            getattr(args, f"{stage}_checkpoint"),
            f"{stage} official released checkpoint",
        )
        record = records[stage]
        file_record = files[stage]
        if (
            not isinstance(record, dict)
            or set(record) != RELEASED_PREREQUISITE_RECORD_KEYS
            or not isinstance(file_record, dict)
            or set(file_record)
            != {
                "path",
                "filename",
                "sha256",
                "bytes",
                "model_class",
                "model_state_schema_sha256",
            }
            or record.get("path") != str(expected_path)
            or file_record.get("path") != str(expected_path)
            or record.get("filename") != specification["filename"]
            or file_record.get("filename") != specification["filename"]
            or record.get("sha256") != specification["sha256"]
            or file_record.get("sha256") != specification["sha256"]
            or record.get("bytes") != expected_path.stat().st_size
            or file_record.get("bytes") != expected_path.stat().st_size
            or record.get("formal_stage") != stage
            or record.get("prerequisite_source")
            != RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
            or record.get("classification")
            != RELEASED_ALL_SPEAKERS_CLASSIFICATION
            or record.get("training_dataset") != "BEAT2"
            or record.get("speaker_scope") != "All-Speakers"
            or record.get("show_trained") is not False
            or record.get("checkpoint_container_schema") != ["model_state"]
            or record.get("model_class") != specification["model_class"]
            or file_record.get("model_class") != specification["model_class"]
            or type(record.get("model_state_tensors")) is not int
            or record["model_state_tensors"] <= 0
            or record.get("model_state_schema_sha256")
            != file_record.get("model_state_schema_sha256")
            or type(record.get("model_state_schema_sha256")) is not str
            or len(record["model_state_schema_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record["model_state_schema_sha256"]
            )
            or record.get("all_model_state_tensors_finite") is not True
            or record.get("strict_state_dict_load") is not True
            or record.get("frozen_eval") is not True
            or record.get("source_receipt") != source_receipt
            or record.get("source_receipt_sha256") != source_receipt_sha
        ):
            raise InferenceContractError(
                f"{stage}: invalid official All-Speakers prerequisite "
                "lineage record"
            )
        normalized_records[stage] = dict(record)
    expected_files = {
        stage: {
            key: normalized_records[stage][key]
            for key in (
                "path",
                "filename",
                "sha256",
                "bytes",
                "model_class",
                "model_state_schema_sha256",
            )
        }
        for stage in sorted(stages)
    }
    if files != expected_files:
        raise InferenceContractError(
            "official All-Speakers import receipt/files mismatch"
        )
    return dict(receipt), normalized_records


def _validate_base_state_schema_pair(
    candidate_state: Mapping[str, Any],
    candidate_path: Path,
    final_state: Mapping[str, Any],
    final_path: Path,
) -> None:
    candidate = _normalize_data_parallel_state(
        candidate_state,
        candidate_path,
    )
    final = _normalize_data_parallel_state(final_state, final_path)
    if set(candidate) != set(final):
        raise InferenceContractError(
            f"{final_path}: completed Base model_state keys differ from the "
            "selected Base candidate schema"
        )
    for key, candidate_value in candidate.items():
        final_value = final[key]
        if (
            candidate_value.dtype != final_value.dtype
            or tuple(candidate_value.shape) != tuple(final_value.shape)
        ):
            raise InferenceContractError(
                f"{final_path}: completed Base model_state schema mismatch "
                f"for {key!r}"
            )


_BASE_MODEL_STATE_SCHEMA: dict[str, tuple[Any, tuple[int, ...]]] | None = None


def _expected_base_model_state_schema(
) -> dict[str, tuple[Any, tuple[int, ...]]]:
    global _BASE_MODEL_STATE_SCHEMA
    if _BASE_MODEL_STATE_SCHEMA is None:
        import torch
        from models.semtalk import semtalk_base

        try:
            with torch.device("meta"):
                reference = semtalk_base(_model_args())
        except Exception as exc:
            raise InferenceContractError(
                "cannot construct the strict SemTalk Base schema on the "
                "PyTorch meta device"
            ) from exc
        _BASE_MODEL_STATE_SCHEMA = {
            key: (value.dtype, tuple(value.shape))
            for key, value in reference.state_dict().items()
        }
        del reference
    return _BASE_MODEL_STATE_SCHEMA


def _validate_base_model_state_schema(
    state: Mapping[str, Any],
    path: Path,
) -> None:
    normalized = _normalize_data_parallel_state(state, path)
    expected = _expected_base_model_state_schema()
    if set(normalized) != set(expected):
        raise InferenceContractError(
            f"{path}: model_state keys are not the exact SemTalk Base schema"
        )
    for key, value in normalized.items():
        expected_dtype, expected_shape = expected[key]
        if value.dtype != expected_dtype or tuple(value.shape) != expected_shape:
            raise InferenceContractError(
                f"{path}: model_state tensor schema mismatch for {key!r}"
            )


def _validate_model_v2_audit(
    audit: Mapping[str, Any],
    *,
    formal_stage: str,
    config_sha256: str,
    lineage_sha256: str,
    dataset_receipt: Mapping[str, Any],
    expected_dataset_summary_sha256: str,
    expected_data_mdb_sha256: str,
    source_receipt: Mapping[str, Any],
    optimizer_updates: int,
    base_candidate_manifest: Mapping[str, Any] | None,
    status: Mapping[str, Any],
    path: Path,
) -> None:
    expected_audit_keys = set(MODEL_V2_AUDIT_KEYS)
    if formal_stage == "lower":
        expected_audit_keys.add(LOWER_TARGET_BACKEND_RECEIPT_KEY)
    for key in MODEL_V2_OPTIONAL_AUDIT_KEYS:
        if key in status:
            expected_audit_keys.add(key)
    if (
        set(audit) != expected_audit_keys
        or audit.get("format") != "semtalk_show_model_v2"
        or audit.get("formal_stage") != formal_stage
        or audit.get("config_sha256") != config_sha256
        or audit.get("lineage_manifest_sha256") != lineage_sha256
        or audit.get("dataset_summary_sha256")
        != expected_dataset_summary_sha256
        or audit.get("data_mdb_sha256") != expected_data_mdb_sha256
        or audit.get("dataset_receipt_sha256")
        != compact_json_sha256(dataset_receipt)
        or audit.get("smplx_asset_receipt")
        != dataset_receipt.get("smplx_asset")
        or audit.get("source_receipt") != source_receipt
        or audit.get("source_receipt_sha256")
        != compact_json_sha256(source_receipt)
        or _require_exact_int(
            audit.get("optimizer_updates"),
            f"{formal_stage} audit optimizer_updates",
        )
        != optimizer_updates
        or audit.get("base_candidate_manifest")
        != base_candidate_manifest
    ):
        raise InferenceContractError(
            f"{path}: invalid model_v2 formal audit for {formal_stage}"
        )
    bound_training_keys = (
        "smplx_training_pool_mode",
        "distributed_training_receipt",
        "optimizer_runtime_receipt",
        "rvq_rank_state_receipt",
        "initialization_receipt",
        "rvq_ema_prior_receipt",
        "latest_representation_candidate",
        "smplx_training_pool_runtime_evidence",
    )
    if any(audit.get(key) != status.get(key) for key in bound_training_keys):
        raise InferenceContractError(
            f"{path}: model_v2 training binding differs from formal status"
        )
    for key in MODEL_V2_OPTIONAL_AUDIT_KEYS:
        if key in status and audit.get(key) != status.get(key):
            raise InferenceContractError(
                f"{path}: model_v2 optional training binding differs"
            )
    gate = dataset_receipt.get("smplx_training_pool_gate")
    expected_pool_mode = (
        gate.get("mode") if isinstance(gate, Mapping) else "disabled"
    )
    if (
        audit.get("smplx_training_pool_mode") != expected_pool_mode
        or audit.get("smplx_training_pool_gate") != gate
        or audit.get("lower_target_joints_cache")
        != dataset_receipt.get("lower_target_joints_cache")
    ):
        raise InferenceContractError(
            f"{path}: model_v2 dataset accelerator binding mismatch"
        )
    runtime_evidence = audit.get("smplx_training_pool_runtime_evidence")
    if expected_pool_mode == "disabled":
        if runtime_evidence is not None:
            raise InferenceContractError(
                f"{path}: disabled SMPL-X pool has runtime evidence"
            )
    else:
        try:
            prerequisite_contract.verify_named_compact_hash(
                runtime_evidence,
                hash_key="receipt_sha256",
                label=f"{formal_stage} SMPL-X pool runtime evidence",
            )
        except prerequisite_contract.ContractError as exc:
            raise InferenceContractError(str(exc)) from exc
    try:
        if formal_stage == "base":
            distributed = prerequisite_contract.exact_keys(
                prerequisite_contract.verify_named_compact_hash(
                    audit.get("distributed_training_receipt"),
                    hash_key="receipt_sha256",
                    label="Base distributed training receipt",
                ),
                (
                    "format",
                    "formal_stage",
                    "world_size",
                    "local_batch_size",
                    "global_batch_size",
                    "train_samples",
                    "available_train_samples",
                    "consumed_samples_per_epoch",
                    "dropped_samples_per_epoch",
                    "padding_or_duplicate_samples_per_epoch",
                    "updates_per_epoch",
                    "loader_drop_last",
                    "sampler",
                    "rvq_ema",
                    "receipt_sha256",
                ),
                "Base distributed training receipt",
            )
            sampler = distributed.get("sampler")
            sampler_seed = prerequisite_contract.require_exact_int(
                sampler.get("seed") if isinstance(sampler, dict) else None,
                "Base distributed training receipt sampler seed",
            )
            if (
                distributed.get("format")
                != "semtalk_show_representation_ddp_v1"
                or distributed.get("formal_stage") != "base"
                or distributed.get("world_size") != 1
                or distributed.get("local_batch_size") != 64
                or distributed.get("global_batch_size") != 64
                or distributed.get("train_samples") != 127_286
                or distributed.get("available_train_samples") != 127_286
                or distributed.get("consumed_samples_per_epoch") != 127_232
                or distributed.get("dropped_samples_per_epoch") != 54
                or distributed.get("padding_or_duplicate_samples_per_epoch") != 0
                or distributed.get("updates_per_epoch") != 1_988
                or distributed.get("loader_drop_last") is not True
                or sampler_seed < 0
                or sampler
                != {
                    "class": "RandomSampler",
                    "shuffle": True,
                    "seed": sampler_seed,
                    "drop_last": True,
                    "set_epoch": None,
                }
                or distributed.get("rvq_ema") != {"enabled": False}
            ):
                raise prerequisite_contract.ContractError(
                    "Base distributed training receipt mismatch"
                )
            if any(
                audit.get(key) is not None
                for key in (
                    "rvq_rank_state_receipt",
                    "initialization_receipt",
                    "rvq_ema_prior_receipt",
                    "latest_representation_candidate",
                )
            ):
                raise prerequisite_contract.ContractError(
                    "Base model_v2 has representation-only bindings"
                )
        else:
            distributed = prerequisite_contract.validate_distributed_training_receipt(
                audit.get("distributed_training_receipt"),
                stage=formal_stage,
                label=f"{formal_stage} distributed training receipt",
            )
            prerequisite_contract.validate_initialization_receipt(
                audit.get("initialization_receipt"),
                stage=formal_stage,
                label=f"{formal_stage} initialization receipt",
                reprove_path=False,
            )
            prerequisite_contract.validate_rvq_ema_prior_receipt(
                audit.get("rvq_ema_prior_receipt"),
                stage=formal_stage,
                label=f"{formal_stage} RVQ EMA-prior receipt",
            )
            prerequisite_contract.validate_rvq_rank_state_receipt(
                audit.get("rvq_rank_state_receipt"),
                stage=formal_stage,
                distributed=distributed,
                label=f"{formal_stage} RVQ rank-state receipt",
            )
        prerequisite_contract.validate_optimizer_runtime_receipt(
            audit.get("optimizer_runtime_receipt"),
            stage=formal_stage,
            label=f"{formal_stage} optimizer runtime receipt",
            required=True,
        )
    except prerequisite_contract.ContractError as exc:
        raise InferenceContractError(str(exc)) from exc
    _validate_lower_target_backend_binding(
        formal_stage=formal_stage,
        audit=audit,
        dataset_receipt=dataset_receipt,
        status=None,
        path=path,
    )


def _base_candidate_relative_path(epoch: int, optimizer_updates: int) -> str:
    return (
        "base_candidates/"
        f"semtalk_base_candidate_epoch_{epoch:04d}"
        f"_step_{optimizer_updates:09d}.bin"
    )


def _base_candidate_audit(
    *,
    epoch: int,
    optimizer_updates: int,
    config_sha256: str,
    lineage_sha256: str,
    dataset_receipt_sha256: str,
    source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": "semtalk_show_base_candidate_model_v1",
        "formal_stage": "base",
        "smplx_training_pool_mode": "disabled",
        "candidate_epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "smplx_asset_receipt": None,
        "source_receipt": dict(source_receipt),
        "source_receipt_sha256": compact_json_sha256(source_receipt),
    }


def _base_candidate_payload_and_receipt(
    *,
    payload: dict[str, Any],
    audit: Mapping[str, Any],
    resolved: Path,
    observed_sha: str,
    observed_bytes: int,
    expected_training_lineage_sha256: str,
    status_path: Path,
    expected_source_receipt: Mapping[str, Any],
    expected_dataset_summary_sha256: str,
    expected_data_mdb_sha256: str,
    manifest_path: Path | None,
    expected_manifest_sha256: str | None,
    expected_formal_status_sha256: str | None,
    expected_final_checkpoint_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        manifest_path is None
        or expected_manifest_sha256 is None
        or expected_formal_status_sha256 is None
        or expected_final_checkpoint_sha256 is None
    ):
        raise InferenceContractError(
            "Base candidate inference requires the immutable candidate "
            "manifest, expected manifest SHA-256, expected formal-status "
            "SHA-256, and expected completed-final-checkpoint SHA-256"
        )
    if set(payload) != {"model_state", "audit"}:
        raise InferenceContractError(
            f"{resolved}: invalid Base candidate checkpoint envelope"
        )
    _validate_base_model_state_schema(payload["model_state"], resolved)
    expected_training_lineage_sha256 = _require_sha256(
        expected_training_lineage_sha256,
        "Base expected training lineage SHA",
    )
    expected_dataset_summary_sha256 = _require_sha256(
        expected_dataset_summary_sha256,
        "Base expected dataset summary SHA",
    )
    expected_data_mdb_sha256 = _require_sha256(
        expected_data_mdb_sha256,
        "Base expected data.mdb SHA",
    )
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "Base candidate manifest SHA",
    )
    expected_formal_status_sha256 = _require_sha256(
        expected_formal_status_sha256,
        "Base formal training status SHA",
    )
    expected_final_checkpoint_sha256 = _require_sha256(
        expected_final_checkpoint_sha256,
        "Base completed final checkpoint SHA",
    )
    manifest_resolved = _resolved_regular_file(
        manifest_path,
        "Base candidate manifest",
    )
    if manifest_resolved.name != "base_candidate_manifest.json":
        raise InferenceContractError(
            "Base candidate manifest basename must be "
            "base_candidate_manifest.json"
        )
    _verify_file_sha(
        manifest_resolved,
        expected_manifest_sha256,
        "Base candidate manifest",
    )
    manifest = load_json(manifest_resolved)
    resolved_status = _resolved_regular_file(
        status_path,
        "base formal training status",
    )
    observed_status_sha256 = _verify_file_sha(
        resolved_status,
        expected_formal_status_sha256,
        "Base formal training status",
    )
    status = load_json(resolved_status)
    dataset_receipt = status.get("dataset_receipt")
    source_receipt = status.get("source_receipt")
    expected_source_triplet = {
        key: expected_source_receipt[key]
        for key in ("origin", "commit", "tree")
    }
    expected_optimizer_updates = BASE_EPOCHS * FORMAL_UPDATES_PER_EPOCH
    manifest_receipt = {
        "path": manifest_resolved.name,
        "sha256": expected_manifest_sha256,
        "entries": BASE_CANDIDATE_COUNT,
        "last_epoch": BASE_EPOCHS,
        "last_optimizer_updates": expected_optimizer_updates,
    }
    if (
        status.get("status") != "complete"
        or status.get("formal_stage") != "base"
        or _require_exact_int(status.get("world_size"), "world_size") != 1
        or _require_exact_int(status.get("epochs"), "epochs") != BASE_EPOCHS
        or _require_exact_int(
            status.get("completed_epochs"),
            "completed_epochs",
        )
        != BASE_EPOCHS
        or _require_exact_int(
            status.get("train_samples"),
            "train_samples",
        )
        != 127_286
        or _require_exact_int(
            status.get("updates_per_epoch"),
            "updates_per_epoch",
        )
        != FORMAL_UPDATES_PER_EPOCH
        or _require_exact_int(
            status.get("optimizer_updates"),
            "optimizer_updates",
        )
        != expected_optimizer_updates
        or status.get("lineage_manifest_sha256")
        != expected_training_lineage_sha256
        or not isinstance(dataset_receipt, dict)
        or dataset_receipt.get("summary_sha256")
        != expected_dataset_summary_sha256
        or dataset_receipt.get("lineage_sha256")
        != expected_training_lineage_sha256
        or dataset_receipt.get("data_mdb_sha256")
        != expected_data_mdb_sha256
        or _require_exact_int(dataset_receipt.get("entries"), "entries")
        != 127_286
        or _require_exact_int(
            dataset_receipt.get("train_clips"),
            "train_clips",
        )
        != 13_687
        or dataset_receipt.get("smplx_asset") is not None
        or status.get("smplx_asset_receipt") is not None
        or not isinstance(source_receipt, dict)
        or {
            key: source_receipt.get(key)
            for key in ("origin", "commit", "tree")
        }
        != expected_source_triplet
        or status.get("source_receipt_sha256")
        != compact_json_sha256(source_receipt)
        or status.get("base_candidate_manifest") != manifest_receipt
        or status.get("final_checkpoint_sha256")
        != expected_final_checkpoint_sha256
    ):
        raise InferenceContractError(
            f"{resolved_status}: invalid complete Base candidate training receipt"
        )
    config_sha256 = _require_sha256(
        str(status.get("config_sha256", "")),
        "Base config SHA",
    )
    if status.get("config_sha256") != config_sha256:
        raise InferenceContractError(
            f"{resolved_status}: Base config SHA is not canonical lowercase"
        )
    dataset_receipt_sha256 = compact_json_sha256(dataset_receipt)

    if (
        set(manifest) != BASE_CANDIDATE_MANIFEST_KEYS
        or manifest.get("format")
        != "semtalk_show_base_candidate_manifest_v2"
        or manifest.get("formal_stage") != "base"
        or _require_exact_int(
            manifest.get("interval_epochs"),
            "candidate interval_epochs",
        )
        != BASE_CANDIDATE_INTERVAL_EPOCHS
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("lineage_manifest_sha256")
        != expected_training_lineage_sha256
        or manifest.get("dataset_receipt_sha256")
        != dataset_receipt_sha256
        or manifest.get("smplx_asset_receipt") is not None
        or manifest.get("source_receipt") != source_receipt
        or manifest.get("source_receipt_sha256")
        != compact_json_sha256(source_receipt)
        or not isinstance(manifest.get("entries"), list)
        or len(manifest["entries"]) != BASE_CANDIDATE_COUNT
    ):
        raise InferenceContractError(
            f"{manifest_resolved}: invalid Base candidate manifest binding"
        )
    if manifest_resolved.parent != resolved_status.parent:
        raise InferenceContractError(
            "Base candidate manifest and training status must share the "
            "immutable checkpoint directory"
        )

    selected_record: Mapping[str, Any] | None = None
    expected_candidate_paths: set[Path] = set()
    candidate_checkpoint_receipts: list[dict[str, str]] = []
    entries = manifest["entries"]
    for index, record in enumerate(entries):
        epoch = (index + 1) * BASE_CANDIDATE_INTERVAL_EPOCHS
        optimizer_updates = epoch * FORMAL_UPDATES_PER_EPOCH
        relative_path = _base_candidate_relative_path(
            epoch,
            optimizer_updates,
        )
        expected_audit = _base_candidate_audit(
            epoch=epoch,
            optimizer_updates=optimizer_updates,
            config_sha256=config_sha256,
            lineage_sha256=expected_training_lineage_sha256,
            dataset_receipt_sha256=dataset_receipt_sha256,
            source_receipt=source_receipt,
        )
        previous_entries = entries[:index]
        previous_manifest_sha256 = (
            training_json_document_sha256(
                {**manifest, "entries": previous_entries}
            )
            if previous_entries
            else None
        )
        expected_core = {
            "format": "semtalk_show_base_candidate_transaction_core_v1",
            "formal_stage": "base",
            "candidate_epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "checkpoint": relative_path,
            "staging_checkpoint": ".base_candidate_checkpoint.staging",
            "previous_manifest_sha256": previous_manifest_sha256,
            "previous_entry_count": index,
            "previous_entries_sha256": compact_json_sha256(
                previous_entries
            ),
            "config_sha256": config_sha256,
            "lineage_manifest_sha256": expected_training_lineage_sha256,
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "smplx_asset_receipt": None,
            "source_receipt": source_receipt,
            "source_receipt_sha256": compact_json_sha256(source_receipt),
            "model_audit_sha256": compact_json_sha256(expected_audit),
        }
        if (
            not isinstance(record, dict)
            or set(record) != BASE_CANDIDATE_RECORD_KEYS
            or _require_exact_int(
                record.get("epoch"),
                "candidate epoch",
            )
            != epoch
            or _require_exact_int(
                record.get("optimizer_updates"),
                "candidate optimizer_updates",
            )
            != optimizer_updates
            or record.get("checkpoint") != relative_path
            or _require_sha256(
                str(record.get("checkpoint_sha256", "")),
                "candidate checkpoint SHA",
            )
            != record.get("checkpoint_sha256")
            or record.get("model_audit_sha256")
            != compact_json_sha256(expected_audit)
            or not isinstance(record.get("transaction_core"), dict)
            or set(record["transaction_core"]) != BASE_CANDIDATE_CORE_KEYS
            or _require_exact_int(
                record["transaction_core"].get("candidate_epoch"),
                "candidate transaction candidate_epoch",
            )
            != epoch
            or _require_exact_int(
                record["transaction_core"].get("optimizer_updates"),
                "candidate transaction optimizer_updates",
            )
            != optimizer_updates
            or _require_exact_int(
                record["transaction_core"].get("previous_entry_count"),
                "candidate transaction previous_entry_count",
            )
            != index
            or record["transaction_core"] != expected_core
            or record.get("transaction_core_sha256")
            != compact_json_sha256(expected_core)
        ):
            raise InferenceContractError(
                f"{manifest_resolved}: invalid candidate record at epoch {epoch}"
            )
        candidate_input = manifest_resolved.parent / relative_path
        if candidate_input.is_symlink() or not candidate_input.is_file():
            raise InferenceContractError(
                f"Base candidate must be a regular non-symlink file: "
                f"{candidate_input}"
            )
        candidate_resolved = candidate_input.resolve()
        candidate_sha256 = (
            observed_sha
            if candidate_resolved == resolved
            else sha256_file(candidate_resolved)
        )
        if candidate_sha256 != record["checkpoint_sha256"]:
            raise InferenceContractError(
                f"{candidate_resolved}: candidate checkpoint SHA differs "
                "from the Base candidate manifest"
            )
        candidate_checkpoint_receipts.append(
            {
                "path": str(candidate_resolved),
                "sha256": candidate_sha256,
            }
        )
        expected_candidate_paths.add(candidate_resolved)
        if candidate_resolved == resolved:
            if selected_record is not None:
                raise InferenceContractError(
                    "selected Base candidate appears more than once"
                )
            selected_record = record

    candidate_dir = manifest_resolved.parent / "base_candidates"
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise InferenceContractError("invalid Base candidate directory")
    actual_candidate_paths = {
        child.resolve()
        for child in candidate_dir.iterdir()
        if child.is_file() and not child.is_symlink()
    }
    if (
        len(list(candidate_dir.iterdir())) != BASE_CANDIDATE_COUNT
        or actual_candidate_paths != expected_candidate_paths
    ):
        raise InferenceContractError(
            "Base candidate directory is not the manifest exact cover"
        )
    if selected_record is None:
        raise InferenceContractError(
            f"{resolved}: selected checkpoint is absent from the "
            "Base candidate manifest"
        )
    if selected_record.get("checkpoint_sha256") != observed_sha:
        raise InferenceContractError(
            f"{resolved}: selected candidate SHA differs from its manifest"
        )
    selected_epoch = _require_exact_int(
        selected_record.get("epoch"),
        "selected candidate epoch",
    )
    selected_updates = _require_exact_int(
        selected_record.get("optimizer_updates"),
        "selected candidate optimizer_updates",
    )
    expected_selected_audit = _base_candidate_audit(
        epoch=selected_epoch,
        optimizer_updates=selected_updates,
        config_sha256=config_sha256,
        lineage_sha256=expected_training_lineage_sha256,
        dataset_receipt_sha256=dataset_receipt_sha256,
        source_receipt=source_receipt,
    )
    if set(audit) != BASE_CANDIDATE_AUDIT_KEYS or audit != expected_selected_audit:
        raise InferenceContractError(
            f"{resolved}: candidate checkpoint audit/manifest mismatch"
        )

    final_path = _resolved_regular_file(
        Path(str(status.get("final_checkpoint", ""))),
        "Base final checkpoint",
    )
    if (
        final_path.parent != resolved_status.parent
        or final_path.name != "semtalk_base_epoch_400.bin"
    ):
        raise InferenceContractError(
            "Base completed final checkpoint is outside the exact formal "
            "checkpoint directory/name"
        )
    final_path, final_snapshot, final_sha = _read_verified_checkpoint_snapshot(
        final_path,
        expected_final_checkpoint_sha256,
        "Base final checkpoint",
    )
    final_payload = _torch_load_checkpoint(final_snapshot, final_path)
    if set(final_payload) != {"model_state", "audit"}:
        raise InferenceContractError(
            f"{final_path}: invalid completed Base checkpoint envelope"
        )
    _finite_state_dict(final_payload["model_state"], final_path)
    _validate_base_model_state_schema(
        final_payload["model_state"],
        final_path,
    )
    _validate_base_state_schema_pair(
        payload["model_state"],
        resolved,
        final_payload["model_state"],
        final_path,
    )
    final_audit = final_payload.get("audit")
    if not isinstance(final_audit, dict):
        raise InferenceContractError(
            f"{final_path}: missing Base final model_v2 audit"
        )
    _validate_model_v2_audit(
        final_audit,
        formal_stage="base",
        config_sha256=config_sha256,
        lineage_sha256=expected_training_lineage_sha256,
        dataset_receipt=dataset_receipt,
        expected_dataset_summary_sha256=expected_dataset_summary_sha256,
        expected_data_mdb_sha256=expected_data_mdb_sha256,
        source_receipt=source_receipt,
        optimizer_updates=expected_optimizer_updates,
        base_candidate_manifest=manifest_receipt,
        status=status,
        path=final_path,
    )
    if (
        final_path == resolved
        or final_sha != expected_final_checkpoint_sha256
        or final_sha != status["final_checkpoint_sha256"]
    ):
        raise InferenceContractError(
            "Base candidate must remain distinct from the completed final model"
        )

    return payload, {
        "path": str(resolved),
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "formal_stage": "base",
        "checkpoint_kind": "immutable_candidate",
        "candidate_epoch": selected_epoch,
        "candidate_optimizer_updates": selected_updates,
        "training_lineage_sha256": expected_training_lineage_sha256,
        "audit": dict(audit),
        "base_candidate_manifest": {
            **manifest_receipt,
            "path": str(manifest_resolved),
        },
        "candidate_checkpoints": candidate_checkpoint_receipts,
        "candidate_transaction_core_sha256": selected_record[
            "transaction_core_sha256"
        ],
        "formal_training_status": str(resolved_status),
        "formal_training_status_sha256": observed_status_sha256,
        "completed_final_checkpoint": {
            "path": str(final_path),
            "sha256": final_sha,
        },
        "training_accounting": {
            "epochs": BASE_EPOCHS,
            "train_samples": 127_286,
            "updates_per_epoch": FORMAL_UPDATES_PER_EPOCH,
            "optimizer_updates": expected_optimizer_updates,
        },
    }


def _require_official_adapt_detached_source(source_root: Path) -> None:
    """Fail closed unless the formal inference checkout is clean and detached."""

    symbolic = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if symbolic.returncode == 0:
        raise InferenceContractError(
            "official_show_adapt_v1 requires a detached source checkout"
        )
    if symbolic.returncode != 1:
        raise InferenceContractError(
            "cannot prove that official_show_adapt_v1 source is detached"
        )


def _reject_withdrawn_adapt_path(value: str | Path, label: str) -> None:
    lowered = str(value).lower()
    if "e30" in lowered or "speaker2" in lowered:
        raise InferenceContractError(
            f"{label} references withdrawn e30 or forbidden Speaker2: {value}"
        )


def _verified_json_object(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    resolved, payload, observed = _parse_verified_json_snapshot(
        Path(path),
        expected_sha256,
        label,
    )
    _reject_withdrawn_adapt_path(resolved, label)
    return resolved, payload, observed


def _validate_adapt_producer_source(
    source: Any,
    *,
    expected_source_receipt: Mapping[str, Any],
    label: str,
    require_detached_receipt: bool,
) -> dict[str, Any]:
    if type(source) is not dict:
        raise InferenceContractError(f"{label}: missing producer source receipt")
    triplet = {
        key: source.get(key) for key in ("origin", "commit", "tree")
    }
    expected_triplet = {
        key: expected_source_receipt.get(key)
        for key in ("origin", "commit", "tree")
    }
    if (
        triplet != expected_triplet
        or source.get("origin") != EXPECTED_ORIGIN
        or source.get("clean") is not True
    ):
        raise InferenceContractError(
            f"{label}: producer source is not the expected clean SemTalk tree"
        )
    if require_detached_receipt and source.get("branch", object()) is not None:
        raise InferenceContractError(
            f"{label}: producer source receipt is not detached"
        )
    entrypoint = source.get("entrypoint")
    entrypoint_sha = source.get("entrypoint_sha256")
    if (
        type(entrypoint) is not str
        or type(entrypoint_sha) is not str
        or not Path(entrypoint).is_absolute()
    ):
        raise InferenceContractError(
            f"{label}: invalid producer entrypoint receipt"
        )
    _verify_file_sha(
        _resolved_regular_file(entrypoint, f"{label} producer entrypoint"),
        entrypoint_sha,
        f"{label} producer entrypoint",
    )
    return dict(source)


def _validate_exact_official_reference(
    receipt: Any,
    *,
    stage: str,
    label: str,
) -> dict[str, Any]:
    specification = RELEASED_ALL_SPEAKERS_MODELS[stage]
    if type(receipt) is not dict:
        raise InferenceContractError(f"{label}: missing official {stage} receipt")
    path_value = receipt.get("path")
    if (
        receipt.get("filename") != specification["filename"]
        or receipt.get("sha256") != specification["sha256"]
        or type(path_value) is not str
    ):
        raise InferenceContractError(
            f"{label}: official {stage} initialization is not hash-pinned"
        )
    resolved = _resolved_regular_file(path_value, f"{label} official {stage}")
    _reject_withdrawn_adapt_path(resolved, f"{label} official {stage}")
    if resolved.name != specification["filename"]:
        raise InferenceContractError(
            f"{label}: official {stage} filename changed"
        )
    _verify_file_sha(
        resolved,
        str(specification["sha256"]),
        f"{label} official {stage}",
    )
    if (
        "official_all_speakers" in receipt
        and receipt.get("official_all_speakers") is not True
    ):
        raise InferenceContractError(
            f"{label}: {stage} is not marked official All-Speakers"
        )
    if (
        "withdrawn_e30_allowed" in receipt
        and receipt.get("withdrawn_e30_allowed") is not False
    ):
        raise InferenceContractError(
            f"{label}: withdrawn e30 is not explicitly forbidden"
        )
    return dict(receipt)


def _validate_official_transfer_metrics(
    *,
    path: Path,
    summary: Mapping[str, Any],
    transfer: Mapping[str, Any],
    stage: str,
) -> None:
    rows = load_jsonl(path)
    epochs = summary.get("epochs")
    if (
        type(epochs) is not int
        or epochs <= 0
        or len(rows) != epochs
        or [row.get("epoch") for row in rows]
        != list(range(1, epochs + 1))
    ):
        raise InferenceContractError(
            f"{path}: incomplete {stage} validation selection history"
        )
    expected_keys = {
        "epoch",
        "train",
        "val",
        "elapsed_seconds",
        "rank_count",
        "finite",
    }
    for row in rows:
        if (
            set(row) != expected_keys
            or row.get("finite") is not True
            or type(row.get("rank_count")) is not int
            or row["rank_count"] <= 0
            or type(row.get("elapsed_seconds")) not in {int, float}
            or isinstance(row.get("elapsed_seconds"), bool)
            or not math.isfinite(float(row["elapsed_seconds"]))
            or float(row["elapsed_seconds"]) < 0.0
            or type(row.get("train")) is not dict
            or type(row.get("val")) is not dict
        ):
            raise InferenceContractError(
                f"{path}: invalid {stage} transfer metric event"
            )
        _validate_finite_measurement_tree(
            row["train"],
            f"{path}: epoch {row['epoch']} train metrics",
        )
        _validate_finite_measurement_tree(
            row["val"],
            f"{path}: epoch {row['epoch']} val metrics",
        )
        total = row["val"].get("total")
        if (
            type(total) not in {int, float}
            or isinstance(total, bool)
            or not math.isfinite(float(total))
        ):
            raise InferenceContractError(
                f"{path}: epoch {row['epoch']} lacks finite val total"
            )
    best = min(rows, key=lambda row: (float(row["val"]["total"]), row["epoch"]))
    if (
        summary.get("best_epoch") != best["epoch"]
        or summary.get("best_validation_total") != best["val"]["total"]
        or transfer.get("validation") != best["val"]
    ):
        raise InferenceContractError(
            f"{path}: {stage} best checkpoint is not the val minimum"
        )


def _validate_official_adapt_base(
    *,
    path: Path,
    expected_sha256: str,
    expected_source_receipt: Mapping[str, Any],
    status_path: Path | None,
    manifest_path: Path | None,
    expected_manifest_sha256: str | None,
    expected_status_sha256: str | None,
    frozen_inputs_path: Path | None,
    expected_frozen_inputs_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if any(
        value is None
        for value in (
            status_path,
            manifest_path,
            expected_manifest_sha256,
            expected_status_sha256,
            frozen_inputs_path,
            expected_frozen_inputs_sha256,
        )
    ):
        raise InferenceContractError(
            "official_show_adapt_v1 Base requires status, candidate manifest, "
            "and frozen-input receipt with external SHA roots"
        )
    resolved, snapshot, observed_sha = _read_verified_checkpoint_snapshot(
        path,
        expected_sha256,
        "official-adapt Base candidate",
    )
    _reject_withdrawn_adapt_path(resolved, "official-adapt Base candidate")
    payload = _torch_load_checkpoint(snapshot, resolved)
    if set(payload) != {"model_state", "audit"}:
        raise InferenceContractError(
            f"{resolved}: invalid official-adapt Base checkpoint envelope"
        )
    _finite_state_dict(payload["model_state"], resolved)
    _validate_base_model_state_schema(payload["model_state"], resolved)

    frozen_resolved, frozen, frozen_file_sha = _verified_json_object(
        frozen_inputs_path,
        str(expected_frozen_inputs_sha256),
        "official-adapt Base frozen inputs",
    )
    frozen_receipt_sha = frozen.get("receipt_sha256")
    frozen_without_sha = dict(frozen)
    frozen_without_sha.pop("receipt_sha256", None)
    if (
        frozen.get("format") != OFFICIAL_SHOW_ADAPT_BASE_FROZEN_FORMAT
        or type(frozen_receipt_sha) is not str
        or official_base_adapt_json_sha256(frozen_without_sha)
        != frozen_receipt_sha
    ):
        raise InferenceContractError(
            f"{frozen_resolved}: invalid frozen-input receipt"
        )
    producer_source = _validate_adapt_producer_source(
        frozen.get("source"),
        expected_source_receipt=expected_source_receipt,
        label="official-adapt Base",
        require_detached_receipt=True,
    )
    official_base = _validate_exact_official_reference(
        frozen.get("official_base"),
        stage="base",
        label="official-adapt Base",
    )
    dataset = frozen.get("dataset")
    protocol = frozen.get("protocol")
    if (
        type(dataset) is not dict
        or dataset.get("format")
        != "semtalk_show_base_official_feature_dataset_receipt_v1"
        or dataset.get("entries") != 127_286
        or dataset.get("train_clips") != 13_687
        or dataset.get("prerequisite_source")
        != RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
        or dataset.get("vq_models_in_training_graph") is not False
        or type(protocol) is not dict
        or protocol.get("format")
        != "semtalk_show_base_official_adapt_protocol_v1"
        or protocol.get("target_dataset") != "SHOW"
        or protocol.get("target_speaker_scope") != "All"
        or protocol.get("target_speakers") != SHOW_SPEAKER_IDS
        or protocol.get("candidate_epochs")
        != list(OFFICIAL_SHOW_ADAPT_BASE_CANDIDATE_EPOCHS)
        or protocol.get("epochs") != 40
        or protocol.get("expected_updates_per_epoch")
        != OFFICIAL_SHOW_ADAPT_BASE_UPDATES_PER_EPOCH
        or protocol.get("vq_models_in_training_graph") is not False
        or protocol.get("initialization", {}).get("sha256")
        != RELEASED_ALL_SPEAKERS_MODELS["base"]["sha256"]
        or protocol.get("initialization", {}).get("speaker_scope")
        != "All-Speakers"
        or protocol.get("forward_contract", {}).get(
            "forwards_per_optimizer_step"
        )
        != 1
        or protocol.get("forward_contract", {}).get(
            "audio_conditioned_main_forward"
        )
        is not True
    ):
        raise InferenceContractError(
            f"{frozen_resolved}: Base adaptation protocol leaks test data or "
            "does not preserve the official initialization contract"
        )

    manifest_resolved, manifest, manifest_file_sha = _verified_json_object(
        manifest_path,
        str(expected_manifest_sha256),
        "official-adapt Base candidate manifest",
    )
    entries = manifest.get("entries")
    if (
        manifest.get("format") != OFFICIAL_SHOW_ADAPT_BASE_MANIFEST_FORMAT
        or manifest.get("status") != "complete"
        or manifest.get("candidate_epochs")
        != list(OFFICIAL_SHOW_ADAPT_BASE_CANDIDATE_EPOCHS)
        or manifest.get("completed_epochs") != 40
        or manifest.get("optimizer_updates")
        != 40 * OFFICIAL_SHOW_ADAPT_BASE_UPDATES_PER_EPOCH
        or manifest.get("frozen_receipt_sha256") != frozen_receipt_sha
        or type(entries) is not list
        or [entry.get("epoch") for entry in entries]
        != list(OFFICIAL_SHOW_ADAPT_BASE_CANDIDATE_EPOCHS)
        or manifest.get("entries_sha256")
        != official_base_adapt_json_sha256(entries)
    ):
        raise InferenceContractError(
            f"{manifest_resolved}: incomplete official-adapt Base manifest"
        )
    selected: dict[str, Any] | None = None
    for entry in entries:
        if type(entry) is not dict:
            raise InferenceContractError(
                f"{manifest_resolved}: invalid candidate entry"
            )
        candidate_path = _resolved_regular_file(
            manifest_resolved.parent / str(entry.get("checkpoint", "")),
            "official-adapt Base candidate entry",
        )
        _reject_withdrawn_adapt_path(
            candidate_path,
            "official-adapt Base candidate entry",
        )
        if (
            entry.get("frozen_receipt_sha256") != frozen_receipt_sha
            or entry.get("checkpoint_container_schema")
            != ["audit", "model_state"]
            or entry.get("all_model_state_tensors_finite") is not True
        ):
            raise InferenceContractError(
                f"{manifest_resolved}: invalid candidate trust receipt"
            )
        candidate_sha = _verify_file_sha(
            candidate_path,
            str(entry.get("checkpoint_sha256", "")),
            "official-adapt Base candidate entry",
        )
        if candidate_path == resolved:
            selected = dict(entry)
            if candidate_sha != observed_sha:
                raise InferenceContractError(
                    "selected Base candidate SHA differs from its manifest"
                )
    if selected is None:
        raise InferenceContractError(
            f"{resolved}: Base candidate is not in the complete manifest"
        )
    audit = payload.get("audit")
    if (
        type(audit) is not dict
        or audit.get("format") != OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT
        or audit.get("completed_epochs") != selected.get("epoch")
        or audit.get("optimizer_updates")
        != selected.get("optimizer_updates")
        or audit.get("frozen_receipt_sha256") != frozen_receipt_sha
        or audit.get("official_base_checkpoint_sha256")
        != RELEASED_ALL_SPEAKERS_MODELS["base"]["sha256"]
        or audit.get("speaker_scope") != "SHOW_All"
        or audit.get("speaker_rows") != [0, 1, 2, 3]
        or audit.get("vq_models_in_training_graph") is not False
        or audit.get("all_model_state_tensors_finite") is not True
    ):
        raise InferenceContractError(
            f"{resolved}: Base candidate audit is not frozen-input bound"
        )

    status_resolved, status, status_file_sha = _verified_json_object(
        status_path,
        str(expected_status_sha256),
        "official-adapt Base status",
    )
    if (
        status.get("format") != OFFICIAL_SHOW_ADAPT_BASE_STATUS_FORMAT
        or status.get("status") != "complete"
        or status.get("completed_epochs") != 40
        or status.get("optimizer_updates")
        != 40 * OFFICIAL_SHOW_ADAPT_BASE_UPDATES_PER_EPOCH
        or status.get("updates_per_epoch")
        != OFFICIAL_SHOW_ADAPT_BASE_UPDATES_PER_EPOCH
        or status.get("candidate_manifest_sha256") != manifest_file_sha
        or status.get("frozen_receipt_sha256") != frozen_receipt_sha
        or status.get("world_size") != 8
        or status.get("global_batch_size") != 512
        or status.get("all_training_state_finite") is not True
    ):
        raise InferenceContractError(
            f"{status_resolved}: Base official adaptation did not finalize"
        )
    return payload, {
        "path": str(resolved),
        "sha256": observed_sha,
        "bytes": len(snapshot),
        "formal_stage": "base",
        "checkpoint_source": OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
        "candidate_epoch": selected["epoch"],
        "candidate_optimizer_updates": selected["optimizer_updates"],
        "audit": dict(audit),
        "official_initialization": official_base,
        "producer_source": producer_source,
        "frozen_inputs": {
            "path": str(frozen_resolved),
            "sha256": frozen_file_sha,
            "receipt_sha256": frozen_receipt_sha,
        },
        "candidate_manifest": {
            "path": str(manifest_resolved),
            "sha256": manifest_file_sha,
            "entries_sha256": manifest["entries_sha256"],
        },
        "formal_training_status": str(status_resolved),
        "formal_training_status_sha256": status_file_sha,
        "all_model_state_tensors_finite": True,
        "strict_state_dict_load": True,
        "frozen_eval": True,
    }


def _validate_official_transfer_checkpoint(
    *,
    path: Path,
    formal_stage: str,
    expected_sha256: str,
    status_path: Path | None,
    expected_status_sha256: str | None,
    expected_source_receipt: Mapping[str, Any],
    expected_canonical_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if formal_stage not in {"face", "global"}:
        raise InferenceContractError(
            "official transfer validation is restricted to Face and Global"
        )
    if status_path is None or expected_status_sha256 is None:
        raise InferenceContractError(
            f"{formal_stage}: official transfer requires a hash-pinned summary"
        )
    expected_name = f"best_{formal_stage}_transfer.bin"
    if Path(path).name != expected_name:
        raise InferenceContractError(
            f"{formal_stage}: adapted checkpoint must be {expected_name}"
        )
    resolved, snapshot, observed_sha = _read_verified_checkpoint_snapshot(
        path,
        expected_sha256,
        f"official-adapt {formal_stage} best checkpoint",
    )
    _reject_withdrawn_adapt_path(
        resolved,
        f"official-adapt {formal_stage} best checkpoint",
    )
    payload = _torch_load_checkpoint(snapshot, resolved)
    if set(payload) != {
        "model_state",
        "optimizer_state",
        "epoch",
        "transfer_receipt",
    }:
        raise InferenceContractError(
            f"{resolved}: invalid official transfer envelope"
        )
    state = _normalize_data_parallel_state(payload["model_state"], resolved)
    _finite_state_dict(state, resolved)
    _validate_released_model_state_schema(
        state,
        formal_stage=formal_stage,
        path=resolved,
    )
    model_state_sha = official_transfer_state_dict_sha256(state)
    frozen_state = {
        key: value
        for key, value in state.items()
        if (
            key.startswith("encoder.")
            or (
                formal_stage == "face"
                and key.startswith("quantizer.")
            )
        )
    }
    if not frozen_state:
        raise InferenceContractError(
            f"{resolved}: transfer checkpoint lacks frozen encoder state"
        )
    frozen_state_sha = official_transfer_state_dict_sha256(frozen_state)
    transfer = payload.get("transfer_receipt")
    if type(transfer) is not dict:
        raise InferenceContractError(
            f"{resolved}: missing transfer receipt"
        )
    transfer_sha = transfer.get("receipt_sha256")
    transfer_without_sha = dict(transfer)
    transfer_without_sha.pop("receipt_sha256", None)
    if (
        transfer.get("format") != OFFICIAL_SHOW_ADAPT_TRANSFER_FORMAT
        or transfer.get("stage") != formal_stage
        or transfer.get("epoch") != payload.get("epoch")
        or transfer.get("test_visible") is not False
        or transfer.get("withdrawn_e30_allowed") is not False
        or transfer.get("model_state_sha256") != model_state_sha
        or transfer.get("frozen_encoder_quantizer_sha256")
        != frozen_state_sha
        or type(transfer_sha) is not str
        or canonical_json_sha256(transfer_without_sha) != transfer_sha
    ):
        raise InferenceContractError(
            f"{resolved}: transfer receipt is incomplete or test-leaking"
        )
    producer_source = _validate_adapt_producer_source(
        transfer.get("source_receipt"),
        expected_source_receipt=expected_source_receipt,
        label=f"official-adapt {formal_stage}",
        require_detached_receipt=True,
    )
    official = transfer.get("official_initialization")
    if type(official) is not dict:
        raise InferenceContractError(
            f"{resolved}: missing official initialization"
        )
    required_official = (
        {"face": "face"}
        if formal_stage == "face"
        else {"global": "global", "lower": "lower"}
    )
    official_records = {
        key: _validate_exact_official_reference(
            official.get(key),
            stage=stage,
            label=f"official-adapt {formal_stage}",
        )
        for key, stage in required_official.items()
    }
    if set(official) != set(required_official):
        raise InferenceContractError(
            f"{resolved}: unexpected official initialization stages"
        )
    cache = transfer.get("cache_receipt")
    protocol = transfer.get("protocol")
    policy = transfer.get("trainable_policy")
    if (
        type(cache) is not dict
        or cache.get("format")
        != "semtalk_show_official_transfer_cache_v1"
        or cache.get("stage") != formal_stage
        or cache.get("canonical_receipt")
        != dict(expected_canonical_receipt)
        or type(cache.get("roots")) is not list
        or any(
            "e30" in str(root).lower() or "speaker2" in str(root).lower()
            for root in cache.get("roots", [])
        )
        or type(protocol) is not dict
        or protocol.get("stage") != formal_stage
        or protocol.get("selection_split") != "val"
        or protocol.get("test_visible") is not False
        or type(policy) is not dict
    ):
        raise InferenceContractError(
            f"{resolved}: transfer cache/protocol is not train+val only"
        )
    trainable_names = policy.get("trainable_parameter_names")
    if (
        type(trainable_names) is not list
        or not trainable_names
        or any(
            type(name) is not str or not name.startswith("decoder.")
            for name in trainable_names
        )
        or policy.get("frozen_encoder") is not True
    ):
        raise InferenceContractError(
            f"{resolved}: {formal_stage} is not a frozen-encoder decoder transfer"
        )
    if formal_stage == "face":
        if (
            policy.get("frozen_quantizer") is not True
            or policy.get("frozen_rvq_ema") is not True
            or protocol.get("trainable") != "decoder only"
            or transfer.get("frozen_encoder_quantizer_sha256") is None
        ):
            raise InferenceContractError(
                f"{resolved}: Face encoder/quantizer freeze is not proven"
            )
    else:
        if (
            policy.get("policy") != "decoder_only"
            or policy.get("full_model_fallback_authorized") is not False
            or protocol.get("trainable") != "decoder_only"
            or protocol.get("locked_inference_interface")
            != "Base-predicted Lower code IDs -> frozen official Lower decode"
            or protocol.get("lower_translation_54_57")
            != "exact zero every frame"
            or protocol.get("world_root_in_lower_recurrent_seed") is not False
            or transfer.get("frozen_encoder_quantizer_sha256") is None
        ):
            raise InferenceContractError(
                f"{resolved}: Global decoder-only locked interface is invalid"
            )

    summary_resolved, summary, summary_file_sha = _verified_json_object(
        status_path,
        expected_status_sha256,
        f"official-adapt {formal_stage} summary",
    )
    summary_without_sha = dict(summary)
    summary_receipt_sha = summary_without_sha.pop("receipt_sha256", None)
    metrics_path = _resolved_regular_file(
        summary.get("metrics_jsonl", ""),
        f"official-adapt {formal_stage} validation metrics",
    )
    _reject_withdrawn_adapt_path(
        metrics_path,
        f"official-adapt {formal_stage} validation metrics",
    )
    if (
        summary.get("format") != OFFICIAL_SHOW_ADAPT_TRANSFER_FORMAT
        or summary.get("status") != "complete"
        or summary.get("stage") != formal_stage
        or summary.get("best_epoch") != payload.get("epoch")
        or summary.get("best_validation_total")
        != transfer.get("validation", {}).get("total")
        or summary.get("official_initialization") != official
        or summary.get("source_receipt") != transfer.get("source_receipt")
        or summary.get("cache_receipt") != cache
        or summary.get("trainable_policy") != policy
        or summary.get("protocol") != protocol
        or summary.get("frozen_encoder_quantizer_sha256")
        != transfer.get("frozen_encoder_quantizer_sha256")
        or summary.get("withdrawn_e30_allowed") is not False
        or summary.get("test_visible") is not False
        or type(summary_receipt_sha) is not str
        or canonical_json_sha256(summary_without_sha) != summary_receipt_sha
        or summary.get("metrics_jsonl_sha256")
        != sha256_file(metrics_path)
    ):
        raise InferenceContractError(
            f"{summary_resolved}: best transfer checkpoint/summary mismatch"
        )
    _validate_official_transfer_metrics(
        path=metrics_path,
        summary=summary,
        transfer=transfer,
        stage=formal_stage,
    )
    return {"model_state": state}, {
        "path": str(resolved),
        "sha256": observed_sha,
        "bytes": len(snapshot),
        "formal_stage": formal_stage,
        "checkpoint_source": OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
        "checkpoint_kind": "val_selected_best_transfer",
        "best_epoch": payload["epoch"],
        "transfer_receipt": dict(transfer),
        "producer_source": producer_source,
        "official_initialization": official_records,
        "formal_training_status": str(summary_resolved),
        "formal_training_status_sha256": summary_file_sha,
        "validation_metrics": {
            "path": str(metrics_path),
            "sha256": summary["metrics_jsonl_sha256"],
        },
        "all_model_state_tensors_finite": True,
        "strict_state_dict_load": True,
        "frozen_eval": True,
    }


def _checkpoint_payload_and_receipt(
    path: Path,
    *,
    formal_stage: str,
    expected_sha256: str,
    expected_training_lineage_sha256: str | None,
    status_path: Path | None,
    expected_source_receipt: Mapping[str, Any] | None,
    expected_dataset_summary_sha256: str | None,
    expected_data_mdb_sha256: str | None,
    checkpoint_source: str = SHOW_TRAINED_CHECKPOINT_SOURCE,
    expected_release_record: Mapping[str, Any] | None = None,
    released_cross_domain_gate_receipt_sha256: str | None = None,
    base_candidate_manifest_path: Path | None = None,
    expected_base_candidate_manifest_sha256: str | None = None,
    expected_base_formal_status_sha256: str | None = None,
    expected_base_final_checkpoint_sha256: str | None = None,
    official_adapt_frozen_inputs_path: Path | None = None,
    expected_official_adapt_frozen_inputs_sha256: str | None = None,
    expected_official_adapt_status_sha256: str | None = None,
    expected_official_adapt_canonical_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint_source == OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE:
        if expected_official_adapt_canonical_receipt is None:
            raise InferenceContractError(
                f"{formal_stage}: official_show_adapt_v1 lacks the canonical "
                "receipt"
            )
        if formal_stage == "base":
            if expected_source_receipt is None:
                raise InferenceContractError(
                    "base: official_show_adapt_v1 lacks the Base-training "
                    "producer source receipt"
                )
            if expected_base_final_checkpoint_sha256 is not None:
                raise InferenceContractError(
                    "official-adapt Base has no scratch final-checkpoint root"
                )
            return _validate_official_adapt_base(
                path=path,
                expected_sha256=expected_sha256,
                expected_source_receipt=expected_source_receipt,
                status_path=status_path,
                manifest_path=base_candidate_manifest_path,
                expected_manifest_sha256=(
                    expected_base_candidate_manifest_sha256
                ),
                expected_status_sha256=(
                    expected_official_adapt_status_sha256
                ),
                frozen_inputs_path=official_adapt_frozen_inputs_path,
                expected_frozen_inputs_sha256=(
                    expected_official_adapt_frozen_inputs_sha256
                ),
            )
        if formal_stage in {"face", "global"}:
            if expected_source_receipt is None:
                raise InferenceContractError(
                    f"{formal_stage}: official_show_adapt_v1 lacks the "
                    "transfer-training producer source receipt"
                )
            return _validate_official_transfer_checkpoint(
                path=path,
                formal_stage=formal_stage,
                expected_sha256=expected_sha256,
                status_path=status_path,
                expected_status_sha256=(
                    expected_official_adapt_status_sha256
                ),
                expected_source_receipt=expected_source_receipt,
                expected_canonical_receipt=(
                    expected_official_adapt_canonical_receipt
                ),
            )
        if formal_stage not in {"hands", "upper", "lower"}:
            raise InferenceContractError(
                f"unsupported official-adapt stage {formal_stage!r}"
            )
        if any(
            value is not None
            for value in (
                status_path,
                expected_source_receipt,
                base_candidate_manifest_path,
                expected_base_candidate_manifest_sha256,
                expected_base_formal_status_sha256,
                expected_base_final_checkpoint_sha256,
                official_adapt_frozen_inputs_path,
                expected_official_adapt_frozen_inputs_sha256,
                expected_official_adapt_status_sha256,
            )
        ):
            raise InferenceContractError(
                f"{formal_stage}: frozen official prerequisite forbids "
                "adaptation receipts"
            )
        specification = RELEASED_ALL_SPEAKERS_MODELS[formal_stage]
        if expected_sha256 != specification["sha256"]:
            raise InferenceContractError(
                f"{formal_stage}: expected SHA is not exact official "
                "All-Speakers"
            )
        state, resolved, snapshot, observed_sha = (
            _load_released_model_state_only(
                path,
                expected_filename=str(specification["filename"]),
                expected_sha256=str(specification["sha256"]),
            )
        )
        _validate_released_model_state_schema(
            state,
            formal_stage=formal_stage,
            path=resolved,
        )
        return {"model_state": state}, {
            "path": str(resolved),
            "filename": specification["filename"],
            "sha256": observed_sha,
            "bytes": len(snapshot),
            "formal_stage": formal_stage,
            "checkpoint_source": OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
            "classification": RELEASED_ALL_SPEAKERS_CLASSIFICATION,
            "training_dataset": "BEAT2",
            "speaker_scope": "All-Speakers",
            "show_trained": False,
            "official_all_speakers": True,
            "withdrawn_e30_allowed": False,
            "strict_state_dict_load": True,
            "frozen_eval": True,
            "all_model_state_tensors_finite": True,
        }
    if checkpoint_source == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE:
        if any(
            value is not None
            for value in (
                status_path,
                base_candidate_manifest_path,
                expected_base_candidate_manifest_sha256,
                expected_base_formal_status_sha256,
                expected_base_final_checkpoint_sha256,
            )
        ):
            raise InferenceContractError(
                "released_all_speakers_v1 forbids SHOW training status and "
                "Base candidate trust-root receipts"
            )
        specification = RELEASED_ALL_SPEAKERS_MODELS[formal_stage]
        pinned_sha = str(specification["sha256"])
        if expected_sha256 != pinned_sha:
            raise InferenceContractError(
                f"{formal_stage}: released expected SHA-256 "
                f"{expected_sha256} != hard-pinned {pinned_sha}"
            )
        auxiliary_receipt: dict[str, Any] | None = None
        if formal_stage == "base":
            (
                state,
                resolved,
                checkpoint_snapshot,
                observed_sha,
                auxiliary_receipt,
            ) = _load_released_base_state(
                path,
                expected_filename=str(specification["filename"]),
                expected_sha256=pinned_sha,
            )
        else:
            state, resolved, checkpoint_snapshot, observed_sha = (
                _load_released_model_state_only(
                    path,
                    expected_filename=str(specification["filename"]),
                    expected_sha256=pinned_sha,
                )
            )
        _validate_released_model_state_schema(
            state,
            formal_stage=formal_stage,
            path=resolved,
        )
        classification = (
            RELEASED_ALL_SPEAKERS_BASE_CLASSIFICATION
            if formal_stage == "base"
            else RELEASED_ALL_SPEAKERS_CLASSIFICATION
        )
        release_source = _released_weight_source_receipt(
            classification=classification,
            checkpoint_container_schema=(
                RELEASED_BASE_CONTAINER_SCHEMA
                if formal_stage == "base"
                else ("model_state",)
            ),
        )
        record: dict[str, Any] = {
            "path": str(resolved),
            "filename": str(specification["filename"]),
            "sha256": observed_sha,
            "bytes": len(checkpoint_snapshot),
            "formal_stage": formal_stage,
            (
                "base_checkpoint_source"
                if formal_stage == "base"
                else "prerequisite_source"
            ): RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
            "classification": classification,
            "training_dataset": "BEAT2",
            "speaker_scope": "All-Speakers",
            "show_trained": False,
            "checkpoint_container_schema": (
                list(RELEASED_BASE_CONTAINER_SCHEMA)
                if formal_stage == "base"
                else ["model_state"]
            ),
            "model_class": str(specification["model_class"]),
            "model_state_tensors": len(state),
            "model_state_schema_sha256": _state_dict_schema_sha256(state),
            "all_model_state_tensors_finite": True,
            "strict_state_dict_load": True,
            "frozen_eval": True,
            "source_receipt": release_source,
            "source_receipt_sha256": canonical_json_sha256(release_source),
        }
        if auxiliary_receipt is not None:
            record["release_auxiliary_state"] = auxiliary_receipt
        if released_cross_domain_gate_receipt_sha256 is not None:
            record["released_cross_domain_gate_receipt_sha256"] = (
                _require_sha256(
                    released_cross_domain_gate_receipt_sha256,
                    f"{formal_stage} released cross-domain gate receipt",
                )
            )
        if formal_stage != "base":
            if (
                expected_release_record is None
                and released_cross_domain_gate_receipt_sha256 is None
            ):
                raise InferenceContractError(
                    f"{formal_stage}: Base feature lineage lacks the exact "
                    "released prerequisite record or cross-domain gate"
                )
            if (
                expected_release_record is not None
                and released_cross_domain_gate_receipt_sha256 is not None
            ):
                raise InferenceContractError(
                    f"{formal_stage}: released checkpoint cannot mix Base "
                    "feature lineage and cross-domain gate authorization"
                )
            if (
                expected_release_record is not None
                and dict(expected_release_record) != record
            ):
                raise InferenceContractError(
                    f"{formal_stage}: released checkpoint does not match the "
                    "Base feature lineage record"
                )
        else:
            if expected_release_record is not None:
                raise InferenceContractError(
                    "official released Base is not a representation "
                    "prerequisite"
                )
            if released_cross_domain_gate_receipt_sha256 is None:
                raise InferenceContractError(
                    "official released Base requires the released "
                    "cross-domain gate"
                )
        return {"model_state": state}, record

    if checkpoint_source != SHOW_TRAINED_CHECKPOINT_SOURCE:
        raise InferenceContractError(
            f"unsupported checkpoint source {checkpoint_source!r}"
        )
    if expected_release_record is not None:
        raise InferenceContractError(
            "show_trained_v1 cannot carry a released checkpoint record"
        )
    if released_cross_domain_gate_receipt_sha256 is not None:
        raise InferenceContractError(
            "show_trained_v1 cannot carry a released cross-domain gate"
        )
    if (
        expected_training_lineage_sha256 is None
        or status_path is None
        or expected_source_receipt is None
        or expected_dataset_summary_sha256 is None
        or expected_data_mdb_sha256 is None
    ):
        raise InferenceContractError(
            f"{formal_stage}: show_trained_v1 requires complete lineage, "
            "source, dataset, and status receipts"
        )
    resolved, checkpoint_snapshot, observed_sha = (
        _read_verified_checkpoint_snapshot(
            path,
            expected_sha256,
            f"{formal_stage} checkpoint",
        )
    )
    payload = _torch_load_checkpoint(checkpoint_snapshot, resolved)
    _finite_state_dict(payload["model_state"], resolved)
    audit = payload.get("audit")
    if (
        isinstance(audit, dict)
        and audit.get("format")
        == "semtalk_show_base_candidate_model_v1"
    ):
        if formal_stage != "base":
            raise InferenceContractError(
                f"{resolved}: candidate checkpoint is restricted to Base"
            )
        return _base_candidate_payload_and_receipt(
            payload=payload,
            audit=audit,
            resolved=resolved,
            observed_sha=observed_sha,
            observed_bytes=len(checkpoint_snapshot),
            expected_training_lineage_sha256=(
                expected_training_lineage_sha256
            ),
            status_path=status_path,
            expected_source_receipt=expected_source_receipt,
            expected_dataset_summary_sha256=(
                expected_dataset_summary_sha256
            ),
            expected_data_mdb_sha256=expected_data_mdb_sha256,
            manifest_path=base_candidate_manifest_path,
            expected_manifest_sha256=(
                expected_base_candidate_manifest_sha256
            ),
            expected_formal_status_sha256=(
                expected_base_formal_status_sha256
            ),
            expected_final_checkpoint_sha256=(
                expected_base_final_checkpoint_sha256
            ),
        )
    if formal_stage == "base":
        raise InferenceContractError(
            "formal Base inference must select one immutable checkpoint from "
            "the complete 40-candidate manifest"
        )
    if (
        base_candidate_manifest_path is not None
        or expected_base_candidate_manifest_sha256 is not None
        or expected_base_formal_status_sha256 is not None
        or expected_base_final_checkpoint_sha256 is not None
    ):
        raise InferenceContractError(
            "Base candidate trust-root arguments are only valid for Base"
        )
    if (
        not isinstance(audit, dict)
        or audit.get("format") != "semtalk_show_model_v2"
        or audit.get("formal_stage") != formal_stage
    ):
        raise InferenceContractError(
            f"{resolved}: formal checkpoint must use semtalk_show_model_v2 "
            f"for {formal_stage}"
        )
    config_sha = audit.get("config_sha256")
    lineage_sha = audit.get("lineage_manifest_sha256")
    _require_sha256(str(config_sha), f"{formal_stage} config SHA")
    expected_training_lineage_sha256 = _require_sha256(
        expected_training_lineage_sha256,
        f"{formal_stage} expected training lineage SHA",
    )
    if lineage_sha != expected_training_lineage_sha256:
        raise InferenceContractError(
            f"{resolved}: training lineage {lineage_sha!r} != "
            f"{expected_training_lineage_sha256!r}"
        )
    expected_dataset_summary_sha256 = _require_sha256(
        expected_dataset_summary_sha256,
        f"{formal_stage} expected dataset summary SHA",
    )
    expected_data_mdb_sha256 = _require_sha256(
        expected_data_mdb_sha256,
        f"{formal_stage} expected data.mdb SHA",
    )
    audit_source = audit.get("source_receipt")
    expected_source_triplet = {
        key: expected_source_receipt[key]
        for key in ("origin", "commit", "tree")
    }
    if (
        not isinstance(audit_source, dict)
        or {
            key: audit_source.get(key)
            for key in ("origin", "commit", "tree")
        }
        != expected_source_triplet
        or audit.get("source_receipt_sha256")
        != compact_json_sha256(audit_source)
        or audit.get("dataset_summary_sha256")
        != expected_dataset_summary_sha256
        or audit.get("data_mdb_sha256") != expected_data_mdb_sha256
    ):
        raise InferenceContractError(
            f"{resolved}: checkpoint source/dataset receipt mismatch"
        )

    resolved_status = _resolved_regular_file(
        status_path,
        f"{formal_stage} formal training status",
    )
    status = load_json(resolved_status)
    expected_epochs = {
        "base": 400,
        "face": 600,
        "hands": 500,
        "upper": 500,
        "lower": 600,
        "global": 1700,
    }[formal_stage]
    expected_updates_per_epoch = (
        497 if formal_stage in prerequisite_contract.RVQ_STAGES else 1_988
    )
    distributed_receipt = status.get("distributed_training_receipt")
    expected_world_size = (
        distributed_receipt.get("world_size")
        if isinstance(distributed_receipt, dict)
        else None
    )
    dataset_receipt = status.get("dataset_receipt")
    status_final_path = _resolved_regular_file(
        Path(str(status.get("final_checkpoint", ""))),
        f"{formal_stage} status final checkpoint",
    )
    if (
        status.get("status") != "complete"
        or status.get("formal_stage") != formal_stage
        or _require_exact_int(status.get("world_size"), "world_size")
        != expected_world_size
        or _require_exact_int(status.get("epochs"), "epochs")
        != expected_epochs
        or _require_exact_int(
            status.get("completed_epochs"),
            "completed_epochs",
        )
        != expected_epochs
        or _require_exact_int(status.get("train_samples"), "train_samples")
        != 127_286
        or _require_exact_int(
            status.get("updates_per_epoch"),
            "updates_per_epoch",
        )
        != expected_updates_per_epoch
        or _require_exact_int(
            status.get("optimizer_updates"),
            "optimizer_updates",
        )
        != expected_epochs * expected_updates_per_epoch
        or status.get("lineage_manifest_sha256")
        != expected_training_lineage_sha256
        or status.get("config_sha256") != config_sha
        or not isinstance(dataset_receipt, dict)
        or dataset_receipt.get("summary_sha256")
        != expected_dataset_summary_sha256
        or dataset_receipt.get("lineage_sha256")
        != expected_training_lineage_sha256
        or dataset_receipt.get("data_mdb_sha256")
        != expected_data_mdb_sha256
        or _require_exact_int(dataset_receipt.get("entries"), "entries")
        != 127_286
        or _require_exact_int(
            dataset_receipt.get("train_clips"),
            "train_clips",
        )
        != 13_687
        or status.get("source_receipt") != audit_source
        or status.get("source_receipt_sha256")
        != audit.get("source_receipt_sha256")
        or status_final_path != resolved
        or status.get("final_checkpoint_sha256") != observed_sha
    ):
        raise InferenceContractError(
            f"{resolved_status}: incomplete or inconsistent formal "
            f"{formal_stage} training receipt"
        )
    if audit.get("format") == "semtalk_show_model_v2":
        candidate_manifest_receipt = status.get("base_candidate_manifest")
        if formal_stage == "base":
            expected_candidate_receipt = {
                "path": "base_candidate_manifest.json",
                "sha256": (
                    candidate_manifest_receipt.get("sha256")
                    if isinstance(candidate_manifest_receipt, dict)
                    else None
                ),
                "entries": BASE_CANDIDATE_COUNT,
                "last_epoch": BASE_EPOCHS,
                "last_optimizer_updates": (
                    BASE_EPOCHS * FORMAL_UPDATES_PER_EPOCH
                ),
            }
            if candidate_manifest_receipt != expected_candidate_receipt:
                raise InferenceContractError(
                    f"{resolved_status}: invalid Base candidate manifest receipt"
                )
            candidate_manifest_path = _resolved_regular_file(
                resolved_status.parent
                / str(candidate_manifest_receipt["path"]),
                "Base candidate manifest",
            )
            _verify_file_sha(
                candidate_manifest_path,
                str(candidate_manifest_receipt["sha256"]),
                "Base candidate manifest",
            )
        elif candidate_manifest_receipt is not None:
            raise InferenceContractError(
                f"{resolved_status}: non-Base model has a candidate manifest"
            )
        if status.get("smplx_asset_receipt") != dataset_receipt.get(
            "smplx_asset"
        ):
            raise InferenceContractError(
                f"{resolved_status}: model_v2 SMPL-X receipt mismatch"
            )
        _validate_lower_target_backend_binding(
            formal_stage=formal_stage,
            audit=audit,
            dataset_receipt=dataset_receipt,
            status=status,
            path=resolved_status,
        )
        _validate_model_v2_audit(
            audit,
            formal_stage=formal_stage,
            config_sha256=str(config_sha),
            lineage_sha256=expected_training_lineage_sha256,
            dataset_receipt=dataset_receipt,
            expected_dataset_summary_sha256=(
                expected_dataset_summary_sha256
            ),
            expected_data_mdb_sha256=expected_data_mdb_sha256,
            source_receipt=audit_source,
            optimizer_updates=expected_epochs * expected_updates_per_epoch,
            base_candidate_manifest=(
                candidate_manifest_receipt
                if formal_stage == "base"
                else None
            ),
            status=status,
            path=resolved,
        )
    parity = dataset_receipt.get("global_fastpath_parity")
    if formal_stage == "global":
        if (
            not isinstance(parity, dict)
            or parity.get("format")
            != "semtalk_show_global_foot_parity_suite_v1"
            or parity.get("status") != "pass"
            or parity.get("speakers") != SHOW_SPEAKER_IDS
        ):
            raise InferenceContractError(
                f"{resolved_status}: invalid Global fastpath parity receipt"
            )
        parity_path = _resolved_regular_file(
            Path(str(parity.get("path", ""))),
            "Global fastpath parity bundle",
        )
        _verify_file_sha(
            parity_path,
            str(parity.get("sha256", "")),
            "Global fastpath parity bundle",
        )
    elif parity is not None:
        raise InferenceContractError(
            f"{resolved_status}: non-Global stage carries a parity receipt"
        )
    return payload, {
        "path": str(resolved),
        "bytes": len(checkpoint_snapshot),
        "sha256": observed_sha,
        "formal_stage": formal_stage,
        "training_lineage_sha256": expected_training_lineage_sha256,
        "audit": audit,
        "formal_training_status": str(resolved_status),
        "formal_training_status_sha256": sha256_file(resolved_status),
        "training_accounting": {
            "epochs": expected_epochs,
            "train_samples": 127_286,
            "updates_per_epoch": expected_updates_per_epoch,
            "optimizer_updates": expected_epochs * expected_updates_per_epoch,
        },
    }


def _model_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=768,
        audio_f=256,
        motion_f=256,
        pose_dims=330,
        pose_length=WINDOW,
        pre_frames=PRE_FRAMES,
        pose_fps=POSE_FPS,
        vae_codebook_size=CODEBOOK_SIZE,
        vae_test_dim=330,
        vae_layer=4,
        vae_length=240,
    )


def _load_models(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch
    from models.motion_representation import VAEConvZero
    from models.rvq import RVQVAE
    from models.semtalk import semtalk_base

    models: dict[str, Any] = {}
    receipts: dict[str, dict[str, Any]] = {}
    validation = inputs["checkpoint_validation"]
    base_payload, receipts["base"] = _checkpoint_payload_and_receipt(
        args.base_checkpoint,
        formal_stage="base",
        expected_sha256=args.expected_base_sha256,
        **validation["base"],
    )
    base = semtalk_base(_model_args()).to(args.device)
    _strict_load_freeze_eval(
        base,
        _normalize_data_parallel_state(
            base_payload["model_state"],
            Path(receipts["base"]["path"]),
        ),
        path=Path(receipts["base"]["path"]),
    )
    models["base"] = base
    del base_payload

    for name, dimension in RVQ_DIMS.items():
        payload, receipts[name] = _checkpoint_payload_and_receipt(
            getattr(args, f"{name}_checkpoint"),
            formal_stage=name,
            expected_sha256=getattr(args, f"expected_{name}_sha256"),
            **validation[name],
        )
        if args.prerequisite_source in {
            RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
            OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
        }:
            specification = RELEASED_ALL_SPEAKERS_MODELS[name]
            model_arguments = SimpleNamespace(
                vae_test_dim=specification["vae_test_dim"],
                vae_layer=specification["vae_layer"],
                vae_length=256,
            )
        else:
            model_arguments = SimpleNamespace(vae_test_dim=dimension)
        model = RVQVAE(model_arguments).to(args.device)
        _strict_load_freeze_eval(
            model,
            _normalize_data_parallel_state(
                payload["model_state"],
                Path(receipts[name]["path"]),
            ),
            path=Path(receipts[name]["path"]),
        )
        models[name] = model
        del payload

    global_payload, receipts["global"] = _checkpoint_payload_and_receipt(
        args.global_checkpoint,
        formal_stage="global",
        expected_sha256=args.expected_global_sha256,
        **validation["global"],
    )
    global_args = SimpleNamespace(
        vae_test_dim=61,
        vae_layer=4,
        vae_length=256,
    )
    global_motion = VAEConvZero(global_args).to(args.device)
    _strict_load_freeze_eval(
        global_motion,
        _normalize_data_parallel_state(
            global_payload["model_state"],
            Path(receipts["global"]["path"]),
        ),
        path=Path(receipts["global"]["path"]),
    )
    models["global"] = global_motion
    del global_payload

    for name, model in models.items():
        model.eval()
        model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError(f"{name}: frozen-model invariant failed")
    torch.cuda.empty_cache()
    return models, receipts


def _checkpoint_receipts_without_models(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for stage in CHECKPOINT_STAGES:
        payload, receipt = _checkpoint_payload_and_receipt(
            getattr(args, f"{stage}_checkpoint"),
            formal_stage=stage,
            expected_sha256=getattr(args, f"expected_{stage}_sha256"),
            **inputs["checkpoint_validation"][stage],
        )
        receipts[stage] = receipt
        del payload
    return receipts


def _callable_source_sha256(function: Any) -> str:
    try:
        lines, _ = inspect.getsourcelines(function)
    except (OSError, TypeError) as error:
        raise InferenceContractError(
            f"cannot inspect pinned callable {function!r}"
        ) from error
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _inference_auxiliary_loss_bypass_receipt() -> dict[str, Any]:
    model_path = (PROJECT_ROOT / PINNED_SEMTALK_MODEL_SOURCE["relative_path"])
    try:
        model_mode = os.lstat(model_path).st_mode
    except FileNotFoundError:
        raise InferenceContractError(
            f"missing pinned SemTalk model source: {model_path}"
        ) from None
    if stat.S_ISLNK(model_mode) or not stat.S_ISREG(model_mode):
        raise InferenceContractError(
            f"pinned SemTalk model source is not a regular file: {model_path}"
        )
    model_payload = model_path.read_bytes()
    model_blob = hashlib.sha1(
        f"blob {len(model_payload)}\0".encode("ascii") + model_payload
    ).hexdigest()
    if (
        hashlib.sha256(model_payload).hexdigest()
        != PINNED_SEMTALK_MODEL_SOURCE["sha256"]
        or model_blob != PINNED_SEMTALK_MODEL_SOURCE["git_blob_sha1"]
    ):
        raise InferenceContractError("pinned SemTalk model source changed")

    from models.semtalk import RhythmicIdentificationLoss, semtalk_base

    model_resolved = model_path.resolve(strict=True)
    semtalk_source = inspect.getsourcefile(semtalk_base)
    loss_source = inspect.getsourcefile(RhythmicIdentificationLoss)
    if (
        semtalk_base.__module__ != "models.semtalk"
        or RhythmicIdentificationLoss.__module__ != "models.semtalk"
        or not isinstance(semtalk_source, str)
        or not isinstance(loss_source, str)
        or Path(semtalk_source).resolve(strict=True) != model_resolved
        or Path(loss_source).resolve(strict=True) != model_resolved
        or str(inspect.signature(semtalk_base.forward))
        != PINNED_SEMTALK_BASE_FORWARD["signature"]
        or _callable_source_sha256(semtalk_base.forward)
        != PINNED_SEMTALK_BASE_FORWARD["source_sha256"]
        or str(inspect.signature(RhythmicIdentificationLoss.forward))
        != PINNED_RHYTHMIC_LOSS_FORWARD["signature"]
        or _callable_source_sha256(RhythmicIdentificationLoss.forward)
        != PINNED_RHYTHMIC_LOSS_FORWARD["source_sha256"]
        or _callable_source_sha256(_rvq_indices)
        != PINNED_RVQ_INDICES_SOURCE_SHA256
        or str(inspect.signature(_infer_clip))
        != PINNED_INFER_CLIP["signature"]
        or _callable_source_sha256(_infer_clip)
        != PINNED_INFER_CLIP["source_sha256"]
    ):
        raise InferenceContractError(
            "inference-only auxiliary-loss bypass source contract changed"
        )
    receipt = {
        "format": INFERENCE_AUXILIARY_LOSS_BYPASS_FORMAT,
        "scope": "validation_and_test_inference_only",
        "model_source": dict(PINNED_SEMTALK_MODEL_SOURCE),
        "base_forward": dict(PINNED_SEMTALK_BASE_FORWARD),
        "loss_forward": dict(PINNED_RHYTHMIC_LOSS_FORWARD),
        "rvq_indices_source_sha256": PINNED_RVQ_INDICES_SOURCE_SHA256,
        "infer_clip": dict(PINNED_INFER_CLIP),
        "patched_attributes": list(INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS),
        "patch_mechanism": "instance_forward_MethodType_with_finally_restore",
        "replacement": "first_input_new_zeros_scalar",
        "module_identity_preserved": True,
        "state_dict_preserved": True,
        "deterministic_algorithms_required": True,
        "deterministic_warn_only_required": False,
        "model_eval_required": True,
        "all_parameters_frozen_required": True,
        "torch_inference_mode_required": True,
    }
    receipt["receipt_sha256"] = compact_json_sha256(receipt)
    return receipt


@contextmanager
def _inference_only_auxiliary_loss_bypass(
    base: Any,
    *,
    expected_calls: int,
) -> Iterable[dict[str, Any]]:
    import torch
    from models.semtalk import RhythmicIdentificationLoss, semtalk_base

    receipt = _inference_auxiliary_loss_bypass_receipt()
    if (
        type(expected_calls) is not int
        or expected_calls < 1
        or type(base) is not semtalk_base
        or base.training
        or any(parameter.requires_grad for parameter in base.parameters())
        or not torch.is_inference_mode_enabled()
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise InferenceContractError(
            "auxiliary-loss bypass requires frozen eval inference under "
            "strict deterministic algorithms"
        )

    modules: dict[str, Any] = {}
    original_forwards: dict[str, Any] = {}
    patched_forwards: dict[str, Any] = {}
    calls = {name: 0 for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS}
    for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS:
        module = getattr(base, name, None)
        if (
            type(module) is not RhythmicIdentificationLoss
            or module.temperature != 0.1
            or module.training
            or module in modules.values()
            or module is not base._modules.get(name)
            or "forward" in module.__dict__
            or module.state_dict()
            or tuple(module.parameters())
            or tuple(module.buffers())
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
        ):
            raise InferenceContractError(
                f"{name}: auxiliary-loss module contract changed"
            )
        modules[name] = module
        original_forwards[name] = module.forward

    def make_zero_loss(name: str, module: Any) -> Any:
        def zero_loss(
            self: Any,
            facial_features: Any,
            audio_features: Any,
        ) -> Any:
            if (
                self is not module
                or not torch.is_inference_mode_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not isinstance(facial_features, torch.Tensor)
                or not isinstance(audio_features, torch.Tensor)
                or tuple(facial_features.shape) != (1, 16, 256)
                or tuple(audio_features.shape) != (1, 64, 256)
                or facial_features.device != audio_features.device
                or facial_features.dtype != audio_features.dtype
                or facial_features.requires_grad
                or audio_features.requires_grad
            ):
                raise InferenceContractError(
                    f"{name}: invalid inference-only auxiliary-loss call"
                )
            calls[name] += 1
            return facial_features.new_zeros(())

        return MethodType(zero_loss, module)

    try:
        for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS:
            module = modules[name]
            patched = make_zero_loss(name, module)
            patched_forwards[name] = patched
            module.forward = patched
            if module.__dict__.get("forward") is not patched:
                raise InferenceContractError(
                    f"{name}: failed to install auxiliary-loss bypass"
                )
    except BaseException:
        for name in reversed(tuple(patched_forwards)):
            module = modules[name]
            if module.__dict__.get("forward") is patched_forwards[name]:
                delattr(module, "forward")
        raise

    activation = {
        **receipt,
        "expected_calls_per_loss": expected_calls,
        "calls": calls,
    }
    completed = False
    try:
        yield activation
        completed = True
    finally:
        restoration_error: str | None = None
        for name in reversed(INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS):
            module = modules[name]
            installed = module.__dict__.get("forward")
            if installed is not patched_forwards[name]:
                restoration_error = (
                    f"{name}: auxiliary-loss bypass was changed while active"
                )
            if "forward" in module.__dict__:
                delattr(module, "forward")
            restored = module.forward
            original = original_forwards[name]
            if (
                getattr(restored, "__self__", None) is not module
                or getattr(restored, "__func__", None)
                is not getattr(original, "__func__", None)
                or getattr(base, name, None) is not module
                or module is not base._modules.get(name)
                or module.state_dict()
            ):
                restoration_error = (
                    f"{name}: auxiliary-loss module restoration failed"
                )
        if restoration_error is not None:
            raise InferenceContractError(restoration_error)
    if completed and any(
        calls[name] != expected_calls
        for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS
    ):
        raise InferenceContractError(
            "auxiliary-loss bypass call count does not match inference rounds"
        )
    if completed:
        activation["activation_sha256"] = compact_json_sha256(activation)


def _runtime_receipt(device: str) -> dict[str, Any]:
    import torch

    device_object = torch.device(device)
    if device_object.type != "cuda":
        raise InferenceContractError("formal inference requires a CUDA device")
    index = device_object.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "device": str(device_object),
        "device_index": int(index),
        "device_properties": {
            "name": properties.name,
            "major": int(properties.major),
            "minor": int(properties.minor),
            "total_memory": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        },
        "window": WINDOW,
        "pre_frames": PRE_FRAMES,
        "stride": STRIDE,
        "rvq_levels": RVQ_LEVELS,
        "rvq_token_frames": RVQ_TOKEN_FRAMES,
        "auxiliary_loss_bypass": _inference_auxiliary_loss_bypass_receipt(),
    }


def _set_deterministic(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _joint_masks(device: Any) -> dict[str, Any]:
    import torch
    from dataloaders.data_tools import joints_list

    source = joints_list["beat_smplx_joints"]
    targets = {
        "upper": joints_list["beat_smplx_upper"],
        "hands": joints_list["beat_smplx_hands"],
        "lower": joints_list["beat_smplx_lower"],
    }
    result: dict[str, Any] = {}
    observed = torch.zeros(POSE_DIM, dtype=torch.int64, device=device)
    expected_counts = {"upper": 13 * 3, "hands": 30 * 3, "lower": 9 * 3}
    for name, target in targets.items():
        mask = np.zeros(POSE_DIM, dtype=bool)
        for joint_name in target:
            width, end = source[joint_name]
            mask[end - width : end] = True
        if int(mask.sum()) != expected_counts[name]:
            raise InferenceContractError(f"{name}: unexpected joint mask width")
        tensor = torch.from_numpy(mask).to(device=device)
        observed += tensor.to(torch.int64)
        result[name] = tensor
    if bool((observed > 1).any().item()):
        raise InferenceContractError("upper/hands/lower joint masks overlap")
    return result


def _aa_to_rotation_6d(axis_angle: Any) -> Any:
    from utils import rotation_conversions as rc

    batch, frames, dimensions = axis_angle.shape
    if dimensions != POSE_DIM:
        raise InferenceContractError(f"axis-angle pose has shape {axis_angle.shape}")
    return rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(axis_angle.reshape(batch, frames, 55, 3))
    ).reshape(batch, frames, 330)


def _scatter_axis_angle(
    values: Any,
    mask: Any,
    *,
    batch: int,
    frames: int,
) -> Any:
    import torch

    result = torch.zeros(
        batch,
        frames,
        POSE_DIM,
        dtype=values.dtype,
        device=values.device,
    )
    result[..., mask] = values.reshape(batch, frames, -1)
    return result


def _decode_body_axis_angle(
    upper: Any,
    lower: Any,
    hands: Any,
    masks: Mapping[str, Any],
) -> tuple[Any, Any]:
    from utils import rotation_conversions as rc

    batch, frames = int(lower.shape[0]), int(lower.shape[1])
    expected = {
        "upper": (batch, frames, 78),
        "lower": (batch, frames, 61),
        "hands": (batch, frames, 180),
    }
    observed = {
        "upper": tuple(upper.shape),
        "lower": tuple(lower.shape),
        "hands": tuple(hands.shape),
    }
    if observed != expected:
        raise InferenceContractError(
            f"decoded RVQ body shapes {observed} != {expected}"
        )
    upper_aa = rc.matrix_to_axis_angle(
        rc.rotation_6d_to_matrix(upper.reshape(batch, frames, 13, 6))
    ).reshape(batch, frames, 39)
    lower_matrix = rc.rotation_6d_to_matrix(
        lower[..., :54].reshape(batch, frames, 9, 6)
    )
    lower_aa = rc.matrix_to_axis_angle(lower_matrix).reshape(batch, frames, 27)
    hands_aa = rc.matrix_to_axis_angle(
        rc.rotation_6d_to_matrix(hands.reshape(batch, frames, 30, 6))
    ).reshape(batch, frames, 90)
    pose = (
        _scatter_axis_angle(
            upper_aa,
            masks["upper"],
            batch=batch,
            frames=frames,
        )
        + _scatter_axis_angle(
            lower_aa,
            masks["lower"],
            batch=batch,
            frames=frames,
        )
        + _scatter_axis_angle(
            hands_aa,
            masks["hands"],
            batch=batch,
            frames=frames,
        )
    )
    lower_projected_6d = rc.matrix_to_rotation_6d(lower_matrix).reshape(
        batch,
        frames,
        54,
    )
    return pose, lower_projected_6d


def _velocity_to_position(velocity: Any, anchor: Any) -> Any:
    import torch

    if velocity.ndim != 3 or velocity.shape[-1] != 1:
        raise InferenceContractError(
            f"velocity must be [B,T,1], got {tuple(velocity.shape)}"
        )
    positions = [anchor.unsqueeze(1)]
    for frame in range(1, int(velocity.shape[1])):
        positions.append(positions[-1] + velocity[:, frame - 1 : frame] / POSE_FPS)
    return torch.cat(positions, dim=1)


def _translation_from_channels(channels: Any, anchor: Any) -> Any:
    import torch

    if channels.ndim != 3 or channels.shape[-1] != 3:
        raise InferenceContractError(
            f"translation channels must be [B,T,3], got {tuple(channels.shape)}"
        )
    x = _velocity_to_position(channels[..., 0:1], anchor[:, 0:1])
    z = _velocity_to_position(channels[..., 2:3], anchor[:, 2:3])
    return torch.cat([x, channels[..., 1:2], z], dim=-1)


def _edge_pad_frames(array: Any, required_frames: int) -> Any:
    import torch

    current = int(array.shape[1])
    if current >= required_frames:
        return array[:, :required_frames]
    if current < 1:
        raise InferenceContractError("cannot edge-pad an empty feature sequence")
    extension = array[:, -1:].expand(
        array.shape[0],
        required_frames - current,
        *array.shape[2:],
    )
    return torch.cat([array, extension], dim=1)


def _rvq_indices(output: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for name in RVQ_DIMS:
        key = f"cls_{name}"
        logits = output.get(key)
        expected = (1, RVQ_TOKENS_PER_WINDOW, CODEBOOK_SIZE, RVQ_LEVELS)
        if logits is None or tuple(logits.shape) != expected:
            shape = None if logits is None else tuple(logits.shape)
            raise InferenceContractError(f"{key} {shape} != {expected}")
        if not bool(logits.isfinite().all().item()):
            raise InferenceContractError(f"{key} contains NaN or Inf")
        indices = logits.argmax(dim=2)
        if (
            tuple(indices.shape)
            != (1, RVQ_TOKENS_PER_WINDOW, RVQ_LEVELS)
            or indices.dtype != torch.int64
        ):
            raise InferenceContractError(
                f"{key} indices have invalid shape/dtype: "
                f"{tuple(indices.shape)}/{indices.dtype}"
            )
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= CODEBOOK_SIZE:
            raise InferenceContractError(f"{key} index outside codebook")
        result[name] = indices
    return result


def _decode_checked(
    model: Any,
    indices: Any,
    *,
    name: str,
    dimension: int,
) -> Any:
    decoded = model.decode(indices)
    expected_frames = int(indices.shape[1]) * RVQ_TOKEN_FRAMES
    expected = (1, expected_frames, dimension)
    if tuple(decoded.shape) != expected:
        raise InferenceContractError(
            f"{name} RVQ decode {tuple(decoded.shape)} != {expected}"
        )
    if not bool(decoded.isfinite().all().item()):
        raise InferenceContractError(f"{name} RVQ decode contains NaN or Inf")
    return decoded


def _infer_clip(
    *,
    pose: np.ndarray,
    trans: np.ndarray,
    beat: np.ndarray,
    hubert: np.ndarray,
    speaker_id: int,
    models: Mapping[str, Any],
    masks: Mapping[str, Any],
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    from utils import rotation_conversions as rc

    frames = int(pose.shape[0])
    if frames < PRE_FRAMES:
        raise InferenceContractError("clip is shorter than the four-frame seed")
    rounds = max(1, math.ceil((frames - PRE_FRAMES) / STRIDE))
    generated_frames = PRE_FRAMES + STRIDE * rounds
    expected_tokens = RVQ_TOKENS_PER_WINDOW + (
        rounds - 1
    ) * (RVQ_TOKENS_PER_WINDOW - 1)
    if expected_tokens * RVQ_TOKEN_FRAMES != generated_frames:
        raise AssertionError("RVQ stitching arithmetic is inconsistent")

    pose_tensor = torch.from_numpy(pose[:PRE_FRAMES]).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    trans_tensor = torch.from_numpy(trans[:PRE_FRAMES]).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    anchor = trans_tensor[:, 0]
    # Contact is intentionally zero: the only GT conditioning contract is the
    # first four pose/translation frames.
    seed = torch.cat(
        [
            _aa_to_rotation_6d(pose_tensor),
            trans_tensor,
            torch.zeros(1, PRE_FRAMES, 4, device=device),
        ],
        dim=-1,
    )
    if tuple(seed.shape) != (1, PRE_FRAMES, 337):
        raise AssertionError("seed shape is not [1,4,337]")

    beat_tensor = _edge_pad_frames(
        torch.from_numpy(beat).to(device=device, dtype=torch.float32).unsqueeze(0),
        generated_frames,
    )
    hubert_tensor = _edge_pad_frames(
        torch.from_numpy(hubert).to(
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0),
        generated_frames,
    )
    in_word = torch.zeros(
        1,
        generated_frames,
        dtype=torch.long,
        device=device,
    )
    if bool(in_word.any().item()):
        raise AssertionError("in_word must be identically zero")

    stitched: dict[str, list[Any]] = {name: [] for name in RVQ_DIMS}
    latent_last = None
    base = models["base"]
    with torch.inference_mode():
        for round_index in range(rounds):
            start = round_index * STRIDE
            motion = torch.zeros(1, WINDOW, 337, device=device)
            motion[:, :PRE_FRAMES] = (
                seed
                if round_index == 0
                else latent_last[:, -PRE_FRAMES:]
            )
            mask = torch.ones_like(motion)
            mask[:, :PRE_FRAMES] = 0.0
            if bool(motion[:, PRE_FRAMES:].any().item()):
                raise AssertionError("future motion payload is not identically zero")
            identifier = torch.full(
                (1, WINDOW, 1),
                speaker_id,
                dtype=torch.long,
                device=device,
            )
            output = base(
                in_audio=beat_tensor[:, start : start + WINDOW],
                in_word=in_word[:, start : start + WINDOW],
                mask=mask,
                in_motion=motion,
                in_id=identifier,
                hubert=hubert_tensor[:, start : start + WINDOW],
                use_attentions=True,
                is_train=False,
            )
            indices = _rvq_indices(output)
            for name, value in indices.items():
                stitched[name].append(
                    value if round_index == 0 else value[:, 1:]
                )

            upper = _decode_checked(
                models["upper"],
                indices["upper"],
                name="upper",
                dimension=RVQ_DIMS["upper"],
            )
            lower = _decode_checked(
                models["lower"],
                indices["lower"],
                name="lower",
                dimension=RVQ_DIMS["lower"],
            )
            hands = _decode_checked(
                models["hands"],
                indices["hands"],
                name="hands",
                dimension=RVQ_DIMS["hands"],
            )
            body_axis_angle, _ = _decode_body_axis_angle(
                upper,
                lower,
                hands,
                masks,
            )
            preliminary_trans = _translation_from_channels(
                lower[..., 54:57],
                anchor,
            )
            latent_last = torch.cat(
                [
                    _aa_to_rotation_6d(body_axis_angle),
                    preliminary_trans,
                    lower[..., 57:61],
                ],
                dim=-1,
            )
            if tuple(latent_last.shape) != (1, WINDOW, 337):
                raise AssertionError("generated recurrent latent has wrong shape")
            if not bool(latent_last.isfinite().all().item()):
                raise InferenceContractError(
                    "generated recurrent latent contains NaN or Inf"
                )

        stitched_indices = {
            name: torch.cat(parts, dim=1)
            for name, parts in stitched.items()
        }
        for name, indices in stitched_indices.items():
            if tuple(indices.shape) != (1, expected_tokens, RVQ_LEVELS):
                raise InferenceContractError(
                    f"{name} stitched indices have shape {tuple(indices.shape)}"
                )
        decoded = {
            name: _decode_checked(
                models[name],
                stitched_indices[name],
                name=name,
                dimension=dimension,
            )
            for name, dimension in RVQ_DIMS.items()
        }
        body_axis_angle, lower_projected_6d = _decode_body_axis_angle(
            decoded["upper"],
            decoded["lower"],
            decoded["hands"],
            masks,
        )
        face = decoded["face"]
        jaw = rc.matrix_to_axis_angle(
            rc.rotation_6d_to_matrix(face[..., :6].reshape(1, generated_frames, 1, 6))
        ).reshape(1, generated_frames, 3)
        body_axis_angle[..., 66:69] = jaw
        body_axis_angle[..., 69:75] = 0.0

        to_global = decoded["lower"].clone()
        to_global[..., :54] = lower_projected_6d
        to_global[..., 54:57] = 0.0
        global_output = models["global"](to_global)
        global_pose = global_output.get("rec_pose")
        if (
            global_pose is None
            or tuple(global_pose.shape) != (1, generated_frames, 61)
            or not bool(global_pose.isfinite().all().item())
        ):
            shape = None if global_pose is None else tuple(global_pose.shape)
            raise InferenceContractError(
                f"global/root VAE output is invalid: {shape}"
            )
        prediction_trans = _translation_from_channels(
            global_pose[..., 54:57],
            anchor,
        )
        prediction_expression = face[..., 6:]

    result = {
        "poses": body_axis_angle[0, :frames].float().cpu().numpy(),
        "expressions": prediction_expression[0, :frames].float().cpu().numpy(),
        "trans": prediction_trans[0, :frames].float().cpu().numpy(),
    }
    expected_shapes = {
        "poses": (frames, POSE_DIM),
        "expressions": (frames, EXPRESSION_DIM),
        "trans": (frames, 3),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(result[name], dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise InferenceContractError(
                f"invalid generated {name}: {array.shape}/{array.dtype}"
            )
        result[name] = array
    if not np.array_equal(
        result["poses"][:, 69:75],
        np.zeros((frames, 6), dtype=np.float32),
    ):
        raise InferenceContractError("unpredicted eye pose dimensions are not zero")
    return result


def _output_arrays(
    *,
    betas: np.ndarray,
    poses: np.ndarray,
    expressions: np.ndarray,
    trans: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "betas": np.asarray(betas, dtype=np.float32).reshape(BETA_DIM),
        "poses": np.asarray(poses, dtype=np.float32),
        "expressions": np.asarray(expressions, dtype=np.float32),
        "trans": np.asarray(trans, dtype=np.float32),
        "model": np.asarray("smplx2020"),
        "gender": np.asarray("neutral"),
        "mocap_frame_rate": np.asarray(POSE_FPS, dtype=np.int64),
    }
    frames = int(arrays["poses"].shape[0])
    expected = {
        "betas": (BETA_DIM,),
        "poses": (frames, POSE_DIM),
        "expressions": (frames, EXPRESSION_DIM),
        "trans": (frames, 3),
        "model": (),
        "gender": (),
        "mocap_frame_rate": (),
    }
    if tuple(arrays) != OUTPUT_FIELDS:
        raise AssertionError("output field order is not canonical")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise InferenceContractError(
                f"output field {name} shape {arrays[name].shape} != {shape}"
            )
        if arrays[name].dtype.kind in "fc" and not np.isfinite(arrays[name]).all():
            raise InferenceContractError(f"output field {name} is non-finite")
    return arrays


def _input_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    fully_released_zero_shot = (
        args.prerequisite_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
        and args.base_checkpoint_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    )
    official_show_adapt = (
        args.prerequisite_source
        == OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
        and args.base_checkpoint_source
        == OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
    )
    if (
        OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
        in {args.prerequisite_source, args.base_checkpoint_source}
        and not official_show_adapt
    ):
        raise InferenceContractError(
            "official_show_adapt_v1 must be selected for both Base and "
            "representation prerequisites"
        )
    source_receipt = _source_receipt(args)
    official_adapt_producer_sources = (
        _official_adapt_producer_source_receipts(args)
        if official_show_adapt
        else {}
    )
    training_source_receipt = (
        None
        if fully_released_zero_shot or official_show_adapt
        else _expected_source_role_receipt(
            role="training",
            commit=args.expected_training_source_commit,
            tree=args.expected_training_source_tree,
        )
    )
    input_source_receipt = _expected_source_role_receipt(
        role="input_artifact",
        commit=args.expected_input_source_commit,
        tree=args.expected_input_source_tree,
    )
    canonical_manifest = _resolved_regular_file(
        args.canonical_manifest,
        "canonical manifest",
    )
    canonical_summary = _resolved_regular_file(
        args.canonical_summary_json,
        "canonical summary",
    )
    canonical_lineage = _resolved_regular_file(
        args.canonical_lineage_json,
        "canonical lineage",
    )
    (
        canonical_summary_payload,
        canonical_lineage_payload,
        canonical_sha,
        canonical_contract_sha,
    ) = _validate_canonical_root_receipts(
        canonical_manifest,
        canonical_summary,
        canonical_lineage,
        args.expected_canonical_manifest_sha256,
        args.expected_canonical_source_commit,
        args.expected_canonical_source_tree,
    )
    canonical_receipt = {
        "manifest": str(canonical_manifest),
        "manifest_sha256": canonical_sha,
        "summary": str(canonical_summary),
        "summary_sha256": sha256_file(canonical_summary),
        "lineage": str(canonical_lineage),
        "lineage_sha256": sha256_file(canonical_lineage),
        "lineage_contract_sha256": canonical_contract_sha,
        "source_receipt": canonical_lineage_payload[
            "lineage_contract"
        ]["source_receipt"],
    }
    canonical_rows, canonical_by_id = _canonical_test_rows(
        canonical_manifest,
        expected_lineage_contract_sha256=canonical_contract_sha,
    )

    audio_manifests = [
        _resolved_regular_file(path, "audio feature manifest")
        for path in args.audio_manifest
    ]
    audio_by_id, audio_hashes, audio_rows_by_manifest = _audio_feature_rows(
        audio_manifests,
        canonical_by_id,
    )
    audio_lineages, audio_summary_hashes, audio_lineage_hashes = (
        _validate_audio_receipts(
            manifests=audio_manifests,
            manifest_hashes=audio_hashes,
            rows_by_manifest=audio_rows_by_manifest,
            summary_paths=args.audio_summary_json,
            lineage_paths=args.audio_lineage_json,
            canonical_manifest=canonical_manifest,
            canonical_manifest_sha=canonical_sha,
            canonical_lineage_contract_sha256=canonical_contract_sha,
            canonical_receipt=canonical_receipt,
            expected_source_commit=args.expected_input_source_commit,
            expected_source_tree=args.expected_input_source_tree,
            expected_hubert_tree_sha256=(
                args.expected_hubert_tree_sha256
            ),
        )
    )
    audio_contract_by_shard = {
        _require_exact_int(
            lineage.get("shard_id"),
            "audio lineage shard_id",
        ): str(
            lineage["audio_lineage_contract_sha256"]
        )
        for lineage in audio_lineages
    }
    for evaluation_index, clip_id in enumerate(sorted(audio_by_id)):
        row = audio_by_id[clip_id]
        expected_shard = evaluation_index % EXPECTED_NUM_SHARDS
        if (
            _require_exact_int(
                row.get("shard_id"),
                f"{clip_id} audio row shard_id",
            )
            != expected_shard
            or _require_exact_int(
                row.get("num_shards"),
                f"{clip_id} audio row num_shards",
            )
            != EXPECTED_NUM_SHARDS
            or row["lineage_contract_sha256"] != canonical_contract_sha
            or row["audio_lineage_contract_sha256"]
            != audio_contract_by_shard[expected_shard]
        ):
            raise InferenceContractError(
                f"{clip_id}: audio shard/lineage ownership mismatch"
            )
    if official_show_adapt:
        _require_official_adapt_detached_source(
            Path(str(source_receipt["source_root"]))
        )
        base_status = _resolved_regular_file(
            args.base_status_json,
            "official-adapt Base status",
        )
        base_manifest = _resolved_regular_file(
            args.base_candidate_manifest,
            "official-adapt Base candidate manifest",
        )
        base_frozen = _resolved_regular_file(
            args.base_frozen_inputs_json,
            "official-adapt Base frozen inputs",
        )
        face_summary = _resolved_regular_file(
            args.face_status_json,
            "official-adapt Face summary",
        )
        global_summary = _resolved_regular_file(
            args.global_status_json,
            "official-adapt Global summary",
        )
        checkpoint_validation: dict[str, dict[str, Any]] = {}
        common = {
            "expected_training_lineage_sha256": None,
            "expected_dataset_summary_sha256": None,
            "expected_data_mdb_sha256": None,
            "checkpoint_source": OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
            "expected_release_record": None,
            "released_cross_domain_gate_receipt_sha256": None,
            "expected_official_adapt_canonical_receipt": canonical_receipt,
        }
        checkpoint_validation["base"] = {
            **common,
            "expected_source_receipt": official_adapt_producer_sources[
                "base_training"
            ],
            "status_path": base_status,
            "base_candidate_manifest_path": base_manifest,
            "expected_base_candidate_manifest_sha256": (
                args.expected_base_candidate_manifest_sha256
            ),
            "expected_base_formal_status_sha256": None,
            "expected_base_final_checkpoint_sha256": None,
            "official_adapt_frozen_inputs_path": base_frozen,
            "expected_official_adapt_frozen_inputs_sha256": (
                args.expected_base_frozen_inputs_sha256
            ),
            "expected_official_adapt_status_sha256": (
                args.expected_base_formal_status_sha256
            ),
        }
        for stage in ("face", "global"):
            checkpoint_validation[stage] = {
                **common,
                "expected_source_receipt": official_adapt_producer_sources[
                    "transfer_training"
                ],
                "status_path": (
                    face_summary if stage == "face" else global_summary
                ),
                "expected_official_adapt_status_sha256": (
                    args.expected_face_status_sha256
                    if stage == "face"
                    else args.expected_global_status_sha256
                ),
            }
        for stage in ("hands", "upper", "lower"):
            checkpoint_validation[stage] = {
                **common,
                "expected_source_receipt": None,
                "status_path": None,
            }
        frozen_receipts = {
            "base_frozen_inputs": {
                "path": str(base_frozen),
                "sha256": args.expected_base_frozen_inputs_sha256,
            },
            "base_candidate_manifest": {
                "path": str(base_manifest),
                "sha256": args.expected_base_candidate_manifest_sha256,
            },
            "base_status": {
                "path": str(base_status),
                "sha256": args.expected_base_formal_status_sha256,
            },
            "face_summary": {
                "path": str(face_summary),
                "sha256": args.expected_face_status_sha256,
            },
            "global_summary": {
                "path": str(global_summary),
                "sha256": args.expected_global_status_sha256,
            },
        }
        return {
            "canonical_manifest": canonical_manifest,
            "canonical_manifest_sha256": canonical_sha,
            "canonical_summary_path": canonical_summary,
            "canonical_summary_sha256": sha256_file(canonical_summary),
            "canonical_summary": canonical_summary_payload,
            "canonical_lineage_path": canonical_lineage,
            "canonical_lineage_sha256": sha256_file(canonical_lineage),
            "canonical_lineage": canonical_lineage_payload,
            "canonical_lineage_contract_sha256": canonical_contract_sha,
            "canonical_receipt": canonical_receipt,
            "canonical_rows": canonical_rows,
            "canonical_by_id": canonical_by_id,
            "audio_by_id": audio_by_id,
            "audio_manifest_sha256": audio_hashes,
            "audio_summary_sha256": audio_summary_hashes,
            "audio_lineage_sha256": audio_lineage_hashes,
            "audio_lineages": audio_lineages,
            "training_lineage_manifest_sha256": frozen_receipts,
            "base_training_summary_path": None,
            "base_training_summary_sha256": None,
            "accepted_training_lineages": {
                "base": {
                    "kind": "official_SHOW_adaptation",
                    "source": OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
                    "selection_split": "val",
                },
                "face": {
                    "kind": "official_decoder_transfer",
                    "selection_split": "val",
                },
                "global": {
                    "kind": "official_decoder_transfer",
                    "selection_split": "val",
                },
                "hands": {"kind": "official_release", "speaker_scope": "All"},
                "upper": {"kind": "official_release", "speaker_scope": "All"},
                "lower": {"kind": "official_release", "speaker_scope": "All"},
            },
            "checkpoint_validation": checkpoint_validation,
            "base_training_lineage": None,
            "representation_training_lineage": None,
            "source": source_receipt,
            "source_roles": _official_adapt_source_roles(
                inference_source=source_receipt,
                producer_sources=official_adapt_producer_sources,
                input_source=input_source_receipt,
                canonical_source=canonical_receipt["source_receipt"],
            ),
            "prerequisite_source": args.prerequisite_source,
            "base_checkpoint_source": args.base_checkpoint_source,
            "released_prerequisite_import_receipt": None,
            "inference_mode": OFFICIAL_SHOW_ADAPT_MODE,
        }
    if fully_released_zero_shot:
        released_gate = _validate_released_cross_domain_gate(
            args=args,
            source_receipt=source_receipt,
            input_source_receipt=input_source_receipt,
            canonical_receipt=canonical_receipt,
        )
        gate_receipt_sha = released_gate["receipt_sha256"]
        accepted_training_lineages = {
            stage: {
                "kind": "official_release",
                "mode": FULLY_RELEASED_ZERO_SHOT_MODE,
                "checkpoint_source": (
                    args.base_checkpoint_source
                    if stage == "base"
                    else args.prerequisite_source
                ),
                "classification": (
                    RELEASED_ALL_SPEAKERS_BASE_CLASSIFICATION
                    if stage == "base"
                    else RELEASED_ALL_SPEAKERS_CLASSIFICATION
                ),
                "cross_domain_gate_receipt_sha256": gate_receipt_sha,
            }
            for stage in CHECKPOINT_STAGES
        }
        checkpoint_validation: dict[str, dict[str, Any]] = {}
        for stage in CHECKPOINT_STAGES:
            checkpoint_validation[stage] = {
                "checkpoint_source": (
                    args.base_checkpoint_source
                    if stage == "base"
                    else args.prerequisite_source
                ),
                "expected_training_lineage_sha256": None,
                "status_path": None,
                "expected_source_receipt": None,
                "expected_dataset_summary_sha256": None,
                "expected_data_mdb_sha256": None,
                "expected_release_record": None,
                "released_cross_domain_gate_receipt_sha256": (
                    gate_receipt_sha
                ),
            }
        checkpoint_validation["base"].update(
            {
                "base_candidate_manifest_path": None,
                "expected_base_candidate_manifest_sha256": None,
                "expected_base_formal_status_sha256": None,
                "expected_base_final_checkpoint_sha256": None,
            }
        )
        return {
            "canonical_manifest": canonical_manifest,
            "canonical_manifest_sha256": canonical_sha,
            "canonical_summary_path": canonical_summary,
            "canonical_summary_sha256": sha256_file(canonical_summary),
            "canonical_summary": canonical_summary_payload,
            "canonical_lineage_path": canonical_lineage,
            "canonical_lineage_sha256": sha256_file(canonical_lineage),
            "canonical_lineage": canonical_lineage_payload,
            "canonical_lineage_contract_sha256": canonical_contract_sha,
            "canonical_receipt": canonical_receipt,
            "canonical_rows": canonical_rows,
            "canonical_by_id": canonical_by_id,
            "audio_by_id": audio_by_id,
            "audio_manifest_sha256": audio_hashes,
            "audio_summary_sha256": audio_summary_hashes,
            "audio_lineage_sha256": audio_lineage_hashes,
            "audio_lineages": audio_lineages,
            "training_lineage_manifest_sha256": {},
            "base_training_summary_path": None,
            "base_training_summary_sha256": None,
            "accepted_training_lineages": accepted_training_lineages,
            "checkpoint_validation": checkpoint_validation,
            "base_training_lineage": None,
            "representation_training_lineage": None,
            "source": source_receipt,
            "source_roles": dict(
                released_gate["payload"]["source_roles"]
            ),
            "prerequisite_source": args.prerequisite_source,
            "base_checkpoint_source": args.base_checkpoint_source,
            "released_prerequisite_import_receipt": None,
            "inference_mode": FULLY_RELEASED_ZERO_SHOT_MODE,
            "fully_released_zero_shot_gate": released_gate,
        }
    base_training_lineage = _resolved_regular_file(
        args.base_training_lineage_manifest,
        "Base training lineage manifest",
    )
    base_training_summary = _resolved_regular_file(
        args.base_training_summary_json,
        "Base training dataset summary",
    )
    representation_training_lineage = _resolved_regular_file(
        args.representation_training_lineage_manifest,
        "representation training lineage manifest",
    )
    base_training_lineage_sha = sha256_file(base_training_lineage)
    base_training_summary_sha = sha256_file(base_training_summary)
    representation_training_lineage_sha = sha256_file(
        representation_training_lineage
    )
    base_lineage_payload = load_json(base_training_lineage)
    base_summary_payload = load_json(base_training_summary)
    representation_lineage_payload = load_json(
        representation_training_lineage
    )
    if (
        base_lineage_payload.get("format")
        != "semtalk_show_base_feature_lineage_v1"
        or base_lineage_payload.get("status") != "complete"
    ):
        raise InferenceContractError(
            "Base training lineage is not a complete Base feature lineage"
        )
    if (
        base_summary_payload.get("format")
        != "semtalk_show_base_lmdb_summary_v1"
        or base_summary_payload.get("status") != "complete"
        or Path(str(base_summary_payload.get("lineage_json", ""))).resolve()
        != base_training_lineage
        or base_summary_payload.get("lineage_json_sha256")
        != base_training_lineage_sha
        or _require_exact_int(
            base_summary_payload.get("train_clips"),
            "Base train_clips",
        )
        != 13_687
        or _require_exact_int(
            base_summary_payload.get("entries"),
            "Base entries",
        )
        != 127_286
    ):
        raise InferenceContractError(
            "Base training dataset summary/lineage binding is invalid"
        )
    if (
        representation_lineage_payload.get("format")
        != "semtalk_show_representation_lmdb_v2_global_foot"
        or representation_lineage_payload.get("status") != "complete"
        or _require_exact_int(
            representation_lineage_payload.get("train_clips"),
            "representation train_clips",
        )
        != 13_687
        or _require_exact_int(
            representation_lineage_payload.get("entries"),
            "representation entries",
        )
        != 127_286
    ):
        raise InferenceContractError(
            "representation training lineage is not complete"
        )
    base_protocol = base_lineage_payload.get("protocol")
    representation_protocol = representation_lineage_payload.get("protocol")
    if (
        not isinstance(base_protocol, dict)
        or base_protocol.get("scope") != "SemTalk Base only"
        or base_protocol.get("split") != "train"
        or base_protocol.get("speakers") != SHOW_SPEAKER_IDS
        or _require_exact_int(
            base_protocol.get("window_length"),
            "Base window_length",
        )
        != 64
        or _require_exact_int(base_protocol.get("stride"), "Base stride")
        != 20
        or base_protocol.get("in_word")
        != "int64_all_zero_unused_placeholder"
        or set(base_protocol.get("forbidden_components", []))
        != FORBIDDEN_COMPONENTS
        or (
            args.prerequisite_source
            == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
            and base_protocol.get("prerequisite_source")
            != RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
        )
        or (
            args.prerequisite_source == SHOW_TRAINED_CHECKPOINT_SOURCE
            and base_protocol.get("prerequisite_source") not in (None, "")
        )
        or not isinstance(representation_protocol, dict)
        or representation_protocol.get("split") != "train"
        or representation_protocol.get("speaker_map")
        != SHOW_SPEAKER_IDS
        or _require_exact_int(
            representation_protocol.get("window_length"),
            "representation window_length",
        )
        != 64
        or _require_exact_int(
            representation_protocol.get("stride"),
            "representation stride",
        )
        != 20
    ):
        raise InferenceContractError(
            "Base/representation four-speaker training protocol mismatch"
        )
    # Inference never reads either training LMDB.  Bind their already-verified
    # SHA receipts through the immutable summaries/checkpoint audits without
    # re-reading roughly 100 GiB of training data in every one of eight shards.
    base_data_sha = _require_sha256(
        str(base_summary_payload.get("data_mdb_sha256", "")),
        "Base training data.mdb receipt",
    )
    representation_data_sha = _require_sha256(
        str(representation_lineage_payload.get("data_mdb_sha256", "")),
        "representation training data.mdb receipt",
    )
    training_source_triplet = {
        key: training_source_receipt[key]
        for key in ("origin", "commit", "tree")
    }
    input_source_triplet = {
        key: input_source_receipt[key]
        for key in ("origin", "commit", "tree")
    }
    lineage_input_source = base_lineage_payload.get(
        "input_artifact_source_receipt"
    )
    lineage_input_source_sha = base_lineage_payload.get(
        "input_artifact_source_receipt_sha256"
    )
    if (
        base_lineage_payload.get("canonical_receipt") != canonical_receipt
        or representation_lineage_payload.get("canonical_receipt")
        != canonical_receipt
        or base_lineage_payload.get("canonical_manifest_sha256")
        != {str(canonical_manifest): canonical_sha}
        or representation_lineage_payload.get(
            "canonical_manifest_sha256"
        )
        != {str(canonical_manifest): canonical_sha}
        or {
            key: base_lineage_payload.get("source_receipt", {}).get(key)
            for key in ("origin", "commit", "tree")
        }
        != training_source_triplet
        or {
            key: representation_lineage_payload.get(
                "source_receipt", {}
            ).get(key)
            for key in ("origin", "commit", "tree")
        }
        != input_source_triplet
        or (
            lineage_input_source is not None
            and lineage_input_source != input_source_receipt
        )
        or (
            lineage_input_source is not None
            and lineage_input_source_sha
            != canonical_json_sha256(input_source_receipt)
        )
        or (
            args.prerequisite_source
            == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
            and lineage_input_source != input_source_receipt
        )
    ):
        raise InferenceContractError(
            "training lineage canonical/source binding mismatch"
        )
    if base_lineage_payload.get(
        "representation_training_lineage_manifest_sha256"
    ) != representation_training_lineage_sha:
        raise InferenceContractError(
            "Base feature lineage does not bind this representation lineage"
        )
    expected_prerequisites = base_lineage_payload.get("formal_checkpoints")
    prerequisite_stages = set(CHECKPOINT_STAGES) - {"base"}
    if (
        not isinstance(expected_prerequisites, dict)
        or set(expected_prerequisites) != prerequisite_stages
    ):
        raise InferenceContractError(
            "Base feature lineage lacks the exact five prerequisite receipts"
        )
    release_import_receipt: dict[str, Any] | None = None
    release_records: dict[str, dict[str, Any]] = {}
    if (
        args.prerequisite_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    ):
        release_import_receipt, release_records = (
            _validate_released_prerequisite_lineage_binding(
                lineage=base_lineage_payload,
                args=args,
            )
        )
    else:
        if base_lineage_payload.get("prerequisite_source_receipt") is not None:
            raise InferenceContractError(
                "show_trained_v1 Base lineage cannot carry an official "
                "release import receipt"
            )
        for stage in sorted(prerequisite_stages):
            record = expected_prerequisites[stage]
            expected_path = _resolved_regular_file(
                getattr(args, f"{stage}_checkpoint"),
                f"{stage} checkpoint",
            )
            expected_sha = _require_sha256(
                str(getattr(args, f"expected_{stage}_sha256")),
                f"{stage} expected SHA",
            )
            expected_status = _resolved_regular_file(
                getattr(args, f"{stage}_status_json"),
                f"{stage} formal training status",
            )
            record_audit_source = record.get("audit", {}).get(
                "source_receipt"
            )
            if (
                not isinstance(record, dict)
                or record.get("formal_stage") != stage
                or Path(str(record.get("path", ""))).resolve()
                != expected_path
                or record.get("sha256") != expected_sha
                or record.get("audit", {}).get(
                    "lineage_manifest_sha256"
                )
                != representation_training_lineage_sha
                or Path(
                    str(record.get("formal_training_status", ""))
                ).resolve()
                != expected_status
                or record.get("formal_training_status_sha256")
                != sha256_file(expected_status)
                or not isinstance(record_audit_source, dict)
                or {
                    key: record_audit_source.get(key)
                    for key in ("origin", "commit", "tree")
                }
                != input_source_triplet
                or record.get("audit", {}).get("source_receipt_sha256")
                != compact_json_sha256(record_audit_source)
            ):
                raise InferenceContractError(
                    f"{stage}: checkpoint is not the prerequisite bound to "
                    "this Base feature lineage"
                )
    training_lineage_hashes = {
        "base": {
            "path": str(base_training_lineage),
            "sha256": base_training_lineage_sha,
        },
        "representation": {
            "path": str(representation_training_lineage),
            "sha256": representation_training_lineage_sha,
        },
    }
    base_is_released = (
        args.base_checkpoint_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    )
    accepted_training_lineages: dict[str, Any] = {}
    accepted_training_lineages["base"] = (
        {
            "kind": "official_release",
            "checkpoint_source": args.base_checkpoint_source,
            "classification": RELEASED_ALL_SPEAKERS_BASE_CLASSIFICATION,
        }
        if base_is_released
        else base_training_lineage_sha
    )
    for stage in sorted(prerequisite_stages):
        accepted_training_lineages[stage] = (
            {
                "kind": "official_release",
                "checkpoint_source": args.prerequisite_source,
                "import_receipt_sha256": (
                    release_import_receipt["receipt_sha256"]
                    if release_import_receipt is not None
                    else None
                ),
            }
            if release_import_receipt is not None
            else representation_training_lineage_sha
        )
    checkpoint_validation: dict[str, dict[str, Any]] = {}
    checkpoint_validation["base"] = {
        "checkpoint_source": args.base_checkpoint_source,
        "expected_training_lineage_sha256": (
            None if base_is_released else base_training_lineage_sha
        ),
        "status_path": (
            None
            if base_is_released
            else _resolved_regular_file(
                args.base_status_json,
                "base formal training status",
            )
        ),
        "expected_source_receipt": (
            None if base_is_released else training_source_receipt
        ),
        "expected_dataset_summary_sha256": (
            None if base_is_released else base_training_summary_sha
        ),
        "expected_data_mdb_sha256": (
            None if base_is_released else base_data_sha
        ),
        "base_candidate_manifest_path": (
            None if base_is_released else args.base_candidate_manifest
        ),
        "expected_base_candidate_manifest_sha256": (
            None
            if base_is_released
            else args.expected_base_candidate_manifest_sha256
        ),
        "expected_base_formal_status_sha256": (
            None
            if base_is_released
            else args.expected_base_formal_status_sha256
        ),
        "expected_base_final_checkpoint_sha256": (
            None
            if base_is_released
            else args.expected_base_final_checkpoint_sha256
        ),
    }
    for stage in sorted(prerequisite_stages):
        released = release_import_receipt is not None
        checkpoint_validation[stage] = {
            "checkpoint_source": args.prerequisite_source,
            "expected_training_lineage_sha256": (
                None if released else representation_training_lineage_sha
            ),
            "status_path": (
                None
                if released
                else _resolved_regular_file(
                    getattr(args, f"{stage}_status_json"),
                    f"{stage} formal training status",
                )
            ),
            "expected_source_receipt": (
                None if released else input_source_receipt
            ),
            "expected_dataset_summary_sha256": (
                None if released else representation_training_lineage_sha
            ),
            "expected_data_mdb_sha256": (
                None if released else representation_data_sha
            ),
            "expected_release_record": (
                release_records[stage] if released else None
            ),
        }
    return {
        "canonical_manifest": canonical_manifest,
        "canonical_manifest_sha256": canonical_sha,
        "canonical_summary_path": canonical_summary,
        "canonical_summary_sha256": sha256_file(canonical_summary),
        "canonical_summary": canonical_summary_payload,
        "canonical_lineage_path": canonical_lineage,
        "canonical_lineage_sha256": sha256_file(canonical_lineage),
        "canonical_lineage": canonical_lineage_payload,
        "canonical_lineage_contract_sha256": canonical_contract_sha,
        "canonical_rows": canonical_rows,
        "canonical_by_id": canonical_by_id,
        "audio_by_id": audio_by_id,
        "audio_manifest_sha256": audio_hashes,
        "audio_summary_sha256": audio_summary_hashes,
        "audio_lineage_sha256": audio_lineage_hashes,
        "audio_lineages": audio_lineages,
        "training_lineage_manifest_sha256": training_lineage_hashes,
        "base_training_summary_path": base_training_summary,
        "base_training_summary_sha256": base_training_summary_sha,
        "accepted_training_lineages": accepted_training_lineages,
        "checkpoint_validation": checkpoint_validation,
        "base_training_lineage": base_lineage_payload,
        "representation_training_lineage": (
            representation_lineage_payload
        ),
        "source": source_receipt,
        "source_roles": {
            "inference": source_receipt,
            "training": training_source_receipt,
            "input_artifact": input_source_receipt,
        },
        "prerequisite_source": args.prerequisite_source,
        "base_checkpoint_source": args.base_checkpoint_source,
        "released_prerequisite_import_receipt": release_import_receipt,
    }


def _checkpoint_args_receipt(
    args: argparse.Namespace,
) -> dict[str, dict[str, str]]:
    return {
        stage: {
            "path": str(Path(getattr(args, f"{stage}_checkpoint")).resolve()),
            "expected_sha256": getattr(args, f"expected_{stage}_sha256"),
        }
        for stage in CHECKPOINT_STAGES
    }


def _stable_contract_receipt(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = {
        "format": "semtalk_show_base_inference_contract_v1",
        "canonical_manifest": str(inputs["canonical_manifest"]),
        "canonical_manifest_sha256": inputs["canonical_manifest_sha256"],
        "canonical_summary_sha256": inputs["canonical_summary_sha256"],
        "canonical_lineage_sha256": inputs["canonical_lineage_sha256"],
        "audio_manifest_sha256": inputs["audio_manifest_sha256"],
        "audio_summary_sha256": inputs["audio_summary_sha256"],
        "audio_lineage_sha256": inputs["audio_lineage_sha256"],
        "training_lineage_manifest_sha256": inputs[
            "training_lineage_manifest_sha256"
        ],
        "base_training_summary_sha256": inputs[
            "base_training_summary_sha256"
        ],
        "training_lineage_stage_mapping": inputs[
            "accepted_training_lineages"
        ],
        "source": inputs["source"],
        "source_roles": inputs["source_roles"],
        "base_checkpoint_source": inputs["base_checkpoint_source"],
        "representation_prerequisite_source": inputs[
            "prerequisite_source"
        ],
        "released_prerequisite_import_receipt": inputs[
            "released_prerequisite_import_receipt"
        ],
        "checkpoints": checkpoints,
        "speaker_mapping": SHOW_SPEAKER_IDS,
        "test_clips": EXPECTED_TEST_CLIPS,
        "num_shards": EXPECTED_NUM_SHARDS,
        "seed": args.seed,
        "autoregression": {
            "window": WINDOW,
            "pre_frames": PRE_FRAMES,
            "stride": STRIDE,
            "first_seed": (
                "GT pose/trans frames 0..3; contact is identically zero, "
                "including where historical public _g_test could preserve it"
            ),
            "later_seed": "generated previous-window last four only",
            "future_motion_payload": "identically zero",
            "in_word": "identically zero",
            "condition_padding": "repeat final audio feature frame",
            "rvq_stitch": "first 16 tokens, then last 15 tokens",
            "final_crop": "exact original canonical frame count",
        },
        "public_g_test_global_compatibility": {
            "recurrent_lower_translation": (
                "integrate decoded lower channels 54:57 across windows"
            ),
            "recurrent_lower_contact": "preserve decoded lower channels 57:61",
            "final_global_input": (
                "projected rotations plus zero translation channels 54:57 "
                "and preserved lower contact channels 57:61"
            ),
            "final_translation": (
                "integrate decoded global channels 54:57 as public _g_test does"
            ),
            "known_training_inference_mismatch": (
                "formal global aelowerfoot training supplied zero channels "
                "54:61, and lower translation was not supervised; inference "
                "intentionally preserves the released public _g_test behavior"
            ),
        },
        "diffsheg_evaluation_boundary": {
            "input": "full canonical SHOW NPZ; never a body-only projection",
            "window_frames": DIFFSHEG_WINDOW,
            "window_stride": DIFFSHEG_WINDOW,
            "metrics": [
                "FMD",
                "FED",
                "expression_diversity",
                "FGD",
                "BA",
                "PCM",
                "gesture_diversity",
            ],
            "talkshow_official_body_protocol": "not used",
        },
        "forbidden_components": sorted(FORBIDDEN_COMPONENTS | {"Speaker2"}),
        "output": {
            "directory": f"{FINAL_BUNDLE_NAME}/npz/test",
            "prediction": "res_<speaker>__<sequence>.npz",
            "ground_truth": (
                "gt_<speaker>__<sequence>.npz deterministically derived from "
                "the frozen canonical SHOW test cache"
            ),
            "fields": list(OUTPUT_FIELDS),
            "pose_convention": "SHOW native local axis-angle",
            "prediction_local_eyes_69_75": "zero",
        },
    }
    if inputs.get("inference_mode") == FULLY_RELEASED_ZERO_SHOT_MODE:
        receipt["inference_mode"] = FULLY_RELEASED_ZERO_SHOT_MODE
        receipt["fully_released_zero_shot_gate"] = inputs[
            "fully_released_zero_shot_gate"
        ]
    elif inputs.get("inference_mode") == OFFICIAL_SHOW_ADAPT_MODE:
        receipt["inference_mode"] = OFFICIAL_SHOW_ADAPT_MODE
        receipt["official_show_adapt_interface"] = {
            "base": (
                "val-selected official-initialized SHOW Base predicts "
                "all four RVQ code streams"
            ),
            "face": "adapted decoder consumes Base-predicted Face codes",
            "lower": (
                "Base-predicted Lower codes are decoded by the exact frozen "
                "official All-Speakers Lower RVQ"
            ),
            "global": (
                "adapted decoder-only Global consumes that decoded Lower "
                "motion with channels 54:57 zeroed"
            ),
            "selection_split": "val",
            "test_evaluations": 1,
            "test_feedback_into_selection": False,
        }
        receipt["public_g_test_global_compatibility"] = {
            "recurrent_lower_translation": (
                "official Lower decode is used only for generated recurrence"
            ),
            "recurrent_lower_contact": (
                "preserve decoded Lower contact channels 57:61"
            ),
            "final_global_input": (
                "Base-predicted Lower decode, SO(3)-projected rotations, "
                "zero channels 54:57, decoded contacts 57:61"
            ),
            "final_translation": (
                "integrate adapted Global channels 54:57"
            ),
            "known_training_inference_mismatch": None,
        }
    return receipt


def _revalidate_frozen_inputs(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    checkpoint_receipts: Mapping[str, Mapping[str, Any]],
    *,
    output_rows: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Prove that every frozen input still matches the start-of-run receipt."""

    if _source_receipt(args) != inputs["source"]:
        raise InferenceContractError("formal source changed during inference")
    fixed_files = {
        str(inputs["canonical_manifest"]): inputs["canonical_manifest_sha256"],
        str(inputs["canonical_summary_path"]): inputs["canonical_summary_sha256"],
        str(inputs["canonical_lineage_path"]): inputs["canonical_lineage_sha256"],
        **inputs["audio_manifest_sha256"],
        **inputs["audio_summary_sha256"],
        **inputs["audio_lineage_sha256"],
    }
    base_training_summary_path = inputs.get("base_training_summary_path")
    base_training_summary_sha = inputs.get("base_training_summary_sha256")
    if (
        base_training_summary_path is None
        or base_training_summary_sha is None
    ):
        if (
            base_training_summary_path is not None
            or base_training_summary_sha is not None
            or inputs.get("inference_mode")
            not in {
                FULLY_RELEASED_ZERO_SHOT_MODE,
                OFFICIAL_SHOW_ADAPT_MODE,
            }
        ):
            raise InferenceContractError(
                "partial or unauthorized absent Base training summary"
            )
    else:
        fixed_files[str(base_training_summary_path)] = str(
            base_training_summary_sha
        )
    for receipt in inputs["training_lineage_manifest_sha256"].values():
        fixed_files[str(receipt["path"])] = str(receipt["sha256"])
    for path_value, expected_sha in fixed_files.items():
        _verify_file_sha(
            _resolved_regular_file(path_value, "frozen inference input"),
            str(expected_sha),
            "frozen inference input",
        )
    if inputs.get("inference_mode") == FULLY_RELEASED_ZERO_SHOT_MODE:
        observed_gate = _validate_released_cross_domain_gate(
            args=args,
            source_receipt=inputs["source"],
            input_source_receipt=inputs["source_roles"][
                "input_artifact"
            ],
            canonical_receipt=inputs["canonical_receipt"],
        )
        if observed_gate != inputs.get("fully_released_zero_shot_gate"):
            raise InferenceContractError(
                "released cross-domain gate changed during inference"
            )
    for stage, receipt in checkpoint_receipts.items():
        _verify_file_sha(
            _resolved_regular_file(receipt["path"], f"{stage} checkpoint"),
            str(receipt["sha256"]),
            f"{stage} checkpoint",
        )
        formal_status = receipt.get("formal_training_status")
        formal_status_sha = receipt.get("formal_training_status_sha256")
        if formal_status is not None or formal_status_sha is not None:
            if formal_status is None or formal_status_sha is None:
                raise InferenceContractError(
                    f"{stage}: partial formal training status receipt"
                )
            _verify_file_sha(
                _resolved_regular_file(
                    formal_status,
                    f"{stage} formal training status",
                ),
                str(formal_status_sha),
                f"{stage} formal training status",
            )
        candidate_manifest = receipt.get("base_candidate_manifest")
        if candidate_manifest is not None:
            _verify_file_sha(
                _resolved_regular_file(
                    candidate_manifest["path"],
                    "Base candidate manifest",
                ),
                str(candidate_manifest["sha256"]),
                "Base candidate manifest",
            )
        completed_final = receipt.get("completed_final_checkpoint")
        if completed_final is not None:
            _verify_file_sha(
                _resolved_regular_file(
                    completed_final["path"],
                    "Base completed final checkpoint",
                ),
                str(completed_final["sha256"]),
                "Base completed final checkpoint",
            )
        candidate_checkpoints = receipt.get("candidate_checkpoints")
        if candidate_checkpoints is not None:
            if (
                not isinstance(candidate_checkpoints, list)
                or len(candidate_checkpoints) != BASE_CANDIDATE_COUNT
            ):
                raise InferenceContractError(
                    "Base candidate receipt lost its exact checkpoint cover"
                )
            for candidate in candidate_checkpoints:
                if not isinstance(candidate, dict):
                    raise InferenceContractError(
                        "invalid Base candidate checkpoint receipt"
                    )
                _verify_file_sha(
                    _resolved_regular_file(
                        candidate.get("path", ""),
                        "Base candidate checkpoint",
                    ),
                    str(candidate.get("sha256", "")),
                    "Base candidate checkpoint",
                )
    for row in output_rows:
        clip_id = str(row["source_clip_id"])
        canonical = inputs["canonical_by_id"].get(clip_id)
        audio = inputs["audio_by_id"].get(clip_id)
        if canonical is None or audio is None:
            raise InferenceContractError(
                f"{clip_id}: output row lost its frozen input binding"
            )
        _verify_file_sha(
            _resolved_regular_file(canonical["canonical_npz"], "canonical NPZ"),
            str(canonical["canonical_npz_sha256"]),
            "canonical NPZ",
        )
        _verify_file_sha(
            _resolved_regular_file(audio["audio_feature_npz"], "audio feature NPZ"),
            str(audio["audio_feature_npz_sha256"]),
            "audio feature NPZ",
        )


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.num_shards != EXPECTED_NUM_SHARDS:
        raise InferenceContractError("formal inference requires exactly eight shards")
    inputs = _input_contract(args)
    _set_deterministic(args.seed)
    runtime = _runtime_receipt(args.device)
    models, checkpoint_receipts = _load_models(
        args,
        inputs,
    )
    contract = _stable_contract_receipt(
        args,
        inputs,
        checkpoint_receipts,
    )
    contract_sha = canonical_json_sha256(contract)

    shard_name = f"shard-{args.shard_id:05d}-of-{args.num_shards:05d}"
    shard_root = args.output_root / "shards" / shard_name
    if shard_root.exists():
        raise FileExistsError(f"refusing to reuse shard output: {shard_root}")
    shard_npz = shard_root / "npz" / "test"
    shard_npz.mkdir(parents=True, exist_ok=False)
    masks = _joint_masks(torch.device(args.device))

    selected = [
        row
        for row in inputs["canonical_rows"]
        if _require_exact_int(
            row.get("global_index"),
            "canonical row global_index",
        )
        % args.num_shards
        == args.shard_id
    ]
    output_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for ordinal, row in enumerate(selected, start=1):
            clip_id = str(row["clip_id"])
            output_id = canonical_clip_id(clip_id)
            canonical, frames = _load_canonical_clip(row)
            audio_row = inputs["audio_by_id"][clip_id]
            audio = _load_audio_features(
                audio_row,
                expected_frames=frames,
            )
            speaker_id = SHOW_SPEAKER_IDS[str(row["speaker"])]
            expected_calls = max(
                1,
                math.ceil((frames - PRE_FRAMES) / STRIDE),
            )
            with _inference_only_auxiliary_loss_bypass(
                models["base"],
                expected_calls=expected_calls,
            ):
                prediction = _infer_clip(
                    pose=canonical["pose"],
                    trans=canonical["trans"],
                    beat=audio["beat"],
                    hubert=audio["hubert"],
                    speaker_id=speaker_id,
                    models=models,
                    masks=masks,
                    device=args.device,
                )
            result_arrays = _output_arrays(
                betas=canonical["beta"][0],
                poses=prediction["poses"],
                expressions=prediction["expressions"],
                trans=prediction["trans"],
            )
            target_arrays = _output_arrays(
                betas=canonical["beta"][0],
                poses=canonical["pose"],
                expressions=canonical["facial"],
                trans=canonical["trans"],
            )
            result_path = shard_npz / f"res_{output_id}.npz"
            target_path = shard_npz / f"gt_{output_id}.npz"
            result_payload = deterministic_npz_bytes(result_arrays)
            target_payload = deterministic_npz_bytes(target_arrays)
            atomic_write_new(result_path, result_payload)
            atomic_write_new(target_path, target_payload)
            result_sha = hashlib.sha256(result_payload).hexdigest()
            target_sha = hashlib.sha256(target_payload).hexdigest()
            if (
                sha256_file(result_path) != result_sha
                or sha256_file(target_path) != target_sha
            ):
                raise InferenceContractError(f"{clip_id}: post-write SHA mismatch")
            output_rows.append(
                {
                    "global_index": _require_exact_int(
                        row.get("global_index"),
                        f"{clip_id} global_index",
                    ),
                    "source_clip_id": clip_id,
                    "canonical_clip_id": output_id,
                    "speaker": str(row["speaker"]),
                    "speaker_id": speaker_id,
                    "frames": frames,
                    "canonical_npz": str(
                        Path(str(row["canonical_npz"])).resolve()
                    ),
                    "canonical_npz_sha256": row["canonical_npz_sha256"],
                    "audio_feature_npz": str(
                        Path(str(audio_row["audio_feature_npz"])).resolve()
                    ),
                    "audio_feature_npz_sha256": audio_row[
                        "audio_feature_npz_sha256"
                    ],
                    "prediction": {
                        "path": str(result_path.resolve()),
                        "bytes": result_path.stat().st_size,
                        "sha256": result_sha,
                    },
                    "ground_truth": {
                        "path": str(target_path.resolve()),
                        "bytes": target_path.stat().st_size,
                        "sha256": target_sha,
                    },
                }
            )
            if (
                ordinal == 1
                or ordinal % args.progress_every == 0
                or ordinal == len(selected)
            ):
                print(
                    f"[{shard_name}] {ordinal}/{len(selected)} {output_id} "
                    f"frames={frames}",
                    flush=True,
                )

    output_rows.sort(
        key=lambda row: _require_exact_int(
            row.get("global_index"),
            "output row global_index",
        )
    )
    expected_files = {
        f"{prefix}_{row['canonical_clip_id']}.npz"
        for row in output_rows
        for prefix in ("res", "gt")
    }
    actual_files = {path.name for path in shard_npz.iterdir()}
    if actual_files != expected_files:
        raise InferenceContractError("shard NPZ directory has an inexact file set")
    manifest_payload = b"".join(
        canonical_json_bytes(row)
        for row in output_rows
    )
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    runtime_sha = canonical_json_sha256(runtime)
    summary = {
        "format": "semtalk_show_base_inference_shard_summary_v1",
        "status": "complete",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "selected_clips": len(output_rows),
        "expected_test_clips": EXPECTED_TEST_CLIPS,
        "manifest_sha256": manifest_sha,
        "contract_sha256": contract_sha,
        "runtime_sha256": runtime_sha,
        "finite": True,
        "exact_once": True,
    }
    lineage = {
        "format": "semtalk_show_base_inference_shard_lineage_v1",
        "status": "complete",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "manifest_sha256": manifest_sha,
        "contract": contract,
        "contract_sha256": contract_sha,
        "runtime": runtime,
        "runtime_sha256": runtime_sha,
        "checkpoints": checkpoint_receipts,
    }
    for row in output_rows:
        frames = _require_exact_int(row.get("frames"), "output row frames")
        prediction_path = Path(row["prediction"]["path"])
        target_path = Path(row["ground_truth"]["path"])
        _load_and_validate_output_npz(
            prediction_path,
            frames=frames,
            prediction=True,
        )
        _load_and_validate_output_npz(
            target_path,
            frames=frames,
            prediction=False,
        )
        canonical, canonical_frames = _load_canonical_clip(
            inputs["canonical_by_id"][str(row["source_clip_id"])]
        )
        if canonical_frames != frames:
            raise InferenceContractError(
                f"{row['source_clip_id']}: canonical/output frame mismatch"
            )
        expected_target = deterministic_npz_bytes(
            _output_arrays(
                betas=canonical["beta"][0],
                poses=canonical["pose"],
                expressions=canonical["facial"],
                trans=canonical["trans"],
            )
        )
        if target_path.read_bytes() != expected_target:
            raise InferenceContractError(
                f"{row['source_clip_id']}: shard ground truth is not canonical"
            )
    _revalidate_frozen_inputs(
        args,
        inputs,
        checkpoint_receipts,
        output_rows=output_rows,
    )
    # The summary is the shard completion marker and must be the final write.
    atomic_write_new(shard_root / "manifest.jsonl", manifest_payload)
    atomic_json_new(shard_root / "lineage.json", lineage)
    atomic_json_new(shard_root / "summary.json", summary)
    return summary


def _safe_shard_artifact(
    path_value: str,
    *,
    shard_root: Path,
    expected_name: str,
) -> Path:
    path = Path(path_value)
    if path.is_symlink():
        raise InferenceContractError(f"shard artifact is a symlink: {path}")
    resolved = path.resolve()
    expected_parent = (shard_root / "npz" / "test").resolve()
    if resolved.parent != expected_parent or resolved.name != expected_name:
        raise InferenceContractError(
            f"shard artifact escapes its exact directory: {resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards != EXPECTED_NUM_SHARDS:
        raise InferenceContractError("formal finalize requires eight shards")
    inputs = _input_contract(args)
    checkpoint_receipts = _checkpoint_receipts_without_models(
        args,
        inputs,
    )
    expected_contract = _stable_contract_receipt(
        args,
        inputs,
        checkpoint_receipts,
    )
    expected_contract_sha = canonical_json_sha256(expected_contract)

    rows_by_index: dict[int, dict[str, Any]] = {}
    shard_receipts: list[dict[str, Any]] = []
    runtime_payload: dict[str, Any] | None = None
    runtime_sha: str | None = None
    for shard_id in range(args.num_shards):
        shard_name = f"shard-{shard_id:05d}-of-{args.num_shards:05d}"
        shard_root = (args.output_root / "shards" / shard_name).resolve()
        manifest_path = shard_root / "manifest.jsonl"
        summary_path = shard_root / "summary.json"
        lineage_path = shard_root / "lineage.json"
        for required in (manifest_path, summary_path, lineage_path):
            if required.is_symlink() or not required.is_file():
                raise InferenceContractError(
                    f"missing or unsafe shard receipt: {required}"
                )
        manifest_sha = sha256_file(manifest_path)
        summary = load_json(summary_path)
        lineage = load_json(lineage_path)
        if (
            summary.get("format")
            != "semtalk_show_base_inference_shard_summary_v1"
            or summary.get("status") != "complete"
            or _require_exact_int(
                summary.get("shard_id"),
                f"{shard_name} summary shard_id",
            )
            != shard_id
            or _require_exact_int(
                summary.get("num_shards"),
                f"{shard_name} summary num_shards",
            )
            != args.num_shards
            or summary.get("manifest_sha256") != manifest_sha
            or summary.get("contract_sha256") != expected_contract_sha
            or summary.get("finite") is not True
            or summary.get("exact_once") is not True
        ):
            raise InferenceContractError(f"{shard_name}: invalid shard summary")
        if (
            lineage.get("format")
            != "semtalk_show_base_inference_shard_lineage_v1"
            or lineage.get("status") != "complete"
            or _require_exact_int(
                lineage.get("shard_id"),
                f"{shard_name} lineage shard_id",
            )
            != shard_id
            or _require_exact_int(
                lineage.get("num_shards"),
                f"{shard_name} lineage num_shards",
            )
            != args.num_shards
            or lineage.get("manifest_sha256") != manifest_sha
            or lineage.get("contract") != expected_contract
            or lineage.get("contract_sha256") != expected_contract_sha
            or lineage.get("checkpoints") != checkpoint_receipts
            or canonical_json_sha256(lineage.get("runtime"))
            != lineage.get("runtime_sha256")
            or summary.get("runtime_sha256")
            != lineage.get("runtime_sha256")
        ):
            raise InferenceContractError(f"{shard_name}: invalid shard lineage")
        if runtime_payload is None:
            runtime_payload = lineage["runtime"]
            runtime_sha = lineage["runtime_sha256"]
        elif (
            lineage["runtime"] != runtime_payload
            or lineage["runtime_sha256"] != runtime_sha
        ):
            raise InferenceContractError("inference shards used different runtimes")

        rows = load_jsonl(manifest_path)
        if (
            _require_exact_int(
                summary.get("selected_clips"),
                f"{shard_name} summary selected_clips",
            )
            != len(rows)
        ):
            raise InferenceContractError(f"{shard_name}: row count mismatch")
        expected_files: set[str] = set()
        for row in rows:
            index = row.get("global_index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise InferenceContractError(
                    f"{shard_name}: invalid global_index {index!r}"
                )
            if index % args.num_shards != shard_id:
                raise InferenceContractError(
                    f"{shard_name}: index {index} belongs to another shard"
                )
            if index in rows_by_index:
                raise InferenceContractError(
                    f"duplicate inference global_index {index}"
                )
            canonical = inputs["canonical_by_id"].get(row.get("source_clip_id"))
            if canonical is None or _require_exact_int(
                canonical.get("global_index"),
                f"{shard_name} canonical global_index",
            ) != index:
                raise InferenceContractError(
                    f"{shard_name}: row does not bind canonical input"
                )
            source_clip_id = str(canonical["clip_id"])
            audio = inputs["audio_by_id"].get(source_clip_id)
            if audio is None:
                raise InferenceContractError(
                    f"{shard_name}: row does not bind frozen audio input"
                )
            expected_id = canonical_clip_id(str(canonical["clip_id"]))
            if row.get("canonical_clip_id") != expected_id:
                raise InferenceContractError(
                    f"{shard_name}: canonical output ID mismatch"
                )
            expected_row_bindings = {
                "source_clip_id": source_clip_id,
                "speaker": str(canonical["speaker"]),
                "speaker_id": _require_exact_int(
                    canonical.get("speaker_id"),
                    f"{source_clip_id} canonical speaker_id",
                ),
                "frames": _require_exact_int(
                    canonical.get("frames"),
                    f"{source_clip_id} canonical frames",
                ),
                "canonical_npz": str(
                    Path(str(canonical["canonical_npz"])).expanduser().resolve()
                ),
                "canonical_npz_sha256": str(
                    canonical["canonical_npz_sha256"]
                ),
                "audio_feature_npz": str(
                    Path(str(audio["audio_feature_npz"])).expanduser().resolve()
                ),
                "audio_feature_npz_sha256": str(
                    audio["audio_feature_npz_sha256"]
                ),
            }
            observed_row_bindings = {
                key: row.get(key)
                for key in expected_row_bindings
            }
            if observed_row_bindings != expected_row_bindings:
                raise InferenceContractError(
                    f"{shard_name}: {expected_id} input receipt mismatch"
                )
            for receipt_name in ("prediction", "ground_truth"):
                receipt = row.get(receipt_name)
                if not isinstance(receipt, dict) or set(receipt) != {
                    "path",
                    "bytes",
                    "sha256",
                }:
                    raise InferenceContractError(
                        f"{expected_id}: invalid {receipt_name} receipt"
                    )
            prediction_name = f"res_{expected_id}.npz"
            target_name = f"gt_{expected_id}.npz"
            prediction = _safe_shard_artifact(
                row["prediction"]["path"],
                shard_root=shard_root,
                expected_name=prediction_name,
            )
            target = _safe_shard_artifact(
                row["ground_truth"]["path"],
                shard_root=shard_root,
                expected_name=target_name,
            )
            _verify_file_sha(
                prediction,
                row["prediction"]["sha256"],
                "shard prediction",
            )
            _verify_file_sha(
                target,
                row["ground_truth"]["sha256"],
                "shard ground truth",
            )
            if (
                prediction.stat().st_size
                != _require_exact_int(
                    row["prediction"].get("bytes"),
                    f"{expected_id} prediction bytes",
                )
                or target.stat().st_size
                != _require_exact_int(
                    row["ground_truth"].get("bytes"),
                    f"{expected_id} ground-truth bytes",
                )
            ):
                raise InferenceContractError(
                    f"{expected_id}: shard output byte count mismatch"
                )
            frames = _require_exact_int(
                canonical.get("frames"),
                f"{expected_id} canonical frames",
            )
            _load_and_validate_output_npz(
                prediction,
                frames=frames,
                prediction=True,
            )
            _load_and_validate_output_npz(
                target,
                frames=frames,
                prediction=False,
            )
            canonical_arrays, canonical_frames = _load_canonical_clip(canonical)
            if canonical_frames != frames:
                raise InferenceContractError(
                    f"{expected_id}: canonical frame count changed"
                )
            _load_audio_features(audio, expected_frames=frames)
            expected_target_payload = deterministic_npz_bytes(
                _output_arrays(
                    betas=canonical_arrays["beta"][0],
                    poses=canonical_arrays["pose"],
                    expressions=canonical_arrays["facial"],
                    trans=canonical_arrays["trans"],
                )
            )
            if (
                target.read_bytes() != expected_target_payload
                or hashlib.sha256(expected_target_payload).hexdigest()
                != row["ground_truth"]["sha256"]
                or len(expected_target_payload)
                != _require_exact_int(
                    row["ground_truth"].get("bytes"),
                    f"{expected_id} ground-truth bytes",
                )
            ):
                raise InferenceContractError(
                    f"{expected_id}: ground truth does not rebuild from canonical input"
                )
            expected_files.update((prediction_name, target_name))
            rows_by_index[index] = row
        actual_files = {
            path.name
            for path in (shard_root / "npz" / "test").iterdir()
        }
        if actual_files != expected_files:
            raise InferenceContractError(
                f"{shard_name}: unexpected or missing NPZ files"
            )
        shard_receipts.append(
            {
                "shard_id": shard_id,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "lineage": str(lineage_path),
                "lineage_sha256": sha256_file(lineage_path),
                "clips": len(rows),
            }
        )

    expected_indices = {
        _require_exact_int(
            row.get("global_index"),
            "canonical row global_index",
        )
        for row in inputs["canonical_rows"]
    }
    if set(rows_by_index) != expected_indices or len(rows_by_index) != EXPECTED_TEST_CLIPS:
        missing = sorted(expected_indices - set(rows_by_index))
        extra = sorted(set(rows_by_index) - expected_indices)
        raise InferenceContractError(
            "inference shards do not cover test exactly once: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    ordered_rows = sorted(
        rows_by_index.values(),
        key=lambda row: str(row["canonical_clip_id"]),
    )
    _revalidate_frozen_inputs(
        args,
        inputs,
        checkpoint_receipts,
        output_rows=ordered_rows,
    )

    with _finalize_lock(args.output_root):
        final_root = args.output_root / FINAL_BUNDLE_NAME
        legacy_targets = (
            args.output_root / "npz",
            args.output_root / "diffsheg_eval_clip_ids.txt",
            args.output_root / "final_manifest.jsonl",
            args.output_root / "final_lineage.json",
            args.output_root / "final_summary.json",
        )
        existing = [
            str(path)
            for path in (final_root, *legacy_targets)
            if os.path.lexists(path)
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite or mix finalized outputs: "
                + ", ".join(existing)
            )

        temporary_root = args.output_root / (
            f".{FINAL_BUNDLE_NAME}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        temporary_test = temporary_root / "npz" / "test"
        temporary_test.mkdir(parents=True, exist_ok=False)
        final_test = final_root / "npz" / "test"
        final_rows: list[dict[str, Any]] = []
        published = False
        try:
            for evaluation_index, row in enumerate(ordered_rows):
                output_id = str(row["canonical_clip_id"])
                frames = _require_exact_int(
                    row.get("frames"),
                    f"{output_id} finalized frames",
                )
                prediction_source = Path(row["prediction"]["path"]).resolve()
                target_source = Path(row["ground_truth"]["path"]).resolve()
                prediction_destination = (
                    temporary_test / f"res_{output_id}.npz"
                )
                target_destination = temporary_test / f"gt_{output_id}.npz"
                _copy_file_fsync_new(
                    prediction_source,
                    prediction_destination,
                    expected_sha256=str(row["prediction"]["sha256"]),
                    expected_bytes=_require_exact_int(
                        row["prediction"].get("bytes"),
                        f"{output_id} prediction bytes",
                    ),
                )
                _copy_file_fsync_new(
                    target_source,
                    target_destination,
                    expected_sha256=str(row["ground_truth"]["sha256"]),
                    expected_bytes=_require_exact_int(
                        row["ground_truth"].get("bytes"),
                        f"{output_id} ground-truth bytes",
                    ),
                )
                _load_and_validate_output_npz(
                    prediction_destination,
                    frames=frames,
                    prediction=True,
                )
                _load_and_validate_output_npz(
                    target_destination,
                    frames=frames,
                    prediction=False,
                )
                canonical, canonical_frames = _load_canonical_clip(
                    inputs["canonical_by_id"][str(row["source_clip_id"])]
                )
                if canonical_frames != frames:
                    raise InferenceContractError(
                        f"{output_id}: canonical frame count changed while publishing"
                    )
                expected_target_payload = deterministic_npz_bytes(
                    _output_arrays(
                        betas=canonical["beta"][0],
                        poses=canonical["pose"],
                        expressions=canonical["facial"],
                        trans=canonical["trans"],
                    )
                )
                if target_destination.read_bytes() != expected_target_payload:
                    raise InferenceContractError(
                        f"{output_id}: copied ground truth is not canonical"
                    )
                final_rows.append(
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key not in {"prediction", "ground_truth"}
                        },
                        "evaluation_index": evaluation_index,
                        "prediction": {
                            "path": str(
                                (final_test / prediction_destination.name).resolve()
                            ),
                            "bytes": _require_exact_int(
                                row["prediction"].get("bytes"),
                                f"{output_id} prediction bytes",
                            ),
                            "sha256": str(row["prediction"]["sha256"]),
                        },
                        "ground_truth": {
                            "path": str(
                                (final_test / target_destination.name).resolve()
                            ),
                            "bytes": _require_exact_int(
                                row["ground_truth"].get("bytes"),
                                f"{output_id} ground-truth bytes",
                            ),
                            "sha256": str(row["ground_truth"]["sha256"]),
                        },
                    }
                )

            expected_final_files = {
                f"{prefix}_{row['canonical_clip_id']}.npz"
                for row in final_rows
                for prefix in ("res", "gt")
            }
            actual_final_files = {
                path.name
                for path in temporary_test.iterdir()
            }
            if actual_final_files != expected_final_files:
                raise InferenceContractError(
                    "unpublished final NPZ directory file set is not exact"
                )
            clip_manifest_payload = "".join(
                f"{row['canonical_clip_id']}\n"
                for row in final_rows
            ).encode("utf-8")
            final_manifest_payload = b"".join(
                canonical_json_bytes(row)
                for row in final_rows
            )
            final_lineage = {
                "format": "semtalk_show_base_inference_final_lineage_v1",
                "status": "complete",
                "contract": expected_contract,
                "contract_sha256": expected_contract_sha,
                "runtime": runtime_payload,
                "runtime_sha256": runtime_sha,
                "shards": shard_receipts,
                "final_manifest_sha256": hashlib.sha256(
                    final_manifest_payload
                ).hexdigest(),
                "clip_manifest_sha256": hashlib.sha256(
                    clip_manifest_payload
                ).hexdigest(),
            }
            final_summary = {
                "format": "semtalk_show_base_inference_final_summary_v1",
                "status": "complete",
                "generator": "SemTalk Base-only",
                "test_clips": len(final_rows),
                "prediction_files": len(final_rows),
                "ground_truth_files": len(final_rows),
                "num_shards": args.num_shards,
                "npz_root": str(final_test.resolve()),
                "manifest_sha256": final_lineage["final_manifest_sha256"],
                "clip_manifest_sha256": final_lineage[
                    "clip_manifest_sha256"
                ],
                "lineage_sha256": canonical_json_sha256(final_lineage),
                "contract_sha256": expected_contract_sha,
                "runtime_sha256": runtime_sha,
                "finite": True,
                "exact_once": True,
                "split_disjoint": True,
            }

            _write_bytes_fsync_new(
                temporary_root / "diffsheg_eval_clip_ids.txt",
                clip_manifest_payload,
            )
            _write_bytes_fsync_new(
                temporary_root / "final_manifest.jsonl",
                final_manifest_payload,
            )
            _write_bytes_fsync_new(
                temporary_root / "final_lineage.json",
                canonical_json_bytes(final_lineage),
            )
            metadata_expectations = {
                temporary_root / "diffsheg_eval_clip_ids.txt": hashlib.sha256(
                    clip_manifest_payload
                ).hexdigest(),
                temporary_root / "final_manifest.jsonl": hashlib.sha256(
                    final_manifest_payload
                ).hexdigest(),
                temporary_root / "final_lineage.json": canonical_json_sha256(
                    final_lineage
                ),
            }
            for metadata_path, metadata_sha in metadata_expectations.items():
                _verify_file_sha(
                    metadata_path,
                    metadata_sha,
                    "unpublished final metadata",
                )
            _fsync_directory(temporary_test)
            _fsync_directory(temporary_root / "npz")
            _revalidate_frozen_inputs(
                args,
                inputs,
                checkpoint_receipts,
                output_rows=ordered_rows,
            )
            # This summary is the only completion marker and is the final file write.
            _write_bytes_fsync_new(
                temporary_root / "final_summary.json",
                canonical_json_bytes(final_summary),
            )
            _fsync_directory(temporary_root)
            _fsync_directory(args.output_root)
            os.rename(temporary_root, final_root)
            published = True
            _fsync_directory(args.output_root)
            return final_summary
        except BaseException:
            if not published:
                shutil.rmtree(temporary_root, ignore_errors=True)
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-summary-json", type=Path, required=True)
    parser.add_argument("--canonical-lineage-json", type=Path, required=True)
    parser.add_argument(
        "--expected-canonical-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--audio-manifest",
        type=Path,
        nargs="+",
        required=True,
        help="Exactly eight frozen SHOW-test audio feature manifests.",
    )
    parser.add_argument(
        "--audio-summary-json",
        type=Path,
        nargs="+",
        required=True,
        help="Exactly one complete summary per audio manifest.",
    )
    parser.add_argument(
        "--audio-lineage-json",
        type=Path,
        nargs="+",
        required=True,
        help="Exactly one complete lineage receipt per audio manifest.",
    )
    parser.add_argument(
        "--base-training-lineage-manifest",
        type=Path,
        help=(
            "Frozen lineage bound only to a SHOW-trained Base checkpoint; "
            "forbidden for fully released zero-shot inference."
        ),
    )
    parser.add_argument(
        "--base-training-summary-json",
        type=Path,
        help=(
            "Frozen SHOW Base LMDB summary; forbidden for fully released "
            "zero-shot inference."
        ),
    )
    parser.add_argument(
        "--representation-training-lineage-manifest",
        type=Path,
        help=(
            "Frozen lineage bound only to face/upper/hands/lower/global "
            "representation checkpoints; forbidden for fully released "
            "zero-shot inference."
        ),
    )
    parser.add_argument(
        "--prerequisite-source",
        choices=CHECKPOINT_SOURCES,
        default=SHOW_TRAINED_CHECKPOINT_SOURCE,
        help="Source contract for face/upper/hands/lower/global weights.",
    )
    parser.add_argument(
        "--base-checkpoint-source",
        choices=CHECKPOINT_SOURCES,
        default=SHOW_TRAINED_CHECKPOINT_SOURCE,
        help="Independent source contract for the Base checkpoint.",
    )
    parser.add_argument(
        "--expected-inference-script-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-source-commit",
        required=True,
    )
    parser.add_argument(
        "--expected-source-tree",
        required=True,
    )
    parser.add_argument("--expected-training-source-commit")
    parser.add_argument("--expected-training-source-tree")
    parser.add_argument("--expected-base-training-source-commit")
    parser.add_argument("--expected-base-training-source-tree")
    parser.add_argument("--expected-transfer-training-source-commit")
    parser.add_argument("--expected-transfer-training-source-tree")
    parser.add_argument("--expected-input-source-commit")
    parser.add_argument("--expected-input-source-tree")
    parser.add_argument(
        "--released-cross-domain-gate-json",
        type=Path,
        help=(
            "Strict cross-domain authorization gate required only when Base "
            "and all five representation checkpoints use the official "
            "BEAT2 All-Speakers release."
        ),
    )
    parser.add_argument(
        "--expected-released-cross-domain-gate-sha256",
        help="External SHA-256 trust root for the released cross-domain gate.",
    )
    parser.add_argument(
        "--expected-canonical-source-commit",
        required=True,
    )
    parser.add_argument(
        "--expected-canonical-source-tree",
        required=True,
    )
    parser.add_argument(
        "--expected-hubert-tree-sha256",
        required=True,
    )
    for stage in CHECKPOINT_STAGES:
        parser.add_argument(
            f"--{stage}-checkpoint",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--expected-{stage}-sha256",
            required=True,
        )
        parser.add_argument(
            f"--{stage}-status-json",
            type=Path,
        )
    parser.add_argument(
        "--base-candidate-manifest",
        type=Path,
        help=(
            "Immutable manifest v2 proving that --base-checkpoint is one of "
            "the 40 formal Base candidates."
        ),
    )
    parser.add_argument(
        "--expected-base-candidate-manifest-sha256",
        help="Expected SHA-256 for --base-candidate-manifest.",
    )
    parser.add_argument(
        "--expected-base-formal-status-sha256",
        help=(
            "External expected SHA-256 for the complete Base formal-status "
            "JSON."
        ),
    )
    parser.add_argument(
        "--expected-base-final-checkpoint-sha256",
        help=(
            "External expected SHA-256 for the completed final Base "
            "checkpoint."
        ),
    )
    parser.add_argument(
        "--base-frozen-inputs-json",
        type=Path,
        help=(
            "Hash-pinned frozen_inputs.json emitted by "
            "train_base_official_adapt.py."
        ),
    )
    parser.add_argument(
        "--expected-base-frozen-inputs-sha256",
        help="External SHA-256 trust root for Base frozen_inputs.json.",
    )
    parser.add_argument(
        "--expected-face-status-sha256",
        help="External SHA-256 trust root for the Face transfer summary.",
    )
    parser.add_argument(
        "--expected-global-status-sha256",
        help="External SHA-256 trust root for the Global transfer summary.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=EXPECTED_NUM_SHARDS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Verify and merge eight already-complete inference shards.",
    )
    args = parser.parse_args(argv)
    if args.num_shards != EXPECTED_NUM_SHARDS:
        parser.error(f"--num-shards must be exactly {EXPECTED_NUM_SHARDS}")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, --num-shards)")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if len(args.audio_manifest) != EXPECTED_NUM_SHARDS:
        parser.error("--audio-manifest must provide exactly eight paths")
    if len(args.audio_summary_json) != EXPECTED_NUM_SHARDS:
        parser.error("--audio-summary-json must provide exactly eight paths")
    if len(args.audio_lineage_json) != EXPECTED_NUM_SHARDS:
        parser.error("--audio-lineage-json must provide exactly eight paths")
    args.expected_canonical_manifest_sha256 = _require_sha256(
        args.expected_canonical_manifest_sha256,
        "--expected-canonical-manifest-sha256",
    )
    args.expected_inference_script_sha256 = _require_sha256(
        args.expected_inference_script_sha256,
        "--expected-inference-script-sha256",
    )
    args.expected_source_commit = _require_git_oid(
        args.expected_source_commit,
        "--expected-source-commit",
    )
    args.expected_source_tree = _require_git_oid(
        args.expected_source_tree,
        "--expected-source-tree",
    )
    source_pairs = (
        (
            "training",
            "expected_training_source_commit",
            "expected_training_source_tree",
        ),
        (
            "base-training",
            "expected_base_training_source_commit",
            "expected_base_training_source_tree",
        ),
        (
            "transfer-training",
            "expected_transfer_training_source_commit",
            "expected_transfer_training_source_tree",
        ),
        (
            "input",
            "expected_input_source_commit",
            "expected_input_source_tree",
        ),
    )
    release_mode = (
        args.prerequisite_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
        or args.base_checkpoint_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    )
    fully_released_zero_shot = (
        args.prerequisite_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
        and args.base_checkpoint_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    )
    official_show_adapt = (
        args.prerequisite_source
        == OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
        and args.base_checkpoint_source
        == OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
    )
    if (
        OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
        in {args.prerequisite_source, args.base_checkpoint_source}
        and not official_show_adapt
    ):
        parser.error(
            "official_show_adapt_v1 must be selected for both Base and "
            "representation prerequisites"
        )
    for label, commit_name, tree_name in source_pairs:
        commit = getattr(args, commit_name)
        tree = getattr(args, tree_name)
        if (commit is None) != (tree is None):
            parser.error(
                f"--expected-{label}-source-commit and "
                f"--expected-{label}-source-tree must be supplied together"
            )

    if official_show_adapt:
        if args.expected_training_source_commit is not None:
            parser.error(
                "official_show_adapt_v1 forbids the ambiguous generic "
                "SHOW training source receipt; supply distinct Base-training "
                "and transfer-training source receipts"
            )
        required_official_sources = (
            (
                "base-training",
                "expected_base_training_source_commit",
            ),
            (
                "transfer-training",
                "expected_transfer_training_source_commit",
            ),
            ("input", "expected_input_source_commit"),
        )
        for label, commit_name in required_official_sources:
            if getattr(args, commit_name) is None:
                parser.error(
                    "official_show_adapt_v1 requires explicit "
                    f"--expected-{label}-source-commit and "
                    f"--expected-{label}-source-tree"
                )
    else:
        for label, commit_name, _ in source_pairs[1:3]:
            if getattr(args, commit_name) is not None:
                parser.error(
                    f"--expected-{label}-source-commit/tree is restricted "
                    "to official_show_adapt_v1"
                )
        if fully_released_zero_shot:
            if args.expected_training_source_commit is not None:
                parser.error(
                    "fully_released_zero_shot_v1 forbids a SHOW training "
                    "source receipt"
                )
        elif args.expected_training_source_commit is None:
            if release_mode:
                parser.error(
                    "non-scratch inference requires explicit "
                    "--expected-training-source-commit and "
                    "--expected-training-source-tree"
                )
            args.expected_training_source_commit = args.expected_source_commit
            args.expected_training_source_tree = args.expected_source_tree
        if args.expected_input_source_commit is None:
            if release_mode:
                parser.error(
                    "non-scratch inference requires explicit "
                    "--expected-input-source-commit and "
                    "--expected-input-source-tree"
                )
            args.expected_input_source_commit = args.expected_source_commit
            args.expected_input_source_tree = args.expected_source_tree

    for label, commit_name, tree_name in source_pairs:
        commit = getattr(args, commit_name)
        tree = getattr(args, tree_name)
        if commit is None:
            continue
        if tree is None:
            parser.error(
                f"--expected-{label}-source-tree is unexpectedly absent"
            )
        setattr(
            args,
            commit_name,
            _require_git_oid(
                commit,
                f"--expected-{label}-source-commit",
            ),
        )
        setattr(
            args,
            tree_name,
            _require_git_oid(
                tree,
                f"--expected-{label}-source-tree",
            ),
        )
    args.expected_canonical_source_commit = _require_git_oid(
        args.expected_canonical_source_commit,
        "--expected-canonical-source-commit",
    )
    args.expected_canonical_source_tree = _require_git_oid(
        args.expected_canonical_source_tree,
        "--expected-canonical-source-tree",
    )
    args.expected_hubert_tree_sha256 = _require_sha256(
        args.expected_hubert_tree_sha256,
        "--expected-hubert-tree-sha256",
    )
    if args.expected_released_cross_domain_gate_sha256 is not None:
        args.expected_released_cross_domain_gate_sha256 = _require_sha256(
            args.expected_released_cross_domain_gate_sha256,
            "--expected-released-cross-domain-gate-sha256",
        )
    if (
        args.base_checkpoint_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
        and args.prerequisite_source
        != RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    ):
        parser.error(
            "released official Base requires all five official released "
            "representation prerequisites"
        )
    training_artifacts = {
        "Base training lineage": args.base_training_lineage_manifest,
        "Base training summary": args.base_training_summary_json,
        "representation training lineage": (
            args.representation_training_lineage_manifest
        ),
    }
    gate_trust_roots = {
        "released cross-domain gate": (
            args.released_cross_domain_gate_json
        ),
        "released cross-domain gate SHA": (
            args.expected_released_cross_domain_gate_sha256
        ),
    }
    official_adapt_artifacts = {
        "Base frozen inputs": args.base_frozen_inputs_json,
        "Base frozen inputs SHA": (
            args.expected_base_frozen_inputs_sha256
        ),
        "Base status JSON": args.base_status_json,
        "Base status SHA": args.expected_base_formal_status_sha256,
        "Base candidate manifest": args.base_candidate_manifest,
        "Base candidate manifest SHA": (
            args.expected_base_candidate_manifest_sha256
        ),
        "Face transfer summary": args.face_status_json,
        "Face transfer summary SHA": args.expected_face_status_sha256,
        "Global transfer summary": args.global_status_json,
        "Global transfer summary SHA": args.expected_global_status_sha256,
    }
    if official_show_adapt:
        adapt_paths = [
            args.output_root,
            args.base_frozen_inputs_json,
            args.base_status_json,
            args.base_candidate_manifest,
            args.face_status_json,
            args.global_status_json,
            *[
                getattr(args, f"{stage}_checkpoint")
                for stage in CHECKPOINT_STAGES
            ],
        ]
        forbidden_path = next(
            (
                value
                for value in adapt_paths
                if value is not None
                and (
                    "e30" in str(value).lower()
                    or "speaker2" in str(value).lower()
                )
            ),
            None,
        )
        if forbidden_path is not None:
            parser.error(
                "official_show_adapt_v1 path references withdrawn e30 or "
                f"forbidden Speaker2: {forbidden_path}"
            )
        unexpected_training = sorted(
            label
            for label, value in training_artifacts.items()
            if value is not None
        )
        missing_adapt = sorted(
            label
            for label, value in official_adapt_artifacts.items()
            if value is None
        )
        unexpected_gate = sorted(
            label for label, value in gate_trust_roots.items()
            if value is not None
        )
        if unexpected_training:
            parser.error(
                "official_show_adapt_v1 forbids legacy training artifacts: "
                + ", ".join(unexpected_training)
            )
        if missing_adapt:
            parser.error(
                "official_show_adapt_v1 requires "
                + ", ".join(missing_adapt)
            )
        if unexpected_gate:
            parser.error(
                "official_show_adapt_v1 forbids released zero-shot gate roots"
            )
        if args.expected_base_final_checkpoint_sha256 is not None:
            parser.error(
                "official_show_adapt_v1 forbids the scratch Base final "
                "checkpoint root"
            )
    elif fully_released_zero_shot:
        unexpected_training = sorted(
            label
            for label, value in training_artifacts.items()
            if value is not None
        )
        missing_gate = sorted(
            label for label, value in gate_trust_roots.items() if value is None
        )
        if unexpected_training:
            parser.error(
                "fully_released_zero_shot_v1 forbids "
                + ", ".join(unexpected_training)
            )
        if missing_gate:
            parser.error(
                "fully_released_zero_shot_v1 requires "
                + ", ".join(missing_gate)
            )
        if any(
            value is not None
            for value in (
                args.base_frozen_inputs_json,
                args.expected_base_frozen_inputs_sha256,
                args.expected_face_status_sha256,
                args.expected_global_status_sha256,
            )
        ):
            parser.error(
                "fully_released_zero_shot_v1 forbids official-adapt receipts"
            )
    else:
        missing_training = sorted(
            label for label, value in training_artifacts.items() if value is None
        )
        unexpected_gate = sorted(
            label for label, value in gate_trust_roots.items() if value is not None
        )
        if missing_training:
            parser.error(
                "SHOW-trained or mixed inference requires "
                + ", ".join(missing_training)
            )
        if unexpected_gate:
            parser.error(
                "released cross-domain gate is restricted to "
                "fully_released_zero_shot_v1"
            )
        if any(
            value is not None
            for value in (
                args.base_frozen_inputs_json,
                args.expected_base_frozen_inputs_sha256,
                args.expected_face_status_sha256,
                args.expected_global_status_sha256,
            )
        ):
            parser.error(
                "legacy inference modes forbid official-adapt-only receipts"
            )
    for name in (
        "expected_base_candidate_manifest_sha256",
        "expected_base_formal_status_sha256",
        "expected_base_final_checkpoint_sha256",
        "expected_base_frozen_inputs_sha256",
        "expected_face_status_sha256",
        "expected_global_status_sha256",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(
                args,
                name,
                _require_sha256(value, f"--{name.replace('_', '-')}"),
            )
    for stage in CHECKPOINT_STAGES:
        setattr(
            args,
            f"expected_{stage}_sha256",
            _require_sha256(
                getattr(args, f"expected_{stage}_sha256"),
                f"--expected-{stage}-sha256",
            ),
        )
    representation_stages = set(CHECKPOINT_STAGES) - {"base"}
    representation_statuses = {
        stage: getattr(args, f"{stage}_status_json")
        for stage in representation_stages
    }
    if args.prerequisite_source == SHOW_TRAINED_CHECKPOINT_SOURCE:
        missing = sorted(
            stage
            for stage, status in representation_statuses.items()
            if status is None
        )
        if missing:
            parser.error(
                "show_trained_v1 requires representation status JSON for "
                + ", ".join(missing)
            )
    elif (
        args.prerequisite_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    ):
        unexpected = sorted(
            stage
            for stage, status in representation_statuses.items()
            if status is not None
        )
        if unexpected:
            parser.error(
                "released_all_speakers_v1 forbids SHOW representation "
                "status JSON for "
                + ", ".join(unexpected)
            )
        for stage in sorted(representation_stages):
            specification = RELEASED_ALL_SPEAKERS_MODELS[stage]
            if (
                getattr(args, f"expected_{stage}_sha256")
                != specification["sha256"]
            ):
                parser.error(
                    f"--expected-{stage}-sha256 must equal the hard-pinned "
                    "official release SHA-256"
                )
            if Path(getattr(args, f"{stage}_checkpoint")).name != (
                specification["filename"]
            ):
                parser.error(
                    f"--{stage}-checkpoint basename must be "
                    f"{specification['filename']}"
                )
    else:
        expected_status_stages = {"face", "global"}
        observed_status_stages = {
            stage
            for stage, status in representation_statuses.items()
            if status is not None
        }
        if observed_status_stages != expected_status_stages:
            parser.error(
                "official_show_adapt_v1 requires status JSON exactly for "
                "Face and Global"
            )
        for stage in ("hands", "upper", "lower"):
            specification = RELEASED_ALL_SPEAKERS_MODELS[stage]
            if (
                getattr(args, f"expected_{stage}_sha256")
                != specification["sha256"]
                or Path(getattr(args, f"{stage}_checkpoint")).name
                != specification["filename"]
            ):
                parser.error(
                    f"official_show_adapt_v1 requires exact official {stage}"
                )
        for stage in ("face", "global"):
            if Path(getattr(args, f"{stage}_checkpoint")).name != (
                f"best_{stage}_transfer.bin"
            ):
                parser.error(
                    f"official_show_adapt_v1 requires "
                    f"best_{stage}_transfer.bin"
                )
    base_trust_roots = {
        "base status JSON": args.base_status_json,
        "Base candidate manifest": args.base_candidate_manifest,
        "Base candidate manifest SHA": (
            args.expected_base_candidate_manifest_sha256
        ),
        "Base formal status SHA": args.expected_base_formal_status_sha256,
        "Base final checkpoint SHA": (
            args.expected_base_final_checkpoint_sha256
        ),
    }
    if args.base_checkpoint_source == SHOW_TRAINED_CHECKPOINT_SOURCE:
        missing = sorted(
            label for label, value in base_trust_roots.items() if value is None
        )
        if missing:
            parser.error(
                "show_trained_v1 Base requires " + ", ".join(missing)
            )
    elif (
        args.base_checkpoint_source
        == RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
    ):
        unexpected = sorted(
            label for label, value in base_trust_roots.items() if value is not None
        )
        if unexpected:
            parser.error(
                "released_all_speakers_v1 Base forbids "
                + ", ".join(unexpected)
            )
        base_specification = RELEASED_ALL_SPEAKERS_MODELS["base"]
        if args.expected_base_sha256 != base_specification["sha256"]:
            parser.error(
                "--expected-base-sha256 must equal the hard-pinned official "
                "release SHA-256"
            )
        if Path(args.base_checkpoint).name != base_specification["filename"]:
            parser.error(
                "--base-checkpoint basename must be "
                f"{base_specification['filename']}"
            )
    else:
        if not Path(args.base_checkpoint).name.startswith(
            "base_official_adapt_epoch_"
        ) or Path(args.base_checkpoint).suffix != ".bin":
            parser.error(
                "official_show_adapt_v1 Base must be an immutable "
                "base_official_adapt_epoch_*.bin candidate"
            )
    args.output_root = args.output_root.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = finalize(args) if args.finalize else run_shard(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
