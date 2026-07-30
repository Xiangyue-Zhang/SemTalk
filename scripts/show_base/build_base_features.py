#!/usr/bin/env python3
"""Build audited SHOW audio caches and the train-only SemTalk Base LMDB.

There are deliberately only two modes:

``audio``
    Run the frozen HuBERT-large model on the *real 16 kHz* waveform and build
    SemTalk's three-channel rhythm/onset feature.  The mode is shardable and
    writes one immutable NPZ per canonical train or test clip plus an audited
    JSONL manifest.

``base``
    Load the four frozen RVQ checkpoints, reproduce the feature construction in
    ``PreprocessProcessor.process`` without importing its CLIP/semantic stack,
    and write the official 64-frame/20-frame-stride training LMDB.

This file must not grow a SemGate, Sparse, ASR, TextGrid, CLIP, emotion, or
semantic dependency.  ``in_word`` is an all-zero compatibility placeholder;
the SemTalk Base forward path does not consume it.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping
import zipfile

# Formal source receipts require the checkout to stay byte-for-byte clean.
sys.dont_write_bytecode = True

import numpy as np


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
CANONICAL_FIELDS = (
    "pose",
    "contact",
    "facial",
    "beta",
    "trans",
    "speaker_id",
)
RVQ_NAMES = ("face", "upper", "hands", "lower")
SHOW_TRAINED_PREREQUISITE_SOURCE = "show_trained_v1"
RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE = (
    "released_all_speakers_v1"
)
PREREQUISITE_SOURCES = (
    SHOW_TRAINED_PREREQUISITE_SOURCE,
    RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE,
)
RELEASED_ALL_SPEAKERS_CLASSIFICATION = (
    "official_BEAT2_All-Speakers_released_weights_not_SHOW-trained"
)
RELEASED_ALL_SPEAKERS_RELEASE_TRUST_ROOT = {
    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
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
RELEASED_ALL_SPEAKERS_MODELS = {
    "face": {
        "filename": "rvq_face_600.bin",
        "sha256": (
            "31b04c88456a25f4d57841c0cb507b4c856daccb3875878d"
            "06545110a6152127"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 106,
        "vae_layer": 2,
    },
    "hands": {
        "filename": "rvq_hands_500.bin",
        "sha256": (
            "08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a3805"
            "5ca47a539b03e436"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 180,
        "vae_layer": 2,
    },
    "upper": {
        "filename": "rvq_upper_500.bin",
        "sha256": (
            "05101461e75b4e9b687ef30437585d56969c6a13d0047b910"
            "00b31d88d08ac17"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 78,
        "vae_layer": 2,
    },
    "lower": {
        "filename": "rvq_lower_600.bin",
        "sha256": (
            "2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e6"
            "2f99c171ae4efce8"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 61,
        "vae_layer": 4,
    },
    "global": {
        "filename": "last_1700_foot.bin",
        "sha256": (
            "6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e"
            "127e19e0144ee12"
        ),
        "model_class": "VAEConvZero",
        "vae_test_dim": 61,
        "vae_layer": 4,
    },
}
SPEAKER_MAP = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
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
    "base_candidate_manifest",
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

# SMPL-X joint order in ``beat_smplx_joints``.  Keeping these explicit avoids
# importing dataloaders.data_tools, whose module-level imports pull unrelated
# text/evaluation packages into this Base-only cache builder.
UPPER_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
LOWER_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified_bytes(
    input_path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes, str]:
    """Read one immutable snapshot and verify the bytes that will be decoded."""
    if input_path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {input_path}")
    path = input_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch for {path}: "
            f"{actual} != {expected_sha256}"
        )
    return path, payload, actual


def canonical_file_payload_sha256(payload: Any) -> str:
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


def compact_payload_sha256(payload: Any) -> str:
    """Match show_base_train._payload_sha256 for training receipts."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return normalized


def require_git_oid(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{label} must be a lowercase Git object ID")
    return normalized


def require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{label} must be an exact integer")
    return value


def require_exact_int_mapping(
    value: Any,
    expected: dict[str, int],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimeError(f"{label} must have exactly {sorted(expected)}")
    for key, expected_value in expected.items():
        if (
            require_exact_int(value[key], f"{label}.{key}")
            != expected_value
        ):
            raise RuntimeError(
                f"{label}.{key} must equal {expected_value}"
            )
    return value


def validate_lower_target_cache_binding(
    *,
    formal_stage: str,
    audit: Mapping[str, Any],
    dataset_receipt: Mapping[str, Any],
    status: Mapping[str, Any],
    path: Path,
) -> None:
    """Require the frozen lower-target receipt on lower and forbid it elsewhere."""

    key = LOWER_TARGET_CACHE_RECEIPT_KEY
    if formal_stage != "lower":
        if (
            key in audit
            or key in dataset_receipt
            or key in status
        ):
            raise RuntimeError(
                f"{path}: non-lower stage carries a lower target cache receipt"
            )
        return

    audit_receipt = audit.get(key)
    dataset_cache_receipt = dataset_receipt.get(key)
    status_receipt = status.get(key)
    if (
        type(audit_receipt) is not dict
        or audit_receipt != dataset_cache_receipt
        or audit_receipt != status_receipt
        or set(audit_receipt) != LOWER_TARGET_CACHE_RECEIPT_KEYS
    ):
        raise RuntimeError(
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
            or not np.isfinite(float(formal_gate[field]))
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
        or receipt_sha != canonical_file_payload_sha256(receipt_without_sha)
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
        raise RuntimeError(
            f"{path}: invalid lower target cache receipt contract"
        )


def validate_lower_target_backend_binding(
    *,
    formal_stage: str,
    audit: Mapping[str, Any],
    dataset_receipt: Mapping[str, Any],
    status: Mapping[str, Any],
    path: Path,
) -> None:
    """Require formal lower's live SMPL-X receipt and reject cache/live mixing."""

    backend_key = LOWER_TARGET_BACKEND_RECEIPT_KEY
    cache_key = LOWER_TARGET_CACHE_RECEIPT_KEY
    if formal_stage != "lower":
        if any(
            key in payload
            for key in (backend_key, cache_key)
            for payload in (audit, dataset_receipt, status)
        ):
            raise RuntimeError(
                f"{path}: non-lower stage carries a lower target receipt"
            )
        return

    if any(
        cache_key in payload
        for payload in (audit, dataset_receipt, status)
    ):
        raise RuntimeError(
            f"{path}: formal lower cannot mix cache and live backend receipts"
        )
    audit_receipt = audit.get(backend_key)
    dataset_backend_receipt = dataset_receipt.get(backend_key)
    status_receipt = status.get(backend_key)
    if (
        type(audit_receipt) is not dict
        or audit_receipt != dataset_backend_receipt
        or audit_receipt != status_receipt
        or set(audit_receipt) != LOWER_TARGET_BACKEND_RECEIPT_KEYS
    ):
        raise RuntimeError(
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
        != sha256(target_forward_source)
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
        or receipt_sha != compact_payload_sha256(receipt_without_sha)
    ):
        raise RuntimeError(
            f"{path}: invalid lower live SMPL-X backend receipt contract"
        )


def source_receipt(
    expected_commit: str,
    expected_tree: str,
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]

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
            "feature builder requires a clean source checkout; first change: "
            f"{status.splitlines()[0]}"
        )
    receipt = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
    }
    if receipt["origin"] != EXPECTED_ORIGIN:
        raise RuntimeError(
            f"feature source origin {receipt['origin']!r} "
            f"!= {EXPECTED_ORIGIN!r}"
        )
    if receipt["commit"] != expected_commit:
        raise RuntimeError(
            f"feature source commit {receipt['commit']} != {expected_commit}"
        )
    if receipt["tree"] != expected_tree:
        raise RuntimeError(
            f"feature source tree {receipt['tree']} != {expected_tree}"
        )
    return receipt


def input_artifact_source_receipt(
    expected_commit: str,
    expected_tree: str,
) -> dict[str, str]:
    """Describe the already-built audio/representation artifact producer."""
    return {
        "format": "semtalk_show_input_artifact_source_v1",
        "origin": EXPECTED_ORIGIN,
        "commit": expected_commit,
        "tree": expected_tree,
    }


def normalize_input_artifact_source_args(
    args: argparse.Namespace,
) -> None:
    """Resolve the Base input producer, defaulting to the builder source."""
    if args.mode != "base":
        return
    commit = args.expected_input_source_commit
    tree = args.expected_input_source_tree
    if (commit is None) != (tree is None):
        raise ValueError(
            "--expected-input-source-commit and "
            "--expected-input-source-tree must be supplied together"
        )
    if commit is None:
        commit = args.expected_source_commit
        tree = args.expected_source_tree
    args.expected_input_source_commit = require_git_oid(
        commit,
        "--expected-input-source-commit",
    )
    args.expected_input_source_tree = require_git_oid(
        tree,
        "--expected-input-source-tree",
    )


def load_canonical_receipt(
    *,
    manifest_paths: list[Path],
    manifest_hashes: dict[str, str],
    summary_path: Path,
    lineage_path: Path,
    expected_manifest_sha256: str,
    expected_summary_sha256: str,
    expected_lineage_sha256: str,
    expected_canonical_source_commit: str,
    expected_canonical_source_tree: str,
) -> dict[str, Any]:
    if len(manifest_paths) != 1:
        raise RuntimeError("formal feature build requires one final manifest")
    input_manifest = manifest_paths[0]
    for path in (input_manifest, summary_path, lineage_path):
        if path.is_symlink():
            raise RuntimeError(f"formal receipt must not be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = input_manifest.resolve()
    summary_path = summary_path.resolve()
    lineage_path = lineage_path.resolve()
    manifest_sha = manifest_hashes[str(manifest)]
    if manifest_sha != expected_manifest_sha256:
        raise RuntimeError(
            f"canonical manifest SHA {manifest_sha} "
            f"!= {expected_manifest_sha256}"
        )
    summary_bytes = summary_path.read_bytes()
    lineage_bytes = lineage_path.read_bytes()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    lineage_sha = hashlib.sha256(lineage_bytes).hexdigest()
    if summary_sha != expected_summary_sha256:
        raise RuntimeError(
            f"canonical summary SHA {summary_sha} "
            f"!= {expected_summary_sha256}"
        )
    if lineage_sha != expected_lineage_sha256:
        raise RuntimeError(
            f"canonical lineage SHA {lineage_sha} "
            f"!= {expected_lineage_sha256}"
        )
    summary = json.loads(summary_bytes.decode("utf-8"))
    lineage = json.loads(lineage_bytes.decode("utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "complete"
        or summary.get("schema_name") != "semtalk-show-canonical-motion"
        or require_exact_int(
            summary.get("schema_version"),
            "canonical summary schema_version",
        )
        != 1
        or summary.get("manifest_sha256") != manifest_sha
        or require_exact_int_mapping(
            summary.get("split_counts"),
            {"train": 13_687, "val": 1_715, "test": 1_708},
            "canonical summary split_counts",
        )
        != {"train": 13_687, "val": 1_715, "test": 1_708}
        or require_exact_int(
            summary.get("clip_count"),
            "canonical summary clip_count",
        )
        != 17_110
        or summary.get("exact_once") is not True
        or summary.get("finite") is not True
        or summary.get("split_disjoint") is not True
    ):
        raise RuntimeError("canonical summary is not a complete formal receipt")
    canonical_contract = (
        lineage.get("lineage_contract")
        if isinstance(lineage, dict)
        else None
    )
    if (
        not isinstance(lineage, dict)
        or lineage.get("final_manifest_sha256") != manifest_sha
        or canonical_file_payload_sha256(lineage)
        != summary.get("lineage_sha256")
        or lineage.get("lineage_contract_sha256")
        != summary.get("lineage_contract_sha256")
        or not isinstance(canonical_contract, dict)
        or canonical_file_payload_sha256(canonical_contract)
        != lineage.get("lineage_contract_sha256")
    ):
        raise RuntimeError("canonical summary/lineage binding mismatch")
    if (
        canonical_contract.get("source_audio_sample_rate")
        != CANONICAL_SOURCE_AUDIO_RATE
        or canonical_contract.get("hubert_target_sample_rate")
        != CANONICAL_HUBERT_TARGET_RATE
        or canonical_contract.get("audio_channel_protocol")
        != CANONICAL_AUDIO_CHANNEL_PROTOCOL
    ):
        raise RuntimeError("canonical audio channel/rate protocol mismatch")
    canonical_source = canonical_contract.get("source_receipt")
    if (
        not isinstance(canonical_source, dict)
        or canonical_source.get("origin") != EXPECTED_ORIGIN
        or canonical_source.get("commit")
        != expected_canonical_source_commit
        or canonical_source.get("tree") != expected_canonical_source_tree
        or summary.get("source_receipt_sha256")
        != canonical_file_payload_sha256(canonical_source)
    ):
        raise RuntimeError("canonical source receipt mismatch")
    return {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "summary": str(summary_path),
        "summary_sha256": summary_sha,
        "lineage": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "lineage_contract_sha256": summary["lineage_contract_sha256"],
        "source_receipt": canonical_source,
    }


def tree_sha256(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash a local model directory while rejecting every symlink."""
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(root)
    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    entries = sorted(root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise RuntimeError(f"model tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"model tree contains a non-file: {path}")
        relative = path.relative_to(root).as_posix()
        file_sha = sha256(path)
        size = path.stat().st_size
        record = {"path": relative, "bytes": size, "sha256": file_sha}
        records.append(record)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(file_sha))
    if not records:
        raise RuntimeError(f"model directory is empty: {root}")
    return aggregate.hexdigest(), records


def _fsync_parent(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_new_file(tmp: Path, path: Path) -> None:
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing file: {path}"
        ) from exc
    tmp.unlink()
    _fsync_parent(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new_file(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new_file(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        with tmp.open("wb") as handle:
            handle.write(deterministic_npz_bytes(arrays))
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new_file(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _npy_bytes(array: np.ndarray) -> bytes:
    with io.BytesIO() as handle:
        np.lib.format.write_array(
            handle,
            np.ascontiguousarray(array),
            allow_pickle=False,
        )
        return handle.getvalue()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Encode an uncompressed NPZ with fixed metadata and exact values."""
    with io.BytesIO() as handle:
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name, array in arrays.items():
                info = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    _npy_bytes(np.asarray(array)),
                    compress_type=zipfile.ZIP_STORED,
                )
        return handle.getvalue()


def refuse_existing(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing outputs: " + ", ".join(existing)
        )


def forbid_nested_output(container: Path, *artifacts: Path) -> None:
    for artifact in artifacts:
        if artifact == container or artifact.is_relative_to(container):
            raise ValueError(
                "sidecar artifact must not be nested in atomic directory "
                f"{container}: {artifact}"
            )


def runtime_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover - only an audit fallback
        record["torch_import_error"] = repr(exc)
    try:
        import transformers

        record["transformers"] = transformers.__version__
    except Exception as exc:  # pragma: no cover - only an audit fallback
        record["transformers_import_error"] = repr(exc)
    try:
        import librosa

        record["librosa"] = librosa.__version__
    except Exception as exc:  # pragma: no cover - only an audit fallback
        record["librosa_import_error"] = repr(exc)
    return record


def load_jsonl(
    paths: list[Path],
    *,
    expected_single_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if expected_single_sha256 is not None and len(paths) != 1:
        raise RuntimeError(
            "a frozen canonical manifest SHA requires exactly one manifest"
        )
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"formal manifest must not be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        payload = resolved.read_bytes()
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        if (
            expected_single_sha256 is not None
            and payload_sha256 != expected_single_sha256
        ):
            raise RuntimeError(
                f"canonical manifest SHA {payload_sha256} "
                f"!= {expected_single_sha256}"
            )
        hashes[str(resolved)] = payload_sha256
        text = payload.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(
                    f"{resolved}:{line_number}: expected JSON object"
                )
            rows.append(row)
    return rows, hashes


def canonical_split_rows(
    paths: list[Path],
    split: str,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if split not in {"train", "test"}:
        raise ValueError(f"unsupported canonical split {split!r}")
    rows, hashes = load_jsonl(
        paths,
        expected_single_sha256=expected_manifest_sha256,
    )
    split_rows = [row for row in rows if row.get("split") == split]
    required = {
        "clip_id",
        "canonical_npz",
        "canonical_npz_sha256",
        "source_wav",
        "source_wav_sha256",
        "lineage_contract_sha256",
        "speaker",
        "speaker_id",
        "frames",
        "wav_channels",
        "wav_sample_width",
        "wav_sample_rate",
        "wav_frames",
        "wav_mono_policy",
    }
    for row in split_rows:
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(
                f"canonical row {row.get('clip_id')!r} missing {missing}"
            )
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or speaker not in SPEAKER_MAP:
            raise RuntimeError(
                f"canonical row {row.get('clip_id')!r} has invalid speaker"
            )
        speaker_id = require_exact_int(
            row.get("speaker_id"),
            f"canonical row {row.get('clip_id')!r} speaker_id",
        )
        if speaker_id != SPEAKER_MAP[speaker]:
            raise RuntimeError(
                f"canonical row {row.get('clip_id')!r} speaker/name mismatch"
            )
        if (
            require_exact_int(
                row.get("frames"),
                f"canonical row {row.get('clip_id')!r} frames",
            )
            <= 0
        ):
            raise RuntimeError(
                f"canonical row {row.get('clip_id')!r} has invalid frames"
            )
        validate_canonical_audio_metadata(
            row,
            f"canonical row {row.get('clip_id')!r}",
        )
    split_rows.sort(key=lambda row: (row["clip_id"], row["canonical_npz"]))
    ids = [str(row["clip_id"]) for row in split_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"duplicate {split} clip_id across canonical manifests"
        )
    if not split_rows:
        raise RuntimeError(
            f"canonical manifests contain no {split} clips"
        )
    lineage_contracts = {
        str(row["lineage_contract_sha256"]) for row in split_rows
    }
    if len(lineage_contracts) != 1:
        raise RuntimeError(
            f"canonical {split} rows disagree on "
            "lineage_contract_sha256"
        )
    return split_rows, hashes


def validate_canonical_audio_metadata(
    row: dict[str, Any],
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
        raise RuntimeError(f"{context}: invalid canonical WAV metadata")


def canonical_frames(row: dict[str, Any]) -> int:
    path, payload, _ = read_verified_bytes(
        Path(row["canonical_npz"]),
        str(row["canonical_npz_sha256"]),
        "canonical NPZ",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if "pose" not in archive.files:
            raise RuntimeError(f"{path}: missing pose")
        pose = archive["pose"]
        if pose.ndim != 2 or pose.shape[1] != 165:
            raise RuntimeError(f"{path}: pose must be [T,165], got {pose.shape}")
        frames = int(pose.shape[0])
        if pose.dtype.kind not in "fc" or not np.isfinite(pose).all():
            raise RuntimeError(f"{path}: pose is non-finite or non-floating")
    if frames <= 0:
        raise RuntimeError(f"{path}: empty canonical clip")
    if (
        row.get("frames") is not None
        and require_exact_int(row["frames"], "canonical row frames")
        != frames
    ):
        raise RuntimeError(f"{path}: frame count disagrees with manifest")
    return frames


def _forward_rolling_max(values: np.ndarray, width: int) -> np.ndarray:
    """Exact O(N) forward-window maximum via prefix/suffix block maxima."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size < width:
        raise ValueError("rolling maximum input is shorter than its window")
    padded_size = ((values.size + width - 1) // width) * width
    padded = np.full(padded_size, -np.inf, dtype=values.dtype)
    padded[: values.size] = values
    blocks = padded.reshape(-1, width)
    prefix = np.maximum.accumulate(blocks, axis=1).reshape(-1)
    suffix = np.maximum.accumulate(
        blocks[:, ::-1], axis=1
    )[:, ::-1].reshape(-1)
    count = values.size - width + 1
    return np.maximum(suffix[:count], prefix[width - 1 : width - 1 + count])


def align_public_audio_prefix(
    native: np.ndarray,
    *,
    target_frames: int,
    max_frame_mismatch: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Match the public loader's shortest-whole-second leading prefix.

    Public SemTalk extracts audio features from the complete source waveform,
    computes the shortest whole-second duration across modalities, and samples
    all modalities from frame zero.  Consequently, a longer audio container is
    prefix-truncated after feature extraction; it must never be time-compressed
    to the shorter pose duration.

    A sub-frame boundary can leave audio one frame short.  Such a shortfall is
    edge-padded only when it cannot remove a whole second that the canonical
    motion would otherwise contribute to training.
    """
    values = np.asarray(native)
    if values.ndim != 2 or values.shape[0] <= 0:
        raise RuntimeError("native audio feature must be non-empty [T,D]")
    if target_frames <= 0:
        raise RuntimeError("canonical audio target must be positive")
    if max_frame_mismatch < 0:
        raise RuntimeError("audio mismatch allowance must be non-negative")
    if values.dtype.kind in "fc" and not np.isfinite(values).all():
        raise RuntimeError("native audio feature contains non-finite values")

    native_frames = int(values.shape[0])
    canonical_usable_frames = (target_frames // 30) * 30
    if native_frames >= target_frames:
        aligned = values[:target_frames].copy()
    else:
        shortfall = target_frames - native_frames
        if (
            shortfall > max_frame_mismatch
            or native_frames < canonical_usable_frames
        ):
            raise RuntimeError(
                "audio/canonical prefix mismatch exceeds public-loader gate: "
                f"audio={native_frames}, canonical={target_frames}, "
                f"allowed_shortfall={max_frame_mismatch}, "
                f"canonical_usable={canonical_usable_frames}"
            )
        aligned = np.concatenate(
            [
                values,
                np.repeat(values[-1:], shortfall, axis=0),
            ],
            axis=0,
        )
    if aligned.shape != (target_frames, values.shape[1]):
        raise AssertionError("internal public audio-prefix alignment mismatch")
    return aligned, {
        "canonical_frames": int(target_frames),
        "canonical_usable_frames": canonical_usable_frames,
        "native_30fps_frames": native_frames,
        "discarded_source_30fps_frames": max(
            native_frames - target_frames,
            0,
        ),
        "edge_padded_tail_frames": max(
            target_frames - native_frames,
            0,
        ),
    }


def semtalk_rhythm_features(
    speech_16k: np.ndarray,
    *,
    target_frames: int,
    max_frame_mismatch: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reproduce ``extract_rhythm_pause_features`` without its ASR imports."""
    import librosa

    target_sr = 16000
    frame_length = 1024
    hop_length = 512
    speech = np.asarray(speech_16k, dtype=np.float32)
    if speech.ndim != 1 or speech.size < frame_length:
        raise RuntimeError("16 kHz speech is too short for rhythm extraction")
    if not np.isfinite(speech).all():
        raise RuntimeError("16 kHz speech contains non-finite values")

    # This is value-identical to the repository's stride-trick rolling view
    # followed by max(abs(...), axis=1), but avoids O(N*1024) work.
    envelope_full = _forward_rolling_max(np.abs(speech), frame_length)
    feature_frames = speech.size // hop_length
    amplitude_envelope = envelope_full[:feature_frames]

    energy = np.asarray(
        [
            np.sum(np.abs(speech[i : i + frame_length] ** 2))
            for i in range(0, speech.size, hop_length)
        ],
        dtype=np.float32,
    )[: amplitude_envelope.size]

    onset_frames = librosa.onset.onset_detect(
        y=speech,
        sr=target_sr,
        hop_length=hop_length,
        units="frames",
    )
    onset_array = np.zeros(amplitude_envelope.size, dtype=np.float64)
    valid_onsets = onset_frames[onset_frames < onset_array.size]
    onset_array[valid_onsets] = 1.0

    features = np.stack(
        [amplitude_envelope, energy, onset_array],
        axis=1,
    )
    native_30fps = int((speech.size / target_sr) * 30)
    if native_30fps <= 0:
        raise RuntimeError("audio duration yields no 30 fps feature frames")
    # The stock implementation first resamples the complete source feature to
    # floor(duration*30).  The public loader later samples the leading prefix
    # shared with pose/facial data; it does not compress a longer audio track.
    native = np.empty((native_30fps, 3), dtype=np.float64)
    native_x = np.linspace(0, features.shape[0] - 1, native_30fps)
    source_x = np.arange(features.shape[0])
    for channel in range(3):
        native[:, channel] = np.interp(
            native_x,
            source_x,
            features[:, channel],
        )
    aligned, timing = align_public_audio_prefix(
        native,
        target_frames=target_frames,
        max_frame_mismatch=max_frame_mismatch,
    )
    aligned = aligned.astype(np.float32, copy=False)
    if aligned.shape != (target_frames, 3) or not np.isfinite(aligned).all():
        raise RuntimeError("invalid aligned rhythm feature")
    return aligned, {
        "audio_samples_16k": int(speech.size),
        **timing,
    }


def hubert_long(
    model: Any,
    feature_extractor: Any,
    speech_16k: np.ndarray,
    *,
    device: str,
) -> Any:
    """Repository ``get_hubert_from_16k_speech_long`` on actual 16 kHz input."""
    import torch

    values = feature_extractor(
        speech_16k,
        return_tensors="pt",
        sampling_rate=16000,
    ).input_values
    if values.ndim != 2 or values.shape[0] != 1:
        raise RuntimeError(f"unexpected HuBERT processor shape {values.shape}")
    values = values.to(device)
    kernel = 400
    stride = 320
    if values.shape[1] < kernel:
        raise RuntimeError("waveform is shorter than HuBERT's 400-sample kernel")
    clip_length = stride * 1000
    num_iter = values.shape[1] // clip_length
    expected_frames = (values.shape[1] - (kernel - stride)) // stride
    chunks = []
    for index in range(num_iter):
        if index == 0:
            start = 0
            end = clip_length - stride + kernel
        else:
            start = clip_length * index
            end = start + clip_length - stride + kernel
        hidden = model(values[:, start:end]).last_hidden_state
        chunks.append(hidden[0])
    remainder = (
        values[:, clip_length * num_iter :] if num_iter > 0 else values
    )
    if remainder.shape[1] >= kernel:
        chunks.append(model(remainder).last_hidden_state[0])
    if not chunks:
        raise RuntimeError("HuBERT produced no chunks")
    result = torch.cat(chunks, dim=0)
    if abs(result.shape[0] - expected_frames) > 1:
        raise RuntimeError(
            f"HuBERT frames {result.shape[0]} != expected {expected_frames}"
        )
    if result.shape[0] < expected_frames:
        result = torch.nn.functional.pad(
            result,
            (0, 0, 0, expected_frames - result.shape[0]),
        )
    else:
        result = result[:expected_frames]
    if result.ndim != 2 or result.shape[1] != 1024:
        raise RuntimeError(f"HuBERT output must be [T,1024], got {result.shape}")
    return result


def align_hubert(hidden: Any, target_frames: int) -> np.ndarray:
    import torch

    aligned = torch.nn.functional.interpolate(
        hidden.transpose(0, 1).unsqueeze(0),
        size=target_frames,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)[0]
    result = aligned.detach().float().cpu().numpy()
    if result.shape != (target_frames, 1024):
        raise RuntimeError(f"aligned HuBERT shape is {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("aligned HuBERT contains non-finite values")
    return result


def _safe_clip_filename(clip_id: str) -> str:
    digest = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def audio_mode(args: argparse.Namespace) -> None:
    import librosa
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0,num-shards)")
    if args.max_frame_mismatch != 1:
        raise ValueError("--max-frame-mismatch must be exactly one")
    formal_expected = {"train": 13_687, "test": 1_708}[args.split]
    if args.expected_total_clips != formal_expected:
        raise RuntimeError(
            f"formal {args.split} audio clip count must be {formal_expected}"
        )

    output_dir = Path(args.output_dir).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    summary_path = Path(args.summary_json).resolve()
    lineage_path = Path(args.lineage_json).resolve()
    forbid_nested_output(
        output_dir,
        output_manifest,
        summary_path,
        lineage_path,
    )
    refuse_existing(output_dir, output_manifest, summary_path, lineage_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_paths = [Path(path) for path in args.canonical_manifest]
    canonical_summary_path = Path(args.canonical_summary)
    canonical_lineage_path = Path(args.canonical_lineage)
    source = source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    rows, canonical_hashes = canonical_split_rows(
        canonical_paths,
        args.split,
        expected_manifest_sha256=(
            args.expected_canonical_manifest_sha256
        ),
    )
    canonical_receipt = load_canonical_receipt(
        manifest_paths=canonical_paths,
        manifest_hashes=canonical_hashes,
        summary_path=canonical_summary_path,
        lineage_path=canonical_lineage_path,
        expected_manifest_sha256=(
            args.expected_canonical_manifest_sha256
        ),
        expected_summary_sha256=args.expected_canonical_summary_sha256,
        expected_lineage_sha256=args.expected_canonical_lineage_sha256,
        expected_canonical_source_commit=(
            args.expected_canonical_source_commit
        ),
        expected_canonical_source_tree=args.expected_canonical_source_tree,
    )
    selected = [
        row for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_id
    ]
    if args.expected_total_clips and len(rows) != args.expected_total_clips:
        raise RuntimeError(
            f"canonical {args.split} clips {len(rows)} != "
            f"--expected-total-clips {args.expected_total_clips}"
        )
    expected_shard = (
        sum(
            index % args.num_shards == args.shard_id
            for index in range(len(rows))
        )
    )
    if len(selected) != expected_shard:
        raise AssertionError("internal deterministic shard count mismatch")

    hubert_input = Path(args.hubert_model)
    if hubert_input.is_symlink():
        raise RuntimeError(f"HuBERT root must not be a symlink: {hubert_input}")
    hubert_dir = hubert_input.resolve()
    model_tree_sha, model_files = tree_sha256(hubert_dir)
    if model_tree_sha != args.expected_hubert_tree_sha256:
        raise RuntimeError(
            f"HuBERT tree {model_tree_sha} "
            f"!= {args.expected_hubert_tree_sha256}"
        )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        str(hubert_dir),
        local_files_only=True,
    )
    model = HubertModel.from_pretrained(
        str(hubert_dir),
        local_files_only=True,
    ).to(args.device)
    loaded_tree_sha, loaded_model_files = tree_sha256(hubert_dir)
    if (
        loaded_tree_sha != model_tree_sha
        or loaded_model_files != model_files
    ):
        raise RuntimeError(
            "HuBERT tree changed while loading feature extractor/model"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_grad_enabled(False)

    runtime = runtime_record()
    protocol = {
        "split": args.split,
        "sample_rate": 16000,
        "fps": 30,
        "hubert": "HubertModel frozen float32 inference",
        "hubert_preprocessing": {
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
        },
        "hubert_long_chunk": {
            "kernel": 400,
            "stride": 320,
            "clip_length": 320000,
        },
        "alignment": AUDIO_ALIGNMENT_PROTOCOL,
        "beat": "SemTalk amplitude_energy_onset",
        "forbidden_components": [
            "ASR",
            "TextGrid",
            "vocabulary",
            "CLIP",
            "emotion",
            "semantic",
            "SemGate",
            "Sparse",
        ],
    }
    audio_lineage_contract = {
        "format": "semtalk_show_audio_lineage_contract_v1",
        "protocol": protocol,
        "canonical_manifest_sha256": canonical_hashes,
        "canonical_receipt": canonical_receipt,
        "source_receipt": source,
        "hubert_model": str(hubert_dir),
        "hubert_model_tree_sha256": model_tree_sha,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "full_split_clips": len(rows),
        "runtime": runtime,
    }
    audio_lineage_contract_sha = canonical_file_payload_sha256(
        audio_lineage_contract
    )

    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    refuse_existing(temp_dir)
    temp_dir.mkdir()
    output_rows: list[dict[str, Any]] = []
    artifact_aggregate = hashlib.sha256()
    started = time.time()
    try:
        with torch.inference_mode():
            for row in selected:
                clip_id = str(row["clip_id"])
                frames = canonical_frames(row)
                source_wav, wav_payload, wav_sha = read_verified_bytes(
                    Path(row["source_wav"]),
                    str(row["source_wav_sha256"]),
                    "source WAV",
                )
                speech_native, sample_rate = librosa.load(
                    io.BytesIO(wav_payload),
                    sr=None,
                    mono=True,
                )
                if sample_rate <= 0:
                    raise RuntimeError(
                        f"{source_wav}: invalid native sample rate {sample_rate}"
                    )
                speech = librosa.resample(
                    np.asarray(speech_native, dtype=np.float32),
                    orig_sr=sample_rate,
                    target_sr=16000,
                ).astype(np.float32, copy=False)
                beat, timing = semtalk_rhythm_features(
                    speech,
                    target_frames=frames,
                    max_frame_mismatch=args.max_frame_mismatch,
                )
                native_hubert = hubert_long(
                    model,
                    feature_extractor,
                    speech,
                    device=args.device,
                )
                hubert_native_30fps = align_hubert(
                    native_hubert,
                    timing["native_30fps_frames"],
                )
                hubert, hubert_timing = align_public_audio_prefix(
                    hubert_native_30fps,
                    target_frames=frames,
                    max_frame_mismatch=args.max_frame_mismatch,
                )
                if hubert_timing != {
                    key: timing[key]
                    for key in hubert_timing
                }:
                    raise AssertionError(
                        f"{clip_id}: beat/HuBERT prefix timing mismatch"
                    )
                arrays = {
                    "beat": beat,
                    "hubert": hubert,
                }
                for name, expected_shape in {
                    "beat": (frames, 3),
                    "hubert": (frames, 1024),
                }.items():
                    array = arrays[name]
                    if (
                        array.shape != expected_shape
                        or array.dtype != np.float32
                        or not np.isfinite(array).all()
                    ):
                        raise RuntimeError(
                            f"{clip_id}: invalid {name} "
                            f"{array.shape}/{array.dtype}"
                        )
                filename = _safe_clip_filename(clip_id)
                temp_npz = temp_dir / filename
                atomic_npz(temp_npz, arrays)
                npz_sha = sha256(temp_npz)
                final_npz = output_dir / filename
                output_row = {
                    "format": "semtalk_show_audio_clip_v1",
                    "split": args.split,
                    "clip_id": clip_id,
                    "canonical_npz": str(
                        Path(row["canonical_npz"]).resolve()
                    ),
                    "canonical_npz_sha256": str(
                        row["canonical_npz_sha256"]
                    ),
                    "source_wav": str(source_wav),
                    "source_wav_sha256": wav_sha,
                    "audio_feature_npz": str(final_npz),
                    "audio_feature_npz_sha256": npz_sha,
                    "frames": frames,
                    "lineage_contract_sha256": str(
                        row["lineage_contract_sha256"]
                    ),
                    "audio_lineage_contract_sha256": (
                        audio_lineage_contract_sha
                    ),
                    "beat_shape": list(beat.shape),
                    "hubert_shape": list(hubert.shape),
                    "hubert_native_frames": int(native_hubert.shape[0]),
                    "audio_samples_16k": timing["audio_samples_16k"],
                    "source_sample_rate": int(sample_rate),
                    "native_30fps_frames": timing["native_30fps_frames"],
                    "canonical_usable_frames": timing[
                        "canonical_usable_frames"
                    ],
                    "discarded_source_30fps_frames": timing[
                        "discarded_source_30fps_frames"
                    ],
                    "edge_padded_tail_frames": timing[
                        "edge_padded_tail_frames"
                    ],
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "source_audio_field": "source_wav",
                }
                output_rows.append(output_row)
                artifact_aggregate.update(clip_id.encode("utf-8"))
                artifact_aggregate.update(bytes.fromhex(npz_sha))
        if len(output_rows) != len(selected):
            raise RuntimeError("audio cache did not cover its shard exactly once")
        final_source = source_receipt(
            args.expected_source_commit,
            args.expected_source_tree,
        )
        final_rows, final_canonical_hashes = canonical_split_rows(
            canonical_paths,
            args.split,
            expected_manifest_sha256=(
                args.expected_canonical_manifest_sha256
            ),
        )
        final_canonical_receipt = load_canonical_receipt(
            manifest_paths=canonical_paths,
            manifest_hashes=final_canonical_hashes,
            summary_path=canonical_summary_path,
            lineage_path=canonical_lineage_path,
            expected_manifest_sha256=(
                args.expected_canonical_manifest_sha256
            ),
            expected_summary_sha256=(
                args.expected_canonical_summary_sha256
            ),
            expected_lineage_sha256=(
                args.expected_canonical_lineage_sha256
            ),
            expected_canonical_source_commit=(
                args.expected_canonical_source_commit
            ),
            expected_canonical_source_tree=(
                args.expected_canonical_source_tree
            ),
        )
        final_tree_sha, final_model_files = tree_sha256(hubert_dir)
        if (
            final_source != source
            or final_rows != rows
            or final_canonical_hashes != canonical_hashes
            or final_canonical_receipt != canonical_receipt
            or final_tree_sha != model_tree_sha
            or final_model_files != model_files
        ):
            raise RuntimeError(
                "formal source/canonical/HuBERT input changed during "
                "audio feature extraction"
            )
        os.replace(temp_dir, output_dir)
        _fsync_parent(output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    atomic_jsonl(output_manifest, output_rows)
    manifest_sha = sha256(output_manifest)
    lineage = {
        "format": "semtalk_show_audio_lineage_v1",
        "status": "complete",
        "protocol": protocol,
        "lineage_contract_sha256": str(
            rows[0]["lineage_contract_sha256"]
        ),
        "audio_lineage_contract": audio_lineage_contract,
        "audio_lineage_contract_sha256": audio_lineage_contract_sha,
        "canonical_manifest_sha256": canonical_hashes,
        "canonical_receipt": canonical_receipt,
        "source_receipt": source,
        "hubert_model": str(hubert_dir),
        "hubert_model_tree_sha256": model_tree_sha,
        "hubert_model_files": model_files,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "full_split_clips": len(rows),
        "shard_clips": len(output_rows),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": manifest_sha,
        "artifact_aggregate_sha256": artifact_aggregate.hexdigest(),
        "runtime": runtime,
        "argv": sys.argv,
    }
    atomic_json(lineage_path, lineage)
    lineage_sha = sha256(lineage_path)
    summary = {
        "format": "semtalk_show_audio_summary_v1",
        "status": "complete",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "full_split_clips": len(rows),
        "shard_clips": len(output_rows),
        "output_dir": str(output_dir),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": manifest_sha,
        "lineage_json": str(lineage_path),
        "lineage_json_sha256": lineage_sha,
        "artifact_aggregate_sha256": artifact_aggregate.hexdigest(),
        "started_unix": started,
        "completed_unix": time.time(),
    }
    atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": "audio",
                "shard_id": args.shard_id,
                "clips": len(output_rows),
            },
            sort_keys=True,
        )
    )


def load_canonical_clip(
    row: dict[str, Any],
) -> tuple[dict[str, np.ndarray], int]:
    path, payload, _ = read_verified_bytes(
        Path(row["canonical_npz"]),
        str(row["canonical_npz_sha256"]),
        "canonical NPZ",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        missing = sorted(set(CANONICAL_FIELDS) - set(archive.files))
        if missing:
            raise RuntimeError(f"{path}: missing canonical fields {missing}")
        arrays = {name: archive[name].copy() for name in CANONICAL_FIELDS}
    frames = int(arrays["pose"].shape[0])
    expected = {
        "pose": (frames, 165),
        "contact": (frames, 4),
        "facial": (frames, 100),
        "beta": (frames, 300),
        "trans": (frames, 3),
        "speaker_id": (frames, 1),
    }
    for name, shape in expected.items():
        array = arrays[name]
        if array.shape != shape:
            raise RuntimeError(f"{path}: {name} {array.shape} != {shape}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise RuntimeError(f"{path}: non-finite canonical {name}")
    speaker = arrays["speaker_id"]
    if speaker.dtype.kind not in "iu":
        raise RuntimeError(f"{path}: speaker_id is not integer")
    if speaker.min() < 0 or speaker.max() > 3:
        raise RuntimeError(f"{path}: SHOW speaker ID outside [0,3]")
    if np.unique(speaker).size != 1:
        raise RuntimeError(f"{path}: speaker ID changes within clip")
    manifest_speaker = row.get("speaker")
    if (
        not isinstance(manifest_speaker, str)
        or manifest_speaker not in SPEAKER_MAP
    ):
        raise RuntimeError(f"{path}: invalid manifest SHOW speaker")
    manifest_speaker_id = require_exact_int(
        row.get("speaker_id"),
        f"{path}: manifest speaker_id",
    )
    if manifest_speaker_id != SPEAKER_MAP[manifest_speaker]:
        raise RuntimeError(f"{path}: manifest speaker/name mismatch")
    if not np.all(speaker == manifest_speaker_id):
        raise RuntimeError(f"{path}: NPZ/manifest speaker ID mismatch")
    if (
        row.get("frames") is not None
        and require_exact_int(row["frames"], "canonical row frames")
        != frames
    ):
        raise RuntimeError(f"{path}: manifest frame mismatch")
    return arrays, frames


def load_audio_rows(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows, hashes = load_jsonl(paths)
    mapping: dict[str, dict[str, Any]] = {}
    required = {
        "format",
        "clip_id",
        "audio_feature_npz",
        "audio_feature_npz_sha256",
        "canonical_npz_sha256",
        "source_wav_sha256",
        "lineage_contract_sha256",
        "audio_lineage_contract_sha256",
        "frames",
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
    for row in rows:
        if row.get("split") != "train":
            raise RuntimeError("audio manifest must contain train rows only")
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(
                f"audio row {row.get('clip_id')!r} missing {missing}"
            )
        clip_id = str(row["clip_id"])
        if row.get("format") != "semtalk_show_audio_clip_v1":
            raise RuntimeError(f"{clip_id}: unexpected audio row format")
        frames = require_exact_int(row.get("frames"), f"{clip_id} frames")
        native_frames = require_exact_int(
            row.get("native_30fps_frames"),
            f"{clip_id} native audio frames",
        )
        canonical_usable = require_exact_int(
            row.get("canonical_usable_frames"),
            f"{clip_id} canonical usable frames",
        )
        discarded = require_exact_int(
            row.get("discarded_source_30fps_frames"),
            f"{clip_id} discarded audio frames",
        )
        edge_padded = require_exact_int(
            row.get("edge_padded_tail_frames"),
            f"{clip_id} edge-padded audio frames",
        )
        audio_samples = require_exact_int(
            row.get("audio_samples_16k"),
            f"{clip_id} 16 kHz audio samples",
        )
        source_sample_rate = require_exact_int(
            row.get("source_sample_rate"),
            f"{clip_id} source sample rate",
        )
        hubert_native_frames = require_exact_int(
            row.get("hubert_native_frames"),
            f"{clip_id} native HuBERT frames",
        )
        if (
            frames <= 0
            or native_frames <= 0
            or audio_samples <= 0
            or source_sample_rate <= 0
            or hubert_native_frames <= 0
            or native_frames != (audio_samples * 30) // 16000
            or row.get("beat_shape") != [frames, 3]
            or row.get("hubert_shape") != [frames, 1024]
            or canonical_usable != (frames // 30) * 30
            or discarded != max(native_frames - frames, 0)
            or edge_padded != max(frames - native_frames, 0)
            or edge_padded > 1
            or (
                native_frames < frames
                and native_frames < canonical_usable
            )
        ):
            raise RuntimeError(f"{clip_id}: invalid public-prefix timing receipt")
        if clip_id in mapping:
            raise RuntimeError(f"duplicate audio clip_id {clip_id}")
        mapping[clip_id] = row
    if not mapping:
        raise RuntimeError("audio manifests contain no rows")
    return mapping, hashes


def load_audio_lineages(
    paths: list[Path],
    *,
    audio_manifest_hashes: dict[str, str],
    canonical_manifest_hashes: dict[str, str],
    canonical_receipt: dict[str, Any],
    canonical_lineage_contract_sha256: str,
    expected_train_clips: int,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_hubert_tree_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not paths:
        raise RuntimeError("audio lineage JSON files are required")
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    manifest_paths = set(audio_manifest_hashes)
    bound_manifests: set[str] = set()
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(
                f"audio lineage must not be a symlink: {path}"
            )
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        hashes[str(resolved)] = sha256(resolved)
        with resolved.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if (
            not isinstance(record, dict)
            or record.get("format") != "semtalk_show_audio_lineage_v1"
            or record.get("status") != "complete"
        ):
            raise RuntimeError(f"{resolved}: invalid audio lineage")
        manifest_input = Path(record["output_manifest"])
        if manifest_input.is_symlink():
            raise RuntimeError(
                f"{resolved}: output manifest must not be a symlink"
            )
        manifest = str(manifest_input.resolve())
        if manifest not in manifest_paths:
            raise RuntimeError(
                f"{resolved}: lineage binds an unrequested manifest {manifest}"
            )
        if manifest in bound_manifests:
            raise RuntimeError(f"duplicate lineage for audio manifest {manifest}")
        if (
            record.get("output_manifest_sha256")
            != audio_manifest_hashes[manifest]
        ):
            raise RuntimeError(f"{resolved}: audio manifest SHA mismatch")
        if record.get("canonical_manifest_sha256") != (
            canonical_manifest_hashes
        ):
            raise RuntimeError(f"{resolved}: canonical lineage mismatch")
        if record.get("canonical_receipt") != canonical_receipt:
            raise RuntimeError(f"{resolved}: canonical receipt mismatch")
        record_source = record.get("source_receipt")
        if (
            not isinstance(record_source, dict)
            or record_source.get("origin") != EXPECTED_ORIGIN
            or record_source.get("commit") != expected_source_commit
            or record_source.get("tree") != expected_source_tree
        ):
            raise RuntimeError(f"{resolved}: feature source receipt mismatch")
        if (
            record.get("hubert_model_tree_sha256")
            != expected_hubert_tree_sha256
        ):
            raise RuntimeError(f"{resolved}: pinned HuBERT tree mismatch")
        if (
            require_exact_int(
                record.get("full_split_clips"),
                "audio lineage full_split_clips",
            )
            != expected_train_clips
        ):
            raise RuntimeError(f"{resolved}: train clip count mismatch")
        protocol = record.get("protocol")
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
            or protocol.get("split") != "train"
            or protocol.get("sample_rate") != 16000
            or protocol.get("hubert_preprocessing")
            != expected_hubert_preprocessing
            or protocol.get("alignment") != AUDIO_ALIGNMENT_PROTOCOL
        ):
            raise RuntimeError(
                f"{resolved}: unsupported HuBERT preprocessing protocol"
            )
        if record.get("lineage_contract_sha256") != (
            canonical_lineage_contract_sha256
        ):
            raise RuntimeError(f"{resolved}: canonical contract mismatch")
        contract = record.get("audio_lineage_contract")
        contract_sha = record.get("audio_lineage_contract_sha256")
        if (
            not isinstance(contract, dict)
            or canonical_file_payload_sha256(contract) != contract_sha
        ):
            raise RuntimeError(
                f"{resolved}: invalid audio lineage contract hash"
            )
        contract_expectations = {
            "format": "semtalk_show_audio_lineage_contract_v1",
            "protocol": record.get("protocol"),
            "canonical_manifest_sha256": canonical_manifest_hashes,
            "canonical_receipt": canonical_receipt,
            "source_receipt": record_source,
            "hubert_model_tree_sha256": record.get(
                "hubert_model_tree_sha256"
            ),
            "shard_id": record.get("shard_id"),
            "num_shards": record.get("num_shards"),
            "full_split_clips": expected_train_clips,
            "runtime": record.get("runtime"),
        }
        for key, expected in contract_expectations.items():
            if contract.get(key) != expected:
                raise RuntimeError(
                    f"{resolved}: audio contract {key} mismatch"
                )
        forbidden = set(record.get("protocol", {}).get(
            "forbidden_components", []
        ))
        required_forbidden = {
            "ASR",
            "TextGrid",
            "vocabulary",
            "CLIP",
            "emotion",
            "semantic",
            "SemGate",
            "Sparse",
        }
        if not required_forbidden.issubset(forbidden):
            raise RuntimeError(f"{resolved}: forbidden-component gate missing")
        bound_manifests.add(manifest)
        records.append(record)
    if bound_manifests != manifest_paths:
        raise RuntimeError("not every audio manifest has exactly one lineage")

    shard_counts = {
        require_exact_int(
            record.get("num_shards"),
            "audio lineage num_shards",
        )
        for record in records
    }
    if len(shard_counts) != 1:
        raise RuntimeError("audio lineages disagree on num_shards")
    num_shards = shard_counts.pop()
    shard_ids = sorted(
        require_exact_int(
            record.get("shard_id"),
            "audio lineage shard_id",
        )
        for record in records
    )
    if shard_ids != list(range(num_shards)):
        raise RuntimeError(
            f"audio lineage shards {shard_ids} do not cover 0..{num_shards - 1}"
        )
    if len(records) != num_shards:
        raise RuntimeError("audio lineage file count differs from num_shards")
    if sum(
        require_exact_int(
            record.get("shard_clips"),
            "audio lineage shard_clips",
        )
        for record in records
    ) != expected_train_clips:
        raise RuntimeError("audio lineage shard counts do not exactly cover train")
    model_hashes = {
        str(record["hubert_model_tree_sha256"]) for record in records
    }
    if len(model_hashes) != 1:
        raise RuntimeError("audio shards used different HuBERT model trees")
    protocols = {
        json.dumps(record["protocol"], sort_keys=True, separators=(",", ":"))
        for record in records
    }
    if len(protocols) != 1:
        raise RuntimeError("audio shards used different feature protocols")
    runtimes = {
        json.dumps(record["runtime"], sort_keys=True, separators=(",", ":"))
        for record in records
    }
    if len(runtimes) != 1:
        raise RuntimeError("audio shards used different software runtimes")
    return records, hashes


def load_audio_clip(
    row: dict[str, Any],
    *,
    expected_frames: int,
) -> dict[str, np.ndarray]:
    path, payload, _ = read_verified_bytes(
        Path(row["audio_feature_npz"]),
        str(row["audio_feature_npz_sha256"]),
        "audio feature NPZ",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        missing = sorted(set(AUDIO_FIELDS) - set(archive.files))
        if missing:
            raise RuntimeError(f"{path}: missing audio fields {missing}")
        arrays = {name: archive[name].copy() for name in AUDIO_FIELDS}
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
            raise RuntimeError(
                f"{path}: invalid {name} {array.shape}/{array.dtype}"
            )
    if require_exact_int(row.get("frames"), "audio manifest frames") != expected_frames:
        raise RuntimeError(f"{path}: audio manifest frame mismatch")
    return arrays


def _torch_load(payload_bytes: bytes, path: Path) -> dict[str, Any]:
    import torch

    buffer = io.BytesIO(payload_bytes)
    try:
        payload = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:  # older supported PyTorch
        buffer.seek(0)
        payload = torch.load(buffer, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(f"{path}: checkpoint lacks model_state")
    if not isinstance(payload["model_state"], dict):
        raise RuntimeError(f"{path}: model_state is not a mapping")
    return payload


def _finite_state_dict(state: dict[str, Any], path: Path) -> None:
    import torch

    for name, value in state.items():
        if torch.is_tensor(value) and (
            value.is_floating_point() or value.is_complex()
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"{path}: non-finite checkpoint tensor {name}")


def _state_dict_schema_sha256(state: Mapping[str, Any]) -> str:
    schema = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(state.items())
    ]
    return canonical_file_payload_sha256(schema)


def _load_released_model_state_only(
    path: Path,
    *,
    expected_filename: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], Path, str, int]:
    """Load one hash-pinned, tensor-only official release checkpoint."""
    import torch

    if path.name != expected_filename:
        raise RuntimeError(
            f"released checkpoint filename {path.name!r} != "
            f"{expected_filename!r}"
        )
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise RuntimeError(
            f"released checkpoint must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    if resolved.name != expected_filename or not resolved.is_file():
        raise RuntimeError(f"unsafe released checkpoint path: {path}")
    checkpoint_bytes = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{resolved}: released checkpoint SHA-256 "
            f"{actual_sha256} != {expected_sha256}"
        )
    buffer = io.BytesIO(checkpoint_bytes)
    try:
        payload = torch.load(
            buffer,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - supported PyTorch has it
        raise RuntimeError(
            "released_all_speakers_v1 requires torch.load(weights_only=True)"
        ) from error
    if type(payload) is not dict or set(payload) != {"model_state"}:
        raise RuntimeError(
            f"{resolved}: released checkpoint must contain only model_state"
        )
    raw_state = payload["model_state"]
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise RuntimeError(
            f"{resolved}: released model_state must be a non-empty mapping"
        )
    normalized: dict[str, Any] = {}
    for key, value in raw_state.items():
        if type(key) is not str or not key:
            raise RuntimeError(
                f"{resolved}: every released model_state key must be a "
                "non-empty string"
            )
        if not torch.is_tensor(value):
            raise RuntimeError(
                f"{resolved}: released model_state value {key!r} is not a "
                "tensor"
            )
        normalized_key = key[7:] if key.startswith("module.") else key
        if not normalized_key or normalized_key in normalized:
            raise RuntimeError(
                f"{resolved}: released model_state key normalization collision"
            )
        if (
            value.is_floating_point() or value.is_complex()
        ) and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(
                f"{resolved}: non-finite released model_state tensor {key!r}"
            )
        normalized[normalized_key] = value
    return normalized, resolved, actual_sha256, len(checkpoint_bytes)


def _strict_load_freeze_eval(
    model: Any,
    state: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{path}: strict released state load was not exact")
    model.eval()
    model.requires_grad_(False)
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError(f"{path}: released model did not freeze in eval mode")


def checkpoint_record(
    path: Path,
    *,
    formal_stage: str,
    expected_lineage_sha256: str,
    status_path: Path,
    expected_source_receipt: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    checkpoint_bytes = resolved.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    payload = _torch_load(checkpoint_bytes, resolved)
    _finite_state_dict(payload["model_state"], resolved)
    audit = payload.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{resolved}: formal checkpoint audit is missing")
    audit_format = audit.get("format")
    if audit_format != "semtalk_show_model_v2":
        raise RuntimeError(
            f"{resolved}: formal checkpoint must use semtalk_show_model_v2"
        )
    if audit.get("formal_stage") != formal_stage:
        raise RuntimeError(
            f"{resolved}: formal_stage={audit.get('formal_stage')!r}, "
            f"expected {formal_stage!r}"
        )
    lineage_sha = audit.get("lineage_manifest_sha256")
    if lineage_sha != expected_lineage_sha256:
        raise RuntimeError(
            f"{resolved}: training lineage {lineage_sha!r} != "
            f"{expected_lineage_sha256!r}"
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
        != compact_payload_sha256(audit_source)
    ):
        raise RuntimeError(
            f"{resolved}: checkpoint source receipt is not bound to the "
            "current formal source"
        )
    resolved_status = status_path.resolve()
    if status_path.is_symlink() or not resolved_status.is_file():
        raise FileNotFoundError(resolved_status)
    with resolved_status.open(encoding="utf-8") as handle:
        status = json.load(handle)
    status_final_input = Path(
        str(status.get("final_checkpoint", ""))
        if isinstance(status, dict)
        else ""
    )
    if status_final_input.is_symlink() or not status_final_input.is_file():
        raise RuntimeError(
            "formal status final_checkpoint must be a regular non-symlink "
            f"file: {status_final_input}"
        )
    status_final_checkpoint = status_final_input.resolve()
    status_dataset_receipt = (
        status.get("dataset_receipt")
        if isinstance(status, dict)
        else None
    )
    expected_epochs = {
        "face": 600,
        "hands": 500,
        "upper": 500,
        "lower": 600,
        "global": 1700,
    }[formal_stage]
    if (
        not isinstance(status, dict)
        or status.get("status") != "complete"
        or status.get("formal_stage") != formal_stage
        or require_exact_int(status.get("world_size"), "world_size") != 1
        or require_exact_int(status.get("epochs"), "epochs")
        != expected_epochs
        or require_exact_int(
            status.get("completed_epochs"),
            "completed_epochs",
        )
        != expected_epochs
        or status.get("lineage_manifest_sha256")
        != expected_lineage_sha256
        or status.get("config_sha256") != audit.get("config_sha256")
        or not isinstance(status_dataset_receipt, dict)
        or status_dataset_receipt.get("summary_sha256")
        != audit.get("dataset_summary_sha256")
        or status_dataset_receipt.get("data_mdb_sha256")
        != audit.get("data_mdb_sha256")
        or require_exact_int(
            status_dataset_receipt.get("entries"),
            "dataset receipt entries",
        )
        != 127_286
        or require_exact_int(
            status_dataset_receipt.get("train_clips"),
            "dataset receipt train_clips",
        )
        != 13_687
        or status.get("source_receipt") != audit.get("source_receipt")
        or status.get("source_receipt_sha256")
        != audit.get("source_receipt_sha256")
        or status_final_checkpoint != resolved
        or status.get("final_checkpoint_sha256") != checkpoint_sha256
    ):
        raise RuntimeError(
            f"{resolved_status}: incomplete or inconsistent formal "
            f"{formal_stage} training receipt"
        )
    parity_receipt = status_dataset_receipt.get("global_fastpath_parity")
    parity_artifact_receipt: dict[str, str] | None = None
    if formal_stage == "global":
        if (
            not isinstance(parity_receipt, dict)
            or parity_receipt.get("format")
            != "semtalk_show_global_foot_parity_suite_v1"
            or parity_receipt.get("status") != "pass"
            or require_exact_int_mapping(
                parity_receipt.get("speakers"),
                SPEAKER_MAP,
                "Global parity speakers",
            )
            != SPEAKER_MAP
        ):
            raise RuntimeError(
                f"{resolved_status}: Global checkpoint lacks the required "
                "four-speaker fastpath parity receipt"
            )
        parity_input = Path(str(parity_receipt.get("path", "")))
        parity_path = parity_input.resolve()
        if (
            parity_input.is_symlink()
            or not parity_path.is_file()
            or sha256(parity_path) != parity_receipt.get("sha256")
        ):
            raise RuntimeError(
                f"{resolved_status}: Global fastpath parity artifact mismatch"
            )
        parity_artifact_receipt = {
            "path": str(parity_path),
            "sha256": str(parity_receipt["sha256"]),
        }
    elif parity_receipt is not None:
        raise RuntimeError(
            f"{resolved_status}: non-Global stage has a parity receipt"
        )
    updates_per_epoch = require_exact_int(
        status.get("updates_per_epoch"),
        "updates_per_epoch",
    )
    train_samples = require_exact_int(
        status.get("train_samples"),
        "train_samples",
    )
    optimizer_updates = require_exact_int(
        status.get("optimizer_updates"),
        "optimizer_updates",
    )
    if (
        train_samples != 127_286
        or updates_per_epoch != 1_988
        or optimizer_updates != expected_epochs * updates_per_epoch
    ):
        raise RuntimeError(
            f"{resolved_status}: invalid sample/update accounting"
        )
    dataset_receipt = status_dataset_receipt
    audit_optimizer_updates = audit.get("optimizer_updates")
    validate_lower_target_backend_binding(
        formal_stage=formal_stage,
        audit=audit,
        dataset_receipt=dataset_receipt,
        status=status,
        path=resolved,
    )
    expected_audit_keys = set(MODEL_V2_AUDIT_KEYS)
    if formal_stage == "lower":
        expected_audit_keys.add(LOWER_TARGET_BACKEND_RECEIPT_KEY)
    if (
        set(audit) != expected_audit_keys
        or not isinstance(dataset_receipt, dict)
        or audit.get("dataset_receipt_sha256")
        != compact_payload_sha256(dataset_receipt)
        or audit.get("smplx_asset_receipt")
        != dataset_receipt.get("smplx_asset")
        or status.get("smplx_asset_receipt")
        != dataset_receipt.get("smplx_asset")
        or type(audit_optimizer_updates) is not int
        or audit_optimizer_updates != optimizer_updates
        or audit.get("base_candidate_manifest") is not None
        or status.get("base_candidate_manifest") is not None
    ):
        raise RuntimeError(
            f"{resolved}: invalid model_v2 dataset/SMPL-X/update audit"
        )
    record = {
        "path": str(resolved),
        "sha256": checkpoint_sha256,
        "formal_stage": formal_stage,
        "audit": audit,
        "formal_training_status": str(resolved_status),
        "formal_training_status_sha256": sha256(resolved_status),
        "training_accounting": {
            "epochs": expected_epochs,
            "train_samples": train_samples,
            "updates_per_epoch": updates_per_epoch,
            "optimizer_updates": optimizer_updates,
        },
    }
    if parity_artifact_receipt is not None:
        record["global_fastpath_parity"] = parity_artifact_receipt
    return payload, record


def load_rvq_models(
    args: argparse.Namespace,
    expected_lineage_sha256: str,
    expected_source_receipt: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from models.rvq import RVQVAE

    dimensions = {
        "face": 106,
        "upper": 78,
        "hands": 180,
        "lower": 61,
    }
    models: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for name in RVQ_NAMES:
        path = Path(getattr(args, f"{name}_checkpoint"))
        payload, record = checkpoint_record(
            path,
            formal_stage=name,
            expected_lineage_sha256=expected_lineage_sha256,
            status_path=Path(getattr(args, f"{name}_status_json")),
            expected_source_receipt=expected_source_receipt,
        )
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in payload["model_state"].items()
        }
        model = RVQVAE(
            SimpleNamespace(vae_test_dim=dimensions[name])
        ).to(args.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
        records[name] = record
        del payload, state

    # Base does not consume the global/root VAE, but its formal completion is a
    # prerequisite and therefore is strictly verified and lineage-bound here.
    global_payload, global_record = checkpoint_record(
        Path(args.global_checkpoint),
        formal_stage="global",
        expected_lineage_sha256=expected_lineage_sha256,
        status_path=Path(args.global_status_json),
        expected_source_receipt=expected_source_receipt,
    )
    records["global"] = global_record
    del global_payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return models, records


def load_released_all_speakers_models(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Import the exact official BEAT2 All-Speakers representation suite."""
    import torch
    from models.motion_representation import VAEConvZero
    from models.rvq import RVQVAE

    source_receipt = {
        "format": "semtalk_released_all_speakers_weight_source_v1",
        "prerequisite_source": RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE,
        "classification": RELEASED_ALL_SPEAKERS_CLASSIFICATION,
        "release_trust_root": dict(
            RELEASED_ALL_SPEAKERS_RELEASE_TRUST_ROOT
        ),
        "official_release": True,
        "training_dataset": "BEAT2",
        "speaker_scope": "All-Speakers",
        "show_trained": False,
        "checkpoint_container_schema": ["model_state"],
    }
    source_receipt_sha256 = canonical_file_payload_sha256(source_receipt)
    models: dict[str, Any] = {}
    records: dict[str, Any] = {}
    receipt_files: dict[str, Any] = {}
    for name in (*RVQ_NAMES, "global"):
        specification = RELEASED_ALL_SPEAKERS_MODELS[name]
        path = Path(getattr(args, f"{name}_checkpoint"))
        state, resolved, checkpoint_sha256, checkpoint_bytes = (
            _load_released_model_state_only(
                path,
                expected_filename=str(specification["filename"]),
                expected_sha256=str(specification["sha256"]),
            )
        )
        model_args = SimpleNamespace(
            vae_test_dim=int(specification["vae_test_dim"]),
            vae_layer=int(specification["vae_layer"]),
            vae_length=256,
        )
        model = (
            VAEConvZero(model_args)
            if name == "global"
            else RVQVAE(model_args)
        ).to(args.device)
        _strict_load_freeze_eval(model, state, path=resolved)
        schema_sha256 = _state_dict_schema_sha256(state)
        record = {
            "path": str(resolved),
            "filename": str(specification["filename"]),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_bytes,
            "formal_stage": name,
            "prerequisite_source": (
                RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE
            ),
            "classification": RELEASED_ALL_SPEAKERS_CLASSIFICATION,
            "training_dataset": "BEAT2",
            "speaker_scope": "All-Speakers",
            "show_trained": False,
            "checkpoint_container_schema": ["model_state"],
            "model_class": str(specification["model_class"]),
            "model_state_tensors": len(state),
            "model_state_schema_sha256": schema_sha256,
            "all_model_state_tensors_finite": True,
            "strict_state_dict_load": True,
            "frozen_eval": True,
            "source_receipt": dict(source_receipt),
            "source_receipt_sha256": source_receipt_sha256,
        }
        records[name] = record
        receipt_files[name] = {
            "path": str(resolved),
            "filename": str(specification["filename"]),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_bytes,
            "model_class": str(specification["model_class"]),
            "model_state_schema_sha256": schema_sha256,
        }
        if name == "global":
            # Base features do not consume Global, but the official VAE must
            # still pass the same exact schema/load/freeze gate.
            del model
        else:
            models[name] = model
        del state
    if set(models) != set(RVQ_NAMES) or set(records) != {
        *RVQ_NAMES,
        "global",
    }:
        raise RuntimeError(
            "released_all_speakers_v1 did not import the exact five models"
        )
    receipt = {
        "format": "semtalk_released_all_speakers_import_receipt_v1",
        "status": "complete",
        "prerequisite_source": RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE,
        "classification": RELEASED_ALL_SPEAKERS_CLASSIFICATION,
        "training_dataset": "BEAT2",
        "speaker_scope": "All-Speakers",
        "show_trained": False,
        "source_receipt": source_receipt,
        "source_receipt_sha256": source_receipt_sha256,
        "files": receipt_files,
        "strict_state_dict_load": True,
        "all_model_state_tensors_finite": True,
        "frozen_eval": True,
    }
    receipt["receipt_sha256"] = canonical_file_payload_sha256(receipt)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return models, records, receipt


def load_prerequisite_models(
    args: argparse.Namespace,
    expected_lineage_sha256: str,
    expected_source_receipt: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    source = args.prerequisite_source
    statuses = {
        name: getattr(args, f"{name}_status_json")
        for name in (*RVQ_NAMES, "global")
    }
    if source == SHOW_TRAINED_PREREQUISITE_SOURCE:
        missing = sorted(name for name, path in statuses.items() if path is None)
        if missing:
            raise RuntimeError(
                "show_trained_v1 requires status JSON for "
                f"{', '.join(missing)}"
            )
        models, records = load_rvq_models(
            args,
            expected_lineage_sha256,
            expected_source_receipt,
        )
        return models, records, None
    if source == RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE:
        unexpected = sorted(
            name for name, path in statuses.items() if path is not None
        )
        if unexpected:
            raise RuntimeError(
                "released_all_speakers_v1 forbids SHOW training status JSON "
                f"for {', '.join(unexpected)}"
            )
        return load_released_all_speakers_models(args)
    raise RuntimeError(f"unsupported prerequisite source: {source!r}")


def revalidate_checkpoint_records(
    records: dict[str, dict[str, Any]],
) -> None:
    if set(records) != {*RVQ_NAMES, "global"}:
        raise RuntimeError("formal checkpoint receipt set changed")
    for name, record in records.items():
        checkpoint = Path(str(record["path"]))
        if (
            record.get("prerequisite_source")
            == RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE
        ):
            specification = RELEASED_ALL_SPEAKERS_MODELS[name]
            try:
                mode = os.lstat(checkpoint).st_mode
            except FileNotFoundError:
                raise RuntimeError(
                    f"{name} released checkpoint disappeared: {checkpoint}"
                ) from None
            if (
                checkpoint.is_symlink()
                or not stat.S_ISREG(mode)
                or checkpoint.name != specification["filename"]
                or record.get("filename") != specification["filename"]
                or record.get("sha256") != specification["sha256"]
                or sha256(checkpoint) != specification["sha256"]
                or record.get("classification")
                != RELEASED_ALL_SPEAKERS_CLASSIFICATION
                or record.get("training_dataset") != "BEAT2"
                or record.get("speaker_scope") != "All-Speakers"
                or record.get("show_trained") is not False
                or record.get("checkpoint_container_schema")
                != ["model_state"]
                or record.get("strict_state_dict_load") is not True
                or record.get("all_model_state_tensors_finite") is not True
                or record.get("frozen_eval") is not True
            ):
                raise RuntimeError(
                    f"{name} released checkpoint receipt changed during "
                    "Base cache build"
                )
            continue
        status = Path(str(record["formal_training_status"]))
        for path, expected, label in (
            (checkpoint, str(record["sha256"]), f"{name} checkpoint"),
            (
                status,
                str(record["formal_training_status_sha256"]),
                f"{name} status",
            ),
        ):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"{label} changed type: {path}")
            if sha256(path) != expected:
                raise RuntimeError(f"{label} changed during Base cache build")
        parity = record.get("global_fastpath_parity")
        if name == "global":
            if not isinstance(parity, dict):
                raise RuntimeError("Global parity receipt disappeared")
            parity_path = Path(str(parity["path"]))
            if parity_path.is_symlink() or not parity_path.is_file():
                raise RuntimeError(
                    f"Global parity artifact changed type: {parity_path}"
                )
            if sha256(parity_path) != parity.get("sha256"):
                raise RuntimeError(
                    "Global parity artifact changed during Base cache build"
                )
        elif parity is not None:
            raise RuntimeError(
                f"non-Global checkpoint {name} acquired a parity receipt"
            )


def _rotation_mask(joint_indices: tuple[int, ...]) -> tuple[int, ...]:
    indices: list[int] = []
    for joint in joint_indices:
        indices.extend((joint * 3, joint * 3 + 1, joint * 3 + 2))
    return tuple(indices)


def construct_rvq_inputs(
    pose: Any,
    facial: Any,
    trans: Any,
    contact: Any,
) -> tuple[dict[str, Any], Any]:
    """Exact tensor algebra from ``PreprocessProcessor.process``."""
    from utils import rotation_conversions as rc
    import torch

    batch, frames, _ = pose.shape
    jaw = pose[:, :, 66:69]
    jaw_matrix = rc.axis_angle_to_matrix(jaw.reshape(batch, frames, 1, 3))
    jaw_6d = rc.matrix_to_rotation_6d(jaw_matrix).reshape(batch, frames, 6)
    face = torch.cat([jaw_6d, facial], dim=2)

    hands_aa = pose[:, :, 25 * 3 : 55 * 3].reshape(
        batch, frames, 30, 3
    )
    hands = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(hands_aa)
    ).reshape(batch, frames, 180)

    upper_indices = torch.tensor(
        _rotation_mask(UPPER_JOINTS),
        dtype=torch.long,
        device=pose.device,
    )
    upper_aa = pose.index_select(2, upper_indices).reshape(
        batch, frames, 13, 3
    )
    upper = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(upper_aa)
    ).reshape(batch, frames, 78)

    lower_indices = torch.tensor(
        _rotation_mask(LOWER_JOINTS),
        dtype=torch.long,
        device=pose.device,
    )
    lower_aa = pose.index_select(2, lower_indices).reshape(
        batch, frames, 9, 3
    )
    lower_rot = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(lower_aa)
    ).reshape(batch, frames, 54)
    lower = torch.cat([lower_rot, trans, contact], dim=2)

    all_aa = pose.reshape(batch, frames, 55, 3)
    all_6d = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(all_aa)
    ).reshape(batch, frames, 330)
    latent_all = torch.cat(
        [all_6d, trans, contact],
        dim=-1,
    )
    expected = {
        "face": 106,
        "upper": 78,
        "hands": 180,
        "lower": 61,
    }
    components = {
        "face": face,
        "upper": upper,
        "hands": hands,
        "lower": lower,
    }
    for name, width in expected.items():
        if components[name].shape != (batch, frames, width):
            raise RuntimeError(
                f"invalid {name} RVQ input {components[name].shape}"
            )
    if latent_all.shape != (batch, frames, 337):
        raise RuntimeError(f"invalid latent_all {latent_all.shape}")
    return components, latent_all


def encode_window_batch(
    windows: list[dict[str, np.ndarray]],
    models: dict[str, Any],
    *,
    device: str,
) -> list[dict[str, np.ndarray]]:
    import torch

    if not windows:
        return []
    pose = torch.from_numpy(np.stack([item["pose"] for item in windows])).to(
        device=device,
        dtype=torch.float32,
    )
    facial = torch.from_numpy(
        np.stack([item["facial"] for item in windows])
    ).to(device=device, dtype=torch.float32)
    trans = torch.from_numpy(
        np.stack([item["trans"] for item in windows])
    ).to(device=device, dtype=torch.float32)
    contact = torch.from_numpy(
        np.stack([item["contact"] for item in windows])
    ).to(device=device, dtype=torch.float32)

    components, latent_all = construct_rvq_inputs(
        pose,
        facial,
        trans,
        contact,
    )
    indices: dict[str, Any] = {}
    zq: dict[str, Any] = {}
    with torch.inference_mode():
        # Preserve stock call order: all indices first, then all zq tensors.
        for name in RVQ_NAMES:
            indices[name] = models[name].map2index(components[name])
        for name in RVQ_NAMES:
            zq[name] = models[name].map2zq(components[name])

    batch_size = len(windows)
    for name in RVQ_NAMES:
        if indices[name].shape != (batch_size, 16, 6):
            raise RuntimeError(
                f"{name} index shape {indices[name].shape} != "
                f"{(batch_size, 16, 6)}"
            )
        if zq[name].shape != (6, batch_size, 16, 256):
            raise RuntimeError(
                f"{name} zq shape {zq[name].shape} != "
                f"{(6, batch_size, 16, 256)}"
            )

    output: list[dict[str, np.ndarray]] = []
    latent_np = latent_all.detach().float().cpu().numpy()
    for batch_index, window in enumerate(windows):
        sample: dict[str, np.ndarray] = {
            "tar_pose": window["pose"].astype(np.float32, copy=False),
            "beat": window["beat"].astype(np.float32, copy=False),
            "in_word": np.zeros(64, dtype=np.int64),
            "tar_id": window["speaker_id"].astype(np.int64, copy=False),
            "latent_all": latent_np[batch_index],
            "hubert": window["hubert"].astype(np.float32, copy=False),
        }
        for name in RVQ_NAMES:
            sample[f"tar_index_value_{name}_top"] = (
                indices[name][batch_index].detach().cpu().numpy().astype(
                    np.int64,
                    copy=False,
                )
            )
            # Retain the singleton batch dimension to match stock
            # to_np_squeezed(map2zq(...)): [Q,1,Tlatent,D].
            sample[f"zq_{name}"] = (
                zq[name][:, batch_index : batch_index + 1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        validate_base_sample(sample)
        output.append(sample)
    return output


def validate_base_sample(sample: dict[str, np.ndarray]) -> None:
    expected = {
        "tar_pose": (64, 165),
        "beat": (64, 3),
        "in_word": (64,),
        "tar_id": (64, 1),
        "latent_all": (64, 337),
        "hubert": (64, 1024),
    }
    for name in RVQ_NAMES:
        expected[f"tar_index_value_{name}_top"] = (16, 6)
        expected[f"zq_{name}"] = (6, 1, 16, 256)
    if set(sample) != set(expected):
        raise RuntimeError(
            f"Base sample fields differ: got={sorted(sample)}, "
            f"expected={sorted(expected)}"
        )
    for name, shape in expected.items():
        array = np.asarray(sample[name])
        if array.shape != shape:
            raise RuntimeError(f"Base {name} {array.shape} != {shape}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise RuntimeError(f"Base {name} contains non-finite values")
    if np.any(sample["in_word"]):
        raise RuntimeError("Base in_word placeholder is not all zero")
    if sample["in_word"].dtype != np.int64:
        raise RuntimeError("Base in_word must be int64")
    if sample["tar_id"].dtype != np.int64:
        raise RuntimeError("Base tar_id must be int64")
    if sample["tar_id"].min() < 0 or sample["tar_id"].max() > 3:
        raise RuntimeError("Base tar_id outside SHOW [0,3]")
    for name in RVQ_NAMES:
        index = sample[f"tar_index_value_{name}_top"]
        if index.dtype != np.int64 or index.min() < 0 or index.max() >= 256:
            raise RuntimeError(f"invalid {name} RVQ index")


def serialize_sample(sample: dict[str, np.ndarray]) -> bytes:
    # Uncompressed NPZ materially reduces preprocessing and DataLoader CPU
    # time.  Fixed ZIP metadata makes reruns byte-reproducible without altering
    # a single tensor value.
    return deterministic_npz_bytes(sample)


def batched(
    source: Iterable[dict[str, np.ndarray]],
    batch_size: int,
) -> Iterator[list[dict[str, np.ndarray]]]:
    batch: list[dict[str, np.ndarray]] = []
    for item in source:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def clip_windows(
    canonical: dict[str, np.ndarray],
    audio: dict[str, np.ndarray],
    *,
    usable_frames: int,
    length: int,
    stride: int,
) -> Iterator[dict[str, np.ndarray]]:
    raw_frames = canonical["pose"].shape[0]
    if (
        audio["beat"].shape[0] != raw_frames
        or audio["hubert"].shape[0] != raw_frames
    ):
        raise RuntimeError("canonical/audio time axes differ")
    if (
        usable_frames < 0
        or usable_frames > raw_frames
        or usable_frames % 30 != 0
    ):
        raise RuntimeError("invalid whole-second usable frame count")
    windows = max(0, (usable_frames - length) // stride + 1)
    for index in range(windows):
        start = index * stride
        end = start + length
        yield {
            "pose": canonical["pose"][start:end],
            "contact": canonical["contact"][start:end],
            "facial": canonical["facial"][start:end],
            "trans": canonical["trans"][start:end],
            "speaker_id": canonical["speaker_id"][start:end],
            "beat": audio["beat"][start:end],
            "hubert": audio["hubert"][start:end],
        }


def base_mode(args: argparse.Namespace) -> None:
    import lmdb
    import torch

    if args.window_length != 64 or args.stride != 20:
        raise RuntimeError("formal SemTalk Base windows are fixed at 64/20")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    if args.commit_interval <= 0:
        raise ValueError("--commit-interval must be positive")
    if args.expected_train_clips != 13_687:
        raise RuntimeError("formal Base train clip count must be 13687")
    if args.expected_entries != 127_286:
        raise RuntimeError("formal Base entry count must be 127286")

    output = Path(args.output_lmdb).resolve()
    summary_path = Path(args.summary_json).resolve()
    lineage_path = Path(args.lineage_json).resolve()
    forbid_nested_output(output, summary_path, lineage_path)
    refuse_existing(output, summary_path, lineage_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_paths = [Path(path) for path in args.canonical_manifest]
    canonical_summary_path = Path(args.canonical_summary)
    canonical_lineage_path = Path(args.canonical_lineage)
    source_receipt_record = source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    input_source_receipt_record = input_artifact_source_receipt(
        args.expected_input_source_commit,
        args.expected_input_source_tree,
    )
    audio_paths = [Path(path) for path in args.audio_manifest]
    audio_lineage_paths = [Path(path) for path in args.audio_lineage_json]
    canonical_rows, canonical_hashes = canonical_split_rows(
        canonical_paths,
        "train",
        expected_manifest_sha256=(
            args.expected_canonical_manifest_sha256
        ),
    )
    canonical_receipt = load_canonical_receipt(
        manifest_paths=canonical_paths,
        manifest_hashes=canonical_hashes,
        summary_path=canonical_summary_path,
        lineage_path=canonical_lineage_path,
        expected_manifest_sha256=(
            args.expected_canonical_manifest_sha256
        ),
        expected_summary_sha256=args.expected_canonical_summary_sha256,
        expected_lineage_sha256=args.expected_canonical_lineage_sha256,
        expected_canonical_source_commit=(
            args.expected_canonical_source_commit
        ),
        expected_canonical_source_tree=args.expected_canonical_source_tree,
    )
    audio_rows, audio_hashes = load_audio_rows(audio_paths)
    audio_lineage_records, audio_lineage_hashes = load_audio_lineages(
        audio_lineage_paths,
        audio_manifest_hashes=audio_hashes,
        canonical_manifest_hashes=canonical_hashes,
        canonical_receipt=canonical_receipt,
        canonical_lineage_contract_sha256=str(
            canonical_rows[0]["lineage_contract_sha256"]
        ),
        expected_train_clips=len(canonical_rows),
        expected_source_commit=args.expected_input_source_commit,
        expected_source_tree=args.expected_input_source_tree,
        expected_hubert_tree_sha256=args.expected_hubert_tree_sha256,
    )
    canonical_ids = [str(row["clip_id"]) for row in canonical_rows]
    if set(canonical_ids) != set(audio_rows):
        missing = sorted(set(canonical_ids) - set(audio_rows))
        extra = sorted(set(audio_rows) - set(canonical_ids))
        raise RuntimeError(
            f"audio/canonical exact-cover mismatch: missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
    audio_num_shards = require_exact_int(
        audio_lineage_records[0].get("num_shards"),
        "audio lineage num_shards",
    )
    audio_contract_by_shard = {
        require_exact_int(
            record.get("shard_id"),
            "audio lineage shard_id",
        ): str(
            record["audio_lineage_contract_sha256"]
        )
        for record in audio_lineage_records
    }
    for index, clip_id in enumerate(canonical_ids):
        row = audio_rows[clip_id]
        expected_shard = index % audio_num_shards
        if (
            require_exact_int(
                row.get("num_shards"),
                f"{clip_id} audio row num_shards",
            )
            != audio_num_shards
        ):
            raise RuntimeError(f"{clip_id}: audio row num_shards mismatch")
        if (
            require_exact_int(
                row.get("shard_id"),
                f"{clip_id} audio row shard_id",
            )
            != expected_shard
        ):
            raise RuntimeError(
                f"{clip_id}: audio row shard_id={row.get('shard_id')} "
                f"expected {expected_shard}"
            )
        if row.get("source_audio_field") != "source_wav":
            raise RuntimeError(f"{clip_id}: audio row did not use source_wav")
        canonical_row = canonical_rows[index]
        if row.get("lineage_contract_sha256") != (
            canonical_row["lineage_contract_sha256"]
        ):
            raise RuntimeError(f"{clip_id}: canonical lineage contract mismatch")
        if row.get("audio_lineage_contract_sha256") != (
            audio_contract_by_shard[expected_shard]
        ):
            raise RuntimeError(f"{clip_id}: audio lineage contract mismatch")
    if args.expected_train_clips and (
        len(canonical_rows) != args.expected_train_clips
    ):
        raise RuntimeError(
            f"train clips {len(canonical_rows)} != "
            f"--expected-train-clips {args.expected_train_clips}"
        )

    training_lineage_input = Path(
        args.representation_training_lineage_manifest
    )
    training_lineage_path = training_lineage_input.resolve()
    if (
        training_lineage_input.is_symlink()
        or not training_lineage_path.is_file()
    ):
        raise FileNotFoundError(training_lineage_path)
    training_lineage_sha = sha256(training_lineage_path)
    with training_lineage_path.open(encoding="utf-8") as handle:
        training_lineage = json.load(handle)
    if (
        not isinstance(training_lineage, dict)
        or training_lineage.get("status") != "complete"
        or training_lineage.get("format")
        != "semtalk_show_representation_lmdb_v2_global_foot"
        or require_exact_int(
            training_lineage.get("train_clips"),
            "representation lineage train_clips",
        )
        != len(canonical_rows)
        or training_lineage.get("canonical_receipt")
        != canonical_receipt
        or training_lineage.get("canonical_manifest_sha256")
        != canonical_hashes
        or training_lineage.get("source_receipt", {}).get("origin")
        != EXPECTED_ORIGIN
        or training_lineage.get("source_receipt", {}).get("commit")
        != args.expected_input_source_commit
        or training_lineage.get("source_receipt", {}).get("tree")
        != args.expected_input_source_tree
        or set(training_lineage.get("speaker_clip_counts", {}))
        != set(SPEAKER_MAP)
        or set(training_lineage.get("speaker_window_counts", {}))
        != set(SPEAKER_MAP)
        or sum(
            require_exact_int(
                value,
                "representation lineage speaker_clip_counts value",
            )
            for value in training_lineage.get(
                "speaker_clip_counts", {}
            ).values()
        )
        != len(canonical_rows)
        or sum(
            require_exact_int(
                value,
                "representation lineage speaker_window_counts value",
            )
            for value in training_lineage.get(
                "speaker_window_counts", {}
            ).values()
        )
        != require_exact_int(
            training_lineage.get("entries"),
            "representation lineage entries",
        )
    ):
        raise RuntimeError(
            "invalid representation training lineage manifest"
        )

    (
        models,
        checkpoint_records,
        prerequisite_source_receipt,
    ) = load_prerequisite_models(
        args,
        training_lineage_sha,
        source_receipt_record,
    )
    torch.set_grad_enabled(False)

    temp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    refuse_existing(temp)
    temp.mkdir()
    env = lmdb.open(
        str(temp),
        subdir=True,
        map_size=args.map_size_gib * 1024**3,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=False,
    )
    transaction = env.begin(write=True)
    entry_count = 0
    aggregate = hashlib.sha256()
    per_clip: list[dict[str, Any]] = []
    skipped_short: list[str] = []
    raw_frames_total = 0
    usable_frames_total = 0
    dropped_tail_frames_total = 0
    started = time.time()
    try:
        for canonical_row in canonical_rows:
            clip_id = str(canonical_row["clip_id"])
            audio_row = audio_rows[clip_id]
            if (
                str(audio_row["canonical_npz_sha256"])
                != str(canonical_row["canonical_npz_sha256"])
            ):
                raise RuntimeError(
                    f"{clip_id}: audio/canonical NPZ lineage mismatch"
                )
            if (
                str(audio_row["source_wav_sha256"])
                != str(canonical_row["source_wav_sha256"])
            ):
                raise RuntimeError(
                    f"{clip_id}: audio/source WAV lineage mismatch"
                )
            canonical, frames = load_canonical_clip(canonical_row)
            audio = load_audio_clip(
                audio_row,
                expected_frames=frames,
            )
            usable_frames = (frames // 30) * 30
            dropped_tail_frames = frames - usable_frames
            raw_frames_total += frames
            usable_frames_total += usable_frames
            dropped_tail_frames_total += dropped_tail_frames
            window_count = max(
                0,
                (usable_frames - args.window_length) // args.stride + 1,
            )
            if window_count == 0:
                skipped_short.append(clip_id)
            window_source = clip_windows(
                canonical,
                audio,
                usable_frames=usable_frames,
                length=args.window_length,
                stride=args.stride,
            )
            emitted = 0
            for input_batch in batched(
                window_source,
                args.inference_batch_size,
            ):
                output_batch = encode_window_batch(
                    input_batch,
                    models,
                    device=args.device,
                )
                for sample in output_batch:
                    value = serialize_sample(sample)
                    key = f"{entry_count:010d}".encode("ascii")
                    if not transaction.put(key, value, overwrite=False):
                        raise RuntimeError(f"duplicate LMDB key {key!r}")
                    aggregate.update(key)
                    aggregate.update(hashlib.sha256(value).digest())
                    entry_count += 1
                    emitted += 1
                    if entry_count % args.commit_interval == 0:
                        transaction.commit()
                        transaction = env.begin(write=True)
            if emitted != window_count:
                raise RuntimeError(
                    f"{clip_id}: emitted {emitted}/{window_count} windows"
                )
            per_clip.append(
                {
                    "clip_id": clip_id,
                    "canonical_npz": str(
                        Path(canonical_row["canonical_npz"]).resolve()
                    ),
                    "canonical_npz_sha256": str(
                        canonical_row["canonical_npz_sha256"]
                    ),
                    "audio_feature_npz": str(
                        Path(audio_row["audio_feature_npz"]).resolve()
                    ),
                    "audio_feature_npz_sha256": str(
                        audio_row["audio_feature_npz_sha256"]
                    ),
                    "raw_frames": frames,
                    "usable_frames": usable_frames,
                    "dropped_tail_frames": dropped_tail_frames,
                    "windows": window_count,
                    "speaker_id": require_exact_int(
                        canonical["speaker_id"][0, 0].item(),
                        f"{clip_id} canonical speaker_id",
                    ),
                }
            )
        transaction.commit()
        transaction = None
        env.sync(True)
    except BaseException:
        if transaction is not None:
            transaction.abort()
        env.close()
        shutil.rmtree(temp, ignore_errors=True)
        raise
    env.close()
    if entry_count <= 0:
        shutil.rmtree(temp, ignore_errors=True)
        raise RuntimeError("Base cache contains no windows")
    if args.expected_entries and entry_count != args.expected_entries:
        shutil.rmtree(temp, ignore_errors=True)
        raise RuntimeError(
            f"Base entries {entry_count} != "
            f"--expected-entries {args.expected_entries}"
        )
    try:
        final_source_receipt = source_receipt(
            args.expected_source_commit,
            args.expected_source_tree,
        )
        final_input_source_receipt = input_artifact_source_receipt(
            args.expected_input_source_commit,
            args.expected_input_source_tree,
        )
        final_canonical_rows, final_canonical_hashes = canonical_split_rows(
            canonical_paths,
            "train",
            expected_manifest_sha256=(
                args.expected_canonical_manifest_sha256
            ),
        )
        final_canonical_receipt = load_canonical_receipt(
            manifest_paths=canonical_paths,
            manifest_hashes=final_canonical_hashes,
            summary_path=canonical_summary_path,
            lineage_path=canonical_lineage_path,
            expected_manifest_sha256=(
                args.expected_canonical_manifest_sha256
            ),
            expected_summary_sha256=(
                args.expected_canonical_summary_sha256
            ),
            expected_lineage_sha256=(
                args.expected_canonical_lineage_sha256
            ),
            expected_canonical_source_commit=(
                args.expected_canonical_source_commit
            ),
            expected_canonical_source_tree=(
                args.expected_canonical_source_tree
            ),
        )
        final_audio_rows, final_audio_hashes = load_audio_rows(audio_paths)
        (
            final_audio_lineage_records,
            final_audio_lineage_hashes,
        ) = load_audio_lineages(
            audio_lineage_paths,
            audio_manifest_hashes=final_audio_hashes,
            canonical_manifest_hashes=final_canonical_hashes,
            canonical_receipt=final_canonical_receipt,
            canonical_lineage_contract_sha256=str(
                final_canonical_rows[0]["lineage_contract_sha256"]
            ),
            expected_train_clips=len(final_canonical_rows),
            expected_source_commit=args.expected_input_source_commit,
            expected_source_tree=args.expected_input_source_tree,
            expected_hubert_tree_sha256=args.expected_hubert_tree_sha256,
        )
        if (
            final_source_receipt != source_receipt_record
            or final_input_source_receipt != input_source_receipt_record
            or final_canonical_rows != canonical_rows
            or final_canonical_hashes != canonical_hashes
            or final_canonical_receipt != canonical_receipt
            or final_audio_rows != audio_rows
            or final_audio_hashes != audio_hashes
            or final_audio_lineage_records != audio_lineage_records
            or final_audio_lineage_hashes != audio_lineage_hashes
        ):
            raise RuntimeError(
                "formal source/canonical/audio receipts changed during "
                "Base cache construction"
            )
        if (
            training_lineage_input.is_symlink()
            or training_lineage_input.resolve() != training_lineage_path
            or not training_lineage_path.is_file()
            or sha256(training_lineage_path) != training_lineage_sha
            or json.loads(training_lineage_path.read_text())
            != training_lineage
        ):
            raise RuntimeError(
                "representation training lineage changed during "
                "Base cache construction"
            )
        revalidate_checkpoint_records(checkpoint_records)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    os.replace(temp, output)
    _fsync_parent(output)

    data_sha = sha256(output / "data.mdb")
    lock_sha = sha256(output / "lock.mdb")
    lineage = {
        "format": "semtalk_show_base_feature_lineage_v1",
        "status": "complete",
        "protocol": {
            "scope": "SemTalk Base only",
            "split": "train",
            "speakers": SPEAKER_MAP,
            "window_length": args.window_length,
            "stride": args.stride,
            "whole_second_policy": "usable_frames=floor(raw_frames/30)*30",
            "tail_policy": "drop_incomplete",
            "in_word": "int64_all_zero_unused_placeholder",
            "rvq": {
                "components": list(RVQ_NAMES),
                "input_dims": {
                    "face": 106,
                    "upper": 78,
                    "hands": 180,
                    "lower": 61,
                },
                "index_shape_per_window": [16, 6],
                "zq_shape_per_window": [6, 1, 16, 256],
            },
            "latent_all": "55xrot6d_plus_translation_plus_contact_337d",
            "global_checkpoint": (
                "verified formal prerequisite; not consumed by Base features"
            ),
            "forbidden_components": [
                "ASR",
                "TextGrid",
                "vocabulary",
                "CLIP",
                "emotion",
                "semantic",
                "SemGate",
                "Sparse",
            ],
        },
        "canonical_manifest_sha256": canonical_hashes,
        "canonical_receipt": canonical_receipt,
        "source_receipt": source_receipt_record,
        "input_artifact_source_receipt": input_source_receipt_record,
        "input_artifact_source_receipt_sha256": (
            canonical_file_payload_sha256(input_source_receipt_record)
        ),
        "audio_manifest_sha256": audio_hashes,
        "audio_lineage_json_sha256": audio_lineage_hashes,
        "audio_lineage": audio_lineage_records,
        "representation_training_lineage_manifest": str(
            training_lineage_path
        ),
        "representation_training_lineage_manifest_sha256": (
            training_lineage_sha
        ),
        "formal_checkpoints": checkpoint_records,
        "train_clips": len(canonical_rows),
        "raw_frames": raw_frames_total,
        "usable_frames": usable_frames_total,
        "dropped_tail_frames": dropped_tail_frames_total,
        "entries": entry_count,
        "entry_aggregate_sha256": aggregate.hexdigest(),
        "inference_batch_size": args.inference_batch_size,
        "runtime": runtime_record(),
        "argv": sys.argv,
    }
    if prerequisite_source_receipt is not None:
        lineage["prerequisite_source_receipt"] = (
            prerequisite_source_receipt
        )
        lineage["protocol"]["prerequisite_source"] = (
            RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE
        )
    atomic_json(lineage_path, lineage)
    lineage_sha = sha256(lineage_path)
    summary = {
        "format": "semtalk_show_base_lmdb_summary_v1",
        "status": "complete",
        "scope": "SemTalk Base only",
        "train_clips": len(canonical_rows),
        "raw_frames": raw_frames_total,
        "usable_frames": usable_frames_total,
        "dropped_tail_frames": dropped_tail_frames_total,
        "entries": entry_count,
        "skipped_short_clip_ids": skipped_short,
        "entry_aggregate_sha256": aggregate.hexdigest(),
        "lmdb": str(output),
        "data_mdb_sha256": data_sha,
        "lock_mdb_sha256": lock_sha,
        "lineage_json": str(lineage_path),
        "lineage_json_sha256": lineage_sha,
        "per_clip": per_clip,
        "started_unix": started,
        "completed_unix": time.time(),
    }
    atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": "base",
                "train_clips": len(canonical_rows),
                "entries": entry_count,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build audited SHOW audio/Base features without SemGate/Sparse",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    audio = subparsers.add_parser("audio", allow_abbrev=False)
    audio.add_argument(
        "--canonical-manifest",
        action="append",
        required=True,
    )
    audio.add_argument("--canonical-summary", required=True)
    audio.add_argument("--canonical-lineage", required=True)
    audio.add_argument(
        "--split",
        choices=("train", "test"),
        required=True,
    )
    audio.add_argument("--hubert-model", required=True)
    audio.add_argument("--output-dir", required=True)
    audio.add_argument("--output-manifest", required=True)
    audio.add_argument("--summary-json", required=True)
    audio.add_argument("--lineage-json", required=True)
    audio.add_argument("--device", required=True)
    audio.add_argument("--shard-id", type=int, required=True)
    audio.add_argument("--num-shards", type=int, required=True)
    audio.add_argument("--expected-total-clips", type=int, default=0)
    audio.add_argument("--expected-hubert-tree-sha256", required=True)
    audio.add_argument("--expected-source-commit", required=True)
    audio.add_argument("--expected-source-tree", required=True)
    audio.add_argument("--expected-canonical-source-commit", required=True)
    audio.add_argument("--expected-canonical-source-tree", required=True)
    audio.add_argument("--expected-canonical-manifest-sha256", required=True)
    audio.add_argument("--expected-canonical-summary-sha256", required=True)
    audio.add_argument("--expected-canonical-lineage-sha256", required=True)
    audio.add_argument("--max-frame-mismatch", type=int, default=1)

    base = subparsers.add_parser("base", allow_abbrev=False)
    base.add_argument(
        "--canonical-manifest",
        action="append",
        required=True,
    )
    base.add_argument("--canonical-summary", required=True)
    base.add_argument("--canonical-lineage", required=True)
    base.add_argument(
        "--audio-manifest",
        action="append",
        required=True,
    )
    base.add_argument(
        "--audio-lineage-json",
        action="append",
        required=True,
    )
    base.add_argument(
        "--representation-training-lineage-manifest",
        required=True,
    )
    base.add_argument(
        "--prerequisite-source",
        choices=PREREQUISITE_SOURCES,
        default=SHOW_TRAINED_PREREQUISITE_SOURCE,
    )
    for name in (*RVQ_NAMES, "global"):
        base.add_argument(f"--{name}-checkpoint", required=True)
        base.add_argument(f"--{name}-status-json")
    base.add_argument("--output-lmdb", required=True)
    base.add_argument("--summary-json", required=True)
    base.add_argument("--lineage-json", required=True)
    base.add_argument("--device", required=True)
    base.add_argument("--window-length", type=int, default=64)
    base.add_argument("--stride", type=int, default=20)
    base.add_argument("--inference-batch-size", type=int, default=16)
    base.add_argument("--map-size-gib", type=int, default=512)
    base.add_argument("--commit-interval", type=int, default=512)
    base.add_argument("--expected-train-clips", type=int, default=0)
    base.add_argument("--expected-entries", type=int, default=0)
    base.add_argument("--expected-hubert-tree-sha256", required=True)
    base.add_argument("--expected-source-commit", required=True)
    base.add_argument("--expected-source-tree", required=True)
    base.add_argument("--expected-input-source-commit")
    base.add_argument("--expected-input-source-tree")
    base.add_argument("--expected-canonical-source-commit", required=True)
    base.add_argument("--expected-canonical-source-tree", required=True)
    base.add_argument("--expected-canonical-manifest-sha256", required=True)
    base.add_argument("--expected-canonical-summary-sha256", required=True)
    base.add_argument("--expected-canonical-lineage-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "base":
        if (
            args.expected_input_source_commit is None
        ) != (
            args.expected_input_source_tree is None
        ):
            parser.error(
                "--expected-input-source-commit and "
                "--expected-input-source-tree must be supplied together"
            )
        statuses = {
            name: getattr(args, f"{name}_status_json")
            for name in (*RVQ_NAMES, "global")
        }
        if args.prerequisite_source == SHOW_TRAINED_PREREQUISITE_SOURCE:
            missing = sorted(
                name for name, path in statuses.items() if path is None
            )
            if missing:
                parser.error(
                    "show_trained_v1 requires status JSON for "
                    f"{', '.join(missing)}"
                )
        elif (
            args.prerequisite_source
            == RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE
        ):
            unexpected = sorted(
                name for name, path in statuses.items() if path is not None
            )
            if unexpected:
                parser.error(
                    "released_all_speakers_v1 forbids SHOW training status "
                    f"JSON for {', '.join(unexpected)}"
                )
    return args


def main() -> None:
    args = parse_args()
    args.expected_hubert_tree_sha256 = require_sha256(
        args.expected_hubert_tree_sha256,
        "--expected-hubert-tree-sha256",
    )
    args.expected_source_commit = require_git_oid(
        args.expected_source_commit,
        "--expected-source-commit",
    )
    args.expected_source_tree = require_git_oid(
        args.expected_source_tree,
        "--expected-source-tree",
    )
    normalize_input_artifact_source_args(args)
    args.expected_canonical_source_commit = require_git_oid(
        args.expected_canonical_source_commit,
        "--expected-canonical-source-commit",
    )
    args.expected_canonical_source_tree = require_git_oid(
        args.expected_canonical_source_tree,
        "--expected-canonical-source-tree",
    )
    args.expected_canonical_manifest_sha256 = require_sha256(
        args.expected_canonical_manifest_sha256,
        "--expected-canonical-manifest-sha256",
    )
    args.expected_canonical_summary_sha256 = require_sha256(
        args.expected_canonical_summary_sha256,
        "--expected-canonical-summary-sha256",
    )
    args.expected_canonical_lineage_sha256 = require_sha256(
        args.expected_canonical_lineage_sha256,
        "--expected-canonical-lineage-sha256",
    )
    if args.mode == "audio":
        audio_mode(args)
    elif args.mode == "base":
        base_mode(args)
    else:  # pragma: no cover - argparse makes this unreachable
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
