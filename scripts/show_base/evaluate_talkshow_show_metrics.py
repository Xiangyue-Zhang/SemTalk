#!/usr/bin/env python3
"""Offline SemTalk NPZ adapter for the released TalkSHOW SHOW metrics.

This module deliberately has no dependency on a SemTalk or GlobalDiff
generator.  It consumes the immutable canonical SHOW manifest, the finalized
SemTalk prediction manifest, and pinned metric-only assets.  A single
deterministic SemTalk prediction is interpreted as a delta distribution:

* released2 uses logical slots 0 and 1;
* paper16 uses logical slots 0 through 15;
* released face and the primary body reference use logical slot 0.

Every logical slot is bound to the same physical prediction artifact.  The
single prediction is repeated before the pinned metric models, producing the
same extractor/SMPL-X batch shapes as the released two- and sixteen-sample
paths while preserving one generator inference.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib
import importlib.abc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
import wave
import zipfile

import numpy as np


SCHEMA_VERSION = 1
REPORT_FORMAT = "semtalk_show_talkshow_metrics_v1"
PRIMARY_REAL_FEATURE_CACHE_FORMAT = (
    "semtalk_show_released2_real_feature_cache_v1"
)
PRIMARY_REAL_FEATURE_CACHE_PRODUCTION_AUTHORITY_FORMAT = (
    "semtalk_show_released2_real_feature_cache_production_authority_v1"
)
PRIMARY_REAL_FEATURE_CACHE_ENTRYPOINT = (
    "scripts/show_base/replay_released2_primary.py"
)
SEMTALK_OFFICIAL_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
PRIMARY_REPLAY_FORMAT = (
    "semtalk_show_released2_primary_fresh_replay_v1"
)
PRIMARY_SCREEN_FORMAT = (
    "semtalk_show_released2_primary_screen_v1"
)
PRIMARY_METRIC_PATH = "body.released2.metrics.FGD"
REPORT_PAYLOAD_HASH_ALGORITHM = (
    "canonical_json_utf8_sorted_compact_newline_v1"
)
DISTRIBUTION_PROTOCOL = (
    "deterministic_replication_of_single_prediction_v1"
)
DISTRIBUTION_RECEIPT_FORMAT = (
    "semtalk_show_deterministic_distribution_receipt_v2"
)
REPLICATION_ALGORITHM = "logical_reference_v1"
NUM_LOGICAL_SLOTS = 16
RELEASED2_SLOTS = (0, 1)
PAPER16_SLOTS = tuple(range(NUM_LOGICAL_SLOTS))
FACE_SLOT = 0
METRIC_INPUT_MATERIALIZATION = {
    "generator_inferences_per_clip": 1,
    "feature_extractor_batches": [2, 16],
    "smplx_body_batch": 16,
    "face_smplx_batch": 2,
    "materialization_operation": (
        "repeat_prediction_before_metric_model_v1"
    ),
}
POSE_FPS = 30
POSE_DIM = 165
EXPRESSION_DIM = 100
BETA_DIM = 300
SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
SPEAKER_NAMES = tuple(SHOW_SPEAKER_IDS)

OUTPUT_FIELDS = (
    "betas",
    "poses",
    "expressions",
    "trans",
    "model",
    "gender",
    "mocap_frame_rate",
)
CANONICAL_FIELDS = (
    "pose",
    "contact",
    "facial",
    "beta",
    "trans",
    "speaker_id",
)
CANONICAL_ROW_KEYS = {
    "global_index",
    "clip_id",
    "split",
    "speaker",
    "speaker_id",
    "video",
    "sequence",
    "source_pkl",
    "source_wav",
    "canonical_npz",
    "canonical_npz_relative",
    "global_foot_fastpath_contract",
    "lower_foot_local",
    "lower_foot_local_relative",
    "frames",
    "pose_fps",
    "wav_channels",
    "wav_sample_width",
    "wav_sample_rate",
    "wav_frames",
    "wav_mono_policy",
    "source_pkl_sha256",
    "source_wav_sha256",
    "canonical_npz_sha256",
    "lower_foot_local_sha256",
    "lineage_contract_sha256",
}
VAL_PREDICTION_ROW_KEYS = {
    "global_index",
    "split",
    "source_clip_id",
    "canonical_clip_id",
    "frames",
    "epoch",
    "candidate_checkpoint_sha256",
    "prediction",
    "ground_truth",
}
TEST_PREDICTION_ROW_KEYS = {
    "global_index",
    "source_clip_id",
    "canonical_clip_id",
    "speaker",
    "speaker_id",
    "frames",
    "canonical_npz",
    "canonical_npz_sha256",
    "audio_feature_npz",
    "audio_feature_npz_sha256",
    "prediction",
    "ground_truth",
    "evaluation_index",
}
SHARD_TEST_PREDICTION_ROW_KEYS = (
    TEST_PREDICTION_ROW_KEYS - {"evaluation_index"}
)
SHARD_SUMMARY_KEYS = {
    "format",
    "status",
    "shard_id",
    "num_shards",
    "selected_clips",
    "expected_test_clips",
    "manifest_sha256",
    "contract_sha256",
    "runtime_sha256",
    "finite",
    "exact_once",
}
SHARD_LINEAGE_KEYS = {
    "format",
    "status",
    "shard_id",
    "num_shards",
    "manifest_sha256",
    "contract",
    "contract_sha256",
    "runtime",
    "runtime_sha256",
    "checkpoints",
}
ARTIFACT_KEYS = {"path", "bytes", "sha256"}
SHA256_LENGTH = 64
PAYLOAD_HASH_ALGORITHM = "canonical_json_utf8_sorted_compact_v1"
BASE_FINAL_AUTHORITY_MODULE = "scripts.show_base.base_final_authority"

FORMAL_SPLITS = {
    "val": {
        "count": 1_715,
        "global_start": 13_687,
        "global_stop": 15_402,
    },
    "test": {
        "count": 1_708,
        "global_start": 15_402,
        "global_stop": 17_110,
    },
}
FORMAL_MIN_FRAMES_EXCLUSIVE = 60
CANONICAL_SOURCE_AUDIO_RATE = 22_000
CANONICAL_WAV_SAMPLE_WIDTH = 2
CANONICAL_WAV_MONO_POLICY = (
    "librosa.load(sr=None,mono=True):arithmetic_channel_mean"
)

TALKSHOW_METRIC_COMMIT = "9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
TALKSHOW_METRIC_TREE = "d993229539e63442a1f1327bae6d80d35f97c521"
FEATURE_EXTRACTOR_SHA256 = (
    "154259bfe8ae1e0fb477ba5afdd5674659ef9eb4d44d4ad49cd2ef4d20ccecfb"
)
SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
TALKSHOW_PATCH_MARKER = {
    "upstream_commit": TALKSHOW_METRIC_COMMIT,
    "patch_version": "globaldiff-show-metric-v1",
    "scope": "released-show-body-feature-extractor-only",
    "source_file": "nets/__init__.py",
    "source_sha256": (
        "d12f3ebd1b8f4b251085061404d72530df58bc972272995a3810236aefdd4d21"
    ),
    "patched_sha256": (
        "e0e277d46475c203d78d355affd01a9e6f55462fbaa1911d140f91c438a0bef2"
    ),
}
TALKSHOW_FGD_ENTRY_MODULES = ("nets.body_ae",)
TALKSHOW_FGD_SOURCE_FILES = (
    "data_utils/__init__.py",
    "data_utils/consts.py",
    "data_utils/dataloader_torch.py",
    "data_utils/lower_body.py",
    "data_utils/mesh_dataset.py",
    "data_utils/rotation_conversion.py",
    "data_utils/utils.py",
    "nets/__init__.py",
    "nets/base.py",
    "nets/body_ae.py",
    "nets/layers.py",
    "nets/spg/s2glayers.py",
    "nets/spg/vqvae_1d.py",
    "nets/spg/vqvae_modules.py",
    "nets/spg/wav2vec.py",
)
TALKSHOW_FGD_IMPORT_ALGORITHM = "python-ast-local-import-closure-v1"
TALKSHOW_TRUSTED_UPSTREAM_FILES = {
    "data_utils/__init__.py": (
        "6d36625d003d93a50fa708c64ba15ba47651148cc81e730e84bf86eaca7e2009",
        "7c3cab8cad67c5e952924944f34319e2c6ff5985",
    ),
    "data_utils/consts.py": (
        "3a57929470837cc55d9be5802fd778435ba71be3e5b6dcb8a469230ec1419097",
        "70406b9d458588030508ca656492d274f48cbf3f",
    ),
    "data_utils/dataloader_torch.py": (
        "93b9c47dae13ffd4be71f0e44451719e2bd57b691b43292b25cc1470bc439f99",
        "dc0bf81c1ccc86580fdd013c0b3e81fae190a65d",
    ),
    "data_utils/lower_body.py": (
        "6eae37ef17760b5ebad60bf25ae7149aa5244cba8d9627eafe77b71616931163",
        "501a83c7c83bbcd97c6dee09b809b1c75be45213",
    ),
    "data_utils/mesh_dataset.py": (
        "ae02bc017e3491a7e678bbad0a1f693a5d9900583310da0450a8d83ea682b193",
        "9d19c1e512e3e1aaed645791acf88345af4c9bb9",
    ),
    "data_utils/rotation_conversion.py": (
        "4f9cd089f2cdc031e16be435f116c10a5498772d834b4da9a03947041fa09f82",
        "770c3bf36f05fcaf89cbb03e17035357f3c0a4df",
    ),
    "data_utils/utils.py": (
        "75a8c844a62d962a88d61af745e0231ac6249138b23ebb602b3de000bd2a47f3",
        "a6b9e713d75ff61dedab869e049dfba2db87ddb6",
    ),
    "nets/__init__.py": (
        "d12f3ebd1b8f4b251085061404d72530df58bc972272995a3810236aefdd4d21",
        "0669d82d7506b314fa3fccd7dd412445fa0b37e1",
    ),
    "nets/base.py": (
        "e5b1931dfdcba45a068818c5543a75f8dcfdc843cfa733e846483b401a0d19bd",
        "08c07caa27ba642dd5a48cebfcf51e4e79edd574",
    ),
    "nets/body_ae.py": (
        "a4065c7a086b48dcb7b161202ebe9756dd8d7a4c0df379e16ad68532368fdc4b",
        "3a9f8bc0ee92f8410da71d711bb19dbcc254d1af",
    ),
    "nets/layers.py": (
        "19c6f29ae94131991575b9fa2cd36d6d8103d064c1abca626f4cc37d552deced",
        "79251b42b6e0fe839ec04dc38472ef36165208ac",
    ),
    "nets/spg/s2glayers.py": (
        "ad2f5d56d25f0e75131e3b524e14bdc7c9f0fa22a696539c89563f62c29d68d7",
        "2a439e6bc0c4973586d39f3b113aa3752ff077fa",
    ),
    "nets/spg/vqvae_1d.py": (
        "c4f8f2a7862a3835b250403e8533c3387523a057e3b244e73266b6f1034a783f",
        "0cd15bd6439b949bf89098af274b3e7ccac9b5f5",
    ),
    "nets/spg/vqvae_modules.py": (
        "6ae6e1c5860afc420876942616cbc547aceb245aa97ba4bf94f4ca51080fa012",
        "5c83bc0399bc3bc034881407ed49d223d8c86ba9",
    ),
    "nets/spg/wav2vec.py": (
        "96533ea3f33dc2908bb35c0b32ebdf919968226d2ce4e06200437a613b34c146",
        "a5d0eff66e67de14ceba283fa6ce43f156c7ddc2",
    ),
}

PROTOCOLS = {
    "released2": {
        "logical_slots": list(RELEASED2_SLOTS),
        "num_samples": len(RELEASED2_SLOTS),
        "bc_sample_policy": "first",
    },
    "paper16": {
        "logical_slots": list(PAPER16_SLOTS),
        "num_samples": len(PAPER16_SLOTS),
        "bc_sample_policy": "all",
    },
}

BC_CHANGE_ANGLE = np.asarray(
    [6.0181e-05, 5.1597e-05, 2.1344e-04, 2.1899e-04],
    dtype=np.float64,
)
TALKSHOW_RELEASED_LOWER_POSE = np.asarray(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0747,
        -0.0158,
        -0.0152,
        -1.1826512813568115,
        0.23866955935955048,
        0.15146760642528534,
        -1.2604516744613647,
        -0.3160211145877838,
        -0.1603458970785141,
        1.1654603481292725,
        0.0,
        0.0,
        1.2521806955337524,
        0.041598282754421234,
        -0.06312154978513718,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


class MetricAdapterContractError(RuntimeError):
    """Raised when the offline metric evidence is not self-consistent."""


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


def compact_canonical_json_bytes(value: Any) -> bytes:
    """Canonical receipt bytes shared with deterministic_replication_gate."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compact_canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(compact_canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fresh_local_control_module(filename: str) -> Any:
    expected = Path(__file__).resolve().parent / filename
    path, payload = _safe_file_snapshot(
        str(expected),
        f"control module {filename}",
    )
    module_name = filename.removesuffix(".py")
    module = ModuleType(f"scripts.show_base.{module_name}")
    module.__file__ = str(path)
    module.__package__ = "scripts.show_base"
    module.__loader__ = None
    try:
        code = compile(
            payload,
            str(path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except BaseException as exc:
        raise MetricAdapterContractError(
            f"cannot source-load control module {filename}"
        ) from exc
    return module


def _replication_gate_module() -> Any:
    """Load the sole deterministic-replication receipt implementation."""

    module = _fresh_local_control_module(
        "deterministic_replication_gate.py",
    )
    if (
        getattr(module, "PAYLOAD_HASH_ALGORITHM", None)
        != PAYLOAD_HASH_ALGORITHM
        or getattr(module, "DISTRIBUTION_FORMAT", None)
        != DISTRIBUTION_RECEIPT_FORMAT
        or getattr(module, "PROTOCOL", None) != DISTRIBUTION_PROTOCOL
        or not callable(
            getattr(module, "variation_policy_receipt", None)
        )
        or not callable(
            getattr(module, "validate_distribution_receipt", None)
        )
        or not callable(
            getattr(
                module,
                "build_distribution_receipt_from_validated_artifacts",
                None,
            )
        )
    ):
        raise MetricAdapterContractError(
            "deterministic replication gate API/schema version mismatch"
        )
    return module


def _base_final_authority_module() -> Any:
    module = _fresh_local_control_module("base_final_authority.py")
    if (
        getattr(module, "FORMAT", None)
        != "semtalk_show_base_final_test_authority_v1"
        or not callable(getattr(module, "validate_test_authority", None))
    ):
        raise MetricAdapterContractError(
            "base_final_authority API/schema mismatch"
        )
    return module


def _fresh_base_selector_module() -> Any:
    authority = _base_final_authority_module()
    loader = getattr(authority, "_control_module", None)
    if not callable(loader):
        raise MetricAdapterContractError(
            "base_final_authority fresh control loader is unavailable"
        )
    try:
        selector = loader("talkshow_base_val_contract")
    except Exception as exc:
        raise MetricAdapterContractError(
            "cannot source-load the Base validation selector"
        ) from exc
    if any(
        not callable(getattr(selector, name, None))
        for name in (
            "validate_val_inputs",
            "validate_pipeline",
            "validate_val_inference_lineage",
        )
    ):
        raise MetricAdapterContractError(
            "Base validation selector API/schema mismatch"
        )
    return selector


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MetricAdapterContractError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _require_exact_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise MetricAdapterContractError(
            f"{label} must be an exact integer, got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise MetricAdapterContractError(
            f"{label} must be >= {minimum}, got {value}"
        )
    return value


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise MetricAdapterContractError(
            f"{name} contains NaN or infinity"
        )
    return array


def _resolved_regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise MetricAdapterContractError(
            f"{label} must be an absolute path: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MetricAdapterContractError(
            f"{label} is not a canonical regular file: {candidate}"
        ) from exc
    if resolved != candidate:
        raise MetricAdapterContractError(
            f"{label} path must be canonical and contain no symlink: "
            f"{candidate}"
        )
    return candidate


def _safe_file_snapshot(
    path: str | Path,
    label: str,
) -> tuple[Path, bytes]:
    candidate = _resolved_regular_file(path, label)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )

    def identity(metadata: os.stat_result) -> tuple[int, ...]:
        return tuple(
            int(getattr(metadata, field)) for field in stable_fields
        )

    try:
        path_before = os.lstat(candidate)
    except OSError as exc:
        raise MetricAdapterContractError(
            f"cannot safely inspect {label}: {candidate}"
        ) from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
        path_before.st_mode
    ):
        raise MetricAdapterContractError(
            f"{label} must be a regular non-symlink file"
        )
    parts = candidate.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise MetricAdapterContractError(
            f"{label} must be below the filesystem root"
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
        file_fd = os.open(
            parts[-1],
            file_flags,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise MetricAdapterContractError(
                f"{label} must be a regular non-symlink file"
            )
        if identity(path_before) != identity(before):
            raise MetricAdapterContractError(
                f"{label} changed before it was opened"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if identity(before) != identity(after):
            raise MetricAdapterContractError(
                f"{label} changed while it was read"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise MetricAdapterContractError(
                f"{label} size changed while it was read"
            )
        leaf_after = os.stat(
            parts[-1],
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        path_after = os.lstat(candidate)
        if (
            identity(after) != identity(leaf_after)
            or identity(after) != identity(path_after)
        ):
            raise MetricAdapterContractError(
                f"{label} path changed while it was read"
            )
        return candidate, payload
    except MetricAdapterContractError:
        raise
    except OSError as exc:
        raise MetricAdapterContractError(
            f"cannot safely read {label}: {candidate}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _resolved_canonical_directory(
    path: str | Path,
    label: str,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise MetricAdapterContractError(
            f"{label} must be an absolute path: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MetricAdapterContractError(
            f"{label} is not a canonical directory: {candidate}"
        ) from exc
    if resolved != candidate:
        raise MetricAdapterContractError(
            f"{label} path must be canonical and contain no symlink: "
            f"{candidate}"
        )
    parts = candidate.parts
    if not parts or parts[0] != os.sep:
        raise MetricAdapterContractError(
            f"{label} must be below the filesystem root"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    directory_fd: int | None = None
    try:
        directory_fd = os.open(os.sep, directory_flags)
        for component in parts[1:]:
            next_fd = os.open(
                component,
                directory_flags,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise MetricAdapterContractError(
                f"{label} must be a non-symlink directory"
            )
        return candidate
    except MetricAdapterContractError:
        raise
    except OSError as exc:
        raise MetricAdapterContractError(
            f"cannot safely resolve {label}: {candidate}"
        ) from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _verified_file_snapshot(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    expected = _require_sha256(expected_sha256, f"{label} expected SHA")
    resolved, payload = _safe_file_snapshot(path, label)
    observed = sha256_bytes(payload)
    if observed != expected:
        raise MetricAdapterContractError(
            f"{label} SHA-256 {observed} != {expected}"
        )
    return resolved, payload


def _strict_json_snapshot(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetricAdapterContractError(
            f"{label} is not valid JSON"
        ) from exc
    if type(value) is not dict:
        raise MetricAdapterContractError(f"{label} must be a JSON object")
    if canonical_json_bytes(value) != payload:
        raise MetricAdapterContractError(
            f"{label} is not canonical JSON bytes"
        )
    return value


def _strict_jsonl_snapshot(
    payload: bytes,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise MetricAdapterContractError(
                f"{label} line {line_number} lacks a newline"
            )
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetricAdapterContractError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if type(value) is not dict or canonical_json_bytes(value) != line:
            raise MetricAdapterContractError(
                f"{label} line {line_number} is not a canonical JSON object"
            )
        rows.append(value)
    if not rows:
        raise MetricAdapterContractError(f"{label} is empty")
    return rows


def canonical_clip_id(source_clip_id: str) -> str:
    pieces = source_clip_id.split("/")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        raise MetricAdapterContractError(
            f"invalid canonical SHOW clip ID: {source_clip_id!r}"
        )
    speaker, _video, sequence = pieces
    if speaker not in SHOW_SPEAKER_IDS:
        raise MetricAdapterContractError(
            f"unknown SHOW speaker {speaker!r}"
        )
    if (
        "__" in speaker
        or sequence in {"", ".", ".."}
        or "/" in sequence
        or "\x00" in sequence
    ):
        raise MetricAdapterContractError(
            f"unsafe canonical SHOW clip ID: {source_clip_id!r}"
        )
    return f"{speaker}__{sequence}"


@dataclass
class FeatureMoments:
    """Mergeable sufficient statistics using the unbiased covariance."""

    count: int = 0
    dimension: int = 0
    feature_sum: np.ndarray | None = None
    feature_outer_sum: np.ndarray | None = None

    def update(self, features: Any) -> None:
        array = _finite_array(features, name="features")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or 0 in array.shape:
            raise MetricAdapterContractError(
                f"features must be a non-empty matrix, got {array.shape}"
            )
        if self.dimension not in (0, int(array.shape[1])):
            raise MetricAdapterContractError(
                "feature dimension changed during evaluation"
            )
        if self.count == 0:
            self.dimension = int(array.shape[1])
            self.feature_sum = np.zeros(self.dimension, dtype=np.float64)
            self.feature_outer_sum = np.zeros(
                (self.dimension, self.dimension),
                dtype=np.float64,
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        self.count += int(array.shape[0])
        self.feature_sum += array.sum(axis=0, dtype=np.float64)
        self.feature_outer_sum += array.T @ array

    def merge(self, other: "FeatureMoments") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.dimension = other.dimension
            assert other.feature_sum is not None
            assert other.feature_outer_sum is not None
            self.feature_sum = other.feature_sum.copy()
            self.feature_outer_sum = other.feature_outer_sum.copy()
            return
        if self.dimension != other.dimension:
            raise MetricAdapterContractError(
                "cannot merge moments with different feature dimensions"
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        assert other.feature_sum is not None
        assert other.feature_outer_sum is not None
        self.count += other.count
        self.feature_sum += other.feature_sum
        self.feature_outer_sum += other.feature_outer_sum

    def mean_and_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise MetricAdapterContractError(
                "at least two feature rows are required"
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        mean = self.feature_sum / self.count
        covariance = (
            self.feature_outer_sum
            - np.outer(self.feature_sum, self.feature_sum) / self.count
        ) / (self.count - 1)
        covariance = (covariance + covariance.T) * 0.5
        if not np.isfinite(covariance).all():
            raise MetricAdapterContractError(
                "unbiased feature covariance is non-finite"
            )
        return mean, covariance

    def to_json(self) -> dict[str, Any]:
        if self.count < 1:
            raise MetricAdapterContractError(
                "cannot serialize empty feature moments"
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        return {
            "count": self.count,
            "dimension": self.dimension,
            "sum": self.feature_sum.tolist(),
            "outer_sum": self.feature_outer_sum.tolist(),
            "covariance_denominator": self.count - 1,
            "covariance_estimator": "unbiased_ddof_1",
        }

    @classmethod
    def from_json(
        cls,
        value: Any,
        *,
        expected_count: int,
        label: str,
    ) -> "FeatureMoments":
        if type(value) is not dict or set(value) != {
            "count",
            "dimension",
            "sum",
            "outer_sum",
            "covariance_denominator",
            "covariance_estimator",
        }:
            raise MetricAdapterContractError(
                f"{label} feature-statistics schema mismatch"
            )
        count = _require_exact_int(
            value["count"],
            f"{label} count",
            minimum=2,
        )
        dimension = _require_exact_int(
            value["dimension"],
            f"{label} dimension",
            minimum=1,
        )
        if (
            count != expected_count
            or value["covariance_denominator"] != count - 1
            or value["covariance_estimator"] != "unbiased_ddof_1"
        ):
            raise MetricAdapterContractError(
                f"{label} unbiased-covariance receipt mismatch"
            )
        feature_sum = _finite_array(
            value["sum"],
            name=f"{label} sum",
        )
        outer_sum = _finite_array(
            value["outer_sum"],
            name=f"{label} outer sum",
        )
        if (
            feature_sum.shape != (dimension,)
            or outer_sum.shape != (dimension, dimension)
        ):
            raise MetricAdapterContractError(
                f"{label} sufficient-statistics shape mismatch"
            )
        return cls(
            count=count,
            dimension=dimension,
            feature_sum=np.asarray(feature_sum, dtype=np.float64).copy(),
            feature_outer_sum=np.asarray(
                outer_sum,
                dtype=np.float64,
            ).copy(),
        )


def frechet_distance(first: FeatureMoments, second: FeatureMoments) -> float:
    mean_1, covariance_1 = first.mean_and_covariance()
    mean_2, covariance_2 = second.mean_and_covariance()
    if mean_1.shape != mean_2.shape:
        raise MetricAdapterContractError("feature dimensions differ")
    eigenvalues_1, eigenvectors_1 = np.linalg.eigh(covariance_1)
    scale = max(
        1.0,
        float(np.linalg.norm(covariance_1, ord=2)),
        float(np.linalg.norm(covariance_2, ord=2)),
    )
    tolerance = 1e-8 * scale
    if float(eigenvalues_1.min(initial=0.0)) < -tolerance:
        raise MetricAdapterContractError(
            "first covariance is not positive semidefinite"
        )
    root_1 = (
        eigenvectors_1
        * np.sqrt(np.clip(eigenvalues_1, 0.0, None))
    ) @ eigenvectors_1.T
    middle = root_1 @ covariance_2 @ root_1
    middle = (middle + middle.T) * 0.5
    middle_eigenvalues = np.linalg.eigvalsh(middle)
    if float(middle_eigenvalues.min(initial=0.0)) < -tolerance:
        raise MetricAdapterContractError(
            "covariance product is not positive semidefinite"
        )
    difference = mean_1 - mean_2
    distance = (
        float(difference @ difference)
        + float(np.trace(covariance_1))
        + float(np.trace(covariance_2))
        - 2.0
        * float(np.sqrt(np.clip(middle_eigenvalues, 0.0, None)).sum())
    )
    if distance < 0.0 and abs(distance) <= 1e-7 * scale:
        distance = 0.0
    if not math.isfinite(distance) or distance < 0.0:
        raise MetricAdapterContractError(
            f"invalid Fréchet distance: {distance}"
        )
    return distance


def variation_from_joints(generated_joints: Any) -> float:
    joints = _finite_array(generated_joints, name="generated joints")
    if joints.ndim != 4 or joints.shape[0] < 2 or joints.shape[-1] != 3:
        raise MetricAdapterContractError(
            "generated joints must be [B,T,J,3] with B >= 2"
        )
    coordinate_variance = np.var(joints, axis=0, ddof=1)
    value = np.linalg.norm(
        coordinate_variance,
        axis=-1,
    ).sum(axis=-1).mean()
    if not math.isfinite(float(value)):
        raise MetricAdapterContractError("Variation is non-finite")
    return float(value)


def bc_components_for_sequence(
    joints: Any,
    audio_beat_times: Any,
    *,
    threshold: float = 0.01,
    sigma: float = 0.1,
) -> tuple[float, int]:
    sequence = _finite_array(joints, name="BC joints").copy()
    beats = _finite_array(
        audio_beat_times,
        name="audio beat times",
    ).reshape(-1)
    if (
        sequence.ndim != 3
        or sequence.shape[1] < 22
        or sequence.shape[2] != 3
    ):
        raise MetricAdapterContractError(
            "BC joints must be [T,J,3] with J >= 22"
        )
    if sequence.shape[0] < 3 or beats.size == 0:
        return 0.0, 0
    sequence[:, 15:21] = sequence[:, 16:22]
    vectors = sequence[:, 15:21] - sequence[:, 13:19]
    inner = np.einsum(
        "kij,kij->ki",
        vectors[:, 2:],
        vectors[:, :-2],
    )
    angles = np.arccos(np.clip(inner, -1.0, 1.0)) / math.pi
    angular_velocity = (
        np.abs(angles[1:] - angles[:-1])
        / BC_CHANGE_ANGLE.reshape(1, -1)
        / len(BC_CHANGE_ANGLE)
    )
    differences = np.concatenate(
        [np.zeros((1, 4), dtype=np.float64), angular_velocity],
        axis=0,
    )
    numerator = 0.0
    denominator = 0
    for channel in range(differences.shape[1]):
        motion_beats = []
        for frame in range(1, sequence.shape[0] - 1):
            current = differences[frame, channel]
            previous = differences[frame - 1, channel]
            following = differences[frame + 1, channel]
            if (
                current < previous
                and current < following
                and (
                    previous - current >= threshold
                    or following - current >= threshold
                )
            ):
                motion_beats.append(frame / POSE_FPS)
        if not motion_beats:
            continue
        motion = np.asarray(motion_beats, dtype=np.float64)
        squared_distance = np.min(
            np.square(beats.reshape(-1, 1) - motion.reshape(1, -1)),
            axis=1,
        )
        numerator += float(
            np.exp(-squared_distance / (2.0 * sigma * sigma)).sum()
        )
        denominator += int(beats.size)
    return numerator, denominator


def bc_components_for_batch(
    generated_joints: Any,
    audio_beat_times: Any,
    sample_policy: str,
) -> tuple[float, int]:
    """Return released BC sufficient statistics for an actual metric batch."""

    joints = _finite_array(generated_joints, name="generated joints")
    if joints.ndim != 4:
        raise MetricAdapterContractError(
            "generated joints must be [B,T,J,3]"
        )
    if sample_policy == "first":
        indices: Iterable[int] = (0,)
    elif sample_policy == "all":
        indices = range(joints.shape[0])
    else:
        raise MetricAdapterContractError(
            f"unknown BC sample policy: {sample_policy}"
        )
    numerator = 0.0
    denominator = 0
    for index in indices:
        partial_numerator, partial_denominator = (
            bc_components_for_sequence(
                joints[index],
                audio_beat_times,
            )
        )
        numerator += partial_numerator
        denominator += partial_denominator
    return numerator, denominator


def released_face_metrics(
    ground_truth: Any,
    prediction: Any,
) -> dict[str, float]:
    gt = _finite_array(ground_truth, name="face ground truth")
    predicted = _finite_array(prediction, name="face prediction")
    if (
        gt.shape != predicted.shape
        or gt.ndim != 3
        or gt.shape[-1] != 3
    ):
        raise MetricAdapterContractError(
            "face joints must have the same [T,J,3] shape"
        )
    if gt.shape[0] < 2 or gt.shape[1] < 75:
        raise MetricAdapterContractError(
            "face metric input is too short or lacks landmarks"
        )
    jaw_gt = gt[:, 22:25]
    jaw_prediction = predicted[:, 22:25]
    landmark_gt = gt[:, 74:]
    landmark_prediction = predicted[:, 74:]
    jaw_l1 = float(
        np.linalg.norm(jaw_gt - jaw_prediction, axis=-1).sum(-1).mean()
    )
    landmark_l1 = float(
        np.linalg.norm(
            landmark_gt - landmark_prediction,
            axis=-1,
        ).sum(-1).mean()
    )
    face_gt = np.concatenate((jaw_gt, landmark_gt), axis=1)
    face_prediction = np.concatenate(
        (jaw_prediction, landmark_prediction),
        axis=1,
    )
    gt_velocity = np.linalg.norm(
        face_gt[1:] - face_gt[:-1],
        axis=-1,
    )
    prediction_velocity = np.linalg.norm(
        face_prediction[1:] - face_prediction[:-1],
        axis=-1,
    )
    lvd = float(
        np.abs(prediction_velocity - gt_velocity).sum(-1).mean()
    )
    result = {
        "jaw_l1": jaw_l1,
        "landmark_l1": landmark_l1,
        "LVD": lvd,
        "face_l2_combined": jaw_l1 + landmark_l1,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise MetricAdapterContractError("face metrics are non-finite")
    return result


def reorder_to_talkshow(
    poses_165: Any,
    expressions_100: Any,
) -> np.ndarray:
    poses = np.asarray(poses_165, dtype=np.float32)
    expressions = np.asarray(expressions_100, dtype=np.float32)
    if poses.shape[:-1] != expressions.shape[:-1]:
        raise MetricAdapterContractError(
            "pose/expression leading shapes differ"
        )
    if poses.shape[-1] != 165 or expressions.shape[-1] != 100:
        raise MetricAdapterContractError(
            "expected pose width 165 and expression width 100"
        )
    output = np.empty((*poses.shape[:-1], 265), dtype=np.float32)
    output[..., 0:3] = poses[..., 66:69]
    output[..., 3:6] = poses[..., 69:72]
    output[..., 6:9] = poses[..., 72:75]
    output[..., 9:12] = poses[..., 0:3]
    output[..., 12:75] = poses[..., 3:66]
    output[..., 75:120] = poses[..., 75:120]
    output[..., 120:165] = poses[..., 120:165]
    output[..., 165:265] = expressions
    return output


def released_body_parameters(parameters_265: Any) -> np.ndarray:
    parameters = np.asarray(parameters_265)
    if parameters.ndim < 2 or parameters.shape[-1] != 265:
        raise MetricAdapterContractError(
            "TalkSHOW body parameters must end in 265 channels"
        )
    if not np.issubdtype(parameters.dtype, np.floating):
        raise MetricAdapterContractError(
            "TalkSHOW body parameters must be floating point"
        )
    if not np.isfinite(parameters).all():
        raise MetricAdapterContractError(
            "TalkSHOW body parameters contain NaN or infinity"
        )
    lower = TALKSHOW_RELEASED_LOWER_POSE.astype(
        parameters.dtype,
        copy=False,
    )
    prefix_shape = parameters.shape[:-1]
    fixed = np.concatenate(
        (
            parameters[..., 0:3],
            np.broadcast_to(lower[0:15], (*prefix_shape, 15)),
            parameters[..., 18:21],
            np.broadcast_to(lower[15:21], (*prefix_shape, 6)),
            parameters[..., 27:30],
            np.broadcast_to(lower[21:27], (*prefix_shape, 6)),
            parameters[..., 36:39],
            np.broadcast_to(lower[27:33], (*prefix_shape, 6)),
            parameters[..., 45:265],
        ),
        axis=-1,
    ).copy()
    if fixed.shape != parameters.shape:
        raise MetricAdapterContractError(
            "TalkSHOW fixed-lower transform changed shape"
        )
    fixed[..., 0:3] = 0.0
    fixed[..., 165:265] = 0.0
    return fixed


def released_face_parameters(
    ground_truth_265: Any,
    prediction_265: Any,
) -> tuple[np.ndarray, np.ndarray]:
    ground_truth = np.asarray(ground_truth_265)
    prediction = np.asarray(prediction_265)
    if (
        ground_truth.shape != prediction.shape
        or ground_truth.ndim < 2
        or ground_truth.shape[-1] != 265
    ):
        raise MetricAdapterContractError(
            "face parameters must have matching [...,T,265] shapes"
        )
    if (
        not np.issubdtype(ground_truth.dtype, np.floating)
        or not np.issubdtype(prediction.dtype, np.floating)
        or not np.isfinite(ground_truth).all()
        or not np.isfinite(prediction).all()
    ):
        raise MetricAdapterContractError(
            "face parameters must be finite floating point"
        )
    isolated_ground_truth = ground_truth.copy()
    isolated_prediction = prediction.copy()
    isolated_ground_truth[..., 3:165] = 0.0
    isolated_prediction[..., 3:165] = 0.0
    return isolated_ground_truth, isolated_prediction


class MetricBackend(Protocol):
    """Pinned metric backend interface used by the numpy adapter."""

    @property
    def asset_receipt(self) -> Mapping[str, Any]:
        ...

    @property
    def runtime_receipt(self) -> Mapping[str, Any]:
        ...

    def extract_body_features(self, parameters_265: np.ndarray) -> np.ndarray:
        ...

    def joints(
        self,
        parameters_265: np.ndarray,
        betas_300: np.ndarray,
    ) -> np.ndarray:
        ...

    def decode_audio_16k(self, snapshot: bytes) -> np.ndarray:
        ...


def _validate_backend_receipts(
    backend: MetricBackend,
    *,
    require_cuda: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = dict(backend.asset_receipt)
    runtime = dict(backend.runtime_receipt)
    expected_asset_keys = {
        "format",
        "status",
        "execution_device",
        "talkshow",
        "feature_extractor",
        "smplx",
    }
    if (
        set(assets) != expected_asset_keys
        or assets.get("format")
        != "semtalk_show_talkshow_metric_assets_v1"
        or assets.get("status") != "pass"
        or not isinstance(assets.get("talkshow"), dict)
        or assets["talkshow"].get("commit") != TALKSHOW_METRIC_COMMIT
        or not isinstance(assets.get("feature_extractor"), dict)
        or assets["feature_extractor"].get("sha256")
        != FEATURE_EXTRACTOR_SHA256
        or not isinstance(assets.get("smplx"), dict)
        or assets["smplx"].get("sha256") != SMPLX_SHA256
    ):
        raise MetricAdapterContractError(
            "metric backend does not attest pinned TalkSHOW assets"
        )
    common_runtime = {
        "python",
        "numpy",
        "torch",
        "smplx",
        "librosa",
        "soundfile",
        "soxr",
    }
    if require_cuda:
        expected_runtime_keys = common_runtime | {
            "scipy",
            "cuda",
            "cudnn",
            "device",
            "device_type",
            "device_index",
            "device_name",
        }
        if (
            set(runtime) != expected_runtime_keys
            or runtime.get("device_type") != "cuda"
            or type(runtime.get("device_index")) is not int
            or runtime["device_index"] < 0
            or runtime.get("device")
            != f"cuda:{runtime['device_index']}"
            or assets.get("execution_device") != runtime.get("device")
            or type(runtime.get("device_name")) is not str
            or not runtime["device_name"]
            or type(runtime.get("cuda")) is not str
            or not runtime["cuda"]
            or type(runtime.get("cudnn")) is not str
            or not runtime["cudnn"]
            or assets["feature_extractor"].get("runtime_dtype")
            != "float32"
            or assets["smplx"].get("runtime_dtype") != "float64"
            or any(
                type(runtime.get(key)) is not str or not runtime[key]
                for key in common_runtime | {"scipy"}
            )
        ):
            raise MetricAdapterContractError(
                "formal metric backend lacks exact CUDA runtime attestation"
            )
    else:
        expected_runtime_keys = common_runtime | {"device"}
        if (
            set(runtime) != expected_runtime_keys
            or runtime.get("device") != "cpu"
            or assets.get("execution_device") != "cpu"
            or any(
                type(runtime.get(key)) is not str or not runtime[key]
                for key in common_runtime
            )
        ):
            raise MetricAdapterContractError(
                "invalid CPU unit-test metric runtime receipt"
            )
    return assets, runtime


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.rstrip("\r\n")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MetricAdapterContractError(
            f"cannot inspect TalkSHOW metric root {root}: {exc}"
        ) from exc


def _local_module_sources(
    root: Path,
    module: str,
) -> list[tuple[str, Path]]:
    if not module:
        return []
    pieces = [piece for piece in module.split(".") if piece]
    if not pieces:
        return []
    result: list[tuple[str, Path]] = []
    for length in range(1, len(pieces)):
        package_module = ".".join(pieces[:length])
        initializer = root.joinpath(
            *pieces[:length],
            "__init__.py",
        )
        if initializer.is_file():
            result.append((package_module, initializer))
    source = root.joinpath(*pieces).with_suffix(".py")
    package = root.joinpath(*pieces, "__init__.py")
    if source.is_file():
        result.append((module, source))
    elif package.is_file():
        result.append((module, package))
    return result


def _relative_import_module(
    current_module: str,
    current_path: Path,
    *,
    level: int,
    module: str | None,
) -> str:
    package = (
        current_module
        if current_path.name == "__init__.py"
        else current_module.rpartition(".")[0]
    )
    pieces = package.split(".") if package else []
    if level > len(pieces) + 1:
        return ""
    prefix = pieces[: len(pieces) - level + 1]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


def _talkshow_fgd_import_closure(root: Path) -> tuple[str, ...]:
    root = _resolved_canonical_directory(
        root,
        "TalkSHOW metric root",
    )
    queue: list[tuple[str, Path]] = []
    queued: set[Path] = set()
    for entry in TALKSHOW_FGD_ENTRY_MODULES:
        for module, source in _local_module_sources(root, entry):
            if source not in queued:
                queued.add(source)
                queue.append((module, source))
    visited: set[Path] = set()
    cursor = 0
    while cursor < len(queue):
        current_module, current_path = queue[cursor]
        cursor += 1
        if current_path in visited:
            continue
        visited.add(current_path)
        try:
            _source_path, source_payload = _safe_file_snapshot(
                current_path,
                f"TalkSHOW metric source {current_path.relative_to(root)}",
            )
            tree = ast.parse(
                source_payload.decode("utf-8"),
                filename=str(current_path),
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetricAdapterContractError(
                f"cannot inspect TalkSHOW metric source {current_path}"
            ) from exc
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    value = _relative_import_module(
                        current_module,
                        current_path,
                        level=node.level,
                        module=node.module,
                    )
                else:
                    value = node.module or ""
                if value:
                    imported.append(value)
                if node.module is None and value:
                    imported.extend(
                        f"{value}.{alias.name}"
                        for alias in node.names
                        if alias.name != "*"
                    )
        for imported_module in imported:
            for module, source in _local_module_sources(
                root,
                imported_module,
            ):
                if source not in queued:
                    queued.add(source)
                    queue.append((module, source))
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in visited)
    )


def validate_talkshow_metric_root(root: str | Path) -> dict[str, Any]:
    resolved = _resolved_canonical_directory(
        root,
        "TalkSHOW metric root",
    )
    commit = _git_output(resolved, "rev-parse", "HEAD^{commit}")
    if commit != TALKSHOW_METRIC_COMMIT:
        raise MetricAdapterContractError(
            f"TalkSHOW metric commit {commit} != {TALKSHOW_METRIC_COMMIT}"
        )
    tree = _git_output(resolved, "rev-parse", "HEAD^{tree}")
    if tree != TALKSHOW_METRIC_TREE:
        raise MetricAdapterContractError(
            f"TalkSHOW metric tree {tree} != {TALKSHOW_METRIC_TREE}"
        )
    dirty = tuple(
        line
        for line in _git_output(
            resolved,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines()
        if line
    )
    if set(dirty) != {
        " M nets/__init__.py",
        "?? .paspa_talkshow_patch.json",
    }:
        raise MetricAdapterContractError(
            "TalkSHOW metric root has an untrusted dirty-file set"
        )
    marker_path, marker_payload = _safe_file_snapshot(
        resolved / ".paspa_talkshow_patch.json",
        "TalkSHOW metric patch marker",
    )
    try:
        marker = json.loads(marker_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetricAdapterContractError(
            "TalkSHOW patch marker is invalid JSON"
        ) from exc
    if (
        marker != TALKSHOW_PATCH_MARKER
        or marker_payload != canonical_json_bytes(TALKSHOW_PATCH_MARKER)
    ):
        raise MetricAdapterContractError(
            f"unexpected TalkSHOW patch marker: {marker}"
        )
    closure = _talkshow_fgd_import_closure(resolved)
    if closure != TALKSHOW_FGD_SOURCE_FILES:
        raise MetricAdapterContractError(
            "TalkSHOW FGD import closure differs from the pinned contract"
        )
    files: dict[str, dict[str, Any]] = {}
    for relative in closure:
        path, payload = _safe_file_snapshot(
            resolved / relative,
            f"TalkSHOW metric source {relative}",
        )
        trusted_sha, trusted_blob = TALKSHOW_TRUSTED_UPSTREAM_FILES[
            relative
        ]
        observed_blob = _git_output(
            resolved,
            "rev-parse",
            f"HEAD:{relative}",
        )
        try:
            upstream_payload = subprocess.run(
                ["git", "-C", str(resolved), "show", f"HEAD:{relative}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise MetricAdapterContractError(
                f"cannot read trusted TalkSHOW blob {relative}"
            ) from exc
        if (
            observed_blob != trusted_blob
            or sha256_bytes(upstream_payload) != trusted_sha
        ):
            raise MetricAdapterContractError(
                f"TalkSHOW trusted Git blob changed: {relative}"
            )
        expected_live_sha = (
            TALKSHOW_PATCH_MARKER["patched_sha256"]
            if relative == "nets/__init__.py"
            else trusted_sha
        )
        if sha256_bytes(payload) != expected_live_sha:
            raise MetricAdapterContractError(
                f"TalkSHOW live metric source changed: {relative}"
            )
        files[relative] = {
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "trusted_upstream_sha256": trusted_sha,
            "trusted_git_blob_sha1": trusted_blob,
        }
    return {
        "path": str(resolved),
        "commit": commit,
        "tree": tree,
        "entry_modules": list(TALKSHOW_FGD_ENTRY_MODULES),
        "import_closure_algorithm": TALKSHOW_FGD_IMPORT_ALGORITHM,
        "patch_marker": marker,
        "marker_sha256": sha256_bytes(marker_payload),
        "files": files,
    }


def _package_version(module: Any, distribution: str) -> str:
    from importlib import metadata

    value = getattr(module, "__version__", None)
    if type(value) is not str or not value.strip():
        value = metadata.version(distribution)
    if type(value) is not str or not value.strip():
        raise MetricAdapterContractError(
            f"cannot attest runtime version for {distribution}"
        )
    return value


def _import_pinned_body_feature_extractor(
    talkshow: Mapping[str, Any],
) -> Any:
    """Import TalkSHOW's body extractor without trusting ``sys.modules``.

    ``nets`` is an intentionally generic top-level package name.  A prior
    import from another checkout must therefore never be allowed to satisfy
    this evaluator import.  Temporarily remove that module subtree, import
    from the attested root, verify every local module against the attested
    closure, and then restore the caller's module table exactly.
    """

    root = _resolved_canonical_directory(
        str(talkshow["path"]),
        "attested TalkSHOW metric root",
    )
    files = talkshow.get("files")
    if type(files) is not dict:
        raise MetricAdapterContractError(
            "TalkSHOW source receipt lacks its pinned file closure"
        )
    prefixes = {
        relative.split("/", 1)[0]
        for relative in files
        if "/" in relative
    }
    if not {"nets", "data_utils"}.issubset(prefixes):
        raise MetricAdapterContractError(
            "TalkSHOW source receipt lacks required local package roots"
        )
    source_map: dict[str, tuple[Path, bytes, bool]] = {}
    namespace_packages: dict[str, Path] = {}
    for relative, expected in sorted(files.items()):
        if type(relative) is not str or type(expected) is not dict:
            raise MetricAdapterContractError(
                "TalkSHOW source receipt has an invalid source entry"
            )
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix != ".py"
        ):
            raise MetricAdapterContractError(
                f"TalkSHOW source receipt has unsafe path {relative!r}"
            )
        source_path, payload = _safe_file_snapshot(
            root / relative_path,
            f"attested TalkSHOW source {relative}",
        )
        if (
            expected.get("sha256") != sha256_bytes(payload)
            or expected.get("bytes") != len(payload)
        ):
            raise MetricAdapterContractError(
                f"TalkSHOW source receipt changed before import: {relative}"
            )
        if relative_path.name == "__init__.py":
            module_parts = relative_path.parts[:-1]
            is_package = True
        else:
            module_parts = (*relative_path.parts[:-1], relative_path.stem)
            is_package = False
        module_name = ".".join(module_parts)
        if not module_name or module_name in source_map:
            raise MetricAdapterContractError(
                f"TalkSHOW source receipt has duplicate module {module_name!r}"
            )
        source_map[module_name] = (source_path, payload, is_package)
        for length in range(1, len(module_parts)):
            parent_name = ".".join(module_parts[:length])
            parent_path = root.joinpath(*module_parts[:length])
            namespace_packages.setdefault(parent_name, parent_path)
    for module_name in source_map:
        namespace_packages.pop(module_name, None)
    for module_name, directory in tuple(namespace_packages.items()):
        namespace_packages[module_name] = _resolved_canonical_directory(
            directory,
            f"TalkSHOW namespace package {module_name}",
        )

    class _PinnedSnapshotLoader(importlib.abc.InspectLoader):
        def __init__(
            self,
            fullname: str,
            source_path: Path,
            payload: bytes,
            package: bool,
        ) -> None:
            self.fullname = fullname
            self.source_path = source_path
            self.payload = payload
            self.package = package

        def get_filename(self, fullname: str) -> str:
            if fullname != self.fullname:
                raise ImportError(fullname)
            return str(self.source_path)

        def get_source(self, fullname: str) -> str:
            if fullname != self.fullname:
                raise ImportError(fullname)
            return self.payload.decode("utf-8")

        def is_package(self, fullname: str) -> bool:
            if fullname != self.fullname:
                raise ImportError(fullname)
            return self.package

        def get_code(self, fullname: str) -> Any:
            if fullname != self.fullname:
                raise ImportError(fullname)
            return compile(
                self.payload,
                str(self.source_path),
                "exec",
                dont_inherit=True,
            )

    namespace_token = object()

    class _PinnedSnapshotFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Any = None,
            target: Any = None,
        ) -> Any:
            entry = source_map.get(fullname)
            if entry is not None:
                source_path, payload, package = entry
                loader = _PinnedSnapshotLoader(
                    fullname,
                    source_path,
                    payload,
                    package,
                )
                specification = importlib.util.spec_from_loader(
                    fullname,
                    loader,
                    origin=str(source_path),
                    is_package=package,
                )
                if specification is None:  # pragma: no cover - fixed loader
                    raise ImportError(fullname)
                # An explicit origin does not imply ``has_location`` on every
                # supported CPython version.  Set it so the module executes
                # with the exact attested source path in ``__file__``.
                specification.has_location = True
                return specification
            namespace_path = namespace_packages.get(fullname)
            if namespace_path is not None:
                specification = importlib.machinery.ModuleSpec(
                    fullname,
                    loader=None,
                    is_package=True,
                )
                specification.submodule_search_locations = [
                    str(namespace_path)
                ]
                specification.loader_state = namespace_token
                return specification
            if any(
                fullname == prefix or fullname.startswith(f"{prefix}.")
                for prefix in prefixes
            ):
                raise ModuleNotFoundError(
                    f"TalkSHOW local import escaped pinned closure: {fullname}"
                )
            return None

    finder = _PinnedSnapshotFinder()

    def reject_bytecode_cache() -> None:
        for prefix in sorted(prefixes):
            package_root = _resolved_canonical_directory(
                root / prefix,
                f"TalkSHOW package root {prefix}",
            )
            for candidate in package_root.rglob("*"):
                if (
                    candidate.name == "__pycache__"
                    or candidate.suffix in {".pyc", ".pyo"}
                ):
                    raise MetricAdapterContractError(
                        "TalkSHOW pinned import forbids cached bytecode: "
                        f"{candidate}"
                    )

    # Timestamp-valid bytecode can execute code that is not represented by
    # the attested source SHA.  A formal import is therefore source-only:
    # reject any pre-existing cache and suppress cache creation throughout
    # the isolated import.
    reject_bytecode_cache()

    def is_local_module(name: str) -> bool:
        return any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in prefixes
        )

    saved_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if is_local_module(name)
    }
    for name in saved_modules:
        del sys.modules[name]
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    original_meta_path = list(sys.meta_path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    imported_modules: dict[str, Any] = {}
    try:
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        os.chdir(root)
        sys.meta_path.insert(0, finder)
        module = importlib.import_module("nets.body_ae")
        imported_modules = {
            name: value
            for name, value in tuple(sys.modules.items())
            if is_local_module(name)
        }
        for name, value in imported_modules.items():
            source = getattr(value, "__file__", None)
            specification = getattr(value, "__spec__", None)
            if source is None and name in namespace_packages:
                locations = getattr(
                    specification,
                    "submodule_search_locations",
                    None,
                )
                if (
                    specification is None
                    or getattr(specification, "origin", None) is not None
                    or getattr(specification, "loader_state", None)
                    is not namespace_token
                    or locations is None
                    or list(locations)
                    != [str(namespace_packages[name])]
                ):
                    raise MetricAdapterContractError(
                        f"TalkSHOW namespace package {name} escaped the "
                        "pinned closure"
                    )
                continue
            if type(source) is not str:
                raise MetricAdapterContractError(
                    f"TalkSHOW imported module {name} has no source file"
                )
            path = Path(source)
            origin = getattr(specification, "origin", None)
            loader = getattr(specification, "loader", None)
            if (
                path.suffix != ".py"
                or type(origin) is not str
                or Path(origin) != path
                or not isinstance(loader, _PinnedSnapshotLoader)
            ):
                raise MetricAdapterContractError(
                    f"TalkSHOW imported module {name} was not loaded "
                    "directly from pinned Python source"
                )
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise MetricAdapterContractError(
                    f"TalkSHOW imported module {name} escaped the pinned root"
                ) from exc
            expected = files.get(relative)
            _source_path, payload = _safe_file_snapshot(
                path,
                f"imported TalkSHOW module {name}",
            )
            if (
                type(expected) is not dict
                or expected.get("sha256") != sha256_bytes(payload)
                or expected.get("bytes") != len(payload)
            ):
                raise MetricAdapterContractError(
                    f"TalkSHOW imported module {name} is outside the pinned "
                    "source closure"
                )
        reject_bytecode_cache()
        extractor = getattr(module, "TrainWrapper", None)
        if not callable(extractor):
            raise MetricAdapterContractError(
                "pinned TalkSHOW body_ae lacks TrainWrapper"
            )
        return extractor
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
        sys.meta_path[:] = original_meta_path
        for name in tuple(sys.modules):
            if is_local_module(name):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


_FORMAL_BACKEND_CONSTRUCTION_TOKEN = object()


class TalkShowCudaMetricBackend:
    """Pinned TalkSHOW feature/SMPL-X backend on one explicit CUDA device."""

    def __init__(
        self,
        *,
        talkshow_root: str | Path,
        feature_extractor: str | Path,
        smplx_asset: str | Path,
        device: str,
        torch_threads: int = 1,
    ) -> None:
        torch_threads = _require_exact_int(
            torch_threads,
            "torch_threads",
            minimum=1,
        )
        talkshow = validate_talkshow_metric_root(talkshow_root)
        feature_path, feature_payload = _safe_file_snapshot(
            feature_extractor,
            "TalkSHOW released feature extractor",
        )
        feature_sha = sha256_bytes(feature_payload)
        if feature_sha != FEATURE_EXTRACTOR_SHA256:
            raise MetricAdapterContractError(
                "TalkSHOW feature extractor SHA differs from the pinned release"
            )
        smplx_path, smplx_payload = _safe_file_snapshot(
            smplx_asset,
            "SMPL-X neutral asset",
        )
        smplx_sha = sha256_bytes(smplx_payload)
        if smplx_sha != SMPLX_SHA256:
            raise MetricAdapterContractError(
                "SMPL-X asset SHA differs from the pinned release"
            )
        try:
            import torch
            import smplx
            import librosa
            import scipy
            import soundfile
            import soxr
        except ImportError as exc:
            raise MetricAdapterContractError(
                "torch, smplx, librosa, scipy, soundfile, and soxr are "
                "required by the formal CUDA metric backend"
            ) from exc
        if (
            type(device) is not str
            or not device.startswith("cuda:")
            or not device[5:].isdigit()
        ):
            raise MetricAdapterContractError(
                "formal metric device must be one explicit cuda:N ordinal"
            )
        torch_device = torch.device(device)
        device_index = int(device[5:])
        if (
            torch_device.type != "cuda"
            or torch_device.index != device_index
            or not torch.cuda.is_available()
            or device_index >= torch.cuda.device_count()
        ):
            raise MetricAdapterContractError(
                f"formal CUDA metric device is unavailable: {device}"
            )
        torch.set_num_threads(torch_threads)
        BodyFeatureExtractor = _import_pinned_body_feature_extractor(
            talkshow
        )
        config = SimpleNamespace(
            Data=SimpleNamespace(
                pose=SimpleNamespace(
                    convert_to_6d=False,
                    pre_pose_length=0,
                    expression=False,
                )
            ),
            Train=SimpleNamespace(
                learning_rate=SimpleNamespace(
                    generator_learning_rate=1e-4,
                    discriminator_learning_rate=1e-4,
                )
            ),
        )
        extractor = BodyFeatureExtractor(
            SimpleNamespace(gpu=str(torch_device)),
            config,
        )
        try:
            checkpoint = torch.load(
                io.BytesIO(feature_payload),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise MetricAdapterContractError(
                "cannot deserialize pinned TalkSHOW feature extractor"
            ) from exc
        if isinstance(checkpoint, Mapping):
            checkpoint = checkpoint.get("generator", checkpoint)
        if not isinstance(checkpoint, Mapping):
            raise MetricAdapterContractError(
                "TalkSHOW feature-extractor checkpoint is not a mapping"
            )
        state = checkpoint.get("g", checkpoint)
        if not isinstance(state, Mapping):
            raise MetricAdapterContractError(
                "TalkSHOW feature-extractor checkpoint lacks g state"
            )
        normalized = {
            (name[7:] if name.startswith("module.") else name): value
            for name, value in state.items()
        }
        extractor.g.load_state_dict(normalized, strict=True)
        extractor.g.to(torch_device).eval()
        extractor.g.requires_grad_(False)
        for name, tensor in extractor.g.state_dict().items():
            if not bool(torch.isfinite(tensor).all().item()):
                raise MetricAdapterContractError(
                    f"non-finite TalkSHOW feature tensor: {name}"
                )
        smplx_model = smplx.create(
            str(smplx_path),
            model_type="smplx",
            gender="neutral",
            ext="npz",
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            num_betas=300,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            use_pca=False,
            flat_hand_mean=False,
            create_expression=True,
            num_expression_coeffs=100,
            num_pca_comps=12,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=False,
            dtype=torch.float64,
        ).to(torch_device).eval()
        smplx_model.requires_grad_(False)
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        if (
            type(cuda_version) is not str
            or not cuda_version.strip()
            or type(cudnn_version) is not int
            or cudnn_version < 1
        ):
            raise MetricAdapterContractError(
                "formal CUDA metric backend cannot attest CUDA/cuDNN"
            )
        self._torch = torch
        self._device = torch_device
        self._extractor = extractor
        self._smplx = smplx_model
        self._librosa = librosa
        self._asset_receipt = {
            "format": "semtalk_show_talkshow_metric_assets_v1",
            "status": "pass",
            "execution_device": str(torch_device),
            "talkshow": talkshow,
            "feature_extractor": {
                "path": str(feature_path),
                "sha256": feature_sha,
                "bytes": len(feature_payload),
                "runtime_dtype": "float32",
            },
            "smplx": {
                "path": str(smplx_path),
                "sha256": smplx_sha,
                "bytes": len(smplx_payload),
                "runtime_dtype": "float64",
            },
        }
        self._runtime_receipt = {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "smplx": _package_version(smplx, "smplx"),
            "librosa": _package_version(librosa, "librosa"),
            "soundfile": _package_version(soundfile, "soundfile"),
            "soxr": _package_version(soxr, "soxr"),
            "scipy": _package_version(scipy, "scipy"),
            "cuda": cuda_version,
            "cudnn": str(cudnn_version),
            "device": str(torch_device),
            "device_type": "cuda",
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(torch_device),
        }
        self._formal_construction_token = (
            _FORMAL_BACKEND_CONSTRUCTION_TOKEN
        )

    @property
    def asset_receipt(self) -> Mapping[str, Any]:
        return self._asset_receipt

    @property
    def runtime_receipt(self) -> Mapping[str, Any]:
        return self._runtime_receipt

    def extract_body_features(self, parameters_265: np.ndarray) -> np.ndarray:
        parameters = np.asarray(parameters_265, dtype=np.float32)
        if parameters.ndim != 3 or parameters.shape[-1] != 265:
            raise MetricAdapterContractError(
                "feature parameters must be [B,T,265]"
            )
        with self._torch.no_grad():
            features, _ = self._extractor.extract(
                self._torch.from_numpy(parameters).to(self._device)
            )
        array = features.detach().cpu().numpy()
        array = array.reshape(-1, array.shape[-1])
        if array.size == 0 or not np.isfinite(array).all():
            raise MetricAdapterContractError(
                "TalkSHOW feature extractor returned invalid features"
            )
        return array

    def joints(
        self,
        parameters_265: np.ndarray,
        betas_300: np.ndarray,
    ) -> np.ndarray:
        parameters = np.asarray(parameters_265)
        betas = np.asarray(betas_300)
        if parameters.ndim != 3 or parameters.shape[-1] != 265:
            raise MetricAdapterContractError(
                "SMPL-X parameters must be [B,T,265]"
            )
        if betas.shape != (BETA_DIM,):
            raise MetricAdapterContractError(
                "SMPL-X betas must have 300 coefficients"
            )
        flat = self._torch.from_numpy(parameters.reshape(-1, 265)).to(
            device=self._device,
            dtype=self._torch.float64,
        )
        beta = self._torch.from_numpy(betas.reshape(1, -1)).to(
            device=self._device,
            dtype=self._torch.float64,
        )
        chunks = []
        with self._torch.no_grad():
            for start in range(0, flat.shape[0], 2048):
                current = flat[start : start + 2048]
                output = self._smplx(
                    betas=beta.expand(current.shape[0], -1),
                    expression=current[:, 165:265],
                    jaw_pose=current[:, 0:3],
                    leye_pose=current[:, 3:6],
                    reye_pose=current[:, 6:9],
                    global_orient=current[:, 9:12],
                    body_pose=current[:, 12:75],
                    left_hand_pose=current[:, 75:120],
                    right_hand_pose=current[:, 120:165],
                    return_verts=False,
                )
                chunks.append(output.joints.detach().cpu())
        array = self._torch.cat(chunks).reshape(
            parameters.shape[0],
            parameters.shape[1],
            -1,
            3,
        ).numpy()
        if not np.isfinite(array).all():
            raise MetricAdapterContractError(
                "SMPL-X backend returned non-finite joints"
            )
        return array

    def audio_beats(self, waveform_16k: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        if not np.isfinite(waveform).all() or waveform.size == 0:
            raise MetricAdapterContractError(
                "audio waveform is empty or non-finite"
            )
        return self._librosa.onset.onset_detect(
            y=waveform,
            sr=16000,
            units="time",
        ).reshape(-1)

    def decode_audio_16k(self, snapshot: bytes) -> np.ndarray:
        if type(snapshot) is not bytes or not snapshot:
            raise MetricAdapterContractError(
                "audio snapshot must be non-empty immutable bytes"
            )
        try:
            waveform, sample_rate = self._librosa.load(
                io.BytesIO(snapshot),
                sr=16_000,
                mono=True,
            )
        except Exception as exc:
            raise MetricAdapterContractError(
                "librosa cannot decode the verified WAV snapshot"
            ) from exc
        array = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000 or array.size < 1 or not np.isfinite(array).all():
            raise MetricAdapterContractError(
                "librosa returned an invalid 16 kHz mono waveform"
            )
        return array


def _strict_npz_members(
    payload: bytes,
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            members = tuple(info.filename for info in archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise MetricAdapterContractError(
            f"cannot inspect verified NPZ snapshot {label}"
        ) from exc
    expected = tuple(f"{field}.npy" for field in fields)
    if members != expected or len(members) != len(set(members)):
        raise MetricAdapterContractError(
            f"{label}: NPZ members {members} != {expected}"
        )


def _load_canonical_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    frames: int,
    speaker_id: int,
) -> dict[str, np.ndarray]:
    resolved, payload = _verified_file_snapshot(
        path,
        expected_sha256,
        "canonical SHOW NPZ",
    )
    _strict_npz_members(
        payload,
        CANONICAL_FIELDS,
        label=str(resolved),
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != CANONICAL_FIELDS:
                raise MetricAdapterContractError(
                    f"{resolved}: canonical NPZ field order mismatch"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in CANONICAL_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise MetricAdapterContractError(
            f"cannot load canonical SHOW NPZ {resolved}"
        ) from exc
    expected = {
        "pose": ((frames, POSE_DIM), np.dtype(np.float32)),
        "contact": ((frames, 4), np.dtype(np.float32)),
        "facial": ((frames, EXPRESSION_DIM), np.dtype(np.float32)),
        "beta": ((frames, BETA_DIM), np.dtype(np.float32)),
        "trans": ((frames, 3), np.dtype(np.float32)),
        "speaker_id": ((frames, 1), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise MetricAdapterContractError(
                f"{resolved}: {name} {array.shape}/{array.dtype} "
                f"!= {shape}/{dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise MetricAdapterContractError(
                f"{resolved}: non-finite canonical {name}"
            )
    if not np.array_equal(
        arrays["speaker_id"],
        np.full((frames, 1), speaker_id, dtype=np.int64),
    ):
        raise MetricAdapterContractError(
            f"{resolved}: canonical speaker_id mismatch"
        )
    if not np.array_equal(
        arrays["beta"],
        np.broadcast_to(arrays["beta"][0], arrays["beta"].shape),
    ):
        raise MetricAdapterContractError(
            f"{resolved}: canonical betas change across frames"
        )
    return arrays


def _load_output_npz(
    artifact: Mapping[str, Any],
    *,
    frames: int,
    prediction: bool,
    expected_name: str,
) -> dict[str, np.ndarray]:
    if type(artifact) is not dict or set(artifact) != ARTIFACT_KEYS:
        raise MetricAdapterContractError(
            f"invalid {'prediction' if prediction else 'ground-truth'} artifact"
        )
    path, payload = _verified_file_snapshot(
        artifact["path"],
        _require_sha256(
            artifact["sha256"],
            "output NPZ SHA",
        ),
        "prediction NPZ" if prediction else "ground-truth NPZ",
    )
    if path.name != expected_name:
        raise MetricAdapterContractError(
            f"output NPZ basename {path.name!r} != {expected_name!r}"
        )
    if _require_exact_int(
        artifact["bytes"],
        "output NPZ bytes",
        minimum=1,
    ) != len(payload):
        raise MetricAdapterContractError(
            f"{path}: output NPZ byte count mismatch"
        )
    _strict_npz_members(payload, OUTPUT_FIELDS, label=str(path))
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != OUTPUT_FIELDS:
                raise MetricAdapterContractError(
                    f"{path}: output NPZ field order mismatch"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in OUTPUT_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise MetricAdapterContractError(
            f"cannot load output NPZ {path}"
        ) from exc
    expected = {
        "betas": ((BETA_DIM,), np.dtype(np.float32)),
        "poses": ((frames, POSE_DIM), np.dtype(np.float32)),
        "expressions": (
            (frames, EXPRESSION_DIM),
            np.dtype(np.float32),
        ),
        "trans": ((frames, 3), np.dtype(np.float32)),
        "model": ((), np.asarray("smplx2020").dtype),
        "gender": ((), np.asarray("neutral").dtype),
        "mocap_frame_rate": ((), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise MetricAdapterContractError(
                f"{path}: {name} {array.shape}/{array.dtype} "
                f"!= {shape}/{dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise MetricAdapterContractError(
                f"{path}: non-finite {name}"
            )
    if (
        arrays["model"].item() != "smplx2020"
        or arrays["gender"].item() != "neutral"
        or int(arrays["mocap_frame_rate"].item()) != POSE_FPS
    ):
        raise MetricAdapterContractError(
            f"{path}: invalid model/gender/frame-rate metadata"
        )
    if prediction and not np.array_equal(
        arrays["poses"][:, 69:75],
        np.zeros((frames, 6), dtype=np.float32),
    ):
        raise MetricAdapterContractError(
            f"{path}: prediction eye pose is not zero"
        )
    return arrays


def _validate_wav_row_and_decode(
    row: Mapping[str, Any],
    *,
    motion_frames: int,
    decode_audio_16k: Callable[[bytes], np.ndarray],
) -> np.ndarray:
    path, payload = _verified_file_snapshot(
        row["source_wav"],
        _require_sha256(
            row["source_wav_sha256"],
            "canonical source WAV SHA",
        ),
        "canonical source WAV",
    )
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            rate = int(handle.getframerate())
            wav_frames = int(handle.getnframes())
            compression = handle.getcomptype()
            handle.readframes(wav_frames)
    except (EOFError, wave.Error) as exc:
        raise MetricAdapterContractError(
            f"cannot decode canonical WAV {path}"
        ) from exc
    if (
        channels not in (1, 2)
        or width != 2
        or rate != CANONICAL_SOURCE_AUDIO_RATE
        or wav_frames < 1
        or compression != "NONE"
        or row.get("wav_channels") != channels
        or row.get("wav_sample_width") != width
        or row.get("wav_sample_rate") != rate
        or row.get("wav_frames") != wav_frames
        or row.get("wav_mono_policy") != CANONICAL_WAV_MONO_POLICY
    ):
        raise MetricAdapterContractError(
            f"{path}: canonical WAV metadata mismatch"
        )
    if not callable(decode_audio_16k):
        raise MetricAdapterContractError(
            "verified WAV decoder is not callable"
        )
    try:
        waveform = np.asarray(
            decode_audio_16k(payload),
            dtype=np.float32,
        ).reshape(-1)
    except MetricAdapterContractError:
        raise
    except Exception as exc:
        raise MetricAdapterContractError(
            f"{path}: verified snapshot audio decode failed"
        ) from exc
    if waveform.size < 1 or not np.isfinite(waveform).all():
        raise MetricAdapterContractError(
            f"{path}: verified snapshot decoder returned invalid audio"
        )
    expected_audio_frames = 16000 * motion_frames // POSE_FPS
    if waveform.shape[0] < expected_audio_frames:
        waveform = np.pad(
            waveform,
            (0, expected_audio_frames - waveform.shape[0]),
            mode="constant",
        )
    else:
        waveform = waveform[:expected_audio_frames]
    if not np.isfinite(waveform).all():
        raise MetricAdapterContractError(
            f"{path}: decoded WAV is non-finite"
        )
    return waveform


def _validate_canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_clip_count: int,
    formal_mode: bool,
    test_only_allow_four_clip_subset: bool,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    global_indices: set[int] = set()
    for raw in rows:
        if type(raw) is not dict or set(raw) != CANONICAL_ROW_KEYS:
            raise MetricAdapterContractError(
                "canonical manifest row schema mismatch"
            )
        row = dict(raw)
        if row.get("split") != split:
            continue
        clip_id = row.get("clip_id")
        if type(clip_id) is not str:
            raise MetricAdapterContractError(
                "canonical clip_id must be a string"
            )
        output_id = canonical_clip_id(clip_id)
        if output_id in selected:
            raise MetricAdapterContractError(
                f"duplicate canonical clip {output_id}"
            )
        index = _require_exact_int(
            row.get("global_index"),
            f"{clip_id} global_index",
            minimum=0,
        )
        if index in global_indices:
            raise MetricAdapterContractError(
                f"duplicate canonical global_index {index}"
            )
        speaker = row.get("speaker")
        if (
            speaker not in SHOW_SPEAKER_IDS
            or row.get("speaker_id") != SHOW_SPEAKER_IDS[speaker]
            or clip_id.split("/", 1)[0] != speaker
            or row.get("pose_fps") != POSE_FPS
        ):
            raise MetricAdapterContractError(
                f"{clip_id}: canonical speaker/frame-rate mismatch"
            )
        frames = _require_exact_int(
            row.get("frames"),
            f"{clip_id} frames",
            minimum=2,
        )
        if frames <= FORMAL_MIN_FRAMES_EXCLUSIVE:
            raise MetricAdapterContractError(
                f"{clip_id}: SHOW metric clip must contain more than "
                f"{FORMAL_MIN_FRAMES_EXCLUSIVE} frames"
            )
        _require_sha256(
            row.get("canonical_npz_sha256"),
            f"{clip_id} canonical NPZ SHA",
        )
        _require_sha256(
            row.get("source_wav_sha256"),
            f"{clip_id} source WAV SHA",
        )
        selected[output_id] = row
        global_indices.add(index)
    if len(selected) != expected_clip_count:
        raise MetricAdapterContractError(
            f"canonical {split} clips {len(selected)} != {expected_clip_count}"
        )
    observed_order = [
        row["global_index"] for row in selected.values()
    ]
    ordered = sorted(
        selected.items(),
        key=lambda item: item[1]["global_index"],
    )
    ordered_indices = [row["global_index"] for _key, row in ordered]
    if formal_mode:
        domain = FORMAL_SPLITS[split]
        if (
            test_only_allow_four_clip_subset
            or expected_clip_count != domain["count"]
            or ordered_indices
            != list(range(domain["global_start"], domain["global_stop"]))
        ):
            raise MetricAdapterContractError(
                f"formal {split} split is not the exact frozen global-index "
                "domain"
            )
    elif test_only_allow_four_clip_subset:
        if expected_clip_count != 4 or len(ordered_indices) != 4:
            raise MetricAdapterContractError(
                "CPU fixture mode requires exactly four clips"
            )
    else:
        raise MetricAdapterContractError(
            "non-formal evaluation is permitted only for the explicit "
            "four-clip CPU fixture"
        )
    if observed_order != ordered_indices:
        raise MetricAdapterContractError(
            f"canonical {split} rows are not deterministically ordered"
        )
    if set(row["speaker"] for row in selected.values()) != set(SPEAKER_NAMES):
        raise MetricAdapterContractError(
            "canonical metric split must contain all four SHOW speakers"
        )
    return dict(ordered)


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    canonical: Mapping[str, Mapping[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    expected_keys = (
        VAL_PREDICTION_ROW_KEYS
        if split == "val"
        else TEST_PREDICTION_ROW_KEYS
    )
    by_id: dict[str, dict[str, Any]] = {}
    observed_order: list[dict[str, Any]] = []
    for raw in rows:
        if type(raw) is not dict or set(raw) != expected_keys:
            raise MetricAdapterContractError(
                f"{split} prediction manifest row schema mismatch"
            )
        row = dict(raw)
        output_id = row.get("canonical_clip_id")
        if type(output_id) is not str or output_id in by_id:
            raise MetricAdapterContractError(
                f"invalid/duplicate prediction clip {output_id!r}"
            )
        canonical_row = canonical.get(output_id)
        if canonical_row is None:
            raise MetricAdapterContractError(
                f"prediction clip {output_id} is not canonical {split}"
            )
        clip_id = canonical_row["clip_id"]
        if (
            row.get("source_clip_id") != clip_id
            or row.get("global_index") != canonical_row["global_index"]
            or row.get("frames") != canonical_row["frames"]
            or output_id != canonical_clip_id(clip_id)
        ):
            raise MetricAdapterContractError(
                f"{output_id}: prediction/canonical identity mismatch"
            )
        if split == "val":
            if (
                row.get("split") != "val"
                or type(row.get("epoch")) is not int
            ):
                raise MetricAdapterContractError(
                    f"{output_id}: invalid validation candidate binding"
                )
            _require_sha256(
                row.get("candidate_checkpoint_sha256"),
                f"{output_id} candidate checkpoint SHA",
            )
        else:
            if (
                row.get("speaker") != canonical_row["speaker"]
                or row.get("speaker_id") != canonical_row["speaker_id"]
                or Path(str(row.get("canonical_npz", ""))).resolve()
                != Path(canonical_row["canonical_npz"]).resolve()
                or row.get("canonical_npz_sha256")
                != canonical_row["canonical_npz_sha256"]
            ):
                raise MetricAdapterContractError(
                    f"{output_id}: test row canonical binding mismatch"
                )
            _require_exact_int(
                row.get("evaluation_index"),
                f"{output_id} evaluation_index",
                minimum=0,
            )
        for role in ("prediction", "ground_truth"):
            artifact = row.get(role)
            if type(artifact) is not dict or set(artifact) != ARTIFACT_KEYS:
                raise MetricAdapterContractError(
                    f"{output_id}: invalid {role} artifact receipt"
                )
            _require_sha256(
                artifact.get("sha256"),
                f"{output_id} {role} SHA",
            )
            _require_exact_int(
                artifact.get("bytes"),
                f"{output_id} {role} bytes",
                minimum=1,
            )
        by_id[output_id] = row
        observed_order.append(row)
    if set(by_id) != set(canonical):
        missing = sorted(set(canonical) - set(by_id))
        extra = sorted(set(by_id) - set(canonical))
        raise MetricAdapterContractError(
            "prediction manifest does not cover canonical split exactly once: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    if split == "test":
        ordered = observed_order
        indices = [row["evaluation_index"] for row in ordered]
        output_ids = [row["canonical_clip_id"] for row in ordered]
        global_indices = sorted(row["global_index"] for row in ordered)
        if (
            indices != list(range(len(ordered)))
            or output_ids != sorted(canonical)
            or global_indices
            != sorted(row["global_index"] for row in canonical.values())
        ):
            raise MetricAdapterContractError(
                "test prediction rows are not in the exact finalized "
                "canonical-clip order"
            )
    else:
        ordered = observed_order
        if [row["global_index"] for row in ordered] != [
            row["global_index"] for row in canonical.values()
        ]:
            raise MetricAdapterContractError(
                "validation prediction rows are not in exact canonical order"
            )
        epochs = {row["epoch"] for row in ordered}
        checkpoints = {
            row["candidate_checkpoint_sha256"] for row in ordered
        }
        if len(epochs) != 1 or len(checkpoints) != 1:
            raise MetricAdapterContractError(
                "validation predictions mix candidates"
            )
    return ordered


def _validate_prediction_lineage(
    lineage: Mapping[str, Any],
    *,
    split: str,
    prediction_manifest_path: Path,
    prediction_manifest_sha256: str,
    expected_clip_count: int,
) -> str | None:
    def reject_randomness_conflicts(value: Any, context: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                current = f"{context}.{key}"
                if (
                    key in {"independent_samples", "stochastic"}
                    and nested is not False
                ):
                    raise MetricAdapterContractError(
                        f"{current} conflicts with deterministic delta metrics"
                    )
                if key == "deterministic" and nested is not True:
                    raise MetricAdapterContractError(
                        f"{current} conflicts with deterministic delta metrics"
                    )
                if (
                    key in {
                        "physical_samples_per_clip",
                        "prediction_samples_per_clip",
                    }
                    and nested != 1
                ):
                    raise MetricAdapterContractError(
                        f"{current} conflicts with one physical prediction"
                    )
                reject_randomness_conflicts(nested, current)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                reject_randomness_conflicts(
                    nested,
                    f"{context}[{index}]",
                )

    reject_randomness_conflicts(lineage, "prediction_lineage")
    if lineage.get("status") != "complete":
        raise MetricAdapterContractError(
            "prediction lineage is not complete"
        )
    payload_sha: str | None = None
    if split == "val":
        if (
            lineage.get("format")
            != "semtalk_show_base_official_adapt_val_inference_lineage_v1"
            or lineage.get("split") != "val"
            or lineage.get("test_visible") is not False
            or lineage.get("exact_once") is not True
            or lineage.get("finite") is not True
            or lineage.get("clip_count") != expected_clip_count
            or not isinstance(lineage.get("final_manifest"), dict)
            or Path(lineage["final_manifest"].get("path", "")).resolve()
            != prediction_manifest_path
            or lineage["final_manifest"].get("sha256")
            != prediction_manifest_sha256
        ):
            raise MetricAdapterContractError(
                "validation prediction lineage/manifest mismatch"
            )
        payload_sha = _require_sha256(
            lineage.get("receipt_payload_sha256"),
            "validation lineage payload SHA",
        )
        unsigned = dict(lineage)
        del unsigned["receipt_payload_sha256"]
        if canonical_json_sha256(unsigned) != payload_sha:
            raise MetricAdapterContractError(
                "validation lineage payload SHA mismatch"
            )
    else:
        if (
            lineage.get("format")
            != "semtalk_show_base_inference_final_lineage_v1"
            or lineage.get("final_manifest_sha256")
            != prediction_manifest_sha256
            or not isinstance(lineage.get("contract"), dict)
            or lineage["contract"].get("test_clips")
            != expected_clip_count
        ):
            raise MetricAdapterContractError(
                "test prediction lineage/manifest mismatch"
            )
    return payload_sha


def _fresh_validate_formal_val_lineage(
    *,
    lineage_path: Path,
    lineage_sha256: str,
    lineage: Mapping[str, Any],
    prediction_path: Path,
    prediction_sha256: str,
) -> None:
    selector = _fresh_base_selector_module()
    try:
        epoch = _require_exact_int(
            lineage.get("epoch"),
            "formal val lineage epoch",
            minimum=1,
        )
        val_inputs_receipt = lineage.get("val_inputs_receipt")
        pipeline_receipt = lineage.get("pipeline_receipt")
        candidate = lineage.get("candidate_checkpoint")
        if (
            type(val_inputs_receipt) is not dict
            or type(pipeline_receipt) is not dict
            or type(candidate) is not dict
        ):
            raise MetricAdapterContractError(
                "formal val lineage omits authority artifacts"
            )
        val_inputs_artifact, coverage = selector.validate_val_inputs(
            Path(val_inputs_receipt["path"]),
            val_inputs_receipt["sha256"],
        )
        pipeline_artifact, _pipeline = selector.validate_pipeline(
            Path(pipeline_receipt["path"]),
            pipeline_receipt["sha256"],
        )
        _lineage_artifact, validated = (
            selector.validate_val_inference_lineage(
                lineage_path,
                lineage_sha256,
                epoch=epoch,
                expected_candidate=candidate,
                val_inputs_artifact=val_inputs_artifact,
                pipeline_artifact=pipeline_artifact,
                expected_coverage=coverage,
            )
        )
    except Exception as exc:
        if isinstance(exc, MetricAdapterContractError):
            raise
        raise MetricAdapterContractError(
            f"formal val lineage fresh selector replay failed: {exc}"
        ) from exc
    final_manifest = validated.get("final_manifest")
    if (
        type(final_manifest) is not dict
        or Path(final_manifest.get("path", "")).resolve()
        != prediction_path
        or final_manifest.get("sha256") != prediction_sha256
    ):
        raise MetricAdapterContractError(
            "formal val selector replay produced another final manifest"
        )


def _validate_external_validation_gate(
    receipt: Mapping[str, Any],
    *,
    expected_scope: str,
    test_only_allow_four_clip_subset: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(receipt) is not dict or set(receipt) != {
        "path",
        "sha256",
        "bytes",
        "receipt_payload_sha256",
    }:
        raise MetricAdapterContractError(
            "validation_gate artifact schema mismatch"
        )
    module = _replication_gate_module()
    try:
        artifact, gate = module.load_gate(
            Path(str(receipt["path"])),
            _require_sha256(
                receipt["sha256"],
                "validation gate file SHA",
            ),
            expected_scope=expected_scope,
            test_only_allow_four_clip_subset=(
                test_only_allow_four_clip_subset
            ),
        )
    except Exception as exc:
        if isinstance(exc, MetricAdapterContractError):
            raise
        raise MetricAdapterContractError(
            f"validation gate fresh replay failed: {exc}"
        ) from exc
    expected_bytes = _require_exact_int(
        receipt["bytes"],
        "validation gate bytes",
        minimum=1,
    )
    expected_payload_sha = _require_sha256(
        receipt["receipt_payload_sha256"],
        "validation gate receipt payload SHA",
    )
    if (
        artifact != dict(receipt)
        or artifact.get("bytes") != expected_bytes
        or artifact.get("receipt_payload_sha256")
        != expected_payload_sha
        or gate.get("payload_hash_algorithm")
        != PAYLOAD_HASH_ALGORITHM
    ):
        raise MetricAdapterContractError(
            "validation gate artifact/payload contract mismatch"
        )
    return dict(artifact), dict(gate)


def _load_gate_subset(gate: Mapping[str, Any]) -> dict[str, Any]:
    artifact = gate.get("subset_manifest")
    if type(artifact) is not dict or not {
        "path",
        "sha256",
        "bytes",
    }.issubset(artifact):
        raise MetricAdapterContractError(
            "replication gate lacks a frozen subset-manifest artifact"
        )
    path, payload = _verified_file_snapshot(
        artifact["path"],
        _require_sha256(
            artifact["sha256"],
            "gate subset-manifest SHA",
        ),
        "gate subset manifest",
    )
    if _require_exact_int(
        artifact["bytes"],
        "gate subset-manifest bytes",
        minimum=1,
    ) != len(payload):
        raise MetricAdapterContractError(
            f"{path}: gate subset-manifest byte count mismatch"
        )
    return _strict_json_snapshot(payload, "gate subset manifest")


def _strong_bind_gate(
    *,
    gate: Mapping[str, Any],
    split: str,
    canonical_path: Path,
    canonical_payload: bytes,
    canonical_rows: Sequence[Mapping[str, Any]],
    ordered_predictions: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    test_authority: Mapping[str, Any] | None,
    formal_mode: bool,
) -> None:
    if not formal_mode:
        return
    source = gate.get("source_closure")
    model_bundle = gate.get("model_bundle")
    if (
        type(source) is not dict
        or source.get("origin")
        != "git@github.com:Xiangyue-Zhang/SemTalk.git"
        or type(model_bundle) is not dict
        or type(model_bundle.get("checkpoints")) is not dict
        or set(model_bundle["checkpoints"])
        != {"base", "face", "hands", "upper", "lower", "global"}
    ):
        raise MetricAdapterContractError(
            "replication gate lacks official Base source/model authority"
        )
    subset = _load_gate_subset(gate)
    parent = subset.get("parent_authority")
    if type(parent) is not dict:
        raise MetricAdapterContractError(
            "replication gate lacks parent val authority"
        )
    if split == "val":
        expected_canonical = {
            "path": str(canonical_path),
            "sha256": sha256_bytes(canonical_payload),
            "bytes": len(canonical_payload),
        }
        parent_canonical = parent.get("canonical_manifest")
        if (
            type(parent_canonical) is not dict
            or {
                key: parent_canonical.get(key)
                for key in ("path", "sha256", "bytes")
            }
            != expected_canonical
        ):
            raise MetricAdapterContractError(
                "val metric canonical manifest differs from gate authority"
            )
        if lineage.get("val_inputs_receipt") != parent.get(
            "val_inputs_receipt"
        ):
            raise MetricAdapterContractError(
                "val inference lineage differs from gate val-input authority"
            )
        subset_rows = subset.get("rows")
        if (
            type(subset_rows) is not list
            or len(subset_rows) != len(ordered_predictions)
            or [
                row.get("canonical_row_sha256")
                for row in subset_rows
            ]
            != [
                compact_canonical_json_sha256(row)
                for row in canonical_rows
                if row.get("split") == "val"
            ]
            or [
                row.get("source_clip_id") for row in subset_rows
            ]
            != [
                row["source_clip_id"] for row in ordered_predictions
            ]
        ):
            raise MetricAdapterContractError(
                "val metric order/rows differ from gate parent authority"
            )
        base_checkpoint = {
            row["candidate_checkpoint_sha256"]
            for row in ordered_predictions
        }
        if (
            len(base_checkpoint) != 1
            or next(iter(base_checkpoint))
            != model_bundle["checkpoints"]["base"].get("sha256")
        ):
            raise MetricAdapterContractError(
                "val candidate checkpoint differs from replication gate"
            )
        pipeline_receipt = lineage.get("pipeline_receipt")
        if type(pipeline_receipt) is not dict:
            raise MetricAdapterContractError(
                "val inference lineage lacks pipeline authority"
            )
        _pipeline_path, pipeline_payload = _verified_file_snapshot(
            pipeline_receipt["path"],
            _require_sha256(
                pipeline_receipt["sha256"],
                "val pipeline receipt SHA",
            ),
            "val pipeline receipt",
        )
        pipeline = _strict_json_snapshot(
            pipeline_payload,
            "val pipeline receipt",
        )
        producer = pipeline.get("source")
        if (
            type(producer) is not dict
            or producer.get("origin") != source["origin"]
            or producer.get("commit") != source["commit"]
            or producer.get("tree") != source["tree"]
        ):
            raise MetricAdapterContractError(
                "val inference source differs from replication gate"
            )
    else:
        if type(test_authority) is not dict:
            raise MetricAdapterContractError(
                "formal test lacks final Base authority"
            )
        authority_source = test_authority["inference_source"]
        if (
            source.get("origin") != authority_source["origin"]
            or source.get("commit") != authority_source["commit"]
            or source.get("tree") != authority_source["tree"]
        ):
            raise MetricAdapterContractError(
                "test authority source differs from replication gate"
            )
        for stage, checkpoint in test_authority["checkpoints"].items():
            if model_bundle["checkpoints"].get(stage) != checkpoint:
                raise MetricAdapterContractError(
                    f"{stage} checkpoint differs between gate and test "
                    "authority"
                )


def _validated_test_authority(
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if type(receipt) is not dict or set(receipt) != {
        "path",
        "sha256",
        "bytes",
        "receipt_payload_sha256",
    }:
        raise MetricAdapterContractError(
            "test authority artifact schema mismatch"
        )
    module = _base_final_authority_module()
    try:
        return module.validate_test_authority(
            receipt["path"],
            expected_file_sha256=_require_sha256(
                receipt["sha256"],
                "test authority file SHA",
            ),
            expected_bytes=_require_exact_int(
                receipt["bytes"],
                "test authority bytes",
                minimum=1,
            ),
            expected_receipt_payload_sha256=_require_sha256(
                receipt["receipt_payload_sha256"],
                "test authority payload SHA",
            ),
        )
    except Exception as exc:
        raise MetricAdapterContractError(
            f"test authority fresh replay failed: {exc}"
        ) from exc


def _reject_forbidden_generator_identity(
    value: Any,
    label: str,
    *,
    semantic_identity: bool = False,
) -> None:
    generator_keys = {
        "generator",
        "generator_identity",
        "generator_name",
        "generator_source",
        "model",
        "model_identity",
        "model_name",
        "model_source",
        "checkpoint",
        "checkpoint_path",
        "checkpoint_source",
        "candidate_checkpoint",
        "selected_checkpoint",
        "fixed_checkpoints",
        "checkpoints",
    }
    evaluator_asset_keys = {
        "evaluator",
        "evaluator_asset",
        "feature_extractor",
        "metric_asset",
        "smplx",
        "smplx_asset",
        "talkshow",
        "talkshow_asset",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = str(key).casefold()
            child_semantic = (
                semantic_identity
                or key_name in generator_keys
                or key_name.endswith("_checkpoint")
                or key_name.endswith("_checkpoint_path")
                or key_name.endswith("_checkpoint_source")
            )
            if (
                key_name in evaluator_asset_keys
                or key_name.endswith("_evaluator")
                or key_name.endswith("_metric_asset")
                or key_name.endswith("_smplx_asset")
            ):
                child_semantic = False
            _reject_forbidden_generator_identity(
                child,
                f"{label}.{key}",
                semantic_identity=child_semantic,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_generator_identity(
                child,
                f"{label}[{index}]",
                semantic_identity=semantic_identity,
            )
    elif isinstance(value, str) and semantic_identity:
        normalized = value.casefold()
        if any(
            token in normalized
            for token in ("speaker2", "sparse", "semgate", "globaldiff")
        ):
            raise MetricAdapterContractError(
                f"{label} contains a forbidden generator identity"
            )


def _validate_exact_artifact_receipt(
    receipt: Any,
    *,
    expected_path: Path,
    label: str,
) -> tuple[Path, bytes]:
    """Verify one exact-path artifact receipt against the live source file."""

    if (
        type(receipt) is not dict
        or set(receipt) != ARTIFACT_KEYS
        or Path(str(receipt.get("path", ""))).resolve()
        != expected_path.resolve()
    ):
        raise MetricAdapterContractError(
            f"{label} escapes its exact artifact path"
        )
    path, payload = _verified_file_snapshot(
        receipt["path"],
        _require_sha256(receipt["sha256"], f"{label} SHA"),
        label,
    )
    if (
        _require_exact_int(
            receipt["bytes"],
            f"{label} bytes",
            minimum=1,
        )
        != len(payload)
    ):
        raise MetricAdapterContractError(f"{label} byte count mismatch")
    return path, payload


def _validate_final_npz_provenance(
    *,
    root: Path,
    ordered_predictions: Sequence[Mapping[str, Any]],
    shard_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Replay the finalizer's exact shard-copy and NPZ-root contract."""

    if (
        len(shard_rows_by_id) != len(ordered_predictions)
        or set(shard_rows_by_id)
        != {row["canonical_clip_id"] for row in ordered_predictions}
    ):
        raise MetricAdapterContractError(
            "test shard manifests do not cover final clips exactly once"
        )
    final_npz_root = root / "npz" / "test"
    expected_final_files: set[str] = set()
    for final_row in ordered_predictions:
        output_id = final_row["canonical_clip_id"]
        shard_row = shard_rows_by_id[output_id]
        if {
            key: final_row[key]
            for key in SHARD_TEST_PREDICTION_ROW_KEYS
            if key not in {"prediction", "ground_truth"}
        } != {
            key: shard_row[key]
            for key in SHARD_TEST_PREDICTION_ROW_KEYS
            if key not in {"prediction", "ground_truth"}
        }:
            raise MetricAdapterContractError(
                f"{output_id}: final/shard identity receipts differ"
            )
        for role_name, prefix in (
            ("prediction", "res"),
            ("ground_truth", "gt"),
        ):
            final_receipt = final_row[role_name]
            shard_receipt = shard_row[role_name]
            expected_name = f"{prefix}_{output_id}.npz"
            expected_final_path = final_npz_root / expected_name
            shard_id = _require_exact_int(
                shard_row.get("global_index"),
                f"{output_id} shard global_index",
            ) % 8
            expected_shard_path = (
                root.parent
                / "shards"
                / f"shard-{shard_id:05d}-of-00008"
                / "npz"
                / "test"
                / expected_name
            )
            _validate_exact_artifact_receipt(
                final_receipt,
                expected_path=expected_final_path,
                label=f"{output_id} finalized {role_name}",
            )
            _validate_exact_artifact_receipt(
                shard_receipt,
                expected_path=expected_shard_path,
                label=f"{output_id} shard {role_name}",
            )
            if {
                key: final_receipt[key]
                for key in ("sha256", "bytes")
            } != {
                key: shard_receipt[key]
                for key in ("sha256", "bytes")
            }:
                raise MetricAdapterContractError(
                    f"{output_id}: finalized {role_name} is not the exact "
                    "authorized shard copy"
                )
            expected_final_files.add(expected_name)
    if (
        not final_npz_root.is_dir()
        or final_npz_root.is_symlink()
        or {
            path.name
            for path in final_npz_root.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        != expected_final_files
        or any(
            path.is_symlink() or not path.is_file()
            for path in final_npz_root.iterdir()
        )
    ):
        raise MetricAdapterContractError(
            "test finalized NPZ root has an inexact regular-file inventory"
        )


def _validate_output_audio_authority(
    *,
    ordered_predictions: Sequence[Mapping[str, Any]],
    authorized_audio_by_clip: Mapping[str, tuple[str, str]],
) -> None:
    if len(authorized_audio_by_clip) != len(ordered_predictions):
        raise MetricAdapterContractError(
            "authorized audio manifests do not cover output rows"
        )
    for row in ordered_predictions:
        if (
            authorized_audio_by_clip.get(row["source_clip_id"])
            != (
                str(Path(row["audio_feature_npz"]).resolve()),
                row["audio_feature_npz_sha256"],
            )
        ):
            raise MetricAdapterContractError(
                f"{row['canonical_clip_id']}: final audio feature differs "
                "from the authorized audio manifests"
            )


def _validate_frozen_inference_evidence(
    *,
    authority: Mapping[str, Any],
    contract: Any | None = None,
    checkpoint_receipts: Any | None = None,
) -> None:
    """Bind only live checkpoint identity; raw producer metadata is telemetry.

    The final authority no longer freezes caller-authored inference contracts
    or nested checkpoint audits.  Base training provenance and winner
    selection are independently reconstructed by the Base-long and published
    winner validators.  Here we accept no provenance from the producer: only
    the six exact live ``path/sha256/bytes`` identities may agree with that
    external authority.  The surrounding final/shard validators separately
    bind source, canonical/audio inputs, output roots and exact-once coverage.
    """

    if contract is not None:
        observed = contract.get("checkpoints")
        if type(observed) is not dict or set(observed) != set(
            authority["checkpoints"]
        ):
            raise MetricAdapterContractError(
                "test final lineage checkpoint projection schema mismatch"
            )
        for stage, expected in authority["checkpoints"].items():
            row = observed[stage]
            if (
                type(row) is not dict
                or row.get("path") != expected["path"]
                or row.get("expected_sha256") != expected["sha256"]
            ):
                raise MetricAdapterContractError(
                    f"test final lineage changed the {stage} checkpoint"
                )
    if checkpoint_receipts is not None:
        if type(checkpoint_receipts) is not dict or set(
            checkpoint_receipts
        ) != set(authority["checkpoints"]):
            raise MetricAdapterContractError(
                "test shard checkpoint projection schema mismatch"
            )
        for stage, expected in authority["checkpoints"].items():
            row = checkpoint_receipts[stage]
            if (
                type(row) is not dict
                or row.get("formal_stage") != stage
                or row.get("path") != expected["path"]
                or row.get("sha256") != expected["sha256"]
                or row.get("bytes") != expected["bytes"]
            ):
                raise MetricAdapterContractError(
                    f"test shard changed the {stage} checkpoint"
                )


def _cross_validate_test_outputs(
    *,
    authority: Mapping[str, Any],
    canonical_path: Path,
    canonical_payload: bytes,
    canonical_rows: Sequence[Mapping[str, Any]],
    prediction_path: Path,
    prediction_payload: bytes,
    ordered_predictions: Sequence[Mapping[str, Any]],
    lineage_path: Path,
    lineage: Mapping[str, Any],
) -> None:
    root = Path(authority["expected_output_root"]).resolve()
    if (
        prediction_path != root / "final_manifest.jsonl"
        or lineage_path != root / "final_lineage.json"
    ):
        raise MetricAdapterContractError(
            "test prediction outputs escape their authorized final root"
        )
    canonical_authority = authority["canonical"]
    if canonical_authority["manifest"] != {
        "path": str(canonical_path),
        "sha256": sha256_bytes(canonical_payload),
        "bytes": len(canonical_payload),
    }:
        raise MetricAdapterContractError(
            "test metric canonical root differs from test authority"
        )
    selected = [
        row for row in canonical_rows if row.get("split") == "test"
    ]
    if [
        canonical_json_sha256(row) for row in selected
    ] != canonical_authority["ordered_row_sha256"]:
        raise MetricAdapterContractError(
            "test metric canonical row hashes differ from test authority"
        )
    if (
        set(lineage)
        != {
            "format",
            "status",
            "contract",
            "contract_sha256",
            "runtime",
            "runtime_sha256",
            "shards",
            "final_manifest_sha256",
            "clip_manifest_sha256",
        }
        or
        lineage.get("format")
        != "semtalk_show_base_inference_final_lineage_v1"
        or lineage.get("status") != "complete"
        or lineage.get("final_manifest_sha256")
        != sha256_bytes(prediction_payload)
        or type(lineage.get("contract")) is not dict
        or lineage.get("contract_sha256")
        != canonical_json_sha256(lineage["contract"])
        or type(lineage.get("runtime")) is not dict
        or lineage.get("runtime_sha256")
        != canonical_json_sha256(lineage["runtime"])
        or lineage.get("clip_manifest_sha256")
        != sha256_bytes(
            "".join(
                f"{row['canonical_clip_id']}\n"
                for row in ordered_predictions
            ).encode("utf-8")
        )
    ):
        raise MetricAdapterContractError(
            "test final lineage is incomplete or self-inconsistent"
        )
    contract = lineage["contract"]
    _validate_frozen_inference_evidence(
        authority=authority,
        contract=contract,
    )
    source = contract.get("source")
    authorized_source = authority["inference_source"]
    if (
        type(source) is not dict
        or any(
            source.get(key) != authorized_source[key]
            for key in ("origin", "commit", "tree", "source_root")
        )
    ):
        raise MetricAdapterContractError(
            "test final lineage source differs from test authority"
        )
    if (
        contract.get("canonical_manifest") != str(canonical_path)
        or contract.get("canonical_manifest_sha256")
        != canonical_authority["manifest"]["sha256"]
        or contract.get("canonical_summary_sha256")
        != canonical_authority["summary"]["sha256"]
        or contract.get("canonical_lineage_sha256")
        != canonical_authority["lineage"]["sha256"]
        or contract.get("test_clips") != FORMAL_SPLITS["test"]["count"]
        or contract.get("num_shards") != 8
        or contract.get("speaker_mapping") != SHOW_SPEAKER_IDS
    ):
        raise MetricAdapterContractError(
            "test final lineage canonical/split contract changed"
        )
    expected_audio_manifest = {
        shard["manifest"]["path"]: shard["manifest"]["sha256"]
        for shard in authority["audio"]
    }
    expected_audio_summary = {
        shard["summary"]["path"]: shard["summary"]["sha256"]
        for shard in authority["audio"]
    }
    expected_audio_lineage = {
        shard["lineage"]["path"]: shard["lineage"]["sha256"]
        for shard in authority["audio"]
    }
    if (
        contract.get("audio_manifest_sha256")
        != expected_audio_manifest
        or contract.get("audio_summary_sha256")
        != expected_audio_summary
        or contract.get("audio_lineage_sha256")
        != expected_audio_lineage
    ):
        raise MetricAdapterContractError(
            "test final lineage audio authorities changed"
        )
    authorized_audio_by_clip: dict[str, tuple[str, str]] = {}
    for audio_shard in authority["audio"]:
        manifest_receipt = audio_shard["manifest"]
        _audio_path, audio_payload = _verified_file_snapshot(
            manifest_receipt["path"],
            manifest_receipt["sha256"],
            f"authorized audio shard {audio_shard['shard_id']} manifest",
        )
        if manifest_receipt["bytes"] != len(audio_payload):
            raise MetricAdapterContractError(
                "authorized audio manifest byte count mismatch"
            )
        for audio_row in _strict_jsonl_snapshot(
            audio_payload,
            f"authorized audio shard {audio_shard['shard_id']} manifest",
        ):
            clip_id = audio_row.get("clip_id")
            feature_path = audio_row.get("audio_feature_npz")
            feature_sha = audio_row.get("audio_feature_npz_sha256")
            if (
                type(clip_id) is not str
                or clip_id in authorized_audio_by_clip
                or type(feature_path) is not str
                or not Path(feature_path).is_absolute()
            ):
                raise MetricAdapterContractError(
                    "authorized audio manifest row identity mismatch"
                )
            authorized_audio_by_clip[clip_id] = (
                str(Path(feature_path).resolve()),
                _require_sha256(
                    feature_sha,
                    f"authorized audio feature {clip_id} SHA",
                ),
            )
    if len(authorized_audio_by_clip) != FORMAL_SPLITS["test"]["count"]:
        raise MetricAdapterContractError(
            "authorized audio manifests do not cover formal test"
        )
    checkpoints = contract.get("checkpoints")
    if type(checkpoints) is not dict or set(checkpoints) != set(
        authority["checkpoints"]
    ):
        raise MetricAdapterContractError(
            "test final lineage checkpoint bundle schema mismatch"
        )
    for stage, authorized in authority["checkpoints"].items():
        observed = checkpoints[stage]
        if (
            type(observed) is not dict
            or observed.get("path") != authorized["path"]
            or observed.get("expected_sha256") != authorized["sha256"]
        ):
            raise MetricAdapterContractError(
                f"test final lineage changed the {stage} checkpoint"
            )
    shards = lineage.get("shards")
    if type(shards) is not list or len(shards) != 8:
        raise MetricAdapterContractError(
            "test final lineage does not contain exactly eight shards"
        )
    shard_rows_by_id: dict[str, dict[str, Any]] = {}
    shard_checkpoint_receipts: dict[str, Any] | None = None
    final_runtime = lineage["runtime"]
    final_runtime_sha256 = lineage["runtime_sha256"]
    for shard_id, shard in enumerate(shards):
        if (
            type(shard) is not dict
            or set(shard) != {
                "shard_id",
                "manifest",
                "manifest_sha256",
                "summary",
                "summary_sha256",
                "lineage",
                "lineage_sha256",
                "clips",
            }
            or shard.get("shard_id") != shard_id
            or type(shard.get("clips")) is not int
            or shard["clips"] < 1
        ):
            raise MetricAdapterContractError(
                f"test final lineage shard {shard_id} schema mismatch"
            )
        expected_parent = (
            root.parent
            / "shards"
            / f"shard-{shard_id:05d}-of-00008"
        )
        payloads: dict[str, bytes] = {}
        expected_names = {
            "manifest": "manifest.jsonl",
            "summary": "summary.json",
            "lineage": "lineage.json",
        }
        for role, expected_name in expected_names.items():
            path, payload = _verified_file_snapshot(
                shard[role],
                _require_sha256(
                    shard[f"{role}_sha256"],
                    f"test shard {shard_id} {role} SHA",
                ),
                f"test shard {shard_id} {role}",
            )
            if path != expected_parent / expected_name:
                raise MetricAdapterContractError(
                    f"test shard {shard_id} {role} escapes its exact root"
                )
            payloads[role] = payload

        rows = _strict_jsonl_snapshot(
            payloads["manifest"],
            f"test shard {shard_id} manifest",
        )
        manifest_sha256 = sha256_bytes(payloads["manifest"])
        summary = _strict_json_snapshot(
            payloads["summary"],
            f"test shard {shard_id} summary",
        )
        shard_lineage = _strict_json_snapshot(
            payloads["lineage"],
            f"test shard {shard_id} lineage",
        )
        if (
            set(summary) != SHARD_SUMMARY_KEYS
            or summary.get("format")
            != "semtalk_show_base_inference_shard_summary_v1"
            or summary.get("status") != "complete"
            or _require_exact_int(
                summary.get("shard_id"),
                f"test shard {shard_id} summary shard_id",
            )
            != shard_id
            or _require_exact_int(
                summary.get("num_shards"),
                f"test shard {shard_id} summary num_shards",
            )
            != 8
            or _require_exact_int(
                summary.get("selected_clips"),
                f"test shard {shard_id} summary selected_clips",
                minimum=1,
            )
            != len(rows)
            or _require_exact_int(
                summary.get("expected_test_clips"),
                f"test shard {shard_id} summary expected_test_clips",
            )
            != FORMAL_SPLITS["test"]["count"]
            or summary.get("manifest_sha256") != manifest_sha256
            or summary.get("contract_sha256")
            != lineage["contract_sha256"]
            or summary.get("runtime_sha256") != final_runtime_sha256
            or summary.get("finite") is not True
            or summary.get("exact_once") is not True
        ):
            raise MetricAdapterContractError(
                f"test shard {shard_id} summary contract mismatch"
            )
        if (
            set(shard_lineage) != SHARD_LINEAGE_KEYS
            or shard_lineage.get("format")
            != "semtalk_show_base_inference_shard_lineage_v1"
            or shard_lineage.get("status") != "complete"
            or _require_exact_int(
                shard_lineage.get("shard_id"),
                f"test shard {shard_id} lineage shard_id",
            )
            != shard_id
            or _require_exact_int(
                shard_lineage.get("num_shards"),
                f"test shard {shard_id} lineage num_shards",
            )
            != 8
            or shard_lineage.get("manifest_sha256") != manifest_sha256
            or shard_lineage.get("contract") != contract
            or shard_lineage.get("contract_sha256")
            != lineage["contract_sha256"]
            or shard_lineage.get("runtime") != final_runtime
            or shard_lineage.get("runtime_sha256")
            != final_runtime_sha256
            or canonical_json_sha256(shard_lineage.get("runtime"))
            != final_runtime_sha256
        ):
            raise MetricAdapterContractError(
                f"test shard {shard_id} lineage contract mismatch"
            )
        observed_checkpoints = shard_lineage.get("checkpoints")
        if (
            type(observed_checkpoints) is not dict
            or set(observed_checkpoints) != set(authority["checkpoints"])
        ):
            raise MetricAdapterContractError(
                f"test shard {shard_id} checkpoint receipt schema mismatch"
            )
        _validate_frozen_inference_evidence(
            authority=authority,
            checkpoint_receipts=observed_checkpoints,
        )
        for stage, authorized in authority["checkpoints"].items():
            observed = observed_checkpoints[stage]
            if (
                type(observed) is not dict
                or observed.get("formal_stage") != stage
                or observed.get("path") != authorized["path"]
                or observed.get("sha256") != authorized["sha256"]
            ):
                raise MetricAdapterContractError(
                    f"test shard {shard_id} changed the {stage} checkpoint "
                    "receipt"
                )
        if shard_checkpoint_receipts is None:
            shard_checkpoint_receipts = observed_checkpoints
        elif observed_checkpoints != shard_checkpoint_receipts:
            raise MetricAdapterContractError(
                "test inference shards used different checkpoint receipts"
            )
        if len(rows) != shard["clips"]:
            raise MetricAdapterContractError(
                f"test shard {shard_id} manifest count mismatch"
            )
        expected_npz_root = expected_parent / "npz" / "test"
        for row in rows:
            if (
                type(row) is not dict
                or set(row) != SHARD_TEST_PREDICTION_ROW_KEYS
            ):
                raise MetricAdapterContractError(
                    f"test shard {shard_id} row schema mismatch"
                )
            output_id = row.get("canonical_clip_id")
            global_index = row.get("global_index")
            if (
                type(output_id) is not str
                or output_id in shard_rows_by_id
                or type(global_index) is not int
                or isinstance(global_index, bool)
                or global_index % 8 != shard_id
            ):
                raise MetricAdapterContractError(
                    f"test shard {shard_id} row identity/sharding mismatch"
                )
            for role_name, prefix in (
                ("prediction", "res"),
                ("ground_truth", "gt"),
            ):
                _validate_exact_artifact_receipt(
                    row.get(role_name),
                    expected_path=(
                        expected_npz_root
                        / f"{prefix}_{output_id}.npz"
                    ),
                    label=(
                        f"test shard {shard_id} {output_id} {role_name}"
                    ),
                )
            shard_rows_by_id[output_id] = row
    if len(shard_rows_by_id) != FORMAL_SPLITS["test"]["count"]:
        raise MetricAdapterContractError(
            "test shard manifests do not cover final clips exactly once"
        )
    _validate_final_npz_provenance(
        root=root,
        ordered_predictions=ordered_predictions,
        shard_rows_by_id=shard_rows_by_id,
    )
    _validate_output_audio_authority(
        ordered_predictions=ordered_predictions,
        authorized_audio_by_clip=authorized_audio_by_clip,
    )
    if sorted(
        row["global_index"] for row in ordered_predictions
    ) != list(
        range(
            FORMAL_SPLITS["test"]["global_start"],
            FORMAL_SPLITS["test"]["global_stop"],
        )
    ):
        raise MetricAdapterContractError(
            "test final manifest is not the exact canonical global domain"
        )
    _reject_forbidden_generator_identity(contract, "test final contract")


def build_distribution_receipt(
    *,
    prediction_manifest_artifact: Mapping[str, Any],
    prediction_artifacts: Sequence[Mapping[str, Any]],
    validation_gate: Mapping[str, Any],
    expected_scope: str = "validation_candidate_family",
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Delegate to the Base gate's sole canonical receipt constructor."""

    gate_artifact, _gate = _validate_external_validation_gate(
        validation_gate,
        expected_scope=expected_scope,
        test_only_allow_four_clip_subset=(
            test_only_allow_four_clip_subset
        ),
    )
    module = _replication_gate_module()
    try:
        return module.build_distribution_receipt_from_validated_artifacts(
            gate_artifact=gate_artifact,
            prediction_manifest_artifact=prediction_manifest_artifact,
            prediction_records=prediction_artifacts,
        )
    except Exception as exc:
        raise MetricAdapterContractError(
            f"cannot build canonical Base distribution receipt: {exc}"
        ) from exc


def validate_distribution_declaration(
    declaration: Mapping[str, Any],
    *,
    expected_gate_artifact: Mapping[str, Any],
    expected_prediction_manifest: Mapping[str, Any],
    expected_prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate through deterministic_replication_gate, never a local copy."""

    module = _replication_gate_module()
    try:
        observed = module.validate_distribution_receipt(
            declaration,
            expected_gate_artifact=expected_gate_artifact,
            expected_prediction_manifest=expected_prediction_manifest,
            expected_prediction_records=expected_prediction_records,
        )
    except Exception as exc:
        raise MetricAdapterContractError(
            f"distribution declaration rejected by Base gate: {exc}"
        ) from exc
    if (
        observed.get("format")
        != DISTRIBUTION_RECEIPT_FORMAT
        or observed.get("payload_hash_algorithm")
        != PAYLOAD_HASH_ALGORITHM
        or observed.get("protocol") != DISTRIBUTION_PROTOCOL
        or observed.get("physical_samples_per_clip") != 1
        or observed.get("independent_samples") is not False
        or observed.get("deterministic_delta_distribution") is not True
        or observed.get("replication_algorithm") != REPLICATION_ALGORITHM
        or observed.get("seed_consumed") is not False
        or observed.get("released2_slots") != list(RELEASED2_SLOTS)
        or observed.get("paper16_slots") != list(PAPER16_SLOTS)
        or observed.get("face_slot") != FACE_SLOT
        or observed.get("variation_policy")
        != module.variation_policy_receipt()
        or observed.get("metric_input_materialization")
        != METRIC_INPUT_MATERIALIZATION
    ):
        raise MetricAdapterContractError(
            "distribution randomness/slot declaration is invalid"
        )
    return dict(observed)


def _ground_truth_matches_canonical(
    ground_truth: Mapping[str, np.ndarray],
    canonical: Mapping[str, np.ndarray],
) -> None:
    expected = {
        "betas": canonical["beta"][0],
        "poses": canonical["pose"],
        "expressions": canonical["facial"],
        "trans": canonical["trans"],
    }
    for name, value in expected.items():
        if not np.array_equal(ground_truth[name], value):
            raise MetricAdapterContractError(
                f"ground-truth NPZ {name} differs from canonical SHOW"
            )


def _require_concrete_formal_backend(backend: MetricBackend) -> None:
    if (
        type(backend) is not TalkShowCudaMetricBackend
        or getattr(backend, "_formal_construction_token", None)
        is not _FORMAL_BACKEND_CONSTRUCTION_TOKEN
    ):
        raise MetricAdapterContractError(
            "formal metrics require the concretely constructed pinned "
            "TalkSHOW CUDA backend"
        )


def _validated_backend_features(
    backend: MetricBackend,
    parameters_265: np.ndarray,
) -> np.ndarray:
    value = _finite_array(
        backend.extract_body_features(parameters_265),
        name="body features",
    )
    if value.ndim != 2 or 0 in value.shape:
        raise MetricAdapterContractError(
            f"body features must be a non-empty matrix, got {value.shape}"
        )
    return value


def _validated_backend_joints(
    backend: MetricBackend,
    parameters_265: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    value = _finite_array(
        backend.joints(parameters_265, betas),
        name="SMPL-X joints",
    )
    expected_prefix = parameters_265.shape[:2]
    if (
        value.ndim != 4
        or value.shape[:2] != expected_prefix
        or value.shape[2] < 75
        or value.shape[3] != 3
    ):
        raise MetricAdapterContractError(
            f"SMPL-X joints have invalid shape {value.shape}"
        )
    return value


def _audio_beats(
    backend: MetricBackend,
    waveform: np.ndarray,
    override: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    function = override
    if function is None:
        function = getattr(backend, "audio_beats", None)
    if not callable(function):
        raise MetricAdapterContractError(
            "metric backend does not provide audio-beat extraction"
        )
    beats = _finite_array(
        function(waveform),
        name="audio beat times",
    ).reshape(-1)
    return beats


def evaluate_canonical_bundle(
    *,
    canonical_manifest: str | Path,
    expected_canonical_manifest_sha256: str,
    prediction_manifest: str | Path,
    expected_prediction_manifest_sha256: str,
    prediction_lineage: str | Path,
    expected_prediction_lineage_sha256: str,
    validation_gate: Mapping[str, Any],
    distribution_declaration: Mapping[str, Any],
    backend: MetricBackend,
    split: str,
    expected_clip_count: int,
    test_authority: Mapping[str, Any] | None = None,
    audio_beat_extractor: Callable[[np.ndarray], np.ndarray] | None = None,
    audio_decoder_16k: Callable[[bytes], np.ndarray] | None = None,
    formal_mode: bool = True,
    test_only_allow_four_clip_gate: bool = False,
) -> dict[str, Any]:
    """Evaluate one immutable SemTalk bundle without invoking a generator."""

    if split not in {"val", "test"}:
        raise MetricAdapterContractError("split must be val or test")
    expected_clip_count = _require_exact_int(
        expected_clip_count,
        "expected_clip_count",
        minimum=1,
    )
    if (
        formal_mode
        and expected_clip_count != FORMAL_SPLITS[split]["count"]
    ):
        raise MetricAdapterContractError(
            "formal evaluation clip count differs from frozen SHOW split"
        )
    if (
        type(formal_mode) is not bool
        or type(test_only_allow_four_clip_gate) is not bool
    ):
        raise MetricAdapterContractError(
            "formal/test-only mode flags must be boolean"
        )
    if formal_mode and test_only_allow_four_clip_gate:
        raise MetricAdapterContractError(
            "formal evaluation cannot use the four-clip CPU fixture"
        )
    if formal_mode:
        _require_concrete_formal_backend(backend)
    if split == "val" and test_authority is not None:
        raise MetricAdapterContractError(
            "validation metrics must not consume test authority"
        )
    validated_test_authority = (
        _validated_test_authority(test_authority)
        if split == "test" and formal_mode
        else None
    )
    if split == "test" and formal_mode and validated_test_authority is None:
        raise MetricAdapterContractError(
            "formal test metrics require an externally pinned test authority"
        )
    if formal_mode and (
        audio_decoder_16k is not None
        or audio_beat_extractor is not None
    ):
        raise MetricAdapterContractError(
            "formal evaluation forbids injected audio decoder/beat extractor"
        )
    decoder = (
        getattr(backend, "decode_audio_16k", None)
        if audio_decoder_16k is None
        else audio_decoder_16k
    )
    if not callable(decoder):
        raise MetricAdapterContractError(
            "metric backend lacks a verified-snapshot 16 kHz decoder"
        )
    assets, runtime = _validate_backend_receipts(
        backend,
        require_cuda=formal_mode,
    )
    canonical_path, canonical_payload = _verified_file_snapshot(
        canonical_manifest,
        expected_canonical_manifest_sha256,
        "canonical SHOW manifest",
    )
    canonical_rows = _strict_jsonl_snapshot(
        canonical_payload,
        "canonical SHOW manifest",
    )
    canonical_by_id = _validate_canonical_rows(
        canonical_rows,
        split=split,
        expected_clip_count=expected_clip_count,
        formal_mode=formal_mode,
        test_only_allow_four_clip_subset=(
            test_only_allow_four_clip_gate
        ),
    )
    prediction_path, prediction_payload = _verified_file_snapshot(
        prediction_manifest,
        expected_prediction_manifest_sha256,
        "SemTalk prediction manifest",
    )
    prediction_rows = _strict_jsonl_snapshot(
        prediction_payload,
        "SemTalk prediction manifest",
    )
    ordered_predictions = _validate_prediction_rows(
        prediction_rows,
        canonical=canonical_by_id,
        split=split,
    )
    lineage_path, lineage_payload = _verified_file_snapshot(
        prediction_lineage,
        expected_prediction_lineage_sha256,
        "SemTalk prediction lineage",
    )
    lineage = _strict_json_snapshot(
        lineage_payload,
        "SemTalk prediction lineage",
    )
    lineage_payload_sha = _validate_prediction_lineage(
        lineage,
        split=split,
        prediction_manifest_path=prediction_path,
        prediction_manifest_sha256=sha256_bytes(prediction_payload),
        expected_clip_count=expected_clip_count,
    )
    if split == "val" and formal_mode:
        _fresh_validate_formal_val_lineage(
            lineage_path=lineage_path,
            lineage_sha256=sha256_bytes(lineage_payload),
            lineage=lineage,
            prediction_path=prediction_path,
            prediction_sha256=sha256_bytes(prediction_payload),
        )
    if split == "test" and formal_mode:
        assert validated_test_authority is not None
        _cross_validate_test_outputs(
            authority=validated_test_authority,
            canonical_path=canonical_path,
            canonical_payload=canonical_payload,
            canonical_rows=canonical_rows,
            prediction_path=prediction_path,
            prediction_payload=prediction_payload,
            ordered_predictions=ordered_predictions,
            lineage_path=lineage_path,
            lineage=lineage,
        )
    artifact_rows = [
        {
            "canonical_clip_id": row["canonical_clip_id"],
            "prediction_sha256": row["prediction"]["sha256"],
            "prediction_bytes": row["prediction"]["bytes"],
        }
        for row in ordered_predictions
    ]
    gate_artifact, gate = _validate_external_validation_gate(
        validation_gate,
        expected_scope=(
            "validation_candidate_family"
            if split == "val"
            else "final_winner"
        ),
        test_only_allow_four_clip_subset=(
            test_only_allow_four_clip_gate
        ),
    )
    _strong_bind_gate(
        gate=gate,
        split=split,
        canonical_path=canonical_path,
        canonical_payload=canonical_payload,
        canonical_rows=canonical_rows,
        ordered_predictions=ordered_predictions,
        lineage=lineage,
        test_authority=validated_test_authority,
        formal_mode=formal_mode,
    )
    prediction_manifest_artifact = {
        "path": str(prediction_path),
        "sha256": sha256_bytes(prediction_payload),
        "bytes": len(prediction_payload),
    }
    distribution = validate_distribution_declaration(
        distribution_declaration,
        expected_gate_artifact=gate_artifact,
        expected_prediction_manifest=prediction_manifest_artifact,
        expected_prediction_records=artifact_rows,
    )
    body = {
        protocol: {
            "real": FeatureMoments(),
            "generated": FeatureMoments(),
            "variation_sum": 0.0,
            "variation_integrity_tolerance_sum": 0.0,
            "bc_numerator": 0.0,
            "bc_denominator": 0,
            "bc_sample_evaluations": 0,
        }
        for protocol in PROTOCOLS
    }
    face_sums = {
        "jaw_l1": 0.0,
        "landmark_l1": 0.0,
        "LVD": 0.0,
        "face_l2_combined": 0.0,
    }
    per_speaker = {name: 0 for name in SPEAKER_NAMES}
    frame_count = 0
    for row in ordered_predictions:
        output_id = row["canonical_clip_id"]
        canonical_row = canonical_by_id[output_id]
        frames = canonical_row["frames"]
        canonical_arrays = _load_canonical_npz(
            canonical_row["canonical_npz"],
            expected_sha256=canonical_row["canonical_npz_sha256"],
            frames=frames,
            speaker_id=canonical_row["speaker_id"],
        )
        prediction = _load_output_npz(
            row["prediction"],
            frames=frames,
            prediction=True,
            expected_name=f"res_{output_id}.npz",
        )
        ground_truth = _load_output_npz(
            row["ground_truth"],
            frames=frames,
            prediction=False,
            expected_name=f"gt_{output_id}.npz",
        )
        _ground_truth_matches_canonical(ground_truth, canonical_arrays)
        if not np.array_equal(
            prediction["betas"],
            canonical_arrays["beta"][0],
        ):
            raise MetricAdapterContractError(
                f"{output_id}: prediction betas differ from canonical SHOW"
            )
        ground_truth_265 = reorder_to_talkshow(
            canonical_arrays["pose"],
            canonical_arrays["facial"],
        )
        prediction_265 = reorder_to_talkshow(
            prediction["poses"],
            prediction["expressions"],
        )
        real_features = _validated_backend_features(
            backend,
            ground_truth_265[None],
        )
        body_parameters = released_body_parameters(
            prediction_265[None],
        )
        generated_joints = _validated_backend_joints(
            backend,
            np.repeat(
                body_parameters,
                NUM_LOGICAL_SLOTS,
                axis=0,
            ),
            prediction["betas"],
        )
        waveform = _validate_wav_row_and_decode(
            canonical_row,
            motion_frames=frames,
            decode_audio_16k=decoder,
        )
        beats = _audio_beats(
            backend,
            waveform,
            audio_beat_extractor,
        )
        for protocol, definition in PROTOCOLS.items():
            samples = int(definition["num_samples"])
            state = body[protocol]
            generated_features = _validated_backend_features(
                backend,
                np.repeat(
                    prediction_265[None],
                    samples,
                    axis=0,
                ),
            )
            state["real"].update(real_features)
            state["generated"].update(generated_features)
            physically_repeated = generated_joints[:samples]
            observed_variation = variation_from_joints(
                physically_repeated
            )
            scale = max(
                1.0,
                float(np.max(np.abs(generated_joints))),
            )
            roundoff_tolerance = (
                np.finfo(np.float64).eps
                * scale
                * scale
                * generated_joints.shape[1]
                * generated_joints.shape[2]
                * 64.0
            )
            if (
                not all(
                    np.array_equal(
                        physically_repeated[index],
                        physically_repeated[0],
                    )
                    for index in range(samples)
                )
                or observed_variation > roundoff_tolerance
            ):
                raise MetricAdapterContractError(
                    f"{output_id}: delta-distribution Variation is not zero"
                )
            # Preserve the exact released primitive result.  The tolerance is
            # only an integrity bound for a physically repeated point mass;
            # it is never used to rewrite the reported value to zero.
            state["variation_sum"] += observed_variation
            state["variation_integrity_tolerance_sum"] += (
                roundoff_tolerance
            )
            bc_numerator, bc_denominator = bc_components_for_batch(
                physically_repeated,
                beats,
                str(definition["bc_sample_policy"]),
            )
            state["bc_numerator"] += bc_numerator
            state["bc_denominator"] += bc_denominator
            state["bc_sample_evaluations"] += (
                1
                if definition["bc_sample_policy"] == "first"
                else samples
            )
        isolated_gt, isolated_prediction = released_face_parameters(
            ground_truth_265,
            prediction_265,
        )
        face_joints = _validated_backend_joints(
            backend,
            np.stack((isolated_gt, isolated_prediction)),
            prediction["betas"],
        )
        clip_face = released_face_metrics(
            face_joints[0],
            face_joints[1],
        )
        for name, value in clip_face.items():
            face_sums[name] += value
        per_speaker[canonical_row["speaker"]] += 1
        frame_count += frames
    if sum(per_speaker.values()) != expected_clip_count:
        raise MetricAdapterContractError(
            "per-speaker counts do not cover the metric split"
        )
    body_report: dict[str, Any] = {}
    for protocol, definition in PROTOCOLS.items():
        state = body[protocol]
        if state["bc_denominator"] < 1:
            raise MetricAdapterContractError(
                f"{protocol} produced zero BC denominator"
            )
        metrics = {
            "FGD": frechet_distance(
                state["real"],
                state["generated"],
            ),
            "Variation": state["variation_sum"] / expected_clip_count,
            "BC": state["bc_numerator"] / state["bc_denominator"],
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise MetricAdapterContractError(
                f"{protocol} metrics violate the finite delta contract"
            )
        body_report[protocol] = {
            "protocol": protocol,
            "logical_slots": list(definition["logical_slots"]),
            "num_samples": definition["num_samples"],
            "bc_sample_policy": definition["bc_sample_policy"],
            "counts": {
                "clips": expected_clip_count,
                "per_speaker": per_speaker,
                "real_features": state["real"].count,
                "generated_features": state["generated"].count,
                "bc_sample_evaluations": state[
                    "bc_sample_evaluations"
                ],
                "bc_denominator": state["bc_denominator"],
            },
            "metrics": metrics,
            "feature_statistics": {
                "real": state["real"].to_json(),
                "generated": state["generated"].to_json(),
            },
            "primitive_receipt": {
                "variation_sum": state["variation_sum"],
                "variation_integrity_tolerance_sum": state[
                    "variation_integrity_tolerance_sum"
                ],
                "bc_numerator": state["bc_numerator"],
                "bc_denominator": state["bc_denominator"],
            },
        }
    face_means = {
        name: value / expected_clip_count
        for name, value in face_sums.items()
    }
    if not all(math.isfinite(value) for value in face_means.values()):
        raise MetricAdapterContractError("face metric means are non-finite")
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "report_payload_hash_algorithm": (
            REPORT_PAYLOAD_HASH_ALGORITHM
        ),
        "status": "complete",
        "generator": "SemTalk Base-only",
        "dataset": {
            "name": "SHOW",
            "release": "ReleaseV1.0",
            "selection": ">2s",
            "speakers": dict(SHOW_SPEAKER_IDS),
        },
        "split": split,
        "selection_protocol": {
            "primary_metric": PRIMARY_METRIC_PATH,
            "mode": "min",
            "validation_only_for_selection": True,
            "test_evaluations": 0 if split == "val" else 1,
        },
        "distribution_receipt": distribution,
        "inputs": {
            "canonical_manifest": {
                "path": str(canonical_path),
                "sha256": sha256_bytes(canonical_payload),
                "bytes": len(canonical_payload),
                "rows": len(canonical_rows),
                "selected_rows": expected_clip_count,
            },
            "prediction_manifest": {
                "path": str(prediction_path),
                "sha256": sha256_bytes(prediction_payload),
                "bytes": len(prediction_payload),
                "rows": len(ordered_predictions),
            },
            "prediction_lineage": {
                "path": str(lineage_path),
                "sha256": sha256_bytes(lineage_payload),
                "bytes": len(lineage_payload),
                "payload_sha256": lineage_payload_sha,
            },
            "test_authority": (
                None
                if validated_test_authority is None
                else {
                    "path": str(Path(test_authority["path"]).resolve()),
                    "sha256": test_authority["sha256"],
                    "bytes": test_authority["bytes"],
                    "receipt_payload_sha256": test_authority[
                        "receipt_payload_sha256"
                    ],
                }
            ),
        },
        "metric_assets": assets,
        "counts": {
            "clips": expected_clip_count,
            "frames": frame_count,
            "physical_prediction_artifacts": expected_clip_count,
            "logical_samples_per_clip": NUM_LOGICAL_SLOTS,
            "per_speaker": per_speaker,
            "all_tensors_finite": True,
            "exact_once": True,
        },
        "body": body_report,
        "face": {
            "protocol": "released_face_sample0",
            "logical_slot": FACE_SLOT,
            "aggregation": "equal_clip_mean",
            "counts": {
                "clips": expected_clip_count,
                "per_speaker": per_speaker,
            },
            "metrics": {
                "released": {
                    "jaw_l1": face_means["jaw_l1"],
                    "landmark_l1": face_means["landmark_l1"],
                    "LVD": face_means["LVD"],
                },
                "derived": {
                    "face_l2_combined": face_means[
                        "face_l2_combined"
                    ],
                },
            },
        },
        "rs": {"status": "N/A/unreleased", "value": None},
        "runtime": runtime,
        "formal_mode": formal_mode,
        "test_only_mode": test_only_allow_four_clip_gate,
    }
    report["report_payload_sha256"] = canonical_json_sha256(report)
    return report


def _report_number(value: Any, label: str, *, nonnegative: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        raise MetricAdapterContractError(
            f"{label} must be a finite"
            f"{' nonnegative' if nonnegative else ''} number"
        )
    return float(value)


def _validate_report_feature_statistics(
    value: Any,
    *,
    expected_count: int,
    label: str,
) -> FeatureMoments:
    return FeatureMoments.from_json(
        value,
        expected_count=expected_count,
        label=label,
    )


def validate_report(
    report: Mapping[str, Any],
    *,
    expected_split: str,
    expected_clip_count: int,
    expected_prediction_manifest: Mapping[str, Any],
    expected_distribution_receipt: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
    expected_test_authority: Mapping[str, Any] | None = None,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Fresh, fail-closed validator for selector/merge consumers.

    The Base selector must separately fresh-validate the shared deterministic
    receipt against its prediction records before calling this function.
    This validator then binds that exact receipt, the formal CUDA runtime,
    pinned metric source/assets, and every report field used for selection.
    """

    if expected_split not in {"val", "test"}:
        raise MetricAdapterContractError("invalid expected report split")
    expected_clip_count = _require_exact_int(
        expected_clip_count,
        "expected report clip count",
        minimum=1,
    )
    fixture_mode = (
        test_only_allow_four_clip_subset
        and expected_clip_count == 4
    )
    if (
        not fixture_mode
        and expected_clip_count != FORMAL_SPLITS[expected_split]["count"]
    ):
        raise MetricAdapterContractError(
            "formal report clip count differs from frozen SHOW split"
        )
    expected_manifest = dict(expected_prediction_manifest)
    if set(expected_manifest) != {"path", "sha256", "bytes"}:
        raise MetricAdapterContractError(
            "expected prediction-manifest artifact schema mismatch"
        )
    if (
        type(expected_manifest["path"]) is not str
        or not Path(expected_manifest["path"]).is_absolute()
    ):
        raise MetricAdapterContractError(
            "expected prediction-manifest path must be absolute"
        )
    _require_sha256(
        expected_manifest["sha256"],
        "expected prediction-manifest SHA",
    )
    _require_exact_int(
        expected_manifest["bytes"],
        "expected prediction-manifest bytes",
        minimum=1,
    )
    frozen_selection_protocol = {
        "primary_metric": PRIMARY_METRIC_PATH,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 0 if expected_split == "val" else 1,
    }
    if dict(expected_selection_protocol) != frozen_selection_protocol:
        raise MetricAdapterContractError(
            "caller selection protocol differs from frozen SHOW policy"
        )
    if type(report) is not dict or set(report) != {
        "format",
        "report_payload_hash_algorithm",
        "status",
        "generator",
        "dataset",
        "split",
        "selection_protocol",
        "distribution_receipt",
        "inputs",
        "metric_assets",
        "counts",
        "body",
        "face",
        "rs",
        "runtime",
        "formal_mode",
        "test_only_mode",
        "report_payload_sha256",
    }:
        raise MetricAdapterContractError("metric report schema mismatch")
    unsigned = dict(report)
    claimed_report_sha = _require_sha256(
        unsigned.pop("report_payload_sha256"),
        "metric report payload SHA",
    )
    if (
        canonical_json_sha256(unsigned) != claimed_report_sha
        or report["report_payload_hash_algorithm"]
        != REPORT_PAYLOAD_HASH_ALGORITHM
    ):
        raise MetricAdapterContractError(
            "metric report payload hash mismatch"
        )
    if (
        report["format"] != REPORT_FORMAT
        or report["status"] != "complete"
        or report["generator"] != "SemTalk Base-only"
        or report["split"] != expected_split
        or report["selection_protocol"] != frozen_selection_protocol
    ):
        raise MetricAdapterContractError(
            "metric report identity/selection protocol mismatch"
        )
    if fixture_mode:
        if (
            report["formal_mode"] is not False
            or report["test_only_mode"] is not True
        ):
            raise MetricAdapterContractError(
                "four-clip report is not explicit CPU fixture output"
            )
    elif (
        report["formal_mode"] is not True
        or report["test_only_mode"] is not False
    ):
        raise MetricAdapterContractError(
            "formal report identity/selection protocol mismatch"
        )
    if report["dataset"] != {
        "name": "SHOW",
        "release": "ReleaseV1.0",
        "selection": ">2s",
        "speakers": dict(SHOW_SPEAKER_IDS),
    }:
        raise MetricAdapterContractError(
            "metric report SHOW dataset contract mismatch"
        )
    distribution = report["distribution_receipt"]
    if (
        distribution != dict(expected_distribution_receipt)
        or distribution.get("format")
        != DISTRIBUTION_RECEIPT_FORMAT
        or distribution.get("payload_hash_algorithm")
        != PAYLOAD_HASH_ALGORITHM
        or distribution.get("protocol") != DISTRIBUTION_PROTOCOL
        or distribution.get("physical_samples_per_clip") != 1
        or distribution.get("independent_samples") is not False
        or distribution.get("deterministic_delta_distribution") is not True
        or distribution.get("replication_algorithm")
        != REPLICATION_ALGORITHM
        or distribution.get("seed_consumed") is not False
        or distribution.get("released2_slots")
        != list(RELEASED2_SLOTS)
        or distribution.get("paper16_slots") != list(PAPER16_SLOTS)
        or distribution.get("face_slot") != FACE_SLOT
        or distribution.get("variation_policy")
        != _replication_gate_module().variation_policy_receipt()
        or distribution.get("metric_input_materialization")
        != METRIC_INPUT_MATERIALIZATION
    ):
        raise MetricAdapterContractError(
            "metric report deterministic distribution receipt mismatch"
        )
    distribution_unsigned = dict(distribution)
    distribution_sha = _require_sha256(
        distribution_unsigned.pop("receipt_payload_sha256"),
        "distribution receipt payload SHA",
    )
    if (
        compact_canonical_json_sha256(distribution_unsigned)
        != distribution_sha
    ):
        raise MetricAdapterContractError(
            "distribution receipt payload hash mismatch"
        )
    inputs = report["inputs"]
    if type(inputs) is not dict or set(inputs) != {
        "canonical_manifest",
        "prediction_manifest",
        "prediction_lineage",
        "test_authority",
    }:
        raise MetricAdapterContractError("metric report inputs schema mismatch")
    if expected_split == "val":
        if (
            expected_test_authority is not None
            or inputs["test_authority"] is not None
        ):
            raise MetricAdapterContractError(
                "validation report must not contain test authority"
            )
    else:
        if (
            type(expected_test_authority) is not dict
            or inputs["test_authority"]
            != dict(expected_test_authority)
            or _validated_test_authority(
                expected_test_authority
            )["receipt_payload_sha256"]
            != expected_test_authority["receipt_payload_sha256"]
        ):
            raise MetricAdapterContractError(
                "test report authority binding mismatch"
            )
    prediction_input = inputs["prediction_manifest"]
    if (
        type(prediction_input) is not dict
        or set(prediction_input) != {
            "path",
            "sha256",
            "bytes",
            "rows",
        }
        or {
            key: prediction_input[key]
            for key in ("path", "sha256", "bytes")
        }
        != expected_manifest
        or distribution.get("prediction_manifest") != expected_manifest
        or prediction_input["rows"] != expected_clip_count
    ):
        raise MetricAdapterContractError(
            "metric report prediction-manifest binding mismatch"
        )
    for role, expected_keys in (
        (
            "canonical_manifest",
            {"path", "sha256", "bytes", "rows", "selected_rows"},
        ),
        (
            "prediction_manifest",
            {"path", "sha256", "bytes", "rows"},
        ),
        (
            "prediction_lineage",
            {"path", "sha256", "bytes", "payload_sha256"},
        ),
    ):
        artifact = inputs[role]
        if type(artifact) is not dict or set(artifact) != expected_keys:
            raise MetricAdapterContractError(
                f"metric report {role} schema mismatch"
            )
        _path, payload = _verified_file_snapshot(
            artifact["path"],
            _require_sha256(
                artifact["sha256"],
                f"metric report {role} SHA",
            ),
            f"metric report {role}",
        )
        if artifact["bytes"] != len(payload):
            raise MetricAdapterContractError(
                f"metric report {role} byte count mismatch"
            )
    canonical_input = inputs["canonical_manifest"]
    if (
        canonical_input["selected_rows"] != expected_clip_count
        or canonical_input["rows"] < expected_clip_count
    ):
        raise MetricAdapterContractError(
            "metric report canonical-manifest counts mismatch"
        )
    assets = report["metric_assets"]
    runtime = report["runtime"]
    _validate_backend_receipts(
        SimpleNamespace(
            asset_receipt=assets,
            runtime_receipt=runtime,
        ),
        require_cuda=not fixture_mode,
    )
    if not fixture_mode:
        talkshow = assets["talkshow"]
        if (
            not isinstance(talkshow, dict)
            or "path" not in talkshow
            or validate_talkshow_metric_root(talkshow["path"]) != talkshow
        ):
            raise MetricAdapterContractError(
                "metric report TalkSHOW source closure changed"
            )
        for role, expected_sha, expected_dtype in (
            ("feature_extractor", FEATURE_EXTRACTOR_SHA256, "float32"),
            ("smplx", SMPLX_SHA256, "float64"),
        ):
            artifact = assets[role]
            if (
                type(artifact) is not dict
                or set(artifact)
                != {"path", "sha256", "bytes", "runtime_dtype"}
                or artifact["runtime_dtype"] != expected_dtype
            ):
                raise MetricAdapterContractError(
                    f"metric report {role} schema/dtype mismatch"
                )
            _path, payload = _verified_file_snapshot(
                artifact["path"],
                expected_sha,
                f"metric report {role}",
            )
            if (
                artifact["sha256"] != expected_sha
                or artifact["bytes"] != len(payload)
            ):
                raise MetricAdapterContractError(
                    f"metric report {role} artifact changed"
                )
    counts = report["counts"]
    if type(counts) is not dict or set(counts) != {
        "clips",
        "frames",
        "physical_prediction_artifacts",
        "logical_samples_per_clip",
        "per_speaker",
        "all_tensors_finite",
        "exact_once",
    }:
        raise MetricAdapterContractError("metric report counts schema mismatch")
    per_speaker = counts["per_speaker"]
    if (
        counts["clips"] != expected_clip_count
        or counts["physical_prediction_artifacts"] != expected_clip_count
        or counts["logical_samples_per_clip"] != NUM_LOGICAL_SLOTS
        or counts["all_tensors_finite"] is not True
        or counts["exact_once"] is not True
        or type(per_speaker) is not dict
        or set(per_speaker) != set(SPEAKER_NAMES)
        or any(type(value) is not int or value < 1 for value in per_speaker.values())
        or sum(per_speaker.values()) != expected_clip_count
        or type(counts["frames"]) is not int
        or counts["frames"]
        < expected_clip_count * (FORMAL_MIN_FRAMES_EXCLUSIVE + 1)
    ):
        raise MetricAdapterContractError(
            "metric report exact-once counts mismatch"
        )
    body = report["body"]
    if type(body) is not dict or set(body) != set(PROTOCOLS):
        raise MetricAdapterContractError("metric report body schema mismatch")
    for protocol, definition in PROTOCOLS.items():
        value = body[protocol]
        if type(value) is not dict or set(value) != {
            "protocol",
            "logical_slots",
            "num_samples",
            "bc_sample_policy",
            "counts",
            "metrics",
            "feature_statistics",
            "primitive_receipt",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} schema mismatch"
            )
        samples = int(definition["num_samples"])
        if (
            value["protocol"] != protocol
            or value["logical_slots"] != definition["logical_slots"]
            or value["num_samples"] != samples
            or value["bc_sample_policy"] != definition["bc_sample_policy"]
        ):
            raise MetricAdapterContractError(
                f"metric report {protocol} protocol mismatch"
            )
        protocol_counts = value["counts"]
        if type(protocol_counts) is not dict or set(protocol_counts) != {
            "clips",
            "per_speaker",
            "real_features",
            "generated_features",
            "bc_sample_evaluations",
            "bc_denominator",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} counts schema mismatch"
            )
        if (
            protocol_counts["clips"] != expected_clip_count
            or protocol_counts["per_speaker"] != per_speaker
            or type(protocol_counts["real_features"]) is not int
            or protocol_counts["real_features"] < 2
            or protocol_counts["generated_features"]
            != protocol_counts["real_features"] * samples
            or protocol_counts["bc_sample_evaluations"]
            != expected_clip_count * (1 if protocol == "released2" else 16)
            or type(protocol_counts["bc_denominator"]) is not int
            or protocol_counts["bc_denominator"] < 1
        ):
            raise MetricAdapterContractError(
                f"metric report {protocol} counts mismatch"
            )
        metrics = value["metrics"]
        if type(metrics) is not dict or set(metrics) != {
            "FGD",
            "Variation",
            "BC",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} metrics schema mismatch"
            )
        fgd = _report_number(
            metrics["FGD"],
            f"{protocol} FGD",
            nonnegative=True,
        )
        variation = _report_number(
            metrics["Variation"],
            f"{protocol} Variation",
            nonnegative=True,
        )
        bc = _report_number(
            metrics["BC"],
            f"{protocol} BC",
            nonnegative=True,
        )
        if bc > 1.0 or not math.isfinite(fgd):
            raise MetricAdapterContractError(
                f"metric report {protocol} metric contract mismatch"
            )
        primitive = value["primitive_receipt"]
        if type(primitive) is not dict or set(primitive) != {
            "variation_sum",
            "variation_integrity_tolerance_sum",
            "bc_numerator",
            "bc_denominator",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} primitive receipt mismatch"
            )
        variation_sum = _report_number(
            primitive["variation_sum"],
            f"{protocol} variation sum",
            nonnegative=True,
        )
        variation_tolerance_sum = _report_number(
            primitive["variation_integrity_tolerance_sum"],
            f"{protocol} variation tolerance sum",
            nonnegative=True,
        )
        bc_numerator = _report_number(
            primitive["bc_numerator"],
            f"{protocol} BC numerator",
            nonnegative=True,
        )
        bc_denominator = _require_exact_int(
            primitive["bc_denominator"],
            f"{protocol} BC denominator",
            minimum=1,
        )
        if (
            bc_denominator != protocol_counts["bc_denominator"]
            or variation_sum > variation_tolerance_sum
            or not math.isclose(
                variation,
                variation_sum / expected_clip_count,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                bc,
                bc_numerator / bc_denominator,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise MetricAdapterContractError(
                f"metric report {protocol} primitive recomputation mismatch"
            )
        statistics = value["feature_statistics"]
        if type(statistics) is not dict or set(statistics) != {
            "real",
            "generated",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} statistics schema mismatch"
            )
        real_moments = _validate_report_feature_statistics(
            statistics["real"],
            expected_count=protocol_counts["real_features"],
            label=f"{protocol} real",
        )
        generated_moments = _validate_report_feature_statistics(
            statistics["generated"],
            expected_count=protocol_counts["generated_features"],
            label=f"{protocol} generated",
        )
        recomputed_fgd = frechet_distance(
            real_moments,
            generated_moments,
        )
        if not math.isclose(
            fgd,
            recomputed_fgd,
            rel_tol=1e-13,
            abs_tol=1e-13 * max(1.0, abs(fgd), abs(recomputed_fgd)),
        ):
            raise MetricAdapterContractError(
                f"metric report {protocol} FGD does not match its moments"
            )
    face = report["face"]
    if (
        type(face) is not dict
        or set(face)
        != {
            "protocol",
            "logical_slot",
            "aggregation",
            "counts",
            "metrics",
        }
        or face["protocol"] != "released_face_sample0"
        or face["logical_slot"] != FACE_SLOT
        or face["aggregation"] != "equal_clip_mean"
        or face["counts"]
        != {"clips": expected_clip_count, "per_speaker": per_speaker}
        or type(face["metrics"]) is not dict
        or set(face["metrics"]) != {"released", "derived"}
        or type(face["metrics"]["released"]) is not dict
        or set(face["metrics"]["released"])
        != {"jaw_l1", "landmark_l1", "LVD"}
        or type(face["metrics"]["derived"]) is not dict
        or set(face["metrics"]["derived"]) != {"face_l2_combined"}
    ):
        raise MetricAdapterContractError("metric report face schema mismatch")
    released_face = face["metrics"]["released"]
    jaw = _report_number(released_face["jaw_l1"], "face jaw_l1", nonnegative=True)
    landmark = _report_number(
        released_face["landmark_l1"],
        "face landmark_l1",
        nonnegative=True,
    )
    _report_number(released_face["LVD"], "face LVD", nonnegative=True)
    combined = _report_number(
        face["metrics"]["derived"]["face_l2_combined"],
        "face combined",
        nonnegative=True,
    )
    if not math.isclose(
        combined,
        jaw + landmark,
        rel_tol=0.0,
        abs_tol=1e-12 * max(1.0, combined, jaw + landmark),
    ):
        raise MetricAdapterContractError(
            "metric report derived face combined mismatch"
        )
    if report["rs"] != {"status": "N/A/unreleased", "value": None}:
        raise MetricAdapterContractError("metric report RS contract mismatch")
    return {
        "status": "pass",
        "split": expected_split,
        "clips": expected_clip_count,
        "primary_metric_path": PRIMARY_METRIC_PATH,
        "primary_metric": float(
            body["released2"]["metrics"]["FGD"]
        ),
        "report_payload_sha256": claimed_report_sha,
    }


def _payload_json_artifact(
    value: Mapping[str, Any],
    *,
    label: str,
    compact_payload_hash: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(value) is not dict or set(value) != {
        "path",
        "sha256",
        "bytes",
        "receipt_payload_sha256",
    }:
        raise MetricAdapterContractError(
            f"{label} payload-artifact schema mismatch"
        )
    path, payload = _verified_file_snapshot(
        value["path"],
        _require_sha256(value["sha256"], f"{label} file SHA"),
        label,
    )
    if (
        _require_exact_int(value["bytes"], f"{label} bytes", minimum=1)
        != len(payload)
    ):
        raise MetricAdapterContractError(f"{label} byte count mismatch")
    decoded = _strict_json_snapshot(payload, label)
    if payload != canonical_json_bytes(decoded):
        raise MetricAdapterContractError(
            f"{label} is not canonical JSON"
        )
    unsigned = dict(decoded)
    payload_sha = _require_sha256(
        unsigned.pop("receipt_payload_sha256", None),
        f"{label} payload SHA",
    )
    payload_hasher = (
        compact_canonical_json_sha256
        if compact_payload_hash
        else canonical_json_sha256
    )
    if (
        payload_sha != value["receipt_payload_sha256"]
        or payload_hasher(unsigned) != payload_sha
    ):
        raise MetricAdapterContractError(
            f"{label} payload hash mismatch"
        )
    return dict(value), decoded


def _validate_primary_metric_assets(
    assets: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    fixture_mode: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_assets, normalized_runtime = _validate_backend_receipts(
        SimpleNamespace(
            asset_receipt=assets,
            runtime_receipt=runtime,
        ),
        require_cuda=not fixture_mode,
    )
    if fixture_mode:
        return normalized_assets, normalized_runtime
    talkshow = normalized_assets["talkshow"]
    if (
        type(talkshow) is not dict
        or "path" not in talkshow
        or validate_talkshow_metric_root(talkshow["path"]) != talkshow
    ):
        raise MetricAdapterContractError(
            "primary replay TalkSHOW source closure changed"
        )
    feature = normalized_assets["feature_extractor"]
    if (
        type(feature) is not dict
        or set(feature)
        != {"path", "sha256", "bytes", "runtime_dtype"}
        or feature["runtime_dtype"] != "float32"
        or feature["sha256"] != FEATURE_EXTRACTOR_SHA256
    ):
        raise MetricAdapterContractError(
            "primary replay feature-extractor receipt mismatch"
        )
    _path, payload = _verified_file_snapshot(
        feature["path"],
        FEATURE_EXTRACTOR_SHA256,
        "primary replay feature extractor",
    )
    if feature["bytes"] != len(payload):
        raise MetricAdapterContractError(
            "primary replay feature-extractor byte count mismatch"
        )
    return normalized_assets, normalized_runtime


def _metric_runtime_signature(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in runtime.items()
        if key not in {"device", "device_index", "device_name"}
    }


def _validate_real_feature_cache_production_authority(
    value: Any,
    *,
    canonical_manifest: Mapping[str, Any],
    split: str,
    clip_count: int,
) -> dict[str, Any]:
    """Validate the immutable producer/input binding of a formal cache."""

    if type(value) is not dict or set(value) != {
        "format",
        "status",
        "source",
        "validation_inputs",
        "receipt_payload_sha256",
    }:
        raise MetricAdapterContractError(
            "released2 real-feature cache production authority schema mismatch"
        )
    claimed = _require_sha256(
        value["receipt_payload_sha256"],
        "released2 real-feature cache production authority payload",
    )
    authority_body = dict(value)
    authority_body.pop("receipt_payload_sha256")
    if compact_canonical_json_sha256(authority_body) != claimed:
        raise MetricAdapterContractError(
            "released2 real-feature cache production authority payload mismatch"
        )
    if (
        value["format"]
        != PRIMARY_REAL_FEATURE_CACHE_PRODUCTION_AUTHORITY_FORMAT
        or value["status"] != "frozen"
    ):
        raise MetricAdapterContractError(
            "released2 real-feature cache production authority mismatch"
        )

    source = value["source"]
    if type(source) is not dict or set(source) != {
        "origin",
        "source_root",
        "commit",
        "tree",
        "clean",
        "detached",
        "local_branches_at_commit",
        "entrypoint",
    }:
        raise MetricAdapterContractError(
            "released2 real-feature cache producer source schema mismatch"
        )
    source_root = source["source_root"]
    if (
        source["origin"] != SEMTALK_OFFICIAL_ORIGIN
        or type(source_root) is not str
        or not Path(source_root).is_absolute()
        or re.fullmatch(r"[0-9a-f]{40}", str(source["commit"])) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(source["tree"])) is None
        or source["clean"] is not True
        or source["detached"] is not True
        or source["local_branches_at_commit"] != []
    ):
        raise MetricAdapterContractError(
            "released2 real-feature cache producer source mismatch"
        )
    entrypoint = source["entrypoint"]
    if type(entrypoint) is not dict or set(entrypoint) != {
        "path",
        "relative",
        "sha256",
        "bytes",
        "git_mode",
        "git_blob_sha1",
    }:
        raise MetricAdapterContractError(
            "released2 real-feature cache producer entrypoint schema mismatch"
        )
    entrypoint_path = entrypoint["path"]
    if (
        entrypoint["relative"] != PRIMARY_REAL_FEATURE_CACHE_ENTRYPOINT
        or type(entrypoint_path) is not str
        or not Path(entrypoint_path).is_absolute()
        or Path(entrypoint_path)
        != Path(source_root) / PRIMARY_REAL_FEATURE_CACHE_ENTRYPOINT
        or _require_exact_int(
            entrypoint["bytes"],
            "released2 real-feature cache producer entrypoint bytes",
            minimum=1,
        )
        != entrypoint["bytes"]
        or entrypoint["git_mode"] not in {"100644", "100755"}
        or re.fullmatch(
            r"[0-9a-f]{40}", str(entrypoint["git_blob_sha1"])
        )
        is None
    ):
        raise MetricAdapterContractError(
            "released2 real-feature cache producer entrypoint mismatch"
        )
    _require_sha256(
        entrypoint["sha256"],
        "released2 real-feature cache producer entrypoint",
    )

    inputs = value["validation_inputs"]
    input_keys = {
        "format",
        "status",
        "split",
        "test_visible",
        "expected_clip_count",
        "canonical_manifest",
        "canonical_summary",
        "canonical_lineage",
        "audio_manifests",
        "audio_summaries",
        "audio_lineages",
        "clip_ids_sha256",
        "talkshow_window_manifest_sha256",
        "receipt_payload_sha256",
    }
    if type(inputs) is not dict or set(inputs) != input_keys:
        raise MetricAdapterContractError(
            "released2 real-feature cache validation-input authority schema "
            "mismatch"
        )
    input_claimed = _require_sha256(
        inputs["receipt_payload_sha256"],
        "released2 real-feature cache validation-input payload",
    )
    input_body = dict(inputs)
    input_body.pop("receipt_payload_sha256")
    if compact_canonical_json_sha256(input_body) != input_claimed:
        raise MetricAdapterContractError(
            "released2 real-feature cache validation-input payload mismatch"
        )
    if (
        inputs["format"] != "semtalk_show_base_talkshow_val_inputs_v2"
        or inputs["status"] != "frozen"
        or inputs["split"] != split
        or inputs["test_visible"] is not False
        or inputs["expected_clip_count"] != clip_count
    ):
        raise MetricAdapterContractError(
            "released2 real-feature cache validation-input authority mismatch"
        )

    def compact_artifact(item: Any, label: str) -> dict[str, str]:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise MetricAdapterContractError(f"{label} schema mismatch")
        path = item["path"]
        if type(path) is not str or not Path(path).is_absolute():
            raise MetricAdapterContractError(f"{label} path mismatch")
        return {
            "path": path,
            "sha256": _require_sha256(item["sha256"], f"{label} SHA-256"),
        }

    canonical = compact_artifact(
        inputs["canonical_manifest"],
        "released2 real-feature cache canonical authority",
    )
    if canonical != {
        "path": canonical_manifest.get("path"),
        "sha256": canonical_manifest.get("sha256"),
    }:
        raise MetricAdapterContractError(
            "released2 real-feature cache canonical authority mismatch"
        )
    compact_artifact(
        inputs["canonical_summary"],
        "released2 real-feature cache canonical summary authority",
    )
    compact_artifact(
        inputs["canonical_lineage"],
        "released2 real-feature cache canonical lineage authority",
    )
    for key in ("audio_manifests", "audio_summaries", "audio_lineages"):
        receipts = inputs[key]
        if type(receipts) is not list or len(receipts) != 8:
            raise MetricAdapterContractError(
                f"released2 real-feature cache {key} must bind eight shards"
            )
        normalized = [
            compact_artifact(
                item,
                f"released2 real-feature cache {key}[{index}]",
            )
            for index, item in enumerate(receipts)
        ]
        if len({item["path"] for item in normalized}) != 8:
            raise MetricAdapterContractError(
                f"released2 real-feature cache {key} paths are not unique"
            )
    _require_sha256(
        inputs["clip_ids_sha256"],
        "released2 real-feature cache validation clip IDs",
    )
    _require_sha256(
        inputs["talkshow_window_manifest_sha256"],
        "released2 real-feature cache TalkSHOW window manifest",
    )
    return dict(value)


def build_released2_real_feature_cache(
    *,
    canonical_manifest: str | Path,
    expected_canonical_manifest_sha256: str,
    backend: MetricBackend,
    split: str,
    expected_clip_count: int,
    formal_mode: bool = True,
    test_only_allow_four_clip_subset: bool = False,
    production_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freshly compute the candidate-independent released2 real moments."""

    if split != "val":
        raise MetricAdapterContractError(
            "released2 primary cache is validation-selection-only"
        )
    fixture_mode = (
        not formal_mode
        and test_only_allow_four_clip_subset
        and expected_clip_count == 4
    )
    if formal_mode:
        _require_concrete_formal_backend(backend)
    assets, runtime = _validate_primary_metric_assets(
        backend.asset_receipt,
        backend.runtime_receipt,
        fixture_mode=fixture_mode,
    )
    canonical_path, canonical_payload = _verified_file_snapshot(
        canonical_manifest,
        expected_canonical_manifest_sha256,
        "primary-cache canonical manifest",
    )
    canonical_rows = _strict_jsonl_snapshot(
        canonical_payload,
        "primary-cache canonical manifest",
    )
    canonical_by_id = _validate_canonical_rows(
        canonical_rows,
        split=split,
        expected_clip_count=expected_clip_count,
        formal_mode=formal_mode,
        test_only_allow_four_clip_subset=(
            test_only_allow_four_clip_subset
        ),
    )
    canonical_receipt = {
        "path": str(canonical_path),
        "sha256": sha256_bytes(canonical_payload),
        "bytes": len(canonical_payload),
        "rows": len(canonical_rows),
        "selected_rows": len(canonical_by_id),
    }
    normalized_production_authority: dict[str, Any] | None = None
    if formal_mode:
        normalized_production_authority = (
            _validate_real_feature_cache_production_authority(
                production_authority,
                canonical_manifest=canonical_receipt,
                split=split,
                clip_count=expected_clip_count,
            )
        )
    elif production_authority is not None:
        raise MetricAdapterContractError(
            "fixture real-feature cache cannot claim production authority"
        )
    moments = FeatureMoments()
    for output_id, row in canonical_by_id.items():
        arrays = _load_canonical_npz(
            row["canonical_npz"],
            expected_sha256=row["canonical_npz_sha256"],
            frames=row["frames"],
            speaker_id=row["speaker_id"],
        )
        features = _validated_backend_features(
            backend,
            reorder_to_talkshow(
                arrays["pose"],
                arrays["facial"],
            )[None],
        )
        moments.update(features)
    if moments.count < 2:
        raise MetricAdapterContractError(
            "primary real-feature cache is empty or singular"
        )
    result: dict[str, Any] = {
        "format": PRIMARY_REAL_FEATURE_CACHE_FORMAT,
        "status": "complete",
        "split": split,
        "clip_count": expected_clip_count,
        "canonical_manifest": canonical_receipt,
        "metric_assets": assets,
        "runtime": runtime,
        "real_feature_statistics": moments.to_json(),
        "formal_mode": formal_mode,
        "test_only_mode": fixture_mode,
    }
    if formal_mode:
        assert normalized_production_authority is not None
        result["production_authority"] = normalized_production_authority
    result["receipt_payload_sha256"] = canonical_json_sha256(result)
    return result


def _validate_released2_real_feature_cache(
    cache: Mapping[str, Any],
    *,
    expected_artifact: Mapping[str, Any],
    expected_canonical_manifest: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    fixture_mode: bool,
) -> tuple[dict[str, Any], FeatureMoments]:
    artifact, decoded = _payload_json_artifact(
        expected_artifact,
        label="released2 real-feature cache",
    )
    if decoded != cache:
        raise MetricAdapterContractError(
            "released2 real-feature cache object/file mismatch"
        )
    expected_keys = {
        "format",
        "status",
        "split",
        "clip_count",
        "canonical_manifest",
        "metric_assets",
        "runtime",
        "real_feature_statistics",
        "formal_mode",
        "test_only_mode",
        "receipt_payload_sha256",
    }
    if not fixture_mode:
        expected_keys.add("production_authority")
    if type(decoded) is not dict or set(decoded) != expected_keys:
        raise MetricAdapterContractError(
            "released2 real-feature cache schema mismatch"
        )
    if (
        decoded["format"] != PRIMARY_REAL_FEATURE_CACHE_FORMAT
        or decoded["status"] != "complete"
        or decoded["split"] != expected_split
        or decoded["clip_count"] != expected_clip_count
        or decoded["canonical_manifest"]
        != dict(expected_canonical_manifest)
        or decoded["formal_mode"] is not (not fixture_mode)
        or decoded["test_only_mode"] is not fixture_mode
    ):
        raise MetricAdapterContractError(
            "released2 real-feature cache authority mismatch"
        )
    _validate_primary_metric_assets(
        decoded["metric_assets"],
        decoded["runtime"],
        fixture_mode=fixture_mode,
    )
    if not fixture_mode:
        _validate_real_feature_cache_production_authority(
            decoded["production_authority"],
            canonical_manifest=decoded["canonical_manifest"],
            split=expected_split,
            clip_count=expected_clip_count,
        )
    statistics = decoded["real_feature_statistics"]
    if type(statistics) is not dict:
        raise MetricAdapterContractError(
            "released2 real-feature cache moments schema mismatch"
        )
    count = _require_exact_int(
        statistics.get("count"),
        "released2 real-feature cache count",
        minimum=2,
    )
    moments = FeatureMoments.from_json(
        statistics,
        expected_count=count,
        label="released2 real-feature cache",
    )
    return artifact, moments


def _fresh_primary_screen_inputs(
    *,
    canonical_manifest: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    distribution_receipt: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    fixture_mode: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    if expected_split != "val":
        raise MetricAdapterContractError(
            "released2 primary screen is validation-selection-only"
        )
    if type(canonical_manifest) is not dict or set(canonical_manifest) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise MetricAdapterContractError(
            "primary screen canonical artifact schema mismatch"
        )
    if type(prediction_manifest) is not dict or set(prediction_manifest) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise MetricAdapterContractError(
            "primary screen prediction artifact schema mismatch"
        )
    canonical_path, canonical_payload = _verified_file_snapshot(
        canonical_manifest["path"],
        canonical_manifest["sha256"],
        "primary screen canonical manifest",
    )
    prediction_path, prediction_payload = _verified_file_snapshot(
        prediction_manifest["path"],
        prediction_manifest["sha256"],
        "primary screen prediction manifest",
    )
    if (
        canonical_manifest["bytes"] != len(canonical_payload)
        or prediction_manifest["bytes"] != len(prediction_payload)
    ):
        raise MetricAdapterContractError(
            "primary screen manifest byte count mismatch"
        )
    canonical_rows = _strict_jsonl_snapshot(
        canonical_payload,
        "primary screen canonical manifest",
    )
    prediction_rows = _strict_jsonl_snapshot(
        prediction_payload,
        "primary screen prediction manifest",
    )
    canonical_by_id = _validate_canonical_rows(
        canonical_rows,
        split=expected_split,
        expected_clip_count=expected_clip_count,
        formal_mode=not fixture_mode,
        test_only_allow_four_clip_subset=fixture_mode,
    )
    ordered_predictions = _validate_prediction_rows(
        prediction_rows,
        canonical=canonical_by_id,
        split=expected_split,
    )
    distribution_artifact, distribution = _payload_json_artifact(
        distribution_receipt,
        label="primary screen distribution receipt",
        compact_payload_hash=True,
    )
    validated_distribution = validate_distribution_declaration(
        distribution,
        expected_gate_artifact=distribution["validation_gate"],
        expected_prediction_manifest={
            "path": str(prediction_path),
            "sha256": sha256_bytes(prediction_payload),
            "bytes": len(prediction_payload),
        },
        expected_prediction_records=[
            {
                "canonical_clip_id": row["canonical_clip_id"],
                "prediction_sha256": row["prediction"]["sha256"],
                "prediction_bytes": row["prediction"]["bytes"],
            }
            for row in ordered_predictions
        ],
    )
    if validated_distribution != distribution:
        raise MetricAdapterContractError(
            "primary screen distribution replay changed"
        )
    canonical_artifact = {
        "path": str(canonical_path),
        "sha256": sha256_bytes(canonical_payload),
        "bytes": len(canonical_payload),
        "rows": len(canonical_rows),
        "selected_rows": len(canonical_by_id),
    }
    prediction_artifact = {
        "path": str(prediction_path),
        "sha256": sha256_bytes(prediction_payload),
        "bytes": len(prediction_payload),
    }
    return (
        canonical_artifact,
        prediction_artifact,
        distribution_artifact,
        canonical_by_id,
        ordered_predictions,
    )


def build_released2_primary_screen(
    backend: MetricBackend,
    *,
    canonical_manifest: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    distribution_receipt: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
    expected_real_feature_cache_artifact: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Compute only the released2 moments/FGD required for selection."""

    fixture_mode = (
        test_only_allow_four_clip_subset and expected_clip_count == 4
    )
    if not fixture_mode:
        _require_concrete_formal_backend(backend)
    selection_protocol = {
        "primary_metric": PRIMARY_METRIC_PATH,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 0,
    }
    if dict(expected_selection_protocol) != selection_protocol:
        raise MetricAdapterContractError(
            "primary screen selection protocol changed"
        )
    (
        canonical_artifact,
        prediction_artifact,
        distribution_artifact,
        canonical_by_id,
        ordered_predictions,
    ) = _fresh_primary_screen_inputs(
        canonical_manifest=canonical_manifest,
        prediction_manifest=prediction_manifest,
        distribution_receipt=distribution_receipt,
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        fixture_mode=fixture_mode,
    )
    assets, runtime = _validate_primary_metric_assets(
        backend.asset_receipt,
        backend.runtime_receipt,
        fixture_mode=fixture_mode,
    )
    cache_artifact, real_moments = _validate_released2_real_feature_cache(
        real_feature_cache,
        expected_artifact=expected_real_feature_cache_artifact,
        expected_canonical_manifest=canonical_artifact,
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        fixture_mode=fixture_mode,
    )
    if any(
        real_feature_cache["metric_assets"][role] != assets[role]
        for role in ("talkshow", "feature_extractor", "smplx")
    ) or _metric_runtime_signature(
        real_feature_cache["runtime"]
    ) != _metric_runtime_signature(runtime):
        raise MetricAdapterContractError(
            "primary screen cache used different metric assets/runtime"
        )
    generated_moments = FeatureMoments()
    for row in ordered_predictions:
        output_id = row["canonical_clip_id"]
        canonical_row = canonical_by_id[output_id]
        arrays = _load_output_npz(
            row["prediction"],
            frames=canonical_row["frames"],
            prediction=True,
            expected_name=f"res_{output_id}.npz",
        )
        canonical_arrays = _load_canonical_npz(
            canonical_row["canonical_npz"],
            expected_sha256=canonical_row["canonical_npz_sha256"],
            frames=canonical_row["frames"],
            speaker_id=canonical_row["speaker_id"],
        )
        if not np.array_equal(arrays["betas"], canonical_arrays["beta"][0]):
            raise MetricAdapterContractError(
                f"{output_id}: primary screen prediction betas changed"
            )
        features = _validated_backend_features(
            backend,
            np.repeat(
                reorder_to_talkshow(
                    arrays["poses"], arrays["expressions"]
                )[None],
                len(RELEASED2_SLOTS),
                axis=0,
            ),
        )
        generated_moments.update(features)
    if generated_moments.count != real_moments.count * len(RELEASED2_SLOTS):
        raise MetricAdapterContractError(
            "primary screen real/generated feature counts disagree"
        )
    primary = frechet_distance(real_moments, generated_moments)
    result: dict[str, Any] = {
        "format": PRIMARY_SCREEN_FORMAT,
        "status": "pass",
        "split": expected_split,
        "test_visible": False,
        "clip_count": expected_clip_count,
        "primary_metric_path": PRIMARY_METRIC_PATH,
        "primary_metric": primary,
        "canonical_manifest": canonical_artifact,
        "prediction_manifest": prediction_artifact,
        "distribution_receipt": distribution_artifact,
        "selection_protocol": selection_protocol,
        "real_feature_cache": cache_artifact,
        "metric_assets": assets,
        "runtime": runtime,
        "real_feature_statistics": real_moments.to_json(),
        "generated_feature_statistics": generated_moments.to_json(),
        "formal_mode": not fixture_mode,
        "test_only_mode": fixture_mode,
    }
    result["receipt_payload_sha256"] = canonical_json_sha256(result)
    return result


def validate_released2_primary_screen_receipt(
    value: Mapping[str, Any],
    *,
    expected_prediction_manifest: Mapping[str, Any],
    expected_distribution_receipt: Mapping[str, Any],
    expected_real_feature_cache: Mapping[str, Any],
    expected_canonical_manifest: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Purely replay one externally byte-pinned primary screen receipt."""

    artifact, receipt = _payload_json_artifact(
        value, label="released2 primary screen receipt"
    )
    fixture_mode = (
        test_only_allow_four_clip_subset and expected_clip_count == 4
    )
    selection_protocol = {
        "primary_metric": PRIMARY_METRIC_PATH,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 0,
    }
    if dict(expected_selection_protocol) != selection_protocol:
        raise MetricAdapterContractError(
            "primary screen expected selection protocol changed"
        )
    expected_keys = {
        "format",
        "status",
        "split",
        "test_visible",
        "clip_count",
        "primary_metric_path",
        "primary_metric",
        "canonical_manifest",
        "prediction_manifest",
        "distribution_receipt",
        "selection_protocol",
        "real_feature_cache",
        "metric_assets",
        "runtime",
        "real_feature_statistics",
        "generated_feature_statistics",
        "formal_mode",
        "test_only_mode",
        "receipt_payload_sha256",
    }
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise MetricAdapterContractError(
            "released2 primary screen receipt schema mismatch"
        )
    if (
        receipt["format"] != PRIMARY_SCREEN_FORMAT
        or receipt["status"] != "pass"
        or receipt["split"] != expected_split
        or receipt["test_visible"] is not False
        or receipt["clip_count"] != expected_clip_count
        or receipt["primary_metric_path"] != PRIMARY_METRIC_PATH
        or receipt["canonical_manifest"] != dict(expected_canonical_manifest)
        or receipt["prediction_manifest"] != dict(expected_prediction_manifest)
        or receipt["distribution_receipt"]
        != dict(expected_distribution_receipt)
        or receipt["selection_protocol"] != selection_protocol
        or receipt["real_feature_cache"]
        != dict(expected_real_feature_cache)
        or receipt["formal_mode"] is not (not fixture_mode)
        or receipt["test_only_mode"] is not fixture_mode
    ):
        raise MetricAdapterContractError(
            "released2 primary screen authority mismatch"
        )
    cache_artifact, cache_payload = _payload_json_artifact(
        expected_real_feature_cache,
        label="primary screen bound real-feature cache",
    )
    _cache, real_moments = _validate_released2_real_feature_cache(
        cache_payload,
        expected_artifact=cache_artifact,
        expected_canonical_manifest=expected_canonical_manifest,
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        fixture_mode=fixture_mode,
    )
    assets, runtime = _validate_primary_metric_assets(
        receipt["metric_assets"],
        receipt["runtime"],
        fixture_mode=fixture_mode,
    )
    if any(
        cache_payload["metric_assets"][role] != assets[role]
        for role in ("talkshow", "feature_extractor", "smplx")
    ) or _metric_runtime_signature(cache_payload["runtime"]) != (
        _metric_runtime_signature(runtime)
    ):
        raise MetricAdapterContractError(
            "primary screen cache assets/runtime differ"
        )
    if receipt["real_feature_statistics"] != real_moments.to_json():
        raise MetricAdapterContractError(
            "primary screen real moments differ from cache"
        )
    generated_statistics = receipt["generated_feature_statistics"]
    generated_count = _require_exact_int(
        generated_statistics.get("count")
        if type(generated_statistics) is dict
        else None,
        "primary screen generated count",
        minimum=2,
    )
    generated_moments = FeatureMoments.from_json(
        generated_statistics,
        expected_count=generated_count,
        label="primary screen generated",
    )
    if generated_count != real_moments.count * len(RELEASED2_SLOTS):
        raise MetricAdapterContractError(
            "primary screen feature counts changed"
        )
    recomputed = frechet_distance(real_moments, generated_moments)
    primary = _report_number(
        receipt["primary_metric"],
        "primary screen metric",
        nonnegative=True,
    )
    # Selection is discrete and can involve near-ties.  A tolerance here
    # would let a rehashed receipt perturb ordering while still passing.  The
    # producer serializes this exact recomputed binary64 value, so authority
    # requires exact equality and returns the recomputed value itself.
    if primary != recomputed:
        raise MetricAdapterContractError(
            "primary screen metric derivation mismatch"
        )
    return {
        "artifact": artifact,
        "primary_metric_path": PRIMARY_METRIC_PATH,
        "primary_metric": recomputed,
        "real_feature_statistics": real_moments.to_json(),
        "generated_feature_statistics": generated_moments.to_json(),
        "metric_assets": assets,
        "runtime": runtime,
    }


def fresh_replay_released2_primary(
    report: Mapping[str, Any],
    backend: MetricBackend,
    *,
    real_feature_cache: Mapping[str, Any],
    expected_real_feature_cache_artifact: Mapping[str, Any],
    expected_prediction_manifest: Mapping[str, Any],
    expected_distribution_receipt: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    expected_test_authority: Mapping[str, Any] | None = None,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Freshly replay only the released2 FGD used for Base selection."""

    fixture_mode = (
        test_only_allow_four_clip_subset
        and expected_clip_count == 4
    )
    if expected_split != "val" or expected_test_authority is not None:
        raise MetricAdapterContractError(
            "released2 primary replay is validation-selection-only"
        )
    if not fixture_mode:
        _require_concrete_formal_backend(backend)
    validated_report = validate_report(
        report,
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        expected_prediction_manifest=expected_prediction_manifest,
        expected_distribution_receipt=expected_distribution_receipt,
        expected_selection_protocol=expected_selection_protocol,
        expected_test_authority=expected_test_authority,
        test_only_allow_four_clip_subset=fixture_mode,
    )
    assets, runtime = _validate_primary_metric_assets(
        backend.asset_receipt,
        backend.runtime_receipt,
        fixture_mode=fixture_mode,
    )
    if (
        assets != report["metric_assets"]
        or runtime != report["runtime"]
    ):
        raise MetricAdapterContractError(
            "primary replay backend differs from report backend authority"
        )
    canonical_input = report["inputs"]["canonical_manifest"]
    cache_artifact, real_moments = (
        _validate_released2_real_feature_cache(
            real_feature_cache,
            expected_artifact=expected_real_feature_cache_artifact,
            expected_canonical_manifest=canonical_input,
            expected_split=expected_split,
            expected_clip_count=expected_clip_count,
            fixture_mode=fixture_mode,
        )
    )
    cache_assets = real_feature_cache["metric_assets"]
    if any(
        cache_assets[role] != assets[role]
        for role in ("talkshow", "feature_extractor", "smplx")
    ) or _metric_runtime_signature(
        real_feature_cache["runtime"]
    ) != _metric_runtime_signature(runtime):
        raise MetricAdapterContractError(
            "primary replay cache used different metric assets/runtime"
        )
    canonical_path, canonical_payload = _verified_file_snapshot(
        canonical_input["path"],
        canonical_input["sha256"],
        "primary replay canonical manifest",
    )
    prediction_path, prediction_payload = _verified_file_snapshot(
        expected_prediction_manifest["path"],
        expected_prediction_manifest["sha256"],
        "primary replay prediction manifest",
    )
    if (
        canonical_input["bytes"] != len(canonical_payload)
        or expected_prediction_manifest["bytes"] != len(prediction_payload)
    ):
        raise MetricAdapterContractError(
            "primary replay manifest byte count mismatch"
        )
    canonical_rows = _strict_jsonl_snapshot(
        canonical_payload,
        "primary replay canonical manifest",
    )
    prediction_rows = _strict_jsonl_snapshot(
        prediction_payload,
        "primary replay prediction manifest",
    )
    canonical_by_id = _validate_canonical_rows(
        canonical_rows,
        split=expected_split,
        expected_clip_count=expected_clip_count,
        formal_mode=not fixture_mode,
        test_only_allow_four_clip_subset=fixture_mode,
    )
    ordered_predictions = _validate_prediction_rows(
        prediction_rows,
        canonical=canonical_by_id,
        split=expected_split,
    )
    generated_moments = FeatureMoments()
    for row in ordered_predictions:
        output_id = row["canonical_clip_id"]
        canonical_row = canonical_by_id[output_id]
        arrays = _load_output_npz(
            row["prediction"],
            frames=canonical_row["frames"],
            prediction=True,
            expected_name=f"res_{output_id}.npz",
        )
        canonical_arrays = _load_canonical_npz(
            canonical_row["canonical_npz"],
            expected_sha256=canonical_row["canonical_npz_sha256"],
            frames=canonical_row["frames"],
            speaker_id=canonical_row["speaker_id"],
        )
        if not np.array_equal(
            arrays["betas"],
            canonical_arrays["beta"][0],
        ):
            raise MetricAdapterContractError(
                f"{output_id}: replay prediction betas changed"
            )
        features = _validated_backend_features(
            backend,
            np.repeat(
                reorder_to_talkshow(
                    arrays["poses"],
                    arrays["expressions"],
                )[None],
                len(RELEASED2_SLOTS),
                axis=0,
            ),
        )
        generated_moments.update(features)
    if generated_moments.count != real_moments.count * len(
        RELEASED2_SLOTS
    ):
        raise MetricAdapterContractError(
            "primary replay real/generated feature counts disagree"
        )
    fresh_fgd = frechet_distance(real_moments, generated_moments)
    report_fgd = float(validated_report["primary_metric"])
    if not math.isclose(
        fresh_fgd,
        report_fgd,
        rel_tol=1e-9,
        abs_tol=1e-9 * max(1.0, abs(fresh_fgd), abs(report_fgd)),
    ):
        raise MetricAdapterContractError(
            "fresh released2 FGD differs from the report"
        )
    result: dict[str, Any] = {
        "format": PRIMARY_REPLAY_FORMAT,
        "status": "pass",
        "split": expected_split,
        "clip_count": expected_clip_count,
        "primary_metric_path": PRIMARY_METRIC_PATH,
        "primary_metric": fresh_fgd,
        "report_primary_metric": report_fgd,
        "report_payload_sha256": validated_report[
            "report_payload_sha256"
        ],
        "canonical_manifest": dict(canonical_input),
        "prediction_manifest": {
            "path": str(prediction_path),
            "sha256": sha256_bytes(prediction_payload),
            "bytes": len(prediction_payload),
        },
        "distribution_receipt_payload_sha256": (
            expected_distribution_receipt["receipt_payload_sha256"]
        ),
        "selection_protocol": dict(expected_selection_protocol),
        "real_feature_cache": cache_artifact,
        "metric_assets": assets,
        "runtime": runtime,
        "real_feature_statistics": real_moments.to_json(),
        "generated_feature_statistics": generated_moments.to_json(),
        "formal_mode": not fixture_mode,
        "test_only_mode": fixture_mode,
    }
    result["receipt_payload_sha256"] = canonical_json_sha256(result)
    return result


def fresh_replay_released2_primary_screen(
    screen: Mapping[str, Any],
    screen_validation: Mapping[str, Any],
    backend: MetricBackend,
    *,
    screen_artifact: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
    expected_real_feature_cache_artifact: Mapping[str, Any],
    expected_prediction_manifest: Mapping[str, Any],
    expected_distribution_receipt: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Independently reopen raw predictions and replay a screen FGD.

    The first screen receipt is intentionally not a ranking authority on its
    own: its serialized moments can be self-consistent after tampering.  This
    second pass reopens every byte-pinned prediction NPZ and recomputes the
    generated TalkSHOW features before the selector is allowed to rank it.
    """

    fixture_mode = (
        test_only_allow_four_clip_subset and expected_clip_count == 4
    )
    if expected_split != "val":
        raise MetricAdapterContractError(
            "released2 primary screen replay is validation-only"
        )
    if not fixture_mode:
        _require_concrete_formal_backend(backend)
    if (
        screen.get("format") != PRIMARY_SCREEN_FORMAT
        or screen_validation.get("artifact") != dict(screen_artifact)
        or screen_validation.get("primary_metric_path")
        != PRIMARY_METRIC_PATH
        or screen.get("prediction_manifest")
        != dict(expected_prediction_manifest)
        or screen.get("distribution_receipt")
        != dict(expected_distribution_receipt)
        or screen.get("selection_protocol")
        != dict(expected_selection_protocol)
        or screen.get("split") != expected_split
        or screen.get("clip_count") != expected_clip_count
    ):
        raise MetricAdapterContractError(
            "released2 primary screen replay authority changed"
        )
    assets, runtime = _validate_primary_metric_assets(
        backend.asset_receipt,
        backend.runtime_receipt,
        fixture_mode=fixture_mode,
    )
    if assets != screen["metric_assets"] or runtime != screen["runtime"]:
        raise MetricAdapterContractError(
            "screen replay backend differs from the first raw pass"
        )
    canonical_input = screen["canonical_manifest"]
    cache_artifact, real_moments = _validate_released2_real_feature_cache(
        real_feature_cache,
        expected_artifact=expected_real_feature_cache_artifact,
        expected_canonical_manifest=canonical_input,
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        fixture_mode=fixture_mode,
    )
    cache_assets = real_feature_cache["metric_assets"]
    if any(
        cache_assets[role] != assets[role]
        for role in ("talkshow", "feature_extractor", "smplx")
    ) or _metric_runtime_signature(real_feature_cache["runtime"]) != (
        _metric_runtime_signature(runtime)
    ):
        raise MetricAdapterContractError(
            "screen replay cache used different metric assets/runtime"
        )
    canonical_path, canonical_payload = _verified_file_snapshot(
        canonical_input["path"],
        canonical_input["sha256"],
        "primary screen replay canonical manifest",
    )
    prediction_path, prediction_payload = _verified_file_snapshot(
        expected_prediction_manifest["path"],
        expected_prediction_manifest["sha256"],
        "primary screen replay prediction manifest",
    )
    if (
        canonical_input["bytes"] != len(canonical_payload)
        or expected_prediction_manifest["bytes"]
        != len(prediction_payload)
    ):
        raise MetricAdapterContractError(
            "primary screen replay manifest byte count mismatch"
        )
    canonical_rows = _strict_jsonl_snapshot(
        canonical_payload,
        "primary screen replay canonical manifest",
    )
    prediction_rows = _strict_jsonl_snapshot(
        prediction_payload,
        "primary screen replay prediction manifest",
    )
    canonical_by_id = _validate_canonical_rows(
        canonical_rows,
        split=expected_split,
        expected_clip_count=expected_clip_count,
        formal_mode=not fixture_mode,
        test_only_allow_four_clip_subset=fixture_mode,
    )
    ordered_predictions = _validate_prediction_rows(
        prediction_rows,
        canonical=canonical_by_id,
        split=expected_split,
    )
    generated_moments = FeatureMoments()
    for row in ordered_predictions:
        output_id = row["canonical_clip_id"]
        canonical_row = canonical_by_id[output_id]
        arrays = _load_output_npz(
            row["prediction"],
            frames=canonical_row["frames"],
            prediction=True,
            expected_name=f"res_{output_id}.npz",
        )
        canonical_arrays = _load_canonical_npz(
            canonical_row["canonical_npz"],
            expected_sha256=canonical_row["canonical_npz_sha256"],
            frames=canonical_row["frames"],
            speaker_id=canonical_row["speaker_id"],
        )
        if not np.array_equal(
            arrays["betas"], canonical_arrays["beta"][0]
        ):
            raise MetricAdapterContractError(
                f"{output_id}: screen replay prediction betas changed"
            )
        features = _validated_backend_features(
            backend,
            np.repeat(
                reorder_to_talkshow(
                    arrays["poses"], arrays["expressions"]
                )[None],
                len(RELEASED2_SLOTS),
                axis=0,
            ),
        )
        generated_moments.update(features)
    if generated_moments.count != real_moments.count * len(
        RELEASED2_SLOTS
    ):
        raise MetricAdapterContractError(
            "screen replay real/generated feature counts disagree"
        )
    fresh_fgd = frechet_distance(real_moments, generated_moments)
    screen_fgd = float(screen_validation["primary_metric"])
    if fresh_fgd != screen_fgd:
        raise MetricAdapterContractError(
            "independent raw screen replay differs from first-pass FGD"
        )
    result: dict[str, Any] = {
        "format": PRIMARY_REPLAY_FORMAT,
        "status": "pass",
        "split": expected_split,
        "clip_count": expected_clip_count,
        "primary_metric_path": PRIMARY_METRIC_PATH,
        "primary_metric": fresh_fgd,
        "report_primary_metric": screen_fgd,
        "report_payload_sha256": screen["receipt_payload_sha256"],
        "canonical_manifest": dict(canonical_input),
        "prediction_manifest": {
            "path": str(prediction_path),
            "sha256": sha256_bytes(prediction_payload),
            "bytes": len(prediction_payload),
        },
        "distribution_receipt_payload_sha256": (
            expected_distribution_receipt["receipt_payload_sha256"]
        ),
        "selection_protocol": dict(expected_selection_protocol),
        "real_feature_cache": cache_artifact,
        "metric_assets": assets,
        "runtime": runtime,
        "real_feature_statistics": real_moments.to_json(),
        "generated_feature_statistics": generated_moments.to_json(),
        "formal_mode": not fixture_mode,
        "test_only_mode": fixture_mode,
    }
    result["receipt_payload_sha256"] = canonical_json_sha256(result)
    return result


def validate_released2_primary_replay_receipt(
    value: Mapping[str, Any],
    *,
    expected_report: Mapping[str, Any],
    expected_prediction_manifest: Mapping[str, Any],
    expected_distribution_receipt: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Purely validate one externally byte-pinned fresh-replay receipt."""

    artifact, receipt = _payload_json_artifact(
        value,
        label="released2 primary replay receipt",
    )
    fixture_mode = (
        test_only_allow_four_clip_subset
        and expected_clip_count == 4
    )
    if expected_split != "val":
        raise MetricAdapterContractError(
            "released2 primary replay receipt is validation-selection-only"
        )
    validated_report = validate_report(
        expected_report,
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        expected_prediction_manifest=expected_prediction_manifest,
        expected_distribution_receipt=expected_distribution_receipt,
        expected_selection_protocol=expected_selection_protocol,
        test_only_allow_four_clip_subset=fixture_mode,
    )
    if type(receipt) is not dict or set(receipt) != {
        "format",
        "status",
        "split",
        "clip_count",
        "primary_metric_path",
        "primary_metric",
        "report_primary_metric",
        "report_payload_sha256",
        "canonical_manifest",
        "prediction_manifest",
        "distribution_receipt_payload_sha256",
        "selection_protocol",
        "real_feature_cache",
        "metric_assets",
        "runtime",
        "real_feature_statistics",
        "generated_feature_statistics",
        "formal_mode",
        "test_only_mode",
        "receipt_payload_sha256",
    }:
        raise MetricAdapterContractError(
            "released2 primary replay receipt schema mismatch"
        )
    if (
        receipt["format"] != PRIMARY_REPLAY_FORMAT
        or receipt["status"] != "pass"
        or receipt["split"] != expected_split
        or receipt["clip_count"] != expected_clip_count
        or receipt["primary_metric_path"] != PRIMARY_METRIC_PATH
        or receipt["report_payload_sha256"]
        != validated_report["report_payload_sha256"]
        or receipt["canonical_manifest"]
        != expected_report["inputs"]["canonical_manifest"]
        or receipt["prediction_manifest"]
        != dict(expected_prediction_manifest)
        or receipt["distribution_receipt_payload_sha256"]
        != expected_distribution_receipt["receipt_payload_sha256"]
        or receipt["selection_protocol"]
        != dict(expected_selection_protocol)
        or receipt["metric_assets"] != expected_report["metric_assets"]
        or receipt["runtime"] != expected_report["runtime"]
        or receipt["formal_mode"] is not (not fixture_mode)
        or receipt["test_only_mode"] is not fixture_mode
    ):
        raise MetricAdapterContractError(
            "released2 primary replay authority mismatch"
        )
    cache_artifact, real_moments = _validate_released2_real_feature_cache(
        _payload_json_artifact(
            receipt["real_feature_cache"],
            label="released2 replay bound real-feature cache",
        )[1],
        expected_artifact=receipt["real_feature_cache"],
        expected_canonical_manifest=receipt["canonical_manifest"],
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        fixture_mode=fixture_mode,
    )
    if cache_artifact != receipt["real_feature_cache"]:
        raise MetricAdapterContractError(
            "released2 primary replay cache binding changed"
        )
    _cache_artifact, cache_payload = _payload_json_artifact(
        receipt["real_feature_cache"],
        label="released2 replay runtime-bound real-feature cache",
    )
    if any(
        cache_payload["metric_assets"][role]
        != receipt["metric_assets"][role]
        for role in ("talkshow", "feature_extractor", "smplx")
    ) or _metric_runtime_signature(
        cache_payload["runtime"]
    ) != _metric_runtime_signature(receipt["runtime"]):
        raise MetricAdapterContractError(
            "released2 primary replay cache assets/runtime differ"
        )
    real_statistics = receipt["real_feature_statistics"]
    generated_statistics = receipt["generated_feature_statistics"]
    if real_statistics != real_moments.to_json():
        raise MetricAdapterContractError(
            "released2 primary replay real moments differ from cache"
        )
    generated_count = _require_exact_int(
        generated_statistics.get("count")
        if type(generated_statistics) is dict
        else None,
        "released2 primary replay generated count",
        minimum=2,
    )
    generated_moments = FeatureMoments.from_json(
        generated_statistics,
        expected_count=generated_count,
        label="released2 primary replay generated",
    )
    if generated_count != real_moments.count * len(RELEASED2_SLOTS):
        raise MetricAdapterContractError(
            "released2 primary replay feature counts changed"
        )
    recomputed = frechet_distance(real_moments, generated_moments)
    primary = _report_number(
        receipt["primary_metric"],
        "released2 primary replay metric",
        nonnegative=True,
    )
    report_primary = _report_number(
        receipt["report_primary_metric"],
        "released2 primary replay report metric",
        nonnegative=True,
    )
    if (
        not math.isclose(
            primary,
            recomputed,
            rel_tol=1e-13,
            abs_tol=1e-13 * max(1.0, abs(primary), abs(recomputed)),
        )
        or not math.isclose(
            report_primary,
            float(validated_report["primary_metric"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            primary,
            report_primary,
            rel_tol=1e-9,
            abs_tol=1e-9
            * max(1.0, abs(primary), abs(report_primary)),
        )
    ):
        raise MetricAdapterContractError(
            "released2 primary replay metric derivation mismatch"
        )
    return {
        "artifact": artifact,
        "receipt_payload_sha256": receipt[
            "receipt_payload_sha256"
        ],
        "primary_metric_path": PRIMARY_METRIC_PATH,
        "primary_metric": primary,
        "report_payload_sha256": receipt["report_payload_sha256"],
        "prediction_manifest": dict(receipt["prediction_manifest"]),
        "real_feature_cache": dict(receipt["real_feature_cache"]),
        "metric_assets": dict(receipt["metric_assets"]),
        "runtime": dict(receipt["runtime"]),
    }


def validate_released2_primary_screen_replay_receipt(
    value: Mapping[str, Any],
    *,
    expected_screen_artifact: Mapping[str, Any],
    expected_screen: Mapping[str, Any],
    screen_validation: Mapping[str, Any],
    expected_prediction_manifest: Mapping[str, Any],
    expected_distribution_receipt: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
    expected_split: str,
    expected_clip_count: int,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Validate one independent raw-NPZ replay of a screen receipt."""

    artifact, receipt = _payload_json_artifact(
        value,
        label="released2 primary screen replay receipt",
    )
    fixture_mode = (
        test_only_allow_four_clip_subset and expected_clip_count == 4
    )
    expected_keys = {
        "format",
        "status",
        "split",
        "clip_count",
        "primary_metric_path",
        "primary_metric",
        "report_primary_metric",
        "report_payload_sha256",
        "canonical_manifest",
        "prediction_manifest",
        "distribution_receipt_payload_sha256",
        "selection_protocol",
        "real_feature_cache",
        "metric_assets",
        "runtime",
        "real_feature_statistics",
        "generated_feature_statistics",
        "formal_mode",
        "test_only_mode",
        "receipt_payload_sha256",
    }
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise MetricAdapterContractError(
            "released2 primary screen replay receipt schema mismatch"
        )
    if (
        expected_screen.get("format") != PRIMARY_SCREEN_FORMAT
        or screen_validation.get("artifact")
        != dict(expected_screen_artifact)
        or receipt["format"] != PRIMARY_REPLAY_FORMAT
        or receipt["status"] != "pass"
        or receipt["split"] != expected_split
        or receipt["clip_count"] != expected_clip_count
        or receipt["primary_metric_path"] != PRIMARY_METRIC_PATH
        or receipt["report_payload_sha256"]
        != expected_screen["receipt_payload_sha256"]
        or receipt["canonical_manifest"]
        != expected_screen["canonical_manifest"]
        or receipt["prediction_manifest"]
        != dict(expected_prediction_manifest)
        or receipt["distribution_receipt_payload_sha256"]
        != expected_distribution_receipt["receipt_payload_sha256"]
        or receipt["selection_protocol"]
        != dict(expected_selection_protocol)
        or receipt["metric_assets"] != expected_screen["metric_assets"]
        or receipt["runtime"] != expected_screen["runtime"]
        or receipt["formal_mode"] is not (not fixture_mode)
        or receipt["test_only_mode"] is not fixture_mode
    ):
        raise MetricAdapterContractError(
            "released2 primary screen replay authority mismatch"
        )
    cache_artifact, cache_payload = _payload_json_artifact(
        receipt["real_feature_cache"],
        label="screen replay bound real-feature cache",
    )
    _cache, real_moments = _validate_released2_real_feature_cache(
        cache_payload,
        expected_artifact=cache_artifact,
        expected_canonical_manifest=receipt["canonical_manifest"],
        expected_split=expected_split,
        expected_clip_count=expected_clip_count,
        fixture_mode=fixture_mode,
    )
    if receipt["real_feature_statistics"] != real_moments.to_json():
        raise MetricAdapterContractError(
            "screen replay real moments differ from frozen cache"
        )
    generated_statistics = receipt["generated_feature_statistics"]
    generated_count = _require_exact_int(
        generated_statistics.get("count")
        if type(generated_statistics) is dict
        else None,
        "screen replay generated count",
        minimum=2,
    )
    generated_moments = FeatureMoments.from_json(
        generated_statistics,
        expected_count=generated_count,
        label="screen replay generated",
    )
    if generated_count != real_moments.count * len(RELEASED2_SLOTS):
        raise MetricAdapterContractError(
            "screen replay feature counts changed"
        )
    replay_fgd = frechet_distance(real_moments, generated_moments)
    declared = _report_number(
        receipt["primary_metric"],
        "screen replay primary metric",
        nonnegative=True,
    )
    first_pass = _report_number(
        screen_validation["primary_metric"],
        "first-pass screen primary metric",
        nonnegative=True,
    )
    if (
        declared != replay_fgd
        or receipt["report_primary_metric"] != first_pass
        or replay_fgd != first_pass
    ):
        raise MetricAdapterContractError(
            "screen and independent raw replay FGD differ"
        )
    return {
        "artifact": artifact,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "primary_metric_path": PRIMARY_METRIC_PATH,
        "primary_metric": replay_fgd,
        "report_payload_sha256": receipt["report_payload_sha256"],
        "prediction_manifest": dict(receipt["prediction_manifest"]),
        "real_feature_cache": dict(receipt["real_feature_cache"]),
        "metric_assets": dict(receipt["metric_assets"]),
        "runtime": dict(receipt["runtime"]),
    }


def _atomic_write_new(path: Path, payload: bytes) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-canonical-manifest-sha256",
        required=True,
    )
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-prediction-manifest-sha256",
        required=True,
    )
    parser.add_argument("--prediction-lineage", type=Path, required=True)
    parser.add_argument(
        "--expected-prediction-lineage-sha256",
        required=True,
    )
    parser.add_argument("--validation-gate-json", type=Path, required=True)
    parser.add_argument(
        "--expected-validation-gate-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-validation-gate-receipt-payload-sha256",
        required=True,
    )
    parser.add_argument(
        "--distribution-declaration-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-distribution-declaration-sha256",
        required=True,
    )
    parser.add_argument("--test-authority-json", type=Path)
    parser.add_argument("--expected-test-authority-sha256")
    parser.add_argument("--expected-test-authority-bytes", type=int)
    parser.add_argument(
        "--expected-test-authority-receipt-payload-sha256"
    )
    parser.add_argument("--talkshow-metric-root", type=Path, required=True)
    parser.add_argument("--feature-extractor", type=Path, required=True)
    parser.add_argument("--smplx-asset", type=Path, required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="Explicit formal CUDA ordinal, for example cuda:0",
    )
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument(
        "--expected-clip-count",
        type=int,
        required=True,
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    authority_values = (
        args.test_authority_json,
        args.expected_test_authority_sha256,
        args.expected_test_authority_bytes,
        args.expected_test_authority_receipt_payload_sha256,
    )
    if args.split == "test" and any(value is None for value in authority_values):
        parser.error(
            "test split requires authority JSON plus SHA/bytes/payload SHA"
        )
    if args.split == "val" and any(
        value is not None for value in authority_values
    ):
        parser.error("val split forbids every test-authority option")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _declaration_path, declaration_payload = _verified_file_snapshot(
        args.distribution_declaration_json,
        args.expected_distribution_declaration_sha256,
        "distribution declaration",
    )
    declaration = _strict_json_snapshot(
        declaration_payload,
        "distribution declaration",
    )
    validation_gate = {
        "path": str(args.validation_gate_json.resolve()),
        "sha256": _require_sha256(
            args.expected_validation_gate_sha256,
            "validation gate file SHA",
        ),
        "bytes": args.validation_gate_json.resolve().stat().st_size,
        "receipt_payload_sha256": _require_sha256(
            args.expected_validation_gate_receipt_payload_sha256,
            "validation gate payload SHA",
        ),
    }
    test_authority = (
        None
        if args.split == "val"
        else {
            "path": str(args.test_authority_json.resolve()),
            "sha256": args.expected_test_authority_sha256,
            "bytes": args.expected_test_authority_bytes,
            "receipt_payload_sha256": (
                args.expected_test_authority_receipt_payload_sha256
            ),
        }
    )
    backend = TalkShowCudaMetricBackend(
        talkshow_root=args.talkshow_metric_root,
        feature_extractor=args.feature_extractor,
        smplx_asset=args.smplx_asset,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    report = evaluate_canonical_bundle(
        canonical_manifest=args.canonical_manifest,
        expected_canonical_manifest_sha256=(
            args.expected_canonical_manifest_sha256
        ),
        prediction_manifest=args.prediction_manifest,
        expected_prediction_manifest_sha256=(
            args.expected_prediction_manifest_sha256
        ),
        prediction_lineage=args.prediction_lineage,
        expected_prediction_lineage_sha256=(
            args.expected_prediction_lineage_sha256
        ),
        validation_gate=validation_gate,
        distribution_declaration=declaration,
        test_authority=test_authority,
        backend=backend,
        split=args.split,
        expected_clip_count=args.expected_clip_count,
    )
    _atomic_write_new(
        args.output_json,
        canonical_json_bytes(report),
    )
    print(
        f"SHOW TalkSHOW metrics complete: split={args.split} "
        f"clips={args.expected_clip_count} "
        f"released2_FGD={report['body']['released2']['metrics']['FGD']:.8f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
