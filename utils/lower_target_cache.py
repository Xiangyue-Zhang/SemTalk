"""Strict runtime contract for frozen SHOW lower-target SMPL-X joints.

The cache is deliberately opt-in.  Formal lower-VQ training may replace only
the target-side SMPL-X forward with a verified, immutable tensor lookup.  The
reconstructed-pose SMPL-X forward remains live so the vertex loss keeps its
original gradient path.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - receipt tooling can run without torch
    torch = None


CACHE_FORMAT = "semtalk_show_lower_target_joints_raw_lmdb_v1"
CHECKER_FORMAT = "semtalk_show_lower_target_joints_checker_v1"
CACHE_VERSION = 1
EXPECTED_ENTRIES = 127_286
WINDOW_LENGTH = 64
JOINT_COUNT = 127
COORDINATE_COUNT = 3
ENTRY_SHAPE = (WINDOW_LENGTH, JOINT_COUNT, COORDINATE_COUNT)
ENTRY_DTYPE = np.dtype("<f4")
ENTRY_BYTES = int(np.prod(ENTRY_SHAPE)) * ENTRY_DTYPE.itemsize
COMPUTE_BATCH_WINDOWS = 64
COMPUTE_ROWS = COMPUTE_BATCH_WINDOWS * WINDOW_LENGTH
FULL_COMPUTE_BATCHES = EXPECTED_ENTRIES // COMPUTE_BATCH_WINDOWS
TAIL_REAL_WINDOWS = EXPECTED_ENTRIES % COMPUTE_BATCH_WINDOWS
TAIL_PADDING_WINDOWS = COMPUTE_BATCH_WINDOWS - TAIL_REAL_WINDOWS
TOTAL_COMPUTE_BATCHES = FULL_COMPUTE_BATCHES + 1
TAIL_PADDING_POLICY = "zero_windows_compute_only_not_stored"
EXPECTED_SPEAKER_IDS = (0, 1, 2, 3)
EXPECTED_SPEAKER_MAP = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
LOWER_JOINT_INDICES = (0, 1, 2, 4, 5, 7, 8, 10, 11)
LOWER_POSE_COLUMNS = tuple(
    column
    for joint in LOWER_JOINT_INDICES
    for column in range(joint * 3, joint * 3 + 3)
)
TARGET_POSE_PREPROCESS = (
    "select_lower27_axis_angle_to_matrix_to_rot6d_to_matrix_to_axis_angle_"
    "then_zero_fill_165"
)
CHECKER_PERMUTATION_SEED = 20_260_729
RECEIPT_KEY = "lower_target_joints_cache"


class LowerTargetCacheError(RuntimeError):
    """Raised when a cache or one of its formal receipts is invalid."""


def _require_torch() -> Any:
    if torch is None:
        raise LowerTargetCacheError(
            "PyTorch is required for lower target tensor operations"
        )
    return torch


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise LowerTargetCacheError(f"{label} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise LowerTargetCacheError(f"{label} must be a lowercase SHA-256")
    return normalized


def require_git_oid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise LowerTargetCacheError(f"{label} must be a Git object ID")
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise LowerTargetCacheError(
            f"{label} must be a lowercase 40-hex Git object ID"
        )
    return normalized


def require_exact_int(
    value: Any,
    label: str,
    *,
    expected: int | None = None,
) -> int:
    """Accept only a JSON integer, never bool/string/integral float."""

    if type(value) is not int:
        raise LowerTargetCacheError(
            f"{label} must be an exact JSON integer, got {type(value).__name__}"
        )
    if expected is not None and value != expected:
        raise LowerTargetCacheError(f"{label}={value}, expected {expected}")
    return value


def require_exact_json_value(
    value: Any,
    expected: Any,
    label: str,
) -> None:
    """Recursively compare JSON while preserving bool/int/float distinctions."""

    if type(expected) is bool:
        if value is not expected:
            raise LowerTargetCacheError(f"{label}={value!r}, expected {expected!r}")
        return
    if type(expected) is int:
        require_exact_int(value, label, expected=expected)
        return
    if type(expected) is str or expected is None:
        if type(value) is not type(expected) or value != expected:
            raise LowerTargetCacheError(f"{label}={value!r}, expected {expected!r}")
        return
    if type(expected) is list:
        if type(value) is not list or len(value) != len(expected):
            raise LowerTargetCacheError(f"{label} must equal {expected!r}")
        for index, (observed_item, expected_item) in enumerate(
            zip(value, expected)
        ):
            require_exact_json_value(
                observed_item,
                expected_item,
                f"{label}[{index}]",
            )
        return
    if type(expected) is dict:
        if type(value) is not dict or set(value) != set(expected):
            raise LowerTargetCacheError(
                f"{label} keys must equal {sorted(expected)!r}"
            )
        for key, expected_item in expected.items():
            require_exact_json_value(
                value[key],
                expected_item,
                f"{label}.{key}",
            )
        return
    raise TypeError(f"unsupported exact JSON contract type: {type(expected)!r}")


def read_regular_file_snapshot(path_value: str | Path, label: str) -> tuple[Path, bytes]:
    """Read one regular-file snapshot without following a final symlink."""

    input_path = Path(path_value)
    if input_path.is_symlink():
        raise LowerTargetCacheError(f"{label} must not be a symlink: {input_path}")
    path = input_path.resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LowerTargetCacheError(f"cannot open regular {label}: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LowerTargetCacheError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_dev != metadata.st_dev
            or final_metadata.st_ino != metadata.st_ino
            or final_metadata.st_size != metadata.st_size
            or final_metadata.st_mtime_ns != metadata.st_mtime_ns
        ):
            raise LowerTargetCacheError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    return path, payload


def cache_key(index: int) -> bytes:
    if index < 0 or index >= EXPECTED_ENTRIES:
        raise IndexError(index)
    return f"{index:010d}".encode("ascii")


def encode_raw_entry(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.shape != ENTRY_SHAPE:
        raise LowerTargetCacheError(
            f"lower target entry shape {array.shape} != {ENTRY_SHAPE}"
        )
    if array.dtype != np.float32:
        raise LowerTargetCacheError(
            f"lower target entry dtype {array.dtype} != float32"
        )
    if not np.isfinite(array).all():
        raise LowerTargetCacheError("lower target entry contains non-finite values")
    little_endian = np.ascontiguousarray(array.astype(ENTRY_DTYPE, copy=False))
    payload = little_endian.tobytes(order="C")
    if len(payload) != ENTRY_BYTES:
        raise AssertionError(f"encoded entry has {len(payload)} bytes")
    return payload


def decode_raw_entry(payload: bytes | bytearray | memoryview) -> np.ndarray:
    raw = bytes(payload)
    if len(raw) != ENTRY_BYTES:
        raise LowerTargetCacheError(
            f"raw lower target payload has {len(raw)} bytes, expected {ENTRY_BYTES}"
        )
    array = np.frombuffer(raw, dtype=ENTRY_DTYPE).reshape(ENTRY_SHAPE).copy()
    if array.dtype != np.float32:
        raise AssertionError(f"decoded dtype is {array.dtype}")
    if not np.isfinite(array).all():
        raise LowerTargetCacheError("raw lower target payload is non-finite")
    return array


def update_entry_aggregate(
    aggregate: "hashlib._Hash",
    index: int,
    payload: bytes | bytearray | memoryview,
) -> None:
    aggregate.update(cache_key(index))
    aggregate.update(hashlib.sha256(bytes(payload)).digest())


def _load_verified_json(
    path_value: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    path, raw = read_regular_file_snapshot(path_value, label)
    expected = require_sha256(expected_sha256, f"expected {label} SHA-256")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise LowerTargetCacheError(
            f"{label} SHA-256 mismatch for {path}: {actual} != {expected}"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LowerTargetCacheError(f"invalid {label} JSON: {path}") from error
    if type(value) is not dict:
        raise LowerTargetCacheError(f"{label} must contain one JSON object")
    return path, value, actual


def _require_protocol(protocol: Any, label: str) -> Mapping[str, Any]:
    if type(protocol) is not dict:
        raise LowerTargetCacheError(f"{label} protocol must be an object")
    expected = {
        "dataset": "show_base",
        "formal_stage": "lower",
        "speaker_scope": "All",
        "speaker_ids": list(EXPECTED_SPEAKER_IDS),
        "split": "train",
        "entries": EXPECTED_ENTRIES,
        "window_length": WINDOW_LENGTH,
        "joints": JOINT_COUNT,
        "coordinates": COORDINATE_COUNT,
        "dtype": "<f4",
        "byte_order": "little",
        "storage": "raw_c_order",
        "key_format": "%010d",
        "compute_batch_windows": COMPUTE_BATCH_WINDOWS,
        "compute_rows": COMPUTE_ROWS,
        "full_compute_batches": FULL_COMPUTE_BATCHES,
        "total_compute_batches": TOTAL_COMPUTE_BATCHES,
        "tail_real_windows": TAIL_REAL_WINDOWS,
        "tail_padding_windows": TAIL_PADDING_WINDOWS,
        "tail_padding_policy": TAIL_PADDING_POLICY,
        "lower_joint_indices": list(LOWER_JOINT_INDICES),
        "lower_pose_columns": list(LOWER_POSE_COLUMNS),
        "target_pose_preprocess": TARGET_POSE_PREPROCESS,
        "exact_once": True,
    }
    expected_keys = set(expected) | {"speaker_map"}
    if set(protocol) != expected_keys:
        raise LowerTargetCacheError(
            f"{label} protocol keys must equal {sorted(expected_keys)!r}"
        )
    for key, value in expected.items():
        if key not in protocol:
            raise LowerTargetCacheError(f"{label} protocol misses {key!r}")
        require_exact_json_value(protocol[key], value, f"{label} protocol.{key}")
    if "speaker_map" not in protocol:
        raise LowerTargetCacheError(f"{label} does not bind the SHOW All speaker map")
    require_exact_json_value(
        protocol["speaker_map"],
        EXPECTED_SPEAKER_MAP,
        f"{label} protocol.speaker_map",
    )
    return protocol


def validate_canonical_receipt_payload(receipt: Mapping[str, Any]) -> None:
    if type(receipt) is not dict:
        raise LowerTargetCacheError("canonical receipt must be an object")
    for key in ("manifest", "summary", "lineage"):
        if type(receipt.get(key)) is not str or not receipt[key]:
            raise LowerTargetCacheError(
                f"canonical receipt {key!r} must be a non-empty path"
            )
    for key in (
        "manifest_sha256",
        "summary_sha256",
        "lineage_sha256",
        "lineage_contract_sha256",
    ):
        require_sha256(receipt.get(key), f"canonical receipt {key}")
    source = receipt.get("source_receipt")
    if type(source) is not dict:
        raise LowerTargetCacheError("canonical receipt source_receipt must be an object")
    if source.get("origin") != "git@github.com:Xiangyue-Zhang/SemTalk.git":
        raise LowerTargetCacheError("canonical receipt source origin is invalid")
    require_git_oid(source.get("commit"), "canonical source commit")
    require_git_oid(source.get("tree"), "canonical source tree")
    if type(source.get("entrypoint")) is not str or not source["entrypoint"]:
        raise LowerTargetCacheError(
            "canonical source entrypoint must be a non-empty path"
        )
    require_sha256(
        source.get("entrypoint_sha256"),
        "canonical source entrypoint SHA-256",
    )


def validate_representation_summary_payload(
    summary: Mapping[str, Any],
    *,
    representation: Mapping[str, Any],
) -> None:
    if type(summary) is not dict:
        raise LowerTargetCacheError("representation summary must be an object")
    require_exact_int(
        summary.get("entries"),
        "representation summary entries",
        expected=EXPECTED_ENTRIES,
    )
    require_exact_int(
        summary.get("train_clips"),
        "representation summary train_clips",
        expected=13_687,
    )
    protocol = summary.get("protocol")
    if type(protocol) is not dict:
        raise LowerTargetCacheError("representation summary protocol is invalid")
    require_exact_json_value(
        protocol.get("speaker_map"),
        EXPECTED_SPEAKER_MAP,
        "representation summary protocol.speaker_map",
    )
    require_exact_int(
        protocol.get("window_length"),
        "representation summary protocol.window_length",
        expected=WINDOW_LENGTH,
    )
    require_exact_int(
        protocol.get("stride"),
        "representation summary protocol.stride",
        expected=20,
    )
    if (
        protocol.get("split") != "train"
        or protocol.get("tail_policy") != "drop_incomplete"
        or protocol.get("clip_boundary_policy")
        != "floor_to_whole_seconds_at_30fps"
        or protocol.get("filtering") != "none"
    ):
        raise LowerTargetCacheError(
            "representation summary preprocessing protocol is invalid"
        )
    fastpath = protocol.get("global_foot_fastpath")
    if type(fastpath) is not dict:
        raise LowerTargetCacheError(
            "representation summary global-foot protocol is invalid"
        )
    if representation.get("format") == (
        "semtalk_show_representation_lmdb_v2_global_foot"
    ):
        require_exact_json_value(
            fastpath,
            {
                "enabled": True,
                "contract": "semtalk_show_global_foot_fastpath_v1",
                "field": "lower_foot_local",
                "shape": [WINDOW_LENGTH, 4, 3],
                "dtype": "float32",
                "activation_env": "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH=1",
            },
            "representation summary protocol.global_foot_fastpath",
        )
    else:
        require_exact_json_value(
            fastpath,
            {"enabled": False, "contract": None},
            "representation summary protocol.global_foot_fastpath",
        )
    if (
        summary.get("format") != representation.get("format")
        or summary.get("status") != "complete"
        or protocol.get("split") != "train"
        or summary.get("data_mdb_sha256")
        != representation.get("data_mdb_sha256")
        or summary.get("lock_mdb_sha256")
        != representation.get("lock_mdb_sha256")
        or summary.get("entry_aggregate_sha256")
        != representation.get("entry_aggregate_sha256")
        or summary.get("canonical_receipt")
        != representation.get("canonical_receipt")
        or summary.get("canonical_manifest_sha256")
        != representation.get("canonical_manifest_sha256")
    ):
        raise LowerTargetCacheError(
            "representation summary does not match the frozen cache receipt"
        )
    source = summary.get("source_receipt")
    representation_source = representation.get("source_receipt")
    canonical = representation.get("canonical_receipt")
    if (
        type(source) is not dict
        or type(representation_source) is not dict
        or type(canonical) is not dict
    ):
        raise LowerTargetCacheError(
            "representation summary lacks source/canonical receipts"
        )
    validate_canonical_receipt_payload(canonical)
    require_sha256(
        source.get("entrypoint_sha256"),
        "representation source entrypoint SHA-256",
    )
    for key in ("origin", "commit", "tree"):
        if source.get(key) != representation_source.get(key):
            raise LowerTargetCacheError(
                f"representation summary/producer source {key} binding is invalid"
            )
    for count_label in ("speaker_clip_counts", "speaker_window_counts"):
        counts = summary.get(count_label)
        if type(counts) is not dict or set(counts) != set(EXPECTED_SPEAKER_MAP):
            raise LowerTargetCacheError(
                f"representation summary {count_label} has the wrong speakers"
            )
        for speaker, count in counts.items():
            value = require_exact_int(
                count,
                f"representation summary {count_label}.{speaker}",
            )
            if value <= 0:
                raise LowerTargetCacheError(
                    f"representation summary {count_label}.{speaker} must be positive"
                )
    if sum(summary["speaker_clip_counts"].values()) != 13_687:
        raise LowerTargetCacheError(
            "representation summary speaker clip counts are inconsistent"
        )
    if sum(summary["speaker_window_counts"].values()) != EXPECTED_ENTRIES:
        raise LowerTargetCacheError(
            "representation summary speaker window counts are inconsistent"
        )


def validate_representation_canonical_coverage(
    summary: Mapping[str, Any],
    canonical_files: Mapping[str, Any],
) -> None:
    require_exact_json_value(
        summary.get("speaker_clip_counts"),
        canonical_files.get("train_speaker_clip_counts"),
        "representation/canonical train speaker clip counts",
    )
    require_exact_json_value(
        summary.get("speaker_window_counts"),
        canonical_files.get("train_speaker_window_counts"),
        "representation/canonical train speaker window counts",
    )
    require_exact_int(
        canonical_files.get("train_clips"),
        "canonical train clips",
        expected=13_687,
    )
    require_exact_int(
        canonical_files.get("train_windows"),
        "canonical train windows",
        expected=EXPECTED_ENTRIES,
    )


def validate_canonical_files(
    representation: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = representation.get("canonical_receipt")
    validate_canonical_receipt_payload(receipt)
    manifest_path, manifest_raw = read_regular_file_snapshot(
        receipt["manifest"],
        "canonical manifest",
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if str(manifest_path) != receipt["manifest"]:
        raise LowerTargetCacheError("canonical manifest path is not normalized")
    if manifest_sha != receipt["manifest_sha256"]:
        raise LowerTargetCacheError("canonical manifest SHA-256 mismatch")
    manifest_hashes = representation.get("canonical_manifest_sha256")
    if type(manifest_hashes) is not dict or not manifest_hashes:
        raise LowerTargetCacheError(
            "representation canonical manifest hash map is invalid"
        )
    for path_value, expected_digest in manifest_hashes.items():
        if type(path_value) is not str or not path_value:
            raise LowerTargetCacheError(
                "canonical manifest hash map path is invalid"
            )
        require_sha256(
            expected_digest,
            f"canonical manifest shard {path_value} SHA-256",
        )
    recorded_hash = manifest_hashes.get(str(manifest_path))
    if recorded_hash != manifest_sha:
        raise LowerTargetCacheError(
            "representation canonical manifest map does not bind the selected manifest"
        )
    for path_value, expected_digest in manifest_hashes.items():
        path, raw = read_regular_file_snapshot(
            path_value,
            "canonical manifest shard",
        )
        if str(path) != path_value:
            raise LowerTargetCacheError(
                "canonical manifest shard path is not normalized"
            )
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise LowerTargetCacheError(
                f"canonical manifest shard SHA-256 mismatch: {path}"
            )

    summary_path, canonical_summary, summary_sha = _load_verified_json(
        receipt["summary"],
        receipt["summary_sha256"],
        "canonical summary",
    )
    lineage_path, canonical_lineage, lineage_sha = _load_verified_json(
        receipt["lineage"],
        receipt["lineage_sha256"],
        "canonical lineage",
    )
    if (
        str(summary_path) != receipt["summary"]
        or str(lineage_path) != receipt["lineage"]
    ):
        raise LowerTargetCacheError(
            "canonical summary/lineage paths are not normalized"
        )
    require_exact_int(
        canonical_summary.get("schema_version"),
        "canonical summary schema_version",
        expected=1,
    )
    require_exact_int(
        canonical_summary.get("clip_count"),
        "canonical summary clip_count",
        expected=17_110,
    )
    require_exact_json_value(
        canonical_summary.get("split_counts"),
        {"train": 13_687, "val": 1_715, "test": 1_708},
        "canonical summary split_counts",
    )
    frame_count = require_exact_int(
        canonical_summary.get("frame_count"),
        "canonical summary frame_count",
    )
    if frame_count <= 0:
        raise LowerTargetCacheError(
            "canonical summary frame_count must be positive"
        )
    num_shards = require_exact_int(
        canonical_summary.get("num_shards"),
        "canonical summary num_shards",
    )
    if num_shards <= 0:
        raise LowerTargetCacheError(
            "canonical summary num_shards must be positive"
        )
    if (
        canonical_summary.get("status") != "complete"
        or canonical_summary.get("schema_name")
        != "semtalk-show-canonical-motion"
        or canonical_summary.get("manifest_sha256") != manifest_sha
        or canonical_summary.get("exact_once") is not True
        or canonical_summary.get("finite") is not True
        or canonical_summary.get("split_disjoint") is not True
        or canonical_summary.get("lineage_sha256")
        != canonical_json_sha256(canonical_lineage)
        or canonical_lineage.get("final_manifest_sha256") != manifest_sha
        or canonical_lineage.get("lineage_contract_sha256")
        != receipt["lineage_contract_sha256"]
        or canonical_summary.get("lineage_contract_sha256")
        != receipt["lineage_contract_sha256"]
    ):
        raise LowerTargetCacheError(
            "canonical summary/lineage/manifest binding is invalid"
        )
    lineage_contract = canonical_lineage.get("lineage_contract")
    if type(lineage_contract) is not dict:
        raise LowerTargetCacheError("canonical lineage contract is invalid")
    if canonical_json_sha256(lineage_contract) != receipt["lineage_contract_sha256"]:
        raise LowerTargetCacheError("canonical lineage contract SHA-256 mismatch")
    require_exact_int(
        lineage_contract.get("schema_version"),
        "canonical lineage schema_version",
        expected=1,
    )
    require_exact_json_value(
        lineage_contract.get("split_counts"),
        {"train": 13_687, "val": 1_715, "test": 1_708},
        "canonical lineage split_counts",
    )
    require_exact_json_value(
        lineage_contract.get("speaker_mapping"),
        EXPECTED_SPEAKER_MAP,
        "canonical lineage speaker_mapping",
    )
    require_exact_int(
        lineage_contract.get("split_missing_count"),
        "canonical lineage split_missing_count",
        expected=0,
    )
    require_exact_int(
        lineage_contract.get("pose_fps"),
        "canonical lineage pose_fps",
        expected=30,
    )
    require_exact_int(
        lineage_contract.get("source_audio_sample_rate"),
        "canonical lineage source_audio_sample_rate",
        expected=22_000,
    )
    require_exact_int(
        lineage_contract.get("hubert_target_sample_rate"),
        "canonical lineage hubert_target_sample_rate",
        expected=16_000,
    )
    if (
        type(lineage_contract.get("smplx_asset_path")) is not str
        or not lineage_contract["smplx_asset_path"]
    ):
        raise LowerTargetCacheError(
            "canonical lineage SMPL-X asset path is invalid"
        )
    canonical_smplx_sha = require_sha256(
        lineage_contract.get("smplx_asset_sha256"),
        "canonical lineage SMPL-X asset SHA-256",
    )
    canonical_source = lineage_contract.get("source_receipt")
    if (
        canonical_source != receipt["source_receipt"]
        or canonical_summary.get("source_receipt_sha256")
        != canonical_json_sha256(canonical_source)
    ):
        raise LowerTargetCacheError("canonical source receipt binding is invalid")
    build_runtime = canonical_lineage.get("build_runtime")
    if (
        type(build_runtime) is not dict
        or canonical_lineage.get("build_runtime_sha256")
        != canonical_json_sha256(build_runtime)
        or canonical_summary.get("build_runtime_sha256")
        != canonical_lineage.get("build_runtime_sha256")
    ):
        raise LowerTargetCacheError("canonical build runtime binding is invalid")

    try:
        manifest_rows = [
            json.loads(line.decode("utf-8"))
            for line in manifest_raw.splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LowerTargetCacheError(
            "canonical manifest contains invalid JSONL"
        ) from error
    if len(manifest_rows) != 17_110 or any(
        type(row) is not dict for row in manifest_rows
    ):
        raise LowerTargetCacheError(
            "canonical manifest must contain exactly 17110 JSON objects"
        )
    split_counts = {"train": 0, "val": 0, "test": 0}
    clip_ids: set[str] = set()
    observed_speakers: set[str] = set()
    train_speaker_clip_counts = {
        speaker: 0 for speaker in EXPECTED_SPEAKER_MAP
    }
    train_speaker_window_counts = {
        speaker: 0 for speaker in EXPECTED_SPEAKER_MAP
    }
    observed_frame_count = 0
    for ordinal, row in enumerate(manifest_rows):
        global_index = require_exact_int(
            row.get("global_index"),
            f"canonical manifest row {ordinal} global_index",
            expected=ordinal,
        )
        del global_index
        clip_id = row.get("clip_id")
        split = row.get("split")
        speaker = row.get("speaker")
        if (
            type(clip_id) is not str
            or not clip_id
            or clip_id in clip_ids
            or split not in split_counts
            or speaker not in EXPECTED_SPEAKER_MAP
        ):
            raise LowerTargetCacheError(
                f"canonical manifest row {ordinal} identity is invalid"
            )
        clip_ids.add(clip_id)
        observed_speakers.add(speaker)
        split_counts[split] += 1
        require_exact_int(
            row.get("speaker_id"),
            f"canonical manifest row {ordinal} speaker_id",
            expected=EXPECTED_SPEAKER_MAP[speaker],
        )
        frames = require_exact_int(
            row.get("frames"),
            f"canonical manifest row {ordinal} frames",
        )
        if frames <= 0:
            raise LowerTargetCacheError(
                f"canonical manifest row {ordinal} frames must be positive"
            )
        observed_frame_count += frames
        if split == "train":
            train_speaker_clip_counts[speaker] += 1
            usable_frames = (frames // 30) * 30
            if usable_frames >= WINDOW_LENGTH:
                train_speaker_window_counts[speaker] += (
                    (usable_frames - WINDOW_LENGTH) // 20 + 1
                )
        require_exact_int(
            row.get("pose_fps"),
            f"canonical manifest row {ordinal} pose_fps",
            expected=30,
        )
        if row.get("lineage_contract_sha256") != (
            receipt["lineage_contract_sha256"]
        ):
            raise LowerTargetCacheError(
                f"canonical manifest row {ordinal} lineage binding is invalid"
            )
        for hash_key in (
            "source_pkl_sha256",
            "source_wav_sha256",
            "canonical_npz_sha256",
            "lower_foot_local_sha256",
        ):
            require_sha256(
                row.get(hash_key),
                f"canonical manifest row {ordinal} {hash_key}",
            )
    require_exact_json_value(
        split_counts,
        {"train": 13_687, "val": 1_715, "test": 1_708},
        "canonical manifest observed split counts",
    )
    if observed_frame_count != frame_count:
        raise LowerTargetCacheError(
            "canonical manifest frame count does not match summary"
        )
    if observed_speakers != set(EXPECTED_SPEAKER_MAP):
        raise LowerTargetCacheError(
            "canonical manifest does not cover every SHOW speaker"
        )

    shard_receipts = canonical_lineage.get("shards")
    if type(shard_receipts) is not list or len(shard_receipts) != num_shards:
        raise LowerTargetCacheError(
            "canonical lineage shard receipts are incomplete"
        )
    shard_clip_count = 0
    for shard_id, shard in enumerate(shard_receipts):
        if type(shard) is not dict:
            raise LowerTargetCacheError(
                f"canonical shard receipt {shard_id} is invalid"
            )
        require_exact_int(
            shard.get("shard_id"),
            f"canonical shard receipt {shard_id} shard_id",
            expected=shard_id,
        )
        shard_clip_count += require_exact_int(
            shard.get("clip_count"),
            f"canonical shard receipt {shard_id} clip_count",
        )
        for path_key, hash_key in (
            ("manifest_path", "manifest_sha256"),
            ("summary_path", "summary_sha256"),
            ("lineage_path", "lineage_sha256"),
        ):
            shard_path, shard_raw = read_regular_file_snapshot(
                shard.get(path_key),
                f"canonical shard {shard_id} {path_key}",
            )
            if (
                str(shard_path) != shard.get(path_key)
                or hashlib.sha256(shard_raw).hexdigest()
                != require_sha256(
                    shard.get(hash_key),
                    f"canonical shard {shard_id} {hash_key}",
                )
            ):
                raise LowerTargetCacheError(
                    f"canonical shard {shard_id} {path_key} binding is invalid"
                )
    if shard_clip_count != 17_110:
        raise LowerTargetCacheError(
            "canonical shard clip counts do not cover the final manifest"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha,
        "lineage_path": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "lineage_contract_sha256": receipt["lineage_contract_sha256"],
        "train_clips": sum(train_speaker_clip_counts.values()),
        "train_windows": sum(train_speaker_window_counts.values()),
        "train_speaker_clip_counts": train_speaker_clip_counts,
        "train_speaker_window_counts": train_speaker_window_counts,
        "smplx_asset_path": lineage_contract["smplx_asset_path"],
        "smplx_asset_sha256": canonical_smplx_sha,
    }


def validate_manifest_payload(manifest: Mapping[str, Any]) -> None:
    if type(manifest) is not dict:
        raise LowerTargetCacheError("lower target cache manifest must be an object")
    require_exact_int(
        manifest.get("cache_version"),
        "cache manifest cache_version",
        expected=CACHE_VERSION,
    )
    require_exact_int(
        manifest.get("entries"),
        "cache manifest entries",
        expected=EXPECTED_ENTRIES,
    )
    require_exact_int(
        manifest.get("entry_bytes"),
        "cache manifest entry_bytes",
        expected=ENTRY_BYTES,
    )
    if (
        manifest.get("format") != CACHE_FORMAT
        or manifest.get("status") != "complete"
        or manifest.get("finite") is not True
        or manifest.get("exact_once") is not True
        or manifest.get("target_requires_grad") is not False
        or manifest.get("target_optimizer_member") is not False
    ):
        raise LowerTargetCacheError("lower target cache manifest is incomplete")
    require_exact_json_value(
        manifest.get("entry_shape"),
        list(ENTRY_SHAPE),
        "cache manifest entry_shape",
    )
    require_exact_json_value(
        manifest.get("observed_speaker_ids"),
        list(EXPECTED_SPEAKER_IDS),
        "cache manifest observed_speaker_ids",
    )
    _require_protocol(manifest.get("protocol"), "cache manifest")
    require_sha256(
        manifest.get("entry_aggregate_sha256"),
        "cache entry aggregate SHA-256",
    )
    source = manifest.get("source_receipt")
    representation = manifest.get("representation_receipt")
    smplx = manifest.get("smplx_receipt")
    runtime = manifest.get("runtime")
    if not all(
        type(value) is dict
        for value in (source, representation, smplx, runtime)
    ):
        raise LowerTargetCacheError(
            "cache manifest lacks source/representation/SMPL-X/runtime receipts"
        )
    require_sha256(
        runtime.get("rotation_conversions_sha256"),
        "rotation conversion source SHA-256",
    )
    runtime_strings = (
        "python",
        "numpy",
        "torch",
        "torch_cuda",
        "smplx",
        "device",
        "device_name",
        "default_dtype",
        "rotation_conversions",
    )
    for key in runtime_strings:
        if type(runtime.get(key)) is not str or not runtime[key]:
            raise LowerTargetCacheError(
                f"cache producer runtime {key!r} must be a non-empty string"
            )
    capability = runtime.get("device_capability")
    if type(capability) is not list or len(capability) != 2:
        raise LowerTargetCacheError(
            "cache producer runtime device_capability must contain two integers"
        )
    for index, value in enumerate(capability):
        require_exact_int(
            value,
            f"cache producer runtime device_capability[{index}]",
        )
    if (
        "H200" not in str(runtime.get("device_name", ""))
        or runtime.get("default_dtype") != "torch.float32"
        or runtime.get("grad_enabled_during_forward") is not False
    ):
        raise LowerTargetCacheError("cache producer runtime is not formal H200 float32")
    if source.get("origin") != "git@github.com:Xiangyue-Zhang/SemTalk.git":
        raise LowerTargetCacheError("cache manifest has the wrong source origin")
    require_git_oid(source.get("commit"), "cache source commit")
    require_git_oid(source.get("tree"), "cache source tree")
    require_sha256(
        source.get("entrypoint_sha256"),
        "cache producer entrypoint SHA-256",
    )
    require_exact_int(
        representation.get("entries"),
        "representation receipt entries",
        expected=EXPECTED_ENTRIES,
    )
    require_exact_int(
        representation.get("window_length"),
        "representation receipt window_length",
        expected=WINDOW_LENGTH,
    )
    if (
        representation.get("format")
        not in {
            "semtalk_show_representation_lmdb_v1",
            "semtalk_show_representation_lmdb_v2_global_foot",
        }
        or representation.get("speaker_scope") != "All"
    ):
        raise LowerTargetCacheError("cache representation receipt is not SHOW All")
    require_exact_json_value(
        representation.get("speaker_ids"),
        list(EXPECTED_SPEAKER_IDS),
        "representation receipt speaker_ids",
    )
    require_exact_json_value(
        representation.get("speaker_map"),
        EXPECTED_SPEAKER_MAP,
        "representation receipt speaker_map",
    )
    require_sha256(
        representation.get("data_mdb_sha256"),
        "representation data.mdb SHA-256",
    )
    require_sha256(
        representation.get("lock_mdb_sha256"),
        "representation lock.mdb SHA-256",
    )
    require_sha256(
        representation.get("summary_sha256"),
        "representation summary SHA-256",
    )
    require_sha256(
        representation.get("lineage_sha256"),
        "representation lineage SHA-256",
    )
    require_sha256(
        representation.get("lineage_payload_sha256"),
        "representation lineage payload SHA-256",
    )
    require_sha256(
        representation.get("entry_aggregate_sha256"),
        "representation entry aggregate SHA-256",
    )
    canonical_smplx_sha = require_sha256(
        representation.get("canonical_smplx_asset_sha256"),
        "representation canonical SMPL-X asset SHA-256",
    )
    representation_source = representation.get("source_receipt")
    if type(representation_source) is not dict:
        raise LowerTargetCacheError(
            "representation receipt lacks its source receipt"
        )
    if representation_source.get("origin") != (
        "git@github.com:Xiangyue-Zhang/SemTalk.git"
    ):
        raise LowerTargetCacheError(
            "representation receipt has the wrong source origin"
        )
    require_git_oid(
        representation_source.get("commit"),
        "representation source commit",
    )
    require_git_oid(
        representation_source.get("tree"),
        "representation source tree",
    )
    if (
        type(representation_source.get("entrypoint")) is not str
        or not representation_source["entrypoint"]
    ):
        raise LowerTargetCacheError(
            "representation source entrypoint must be a non-empty path"
        )
    require_sha256(
        representation_source.get("entrypoint_sha256"),
        "representation source entrypoint SHA-256",
    )
    for key in ("origin", "commit", "tree"):
        if representation_source.get(key) != source.get(key):
            raise LowerTargetCacheError(
                f"representation/cache source {key} binding is invalid"
            )
    for key in ("lmdb_path", "summary_path", "lineage_path"):
        if type(representation.get(key)) is not str or not representation[key]:
            raise LowerTargetCacheError(
                f"representation receipt {key!r} must be a non-empty path"
            )
    canonical = representation.get("canonical_receipt")
    if type(canonical) is not dict:
        raise LowerTargetCacheError(
            "representation receipt lacks its canonical cache receipt"
        )
    validate_canonical_receipt_payload(canonical)
    canonical_manifest_hashes = representation.get(
        "canonical_manifest_sha256"
    )
    if type(canonical_manifest_hashes) is not dict or not canonical_manifest_hashes:
        raise LowerTargetCacheError(
            "representation receipt lacks canonical manifest hashes"
        )
    for path, digest in canonical_manifest_hashes.items():
        if type(path) is not str or not path:
            raise LowerTargetCacheError(
                "canonical manifest hash map contains an invalid path"
            )
        require_sha256(digest, f"canonical manifest {path} SHA-256")
    require_sha256(smplx.get("asset_sha256"), "SMPL-X asset SHA-256")
    if smplx.get("asset_sha256") != canonical_smplx_sha:
        raise LowerTargetCacheError(
            "lower target SMPL-X asset differs from canonical SHOW lineage"
        )
    for key in ("model_dir", "asset_path"):
        if type(smplx.get(key)) is not str or not smplx[key]:
            raise LowerTargetCacheError(
                f"SMPL-X receipt {key!r} must be a non-empty path"
            )
    require_exact_int(
        smplx.get("num_betas"),
        "SMPL-X receipt num_betas",
        expected=300,
    )
    require_exact_int(
        smplx.get("num_expression_coeffs"),
        "SMPL-X receipt num_expression_coeffs",
        expected=100,
    )
    require_exact_int(
        smplx.get("output_joints"),
        "SMPL-X receipt output_joints",
        expected=JOINT_COUNT,
    )
    if (
        smplx.get("model_type") != "smplx"
        or smplx.get("gender") != "NEUTRAL_2020"
        or smplx.get("use_face_contour") is not False
        or smplx.get("use_pca") is not False
        or smplx.get("ext") != "npz"
        or smplx.get("return_verts") is not False
        or smplx.get("return_joints") is not True
        or smplx.get("return_shaped") is not False
        or smplx.get("target_forward")
        != "torch.no_grad+return_shaped_false"
        or smplx.get("translation") != "tar_trans_minus_itself"
        or smplx.get("expression") != "zeros_float32"
        or smplx.get("lower_joint_indices") != list(LOWER_JOINT_INDICES)
        or smplx.get("lower_pose_columns") != list(LOWER_POSE_COLUMNS)
        or smplx.get("target_pose_preprocess") != TARGET_POSE_PREPROCESS
    ):
        raise LowerTargetCacheError("cache SMPL-X receipt is not trainer-exact")
    lmdb_receipt = manifest.get("lmdb")
    if type(lmdb_receipt) is not dict:
        raise LowerTargetCacheError("cache manifest lacks LMDB receipt")
    require_sha256(lmdb_receipt.get("data_mdb_sha256"), "cache data.mdb SHA-256")
    require_sha256(lmdb_receipt.get("lock_mdb_sha256"), "cache lock.mdb SHA-256")
    require_exact_int(
        lmdb_receipt.get("entries"),
        "cache LMDB entries",
        expected=EXPECTED_ENTRIES,
    )
    require_exact_int(
        lmdb_receipt.get("map_size_bytes"),
        "cache LMDB map_size_bytes",
        expected=16 * 1024**3,
    )
    if type(lmdb_receipt.get("path")) is not str or not lmdb_receipt["path"]:
        raise LowerTargetCacheError(
            "cache LMDB receipt path must be a non-empty path"
        )


def validate_checker_payload(
    checker: Mapping[str, Any],
    *,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
) -> None:
    if type(checker) is not dict:
        raise LowerTargetCacheError("lower target checker receipt must be an object")
    require_exact_int(
        checker.get("cache_version"),
        "checker cache_version",
        expected=CACHE_VERSION,
    )
    require_exact_int(
        checker.get("mismatch_count"),
        "checker mismatch_count",
        expected=0,
    )
    require_exact_int(
        checker.get("checked_entries"),
        "checker checked_entries",
        expected=EXPECTED_ENTRIES,
    )
    if (
        checker.get("format") != CHECKER_FORMAT
        or checker.get("status") != "complete"
        or checker.get("finite") is not True
        or checker.get("exact_once") is not True
        or checker.get("torch_equal_all") is not True
    ):
        raise LowerTargetCacheError("lower target checker receipt is incomplete")
    _require_protocol(checker.get("protocol"), "checker")
    if checker.get("cache_manifest_sha256") != manifest_sha256:
        raise LowerTargetCacheError("checker does not bind the selected manifest")
    if (
        checker.get("cache_data_mdb_sha256")
        != manifest["lmdb"]["data_mdb_sha256"]
        or checker.get("entry_aggregate_sha256")
        != manifest["entry_aggregate_sha256"]
    ):
        raise LowerTargetCacheError("checker/cache byte receipts disagree")
    traversal = checker.get("traversal")
    if (
        type(traversal) is not dict
        or traversal.get("method") != "numpy_default_rng_permutation"
        or traversal.get("covers_all_entries") is not True
    ):
        raise LowerTargetCacheError(
            "checker did not traverse all cache entries once in randomized order"
        )
    for key, expected in (
        ("duplicates", 0),
        ("missing", 0),
        ("entries", EXPECTED_ENTRIES),
        ("seed", CHECKER_PERMUTATION_SEED),
        ("batch_windows", COMPUTE_BATCH_WINDOWS),
        ("tail_real_windows", TAIL_REAL_WINDOWS),
        ("tail_padding_windows", TAIL_PADDING_WINDOWS),
    ):
        require_exact_int(
            traversal.get(key),
            f"checker traversal {key}",
            expected=expected,
        )
    if traversal.get("tail_padding_policy") != TAIL_PADDING_POLICY:
        raise LowerTargetCacheError(
            "checker traversal tail padding policy is invalid"
        )
    if checker.get("cache_path") != manifest["lmdb"]["path"]:
        raise LowerTargetCacheError("checker cache path does not match producer")
    require_exact_json_value(
        checker.get("observed_speaker_ids"),
        list(EXPECTED_SPEAKER_IDS),
        "checker observed_speaker_ids",
    )
    checker_source = checker.get("source_receipt")
    producer_source = manifest.get("source_receipt")
    if type(checker_source) is not dict:
        raise LowerTargetCacheError("checker lacks an independent source receipt")
    require_sha256(
        checker_source.get("entrypoint_sha256"),
        "checker entrypoint SHA-256",
    )
    for key in ("origin", "commit", "tree"):
        if checker_source.get(key) != producer_source.get(key):
            raise LowerTargetCacheError(
                f"checker source {key} does not match producer"
            )
    if checker_source.get("entrypoint") == producer_source.get("entrypoint"):
        raise LowerTargetCacheError(
            "checker must use an entrypoint independent of the producer"
        )
    for key in ("representation_receipt", "smplx_receipt"):
        if checker.get(key) != manifest.get(key):
            raise LowerTargetCacheError(f"checker {key} does not match producer")
    if checker.get("runtime") != manifest.get("runtime"):
        raise LowerTargetCacheError("checker runtime does not match producer")


def current_runtime_receipt(device: Any) -> dict[str, Any]:
    torch_module = _require_torch()
    from utils import rotation_conversions

    resolved_device = torch_module.device(device)
    if resolved_device.type != "cuda":
        raise LowerTargetCacheError("lower target cache runtime requires CUDA")
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
        "smplx": importlib.metadata.version("smplx"),
        "device": str(resolved_device),
        "device_name": torch_module.cuda.get_device_name(resolved_device),
        "device_capability": list(
            torch_module.cuda.get_device_capability(resolved_device)
        ),
        "default_dtype": str(torch_module.get_default_dtype()),
        "grad_enabled_during_forward": False,
        "rotation_conversions": str(
            Path(rotation_conversions.__file__).resolve()
        ),
        "rotation_conversions_sha256": sha256_file(
            Path(rotation_conversions.__file__).resolve()
        ),
    }


def validate_current_inputs_against_manifest(
    args: Any,
    manifest: Mapping[str, Any],
    *,
    device: Any,
) -> dict[str, Any]:
    """Bind cached targets to the exact dataset, SMPL-X asset and runtime used now."""

    representation = manifest["representation_receipt"]
    train_input = Path(getattr(args, "train_path", ""))
    if train_input.is_symlink():
        raise LowerTargetCacheError(
            f"training representation LMDB must not be a symlink: {train_input}"
        )
    train_path = train_input.resolve()
    if not train_path.is_dir():
        raise FileNotFoundError(train_path)
    if train_path != Path(representation["lmdb_path"]).resolve():
        raise LowerTargetCacheError(
            "lower target cache representation LMDB path does not match "
            "the current training dataset"
        )
    representation_artifacts: dict[str, str] = {}
    for filename, receipt_key in (
        ("data.mdb", "data_mdb_sha256"),
        ("lock.mdb", "lock_mdb_sha256"),
    ):
        artifact = train_path / filename
        if artifact.is_symlink() or not artifact.is_file():
            raise LowerTargetCacheError(
                f"invalid current representation artifact: {artifact}"
            )
        actual = sha256_file(artifact)
        if actual != representation[receipt_key]:
            raise LowerTargetCacheError(
                f"current representation {filename} SHA-256 mismatch"
            )
        representation_artifacts[receipt_key] = actual

    summary_value = getattr(args, "dataset_summary", None)
    lineage_value = getattr(args, "lineage_manifest", None)
    if not summary_value or not lineage_value:
        raise LowerTargetCacheError(
            "lower target cache requires current dataset summary and lineage paths"
        )
    summary_path, summary, summary_sha = _load_verified_json(
        summary_value,
        representation["summary_sha256"],
        "current representation summary",
    )
    lineage_path, lineage, lineage_sha = _load_verified_json(
        lineage_value,
        representation["lineage_sha256"],
        "current representation lineage",
    )
    if (
        summary_path != Path(representation["summary_path"]).resolve()
        or lineage_path != Path(representation["lineage_path"]).resolve()
        or summary_path != lineage_path
        or summary_sha != lineage_sha
        or summary != lineage
    ):
        raise LowerTargetCacheError(
            "current representation summary/lineage paths do not match "
            "the cache producer receipt"
        )
    if canonical_json_sha256(lineage) != representation["lineage_payload_sha256"]:
        raise LowerTargetCacheError(
            "current representation lineage payload SHA-256 mismatch"
        )
    validate_representation_summary_payload(
        summary,
        representation=representation,
    )
    canonical_files = validate_canonical_files(representation)
    validate_representation_canonical_coverage(
        summary,
        canonical_files,
    )

    expected_smplx_sha = require_sha256(
        getattr(args, "expected_smplx_asset_sha256", None),
        "expected current SMPL-X asset SHA-256",
    )
    from utils.project_paths import smplx_model_dir

    model_input = smplx_model_dir(args)
    if model_input.is_symlink():
        raise LowerTargetCacheError(
            f"current SMPL-X model directory must not be a symlink: {model_input}"
        )
    model_dir = model_input.resolve()
    asset = model_dir / "smplx" / "SMPLX_NEUTRAL_2020.npz"
    if asset.is_symlink() or not asset.is_file():
        raise LowerTargetCacheError(f"invalid current SMPL-X asset: {asset}")
    asset = asset.resolve()
    actual_smplx_sha = sha256_file(asset)
    smplx_receipt = manifest["smplx_receipt"]
    if (
        expected_smplx_sha != actual_smplx_sha
        or actual_smplx_sha != smplx_receipt["asset_sha256"]
        or model_dir != Path(smplx_receipt["model_dir"]).resolve()
        or asset != Path(smplx_receipt["asset_path"]).resolve()
    ):
        raise LowerTargetCacheError(
            "lower target cache SMPL-X receipt does not match the current "
            "trainer asset"
        )

    runtime = current_runtime_receipt(device)
    if runtime != manifest["runtime"]:
        raise LowerTargetCacheError(
            "lower target cache producer/checker runtime does not match "
            "the current training runtime"
        )
    for filename, receipt_key in (
        ("data.mdb", "data_mdb_sha256"),
        ("lock.mdb", "lock_mdb_sha256"),
    ):
        artifact = train_path / filename
        if (
            artifact.is_symlink()
            or sha256_file(artifact) != representation_artifacts[receipt_key]
        ):
            raise LowerTargetCacheError(
                f"current representation {filename} changed during validation"
            )
    if asset.is_symlink() or sha256_file(asset) != actual_smplx_sha:
        raise LowerTargetCacheError(
            "current SMPL-X asset changed during validation"
        )
    final_summary_path, final_summary, final_summary_sha = _load_verified_json(
        summary_value,
        representation["summary_sha256"],
        "final current representation summary",
    )
    final_lineage_path, final_lineage, final_lineage_sha = _load_verified_json(
        lineage_value,
        representation["lineage_sha256"],
        "final current representation lineage",
    )
    if (
        final_summary_path != summary_path
        or final_lineage_path != lineage_path
        or final_summary_sha != summary_sha
        or final_lineage_sha != lineage_sha
        or final_summary != summary
        or final_lineage != lineage
    ):
        raise LowerTargetCacheError(
            "current representation summary/lineage changed during validation"
        )
    return {
        "representation_lmdb_path": str(train_path),
        **representation_artifacts,
        "representation_summary_path": str(summary_path),
        "representation_summary_sha256": summary_sha,
        "representation_lineage_path": str(lineage_path),
        "representation_lineage_sha256": lineage_sha,
        "canonical_files": canonical_files,
        "smplx_model_dir": str(model_dir),
        "smplx_asset_path": str(asset),
        "smplx_asset_sha256": actual_smplx_sha,
        "runtime": runtime,
    }


def _activation_values(args: Any) -> dict[str, Any]:
    return {
        "cache": getattr(args, "lower_target_joints_cache", None),
        "manifest": getattr(args, "lower_target_joints_cache_manifest", None),
        "manifest_sha256": getattr(
            args,
            "expected_lower_target_joints_cache_manifest_sha256",
            None,
        ),
        "checker": getattr(
            args,
            "lower_target_joints_cache_checker_receipt",
            None,
        ),
        "checker_sha256": getattr(
            args,
            "expected_lower_target_joints_cache_checker_sha256",
            None,
        ),
    }


def validate_activation_args(args: Any) -> bool:
    enabled_value = getattr(args, "use_lower_target_joints_cache", False)
    if type(enabled_value) is not bool:
        raise LowerTargetCacheError(
            "use_lower_target_joints_cache must be an exact boolean"
        )
    enabled = enabled_value
    values = _activation_values(args)
    present = {key: value is not None and str(value) != "" for key, value in values.items()}
    if not enabled:
        if any(present.values()):
            raise LowerTargetCacheError(
                "lower target cache paths/SHAs require "
                "--use_lower_target_joints_cache true"
            )
        return False
    missing = sorted(key for key, is_present in present.items() if not is_present)
    if missing:
        raise LowerTargetCacheError(
            f"enabled lower target cache is missing explicit arguments: {missing}"
        )
    required_args = {
        "dataset": "show_base",
        "formal_stage": "lower",
        "tar_joints": "beat_smplx_lower",
        "train_only": True,
        "pose_length": WINDOW_LENGTH,
        "batch_size": COMPUTE_BATCH_WINDOWS,
    }
    for name, expected in required_args.items():
        observed = getattr(args, name, None)
        if type(observed) is not type(expected) or observed != expected:
            raise LowerTargetCacheError(
                f"lower target cache requires {name}={expected!r}, "
                f"got {observed!r}"
            )
    raw_speakers = getattr(args, "training_speakers", ())
    if type(raw_speakers) not in (list, tuple):
        raise LowerTargetCacheError("training_speakers must be a list or tuple")
    if any(type(value) is not int for value in raw_speakers):
        raise LowerTargetCacheError(
            "training_speakers must contain exact integers"
        )
    speakers = tuple(raw_speakers)
    if speakers != EXPECTED_SPEAKER_IDS:
        raise LowerTargetCacheError(
            "lower target cache is restricted to SHOW All speakers [0,1,2,3]"
        )
    rec_ver_weight = getattr(args, "rec_ver_weight", 0.0)
    if type(rec_ver_weight) not in (int, float) or type(rec_ver_weight) is bool:
        raise LowerTargetCacheError(
            "rec_ver_weight must be an exact finite number"
        )
    if not math.isfinite(rec_ver_weight) or rec_ver_weight <= 0.0:
        raise LowerTargetCacheError(
            "lower target cache requires a finite positive target-joint loss weight"
        )
    for name in (
        "train_path",
        "dataset_summary",
        "lineage_manifest",
        "expected_smplx_asset_sha256",
    ):
        value = getattr(args, name, None)
        if type(value) is not str or not value:
            raise LowerTargetCacheError(
                f"lower target cache requires an explicit {name}"
            )
    return True


def formal_lower_target_cache_receipt(args: Any) -> dict[str, Any] | None:
    """Validate formal artifacts and return a checkpoint-safe small receipt."""

    if not validate_activation_args(args):
        return None
    values = _activation_values(args)
    manifest_path, manifest, manifest_sha = _load_verified_json(
        values["manifest"],
        values["manifest_sha256"],
        "lower target cache manifest",
    )
    validate_manifest_payload(manifest)
    checker_path, checker, checker_sha = _load_verified_json(
        values["checker"],
        values["checker_sha256"],
        "lower target cache checker receipt",
    )
    validate_checker_payload(
        checker,
        manifest_sha256=manifest_sha,
        manifest=manifest,
    )
    torch_module = _require_torch()
    if not torch_module.cuda.is_available():
        raise LowerTargetCacheError(
            "formal lower target cache receipt validation requires CUDA"
        )
    current_device = torch_module.device(
        "cuda",
        torch_module.cuda.current_device(),
    )
    current_inputs = validate_current_inputs_against_manifest(
        args,
        manifest,
        device=current_device,
    )
    repository = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    current_source = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
    }
    source_status = git("status", "--porcelain=v1", "--untracked-files=all")
    if source_status:
        raise LowerTargetCacheError(
            "formal lower target cache consumption requires a clean checkout"
        )
    for key in ("origin", "commit", "tree"):
        if current_source[key] != manifest["source_receipt"][key]:
            raise LowerTargetCacheError(
                f"current source {key} does not match cache production"
            )

    input_cache = Path(values["cache"])
    if input_cache.is_symlink():
        raise LowerTargetCacheError(
            f"lower target cache LMDB must not be a symlink: {input_cache}"
        )
    cache_path = input_cache.resolve()
    if not cache_path.is_dir():
        raise FileNotFoundError(cache_path)
    recorded_cache_path = Path(manifest["lmdb"]["path"]).resolve()
    if cache_path != recorded_cache_path:
        raise LowerTargetCacheError(
            f"cache path {cache_path} != manifest path {recorded_cache_path}"
        )
    for name in ("data.mdb", "lock.mdb"):
        artifact = cache_path / name
        if artifact.is_symlink() or not artifact.is_file():
            raise LowerTargetCacheError(f"invalid cache LMDB artifact: {artifact}")
    data_sha = sha256_file(cache_path / "data.mdb")
    lock_sha = sha256_file(cache_path / "lock.mdb")
    if data_sha != manifest["lmdb"]["data_mdb_sha256"]:
        raise LowerTargetCacheError("cache data.mdb changed after production")
    if lock_sha != manifest["lmdb"]["lock_mdb_sha256"]:
        raise LowerTargetCacheError("cache lock.mdb changed after production")

    receipt = {
        "format": CACHE_FORMAT,
        "cache_version": CACHE_VERSION,
        "cache_path": str(cache_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checker_receipt_path": str(checker_path),
        "checker_receipt_sha256": checker_sha,
        "data_mdb_sha256": data_sha,
        "lock_mdb_sha256": lock_sha,
        "entry_aggregate_sha256": manifest["entry_aggregate_sha256"],
        "entries": EXPECTED_ENTRIES,
        "entry_shape": list(ENTRY_SHAPE),
        "dtype": "<f4",
        "speaker_scope": "All",
        "speaker_ids": list(EXPECTED_SPEAKER_IDS),
        "source_receipt": current_source,
        "current_inputs": current_inputs,
        "exact_once": True,
        "finite": True,
        "torch_equal_checked": True,
        "target_requires_grad": False,
        "target_optimizer_excluded": True,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def attach_lower_target_cache_receipt(
    payload: MutableMapping[str, Any],
    receipt: Mapping[str, Any] | None,
) -> None:
    """Attach the small cache receipt without permitting silent replacement."""

    if receipt is None:
        if RECEIPT_KEY in payload:
            raise LowerTargetCacheError(
                "checkpoint unexpectedly contains a lower target cache receipt"
            )
        return
    frozen = json.loads(canonical_json_bytes(dict(receipt)))
    if RECEIPT_KEY in payload and payload[RECEIPT_KEY] != frozen:
        raise LowerTargetCacheError(
            "refusing to replace a different lower target cache receipt"
        )
    payload[RECEIPT_KEY] = frozen


def verify_lower_target_cache_resume_receipt(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> None:
    if expected is None:
        if RECEIPT_KEY in payload:
            raise LowerTargetCacheError(
                "resume checkpoint was created with a lower target cache"
            )
        return
    if RECEIPT_KEY not in payload:
        raise LowerTargetCacheError(
            "resume checkpoint is missing its lower target cache receipt"
        )
    observed = payload[RECEIPT_KEY]
    if observed != dict(expected):
        raise LowerTargetCacheError(
            "resume checkpoint lower target cache receipt mismatch"
        )


def assert_optimizer_excludes_tensor(
    optimizer: torch.optim.Optimizer,
    tensor: torch.Tensor,
) -> None:
    torch_module = _require_torch()
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):
            if parameter is tensor or (
                isinstance(parameter, torch_module.Tensor)
                and parameter.data_ptr() == tensor.data_ptr()
            ):
                raise LowerTargetCacheError(
                    "lower target cache tensor appears in an optimizer group"
                )


def index_select_frozen_targets(
    tensor: torch.Tensor,
    sample_index: torch.Tensor | Sequence[int],
    *,
    expected_entries: int = EXPECTED_ENTRIES,
) -> torch.Tensor:
    """Select frozen target windows with strict, testable index semantics."""

    torch_module = _require_torch()
    if tuple(tensor.shape) != (
        expected_entries,
        WINDOW_LENGTH,
        JOINT_COUNT,
        COORDINATE_COUNT,
    ):
        raise LowerTargetCacheError(
            f"target tensor shape {tuple(tensor.shape)} is invalid"
        )
    if tensor.dtype != torch_module.float32 or not tensor.is_contiguous():
        raise LowerTargetCacheError(
            "target tensor must be contiguous float32"
        )
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise LowerTargetCacheError("target tensor must be outside autograd")
    indices = torch_module.as_tensor(sample_index)
    if indices.ndim != 1:
        raise LowerTargetCacheError(
            f"sample_index must be rank one, got {tuple(indices.shape)}"
        )
    if indices.dtype != torch_module.int64:
        raise LowerTargetCacheError(
            f"sample_index must be int64, got {indices.dtype}"
        )
    if indices.device.type != "cpu":
        raise LowerTargetCacheError(
            "sample_index must remain on CPU until the cache lookup"
        )
    if indices.numel() == 0:
        raise LowerTargetCacheError("sample_index batch must not be empty")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= expected_entries:
        raise LowerTargetCacheError("sample_index is outside the frozen cache")
    indices = indices.to(device=tensor.device, non_blocking=True)
    with torch_module.no_grad():
        selected = torch_module.index_select(tensor, 0, indices)
    if selected.requires_grad or selected.grad_fn is not None:
        raise AssertionError("target cache lookup unexpectedly entered autograd")
    return selected


class LowerTargetJointsCache:
    """An immutable preloaded target tensor addressed by dataset sample index."""

    def __init__(
        self,
        tensor: torch.Tensor,
        receipt: Mapping[str, Any],
    ) -> None:
        torch_module = _require_torch()
        if tuple(tensor.shape) != (
            EXPECTED_ENTRIES,
            WINDOW_LENGTH,
            JOINT_COUNT,
            COORDINATE_COUNT,
        ):
            raise LowerTargetCacheError(
                f"preloaded cache tensor has invalid shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch_module.float32 or not tensor.is_contiguous():
            raise LowerTargetCacheError(
                "preloaded cache tensor must be contiguous float32"
            )
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise LowerTargetCacheError(
                "preloaded lower target tensor must be outside autograd"
            )
        self._tensor = tensor.detach()
        self._receipt = dict(receipt)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(self._receipt))

    def assert_optimizer_excluded(self, optimizer: torch.optim.Optimizer) -> None:
        assert_optimizer_excludes_tensor(optimizer, self._tensor)

    def index_select(self, sample_index: torch.Tensor | Sequence[int]) -> torch.Tensor:
        return index_select_frozen_targets(
            self._tensor,
            sample_index,
            expected_entries=EXPECTED_ENTRIES,
        )

    @classmethod
    def from_formal_args(
        cls,
        args: Any,
        *,
        optimizer: torch.optim.Optimizer,
        preload_chunk_entries: int = 64,
    ) -> "LowerTargetJointsCache":
        torch_module = _require_torch()
        receipt = formal_lower_target_cache_receipt(args)
        if receipt is None:
            raise LowerTargetCacheError("lower target cache activation is disabled")
        if not torch_module.cuda.is_available():
            raise LowerTargetCacheError("formal lower target preload requires CUDA")
        if preload_chunk_entries <= 0:
            raise ValueError("preload_chunk_entries must be positive")
        device = torch_module.device(
            "cuda",
            torch_module.cuda.current_device(),
        )
        device_name = torch_module.cuda.get_device_name(device)
        if "H200" not in device_name:
            raise LowerTargetCacheError(
                f"formal lower target preload requires H200, got {device_name!r}"
            )

        try:
            import lmdb
        except ImportError as error:
            raise LowerTargetCacheError(
                "python-lmdb is required for lower target cache preload"
            ) from error

        cache_path = Path(receipt["cache_path"])
        env = lmdb.open(
            str(cache_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=64,
            subdir=True,
        )
        aggregate = hashlib.sha256()
        loaded_entries = 0
        try:
            with env.begin(buffers=True) as txn:
                if int(txn.stat()["entries"]) != EXPECTED_ENTRIES:
                    raise LowerTargetCacheError(
                        "runtime cache LMDB does not have exactly 127286 entries"
                    )
            with torch_module.no_grad():
                tensor = torch_module.empty(
                    (
                        EXPECTED_ENTRIES,
                        WINDOW_LENGTH,
                        JOINT_COUNT,
                        COORDINATE_COUNT,
                    ),
                    dtype=torch_module.float32,
                    device=device,
                    requires_grad=False,
                )
                for start in range(0, EXPECTED_ENTRIES, preload_chunk_entries):
                    end = min(start + preload_chunk_entries, EXPECTED_ENTRIES)
                    host = np.empty(
                        (end - start, *ENTRY_SHAPE),
                        dtype=np.float32,
                    )
                    with env.begin(buffers=True) as txn:
                        for offset, index in enumerate(range(start, end)):
                            value = txn.get(cache_key(index))
                            if value is None:
                                raise LowerTargetCacheError(
                                    f"runtime cache misses key {cache_key(index)!r}"
                                )
                            payload = bytes(value)
                            update_entry_aggregate(
                                aggregate,
                                index,
                                payload,
                            )
                            host[offset] = decode_raw_entry(payload)
                            loaded_entries += 1
                    tensor[start:end].copy_(
                        torch_module.from_numpy(host).to(
                            device=device,
                            dtype=torch_module.float32,
                            non_blocking=False,
                        )
                    )
        except BaseException:
            if "tensor" in locals():
                del tensor
            raise
        finally:
            env.close()
        if (
            loaded_entries != EXPECTED_ENTRIES
            or aggregate.hexdigest() != receipt["entry_aggregate_sha256"]
        ):
            if "tensor" in locals():
                del tensor
            raise LowerTargetCacheError(
                "preloaded cache entries do not match the formal aggregate"
            )
        final_data_sha = sha256_file(cache_path / "data.mdb")
        if final_data_sha != receipt["data_mdb_sha256"]:
            del tensor
            raise LowerTargetCacheError(
                "cache data.mdb changed during the preload"
            )
        cache = cls(tensor, receipt)
        cache.assert_optimizer_excluded(optimizer)
        return cache
