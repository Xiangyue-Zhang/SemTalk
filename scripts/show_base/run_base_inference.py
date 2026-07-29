#!/usr/bin/env python3
"""Run frozen SemTalk Base-only inference on the canonical SHOW test split.

This entry point deliberately does not instantiate a trainer.  It loads exactly
the formal Base checkpoint, the four formal RVQ checkpoints, and the formal
global/root VAE checkpoint.  SemGate, Sparse, Speaker2 remapping, CLIP,
emotion, semantic, ASR, TextGrid, and vocabulary components are forbidden.

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
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import uuid
import zipfile

import numpy as np


# Formal source cleanliness is rechecked at completion; local model imports must
# therefore never create untracked bytecode inside the immutable source tree.
sys.dont_write_bytecode = True


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
BASE_CANDIDATE_AUDIT_KEYS = {
    "format",
    "formal_stage",
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
FORMAL_UPDATES_PER_EPOCH = 1_989
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    resolved = _resolved_regular_file(path, label)
    expected = _require_sha256(expected, f"{label} expected SHA")
    payload = resolved.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise InferenceContractError(
            f"{label} SHA mismatch: {observed} != {expected} ({resolved})"
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
            "resample": "native_to_true_16000hz_before_processor",
            "processor_sampling_rate": 16000,
            "released_code_difference": (
                "The public loader decodes with librosa's 22050 Hz default "
                "and passes that waveform to a processor declared as 16000 Hz. "
                "This run corrects that sample-rate mismatch."
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
    path: Path,
) -> None:
    if (
        set(audit) != MODEL_V2_AUDIT_KEYS
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
        != 127_309
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
        != 127_309
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
            "train_samples": 127_309,
            "updates_per_epoch": FORMAL_UPDATES_PER_EPOCH,
            "optimizer_updates": expected_optimizer_updates,
        },
    }


def _checkpoint_payload_and_receipt(
    path: Path,
    *,
    formal_stage: str,
    expected_sha256: str,
    expected_training_lineage_sha256: str,
    status_path: Path,
    expected_source_receipt: Mapping[str, Any],
    expected_dataset_summary_sha256: str,
    expected_data_mdb_sha256: str,
    base_candidate_manifest_path: Path | None = None,
    expected_base_candidate_manifest_sha256: str | None = None,
    expected_base_formal_status_sha256: str | None = None,
    expected_base_final_checkpoint_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    dataset_receipt = status.get("dataset_receipt")
    status_final_path = _resolved_regular_file(
        Path(str(status.get("final_checkpoint", ""))),
        f"{formal_stage} status final checkpoint",
    )
    if (
        status.get("status") != "complete"
        or status.get("formal_stage") != formal_stage
        or _require_exact_int(status.get("world_size"), "world_size") != 1
        or _require_exact_int(status.get("epochs"), "epochs")
        != expected_epochs
        or _require_exact_int(
            status.get("completed_epochs"),
            "completed_epochs",
        )
        != expected_epochs
        or _require_exact_int(status.get("train_samples"), "train_samples")
        != 127_309
        or _require_exact_int(
            status.get("updates_per_epoch"),
            "updates_per_epoch",
        )
        != 1_989
        or _require_exact_int(
            status.get("optimizer_updates"),
            "optimizer_updates",
        )
        != expected_epochs * 1_989
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
        != 127_309
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
            optimizer_updates=expected_epochs * FORMAL_UPDATES_PER_EPOCH,
            base_candidate_manifest=(
                candidate_manifest_receipt
                if formal_stage == "base"
                else None
            ),
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
            "train_samples": 127_309,
            "updates_per_epoch": 1_989,
            "optimizer_updates": expected_epochs * 1_989,
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
    base.load_state_dict(
        _normalize_data_parallel_state(
            base_payload["model_state"],
            Path(receipts["base"]["path"]),
        ),
        strict=True,
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
        model = RVQVAE(SimpleNamespace(vae_test_dim=dimension)).to(args.device)
        model.load_state_dict(
            _normalize_data_parallel_state(
                payload["model_state"],
                Path(receipts[name]["path"]),
            ),
            strict=True,
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
    global_motion.load_state_dict(
        _normalize_data_parallel_state(
            global_payload["model_state"],
            Path(receipts["global"]["path"]),
        ),
        strict=True,
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
    source_receipt = _source_receipt(args)
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
            expected_source_commit=args.expected_source_commit,
            expected_source_tree=args.expected_source_tree,
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
        != 127_309
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
        != 127_309
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
    source_triplet = {
        key: source_receipt[key] for key in ("origin", "commit", "tree")
    }
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
        != source_triplet
        or {
            key: representation_lineage_payload.get(
                "source_receipt", {}
            ).get(key)
            for key in ("origin", "commit", "tree")
        }
        != source_triplet
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
        record_audit_source = record.get("audit", {}).get("source_receipt")
        if (
            not isinstance(record, dict)
            or record.get("formal_stage") != stage
            or Path(str(record.get("path", ""))).resolve() != expected_path
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
            != source_triplet
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
    accepted_training_lineages = {
        stage: (
            base_training_lineage_sha
            if stage == "base"
            else representation_training_lineage_sha
        )
        for stage in CHECKPOINT_STAGES
    }
    checkpoint_validation = {
        stage: {
            "expected_training_lineage_sha256": (
                base_training_lineage_sha
                if stage == "base"
                else representation_training_lineage_sha
            ),
            "status_path": _resolved_regular_file(
                getattr(args, f"{stage}_status_json"),
                f"{stage} formal training status",
            ),
            "expected_source_receipt": source_receipt,
            "expected_dataset_summary_sha256": (
                base_training_summary_sha
                if stage == "base"
                else representation_training_lineage_sha
            ),
            "expected_data_mdb_sha256": (
                base_data_sha
                if stage == "base"
                else representation_data_sha
            ),
        }
        for stage in CHECKPOINT_STAGES
    }
    checkpoint_validation["base"].update(
        {
            "base_candidate_manifest_path": args.base_candidate_manifest,
            "expected_base_candidate_manifest_sha256": (
                args.expected_base_candidate_manifest_sha256
            ),
            "expected_base_formal_status_sha256": (
                args.expected_base_formal_status_sha256
            ),
            "expected_base_final_checkpoint_sha256": (
                args.expected_base_final_checkpoint_sha256
            ),
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
    return {
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
        str(inputs["base_training_summary_path"]): inputs[
            "base_training_summary_sha256"
        ],
        **inputs["audio_manifest_sha256"],
        **inputs["audio_summary_sha256"],
        **inputs["audio_lineage_sha256"],
    }
    for receipt in inputs["training_lineage_manifest_sha256"].values():
        fixed_files[str(receipt["path"])] = str(receipt["sha256"])
    for path_value, expected_sha in fixed_files.items():
        _verify_file_sha(
            _resolved_regular_file(path_value, "frozen inference input"),
            str(expected_sha),
            "frozen inference input",
        )
    for stage, receipt in checkpoint_receipts.items():
        _verify_file_sha(
            _resolved_regular_file(receipt["path"], f"{stage} checkpoint"),
            str(receipt["sha256"]),
            f"{stage} checkpoint",
        )
        _verify_file_sha(
            _resolved_regular_file(
                receipt["formal_training_status"],
                f"{stage} formal training status",
            ),
            str(receipt["formal_training_status_sha256"]),
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
        required=True,
        help="Frozen lineage bound only to the formal Base checkpoint.",
    )
    parser.add_argument(
        "--base-training-summary-json",
        type=Path,
        required=True,
        help="Frozen Base LMDB summary bound to its training lineage.",
    )
    parser.add_argument(
        "--representation-training-lineage-manifest",
        type=Path,
        required=True,
        help=(
            "Frozen lineage bound only to face/upper/hands/lower/global "
            "representation checkpoints."
        ),
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
            required=True,
        )
    parser.add_argument(
        "--base-candidate-manifest",
        type=Path,
        required=True,
        help=(
            "Immutable manifest v2 proving that --base-checkpoint is one of "
            "the 40 formal Base candidates."
        ),
    )
    parser.add_argument(
        "--expected-base-candidate-manifest-sha256",
        required=True,
        help="Expected SHA-256 for --base-candidate-manifest.",
    )
    parser.add_argument(
        "--expected-base-formal-status-sha256",
        required=True,
        help=(
            "External expected SHA-256 for the complete Base formal-status "
            "JSON."
        ),
    )
    parser.add_argument(
        "--expected-base-final-checkpoint-sha256",
        required=True,
        help=(
            "External expected SHA-256 for the completed final Base "
            "checkpoint."
        ),
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
    args.expected_base_candidate_manifest_sha256 = _require_sha256(
        args.expected_base_candidate_manifest_sha256,
        "--expected-base-candidate-manifest-sha256",
    )
    args.expected_base_formal_status_sha256 = _require_sha256(
        args.expected_base_formal_status_sha256,
        "--expected-base-formal-status-sha256",
    )
    args.expected_base_final_checkpoint_sha256 = _require_sha256(
        args.expected_base_final_checkpoint_sha256,
        "--expected-base-final-checkpoint-sha256",
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
    args.output_root = args.output_root.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = finalize(args) if args.finalize else run_shard(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
