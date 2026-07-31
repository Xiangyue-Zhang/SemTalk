#!/usr/bin/env python3
"""Strict TalkSHOW released2 validation authority for SemTalk Base.

This module is the only formal validation contract used by fresh Base
training, validation inference, winner selection, and independent metric
replay. Historical evaluator contracts are intentionally outside this
authority: selection is TalkSHOW body.released2 FGD, followed by one complete
TalkSHOW body/face evaluation of the frozen winner.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

from scripts.show_base import gate_task_space_on_show_v2 as _receipt
from scripts.show_base import selected_prerequisites as _selected_prerequisites


EXPECTED_VAL_CLIPS = 1_715
EXPECTED_AUDIO_SHARDS = 8
TALKSHOW_WINDOW = 88
TALKSHOW_STRIDE = 88
SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}

VAL_INPUTS_FORMAT = "semtalk_show_base_talkshow_val_inputs_v2"
VAL_CANONICAL_SUMMARY_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_summary_v1"
)
VAL_CANONICAL_LINEAGE_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_lineage_v1"
)
VAL_INFERENCE_LINEAGE_FORMAT = (
    "semtalk_show_base_talkshow_val_inference_lineage_v2"
)
FRESH_PIPELINE_FORMAT = "semtalk_show_base_fresh_val_pipeline_v2"
FRESH_PIPELINE_MODE = "show_val_selected_five_prerequisites_v2"
INFERENCE_HELPERS = (
    "_load_canonical_clip",
    "_load_audio_features",
    "_infer_clip",
    "_inference_only_auxiliary_loss_bypass",
    "_inference_auxiliary_loss_bypass_receipt",
    "_output_arrays",
)
VAL_INFERENCE_SOURCE = {
    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
    "entrypoint": "semtalk_base_inference_core.py",
}
FRESH_PREREQUISITE_CONSUMPTION = {
    "base_training_feature_graph": {
        "live_prerequisite_models": [],
        "consumed_precomputed_selected_outputs": [
            "face",
            "hands",
            "upper",
            "lower",
        ],
        "global_model_consumed": False,
    },
    "official_base_inference": {
        "strict_loaded_models": [
            "face",
            "hands",
            "upper",
            "lower",
            "global",
        ],
        "decoded_models": [
            "face",
            "hands",
            "upper",
            "lower",
            "global",
        ],
        "global_translation_reconstruction": True,
    },
}
FRESH_PIPELINE_SOURCE_FILES = (
    "scripts/show_base/__init__.py",
    "scripts/show_base/talkshow_base_val_contract.py",
    "scripts/show_base/base_long_val_contract.py",
    "scripts/show_base/base_fresh_val_orchestrator.py",
    "scripts/show_base/base_fresh_probe_producer.py",
    "scripts/show_base/base_fresh_probe_workload.py",
    "scripts/show_base/published_test_winner_claim.py",
    "scripts/show_base/select_published_base_winner.py",
    "scripts/show_base/run_base_val_inference.py",
    "scripts/show_base/semtalk_base_inference_core.py",
    "scripts/show_base/evaluate_talkshow_show_metrics.py",
    "scripts/show_base/build_talkshow_metric_root.py",
    "scripts/show_base/replay_released2_primary.py",
    "scripts/show_base/build_base_features.py",
    "scripts/show_base/selected_prerequisites.py",
    "scripts/show_base/prerequisite_val_contract.py",
    "scripts/show_base/merge_prerequisite_val_shards.py",
    "scripts/show_base/gate_task_space_on_show_v2.py",
    "scripts/show_base/gate_released_all_speakers_on_show.py",
    "utils/show_base_joints.py",
    "utils/rotation_conversions.py",
    "utils/__init__.py",
    "dataloaders/__init__.py",
    "dataloaders/data_tools.py",
    "models/__init__.py",
    "models/semtalk.py",
    "models/motion_encoder.py",
    "models/motion_representation.py",
    "models/rvq.py",
    "models/encdec.py",
    "models/residual_vq.py",
    "models/quantizer.py",
    "models/resnet.py",
    "models/utils/__init__.py",
    "models/utils/layer.py",
    "models/utils/skeleton.py",
)
_TEST_PATH_LABEL_TOKENS = frozenset(
    {"test", "tests", "testset", "testsets"}
)
BASE_PRODUCER_SOURCE = {
    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
}


class SelectionContractError(RuntimeError):
    """Raised when validation-only selection cannot be proven."""


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_file_snapshot(
    value: Any,
    label: str,
) -> tuple[Path, bytes]:
    if not isinstance(value, (str, os.PathLike)):
        raise SelectionContractError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SelectionContractError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SelectionContractError(f"{label} does not exist: {path}") from error
    if resolved != path:
        raise SelectionContractError(
            f"{label} must be canonical with no symlink ancestor: {path}"
        )
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise SelectionContractError(
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
        file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SelectionContractError(
                f"{label} must be a regular non-symlink file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in fields
        ):
            raise SelectionContractError(
                f"{label} changed while it was read"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise SelectionContractError(
                f"{label} size changed while it was read"
            )
        return path, payload
    except SelectionContractError:
        raise
    except OSError as error:
        raise SelectionContractError(
            f"cannot safely read {label}: {path}"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _safe_directory(value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise SelectionContractError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SelectionContractError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SelectionContractError(f"{label} does not exist: {path}") from error
    if resolved != path:
        raise SelectionContractError(
            f"{label} must be canonical with no symlink ancestor: {path}"
        )
    parts = path.parts
    if not parts or parts[0] != os.sep:
        raise SelectionContractError(
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
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise SelectionContractError(
                f"{label} must be a non-symlink directory"
            )
        return path
    except SelectionContractError:
        raise
    except OSError as error:
        raise SelectionContractError(
            f"cannot safely resolve {label}: {path}"
        ) from error
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def sha256_file(path: Path) -> str:
    _resolved, payload = _safe_file_snapshot(
        path,
        f"SHA-256 input {path}",
    )
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionContractError(
            f"{label} must be a canonical lowercase SHA-256"
        )
    return value


def require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionContractError(
            f"{label} must be a canonical lowercase Git object ID"
        )
    return value


def require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionContractError(f"{label} must be an exact integer")
    return value


def require_exact_keys(
    value: Any,
    keys: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SelectionContractError(f"{label} schema mismatch")
    return value


def reject_forbidden_source_labels(*values: object) -> None:
    """Hard reject withdrawn e30 and forbidden Speaker2 labels/paths."""

    for value in values:
        for component in str(value).replace("\\", "/").split("/"):
            stem = component.rsplit(".", 1)[0]
            if len(stem) in {40, 64} and all(
                character in "0123456789abcdefABCDEF"
                for character in stem
            ):
                continue
            normalized = component.casefold()
            if (
                re.search(r"(^|[^a-z0-9])e[-_]?30([^a-z0-9]|$)", normalized)
                or re.search(
                    r"(^|[^a-z0-9])speaker[-_]?2([^a-z0-9]|$)",
                    normalized,
                )
            ):
                raise SelectionContractError(
                    "withdrawn e30 or forbidden Speaker2 input: "
                    f"{value}"
                )


def require_absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SelectionContractError(f"{label} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        raise SelectionContractError(f"{label} must be absolute")
    reject_forbidden_source_labels(path)
    return path


def require_val_only_path(value: Any, label: str) -> Path:
    path = require_absolute_path(value, label)
    reject_test_path(path, label)
    return path


def reject_test_path(path: Path, label: str) -> None:
    for piece in path.parts:
        tokens = set(re.findall(r"[a-z0-9]+", piece.casefold()))
        if tokens & _TEST_PATH_LABEL_TOKENS:
            raise SelectionContractError(
                f"{label} must not expose a test-labeled path: {path}"
            )


def require_directory(value: Any, label: str) -> Path:
    path = require_val_only_path(value, label)
    resolved = _safe_directory(path, label)
    reject_forbidden_source_labels(resolved)
    reject_test_path(resolved, label)
    return resolved


def canonical_clip_id(source_clip_id: str) -> str:
    """Mirror the final inference output-ID mapping without importing numpy."""

    pieces = source_clip_id.split("/")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        raise SelectionContractError(
            f"invalid canonical SHOW source clip ID: {source_clip_id!r}"
        )
    speaker, _video, sequence = pieces
    if speaker not in SHOW_SPEAKER_IDS:
        raise SelectionContractError(f"unknown SHOW speaker {speaker!r}")
    if (
        "__" in speaker
        or "/" in sequence
        or sequence in {"", ".", ".."}
        or "\x00" in sequence
    ):
        raise SelectionContractError(
            f"unsafe canonical SHOW source clip ID: {source_clip_id!r}"
        )
    return f"{speaker}__{sequence}"


def _regular_file(path: Path, label: str) -> Path:
    reject_forbidden_source_labels(path)
    resolved, _payload = _safe_file_snapshot(path, label)
    reject_forbidden_source_labels(resolved)
    return resolved


def _verified_bytes(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes, str]:
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    resolved, payload = _safe_file_snapshot(path, label)
    reject_forbidden_source_labels(resolved)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise SelectionContractError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return resolved, payload, observed


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return _receipt.strict_json_loads(payload, label)
    except (TypeError, ValueError, RuntimeError) as error:
        raise SelectionContractError(str(error)) from error


def _verified_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    resolved, payload, observed = _verified_bytes(
        path,
        expected_sha256,
        label,
    )
    value = _strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise SelectionContractError(f"{label} must be a JSON object")
    return resolved, value, observed


def _strict_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SelectionContractError(
            f"{label} is not UTF-8: {error}"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_json_bytes(
            line.encode("utf-8"),
            f"{label}:{line_number}",
        )
        if not isinstance(value, dict):
            raise SelectionContractError(
                f"{label}:{line_number} must be a JSON object"
            )
        rows.append(value)
    return rows


def _payload_hash_without(
    payload: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    claimed = require_sha256(payload.get(field), f"{label}.{field}")
    body = dict(payload)
    body.pop(field, None)
    observed = canonical_json_sha256(body)
    if observed != claimed:
        raise SelectionContractError(
            f"{label} payload hash mismatch: {observed} != {claimed}"
        )
    return claimed


def _artifact_fields(
    value: Any,
    label: str,
    *,
    payload_hash: bool = False,
) -> tuple[Path, str, str | None]:
    expected = {"path", "sha256"}
    if payload_hash:
        expected.add("receipt_payload_sha256")
    value = require_exact_keys(value, expected, label)
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise SelectionContractError(f"{label}.path must be nonempty")
    path = Path(raw_path)
    if not path.is_absolute():
        raise SelectionContractError(f"{label}.path must be absolute")
    reject_forbidden_source_labels(path)
    digest = require_sha256(value["sha256"], f"{label}.sha256")
    receipt_hash = (
        require_sha256(
            value["receipt_payload_sha256"],
            f"{label}.receipt_payload_sha256",
        )
        if payload_hash
        else None
    )
    return path, digest, receipt_hash


def _verify_artifact(
    value: Any,
    label: str,
    *,
    payload_hash: bool = False,
) -> tuple[dict[str, Any], Path, bytes]:
    path, digest, receipt_hash = _artifact_fields(
        value,
        label,
        payload_hash=payload_hash,
    )
    resolved, payload, _ = _verified_bytes(path, digest, label)
    receipt = {"path": str(resolved), "sha256": digest}
    if receipt_hash is not None:
        receipt["receipt_payload_sha256"] = receipt_hash
    return receipt, resolved, payload


def _canonical_coverage(
    payload: bytes,
    label: str,
) -> tuple[set[str], dict[str, Any]]:
    rows = _strict_jsonl(payload, label)
    if len(rows) != EXPECTED_VAL_CLIPS:
        raise SelectionContractError(
            f"canonical validation rows {len(rows)} != {EXPECTED_VAL_CLIPS}"
        )
    seen_indices: set[int] = set()
    seen_source_ids: set[str] = set()
    seen_output_ids: set[str] = set()
    ordered: list[tuple[int, str, int]] = []
    ordered_clips: list[dict[str, Any]] = []
    speakers: set[str] = set()
    for row in rows:
        if row.get("split") != "val":
            raise SelectionContractError(
                "canonical selection manifest must contain val rows only"
            )
        index = require_exact_int(
            row.get("global_index"),
            "canonical validation global_index",
        )
        frames = require_exact_int(
            row.get("frames"),
            "canonical validation frames",
        )
        clip_id = row.get("clip_id")
        if (
            index < 0
            or index in seen_indices
            or frames < TALKSHOW_WINDOW
            or not isinstance(clip_id, str)
            or not clip_id
            or clip_id in seen_source_ids
        ):
            raise SelectionContractError(
                f"invalid canonical validation row {clip_id!r}"
            )
        output_id = canonical_clip_id(clip_id)
        if output_id in seen_output_ids:
            raise SelectionContractError(
                f"canonical validation output-ID collision: {output_id}"
            )
        seen_indices.add(index)
        seen_source_ids.add(clip_id)
        seen_output_ids.add(output_id)
        speakers.add(clip_id.split("/", 1)[0])
        ordered.append((index, output_id, frames))
        ordered_clips.append(
            {
                "global_index": index,
                "source_clip_id": clip_id,
                "canonical_clip_id": output_id,
                "frames": frames,
            }
        )
    if [item[0] for item in ordered] != sorted(seen_indices):
        raise SelectionContractError(
            "canonical validation manifest is not in global-index order"
        )
    if speakers != set(SHOW_SPEAKER_IDS):
        raise SelectionContractError(
            "canonical validation coverage is not the four SHOW speakers"
        )
    clip_digest = hashlib.sha256()
    talkshow_window_digest = hashlib.sha256()
    frame_count = 0
    window_count = 0
    uncovered_tail_frames = 0
    for _, output_id, frames in ordered:
        clip_digest.update(output_id.encode("utf-8"))
        clip_digest.update(b"\n")
        starts = tuple(
            range(0, frames - TALKSHOW_WINDOW + 1, TALKSHOW_STRIDE)
        )
        if not starts:
            raise AssertionError("frame lower bound did not provide a window")
        frame_count += frames
        window_count += len(starts)
        uncovered_tail_frames += frames - (
            starts[-1] + TALKSHOW_WINDOW
        )
        talkshow_window_digest.update(output_id.encode("utf-8"))
        talkshow_window_digest.update(b"\0")
        talkshow_window_digest.update(str(frames).encode("ascii"))
        talkshow_window_digest.update(b"\0")
        talkshow_window_digest.update(
            ",".join(str(value) for value in starts).encode("ascii")
        )
        talkshow_window_digest.update(b"\n")
    return seen_source_ids, {
        "split": "val",
        "clip_count": EXPECTED_VAL_CLIPS,
        "frame_count": frame_count,
        "window_count": window_count,
        "uncovered_tail_frames": uncovered_tail_frames,
        "clip_ids_sha256": clip_digest.hexdigest(),
        "talkshow_window_manifest_sha256": talkshow_window_digest.hexdigest(),
        "_ordered_clips": ordered_clips,
    }


def public_val_coverage(coverage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: coverage[key]
        for key in (
            "split",
            "clip_count",
            "frame_count",
            "window_count",
            "uncovered_tail_frames",
            "clip_ids_sha256",
            "talkshow_window_manifest_sha256",
        )
    }


def _audio_coverage(
    manifest_receipts: Sequence[Any],
    summary_receipts: Sequence[Any],
    lineage_receipts: Sequence[Any],
    canonical_ids: set[str],
) -> dict[str, Any]:
    if not (
        len(manifest_receipts)
        == len(summary_receipts)
        == len(lineage_receipts)
        == EXPECTED_AUDIO_SHARDS
    ):
        raise SelectionContractError(
            "validation audio inputs require exactly eight "
            "manifest/summary/lineage receipts"
        )

    manifests: dict[int, dict[str, Any]] = {}
    audio_ids: set[str] = set()
    audio_paths: set[Path] = set()
    for receipt_value in manifest_receipts:
        artifact, resolved, payload = _verify_artifact(
            receipt_value,
            "validation audio manifest",
        )
        reject_test_path(resolved, "validation audio manifest")
        rows = _strict_jsonl(payload, f"validation audio manifest {resolved}")
        shard_ids: set[int] = set()
        for row in rows:
            clip_id = row.get("clip_id")
            shard_id = require_exact_int(
                row.get("shard_id"),
                "validation audio shard_id",
            )
            num_shards = require_exact_int(
                row.get("num_shards"),
                "validation audio num_shards",
            )
            if (
                row.get("format") != "semtalk_show_audio_clip_v1"
                or row.get("split") != "val"
                or num_shards != EXPECTED_AUDIO_SHARDS
                or not isinstance(clip_id, str)
                or not clip_id
                or clip_id in audio_ids
            ):
                raise SelectionContractError(
                    f"invalid validation audio row {clip_id!r}"
                )
            frames = require_exact_int(
                row.get("frames"),
                f"{clip_id} audio frames",
            )
            if (
                frames < TALKSHOW_WINDOW
                or row.get("beat_shape") != [frames, 3]
                or row.get("hubert_shape") != [frames, 1024]
            ):
                raise SelectionContractError(
                    f"{clip_id}: invalid audio feature shapes"
                )
            audio_sha = require_sha256(
                row.get("audio_feature_npz_sha256"),
                f"{clip_id} audio feature SHA",
            )
            audio_path = require_val_only_path(
                row.get("audio_feature_npz"),
                f"{clip_id} audio feature path",
            )
            resolved_audio = _regular_file(
                audio_path,
                f"{clip_id} audio feature",
            )
            reject_test_path(
                resolved_audio,
                f"{clip_id} audio feature",
            )
            if (
                resolved_audio.name
                != f"{hashlib.sha256(clip_id.encode('utf-8')).hexdigest()}.npz"
                or resolved_audio in audio_paths
                or sha256_file(resolved_audio) != audio_sha
            ):
                raise SelectionContractError(
                    f"{clip_id}: audio feature receipt mismatch"
                )
            shard_ids.add(shard_id)
            audio_ids.add(clip_id)
            audio_paths.add(resolved_audio)
        if len(shard_ids) != 1:
            raise SelectionContractError(
                f"{resolved}: audio manifest must contain exactly one shard"
            )
        shard_id = shard_ids.pop()
        if shard_id in manifests:
            raise SelectionContractError(
                f"duplicate validation audio shard {shard_id}"
            )
        manifests[shard_id] = {
            "artifact": artifact,
            "rows": len(rows),
            "path": str(resolved),
        }
    if set(manifests) != set(range(EXPECTED_AUDIO_SHARDS)):
        raise SelectionContractError(
            "validation audio manifests do not cover shards 0..7"
        )
    if audio_ids != canonical_ids or len(audio_ids) != EXPECTED_VAL_CLIPS:
        raise SelectionContractError(
            "validation audio/canonical clip coverage is not exact"
        )

    summaries: dict[int, dict[str, Any]] = {}
    for receipt_value in summary_receipts:
        artifact, resolved, payload = _verify_artifact(
            receipt_value,
            "validation audio summary",
        )
        reject_test_path(resolved, "validation audio summary")
        summary = _strict_json_bytes(
            payload,
            f"validation audio summary {resolved}",
        )
        if not isinstance(summary, dict):
            raise SelectionContractError("audio summary must be an object")
        shard_id = require_exact_int(
            summary.get("shard_id"),
            "validation audio summary shard_id",
        )
        if (
            summary.get("format") != "semtalk_show_audio_summary_v1"
            or summary.get("status") != "complete"
            or summary.get("num_shards") != EXPECTED_AUDIO_SHARDS
            or summary.get("full_split_clips") != EXPECTED_VAL_CLIPS
            or shard_id not in manifests
            or shard_id in summaries
            or summary.get("output_manifest_sha256")
            != manifests[shard_id]["artifact"]["sha256"]
            or summary.get("shard_clips") != manifests[shard_id]["rows"]
        ):
            raise SelectionContractError(
                f"{resolved}: invalid validation audio summary"
            )
        summaries[shard_id] = artifact

    lineages: dict[int, dict[str, Any]] = {}
    for receipt_value in lineage_receipts:
        artifact, resolved, payload = _verify_artifact(
            receipt_value,
            "validation audio lineage",
        )
        reject_test_path(resolved, "validation audio lineage")
        lineage = _strict_json_bytes(
            payload,
            f"validation audio lineage {resolved}",
        )
        if not isinstance(lineage, dict):
            raise SelectionContractError("audio lineage must be an object")
        shard_id = require_exact_int(
            lineage.get("shard_id"),
            "validation audio lineage shard_id",
        )
        protocol = lineage.get("protocol")
        if (
            lineage.get("format") != "semtalk_show_audio_lineage_v1"
            or lineage.get("status") != "complete"
            or lineage.get("num_shards") != EXPECTED_AUDIO_SHARDS
            or lineage.get("full_split_clips") != EXPECTED_VAL_CLIPS
            or lineage.get("shard_clips") != manifests.get(
                shard_id,
                {},
            ).get("rows")
            or lineage.get("output_manifest_sha256")
            != manifests.get(shard_id, {}).get("artifact", {}).get("sha256")
            or not isinstance(protocol, dict)
            or protocol.get("split") != "val"
            or shard_id in lineages
        ):
            raise SelectionContractError(
                f"{resolved}: invalid validation audio lineage"
            )
        lineages[shard_id] = artifact
    if (
        set(summaries) != set(range(EXPECTED_AUDIO_SHARDS))
        or set(lineages) != set(range(EXPECTED_AUDIO_SHARDS))
    ):
        raise SelectionContractError(
            "validation audio summary/lineage coverage is incomplete"
        )
    return {
        "manifests": [manifests[index]["artifact"] for index in range(8)],
        "summaries": [summaries[index] for index in range(8)],
        "lineages": [lineages[index] for index in range(8)],
        "num_shards": EXPECTED_AUDIO_SHARDS,
        "clip_count": len(audio_ids),
        "exact_once": True,
    }


def _validate_val_canonical_receipts(
    *,
    summary_value: Any,
    lineage_value: Any,
    canonical_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_artifact, summary_path, summary_payload = _verify_artifact(
        summary_value,
        "validation canonical summary",
    )
    lineage_artifact, lineage_path, lineage_payload = _verify_artifact(
        lineage_value,
        "validation canonical lineage",
    )
    reject_test_path(summary_path, "validation canonical summary")
    reject_test_path(lineage_path, "validation canonical lineage")
    summary = require_exact_keys(
        _strict_json_bytes(summary_payload, str(summary_path)),
        {
            "format",
            "status",
            "split",
            "test_visible",
            "clip_count",
            "manifest_sha256",
            "lineage_sha256",
            "lineage_contract_sha256",
            "receipt_payload_sha256",
        },
        "validation canonical summary",
    )
    lineage = require_exact_keys(
        _strict_json_bytes(lineage_payload, str(lineage_path)),
        {
            "format",
            "status",
            "split",
            "test_visible",
            "clip_count",
            "manifest_sha256",
            "lineage_contract_sha256",
            "projection",
            "source_receipt",
            "receipt_payload_sha256",
        },
        "validation canonical lineage",
    )
    _payload_hash_without(
        summary,
        "receipt_payload_sha256",
        "validation canonical summary",
    )
    _payload_hash_without(
        lineage,
        "receipt_payload_sha256",
        "validation canonical lineage",
    )
    projection = require_exact_keys(
        lineage["projection"],
        {"operation", "split", "test_rows_materialized"},
        "validation canonical projection",
    )
    source = require_exact_keys(
        lineage["source_receipt"],
        {
            "origin",
            "commit",
            "tree",
            "full_manifest_sha256",
            "full_summary_sha256",
            "full_lineage_sha256",
        },
        "validation canonical source receipt",
    )
    if (
        summary["format"] != VAL_CANONICAL_SUMMARY_FORMAT
        or lineage["format"] != VAL_CANONICAL_LINEAGE_FORMAT
        or summary["status"] != "complete"
        or lineage["status"] != "complete"
        or summary["split"] != "val"
        or lineage["split"] != "val"
        or summary["test_visible"] is not False
        or lineage["test_visible"] is not False
        or require_exact_int(
            summary["clip_count"],
            "validation canonical summary clip_count",
        )
        != EXPECTED_VAL_CLIPS
        or require_exact_int(
            lineage["clip_count"],
            "validation canonical lineage clip_count",
        )
        != EXPECTED_VAL_CLIPS
        or summary["manifest_sha256"] != canonical_manifest_sha256
        or lineage["manifest_sha256"] != canonical_manifest_sha256
        or summary["lineage_sha256"] != lineage_artifact["sha256"]
        or require_sha256(
            summary["lineage_contract_sha256"],
            "validation canonical summary lineage contract SHA",
        )
        != require_sha256(
            lineage["lineage_contract_sha256"],
            "validation canonical lineage contract SHA",
        )
        or projection
        != {
            "operation": "filter_exact_split",
            "split": "val",
            "test_rows_materialized": False,
        }
        or source["origin"] != BASE_PRODUCER_SOURCE["origin"]
    ):
        raise SelectionContractError(
            "canonical val-view summary/lineage binding mismatch"
        )
    require_git_oid(
        source["commit"],
        "canonical val-view source commit",
    )
    require_git_oid(
        source["tree"],
        "canonical val-view source tree",
    )
    for key in (
        "full_manifest_sha256",
        "full_summary_sha256",
        "full_lineage_sha256",
    ):
        require_sha256(source[key], f"canonical val-view source {key}")
    return summary_artifact, lineage_artifact


def validate_val_inputs(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, inputs, file_sha = _verified_json(
        path,
        expected_sha256,
        "Base validation inputs",
    )
    reject_test_path(resolved, "Base validation inputs")
    inputs = require_exact_keys(
        inputs,
        {
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
        },
        "Base validation inputs",
    )
    receipt_payload_sha = _payload_hash_without(
        inputs,
        "receipt_payload_sha256",
        "Base validation inputs",
    )
    if (
        inputs["format"] != VAL_INPUTS_FORMAT
        or inputs["status"] != "frozen"
        or inputs["split"] != "val"
        or inputs["test_visible"] is not False
        or inputs["expected_clip_count"] != EXPECTED_VAL_CLIPS
    ):
        raise SelectionContractError(
            "Base selector accepts only the frozen validation split"
        )
    canonical_artifact, canonical_path, canonical_payload = _verify_artifact(
        inputs["canonical_manifest"],
        "validation canonical manifest",
    )
    reject_test_path(canonical_path, "validation canonical manifest")
    canonical_ids, coverage = _canonical_coverage(
        canonical_payload,
        str(canonical_path),
    )
    if (
        require_sha256(
            inputs["clip_ids_sha256"],
            "validation clip ID SHA",
        )
        != coverage["clip_ids_sha256"]
        or require_sha256(
            inputs["talkshow_window_manifest_sha256"],
            "validation TalkSHOW released2 clip-manifest SHA",
        )
        != coverage["talkshow_window_manifest_sha256"]
    ):
        raise SelectionContractError(
            "validation input coverage digest mismatch"
        )
    canonical_summary, canonical_lineage = _validate_val_canonical_receipts(
        summary_value=inputs["canonical_summary"],
        lineage_value=inputs["canonical_lineage"],
        canonical_manifest_sha256=canonical_artifact["sha256"],
    )
    audio = _audio_coverage(
        inputs["audio_manifests"],
        inputs["audio_summaries"],
        inputs["audio_lineages"],
        canonical_ids,
    )
    artifact = {
        "path": str(resolved),
        "sha256": file_sha,
        "receipt_payload_sha256": receipt_payload_sha,
    }
    return artifact, {
        **coverage,
        "canonical_manifest": canonical_artifact,
        "canonical_summary": canonical_summary,
        "canonical_lineage": canonical_lineage,
        "audio": audio,
    }


def _git_output(
    source_root: Path,
    arguments: Sequence[str],
    *,
    label: str,
    allow_failure: bool = False,
) -> tuple[int, bytes]:
    process = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0 and not allow_failure:
        raise SelectionContractError(
            f"cannot inspect fresh pipeline {label}: "
            f"{process.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return process.returncode, process.stdout


def build_fresh_pipeline_source_receipt(source_root: Path) -> dict[str, Any]:
    """Freeze the single detached official SemTalk source used by Base val."""

    root = require_directory(source_root, "fresh Base source root")
    _, remotes_raw = _git_output(root, ["remote"], label="remote names")
    remotes = [
        line
        for line in remotes_raw.decode("utf-8").splitlines()
        if line
    ]
    _, origin_raw = _git_output(
        root, ["remote", "get-url", "origin"], label="origin fetch URL"
    )
    _, origin_push_raw = _git_output(
        root,
        ["remote", "get-url", "--push", "origin"],
        label="origin push URL",
    )
    _, commit_raw = _git_output(root, ["rev-parse", "HEAD"], label="HEAD")
    _, tree_raw = _git_output(
        root, ["rev-parse", "HEAD^{tree}"], label="tree"
    )
    origin = origin_raw.decode("utf-8").strip()
    origin_push = origin_push_raw.decode("utf-8").strip()
    commit = require_git_oid(commit_raw.decode("ascii").strip(), "source commit")
    tree = require_git_oid(tree_raw.decode("ascii").strip(), "source tree")
    _, status = _git_output(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        label="checkout cleanliness",
    )
    symbolic_rc, _symbolic = _git_output(
        root,
        ["symbolic-ref", "-q", "HEAD"],
        label="detached HEAD",
        allow_failure=True,
    )
    _, branch_raw = _git_output(
        root,
        [
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
        ],
        label="local branches",
    )
    branches = [
        line
        for line in branch_raw.decode("utf-8").splitlines()
        if line
    ]
    if (
        remotes != ["origin"]
        or origin != "git@github.com:Xiangyue-Zhang/SemTalk.git"
        or origin_push != origin
        or status
        or symbolic_rc == 0
        or branches
    ):
        raise SelectionContractError(
            "fresh Base source must be the clean detached official SemTalk "
            "checkout with no local branch at HEAD"
        )
    files: dict[str, dict[str, Any]] = {}
    for relative in FRESH_PIPELINE_SOURCE_FILES:
        path = root / relative
        resolved, payload = _safe_file_snapshot(
            path, f"fresh Base source {relative}"
        )
        if resolved != path:
            raise SelectionContractError(
                f"fresh Base source {relative} escaped its checkout"
            )
        _, tree_entry_raw = _git_output(
            root,
            ["ls-tree", commit, "--", relative],
            label=f"tracked source {relative}",
        )
        tree_entry = tree_entry_raw.decode("utf-8").rstrip("\n")
        match = re.fullmatch(
            rf"(100644|100755) blob ([0-9a-f]{{40}})\t{re.escape(relative)}",
            tree_entry,
        )
        if match is None:
            raise SelectionContractError(
                f"fresh Base source {relative} is not one regular tracked blob"
            )
        mode = match.group(1)
        blob = match.group(2)
        _, committed = _git_output(
            root,
            ["show", f"{commit}:{relative}"],
            label=f"committed source {relative}",
        )
        if committed != payload:
            raise SelectionContractError(
                f"fresh Base source {relative} differs from commit {commit}"
            )
        files[relative] = {
            "path": str(resolved),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "git_mode": mode,
            "git_blob_sha1": blob,
        }
    return {
        "origin": origin,
        "source_root": str(root),
        "commit": commit,
        "tree": tree,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "files": files,
    }


def _fresh_fixed_checkpoints(
    bridge: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    selected = bridge.get("selected")
    if not isinstance(selected, dict) or set(selected) != {
        "face",
        "hands",
        "upper",
        "lower",
        "global",
    }:
        raise SelectionContractError(
            "fresh Base pipeline prerequisite bridge is not five-stage"
        )
    fixed: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for stage in ("face", "hands", "upper", "lower", "global"):
        row = selected[stage]
        checkpoint = row.get("candidate_checkpoint")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise SelectionContractError(
                f"fresh Base fixed {stage} checkpoint schema changed"
            )
        epoch = require_exact_int(row.get("epoch"), f"fixed {stage} epoch")
        updates = require_exact_int(
            row.get("optimizer_updates"),
            f"fixed {stage} optimizer updates",
        )
        updates_per_epoch = require_exact_int(
            row.get("updates_per_epoch"),
            f"fixed {stage} updates per epoch",
        )
        if (
            epoch <= 0
            or updates_per_epoch <= 0
            or updates != epoch * updates_per_epoch
        ):
            raise SelectionContractError(
                f"fresh Base fixed {stage} update topology changed"
            )
        path = str(checkpoint["path"])
        if path in paths:
            raise SelectionContractError(
                "fresh Base five prerequisites reused one checkpoint path"
            )
        paths.add(path)
        fixed[stage] = {
            "stage": stage,
            "path": path,
            "sha256": require_sha256(
                checkpoint["sha256"], f"fixed {stage} checkpoint SHA"
            ),
            "bytes": require_exact_int(
                checkpoint["bytes"], f"fixed {stage} checkpoint bytes"
            ),
            "source": "show_val_selected_v1",
            "selection_split": "val",
            "test_visible": False,
            "epoch": epoch,
            "optimizer_updates": updates,
            "updates_per_epoch": updates_per_epoch,
            "candidate_audit_sha256": require_sha256(
                row.get("candidate_audit_sha256"),
                f"fixed {stage} candidate audit SHA",
            ),
            "selection_metric": row.get("selection_metric"),
            "measurement_receipt": dict(row.get("measurement_receipt", {})),
        }
    return fixed


def build_fresh_pipeline_payload(
    *,
    source_root: Path,
    prerequisite_selection: Path,
    expected_prerequisite_selection_sha256: str,
) -> dict[str, Any]:
    """Build the fresh Base pipeline from the five selected SHOW models."""

    bridge = _selected_prerequisites.load_selected_prerequisites(
        prerequisite_selection,
        require_sha256(
            expected_prerequisite_selection_sha256,
            "fresh pipeline prerequisite selection SHA",
        ),
    )
    selection_path, selection_payload = _safe_file_snapshot(
        prerequisite_selection, "fresh pipeline prerequisite selection"
    )
    selection_artifact = {
        "path": str(selection_path),
        "sha256": hashlib.sha256(selection_payload).hexdigest(),
        "bytes": len(selection_payload),
        "receipt_payload_sha256": bridge["selection"][
            "receipt_payload_sha256"
        ],
    }
    source = build_fresh_pipeline_source_receipt(source_root)
    files = source["files"]
    payload = {
        "format": FRESH_PIPELINE_FORMAT,
        "status": "frozen",
        "split": "val",
        "test_visible": False,
        "mode": FRESH_PIPELINE_MODE,
        "base_candidate_variable_only": True,
        "source": {
            key: source[key]
            for key in (
                "origin",
                "source_root",
                "commit",
                "tree",
                "clean",
                "detached",
                "local_branches_at_commit",
            )
        },
        "source_closure": files,
        "inference_entrypoint": files[
            "scripts/show_base/run_base_val_inference.py"
        ],
        "inference_helper": files[
            "scripts/show_base/semtalk_base_inference_core.py"
        ],
        "generator_module": "models.semtalk.semtalk_base",
        "prerequisite_consumption": FRESH_PREREQUISITE_CONSUMPTION,
        "prerequisite_selection": selection_artifact,
        "fixed_checkpoints": _fresh_fixed_checkpoints(bridge),
    }
    payload["receipt_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_fresh_pipeline(
    path: Path,
    expected_sha256: str,
    *,
    expected_prerequisite_selection: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freshly replay the five-SHOW-checkpoint validation pipeline."""

    resolved, pipeline, file_sha = _verified_json(
        path, expected_sha256, "fresh Base validation pipeline"
    )
    reject_test_path(resolved, "fresh Base validation pipeline")
    pipeline = require_exact_keys(
        pipeline,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "mode",
            "base_candidate_variable_only",
            "source",
            "source_closure",
            "inference_entrypoint",
            "inference_helper",
            "generator_module",
            "prerequisite_consumption",
            "prerequisite_selection",
            "fixed_checkpoints",
            "receipt_payload_sha256",
        },
        "fresh Base validation pipeline",
    )
    payload_sha = _payload_hash_without(
        pipeline,
        "receipt_payload_sha256",
        "fresh Base validation pipeline",
    )
    if (
        pipeline["format"] != FRESH_PIPELINE_FORMAT
        or pipeline["status"] != "frozen"
        or pipeline["split"] != "val"
        or pipeline["test_visible"] is not False
        or pipeline["mode"] != FRESH_PIPELINE_MODE
        or pipeline["base_candidate_variable_only"] is not True
        or pipeline["generator_module"] != "models.semtalk.semtalk_base"
        or pipeline["prerequisite_consumption"]
        != FRESH_PREREQUISITE_CONSUMPTION
    ):
        raise SelectionContractError(
            "fresh Base validation pipeline identity changed"
        )
    live_source = build_fresh_pipeline_source_receipt(
        Path(str(pipeline.get("source", {}).get("source_root", "")))
    )
    expected_source_public = {
        key: live_source[key]
        for key in (
            "origin",
            "source_root",
            "commit",
            "tree",
            "clean",
            "detached",
            "local_branches_at_commit",
        )
    }
    if (
        pipeline["source"] != expected_source_public
        or pipeline["source_closure"] != live_source["files"]
        or pipeline["inference_entrypoint"]
        != live_source["files"][
            "scripts/show_base/run_base_val_inference.py"
        ]
        or pipeline["inference_helper"]
        != live_source["files"][
            "scripts/show_base/semtalk_base_inference_core.py"
        ]
    ):
        raise SelectionContractError(
            "fresh Base validation pipeline mixed source checkouts"
        )
    if expected_source is not None:
        for key in ("origin", "commit", "tree"):
            if pipeline["source"].get(key) != expected_source.get(key):
                raise SelectionContractError(
                    "fresh Base pipeline source differs from run source"
                )
    selection = pipeline["prerequisite_selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "path",
        "sha256",
        "bytes",
        "receipt_payload_sha256",
    }:
        raise SelectionContractError(
            "fresh Base pipeline prerequisite selection schema changed"
        )
    bridge = _selected_prerequisites.load_selected_prerequisites(
        selection["path"],
        require_sha256(selection["sha256"], "pipeline prerequisite SHA"),
    )
    selection_path, selection_payload = _safe_file_snapshot(
        selection["path"], "pipeline prerequisite selection"
    )
    canonical_selection = {
        "path": str(selection_path),
        "sha256": hashlib.sha256(selection_payload).hexdigest(),
        "bytes": len(selection_payload),
        "receipt_payload_sha256": bridge["selection"][
            "receipt_payload_sha256"
        ],
    }
    if selection != canonical_selection:
        raise SelectionContractError(
            "fresh Base pipeline prerequisite selection changed"
        )
    if expected_prerequisite_selection is not None and any(
        selection.get(key) != expected_prerequisite_selection.get(key)
        for key in (
            "path",
            "sha256",
            "bytes",
            "receipt_payload_sha256",
        )
    ):
        raise SelectionContractError(
            "fresh Base pipeline uses another prerequisite selection"
        )
    fixed = _fresh_fixed_checkpoints(bridge)
    if pipeline["fixed_checkpoints"] != fixed:
        raise SelectionContractError(
            "fresh Base pipeline fixed five differ from selected SHOW winners"
        )
    return {
        "path": str(resolved),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
    }, pipeline


def _validate_output_file_receipt(
    value: Any,
    *,
    expected_directory: Path,
    expected_filename: str,
    label: str,
) -> dict[str, Any]:
    value = require_exact_keys(
        value,
        {"path", "sha256", "bytes"},
        label,
    )
    path = require_val_only_path(value["path"], f"{label}.path")
    resolved = _regular_file(path, label)
    reject_test_path(resolved, label)
    if (
        resolved.parent != expected_directory
        or resolved.name != expected_filename
    ):
        raise SelectionContractError(f"{label} path mismatch")
    size = require_exact_int(value["bytes"], f"{label}.bytes")
    digest = require_sha256(value["sha256"], f"{label}.sha256")
    if (
        size <= 0
        or resolved.stat().st_size != size
        or sha256_file(resolved) != digest
    ):
        raise SelectionContractError(
            f"{label} bytes do not match the bound output artifact"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "bytes": size,
    }


def validate_val_inference_lineage(
    path: Path,
    expected_sha256: str,
    *,
    epoch: int,
    expected_candidate: Mapping[str, Any],
    val_inputs_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
    expected_coverage: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one candidate's exact val output manifest without importing torch."""

    resolved, lineage, file_sha = _verified_json(
        path,
        expected_sha256,
        f"epoch {epoch} validation inference lineage",
    )
    reject_test_path(
        resolved,
        f"epoch {epoch} validation inference lineage",
    )
    lineage = require_exact_keys(
        lineage,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "epoch",
            "candidate_checkpoint",
            "val_inputs_receipt",
            "pipeline_receipt",
            "prediction_dir",
            "ground_truth_dir",
            "final_manifest",
            "clip_manifest",
            "clip_count",
            "frame_count",
            "window_count",
            "uncovered_tail_frames",
            "clip_ids_sha256",
            "talkshow_window_manifest_sha256",
            "prediction_files",
            "ground_truth_files",
            "exact_once",
            "finite",
            "receipt_payload_sha256",
        },
        f"epoch {epoch} validation inference lineage",
    )
    payload_sha = _payload_hash_without(
        lineage,
        "receipt_payload_sha256",
        f"epoch {epoch} validation inference lineage",
    )
    lineage_epoch = require_exact_int(
        lineage["epoch"],
        "validation inference lineage epoch",
    )
    candidate_path, candidate_sha, _ = _artifact_fields(
        lineage["candidate_checkpoint"],
        f"epoch {epoch} lineage candidate checkpoint",
    )
    if (
        lineage["format"] != VAL_INFERENCE_LINEAGE_FORMAT
        or lineage["status"] != "complete"
        or lineage["split"] != "val"
        or lineage["test_visible"] is not False
        or lineage_epoch != epoch
        or candidate_path.resolve() != Path(expected_candidate["path"])
        or candidate_sha != expected_candidate["sha256"]
        or lineage["val_inputs_receipt"] != val_inputs_artifact
        or lineage["pipeline_receipt"] != pipeline_artifact
        or lineage["exact_once"] is not True
        or lineage["finite"] is not True
    ):
        raise SelectionContractError(
            f"epoch {epoch} inference lineage is not candidate-bound val-only"
        )

    prediction_dir = require_directory(
        lineage["prediction_dir"],
        f"epoch {epoch} prediction directory",
    )
    ground_truth_dir = require_directory(
        lineage["ground_truth_dir"],
        f"epoch {epoch} ground-truth directory",
    )
    final_artifact, final_path, final_payload = _verify_artifact(
        lineage["final_manifest"],
        f"epoch {epoch} final inference manifest",
    )
    clip_artifact, clip_path, clip_payload = _verify_artifact(
        lineage["clip_manifest"],
        f"epoch {epoch} canonical evaluation clip manifest",
    )
    reject_test_path(final_path, "final inference manifest")
    reject_test_path(clip_path, "canonical evaluation clip manifest")
    if final_path.name != "final_manifest.jsonl":
        raise SelectionContractError("final inference manifest basename mismatch")
    if clip_path.name != "diffsheg_eval_clip_ids.txt":
        raise SelectionContractError("DiffSHEG clip manifest basename mismatch")

    expected_rows = expected_coverage.get("_ordered_clips")
    if (
        not isinstance(expected_rows, list)
        or len(expected_rows) != EXPECTED_VAL_CLIPS
    ):
        raise SelectionContractError("canonical val row coverage is unavailable")
    rows = _strict_jsonl(
        final_payload,
        f"epoch {epoch} final inference manifest {final_path}",
    )
    if len(rows) != EXPECTED_VAL_CLIPS:
        raise SelectionContractError(
            f"epoch {epoch} final inference rows must equal "
            f"{EXPECTED_VAL_CLIPS}"
        )
    prediction_paths: set[str] = set()
    ground_truth_paths: set[str] = set()
    row_keys = {
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
    for row_number, (raw_row, canonical_row) in enumerate(
        zip(rows, expected_rows),
    ):
        row = require_exact_keys(
            raw_row,
            row_keys,
            f"epoch {epoch} inference row {row_number}",
        )
        output_id = canonical_row["canonical_clip_id"]
        if (
            require_exact_int(
                row["global_index"],
                f"epoch {epoch} inference row global_index",
            )
            != canonical_row["global_index"]
            or row["split"] != "val"
            or row["source_clip_id"] != canonical_row["source_clip_id"]
            or row["canonical_clip_id"] != output_id
            or require_exact_int(
                row["frames"],
                f"epoch {epoch} inference row frames",
            )
            != canonical_row["frames"]
            or require_exact_int(
                row["epoch"],
                f"epoch {epoch} inference row candidate epoch",
            )
            != epoch
            or row["candidate_checkpoint_sha256"]
            != expected_candidate["sha256"]
        ):
            raise SelectionContractError(
                f"epoch {epoch} inference row {row_number} "
                "does not match canonical val/candidate lineage"
            )
        prediction = _validate_output_file_receipt(
            row["prediction"],
            expected_directory=prediction_dir,
            expected_filename=f"res_{output_id}.npz",
            label=f"epoch {epoch} {output_id} prediction",
        )
        ground_truth = _validate_output_file_receipt(
            row["ground_truth"],
            expected_directory=ground_truth_dir,
            expected_filename=f"gt_{output_id}.npz",
            label=f"epoch {epoch} {output_id} ground truth",
        )
        if (
            prediction["path"] in prediction_paths
            or ground_truth["path"] in ground_truth_paths
        ):
            raise SelectionContractError(
                f"epoch {epoch} inference file coverage is not exact-once"
            )
        prediction_paths.add(prediction["path"])
        ground_truth_paths.add(ground_truth["path"])

    for directory, expected_paths, role in (
        (prediction_dir, prediction_paths, "prediction"),
        (ground_truth_dir, ground_truth_paths, "ground-truth"),
    ):
        children = list(directory.iterdir())
        actual_paths = {
            str(child.resolve())
            for child in children
            if child.is_file() and not child.is_symlink()
        }
        if (
            len(children) != EXPECTED_VAL_CLIPS
            or actual_paths != expected_paths
        ):
            raise SelectionContractError(
                f"epoch {epoch} {role} directory is not the exact "
                "1,715-file cover"
            )

    expected_clip_payload = "".join(
        f"{row['canonical_clip_id']}\n" for row in expected_rows
    ).encode("utf-8")
    public_coverage = public_val_coverage(expected_coverage)
    counts = {
        key: require_exact_int(
            lineage[key],
            f"validation inference lineage {key}",
        )
        for key in (
            "clip_count",
            "frame_count",
            "window_count",
            "uncovered_tail_frames",
        )
    }
    if (
        clip_payload != expected_clip_payload
        or clip_artifact["sha256"] != public_coverage["clip_ids_sha256"]
        or counts
        != {
            key: public_coverage[key]
            for key in (
                "clip_count",
                "frame_count",
                "window_count",
                "uncovered_tail_frames",
            )
        }
        or require_exact_int(
            lineage["prediction_files"],
            "validation inference prediction_files",
        )
        != EXPECTED_VAL_CLIPS
        or require_exact_int(
            lineage["ground_truth_files"],
            "validation inference ground_truth_files",
        )
        != EXPECTED_VAL_CLIPS
        or lineage["clip_ids_sha256"]
        != public_coverage["clip_ids_sha256"]
        or lineage["talkshow_window_manifest_sha256"]
        != public_coverage["talkshow_window_manifest_sha256"]
    ):
        raise SelectionContractError(
            f"epoch {epoch} inference lineage does not exactly cover "
            "the 1,715 canonical val clips"
        )
    artifact = {
        "path": str(resolved),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
    }
    return artifact, {
        "prediction_dir": str(prediction_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "final_manifest": final_artifact,
        "clip_manifest": clip_artifact,
        "coverage": public_coverage,
    }


# Formal consumers use the neutral name; there is only one accepted pipeline.
validate_pipeline = validate_fresh_pipeline
