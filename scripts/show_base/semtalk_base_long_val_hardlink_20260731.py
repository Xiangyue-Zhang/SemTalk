#!/usr/bin/env python3
"""Publish and audit immutable flat validation views without copying NPZ data.

The GPU shard producer writes one regular prediction and ground-truth NPZ per
validation clip.  This helper validates the complete 16-shard exact-once cover
and publishes a flat DiffSHEG-compatible directory by creating same-filesystem
hardlinks to those already-audited files.  No metric math is implemented here.

The ``gate-e40`` command proves the optimization against the frozen legacy e40
validation result.  It requires all 3,430 NPZ files and the DiffSHEG FGD scalar
to be exactly equal.  Gate artifacts are explicitly selection-ineligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid


SCHEDULE_FORMAT = "semtalk_show_base_long_schedule_v1"
SCHEDULE_SHA256 = (
    "013f8ade256f20579f9d545ac44da681238ed56af1cb687fc6fe9356c0bbefa3"
)
CANDIDATE_EPOCHS = (
    1,
    2,
    4,
    8,
    16,
    32,
    40,
    50,
    60,
    70,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    240,
    280,
    320,
    360,
    400,
)
VALIDATION_WAVES = (
    (1, 2, 4, 8),
    (16, 32, 40, 50),
    (60, 70, 80, 100),
    (120, 140, 160, 180),
    (200, 240, 280, 320),
    (360, 400),
)
EXPECTED_CLIPS = 1_715
EXPECTED_NUM_SHARDS = 16
SHARD_FORMAT = "semtalk_show_base_official_adapt_val_inference_shard_v1"
ASSIGNMENT = "canonical_position_modulo_num_shards"
VIEW_FORMAT = "semtalk_show_base_long_val_hardlink_view_v1"
GATE_FORMAT = "semtalk_show_base_long_e40_hardlink_equivalence_gate_v1"
CLIP_MANIFEST_SHA256 = (
    "f6ab4334c13f461b99b5d6f8024fcd270ccb812724a8e83f2e5f12c0c57c1fee"
)
OLD_E40_CHECKPOINT_SHA256 = (
    "027523157e81291082243b93e3b8cd57adba57e421fd2862dd6db83feb3d3e6d"
)
OLD_E40_LINEAGE_SHA256 = (
    "3d5091e086b14a65a5e50ca5ff80e9595e6d06016297307a70ddd3917c74f7c9"
)
OLD_E40_FINAL_MANIFEST_SHA256 = (
    "f18ccf81213aa01b80607f4df65272d2679e168114fdf89e6a2e4cde64dca717"
)
OLD_E40_FGD_REPORT_SHA256 = (
    "248653e7582fa5529c30e33cd33bf7791836b197ecad7b851352d021d3586328"
)
OLD_E40_FGD = 0.015321176277549198
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_COMPONENT = re.compile(
    r"(^|[^a-z0-9])(?:e(?:poch)?[-_]?30|speaker[-_]?2|semgate|sparse)"
    r"([^a-z0-9]|$)",
    re.IGNORECASE,
)


class ContractError(RuntimeError):
    """Raised when the lossless publication contract cannot be proved."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def payload_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_payload_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body).rstrip(b"\n")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ContractError(f"{label} is not a canonical SHA-256")
    return value


def reject_forbidden(value: object, label: str) -> None:
    text = str(value).replace("\\", "/")
    for component in text.split("/"):
        lowered = component.casefold()
        if "test" in lowered or FORBIDDEN_COMPONENT.search(lowered):
            raise ContractError(f"{label} has a forbidden component: {value}")


def absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ContractError(f"{label} must be a path")
    path = Path(value)
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute: {path}")
    reject_forbidden(path, label)
    return path


def regular_file(value: object, label: str) -> Path:
    path = absolute_path(value, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise ContractError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ContractError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    reject_forbidden(resolved, label)
    return resolved


def directory(value: object, label: str) -> Path:
    path = absolute_path(value, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise ContractError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ContractError(
            f"{label} must be a regular non-symlink directory: {path}"
        )
    resolved = path.resolve(strict=True)
    reject_forbidden(resolved, label)
    return resolved


def verified_file(value: object, expected_sha256: str, label: str) -> Path:
    expected = require_sha256(expected_sha256, f"{label} expected SHA")
    path = regular_file(value, label)
    observed = sha256_file(path)
    if observed != expected:
        raise ContractError(f"{label} SHA mismatch: {observed} != {expected}")
    return path


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON token {token!r} in {path}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_inside(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_new_json(path_value: object, value: Mapping[str, Any]) -> Path:
    path = absolute_path(path_value, "output receipt")
    parent = directory(path.parent, "output receipt parent")
    output = parent / path.name
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.partial-", suffix=".json", dir=parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {output}") from None
        fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_schedule(path_value: object, expected_sha256: str) -> tuple[
    Path, dict[str, Any]
]:
    if expected_sha256 != SCHEDULE_SHA256:
        raise ContractError("launcher only accepts the frozen schedule SHA")
    path = verified_file(path_value, expected_sha256, "long schedule")
    value = strict_json(path)
    selection = value.get("selection", {})
    training = value.get("training", {})
    initialization = value.get("initialization", {})
    if (
        value.get("format") != SCHEDULE_FORMAT
        or value.get("scope") != "SemTalk Base only"
        or value.get("target_dataset") != "SHOW"
        or value.get("target_speaker_scope") != "All"
        or tuple(value.get("candidate_epochs", ())) != CANDIDATE_EPOCHS
        or tuple(tuple(wave) for wave in value.get("validation_waves", ()))
        != VALIDATION_WAVES
        or value.get("trajectory_anchor_epochs") != [1, 2, 4, 8, 16, 32, 40]
        or training.get("total_epochs") != 400
        or training.get("world_size") != 8
        or training.get("updates_per_epoch") != 248
        or training.get("seed") != 43
        or training.get("vq_models_in_training_graph") is not False
        or initialization.get("checkpoint_sha256")
        != "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
        or initialization.get("forbidden_epoch") != 30
        or initialization.get("forbidden_sources")
        != ["Speaker2", "SemGate", "Sparse"]
        or selection
        != {
            "split": "val",
            "test_visible": False,
            "ordering": ["FGD", "epoch"],
            "direction": ["min", "min"],
            "test_runs": 1,
        }
    ):
        raise ContractError("long schedule does not match the frozen protocol")
    return path, value


def read_clip_ids(path_value: object, expected_sha256: str) -> tuple[
    Path, tuple[str, ...], bytes
]:
    if expected_sha256 != CLIP_MANIFEST_SHA256:
        raise ContractError("unexpected validation clip-manifest SHA")
    path = verified_file(path_value, expected_sha256, "validation clip manifest")
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise ContractError("clip manifest must end in exactly one newline")
    try:
        rows = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as error:
        raise ContractError("clip manifest is not UTF-8") from error
    if len(rows) != EXPECTED_CLIPS or len(set(rows)) != EXPECTED_CLIPS:
        raise ContractError("clip manifest is not the exact 1,715-clip cover")
    for clip_id in rows:
        if (
            not clip_id
            or clip_id != clip_id.strip()
            or "/" in clip_id
            or "\\" in clip_id
            or "\x00" in clip_id
        ):
            raise ContractError(f"unsafe canonical clip ID: {clip_id!r}")
        reject_forbidden(clip_id, "canonical clip ID")
    return path, tuple(rows), payload


def validate_preflight(path_value: object, expected_sha256: str) -> tuple[
    Path, dict[str, Any], dict[str, Any]
]:
    path = verified_file(path_value, expected_sha256, "validation preflight")
    value = strict_json(path)
    claimed = require_sha256(
        value.get("receipt_payload_sha256"), "preflight payload SHA"
    )
    if (
        payload_sha256(value) != claimed
        or value.get("status") != "complete"
        or value.get("split") != "val"
        or value.get("test_visible") is not False
    ):
        raise ContractError("invalid validation-only preflight")
    candidate_bundle = value.get("candidate_bundle")
    if not isinstance(candidate_bundle, dict):
        raise ContractError("preflight has no candidate bundle")
    return (
        path,
        value,
        {
            "path": str(path),
            "sha256": expected_sha256,
            "receipt_payload_sha256": claimed,
        },
    )


def validate_artifact(
    value: object,
    *,
    expected_parent: Path,
    expected_name: str,
    role: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        raise ContractError(f"invalid {role} artifact")
    path = regular_file(value["path"], f"shard {role}")
    expected_sha = require_sha256(value["sha256"], f"shard {role} SHA")
    expected_bytes = value["bytes"]
    if (
        path.parent != expected_parent
        or path.name != expected_name
        or type(expected_bytes) is not int
        or expected_bytes < 1
        or path.stat().st_size != expected_bytes
        or sha256_file(path) != expected_sha
    ):
        raise ContractError(f"invalid or changed shard {role}: {path}")
    return path, {
        "path": str(path),
        "sha256": expected_sha,
        "bytes": expected_bytes,
    }


def validate_shard(
    *,
    candidate_root: Path,
    epoch: int,
    shard_id: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    preflight_artifact: Mapping[str, Any],
    clip_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    name = f"shard-{shard_id:05d}-of-{EXPECTED_NUM_SHARDS:05d}"
    root = directory(candidate_root / "shards" / name, f"shard {shard_id}")
    receipt_path = regular_file(
        root / "shard_receipt.json", f"shard {shard_id} receipt"
    )
    receipt = strict_json(receipt_path)
    expected_receipt_keys = {
        "format",
        "status",
        "split",
        "test_visible",
        "epoch",
        "candidate_checkpoint",
        "preflight_receipt",
        "assignment",
        "shard_id",
        "num_shards",
        "clip_count",
        "frame_count",
        "prediction_files",
        "ground_truth_files",
        "manifest",
        "model_receipts",
        "model_receipts_sha256",
        "runtime_contract",
        "runtime_contract_sha256",
        "device",
        "exact_once",
        "finite",
        "receipt_payload_sha256",
    }
    claimed = require_sha256(
        receipt.get("receipt_payload_sha256"), f"shard {shard_id} payload SHA"
    )
    candidate_receipt = receipt.get("candidate_checkpoint")
    model_receipts = receipt.get("model_receipts")
    runtime_contract = receipt.get("runtime_contract")
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("format") != SHARD_FORMAT
        or receipt.get("status") != "complete"
        or receipt.get("split") != "val"
        or receipt.get("test_visible") is not False
        or receipt.get("epoch") != epoch
        or candidate_receipt
        != {"path": str(checkpoint_path), "sha256": checkpoint_sha256}
        or receipt.get("preflight_receipt") != preflight_artifact
        or receipt.get("assignment") != ASSIGNMENT
        or receipt.get("shard_id") != shard_id
        or receipt.get("num_shards") != EXPECTED_NUM_SHARDS
        or receipt.get("exact_once") is not True
        or receipt.get("finite") is not True
        or payload_sha256(receipt) != claimed
        or not isinstance(model_receipts, dict)
        or receipt.get("model_receipts_sha256")
        != payload_sha256(model_receipts)
        or not isinstance(runtime_contract, dict)
        or receipt.get("runtime_contract_sha256")
        != payload_sha256(runtime_contract)
    ):
        raise ContractError(f"invalid shard receipt {shard_id}")
    manifest_value = receipt.get("manifest")
    if (
        not isinstance(manifest_value, dict)
        or set(manifest_value) != {"path", "sha256"}
    ):
        raise ContractError(f"invalid shard manifest receipt {shard_id}")
    manifest_path = verified_file(
        manifest_value["path"],
        manifest_value["sha256"],
        f"shard {shard_id} manifest",
    )
    if manifest_path != root / "shard_manifest.jsonl":
        raise ContractError(f"shard {shard_id} manifest path mismatch")
    def row_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(
                    f"duplicate JSONL key {key!r} in {manifest_path}"
                )
            result[key] = value
        return result

    manifest_rows = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8", errors="strict").splitlines(),
        1,
    ):
        try:
            row = json.loads(
                line,
                object_pairs_hook=row_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ContractError(
                        f"non-finite token {token!r} in {manifest_path}"
                    )
                ),
            )
        except json.JSONDecodeError as error:
            raise ContractError(
                f"invalid JSONL row {line_number} in {manifest_path}"
            ) from error
        if not isinstance(row, dict):
            raise ContractError(f"non-object shard row {line_number}")
        manifest_rows.append(row)
    expected_positions = [
        position
        for position in range(EXPECTED_CLIPS)
        if position % EXPECTED_NUM_SHARDS == shard_id
    ]
    if (
        [row.get("canonical_position") for row in manifest_rows]
        != expected_positions
        or receipt.get("clip_count") != len(manifest_rows)
        or receipt.get("prediction_files") != len(manifest_rows)
        or receipt.get("ground_truth_files") != len(manifest_rows)
        or receipt.get("frame_count")
        != sum(
            row.get("frames")
            for row in manifest_rows
            if type(row.get("frames")) is int
        )
    ):
        raise ContractError(f"shard {shard_id} coverage mismatch")
    prediction_dir = directory(
        root / "predictions" / "val", f"shard {shard_id} predictions"
    )
    ground_truth_dir = directory(
        root / "ground-truth" / "val", f"shard {shard_id} ground truth"
    )
    normalized_rows = []
    expected_prediction_files: set[Path] = set()
    expected_ground_truth_files: set[Path] = set()
    expected_row_keys = {
        "canonical_position",
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
    for row, position in zip(manifest_rows, expected_positions):
        clip_id = clip_ids[position]
        if (
            set(row) != expected_row_keys
            or row.get("canonical_position") != position
            or row.get("split") != "val"
            or row.get("canonical_clip_id") != clip_id
            or type(row.get("frames")) is not int
            or row.get("frames") < 1
            or row.get("epoch") != epoch
            or row.get("candidate_checkpoint_sha256") != checkpoint_sha256
        ):
            raise ContractError(
                f"shard {shard_id} row mismatch at canonical position {position}"
            )
        prediction_path, prediction = validate_artifact(
            row.get("prediction"),
            expected_parent=prediction_dir,
            expected_name=f"res_{clip_id}.npz",
            role="prediction",
        )
        ground_truth_path, ground_truth = validate_artifact(
            row.get("ground_truth"),
            expected_parent=ground_truth_dir,
            expected_name=f"gt_{clip_id}.npz",
            role="ground truth",
        )
        expected_prediction_files.add(prediction_path)
        expected_ground_truth_files.add(ground_truth_path)
        normalized_rows.append(
            {
                "canonical_position": position,
                "canonical_clip_id": clip_id,
                "source_clip_id": row.get("source_clip_id"),
                "global_index": row.get("global_index"),
                "frames": row.get("frames"),
                "prediction": {**prediction, "_source": prediction_path},
                "ground_truth": {**ground_truth, "_source": ground_truth_path},
            }
        )
    if (
        set(prediction_dir.iterdir()) != expected_prediction_files
        or set(ground_truth_dir.iterdir()) != expected_ground_truth_files
    ):
        raise ContractError(f"shard {shard_id} output directories have extras")
    expected_entries = {
        root / "shard_receipt.json",
        root / "shard_manifest.jsonl",
        root / "predictions",
        root / "ground-truth",
    }
    if set(root.iterdir()) != expected_entries:
        raise ContractError(f"shard {shard_id} contains unexpected entries")
    artifact = {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "receipt_payload_sha256": claimed,
    }
    return receipt, normalized_rows, artifact


def link_regular(source: Path, destination: Path, role: str) -> None:
    source_stat = source.stat()
    if stat.S_ISLNK(os.lstat(source).st_mode) or not stat.S_ISREG(
        source_stat.st_mode
    ):
        raise ContractError(f"{role} source is not a regular file: {source}")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        raise ContractError(
            f"same-filesystem hardlink failed for {role}: {source} -> "
            f"{destination}: {error}"
        ) from error
    destination_stat = destination.stat()
    if (
        source_stat.st_dev != destination_stat.st_dev
        or source_stat.st_ino != destination_stat.st_ino
        or source_stat.st_size != destination_stat.st_size
        or stat.S_ISLNK(os.lstat(destination).st_mode)
        or not stat.S_ISREG(destination_stat.st_mode)
    ):
        destination.unlink(missing_ok=True)
        raise ContractError(f"{role} destination is not the exact hardlink")


def publish_view(args: argparse.Namespace) -> dict[str, Any]:
    schedule_path, _schedule = validate_schedule(
        args.schedule, args.expected_schedule_sha256
    )
    if args.epoch not in CANDIDATE_EPOCHS:
        raise ContractError("epoch is outside the frozen candidate schedule")
    checkpoint_sha = require_sha256(
        args.expected_checkpoint_sha256, "candidate checkpoint SHA"
    )
    checkpoint_path = verified_file(
        args.checkpoint, checkpoint_sha, "candidate checkpoint"
    )
    if (
        type(args.expected_checkpoint_bytes) is not int
        or args.expected_checkpoint_bytes < 1
        or checkpoint_path.stat().st_size != args.expected_checkpoint_bytes
    ):
        raise ContractError("candidate checkpoint byte size mismatch")
    preflight_path, preflight, preflight_artifact = validate_preflight(
        args.preflight, args.expected_preflight_sha256
    )
    candidates = preflight["candidate_bundle"].get("candidates")
    candidate = None
    if isinstance(candidates, dict):
        candidate = candidates.get(str(args.epoch))
        if candidate is None:
            candidate = candidates.get(args.epoch)
    if (
        not isinstance(candidate, dict)
        or candidate.get("path") != str(checkpoint_path)
        or candidate.get("sha256") != checkpoint_sha
    ):
        raise ContractError("preflight does not bind the requested candidate")
    _clip_path, clip_ids, clip_payload = read_clip_ids(
        args.clip_manifest, args.expected_clip_manifest_sha256
    )
    candidate_root = directory(args.candidate_root, "candidate shard root")
    output = absolute_path(args.output_root, "hardlink view output")
    parent = directory(output.parent, "hardlink view parent")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite hardlink view: {output}")
    if candidate_root.stat().st_dev != parent.stat().st_dev:
        raise ContractError("candidate shards and hardlink view are not on one FS")

    rows_by_position: dict[int, dict[str, Any]] = {}
    shard_artifacts = []
    common_model_sha: str | None = None
    common_runtime_sha: str | None = None
    for shard_id in range(EXPECTED_NUM_SHARDS):
        receipt, rows, receipt_artifact = validate_shard(
            candidate_root=candidate_root,
            epoch=args.epoch,
            shard_id=shard_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            preflight_artifact=preflight_artifact,
            clip_ids=clip_ids,
        )
        model_sha = require_sha256(
            receipt.get("model_receipts_sha256"), "model receipts SHA"
        )
        runtime_sha = require_sha256(
            receipt.get("runtime_contract_sha256"), "runtime contract SHA"
        )
        if common_model_sha is None:
            common_model_sha = model_sha
            common_runtime_sha = runtime_sha
        elif model_sha != common_model_sha or runtime_sha != common_runtime_sha:
            raise ContractError("16 shards disagree on model/runtime receipts")
        shard_artifacts.append(receipt_artifact)
        for row in rows:
            position = row["canonical_position"]
            if position in rows_by_position:
                raise ContractError(f"duplicate canonical position {position}")
            rows_by_position[position] = row
    if set(rows_by_position) != set(range(EXPECTED_CLIPS)):
        raise ContractError("16 shards do not exactly cover all validation clips")

    stage = parent / f".{output.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir()
    prediction_stage = stage / "predictions" / "val"
    ground_truth_stage = stage / "ground-truth" / "val"
    prediction_stage.mkdir(parents=True)
    ground_truth_stage.mkdir(parents=True)
    prediction_final = output / "predictions" / "val"
    ground_truth_final = output / "ground-truth" / "val"
    final_rows = []
    try:
        for position in range(EXPECTED_CLIPS):
            row = rows_by_position[position]
            clip_id = clip_ids[position]
            prediction_source = row["prediction"].pop("_source")
            ground_truth_source = row["ground_truth"].pop("_source")
            prediction_name = f"res_{clip_id}.npz"
            ground_truth_name = f"gt_{clip_id}.npz"
            prediction_destination = prediction_stage / prediction_name
            ground_truth_destination = ground_truth_stage / ground_truth_name
            link_regular(
                prediction_source, prediction_destination, "prediction"
            )
            link_regular(
                ground_truth_source, ground_truth_destination, "ground truth"
            )
            final_rows.append(
                {
                    "canonical_position": position,
                    "global_index": row["global_index"],
                    "source_clip_id": row["source_clip_id"],
                    "canonical_clip_id": clip_id,
                    "frames": row["frames"],
                    "epoch": args.epoch,
                    "candidate_checkpoint_sha256": checkpoint_sha,
                    "prediction": {
                        **row["prediction"],
                        "path": str(prediction_final / prediction_name),
                    },
                    "ground_truth": {
                        **row["ground_truth"],
                        "path": str(ground_truth_final / ground_truth_name),
                    },
                }
            )
        write_inside(stage / "diffsheg_eval_clip_ids.txt", clip_payload)
        manifest_payload = canonical_jsonl_bytes(final_rows)
        write_inside(stage / "view_manifest.jsonl", manifest_payload)
        shard_cover_sha = sha256_bytes(
            canonical_json_bytes(shard_artifacts).rstrip(b"\n")
        )
        receipt_body = {
            "format": VIEW_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "selection_eligible": True,
            "publication_mode": "verified_same_filesystem_hardlinks",
            "epoch": args.epoch,
            "schedule": {
                "path": str(schedule_path),
                "sha256": SCHEDULE_SHA256,
            },
            "candidate_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "bytes": args.expected_checkpoint_bytes,
            },
            "preflight": {
                "path": str(preflight_path),
                "sha256": args.expected_preflight_sha256,
                "receipt_payload_sha256": preflight_artifact[
                    "receipt_payload_sha256"
                ],
            },
            "shards": {
                "num_shards": EXPECTED_NUM_SHARDS,
                "assignment": ASSIGNMENT,
                "exact_once": True,
                "finite": True,
                "receipt_cover_sha256": shard_cover_sha,
                "receipts": shard_artifacts,
                "common_model_receipts_sha256": common_model_sha,
                "common_runtime_contract_sha256": common_runtime_sha,
            },
            "coverage": {
                "clip_count": EXPECTED_CLIPS,
                "prediction_files": EXPECTED_CLIPS,
                "ground_truth_files": EXPECTED_CLIPS,
                "clip_manifest_sha256": CLIP_MANIFEST_SHA256,
            },
            "view": {
                "root": str(output),
                "prediction_dir": str(prediction_final),
                "ground_truth_dir": str(ground_truth_final),
                "clip_manifest": {
                    "path": str(output / "diffsheg_eval_clip_ids.txt"),
                    "sha256": CLIP_MANIFEST_SHA256,
                },
                "manifest": {
                    "path": str(output / "view_manifest.jsonl"),
                    "sha256": sha256_bytes(manifest_payload),
                },
            },
        }
        receipt = {
            **receipt_body,
            "receipt_payload_sha256": payload_sha256(receipt_body),
        }
        write_inside(stage / "hardlink-view-receipt.json", canonical_json_bytes(receipt))
        fsync_directory(prediction_stage)
        fsync_directory(prediction_stage.parent)
        fsync_directory(ground_truth_stage)
        fsync_directory(ground_truth_stage.parent)
        fsync_directory(stage)
        fsync_directory(parent)
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to overwrite hardlink view: {output}")
        os.rename(stage, output)
        fsync_directory(parent)
    except BaseException:
        if os.path.lexists(stage):
            shutil.rmtree(stage)
        raise
    receipt_path = output / "hardlink-view-receipt.json"
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }


def exact_directory_artifacts(
    path_value: object, prefix: str, clip_ids: Sequence[str], label: str
) -> dict[str, Path]:
    root = directory(path_value, label)
    expected = {f"{prefix}{clip_id}.npz" for clip_id in clip_ids}
    observed: dict[str, Path] = {}
    for entry in root.iterdir():
        file_path = regular_file(entry, f"{label} entry")
        observed[file_path.name] = file_path
    if set(observed) != expected:
        raise ContractError(f"{label} is not the exact {len(expected)}-file cover")
    return observed


def fgd_value(path: Path) -> float:
    value = strict_json(path)
    metric = value.get("metrics", {}).get("fgd")
    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(float(metric))
        or float(metric) < 0
    ):
        raise ContractError(f"invalid FGD report: {path}")
    return float(metric)


def gate_e40(args: argparse.Namespace) -> dict[str, Any]:
    schedule_path, _schedule = validate_schedule(
        args.schedule, args.expected_schedule_sha256
    )
    _clip_path, clip_ids, _clip_payload = read_clip_ids(
        args.clip_manifest, args.expected_clip_manifest_sha256
    )
    new_root = directory(args.new_view_root, "new e40 hardlink view")
    new_receipt_path = regular_file(
        new_root / "hardlink-view-receipt.json", "new e40 view receipt"
    )
    new_receipt = strict_json(new_receipt_path)
    if (
        new_receipt.get("format") != VIEW_FORMAT
        or new_receipt.get("status") != "complete"
        or new_receipt.get("epoch") != 40
        or new_receipt.get("candidate_checkpoint", {}).get("sha256")
        != OLD_E40_CHECKPOINT_SHA256
        or new_receipt.get("publication_mode")
        != "verified_same_filesystem_hardlinks"
        or payload_sha256(new_receipt)
        != new_receipt.get("receipt_payload_sha256")
    ):
        raise ContractError("new e40 view is not trajectory/publisher anchored")
    old_root = directory(args.old_final_root, "frozen old e40 final")
    old_lineage = verified_file(
        old_root / "val-inference-lineage.json",
        OLD_E40_LINEAGE_SHA256,
        "frozen old e40 lineage",
    )
    old_manifest = verified_file(
        old_root / "final_manifest.jsonl",
        OLD_E40_FINAL_MANIFEST_SHA256,
        "frozen old e40 final manifest",
    )
    for root, label in (
        (new_root, "new e40"),
        (old_root, "frozen old e40"),
    ):
        verified_file(
            root / "diffsheg_eval_clip_ids.txt",
            CLIP_MANIFEST_SHA256,
            f"{label} clip manifest",
        )
    comparisons = []
    cover_rows = []
    for role, prefix in (("predictions", "res_"), ("ground-truth", "gt_")):
        new_files = exact_directory_artifacts(
            new_root / role / "val", prefix, clip_ids, f"new e40 {role}"
        )
        old_files = exact_directory_artifacts(
            old_root / role / "val", prefix, clip_ids, f"old e40 {role}"
        )
        for name in sorted(new_files):
            new_path = new_files[name]
            old_path = old_files[name]
            new_bytes = new_path.stat().st_size
            old_bytes = old_path.stat().st_size
            new_sha = sha256_file(new_path)
            old_sha = sha256_file(old_path)
            if new_bytes != old_bytes or new_sha != old_sha:
                raise ContractError(f"e40 {role} differs for {name}")
            cover_rows.append((role, name, new_bytes, new_sha))
        comparisons.append({"role": role, "files": len(new_files), "exact": True})
    cover_sha = sha256_bytes(
        json.dumps(
            cover_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )
    old_fgd_report = verified_file(
        args.old_fgd_report,
        OLD_E40_FGD_REPORT_SHA256,
        "frozen old e40 FGD report",
    )
    new_fgd_report = regular_file(args.new_fgd_report, "new e40 FGD report")
    old_fgd = fgd_value(old_fgd_report)
    new_fgd = fgd_value(new_fgd_report)
    if old_fgd != OLD_E40_FGD or new_fgd != old_fgd:
        raise ContractError(
            f"e40 FGD is not exact: new={new_fgd}, old={old_fgd}"
        )
    body = {
        "format": GATE_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "selection_eligible": False,
        "schedule": {"path": str(schedule_path), "sha256": SCHEDULE_SHA256},
        "new_e40": {
            "view_receipt": {
                "path": str(new_receipt_path),
                "sha256": sha256_file(new_receipt_path),
                "receipt_payload_sha256": new_receipt[
                    "receipt_payload_sha256"
                ],
            },
            "fgd_report": {
                "path": str(new_fgd_report),
                "sha256": sha256_file(new_fgd_report),
            },
        },
        "frozen_reference_e40": {
            "lineage": {
                "path": str(old_lineage),
                "sha256": OLD_E40_LINEAGE_SHA256,
            },
            "final_manifest": {
                "path": str(old_manifest),
                "sha256": OLD_E40_FINAL_MANIFEST_SHA256,
            },
            "fgd_report": {
                "path": str(old_fgd_report),
                "sha256": OLD_E40_FGD_REPORT_SHA256,
            },
        },
        "equivalence": {
            "prediction_files": EXPECTED_CLIPS,
            "ground_truth_files": EXPECTED_CLIPS,
            "all_file_bytes_and_sha256_exact": True,
            "file_cover_sha256": cover_sha,
            "directory_comparisons": comparisons,
            "clip_manifest_sha256": CLIP_MANIFEST_SHA256,
            "fgd": new_fgd,
            "fgd_exact": True,
        },
        "selection_firewall": {
            "old_e40_measurement_reused": False,
            "old_e40_selection_reused": False,
            "old_e40_test_claim_reused": False,
            "gate_receipt_selection_eligible": False,
        },
    }
    receipt = {**body, "receipt_payload_sha256": payload_sha256(body)}
    output = write_new_json(args.output, receipt)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    link = commands.add_parser("link-view", allow_abbrev=False)
    link.add_argument("--schedule", type=Path, required=True)
    link.add_argument("--expected-schedule-sha256", required=True)
    link.add_argument("--epoch", type=int, required=True)
    link.add_argument("--candidate-root", type=Path, required=True)
    link.add_argument("--checkpoint", type=Path, required=True)
    link.add_argument("--expected-checkpoint-sha256", required=True)
    link.add_argument("--expected-checkpoint-bytes", type=int, required=True)
    link.add_argument("--preflight", type=Path, required=True)
    link.add_argument("--expected-preflight-sha256", required=True)
    link.add_argument("--clip-manifest", type=Path, required=True)
    link.add_argument(
        "--expected-clip-manifest-sha256",
        default=CLIP_MANIFEST_SHA256,
    )
    link.add_argument("--output-root", type=Path, required=True)

    gate = commands.add_parser("gate-e40", allow_abbrev=False)
    gate.add_argument("--schedule", type=Path, required=True)
    gate.add_argument("--expected-schedule-sha256", required=True)
    gate.add_argument("--new-view-root", type=Path, required=True)
    gate.add_argument("--new-fgd-report", type=Path, required=True)
    gate.add_argument("--old-final-root", type=Path, required=True)
    gate.add_argument("--old-fgd-report", type=Path, required=True)
    gate.add_argument("--clip-manifest", type=Path, required=True)
    gate.add_argument(
        "--expected-clip-manifest-sha256",
        default=CLIP_MANIFEST_SHA256,
    )
    gate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "link-view":
        result = publish_view(args)
    elif args.command == "gate-e40":
        result = gate_e40(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
