#!/usr/bin/env python3
"""Fail-closed bridge from SHOW prerequisite selection into SemTalk Base.

This module is the consumer-side mirror of the frozen prerequisite validation
contract.  It intentionally revalidates the selector's exact receipt schema
and every referenced immutable artifact before Base imports PyTorch.

Base consumes Face/Hands/Upper/Lower RVQ codes.  Global remains a mandatory,
fully selected prerequisite and is audited even though it is not part of the
Base feature graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SELECTION_FORMAT = "semtalk_show_prerequisite_val_selection_v1"
CANDIDATE_INDEX_FORMAT = "semtalk_show_prerequisite_candidate_index_v1"
MEASUREMENT_INDEX_FORMAT = (
    "semtalk_show_prerequisite_val_measurement_index_v1"
)
STAGE_MEASUREMENT_FORMAT = (
    "semtalk_show_prerequisite_val_stage_measurement_v1"
)
SHARD_FORMAT = "semtalk_show_prerequisite_val_shard_v1"
TRAINING_SOURCE_FREEZE_FORMAT = "semtalk_show_training_source_freeze_v1"
STAGES = ("face", "hands", "upper", "lower", "global")
RVQ_STAGES = ("face", "hands", "upper", "lower")
EXPECTED_CANDIDATE_EPOCHS = tuple(range(20, 201, 20))
EXPECTED_UPDATES_PER_EPOCH = 497
EXPECTED_VAL_CLIPS = 1_715
EXPECTED_SHARDS = 8
EXPECTED_SPEAKER_SCOPE = "all_speakers_0_1_2_3"
WINDOW_LENGTH = 64
WINDOW_STRIDE = 20
EXPECTED_METRICS = {
    "face": "face_geometry_expression_objective_v1",
    "hands": "hands_rotation_geometry_objective_v1",
    "upper": "upper_rotation_geometry_objective_v1",
    "lower": "lower_rotation_contact_objective_v1",
    "global": "global_root_contact_objective_v1",
}

TOP_KEYS = {
    "format",
    "status",
    "target_dataset",
    "target_speaker_scope",
    "split",
    "test_visible",
    "protocol",
    "selection_policy",
    "canonical_view",
    "producer_sources",
    "training_sources",
    "config_sha256",
    "candidate_index_receipt",
    "measurement_index_receipt",
    "stages",
    "receipt_payload_sha256",
}
STAGE_KEYS = {
    "stage",
    "selection_metric",
    "epoch",
    "optimizer_updates",
    "candidate_index",
    "selection_score",
    "candidate_checkpoint",
    "measurement_receipt",
    "coverage",
}
CHECKPOINT_KEYS = {"path", "sha256", "bytes"}
PAYLOAD_ARTIFACT_KEYS = {"path", "sha256", "receipt_payload_sha256"}
ARTIFACT_KEYS = {"path", "sha256"}
COVERAGE_KEYS = {
    "split",
    "test_visible",
    "clips",
    "shards",
    "exact_once",
    "all_finite",
}
CANDIDATE_INDEX_KEYS = {
    "format",
    "status",
    "target_dataset",
    "target_speaker_scope",
    "selection_split",
    "test_visible",
    "candidate_epochs",
    "updates_per_epoch",
    "source_receipts",
    "config_sha256",
    "dataset_receipt_sha256",
    "formal_training_status",
    "stages",
    "receipt_payload_sha256",
}
MEASUREMENT_INDEX_KEYS = {
    "format",
    "status",
    "target_dataset",
    "target_speaker_scope",
    "split",
    "test_visible",
    "protocol",
    "canonical_view",
    "producer_sources",
    "training_sources",
    "config_sha256",
    "candidate_index_receipt",
    "stages",
    "coverage",
    "receipt_payload_sha256",
}
STAGE_MEASUREMENT_KEYS = {
    "format",
    "status",
    "target_dataset",
    "target_speaker_scope",
    "split",
    "test_visible",
    "stage",
    "selection_metric",
    "protocol",
    "canonical_view",
    "producer_sources",
    "training_source",
    "config_sha256",
    "candidate_index_receipt",
    "candidates",
    "coverage",
    "receipt_payload_sha256",
}
CANDIDATE_KEYS = {
    "candidate_index",
    "epoch",
    "optimizer_updates",
    "selection_score",
    "selection_metric",
    "candidate_checkpoint",
    "metrics",
    "codebook_histograms",
    "coverage",
    "shard_receipts",
}
CANONICAL_VIEW_KEYS = {
    "manifest",
    "summary",
    "lineage",
    "clip_count",
    "clip_ids_sha256",
    "split",
    "test_visible",
}
PRODUCER_SOURCE_KEYS = {
    "source_root",
    "origin",
    "commit",
    "tree",
    "clean",
    "script",
    "script_relative",
    "script_sha256",
}

_FORBIDDEN_E30 = re.compile(
    r"(^|[^a-z0-9])e[-_]?30([^a-z0-9]|$)",
    re.IGNORECASE,
)
_FORBIDDEN_SPEAKER2 = re.compile(
    r"(^|[^a-z0-9])speaker[-_]?2([^a-z0-9]|$)",
    re.IGNORECASE,
)
_FORBIDDEN_COMPONENTS = frozenset(
    {"test", "tests", "testset", "testsets", "semgate", "sparse"}
)


class SelectedPrerequisiteError(RuntimeError):
    """Raised before Base work when selected prerequisites are not proven."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelectedPrerequisiteError(
            "receipt is not canonical finite JSON"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectedPrerequisiteError(
            f"{label} must be a canonical lowercase SHA-256"
        )
    return value


def require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectedPrerequisiteError(
            f"{label} must be a canonical lowercase Git object ID"
        )
    return value


def require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectedPrerequisiteError(f"{label} must be an exact integer")
    return value


def require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise SelectedPrerequisiteError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SelectedPrerequisiteError(
            f"{label} must be finite and nonnegative"
        )
    return result


def exact_keys(
    value: Any,
    keys: Sequence[str] | set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise SelectedPrerequisiteError(f"{label} schema mismatch")
    return value


def reject_forbidden(value: Any, label: str) -> None:
    text = str(value).replace("\\", "/")
    for component in text.split("/"):
        normalized = component.casefold()
        stem = normalized.rsplit(".", 1)[0]
        tokens = frozenset(filter(None, re.split(r"[^a-z0-9]+", stem)))
        if (
            tokens & _FORBIDDEN_COMPONENTS
            or _FORBIDDEN_E30.search(normalized)
            or _FORBIDDEN_SPEAKER2.search(normalized)
        ):
            raise SelectedPrerequisiteError(
                f"{label} contains a forbidden/test source: {value}"
            )


def reject_absolute_paths_in_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            reject_absolute_paths_in_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_absolute_paths_in_tree(child, f"{label}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        reject_forbidden(value, label)


def regular_file(
    value: Any,
    label: str,
    *,
    val_only: bool = True,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise SelectedPrerequisiteError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SelectedPrerequisiteError(f"{label} must be absolute")
    if val_only:
        reject_forbidden(path, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise SelectedPrerequisiteError(
            f"{label} does not exist: {path}"
        ) from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SelectedPrerequisiteError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    if val_only:
        reject_forbidden(resolved, label)
    return resolved


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SelectedPrerequisiteError(
                    f"{label}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SelectedPrerequisiteError(
                    f"{label}: non-finite JSON token {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectedPrerequisiteError(
            f"{label}: invalid strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise SelectedPrerequisiteError(f"{label} must be a JSON object")
    return value


def strict_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        result.append(strict_json_bytes(line, f"{label}:{line_number}"))
    return result


def _verify_self_hash(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectedPrerequisiteError(f"{label} must be an object")
    claimed = require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} receipt payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256", None)
    observed = canonical_json_sha256(unsigned)
    if claimed != observed:
        raise SelectedPrerequisiteError(
            f"{label} receipt payload SHA-256 mismatch"
        )
    return value


def _read_artifact(
    value: Any,
    *,
    keys: set[str],
    label: str,
    val_only: bool = True,
    parse_json: bool = True,
) -> tuple[Path, bytes, dict[str, Any] | None, dict[str, Any]]:
    binding = exact_keys(value, keys, f"{label} artifact")
    path = regular_file(binding["path"], label, val_only=val_only)
    expected_sha = require_sha256(binding["sha256"], f"{label} SHA-256")
    payload = path.read_bytes()
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != expected_sha:
        raise SelectedPrerequisiteError(
            f"{label} SHA-256 mismatch: {observed_sha} != {expected_sha}"
        )
    parsed = strict_json_bytes(payload, str(path)) if parse_json else None
    if "receipt_payload_sha256" in keys:
        assert parsed is not None
        parsed = _verify_self_hash(parsed, label)
        if (
            parsed["receipt_payload_sha256"]
            != require_sha256(
                binding["receipt_payload_sha256"],
                f"{label} bound payload SHA-256",
            )
        ):
            raise SelectedPrerequisiteError(
                f"{label} bound receipt payload SHA-256 mismatch"
            )
    return path, payload, parsed, dict(binding)


def _validate_training_source(value: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("origin") != EXPECTED_ORIGIN
        or value.get("clean") is not True
    ):
        raise SelectedPrerequisiteError(
            f"{label} is not a clean SemTalk source"
        )
    require_git_oid(value.get("commit"), f"{label}.commit")
    require_git_oid(value.get("tree"), f"{label}.tree")
    reject_absolute_paths_in_tree(value, label)
    return dict(value)


def _validate_frozen_training_source(
    value: Any,
    label: str,
) -> dict[str, Any]:
    source = exact_keys(
        _verify_self_hash(value, label),
        {
            "format",
            "training_audit",
            "source_root",
            "origin",
            "commit",
            "tree",
            "clean",
            "detached",
            "local_branch_count",
            "entrypoint_relative",
            "entrypoint_sha256",
            "receipt_payload_sha256",
        },
        label,
    )
    audit = exact_keys(
        source["training_audit"],
        {
            "commit",
            "tree",
            "origin",
            "entrypoint",
            "entrypoint_sha256",
        },
        f"{label}.training_audit",
    )
    if (
        source["format"] != TRAINING_SOURCE_FREEZE_FORMAT
        or source["origin"] != EXPECTED_ORIGIN
        or audit["origin"] != EXPECTED_ORIGIN
        or source["origin"] != audit["origin"]
        or source["commit"] != audit["commit"]
        or source["tree"] != audit["tree"]
        or source["clean"] is not True
        or source["detached"] is not True
        or source["local_branch_count"] != 0
        or source["entrypoint_sha256"] != audit["entrypoint_sha256"]
    ):
        raise SelectedPrerequisiteError(
            f"{label} frozen training source binding mismatch"
        )
    require_git_oid(source["commit"], f"{label}.commit")
    require_git_oid(source["tree"], f"{label}.tree")
    root = Path(source["source_root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SelectedPrerequisiteError(f"{label}.source_root is invalid")
    root = root.resolve(strict=True)
    entrypoint = regular_file(
        audit["entrypoint"],
        f"{label}.entrypoint",
        val_only=False,
    )
    try:
        relative = str(entrypoint.relative_to(root))
    except ValueError as error:
        raise SelectedPrerequisiteError(
            f"{label}.entrypoint escapes source root"
        ) from error
    if (
        relative != source["entrypoint_relative"]
        or sha256_file(entrypoint)
        != require_sha256(
            source["entrypoint_sha256"],
            f"{label}.entrypoint SHA-256",
        )
    ):
        raise SelectedPrerequisiteError(
            f"{label} frozen entrypoint binding changed"
        )
    reject_absolute_paths_in_tree(source, label)
    return dict(source)


def _validate_producer_source(value: Any, label: str) -> dict[str, Any]:
    source = exact_keys(value, PRODUCER_SOURCE_KEYS, label)
    _validate_training_source(source, label)
    root = Path(source["source_root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SelectedPrerequisiteError(f"{label}.source_root is invalid")
    root = root.resolve(strict=True)
    script = regular_file(source["script"], f"{label}.script", val_only=False)
    try:
        relative = str(script.relative_to(root))
    except ValueError as error:
        raise SelectedPrerequisiteError(
            f"{label}.script escapes source root"
        ) from error
    if (
        relative != source["script_relative"]
        or sha256_file(script)
        != require_sha256(
            source["script_sha256"],
            f"{label}.script SHA-256",
        )
    ):
        raise SelectedPrerequisiteError(f"{label} script binding changed")
    return dict(source)


def _validate_canonical_view(value: Any) -> dict[str, Any]:
    view = exact_keys(value, CANONICAL_VIEW_KEYS, "canonical validation view")
    if (
        view["clip_count"] != EXPECTED_VAL_CLIPS
        or view["split"] != "val"
        or view["test_visible"] is not False
    ):
        raise SelectedPrerequisiteError(
            "canonical validation view protocol mismatch"
        )
    manifest_path, manifest_bytes, _, manifest_binding = _read_artifact(
        view["manifest"],
        keys=ARTIFACT_KEYS,
        label="canonical validation manifest",
        parse_json=False,
    )
    _, _, summary, summary_binding = _read_artifact(
        view["summary"],
        keys=ARTIFACT_KEYS,
        label="canonical validation summary",
    )
    _, _, lineage, lineage_binding = _read_artifact(
        view["lineage"],
        keys=ARTIFACT_KEYS,
        label="canonical validation lineage",
    )
    assert summary is not None and lineage is not None
    _verify_self_hash(summary, "canonical validation summary")
    _verify_self_hash(lineage, "canonical validation lineage")
    if (
        summary.get("format")
        != "semtalk_show_base_official_adapt_val_canonical_summary_v1"
        or summary.get("status") != "complete"
        or summary.get("split") != "val"
        or summary.get("test_visible") is not False
        or summary.get("clip_count") != EXPECTED_VAL_CLIPS
        or summary.get("manifest_sha256") != manifest_binding["sha256"]
        or summary.get("lineage_sha256") != lineage_binding["sha256"]
        or lineage.get("format")
        != "semtalk_show_base_official_adapt_val_canonical_lineage_v1"
        or lineage.get("status") != "complete"
        or lineage.get("split") != "val"
        or lineage.get("test_visible") is not False
        or lineage.get("clip_count") != EXPECTED_VAL_CLIPS
        or lineage.get("manifest_sha256") != manifest_binding["sha256"]
    ):
        raise SelectedPrerequisiteError(
            "canonical validation receipt binding mismatch"
        )
    rows = strict_jsonl_bytes(manifest_bytes, str(manifest_path))
    if len(rows) != EXPECTED_VAL_CLIPS:
        raise SelectedPrerequisiteError(
            "canonical validation manifest clip count mismatch"
        )
    indices: list[int] = []
    clip_ids: list[str] = []
    for line_number, row in enumerate(rows, 1):
        index = require_exact_int(
            row.get("global_index"),
            f"canonical row {line_number}.global_index",
        )
        clip_id = row.get("clip_id")
        if (
            row.get("split") != "val"
            or not isinstance(clip_id, str)
            or not clip_id
        ):
            raise SelectedPrerequisiteError(
                f"canonical row {line_number} is not val-only"
            )
        reject_forbidden(clip_id, f"canonical row {line_number}.clip_id")
        indices.append(index)
        clip_ids.append(clip_id)
    if (
        indices != sorted(indices)
        or len(set(indices)) != EXPECTED_VAL_CLIPS
        or len(set(clip_ids)) != EXPECTED_VAL_CLIPS
        or canonical_json_sha256(clip_ids)
        != require_sha256(
            view["clip_ids_sha256"],
            "canonical validation clip ID SHA-256",
        )
    ):
        raise SelectedPrerequisiteError(
            "canonical validation rows are not exact-once"
        )
    return {
        "manifest": manifest_binding,
        "summary": summary_binding,
        "lineage": lineage_binding,
        "clip_count": EXPECTED_VAL_CLIPS,
        "clip_ids_sha256": view["clip_ids_sha256"],
        "split": "val",
        "test_visible": False,
    }


def _validate_candidate_index(
    artifact: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[int, dict[str, Any]]],
]:
    _, _, index, receipt = _read_artifact(
        artifact,
        keys=PAYLOAD_ARTIFACT_KEYS,
        label="prerequisite candidate index",
    )
    assert index is not None
    exact_keys(index, CANDIDATE_INDEX_KEYS, "candidate index")
    if (
        index["format"] != CANDIDATE_INDEX_FORMAT
        or index["status"] != "complete"
        or index["target_dataset"] != "SHOW"
        or index["target_speaker_scope"] != EXPECTED_SPEAKER_SCOPE
        or index["selection_split"] != "val"
        or index["test_visible"] is not False
        or index["candidate_epochs"] != list(EXPECTED_CANDIDATE_EPOCHS)
        or index["updates_per_epoch"] != EXPECTED_UPDATES_PER_EPOCH
    ):
        raise SelectedPrerequisiteError(
            "prerequisite candidate index protocol mismatch"
        )
    for mapping_name in (
        "source_receipts",
        "config_sha256",
        "dataset_receipt_sha256",
        "formal_training_status",
        "stages",
    ):
        mapping = index[mapping_name]
        if not isinstance(mapping, dict) or set(mapping) != set(STAGES):
            raise SelectedPrerequisiteError(
                f"candidate index {mapping_name} stage coverage mismatch"
            )
    for stage in STAGES:
        _validate_frozen_training_source(
            index["source_receipts"][stage],
            f"candidate index {stage} training source",
        )
        require_sha256(
            index["config_sha256"][stage],
            f"candidate index {stage} config SHA-256",
        )
        require_sha256(
            index["dataset_receipt_sha256"][stage],
            f"candidate index {stage} dataset receipt SHA-256",
        )
        _, _, status, _ = _read_artifact(
            index["formal_training_status"][stage],
            keys=ARTIFACT_KEYS,
            label=f"{stage} formal training status",
        )
        assert status is not None
        if (
            status.get("status") != "complete"
            or status.get("formal_stage") != stage
            or status.get("completed_epochs") != 200
            or status.get("updates_per_epoch") != EXPECTED_UPDATES_PER_EPOCH
            or status.get("optimizer_updates")
            != 200 * EXPECTED_UPDATES_PER_EPOCH
            or status.get("all_training_state_finite", True) is not True
        ):
            raise SelectedPrerequisiteError(
                f"{stage} formal training status is incomplete"
            )

    normalized: dict[str, dict[int, dict[str, Any]]] = {}
    observed_paths: set[Path] = set()
    for stage in STAGES:
        rows = index["stages"][stage]
        if not isinstance(rows, list) or len(rows) != len(
            EXPECTED_CANDIDATE_EPOCHS
        ):
            raise SelectedPrerequisiteError(
                f"{stage} candidate index coverage mismatch"
            )
        normalized[stage] = {}
        for expected_epoch, row in zip(EXPECTED_CANDIDATE_EPOCHS, rows):
            row = exact_keys(
                row,
                {
                    "epoch",
                    "optimizer_updates",
                    "checkpoint",
                    "checkpoint_sha256",
                    "checkpoint_bytes",
                    "checkpoint_audit_sha256",
                },
                f"{stage} candidate index row",
            )
            epoch = require_exact_int(row["epoch"], f"{stage} epoch")
            updates = require_exact_int(
                row["optimizer_updates"],
                f"{stage} optimizer updates",
            )
            checkpoint = regular_file(
                row["checkpoint"],
                f"{stage} candidate checkpoint",
            )
            checkpoint_sha = require_sha256(
                row["checkpoint_sha256"],
                f"{stage} candidate checkpoint SHA-256",
            )
            checkpoint_bytes = require_exact_int(
                row["checkpoint_bytes"],
                f"{stage} candidate checkpoint bytes",
            )
            checkpoint_audit_sha = require_sha256(
                row["checkpoint_audit_sha256"],
                f"{stage} candidate audit SHA-256",
            )
            if (
                epoch != expected_epoch
                or updates != epoch * EXPECTED_UPDATES_PER_EPOCH
                or checkpoint in observed_paths
                or checkpoint.stat().st_size != checkpoint_bytes
                or sha256_file(checkpoint) != checkpoint_sha
            ):
                raise SelectedPrerequisiteError(
                    f"{stage} candidate index binding mismatch at e{epoch}"
                )
            observed_paths.add(checkpoint)
            normalized[stage][epoch] = {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "bytes": checkpoint_bytes,
                "optimizer_updates": updates,
                "checkpoint_audit_sha256": checkpoint_audit_sha,
            }
    return index, receipt, normalized


def _validate_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif type(value) in {int, float}:
        if not math.isfinite(float(value)):
            raise SelectedPrerequisiteError(f"{label} is non-finite")
    else:
        raise SelectedPrerequisiteError(f"{label} has unsupported JSON type")


def _validate_shard_receipts(
    value: Any,
    *,
    stage: str,
    epoch: int,
    expected_checkpoint: Mapping[str, Any],
    expected_windows: int,
) -> None:
    if not isinstance(value, list) or len(value) != EXPECTED_SHARDS:
        raise SelectedPrerequisiteError(
            f"{stage} e{epoch} shard receipt coverage mismatch"
        )
    observed_sha: set[str] = set()
    total_clips = 0
    total_windows = 0
    for expected_index, receipt in enumerate(value):
        binding = exact_keys(
            receipt,
            {
                "path",
                "sha256",
                "receipt_payload_sha256",
                "shard_index",
                "clips",
                "windows",
            },
            f"{stage} e{epoch} shard receipt",
        )
        if binding["shard_index"] != expected_index:
            raise SelectedPrerequisiteError(
                f"{stage} e{epoch} shard receipt order changed"
            )
        _, _, shard, _ = _read_artifact(
            {
                key: binding[key]
                for key in PAYLOAD_ARTIFACT_KEYS
            },
            keys=PAYLOAD_ARTIFACT_KEYS,
            label=f"{stage} e{epoch} shard {expected_index}",
        )
        assert shard is not None
        shard_sha = binding["sha256"]
        shard_coverage = shard.get("shard")
        if (
            shard_sha in observed_sha
            or shard.get("format") != SHARD_FORMAT
            or shard.get("status") != "complete"
            or shard.get("stage") != stage
            or shard.get("split") != "val"
            or shard.get("test_visible") is not False
            or shard.get("epoch") != epoch
            or shard.get("optimizer_updates")
            != expected_checkpoint["optimizer_updates"]
            or shard.get("checkpoint")
            != {
                "path": expected_checkpoint["path"],
                "sha256": expected_checkpoint["sha256"],
                "bytes": expected_checkpoint["bytes"],
            }
            or shard.get("finite") is not True
            or shard.get("exact_once_within_shard") is not True
            or not isinstance(shard_coverage, dict)
            or shard_coverage.get("index") != expected_index
            or shard_coverage.get("count") != EXPECTED_SHARDS
            or shard_coverage.get("clip_count") != binding["clips"]
            or shard_coverage.get("window_count") != binding["windows"]
        ):
            raise SelectedPrerequisiteError(
                f"{stage} e{epoch} shard {expected_index} binding changed"
            )
        observed_sha.add(shard_sha)
        total_clips += require_exact_int(
            binding["clips"],
            f"{stage} e{epoch} shard {expected_index} clips",
        )
        total_windows += require_exact_int(
            binding["windows"],
            f"{stage} e{epoch} shard {expected_index} windows",
        )
    if total_clips != EXPECTED_VAL_CLIPS or total_windows != expected_windows:
        raise SelectedPrerequisiteError(
            f"{stage} e{epoch} shard totals changed"
        )


def _validate_stage_measurement(
    *,
    stage: str,
    artifact: Any,
    measurement_index: Mapping[str, Any],
    candidate_index: Mapping[str, Any],
    candidates: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    _, _, measurement, binding = _read_artifact(
        artifact,
        keys=PAYLOAD_ARTIFACT_KEYS,
        label=f"{stage} stage measurement",
    )
    assert measurement is not None
    exact_keys(measurement, STAGE_MEASUREMENT_KEYS, f"{stage} measurement")
    if (
        measurement["format"] != STAGE_MEASUREMENT_FORMAT
        or measurement["status"] != "complete"
        or measurement["target_dataset"] != "SHOW"
        or measurement["target_speaker_scope"] != EXPECTED_SPEAKER_SCOPE
        or measurement["split"] != "val"
        or measurement["test_visible"] is not False
        or measurement["stage"] != stage
        or measurement["selection_metric"] != EXPECTED_METRICS[stage]
        or measurement["canonical_view"]
        != measurement_index["canonical_view"]
        or measurement["producer_sources"]
        != measurement_index["producer_sources"]
        or measurement["training_source"]
        != candidate_index["source_receipts"][stage]
        or measurement["config_sha256"]
        != candidate_index["config_sha256"][stage]
        or measurement["candidate_index_receipt"]
        != measurement_index["candidate_index_receipt"]
    ):
        raise SelectedPrerequisiteError(
            f"{stage} measurement binding mismatch"
        )
    protocol = exact_keys(
        measurement["protocol"],
        {
            "per_stage_independent",
            "candidate_variable_only",
            "candidate_epochs",
            "window_length",
            "window_stride",
            "full_base_fgd_used",
            "test_feedback_into_selection",
        },
        f"{stage} measurement protocol",
    )
    if protocol != {
        "per_stage_independent": True,
        "candidate_variable_only": True,
        "candidate_epochs": list(EXPECTED_CANDIDATE_EPOCHS),
        "window_length": WINDOW_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "full_base_fgd_used": False,
        "test_feedback_into_selection": False,
    }:
        raise SelectedPrerequisiteError(
            f"{stage} measurement protocol changed"
        )
    coverage = exact_keys(
        measurement["coverage"],
        {
            "split",
            "test_visible",
            "clips_per_candidate",
            "candidates",
            "shards_per_candidate",
            "shard_jobs",
            "windows_per_candidate",
            "exact_once_per_candidate",
            "all_finite",
        },
        f"{stage} measurement coverage",
    )
    expected_windows = require_exact_int(
        coverage["windows_per_candidate"],
        f"{stage} windows per candidate",
    )
    if coverage != {
        "split": "val",
        "test_visible": False,
        "clips_per_candidate": EXPECTED_VAL_CLIPS,
        "candidates": len(EXPECTED_CANDIDATE_EPOCHS),
        "shards_per_candidate": EXPECTED_SHARDS,
        "shard_jobs": len(EXPECTED_CANDIDATE_EPOCHS) * EXPECTED_SHARDS,
        "windows_per_candidate": expected_windows,
        "exact_once_per_candidate": True,
        "all_finite": True,
    } or expected_windows <= 0:
        raise SelectedPrerequisiteError(
            f"{stage} measurement coverage changed"
        )
    rows = measurement["candidates"]
    if not isinstance(rows, list) or len(rows) != len(
        EXPECTED_CANDIDATE_EPOCHS
    ):
        raise SelectedPrerequisiteError(
            f"{stage} measurement candidate coverage mismatch"
        )
    validated: list[dict[str, Any]] = []
    for expected_index, expected_epoch in enumerate(
        EXPECTED_CANDIDATE_EPOCHS
    ):
        row = exact_keys(
            rows[expected_index],
            CANDIDATE_KEYS,
            f"{stage} candidate measurement",
        )
        expected = candidates[expected_epoch]
        expected_checkpoint = {
            "path": expected["path"],
            "sha256": expected["sha256"],
            "bytes": expected["bytes"],
        }
        row_coverage = exact_keys(
            row["coverage"],
            {
                "split",
                "test_visible",
                "clips",
                "shards",
                "windows",
                "exact_once",
                "all_finite",
            },
            f"{stage} e{expected_epoch} coverage",
        )
        score = require_finite(
            row["selection_score"],
            f"{stage} e{expected_epoch} selection score",
        )
        if (
            row["candidate_index"] != expected_index
            or row["epoch"] != expected_epoch
            or row["optimizer_updates"]
            != expected["optimizer_updates"]
            or row["selection_metric"] != EXPECTED_METRICS[stage]
            or exact_keys(
                row["candidate_checkpoint"],
                CHECKPOINT_KEYS,
                f"{stage} e{expected_epoch} checkpoint",
            )
            != expected_checkpoint
            or row_coverage
            != {
                "split": "val",
                "test_visible": False,
                "clips": EXPECTED_VAL_CLIPS,
                "shards": EXPECTED_SHARDS,
                "windows": expected_windows,
                "exact_once": True,
                "all_finite": True,
            }
        ):
            raise SelectedPrerequisiteError(
                f"{stage} e{expected_epoch} measurement changed"
            )
        _validate_finite_tree(row["metrics"], f"{stage} e{expected_epoch} metrics")
        _validate_finite_tree(
            row["codebook_histograms"],
            f"{stage} e{expected_epoch} codebook histograms",
        )
        _validate_shard_receipts(
            row["shard_receipts"],
            stage=stage,
            epoch=expected_epoch,
            expected_checkpoint=expected,
            expected_windows=expected_windows,
        )
        validated.append({**dict(row), "selection_score": score})
    return measurement, binding, validated


def load_selected_prerequisites(
    path_value: Any,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load and fully revalidate one five-stage selection transaction."""

    path = regular_file(path_value, "prerequisite selection")
    expected_file_sha = require_sha256(
        expected_sha256,
        "prerequisite selection expected SHA-256",
    )
    payload_bytes = path.read_bytes()
    observed_file_sha = hashlib.sha256(payload_bytes).hexdigest()
    if observed_file_sha != expected_file_sha:
        raise SelectedPrerequisiteError(
            "prerequisite selection file SHA-256 mismatch"
        )
    selection = _verify_self_hash(
        strict_json_bytes(payload_bytes, str(path)),
        "prerequisite selection",
    )
    exact_keys(selection, TOP_KEYS, "prerequisite selection")
    claimed_payload_sha = selection["receipt_payload_sha256"]
    if (
        selection["format"] != SELECTION_FORMAT
        or selection["status"] != "selected"
        or selection["target_dataset"] != "SHOW"
        or selection["target_speaker_scope"] != EXPECTED_SPEAKER_SCOPE
        or selection["split"] != "val"
        or selection["test_visible"] is not False
    ):
        raise SelectedPrerequisiteError(
            "prerequisite selection protocol mismatch"
        )
    reject_absolute_paths_in_tree(selection, "prerequisite selection")
    if selection["protocol"] != {
        "name": "five_independent_show_prerequisite_validation_v1",
        "candidate_epochs": list(EXPECTED_CANDIDATE_EPOCHS),
        "candidates_per_stage": len(EXPECTED_CANDIDATE_EPOCHS),
        "clips_per_candidate": EXPECTED_VAL_CLIPS,
        "shards_per_candidate": EXPECTED_SHARDS,
        "window_length": WINDOW_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "full_base_fgd_used": False,
    }:
        raise SelectedPrerequisiteError(
            "prerequisite selection protocol changed"
        )
    if selection["selection_policy"] != {
        "per_stage_independent": True,
        "ordering": [
            "selection_score",
            "epoch",
            "optimizer_updates",
            "checkpoint_sha256",
        ],
        "test_feedback_into_selection": False,
    }:
        raise SelectedPrerequisiteError(
            "prerequisite selection policy changed"
        )

    candidate_index, candidate_receipt, candidates = (
        _validate_candidate_index(selection["candidate_index_receipt"])
    )
    _, _, measurement_index, measurement_index_receipt = _read_artifact(
        selection["measurement_index_receipt"],
        keys=PAYLOAD_ARTIFACT_KEYS,
        label="prerequisite measurement index",
    )
    assert measurement_index is not None
    exact_keys(
        measurement_index,
        MEASUREMENT_INDEX_KEYS,
        "prerequisite measurement index",
    )
    if (
        measurement_index["format"] != MEASUREMENT_INDEX_FORMAT
        or measurement_index["status"] != "complete"
        or measurement_index["target_dataset"] != "SHOW"
        or measurement_index["target_speaker_scope"]
        != EXPECTED_SPEAKER_SCOPE
        or measurement_index["split"] != "val"
        or measurement_index["test_visible"] is not False
        or measurement_index["candidate_index_receipt"] != candidate_receipt
        or measurement_index["training_sources"]
        != candidate_index["source_receipts"]
        or measurement_index["config_sha256"]
        != candidate_index["config_sha256"]
    ):
        raise SelectedPrerequisiteError(
            "prerequisite measurement index binding mismatch"
        )
    if measurement_index["protocol"] != {
        "per_stage_independent": True,
        "candidate_epochs": list(EXPECTED_CANDIDATE_EPOCHS),
        "candidates_per_stage": len(EXPECTED_CANDIDATE_EPOCHS),
        "shards_per_candidate": EXPECTED_SHARDS,
        "full_base_fgd_used": False,
        "test_feedback_into_selection": False,
    }:
        raise SelectedPrerequisiteError(
            "prerequisite measurement index protocol changed"
        )
    index_coverage = exact_keys(
        measurement_index["coverage"],
        {
            "stages",
            "candidates_per_stage",
            "shards_per_candidate",
            "total_shard_jobs",
            "clips_per_candidate",
            "windows_per_candidate",
            "exact_once",
            "all_finite",
        },
        "prerequisite measurement index coverage",
    )
    windows_per_candidate = require_exact_int(
        index_coverage["windows_per_candidate"],
        "prerequisite windows per candidate",
    )
    if index_coverage != {
        "stages": len(STAGES),
        "candidates_per_stage": len(EXPECTED_CANDIDATE_EPOCHS),
        "shards_per_candidate": EXPECTED_SHARDS,
        "total_shard_jobs": (
            len(STAGES) * len(EXPECTED_CANDIDATE_EPOCHS) * EXPECTED_SHARDS
        ),
        "clips_per_candidate": EXPECTED_VAL_CLIPS,
        "windows_per_candidate": windows_per_candidate,
        "exact_once": True,
        "all_finite": True,
    } or windows_per_candidate <= 0:
        raise SelectedPrerequisiteError(
            "prerequisite measurement index coverage changed"
        )

    canonical_view = _validate_canonical_view(
        measurement_index["canonical_view"]
    )
    if selection["canonical_view"] != canonical_view:
        raise SelectedPrerequisiteError(
            "selection canonical validation binding changed"
        )

    producer_sources = exact_keys(
        measurement_index["producer_sources"],
        {"evaluator", "merge"},
        "measurement producer sources",
    )
    for role in ("evaluator", "merge"):
        _validate_producer_source(
            producer_sources[role],
            f"measurement {role} source",
        )
    selection_sources = exact_keys(
        selection["producer_sources"],
        {"evaluator", "merge", "selector"},
        "selection producer sources",
    )
    if (
        selection_sources["evaluator"] != producer_sources["evaluator"]
        or selection_sources["merge"] != producer_sources["merge"]
    ):
        raise SelectedPrerequisiteError(
            "selection producer lineage changed"
        )
    _validate_producer_source(
        selection_sources["selector"],
        "selection selector source",
    )
    if (
        selection["training_sources"] != candidate_index["source_receipts"]
        or selection["config_sha256"] != candidate_index["config_sha256"]
    ):
        raise SelectedPrerequisiteError(
            "selection training/config lineage changed"
        )

    stage_receipts = measurement_index["stages"]
    if not isinstance(stage_receipts, dict) or set(stage_receipts) != set(
        STAGES
    ):
        raise SelectedPrerequisiteError(
            "measurement index does not exactly cover five stages"
        )
    selection_stages = selection["stages"]
    if not isinstance(selection_stages, list) or len(selection_stages) != len(
        STAGES
    ):
        raise SelectedPrerequisiteError(
            "selection does not contain the ordered five stages"
        )

    selected: dict[str, Any] = {}
    selected_paths: set[Path] = set()
    for expected_stage_index, stage in enumerate(STAGES):
        index_receipt = exact_keys(
            stage_receipts[stage],
            {"stage", "path", "sha256", "receipt_payload_sha256"},
            f"{stage} measurement index receipt",
        )
        if index_receipt["stage"] != stage:
            raise SelectedPrerequisiteError(
                f"{stage} measurement index receipt changed"
            )
        stage_measurement, measurement_receipt, measured_candidates = (
            _validate_stage_measurement(
                stage=stage,
                artifact={
                    key: index_receipt[key]
                    for key in PAYLOAD_ARTIFACT_KEYS
                },
                measurement_index=measurement_index,
                candidate_index=candidate_index,
                candidates=candidates[stage],
            )
        )
        row = exact_keys(
            selection_stages[expected_stage_index],
            STAGE_KEYS,
            f"{stage} selection",
        )
        winner = min(
            measured_candidates,
            key=lambda candidate: (
                candidate["selection_score"],
                candidate["epoch"],
                candidate["optimizer_updates"],
                candidate["candidate_checkpoint"]["sha256"],
            ),
        )
        epoch = require_exact_int(row["epoch"], f"{stage} selected epoch")
        updates = require_exact_int(
            row["optimizer_updates"],
            f"{stage} selected optimizer updates",
        )
        candidate_index_number = require_exact_int(
            row["candidate_index"],
            f"{stage} selected candidate index",
        )
        score = require_finite(
            row["selection_score"],
            f"{stage} selected score",
        )
        checkpoint = exact_keys(
            row["candidate_checkpoint"],
            CHECKPOINT_KEYS,
            f"{stage} selected checkpoint",
        )
        checkpoint_path = regular_file(
            checkpoint["path"],
            f"{stage} selected checkpoint",
        )
        checkpoint_binding = {
            "path": str(checkpoint_path),
            "sha256": require_sha256(
                checkpoint["sha256"],
                f"{stage} selected checkpoint SHA-256",
            ),
            "bytes": require_exact_int(
                checkpoint["bytes"],
                f"{stage} selected checkpoint bytes",
            ),
        }
        expected_candidate = candidates[stage].get(epoch)
        expected_coverage = {
            "split": "val",
            "test_visible": False,
            "clips": EXPECTED_VAL_CLIPS,
            "shards": EXPECTED_SHARDS,
            "exact_once": True,
            "all_finite": True,
        }
        measurement_binding_from_row = exact_keys(
            row["measurement_receipt"],
            PAYLOAD_ARTIFACT_KEYS,
            f"{stage} selected measurement receipt",
        )
        if (
            row["stage"] != stage
            or row["selection_metric"] != EXPECTED_METRICS[stage]
            or expected_candidate is None
            or updates != expected_candidate["optimizer_updates"]
            or candidate_index_number
            != EXPECTED_CANDIDATE_EPOCHS.index(epoch)
            or checkpoint_binding
            != {
                key: expected_candidate[key]
                for key in ("path", "sha256", "bytes")
            }
            or checkpoint_binding != winner["candidate_checkpoint"]
            or score != winner["selection_score"]
            or epoch != winner["epoch"]
            or measurement_binding_from_row != measurement_receipt
            or exact_keys(
                row["coverage"],
                COVERAGE_KEYS,
                f"{stage} selected coverage",
            )
            != expected_coverage
            or stage_measurement["coverage"]["windows_per_candidate"]
            != windows_per_candidate
            or checkpoint_path in selected_paths
        ):
            raise SelectedPrerequisiteError(
                f"{stage} selected winner binding mismatch"
            )
        selected_paths.add(checkpoint_path)
        selected[stage] = {
            "stage": stage,
            "selection_metric": row["selection_metric"],
            "epoch": epoch,
            "optimizer_updates": updates,
            "candidate_index": candidate_index_number,
            "selection_score": score,
            "candidate_audit_sha256": expected_candidate[
                "checkpoint_audit_sha256"
            ],
            "candidate_checkpoint": checkpoint_binding,
            "measurement_receipt": measurement_receipt,
            "coverage": expected_coverage,
        }

    return {
        "format": "semtalk_show_selected_prerequisite_bridge_v1",
        "selection": {
            "path": str(path),
            "sha256": observed_file_sha,
            "receipt_payload_sha256": claimed_payload_sha,
        },
        "candidate_index_receipt": candidate_receipt,
        "measurement_index_receipt": measurement_index_receipt,
        "canonical_view": canonical_view,
        "producer_sources": dict(selection_sources),
        "training_sources": dict(selection["training_sources"]),
        "config_sha256": dict(selection["config_sha256"]),
        "selected": selected,
        "global_verified_not_consumed": True,
        "test_visible": False,
    }


def revalidate_selected_prerequisites(receipt: Mapping[str, Any]) -> None:
    """Revalidate every selected file after a long Base feature transaction."""

    selection = receipt.get("selection")
    if not isinstance(selection, dict):
        raise SelectedPrerequisiteError("selection bridge receipt is missing")
    refreshed = load_selected_prerequisites(
        selection.get("path"),
        require_sha256(
            selection.get("sha256"),
            "selection bridge file SHA-256",
        ),
    )
    if refreshed != dict(receipt):
        raise SelectedPrerequisiteError(
            "selected prerequisite transaction changed while consumed"
        )
