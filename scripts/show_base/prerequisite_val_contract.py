#!/usr/bin/env python3
"""Shared fail-closed contract for SHOW prerequisite validation.

The five representation prerequisites are selected independently on the
frozen SHOW validation split.  This module is intentionally stdlib-only so
candidate merging and selection never import PyTorch or inspect test data.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_VAL_CLIPS = 1_715
EXPECTED_SHARDS = 8
EXPECTED_CANDIDATE_EPOCHS = tuple(range(20, 201, 20))
EXPECTED_UPDATES_PER_EPOCH = 497
TARGET_SPEAKER_SCOPE = "all_speakers_0_1_2_3"
FPS = 30
WINDOW_LENGTH = 64
WINDOW_STRIDE = 20
RVQ_LEVELS = 6
CODEBOOK_SIZE = 256
STAGES = ("face", "hands", "upper", "lower", "global")
RVQ_STAGES = ("face", "hands", "upper", "lower")

VAL_CANONICAL_SUMMARY_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_summary_v1"
)
VAL_CANONICAL_LINEAGE_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_lineage_v1"
)
CANDIDATE_INDEX_FORMAT = "semtalk_show_prerequisite_candidate_index_v1"
SHARD_FORMAT = "semtalk_show_prerequisite_val_shard_v1"
STAGE_MEASUREMENT_FORMAT = (
    "semtalk_show_prerequisite_val_stage_measurement_v1"
)
MEASUREMENT_FORMAT = "semtalk_show_prerequisite_val_measurement_index_v1"
SELECTION_FORMAT = "semtalk_show_prerequisite_val_selection_v1"
TRAINING_SOURCE_FREEZE_FORMAT = "semtalk_show_training_source_freeze_v1"

SELECTION_METRICS = {
    "face": "face_geometry_expression_objective_v1",
    "hands": "hands_rotation_geometry_objective_v1",
    "upper": "upper_rotation_geometry_objective_v1",
    "lower": "lower_rotation_contact_objective_v1",
    "global": "global_root_contact_objective_v1",
}

ACCUMULATOR_KEYS = {
    "face": (
        "rotation_geodesic",
        "rotation_velocity",
        "rotation_acceleration",
        "expression",
        "expression_velocity",
        "expression_acceleration",
    ),
    "hands": (
        "rotation_geodesic",
        "rotation_velocity",
        "rotation_acceleration",
    ),
    "upper": (
        "rotation_geodesic",
        "rotation_velocity",
        "rotation_acceleration",
    ),
    "lower": (
        "rotation_geodesic",
        "rotation_velocity",
        "rotation_acceleration",
        "contact",
        "translation",
    ),
    "global": (
        "contact",
        "velocity_x",
        "velocity_z",
        "velocity_delta_x",
        "velocity_delta_z",
        "velocity_acceleration_x",
        "velocity_acceleration_z",
        "integrated_translation_velocity",
        "integrated_translation_acceleration",
        "integrated_translation",
    ),
}

_FORBIDDEN_E30 = re.compile(
    r"(^|[^a-z0-9])e[-_]?30([^a-z0-9]|$)",
    re.IGNORECASE,
)
_FORBIDDEN_SPEAKER2 = re.compile(
    r"(^|[^a-z0-9])speaker[-_]?2([^a-z0-9]|$)",
    re.IGNORECASE,
)
_TEST_COMPONENTS = frozenset({"test", "tests", "testset", "testsets"})


class ContractError(RuntimeError):
    """Raised when validation evidence is incomplete or ambiguous."""


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


def canonical_payload_sha256(value: Any) -> str:
    # Receipt payload hashes intentionally exclude the JSON-file newline.
    # This is the frozen convention used by build_val_canonical_view.py.
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def window_count(frames: Any) -> int:
    frames = require_exact_int(frames, "frames")
    if frames <= 0:
        raise ContractError("frames must be positive")
    usable = (frames // FPS) * FPS
    return max(0, (usable - WINDOW_LENGTH) // WINDOW_STRIDE + 1)


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase Git object ID")
    return value


def require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an exact integer")
    return value


def require_finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ContractError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ContractError(f"{label} must be finite and valid")
    return result


def strict_json_bytes(payload: bytes, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ContractError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"{label}: non-finite JSON constant {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ContractError(f"{label}: invalid UTF-8") from error


def strict_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        value = strict_json_bytes(line, f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise ContractError(f"{label}:{line_number}: expected object")
        result.append(value)
    return result


def reject_forbidden_label(value: Any, label: str) -> None:
    text = str(value).replace("\\", "/")
    for component in text.split("/"):
        normalized = component.casefold()
        stem = normalized.rsplit(".", 1)[0]
        tokens = frozenset(filter(None, re.split(r"[^a-z0-9]+", stem)))
        if tokens & _TEST_COMPONENTS:
            raise ContractError(f"{label} exposes a test-labelled path")
        if _FORBIDDEN_E30.search(normalized):
            raise ContractError(f"{label} exposes withdrawn e30")
        if _FORBIDDEN_SPEAKER2.search(normalized):
            raise ContractError(f"{label} exposes forbidden Speaker2")


def regular_file(
    value: Any,
    label: str,
    *,
    val_only: bool = True,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ContractError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    if val_only:
        reject_forbidden_label(path, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise ContractError(f"{label} does not exist: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ContractError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if val_only:
        reject_forbidden_label(resolved, label)
    return resolved


def read_verified_file(
    value: Any,
    expected_sha256: Any,
    label: str,
    *,
    val_only: bool = True,
) -> tuple[Path, bytes, str]:
    expected = require_sha256(expected_sha256, f"{label} expected SHA-256")
    path = regular_file(value, label, val_only=val_only)
    payload = path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return path, payload, observed


def validate_training_audit_source(
    value: Any,
    label: str,
) -> dict[str, Any]:
    """Validate the exact limited source receipt embedded by e2d training."""
    value = exact_keys(
        value,
        (
            "commit",
            "tree",
            "origin",
            "entrypoint",
            "entrypoint_sha256",
        ),
        label,
    )
    if value["origin"] != EXPECTED_ORIGIN:
        raise ContractError(f"{label} origin mismatch")
    require_git_oid(value["commit"], f"{label}.commit")
    require_git_oid(value["tree"], f"{label}.tree")
    entrypoint = regular_file(
        value["entrypoint"],
        f"{label}.entrypoint",
        val_only=False,
    )
    expected_sha = require_sha256(
        value["entrypoint_sha256"],
        f"{label}.entrypoint_sha256",
    )
    if sha256_file(entrypoint) != expected_sha:
        raise ContractError(f"{label} entrypoint changed")
    return dict(value)


def freeze_training_audit_source(
    value: Any,
    label: str,
) -> dict[str, Any]:
    """Prove and enrich the limited training receipt at index-freeze time."""
    training_audit = validate_training_audit_source(value, label)
    entrypoint = Path(training_audit["entrypoint"]).resolve(strict=True)

    def git(*arguments: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", str(entrypoint.parent), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        source_root = Path(git("rev-parse", "--show-toplevel")).resolve(
            strict=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ContractError(f"{label} entrypoint is not in Git") from error
    try:
        relative = str(entrypoint.relative_to(source_root))
    except ValueError as error:
        raise ContractError(f"{label} entrypoint escapes source root") from error
    if git("ls-files", "--error-unmatch", relative) != relative:
        raise ContractError(f"{label} entrypoint is not tracked")
    origin = git("remote", "get-url", "origin")
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    symbolic = git("symbolic-ref", "-q", "--short", "HEAD", check=False)
    local_heads = [
        line
        for line in git(
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
        ).splitlines()
        if line
    ]
    if (
        origin != training_audit["origin"]
        or commit != training_audit["commit"]
        or tree != training_audit["tree"]
        or dirty
        or symbolic
        or local_heads
    ):
        raise ContractError(
            f"{label} checkout is not exact, clean, detached, branch-free"
        )
    return receipt_payload(
        {
            "format": TRAINING_SOURCE_FREEZE_FORMAT,
            "training_audit": training_audit,
            "source_root": str(source_root),
            "origin": origin,
            "commit": commit,
            "tree": tree,
            "clean": True,
            "detached": True,
            "local_branch_count": 0,
            "entrypoint_relative": relative,
            "entrypoint_sha256": training_audit["entrypoint_sha256"],
        }
    )


def validate_frozen_training_source(
    value: Any,
    label: str,
    *,
    reprove_checkout: bool,
) -> dict[str, Any]:
    value = verify_receipt_payload(value, label)
    value = exact_keys(
        value,
        (
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
        ),
        label,
    )
    training_audit = validate_training_audit_source(
        value["training_audit"],
        f"{label}.training_audit",
    )
    if (
        value["format"] != TRAINING_SOURCE_FREEZE_FORMAT
        or value["origin"] != EXPECTED_ORIGIN
        or value["origin"] != training_audit["origin"]
        or value["commit"] != training_audit["commit"]
        or value["tree"] != training_audit["tree"]
        or value["clean"] is not True
        or value["detached"] is not True
        or value["local_branch_count"] != 0
        or value["entrypoint_sha256"]
        != training_audit["entrypoint_sha256"]
    ):
        raise ContractError(f"{label} freeze binding mismatch")
    root = Path(value["source_root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ContractError(f"{label}.source_root is invalid")
    root = root.resolve(strict=True)
    entrypoint = Path(training_audit["entrypoint"]).resolve(strict=True)
    try:
        relative = str(entrypoint.relative_to(root))
    except ValueError as error:
        raise ContractError(f"{label} entrypoint escapes source root") from error
    if relative != value["entrypoint_relative"]:
        raise ContractError(f"{label} entrypoint relative path mismatch")
    if reprove_checkout:
        observed = freeze_training_audit_source(training_audit, label)
        if observed != value:
            raise ContractError(f"{label} live freeze proof changed")
    return dict(value)


def receipt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_payload_sha256"] = canonical_payload_sha256(result)
    return result


def verify_receipt_payload(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be an object")
    claimed = require_sha256(
        payload.get("receipt_payload_sha256"),
        f"{label} receipt payload SHA-256",
    )
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256", None)
    observed = canonical_payload_sha256(unsigned)
    if claimed != observed:
        raise ContractError(
            f"{label} receipt payload mismatch: {claimed} != {observed}"
        )
    return payload


def atomic_json_new(path: Path, payload: Mapping[str, Any]) -> str:
    if not path.is_absolute():
        raise ContractError("output path must be absolute")
    reject_forbidden_label(path, "output path")
    parent = path.parent.resolve(strict=True)
    if path.parent != parent:
        raise ContractError("output parent must be canonical")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite output: {path}")
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise FileExistsError(path) from error
            raise
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return hashlib.sha256(encoded).hexdigest()
    except Exception:
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _artifact(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256}


def load_val_canonical(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    summary_path: Path,
    summary_sha256: str,
    lineage_path: Path,
    lineage_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, manifest_bytes, manifest_sha = read_verified_file(
        manifest_path,
        manifest_sha256,
        "validation canonical manifest",
    )
    summary, summary_bytes, summary_sha = read_verified_file(
        summary_path,
        summary_sha256,
        "validation canonical summary",
    )
    lineage, lineage_bytes, lineage_sha = read_verified_file(
        lineage_path,
        lineage_sha256,
        "validation canonical lineage",
    )
    summary_value = verify_receipt_payload(
        strict_json_bytes(summary_bytes, str(summary)),
        "validation canonical summary",
    )
    lineage_value = verify_receipt_payload(
        strict_json_bytes(lineage_bytes, str(lineage)),
        "validation canonical lineage",
    )
    if (
        summary_value.get("format") != VAL_CANONICAL_SUMMARY_FORMAT
        or summary_value.get("status") != "complete"
        or summary_value.get("split") != "val"
        or summary_value.get("test_visible") is not False
        or summary_value.get("clip_count") != EXPECTED_VAL_CLIPS
        or summary_value.get("manifest_sha256") != manifest_sha
        or summary_value.get("lineage_sha256") != lineage_sha
        or lineage_value.get("format") != VAL_CANONICAL_LINEAGE_FORMAT
        or lineage_value.get("status") != "complete"
        or lineage_value.get("split") != "val"
        or lineage_value.get("test_visible") is not False
        or lineage_value.get("clip_count") != EXPECTED_VAL_CLIPS
        or lineage_value.get("manifest_sha256") != manifest_sha
    ):
        raise ContractError("validation canonical receipt binding mismatch")
    rows = strict_jsonl_bytes(manifest_bytes, str(manifest))
    if len(rows) != EXPECTED_VAL_CLIPS:
        raise ContractError("validation canonical clip count mismatch")
    indices: list[int] = []
    clip_ids: list[str] = []
    for line_number, row in enumerate(rows, 1):
        index = require_exact_int(
            row.get("global_index"),
            f"validation row {line_number} global_index",
        )
        clip_id = row.get("clip_id")
        if (
            row.get("split") != "val"
            or not isinstance(clip_id, str)
            or not clip_id
        ):
            raise ContractError(f"validation row {line_number} is not val-only")
        reject_forbidden_label(clip_id, f"validation row {line_number} clip_id")
        indices.append(index)
        clip_ids.append(clip_id)
    if (
        indices != sorted(indices)
        or len(set(indices)) != EXPECTED_VAL_CLIPS
        or len(set(clip_ids)) != EXPECTED_VAL_CLIPS
    ):
        raise ContractError("validation rows are not exact-once ordered")
    receipt = {
        "manifest": _artifact(manifest, manifest_sha),
        "summary": _artifact(summary, summary_sha),
        "lineage": _artifact(lineage, lineage_sha),
        "clip_count": EXPECTED_VAL_CLIPS,
        "clip_ids_sha256": canonical_payload_sha256(clip_ids),
        "split": "val",
        "test_visible": False,
    }
    return rows, receipt


def error_accumulator() -> dict[str, Any]:
    return {"count": 0, "sum_abs": 0.0, "sum_squared": 0.0, "max_abs": 0.0}


def validate_accumulator(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "count",
        "sum_abs",
        "sum_squared",
        "max_abs",
    }:
        raise ContractError(f"{label} accumulator schema mismatch")
    count = require_exact_int(value["count"], f"{label}.count")
    if count <= 0:
        raise ContractError(f"{label}.count must be positive")
    result = {"count": count}
    for key in ("sum_abs", "sum_squared", "max_abs"):
        result[key] = require_finite(
            value[key],
            f"{label}.{key}",
            nonnegative=True,
        )
    return result


def merge_accumulators(
    values: Iterable[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    result = error_accumulator()
    seen = 0
    for raw in values:
        value = validate_accumulator(raw, f"{label}[{seen}]")
        result["count"] += value["count"]
        result["sum_abs"] += value["sum_abs"]
        result["sum_squared"] += value["sum_squared"]
        result["max_abs"] = max(result["max_abs"], value["max_abs"])
        seen += 1
    if seen != EXPECTED_SHARDS:
        raise ContractError(f"{label} must cover exactly {EXPECTED_SHARDS} shards")
    return result


def finalize_accumulator(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    item = validate_accumulator(dict(value), label)
    count = item["count"]
    return {
        **item,
        "mae": item["sum_abs"] / count,
        "mse": item["sum_squared"] / count,
        "rmse": math.sqrt(item["sum_squared"] / count),
    }


def stage_metrics(
    stage: str,
    accumulators: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    if stage not in STAGES:
        raise ContractError(f"unknown prerequisite stage {stage!r}")
    expected = set(ACCUMULATOR_KEYS[stage])
    if not isinstance(accumulators, Mapping) or set(accumulators) != expected:
        raise ContractError(f"{stage} accumulator names mismatch")
    metrics = {
        name: finalize_accumulator(accumulators[name], f"{stage}.{name}")
        for name in ACCUMULATOR_KEYS[stage]
    }
    if stage == "face":
        score = (
            metrics["rotation_geodesic"]["mae"]
            + metrics["rotation_velocity"]["mae"]
            + metrics["rotation_acceleration"]["mae"]
            + metrics["expression"]["mse"]
            + metrics["expression_velocity"]["mae"]
            + metrics["expression_acceleration"]["mae"]
        )
    elif stage in {"hands", "upper"}:
        score = (
            metrics["rotation_geodesic"]["mae"]
            + metrics["rotation_velocity"]["mae"]
            + metrics["rotation_acceleration"]["mae"]
        )
    elif stage == "lower":
        score = (
            metrics["rotation_geodesic"]["mae"]
            + metrics["rotation_velocity"]["mae"]
            + metrics["rotation_acceleration"]["mae"]
            + metrics["contact"]["mse"]
        )
    else:
        score = (
            metrics["contact"]["mse"]
            + metrics["velocity_x"]["mae"]
            + metrics["velocity_z"]["mae"]
            + 5.0 * metrics["velocity_delta_x"]["mae"]
            + 5.0 * metrics["velocity_delta_z"]["mae"]
            + 5.0 * metrics["velocity_acceleration_x"]["mae"]
            + 5.0 * metrics["velocity_acceleration_z"]["mae"]
            + 5.0 * metrics["integrated_translation_velocity"]["mae"]
            + 5.0 * metrics["integrated_translation_acceleration"]["mae"]
            + metrics["integrated_translation"]["mae"]
        )
    score = require_finite(
        score,
        f"{stage} selection score",
        nonnegative=True,
    )
    return metrics, score


def validate_candidate_index(
    value: Any,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    payload = verify_receipt_payload(value, "candidate index")
    if (
        payload.get("format") != CANDIDATE_INDEX_FORMAT
        or payload.get("status") != "complete"
        or payload.get("target_dataset") != "SHOW"
        or payload.get("target_speaker_scope") != TARGET_SPEAKER_SCOPE
        or payload.get("selection_split") != "val"
        or payload.get("test_visible") is not False
        or payload.get("candidate_epochs")
        != list(EXPECTED_CANDIDATE_EPOCHS)
        or payload.get("updates_per_epoch") != EXPECTED_UPDATES_PER_EPOCH
    ):
        raise ContractError("candidate index protocol mismatch")
    stages = payload.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        raise ContractError("candidate index stage coverage mismatch")
    source_receipts = payload.get("source_receipts")
    if not isinstance(source_receipts, dict) or set(source_receipts) != set(
        STAGES
    ):
        raise ContractError("candidate index training source coverage mismatch")
    for stage in STAGES:
        validate_frozen_training_source(
            source_receipts[stage],
            f"{stage} frozen training source",
            reprove_checkout=False,
        )
    observed_paths: set[Path] = set()
    for stage in STAGES:
        entries = stages[stage]
        if not isinstance(entries, list) or len(entries) != len(
            EXPECTED_CANDIDATE_EPOCHS
        ):
            raise ContractError(f"{stage} candidate coverage mismatch")
        for expected_epoch, entry in zip(EXPECTED_CANDIDATE_EPOCHS, entries):
            if not isinstance(entry, dict) or set(entry) != {
                "epoch",
                "optimizer_updates",
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_bytes",
                "checkpoint_audit_sha256",
            }:
                raise ContractError(f"{stage} candidate schema mismatch")
            epoch = require_exact_int(entry["epoch"], f"{stage} epoch")
            updates = require_exact_int(
                entry["optimizer_updates"],
                f"{stage} optimizer updates",
            )
            if (
                epoch != expected_epoch
                or updates != epoch * EXPECTED_UPDATES_PER_EPOCH
            ):
                raise ContractError(f"{stage} candidate schedule mismatch")
            checkpoint = regular_file(
                entry["checkpoint"],
                f"{stage} candidate checkpoint",
            )
            if checkpoint in observed_paths:
                raise ContractError("candidate checkpoint path is reused")
            observed_paths.add(checkpoint)
            expected_sha = require_sha256(
                entry["checkpoint_sha256"],
                f"{stage} checkpoint SHA-256",
            )
            require_sha256(
                entry["checkpoint_audit_sha256"],
                f"{stage} checkpoint audit SHA-256",
            )
            if (
                sha256_file(checkpoint) != expected_sha
                or checkpoint.stat().st_size
                != require_exact_int(
                    entry["checkpoint_bytes"],
                    f"{stage} checkpoint bytes",
                )
            ):
                raise ContractError(f"{stage} candidate file changed")
    if path is not None:
        regular_file(path, "candidate index")
    return payload


def load_candidate_index(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved, payload, sha = read_verified_file(
        path,
        expected_sha256,
        "candidate index",
    )
    value = strict_json_bytes(payload, str(resolved))
    return validate_candidate_index(value, path=resolved), _artifact(resolved, sha)


def candidate_lookup(
    candidate_index: Mapping[str, Any],
    stage: str,
    epoch: int,
) -> dict[str, Any]:
    if stage not in STAGES or epoch not in EXPECTED_CANDIDATE_EPOCHS:
        raise ContractError("candidate lookup outside the frozen schedule")
    matches = [
        entry
        for entry in candidate_index["stages"][stage]
        if entry["epoch"] == epoch
    ]
    if len(matches) != 1:
        raise ContractError("candidate index is not exact-once")
    return dict(matches[0])


def exact_keys(value: Any, keys: Sequence[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ContractError(f"{label} schema mismatch")
    return value
