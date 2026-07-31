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
OFFICIAL_BASELINE_COMMIT = "806b008c97bf51fce203e54109e4c22325253618"
EXPECTED_VAL_CLIPS = 1_715
EXPECTED_SHOW_SPLIT_COUNTS = {"train": 13_687, "val": 1_715, "test": 1_708}
EXPECTED_VAL_GLOBAL_INDEX_START = EXPECTED_SHOW_SPLIT_COUNTS["train"]
EXPECTED_VAL_GLOBAL_INDEX_STOP = (
    EXPECTED_VAL_GLOBAL_INDEX_START + EXPECTED_VAL_CLIPS
)
EXPECTED_SHARDS = 8
REQUIRED_CANDIDATE_EPOCHS = tuple(range(20, 201, 20))
# Backwards-compatible name for callers that only need the mandatory
# production schedule.  Receipt consumers must use ``candidate_epochs()``
# so a later formal run can append e220, e240, ... without weakening the
# required e20..e200 prefix.
EXPECTED_CANDIDATE_EPOCHS = REQUIRED_CANDIDATE_EPOCHS
RVQ_UPDATES_PER_EPOCH = 497
GLOBAL_UPDATES_PER_EPOCH = 1_988
# Compatibility alias for RVQ-only callers.  Five-stage consumers must call
# ``updates_per_epoch(stage)`` and must never apply this value to Global.
EXPECTED_UPDATES_PER_EPOCH = RVQ_UPDATES_PER_EPOCH
TARGET_SPEAKER_SCOPE = "all_speakers_0_1_2_3"
FPS = 30
WINDOW_LENGTH = 64
WINDOW_STRIDE = 20
RVQ_LEVELS = 6
CODEBOOK_SIZE = 256
STAGES = ("face", "hands", "upper", "lower", "global")
RVQ_STAGES = ("face", "hands", "upper", "lower")


def updates_per_epoch(stage: str) -> int:
    if stage in RVQ_STAGES:
        return RVQ_UPDATES_PER_EPOCH
    if stage == "global":
        return GLOBAL_UPDATES_PER_EPOCH
    raise ContractError(f"unknown prerequisite stage {stage!r}")


def updates_per_epoch_map(stages: Iterable[str]) -> dict[str, int]:
    return {stage: updates_per_epoch(stage) for stage in stages}


def validate_candidate_epochs(value: Any) -> tuple[int, ...]:
    """Validate the receipt-driven candidate inventory.

    Every formal receipt must contain the complete e20..e200 schedule.
    Future training may append strictly increasing 20-epoch boundaries, but
    may never remove, reorder, or insert candidates into the mandatory
    prefix.
    """

    if not isinstance(value, list):
        raise ContractError("candidate epochs must be a JSON list")
    epochs = tuple(
        require_exact_int(epoch, "candidate epoch") for epoch in value
    )
    required = REQUIRED_CANDIDATE_EPOCHS
    if (
        len(epochs) < len(required)
        or epochs[: len(required)] != required
        or any(
            epoch <= 0
            or epoch % 20 != 0
            or (index and epoch != epochs[index - 1] + 20)
            for index, epoch in enumerate(epochs)
        )
        or any(epoch <= required[-1] for epoch in epochs[len(required) :])
    ):
        raise ContractError(
            "candidate epochs must be the e20..e200 prefix followed by "
            "contiguous +20-epoch append-only boundaries"
        )
    return epochs


def candidate_epochs(candidate_index: Mapping[str, Any]) -> tuple[int, ...]:
    return validate_candidate_epochs(candidate_index.get("candidate_epochs"))


def candidate_epochs_by_stage(
    candidate_index: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Return the exact append-only validation inventory for every stage.

    Initial e20..e200 indexes intentionally keep the v2 common-schedule
    schema.  A segmented v3 index carries one independent schedule per stage
    so a frozen winner is never fabricated at a later boundary merely because
    another prerequisite still needs continuation.
    """

    stages = candidate_index.get("stages")
    if not isinstance(stages, Mapping):
        raise ContractError("candidate index stage coverage is missing")
    raw = candidate_index.get("candidate_epochs_by_stage")
    if raw is None:
        common = candidate_epochs(candidate_index)
        return {stage: common for stage in stages}
    if not isinstance(raw, dict) or set(raw) != set(stages):
        raise ContractError("per-stage candidate schedule coverage mismatch")
    return {
        stage: validate_candidate_epochs(raw[stage])
        for stage in stages
    }


def candidate_epochs_for_stage(
    candidate_index: Mapping[str, Any], stage: str
) -> tuple[int, ...]:
    schedules = candidate_epochs_by_stage(candidate_index)
    if stage not in schedules:
        raise ContractError(f"unknown candidate stage {stage!r}")
    return schedules[stage]


def is_candidate_epoch(epoch: Any) -> bool:
    return (
        type(epoch) is int
        and epoch >= REQUIRED_CANDIDATE_EPOCHS[0]
        and epoch % 20 == 0
    )

VAL_CANONICAL_SUMMARY_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_summary_v1"
)
VAL_CANONICAL_LINEAGE_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_lineage_v1"
)
CANDIDATE_INDEX_FORMAT = "semtalk_show_prerequisite_candidate_index_v2"
PARTIAL_CANDIDATE_INDEX_FORMAT = (
    "semtalk_show_prerequisite_nonglobal_candidate_index_v2"
)
SEGMENTED_CANDIDATE_INDEX_FORMAT = (
    "semtalk_show_prerequisite_candidate_index_v3"
)
SEGMENTED_PARTIAL_CANDIDATE_INDEX_FORMAT = (
    "semtalk_show_prerequisite_nonglobal_candidate_index_v3"
)
SEGMENTED_UNION_FORMAT = (
    "semtalk_show_prerequisite_segmented_candidate_union_v2"
)
CONTINUATION_WAVE_FORMAT = "semtalk_show_prerequisite_continuation_wave_v1"
PER_STAGE_CONTINUATION_WAVE_FORMAT = (
    "semtalk_show_prerequisite_continuation_wave_v2"
)
SHARD_FORMAT = "semtalk_show_prerequisite_val_shard_v1"
STAGE_MEASUREMENT_FORMAT = (
    "semtalk_show_prerequisite_val_stage_measurement_v1"
)
MEASUREMENT_FORMAT = "semtalk_show_prerequisite_val_measurement_index_v1"
SELECTION_FORMAT = "semtalk_show_prerequisite_val_selection_v1"
PER_STAGE_MEASUREMENT_FORMAT = (
    "semtalk_show_prerequisite_val_measurement_index_v2"
)
PER_STAGE_SELECTION_FORMAT = "semtalk_show_prerequisite_val_selection_v2"
TRAINING_SOURCE_FREEZE_FORMAT = "semtalk_show_training_source_freeze_v2"
SOURCE_POLICY_FORMAT = "semtalk_show_prerequisite_source_policy_v1"
PER_STAGE_SOURCE_POLICY_FORMAT = (
    "semtalk_show_prerequisite_source_policy_v2"
)

SHOW_SPEAKERS = {"oliver": 0, "chemistry": 1, "seth": 2, "conan": 3}
OFFICIAL_INITIALIZATION = {
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

REPRESENTATION_CANDIDATE_AUDIT_KEYS = (
    "format",
    "formal_stage",
    "completed_epochs",
    "optimizer_updates",
    "config_sha256",
    "lineage_manifest_sha256",
    "dataset_receipt_sha256",
    "source_receipt",
    "source_receipt_sha256",
    "initialization_receipt",
    "rvq_ema_prior_receipt",
    "distributed_training_receipt",
    "rvq_rank_state_receipt",
    "selection_status",
)
CONTINUATION_WAVE_BINDING_KEY = "continuation_wave_receipt"
OPTIMIZER_RUNTIME_BINDING_KEY = "optimizer_runtime_receipt"

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


def _safe_file_snapshot(
    value: Any,
    label: str,
    *,
    val_only: bool = True,
    require_path_identity: bool = False,
) -> tuple[Path, bytes]:
    if not isinstance(value, (str, os.PathLike)):
        raise ContractError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    if val_only:
        reject_forbidden_label(path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{label} does not exist: {path}") from error
    if val_only:
        reject_forbidden_label(resolved, label)
    if resolved != path:
        raise ContractError(
            f"{label} must be a regular non-symlink file at a canonical "
            f"path with no symlink ancestor: {path}"
        )
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise ContractError(f"{label} must be below the filesystem root")
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
            raise ContractError(
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
            raise ContractError(f"{label} changed while it was read")
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise ContractError(f"{label} size changed while it was read")
        if require_path_identity:
            current = os.stat(
                parts[-1],
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(current.st_mode) or any(
                getattr(after, field) != getattr(current, field)
                for field in fields
            ):
                raise ContractError(f"{label} path changed while it was read")
        if val_only:
            reject_forbidden_label(path, label)
        return path, payload
    except ContractError:
        raise
    except OSError as error:
        raise ContractError(f"cannot safely read {label}: {path}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _safe_directory(value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ContractError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{label} does not exist: {path}") from error
    if resolved != path:
        raise ContractError(
            f"{label} must be canonical with no symlink ancestor: {path}"
        )
    parts = path.parts
    if not parts or parts[0] != os.sep:
        raise ContractError(f"{label} must be below the filesystem root")
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
            raise ContractError(
                f"{label} must be a non-symlink directory"
            )
        return path
    except ContractError:
        raise
    except OSError as error:
        raise ContractError(
            f"cannot safely resolve {label}: {path}"
        ) from error
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def sha256_file(path: Path) -> str:
    _resolved, payload = _safe_file_snapshot(
        path,
        f"SHA-256 input {path}",
        val_only=False,
    )
    return hashlib.sha256(payload).hexdigest()


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
    path, _payload = _safe_file_snapshot(
        value,
        label,
        val_only=val_only,
    )
    return path


def read_file_snapshot(
    value: Any,
    label: str,
    *,
    val_only: bool = True,
) -> tuple[Path, bytes]:
    """Return one O_NOFOLLOW, same-descriptor snapshot of a regular file."""

    return _safe_file_snapshot(
        value,
        label,
        val_only=val_only,
        require_path_identity=True,
    )


def read_verified_file(
    value: Any,
    expected_sha256: Any,
    label: str,
    *,
    val_only: bool = True,
    require_path_identity: bool = False,
) -> tuple[Path, bytes, str]:
    expected = require_sha256(expected_sha256, f"{label} expected SHA-256")
    path, payload = _safe_file_snapshot(
        value,
        label,
        val_only=val_only,
        require_path_identity=require_path_identity,
    )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return path, payload, observed


def validate_training_audit_source(
    value: Any,
    label: str,
    *,
    reprove_entrypoint: bool = True,
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
    entrypoint_value = value["entrypoint"]
    if (
        not isinstance(entrypoint_value, str)
        or not Path(entrypoint_value).is_absolute()
    ):
        raise ContractError(f"{label}.entrypoint must be absolute")
    expected_sha = require_sha256(
        value["entrypoint_sha256"],
        f"{label}.entrypoint_sha256",
    )
    if reprove_entrypoint:
        entrypoint = regular_file(
            entrypoint_value,
            f"{label}.entrypoint",
            val_only=False,
        )
        if sha256_file(entrypoint) != expected_sha:
            raise ContractError(f"{label} entrypoint changed")
    return dict(value)


def _portable_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a nonempty relative path")
    path = Path(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ContractError(f"{label} is not a canonical relative path")
    return value


def portable_training_source_identity(
    value: Any,
    label: str,
) -> dict[str, str]:
    value = exact_keys(
        value,
        (
            "origin",
            "commit",
            "tree",
            "script_relative",
            "script_sha256",
        ),
        label,
    )
    if value["origin"] != EXPECTED_ORIGIN:
        raise ContractError(f"{label}.origin mismatch")
    require_git_oid(value["commit"], f"{label}.commit")
    require_git_oid(value["tree"], f"{label}.tree")
    _portable_relative(value["script_relative"], f"{label}.script_relative")
    require_sha256(value["script_sha256"], f"{label}.script_sha256")
    return dict(value)


def freeze_training_audit_source(
    value: Any,
    label: str,
) -> dict[str, Any]:
    """Prove and enrich the limited training receipt at index-freeze time."""
    training_audit = validate_training_audit_source(
        value,
        label,
        reprove_entrypoint=True,
    )
    entrypoint, _entrypoint_payload = _safe_file_snapshot(
        training_audit["entrypoint"],
        f"{label}.entrypoint",
        val_only=False,
    )

    def git(*arguments: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", str(entrypoint.parent), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        source_root = _safe_directory(
            git("rev-parse", "--show-toplevel"),
            f"{label}.source_root",
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
    baseline_result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "merge-base",
            "--is-ancestor",
            OFFICIAL_BASELINE_COMMIT,
            commit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        origin != training_audit["origin"]
        or commit != training_audit["commit"]
        or tree != training_audit["tree"]
        or dirty
        or symbolic
        or local_heads
        or baseline_result.returncode != 0
    ):
        raise ContractError(
            f"{label} checkout is not exact, clean, detached, branch-free"
        )
    return receipt_payload(
        {
            "format": TRAINING_SOURCE_FREEZE_FORMAT,
            "training_audit": training_audit,
            "source_root": str(source_root),
            "portable_identity": {
                "origin": origin,
                "commit": commit,
                "tree": tree,
                "script_relative": relative,
                "script_sha256": training_audit["entrypoint_sha256"],
            },
            "clean": True,
            "detached": True,
            "local_branch_count": 0,
            "official_baseline_commit": OFFICIAL_BASELINE_COMMIT,
            "official_baseline_is_ancestor": True,
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
            "portable_identity",
            "clean",
            "detached",
            "local_branch_count",
            "official_baseline_commit",
            "official_baseline_is_ancestor",
            "receipt_payload_sha256",
        ),
        label,
    )
    training_audit = validate_training_audit_source(
        value["training_audit"],
        f"{label}.training_audit",
        reprove_entrypoint=False,
    )
    portable = portable_training_source_identity(
        value["portable_identity"],
        f"{label}.portable_identity",
    )
    if (
        value["format"] != TRAINING_SOURCE_FREEZE_FORMAT
        or portable["origin"] != training_audit["origin"]
        or portable["commit"] != training_audit["commit"]
        or portable["tree"] != training_audit["tree"]
        or portable["script_sha256"]
        != training_audit["entrypoint_sha256"]
        or value["clean"] is not True
        or value["detached"] is not True
        or value["local_branch_count"] != 0
        or value["official_baseline_commit"] != OFFICIAL_BASELINE_COMMIT
        or value["official_baseline_is_ancestor"] is not True
    ):
        raise ContractError(f"{label} freeze binding mismatch")
    root = Path(value["source_root"])
    if not root.is_absolute():
        raise ContractError(f"{label}.source_root must be absolute")
    entrypoint = Path(training_audit["entrypoint"])
    if not entrypoint.is_absolute():
        raise ContractError(f"{label} entrypoint must be absolute")
    try:
        relative = entrypoint.relative_to(root).as_posix()
    except ValueError as error:
        raise ContractError(f"{label} entrypoint escapes source root") from error
    if relative != portable["script_relative"]:
        raise ContractError(f"{label} portable script path mismatch")
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


def build_source_policy(
    source_receipts: Mapping[str, Any],
    *,
    stages: Sequence[str],
    reprove_ancestry: bool,
    independent_stage_sources: bool = False,
) -> dict[str, Any]:
    expected_stages = tuple(stages)
    if (
        not expected_stages
        or any(stage not in STAGES for stage in expected_stages)
        or len(set(expected_stages)) != len(expected_stages)
        or not isinstance(source_receipts, Mapping)
        or set(source_receipts) != set(expected_stages)
    ):
        raise ContractError("source policy stage coverage mismatch")
    portable = {
        stage: portable_training_source_identity(
            source_receipts[stage]["portable_identity"],
            f"{stage} source policy identity",
        )
        for stage in expected_stages
    }
    if independent_stage_sources:
        # A per-stage continuation wave proves old -> new ancestry for every
        # active stage independently.  Frozen stages intentionally retain
        # their prior immutable source receipt, so requiring all four RVQs to
        # finish on one commit would make a heterogeneous continuation
        # impossible to publish.  The v2 policy records every terminal source
        # explicitly; the segmented-union replay supplies the ancestry proof.
        policy = {
            "format": PER_STAGE_SOURCE_POLICY_FORMAT,
            "mode": "independent_stage_source_ancestry_v2",
            "stages": list(expected_stages),
            "origin": EXPECTED_ORIGIN,
            "portable_identity_sha256_by_stage": {
                stage: canonical_payload_sha256(portable[stage])
                for stage in expected_stages
            },
            "commit_by_stage": {
                stage: portable[stage]["commit"] for stage in expected_stages
            },
            "tree_by_stage": {
                stage: portable[stage]["tree"] for stage in expected_stages
            },
            "same_origin": True,
            "ancestry_authority": PER_STAGE_CONTINUATION_WAVE_FORMAT,
        }
        policy["receipt_payload_sha256"] = canonical_payload_sha256(policy)
        return policy
    rvq_stages = [stage for stage in expected_stages if stage in RVQ_STAGES]
    rvq_identity_hashes = {
        canonical_payload_sha256(portable[stage]) for stage in rvq_stages
    }
    if len(rvq_identity_hashes) != 1:
        raise ContractError("source policy RVQ identities differ")
    rvq_identity = portable[rvq_stages[0]]
    global_identity = portable.get("global")
    descendant: bool | None = None
    if global_identity is not None:
        if global_identity["origin"] != rvq_identity["origin"]:
            raise ContractError("source policy origins differ")
        if reprove_ancestry:
            global_root = Path(source_receipts["global"]["source_root"])
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(global_root),
                    "merge-base",
                    "--is-ancestor",
                    rvq_identity["commit"],
                    global_identity["commit"],
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ContractError(
                    "Global training source is not an RVQ-source descendant"
                )
        descendant = True
    policy = {
        "format": SOURCE_POLICY_FORMAT,
        "mode": "rvq_common_global_descendant_v1",
        "stages": list(expected_stages),
        "origin": rvq_identity["origin"],
        "rvq_portable_identity_sha256": canonical_payload_sha256(
            rvq_identity
        ),
        "rvq_commit": rvq_identity["commit"],
        "rvq_tree": rvq_identity["tree"],
        "global_portable_identity_sha256": (
            canonical_payload_sha256(global_identity)
            if global_identity is not None
            else None
        ),
        "global_commit": (
            global_identity["commit"] if global_identity is not None else None
        ),
        "global_tree": (
            global_identity["tree"] if global_identity is not None else None
        ),
        "same_origin": True if global_identity is not None else None,
        "global_descends_from_rvq": descendant,
    }
    policy["receipt_payload_sha256"] = canonical_payload_sha256(policy)
    return policy


def verify_named_compact_hash(
    value: Any,
    *,
    hash_key: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    claimed = require_sha256(value.get(hash_key), f"{label}.{hash_key}")
    unsigned = dict(value)
    unsigned.pop(hash_key, None)
    if canonical_payload_sha256(unsigned) != claimed:
        raise ContractError(f"{label} compact self-hash mismatch")
    return dict(value)


def validate_initialization_receipt(
    value: Any,
    *,
    stage: str,
    label: str,
    reprove_path: bool,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        (
            "stage",
            "path",
            "filename",
            "sha256",
            "official_all_speakers",
            "withdrawn_e30_allowed",
            "model_state_sha256",
        ),
        label,
    )
    spec = OFFICIAL_INITIALIZATION.get(stage)
    if spec is None:
        raise ContractError(f"{label}: unsupported stage")
    path_value = value["path"]
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
    ):
        raise ContractError(f"{label}.path must be absolute")
    reject_forbidden_label(path_value, f"{label}.path")
    checkpoint_sha = require_sha256(value["sha256"], f"{label}.sha256")
    require_sha256(
        value["model_state_sha256"],
        f"{label}.model_state_sha256",
    )
    if (
        value["stage"] != stage
        or value["filename"] != spec["filename"]
        or Path(path_value).name != spec["filename"]
        or checkpoint_sha != spec["sha256"]
        or value["official_all_speakers"] is not True
        or value["withdrawn_e30_allowed"] is not False
    ):
        raise ContractError(f"{label} official All-Speakers binding mismatch")
    if reprove_path:
        resolved = regular_file(path_value, f"{label}.path")
        if sha256_file(resolved) != checkpoint_sha:
            raise ContractError(f"{label} checkpoint changed")
    return dict(value)


def validate_distributed_training_receipt(
    value: Any,
    *,
    stage: str,
    label: str,
) -> dict[str, Any]:
    value = exact_keys(
        verify_named_compact_hash(
            value,
            hash_key="receipt_sha256",
            label=label,
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
        label,
    )
    world_size = require_exact_int(value["world_size"], f"{label}.world_size")
    local_batch = require_exact_int(
        value["local_batch_size"],
        f"{label}.local_batch_size",
    )
    seed = None
    sampler = value["sampler"]
    if not isinstance(sampler, dict):
        raise ContractError(f"{label}.sampler must be an object")
    seed = require_exact_int(sampler.get("seed"), f"{label}.sampler.seed")
    if seed < 0:
        raise ContractError(f"{label}.sampler.seed must be nonnegative")
    if stage in RVQ_STAGES:
        expected_sampler = {
            "class": "torch.utils.data.distributed.DistributedSampler",
            "shuffle": True,
            "seed": seed,
            "drop_last": True,
            "set_epoch": "before every epoch",
        }
        expected_ema = {
            "enabled": True,
            "assignment": "rank-local Gumbel samples from seed + global rank",
            "statistics": ["code_count", "code_sum"],
            "collective": "all_reduce SUM",
            "initialization": "global rank-ordered prefix; rank0 broadcast",
            "dead_code_reset": (
                "global rank-ordered prefix on demand; rank0 broadcast"
            ),
            "perplexity": "global code_count all_reduce SUM",
            "rank_state": "exact SHA-256 agreement at save/resume/finalize",
        }
        valid_parallelism = (
            world_size in {2, 4}
            and local_batch * world_size == 256
        )
    else:
        expected_sampler = {
            "class": "RandomSampler",
            "shuffle": True,
            "seed": seed,
            "drop_last": True,
            "set_epoch": None,
        }
        expected_ema = {"enabled": False}
        valid_parallelism = world_size == 1 and local_batch == 64
    stage_global_batch = 256 if stage in RVQ_STAGES else 64
    stage_updates = updates_per_epoch(stage)
    integer_expectations = {
        "global_batch_size": stage_global_batch,
        "train_samples": 127_286,
        "available_train_samples": 127_286,
        "consumed_samples_per_epoch": stage_updates * stage_global_batch,
        "dropped_samples_per_epoch": (
            127_286 - stage_updates * stage_global_batch
        ),
        "padding_or_duplicate_samples_per_epoch": 0,
        "updates_per_epoch": stage_updates,
    }
    if (
        value["format"] != "semtalk_show_representation_ddp_v1"
        or value["formal_stage"] != stage
        or not valid_parallelism
        or value["loader_drop_last"] is not True
        or value["sampler"] != expected_sampler
        or value["rvq_ema"] != expected_ema
    ):
        raise ContractError(f"{label} distributed protocol mismatch")
    for key, expected in integer_expectations.items():
        if require_exact_int(value[key], f"{label}.{key}") != expected:
            raise ContractError(f"{label}.{key} mismatch")
    return dict(value)


def validate_optimizer_runtime_receipt(
    value: Any,
    *,
    stage: str,
    label: str,
    required: bool = False,
) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise ContractError(f"{label} is required")
        return None
    value = exact_keys(
        verify_named_compact_hash(
            value,
            hash_key="receipt_sha256",
            label=label,
        ),
        (
            "format",
            "formal_stage",
            "class",
            "base_learning_rate",
            "betas",
            "weight_decay",
            "eps",
            "amsgrad",
            "parameter_groups",
            "trained_parameter_tensors",
            "receipt_sha256",
        ),
        label,
    )
    expected_base_lr = 6e-4 if stage in RVQ_STAGES else (
        1.5e-4 if stage == "global" else 5e-5
    )
    groups = require_exact_int(
        value["parameter_groups"], f"{label}.parameter_groups"
    )
    trained = require_exact_int(
        value["trained_parameter_tensors"],
        f"{label}.trained_parameter_tensors",
    )
    if (
        value["format"] != "semtalk_show_optimizer_runtime_v1"
        or value["formal_stage"] != stage
        or value["class"] != "torch.optim.Adam"
        or value["base_learning_rate"] != expected_base_lr
        or value["betas"] != [0.5, 0.999]
        or value["weight_decay"] != 0.0
        or value["eps"] != 1e-8
        or value["amsgrad"] is not False
        or groups <= 0
        or trained <= 0
    ):
        raise ContractError(f"{label} optimizer protocol mismatch")
    return dict(value)


def validate_rvq_ema_prior_receipt(
    value: Any,
    *,
    stage: str,
    label: str,
) -> dict[str, Any] | None:
    if stage == "global":
        if value is not None:
            raise ContractError(f"{label} must be null for Global")
        return None
    value = exact_keys(value, ("format", "layers"), label)
    layers = value["layers"]
    if (
        value["format"] != "semtalk_show_official_rvq_ema_prior_v2"
        or not isinstance(layers, list)
        or len(layers) != RVQ_LEVELS
    ):
        raise ContractError(f"{label} RVQ EMA-prior schema mismatch")
    names: set[str] = set()
    for index, layer in enumerate(layers):
        layer = exact_keys(
            layer,
            ("name", "ema_decay", "prior_count"),
            f"{label}.layers[{index}]",
        )
        name = layer["name"]
        decay = require_finite(
            layer["ema_decay"],
            f"{label}.layers[{index}].ema_decay",
        )
        prior = require_finite(
            layer["prior_count"],
            f"{label}.layers[{index}].prior_count",
        )
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not 0.0 < decay < 1.0
            or prior <= 1.0
            or not math.isclose(prior, 1.0 / (1.0 - decay))
        ):
            raise ContractError(f"{label} invalid RVQ EMA-prior layer")
        names.add(name)
    return dict(value)


def validate_rvq_rank_state_receipt(
    value: Any,
    *,
    stage: str,
    distributed: Mapping[str, Any],
    label: str,
) -> dict[str, Any] | None:
    if stage == "global":
        if value is not None:
            raise ContractError(f"{label} must be null for Global")
        return None
    value = exact_keys(
        value,
        ("format", "world_size", "state_sha256", "all_ranks_exact"),
        label,
    )
    if (
        value["format"] != "semtalk_show_rvq_rank_state_v1"
        or require_exact_int(value["world_size"], f"{label}.world_size")
        != distributed["world_size"]
        or value["all_ranks_exact"] is not True
    ):
        raise ContractError(f"{label} rank-state mismatch")
    require_sha256(value["state_sha256"], f"{label}.state_sha256")
    return dict(value)


def validate_representation_candidate_audit(
    value: Any,
    *,
    stage: str,
    epoch: int,
    label: str,
    reprove_paths: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    expected_keys = set(REPRESENTATION_CANDIDATE_AUDIT_KEYS)
    optional_keys = {
        CONTINUATION_WAVE_BINDING_KEY,
        OPTIMIZER_RUNTIME_BINDING_KEY,
    }
    if not expected_keys.issubset(value) or (
        set(value) - expected_keys
    ) - optional_keys:
        raise ContractError(f"{label} schema mismatch")
    if (
        stage == "global"
        and OPTIMIZER_RUNTIME_BINDING_KEY not in value
    ):
        raise ContractError(f"{label} lacks Global optimizer binding")
    continuation_wave = value.get(CONTINUATION_WAVE_BINDING_KEY)
    if continuation_wave is not None:
        continuation_wave = exact_keys(
            continuation_wave,
            ("path", "sha256", "receipt_payload_sha256"),
            f"{label}.continuation_wave_receipt",
        )
        wave_path = continuation_wave["path"]
        if (
            not isinstance(wave_path, str)
            or not Path(wave_path).is_absolute()
        ):
            raise ContractError(
                f"{label} continuation wave path must be absolute"
            )
        require_sha256(
            continuation_wave["sha256"],
            f"{label} continuation wave SHA-256",
        )
        require_sha256(
            continuation_wave["receipt_payload_sha256"],
            f"{label} continuation wave payload SHA-256",
        )
    updates = epoch * updates_per_epoch(stage)
    if (
        stage not in STAGES
        or not is_candidate_epoch(epoch)
        or value["format"] != "semtalk_show_representation_candidate_v1"
        or value["formal_stage"] != stage
        or require_exact_int(
            value["completed_epochs"],
            f"{label}.completed_epochs",
        )
        != epoch
        or require_exact_int(
            value["optimizer_updates"],
            f"{label}.optimizer_updates",
        )
        != updates
        or value["selection_status"] != "offline_validation_pending"
    ):
        raise ContractError(f"{label} candidate protocol mismatch")
    for key in (
        "config_sha256",
        "lineage_manifest_sha256",
        "dataset_receipt_sha256",
    ):
        require_sha256(value[key], f"{label}.{key}")
    source = validate_training_audit_source(
        value["source_receipt"],
        f"{label}.source_receipt",
        reprove_entrypoint=reprove_paths,
    )
    if (
        require_sha256(
            value["source_receipt_sha256"],
            f"{label}.source_receipt_sha256",
        )
        != canonical_payload_sha256(source)
    ):
        raise ContractError(f"{label} source payload hash mismatch")
    validate_initialization_receipt(
        value["initialization_receipt"],
        stage=stage,
        label=f"{label}.initialization_receipt",
        reprove_path=reprove_paths,
    )
    distributed = validate_distributed_training_receipt(
        value["distributed_training_receipt"],
        stage=stage,
        label=f"{label}.distributed_training_receipt",
    )
    validate_optimizer_runtime_receipt(
        value.get(OPTIMIZER_RUNTIME_BINDING_KEY),
        stage=stage,
        label=f"{label}.optimizer_runtime_receipt",
        required=stage == "global",
    )
    validate_rvq_ema_prior_receipt(
        value["rvq_ema_prior_receipt"],
        stage=stage,
        label=f"{label}.rvq_ema_prior_receipt",
    )
    validate_rvq_rank_state_receipt(
        value["rvq_rank_state_receipt"],
        stage=stage,
        distributed=distributed,
        label=f"{label}.rvq_rank_state_receipt",
    )
    return dict(value)


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
    summary_index_start = require_exact_int(
        summary_value.get("global_index_start"),
        "validation canonical summary global_index_start",
    )
    summary_index_stop = require_exact_int(
        summary_value.get("global_index_stop_exclusive"),
        "validation canonical summary global_index_stop_exclusive",
    )
    lineage_index_start = require_exact_int(
        lineage_value.get("global_index_start"),
        "validation canonical lineage global_index_start",
    )
    lineage_index_stop = require_exact_int(
        lineage_value.get("global_index_stop_exclusive"),
        "validation canonical lineage global_index_stop_exclusive",
    )
    summary_indices_sha = require_sha256(
        summary_value.get("global_indices_sha256"),
        "validation canonical summary global_indices_sha256",
    )
    lineage_indices_sha = require_sha256(
        lineage_value.get("global_indices_sha256"),
        "validation canonical lineage global_indices_sha256",
    )
    if (
        summary_value.get("format") != VAL_CANONICAL_SUMMARY_FORMAT
        or summary_value.get("status") != "complete"
        or summary_value.get("split") != "val"
        or summary_value.get("test_visible") is not False
        or summary_value.get("clip_count") != EXPECTED_VAL_CLIPS
        or summary_value.get("manifest_sha256") != manifest_sha
        or summary_value.get("lineage_sha256") != lineage_sha
        or summary_index_start != EXPECTED_VAL_GLOBAL_INDEX_START
        or summary_index_stop != EXPECTED_VAL_GLOBAL_INDEX_STOP
        or lineage_index_start != summary_index_start
        or lineage_index_stop != summary_index_stop
        or lineage_indices_sha != summary_indices_sha
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
    canonical_paths: list[Path] = []
    speakers: set[str] = set()
    required_row_keys = {
        "global_index",
        "clip_id",
        "split",
        "speaker",
        "speaker_id",
        "frames",
        "canonical_npz",
        "canonical_npz_sha256",
    }
    for line_number, row in enumerate(rows, 1):
        if not required_row_keys <= set(row):
            raise ContractError(
                f"validation row {line_number} schema is incomplete"
            )
        index = require_exact_int(
            row.get("global_index"),
            f"validation row {line_number} global_index",
        )
        clip_id = row.get("clip_id")
        speaker = row.get("speaker")
        speaker_id = require_exact_int(
            row.get("speaker_id"),
            f"validation row {line_number} speaker_id",
        )
        frames = require_exact_int(
            row.get("frames"),
            f"validation row {line_number} frames",
        )
        if (
            row.get("split") != "val"
            or not isinstance(clip_id, str)
            or not clip_id
            or not isinstance(speaker, str)
            or SHOW_SPEAKERS.get(speaker) != speaker_id
            or frames <= 0
            or window_count(frames) <= 0
        ):
            raise ContractError(f"validation row {line_number} is not val-only")
        reject_forbidden_label(clip_id, f"validation row {line_number} clip_id")
        canonical_path = regular_file(
            row.get("canonical_npz"),
            f"validation row {line_number} canonical_npz",
        )
        require_sha256(
            row.get("canonical_npz_sha256"),
            f"validation row {line_number} canonical_npz_sha256",
        )
        indices.append(index)
        clip_ids.append(clip_id)
        canonical_paths.append(canonical_path)
        speakers.add(speaker)
    if (
        indices != list(range(summary_index_start, summary_index_stop))
        or canonical_payload_sha256(indices) != summary_indices_sha
        or len(set(clip_ids)) != EXPECTED_VAL_CLIPS
        or len(set(canonical_paths)) != EXPECTED_VAL_CLIPS
        or speakers != set(SHOW_SPEAKERS)
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


def _validate_bound_prior_candidate_index(
    value: Any,
    *,
    path: Path,
    artifact_stack: frozenset[Path],
) -> dict[str, Any]:
    return validate_candidate_index(
        value,
        path=path,
        allow_partial=False,
        _artifact_stack=artifact_stack,
    )


def _replay_bound_continuation_wave(
    value: Any,
    *,
    path: Path,
    expected_sha256: str,
    artifact_stack: frozenset[Path],
) -> dict[str, Any]:
    # Late import avoids the contract/wave module import cycle.
    from scripts.show_base import prerequisite_continuation_wave as wave_contract

    try:
        replayed = wave_contract.replay_wave_file(
            path,
            expected_sha256,
            _artifact_stack=artifact_stack,
        )
    except (wave_contract.ContinuationWaveError, ContractError) as error:
        raise ContractError(
            f"continuation wave semantic replay failed: {error}"
        ) from error
    if replayed != value:
        raise ContractError("continuation wave semantic replay changed payload")
    return replayed


def validate_candidate_index(
    value: Any,
    *,
    path: Path | None = None,
    allow_partial: bool = False,
    reprove_source_ancestry: bool = False,
    _artifact_stack: frozenset[Path] | None = None,
) -> dict[str, Any]:
    artifact_stack = frozenset() if _artifact_stack is None else _artifact_stack
    if path is not None:
        candidate_path = Path(path)
        if candidate_path in artifact_stack:
            raise ContractError("segmented candidate index cycle detected")
        artifact_stack = artifact_stack | {candidate_path}
    payload = verify_receipt_payload(value, "candidate index")
    candidate_format = payload.get("format")
    is_segmented = candidate_format in {
        SEGMENTED_CANDIDATE_INDEX_FORMAT,
        SEGMENTED_PARTIAL_CANDIDATE_INDEX_FORMAT,
    }
    is_partial = candidate_format in {
        PARTIAL_CANDIDATE_INDEX_FORMAT,
        SEGMENTED_PARTIAL_CANDIDATE_INDEX_FORMAT,
    }
    expected_stages = STAGES[:-1] if is_partial else STAGES
    if (
        candidate_format
        not in {
            CANDIDATE_INDEX_FORMAT,
            PARTIAL_CANDIDATE_INDEX_FORMAT,
            SEGMENTED_CANDIDATE_INDEX_FORMAT,
            SEGMENTED_PARTIAL_CANDIDATE_INDEX_FORMAT,
        }
        or (is_partial and not allow_partial)
        or payload.get("status") != "complete"
        or payload.get("target_dataset") != "SHOW"
        or payload.get("target_speaker_scope") != TARGET_SPEAKER_SCOPE
        or payload.get("selection_split") != "val"
        or payload.get("test_visible") is not False
        or payload.get("updates_per_epoch")
        != updates_per_epoch_map(expected_stages)
    ):
        raise ContractError("candidate index protocol mismatch")
    schedule = validate_candidate_epochs(payload.get("candidate_epochs"))
    stages = payload.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(expected_stages):
        raise ContractError("candidate index stage coverage mismatch")
    if is_segmented:
        schedules = candidate_epochs_by_stage(payload)
        union = sorted({epoch for values in schedules.values() for epoch in values})
        if list(schedule) != union or payload.get("segmented_union") is None:
            raise ContractError("segmented candidate schedule union mismatch")
    else:
        if "candidate_epochs_by_stage" in payload or "segmented_union" in payload:
            raise ContractError("initial candidate index cannot bypass v3")
        schedules = {stage: schedule for stage in expected_stages}
    source_receipts = payload.get("source_receipts")
    if not isinstance(source_receipts, dict) or set(source_receipts) != set(
        expected_stages
    ):
        raise ContractError("candidate index training source coverage mismatch")
    for stage in expected_stages:
        validate_frozen_training_source(
            source_receipts[stage],
            f"{stage} frozen training source",
            reprove_checkout=False,
        )
    expected_source_policy = build_source_policy(
        source_receipts,
        stages=expected_stages,
        reprove_ancestry=reprove_source_ancestry,
        independent_stage_sources=is_segmented,
    )
    if payload.get("source_policy") != expected_source_policy:
        raise ContractError("candidate index source policy mismatch")
    rvq_portable_sources = {
        canonical_payload_sha256(
            portable_training_source_identity(
                source_receipts[stage]["portable_identity"],
                f"{stage} portable training source",
            )
        )
        for stage in expected_stages
        if stage in RVQ_STAGES
    }
    # The four RVQs are one synchronized DDP production group and must be
    # byte-identical in source.  Global is an independent VAEConvZero model;
    # it may use a separately frozen official descendant source so a
    # Global-only contract correction never forces scientifically unrelated
    # RVQs to be retrained or relabelled.
    if not is_segmented and len(rvq_portable_sources) != 1:
        raise ContractError("candidate index RVQ training sources differ")
    observed_paths: set[Path] = set()
    for stage in expected_stages:
        entries = stages[stage]
        stage_schedule = schedules[stage]
        if not isinstance(entries, list) or len(entries) != len(stage_schedule):
            raise ContractError(f"{stage} candidate coverage mismatch")
        for expected_epoch, entry in zip(stage_schedule, entries):
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
                or updates != epoch * updates_per_epoch(stage)
            ):
                raise ContractError(f"{stage} candidate schedule mismatch")
            checkpoint, checkpoint_payload = read_file_snapshot(
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
                hashlib.sha256(checkpoint_payload).hexdigest() != expected_sha
                or len(checkpoint_payload)
                != require_exact_int(
                    entry["checkpoint_bytes"],
                    f"{stage} checkpoint bytes",
                )
            ):
                raise ContractError(f"{stage} candidate file changed")
    segmented = payload.get("segmented_union")
    if segmented is not None:
        segmented = exact_keys(
            segmented,
            (
                "format",
                "status",
                "prior_candidate_index",
                "continuation_waves",
                "candidate_segment_chain",
                "receipt_payload_sha256",
            ),
            "segmented candidate union",
        )
        segmented_unsigned = dict(segmented)
        segmented_claimed = segmented_unsigned.pop(
            "receipt_payload_sha256"
        )
        if (
            segmented["format"] != SEGMENTED_UNION_FORMAT
            or segmented["status"] != "complete"
            or require_sha256(
                segmented_claimed,
                "segmented candidate union payload SHA",
            )
            != canonical_payload_sha256(segmented_unsigned)
        ):
            raise ContractError("segmented candidate union protocol mismatch")

        def validate_binding(
            value: Any,
            label: str,
            *,
            expected_format: str | tuple[str, ...],
        ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
            binding = exact_keys(
                value,
                ("path", "sha256", "bytes", "receipt_payload_sha256"),
                label,
            )
            artifact_sha = require_sha256(binding["sha256"], f"{label} SHA")
            artifact_bytes = require_exact_int(
                binding["bytes"], f"{label} bytes"
            )
            expected_payload_sha = require_sha256(
                binding["receipt_payload_sha256"],
                f"{label} payload SHA",
            )
            artifact_path, artifact_payload, _observed_sha = read_verified_file(
                binding["path"],
                artifact_sha,
                label,
                require_path_identity=True,
            )
            if artifact_bytes <= 0 or len(artifact_payload) != artifact_bytes:
                raise ContractError(f"{label} changed")
            artifact_value = strict_json_bytes(
                artifact_payload, str(artifact_path)
            )
            artifact_value = verify_receipt_payload(artifact_value, label)
            expected_formats = (
                (expected_format,)
                if isinstance(expected_format, str)
                else expected_format
            )
            if artifact_value.get("format") not in expected_formats:
                raise ContractError(f"{label} format mismatch")
            if (
                artifact_value["receipt_payload_sha256"]
                != expected_payload_sha
            ):
                raise ContractError(f"{label} payload SHA mismatch")
            return dict(binding), artifact_path, artifact_value

        prior, prior_path, prior_value = validate_binding(
            segmented["prior_candidate_index"],
            "prior candidate index",
            expected_format=(
                CANDIDATE_INDEX_FORMAT,
                SEGMENTED_CANDIDATE_INDEX_FORMAT,
            ),
        )
        if path is not None and prior["path"] == str(path):
            raise ContractError("segmented candidate index is self-referential")
        _validate_bound_prior_candidate_index(
            prior_value,
            path=prior_path,
            artifact_stack=artifact_stack,
        )
        waves = segmented["continuation_waves"]
        if not isinstance(waves, list) or not waves:
            raise ContractError("segmented candidate union has no waves")
        normalized_waves = []
        for index, value in enumerate(waves):
            binding, wave_path, wave_value = validate_binding(
                value,
                f"continuation wave {index}",
                expected_format=PER_STAGE_CONTINUATION_WAVE_FORMAT,
            )
            _replay_bound_continuation_wave(
                wave_value,
                path=wave_path,
                expected_sha256=binding["sha256"],
                artifact_stack=artifact_stack,
            )
            normalized_waves.append(binding)
        if len({value["path"] for value in normalized_waves}) != len(waves):
            raise ContractError("segmented continuation wave path was reused")
        chains = segmented["candidate_segment_chain"]
        if not isinstance(chains, dict) or set(chains) != set(expected_stages):
            raise ContractError("segmented candidate chain coverage mismatch")
        for stage in expected_stages:
            chain = chains[stage]
            if not isinstance(chain, list) or not chain:
                raise ContractError(f"{stage} segmented chain is empty")
            expected_start = REQUIRED_CANDIDATE_EPOCHS[0]
            predecessor: str | None = None
            covered: list[int] = []
            runs: set[str] = set()
            for index, raw_segment in enumerate(chain):
                segment = exact_keys(
                    raw_segment,
                    (
                        "run_path",
                        "start_epoch",
                        "end_epoch",
                        "candidate_epochs",
                        "predecessor_segment_id",
                        "segment_id",
                    ),
                    f"{stage} segment {index}",
                )
                run_path = Path(segment["run_path"])
                start = require_exact_int(
                    segment["start_epoch"], f"{stage} segment start"
                )
                end = require_exact_int(
                    segment["end_epoch"], f"{stage} segment end"
                )
                epochs = segment["candidate_epochs"]
                if (
                    not run_path.is_absolute()
                    or ".." in run_path.parts
                    or str(run_path) != segment["run_path"]
                    or segment["run_path"] in runs
                    or start != expected_start
                    or end < start
                    or not isinstance(epochs, list)
                    or epochs != list(range(start, end + 1, 20))
                    or segment["predecessor_segment_id"] != predecessor
                ):
                    raise ContractError(f"{stage} segmented chain mismatch")
                expected_id = canonical_payload_sha256(
                    {
                        key: value
                        for key, value in segment.items()
                        if key != "segment_id"
                    }
                )
                if (
                    require_sha256(
                        segment["segment_id"],
                        f"{stage} segment ID",
                    )
                    != expected_id
                ):
                    raise ContractError(f"{stage} segment ID changed")
                runs.add(segment["run_path"])
                covered.extend(epochs)
                predecessor = segment["segment_id"]
                expected_start = end + 20
            if covered != list(schedules[stage]):
                raise ContractError(f"{stage} segmented schedule mismatch")
            for entry in stages[stage]:
                matching = [
                    segment
                    for segment in chain
                    if entry["epoch"] in segment["candidate_epochs"]
                ]
                if (
                    len(matching) != 1
                    or Path(entry["checkpoint"]).parent
                    != Path(matching[0]["run_path"])
                    / "representation_candidates"
                ):
                    raise ContractError(
                        f"{stage} candidate escapes its immutable segment"
                    )
    return payload


def load_candidate_index(
    path: Path,
    expected_sha256: str,
    *,
    allow_partial: bool = False,
    _artifact_stack: frozenset[Path] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved, payload, sha = read_verified_file(
        path,
        expected_sha256,
        "candidate index",
        require_path_identity=True,
    )
    value = strict_json_bytes(payload, str(resolved))
    return (
        validate_candidate_index(
            value,
            path=resolved,
            allow_partial=allow_partial,
            _artifact_stack=_artifact_stack,
        ),
        _artifact(resolved, sha),
    )


def candidate_lookup(
    candidate_index: Mapping[str, Any],
    stage: str,
    epoch: int,
) -> dict[str, Any]:
    if (
        stage not in STAGES
        or stage not in candidate_index.get("stages", {})
        or epoch not in candidate_epochs_for_stage(candidate_index, stage)
    ):
        raise ContractError("candidate lookup outside the frozen schedule")
    matches = [
        entry
        for entry in candidate_index["stages"][stage]
        if entry["epoch"] == epoch
    ]
    if len(matches) != 1:
        raise ContractError("candidate index is not exact-once")
    return dict(matches[0])


def validate_selection_receipt(value: Any) -> dict[str, Any]:
    payload = verify_receipt_payload(value, "selection receipt")
    payload = exact_keys(
        payload,
        (
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
        ),
        "selection receipt",
    )
    if (
        payload["format"] not in {SELECTION_FORMAT, PER_STAGE_SELECTION_FORMAT}
        or payload["status"] != "selected"
        or payload["target_dataset"] != "SHOW"
        or payload["target_speaker_scope"] != TARGET_SPEAKER_SCOPE
        or payload["split"] != "val"
        or payload["test_visible"] is not False
    ):
        raise ContractError("selection receipt protocol mismatch")
    if payload["format"] == PER_STAGE_SELECTION_FORMAT:
        protocol = exact_keys(
            payload["protocol"],
            (
                "name",
                "candidate_epochs",
                "candidate_epochs_by_stage",
                "candidates_per_stage",
                "clips_per_candidate",
                "shards_per_candidate",
                "window_length",
                "window_stride",
                "full_base_fgd_used",
            ),
            "selection receipt protocol",
        )
        epochs = validate_candidate_epochs(protocol["candidate_epochs"])
        raw_schedules = protocol["candidate_epochs_by_stage"]
        if not isinstance(raw_schedules, dict) or set(raw_schedules) != set(STAGES):
            raise ContractError("selection per-stage schedule coverage mismatch")
        schedules = {
            stage: validate_candidate_epochs(raw_schedules[stage])
            for stage in STAGES
        }
        union = sorted({epoch for values in schedules.values() for epoch in values})
        if protocol != {
            "name": "five_independent_show_prerequisite_validation_v2",
            "candidate_epochs": union,
            "candidate_epochs_by_stage": {
                stage: list(schedules[stage]) for stage in STAGES
            },
            "candidates_per_stage": {
                stage: len(schedules[stage]) for stage in STAGES
            },
            "clips_per_candidate": EXPECTED_VAL_CLIPS,
            "shards_per_candidate": EXPECTED_SHARDS,
            "window_length": WINDOW_LENGTH,
            "window_stride": WINDOW_STRIDE,
            "full_base_fgd_used": False,
        }:
            raise ContractError("selection receipt candidate protocol mismatch")
    else:
        protocol = exact_keys(
            payload["protocol"],
            (
                "name",
                "candidate_epochs",
                "candidates_per_stage",
                "clips_per_candidate",
                "shards_per_candidate",
                "window_length",
                "window_stride",
                "full_base_fgd_used",
            ),
            "selection receipt protocol",
        )
        epochs = validate_candidate_epochs(protocol["candidate_epochs"])
        schedules = {stage: epochs for stage in STAGES}
        if protocol != {
            "name": "five_independent_show_prerequisite_validation_v1",
            "candidate_epochs": list(epochs),
            "candidates_per_stage": len(epochs),
            "clips_per_candidate": EXPECTED_VAL_CLIPS,
            "shards_per_candidate": EXPECTED_SHARDS,
            "window_length": WINDOW_LENGTH,
            "window_stride": WINDOW_STRIDE,
            "full_base_fgd_used": False,
        }:
            raise ContractError("selection receipt candidate protocol mismatch")
    stages = payload["stages"]
    if (
        not isinstance(stages, list)
        or len(stages) != len(STAGES)
        or any(not isinstance(stage, dict) for stage in stages)
        or [stage.get("stage") for stage in stages] != list(STAGES)
    ):
        raise ContractError("selection receipt stage coverage mismatch")
    for stage, result in zip(STAGES, stages):
        result = exact_keys(
            result,
            (
                "stage",
                "selection_metric",
                "epoch",
                "optimizer_updates",
                "candidate_index",
                "selection_score",
                "candidate_checkpoint",
                "measurement_receipt",
                "coverage",
            ),
            f"{stage} selection result",
        )
        epoch = require_exact_int(result["epoch"], f"{stage} selected epoch")
        if (
            result["stage"] != stage
            or result["selection_metric"] != SELECTION_METRICS[stage]
            or epoch not in schedules[stage]
            or result["candidate_index"] != schedules[stage].index(epoch)
            or result["optimizer_updates"]
            != epoch * updates_per_epoch(stage)
        ):
            raise ContractError(f"{stage} selection result mismatch")
        require_finite(
            result["selection_score"],
            f"{stage} selection score",
            nonnegative=True,
        )
    return payload


def exact_keys(value: Any, keys: Sequence[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ContractError(f"{label} schema mismatch")
    return value
