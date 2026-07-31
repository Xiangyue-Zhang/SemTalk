#!/usr/bin/env python3
"""Plan and seal the immutable fresh SemTalk-Base SHOW validation run.

This module never generates motion itself.  The guarded shell launcher runs
``run_base_val_inference.py`` on exactly eight CUDA shards and this module
turns the resulting byte-pinned artifacts into one auditable candidate
transaction.  All 22 frozen candidates carry only a fresh released2 primary
screen receipt; the expensive full TalkSHOW report is reserved for the one
immutable selected winner.

The public commands are deliberately CPU-only.  Every output is create-new,
validation-only, and payload hashed.  No command accepts a test artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import published_test_winner_claim as authority
from scripts.show_base import select_published_base_winner as selector
from scripts.show_base import base_long_val_contract as val_contract


PLAN_FORMAT = "semtalk_show_base_fresh_val_plan_v1"
RUN_SPEC_FORMAT = "semtalk_show_base_fresh_val_run_spec_v1"
RUN_SPEC_AUDIT_FORMAT = "semtalk_show_base_fresh_val_run_spec_audit_v1"
CANDIDATE_SEAL_FORMAT = "semtalk_show_base_fresh_val_candidate_seal_v1"
PARTITION_FORMAT = "semtalk_show_base_fresh_val_partition_v1"
PARTITION_UNION_FORMAT = "semtalk_show_base_fresh_val_partition_union_v1"
MULTICANDIDATE_GATE_FORMAT = (
    "semtalk_show_base_fresh_val_multicandidate_gate_v1"
)
MULTICANDIDATE_COMPARISON_FORMAT = (
    "semtalk_show_base_fresh_val_multicandidate_comparison_v1"
)
MULTICANDIDATE_PROBE_RUN_FORMAT = (
    "semtalk_show_base_fresh_val_multicandidate_probe_run_v1"
)
MULTICANDIDATE_PROBE_METRIC_FORMAT = (
    "semtalk_show_base_fresh_val_multicandidate_probe_metric_v1"
)
CONCURRENCY_SELECTION_ORDER = (1, 2, 4)
CONCURRENCY_SELECTION_SAFETY_MARGIN = 0.02
PROBE_EPOCHS = (1, 2, 4, 8)
PROBE_CLIPS_PER_CANDIDATE = 64
PROBE_GPU_MEMORY_LIMIT_FRACTION = 0.95
CONCURRENCY_EXCLUDED_FIELDS = (
    "elapsed_seconds",
    "finished_at",
    "gpu_process_pid",
    "output_root",
    "started_at",
)
GENERATOR_MODULE = authority.FRESH_VAL_GENERATOR_MODULE
EXPECTED_EPOCHS = authority.BASE_CANDIDATE_EPOCHS
EXPECTED_SHARDS = authority.EXPECTED_SHARDS
EXPECTED_CLIPS = authority.EXPECTED_VAL_CLIPS
PAYLOAD_HASH_ALGORITHM = authority.PAYLOAD_HASH_ALGORITHM
FORMAL_HOST_BY_PARTITION = {
    0: authority.FRESH_VAL_FORMAL_HOSTS[0],
    1: authority.FRESH_VAL_FORMAL_HOSTS[1],
}


class BaseFreshValOrchestratorError(RuntimeError):
    """Raised when a fresh Base validation closure is incomplete."""


def _canonical_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise BaseFreshValOrchestratorError(
            "artifact is not finite canonical JSON"
        ) from error


def _payload_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _with_payload_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_payload_sha256"] = _payload_sha(result)
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_snapshot(path_value: Any, label: str) -> tuple[Path, bytes]:
    try:
        path = Path(path_value)
    except TypeError as error:
        raise BaseFreshValOrchestratorError(
            f"{label} must be path-like"
        ) from error
    if not path.is_absolute():
        raise BaseFreshValOrchestratorError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BaseFreshValOrchestratorError(
            f"cannot resolve {label}: {path}"
        ) from error
    if resolved != path or path.is_symlink():
        raise BaseFreshValOrchestratorError(
            f"{label} must be canonical and non-symlink"
        )
    try:
        resolved_identity = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise BaseFreshValOrchestratorError(
            f"cannot stat {label}: {path}"
        ) from error
    if not stat.S_ISREG(resolved_identity.st_mode):
        raise BaseFreshValOrchestratorError(
            f"{label} must be a regular non-symlink file"
        )
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise BaseFreshValOrchestratorError(
            f"{label} must be below the filesystem root"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | cloexec
    )
    file_flags = os.O_RDONLY | nofollow | cloexec
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
            raise BaseFreshValOrchestratorError(
                f"{label} must be a regular non-symlink file"
            )
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(resolved_identity, field) != getattr(before, field)
            for field in stable
        ):
            raise BaseFreshValOrchestratorError(
                f"{label} changed before it was opened"
            )
        opened_path = os.stat(path, follow_symlinks=False)
        identity = ("st_dev", "st_ino", "st_mode")
        if any(
            getattr(before, field) != getattr(opened_path, field)
            for field in identity
        ):
            raise BaseFreshValOrchestratorError(
                f"{label} path changed while it was opened"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        final_path = os.stat(path, follow_symlinks=False)
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(final_path, field)
            for field in stable
        ):
            raise BaseFreshValOrchestratorError(
                f"{label} changed while it was read"
            )
        try:
            final_resolved = path.resolve(strict=True)
        except OSError as error:
            raise BaseFreshValOrchestratorError(
                f"{label} path changed while it was read"
            ) from error
        if final_resolved != path:
            raise BaseFreshValOrchestratorError(
                f"{label} acquired a symlink while it was read"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise BaseFreshValOrchestratorError(
                f"{label} size changed while it was read"
            )
        return path, payload
    except BaseFreshValOrchestratorError:
        raise
    except OSError as error:
        raise BaseFreshValOrchestratorError(
            f"cannot safely read {label}: {path}"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _strict_json(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BaseFreshValOrchestratorError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except BaseFreshValOrchestratorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BaseFreshValOrchestratorError(
            f"{label} is not strict JSON: {error}"
        ) from error


def _exact(value: Any, keys: Iterable[str], label: str) -> dict[str, Any]:
    expected = set(keys)
    if not isinstance(value, dict) or set(value) != expected:
        raise BaseFreshValOrchestratorError(f"{label} schema mismatch")
    return value


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise BaseFreshValOrchestratorError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _artifact(
    path_value: Any,
    expected_sha: str | None = None,
    *,
    payload_receipt: bool,
) -> tuple[dict[str, Any], Any]:
    path, payload = _safe_snapshot(path_value, "artifact")
    observed = _sha256_bytes(payload)
    if expected_sha is not None and observed != _require_sha(
        expected_sha, "artifact SHA-256"
    ):
        raise BaseFreshValOrchestratorError(
            f"artifact SHA-256 changed: {path}"
        )
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": observed,
        "bytes": len(payload),
    }
    value: Any = None
    if payload_receipt:
        value = _strict_json(payload, str(path))
        if not isinstance(value, dict):
            raise BaseFreshValOrchestratorError(
                f"payload receipt is not an object: {path}"
            )
        claimed = _require_sha(
            value.get("receipt_payload_sha256"),
            f"{path} receipt payload SHA-256",
        )
        if _payload_sha(value) != claimed:
            raise BaseFreshValOrchestratorError(
                f"payload receipt hash mismatch: {path}"
            )
        result["receipt_payload_sha256"] = claimed
    return result, value


def _write_new(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_absolute():
        raise BaseFreshValOrchestratorError("output path must be absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = _canonical_bytes(value)
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise BaseFreshValOrchestratorError(
            "output must be below the filesystem root"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | cloexec
    )
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | cloexec
    directory_fd: int | None = None
    descriptor: int | None = None
    created = False
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
        descriptor = os.open(
            parts[-1], file_flags, 0o600, dir_fd=directory_fd
        )
        created = True
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            after = os.fstat(handle.fileno())
        final = os.stat(
            parts[-1], dir_fd=directory_fd, follow_symlinks=False
        )
        stable = ("st_dev", "st_ino", "st_mode")
        if (
            not stat.S_ISREG(before.st_mode)
            or any(
                getattr(before, field) != getattr(after, field)
                or getattr(after, field) != getattr(final, field)
                for field in stable
            )
            or after.st_size != len(payload)
            or final.st_size != len(payload)
        ):
            raise BaseFreshValOrchestratorError(
                "output identity changed while it was written"
            )
    except BaseException:
        if created and directory_fd is not None:
            try:
                os.unlink(parts[-1], dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)
    artifact, _ = _artifact(path, payload_receipt=True)
    return artifact


def _create_new_directory_tree(
    path: Path,
    *,
    subdirectories: Sequence[str] = (),
) -> tuple[Path, int, int]:
    """Create one formal run root without a pathname TOCTOU window.

    Every existing parent component is opened relative to the preceding
    directory with ``O_NOFOLLOW``.  The absent basename is created with
    ``mkdirat``, immediately reopened through the pinned parent descriptor,
    and compared to a non-following ``fstatat`` identity.  Requested direct
    children are created through the new root descriptor by the same rule.
    A failure deliberately leaves the newly created root in place so no
    formal caller can silently reuse a partial transaction.
    """

    if not isinstance(path, Path) or not path.is_absolute():
        raise BaseFreshValOrchestratorError(
            "formal run root must be an absolute Path"
        )
    if path == Path(os.sep) or path.name in {"", ".", ".."}:
        raise BaseFreshValOrchestratorError(
            "formal run root must be below the filesystem root"
        )
    try:
        canonical_parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise BaseFreshValOrchestratorError(
            "formal run-root parent does not exist"
        ) from error
    if canonical_parent / path.name != path:
        raise BaseFreshValOrchestratorError(
            "formal run root must use a canonical non-symlink parent"
        )
    children: list[str] = []
    for value in subdirectories:
        if (
            not isinstance(value, str)
            or value in {"", ".", ".."}
            or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
            or value in children
        ):
            raise BaseFreshValOrchestratorError(
                "formal run-root child must be one unique safe basename"
            )
        children.append(value)

    parts = path.parts
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | cloexec
    )
    parent_fd: int | None = None
    root_fd: int | None = None
    try:
        parent_fd = os.open(os.sep, directory_flags)
        for component in parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            identity = os.fstat(next_fd)
            if not stat.S_ISDIR(identity.st_mode):
                os.close(next_fd)
                raise BaseFreshValOrchestratorError(
                    "formal run-root parent traversal left a directory"
                )
            os.close(parent_fd)
            parent_fd = next_fd
        try:
            os.mkdir(parts[-1], mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to reuse formal run root {path}"
            ) from error
        root_fd = os.open(parts[-1], directory_flags, dir_fd=parent_fd)
        opened = os.fstat(root_fd)
        linked = os.stat(
            parts[-1], dir_fd=parent_fd, follow_symlinks=False
        )
        identity_fields = ("st_dev", "st_ino", "st_mode")
        if (
            not stat.S_ISDIR(opened.st_mode)
            or any(
                getattr(opened, field) != getattr(linked, field)
                for field in identity_fields
            )
        ):
            raise BaseFreshValOrchestratorError(
                "formal run-root identity changed during mkdirat"
            )
        for child in children:
            os.mkdir(child, mode=0o700, dir_fd=root_fd)
            child_fd = os.open(child, directory_flags, dir_fd=root_fd)
            try:
                child_opened = os.fstat(child_fd)
                child_linked = os.stat(
                    child, dir_fd=root_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISDIR(child_opened.st_mode)
                    or any(
                        getattr(child_opened, field)
                        != getattr(child_linked, field)
                        for field in identity_fields
                    )
                ):
                    raise BaseFreshValOrchestratorError(
                        "formal run-root child identity changed"
                    )
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
        os.fsync(root_fd)
        os.fsync(parent_fd)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise BaseFreshValOrchestratorError(
                "formal run root disappeared after creation"
            ) from error
        final = os.stat(path, follow_symlinks=False)
        if (
            resolved != path
            or any(
                getattr(opened, field) != getattr(final, field)
                for field in identity_fields
            )
        ):
            raise BaseFreshValOrchestratorError(
                "formal run-root pathname differs from mkdirat identity"
            )
        return path, opened.st_dev, opened.st_ino
    except (BaseFreshValOrchestratorError, FileExistsError):
        raise
    except OSError as error:
        raise BaseFreshValOrchestratorError(
            f"cannot safely create formal run root {path}: {error}"
        ) from error
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _compact_from_args(
    args: argparse.Namespace, prefix: str
) -> dict[str, Any]:
    attr = prefix.replace("-", "_")
    artifact, _ = _artifact(
        getattr(args, f"{attr}_path"),
        getattr(args, f"{attr}_sha256"),
        payload_receipt=True,
    )
    claimed_payload = getattr(args, f"{attr}_payload_sha256")
    if artifact["receipt_payload_sha256"] != claimed_payload:
        raise BaseFreshValOrchestratorError(
            f"{prefix} payload SHA-256 changed"
        )
    return artifact


def _preflight_candidate(
    preflight_artifact: Mapping[str, Any], epoch: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized, preflight = authority._verify_compact_receipt(
        preflight_artifact, "fresh Base validation preflight"
    )
    if (
        preflight.get("format") != authority.FRESH_VAL_PREFLIGHT_FORMAT
        or preflight.get("status") != "complete"
        or preflight.get("split") != "val"
        or preflight.get("test_visible") is not False
        or preflight.get("candidate_epochs") != list(EXPECTED_EPOCHS)
        or preflight.get("coverage", {}).get("clip_count") != EXPECTED_CLIPS
    ):
        raise BaseFreshValOrchestratorError(
            "preflight is not the frozen 22-candidate validation contract"
        )
    candidate = (
        preflight.get("candidate_bundle", {})
        .get("candidates", {})
        .get(str(epoch))
    )
    if not isinstance(candidate, dict):
        raise BaseFreshValOrchestratorError(
            f"preflight lacks Base e{epoch}"
        )
    checkpoint = authority._validate_checkpoint(
        candidate, f"Base e{epoch} checkpoint"
    )
    return normalized, {**preflight, "_candidate": checkpoint}


def _selected_updates_per_epoch(preflight: Mapping[str, Any]) -> int:
    bundle = preflight.get("candidate_bundle")
    if not isinstance(bundle, dict):
        raise BaseFreshValOrchestratorError(
            "preflight candidate bundle is absent"
        )
    updates = bundle.get("updates_per_epoch")
    topology = bundle.get("selected_topology")
    if (
        type(updates) is not int
        or updates not in {248, 1988}
        or not isinstance(topology, dict)
        or topology.get("updates_per_epoch") != updates
        or topology.get("mode") not in {
            "official_objective_w8_l8_g64_ddp_adaptation",
            "official_objective_w16_l4_g64_ddp_adaptation",
            "validation_gated_w8_l64_g512_empirical_acceleration",
            "validation_gated_w16_l32_g512_empirical_acceleration",
        }
    ):
        raise BaseFreshValOrchestratorError(
            "preflight lacks one selected non-W1 Base topology"
        )
    return updates


def build_plan(
    *,
    preflight_artifact: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_preflight, preflight = _preflight_candidate(
        preflight_artifact, EXPECTED_EPOCHS[0]
    )
    normalized_prerequisite, prerequisite, _fixed = (
        authority._validate_prerequisite_selection(prerequisite_artifact)
    )
    normalized_continuation, _decision = (
        authority._validate_continuation_decision(
            continuation_artifact,
            prerequisite_artifact=normalized_prerequisite,
            prerequisite_selection=prerequisite,
        )
    )
    candidates = preflight["candidate_bundle"]["candidates"]
    updates_per_epoch = _selected_updates_per_epoch(preflight)
    plan_rows = []
    for epoch in EXPECTED_EPOCHS:
        checkpoint = authority._validate_checkpoint(
            candidates[str(epoch)], f"Base e{epoch} checkpoint"
        )
        plan_rows.append(
            {
                "epoch": epoch,
                "optimizer_updates": epoch
                * updates_per_epoch,
                "candidate_checkpoint": checkpoint,
                "shards": [
                    {"shard_id": shard_id, "device": f"cuda:{shard_id}"}
                    for shard_id in range(EXPECTED_SHARDS)
                ],
            }
        )
    result = {
        "format": PLAN_FORMAT,
        "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
        "status": "frozen",
        "generator": "SemTalk Base Motion Generation",
        "generator_module": GENERATOR_MODULE,
        "dataset": "SHOW",
        "target_speaker_scope": authority.EXPECTED_SCOPE,
        "split": "val",
        "test_visible": False,
        "candidate_epochs": list(EXPECTED_EPOCHS),
        "updates_per_epoch": updates_per_epoch,
        "clip_count": EXPECTED_CLIPS,
        "num_shards": EXPECTED_SHARDS,
        "assignment": authority.FRESH_VAL_ASSIGNMENT,
        "preflight_receipt": normalized_preflight,
        "prerequisite_selection": normalized_prerequisite,
        "continuation_decision": normalized_continuation,
        "candidates": plan_rows,
    }
    return _with_payload_sha(result)


def _probe_binding(value: Any) -> dict[str, Any]:
    binding = _exact(
        value,
        {
            "source",
            "pipeline",
            "prerequisite_selection",
            "val_inputs",
            "candidate_checkpoints",
            "subset_manifest",
            "seed",
            "clips_per_candidate",
            "shards_per_candidate",
        },
        "fresh Base throughput probe binding",
    )
    source = _exact(
        binding["source"],
        {
            "origin",
            "commit",
            "tree",
            "clean",
            "detached",
            "local_branches_at_commit",
        },
        "fresh Base throughput probe source",
    )
    if (
        source["origin"] != authority.EXPECTED_ORIGIN
        or re.fullmatch(r"[0-9a-f]{40}", str(source["commit"])) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(source["tree"])) is None
        or source["clean"] is not True
        or source["detached"] is not True
        or source["local_branches_at_commit"] != []
        or binding["seed"] != 20260731
        or binding["clips_per_candidate"] != PROBE_CLIPS_PER_CANDIDATE
        or binding["shards_per_candidate"] != EXPECTED_SHARDS
    ):
        raise BaseFreshValOrchestratorError(
            "fresh Base throughput probe identity changed"
        )
    normalized_payloads = {}
    for role in ("pipeline", "prerequisite_selection", "val_inputs"):
        normalized, _payload = authority._verify_compact_receipt(
            binding[role], f"fresh Base throughput probe {role}"
        )
        normalized_payloads[role] = normalized
    checkpoint_rows = binding["candidate_checkpoints"]
    if (
        not isinstance(checkpoint_rows, list)
        or len(checkpoint_rows) != len(PROBE_EPOCHS)
    ):
        raise BaseFreshValOrchestratorError(
            "throughput probe candidate coverage changed"
        )
    normalized_checkpoints = []
    for expected_epoch, raw in zip(PROBE_EPOCHS, checkpoint_rows):
        row = _exact(
            raw,
            {"epoch", "candidate_checkpoint"},
            f"throughput probe Base e{expected_epoch}",
        )
        checkpoint = authority._validate_checkpoint(
            row["candidate_checkpoint"],
            f"throughput probe Base e{expected_epoch} checkpoint",
        )
        if row["epoch"] != expected_epoch:
            raise BaseFreshValOrchestratorError(
                "throughput probe candidate order changed"
            )
        normalized_checkpoints.append(
            {"epoch": expected_epoch, "candidate_checkpoint": checkpoint}
        )
    subset, subset_payload = authority._normalize_artifact(
        binding["subset_manifest"],
        "fresh Base throughput probe subset",
        with_payload=False,
    )
    subset_rows = authority._strict_jsonl_bytes(
        subset_payload, "fresh Base throughput probe subset"
    )
    subset_ids = [row.get("canonical_clip_id") for row in subset_rows]
    if (
        len(subset_rows) != PROBE_CLIPS_PER_CANDIDATE
        or len(set(subset_ids)) != PROBE_CLIPS_PER_CANDIDATE
        or any(
            not isinstance(clip_id, str) or not clip_id
            for clip_id in subset_ids
        )
        or any(row.get("split") != "val" for row in subset_rows)
    ):
        raise BaseFreshValOrchestratorError(
            "fresh Base throughput probe subset changed"
        )
    return {
        **binding,
        "source": source,
        **normalized_payloads,
        "candidate_checkpoints": normalized_checkpoints,
        "subset_manifest": subset,
    }


def _probe_metric(
    value: Any,
    *,
    epoch: int,
    checkpoint: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    subset_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, metric = authority._verify_compact_receipt(
        value, f"throughput probe Base e{epoch} metric"
    )
    _exact(
        metric,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "epoch",
            "candidate_checkpoint",
            "subset_manifest",
            "prediction_manifest",
            "metric_values",
            "receipt_payload_sha256",
        },
        f"throughput probe Base e{epoch} metric",
    )
    values = metric["metric_values"]
    if (
        metric["format"] != MULTICANDIDATE_PROBE_METRIC_FORMAT
        or metric["status"] != "complete"
        or metric["split"] != "val"
        or metric["test_visible"] is not False
        or metric["epoch"] != epoch
        or metric["candidate_checkpoint"] != checkpoint
        or metric["subset_manifest"] != subset_manifest
        or metric["prediction_manifest"] != prediction_manifest
        or not isinstance(values, dict)
        or not values
        or any(
            not isinstance(key, str)
            or not key
            or type(number) not in (int, float)
            or not math.isfinite(float(number))
            for key, number in values.items()
        )
    ):
        raise BaseFreshValOrchestratorError(
            f"throughput probe Base e{epoch} metric changed"
        )
    return artifact, metric


def _probe_candidate_output(
    value: Any,
    *,
    expected_epoch: int,
    expected_checkpoint: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    row = _exact(
        value,
        {
            "epoch",
            "candidate_checkpoint",
            "prediction_manifest",
            "metric_receipt",
        },
        f"throughput probe Base e{expected_epoch} output",
    )
    if (
        row["epoch"] != expected_epoch
        or row["candidate_checkpoint"] != expected_checkpoint
    ):
        raise BaseFreshValOrchestratorError(
            "throughput probe candidate output binding changed"
        )
    manifest, manifest_payload = authority._normalize_artifact(
        row["prediction_manifest"],
        f"throughput probe Base e{expected_epoch} prediction manifest",
        with_payload=False,
    )
    rows = authority._strict_jsonl_bytes(
        manifest_payload,
        f"throughput probe Base e{expected_epoch} prediction manifest",
    )
    expected_subset_path = binding["subset_manifest"]["path"]
    subset_path, subset_payload = _safe_snapshot(
        Path(expected_subset_path), "throughput probe subset"
    )
    del subset_path
    subset_rows = authority._strict_jsonl_bytes(
        subset_payload, "throughput probe subset"
    )
    expected_ids = [row["canonical_clip_id"] for row in subset_rows]
    if len(rows) != len(expected_ids):
        raise BaseFreshValOrchestratorError(
            "throughput probe prediction coverage changed"
        )
    prediction_fingerprint = []
    output_paths: set[str] = {manifest["path"]}
    for position, (prediction_row, expected_id) in enumerate(
        zip(rows, expected_ids)
    ):
        prediction = prediction_row.get("prediction")
        normalized_prediction, _payload = authority._normalize_artifact(
            prediction,
            f"throughput probe Base e{expected_epoch} prediction {position}",
            with_payload=False,
        )
        if (
            prediction_row.get("canonical_clip_id") != expected_id
            or normalized_prediction["path"] in output_paths
        ):
            raise BaseFreshValOrchestratorError(
                "throughput probe prediction order/path changed"
            )
        output_paths.add(normalized_prediction["path"])
        prediction_fingerprint.append(
            {
                "canonical_clip_id": expected_id,
                "sha256": normalized_prediction["sha256"],
                "bytes": normalized_prediction["bytes"],
            }
        )
    metric_artifact, metric = _probe_metric(
        row["metric_receipt"],
        epoch=expected_epoch,
        checkpoint=expected_checkpoint,
        prediction_manifest=manifest,
        subset_manifest=binding["subset_manifest"],
    )
    if metric_artifact["path"] in output_paths:
        raise BaseFreshValOrchestratorError(
            "throughput probe metric reused an output path"
        )
    output_paths.add(metric_artifact["path"])
    return (
        {
            "epoch": expected_epoch,
            "candidate_checkpoint": dict(expected_checkpoint),
            "prediction_fingerprint_sha256": authority.canonical_json_sha256(
                prediction_fingerprint
            ),
            "metric_values_sha256": authority.canonical_json_sha256(
                metric["metric_values"]
            ),
        },
        output_paths,
    )


def _replay_probe_run(
    artifact_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact, run = authority._verify_compact_receipt(
        artifact_value, "fresh Base multi-candidate probe run"
    )
    _exact(
        run,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "formal_host",
            "candidates_per_wave",
            "execution_mode",
            "probe_binding",
            "candidate_outputs",
            "execution_trace",
            "process_evidence",
            "receipt_payload_sha256",
        },
        "fresh Base multi-candidate probe run",
    )
    binding = _probe_binding(run["probe_binding"])
    trace = _exact(
        run["execution_trace"],
        {
            "started_monotonic_ns",
            "finished_monotonic_ns",
            "gpu_peak_memory_bytes",
            "gpu_total_memory_bytes",
        },
        "fresh Base throughput probe execution trace",
    )
    process = _exact(
        run["process_evidence"],
        {"runner_rc", "oom", "descendants_exited", "guards_restored"},
        "fresh Base throughput probe process evidence",
    )
    started = trace["started_monotonic_ns"]
    finished = trace["finished_monotonic_ns"]
    peaks = trace["gpu_peak_memory_bytes"]
    totals = trace["gpu_total_memory_bytes"]
    if (
        run["format"] != MULTICANDIDATE_PROBE_RUN_FORMAT
        or run["status"] not in {"complete", "failed"}
        or run["split"] != "val"
        or run["test_visible"] is not False
        or run["formal_host"] not in FORMAL_HOST_BY_PARTITION.values()
        or run["candidates_per_wave"] not in CONCURRENCY_SELECTION_ORDER
        or run["execution_mode"] not in {"serial", "concurrent"}
        or type(started) is not int
        or type(finished) is not int
        or started < 0
        or finished <= started
        or not isinstance(peaks, list)
        or not isinstance(totals, list)
        or len(peaks) != EXPECTED_SHARDS
        or len(totals) != EXPECTED_SHARDS
        or any(type(value) is not int or value < 0 for value in peaks)
        or any(type(value) is not int or value <= 0 for value in totals)
        or any(peak > total for peak, total in zip(peaks, totals))
        or type(process["runner_rc"]) is not int
        or type(process["oom"]) is not bool
        or type(process["descendants_exited"]) is not bool
        or type(process["guards_restored"]) is not bool
    ):
        raise BaseFreshValOrchestratorError(
            "fresh Base throughput probe run identity/evidence changed"
        )
    process_success = (
        process["runner_rc"] == 0
        and process["oom"] is False
        and process["descendants_exited"] is True
        and process["guards_restored"] is True
    )
    outputs = run["candidate_outputs"]
    fingerprints: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    if run["status"] == "complete":
        if not process_success or not isinstance(outputs, list) or len(outputs) != len(PROBE_EPOCHS):
            raise BaseFreshValOrchestratorError(
                "complete throughput probe lacks successful exact outputs"
            )
        checkpoint_by_epoch = {
            row["epoch"]: row["candidate_checkpoint"]
            for row in binding["candidate_checkpoints"]
        }
        for expected_epoch, value in zip(PROBE_EPOCHS, outputs):
            fingerprint, paths = _probe_candidate_output(
                value,
                expected_epoch=expected_epoch,
                expected_checkpoint=checkpoint_by_epoch[expected_epoch],
                binding=binding,
            )
            if output_paths & paths:
                raise BaseFreshValOrchestratorError(
                    "throughput probe candidates reused output paths"
                )
            output_paths.update(paths)
            fingerprints.append(fingerprint)
    elif outputs != [] or process_success:
        raise BaseFreshValOrchestratorError(
            "failed throughput probe must expose no publishable outputs"
        )
    memory_within_limit = all(
        peak <= int(total * PROBE_GPU_MEMORY_LIMIT_FRACTION)
        for peak, total in zip(peaks, totals)
    )
    summary = {
        "binding": binding,
        "fingerprints": fingerprints,
        "output_paths": output_paths,
        "elapsed_seconds": (finished - started) / 1_000_000_000.0,
        "gpu_peak_memory_bytes": list(peaks),
        "gpu_total_memory_bytes": list(totals),
        "memory_within_limit": memory_within_limit,
        "process_success": process_success,
    }
    return artifact, run, summary


def build_multicandidate_comparison(
    *,
    serial_run: Mapping[str, Any],
    concurrent_run: Mapping[str, Any],
) -> dict[str, Any]:
    serial_artifact, serial, serial_summary = _replay_probe_run(serial_run)
    concurrent_artifact, concurrent, concurrent_summary = _replay_probe_run(
        concurrent_run
    )
    if (
        serial_artifact["path"] == concurrent_artifact["path"]
        or serial["execution_mode"] != "serial"
        or serial["candidates_per_wave"] != 1
        or serial["status"] != "complete"
        or concurrent["execution_mode"] != "concurrent"
        or concurrent["candidates_per_wave"]
        not in CONCURRENCY_SELECTION_ORDER
        or serial["formal_host"] != concurrent["formal_host"]
        or serial_summary["binding"] != concurrent_summary["binding"]
        or serial_summary["output_paths"] & concurrent_summary["output_paths"]
    ):
        raise BaseFreshValOrchestratorError(
            "serial/concurrent throughput probe pairing changed"
        )
    prediction_equal = (
        concurrent["status"] == "complete"
        and [
            row["prediction_fingerprint_sha256"]
            for row in serial_summary["fingerprints"]
        ]
        == [
            row["prediction_fingerprint_sha256"]
            for row in concurrent_summary["fingerprints"]
        ]
    )
    metric_equal = (
        concurrent["status"] == "complete"
        and [
            row["metric_values_sha256"]
            for row in serial_summary["fingerprints"]
        ]
        == [
            row["metric_values_sha256"]
            for row in concurrent_summary["fingerprints"]
        ]
    )
    semantics_equal = (
        prediction_equal
        and metric_equal
        and serial_summary["binding"] == concurrent_summary["binding"]
    )
    comparison_pass = (
        semantics_equal
        and concurrent_summary["process_success"]
        and concurrent_summary["memory_within_limit"]
    )
    failures = []
    if not concurrent_summary["process_success"]:
        failures.append("concurrent_process_evidence_failed")
    if not concurrent_summary["memory_within_limit"]:
        failures.append("concurrent_gpu_memory_headroom_failed")
    if not prediction_equal:
        failures.append("prediction_sha_or_bytes_mismatch")
    if not metric_equal:
        failures.append("metric_values_mismatch")
    serial_elapsed = serial_summary["elapsed_seconds"]
    concurrent_elapsed = concurrent_summary["elapsed_seconds"]
    return _with_payload_sha(
        {
            "format": MULTICANDIDATE_COMPARISON_FORMAT,
            "status": "pass" if comparison_pass else "fail",
            "split": "val",
            "test_visible": False,
            "formal_host": serial["formal_host"],
            "candidates_per_wave": concurrent["candidates_per_wave"],
            "probe_binding_sha256": authority.canonical_json_sha256(
                serial_summary["binding"]
            ),
            "serial_run": serial_artifact,
            "concurrent_run": concurrent_artifact,
            "equivalence": {
                "excluded_fields": list(CONCURRENCY_EXCLUDED_FIELDS),
                "prediction_sha_bytes_equal": prediction_equal,
                "metric_receipts_equal": metric_equal,
                "deterministic_semantics_equal": semantics_equal,
            },
            "throughput": {
                "serial_elapsed_seconds": serial_elapsed,
                "concurrent_elapsed_seconds": concurrent_elapsed,
                "concurrent_over_serial_ratio": (
                    concurrent_elapsed / serial_elapsed
                ),
                "concurrent_peak_gpu_memory_bytes": concurrent_summary[
                    "gpu_peak_memory_bytes"
                ],
                "concurrent_total_gpu_memory_bytes": concurrent_summary[
                    "gpu_total_memory_bytes"
                ],
                "memory_limit_fraction": PROBE_GPU_MEMORY_LIMIT_FRACTION,
                "memory_within_limit": concurrent_summary[
                    "memory_within_limit"
                ],
            },
            "failure_reasons": failures,
        }
    )


def _replay_multicandidate_comparison(
    artifact_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, comparison = authority._verify_compact_receipt(
        artifact_value, "fresh Base multi-candidate comparison"
    )
    _exact(
        comparison,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "formal_host",
            "candidates_per_wave",
            "probe_binding_sha256",
            "serial_run",
            "concurrent_run",
            "equivalence",
            "throughput",
            "failure_reasons",
            "receipt_payload_sha256",
        },
        "fresh Base multi-candidate comparison",
    )
    rebuilt = build_multicandidate_comparison(
        serial_run=comparison["serial_run"],
        concurrent_run=comparison["concurrent_run"],
    )
    if rebuilt != comparison:
        raise BaseFreshValOrchestratorError(
            "multi-candidate comparison differs from real run replay"
        )
    return artifact, comparison


def build_multicandidate_gate(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_count = len(CONCURRENCY_SELECTION_ORDER) * len(
        FORMAL_HOST_BY_PARTITION
    )
    if len(comparisons) != expected_count:
        raise BaseFreshValOrchestratorError(
            "multi-candidate gate requires every mode on both hosts"
        )
    by_key: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    common_binding: dict[str, Any] | None = None
    artifact_paths: set[str] = set()
    for value in comparisons:
        artifact, comparison = _replay_multicandidate_comparison(value)
        key = (
            comparison["candidates_per_wave"],
            comparison["formal_host"],
        )
        _serial_artifact, _serial, serial_summary = _replay_probe_run(
            comparison["serial_run"]
        )
        if (
            key in by_key
            or artifact["path"] in artifact_paths
            or key[0] not in CONCURRENCY_SELECTION_ORDER
            or key[1] not in FORMAL_HOST_BY_PARTITION.values()
        ):
            raise BaseFreshValOrchestratorError(
                "multi-candidate comparison coverage/path changed"
            )
        artifact_paths.add(artifact["path"])
        if common_binding is None:
            common_binding = serial_summary["binding"]
        elif common_binding != serial_summary["binding"]:
            raise BaseFreshValOrchestratorError(
                "multi-candidate comparisons use different probe bindings"
            )
        by_key[key] = (artifact, comparison)
    expected_keys = {
        (mode, host)
        for mode in CONCURRENCY_SELECTION_ORDER
        for host in FORMAL_HOST_BY_PARTITION.values()
    }
    if set(by_key) != expected_keys or common_binding is None:
        raise BaseFreshValOrchestratorError(
            "multi-candidate comparison matrix is incomplete"
        )
    attempts = []
    safe_scores: dict[int, float] = {}
    for mode in CONCURRENCY_SELECTION_ORDER:
        host_artifacts = []
        host_payloads = []
        for host in FORMAL_HOST_BY_PARTITION.values():
            artifact, comparison = by_key[(mode, host)]
            host_artifacts.append(artifact)
            host_payloads.append(comparison)
        passed = all(row["status"] == "pass" for row in host_payloads)
        bottleneck = max(
            float(row["throughput"]["concurrent_elapsed_seconds"])
            for row in host_payloads
        )
        if passed:
            safe_scores[mode] = bottleneck
        attempts.append(
            {
                "candidates_per_wave": mode,
                "status": "pass" if passed else "fail",
                "host_comparisons": host_artifacts,
                "bottleneck_elapsed_seconds": bottleneck,
            }
        )
    if not safe_scores:
        raise BaseFreshValOrchestratorError(
            "no safe measured multi-candidate execution mode"
        )
    fastest_elapsed = min(safe_scores.values())
    selected = min(
        mode
        for mode, elapsed in safe_scores.items()
        if elapsed
        <= fastest_elapsed * (1.0 + CONCURRENCY_SELECTION_SAFETY_MARGIN)
    )
    return _with_payload_sha(
        {
            "format": MULTICANDIDATE_GATE_FORMAT,
            "status": "pass",
            "split": "val",
            "test_visible": False,
            "formal_hosts": list(FORMAL_HOST_BY_PARTITION.values()),
            "probe_binding": common_binding,
            "protocol": {
                "evaluated_candidates_per_wave": list(
                    CONCURRENCY_SELECTION_ORDER
                ),
                "probe_candidates_per_host": len(PROBE_EPOCHS),
                "probe_clips_per_candidate": PROBE_CLIPS_PER_CANDIDATE,
                "shards_per_candidate": EXPECTED_SHARDS,
                "numeric_equivalence": (
                    "prediction_sha_bytes_and_metric_values_exact"
                ),
                "excluded_fields": list(CONCURRENCY_EXCLUDED_FIELDS),
                "selection_rule": (
                    "lowest_concurrency_within_2_percent_of_minimum_"
                    "two_host_bottleneck_elapsed_v1"
                ),
                "selection_safety_margin_fraction": (
                    CONCURRENCY_SELECTION_SAFETY_MARGIN
                ),
                "gpu_memory_limit_fraction": (
                    PROBE_GPU_MEMORY_LIMIT_FRACTION
                ),
            },
            "attempts": attempts,
            "selected_candidates_per_wave": selected,
            "selected_bottleneck_elapsed_seconds": safe_scores[selected],
        }
    )


def _validate_multicandidate_gate(
    artifact_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, gate = authority._verify_compact_receipt(
        artifact_value, "fresh Base multi-candidate throughput gate"
    )
    _exact(
        gate,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "formal_hosts",
            "probe_binding",
            "protocol",
            "attempts",
            "selected_candidates_per_wave",
            "selected_bottleneck_elapsed_seconds",
            "receipt_payload_sha256",
        },
        "fresh Base multi-candidate throughput gate",
    )
    if (
        gate["format"] != MULTICANDIDATE_GATE_FORMAT
        or gate["status"] != "pass"
        or gate["split"] != "val"
        or gate["test_visible"] is not False
    ):
        raise BaseFreshValOrchestratorError(
            "fresh Base multi-candidate gate identity changed"
        )
    attempts = gate["attempts"]
    if (
        not isinstance(attempts, list)
        or len(attempts) != len(CONCURRENCY_SELECTION_ORDER)
    ):
        raise BaseFreshValOrchestratorError(
            "fresh Base multi-candidate gate attempt coverage changed"
        )
    comparison_artifacts = []
    for expected_mode, attempt in zip(CONCURRENCY_SELECTION_ORDER, attempts):
        attempt = _exact(
            attempt,
            {
                "candidates_per_wave",
                "status",
                "host_comparisons",
                "bottleneck_elapsed_seconds",
            },
            f"fresh Base C={expected_mode} gate attempt",
        )
        if (
            attempt["candidates_per_wave"] != expected_mode
            or not isinstance(attempt["host_comparisons"], list)
            or len(attempt["host_comparisons"])
            != len(FORMAL_HOST_BY_PARTITION)
        ):
            raise BaseFreshValOrchestratorError(
                "fresh Base multi-candidate gate attempt changed"
            )
        comparison_artifacts.extend(attempt["host_comparisons"])
    rebuilt = build_multicandidate_gate(comparison_artifacts)
    if rebuilt != gate:
        raise BaseFreshValOrchestratorError(
            "multi-candidate gate differs from serial/concurrent replay"
        )
    return artifact, gate


def validate_run_spec(
    spec_artifact: Mapping[str, Any],
    *,
    expected_source_commit: str,
    expected_source_tree: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the sole frozen input surface consumed by the shell runner."""

    if (
        re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", expected_source_tree) is None
    ):
        raise BaseFreshValOrchestratorError(
            "source commit/tree must be lowercase Git object IDs"
        )
    normalized, spec = authority._verify_compact_receipt(
        spec_artifact, "fresh Base validation run specification"
    )
    _exact(
        spec,
        {
            "format",
            "payload_hash_algorithm",
            "status",
            "generator",
            "generator_module",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "formal_hosts",
            "multi_candidate_gate",
            "source",
            "candidate_bundle",
            "val_inputs",
            "pipeline",
            "prerequisite_selection",
            "continuation_decision",
            "continuation_waves",
            "canonical_manifest",
            "validation_gates",
            "metric_assets",
            "real_feature_cache",
            "seed",
            "receipt_payload_sha256",
        },
        "fresh Base validation run specification",
    )
    source = _exact(
        spec["source"],
        {
            "origin",
            "commit",
            "tree",
            "clean",
            "detached",
            "local_branches_at_commit",
        },
        "fresh Base validation source",
    )
    if (
        spec["format"] != RUN_SPEC_FORMAT
        or spec["payload_hash_algorithm"] != PAYLOAD_HASH_ALGORITHM
        or spec["status"] != "frozen"
        or spec["generator"] != "SemTalk Base Motion Generation"
        or spec["generator_module"] != GENERATOR_MODULE
        or spec["dataset"] != "SHOW"
        or spec["target_speaker_scope"] != authority.EXPECTED_SCOPE
        or spec["split"] != "val"
        or spec["test_visible"] is not False
        or spec["formal_hosts"]
        != {
            "partition_count": 2,
            "partition_id_to_hostname": {
                "0": FORMAL_HOST_BY_PARTITION[0],
                "1": FORMAL_HOST_BY_PARTITION[1],
            },
        }
        or source
        != {
            "origin": authority.EXPECTED_ORIGIN,
            "commit": expected_source_commit,
            "tree": expected_source_tree,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        or spec["seed"] != 20260731
    ):
        raise BaseFreshValOrchestratorError(
            "fresh Base validation run identity changed"
        )
    bundle = _exact(
        spec["candidate_bundle"],
        {"manifest", "status", "frozen_inputs"},
        "candidate bundle",
    )
    normalized_bundle: dict[str, dict[str, Any]] = {}
    for role, artifact_value in bundle.items():
        artifact, _ = authority._normalize_artifact(
            artifact_value,
            f"candidate bundle {role}",
            with_payload=False,
        )
        normalized_bundle[role] = artifact
    normalized_val, _ = authority._verify_compact_receipt(
        spec["val_inputs"], "frozen validation inputs"
    )
    normalized_pipeline, _ = authority._verify_compact_receipt(
        spec["pipeline"], "frozen validation pipeline"
    )
    prerequisite, prerequisite_payload, fixed_checkpoints = (
        authority._validate_prerequisite_selection(
            spec["prerequisite_selection"]
        )
    )
    try:
        audited_val, val_coverage = val_contract.validate_val_inputs(
            Path(normalized_val["path"]), normalized_val["sha256"]
        )
        audited_pipeline, pipeline_payload = val_contract.validate_pipeline(
            Path(normalized_pipeline["path"]),
            normalized_pipeline["sha256"],
            expected_prerequisite_selection=prerequisite,
            expected_source=source,
        )
        expected_selected = {
            stage: pipeline_payload["fixed_checkpoints"][stage]["sha256"]
            for stage in authority.STAGES
        }
        audited_bundle = val_contract.validate_candidate_bundle(
            manifest_path=Path(normalized_bundle["manifest"]["path"]),
            expected_manifest_sha256=normalized_bundle["manifest"]["sha256"],
            status_path=Path(normalized_bundle["status"]["path"]),
            expected_status_sha256=normalized_bundle["status"]["sha256"],
            frozen_inputs_path=Path(
                normalized_bundle["frozen_inputs"]["path"]
            ),
            expected_frozen_inputs_sha256=normalized_bundle[
                "frozen_inputs"
            ]["sha256"],
            expected_selected_prerequisite_sha256=expected_selected,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise BaseFreshValOrchestratorError(
            "audited frozen validation authority replay failed"
        ) from error
    for expected, observed, role in (
        (normalized_val, audited_val, "val inputs"),
        (normalized_pipeline, audited_pipeline, "pipeline"),
    ):
        if any(
            expected.get(key) != observed.get(key)
            for key in ("path", "sha256", "receipt_payload_sha256")
        ):
            raise BaseFreshValOrchestratorError(
                f"audited frozen {role} binding changed"
            )
    for role in ("manifest", "status"):
        if audited_bundle[role] != normalized_bundle[role]:
            raise BaseFreshValOrchestratorError(
                f"audited candidate bundle {role} changed"
            )
    if (
        any(
            audited_bundle["frozen_inputs"].get(key)
            != normalized_bundle["frozen_inputs"].get(key)
            for key in ("path", "sha256")
        )
        or audited_bundle.get("producer_source") != source
        or audited_bundle.get("selected_prerequisite_sha256")
        != expected_selected
        or {
            stage: {
                key: pipeline_payload["fixed_checkpoints"][stage][key]
                for key in ("path", "sha256", "bytes")
            }
            for stage in authority.STAGES
        }
        != fixed_checkpoints
    ):
        raise BaseFreshValOrchestratorError(
            "Base candidate/fresh pipeline/selected-five binding changed"
        )
    normalized_concurrency_gate, concurrency_gate = (
        _validate_multicandidate_gate(spec["multi_candidate_gate"])
    )
    probe_binding = concurrency_gate["probe_binding"]
    expected_probe_checkpoints = [
        {
            "epoch": epoch,
            "candidate_checkpoint": audited_bundle["candidates"][epoch],
        }
        for epoch in PROBE_EPOCHS
    ]
    if (
        probe_binding["source"] != source
        or probe_binding["pipeline"] != normalized_pipeline
        or probe_binding["prerequisite_selection"] != prerequisite
        or probe_binding["val_inputs"] != normalized_val
        or probe_binding["candidate_checkpoints"]
        != expected_probe_checkpoints
    ):
        raise BaseFreshValOrchestratorError(
            "multi-candidate gate differs from frozen run inputs"
        )
    continuation, _decision = authority._validate_continuation_decision(
        spec["continuation_decision"],
        prerequisite_artifact=prerequisite,
        prerequisite_selection=prerequisite_payload,
    )
    continuation_waves = authority._validate_continuation_waves(
        spec["continuation_waves"],
        prerequisite_selection=prerequisite_payload,
    )
    canonical_manifest, canonical_payload = authority._normalize_artifact(
        spec["canonical_manifest"],
        "canonical SHOW validation manifest",
        with_payload=False,
    )
    canonical_rows = authority._strict_jsonl_bytes(
        canonical_payload, "canonical SHOW validation manifest"
    )
    subset_path, subset_payload = _safe_snapshot(
        Path(probe_binding["subset_manifest"]["path"]),
        "fresh Base throughput probe subset",
    )
    del subset_path
    subset_rows = authority._strict_jsonl_bytes(
        subset_payload, "fresh Base throughput probe subset"
    )
    if (
        len(canonical_rows) != EXPECTED_CLIPS
        or subset_rows != canonical_rows[:PROBE_CLIPS_PER_CANDIDATE]
    ):
        raise BaseFreshValOrchestratorError(
            "multi-candidate probe subset is not the frozen val prefix"
        )
    cache, cache_payload = authority._verify_compact_receipt(
        spec["real_feature_cache"], "released2 real-feature cache"
    )
    canonical_from_val = val_coverage.get("canonical_manifest")
    if (
        not isinstance(canonical_from_val, dict)
        or any(
            canonical_manifest.get(key) != canonical_from_val.get(key)
            for key in ("path", "sha256")
        )
    ):
        raise BaseFreshValOrchestratorError(
            "metric canonical manifest differs from audited val inputs"
        )
    expected_cache_canonical = {
        "path": canonical_manifest["path"],
        "sha256": canonical_manifest["sha256"],
        "bytes": canonical_manifest["bytes"],
        "rows": EXPECTED_CLIPS,
        "selected_rows": EXPECTED_CLIPS,
    }
    if (
        cache_payload.get("format")
        != "semtalk_show_released2_real_feature_cache_v1"
        or cache_payload.get("status") != "complete"
        or cache_payload.get("split") != "val"
        or cache_payload.get("clip_count") != EXPECTED_CLIPS
        or cache_payload.get("canonical_manifest")
        != expected_cache_canonical
        or cache_payload.get("formal_mode") is not True
        or cache_payload.get("test_only_mode") is not False
    ):
        raise BaseFreshValOrchestratorError(
            "released2 real-feature cache differs from audited val canonical"
        )
    gates = spec["validation_gates"]
    if not isinstance(gates, list) or len(gates) != len(EXPECTED_EPOCHS):
        raise BaseFreshValOrchestratorError(
            "run specification requires 22 candidate-bound gates"
        )
    normalized_gates: list[dict[str, Any]] = []
    gate_paths: set[str] = set()
    for expected_epoch, raw in zip(EXPECTED_EPOCHS, gates):
        row = _exact(
            raw,
            {"epoch", "validation_gate"},
            f"Base e{expected_epoch} gate row",
        )
        gate_artifact, gate = authority._verify_compact_receipt(
            row["validation_gate"], f"Base e{expected_epoch} gate"
        )
        if (
            row["epoch"] != expected_epoch
            or gate.get("format")
            != "semtalk_show_deterministic_replication_gate_v2"
            or gate.get("status") != "pass"
            or gate.get("split") != "val"
            or gate.get("test_visible") is not False
            or gate.get("scope") != "validation_candidate_family"
            or gate.get("coverage_mode") != "full_frozen_val_1715"
            or gate.get("proof", {}).get("clip_count") != EXPECTED_CLIPS
            or gate_artifact["path"] in gate_paths
        ):
            raise BaseFreshValOrchestratorError(
                f"Base e{expected_epoch} gate changed or was reused"
            )
        gate_paths.add(gate_artifact["path"])
        normalized_gates.append(
            {"epoch": expected_epoch, "validation_gate": gate_artifact}
        )
    metric_assets = _exact(
        spec["metric_assets"],
        {"talkshow_metric_root", "feature_extractor", "smplx_asset"},
        "TalkSHOW metric assets",
    )
    metric_root = Path(metric_assets["talkshow_metric_root"])
    if (
        not metric_root.is_absolute()
        or metric_root.is_symlink()
        or not metric_root.is_dir()
        or metric_root.resolve(strict=True) != metric_root
    ):
        raise BaseFreshValOrchestratorError(
            "TalkSHOW metric root must be a canonical non-symlink directory"
        )
    normalized_metric_assets = {
        "talkshow_metric_root": str(metric_root),
    }
    for role in ("feature_extractor", "smplx_asset"):
        artifact, _ = authority._normalize_artifact(
            metric_assets[role], f"TalkSHOW {role}", with_payload=False
        )
        normalized_metric_assets[role] = artifact
    canonical_spec = {
        **spec,
        "candidate_bundle": normalized_bundle,
        "val_inputs": normalized_val,
        "pipeline": normalized_pipeline,
        "multi_candidate_gate": normalized_concurrency_gate,
        "prerequisite_selection": prerequisite,
        "continuation_decision": continuation,
        "continuation_waves": continuation_waves,
        "canonical_manifest": canonical_manifest,
        "validation_gates": normalized_gates,
        "metric_assets": normalized_metric_assets,
        "real_feature_cache": cache,
    }
    if canonical_spec != spec:
        raise BaseFreshValOrchestratorError(
            "run specification is not canonical"
        )
    return normalized, spec


def _prediction_records(
    manifest_artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    normalized, payload = authority._normalize_artifact(
        manifest_artifact, "prediction manifest", with_payload=False
    )
    del normalized
    rows = authority._strict_jsonl_bytes(payload, "prediction manifest")
    if len(rows) != EXPECTED_CLIPS:
        raise BaseFreshValOrchestratorError(
            "prediction manifest must contain exactly 1715 rows"
        )
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prediction = row.get("prediction")
        if not isinstance(prediction, dict):
            raise BaseFreshValOrchestratorError(
                f"prediction row {index} lacks an artifact"
            )
        records.append(
            {
                "canonical_clip_id": row.get("canonical_clip_id"),
                "prediction_sha256": prediction.get("sha256"),
                "prediction_bytes": prediction.get("bytes"),
            }
        )
    return records


def build_distribution(
    *,
    lineage_artifact: Mapping[str, Any],
    gate_path: Path,
    expected_gate_sha256: str,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    # Kept lazy so every planning/sealing contract remains standard-library
    # only; NumPy is needed solely when the real gate replays NPZ arrays.
    from scripts.show_base import deterministic_replication_gate as replication

    _lineage_artifact, lineage = authority._verify_compact_receipt(
        lineage_artifact, "legacy validation lineage"
    )
    if (
        lineage.get("format") != authority.FORMAL_VAL_LINEAGE_FORMAT
        or lineage.get("split") != "val"
        or lineage.get("test_visible") is not False
        or lineage.get("clip_count") != EXPECTED_CLIPS
    ):
        raise BaseFreshValOrchestratorError(
            "distribution input is not complete validation lineage"
        )
    manifest_path = Path(lineage["final_manifest"]["path"])
    manifest_artifact, _ = _artifact(
        manifest_path,
        lineage["final_manifest"]["sha256"],
        payload_receipt=False,
    )
    gate_artifact, gate = replication.load_gate(
        gate_path,
        expected_gate_sha256,
        expected_scope="validation_candidate_family",
    )
    if gate["model_bundle"]["checkpoints"]["base"] != checkpoint:
        raise BaseFreshValOrchestratorError(
            "deterministic gate is not bound to this Base checkpoint"
        )
    return replication.build_distribution_receipt_from_validated_artifacts(
        gate_artifact=gate_artifact,
        prediction_manifest_artifact=manifest_artifact,
        prediction_records=_prediction_records(manifest_artifact),
    )


def build_failure_manifest(epoch: int) -> dict[str, Any]:
    if epoch not in EXPECTED_EPOCHS:
        raise BaseFreshValOrchestratorError("epoch is outside frozen schedule")
    return _with_payload_sha(
        {
            "format": authority.FRESH_VAL_FAILURE_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "epoch": epoch,
            "failure_count": 0,
            "failures": [],
        }
    )


def build_candidate_transaction(
    *,
    epoch: int,
    preflight_artifact: Mapping[str, Any],
    output_root: Path,
    lineage_artifact: Mapping[str, Any],
    distribution_artifact: Mapping[str, Any],
    failure_artifact: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
    formal_host: str,
) -> dict[str, Any]:
    if formal_host not in FORMAL_HOST_BY_PARTITION.values():
        raise BaseFreshValOrchestratorError(
            "candidate formal host is outside the exact dual-host allowlist"
        )
    normalized_preflight, preflight = _preflight_candidate(
        preflight_artifact, epoch
    )
    checkpoint = preflight["_candidate"]
    normalized_prerequisite, prerequisite, _fixed = (
        authority._validate_prerequisite_selection(prerequisite_artifact)
    )
    normalized_continuation, _decision = (
        authority._validate_continuation_decision(
            continuation_artifact,
            prerequisite_artifact=normalized_prerequisite,
            prerequisite_selection=prerequisite,
        )
    )
    normalized_lineage, lineage = authority._verify_compact_receipt(
        lineage_artifact, f"Base e{epoch} inference lineage"
    )
    manifest_artifact, _ = _artifact(
        Path(lineage["final_manifest"]["path"]),
        lineage["final_manifest"]["sha256"],
        payload_receipt=False,
    )
    normalized_distribution, distribution = authority._validate_distribution(
        distribution_artifact,
        prediction_manifest=manifest_artifact,
    )
    normalized_failure, failures = authority._verify_compact_receipt(
        failure_artifact, f"Base e{epoch} failure manifest"
    )
    if failures != build_failure_manifest(epoch):
        raise BaseFreshValOrchestratorError(
            f"Base e{epoch} failure manifest is not canonical/empty"
        )
    val_reference = preflight["val_inputs_receipt"]
    val_artifact, _ = _artifact(
        Path(val_reference["path"]),
        val_reference["sha256"],
        payload_receipt=True,
    )
    pipeline_reference = preflight["pipeline_receipt"]
    pipeline_artifact, _ = _artifact(
        Path(pipeline_reference["path"]),
        pipeline_reference["sha256"],
        payload_receipt=True,
    )
    shards: list[dict[str, Any]] = []
    common_model: str | None = None
    common_runtime: str | None = None
    total_clips = 0
    for shard_id in range(EXPECTED_SHARDS):
        shard_root = output_root / "shards" / (
            f"shard-{shard_id:05d}-of-{EXPECTED_SHARDS:05d}"
        )
        receipt_artifact, receipt = _artifact(
            shard_root / "shard_receipt.json", payload_receipt=True
        )
        manifest_artifact, manifest_payload = _artifact(
            shard_root / "shard_manifest.jsonl", payload_receipt=False
        )
        rows = authority._strict_jsonl_bytes(
            _safe_snapshot(
                Path(manifest_artifact["path"]),
                f"Base e{epoch} shard manifest",
            )[1],
            f"Base e{epoch} shard manifest",
        )
        if (
            receipt.get("shard_id") != shard_id
            or receipt.get("num_shards") != EXPECTED_SHARDS
            or receipt.get("epoch") != epoch
            or receipt.get("clip_count") != len(rows)
        ):
            raise BaseFreshValOrchestratorError(
                f"Base e{epoch} shard {shard_id} identity changed"
            )
        common_model = common_model or receipt["model_receipts_sha256"]
        common_runtime = common_runtime or receipt["runtime_contract_sha256"]
        if (
            receipt["model_receipts_sha256"] != common_model
            or receipt["runtime_contract_sha256"] != common_runtime
        ):
            raise BaseFreshValOrchestratorError(
                f"Base e{epoch} shard model/runtime mismatch"
            )
        total_clips += len(rows)
        shards.append(
            {
                "shard_id": shard_id,
                "receipt": receipt_artifact,
                "manifest": manifest_artifact,
            }
        )
    if total_clips != EXPECTED_CLIPS:
        raise BaseFreshValOrchestratorError(
            f"Base e{epoch} shards do not cover 1715 clips"
        )
    transaction = _with_payload_sha(
        {
            "format": authority.FRESH_VAL_TRANSACTION_FORMAT,
            "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "generator": "SemTalk Base Motion Generation",
            "generator_module": GENERATOR_MODULE,
            "dataset": "SHOW",
            "target_speaker_scope": authority.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "formal_host": formal_host,
            "epoch": epoch,
            "optimizer_updates": epoch
            * _selected_updates_per_epoch(preflight),
            "candidate_checkpoint": checkpoint,
            "prerequisite_selection": normalized_prerequisite,
            "continuation_decision": normalized_continuation,
            "preflight_receipt": normalized_preflight,
            "val_inputs_receipt": val_artifact,
            "pipeline_receipt": pipeline_artifact,
            "prediction_manifest": manifest_artifact,
            "inference_lineage": normalized_lineage,
            "distribution_receipt": normalized_distribution,
            "validation_gate": distribution["validation_gate"],
            "shards": shards,
            "failure_manifest": normalized_failure,
            "source_runtime_input_pins": {
                "source": preflight["candidate_bundle"]["producer_source"],
                "preflight_receipt_payload_sha256": normalized_preflight[
                    "receipt_payload_sha256"
                ],
                "val_inputs_receipt": val_artifact,
                "pipeline_receipt": pipeline_artifact,
                "model_receipts_sha256": common_model,
                "runtime_contract_sha256": common_runtime,
                "prediction_manifest_sha256": manifest_artifact["sha256"],
                "distribution_receipt_payload_sha256": (
                    normalized_distribution["receipt_payload_sha256"]
                ),
            },
            "coverage": {
                "clip_count": EXPECTED_CLIPS,
                "num_shards": EXPECTED_SHARDS,
                "exact_once": True,
                "all_finite": True,
                "failure_count": 0,
            },
        }
    )
    return transaction


def build_candidate_seal(
    *,
    transaction_artifact: Mapping[str, Any],
    primary_screen_artifact: Mapping[str, Any],
    primary_replay_artifact: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
) -> dict[str, Any]:
    return _validate_one_candidate_for_seal(
        transaction_artifact=transaction_artifact,
        primary_screen_artifact=primary_screen_artifact,
        primary_replay_artifact=primary_replay_artifact,
        prerequisite_artifact=prerequisite_artifact,
        continuation_artifact=continuation_artifact,
        real_feature_cache=real_feature_cache,
    )


def _validate_one_candidate_for_seal(
    *,
    transaction_artifact: Mapping[str, Any],
    primary_screen_artifact: Mapping[str, Any],
    primary_replay_artifact: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
) -> dict[str, Any]:
    tx_artifact, transaction = authority._verify_compact_receipt(
        transaction_artifact, "candidate transaction"
    )
    epoch = transaction["epoch"]
    updates = transaction["optimizer_updates"]
    checkpoint = authority._validate_checkpoint(
        transaction["candidate_checkpoint"], f"Base e{epoch} checkpoint"
    )
    prediction = transaction["prediction_manifest"]
    distribution_artifact, distribution = authority._validate_distribution(
        transaction["distribution_receipt"], prediction_manifest=prediction
    )
    prerequisite, prerequisite_payload, _fixed = (
        authority._validate_prerequisite_selection(prerequisite_artifact)
    )
    continuation, _decision = authority._validate_continuation_decision(
        continuation_artifact,
        prerequisite_artifact=prerequisite,
        prerequisite_selection=prerequisite_payload,
    )
    (
        _tx,
        _payload,
        lineage,
        expected_canonical_manifest,
    ) = authority._validate_candidate_transaction(
        tx_artifact,
        epoch=epoch,
        updates=updates,
        checkpoint=checkpoint,
        prediction_manifest=prediction,
        distribution_artifact=distribution_artifact,
        distribution=distribution,
        prerequisite_artifact=prerequisite,
        continuation_artifact=continuation,
    )
    cache, _cache = authority._normalize_artifact(
        real_feature_cache, "released2 real-feature cache", with_payload=True
    )
    screen, screen_fgd, screen_validation = (
        authority._validate_primary_screen_receipt(
            primary_screen_artifact,
            prediction_manifest=prediction,
            distribution_artifact=distribution_artifact,
            expected_real_feature_cache=cache,
            expected_canonical_manifest=expected_canonical_manifest,
        )
    )
    replay, fgd = authority._validate_primary_screen_replay_receipt(
        primary_replay_artifact,
        screen_artifact=screen,
        screen_validation=screen_validation,
        prediction_manifest=prediction,
        distribution_artifact=distribution_artifact,
    )
    if fgd != screen_fgd:
        raise BaseFreshValOrchestratorError(
            f"Base e{epoch} screen/replay FGD mismatch"
        )
    return _with_payload_sha(
        {
            "format": CANDIDATE_SEAL_FORMAT,
            "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "row": {
                "epoch": epoch,
                "optimizer_updates": updates,
                "candidate_checkpoint": checkpoint,
                "candidate_transaction": tx_artifact,
                "prediction_manifest": prediction,
                "inference_lineage": lineage,
                "distribution_receipt": distribution_artifact,
                "primary_screen_receipt": screen,
                "primary_replay_receipt": replay,
            },
            "body_released2_fgd": fgd,
        }
    )


def build_candidate_evidence(
    *,
    candidate_seals: Sequence[Mapping[str, Any]],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
) -> dict[str, Any]:
    prerequisite, prerequisite_payload, _fixed = (
        authority._validate_prerequisite_selection(prerequisite_artifact)
    )
    continuation, _decision = authority._validate_continuation_decision(
        continuation_artifact,
        prerequisite_artifact=prerequisite,
        prerequisite_selection=prerequisite_payload,
    )
    cache, _cache_payload = authority._normalize_artifact(
        real_feature_cache, "released2 real-feature cache", with_payload=True
    )
    if len(candidate_seals) != len(EXPECTED_EPOCHS):
        raise BaseFreshValOrchestratorError(
            "candidate evidence requires exactly 22 seals"
        )
    rows: list[dict[str, Any]] = []
    seal_paths: set[str] = set()
    for expected_epoch, seal_value in zip(EXPECTED_EPOCHS, candidate_seals):
        seal_artifact, seal = authority._verify_compact_receipt(
            seal_value, f"Base e{expected_epoch} candidate seal"
        )
        if (
            seal.get("format") != CANDIDATE_SEAL_FORMAT
            or seal.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM
            or seal.get("status") != "complete"
            or seal.get("split") != "val"
            or seal.get("test_visible") is not False
            or seal.get("row", {}).get("epoch") != expected_epoch
            or seal_artifact["path"] in seal_paths
        ):
            raise BaseFreshValOrchestratorError(
                f"Base e{expected_epoch} candidate seal changed/reused"
            )
        seal_paths.add(seal_artifact["path"])
        rows.append(seal["row"])
    update_rates = {
        row["optimizer_updates"] // row["epoch"] for row in rows
    }
    if len(update_rates) != 1 or next(iter(update_rates)) not in {248, 1988}:
        raise BaseFreshValOrchestratorError(
            "candidate seals mix Base training topologies"
        )
    updates_per_epoch = next(iter(update_rates))
    selector._validate_candidate_rows(
        rows,
        prerequisite_artifact=prerequisite,
        continuation_artifact=continuation,
        expected_real_feature_cache=cache,
        require_fresh_transaction=True,
        expected_updates_per_epoch=updates_per_epoch,
    )
    evidence = _with_payload_sha(
        {
            "format": selector.FRESH_CANDIDATE_EVIDENCE_FORMAT,
            "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "generator": "SemTalk Base Motion Generation",
            "dataset": "SHOW",
            "target_speaker_scope": authority.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "candidate_epochs": list(EXPECTED_EPOCHS),
            "updates_per_epoch": updates_per_epoch,
            "prerequisite_selection": prerequisite,
            "continuation_decision": continuation,
            "real_feature_cache": cache,
            "candidates": rows,
        }
    )
    return evidence


def _partition_epochs(partition_id: int, partition_count: int) -> list[int]:
    if (
        isinstance(partition_id, bool)
        or isinstance(partition_count, bool)
        or not isinstance(partition_id, int)
        or not isinstance(partition_count, int)
        or partition_count < 1
        or partition_count > len(EXPECTED_EPOCHS)
        or not 0 <= partition_id < partition_count
    ):
        raise BaseFreshValOrchestratorError("invalid static partition")
    return [
        epoch
        for index, epoch in enumerate(EXPECTED_EPOCHS)
        if index % partition_count == partition_id
    ]


def _candidate_seal_envelope(
    seal_value: Mapping[str, Any], expected_epoch: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact, seal = authority._verify_compact_receipt(
        seal_value, f"Base e{expected_epoch} candidate seal"
    )
    row = seal.get("row")
    if (
        seal.get("format") != CANDIDATE_SEAL_FORMAT
        or seal.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM
        or seal.get("status") != "complete"
        or seal.get("split") != "val"
        or seal.get("test_visible") is not False
        or not isinstance(row, dict)
        or row.get("epoch") != expected_epoch
        or row.get("optimizer_updates")
        not in {expected_epoch * 248, expected_epoch * 1988}
    ):
        raise BaseFreshValOrchestratorError(
            f"Base e{expected_epoch} candidate seal identity changed"
        )
    tx_artifact, transaction = authority._verify_compact_receipt(
        row.get("candidate_transaction"),
        f"Base e{expected_epoch} candidate transaction",
    )
    if (
        transaction.get("format") != authority.FRESH_VAL_TRANSACTION_FORMAT
        or transaction.get("generator_module") != GENERATOR_MODULE
        or transaction.get("split") != "val"
        or transaction.get("test_visible") is not False
        or transaction.get("formal_host")
        not in FORMAL_HOST_BY_PARTITION.values()
        or transaction.get("epoch") != expected_epoch
        or transaction.get("optimizer_updates")
        != row.get("optimizer_updates")
        or transaction.get("coverage")
        != {
            "clip_count": EXPECTED_CLIPS,
            "num_shards": EXPECTED_SHARDS,
            "exact_once": True,
            "all_finite": True,
            "failure_count": 0,
        }
        or len(transaction.get("shards", [])) != EXPECTED_SHARDS
        or tx_artifact != row.get("candidate_transaction")
    ):
        raise BaseFreshValOrchestratorError(
            f"Base e{expected_epoch} candidate transaction envelope changed"
        )
    return artifact, seal, transaction


def build_partition_receipt(
    *,
    partition_id: int,
    partition_count: int,
    candidate_seals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if partition_count != 2 or partition_id not in FORMAL_HOST_BY_PARTITION:
        raise BaseFreshValOrchestratorError(
            "formal Base validation requires exact partition 0/2 or 1/2"
        )
    expected_host = FORMAL_HOST_BY_PARTITION[partition_id]
    expected_epochs = _partition_epochs(partition_id, partition_count)
    if len(candidate_seals) != len(expected_epochs):
        raise BaseFreshValOrchestratorError(
            f"partition {partition_id}/{partition_count} must contain "
            f"{len(expected_epochs)} candidate seals"
        )
    normalized_seals: list[dict[str, Any]] = []
    seal_paths: set[str] = set()
    transaction_paths: set[str] = set()
    common: dict[str, Any] | None = None
    for epoch, seal_value in zip(expected_epochs, candidate_seals):
        seal_artifact, _seal, transaction = _candidate_seal_envelope(
            seal_value, epoch
        )
        transaction_path = transaction["inference_lineage"]["path"]
        if (
            seal_artifact["path"] in seal_paths
            or transaction_path in transaction_paths
            or transaction["formal_host"] != expected_host
        ):
            raise BaseFreshValOrchestratorError(
                "partition candidate evidence path was reused"
            )
        seal_paths.add(seal_artifact["path"])
        transaction_paths.add(transaction_path)
        pins = transaction["source_runtime_input_pins"]
        observed_common = {
            "preflight_sha256": transaction["preflight_receipt"]["sha256"],
            "preflight_receipt_payload_sha256": transaction[
                "preflight_receipt"
            ]["receipt_payload_sha256"],
            "prerequisite_selection": transaction[
                "prerequisite_selection"
            ],
            "continuation_decision": transaction["continuation_decision"],
            "source": pins["source"],
            "val_inputs_receipt": pins["val_inputs_receipt"],
            "pipeline_receipt": pins["pipeline_receipt"],
            "runtime_contract_sha256": pins["runtime_contract_sha256"],
        }
        if common is None:
            common = observed_common
        elif observed_common != common:
            raise BaseFreshValOrchestratorError(
                "partition candidates disagree on source/runtime/input pins"
            )
        normalized_seals.append(seal_artifact)
    if common is None:
        raise BaseFreshValOrchestratorError("partition cannot be empty")
    return _with_payload_sha(
        {
            "format": PARTITION_FORMAT,
            "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "generator_module": GENERATOR_MODULE,
            "dataset": "SHOW",
            "target_speaker_scope": authority.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "formal_host": expected_host,
            "partition": {
                "partition_id": partition_id,
                "partition_count": partition_count,
                "assignment": "candidate_index_modulo_partition_count",
            },
            "candidate_epochs": expected_epochs,
            "candidate_seals": normalized_seals,
            "common_source_runtime_inputs": common,
            "coverage": {
                "candidate_count": len(expected_epochs),
                "shard_count": len(expected_epochs) * EXPECTED_SHARDS,
                "clip_evaluations": len(expected_epochs) * EXPECTED_CLIPS,
                "exact_once": True,
                "all_finite": True,
                "failure_count": 0,
            },
        }
    )


def build_partition_union(
    *,
    partition_count: int,
    partition_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if partition_count != 2:
        raise BaseFreshValOrchestratorError(
            "formal Base validation union requires exactly two hosts"
        )
    if len(partition_receipts) != partition_count:
        raise BaseFreshValOrchestratorError(
            "partition union receipt count changed"
        )
    normalized_partitions: list[dict[str, Any]] = []
    seals_by_epoch: dict[int, dict[str, Any]] = {}
    common: dict[str, Any] | None = None
    partition_paths: set[str] = set()
    nested_paths: set[str] = set()
    observed_hosts: set[str] = set()
    for partition_id, value in enumerate(partition_receipts):
        artifact, receipt = authority._verify_compact_receipt(
            value, f"partition {partition_id}/{partition_count}"
        )
        expected_epochs = _partition_epochs(partition_id, partition_count)
        expected_coverage = {
            "candidate_count": len(expected_epochs),
            "shard_count": len(expected_epochs) * EXPECTED_SHARDS,
            "clip_evaluations": len(expected_epochs) * EXPECTED_CLIPS,
            "exact_once": True,
            "all_finite": True,
            "failure_count": 0,
        }
        if (
            receipt.get("format") != PARTITION_FORMAT
            or receipt.get("payload_hash_algorithm")
            != PAYLOAD_HASH_ALGORITHM
            or receipt.get("status") != "complete"
            or receipt.get("generator_module") != GENERATOR_MODULE
            or receipt.get("dataset") != "SHOW"
            or receipt.get("target_speaker_scope")
            != authority.EXPECTED_SCOPE
            or receipt.get("split") != "val"
            or receipt.get("test_visible") is not False
            or receipt.get("formal_host")
            != FORMAL_HOST_BY_PARTITION[partition_id]
            or receipt.get("partition")
            != {
                "partition_id": partition_id,
                "partition_count": partition_count,
                "assignment": "candidate_index_modulo_partition_count",
            }
            or receipt.get("candidate_epochs") != expected_epochs
            or receipt.get("coverage") != expected_coverage
            or not isinstance(receipt.get("candidate_seals"), list)
            or len(receipt["candidate_seals"]) != len(expected_epochs)
            or artifact["path"] in partition_paths
        ):
            raise BaseFreshValOrchestratorError(
                f"partition {partition_id}/{partition_count} changed"
            )
        partition_paths.add(artifact["path"])
        observed_hosts.add(receipt["formal_host"])
        observed_common = receipt.get("common_source_runtime_inputs")
        if common is None:
            common = observed_common
        elif observed_common != common:
            raise BaseFreshValOrchestratorError(
                "partitions disagree on source/runtime/input pins"
            )
        for epoch, seal_value in zip(
            expected_epochs, receipt["candidate_seals"]
        ):
            seal_artifact, _seal, transaction = _candidate_seal_envelope(
                seal_value, epoch
            )
            candidate_paths = {
                seal_artifact["path"],
                transaction["inference_lineage"]["path"],
                transaction["prediction_manifest"]["path"],
                transaction["distribution_receipt"]["path"],
            }
            if (
                epoch in seals_by_epoch
                or candidate_paths & nested_paths
                or transaction["source_runtime_input_pins"]
                ["runtime_contract_sha256"]
                != common["runtime_contract_sha256"]
                or transaction["formal_host"] != receipt["formal_host"]
            ):
                raise BaseFreshValOrchestratorError(
                    "partition union has duplicate/reused/mismatched candidate"
                )
            nested_paths.update(candidate_paths)
            seals_by_epoch[epoch] = seal_artifact
        normalized_partitions.append(artifact)
    if (
        set(seals_by_epoch) != set(EXPECTED_EPOCHS)
        or common is None
        or observed_hosts != set(FORMAL_HOST_BY_PARTITION.values())
    ):
        raise BaseFreshValOrchestratorError(
            "partition union is missing or duplicates frozen candidates"
        )
    ordered_seals = [seals_by_epoch[epoch] for epoch in EXPECTED_EPOCHS]
    return _with_payload_sha(
        {
            "format": PARTITION_UNION_FORMAT,
            "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "generator_module": GENERATOR_MODULE,
            "dataset": "SHOW",
            "target_speaker_scope": authority.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "formal_hosts": [
                FORMAL_HOST_BY_PARTITION[index] for index in range(2)
            ],
            "partition_count": partition_count,
            "partition_receipts": normalized_partitions,
            "candidate_epochs": list(EXPECTED_EPOCHS),
            "candidate_seals": ordered_seals,
            "common_source_runtime_inputs": common,
            "coverage": {
                "candidate_count": len(EXPECTED_EPOCHS),
                "shard_count": len(EXPECTED_EPOCHS) * EXPECTED_SHARDS,
                "clip_evaluations": len(EXPECTED_EPOCHS) * EXPECTED_CLIPS,
                "exact_once": True,
                "all_finite": True,
                "failure_count": 0,
            },
        }
    )


def candidate_seals_from_partition_union(
    union_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _artifact_value, receipt = authority._verify_compact_receipt(
        union_artifact, "fresh Base validation partition union"
    )
    rebuilt = build_partition_union(
        partition_count=receipt.get("partition_count"),
        partition_receipts=receipt.get("partition_receipts", []),
    )
    if rebuilt != receipt:
        raise BaseFreshValOrchestratorError(
            "partition union differs from fresh exact-union replay"
        )
    return list(receipt["candidate_seals"])


def _emit_nul_fields(values: Sequence[Any]) -> None:
    payload = b"".join(os.fsencode(str(value)) + b"\0" for value in values)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _validated_run_spec_from_args(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return validate_run_spec(
        _compact_from_args(args, "run-spec"),
        expected_source_commit=args.source_commit,
        expected_source_tree=args.source_tree,
    )


def _run_spec_fields(spec: Mapping[str, Any], profile: str) -> list[Any]:
    artifacts = (
        spec["prerequisite_selection"],
        spec["continuation_decision"],
        spec["real_feature_cache"],
    )
    if profile == "finalizer":
        fields = [
            artifact[key]
            for artifact in artifacts
            for key in (
                "path",
                "sha256",
                "bytes",
                "receipt_payload_sha256",
            )
        ]
        fields.extend(
            wave[key]
            for wave in spec["continuation_waves"]
            for key in (
                "path",
                "sha256",
                "bytes",
                "receipt_payload_sha256",
            )
        )
        return fields
    if profile != "launcher":
        raise BaseFreshValOrchestratorError(
            "run-spec extraction profile must be launcher or finalizer"
        )
    _gate_artifact, concurrency_gate = _validate_multicandidate_gate(
        spec["multi_candidate_gate"]
    )
    return [
        spec["candidate_bundle"]["manifest"]["path"],
        spec["candidate_bundle"]["manifest"]["sha256"],
        spec["candidate_bundle"]["status"]["path"],
        spec["candidate_bundle"]["status"]["sha256"],
        spec["candidate_bundle"]["frozen_inputs"]["path"],
        spec["candidate_bundle"]["frozen_inputs"]["sha256"],
        spec["val_inputs"]["path"],
        spec["val_inputs"]["sha256"],
        spec["pipeline"]["path"],
        spec["pipeline"]["sha256"],
        spec["prerequisite_selection"]["path"],
        spec["prerequisite_selection"]["sha256"],
        spec["prerequisite_selection"]["bytes"],
        spec["prerequisite_selection"]["receipt_payload_sha256"],
        spec["continuation_decision"]["path"],
        spec["continuation_decision"]["sha256"],
        spec["continuation_decision"]["bytes"],
        spec["continuation_decision"]["receipt_payload_sha256"],
        spec["canonical_manifest"]["path"],
        spec["canonical_manifest"]["sha256"],
        spec["canonical_manifest"]["bytes"],
        spec["metric_assets"]["talkshow_metric_root"],
        spec["metric_assets"]["feature_extractor"]["path"],
        spec["metric_assets"]["smplx_asset"]["path"],
        spec["real_feature_cache"]["path"],
        spec["real_feature_cache"]["sha256"],
        spec["real_feature_cache"]["bytes"],
        spec["real_feature_cache"]["receipt_payload_sha256"],
        spec["seed"],
        concurrency_gate["selected_candidates_per_wave"],
    ]


def _candidate_context_fields(
    spec: Mapping[str, Any],
    preflight_artifact: Mapping[str, Any],
    epoch: int,
) -> list[Any]:
    if epoch not in EXPECTED_EPOCHS:
        raise BaseFreshValOrchestratorError(
            "candidate context epoch is outside the frozen schedule"
        )
    _normalized_preflight, preflight = _preflight_candidate(
        preflight_artifact, epoch
    )
    def same_payload_reference(left: Any, right: Any) -> bool:
        return isinstance(left, dict) and isinstance(right, dict) and all(
            left.get(key) == right.get(key)
            for key in ("path", "sha256", "receipt_payload_sha256")
        )

    if (
        not same_payload_reference(
            preflight["val_inputs_receipt"], spec["val_inputs"]
        )
        or not same_payload_reference(
            preflight["pipeline_receipt"], spec["pipeline"]
        )
        or preflight["candidate_bundle"].get("producer_source")
        != spec["source"]
    ):
        raise BaseFreshValOrchestratorError(
            f"Base e{epoch} preflight differs from the validated run spec"
        )
    matches = [
        row["validation_gate"]
        for row in spec["validation_gates"]
        if row["epoch"] == epoch
    ]
    if len(matches) != 1:
        raise BaseFreshValOrchestratorError(
            f"Base e{epoch} candidate gate is not exact-once"
        )
    gate = matches[0]
    checkpoint = preflight["_candidate"]
    return [
        gate["path"],
        gate["sha256"],
        gate["receipt_payload_sha256"],
        checkpoint["path"],
        checkpoint["sha256"],
        checkpoint["bytes"],
    ]


def _winner_full_context_fields(
    spec: Mapping[str, Any],
    winner_selection: Mapping[str, Any],
) -> list[Any]:
    """Return the exact, freshly replayed winner-only metric inputs.

    This extraction is intentionally unavailable before the immutable
    22-candidate primary-screen selection exists.  It replays that selection
    against the run specification and emits no caller-provided checkpoint or
    metric input.
    """

    prerequisite = spec["prerequisite_selection"]
    continuation = spec["continuation_decision"]
    cache = spec["real_feature_cache"]
    selection, _selection_payload, winner = (
        authority._validate_winner_selection(
            winner_selection,
            prerequisite_artifact=prerequisite,
            continuation_artifact=continuation,
            expected_real_feature_cache=cache,
            require_fresh_transaction=True,
        )
    )
    del selection
    epoch = winner["epoch"]
    updates = winner["optimizer_updates"]
    checkpoint = winner["candidate_checkpoint"]
    prediction = winner["prediction_manifest"]
    distribution_artifact, distribution = authority._validate_distribution(
        winner["distribution_receipt"],
        prediction_manifest=prediction,
    )
    (
        transaction,
        transaction_payload,
        lineage,
        canonical_manifest,
    ) = authority._validate_candidate_transaction(
        winner["candidate_transaction"],
        epoch=epoch,
        updates=updates,
        checkpoint=checkpoint,
        prediction_manifest=prediction,
        distribution_artifact=distribution_artifact,
        distribution=distribution,
        prerequisite_artifact=prerequisite,
        continuation_artifact=continuation,
    )
    screen, _fgd, _screen_validation = (
        authority._validate_primary_screen_receipt(
            winner["primary_screen_receipt"],
            prediction_manifest=prediction,
            distribution_artifact=distribution_artifact,
            expected_real_feature_cache=cache,
            expected_canonical_manifest=canonical_manifest,
        )
    )
    canonical = spec["canonical_manifest"]
    if any(
        canonical_manifest.get(key) != canonical.get(key)
        for key in ("path", "sha256", "bytes")
    ):
        raise BaseFreshValOrchestratorError(
            "selected winner canonical manifest differs from run spec"
        )
    gate_artifact, _gate = authority._verify_compact_receipt(
        transaction_payload["validation_gate"],
        "selected winner deterministic validation gate",
    )
    metric_assets = spec["metric_assets"]
    return [
        epoch,
        updates,
        prediction["path"],
        prediction["sha256"],
        prediction["bytes"],
        lineage["path"],
        lineage["sha256"],
        lineage["bytes"],
        lineage["receipt_payload_sha256"],
        distribution_artifact["path"],
        distribution_artifact["sha256"],
        distribution_artifact["bytes"],
        distribution_artifact["receipt_payload_sha256"],
        gate_artifact["path"],
        gate_artifact["sha256"],
        gate_artifact["bytes"],
        gate_artifact["receipt_payload_sha256"],
        canonical["path"],
        canonical["sha256"],
        canonical["bytes"],
        metric_assets["talkshow_metric_root"],
        metric_assets["feature_extractor"]["path"],
        metric_assets["smplx_asset"]["path"],
        cache["path"],
        cache["sha256"],
        cache["bytes"],
        cache["receipt_payload_sha256"],
        transaction["path"],
        transaction["sha256"],
        transaction["bytes"],
        transaction["receipt_payload_sha256"],
        screen["path"],
        screen["sha256"],
        screen["bytes"],
        screen["receipt_payload_sha256"],
        transaction_payload["formal_host"],
    ]


def _add_payload_artifact(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-path", type=Path, required=True)
    parser.add_argument(f"--{prefix}-sha256", required=True)
    parser.add_argument(f"--{prefix}-payload-sha256", required=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    create_root = commands.add_parser(
        "create-run-root", allow_abbrev=False
    )
    create_root.add_argument("--path", type=Path, required=True)
    create_root.add_argument("--subdirectory", action="append", default=[])
    spec = commands.add_parser("validate-run-spec", allow_abbrev=False)
    _add_payload_artifact(spec, "run-spec")
    spec.add_argument("--source-commit", required=True)
    spec.add_argument("--source-tree", required=True)
    spec.add_argument("--output-json", type=Path, required=True)

    extract_spec = commands.add_parser(
        "extract-run-spec", allow_abbrev=False
    )
    _add_payload_artifact(extract_spec, "run-spec")
    extract_spec.add_argument("--source-commit", required=True)
    extract_spec.add_argument("--source-tree", required=True)
    extract_spec.add_argument(
        "--profile", choices=("launcher", "finalizer"), required=True
    )

    extract_candidate = commands.add_parser(
        "extract-candidate-context", allow_abbrev=False
    )
    _add_payload_artifact(extract_candidate, "run-spec")
    _add_payload_artifact(extract_candidate, "preflight")
    extract_candidate.add_argument("--source-commit", required=True)
    extract_candidate.add_argument("--source-tree", required=True)
    extract_candidate.add_argument("--epoch", type=int, required=True)

    extract_winner = commands.add_parser(
        "extract-winner-full-context", allow_abbrev=False
    )
    _add_payload_artifact(extract_winner, "run-spec")
    _add_payload_artifact(extract_winner, "winner-selection")
    extract_winner.add_argument("--source-commit", required=True)
    extract_winner.add_argument("--source-tree", required=True)

    artifact_fields = commands.add_parser(
        "artifact-fields", allow_abbrev=False
    )
    artifact_fields.add_argument("--artifact-path", type=Path, required=True)
    artifact_fields.add_argument(
        "--payload-receipt", action="store_true"
    )

    plan = commands.add_parser("plan", allow_abbrev=False)
    for prefix in ("preflight", "prerequisite", "continuation"):
        _add_payload_artifact(plan, prefix)
    plan.add_argument("--output-json", type=Path, required=True)

    distribution = commands.add_parser("distribution", allow_abbrev=False)
    _add_payload_artifact(distribution, "lineage")
    distribution.add_argument("--gate-path", type=Path, required=True)
    distribution.add_argument("--gate-sha256", required=True)
    distribution.add_argument("--checkpoint-path", type=Path, required=True)
    distribution.add_argument("--checkpoint-sha256", required=True)
    distribution.add_argument("--checkpoint-bytes", type=int, required=True)
    distribution.add_argument("--output-json", type=Path, required=True)

    failure = commands.add_parser("failure-manifest", allow_abbrev=False)
    failure.add_argument("--epoch", type=int, required=True)
    failure.add_argument("--output-json", type=Path, required=True)

    transaction = commands.add_parser(
        "candidate-transaction", allow_abbrev=False
    )
    transaction.add_argument("--epoch", type=int, required=True)
    transaction.add_argument("--formal-host", required=True)
    transaction.add_argument("--output-root", type=Path, required=True)
    for prefix in (
        "preflight",
        "lineage",
        "distribution",
        "failure",
        "prerequisite",
        "continuation",
    ):
        _add_payload_artifact(transaction, prefix)
    transaction.add_argument("--output-json", type=Path, required=True)

    seal = commands.add_parser("candidate-seal", allow_abbrev=False)
    for prefix in (
        "transaction",
        "primary-screen",
        "primary-replay",
        "prerequisite",
        "continuation",
        "real-feature-cache",
    ):
        _add_payload_artifact(seal, prefix)
    seal.add_argument("--output-json", type=Path, required=True)

    partition = commands.add_parser("seal-partition", allow_abbrev=False)
    partition.add_argument("--partition-id", type=int, required=True)
    partition.add_argument("--partition-count", type=int, required=True)
    partition.add_argument(
        "--candidate-seal", action="append", type=Path, required=True
    )
    partition.add_argument(
        "--candidate-seal-sha256", action="append", required=True
    )
    partition.add_argument(
        "--candidate-seal-payload-sha256", action="append", required=True
    )
    partition.add_argument("--output-json", type=Path, required=True)

    union = commands.add_parser("union-partitions", allow_abbrev=False)
    union.add_argument("--partition-count", type=int, required=True)
    union.add_argument(
        "--partition-receipt", action="append", type=Path, required=True
    )
    union.add_argument(
        "--partition-receipt-sha256", action="append", required=True
    )
    union.add_argument(
        "--partition-receipt-payload-sha256",
        action="append",
        required=True,
    )
    union.add_argument("--output-json", type=Path, required=True)

    evidence = commands.add_parser("publish-evidence", allow_abbrev=False)
    evidence.add_argument(
        "--candidate-seal", action="append", type=Path
    )
    evidence.add_argument(
        "--candidate-seal-sha256", action="append"
    )
    evidence.add_argument(
        "--candidate-seal-payload-sha256", action="append"
    )
    _add_payload_artifact(evidence, "partition-union")
    for prefix in ("prerequisite", "continuation", "real-feature-cache"):
        _add_payload_artifact(evidence, prefix)
    evidence.add_argument("--output-json", type=Path, required=True)

    winner_full = commands.add_parser(
        "winner-full-closure", allow_abbrev=False
    )
    for prefix in (
        "winner-selection",
        "prerequisite",
        "continuation",
        "real-feature-cache",
        "primary-replay",
    ):
        _add_payload_artifact(winner_full, prefix)
    winner_full.add_argument("--report-path", type=Path, required=True)
    winner_full.add_argument("--report-sha256", required=True)
    winner_full.add_argument("--output-json", type=Path, required=True)

    claim = commands.add_parser("publish-fresh-claim", allow_abbrev=False)
    for prefix in (
        "winner-selection",
        "prerequisite",
        "continuation",
        "real-feature-cache",
        "winner-full-closure",
    ):
        _add_payload_artifact(claim, prefix)
    claim.add_argument(
        "--continuation-wave-path", action="append", type=Path, default=[]
    )
    claim.add_argument(
        "--continuation-wave-sha256", action="append", default=[]
    )
    claim.add_argument(
        "--continuation-wave-bytes", action="append", type=int, default=[]
    )
    claim.add_argument(
        "--continuation-wave-payload-sha256",
        action="append",
        default=[],
    )
    claim.add_argument("--expected-test-output-root", type=Path, required=True)
    claim.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "create-run-root":
        path, device, inode = _create_new_directory_tree(
            args.path,
            subdirectories=args.subdirectory,
        )
        _emit_nul_fields([path, device, inode])
        return 0
    if args.command == "extract-run-spec":
        _normalized, spec = _validated_run_spec_from_args(args)
        _emit_nul_fields(_run_spec_fields(spec, args.profile))
        return 0
    if args.command == "extract-candidate-context":
        _normalized, spec = _validated_run_spec_from_args(args)
        _emit_nul_fields(
            _candidate_context_fields(
                spec,
                _compact_from_args(args, "preflight"),
                args.epoch,
            )
        )
        return 0
    if args.command == "extract-winner-full-context":
        _normalized, spec = _validated_run_spec_from_args(args)
        _emit_nul_fields(
            _winner_full_context_fields(
                spec,
                _compact_from_args(args, "winner-selection"),
            )
        )
        return 0
    if args.command == "artifact-fields":
        artifact, _value = _artifact(
            args.artifact_path,
            payload_receipt=args.payload_receipt,
        )
        _emit_nul_fields(
            [
                artifact["sha256"],
                artifact["bytes"],
                artifact.get("receipt_payload_sha256", ""),
            ]
        )
        return 0
    if args.command == "validate-run-spec":
        normalized, _spec = validate_run_spec(
            _compact_from_args(args, "run-spec"),
            expected_source_commit=args.source_commit,
            expected_source_tree=args.source_tree,
        )
        result = _with_payload_sha(
            {
                "format": RUN_SPEC_AUDIT_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "run_spec": normalized,
                "source_commit": args.source_commit,
                "source_tree": args.source_tree,
            }
        )
    elif args.command == "plan":
        result = build_plan(
            preflight_artifact=_compact_from_args(args, "preflight"),
            prerequisite_artifact=_compact_from_args(args, "prerequisite"),
            continuation_artifact=_compact_from_args(args, "continuation"),
        )
    elif args.command == "distribution":
        checkpoint, _ = _artifact(
            args.checkpoint_path,
            args.checkpoint_sha256,
            payload_receipt=False,
        )
        if checkpoint["bytes"] != args.checkpoint_bytes:
            raise BaseFreshValOrchestratorError(
                "checkpoint byte count changed"
            )
        result = build_distribution(
            lineage_artifact=_compact_from_args(args, "lineage"),
            gate_path=args.gate_path,
            expected_gate_sha256=args.gate_sha256,
            checkpoint=checkpoint,
        )
    elif args.command == "failure-manifest":
        result = build_failure_manifest(args.epoch)
    elif args.command == "candidate-transaction":
        result = build_candidate_transaction(
            epoch=args.epoch,
            preflight_artifact=_compact_from_args(args, "preflight"),
            output_root=args.output_root,
            lineage_artifact=_compact_from_args(args, "lineage"),
            distribution_artifact=_compact_from_args(args, "distribution"),
            failure_artifact=_compact_from_args(args, "failure"),
            prerequisite_artifact=_compact_from_args(args, "prerequisite"),
            continuation_artifact=_compact_from_args(args, "continuation"),
            formal_host=args.formal_host,
        )
    elif args.command == "candidate-seal":
        result = _validate_one_candidate_for_seal(
            transaction_artifact=_compact_from_args(args, "transaction"),
            primary_screen_artifact=_compact_from_args(
                args, "primary-screen"
            ),
            primary_replay_artifact=_compact_from_args(
                args, "primary-replay"
            ),
            prerequisite_artifact=_compact_from_args(args, "prerequisite"),
            continuation_artifact=_compact_from_args(args, "continuation"),
            real_feature_cache=_compact_from_args(
                args, "real-feature-cache"
            ),
        )
    elif args.command == "seal-partition":
        if not (
            len(args.candidate_seal)
            == len(args.candidate_seal_sha256)
            == len(args.candidate_seal_payload_sha256)
        ):
            raise BaseFreshValOrchestratorError(
                "candidate seal path/SHA/payload lists differ"
            )
        seals = []
        for path, digest, payload_digest in zip(
            args.candidate_seal,
            args.candidate_seal_sha256,
            args.candidate_seal_payload_sha256,
        ):
            artifact, _ = _artifact(path, digest, payload_receipt=True)
            if artifact["receipt_payload_sha256"] != payload_digest:
                raise BaseFreshValOrchestratorError(
                    "candidate seal payload SHA-256 changed"
                )
            seals.append(artifact)
        result = build_partition_receipt(
            partition_id=args.partition_id,
            partition_count=args.partition_count,
            candidate_seals=seals,
        )
    elif args.command == "union-partitions":
        if not (
            len(args.partition_receipt)
            == len(args.partition_receipt_sha256)
            == len(args.partition_receipt_payload_sha256)
        ):
            raise BaseFreshValOrchestratorError(
                "partition receipt path/SHA/payload lists differ"
            )
        partitions = []
        for path, digest, payload_digest in zip(
            args.partition_receipt,
            args.partition_receipt_sha256,
            args.partition_receipt_payload_sha256,
        ):
            artifact, _ = _artifact(path, digest, payload_receipt=True)
            if artifact["receipt_payload_sha256"] != payload_digest:
                raise BaseFreshValOrchestratorError(
                    "partition receipt payload SHA-256 changed"
                )
            partitions.append(artifact)
        result = build_partition_union(
            partition_count=args.partition_count,
            partition_receipts=partitions,
        )
    elif args.command == "publish-evidence":
        union_artifact = _compact_from_args(args, "partition-union")
        seals = candidate_seals_from_partition_union(union_artifact)
        if any(
            value is not None
            for value in (
                args.candidate_seal,
                args.candidate_seal_sha256,
                args.candidate_seal_payload_sha256,
            )
        ):
            if not (
                args.candidate_seal is not None
                and args.candidate_seal_sha256 is not None
                and args.candidate_seal_payload_sha256 is not None
                and len(args.candidate_seal)
                == len(args.candidate_seal_sha256)
                == len(args.candidate_seal_payload_sha256)
                == len(seals)
            ):
                raise BaseFreshValOrchestratorError(
                    "optional direct seals differ from exact partition union"
                )
            direct = []
            for path, digest, payload_digest in zip(
                args.candidate_seal,
                args.candidate_seal_sha256,
                args.candidate_seal_payload_sha256,
            ):
                artifact, _ = _artifact(path, digest, payload_receipt=True)
                if artifact["receipt_payload_sha256"] != payload_digest:
                    raise BaseFreshValOrchestratorError(
                        "candidate seal payload SHA-256 changed"
                    )
                direct.append(artifact)
            if direct != seals:
                raise BaseFreshValOrchestratorError(
                    "direct seals differ from exact partition union"
                )
        result = build_candidate_evidence(
            candidate_seals=seals,
            prerequisite_artifact=_compact_from_args(args, "prerequisite"),
            continuation_artifact=_compact_from_args(args, "continuation"),
            real_feature_cache=_compact_from_args(
                args, "real-feature-cache"
            ),
        )
    elif args.command == "winner-full-closure":
        report, _report_payload = _artifact(
            args.report_path,
            args.report_sha256,
            payload_receipt=False,
        )
        result = authority.build_winner_full_metric_closure(
            winner_selection=_compact_from_args(args, "winner-selection"),
            prerequisite_selection=_compact_from_args(args, "prerequisite"),
            continuation_decision=_compact_from_args(args, "continuation"),
            real_feature_cache=_compact_from_args(
                args, "real-feature-cache"
            ),
            talkshow_metric_report=report,
            primary_replay_receipt=_compact_from_args(
                args, "primary-replay"
            ),
        )
    elif args.command == "publish-fresh-claim":
        if not (
            len(args.continuation_wave_path)
            == len(args.continuation_wave_sha256)
            == len(args.continuation_wave_bytes)
            == len(args.continuation_wave_payload_sha256)
        ):
            raise BaseFreshValOrchestratorError(
                "continuation wave path/SHA/bytes/payload lists differ"
            )
        waves = []
        for path, digest, expected_bytes, payload_digest in zip(
            args.continuation_wave_path,
            args.continuation_wave_sha256,
            args.continuation_wave_bytes,
            args.continuation_wave_payload_sha256,
        ):
            wave, _payload = _artifact(
                path, digest, payload_receipt=True
            )
            if (
                wave["bytes"] != expected_bytes
                or wave["receipt_payload_sha256"] != payload_digest
            ):
                raise BaseFreshValOrchestratorError(
                    "continuation wave bytes/payload SHA-256 changed"
                )
            waves.append(wave)
        result = authority.build_fresh_published_test_winner_claim(
            winner_selection=_compact_from_args(
                args, "winner-selection"
            ),
            prerequisite_selection=_compact_from_args(args, "prerequisite"),
            continuation_decision=_compact_from_args(args, "continuation"),
            continuation_waves=waves,
            real_feature_cache=_compact_from_args(
                args, "real-feature-cache"
            ),
            winner_full_metric_closure=_compact_from_args(
                args, "winner-full-closure"
            ),
            expected_output_root=args.expected_test_output_root,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    artifact = _write_new(args.output_json.resolve(), result)
    if args.command == "publish-evidence":
        selector.build_published_base_winner_selection(
            artifact,
            prerequisite_selection=_compact_from_args(args, "prerequisite"),
            continuation_decision=_compact_from_args(args, "continuation"),
        )
    elif args.command == "publish-fresh-claim":
        authority.validate_published_test_winner_claim(
            artifact["path"],
            expected_claim_sha256=artifact["sha256"],
            expected_claim_bytes=artifact["bytes"],
            expected_claim_payload_sha256=artifact[
                "receipt_payload_sha256"
            ],
            expected_output_root=args.expected_test_output_root,
            prerequisite_selection=_compact_from_args(args, "prerequisite"),
            continuation_decision=_compact_from_args(args, "continuation"),
            continuation_waves=waves,
            winner_full_metric_closure=_compact_from_args(
                args, "winner-full-closure"
            ),
            expected_claim_format=authority.FRESH_CLAIM_FORMAT,
        )
    print(json.dumps(artifact, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
