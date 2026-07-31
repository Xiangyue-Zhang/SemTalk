#!/usr/bin/env python3
"""Pure authorization contract for one synchronized prerequisite wave.

This module deliberately does not move checkpoints or launch training.  It
turns one freshly replayed continuation decision into a single authorization
covering all five SHOW prerequisite stages.  If any stage requests
continuation, every stage is authorized for exactly one additional 20-epoch
boundary.

The independently selected validation winner and the latest resume boundary
are separate bindings.  A winner may therefore be older than the boundary.
Old candidates remain in their immutable source segment; the new segment is
authorized to create only the new target candidate.
"""

from __future__ import annotations

import hashlib
import io
import json
import copy
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from scripts.show_base import (
    decide_prerequisite_continuation as continuation_decision,
)
from scripts.show_base import prerequisite_boundary_state as boundary_state
from scripts.show_base import prerequisite_val_contract as val_contract
from scripts.show_base import selected_prerequisites as selected_contract


FORMAT = val_contract.CONTINUATION_WAVE_FORMAT
INTERVAL_EPOCHS = 20
MINIMUM_BOUNDARY_EPOCH = 200
EXPECTED_UPDATES_PER_EPOCH = 497
STAGES = ("face", "hands", "upper", "lower", "global")
RVQ_STAGES = frozenset(("face", "hands", "upper", "lower"))


def updates_per_epoch(stage: str) -> int:
    try:
        return val_contract.updates_per_epoch(stage)
    except val_contract.ContractError as error:
        raise ContinuationWaveError(str(error)) from error
FORMAL_HOSTS = frozenset(
    (
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0",
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0",
    )
)
FORMAL_SMPLX_FILENAME = "SMPLX_NEUTRAL_2020.npz"
FORMAL_SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
FORMAL_SMPLX_BYTES = 167_264_530
OFFICIAL_BASELINE_COMMIT = "806b008c97bf51fce203e54109e4c22325253618"
SOURCE_ANCESTRY_FORMAT = "semtalk_show_source_ancestry_v1"

PROTOCOL = {
    "authorization": (
        "fresh_replayed_decision_any_trigger_authorizes_all_five"
    ),
    "wave_scope": "all_five_synchronized_exact_plus_20",
    "selection": (
        "independent_validation_winner_may_precede_resume_boundary"
    ),
    "resume": "latest_boundary_candidate_only",
    "storage": (
        "immutable_segments_new_segment_contains_target_candidate_only"
    ),
    "interval_epochs": INTERVAL_EPOCHS,
}

TOP_KEYS = {
    "format",
    "status",
    "test_visible",
    "protocol",
    "decision",
    "trigger_stages",
    "boundary_epoch",
    "target_epoch",
    "stages",
    "receipt_payload_sha256",
}
DECISION_BINDING_KEYS = {
    "path",
    "sha256",
    "receipt_payload_sha256",
}
STAGE_KEYS = {
    "stage",
    "triggered",
    "selection_metric",
    "training_topology",
    "independent_selected_candidate",
    "resume_boundary_candidate",
    "old_segment",
    "new_segment",
}
TOPOLOGY_KEYS = {
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "updates_per_epoch",
}
CANDIDATE_KEYS = {
    "epoch",
    "optimizer_updates",
    "checkpoint",
    "checkpoint_audit_sha256",
}
CHECKPOINT_KEYS = {"path", "sha256", "bytes"}
SOURCE_KEYS = {"commit", "tree", "source_receipt_sha256"}
SOURCE_ANCESTRY_KEYS = {
    "format",
    "repository_path",
    "origin",
    "head_commit",
    "head_tree",
    "old_commit",
    "baseline_commit",
    "clean",
    "detached_head",
    "local_branch_ref_count",
    "old_is_ancestor",
    "baseline_is_ancestor",
    "receipt_payload_sha256",
}
SMPLX_ASSET_KEYS = {
    "format",
    "filename",
    "path",
    "sha256",
    "bytes",
    "regular_file",
    "symlink",
}
SEGMENT_CHAIN_ENTRY_KEYS = {
    "run_path",
    "start_epoch",
    "end_epoch",
    "candidate_epochs",
    "predecessor_segment_id",
    "segment_id",
}
OLD_SEGMENT_KEYS = {
    "run_path",
    "segment_id",
    "terminal_epoch",
    "source",
    "host",
    "smplx_asset",
    "config_sha256",
    "source_audit_sha256",
    "dataset_receipt_sha256",
    "config_semantic_sha256",
    "dataset_semantic_sha256",
    "candidate_catalog_receipt",
    "candidate_inventory_sha256",
    "candidate_segment_chain",
    "predecessor_wave",
    "formal_status",
    "boundary_resume",
    "final_checkpoint",
    "boundary_state",
}
NEW_SEGMENT_KEYS = {
    "run_path",
    "segment_id",
    "target_epoch",
    "source",
    "source_ancestry",
    "host",
    "smplx_asset",
    "config_sha256",
    "config_semantic_sha256",
    "dataset_semantic_sha256",
    "authorized_candidate_epochs",
    "predecessor_segment_id",
    "chain_segment_id",
}
STAGE_PLAN_KEYS = {
    "stage",
    "old_run_path",
    "new_run_path",
    "old_source",
    "new_source",
    "source_ancestry",
    "old_host",
    "new_host",
    "old_smplx_asset",
    "new_smplx_asset",
    "old_config_sha256",
    "new_config_sha256",
    "old_source_audit_sha256",
    "old_dataset_receipt_sha256",
    "old_config_semantic_sha256",
    "new_config_semantic_sha256",
    "old_dataset_semantic_sha256",
    "new_dataset_semantic_sha256",
    "candidate_catalog_receipt",
    "candidate_segment_chain",
    "predecessor_wave",
    "formal_status",
    "boundary_resume",
    "final_checkpoint",
    "boundary_state",
}
ADAPTER_STAGE_PLAN_KEYS = (
    STAGE_PLAN_KEYS
    - {
    "candidate_catalog_receipt",
    "old_source_audit_sha256",
    "old_dataset_receipt_sha256",
    "source_ancestry",
    "formal_status",
    "boundary_resume",
    "final_checkpoint",
    "boundary_state",
    }
) | {"new_source_repository"}


class ContinuationWaveError(RuntimeError):
    """Raised when a synchronized continuation wave is not exact."""


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
        raise ContinuationWaveError(
            "continuation wave is not canonical finite JSON"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_keys(
    value: Any,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContinuationWaveError(f"{label} schema mismatch")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContinuationWaveError(f"{label} must be an exact integer")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ContinuationWaveError(f"{label} must be a boolean")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuationWaveError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuationWaveError(
            f"{label} must be a lowercase 40-hex Git object ID"
        )
    return value


def _absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContinuationWaveError(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ContinuationWaveError(
            f"{label} must be an absolute normalized path"
        )
    return value


def _is_within(path: str, parent: str) -> bool:
    try:
        Path(path).relative_to(Path(parent))
    except ValueError:
        return False
    return True


def _validate_payload_binding(value: Any, label: str) -> dict[str, Any]:
    binding = _exact_keys(value, DECISION_BINDING_KEYS, label)
    _absolute_path(binding["path"], f"{label} path")
    _require_sha256(binding["sha256"], f"{label} SHA-256")
    _require_sha256(
        binding["receipt_payload_sha256"],
        f"{label} payload SHA-256",
    )
    return dict(binding)


def _validate_checkpoint(value: Any, label: str) -> dict[str, Any]:
    checkpoint = _exact_keys(value, CHECKPOINT_KEYS, label)
    _absolute_path(checkpoint["path"], f"{label} path")
    _require_sha256(checkpoint["sha256"], f"{label} SHA-256")
    byte_count = _require_int(checkpoint["bytes"], f"{label} bytes")
    if byte_count <= 0:
        raise ContinuationWaveError(f"{label} bytes must be positive")
    return dict(checkpoint)


def _validate_candidate(
    value: Any,
    *,
    stage: str,
    label: str,
) -> dict[str, Any]:
    candidate = _exact_keys(value, CANDIDATE_KEYS, label)
    epoch = _require_int(candidate["epoch"], f"{label} epoch")
    updates = _require_int(
        candidate["optimizer_updates"],
        f"{label} optimizer updates",
    )
    if (
        epoch <= 0
        or epoch % INTERVAL_EPOCHS != 0
        or updates != epoch * updates_per_epoch(stage)
    ):
        raise ContinuationWaveError(f"{label} epoch/update mismatch")
    checkpoint = _validate_checkpoint(
        candidate["checkpoint"],
        f"{label} checkpoint",
    )
    _require_sha256(
        candidate["checkpoint_audit_sha256"],
        f"{label} checkpoint audit SHA-256",
    )
    return {
        "epoch": epoch,
        "optimizer_updates": updates,
        "checkpoint": checkpoint,
        "checkpoint_audit_sha256": candidate[
            "checkpoint_audit_sha256"
        ],
    }


def _validate_source(value: Any, label: str) -> dict[str, Any]:
    source = _exact_keys(value, SOURCE_KEYS, label)
    _require_git_oid(source["commit"], f"{label} commit")
    _require_git_oid(source["tree"], f"{label} tree")
    _require_sha256(
        source["source_receipt_sha256"],
        f"{label} source receipt SHA-256",
    )
    return dict(source)


def _validate_source_ancestry(
    value: Any,
    *,
    old_source: Mapping[str, Any],
    new_source: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    proof = _exact_keys(value, SOURCE_ANCESTRY_KEYS, label)
    repository_path = _absolute_path(
        proof["repository_path"],
        f"{label} repository path",
    )
    if (
        proof["format"] != SOURCE_ANCESTRY_FORMAT
        or proof["origin"] != val_contract.EXPECTED_ORIGIN
        or proof["head_commit"] != new_source["commit"]
        or proof["head_tree"] != new_source["tree"]
        or proof["old_commit"] != old_source["commit"]
        or proof["baseline_commit"] != OFFICIAL_BASELINE_COMMIT
        or _require_bool(proof["clean"], f"{label} clean") is not True
        or _require_bool(
            proof["detached_head"],
            f"{label} detached_head",
        )
        is not True
        or _require_int(
            proof["local_branch_ref_count"],
            f"{label} local branch ref count",
        )
        != 0
        or _require_bool(
            proof["old_is_ancestor"],
            f"{label} old_is_ancestor",
        )
        is not True
        or _require_bool(
            proof["baseline_is_ancestor"],
            f"{label} baseline_is_ancestor",
        )
        is not True
    ):
        raise ContinuationWaveError(
            f"{label} does not prove the exact source ancestry"
        )
    claimed = _require_sha256(
        proof["receipt_payload_sha256"],
        f"{label} payload SHA-256",
    )
    unsigned = dict(proof)
    unsigned.pop("receipt_payload_sha256")
    if canonical_json_sha256(unsigned) != claimed:
        raise ContinuationWaveError(
            f"{label} payload SHA-256 mismatch"
        )
    return {**dict(proof), "repository_path": repository_path}


def prove_source_ancestry(
    repository_path: Path,
    *,
    old_source: Mapping[str, Any],
    new_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove ancestry from immutable OIDs in a clean detached checkout."""

    repository_text = _absolute_path(
        str(repository_path),
        "source ancestry repository",
    )
    repository = Path(repository_text)
    try:
        mode = repository.lstat().st_mode
    except FileNotFoundError:
        raise ContinuationWaveError(
            "source ancestry repository is missing"
        ) from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ContinuationWaveError(
            "source ancestry repository must be a real directory"
        )
    try:
        resolved = repository.resolve(strict=True)
    except OSError as error:
        raise ContinuationWaveError(
            f"source ancestry repository cannot resolve: {error}"
        ) from error
    if resolved != repository:
        raise ContinuationWaveError(
            "source ancestry repository path changed during resolution"
        )

    def git(
        *arguments: str,
        accepted: tuple[int, ...] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", repository_text, *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in accepted:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ContinuationWaveError(
                "source ancestry git command failed: "
                f"git {' '.join(arguments)}: {detail}"
            )
        return result

    origin = git("remote", "get-url", "origin").stdout.strip()
    head_commit = git("rev-parse", "HEAD").stdout.strip()
    head_tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
    status_text = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    symbolic = git(
        "symbolic-ref",
        "-q",
        "HEAD",
        accepted=(0, 1),
    )
    branch_refs = [
        line
        for line in git(
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
        ).stdout.splitlines()
        if line
    ]
    old_ancestor = git(
        "merge-base",
        "--is-ancestor",
        old_source["commit"],
        new_source["commit"],
        accepted=(0, 1),
    ).returncode == 0
    baseline_ancestor = git(
        "merge-base",
        "--is-ancestor",
        OFFICIAL_BASELINE_COMMIT,
        new_source["commit"],
        accepted=(0, 1),
    ).returncode == 0
    proof: dict[str, Any] = {
        "format": SOURCE_ANCESTRY_FORMAT,
        "repository_path": repository_text,
        "origin": origin,
        "head_commit": head_commit,
        "head_tree": head_tree,
        "old_commit": old_source["commit"],
        "baseline_commit": OFFICIAL_BASELINE_COMMIT,
        "clean": status_text == "",
        "detached_head": symbolic.returncode == 1,
        "local_branch_ref_count": len(branch_refs),
        "old_is_ancestor": old_ancestor,
        "baseline_is_ancestor": baseline_ancestor,
    }
    proof["receipt_payload_sha256"] = canonical_json_sha256(proof)
    return _validate_source_ancestry(
        proof,
        old_source=old_source,
        new_source=new_source,
        label="source ancestry proof",
    )


def _validate_host(value: Any, label: str) -> str:
    if not isinstance(value, str) or value not in FORMAL_HOSTS:
        raise ContinuationWaveError(
            f"{label} must be one of the two exact formal hosts"
        )
    return value


def _validate_smplx_asset(
    value: Any,
    *,
    stage: str,
    host: str,
    label: str,
) -> dict[str, Any] | None:
    """Validate the host-local immutable SMPL-X binding.

    Global has no SMPL-X dependency.  Every RVQ segment binds the same frozen
    bytes while retaining the absolute host-local path that was actually
    used.
    """

    _validate_host(host, f"{label} host")
    if stage == "global":
        if value is not None:
            raise ContinuationWaveError(
                f"{label} must be null for the Global stage"
            )
        return None
    if stage not in RVQ_STAGES:
        raise ContinuationWaveError(f"unknown prerequisite stage {stage!r}")
    asset = _exact_keys(value, SMPLX_ASSET_KEYS, label)
    path = _absolute_path(asset["path"], f"{label} path")
    byte_count = _require_int(asset["bytes"], f"{label} bytes")
    regular_file = _require_bool(
        asset["regular_file"],
        f"{label} regular_file",
    )
    symlink = _require_bool(asset["symlink"], f"{label} symlink")
    if (
        asset["format"] != "semtalk_show_smplx_asset_v1"
        or asset["filename"] != FORMAL_SMPLX_FILENAME
        or Path(path).name != FORMAL_SMPLX_FILENAME
        or asset["sha256"] != FORMAL_SMPLX_SHA256
        or byte_count != FORMAL_SMPLX_BYTES
        or regular_file is not True
        or symlink is not False
    ):
        raise ContinuationWaveError(
            f"{label} differs from the pinned SMPL-X asset"
        )
    return {
        **dict(asset),
        "path": path,
        "bytes": byte_count,
        "regular_file": True,
        "symlink": False,
    }


def _topology(stage: str) -> dict[str, int]:
    if stage in RVQ_STAGES:
        return {
            "world_size": 4,
            "local_batch_size": 64,
            "global_batch_size": 256,
            "updates_per_epoch": updates_per_epoch(stage),
        }
    if stage == "global":
        return {
            "world_size": 1,
            "local_batch_size": 64,
            "global_batch_size": 64,
            "updates_per_epoch": updates_per_epoch(stage),
        }
    raise ContinuationWaveError(f"unknown prerequisite stage {stage!r}")


def _validate_topology(
    value: Any,
    *,
    stage: str,
    label: str,
) -> dict[str, int]:
    topology = _exact_keys(value, TOPOLOGY_KEYS, label)
    for key in TOPOLOGY_KEYS:
        _require_int(topology[key], f"{label} {key}")
    expected = _topology(stage)
    if topology != expected:
        raise ContinuationWaveError(f"{label} protocol mismatch")
    return expected


def _segment_payload(
    segment: Mapping[str, Any],
    *,
    id_key: str,
) -> dict[str, Any]:
    payload = dict(segment)
    payload.pop(id_key, None)
    return payload


def _segment_id(segment: Mapping[str, Any], *, id_key: str) -> str:
    return canonical_json_sha256(_segment_payload(segment, id_key=id_key))


def _chain_segment_id(value: Mapping[str, Any]) -> str:
    return _segment_id(value, id_key="segment_id")


def _make_chain_segment(
    *,
    run_path: str,
    start_epoch: int,
    end_epoch: int,
    predecessor_segment_id: str | None,
) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "run_path": _absolute_path(run_path, "chain segment run"),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "candidate_epochs": list(
            range(start_epoch, end_epoch + 1, INTERVAL_EPOCHS)
        ),
        "predecessor_segment_id": predecessor_segment_id,
    }
    segment["segment_id"] = _chain_segment_id(segment)
    return segment


def _validate_segment_chain(
    value: Any,
    *,
    stage: str,
    boundary: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContinuationWaveError(
            f"{stage} candidate segment chain is empty"
        )
    result: list[dict[str, Any]] = []
    expected_start = INTERVAL_EPOCHS
    observed_runs: set[str] = set()
    predecessor: str | None = None
    for index, raw in enumerate(value):
        segment = _exact_keys(
            raw,
            SEGMENT_CHAIN_ENTRY_KEYS,
            f"{stage} candidate segment chain {index}",
        )
        run_path = _absolute_path(
            segment["run_path"],
            f"{stage} candidate segment chain {index} run",
        )
        start = _require_int(
            segment["start_epoch"],
            f"{stage} candidate segment chain {index} start",
        )
        end = _require_int(
            segment["end_epoch"],
            f"{stage} candidate segment chain {index} end",
        )
        expected_epochs = list(
            range(start, end + 1, INTERVAL_EPOCHS)
        )
        if (
            run_path in observed_runs
            or start != expected_start
            or start <= 0
            or start % INTERVAL_EPOCHS
            or end < start
            or end % INTERVAL_EPOCHS
            or segment["candidate_epochs"] != expected_epochs
            or segment["predecessor_segment_id"] != predecessor
            or (index > 0 and start != end)
        ):
            raise ContinuationWaveError(
                f"{stage} candidate segment chain is not append-only"
            )
        segment_id = _require_sha256(
            segment["segment_id"],
            f"{stage} candidate segment chain {index} ID",
        )
        normalized = {
            **dict(segment),
            "run_path": run_path,
            "start_epoch": start,
            "end_epoch": end,
            "candidate_epochs": expected_epochs,
        }
        if segment_id != _chain_segment_id(normalized):
            raise ContinuationWaveError(
                f"{stage} candidate segment chain {index} ID mismatch"
            )
        result.append(normalized)
        observed_runs.add(run_path)
        predecessor = segment_id
        expected_start = end + INTERVAL_EPOCHS
    if result[-1]["end_epoch"] != boundary:
        raise ContinuationWaveError(
            f"{stage} candidate segment chain boundary mismatch"
        )
    return result


def _segment_for_epoch(
    chain: Sequence[Mapping[str, Any]],
    epoch: int,
    *,
    stage: str,
) -> Mapping[str, Any]:
    matches = [
        segment
        for segment in chain
        if epoch in segment["candidate_epochs"]
    ]
    if len(matches) != 1:
        raise ContinuationWaveError(
            f"{stage} candidate epoch has no exact chain segment"
        )
    return matches[0]


def _validate_decision(
    value: Any,
) -> tuple[int, tuple[str, ...], str]:
    if not isinstance(value, dict):
        raise ContinuationWaveError("continuation decision must be an object")
    claimed_payload_sha = _require_sha256(
        value.get("receipt_payload_sha256"),
        "continuation decision payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256")
    if canonical_json_sha256(unsigned) != claimed_payload_sha:
        raise ContinuationWaveError(
            "continuation decision payload SHA-256 mismatch"
        )
    if (
        value.get("status") != "complete"
        or value.get("decision") != "continue"
        or value.get("test_visible") is not False
    ):
        raise ContinuationWaveError(
            "continuation decision does not authorize a wave"
        )
    stages = value.get("stages")
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise ContinuationWaveError(
            "continuation decision must cover ordered five stages"
        )
    boundaries: list[int] = []
    triggers: list[str] = []
    for expected_stage, stage_value in zip(STAGES, stages):
        if not isinstance(stage_value, dict):
            raise ContinuationWaveError(
                f"continuation decision {expected_stage} stage is invalid"
            )
        if stage_value.get("stage") != expected_stage:
            raise ContinuationWaveError(
                "continuation decision stage order changed"
            )
        boundary = _require_int(
            stage_value.get("latest_epoch"),
            f"continuation decision {expected_stage} latest epoch",
        )
        triggered = _require_bool(
            stage_value.get("requests_continuation"),
            (
                f"continuation decision {expected_stage} "
                "requests_continuation"
            ),
        )
        boundaries.append(boundary)
        if triggered:
            triggers.append(expected_stage)
    if len(set(boundaries)) != 1:
        raise ContinuationWaveError(
            "all five stages must share one resume boundary"
        )
    boundary = boundaries[0]
    if (
        boundary < MINIMUM_BOUNDARY_EPOCH
        or boundary % INTERVAL_EPOCHS != 0
    ):
        raise ContinuationWaveError("resume boundary is not formal")
    if not triggers:
        raise ContinuationWaveError(
            "continue decision has no triggering stage"
        )
    return boundary, tuple(triggers), claimed_payload_sha


def _require_disjoint_run_paths(
    old_runs: set[str],
    new_runs: set[str],
) -> None:
    labelled = [
        *(("old", path) for path in sorted(old_runs)),
        *(("new", path) for path in sorted(new_runs)),
    ]
    for index, (left_kind, left) in enumerate(labelled):
        for right_kind, right in labelled[index + 1 :]:
            if (
                left == right
                or _is_within(left, right)
                or _is_within(right, left)
            ):
                raise ContinuationWaveError(
                    "stage run paths are not globally disjoint: "
                    f"{left_kind}={left!r}, {right_kind}={right!r}"
                )


def _normalize_catalog(
    value: Any,
    *,
    stage: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContinuationWaveError(f"{stage} candidate catalog is empty")
    result = [
        _validate_candidate(
            candidate,
            stage=stage,
            label=f"{stage} candidate catalog {index}",
        )
        for index, candidate in enumerate(value)
    ]
    epochs = [candidate["epoch"] for candidate in result]
    expected_epochs = list(
        range(INTERVAL_EPOCHS, epochs[-1] + 1, INTERVAL_EPOCHS)
    )
    if epochs != expected_epochs:
        raise ContinuationWaveError(
            f"{stage} candidate catalog must exactly cover e20..boundary"
        )
    return result


def _candidate_at(
    candidates: Sequence[Mapping[str, Any]],
    epoch: int,
    *,
    stage: str,
    label: str,
) -> dict[str, Any]:
    matches = [
        dict(candidate)
        for candidate in candidates
        if candidate["epoch"] == epoch
    ]
    if len(matches) != 1:
        raise ContinuationWaveError(
            f"{stage} {label} candidate is not exact-once"
        )
    return matches[0]


def _normalize_plan(
    value: Any,
    *,
    stage: str,
    boundary: int,
) -> dict[str, Any]:
    plan = _exact_keys(value, STAGE_PLAN_KEYS, f"{stage} stage plan")
    if plan["stage"] != stage:
        raise ContinuationWaveError(f"{stage} stage plan changed")
    old_run = _absolute_path(
        plan["old_run_path"],
        f"{stage} old run",
    )
    new_run = _absolute_path(
        plan["new_run_path"],
        f"{stage} new run",
    )
    if (
        old_run == new_run
        or _is_within(old_run, new_run)
        or _is_within(new_run, old_run)
    ):
        raise ContinuationWaveError(
            f"{stage} new segment must be disjoint from old run"
        )
    candidate_segment_chain = _validate_segment_chain(
        plan["candidate_segment_chain"],
        stage=stage,
        boundary=boundary,
    )
    predecessor_wave_value = plan["predecessor_wave"]
    predecessor_wave = (
        None
        if predecessor_wave_value is None
        else _validate_payload_binding(
            predecessor_wave_value,
            f"{stage} predecessor wave",
        )
    )
    if (
        candidate_segment_chain[-1]["run_path"] != old_run
        or (len(candidate_segment_chain) == 1)
        != (predecessor_wave is None)
    ):
        raise ContinuationWaveError(
            f"{stage} old run/predecessor wave differs from segment chain"
        )
    old_source = _validate_source(
        plan["old_source"],
        f"{stage} old source",
    )
    new_source = _validate_source(
        plan["new_source"],
        f"{stage} new source",
    )
    source_ancestry = _validate_source_ancestry(
        plan["source_ancestry"],
        old_source=old_source,
        new_source=new_source,
        label=f"{stage} source ancestry",
    )
    old_host = _validate_host(
        plan["old_host"],
        f"{stage} old host",
    )
    new_host = _validate_host(
        plan["new_host"],
        f"{stage} new host",
    )
    old_smplx_asset = _validate_smplx_asset(
        plan["old_smplx_asset"],
        stage=stage,
        host=old_host,
        label=f"{stage} old SMPL-X asset",
    )
    new_smplx_asset = _validate_smplx_asset(
        plan["new_smplx_asset"],
        stage=stage,
        host=new_host,
        label=f"{stage} new SMPL-X asset",
    )
    for key in (
        "old_config_sha256",
        "new_config_sha256",
        "old_source_audit_sha256",
        "old_dataset_receipt_sha256",
        "old_config_semantic_sha256",
        "new_config_semantic_sha256",
        "old_dataset_semantic_sha256",
        "new_dataset_semantic_sha256",
    ):
        _require_sha256(plan[key], f"{stage} {key}")
    if (
        plan["old_config_semantic_sha256"]
        != plan["new_config_semantic_sha256"]
        or plan["old_dataset_semantic_sha256"]
        != plan["new_dataset_semantic_sha256"]
    ):
        raise ContinuationWaveError(
            f"{stage} training semantics changed across wave"
        )
    catalog_receipt = _validate_payload_binding(
        plan["candidate_catalog_receipt"],
        f"{stage} candidate catalog receipt",
    )
    runtime_bindings = {
        key: _validate_checkpoint(
            plan[key],
            f"{stage} {key.replace('_', ' ')}",
        )
        for key in (
            "formal_status",
            "boundary_resume",
            "final_checkpoint",
        )
    }
    for key, binding in runtime_bindings.items():
        if not _is_within(binding["path"], old_run):
            raise ContinuationWaveError(
                f"{stage} {key.replace('_', ' ')} is outside old run"
            )
    try:
        state_proof = boundary_state.validate_boundary_state_proof(
            plan["boundary_state"]
        )
    except boundary_state.BoundaryStateError as error:
        raise ContinuationWaveError(
            f"{stage} boundary state proof is invalid: {error}"
        ) from error
    if state_proof["stage"] != stage:
        raise ContinuationWaveError(
            f"{stage} boundary state proof stage mismatch"
        )
    return {
        **dict(plan),
        "old_run_path": old_run,
        "new_run_path": new_run,
        "old_source": old_source,
        "new_source": new_source,
        "source_ancestry": source_ancestry,
        "candidate_segment_chain": candidate_segment_chain,
        "predecessor_wave": predecessor_wave,
        "old_host": old_host,
        "new_host": new_host,
        "old_smplx_asset": old_smplx_asset,
        "new_smplx_asset": new_smplx_asset,
        "candidate_catalog_receipt": catalog_receipt,
        **runtime_bindings,
        "boundary_state": state_proof,
    }


def _old_segment(
    *,
    plan: Mapping[str, Any],
    boundary: int,
    catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "run_path": plan["old_run_path"],
        "terminal_epoch": boundary,
        "source": dict(plan["old_source"]),
        "host": plan["old_host"],
        "smplx_asset": copy.deepcopy(plan["old_smplx_asset"]),
        "config_sha256": plan["old_config_sha256"],
        "source_audit_sha256": plan["old_source_audit_sha256"],
        "dataset_receipt_sha256": plan[
            "old_dataset_receipt_sha256"
        ],
        "config_semantic_sha256": plan[
            "old_config_semantic_sha256"
        ],
        "dataset_semantic_sha256": plan[
            "old_dataset_semantic_sha256"
        ],
        "candidate_catalog_receipt": dict(
            plan["candidate_catalog_receipt"]
        ),
        "candidate_inventory_sha256": canonical_json_sha256(
            list(catalog)
        ),
        "candidate_segment_chain": copy.deepcopy(
            plan["candidate_segment_chain"]
        ),
        "predecessor_wave": copy.deepcopy(plan["predecessor_wave"]),
        "formal_status": dict(plan["formal_status"]),
        "boundary_resume": dict(plan["boundary_resume"]),
        "final_checkpoint": dict(plan["final_checkpoint"]),
        "boundary_state": dict(plan["boundary_state"]),
    }
    segment["segment_id"] = _segment_id(segment, id_key="segment_id")
    return segment


def _new_segment(
    *,
    plan: Mapping[str, Any],
    target: int,
    predecessor_segment_id: str,
) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "run_path": plan["new_run_path"],
        "target_epoch": target,
        "source": dict(plan["new_source"]),
        "source_ancestry": dict(plan["source_ancestry"]),
        "host": plan["new_host"],
        "smplx_asset": copy.deepcopy(plan["new_smplx_asset"]),
        "config_sha256": plan["new_config_sha256"],
        "config_semantic_sha256": plan[
            "new_config_semantic_sha256"
        ],
        "dataset_semantic_sha256": plan[
            "new_dataset_semantic_sha256"
        ],
        "authorized_candidate_epochs": [target],
        "predecessor_segment_id": predecessor_segment_id,
    }
    chain_segment = _make_chain_segment(
        run_path=plan["new_run_path"],
        start_epoch=target,
        end_epoch=target,
        predecessor_segment_id=predecessor_segment_id,
    )
    segment["chain_segment_id"] = chain_segment["segment_id"]
    segment["segment_id"] = _segment_id(segment, id_key="segment_id")
    return segment


def authorize_wave(
    *,
    decision_receipt: Mapping[str, Any],
    decision_binding: Mapping[str, Any],
    selected_by_stage: Mapping[str, Any],
    candidate_catalog_by_stage: Mapping[str, Any],
    stage_plans: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic all-five +20 authorization receipt."""

    boundary, triggers, decision_payload_sha = _validate_decision(
        decision_receipt
    )
    target = boundary + INTERVAL_EPOCHS
    binding = _validate_payload_binding(
        decision_binding,
        "continuation decision binding",
    )
    if binding["receipt_payload_sha256"] != decision_payload_sha:
        raise ContinuationWaveError(
            "continuation decision binding payload mismatch"
        )
    if (
        not isinstance(selected_by_stage, Mapping)
        or set(selected_by_stage) != set(STAGES)
        or not isinstance(candidate_catalog_by_stage, Mapping)
        or set(candidate_catalog_by_stage) != set(STAGES)
        or not isinstance(stage_plans, Mapping)
        or set(stage_plans) != set(STAGES)
    ):
        raise ContinuationWaveError(
            "wave inputs must exactly cover all five stages"
        )

    stage_entries: list[dict[str, Any]] = []
    old_runs: set[str] = set()
    new_runs: set[str] = set()
    new_source_identities: set[tuple[str, str]] = set()
    segment_hosts: set[str] = set()
    smplx_hosts: set[str] = set()
    predecessor_wave_payloads: set[str | None] = set()
    for stage in STAGES:
        plan = _normalize_plan(
            stage_plans[stage],
            stage=stage,
            boundary=boundary,
        )
        segment_hosts.update((plan["old_host"], plan["new_host"]))
        if plan["old_smplx_asset"] is not None:
            smplx_hosts.add(plan["old_host"])
        if plan["new_smplx_asset"] is not None:
            smplx_hosts.add(plan["new_host"])
        predecessor_wave_payloads.add(
            None
            if plan["predecessor_wave"] is None
            else canonical_json_sha256(plan["predecessor_wave"])
        )
        if (
            plan["boundary_state"]["boundary_epoch"] != boundary
            or plan["boundary_state"]["world_size"]
            != _topology(stage)["world_size"]
        ):
            raise ContinuationWaveError(
                f"{stage} boundary state differs from decision/topology"
            )
        new_source_identities.add(
            (
                plan["new_source"]["commit"],
                plan["new_source"]["tree"],
            )
        )
        if plan["old_run_path"] in old_runs:
            raise ContinuationWaveError("old stage runs must be unique")
        if plan["new_run_path"] in new_runs:
            raise ContinuationWaveError("new stage runs must be unique")
        old_runs.add(plan["old_run_path"])
        new_runs.add(plan["new_run_path"])

        catalog = _normalize_catalog(
            candidate_catalog_by_stage[stage],
            stage=stage,
        )
        catalog_boundary = catalog[-1]["epoch"]
        if catalog_boundary != boundary:
            raise ContinuationWaveError(
                f"{stage} catalog boundary differs from decision"
            )
        selected = _validate_candidate(
            selected_by_stage[stage],
            stage=stage,
            label=f"{stage} independently selected candidate",
        )
        expected_selected = _candidate_at(
            catalog,
            selected["epoch"],
            stage=stage,
            label="selected",
        )
        if selected != expected_selected:
            raise ContinuationWaveError(
                f"{stage} selected candidate differs from catalog"
            )
        boundary_candidate = _candidate_at(
            catalog,
            boundary,
            stage=stage,
            label="resume boundary",
        )
        for candidate in catalog:
            candidate_segment = _segment_for_epoch(
                plan["candidate_segment_chain"],
                candidate["epoch"],
                stage=stage,
            )
            if not _is_within(
                candidate["checkpoint"]["path"],
                str(
                    Path(candidate_segment["run_path"])
                    / "representation_candidates"
                ),
            ):
                raise ContinuationWaveError(
                    f"{stage} checkpoint is outside its immutable segment"
                )

        old_segment = _old_segment(
            plan=plan,
            boundary=boundary,
            catalog=catalog,
        )
        new_segment = _new_segment(
            plan=plan,
            target=target,
            predecessor_segment_id=plan["candidate_segment_chain"][-1][
                "segment_id"
            ],
        )
        stage_entries.append(
            {
                "stage": stage,
                "triggered": stage in triggers,
                "selection_metric": val_contract.SELECTION_METRICS[stage],
                "training_topology": _topology(stage),
                "independent_selected_candidate": selected,
                "resume_boundary_candidate": boundary_candidate,
                "old_segment": old_segment,
                "new_segment": new_segment,
            }
        )
    _require_disjoint_run_paths(old_runs, new_runs)
    if len(new_source_identities) != 1:
        raise ContinuationWaveError(
            "all five stages must use one new source commit/tree"
        )
    if segment_hosts != set(FORMAL_HOSTS) or smplx_hosts != set(
        FORMAL_HOSTS
    ):
        raise ContinuationWaveError(
            "wave must bind both formal hosts to the pinned SMPL-X bytes"
        )
    if len(predecessor_wave_payloads) != 1:
        raise ContinuationWaveError(
            "all five stages must bind one predecessor wave"
        )

    receipt: dict[str, Any] = {
        "format": FORMAT,
        "status": "authorized",
        "test_visible": False,
        "protocol": dict(PROTOCOL),
        "decision": binding,
        "trigger_stages": list(triggers),
        "boundary_epoch": boundary,
        "target_epoch": target,
        "stages": stage_entries,
    }
    receipt["receipt_payload_sha256"] = canonical_json_sha256(receipt)
    validate_wave_schema(receipt)
    return receipt


def _catalog_candidate_from_index(
    value: Any,
    *,
    stage: str,
    index: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "epoch",
        "optimizer_updates",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_audit_sha256",
    }:
        raise ContinuationWaveError(
            f"{stage} candidate index entry {index} schema mismatch"
        )
    return _validate_candidate(
        {
            "epoch": value["epoch"],
            "optimizer_updates": value["optimizer_updates"],
            "checkpoint": {
                "path": value["checkpoint"],
                "sha256": value["checkpoint_sha256"],
                "bytes": value["checkpoint_bytes"],
            },
            "checkpoint_audit_sha256": value[
                "checkpoint_audit_sha256"
            ],
        },
        stage=stage,
        label=f"{stage} candidate index entry {index}",
    )


def _read_file_binding(
    path_value: Any,
    expected_sha256: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    path_text = _absolute_path(path_value, f"{label} path")
    expected_sha = _require_sha256(
        expected_sha256,
        f"{label} SHA-256",
    )
    try:
        resolved, payload, observed = val_contract.read_verified_file(
            Path(path_text),
            expected_sha,
            label,
            val_only=False,
        )
    except (val_contract.ContractError, OSError) as error:
        raise ContinuationWaveError(str(error)) from error
    if str(resolved) != path_text or observed != expected_sha or not payload:
        raise ContinuationWaveError(f"{label} binding changed")
    return {
        "path": path_text,
        "sha256": expected_sha,
        "bytes": len(payload),
    }, payload


def _formal_status_artifact(value: Any, *, stage: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) not in (
        {"path", "sha256"},
        {"path", "sha256", "finite_evidence"},
    ):
        raise ContinuationWaveError(
            f"{stage} formal training status binding schema mismatch"
        )
    result = {
        "path": _absolute_path(
            value["path"],
            f"{stage} formal training status path",
        ),
        "sha256": _require_sha256(
            value["sha256"],
            f"{stage} formal training status SHA-256",
        ),
    }
    if "finite_evidence" in value:
        if not isinstance(value["finite_evidence"], dict):
            raise ContinuationWaveError(
                f"{stage} formal finite evidence is invalid"
            )
        result["finite_evidence"] = dict(value["finite_evidence"])
    return result


def _load_old_runtime_evidence(
    *,
    stage: str,
    boundary: int,
    old_run_path: str,
    old_source_receipt: Mapping[str, Any],
    old_config_sha256: str,
    old_dataset_receipt_sha256: str,
    formal_status_artifact: Mapping[str, Any],
    boundary_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Reprove the immutable old boundary and its complete train state."""

    old_run = _absolute_path(old_run_path, f"{stage} old run")
    status_artifact = _formal_status_artifact(
        formal_status_artifact,
        stage=stage,
    )
    status_binding, status_payload = _read_file_binding(
        status_artifact["path"],
        status_artifact["sha256"],
        label=f"{stage} formal training status",
    )
    if Path(status_binding["path"]).parent != Path(old_run):
        raise ContinuationWaveError(
            f"{stage} formal training status is outside old run"
        )
    try:
        status = val_contract.strict_json_bytes(
            status_payload,
            status_binding["path"],
        )
    except val_contract.ContractError as error:
        raise ContinuationWaveError(str(error)) from error
    stage_updates = updates_per_epoch(stage)
    expected_updates = boundary * stage_updates
    expected_world = _topology(stage)["world_size"]
    if (
        not isinstance(status, dict)
        or status.get("status") != "complete"
        or status.get("formal_stage") != stage
        or status.get("completed_epochs") != boundary
        or status.get("epochs") not in (None, boundary)
        or status.get("updates_per_epoch") != stage_updates
        or status.get("optimizer_updates") != expected_updates
        or status.get("world_size") not in (None, expected_world)
        or status.get("config_sha256") != old_config_sha256
    ):
        raise ContinuationWaveError(
            f"{stage} formal training status boundary mismatch"
        )
    old_host = _validate_host(
        status.get("hostname"),
        f"{stage} formal training status hostname",
    )
    try:
        source = val_contract.validate_training_audit_source(
            status.get("source_receipt"),
            f"{stage} formal status source",
            reprove_entrypoint=False,
        )
    except val_contract.ContractError as error:
        raise ContinuationWaveError(str(error)) from error
    portable = old_source_receipt.get("portable_identity")
    frozen_audit = old_source_receipt.get("training_audit")
    if (
        not isinstance(portable, dict)
        or not isinstance(frozen_audit, dict)
        or source != frozen_audit
        or {
            key: source.get(key)
            for key in ("origin", "commit", "tree")
        }
        != {
            "origin": val_contract.EXPECTED_ORIGIN,
            "commit": portable.get("commit"),
            "tree": portable.get("tree"),
        }
        or status.get("source_receipt_sha256")
        != val_contract.canonical_payload_sha256(source)
    ):
        raise ContinuationWaveError(
            f"{stage} formal status source differs from candidate index"
        )
    source_audit_sha = val_contract.canonical_payload_sha256(source)
    dataset_receipt = status.get("dataset_receipt")
    if (
        not isinstance(dataset_receipt, dict)
        or val_contract.canonical_payload_sha256(dataset_receipt)
        != old_dataset_receipt_sha256
    ):
        raise ContinuationWaveError(
            f"{stage} formal status dataset differs from candidate index"
        )
    old_smplx_asset = _validate_smplx_asset(
        dataset_receipt.get("smplx_asset"),
        stage=stage,
        host=old_host,
        label=f"{stage} formal status SMPL-X asset",
    )
    if status.get("smplx_asset_receipt") != old_smplx_asset:
        raise ContinuationWaveError(
            f"{stage} status and dataset bind different SMPL-X assets"
        )
    latest = status.get("latest_representation_candidate")
    checkpoint = boundary_candidate["checkpoint"]
    if latest != {
        "path": checkpoint["path"],
        "sha256": checkpoint["sha256"],
        "completed_epochs": boundary,
        "optimizer_updates": expected_updates,
        "selection_status": "offline_validation_pending",
    }:
        raise ContinuationWaveError(
            f"{stage} formal status latest candidate is not boundary"
        )

    resume_path = str(Path(old_run) / "latest_resume.pt")
    resume_binding, resume_payload = _read_file_binding(
        resume_path,
        status.get("latest_resume_sha256"),
        label=f"{stage} boundary resume",
    )
    final_binding, _ = _read_file_binding(
        status.get("final_checkpoint"),
        status.get("final_checkpoint_sha256"),
        label=f"{stage} final checkpoint",
    )
    if Path(final_binding["path"]).parent != Path(old_run):
        raise ContinuationWaveError(
            f"{stage} final checkpoint is outside old run"
        )

    finite_evidence = status_artifact.get("finite_evidence")
    if finite_evidence is not None:
        if (
            set(finite_evidence)
            != {
                "final_checkpoint",
                "latest_resume",
                "last_metrics",
                "all_candidate_model_tensors_finite",
            }
            or finite_evidence["all_candidate_model_tensors_finite"]
            is not True
            or not isinstance(finite_evidence["last_metrics"], dict)
            or {
                key: finite_evidence["latest_resume"].get(key)
                for key in CHECKPOINT_KEYS
            }
            != resume_binding
            or {
                key: finite_evidence["final_checkpoint"].get(key)
                for key in CHECKPOINT_KEYS
            }
            != final_binding
        ):
            raise ContinuationWaveError(
                f"{stage} formal finite evidence differs from live files"
            )

    try:
        import torch

        try:
            resume = torch.load(
                io.BytesIO(resume_payload),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            resume = torch.load(
                io.BytesIO(resume_payload),
                map_location="cpu",
            )
        state_proof = boundary_state.build_boundary_state_proof(
            resume,
            stage=stage,
            boundary_epoch=boundary,
            world_size=expected_world,
        )
    except (
        ImportError,
        RuntimeError,
        ValueError,
        boundary_state.BoundaryStateError,
    ) as error:
        raise ContinuationWaveError(
            f"{stage} boundary resume state proof failed: {error}"
        ) from error
    return {
        "old_host": old_host,
        "old_smplx_asset": old_smplx_asset,
        "old_source_audit_sha256": source_audit_sha,
        "old_dataset_receipt_sha256": old_dataset_receipt_sha256,
        "formal_status": status_binding,
        "boundary_resume": resume_binding,
        "final_checkpoint": final_binding,
        "boundary_state": state_proof,
    }


def _selected_candidate_from_bridge(
    value: Any,
    *,
    stage: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContinuationWaveError(
            f"{stage} selected bridge entry is missing"
        )
    checkpoint = value.get("candidate_checkpoint")
    return _validate_candidate(
        {
            "epoch": value.get("epoch"),
            "optimizer_updates": value.get("optimizer_updates"),
            "checkpoint": checkpoint,
            "checkpoint_audit_sha256": value.get(
                "candidate_audit_sha256"
            ),
        },
        stage=stage,
        label=f"{stage} selected bridge entry",
    )


def authorize_wave_from_replayed_inputs(
    *,
    decision_path: Path,
    expected_decision_sha256: str,
    stage_plans: Mapping[str, Any],
    _wave_stack: frozenset[Path] | None = None,
    _artifact_stack: frozenset[Path] | None = None,
) -> dict[str, Any]:
    """Adapt real replayed decision/bridge/index evidence into one wave.

    ``decision.replay_decision`` and
    ``selected_prerequisites.load_selected_prerequisites`` reprove the
    validation authority.  ``load_candidate_index`` then re-hashes every
    checkpoint.  Only after those three independent replays are complete are
    their values normalized into :func:`authorize_wave`.
    """

    wave_stack = frozenset() if _wave_stack is None else _wave_stack
    artifact_stack = (
        frozenset() if _artifact_stack is None else _artifact_stack
    )
    expected_decision_sha = _require_sha256(
        expected_decision_sha256,
        "continuation decision expected SHA-256",
    )
    try:
        replayed_decision = continuation_decision.replay_decision(
            decision_path,
            expected_decision_sha,
        )
    except continuation_decision.ContinuationDecisionError as error:
        raise ContinuationWaveError(
            f"continuation decision replay failed: {error}"
        ) from error
    try:
        resolved_decision = Path(decision_path).resolve(strict=True)
    except OSError as error:
        raise ContinuationWaveError(
            "continuation decision path disappeared after replay"
        ) from error
    selection_binding = replayed_decision.get("inputs", {}).get(
        "selection"
    )
    selection_binding = _validate_payload_binding(
        selection_binding,
        "replayed selection binding",
    )
    try:
        bridge = selected_contract.load_selected_prerequisites(
            Path(selection_binding["path"]),
            selection_binding["sha256"],
        )
    except selected_contract.SelectedPrerequisiteError as error:
        raise ContinuationWaveError(
            f"selected prerequisite replay failed: {error}"
        ) from error
    bridge_selection = _validate_payload_binding(
        bridge.get("selection"),
        "selected bridge selection",
    )
    if bridge_selection != selection_binding:
        raise ContinuationWaveError(
            "decision and selected bridge bind different selections"
        )

    candidate_index_binding = _validate_payload_binding(
        bridge.get("candidate_index_receipt"),
        "selected bridge candidate index",
    )
    try:
        candidate_index, candidate_index_artifact = (
            val_contract.load_candidate_index(
                Path(candidate_index_binding["path"]),
                candidate_index_binding["sha256"],
                _artifact_stack=artifact_stack,
            )
        )
    except (val_contract.ContractError, OSError) as error:
        raise ContinuationWaveError(
            f"candidate index replay failed: {error}"
        ) from error
    if (
        candidate_index_artifact.get("path")
        != candidate_index_binding["path"]
        or candidate_index_artifact.get("sha256")
        != candidate_index_binding["sha256"]
        or candidate_index.get("receipt_payload_sha256")
        != candidate_index_binding["receipt_payload_sha256"]
    ):
        raise ContinuationWaveError(
            "candidate index binding changed during replay"
        )

    selected_value = bridge.get("selected")
    indexed_stages = candidate_index.get("stages")
    source_receipts = candidate_index.get("source_receipts")
    config_sha256 = candidate_index.get("config_sha256")
    dataset_receipt_sha256 = candidate_index.get(
        "dataset_receipt_sha256"
    )
    formal_training_status = candidate_index.get(
        "formal_training_status"
    )
    if (
        not isinstance(selected_value, dict)
        or set(selected_value) != set(STAGES)
        or not isinstance(indexed_stages, dict)
        or set(indexed_stages) != set(STAGES)
        or not isinstance(source_receipts, dict)
        or set(source_receipts) != set(STAGES)
        or not isinstance(config_sha256, dict)
        or set(config_sha256) != set(STAGES)
        or not isinstance(dataset_receipt_sha256, dict)
        or set(dataset_receipt_sha256) != set(STAGES)
        or not isinstance(formal_training_status, dict)
        or set(formal_training_status) != set(STAGES)
        or not isinstance(stage_plans, Mapping)
        or set(stage_plans) != set(STAGES)
    ):
        raise ContinuationWaveError(
            "replayed wave inputs do not exactly cover five stages"
        )

    predecessor_values = [
        stage_plans[stage].get("predecessor_wave")
        if isinstance(stage_plans[stage], Mapping)
        else object()
        for stage in STAGES
    ]
    predecessor_wave: dict[str, Any] | None = None
    predecessor_receipt: dict[str, Any] | None = None
    if any(value is not None for value in predecessor_values):
        if any(value is None for value in predecessor_values):
            raise ContinuationWaveError(
                "all five stages must bind the same predecessor wave"
            )
        predecessor_wave = _validate_payload_binding(
            predecessor_values[0],
            "predecessor continuation wave",
        )
        if any(
            _validate_payload_binding(
                value,
                "stage predecessor continuation wave",
            )
            != predecessor_wave
            for value in predecessor_values[1:]
        ):
            raise ContinuationWaveError(
                "all five stages must bind the same predecessor wave"
            )
        try:
            predecessor_receipt = replay_wave_file(
                Path(predecessor_wave["path"]),
                predecessor_wave["sha256"],
                _wave_stack=wave_stack,
                _artifact_stack=artifact_stack,
            )
        except ContinuationWaveError as error:
            raise ContinuationWaveError(
                f"predecessor wave replay failed: {error}"
            ) from error
        if (
            predecessor_receipt["receipt_payload_sha256"]
            != predecessor_wave["receipt_payload_sha256"]
        ):
            raise ContinuationWaveError(
                "predecessor wave payload binding changed"
            )

    selected_by_stage: dict[str, Any] = {}
    candidate_catalog_by_stage: dict[str, Any] = {}
    normalized_plans: dict[str, Any] = {}
    boundary, _, _ = _validate_decision(replayed_decision)
    for stage in STAGES:
        entries = indexed_stages[stage]
        if not isinstance(entries, list):
            raise ContinuationWaveError(
                f"{stage} candidate index stage is not a list"
            )
        catalog = [
            _catalog_candidate_from_index(
                entry,
                stage=stage,
                index=index,
            )
            for index, entry in enumerate(entries)
        ]
        candidate_catalog_by_stage[stage] = catalog
        selected_by_stage[stage] = _selected_candidate_from_bridge(
            selected_value[stage],
            stage=stage,
        )
        raw_plan = _exact_keys(
            stage_plans[stage],
            ADAPTER_STAGE_PLAN_KEYS,
            f"{stage} adapter stage plan",
        )
        if predecessor_receipt is None:
            if raw_plan["predecessor_wave"] is not None:
                raise ContinuationWaveError(
                    f"{stage} unexpected predecessor wave"
                )
        else:
            if predecessor_receipt["target_epoch"] != boundary:
                raise ContinuationWaveError(
                    "predecessor wave target differs from decision boundary"
                )
            previous_matches = [
                entry
                for entry in predecessor_receipt["stages"]
                if entry["stage"] == stage
            ]
            if len(previous_matches) != 1:
                raise ContinuationWaveError(
                    f"{stage} predecessor wave stage is not exact-once"
                )
            previous_entry = previous_matches[0]
            previous_new = previous_entry["new_segment"]
            appended = _make_chain_segment(
                run_path=previous_new["run_path"],
                start_epoch=boundary,
                end_epoch=boundary,
                predecessor_segment_id=previous_new[
                    "predecessor_segment_id"
                ],
            )
            if (
                appended["segment_id"]
                != previous_new["chain_segment_id"]
                or raw_plan["candidate_segment_chain"]
                != (
                    previous_entry["old_segment"][
                        "candidate_segment_chain"
                    ]
                    + [appended]
                )
                or raw_plan["old_run_path"] != previous_new["run_path"]
                or raw_plan["predecessor_wave"] != predecessor_wave
            ):
                raise ContinuationWaveError(
                    f"{stage} recursive segment chain differs from "
                    "predecessor wave"
                )
        old_run = _absolute_path(
            raw_plan["old_run_path"],
            f"{stage} adapter old run",
        )
        candidate_segment_chain = _validate_segment_chain(
            raw_plan["candidate_segment_chain"],
            stage=stage,
            boundary=boundary,
        )
        for candidate in catalog:
            segment = _segment_for_epoch(
                candidate_segment_chain,
                candidate["epoch"],
                stage=stage,
            )
            if Path(candidate["checkpoint"]["path"]).parent != (
                Path(segment["run_path"]) / "representation_candidates"
            ):
                raise ContinuationWaveError(
                    f"{stage} candidate index differs from segment chain"
                )
        if candidate_segment_chain[-1]["run_path"] != old_run:
            raise ContinuationWaveError(
                f"{stage} old run differs from terminal segment"
            )
        source = source_receipts[stage]
        if not isinstance(source, dict):
            raise ContinuationWaveError(
                f"{stage} frozen source receipt is invalid"
            )
        portable = source.get("portable_identity")
        if not isinstance(portable, dict):
            raise ContinuationWaveError(
                f"{stage} portable source identity is unavailable"
            )
        # The legacy e20..e200 index predates segmented continuations and
        # identifies its frozen wrapper receipt.  A segmented union's terminal
        # source is exactly the preceding wave's ``new_source`` training-audit
        # receipt, so recursive e220->e240 replay must preserve that SHA rather
        # than silently substituting the wrapper hash created at union time.
        source_receipt_sha256 = source.get("receipt_payload_sha256")
        if "segmented_union" in candidate_index:
            training_audit = source.get("training_audit")
            if not isinstance(training_audit, dict):
                raise ContinuationWaveError(
                    f"{stage} segmented source training audit is unavailable"
                )
            source_receipt_sha256 = val_contract.canonical_payload_sha256(
                training_audit
            )
        expected_old_source = {
            "commit": portable.get("commit"),
            "tree": portable.get("tree"),
            "source_receipt_sha256": source_receipt_sha256,
        }
        expected_old_source = _validate_source(
            expected_old_source,
            f"{stage} indexed old source",
        )
        raw_old_source = _validate_source(
            raw_plan["old_source"],
            f"{stage} adapter old source",
        )
        if (
            raw_old_source != expected_old_source
            or raw_plan["old_config_sha256"] != config_sha256[stage]
        ):
            raise ContinuationWaveError(
                f"{stage} old source/config differs from candidate index"
            )
        indexed_dataset_sha = _require_sha256(
            dataset_receipt_sha256[stage],
            f"{stage} indexed dataset receipt SHA-256",
        )
        boundary_candidate = _candidate_at(
            catalog,
            boundary,
            stage=stage,
            label="adapter resume boundary",
        )
        runtime_evidence = _load_old_runtime_evidence(
            stage=stage,
            boundary=boundary,
            old_run_path=old_run,
            old_source_receipt=source,
            old_config_sha256=config_sha256[stage],
            old_dataset_receipt_sha256=indexed_dataset_sha,
            formal_status_artifact=formal_training_status[stage],
            boundary_candidate=boundary_candidate,
        )
        new_source = _validate_source(
            raw_plan["new_source"],
            f"{stage} adapter new source",
        )
        source_ancestry = prove_source_ancestry(
            Path(
                _absolute_path(
                    raw_plan["new_source_repository"],
                    f"{stage} adapter new source repository",
                )
            ),
            old_source=raw_old_source,
            new_source=new_source,
        )
        plan = _normalize_plan(
            {
                **{
                    key: value
                    for key, value in raw_plan.items()
                    if key != "new_source_repository"
                },
                "source_ancestry": source_ancestry,
                "candidate_catalog_receipt": dict(
                    candidate_index_binding
                ),
                **runtime_evidence,
            },
            stage=stage,
            boundary=boundary,
        )
        normalized_plans[stage] = {
            **plan,
        }

    decision_binding = {
        "path": str(resolved_decision),
        "sha256": expected_decision_sha,
        "receipt_payload_sha256": replayed_decision[
            "receipt_payload_sha256"
        ],
    }
    return authorize_wave(
        decision_receipt=replayed_decision,
        decision_binding=decision_binding,
        selected_by_stage=selected_by_stage,
        candidate_catalog_by_stage=candidate_catalog_by_stage,
        stage_plans=normalized_plans,
    )


def adapter_stage_plans_from_wave(
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Recover the exact semantic plan needed for a fresh adapter replay."""

    validate_wave_schema(receipt)
    result: dict[str, dict[str, Any]] = {}
    for stage_entry in receipt["stages"]:
        stage = stage_entry["stage"]
        old = stage_entry["old_segment"]
        new = stage_entry["new_segment"]
        result[stage] = {
            "stage": stage,
            "old_run_path": old["run_path"],
            "new_run_path": new["run_path"],
            "old_source": dict(old["source"]),
            "new_source": dict(new["source"]),
            "old_host": old["host"],
            "new_host": new["host"],
            "old_smplx_asset": copy.deepcopy(old["smplx_asset"]),
            "new_smplx_asset": copy.deepcopy(new["smplx_asset"]),
            "new_source_repository": new["source_ancestry"][
                "repository_path"
            ],
            "candidate_segment_chain": copy.deepcopy(
                old["candidate_segment_chain"]
            ),
            "predecessor_wave": copy.deepcopy(
                old["predecessor_wave"]
            ),
            "old_config_sha256": old["config_sha256"],
            "new_config_sha256": new["config_sha256"],
            "old_config_semantic_sha256": old[
                "config_semantic_sha256"
            ],
            "new_config_semantic_sha256": new[
                "config_semantic_sha256"
            ],
            "old_dataset_semantic_sha256": old[
                "dataset_semantic_sha256"
            ],
            "new_dataset_semantic_sha256": new[
                "dataset_semantic_sha256"
            ],
        }
    return result


def replay_wave_file(
    path: Path,
    expected_sha256: str,
    *,
    _wave_stack: frozenset[Path] | None = None,
    _artifact_stack: frozenset[Path] | None = None,
) -> dict[str, Any]:
    """Reopen a published wave and re-run decision/bridge/index authority."""

    expected_sha = _require_sha256(
        expected_sha256,
        "continuation wave expected SHA-256",
    )
    try:
        resolved, payload, observed_sha = val_contract.read_verified_file(
            path,
            expected_sha,
            "continuation wave",
            require_path_identity=True,
        )
        receipt = val_contract.strict_json_bytes(payload, str(resolved))
    except (val_contract.ContractError, OSError) as error:
        raise ContinuationWaveError(str(error)) from error
    if observed_sha != expected_sha:
        raise ContinuationWaveError(
            "continuation wave file SHA-256 mismatch"
        )
    wave_stack = frozenset() if _wave_stack is None else _wave_stack
    if resolved in wave_stack:
        raise ContinuationWaveError("continuation wave cycle detected")
    wave_stack = wave_stack | {resolved}
    validate_wave_schema(receipt)
    expected = authorize_wave_from_replayed_inputs(
        decision_path=Path(receipt["decision"]["path"]),
        expected_decision_sha256=receipt["decision"]["sha256"],
        stage_plans=adapter_stage_plans_from_wave(receipt),
        _wave_stack=wave_stack,
        _artifact_stack=_artifact_stack,
    )
    if canonical_json_bytes(receipt) != canonical_json_bytes(expected):
        raise ContinuationWaveError(
            "published continuation wave differs from fresh replay"
        )
    return dict(receipt)


def _validate_old_segment(
    value: Any,
    *,
    stage: str,
    boundary: int,
) -> dict[str, Any]:
    segment = _exact_keys(
        value,
        OLD_SEGMENT_KEYS,
        f"{stage} old segment",
    )
    old_run = _absolute_path(
        segment["run_path"],
        f"{stage} old segment run",
    )
    if (
        _require_int(
            segment["terminal_epoch"],
            f"{stage} old segment terminal epoch",
        )
        != boundary
    ):
        raise ContinuationWaveError(
            f"{stage} old segment terminal epoch mismatch"
        )
    _validate_source(segment["source"], f"{stage} old segment source")
    host = _validate_host(
        segment["host"],
        f"{stage} old segment host",
    )
    smplx_asset = _validate_smplx_asset(
        segment["smplx_asset"],
        stage=stage,
        host=host,
        label=f"{stage} old segment SMPL-X asset",
    )
    _require_sha256(
        segment["config_sha256"],
        f"{stage} old segment config SHA-256",
    )
    _require_sha256(
        segment["source_audit_sha256"],
        f"{stage} old segment source audit SHA-256",
    )
    _require_sha256(
        segment["dataset_receipt_sha256"],
        f"{stage} old segment dataset receipt SHA-256",
    )
    _require_sha256(
        segment["config_semantic_sha256"],
        f"{stage} old segment config semantic SHA-256",
    )
    _require_sha256(
        segment["dataset_semantic_sha256"],
        f"{stage} old segment dataset semantic SHA-256",
    )
    _validate_payload_binding(
        segment["candidate_catalog_receipt"],
        f"{stage} old segment catalog receipt",
    )
    _require_sha256(
        segment["candidate_inventory_sha256"],
        f"{stage} old segment candidate inventory SHA-256",
    )
    candidate_segment_chain = _validate_segment_chain(
        segment["candidate_segment_chain"],
        stage=stage,
        boundary=boundary,
    )
    predecessor_wave = (
        None
        if segment["predecessor_wave"] is None
        else _validate_payload_binding(
            segment["predecessor_wave"],
            f"{stage} old segment predecessor wave",
        )
    )
    if (
        candidate_segment_chain[-1]["run_path"] != old_run
        or (len(candidate_segment_chain) == 1)
        != (predecessor_wave is None)
    ):
        raise ContinuationWaveError(
            f"{stage} old segment chain/predecessor mismatch"
        )
    runtime_bindings = {
        key: _validate_checkpoint(
            segment[key],
            f"{stage} old segment {key.replace('_', ' ')}",
        )
        for key in (
            "formal_status",
            "boundary_resume",
            "final_checkpoint",
        )
    }
    for key, binding in runtime_bindings.items():
        if not _is_within(binding["path"], old_run):
            raise ContinuationWaveError(
                f"{stage} old segment {key.replace('_', ' ')} "
                "is outside old run"
            )
    try:
        state_proof = boundary_state.validate_boundary_state_proof(
            segment["boundary_state"]
        )
    except boundary_state.BoundaryStateError as error:
        raise ContinuationWaveError(
            f"{stage} old segment boundary state is invalid: {error}"
        ) from error
    if (
        state_proof["stage"] != stage
        or state_proof["boundary_epoch"] != boundary
    ):
        raise ContinuationWaveError(
            f"{stage} old segment boundary state mismatch"
        )
    segment_id = _require_sha256(
        segment["segment_id"],
        f"{stage} old segment ID",
    )
    if segment_id != _segment_id(segment, id_key="segment_id"):
        raise ContinuationWaveError(f"{stage} old segment ID mismatch")
    return {
        **dict(segment),
        "run_path": old_run,
        "host": host,
        "smplx_asset": smplx_asset,
        "candidate_segment_chain": candidate_segment_chain,
        "predecessor_wave": predecessor_wave,
        **runtime_bindings,
        "boundary_state": state_proof,
    }


def _validate_new_segment(
    value: Any,
    *,
    stage: str,
    target: int,
    predecessor_segment_id: str,
    predecessor_source: Mapping[str, Any],
) -> dict[str, Any]:
    segment = _exact_keys(
        value,
        NEW_SEGMENT_KEYS,
        f"{stage} new segment",
    )
    new_run = _absolute_path(
        segment["run_path"],
        f"{stage} new segment run",
    )
    if (
        _require_int(
            segment["target_epoch"],
            f"{stage} new segment target epoch",
        )
        != target
        or segment["authorized_candidate_epochs"] != [target]
        or segment["predecessor_segment_id"] != predecessor_segment_id
    ):
        raise ContinuationWaveError(
            f"{stage} new segment boundary protocol mismatch"
        )
    source = _validate_source(
        segment["source"],
        f"{stage} new segment source",
    )
    source_ancestry = _validate_source_ancestry(
        segment["source_ancestry"],
        old_source=predecessor_source,
        new_source=source,
        label=f"{stage} new segment source ancestry",
    )
    host = _validate_host(
        segment["host"],
        f"{stage} new segment host",
    )
    smplx_asset = _validate_smplx_asset(
        segment["smplx_asset"],
        stage=stage,
        host=host,
        label=f"{stage} new segment SMPL-X asset",
    )
    _require_sha256(
        segment["config_sha256"],
        f"{stage} new segment config SHA-256",
    )
    _require_sha256(
        segment["config_semantic_sha256"],
        f"{stage} new segment config semantic SHA-256",
    )
    _require_sha256(
        segment["dataset_semantic_sha256"],
        f"{stage} new segment dataset semantic SHA-256",
    )
    segment_id = _require_sha256(
        segment["segment_id"],
        f"{stage} new segment ID",
    )
    if segment_id != _segment_id(segment, id_key="segment_id"):
        raise ContinuationWaveError(f"{stage} new segment ID mismatch")
    expected_chain_segment = _make_chain_segment(
        run_path=new_run,
        start_epoch=target,
        end_epoch=target,
        predecessor_segment_id=predecessor_segment_id,
    )
    if (
        _require_sha256(
            segment["chain_segment_id"],
            f"{stage} new segment chain ID",
        )
        != expected_chain_segment["segment_id"]
    ):
        raise ContinuationWaveError(
            f"{stage} new segment chain ID mismatch"
        )
    return {
        **dict(segment),
        "run_path": new_run,
        "source": source,
        "source_ancestry": source_ancestry,
        "host": host,
        "smplx_asset": smplx_asset,
    }


def validate_wave_schema(value: Any) -> dict[str, Any]:
    """Validate the closed wave schema and all internal invariants."""

    receipt = _exact_keys(value, TOP_KEYS, "continuation wave")
    if (
        receipt["format"] != FORMAT
        or receipt["status"] != "authorized"
        or receipt["test_visible"] is not False
        or receipt["protocol"] != PROTOCOL
    ):
        raise ContinuationWaveError(
            "continuation wave protocol mismatch"
        )
    _validate_payload_binding(
        receipt["decision"],
        "continuation wave decision",
    )
    boundary = _require_int(
        receipt["boundary_epoch"],
        "continuation wave boundary epoch",
    )
    target = _require_int(
        receipt["target_epoch"],
        "continuation wave target epoch",
    )
    if (
        boundary < MINIMUM_BOUNDARY_EPOCH
        or boundary % INTERVAL_EPOCHS != 0
        or target != boundary + INTERVAL_EPOCHS
    ):
        raise ContinuationWaveError(
            "continuation wave is not exact +20"
        )
    triggers = receipt["trigger_stages"]
    if (
        not isinstance(triggers, list)
        or not triggers
        or len(triggers) != len(set(triggers))
        or triggers
        != [stage for stage in STAGES if stage in set(triggers)]
    ):
        raise ContinuationWaveError(
            "continuation wave trigger inventory mismatch"
        )

    stage_values = receipt["stages"]
    if not isinstance(stage_values, list) or len(stage_values) != len(
        STAGES
    ):
        raise ContinuationWaveError(
            "continuation wave must cover ordered five stages"
        )
    old_runs: set[str] = set()
    new_runs: set[str] = set()
    new_source_identities: set[tuple[str, str]] = set()
    segment_hosts: set[str] = set()
    smplx_hosts: set[str] = set()
    predecessor_wave_payloads: set[str | None] = set()
    for expected_stage, stage_value in zip(STAGES, stage_values):
        stage = _exact_keys(
            stage_value,
            STAGE_KEYS,
            f"{expected_stage} continuation wave stage",
        )
        if (
            stage["stage"] != expected_stage
            or _require_bool(
                stage["triggered"],
                f"{expected_stage} triggered",
            )
            != (expected_stage in triggers)
            or stage["selection_metric"]
            != val_contract.SELECTION_METRICS[expected_stage]
        ):
            raise ContinuationWaveError(
                f"{expected_stage} continuation wave binding mismatch"
            )
        _validate_topology(
            stage["training_topology"],
            stage=expected_stage,
            label=f"{expected_stage} training topology",
        )
        selected = _validate_candidate(
            stage["independent_selected_candidate"],
            stage=expected_stage,
            label=f"{expected_stage} selected candidate",
        )
        resume = _validate_candidate(
            stage["resume_boundary_candidate"],
            stage=expected_stage,
            label=f"{expected_stage} resume candidate",
        )
        if selected["epoch"] > boundary or resume["epoch"] != boundary:
            raise ContinuationWaveError(
                f"{expected_stage} selected/resume boundary mismatch"
            )
        old = _validate_old_segment(
            stage["old_segment"],
            stage=expected_stage,
            boundary=boundary,
        )
        new = _validate_new_segment(
            stage["new_segment"],
            stage=expected_stage,
            target=target,
            predecessor_segment_id=old["candidate_segment_chain"][-1][
                "segment_id"
            ],
            predecessor_source=old["source"],
        )
        if (
            old["config_semantic_sha256"]
            != new["config_semantic_sha256"]
            or old["dataset_semantic_sha256"]
            != new["dataset_semantic_sha256"]
        ):
            raise ContinuationWaveError(
                f"{expected_stage} training semantics changed"
            )
        segment_hosts.update((old["host"], new["host"]))
        if old["smplx_asset"] is not None:
            smplx_hosts.add(old["host"])
        if new["smplx_asset"] is not None:
            smplx_hosts.add(new["host"])
        predecessor_wave_payloads.add(
            None
            if old["predecessor_wave"] is None
            else canonical_json_sha256(old["predecessor_wave"])
        )
        new_source_identities.add(
            (
                new["source"]["commit"],
                new["source"]["tree"],
            )
        )
        if (
            old["run_path"] == new["run_path"]
            or _is_within(old["run_path"], new["run_path"])
            or _is_within(new["run_path"], old["run_path"])
            or old["run_path"] in old_runs
            or new["run_path"] in new_runs
        ):
            raise ContinuationWaveError(
                f"{expected_stage} segment path reuse detected"
            )
        old_runs.add(old["run_path"])
        new_runs.add(new["run_path"])
        for label, candidate in (("selected", selected), ("resume", resume)):
            candidate_segment = _segment_for_epoch(
                old["candidate_segment_chain"],
                candidate["epoch"],
                stage=expected_stage,
            )
            expected_parent = (
                Path(candidate_segment["run_path"])
                / "representation_candidates"
            )
            if not _is_within(
                candidate["checkpoint"]["path"],
                str(expected_parent),
            ):
                raise ContinuationWaveError(
                    f"{expected_stage} {label} checkpoint escaped old segment"
                )
    _require_disjoint_run_paths(old_runs, new_runs)
    if len(new_source_identities) != 1:
        raise ContinuationWaveError(
            "all five stages must use one new source commit/tree"
        )
    if segment_hosts != set(FORMAL_HOSTS) or smplx_hosts != set(
        FORMAL_HOSTS
    ):
        raise ContinuationWaveError(
            "wave must bind both formal hosts to the pinned SMPL-X bytes"
        )
    if len(predecessor_wave_payloads) != 1:
        raise ContinuationWaveError(
            "all five stages must bind one predecessor wave"
        )

    payload_sha = _require_sha256(
        receipt["receipt_payload_sha256"],
        "continuation wave payload SHA-256",
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_payload_sha256")
    if canonical_json_sha256(unsigned) != payload_sha:
        raise ContinuationWaveError(
            "continuation wave payload SHA-256 mismatch"
        )
    return receipt


def replay_wave_receipt(
    receipt: Mapping[str, Any],
    *,
    decision_receipt: Mapping[str, Any],
    decision_binding: Mapping[str, Any],
    selected_by_stage: Mapping[str, Any],
    candidate_catalog_by_stage: Mapping[str, Any],
    stage_plans: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a receipt equals a fresh deterministic rebuild."""

    validate_wave_schema(receipt)
    expected = authorize_wave(
        decision_receipt=decision_receipt,
        decision_binding=decision_binding,
        selected_by_stage=selected_by_stage,
        candidate_catalog_by_stage=candidate_catalog_by_stage,
        stage_plans=stage_plans,
    )
    if canonical_json_bytes(receipt) != canonical_json_bytes(expected):
        raise ContinuationWaveError(
            "continuation wave differs from fresh replay"
        )
    return dict(receipt)
