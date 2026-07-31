#!/usr/bin/env python3
"""No-copy runtime authorization for one prerequisite continuation stage.

The wave receipt is the authority.  The old boundary resume and boundary
candidate remain in their immutable old segment.  A fresh continuation run
must start empty and may create only the single target candidate authorized
for its new segment.
"""

from __future__ import annotations

import hashlib
import io
import copy
from pathlib import Path
import stat
from typing import Any, Mapping

from scripts.show_base import prerequisite_boundary_state as boundary_state
from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as val_contract


FORMAT = "semtalk_show_prerequisite_continuation_runtime_v2"
OLD_SNAPSHOT_FORMAT = "semtalk_show_old_segment_snapshot_v1"
OLD_BEFORE_AFTER_FORMAT = (
    "semtalk_show_old_segment_read_only_before_after_v1"
)
CONFIG_RECEIPT_FORMAT = "semtalk_show_formal_config_preflight_v2"
STAGES = wave.STAGES
RUNTIME_KEYS = {
    "format",
    "stage",
    "boundary_epoch",
    "target_epoch",
    "cap_epoch",
    "wave",
    "old_run_path",
    "new_run_path",
    "old_host",
    "new_host",
    "old_smplx_asset",
    "new_smplx_asset",
    "old_config_sha256",
    "old_source_audit_sha256",
    "old_dataset_receipt_sha256",
    "resume_boundary_candidate",
    "boundary_resume",
    "boundary_state",
    "old_segment_snapshot",
    "new_source",
    "new_config_sha256",
    "new_config_semantic_sha256",
    "new_dataset_semantic_sha256",
    "world_size",
    "receipt_payload_sha256",
}
OLD_SNAPSHOT_KEYS = {
    "format",
    "stage",
    "bindings",
    "receipt_payload_sha256",
}
OLD_SNAPSHOT_LABELS = frozenset(
    {
        "candidate_catalog_receipt",
        "formal_status",
        "boundary_resume",
        "final_checkpoint",
        "selected_candidate",
        "resume_candidate",
    }
)
OLD_BEFORE_AFTER_KEYS = {
    "format",
    "stage",
    "before",
    "after",
    "unchanged",
    "receipt_payload_sha256",
}


class ContinuationRuntimeError(RuntimeError):
    """Raised when a process is not an exact no-copy continuation."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def config_semantic_sha256(snapshot: Any) -> str:
    """Hash all config fields except the three authorized segment labels."""

    if not isinstance(snapshot, dict):
        raise ContinuationRuntimeError(
            "formal config snapshot must be an object"
        )
    ignored = {"epochs", "run_name", "final_ckpt_name"}
    missing = ignored - set(snapshot)
    if missing:
        raise ContinuationRuntimeError(
            f"formal config snapshot lacks semantic fields: {sorted(missing)}"
        )
    return wave.canonical_json_sha256(
        {
            key: value
            for key, value in snapshot.items()
            if key not in ignored
        }
    )


def _source_neutral_nested_receipt(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or "source_binding" not in value
        or "receipt_sha256" not in value
    ):
        raise ContinuationRuntimeError(
            f"{label} lacks its explicit source/self-hash fields"
        )
    result = copy.deepcopy(value)
    result.pop("source_binding")
    result.pop("receipt_sha256")
    return result


def dataset_semantic_projection(value: Any) -> dict[str, Any]:
    """Project only enumerated source-bound fields from a dataset receipt."""

    if not isinstance(value, dict) or "source_binding" not in value:
        raise ContinuationRuntimeError(
            "dataset receipt lacks top-level source_binding"
        )
    result = copy.deepcopy(value)
    result.pop("source_binding")
    if "lower_target_joints_cache" in result:
        raise ContinuationRuntimeError(
            "prerequisite continuation forbids lower target cache"
        )
    for key in ("smplx_training_pool_gate", "lower_target_backend"):
        if key in result:
            result[key] = _source_neutral_nested_receipt(
                result[key],
                label=key,
            )
    return result


def dataset_semantic_sha256(value: Any) -> str:
    return wave.canonical_json_sha256(
        dataset_semantic_projection(value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuationRuntimeError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContinuationRuntimeError(f"{label} must be an exact integer")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContinuationRuntimeError(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ContinuationRuntimeError(
            f"{label} must be an absolute normalized path"
        )
    return path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _regular_file_bytes(
    binding: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, bytes]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise ContinuationRuntimeError(f"{label} schema mismatch")
    path = _absolute_path(binding["path"], f"{label} path")
    expected_sha = _require_sha256(
        binding["sha256"],
        f"{label} SHA-256",
    )
    expected_bytes = _require_int(binding["bytes"], f"{label} bytes")
    if expected_bytes <= 0:
        raise ContinuationRuntimeError(f"{label} bytes must be positive")
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ContinuationRuntimeError(
                f"{label} must be a regular non-symlink file"
            )
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
    except FileNotFoundError:
        raise ContinuationRuntimeError(f"{label} is missing") from None
    if (
        resolved != path
        or len(payload) != expected_bytes
        or _sha256(payload) != expected_sha
    ):
        raise ContinuationRuntimeError(f"{label} binding changed")
    return path, payload


def _binding_from_path_sha(
    path_value: Any,
    sha256_value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    path = _absolute_path(path_value, f"{label} path")
    expected_sha = _require_sha256(
        sha256_value,
        f"{label} SHA-256",
    )
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ContinuationRuntimeError(
                f"{label} must be a regular non-symlink file"
            )
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
    except FileNotFoundError:
        raise ContinuationRuntimeError(f"{label} is missing") from None
    if resolved != path or _sha256(payload) != expected_sha:
        raise ContinuationRuntimeError(f"{label} binding changed")
    return {
        "path": str(path),
        "sha256": expected_sha,
        "bytes": len(payload),
    }


def validate_old_segment_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != OLD_SNAPSHOT_KEYS:
        raise ContinuationRuntimeError(
            "old segment snapshot schema mismatch"
        )
    if value["format"] != OLD_SNAPSHOT_FORMAT or value["stage"] not in STAGES:
        raise ContinuationRuntimeError(
            "old segment snapshot protocol mismatch"
        )
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != set(
        OLD_SNAPSHOT_LABELS
    ):
        raise ContinuationRuntimeError(
            "old segment snapshot binding coverage mismatch"
        )
    for label, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ContinuationRuntimeError(
                f"old segment snapshot {label} schema mismatch"
            )
        _absolute_path(binding["path"], f"old snapshot {label} path")
        _require_sha256(
            binding["sha256"],
            f"old snapshot {label} SHA-256",
        )
        if _require_int(
            binding["bytes"],
            f"old snapshot {label} bytes",
        ) <= 0:
            raise ContinuationRuntimeError(
                f"old snapshot {label} bytes must be positive"
            )
    claimed = _require_sha256(
        value["receipt_payload_sha256"],
        "old segment snapshot payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256")
    if wave.canonical_json_sha256(unsigned) != claimed:
        raise ContinuationRuntimeError(
            "old segment snapshot payload mismatch"
        )
    return copy.deepcopy(value)


def _build_old_segment_snapshot(
    entry: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    old = entry["old_segment"]
    bindings: dict[str, dict[str, Any]] = {}
    source_bindings = {
        "candidate_catalog_receipt": old["candidate_catalog_receipt"],
        "formal_status": old["formal_status"],
        "boundary_resume": old["boundary_resume"],
        "final_checkpoint": old["final_checkpoint"],
        "selected_candidate": entry["independent_selected_candidate"][
            "checkpoint"
        ],
        "resume_candidate": entry["resume_boundary_candidate"][
            "checkpoint"
        ],
    }
    for label, binding in source_bindings.items():
        bindings[label] = _binding_from_path_sha(
            binding["path"],
            binding["sha256"],
            label=f"{stage} old {label}",
        )
        if "bytes" in binding and bindings[label]["bytes"] != binding["bytes"]:
            raise ContinuationRuntimeError(
                f"{stage} old {label} byte count changed"
            )
    snapshot: dict[str, Any] = {
        "format": OLD_SNAPSHOT_FORMAT,
        "stage": stage,
        "bindings": bindings,
    }
    snapshot["receipt_payload_sha256"] = wave.canonical_json_sha256(
        snapshot
    )
    return validate_old_segment_snapshot(snapshot)


def verify_old_segment_unchanged(
    runtime_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_value = validate_runtime_receipt(runtime_receipt)
    before = validate_old_segment_snapshot(
        runtime_value["old_segment_snapshot"]
    )
    after_bindings: dict[str, dict[str, Any]] = {}
    for label, binding in before["bindings"].items():
        _, payload = _regular_file_bytes(
            binding,
            label=f"{before['stage']} old {label} postflight",
        )
        after_bindings[label] = {
            "path": binding["path"],
            "sha256": _sha256(payload),
            "bytes": len(payload),
        }
    after: dict[str, Any] = {
        "format": OLD_SNAPSHOT_FORMAT,
        "stage": before["stage"],
        "bindings": after_bindings,
    }
    after["receipt_payload_sha256"] = wave.canonical_json_sha256(after)
    validate_old_segment_snapshot(after)
    if after != before:
        raise ContinuationRuntimeError(
            "old segment changed between continuation pre/postflight"
        )
    proof: dict[str, Any] = {
        "format": OLD_BEFORE_AFTER_FORMAT,
        "stage": before["stage"],
        "before": before,
        "after": after,
        "unchanged": True,
    }
    proof["receipt_payload_sha256"] = wave.canonical_json_sha256(proof)
    return proof


def _verify_current_smplx_asset(
    value: Any,
    *,
    stage: str,
    host: str,
) -> dict[str, Any] | None:
    try:
        asset = wave._validate_smplx_asset(
            value,
            stage=stage,
            host=host,
            label=f"{stage} current SMPL-X asset",
        )
    except wave.ContinuationWaveError as error:
        raise ContinuationRuntimeError(str(error)) from error
    if asset is None:
        return None
    path = _absolute_path(asset["path"], f"{stage} current SMPL-X path")
    try:
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(
            file_stat.st_mode
        ):
            raise ContinuationRuntimeError(
                f"{stage} current SMPL-X asset is not a regular file"
            )
        resolved = path.resolve(strict=True)
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        raise ContinuationRuntimeError(
            f"{stage} current SMPL-X asset is missing"
        ) from None
    if (
        resolved != path
        or file_stat.st_size != asset["bytes"]
        or digest.hexdigest() != asset["sha256"]
    ):
        raise ContinuationRuntimeError(
            f"{stage} current SMPL-X asset binding changed"
        )
    return asset


def _load_resume_proof(
    payload: bytes,
    *,
    stage: str,
    boundary_epoch: int,
    world_size: int,
) -> dict[str, Any]:
    try:
        import torch

        try:
            resume = torch.load(
                io.BytesIO(payload),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            resume = torch.load(io.BytesIO(payload), map_location="cpu")
        return boundary_state.build_boundary_state_proof(
            resume,
            stage=stage,
            boundary_epoch=boundary_epoch,
            world_size=world_size,
        )
    except (
        ImportError,
        RuntimeError,
        ValueError,
        boundary_state.BoundaryStateError,
    ) as error:
        raise ContinuationRuntimeError(
            f"{stage} boundary resume proof failed: {error}"
        ) from error


def verify_new_segment_preflight(
    new_run: Path,
    *,
    stage: str,
    boundary_epoch: int,
    target_epoch: int,
) -> Path:
    """Reject copied prefix artifacts or a reused/partially trained new run."""

    if stage not in STAGES:
        raise ContinuationRuntimeError(f"unknown stage {stage!r}")
    if (
        boundary_epoch < 200
        or boundary_epoch % wave.INTERVAL_EPOCHS
        or target_epoch != boundary_epoch + wave.INTERVAL_EPOCHS
    ):
        raise ContinuationRuntimeError("new segment boundary mismatch")
    try:
        mode = new_run.lstat().st_mode
    except FileNotFoundError:
        new_run.mkdir(parents=True, exist_ok=False)
        mode = new_run.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ContinuationRuntimeError(
            "new segment must be a real directory"
        )
    resolved = new_run.resolve(strict=True)
    if resolved != new_run:
        raise ContinuationRuntimeError(
            "new segment path changed during resolution"
        )
    forbidden_exact = {
        "latest_resume.pt",
        "formal_training_status.json",
        "base_candidate_manifest.json",
    }
    candidate_dir = resolved / "representation_candidates"
    if candidate_dir.exists() or candidate_dir.is_symlink():
        mode = candidate_dir.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ContinuationRuntimeError(
                "new representation candidate path is invalid"
            )
        if any(candidate_dir.iterdir()):
            raise ContinuationRuntimeError(
                "new segment already contains representation candidates"
            )
    for path in resolved.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ContinuationRuntimeError(
                "new segment contains a symbolic link"
            )
        if not stat.S_ISREG(mode):
            continue
        if path.name in forbidden_exact:
            raise ContinuationRuntimeError(
                f"new segment already contains {path.name}"
            )
        normalized = path.name.casefold()
        for epoch in range(
            wave.INTERVAL_EPOCHS,
            boundary_epoch + 1,
            wave.INTERVAL_EPOCHS,
        ):
            if (
                f"epoch_{epoch:04d}" in normalized
                or f"epoch-{epoch:04d}" in normalized
                or f"_e{epoch}_" in normalized
            ):
                raise ContinuationRuntimeError(
                    "new segment contains a copied prefix candidate"
                )
        if (
            f"epoch_{target_epoch:04d}" in normalized
            or f"epoch-{target_epoch:04d}" in normalized
            or f"_e{target_epoch}_" in normalized
        ):
            raise ContinuationRuntimeError(
                "new segment target candidate already exists"
            )
    return resolved


def verify_runtime_wave_stage(
    *,
    wave_path: Path,
    expected_wave_sha256: str,
    stage: str,
    target_epoch: int,
    current_new_run: Path,
    resume_path: Path,
    current_source: Mapping[str, Any],
    current_host: str,
    current_smplx_asset: Mapping[str, Any] | None,
    current_config_sha256: str,
    current_config_semantic_sha256: str,
    current_dataset_semantic_sha256: str,
    world_size: int,
) -> dict[str, Any]:
    """Return an exact runtime authorization after fresh full replay."""

    if stage not in STAGES:
        raise ContinuationRuntimeError(f"unknown stage {stage!r}")
    try:
        receipt = wave.replay_wave_file(
            wave_path,
            expected_wave_sha256,
        )
    except wave.ContinuationWaveError as error:
        raise ContinuationRuntimeError(
            f"continuation wave replay failed: {error}"
        ) from error
    matches = [
        value for value in receipt["stages"] if value["stage"] == stage
    ]
    if len(matches) != 1:
        raise ContinuationRuntimeError(
            f"{stage} is not exact-once in continuation wave"
        )
    entry = matches[0]
    old = entry["old_segment"]
    new = entry["new_segment"]
    boundary = entry.get("boundary_epoch", receipt.get("boundary_epoch"))
    boundary = _require_int(boundary, "boundary epoch")
    cap = entry.get(
        "cap_epoch",
        wave.continuation_decision.STAGE_CAP_EPOCHS[stage],
    )
    cap = _require_int(cap, "cap epoch")
    target = _require_int(target_epoch, "target epoch")
    world = _require_int(world_size, "world size")
    new_run = _absolute_path(
        str(current_new_run),
        "current new run",
    )
    resume = _absolute_path(str(resume_path), "resume path")
    try:
        host = wave._validate_host(
            current_host,
            f"{stage} current host",
        )
    except wave.ContinuationWaveError as error:
        raise ContinuationRuntimeError(str(error)) from error
    asset = _verify_current_smplx_asset(
        current_smplx_asset,
        stage=stage,
        host=host,
    )
    if (
        target != entry.get("target_epoch", receipt.get("target_epoch"))
        or new["target_epoch"] != target
        or target > cap
        or cap != wave.continuation_decision.STAGE_CAP_EPOCHS[stage]
        or new["authorized_candidate_epochs"] != [target]
        or new_run != Path(new["run_path"])
        or resume != Path(old["boundary_resume"]["path"])
        or _is_within(resume, new_run)
        or world != entry["training_topology"]["world_size"]
        or host != new["host"]
        or asset != new["smplx_asset"]
        or dict(current_source) != new["source"]
        or current_config_sha256 != new["config_sha256"]
        or current_config_semantic_sha256
        != new["config_semantic_sha256"]
        or current_dataset_semantic_sha256
        != new["dataset_semantic_sha256"]
    ):
        raise ContinuationRuntimeError(
            f"{stage} runtime differs from continuation authorization"
        )
    _require_sha256(
        current_config_sha256,
        "current config SHA-256",
    )
    _require_sha256(
        current_config_semantic_sha256,
        "current config semantic SHA-256",
    )
    _require_sha256(
        current_dataset_semantic_sha256,
        "current dataset semantic SHA-256",
    )
    _, resume_payload = _regular_file_bytes(
        old["boundary_resume"],
        label=f"{stage} boundary resume",
    )
    observed_proof = _load_resume_proof(
        resume_payload,
        stage=stage,
        boundary_epoch=boundary,
        world_size=world,
    )
    if observed_proof != old["boundary_state"]:
        raise ContinuationRuntimeError(
            f"{stage} boundary state differs from wave receipt"
        )
    old_segment_snapshot = _build_old_segment_snapshot(
        entry,
        stage=stage,
    )
    verify_new_segment_preflight(
        new_run,
        stage=stage,
        boundary_epoch=boundary,
        target_epoch=target,
    )
    expected_wave_sha = _require_sha256(
        expected_wave_sha256,
        "wave SHA-256",
    )
    wave_resolved = _absolute_path(str(wave_path), "wave path")
    result: dict[str, Any] = {
        "format": FORMAT,
        "stage": stage,
        "boundary_epoch": boundary,
        "target_epoch": target,
        "cap_epoch": cap,
        "wave": {
            "path": str(wave_resolved),
            "sha256": expected_wave_sha,
            "receipt_payload_sha256": receipt[
                "receipt_payload_sha256"
            ],
        },
        "old_run_path": old["run_path"],
        "new_run_path": new["run_path"],
        "old_host": old["host"],
        "new_host": new["host"],
        "old_smplx_asset": copy.deepcopy(old["smplx_asset"]),
        "new_smplx_asset": copy.deepcopy(new["smplx_asset"]),
        "old_config_sha256": old["config_sha256"],
        "old_source_audit_sha256": old["source_audit_sha256"],
        "old_dataset_receipt_sha256": old[
            "dataset_receipt_sha256"
        ],
        "resume_boundary_candidate": dict(
            entry["resume_boundary_candidate"]
        ),
        "boundary_resume": dict(old["boundary_resume"]),
        "boundary_state": dict(old["boundary_state"]),
        "old_segment_snapshot": old_segment_snapshot,
        "new_source": dict(new["source"]),
        "new_config_sha256": new["config_sha256"],
        "new_config_semantic_sha256": new[
            "config_semantic_sha256"
        ],
        "new_dataset_semantic_sha256": new[
            "dataset_semantic_sha256"
        ],
        "world_size": world,
    }
    result["receipt_payload_sha256"] = wave.canonical_json_sha256(
        result
    )
    validate_runtime_receipt(result)
    return result


def validate_runtime_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUNTIME_KEYS:
        raise ContinuationRuntimeError(
            "continuation runtime receipt schema mismatch"
        )
    if value["format"] != FORMAT or value["stage"] not in STAGES:
        raise ContinuationRuntimeError(
            "continuation runtime receipt protocol mismatch"
        )
    boundary = _require_int(value["boundary_epoch"], "boundary epoch")
    target = _require_int(value["target_epoch"], "target epoch")
    cap = _require_int(value["cap_epoch"], "cap epoch")
    world = _require_int(value["world_size"], "world size")
    if (
        boundary < 200
        or boundary % wave.INTERVAL_EPOCHS
        or target != boundary + wave.INTERVAL_EPOCHS
        or target > cap
        or cap
        != wave.continuation_decision.STAGE_CAP_EPOCHS[value["stage"]]
        or world != (4 if value["stage"] in wave.RVQ_STAGES else 1)
    ):
        raise ContinuationRuntimeError(
            "continuation runtime receipt accounting mismatch"
        )
    try:
        old_host = wave._validate_host(
            value["old_host"],
            "runtime receipt old host",
        )
        new_host = wave._validate_host(
            value["new_host"],
            "runtime receipt new host",
        )
        wave._validate_smplx_asset(
            value["old_smplx_asset"],
            stage=value["stage"],
            host=old_host,
            label="runtime receipt old SMPL-X asset",
        )
        wave._validate_smplx_asset(
            value["new_smplx_asset"],
            stage=value["stage"],
            host=new_host,
            label="runtime receipt new SMPL-X asset",
        )
    except wave.ContinuationWaveError as error:
        raise ContinuationRuntimeError(str(error)) from error
    for key in (
        "old_config_sha256",
        "old_source_audit_sha256",
        "old_dataset_receipt_sha256",
        "new_config_sha256",
        "new_config_semantic_sha256",
        "new_dataset_semantic_sha256",
    ):
        _require_sha256(value[key], f"runtime receipt {key}")
    try:
        proof = boundary_state.validate_boundary_state_proof(
            value["boundary_state"]
        )
    except boundary_state.BoundaryStateError as error:
        raise ContinuationRuntimeError(str(error)) from error
    if (
        proof["stage"] != value["stage"]
        or proof["boundary_epoch"] != boundary
        or proof["world_size"] != world
    ):
        raise ContinuationRuntimeError(
            "continuation runtime boundary proof mismatch"
        )
    snapshot = validate_old_segment_snapshot(
        value["old_segment_snapshot"]
    )
    if snapshot["stage"] != value["stage"]:
        raise ContinuationRuntimeError(
            "continuation runtime old snapshot stage mismatch"
        )
    claimed = _require_sha256(
        value["receipt_payload_sha256"],
        "runtime receipt payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256")
    if wave.canonical_json_sha256(unsigned) != claimed:
        raise ContinuationRuntimeError(
            "continuation runtime receipt payload mismatch"
        )
    return dict(value)
