#!/usr/bin/env python3
"""Merge the exact 8-shard SHOW prerequisite validation measurements.

This program is deliberately CPU-only.  It verifies every immutable shard,
recomputes exact clip/window coverage from the frozen validation view, merges
only sufficient statistics, and publishes one independently attributable
measurement file per prerequisite stage plus a signed measurement index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import gate_released_all_speakers_on_show as gate
from scripts.show_base import prerequisite_val_contract as contract


SHARD_KEYS = {
    "format",
    "status",
    "stage",
    "split",
    "test_visible",
    "epoch",
    "optimizer_updates",
    "checkpoint",
    "checkpoint_audit_sha256",
    "canonical_receipt",
    "producer_source",
    "protocol",
    "shard",
    "finite",
    "exact_once_within_shard",
    "determinism",
    "accumulators",
    "codebook_histograms",
    "runtime",
    "receipt_payload_sha256",
}


def _artifact(path: Path, sha256: str) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256}


def _payload_artifact(
    path: Path,
    sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256,
        "receipt_payload_sha256": contract.require_sha256(
            payload.get("receipt_payload_sha256"),
            f"{path} receipt payload SHA-256",
        ),
    }


def _validate_source_receipt(
    value: Any,
    label: str,
    *,
    reprove_local: bool = False,
) -> dict[str, Any]:
    value = contract.exact_keys(
        value,
        (
            "source_root",
            "origin",
            "commit",
            "tree",
            "clean",
            "script",
            "script_relative",
            "script_sha256",
        ),
        label,
    )
    if value["origin"] != contract.EXPECTED_ORIGIN or value["clean"] is not True:
        raise contract.ContractError(f"{label} is not clean SemTalk source")
    contract.require_git_oid(value["commit"], f"{label}.commit")
    contract.require_git_oid(value["tree"], f"{label}.tree")
    if (
        not isinstance(value["source_root"], str)
        or not Path(value["source_root"]).is_absolute()
        or not isinstance(value["script"], str)
        or not Path(value["script"]).is_absolute()
    ):
        raise contract.ContractError(f"{label} source paths must be absolute")
    relative = contract._portable_relative(
        value["script_relative"],
        f"{label}.script_relative",
    )
    script_sha = contract.require_sha256(
        value["script_sha256"],
        f"{label}.script_sha256",
    )
    root = Path(value["source_root"])
    script = Path(value["script"])
    try:
        observed_relative = script.relative_to(root).as_posix()
    except ValueError as error:
        raise contract.ContractError(f"{label} script escapes source root") from error
    if observed_relative != relative:
        raise contract.ContractError(f"{label}.script_relative mismatch")
    if reprove_local:
        if root.is_symlink() or not root.is_dir():
            raise contract.ContractError(f"{label}.source_root is invalid")
        resolved_root = root.resolve(strict=True)
        resolved_script = contract.regular_file(
            script,
            f"{label}.script",
            val_only=False,
        )
        try:
            resolved_relative = resolved_script.relative_to(
                resolved_root
            ).as_posix()
        except ValueError as error:
            raise contract.ContractError(
                f"{label} resolved script escapes source root"
            ) from error
        if (
            resolved_relative != relative
            or contract.sha256_file(resolved_script) != script_sha
        ):
            raise contract.ContractError(f"{label} live script changed")
    return dict(value)


def _portable_source_identity(
    value: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    value = _validate_source_receipt(value, label, reprove_local=False)
    return {
        "origin": value["origin"],
        "commit": value["commit"],
        "tree": value["tree"],
        "script_relative": value["script_relative"],
        "script_sha256": value["script_sha256"],
    }


def _repository_identity(
    value: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    source = _validate_source_receipt(value, label, reprove_local=False)
    return {
        "origin": source["origin"],
        "commit": source["commit"],
        "tree": source["tree"],
    }


def _expected_shard_paths() -> set[Path]:
    return {
        Path("shards")
        / stage
        / f"epoch_{epoch:04d}"
        / f"shard_{shard_index:02d}.json"
        for stage in contract.STAGES
        for epoch in contract.EXPECTED_CANDIDATE_EPOCHS
        for shard_index in range(contract.EXPECTED_SHARDS)
    }


def _inventory_shard_subtree(
    root: Path,
    *,
    expected_files: set[Path],
) -> set[Path]:
    expected_directories = {Path("shards")}
    for path in expected_files:
        expected_directories.update(path.parents)
    expected_directories.discard(Path("."))
    shard_directory = root / "shards"
    try:
        shard_mode = os.lstat(shard_directory).st_mode
    except FileNotFoundError:
        raise contract.ContractError(
            f"shard root lacks its shards subtree: {root}"
        ) from None
    if stat.S_ISLNK(shard_mode) or not stat.S_ISDIR(shard_mode):
        raise contract.ContractError("shards subtree must be a non-symlink directory")
    observed_files: set[Path] = set()
    stack = [(shard_directory, Path("shards"))]
    while stack:
        directory, relative_directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = relative_directory / entry.name
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise contract.ContractError(
                        f"shard subtree contains a symlink: {relative}"
                    )
                if stat.S_ISDIR(mode):
                    if relative not in expected_directories:
                        raise contract.ContractError(
                            f"unexpected shard directory: {relative}"
                        )
                    stack.append((Path(entry.path), relative))
                elif stat.S_ISREG(mode):
                    if relative not in expected_files:
                        raise contract.ContractError(
                            f"unexpected shard file: {relative}"
                        )
                    observed_files.add(relative)
                else:
                    raise contract.ContractError(
                        f"shard subtree contains a special file: {relative}"
                    )
    return observed_files


def _validate_artifact(
    value: Any,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    value = contract.exact_keys(value, ("path", "sha256"), label)
    if value != dict(expected):
        raise contract.ContractError(f"{label} binding mismatch")


def _validate_checkpoint(
    value: Any,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    value = contract.exact_keys(value, ("path", "sha256", "bytes"), label)
    if value != {
        "path": expected["checkpoint"],
        "sha256": expected["checkpoint_sha256"],
        "bytes": expected["checkpoint_bytes"],
    }:
        raise contract.ContractError(f"{label} binding mismatch")


def _expected_window_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    digest = hashlib.sha256()
    windows = 0
    for row in rows:
        global_index = contract.require_exact_int(
            row.get("global_index"),
            "canonical global_index",
        )
        frames = contract.require_exact_int(
            row.get("frames"),
            "canonical frames",
        )
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise contract.ContractError("canonical clip_id is invalid")
        canonical_sha = contract.require_sha256(
            row.get("canonical_npz_sha256"),
            "canonical NPZ SHA-256",
        )
        count = contract.window_count(frames)
        for index in range(count):
            start = index * contract.WINDOW_STRIDE
            digest.update(
                contract.canonical_json_bytes(
                    {
                        "global_index": global_index,
                        "clip_id": clip_id,
                        "start": start,
                        "end": start + contract.WINDOW_LENGTH,
                        "canonical_npz_sha256": canonical_sha,
                    }
                )
            )
        windows += count
    return windows, digest.hexdigest()


def _expected_accumulator_counts(stage: str, windows: int) -> dict[str, int]:
    joints = {"face": 1, "hands": 30, "upper": 13, "lower": 9}
    if stage in contract.RVQ_STAGES:
        result = {
            "rotation_geodesic": windows * 64 * joints[stage],
            "rotation_velocity": windows * 63 * joints[stage] * 9,
            "rotation_acceleration": windows * 62 * joints[stage] * 9,
        }
        if stage == "face":
            result.update(
                {
                    "expression": windows * 64 * 100,
                    "expression_velocity": windows * 63 * 100,
                    "expression_acceleration": windows * 62 * 100,
                }
            )
        elif stage == "lower":
            result.update(
                {
                    "contact": windows * 64 * 4,
                    "translation": windows * 64 * 3,
                }
            )
        return result
    return {
        "contact": windows * 64 * 4,
        "velocity_x": windows * 64,
        "velocity_z": windows * 64,
        "velocity_delta_x": windows * 63,
        "velocity_delta_z": windows * 63,
        "velocity_acceleration_x": windows * 62,
        "velocity_acceleration_z": windows * 62,
        "integrated_translation_velocity": windows * 63 * 3,
        "integrated_translation_acceleration": windows * 62 * 3,
        "integrated_translation": windows * 64 * 3,
    }


def _validate_histograms(
    value: Any,
    *,
    stage: str,
    windows: int,
    label: str,
) -> list[list[int]] | None:
    if stage == "global":
        if value is not None:
            raise contract.ContractError(f"{label} must be null")
        return None
    if not isinstance(value, list) or len(value) != contract.RVQ_LEVELS:
        raise contract.ContractError(f"{label} RVQ level coverage mismatch")
    result: list[list[int]] = []
    expected_tokens = windows * (contract.WINDOW_LENGTH // 4)
    for level, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != contract.CODEBOOK_SIZE:
            raise contract.ContractError(f"{label}[{level}] schema mismatch")
        histogram = [
            contract.require_exact_int(item, f"{label}[{level}] code count")
            for item in raw
        ]
        if any(item < 0 for item in histogram) or sum(histogram) != expected_tokens:
            raise contract.ContractError(f"{label}[{level}] token count mismatch")
        result.append(histogram)
    return result


def _codebook_summary(histograms: list[list[int]] | None) -> Any:
    if histograms is None:
        return None
    result = []
    for level, histogram in enumerate(histograms):
        tokens = sum(histogram)
        occupied = sum(count > 0 for count in histogram)
        entropy = 0.0
        for count in histogram:
            if count:
                probability = count / tokens
                entropy -= probability * math.log(probability)
        result.append(
            {
                "level": level,
                "tokens": tokens,
                "occupied_codes": occupied,
                "occupancy_fraction": occupied / contract.CODEBOOK_SIZE,
                "dead_fraction": 1.0 - occupied / contract.CODEBOOK_SIZE,
                "entropy_nats": entropy,
                "normalized_entropy": entropy
                / math.log(contract.CODEBOOK_SIZE),
                "histogram": histogram,
            }
        )
    return result


def _read_shard(
    path: Path,
    *,
    stage: str,
    epoch: int,
    shard_index: int,
    candidate: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    canonical_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = contract.regular_file(path, f"{stage} epoch {epoch} shard")
    payload_bytes = resolved.read_bytes()
    payload = contract.verify_receipt_payload(
        contract.strict_json_bytes(payload_bytes, str(resolved)),
        f"{stage} epoch {epoch} shard {shard_index}",
    )
    if set(payload) != SHARD_KEYS:
        raise contract.ContractError("validation shard top-level schema mismatch")
    if (
        payload["format"] != contract.SHARD_FORMAT
        or payload["status"] != "complete"
        or payload["stage"] != stage
        or payload["split"] != "val"
        or payload["test_visible"] is not False
        or payload["epoch"] != epoch
        or payload["optimizer_updates"]
        != epoch * contract.EXPECTED_UPDATES_PER_EPOCH
        or payload["finite"] is not True
        or payload["exact_once_within_shard"] is not True
    ):
        raise contract.ContractError("validation shard protocol mismatch")
    if (
        contract.require_sha256(
            payload["checkpoint_audit_sha256"],
            "checkpoint audit SHA-256",
        )
        != candidate["checkpoint_audit_sha256"]
    ):
        raise contract.ContractError("checkpoint audit binding mismatch")
    _validate_checkpoint(payload["checkpoint"], candidate, "shard checkpoint")
    if payload["canonical_receipt"] != canonical_receipt:
        raise contract.ContractError("shard canonical view binding mismatch")
    source = _validate_source_receipt(
        payload["producer_source"],
        "shard evaluator source",
    )
    protocol = contract.exact_keys(
        payload["protocol"],
        (
            "name",
            "stage_independent",
            "candidate_variable_only",
            "window_length",
            "window_stride",
            "selection_split",
            "test_visible",
            "full_base_fgd_used",
            "inference_only_exclusions",
        ),
        "shard protocol",
    )
    if protocol != {
        "name": contract.SELECTION_METRICS[stage],
        "stage_independent": True,
        "candidate_variable_only": True,
        "window_length": contract.WINDOW_LENGTH,
        "window_stride": contract.WINDOW_STRIDE,
        "selection_split": "val",
        "test_visible": False,
        "full_base_fgd_used": False,
        "inference_only_exclusions": [
            "quantizer_embedding_loss",
            "smplx_vertex_loss",
        ],
    }:
        raise contract.ContractError("shard stage metric protocol mismatch")
    expected_clip_ids = [str(row["clip_id"]) for row in expected_rows]
    expected_windows, expected_records_sha = _expected_window_evidence(
        expected_rows
    )
    shard = contract.exact_keys(
        payload["shard"],
        (
            "index",
            "count",
            "clip_count",
            "clip_ids",
            "clip_ids_sha256",
            "window_count",
            "window_records_sha256",
        ),
        "shard coverage",
    )
    if (
        shard["index"] != shard_index
        or shard["count"] != contract.EXPECTED_SHARDS
        or shard["clip_count"] != len(expected_rows)
        or shard["clip_ids"] != expected_clip_ids
        or shard["clip_ids_sha256"]
        != contract.canonical_payload_sha256(expected_clip_ids)
        or shard["window_count"] != expected_windows
        or shard["window_records_sha256"] != expected_records_sha
    ):
        raise contract.ContractError("shard exact coverage mismatch")
    determinism = contract.exact_keys(
        payload["determinism"],
        (
            "seed",
            "torch_deterministic_algorithms",
            "tf32",
            "first_batch_exact_replay",
            "output_digest_sha256",
        ),
        "shard determinism",
    )
    if (
        contract.require_exact_int(determinism["seed"], "shard seed") < 0
        or determinism["torch_deterministic_algorithms"] is not True
        or determinism["tf32"] is not False
        or determinism["first_batch_exact_replay"] is not True
    ):
        raise contract.ContractError("shard determinism evidence mismatch")
    contract.require_sha256(
        determinism["output_digest_sha256"],
        "shard output digest",
    )
    runtime = contract.exact_keys(
        payload["runtime"],
        ("python", "torch", "numpy", "device", "batch_size", "batches"),
        "shard runtime",
    )
    batch_size = contract.require_exact_int(
        runtime["batch_size"],
        "shard batch size",
    )
    batches = contract.require_exact_int(runtime["batches"], "shard batches")
    if (
        batch_size <= 0
        or batches != math.ceil(expected_windows / batch_size)
        or not all(
            isinstance(runtime[key], str) and runtime[key]
            for key in ("python", "torch", "numpy", "device")
        )
    ):
        raise contract.ContractError("shard runtime coverage mismatch")
    raw_accumulators = payload["accumulators"]
    if (
        not isinstance(raw_accumulators, dict)
        or set(raw_accumulators) != set(contract.ACCUMULATOR_KEYS[stage])
    ):
        raise contract.ContractError("shard accumulator names mismatch")
    expected_counts = _expected_accumulator_counts(stage, expected_windows)
    accumulators = {}
    for name in contract.ACCUMULATOR_KEYS[stage]:
        accumulator = contract.validate_accumulator(
            raw_accumulators[name],
            f"{stage}.{name}",
        )
        if accumulator["count"] != expected_counts[name]:
            raise contract.ContractError(
                f"{stage}.{name} element coverage mismatch"
            )
        accumulators[name] = accumulator
    histograms = _validate_histograms(
        payload["codebook_histograms"],
        stage=stage,
        windows=expected_windows,
        label=f"{stage} epoch {epoch} shard histograms",
    )
    receipt = {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "receipt_payload_sha256": payload["receipt_payload_sha256"],
        "shard_index": shard_index,
        "clips": len(expected_rows),
        "windows": expected_windows,
    }
    return {
        "source": source,
        "runtime": dict(runtime),
        "seed": determinism["seed"],
        "accumulators": accumulators,
        "histograms": histograms,
        "clip_ids": expected_clip_ids,
        "windows": expected_windows,
    }, receipt


def _merge_histograms(
    values: Sequence[list[list[int]] | None],
    stage: str,
) -> list[list[int]] | None:
    if stage == "global":
        if any(value is not None for value in values):
            raise contract.ContractError("Global unexpectedly has codebook evidence")
        return None
    result = [
        [0 for _ in range(contract.CODEBOOK_SIZE)]
        for _ in range(contract.RVQ_LEVELS)
    ]
    for value in values:
        if value is None:
            raise contract.ContractError("RVQ histogram is missing")
        for level in range(contract.RVQ_LEVELS):
            for index in range(contract.CODEBOOK_SIZE):
                result[level][index] += value[level][index]
    return result


def _publish_directory(
    output_root: Path,
    stage_payloads: Mapping[str, Mapping[str, Any]],
    index_payload: Mapping[str, Any],
) -> None:
    if not output_root.is_absolute():
        raise contract.ContractError("measurement output root must be absolute")
    contract.reject_forbidden_label(output_root, "measurement output root")
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing existing output root: {output_root}")
    parent = output_root.parent.resolve(strict=True)
    staging = parent / (
        f".{output_root.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    published = False
    try:
        for stage in contract.STAGES:
            contract.atomic_json_new(
                staging / f"{stage}_measurements.json",
                stage_payloads[stage],
            )
        contract.atomic_json_new(staging / "measurement_index.json", index_payload)
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(staging, output_root)
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def merge(
    *,
    candidate_index_path: Path,
    candidate_index_sha256: str,
    canonical_manifest: Path,
    canonical_manifest_sha256: str,
    canonical_summary: Path,
    canonical_summary_sha256: str,
    canonical_lineage: Path,
    canonical_lineage_sha256: str,
    shard_roots: Sequence[Path],
    output_root: Path,
    merge_source: Mapping[str, Any],
) -> dict[str, Any]:
    if not output_root.is_absolute():
        raise contract.ContractError("measurement output root must be absolute")
    output_parent = output_root.parent.resolve(strict=True)
    output_root = output_parent / output_root.name
    merge_source = _validate_source_receipt(
        merge_source,
        "merge source",
        reprove_local=True,
    )
    candidate_index, candidate_artifact = contract.load_candidate_index(
        candidate_index_path,
        candidate_index_sha256,
    )
    rows, canonical_receipt = contract.load_val_canonical(
        manifest_path=canonical_manifest,
        manifest_sha256=canonical_manifest_sha256,
        summary_path=canonical_summary,
        summary_sha256=canonical_summary_sha256,
        lineage_path=canonical_lineage,
        lineage_sha256=canonical_lineage_sha256,
    )
    if not isinstance(shard_roots, Sequence) or not shard_roots:
        raise contract.ContractError("at least one shard root is required")
    resolved_shard_roots = []
    for raw_root in shard_roots:
        shard_root = Path(raw_root)
        if (
            not shard_root.is_absolute()
            or shard_root.is_symlink()
            or not shard_root.is_dir()
        ):
            raise contract.ContractError(
                "shard root must be an existing directory"
            )
        shard_root = shard_root.resolve(strict=True)
        contract.reject_forbidden_label(shard_root, "shard root")
        if shard_root in resolved_shard_roots:
            raise contract.ContractError("shard root is duplicated")
        resolved_shard_roots.append(shard_root)
    expected_shard_paths = _expected_shard_paths()
    shard_locations: dict[Path, Path] = {}
    for shard_root in resolved_shard_roots:
        observed = _inventory_shard_subtree(
            shard_root,
            expected_files=expected_shard_paths,
        )
        for relative in observed:
            if relative in shard_locations:
                raise contract.ContractError(
                    f"duplicate shard path across roots: {relative}"
                )
            shard_locations[relative] = shard_root / relative
    observed_shard_paths = set(shard_locations)
    if observed_shard_paths != expected_shard_paths:
        missing = sorted(expected_shard_paths - observed_shard_paths)
        extra = sorted(observed_shard_paths - expected_shard_paths)
        raise contract.ContractError(
            "shard subtree inventory is not the exact expected 400 files; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    candidate_receipt = {
        **candidate_artifact,
        "receipt_payload_sha256": candidate_index[
            "receipt_payload_sha256"
        ],
    }
    stage_payloads: dict[str, dict[str, Any]] = {}
    evaluator_source: dict[str, Any] | None = None
    common_runtime_versions: dict[str, str] | None = None
    common_seed: int | None = None
    total_windows = sum(contract.window_count(row["frames"]) for row in rows)
    for stage in contract.STAGES:
        candidates = []
        for candidate_index_number, epoch in enumerate(
            contract.EXPECTED_CANDIDATE_EPOCHS
        ):
            candidate = contract.candidate_lookup(
                candidate_index,
                stage,
                epoch,
            )
            shard_values = []
            shard_receipts = []
            union_clip_ids: list[str] = []
            for shard_index in range(contract.EXPECTED_SHARDS):
                expected_rows = [
                    row
                    for row in rows
                    if int(row["global_index"]) % contract.EXPECTED_SHARDS
                    == shard_index
                ]
                relative = (
                    Path("shards")
                    / stage
                    / f"epoch_{epoch:04d}"
                    / f"shard_{shard_index:02d}.json"
                )
                path = shard_locations[relative]
                value, receipt = _read_shard(
                    path,
                    stage=stage,
                    epoch=epoch,
                    shard_index=shard_index,
                    candidate=candidate,
                    expected_rows=expected_rows,
                    canonical_receipt=canonical_receipt,
                )
                if evaluator_source is None:
                    evaluator_source = value["source"]
                elif _portable_source_identity(
                    value["source"],
                    "shard evaluator source",
                ) != _portable_source_identity(
                    evaluator_source,
                    "first shard evaluator source",
                ):
                    raise contract.ContractError(
                        "evaluator portable source changed across shards"
                    )
                versions = {
                    key: value["runtime"][key]
                    for key in ("python", "torch", "numpy")
                }
                if common_runtime_versions is None:
                    common_runtime_versions = versions
                elif versions != common_runtime_versions:
                    raise contract.ContractError(
                        "runtime versions changed across shards"
                    )
                if common_seed is None:
                    common_seed = value["seed"]
                elif value["seed"] != common_seed:
                    raise contract.ContractError(
                        "deterministic seed changed across shards"
                    )
                shard_values.append(value)
                shard_receipts.append(receipt)
                union_clip_ids.extend(value["clip_ids"])
            canonical_clip_ids = [str(row["clip_id"]) for row in rows]
            if (
                len(union_clip_ids) != contract.EXPECTED_VAL_CLIPS
                or len(set(union_clip_ids)) != contract.EXPECTED_VAL_CLIPS
                or set(union_clip_ids) != set(canonical_clip_ids)
                or sum(value["windows"] for value in shard_values)
                != total_windows
            ):
                raise contract.ContractError(
                    f"{stage} epoch {epoch} merged coverage is not exact-once"
                )
            merged_accumulators = {
                name: contract.merge_accumulators(
                    [value["accumulators"][name] for value in shard_values],
                    f"{stage} epoch {epoch}.{name}",
                )
                for name in contract.ACCUMULATOR_KEYS[stage]
            }
            metrics, score = contract.stage_metrics(
                stage,
                merged_accumulators,
            )
            merged_histograms = _merge_histograms(
                [value["histograms"] for value in shard_values],
                stage,
            )
            candidates.append(
                {
                    "candidate_index": candidate_index_number,
                    "epoch": epoch,
                    "optimizer_updates": candidate["optimizer_updates"],
                    "selection_score": score,
                    "selection_metric": contract.SELECTION_METRICS[stage],
                    "candidate_checkpoint": {
                        "path": candidate["checkpoint"],
                        "sha256": candidate["checkpoint_sha256"],
                        "bytes": candidate["checkpoint_bytes"],
                    },
                    "metrics": metrics,
                    "codebook_histograms": _codebook_summary(
                        merged_histograms
                    ),
                    "coverage": {
                        "split": "val",
                        "test_visible": False,
                        "clips": contract.EXPECTED_VAL_CLIPS,
                        "shards": contract.EXPECTED_SHARDS,
                        "windows": total_windows,
                        "exact_once": True,
                        "all_finite": True,
                    },
                    "shard_receipts": shard_receipts,
                }
            )
        assert evaluator_source is not None
        stage_payloads[stage] = contract.receipt_payload(
            {
                "format": contract.STAGE_MEASUREMENT_FORMAT,
                "status": "complete",
                "target_dataset": "SHOW",
                "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
                "split": "val",
                "test_visible": False,
                "stage": stage,
                "selection_metric": contract.SELECTION_METRICS[stage],
                "protocol": {
                    "per_stage_independent": True,
                    "candidate_variable_only": True,
                    "candidate_epochs": list(
                        contract.EXPECTED_CANDIDATE_EPOCHS
                    ),
                    "window_length": contract.WINDOW_LENGTH,
                    "window_stride": contract.WINDOW_STRIDE,
                    "full_base_fgd_used": False,
                    "test_feedback_into_selection": False,
                },
                "canonical_view": canonical_receipt,
                "producer_sources": {
                    "evaluator": evaluator_source,
                    "merge": merge_source,
                },
                "training_source": candidate_index["source_receipts"][
                    stage
                ],
                "config_sha256": candidate_index["config_sha256"][stage],
                "candidate_index_receipt": candidate_receipt,
                "candidates": candidates,
                "coverage": {
                    "split": "val",
                    "test_visible": False,
                    "clips_per_candidate": contract.EXPECTED_VAL_CLIPS,
                    "candidates": len(contract.EXPECTED_CANDIDATE_EPOCHS),
                    "shards_per_candidate": contract.EXPECTED_SHARDS,
                    "shard_jobs": len(contract.EXPECTED_CANDIDATE_EPOCHS)
                    * contract.EXPECTED_SHARDS,
                    "windows_per_candidate": total_windows,
                    "exact_once_per_candidate": True,
                    "all_finite": True,
                },
            }
        )
    assert evaluator_source is not None
    if _repository_identity(
        evaluator_source,
        "evaluator source",
    ) != _repository_identity(merge_source, "merge source"):
        raise contract.ContractError(
            "evaluator and merge repository identities differ"
        )
    final_stage_receipts = {}
    for stage in contract.STAGES:
        encoded = (
            json.dumps(
                stage_payloads[stage],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        final_path = output_root / f"{stage}_measurements.json"
        final_stage_receipts[stage] = {
            "stage": stage,
            "path": str(final_path),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "receipt_payload_sha256": stage_payloads[stage][
                "receipt_payload_sha256"
            ],
        }
    index_payload = contract.receipt_payload(
        {
            "format": contract.MEASUREMENT_FORMAT,
            "status": "complete",
            "target_dataset": "SHOW",
            "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
            "split": "val",
            "test_visible": False,
            "protocol": {
                "per_stage_independent": True,
                "candidate_epochs": list(
                    contract.EXPECTED_CANDIDATE_EPOCHS
                ),
                "candidates_per_stage": len(
                    contract.EXPECTED_CANDIDATE_EPOCHS
                ),
                "shards_per_candidate": contract.EXPECTED_SHARDS,
                "full_base_fgd_used": False,
                "test_feedback_into_selection": False,
            },
            "canonical_view": canonical_receipt,
            "producer_sources": {
                "evaluator": evaluator_source,
                "merge": merge_source,
            },
            "training_sources": candidate_index["source_receipts"],
            "config_sha256": candidate_index["config_sha256"],
            "candidate_index_receipt": candidate_receipt,
            "stages": final_stage_receipts,
            "coverage": {
                "stages": len(contract.STAGES),
                "candidates_per_stage": len(
                    contract.EXPECTED_CANDIDATE_EPOCHS
                ),
                "shards_per_candidate": contract.EXPECTED_SHARDS,
                "total_shard_jobs": len(contract.STAGES)
                * len(contract.EXPECTED_CANDIDATE_EPOCHS)
                * contract.EXPECTED_SHARDS,
                "clips_per_candidate": contract.EXPECTED_VAL_CLIPS,
                "windows_per_candidate": total_windows,
                "exact_once": True,
                "all_finite": True,
            },
        }
    )
    _publish_directory(output_root, stage_payloads, index_payload)
    return index_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge exact SHOW prerequisite validation shards",
        allow_abbrev=False,
    )
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--expected-candidate-index-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        required=True,
        help="repeat for disjoint multi-host shard roots",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    merge_source = gate.git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
        script=Path(__file__).resolve(),
    )
    result = merge(
        candidate_index_path=args.candidate_index,
        candidate_index_sha256=args.expected_candidate_index_sha256,
        canonical_manifest=args.canonical_manifest,
        canonical_manifest_sha256=args.expected_manifest_sha256,
        canonical_summary=args.canonical_summary,
        canonical_summary_sha256=args.expected_summary_sha256,
        canonical_lineage=args.canonical_lineage,
        canonical_lineage_sha256=args.expected_lineage_sha256,
        shard_roots=args.shard_root,
        output_root=args.output_root,
        merge_source=merge_source,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "stages": list(contract.STAGES),
                "shard_jobs": result["coverage"]["total_shard_jobs"],
                "receipt_payload_sha256": result[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
