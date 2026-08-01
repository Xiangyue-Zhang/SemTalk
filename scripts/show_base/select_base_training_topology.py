#!/usr/bin/env python3
"""Seal the fastest safe SemTalk Base topology after all nine real probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_long_val_contract as formal_validation
from scripts.show_base import train_base_official_adapt_long as contract


class TopologySelectionError(RuntimeError):
    """Raised when any measured topology is absent, unsafe, or stale."""


QUALITY_GATE_FORMAT = "semtalk_show_base_topology_quality_gate_spec_v3"
QUALITY_REPORT_FORMAT = "semtalk_show_base_topology_quality_report_v3"
QUALITY_SKIP_FORMAT = (
    "semtalk_show_base_topology_quality_skipped_over_eta_budget_v2"
)
SHORT_TRAJECTORY_FORMAT = (
    "semtalk_show_base_topology_short_trajectory_v2"
)
QUALITY_PROVENANCE_FORMAT = (
    "semtalk_show_base_topology_quality_provenance_v3"
)
QUALITY_PROTOCOL_VERSION = 3
QUALITY_ARTIFACT_ROOT_NAMESPACE = "semtalk_show_base_topology_quality_v3"
W1_REFERENCE_EPOCHS = (1, 2, 4, 8)
CANDIDATE_QUALITY_EPOCHS = (1, 2, 4, 8, 16, 32)
SAME_EPOCH_COMPARISON_EPOCHS = W1_REFERENCE_EPOCHS
TAIL_EPOCHS = (16, 32)
EXPECTED_VAL_CLIPS = formal_validation.EXPECTED_VAL_CLIPS
PRIMARY_METRIC_PATH = "validation.diffsheg.metrics.fgd"
VALIDATION_PROTOCOL = "diffsheg_show_validation_fgd_v1"
MAX_TRAINING_SECONDS = 24 * 60 * 60
MAX_P99_TRAINING_SECONDS = 22 * 60 * 60
MAX_ABSOLUTE_FGD_REGRESSION = 0.01
MAX_RELATIVE_FGD_REGRESSION = 0.02
SELECTION_PROTOCOL = {
    "primary_metric": PRIMARY_METRIC_PATH,
    "validation_protocol": VALIDATION_PROTOCOL,
    "mode": "min",
    "validation_only_for_selection": True,
    "test_evaluations": 0,
}


def quality_epochs_for_mode(mode: str) -> tuple[int, ...]:
    if mode not in contract.TOPOLOGY_SPECS:
        raise TopologySelectionError(f"unknown topology mode {mode!r}")
    if mode == contract.OFFICIAL_W1_REFERENCE_MODE:
        return W1_REFERENCE_EPOCHS
    return CANDIDATE_QUALITY_EPOCHS


def quality_role_for_mode(mode: str) -> str:
    return (
        "w1_reference_only"
        if mode == contract.OFFICIAL_W1_REFERENCE_MODE
        else "candidate_quality"
    )


def _artifact(
    value: Any,
    label: str,
    *,
    payload: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    required = {"path", "sha256", "bytes"}
    if payload:
        required.add("receipt_payload_sha256")
    if not isinstance(value, dict) or set(value) != required:
        raise TopologySelectionError(f"{label} artifact schema mismatch")
    raw_path = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        or type(size) is not int
        or size <= 0
    ):
        raise TopologySelectionError(f"{label} artifact identity is invalid")
    path = Path(raw_path)
    try:
        resolved, data, _identity = contract._read_regular_file_bytes(
            path,
            label,
        )
    except Exception as error:
        raise TopologySelectionError(f"{label} artifact is absent") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        raise TopologySelectionError(
            f"{label} artifact must be one canonical regular file"
        )
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise TopologySelectionError(f"{label} artifact changed")
    normalized = dict(value)
    if not payload:
        return normalized, None
    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise TopologySelectionError(f"{label} is not strict JSON") from error
    if not isinstance(parsed, dict):
        raise TopologySelectionError(f"{label} must contain a JSON object")
    claimed = value.get("receipt_payload_sha256")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(claimed)) is None
        or parsed.get("receipt_payload_sha256") != claimed
        or contract.canonical_json_sha256(
            {
                key: item
                for key, item in parsed.items()
                if key != "receipt_payload_sha256"
            }
        )
        != claimed
    ):
        raise TopologySelectionError(f"{label} payload hash changed")
    return normalized, parsed


def _strict_jsonl_rows(
    artifact: Mapping[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    path = Path(str(artifact["path"]))
    try:
        resolved, payload, _identity = contract._read_regular_file_bytes(
            path,
            label,
        )
    except Exception as error:
        raise TopologySelectionError(f"{label} is absent") from error
    if (
        resolved != path
        or len(payload) != artifact["bytes"]
        or hashlib.sha256(payload).hexdigest() != artifact["sha256"]
    ):
        raise TopologySelectionError(f"{label} changed during replay")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(payload.splitlines(), 1):
            if not raw.strip():
                raise ValueError(f"blank line {line_number}")
            row = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=no_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append(row)
    except (UnicodeDecodeError, ValueError) as error:
        raise TopologySelectionError(f"{label} is not strict JSONL") from error
    return rows


def _fresh_trajectory_probe(
    mode: str,
    value: Any,
    label: str,
) -> dict[str, Any]:
    """Replay the trainer's fresh all-rank trajectory aggregate."""

    specification = contract.TOPOLOGY_SPECS[mode]
    world_size = int(specification["world_size"])
    local_batch_size = int(specification["local_batch_size"])
    required = {
        "format",
        "optimizer_updates",
        "world_size",
        "rank_order",
        "all_rank_model_state_identical",
        "all_rank_parameter_state_identical",
        "all_rank_buffer_state_identical",
        "all_rank_optimizer_state_identical",
        "ranks",
    }
    rank_required = {
        "rank",
        "optimizer_updates",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
        "parameter_state_tensors",
        "parameter_state_schema_sha256",
        "parameter_state_semantic_sha256",
        "buffer_state_tensors",
        "buffer_state_schema_sha256",
        "buffer_state_semantic_sha256",
        "optimizer_state_semantic_sha256",
        "python_random_state_sha256",
        "numpy_random_state_sha256",
        "torch_cpu_rng_state_sha256",
        "torch_cuda_rng_state_sha256",
        "sample_order_sha256",
        "sample_count",
    }
    ranks = value.get("ranks") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("format") != contract.TRAJECTORY_PROBE_FORMAT
        or type(value.get("optimizer_updates")) is not int
        or value["optimizer_updates"]
        != contract.TRAJECTORY_PROBE_UPDATES
        or type(value.get("world_size")) is not int
        or value["world_size"] != world_size
        or type(value.get("rank_order")) is not list
        or any(type(rank) is not int for rank in value["rank_order"])
        or value["rank_order"] != list(range(world_size))
        or type(value.get("all_rank_model_state_identical")) is not bool
        or value.get("all_rank_parameter_state_identical") is not True
        or type(value.get("all_rank_buffer_state_identical")) is not bool
        or value.get("all_rank_optimizer_state_identical") is not True
        or not isinstance(ranks, list)
        or len(ranks) != world_size
        or [rank.get("rank") for rank in ranks if isinstance(rank, dict)]
        != list(range(world_size))
    ):
        raise TopologySelectionError(f"{label} changed")
    for rank in ranks:
        if (
            not isinstance(rank, dict)
            or set(rank) != rank_required
            or type(rank.get("rank")) is not int
            or type(rank.get("optimizer_updates")) is not int
            or rank["optimizer_updates"]
            != contract.TRAJECTORY_PROBE_UPDATES
            or type(rank.get("model_state_tensors")) is not int
            or rank["model_state_tensors"] <= 0
            or type(rank.get("parameter_state_tensors")) is not int
            or rank["parameter_state_tensors"] <= 0
            or type(rank.get("buffer_state_tensors")) is not int
            or rank["buffer_state_tensors"] <= 0
            or rank["model_state_tensors"]
            != rank["parameter_state_tensors"] + rank["buffer_state_tensors"]
            or type(rank.get("sample_count")) is not int
            or rank["sample_count"]
            != contract.TRAJECTORY_PROBE_UPDATES * local_batch_size
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(item)) is None
                for key, item in rank.items()
                if key.endswith("_sha256")
            )
        ):
            raise TopologySelectionError(f"{label} changed")
    model_consensus = {
        (
            rank["model_state_tensors"],
            rank["model_state_schema_sha256"],
            rank["model_state_semantic_sha256"],
        )
        for rank in ranks
    }
    model_schema_consensus = {
        (
            rank["model_state_tensors"],
            rank["model_state_schema_sha256"],
        )
        for rank in ranks
    }
    parameter_consensus = {
        (
            rank["parameter_state_tensors"],
            rank["parameter_state_schema_sha256"],
            rank["parameter_state_semantic_sha256"],
        )
        for rank in ranks
    }
    buffer_consensus = {
        (
            rank["buffer_state_tensors"],
            rank["buffer_state_schema_sha256"],
            rank["buffer_state_semantic_sha256"],
        )
        for rank in ranks
    }
    buffer_schema_consensus = {
        (
            rank["buffer_state_tensors"],
            rank["buffer_state_schema_sha256"],
        )
        for rank in ranks
    }
    optimizer_consensus = {
        rank["optimizer_state_semantic_sha256"] for rank in ranks
    }
    if (
        len(model_schema_consensus) != 1
        or len(parameter_consensus) != 1
        or len(buffer_schema_consensus) != 1
        or len(optimizer_consensus) != 1
        or value["all_rank_model_state_identical"]
        is not (len(model_consensus) == 1)
        or value["all_rank_buffer_state_identical"]
        is not (len(buffer_consensus) == 1)
    ):
        raise TopologySelectionError(f"{label} changed")
    return dict(value)


def validate_candidate_ready_receipts(
    mode: str,
    values: Sequence[Any],
    *,
    topology_gate_spec_sha256: str,
    quality_gate_spec_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    str,
    dict[int, dict[str, Any]],
]:
    """Freshly replay the mode-specific v3 short-quality publications."""

    quality_epochs = quality_epochs_for_mode(mode)
    quality_role = quality_role_for_mode(mode)
    reference_only = mode == contract.OFFICIAL_W1_REFERENCE_MODE
    if len(values) != len(quality_epochs):
        raise TopologySelectionError(
            f"{mode} candidate-ready coverage changed"
        )
    ready_artifacts: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    semantic_sha: str | None = None
    common: dict[str, Any] | None = None
    seen_ready: set[str] = set()
    required = {
        "format",
        "status",
        "run_purpose",
        "target_epochs",
        "quality_protocol_version",
        "artifact_root_namespace",
        "artifact_root",
        "quality_role",
        "reference_only",
        "late_w1_status",
        "w1_tail_equivalence_claimed",
        "selection_eligible",
        "test_visible",
        "epoch",
        "optimizer_updates",
        "candidate_checkpoint",
        "candidate_manifest",
        "frozen_inputs",
        "protocol",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "trajectory_anchor_match",
        "published_unix",
        "receipt_payload_sha256",
    }
    checkpoint_keys = {
        "path",
        "relative_path",
        "sha256",
        "bytes",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
    }
    manifest_keys = {
        "format",
        "status",
        "run_purpose",
        "target_epochs",
        "candidate_epochs",
        "quality_protocol_version",
        "artifact_root_namespace",
        "artifact_root",
        "quality_role",
        "reference_only",
        "late_w1_status",
        "w1_tail_equivalence_claimed",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "throughput_gate",
        "trajectory_mode",
        "trajectory_probe_verified",
        "trajectory_probe",
        "entries",
        "entries_sha256",
    }
    entry_keys = {
        "epoch",
        "optimizer_updates",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_container_schema",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
        "all_model_state_tensors_finite",
        "frozen_receipt_sha256",
        "trajectory_anchor_match",
        "trajectory_probe_verified",
        "run_purpose",
    }
    expected_updates = int(contract.TOPOLOGY_SPECS[mode]["updates_per_epoch"])
    throughput_keys = {
        "path",
        "sha256",
        "topology_mode",
        "samples_per_second",
        "seconds_per_update",
        "median_seconds",
        "p90_seconds",
        "p99_seconds",
        "estimated_training_seconds",
        "trajectory_mode",
        "trajectory_probe",
        "gate_frozen_receipt_sha256",
        "frozen_gate_compatibility_sha256",
    }
    frozen_keys = {
        "format",
        "run_purpose",
        "target_epochs",
        "source",
        "official_base",
        "speaker_initialization",
        "dataset",
        "protocol",
        "long_contract",
        "topology",
        "receipt_sha256",
    }
    for position, (epoch, value) in enumerate(zip(quality_epochs, values)):
        label = f"{mode} e{epoch} candidate-ready"
        ready, payload = _artifact(value, label, payload=True)
        assert payload is not None
        if ready["sha256"] in seen_ready:
            raise TopologySelectionError(
                f"{label} receipt was reused"
            )
        seen_ready.add(ready["sha256"])
        checkpoint_value = payload.get("candidate_checkpoint")
        manifest_value = payload.get("candidate_manifest")
        frozen_value = payload.get("frozen_inputs")
        protocol = payload.get("protocol")
        expected_optimizer_updates = epoch * expected_updates
        if (
            set(payload) != required
            or payload.get("format")
            != contract.SHORT_QUALITY_READY_RECEIPT_FORMAT
            or payload.get("status") != "ready"
            or payload.get("run_purpose")
            != contract.RUN_PURPOSE_SHORT_QUALITY
            or payload.get("target_epochs") != list(quality_epochs)
            or payload.get("quality_protocol_version")
            != QUALITY_PROTOCOL_VERSION
            or payload.get("artifact_root_namespace")
            != QUALITY_ARTIFACT_ROOT_NAMESPACE
            or not isinstance(payload.get("artifact_root"), str)
            or not Path(payload["artifact_root"]).is_absolute()
            or payload.get("quality_role") != quality_role
            or payload.get("reference_only") is not reference_only
            or payload.get("late_w1_status") != "not_measured"
            or payload.get("w1_tail_equivalence_claimed") is not False
            or payload.get("selection_eligible") is not False
            or payload.get("test_visible") is not False
            or payload.get("epoch") != epoch
            or payload.get("optimizer_updates")
            != expected_optimizer_updates
            or payload.get("trajectory_anchor_match") is not None
            or isinstance(payload.get("published_unix"), bool)
            or not isinstance(payload.get("published_unix"), (int, float))
            or not math.isfinite(float(payload["published_unix"]))
            or float(payload["published_unix"]) <= 0.0
            or not isinstance(checkpoint_value, dict)
            or set(checkpoint_value) != checkpoint_keys
            or checkpoint_value.get("relative_path")
            != (
                "provisional_candidates/"
                f"base_official_adapt_short_quality_epoch_{epoch:02d}.bin"
            )
            or type(checkpoint_value.get("model_state_tensors")) is not int
            or checkpoint_value["model_state_tensors"] <= 0
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(checkpoint_value.get("model_state_schema_sha256")),
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(checkpoint_value.get("model_state_semantic_sha256")),
            )
            is None
            or not isinstance(manifest_value, dict)
            or set(manifest_value)
            != {
                "path",
                "sha256_at_ready",
                "entries_sha256_at_ready",
                "immutable_snapshot",
                "live_path",
            }
            or manifest_value.get("immutable_snapshot") is not True
            or not isinstance(frozen_value, dict)
            or set(frozen_value)
            != {"path", "sha256", "receipt_payload_sha256"}
            or not isinstance(protocol, dict)
            or set(protocol) != {"format", "payload_sha256"}
            or protocol.get("format") != contract.PROTOCOL_FORMAT
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key)))
                is None
                for key in (
                    "frozen_receipt_sha256",
                    "schedule_sha256",
                    "trajectory_anchor_sha256",
                )
            )
        ):
            raise TopologySelectionError(f"{label} protocol changed")

        checkpoint, _ = _artifact(
            {
                key: checkpoint_value[key]
                for key in ("path", "sha256", "bytes")
            },
            f"{label} checkpoint",
        )
        manifest_path = Path(str(manifest_value["path"]))
        artifact_root = Path(payload["artifact_root"])
        expected_checkpoint_path = (
            artifact_root
            / "provisional_candidates"
            / f"base_official_adapt_short_quality_epoch_{epoch:02d}.bin"
        )
        expected_snapshot_path = (
            artifact_root
            / "short_quality_candidate_manifest_snapshots"
            / f"epoch-{epoch:04d}.json"
        )
        expected_ready_path = (
            artifact_root
            / "short_quality_candidate_receipts"
            / f"epoch-{epoch:04d}.json"
        )
        if (
            not artifact_root.is_dir()
            or artifact_root.is_symlink()
            or artifact_root.resolve(strict=True) != artifact_root
            or Path(ready["path"]) != expected_ready_path
            or Path(checkpoint["path"]) != expected_checkpoint_path
            or manifest_path != expected_snapshot_path
            or Path(str(manifest_value["live_path"]))
            != artifact_root / "short_quality_candidate_manifest.json"
            or Path(str(frozen_value["path"]))
            != artifact_root / "frozen_inputs.json"
        ):
            raise TopologySelectionError(f"{label} v3 artifact root changed")
        if not manifest_path.is_absolute():
            raise TopologySelectionError(f"{label} manifest path changed")
        try:
            manifest_resolved, manifest_bytes, _manifest_identity = (
                contract._read_regular_file_bytes(
                    manifest_path,
                    f"{label} manifest",
                )
            )
        except Exception as error:
            raise TopologySelectionError(f"{label} manifest is absent") from error
        if (
            manifest_resolved != manifest_path
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or hashlib.sha256(manifest_bytes).hexdigest()
            != manifest_value["sha256_at_ready"]
        ):
            raise TopologySelectionError(
                f"{label} manifest is not canonical"
            )
        try:
            manifest = contract._strict_json_bytes(
                manifest_bytes,
                f"{label} manifest",
            )
        except Exception as error:
            raise TopologySelectionError(
                f"{label} manifest is not strict JSON"
            ) from error
        entries = manifest.get("entries")
        expected_entry_epochs = list(quality_epochs[: position + 1])
        if (
            set(manifest) != manifest_keys
            or manifest.get("format")
            != contract.SHORT_QUALITY_MANIFEST_FORMAT
            or manifest.get("status") != "running"
            or manifest.get("run_purpose")
            != contract.RUN_PURPOSE_SHORT_QUALITY
            or manifest.get("target_epochs") != list(quality_epochs)
            or manifest.get("candidate_epochs") != list(quality_epochs)
            or manifest.get("quality_protocol_version")
            != QUALITY_PROTOCOL_VERSION
            or manifest.get("artifact_root_namespace")
            != QUALITY_ARTIFACT_ROOT_NAMESPACE
            or manifest.get("artifact_root") != payload["artifact_root"]
            or manifest.get("quality_role") != quality_role
            or manifest.get("reference_only") is not reference_only
            or manifest.get("late_w1_status") != "not_measured"
            or manifest.get("w1_tail_equivalence_claimed") is not False
            or manifest.get("frozen_receipt_sha256")
            != payload["frozen_receipt_sha256"]
            or manifest.get("schedule_sha256") != payload["schedule_sha256"]
            or manifest.get("trajectory_anchor_sha256")
            != payload["trajectory_anchor_sha256"]
            or manifest.get("trajectory_mode")
            != contract.FRESH_TRAJECTORY_MODE
            or manifest.get("trajectory_probe_verified") is not True
            or not isinstance(entries, list)
            or [entry.get("epoch") for entry in entries]
            != expected_entry_epochs
            or manifest.get("entries_sha256")
            != contract.canonical_json_sha256(entries)
            or manifest_value.get("entries_sha256_at_ready")
            != manifest.get("entries_sha256")
        ):
            raise TopologySelectionError(f"{label} manifest changed")
        entry = entries[-1]
        if (
            not isinstance(entry, dict)
            or set(entry) != entry_keys
            or entry.get("epoch") != epoch
            or entry.get("optimizer_updates") != expected_optimizer_updates
            or entry.get("checkpoint")
            != checkpoint_value["relative_path"]
            or entry.get("checkpoint_sha256") != checkpoint["sha256"]
            or entry.get("checkpoint_bytes") != checkpoint["bytes"]
            or entry.get("checkpoint_container_schema")
            != ["audit", "model_state"]
            or entry.get("model_state_tensors")
            != checkpoint_value["model_state_tensors"]
            or entry.get("model_state_schema_sha256")
            != checkpoint_value["model_state_schema_sha256"]
            or entry.get("model_state_semantic_sha256")
            != checkpoint_value["model_state_semantic_sha256"]
            or entry.get("all_model_state_tensors_finite") is not True
            or entry.get("frozen_receipt_sha256")
            != payload["frozen_receipt_sha256"]
            or entry.get("trajectory_anchor_match") is not None
            or entry.get("run_purpose")
            != contract.RUN_PURPOSE_SHORT_QUALITY
            or entry.get("trajectory_probe_verified") is not True
        ):
            raise TopologySelectionError(
                f"{label} manifest checkpoint changed"
            )

        frozen, _frozen_path, frozen_sha = contract._load_json_receipt(
            Path(str(frozen_value["path"])),
            str(frozen_value["sha256"]),
            f"{label} frozen inputs",
        )
        throughput = manifest.get("throughput_gate")
        frozen_protocol = frozen.get("protocol")
        frozen_long_contract = frozen.get("long_contract")
        frozen_schedule = (
            frozen_long_contract.get("schedule")
            if isinstance(frozen_long_contract, dict)
            else None
        )
        frozen_trajectory = (
            frozen_long_contract.get("trajectory_anchor")
            if isinstance(frozen_long_contract, dict)
            else None
        )
        if (
            frozen_sha != frozen_value["sha256"]
            or set(frozen) != frozen_keys
            or frozen.get("format")
            != "semtalk_show_base_official_adapt_frozen_inputs_v1"
            or frozen.get("run_purpose")
            != contract.RUN_PURPOSE_SHORT_QUALITY
            or frozen.get("target_epochs") != list(quality_epochs)
            or frozen.get("receipt_sha256")
            != frozen_value["receipt_payload_sha256"]
            or frozen.get("receipt_sha256")
            != payload["frozen_receipt_sha256"]
            or not isinstance(frozen_protocol, dict)
            or frozen_protocol.get("format") != contract.PROTOCOL_FORMAT
            or protocol
            != {
                "format": frozen_protocol["format"],
                "payload_sha256": contract.canonical_json_sha256(
                    frozen_protocol
                ),
            }
            or not isinstance(frozen_long_contract, dict)
            or frozen_long_contract.get("format")
            != "semtalk_show_base_long_contract_receipts_v1"
            or not isinstance(frozen_schedule, dict)
            or frozen_schedule.get("sha256") != payload["schedule_sha256"]
            or not isinstance(frozen_trajectory, dict)
            or frozen_trajectory.get("sha256")
            != payload["trajectory_anchor_sha256"]
            or not isinstance(throughput, dict)
            or set(throughput) != throughput_keys
            or throughput.get("topology_mode") != mode
            or throughput.get("trajectory_mode")
            != contract.FRESH_TRAJECTORY_MODE
            or throughput.get("trajectory_probe")
            != manifest.get("trajectory_probe")
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(throughput.get("gate_frozen_receipt_sha256")),
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(throughput.get("frozen_gate_compatibility_sha256")),
            )
            is None
        ):
            raise TopologySelectionError(
                f"{label} frozen/topology authority changed"
            )

        try:
            full_probe, full_probe_path, full_probe_sha = (
                contract._load_json_receipt(
                    Path(str(throughput["path"])),
                    str(throughput["sha256"]),
                    f"{label} throughput gate",
                )
            )
            normalized_trajectory = _fresh_trajectory_probe(
                mode,
                throughput["trajectory_probe"],
                f"{label} trajectory probe",
            )
            normalized_probe = validate_probe(
                mode,
                Path(str(throughput["path"])),
                str(throughput["sha256"]),
                gate_spec_sha256=topology_gate_spec_sha256,
            )
        except Exception as error:
            raise TopologySelectionError(
                f"{label} frozen/topology authority changed"
            ) from error

        numeric_projection = (
            "samples_per_second",
            "seconds_per_update",
            "median_seconds",
            "p90_seconds",
            "p99_seconds",
            "estimated_training_seconds",
        )
        if (
            full_probe_path != Path(str(throughput["path"]))
            or full_probe_sha != throughput["sha256"]
            or normalized_probe["report_path"] != throughput["path"]
            or normalized_probe["report_sha256"] != throughput["sha256"]
            or full_probe.get("frozen_receipt_sha256")
            != throughput["gate_frozen_receipt_sha256"]
            or full_probe.get("frozen_gate_compatibility_sha256")
            != throughput["frozen_gate_compatibility_sha256"]
            or throughput["frozen_gate_compatibility_sha256"]
            != contract._frozen_gate_compatibility_sha256(frozen)
            or full_probe.get("topology_receipt_sha256")
            != frozen.get("topology", {}).get("receipt_sha256")
            or full_probe.get("topology_independent_input_sha256")
            != contract._topology_independent_gate_semantic_sha256(frozen)
            or full_probe.get("updates_per_epoch") != expected_updates
            or full_probe.get("trajectory_mode")
            != contract.FRESH_TRAJECTORY_MODE
            or full_probe.get("trajectory_probe")
            != throughput["trajectory_probe"]
            or normalized_trajectory != throughput["trajectory_probe"]
            or any(
                not isinstance(throughput.get(key), (int, float))
                or isinstance(throughput.get(key), bool)
                or not math.isfinite(float(throughput[key]))
                or float(throughput[key]) <= 0.0
                or float(throughput[key]) != float(full_probe.get(key))
                for key in numeric_projection
            )
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(full_probe.get("topology_independent_input_sha256")),
            )
            is None
        ):
            raise TopologySelectionError(
                f"{label} frozen/topology authority changed"
            )
        current_semantic = full_probe[
            "topology_independent_input_sha256"
        ]
        current_common = {
            "frozen_receipt_sha256": payload["frozen_receipt_sha256"],
            "schedule_sha256": payload["schedule_sha256"],
            "trajectory_anchor_sha256": payload[
                "trajectory_anchor_sha256"
            ],
            "protocol": protocol,
            "throughput_gate_sha256": throughput["sha256"],
            "frozen_gate_compatibility_sha256": throughput[
                "frozen_gate_compatibility_sha256"
            ],
        }
        if semantic_sha is None:
            semantic_sha = current_semantic
            common = current_common
        elif semantic_sha != current_semantic or common != current_common:
            raise TopologySelectionError(
                f"{mode} candidate-ready receipts mix training authorities"
            )
        ready_artifacts.append(ready)
        checkpoints[epoch] = checkpoint
    assert semantic_sha is not None
    return ready_artifacts, semantic_sha, checkpoints


def validate_short_quality_training_bundle(
    mode: str,
    value: Any,
    candidate_ready_receipts: Sequence[Any],
    *,
    topology_gate_spec_sha256: str,
    quality_gate_spec_sha256: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    dict[int, dict[str, Any]],
]:
    """Validate one completed mode-specific v3 provisional bundle."""

    artifact, status = _artifact(
        value,
        f"{mode} short-quality training status",
        payload=True,
    )
    assert status is not None
    quality_epochs = quality_epochs_for_mode(mode)
    quality_role = quality_role_for_mode(mode)
    reference_only = mode == contract.OFFICIAL_W1_REFERENCE_MODE
    quality_total_epochs = quality_epochs[-1]
    expected_updates = int(contract.TOPOLOGY_SPECS[mode]["updates_per_epoch"])
    required = {
        "format",
        "status",
        "run_purpose",
        "target_epochs",
        "quality_protocol_version",
        "artifact_root_namespace",
        "artifact_root",
        "quality_role",
        "reference_only",
        "late_w1_status",
        "w1_tail_equivalence_claimed",
        "completed_epochs",
        "optimizer_updates",
        "updates_per_epoch",
        "candidate_manifest_sha256",
        "frozen_receipt_sha256",
        "throughput_gate",
        "world_size",
        "local_batch_size",
        "global_batch_size",
        "all_training_state_finite",
        "epoch_metrics_jsonl",
        "epoch_metrics_sha256",
        "epoch_metrics_records",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "trajectory_mode",
        "trajectory_probe_verified",
        "trajectory_probe",
        "started_unix",
        "completed_unix",
        "candidate_manifest",
        "epoch_metrics",
        "candidate_ready_receipts",
        "receipt_payload_sha256",
    }
    manifest_value = status.get("candidate_manifest")
    metrics_value = status.get("epoch_metrics")
    embedded_ready = status.get("candidate_ready_receipts")
    if (
        set(status) != required
        or status.get("format") != contract.SHORT_QUALITY_STATUS_FORMAT
        or status.get("status") != "complete"
        or status.get("run_purpose")
        != contract.RUN_PURPOSE_SHORT_QUALITY
        or status.get("target_epochs") != list(quality_epochs)
        or status.get("quality_protocol_version") != QUALITY_PROTOCOL_VERSION
        or status.get("artifact_root_namespace")
        != QUALITY_ARTIFACT_ROOT_NAMESPACE
        or not isinstance(status.get("artifact_root"), str)
        or not Path(status["artifact_root"]).is_absolute()
        or status.get("quality_role") != quality_role
        or status.get("reference_only") is not reference_only
        or status.get("late_w1_status") != "not_measured"
        or status.get("w1_tail_equivalence_claimed") is not False
        or status.get("completed_epochs")
        != quality_total_epochs
        or status.get("optimizer_updates")
        != quality_total_epochs * expected_updates
        or status.get("updates_per_epoch") != expected_updates
        or status.get("world_size")
        != contract.TOPOLOGY_SPECS[mode]["world_size"]
        or status.get("local_batch_size")
        != contract.TOPOLOGY_SPECS[mode]["local_batch_size"]
        or status.get("global_batch_size")
        != contract.TOPOLOGY_SPECS[mode]["global_batch_size"]
        or status.get("all_training_state_finite") is not True
        or status.get("epoch_metrics_records")
        != quality_total_epochs
        or status.get("trajectory_mode")
        != contract.FRESH_TRAJECTORY_MODE
        or status.get("trajectory_probe_verified") is not True
        or not isinstance(manifest_value, dict)
        or set(manifest_value) != {"path", "sha256", "bytes"}
        or not isinstance(metrics_value, dict)
        or set(metrics_value) != {"path", "sha256", "bytes", "records"}
        or metrics_value.get("records")
        != quality_total_epochs
        or not isinstance(embedded_ready, list)
    ):
        raise TopologySelectionError(
            f"{mode} provisional short-quality status changed"
        )

    ready, semantic_sha, checkpoints = validate_candidate_ready_receipts(
        mode,
        candidate_ready_receipts,
        topology_gate_spec_sha256=topology_gate_spec_sha256,
        quality_gate_spec_sha256=quality_gate_spec_sha256,
    )
    if embedded_ready != ready:
        raise TopologySelectionError(
            f"{mode} provisional candidate-ready authority changed"
        )
    manifest_artifact, _ = _artifact(
        manifest_value,
        f"{mode} final short-quality manifest",
    )
    metrics_artifact, _ = _artifact(
        {
            key: metrics_value[key]
            for key in ("path", "sha256", "bytes")
        },
        f"{mode} short-quality metrics",
    )
    if (
        status["candidate_manifest_sha256"]
        != manifest_artifact["sha256"]
        or status["epoch_metrics_jsonl"] != metrics_artifact["path"]
        or status["epoch_metrics_sha256"] != metrics_artifact["sha256"]
    ):
        raise TopologySelectionError(
            f"{mode} provisional output identity changed"
        )
    manifest_path = Path(str(manifest_artifact["path"]))
    artifact_root = Path(status["artifact_root"])
    if (
        not artifact_root.is_dir()
        or artifact_root.is_symlink()
        or artifact_root.resolve(strict=True) != artifact_root
        or Path(artifact["path"])
        != artifact_root / "short_quality_status.json"
        or manifest_path
        != artifact_root / "short_quality_candidate_manifest.json"
        or Path(metrics_artifact["path"])
        != artifact_root / "epoch_metrics.jsonl"
    ):
        raise TopologySelectionError(
            f"{mode} short-quality v3 artifact root changed"
        )
    try:
        manifest = contract._strict_json_bytes(
            manifest_path.read_bytes(),
            f"{mode} final short-quality manifest",
        )
    except Exception as error:
        raise TopologySelectionError(
            f"{mode} final short-quality manifest is invalid"
        ) from error
    manifest_required = {
        "format",
        "status",
        "run_purpose",
        "target_epochs",
        "candidate_epochs",
        "quality_protocol_version",
        "artifact_root_namespace",
        "artifact_root",
        "quality_role",
        "reference_only",
        "late_w1_status",
        "w1_tail_equivalence_claimed",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "throughput_gate",
        "trajectory_mode",
        "trajectory_probe_verified",
        "trajectory_probe",
        "entries",
        "entries_sha256",
        "completed_epochs",
        "optimizer_updates",
    }
    entries = manifest.get("entries")
    if (
        set(manifest) != manifest_required
        or manifest.get("format")
        != contract.SHORT_QUALITY_MANIFEST_FORMAT
        or manifest.get("status") != "complete"
        or manifest.get("run_purpose")
        != contract.RUN_PURPOSE_SHORT_QUALITY
        or manifest.get("target_epochs") != list(quality_epochs)
        or manifest.get("candidate_epochs") != list(quality_epochs)
        or manifest.get("quality_protocol_version")
        != QUALITY_PROTOCOL_VERSION
        or manifest.get("artifact_root_namespace")
        != QUALITY_ARTIFACT_ROOT_NAMESPACE
        or manifest.get("artifact_root") != status["artifact_root"]
        or manifest.get("quality_role") != quality_role
        or manifest.get("reference_only") is not reference_only
        or manifest.get("late_w1_status") != "not_measured"
        or manifest.get("w1_tail_equivalence_claimed") is not False
        or manifest.get("completed_epochs")
        != quality_total_epochs
        or manifest.get("optimizer_updates")
        != quality_total_epochs * expected_updates
        or manifest.get("frozen_receipt_sha256")
        != status["frozen_receipt_sha256"]
        or manifest.get("schedule_sha256") != status["schedule_sha256"]
        or manifest.get("trajectory_anchor_sha256")
        != status["trajectory_anchor_sha256"]
        or manifest.get("throughput_gate") != status["throughput_gate"]
        or manifest.get("trajectory_probe") != status["trajectory_probe"]
        or manifest.get("trajectory_probe_verified") is not True
        or not isinstance(entries, list)
        or [entry.get("epoch") for entry in entries]
        != list(quality_epochs)
        or manifest.get("entries_sha256")
        != contract.canonical_json_sha256(entries)
    ):
        raise TopologySelectionError(
            f"{mode} final short-quality manifest changed"
        )
    for epoch, entry in zip(quality_epochs, entries):
        checkpoint = checkpoints[epoch]
        if (
            entry.get("checkpoint_sha256") != checkpoint["sha256"]
            or entry.get("checkpoint_bytes") != checkpoint["bytes"]
            or entry.get("run_purpose")
            != contract.RUN_PURPOSE_SHORT_QUALITY
        ):
            raise TopologySelectionError(
                f"{mode} final short-quality checkpoint changed"
            )

    rows = _strict_jsonl_rows(
        metrics_artifact,
        f"{mode} short-quality metrics",
    )
    metric_keys = {*contract.LOSS_COMPONENTS, "total", "gradient_norm_preclip"}
    for epoch, row in enumerate(rows, 1):
        metrics = row.get("metrics") if isinstance(row, dict) else None
        if (
            set(row)
            != {
                "format",
                "run_purpose",
                "target_epochs",
                "quality_protocol_version",
                "artifact_root_namespace",
                "artifact_root",
                "quality_role",
                "reference_only",
                "late_w1_status",
                "w1_tail_equivalence_claimed",
                "epoch",
                "optimizer_updates",
                "updates_per_epoch",
                "learning_rate",
                "metrics",
                "all_finite",
                "completed_unix",
            }
            or row.get("format")
            != contract.SHORT_QUALITY_EPOCH_METRIC_FORMAT
            or row.get("run_purpose")
            != contract.RUN_PURPOSE_SHORT_QUALITY
            or row.get("target_epochs") != list(quality_epochs)
            or row.get("quality_protocol_version")
            != QUALITY_PROTOCOL_VERSION
            or row.get("artifact_root_namespace")
            != QUALITY_ARTIFACT_ROOT_NAMESPACE
            or row.get("artifact_root") != status["artifact_root"]
            or row.get("quality_role") != quality_role
            or row.get("reference_only") is not reference_only
            or row.get("late_w1_status") != "not_measured"
            or row.get("w1_tail_equivalence_claimed") is not False
            or row.get("epoch") != epoch
            or row.get("optimizer_updates") != epoch * expected_updates
            or row.get("updates_per_epoch") != expected_updates
            or row.get("all_finite") is not True
            or not isinstance(metrics, dict)
            or set(metrics) != metric_keys
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in metrics.values()
            )
        ):
            raise TopologySelectionError(
                f"{mode} short-quality metrics are not exact e1..e{quality_total_epochs}"
            )
    if len(rows) != quality_total_epochs:
        raise TopologySelectionError(
            f"{mode} short-quality metrics are not exact e1..e{quality_total_epochs}"
        )
    return artifact, ready, semantic_sha, checkpoints


def validate_short_trajectory_receipt(
    mode: str,
    value: Any,
    *,
    topology_gate_spec_sha256: str,
    quality_gate_spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    artifact, payload = _artifact(
        value,
        f"{mode} short trajectory",
        payload=True,
    )
    assert payload is not None
    quality_epochs = quality_epochs_for_mode(mode)
    quality_role = quality_role_for_mode(mode)
    reference_only = mode == contract.OFFICIAL_W1_REFERENCE_MODE
    required = {
        "format",
        "status",
        "topology_mode",
        "topology_gate_spec_sha256",
        "quality_gate_spec_sha256",
        "quality_protocol_version",
        "artifact_root_namespace",
        "artifact_root",
        "quality_role",
        "reference_only",
        "late_w1_status",
        "w1_tail_equivalence_claimed",
        "candidate_epochs",
        "split",
        "test_visible",
        "topology_independent_input_sha256",
        "candidate_ready_receipts",
        "candidates",
        "receipt_payload_sha256",
    }
    candidates = payload.get("candidates")
    candidate_ready_receipts = payload.get("candidate_ready_receipts")
    if (
        set(payload) != required
        or payload.get("format") != SHORT_TRAJECTORY_FORMAT
        or payload.get("status") != "complete"
        or payload.get("topology_mode") != mode
        or payload.get("topology_gate_spec_sha256")
        != topology_gate_spec_sha256
        or payload.get("quality_gate_spec_sha256")
        != quality_gate_spec_sha256
        or payload.get("quality_protocol_version") != QUALITY_PROTOCOL_VERSION
        or payload.get("artifact_root_namespace")
        != QUALITY_ARTIFACT_ROOT_NAMESPACE
        or not isinstance(payload.get("artifact_root"), str)
        or not Path(payload["artifact_root"]).is_absolute()
        or payload.get("quality_role") != quality_role
        or payload.get("reference_only") is not reference_only
        or payload.get("late_w1_status") != "not_measured"
        or payload.get("w1_tail_equivalence_claimed") is not False
        or payload.get("candidate_epochs") != list(quality_epochs)
        or payload.get("split") != "val"
        or payload.get("test_visible") is not False
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(payload.get("topology_independent_input_sha256")),
        )
        is None
        or not isinstance(candidates, list)
        or len(candidates) != len(quality_epochs)
        or not isinstance(candidate_ready_receipts, list)
        or len(candidate_ready_receipts) != len(quality_epochs)
    ):
        raise TopologySelectionError(f"{mode} short trajectory changed")
    ready, ready_semantic, ready_checkpoints = (
        validate_candidate_ready_receipts(
            mode,
            candidate_ready_receipts,
            topology_gate_spec_sha256=topology_gate_spec_sha256,
            quality_gate_spec_sha256=quality_gate_spec_sha256,
        )
    )
    if (
        ready != candidate_ready_receipts
        or ready_semantic
        != payload["topology_independent_input_sha256"]
    ):
        raise TopologySelectionError(
            f"{mode} short trajectory candidate-ready authority changed"
        )
    first_ready_artifact, first_ready_payload = _artifact(
        candidate_ready_receipts[0],
        f"{mode} first candidate-ready root authority",
        payload=True,
    )
    assert first_ready_payload is not None
    artifact_root = Path(payload["artifact_root"])
    if (
        first_ready_artifact != ready[0]
        or first_ready_payload.get("artifact_root") != payload["artifact_root"]
        or not artifact_root.is_dir()
        or artifact_root.is_symlink()
        or artifact_root.resolve(strict=True) != artifact_root
    ):
        raise TopologySelectionError(
            f"{mode} short trajectory v3 artifact root changed"
        )
    expected_updates = int(contract.TOPOLOGY_SPECS[mode]["updates_per_epoch"])
    normalized: dict[int, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for expected_epoch, row in zip(quality_epochs, candidates):
        if not isinstance(row, dict) or set(row) != {
            "epoch",
            "optimizer_updates",
            "candidate_checkpoint",
        }:
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} short checkpoint schema mismatch"
            )
        checkpoint, _ = _artifact(
            row["candidate_checkpoint"],
            f"{mode} e{expected_epoch} short checkpoint",
        )
        if (
            row.get("epoch") != expected_epoch
            or row.get("optimizer_updates")
            != expected_epoch * expected_updates
            or checkpoint != ready_checkpoints[expected_epoch]
            or checkpoint["path"] in seen_paths
            or checkpoint["sha256"] in seen_hashes
        ):
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} short checkpoint identity changed"
            )
        seen_paths.add(checkpoint["path"])
        seen_hashes.add(checkpoint["sha256"])
        normalized[expected_epoch] = checkpoint
    return artifact, payload, normalized


def _formal_artifact_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": value["path"],
        "sha256": value["sha256"],
        "receipt_payload_sha256": value["receipt_payload_sha256"],
    }


def _validate_formal_receipt(
    value: Any,
    label: str,
    validator: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_artifact, _payload = _artifact(value, label, payload=True)
    try:
        formal_artifact, validated = validator(
            Path(source_artifact["path"]), source_artifact["sha256"]
        )
    except Exception as error:
        raise TopologySelectionError(f"{label} validation failed") from error
    if formal_artifact != _formal_artifact_projection(source_artifact):
        raise TopologySelectionError(f"{label} identity changed")
    return source_artifact, formal_artifact, validated


def validate_quality_common_authority(
    val_inputs_receipt: Any,
    pipeline_receipt: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Replay the exact formal val-input and inference-pipeline authority."""

    val_source, val_artifact, coverage = _validate_formal_receipt(
        val_inputs_receipt,
        "topology quality val-input receipt",
        formal_validation.validate_val_inputs,
    )
    pipeline_source, pipeline_artifact, pipeline = _validate_formal_receipt(
        pipeline_receipt,
        "topology quality pipeline receipt",
        formal_validation.validate_pipeline,
    )
    return (
        val_source,
        val_artifact,
        coverage,
        pipeline_source,
        pipeline_artifact,
        pipeline,
    )


def _strict_diffsheg_report(
    value: Any,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, _ = _artifact(value, label)
    path = Path(artifact["path"])
    try:
        formal_validation.reject_test_path(path, label)
        payload = path.read_bytes()
        if (
            len(payload) != artifact["bytes"]
            or hashlib.sha256(payload).hexdigest() != artifact["sha256"]
        ):
            raise TopologySelectionError(f"{label} changed during validation")
        report = formal_validation._strict_json_bytes(payload, label)
    except TopologySelectionError:
        raise
    except Exception as error:
        raise TopologySelectionError(f"{label} validation failed") from error
    return artifact, report


def validate_quality_candidate_provenance(
    mode: str,
    epoch: int,
    *,
    topology_independent_input_sha256: str,
    short_trajectory_receipt: Mapping[str, Any],
    expected_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Any,
    val_inputs_receipt: Mapping[str, Any],
    val_inputs_artifact: Mapping[str, Any],
    pipeline_receipt: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
    pipeline_payload: Mapping[str, Any],
    expected_coverage: Mapping[str, Any],
    inference_lineage: Any,
    diffsheg_report: Any,
) -> dict[str, Any]:
    """Bind one short checkpoint to the formal DiffSHEG val-FGD chain."""

    label = f"{mode} e{epoch}"
    short_artifact, short_payload = _artifact(
        short_trajectory_receipt,
        f"{label} short trajectory provenance authority",
        payload=True,
    )
    assert short_payload is not None
    quality_role = quality_role_for_mode(mode)
    reference_only = mode == contract.OFFICIAL_W1_REFERENCE_MODE
    if (
        short_artifact != dict(short_trajectory_receipt)
        or short_payload.get("quality_protocol_version")
        != QUALITY_PROTOCOL_VERSION
        or short_payload.get("artifact_root_namespace")
        != QUALITY_ARTIFACT_ROOT_NAMESPACE
        or not isinstance(short_payload.get("artifact_root"), str)
        or not Path(short_payload["artifact_root"]).is_absolute()
        or short_payload.get("quality_role") != quality_role
        or short_payload.get("reference_only") is not reference_only
        or short_payload.get("late_w1_status") != "not_measured"
        or short_payload.get("w1_tail_equivalence_claimed") is not False
    ):
        raise TopologySelectionError(
            f"{label} short trajectory v3 provenance authority changed"
        )
    checkpoint, _ = _artifact(candidate_checkpoint, f"{label} checkpoint")
    if checkpoint != dict(expected_checkpoint):
        raise TopologySelectionError(
            f"{label} checkpoint differs from short trajectory"
        )
    if (
        _formal_artifact_projection(val_inputs_receipt)
        != dict(val_inputs_artifact)
        or _formal_artifact_projection(pipeline_receipt)
        != dict(pipeline_artifact)
    ):
        raise TopologySelectionError(f"{label} common validation authority changed")

    lineage_source, _ = _artifact(
        inference_lineage,
        f"{label} inference lineage",
        payload=True,
    )
    try:
        lineage_artifact, lineage = (
            formal_validation.validate_val_inference_lineage(
                Path(lineage_source["path"]),
                lineage_source["sha256"],
                epoch=epoch,
                expected_candidate=checkpoint,
                val_inputs_artifact=val_inputs_artifact,
                pipeline_artifact=pipeline_artifact,
                expected_coverage=expected_coverage,
            )
        )
    except Exception as error:
        raise TopologySelectionError(
            f"{label} DiffSHEG inference lineage validation failed"
        ) from error
    if lineage_artifact != _formal_artifact_projection(lineage_source):
        raise TopologySelectionError(f"{label} inference lineage identity changed")

    report_artifact, report = _strict_diffsheg_report(
        diffsheg_report,
        f"{label} DiffSHEG report",
    )
    try:
        metrics, coverage = formal_validation.validate_diffsheg_report(
            report,
            expected_coverage=expected_coverage,
            inference_lineage=lineage,
            expected_pipeline=pipeline_payload,
        )
    except Exception as error:
        raise TopologySelectionError(
            f"{label} DiffSHEG report validation failed"
        ) from error
    value = float(metrics["fgd"])
    if (
        not math.isfinite(value)
        or value < 0.0
        or coverage.get("clip_count") != EXPECTED_VAL_CLIPS
    ):
        raise TopologySelectionError(f"{label} DiffSHEG FGD is invalid")

    chain = {
        "short_trajectory_receipt": dict(short_trajectory_receipt),
        "candidate_checkpoint": checkpoint,
        "val_inputs_receipt": dict(val_inputs_receipt),
        "pipeline_receipt": dict(pipeline_receipt),
        "inference_lineage": lineage_source,
        "diffsheg_report": report_artifact,
    }
    provenance = {
        "format": QUALITY_PROVENANCE_FORMAT,
        "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
        "artifact_root_namespace": QUALITY_ARTIFACT_ROOT_NAMESPACE,
        "artifact_root": short_payload["artifact_root"],
        "quality_role": quality_role,
        "reference_only": reference_only,
        "late_w1_status": "not_measured",
        "w1_tail_equivalence_claimed": False,
        "topology_mode": mode,
        "epoch": epoch,
        "split": "val",
        "test_visible": False,
        "clip_count": coverage["clip_count"],
        "topology_independent_input_sha256": (
            topology_independent_input_sha256
        ),
        "short_trajectory_receipt_sha256": short_trajectory_receipt[
            "sha256"
        ],
        "short_trajectory_receipt_payload_sha256": (
            short_trajectory_receipt["receipt_payload_sha256"]
        ),
        "checkpoint_sha256": checkpoint["sha256"],
        "val_inputs_receipt_sha256": val_inputs_receipt["sha256"],
        "val_inputs_receipt_payload_sha256": val_inputs_receipt[
            "receipt_payload_sha256"
        ],
        "pipeline_receipt_sha256": pipeline_receipt["sha256"],
        "pipeline_receipt_payload_sha256": pipeline_receipt[
            "receipt_payload_sha256"
        ],
        "inference_lineage_sha256": lineage_source["sha256"],
        "inference_lineage_payload_sha256": lineage_source[
            "receipt_payload_sha256"
        ],
        "diffsheg_report_sha256": report_artifact["sha256"],
        "primary_metric": PRIMARY_METRIC_PATH,
        "validation_protocol": VALIDATION_PROTOCOL,
        "chain_sha256": contract.canonical_json_sha256(chain),
    }
    return {
        "candidate_checkpoint": checkpoint,
        "inference_lineage": lineage_source,
        "diffsheg_report": report_artifact,
        "provenance": provenance,
        "diffsheg_fgd": value,
    }


def validate_quality_gate_spec(
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    payload, resolved, observed = contract._load_json_receipt(
        path,
        expected_sha256,
        "Base topology quality gate specification",
    )
    expected = {
        "format": QUALITY_GATE_FORMAT,
        "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
        "artifact_root_namespace": QUALITY_ARTIFACT_ROOT_NAMESPACE,
        "scope": "SemTalk Base on SHOW validation only",
        "split": "val",
        "test_visible": False,
        "w1_reference_epochs": list(W1_REFERENCE_EPOCHS),
        "candidate_quality_epochs": list(CANDIDATE_QUALITY_EPOCHS),
        "same_epoch_comparison_epochs": list(SAME_EPOCH_COMPARISON_EPOCHS),
        "tail_epochs": list(TAIL_EPOCHS),
        "primary_metric": PRIMARY_METRIC_PATH,
        "validation_protocol": VALIDATION_PROTOCOL,
        "diffsheg_validation_measurement_required": True,
        "comparison_reference": contract.OFFICIAL_W1_REFERENCE_MODE,
        "measured_eta_constraint": {
            "modes": list(contract.TOPOLOGY_SPECS),
            "maximum_estimated_training_seconds": MAX_TRAINING_SECONDS,
            "maximum_p99_training_seconds": MAX_P99_TRAINING_SECONDS,
            "p99_total_updates_required": True,
            "finite_probe_required": True,
            "policy": (
                "non_reference_quality_safe_finite_modes_compete_by_"
                "measured_eta_under_24h"
            ),
        },
        "w1_reference_gate": {
            "mode": contract.OFFICIAL_W1_REFERENCE_MODE,
            "role": "w1_reference_only",
            "selection_eligible": False,
            "reference_epochs": list(W1_REFERENCE_EPOCHS),
            "late_w1_status": "not_measured",
            "w1_tail_equivalence_claimed": False,
            "full400_eta_over_budget_policy": (
                "reference_only_and_never_selection_eligible"
            ),
        },
        "candidate_quality_gate": {
            "modes": [
                mode
                for mode in contract.TOPOLOGY_SPECS
                if mode != contract.OFFICIAL_W1_REFERENCE_MODE
            ],
            "role": "candidate_quality",
            "candidate_epochs": list(CANDIDATE_QUALITY_EPOCHS),
            "same_epoch_comparison_epochs": list(
                SAME_EPOCH_COMPARISON_EPOCHS
            ),
            "tail_epochs": list(TAIL_EPOCHS),
            "tail_reference_epochs": list(W1_REFERENCE_EPOCHS),
            "tail_reference_reducer": "minimum",
            "tail_maximum_allowed_function": (
                "reference_plus_max_absolute_or_relative_margin"
            ),
            "per_epoch_comparison": (
                "candidate_fgd_lte_reference_fgd_plus_max_of_"
                "absolute_or_relative_margin"
            ),
            "tail_absolute_envelope": (
                "candidate_fgd_lte_maximum_allowed_fgd_of_minimum_"
                "w1_reference_fgd"
            ),
            "maximum_absolute_fgd_regression": MAX_ABSOLUTE_FGD_REGRESSION,
            "maximum_relative_fgd_regression": MAX_RELATIVE_FGD_REGRESSION,
            "all_same_epoch_and_tail_epochs_must_pass": True,
            "late_w1_status": "not_measured",
            "w1_tail_equivalence_claimed": False,
        },
        "over_eta_budget_quality_skip": {
            "receipt_format": QUALITY_SKIP_FORMAT,
            "eligible_modes": [
                mode
                for mode in contract.TOPOLOGY_SPECS
                if mode != contract.OFFICIAL_W1_REFERENCE_MODE
            ],
            "condition": (
                "estimated_training_seconds_strictly_greater_than_"
                "maximum"
            ),
            "w1_quality_report_required": True,
            "within_budget_quality_report_required": True,
            "required_sha256_bindings": [
                "source_binding.frozen_receipt_sha256",
                "source_binding.frozen_gate_compatibility_sha256",
                "source_binding.topology_receipt_sha256",
                "topology_gate_spec_sha256",
                "quality_gate_spec_sha256",
                "probe_report.sha256",
                "topology_independent_input_sha256",
            ],
        },
        "selection_tiebreak": [
            "estimated_training_seconds",
            "p99_seconds",
            "topology_matrix_order",
        ],
    }
    if payload != expected:
        raise TopologySelectionError(
            "Base topology quality gate specification changed"
        )
    return {"path": str(resolved), "sha256": observed, "payload": payload}


def validate_quality_report(
    mode: str,
    path: Path,
    expected_sha256: str,
    *,
    quality_gate_spec_sha256: str,
    topology_gate_spec_sha256: str,
) -> dict[str, Any]:
    report, resolved, observed = contract._load_json_receipt(
        path,
        expected_sha256,
        f"Base topology quality report {mode}",
    )
    quality_epochs = quality_epochs_for_mode(mode)
    quality_role = quality_role_for_mode(mode)
    reference_only = mode == contract.OFFICIAL_W1_REFERENCE_MODE
    required = {
        "format",
        "status",
        "mode",
        "quality_gate_spec_sha256",
        "quality_protocol_version",
        "artifact_root_namespace",
        "artifact_root",
        "quality_role",
        "reference_only",
        "late_w1_status",
        "w1_tail_equivalence_claimed",
        "split",
        "test_visible",
        "trajectory_epochs",
        "topology_independent_input_sha256",
        "short_trajectory_receipt",
        "val_inputs_receipt",
        "pipeline_receipt",
        "candidates",
        "receipt_sha256",
    }
    candidates = report.get("candidates")
    if (
        not isinstance(report, dict)
        or set(report) != required
        or report.get("format") != QUALITY_REPORT_FORMAT
        or report.get("status") != "pass"
        or report.get("mode") != mode
        or report.get("quality_gate_spec_sha256")
        != quality_gate_spec_sha256
        or report.get("quality_protocol_version") != QUALITY_PROTOCOL_VERSION
        or report.get("artifact_root_namespace")
        != QUALITY_ARTIFACT_ROOT_NAMESPACE
        or not isinstance(report.get("artifact_root"), str)
        or not Path(report["artifact_root"]).is_absolute()
        or report.get("quality_role") != quality_role
        or report.get("reference_only") is not reference_only
        or report.get("late_w1_status") != "not_measured"
        or report.get("w1_tail_equivalence_claimed") is not False
        or report.get("split") != "val"
        or report.get("test_visible") is not False
        or report.get("trajectory_epochs") != list(quality_epochs)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(report.get("topology_independent_input_sha256")),
        )
        is None
        or not isinstance(candidates, list)
        or len(candidates) != len(quality_epochs)
        or report.get("receipt_sha256")
        != contract.canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "receipt_sha256"
            }
        )
    ):
        raise TopologySelectionError(
            f"Base topology quality report {mode} is forged or stale"
        )
    (
        short_trajectory,
        short_payload,
        short_checkpoints,
    ) = validate_short_trajectory_receipt(
        mode,
        report["short_trajectory_receipt"],
        topology_gate_spec_sha256=topology_gate_spec_sha256,
        quality_gate_spec_sha256=quality_gate_spec_sha256,
    )
    if (
        short_payload["topology_independent_input_sha256"]
        != report["topology_independent_input_sha256"]
        or short_payload["artifact_root"] != report["artifact_root"]
    ):
        raise TopologySelectionError(f"{mode} short trajectory changed")
    (
        val_inputs_source,
        val_inputs_artifact,
        coverage,
        pipeline_source,
        pipeline_artifact,
        pipeline_payload,
    ) = validate_quality_common_authority(
        report["val_inputs_receipt"],
        report["pipeline_receipt"],
    )
    values: dict[int, float] = {}
    normalized_rows: list[dict[str, Any]] = []
    for expected_epoch, row in zip(quality_epochs, candidates):
        row_keys = {
            "epoch",
            "candidate_checkpoint",
            "inference_lineage",
            "diffsheg_report",
            "provenance",
        }
        if not isinstance(row, dict) or set(row) != row_keys:
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} quality row schema mismatch"
            )
        if row.get("epoch") != expected_epoch:
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} validation identity changed"
            )
        validated = validate_quality_candidate_provenance(
            mode,
            expected_epoch,
            topology_independent_input_sha256=report[
                "topology_independent_input_sha256"
            ],
            short_trajectory_receipt=short_trajectory,
            expected_checkpoint=short_checkpoints[expected_epoch],
            candidate_checkpoint=row["candidate_checkpoint"],
            val_inputs_receipt=val_inputs_source,
            val_inputs_artifact=val_inputs_artifact,
            pipeline_receipt=pipeline_source,
            pipeline_artifact=pipeline_artifact,
            pipeline_payload=pipeline_payload,
            expected_coverage=coverage,
            inference_lineage=row["inference_lineage"],
            diffsheg_report=row["diffsheg_report"],
        )
        if row["provenance"] != validated["provenance"]:
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} provenance changed"
            )
        value = validated["diffsheg_fgd"]
        values[expected_epoch] = value
        normalized_rows.append(
            {
                "epoch": expected_epoch,
                **validated,
            }
        )
    return {
        "mode": mode,
        "report_path": str(resolved),
        "report_sha256": observed,
        "topology_independent_input_sha256": report[
            "topology_independent_input_sha256"
        ],
        "short_trajectory_receipt": short_trajectory,
        "val_inputs_receipt": val_inputs_source,
        "pipeline_receipt": pipeline_source,
        "candidate_fgd": {
            str(epoch): values[epoch] for epoch in quality_epochs
        },
        "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
        "artifact_root_namespace": QUALITY_ARTIFACT_ROOT_NAMESPACE,
        "artifact_root": report["artifact_root"],
        "quality_role": quality_role,
        "reference_only": reference_only,
        "late_w1_status": "not_measured",
        "w1_tail_equivalence_claimed": False,
        "candidates": normalized_rows,
    }


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def maximum_allowed_fgd(reference_fgd: float) -> float:
    if not math.isfinite(reference_fgd) or reference_fgd < 0.0:
        raise TopologySelectionError("reference FGD must be finite/nonnegative")
    return reference_fgd + max(
        MAX_ABSOLUTE_FGD_REGRESSION,
        abs(reference_fgd) * MAX_RELATIVE_FGD_REGRESSION,
    )


def validate_probe(
    mode: str,
    path: Path,
    expected_sha256: str,
    *,
    gate_spec_sha256: str,
) -> dict[str, Any]:
    specification = contract.TOPOLOGY_SPECS[mode]
    payload, resolved, observed_sha = contract._load_json_receipt(
        path,
        expected_sha256,
        f"Base topology probe {mode}",
    )
    sample_inventory = payload.get("sample_inventory")
    data_wait = payload.get("data_wait_seconds")
    collective = payload.get("collective_seconds")
    batchnorm = payload.get("batchnorm_inventory")
    rng = payload.get("rng_inventory")
    probe_epoch_updates = list(
        contract._probe_epoch_update_counts(
            int(specification["updates_per_epoch"])
        )
    )
    probe_epoch_samples = [
        updates * int(specification["global_batch_size"])
        for updates in probe_epoch_updates
    ]
    if (
        payload.get("format") != contract.GATE_FORMAT
        or payload.get("status") != "pass"
        or payload.get("topology_mode") != mode
        or payload.get("topology_classification")
        != specification["classification"]
        or payload.get("topology_gate_spec_sha256") != gate_spec_sha256
        or payload.get("node_count") != specification["node_count"]
        or payload.get("local_world_size")
        != specification["local_world_size"]
        or payload.get("world_size") != specification["world_size"]
        or payload.get("local_batch_size")
        != specification["local_batch_size"]
        or payload.get("global_batch_size")
        != specification["global_batch_size"]
        or payload.get("updates_per_epoch")
        != specification["updates_per_epoch"]
        or payload.get("unique_samples_per_epoch")
        != specification["unique_samples_per_epoch"]
        or float(payload.get("learning_rate", math.nan))
        != float(specification["learning_rate"])
        or payload.get("warmup_updates")
        != contract.THROUGHPUT_WARMUP_UPDATES
        or payload.get("timed_updates")
        != contract.THROUGHPUT_TIMED_UPDATES
        or payload.get("optimizer_updates")
        != contract.TRAJECTORY_PROBE_UPDATES
        or payload.get("precision") != specification["precision"]
        or payload.get("all_losses_finite") is not True
        or payload.get("all_gradients_finite") is not True
        or payload.get("oom") is not False
        or not all(
            _finite_positive(payload.get(key))
            for key in (
                "seconds_per_update",
                "median_seconds",
                "p90_seconds",
                "p99_seconds",
                "estimated_training_seconds",
                "samples_per_second",
            )
        )
        or not (
            float(payload["median_seconds"])
            <= float(payload["p90_seconds"])
            <= float(payload["p99_seconds"])
        )
        or float(payload["seconds_per_update"])
        != float(payload["median_seconds"])
        or float(payload["samples_per_second"])
        != (
            float(specification["global_batch_size"])
            / float(payload["median_seconds"])
        )
        or float(payload["estimated_training_seconds"])
        != (
            float(payload["median_seconds"])
            * int(specification["updates_per_epoch"])
            * contract.TOTAL_EPOCHS
        )
        or payload.get("estimated_epochs") != contract.TOTAL_EPOCHS
        or float(payload["p99_seconds"])
        > 4.0 * float(payload["median_seconds"])
        or not isinstance(payload.get("peak_cuda_memory_bytes_all_ranks"), list)
        or len(payload["peak_cuda_memory_bytes_all_ranks"])
        != specification["world_size"]
        or any(
            type(value) is not int or value <= 0
            for value in payload["peak_cuda_memory_bytes_all_ranks"]
        )
        or not isinstance(data_wait, dict)
        or not all(_finite_positive(data_wait.get(key)) for key in ("median", "p99"))
        or not isinstance(collective, dict)
        or collective.get("probe") != "ten_scalar_nccl_all_reduce_calls"
        or not all(
            _finite_positive(collective.get(key)) for key in ("median", "p99")
        )
        or not isinstance(batchnorm, list)
        or len(batchnorm) != specification["world_size"]
        or [item.get("rank") for item in batchnorm]
        != list(range(specification["world_size"]))
        or not batchnorm
        or any(item.get("before") != batchnorm[0].get("before") for item in batchnorm)
        or not isinstance(rng, list)
        or len(rng) != specification["world_size"]
        or [item.get("rank") for item in rng]
        != list(range(specification["world_size"]))
        or (
            specification["world_size"] > 1
            and len(
                {
                    item.get("torch_cuda_rng_state_sha256")
                    for item in rng
                }
            )
            != specification["world_size"]
        )
        or not isinstance(sample_inventory, dict)
        or sample_inventory.get("sampler_drop_last") is not True
        or sample_inventory.get("padding_duplicates") != 0
        or sample_inventory.get("probe_samples")
        != contract.TRAJECTORY_PROBE_UPDATES
        * specification["global_batch_size"]
        or type(sample_inventory.get("probe_unique_samples")) is not int
        or sample_inventory["probe_unique_samples"] <= 0
        or sample_inventory["probe_unique_samples"]
        > sample_inventory["probe_samples"]
        or sample_inventory.get("probe_sampler_epochs")
        != list(range(len(probe_epoch_updates)))
        or sample_inventory.get("probe_epoch_updates")
        != probe_epoch_updates
        or sample_inventory.get("probe_epoch_samples")
        != probe_epoch_samples
        or sample_inventory.get("probe_epoch_unique_samples")
        != probe_epoch_samples
        or type(sample_inventory.get("probe_cross_epoch_duplicates"))
        is not int
        or sample_inventory["probe_cross_epoch_duplicates"] < 0
        or sample_inventory["probe_unique_samples"]
        + sample_inventory["probe_cross_epoch_duplicates"]
        != sample_inventory["probe_samples"]
        or (
            len(probe_epoch_updates) == 1
            and sample_inventory["probe_cross_epoch_duplicates"] != 0
        )
        or sample_inventory.get("full_epoch_samples")
        != specification["unique_samples_per_epoch"]
        or sample_inventory.get("full_epoch_unique_samples")
        != specification["unique_samples_per_epoch"]
        or sample_inventory.get("dataset_samples")
        != contract.EXPECTED_TRAIN_SAMPLES
        or sample_inventory.get("dropped_tail_samples")
        != contract.EXPECTED_TRAIN_SAMPLES
        - specification["unique_samples_per_epoch"]
        or not isinstance(payload.get("topology_independent_input_sha256"), str)
        or len(payload["topology_independent_input_sha256"]) != 64
        or payload.get("receipt_sha256")
        != contract.canonical_json_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_sha256"
            }
        )
    ):
        raise TopologySelectionError(
            f"unsafe, incomplete, or stale topology probe: {mode}"
        )
    return {
        "mode": mode,
        "status": "pass",
        "report_path": str(resolved),
        "report_sha256": observed_sha,
        "classification": specification["classification"],
        "precision": specification["precision"],
        "formal_training_eligible": specification[
            "formal_training_eligible"
        ],
        "topology_independent_input_sha256": payload[
            "topology_independent_input_sha256"
        ],
        "median_seconds": float(payload["median_seconds"]),
        "p90_seconds": float(payload["p90_seconds"]),
        "p99_seconds": float(payload["p99_seconds"]),
        "estimated_training_seconds": float(
            payload["estimated_training_seconds"]
        ),
        "samples_per_second": float(payload["samples_per_second"]),
    }


def _validated_probe_receipt_binding(
    mode: str,
    path: Path,
    expected_sha256: str,
    *,
    gate_spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Return a validated probe plus its immutable source/hash binding."""

    probe = validate_probe(
        mode,
        path,
        expected_sha256,
        gate_spec_sha256=gate_spec_sha256,
    )
    raw, resolved, observed_sha = contract._load_json_receipt(
        path,
        expected_sha256,
        f"Base topology probe binding {mode}",
    )
    source_binding = {
        "frozen_receipt_sha256": raw.get("frozen_receipt_sha256"),
        "frozen_gate_compatibility_sha256": raw.get(
            "frozen_gate_compatibility_sha256"
        ),
        "topology_receipt_sha256": raw.get("topology_receipt_sha256"),
    }
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in source_binding.values()
    ) or re.fullmatch(
        r"[0-9a-f]{64}",
        str(probe.get("topology_independent_input_sha256")),
    ) is None:
        raise TopologySelectionError(
            f"Base topology probe {mode} lacks one immutable source binding"
        )
    try:
        canonical, data, _identity = contract._read_regular_file_bytes(
            resolved,
            f"Base topology probe binding {mode}",
        )
    except Exception as error:
        raise TopologySelectionError(
            f"Base topology probe binding {mode} is absent"
        ) from error
    probe_receipt_sha256 = raw.get("receipt_sha256")
    if (
        canonical != resolved
        or resolved.is_symlink()
        or not resolved.is_file()
        or hashlib.sha256(data).hexdigest() != observed_sha
        or re.fullmatch(
            r"[0-9a-f]{64}", str(probe_receipt_sha256)
        )
        is None
    ):
        raise TopologySelectionError(
            f"Base topology probe binding {mode} changed"
        )
    probe_artifact = {
        "path": str(resolved),
        "sha256": observed_sha,
        "bytes": len(data),
        "receipt_sha256": probe_receipt_sha256,
    }
    return probe, source_binding, probe_artifact


def build_quality_skip_receipt(
    mode: str,
    probe_path: Path,
    expected_probe_sha256: str,
    *,
    topology_gate_spec_sha256: str,
    quality_gate_spec_sha256: str,
) -> dict[str, Any]:
    """Build one create-new authority for an uncompetitive over-budget mode."""

    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (
            topology_gate_spec_sha256,
            quality_gate_spec_sha256,
        )
    ):
        raise TopologySelectionError(
            "quality skip requires lowercase hash-pinned gate specifications"
        )
    if mode == contract.OFFICIAL_W1_REFERENCE_MODE:
        raise TopologySelectionError(
            "W1 is the mandatory quality reference and cannot be skipped"
        )
    probe, source_binding, probe_artifact = (
        _validated_probe_receipt_binding(
            mode,
            probe_path,
            expected_probe_sha256,
            gate_spec_sha256=topology_gate_spec_sha256,
        )
    )
    estimated = float(probe["estimated_training_seconds"])
    if estimated <= MAX_TRAINING_SECONDS:
        raise TopologySelectionError(
            f"{mode} ETA is within 24 hours and requires full "
            "e1/e2/e4/e8/e16/e32 "
            "quality"
        )
    receipt: dict[str, Any] = {
        "format": QUALITY_SKIP_FORMAT,
        "status": "skipped_over_eta_budget",
        "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
        "artifact_root_namespace": QUALITY_ARTIFACT_ROOT_NAMESPACE,
        "artifact_root_semantics": "not_applicable_eta_skip",
        "quality_role": "candidate_quality",
        "reference_only": False,
        "late_w1_status": "not_measured",
        "w1_tail_equivalence_claimed": False,
        "mode": mode,
        "reference_mode": contract.OFFICIAL_W1_REFERENCE_MODE,
        "topology_gate_spec_sha256": topology_gate_spec_sha256,
        "quality_gate_spec_sha256": quality_gate_spec_sha256,
        "maximum_estimated_training_seconds": MAX_TRAINING_SECONDS,
        "estimated_training_seconds": estimated,
        "topology_independent_input_sha256": probe[
            "topology_independent_input_sha256"
        ],
        "source_binding": source_binding,
        "probe_report": probe_artifact,
    }
    receipt["receipt_sha256"] = contract.canonical_json_sha256(receipt)
    return receipt


def validate_quality_skip(
    mode: str,
    path: Path,
    expected_sha256: str,
    *,
    topology_gate_spec_sha256: str,
    quality_gate_spec_sha256: str,
) -> dict[str, Any]:
    """Validate one immutable over-budget quality-skip receipt."""

    receipt, resolved, observed_sha = contract._load_json_receipt(
        path,
        expected_sha256,
        f"Base topology over-budget quality skip {mode}",
    )
    required = {
        "format",
        "status",
        "quality_protocol_version",
        "artifact_root_namespace",
        "artifact_root_semantics",
        "quality_role",
        "reference_only",
        "late_w1_status",
        "w1_tail_equivalence_claimed",
        "mode",
        "reference_mode",
        "topology_gate_spec_sha256",
        "quality_gate_spec_sha256",
        "maximum_estimated_training_seconds",
        "estimated_training_seconds",
        "topology_independent_input_sha256",
        "source_binding",
        "probe_report",
        "receipt_sha256",
    }
    probe_report = receipt.get("probe_report")
    source_binding = receipt.get("source_binding")
    if (
        mode == contract.OFFICIAL_W1_REFERENCE_MODE
        or not isinstance(receipt, dict)
        or set(receipt) != required
        or receipt.get("format") != QUALITY_SKIP_FORMAT
        or receipt.get("status") != "skipped_over_eta_budget"
        or receipt.get("quality_protocol_version")
        != QUALITY_PROTOCOL_VERSION
        or receipt.get("artifact_root_namespace")
        != QUALITY_ARTIFACT_ROOT_NAMESPACE
        or receipt.get("artifact_root_semantics")
        != "not_applicable_eta_skip"
        or receipt.get("quality_role") != "candidate_quality"
        or receipt.get("reference_only") is not False
        or receipt.get("late_w1_status") != "not_measured"
        or receipt.get("w1_tail_equivalence_claimed") is not False
        or receipt.get("mode") != mode
        or receipt.get("reference_mode")
        != contract.OFFICIAL_W1_REFERENCE_MODE
        or receipt.get("topology_gate_spec_sha256")
        != topology_gate_spec_sha256
        or receipt.get("quality_gate_spec_sha256")
        != quality_gate_spec_sha256
        or receipt.get("maximum_estimated_training_seconds")
        != MAX_TRAINING_SECONDS
        or not _finite_positive(receipt.get("estimated_training_seconds"))
        or float(receipt["estimated_training_seconds"])
        <= MAX_TRAINING_SECONDS
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(receipt.get("topology_independent_input_sha256")),
        )
        is None
        or not isinstance(source_binding, dict)
        or set(source_binding)
        != {
            "frozen_receipt_sha256",
            "frozen_gate_compatibility_sha256",
            "topology_receipt_sha256",
        }
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in source_binding.values()
        )
        or not isinstance(probe_report, dict)
        or set(probe_report)
        != {"path", "sha256", "bytes", "receipt_sha256"}
        or receipt.get("receipt_sha256")
        != contract.canonical_json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )
    ):
        raise TopologySelectionError(
            f"Base topology over-budget quality skip {mode} is forged or stale"
        )
    probe_artifact, _ = _artifact(
        {
            key: probe_report[key]
            for key in ("path", "sha256", "bytes")
        },
        f"{mode} skipped-over-budget probe report",
    )
    probe, observed_source, observed_probe_artifact = (
        _validated_probe_receipt_binding(
            mode,
            Path(probe_artifact["path"]),
            probe_artifact["sha256"],
            gate_spec_sha256=topology_gate_spec_sha256,
        )
    )
    if (
        probe_report != observed_probe_artifact
        or source_binding != observed_source
        or receipt["topology_independent_input_sha256"]
        != probe["topology_independent_input_sha256"]
        or float(receipt["estimated_training_seconds"])
        != float(probe["estimated_training_seconds"])
    ):
        raise TopologySelectionError(
            f"Base topology over-budget quality skip {mode} probe binding changed"
        )
    return {
        "mode": mode,
        "status": "skipped_over_eta_budget",
        "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
        "artifact_root_namespace": QUALITY_ARTIFACT_ROOT_NAMESPACE,
        "artifact_root_semantics": "not_applicable_eta_skip",
        "quality_role": "candidate_quality",
        "reference_only": False,
        "late_w1_status": "not_measured",
        "w1_tail_equivalence_claimed": False,
        "receipt_path": str(resolved),
        "receipt_sha256": observed_sha,
        "receipt_payload_sha256": receipt["receipt_sha256"],
        "topology_gate_spec_sha256": topology_gate_spec_sha256,
        "quality_gate_spec_sha256": quality_gate_spec_sha256,
        "topology_independent_input_sha256": receipt[
            "topology_independent_input_sha256"
        ],
        "source_binding": dict(source_binding),
        "probe_report": dict(probe_report),
        "estimated_training_seconds": float(
            receipt["estimated_training_seconds"]
        ),
        "maximum_estimated_training_seconds": MAX_TRAINING_SECONDS,
    }


def select_topology(
    probes: Sequence[Mapping[str, Any]],
    quality_reports: Sequence[Mapping[str, Any]],
    *,
    gate_spec_sha256: str,
    quality_gate_spec_sha256: str,
    quality_skips: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if [probe.get("mode") for probe in probes] != list(
        contract.TOPOLOGY_SPECS
    ):
        raise TopologySelectionError("all nine probes must be ordered exactly")
    semantic_hashes = {
        probe["topology_independent_input_sha256"] for probe in probes
    }
    if len(semantic_hashes) != 1:
        raise TopologySelectionError(
            "topology probes do not share one fresh five-stage authority"
        )
    eligible = [
        probe for probe in probes if probe["formal_training_eligible"] is True
    ]
    if [probe["mode"] for probe in eligible] != list(contract.TOPOLOGY_SPECS):
        raise TopologySelectionError("formal candidate topology set changed")
    order = list(contract.TOPOLOGY_SPECS)
    report_modes = [report.get("mode") for report in quality_reports]
    skip_modes = [skip.get("mode") for skip in quality_skips]
    if (
        len(set(report_modes)) != len(report_modes)
        or len(set(skip_modes)) != len(skip_modes)
        or report_modes != [mode for mode in order if mode in report_modes]
        or skip_modes != [mode for mode in order if mode in skip_modes]
        or set(report_modes) & set(skip_modes)
        or set(report_modes) | set(skip_modes) != set(order)
    ):
        raise TopologySelectionError(
            "quality reports/skips must cover all nine modes exactly once "
            "in matrix order"
        )
    if contract.OFFICIAL_W1_REFERENCE_MODE not in report_modes:
        raise TopologySelectionError(
            "W1 reference-only e1/e2/e4/e8 quality report is mandatory"
        )
    probe_by_mode = {probe["mode"]: probe for probe in eligible}
    skip_by_mode = {skip["mode"]: skip for skip in quality_skips}
    for mode, skip in skip_by_mode.items():
        probe = probe_by_mode[mode]
        probe_report = skip.get("probe_report")
        if (
            mode == contract.OFFICIAL_W1_REFERENCE_MODE
            or float(probe["estimated_training_seconds"])
            <= MAX_TRAINING_SECONDS
            or skip.get("status") != "skipped_over_eta_budget"
            or skip.get("quality_protocol_version")
            != QUALITY_PROTOCOL_VERSION
            or skip.get("artifact_root_namespace")
            != QUALITY_ARTIFACT_ROOT_NAMESPACE
            or skip.get("artifact_root_semantics")
            != "not_applicable_eta_skip"
            or skip.get("quality_role") != "candidate_quality"
            or skip.get("reference_only") is not False
            or skip.get("late_w1_status") != "not_measured"
            or skip.get("w1_tail_equivalence_claimed") is not False
            or skip.get("topology_gate_spec_sha256")
            != gate_spec_sha256
            or skip.get("quality_gate_spec_sha256")
            != quality_gate_spec_sha256
            or skip.get("topology_independent_input_sha256")
            != probe["topology_independent_input_sha256"]
            or not _finite_positive(
                skip.get("estimated_training_seconds")
            )
            or float(skip["estimated_training_seconds"])
            != float(probe["estimated_training_seconds"])
            or skip.get("maximum_estimated_training_seconds")
            != MAX_TRAINING_SECONDS
            or re.fullmatch(
                r"[0-9a-f]{64}", str(skip.get("receipt_sha256"))
            )
            is None
            or not isinstance(skip.get("receipt_path"), str)
            or not Path(skip["receipt_path"]).is_absolute()
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(skip.get("receipt_payload_sha256")),
            )
            is None
            or not isinstance(skip.get("source_binding"), dict)
            or set(skip["source_binding"])
            != {
                "frozen_receipt_sha256",
                "frozen_gate_compatibility_sha256",
                "topology_receipt_sha256",
            }
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                for value in skip["source_binding"].values()
            )
            or not isinstance(probe_report, dict)
            or set(probe_report)
            != {"path", "sha256", "bytes", "receipt_sha256"}
            or probe_report.get("path") != probe["report_path"]
            or probe_report.get("sha256") != probe["report_sha256"]
            or type(probe_report.get("bytes")) is not int
            or probe_report["bytes"] <= 0
            or re.fullmatch(
                r"[0-9a-f]{64}", str(probe_report.get("receipt_sha256"))
            )
            is None
        ):
            raise TopologySelectionError(
                f"{mode} quality skip is not bound to one over-budget probe"
            )
    quality_semantic_hashes = {
        report["topology_independent_input_sha256"]
        for report in quality_reports
    } | {
        skip["topology_independent_input_sha256"] for skip in quality_skips
    }
    val_input_receipts = {
        (
            report["val_inputs_receipt"]["sha256"],
            report["val_inputs_receipt"]["receipt_payload_sha256"],
        )
        for report in quality_reports
    }
    pipeline_receipts = {
        (
            report["pipeline_receipt"]["sha256"],
            report["pipeline_receipt"]["receipt_payload_sha256"],
        )
        for report in quality_reports
    }
    if (
        quality_semantic_hashes != semantic_hashes
        or len(val_input_receipts) != 1
        or len(pipeline_receipts) != 1
    ):
        raise TopologySelectionError(
            "quality reports do not share the formal DiffSHEG val/pipeline "
            "authority"
        )
    by_mode = {report["mode"]: report for report in quality_reports}
    reference = by_mode[contract.OFFICIAL_W1_REFERENCE_MODE]
    if (
        reference.get("quality_protocol_version")
        != QUALITY_PROTOCOL_VERSION
        or reference.get("artifact_root_namespace")
        != QUALITY_ARTIFACT_ROOT_NAMESPACE
        or reference.get("quality_role") != "w1_reference_only"
        or reference.get("reference_only") is not True
        or reference.get("late_w1_status") != "not_measured"
        or reference.get("w1_tail_equivalence_claimed") is not False
        or set(reference.get("candidate_fgd", {}))
        != {str(epoch) for epoch in W1_REFERENCE_EPOCHS}
    ):
        raise TopologySelectionError(
            "W1 quality authority is not an honest four-point reference"
        )
    reference_values = {
        epoch: float(reference["candidate_fgd"][str(epoch)])
        for epoch in W1_REFERENCE_EPOCHS
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in reference_values.values()
    ):
        raise TopologySelectionError("W1 reference FGD is invalid")
    tail_reference_fgd = min(reference_values.values())
    tail_maximum_allowed_fgd = maximum_allowed_fgd(tail_reference_fgd)
    quality_decisions: dict[str, dict[str, Any]] = {}
    for probe in eligible:
        mode = probe["mode"]
        if mode in skip_by_mode:
            skip = skip_by_mode[mode]
            quality_decisions[mode] = {
                "status": "skipped_over_eta_budget",
                "quality_evaluated": False,
                "selection_eligible": False,
                "quality_role": "candidate_quality",
                "reference_only": False,
                "late_w1_status": "not_measured",
                "w1_tail_equivalence_claimed": False,
                "skip_receipt_path": skip["receipt_path"],
                "skip_receipt_sha256": skip["receipt_sha256"],
                "estimated_training_seconds": float(
                    probe["estimated_training_seconds"]
                ),
                "maximum_estimated_training_seconds": (
                    MAX_TRAINING_SECONDS
                ),
            }
            continue
        report = by_mode[mode]
        if mode == contract.OFFICIAL_W1_REFERENCE_MODE:
            quality_decisions[mode] = {
                "status": "reference_only",
                "quality_evaluated": True,
                "selection_eligible": False,
                "quality_role": "w1_reference_only",
                "reference_only": True,
                "late_w1_status": "not_measured",
                "w1_tail_equivalence_claimed": False,
                "report_path": report["report_path"],
                "report_sha256": report["report_sha256"],
                "reference_epochs": list(W1_REFERENCE_EPOCHS),
                "reference_fgd": {
                    str(epoch): reference_values[epoch]
                    for epoch in W1_REFERENCE_EPOCHS
                },
                "full400_estimated_training_seconds": float(
                    probe["estimated_training_seconds"]
                ),
                "full400_eta_over_24h": (
                    float(probe["estimated_training_seconds"])
                    > MAX_TRAINING_SECONDS
                ),
            }
            continue
        if (
            report.get("quality_protocol_version")
            != QUALITY_PROTOCOL_VERSION
            or report.get("artifact_root_namespace")
            != QUALITY_ARTIFACT_ROOT_NAMESPACE
            or report.get("quality_role") != "candidate_quality"
            or report.get("reference_only") is not False
            or report.get("late_w1_status") != "not_measured"
            or report.get("w1_tail_equivalence_claimed") is not False
            or set(report.get("candidate_fgd", {}))
            != {str(epoch) for epoch in CANDIDATE_QUALITY_EPOCHS}
        ):
            raise TopologySelectionError(
                f"{mode} quality authority is not an honest six-point candidate"
            )
        same_epoch_comparisons = []
        for epoch in SAME_EPOCH_COMPARISON_EPOCHS:
            reference_fgd = reference_values[epoch]
            candidate_fgd = float(report["candidate_fgd"][str(epoch)])
            allowed = maximum_allowed_fgd(reference_fgd)
            same_epoch_comparisons.append(
                {
                    "comparison_kind": "same_epoch_w1_reference",
                    "epoch": epoch,
                    "reference_fgd": reference_fgd,
                    "candidate_fgd": candidate_fgd,
                    "maximum_allowed_fgd": allowed,
                    "pass": candidate_fgd <= allowed,
                }
            )
        tail_comparisons = []
        for epoch in TAIL_EPOCHS:
            candidate_fgd = float(report["candidate_fgd"][str(epoch)])
            tail_comparisons.append(
                {
                    "comparison_kind": "tail_absolute_envelope",
                    "epoch": epoch,
                    "tail_reference_epochs": list(W1_REFERENCE_EPOCHS),
                    "tail_reference_reducer": "minimum",
                    "tail_reference_fgd": tail_reference_fgd,
                    "candidate_fgd": candidate_fgd,
                    "maximum_allowed_fgd": tail_maximum_allowed_fgd,
                    "pass": candidate_fgd <= tail_maximum_allowed_fgd,
                }
            )
        all_same_epoch_pass = all(
            item["pass"] for item in same_epoch_comparisons
        )
        all_tail_pass = all(item["pass"] for item in tail_comparisons)
        all_quality_epochs_pass = all_same_epoch_pass and all_tail_pass
        quality_decisions[mode] = {
            "status": "measured",
            "quality_evaluated": True,
            "selection_eligible": all_quality_epochs_pass,
            "quality_role": "candidate_quality",
            "reference_only": False,
            "late_w1_status": "not_measured",
            "w1_tail_equivalence_claimed": False,
            "report_path": report["report_path"],
            "report_sha256": report["report_sha256"],
            "same_epoch_comparisons": same_epoch_comparisons,
            "tail_reference_epochs": list(W1_REFERENCE_EPOCHS),
            "tail_reference_reducer": "minimum",
            "tail_reference_fgd": tail_reference_fgd,
            "tail_maximum_allowed_fgd": tail_maximum_allowed_fgd,
            "tail_comparisons": tail_comparisons,
            "all_same_epoch_comparisons_pass": all_same_epoch_pass,
            "all_tail_comparisons_pass": all_tail_pass,
            "all_quality_epochs_pass": all_quality_epochs_pass,
        }

    rank_key = lambda probe: (
        probe["estimated_training_seconds"],
        probe["p99_seconds"],
        order.index(probe["mode"]),
    )
    safe_under_budget = [
        probe
        for probe in eligible
        if _finite_positive(probe.get("estimated_training_seconds"))
        and _finite_positive(probe.get("p99_seconds"))
        and float(probe["estimated_training_seconds"])
        <= MAX_TRAINING_SECONDS
        and float(probe["p99_seconds"])
        * int(contract.TOPOLOGY_SPECS[probe["mode"]]["updates_per_epoch"])
        * contract.TOTAL_EPOCHS
        <= MAX_P99_TRAINING_SECONDS
        and probe["mode"] != contract.OFFICIAL_W1_REFERENCE_MODE
        and probe["mode"] in by_mode
        and quality_decisions[probe["mode"]]["selection_eligible"]
        and quality_decisions[probe["mode"]]["all_quality_epochs_pass"]
    ]
    if not safe_under_budget:
        raise TopologySelectionError(
            "no quality-safe finite topology meets the 24-hour median and "
            "22-hour p99 measured ETA limits"
        )
    selected = min(safe_under_budget, key=rank_key)
    decision_branch = "fastest_quality_safe_finite_under_24h"
    payload = {
        "format": contract.TOPOLOGY_SELECTION_FORMAT,
        "status": "pass",
        "topology_gate_spec_sha256": gate_spec_sha256,
        "quality_gate_spec_sha256": quality_gate_spec_sha256,
        "reference_mode": contract.OFFICIAL_W1_REFERENCE_MODE,
        "matrix_modes": list(contract.TOPOLOGY_SPECS),
        "candidate_modes": [
            mode
            for mode in contract.TOPOLOGY_SPECS
            if mode != contract.OFFICIAL_W1_REFERENCE_MODE
        ],
        "topology_independent_input_sha256": next(iter(semantic_hashes)),
        "selection_policy": (
            "fastest_quality_safe_non_reference_under_24h_eta_pruned_quality_v4"
        ),
        "selection_decision_branch": decision_branch,
        "quality_gate_policy": {
            "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
            "artifact_root_namespace": QUALITY_ARTIFACT_ROOT_NAMESPACE,
            "w1_reference_epochs": list(W1_REFERENCE_EPOCHS),
            "candidate_quality_epochs": list(CANDIDATE_QUALITY_EPOCHS),
            "same_epoch_comparison_epochs": list(
                SAME_EPOCH_COMPARISON_EPOCHS
            ),
            "tail_epochs": list(TAIL_EPOCHS),
            "tail_reference_epochs": list(W1_REFERENCE_EPOCHS),
            "tail_reference_reducer": "minimum",
            "tail_reference_fgd": tail_reference_fgd,
            "tail_maximum_allowed_fgd": tail_maximum_allowed_fgd,
            "primary_metric": PRIMARY_METRIC_PATH,
            "validation_protocol": VALIDATION_PROTOCOL,
            "diffsheg_validation_measurement_required": True,
            "maximum_training_seconds": MAX_TRAINING_SECONDS,
            "maximum_p99_training_seconds": MAX_P99_TRAINING_SECONDS,
            "p99_total_updates_required": True,
            "maximum_absolute_fgd_regression": (
                MAX_ABSOLUTE_FGD_REGRESSION
            ),
            "maximum_relative_fgd_regression": (
                MAX_RELATIVE_FGD_REGRESSION
            ),
            "w1_quality_role": "w1_reference_only",
            "w1_selection_eligible": False,
            "late_w1_status": "not_measured",
            "w1_tail_equivalence_claimed": False,
            "w1_quality_report_required": True,
            "within_budget_quality_report_required": True,
            "over_budget_non_w1_skip_status": (
                "skipped_over_eta_budget"
            ),
        },
        "late_w1_status": "not_measured",
        "w1_tail_equivalence_claimed": False,
        "w1_trajectory_equivalence_claimed_for_selected": False,
        "probes": [dict(probe) for probe in probes],
        "quality_reports": [dict(report) for report in quality_reports],
        "quality_skips": [dict(skip) for skip in quality_skips],
        "quality_decisions": quality_decisions,
        "selected": dict(selected),
    }
    payload["receipt_sha256"] = contract.canonical_json_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal the nine-mode SemTalk Base topology gate"
    )
    parser.allow_abbrev = False
    parser.add_argument("--gate-spec", type=Path, required=True)
    parser.add_argument("--expected-gate-spec-sha256", required=True)
    parser.add_argument("--quality-gate-spec", type=Path, required=True)
    parser.add_argument(
        "--expected-quality-gate-spec-sha256", required=True
    )
    parser.add_argument(
        "--probe",
        nargs=3,
        action="append",
        metavar=("MODE", "REPORT", "SHA256"),
        required=True,
    )
    parser.add_argument(
        "--quality-report",
        nargs=3,
        action="append",
        metavar=("MODE", "REPORT", "SHA256"),
        default=[],
    )
    parser.add_argument(
        "--quality-skip",
        nargs=3,
        action="append",
        metavar=("MODE", "RECEIPT", "SHA256"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = [probe[0] for probe in args.probe]
    if modes != list(contract.TOPOLOGY_SPECS):
        raise TopologySelectionError(
            "--probe must name all nine measured modes exactly once"
        )
    quality_modes = [report[0] for report in args.quality_report]
    skip_modes = [receipt[0] for receipt in args.quality_skip]
    known_modes = set(contract.TOPOLOGY_SPECS)
    if any(mode not in known_modes for mode in quality_modes + skip_modes):
        raise TopologySelectionError(
            "quality authority names an unknown topology mode"
        )
    gate_spec = contract.validate_topology_gate_spec(
        SimpleNamespace(
            topology_gate_spec=args.gate_spec,
            expected_topology_gate_spec_sha256=(
                args.expected_gate_spec_sha256
            ),
            topology_mode=contract.OFFICIAL_W1_REFERENCE_MODE,
        )
    )
    probes = [
        validate_probe(
            mode,
            Path(report),
            sha256,
            gate_spec_sha256=gate_spec["sha256"],
        )
        for mode, report, sha256 in args.probe
    ]
    quality_gate = validate_quality_gate_spec(
        args.quality_gate_spec,
        args.expected_quality_gate_spec_sha256,
    )
    quality_reports = [
        validate_quality_report(
            mode,
            Path(report),
            sha256,
            quality_gate_spec_sha256=quality_gate["sha256"],
            topology_gate_spec_sha256=gate_spec["sha256"],
        )
        for mode, report, sha256 in args.quality_report
    ]
    quality_skips = [
        validate_quality_skip(
            mode,
            Path(receipt),
            sha256,
            topology_gate_spec_sha256=gate_spec["sha256"],
            quality_gate_spec_sha256=quality_gate["sha256"],
        )
        for mode, receipt, sha256 in args.quality_skip
    ]
    payload = select_topology(
        probes,
        quality_reports,
        gate_spec_sha256=gate_spec["sha256"],
        quality_gate_spec_sha256=quality_gate["sha256"],
        quality_skips=quality_skips,
    )
    contract._write_new_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
