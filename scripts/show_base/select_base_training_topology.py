#!/usr/bin/env python3
"""Seal the fastest safe SemTalk Base topology after all five real probes."""

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

from scripts.show_base import train_base_official_adapt_long as contract


class TopologySelectionError(RuntimeError):
    """Raised when any measured topology is absent, unsafe, or stale."""


QUALITY_GATE_FORMAT = "semtalk_show_base_topology_quality_gate_spec_v1"
QUALITY_REPORT_FORMAT = "semtalk_show_base_topology_quality_report_v1"
SHORT_TRAJECTORY_FORMAT = (
    "semtalk_show_base_topology_short_trajectory_v1"
)
QUALITY_PROVENANCE_FORMAT = (
    "semtalk_show_base_topology_quality_provenance_v1"
)
QUALITY_EPOCHS = (1, 2, 4, 8)
EXPECTED_VAL_CLIPS = 1_715
PRIMARY_METRIC_PATH = "body.released2.metrics.FGD"
MAX_TRAINING_SECONDS = 24 * 60 * 60
MAX_ABSOLUTE_FGD_REGRESSION = 0.01
MAX_RELATIVE_FGD_REGRESSION = 0.02
SELECTION_PROTOCOL = {
    "primary_metric": PRIMARY_METRIC_PATH,
    "mode": "min",
    "validation_only_for_selection": True,
    "test_evaluations": 0,
}


def _metrics_module() -> Any:
    # Topology policy and its CPU tests remain stdlib-only.  NumPy/torch are
    # loaded only when the selector validates actual raw-NPZ replay receipts.
    from scripts.show_base import evaluate_talkshow_show_metrics as module

    return module


def _replication_module() -> Any:
    # NumPy is loaded only when a formal quality row freshly replays its
    # deterministic-replication gate.
    from scripts.show_base import deterministic_replication_gate as module

    return module


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


def _canonical_manifest_artifact(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "bytes",
        "rows",
        "selected_rows",
    }:
        raise TopologySelectionError(f"{label} artifact schema mismatch")
    normalized, _ = _artifact(
        {key: value[key] for key in ("path", "sha256", "bytes")},
        label,
    )
    rows = value.get("rows")
    selected = value.get("selected_rows")
    if (
        type(rows) is not int
        or rows < EXPECTED_VAL_CLIPS
        or type(selected) is not int
        or selected != EXPECTED_VAL_CLIPS
    ):
        raise TopologySelectionError(f"{label} coverage changed")
    return {**normalized, "rows": rows, "selected_rows": selected}


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
        "all_rank_optimizer_state_identical",
        "ranks",
    }
    rank_required = {
        "rank",
        "optimizer_updates",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
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
        or value.get("optimizer_updates")
        != contract.TRAJECTORY_PROBE_UPDATES
        or value.get("world_size") != world_size
        or value.get("rank_order") != list(range(world_size))
        or value.get("all_rank_model_state_identical") is not True
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
            or rank.get("optimizer_updates")
            != contract.TRAJECTORY_PROBE_UPDATES
            or type(rank.get("model_state_tensors")) is not int
            or rank["model_state_tensors"] <= 0
            or rank.get("sample_count")
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
    optimizer_consensus = {
        rank["optimizer_state_semantic_sha256"] for rank in ranks
    }
    if len(model_consensus) != 1 or len(optimizer_consensus) != 1:
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
    """Freshly replay the trainer-owned e1/e2/e4/e8 publications."""

    if len(values) != len(QUALITY_EPOCHS):
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
        "candidate_epochs",
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
        "topology_selection",
    }
    topology_selection_keys = {
        "path",
        "sha256",
        "selected",
        "probe_report_sha256",
    }
    frozen_keys = {
        "format",
        "source",
        "official_base",
        "speaker_initialization",
        "dataset",
        "protocol",
        "long_contract",
        "topology",
        "receipt_sha256",
    }
    for position, (epoch, value) in enumerate(zip(QUALITY_EPOCHS, values)):
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
            or payload.get("format") != contract.READY_RECEIPT_FORMAT
            or payload.get("status") != "ready"
            or payload.get("selection_eligible") is not False
            or payload.get("test_visible") is not False
            or payload.get("epoch") != epoch
            or payload.get("optimizer_updates")
            != expected_optimizer_updates
            or payload.get("trajectory_anchor_match") is not True
            or isinstance(payload.get("published_unix"), bool)
            or not isinstance(payload.get("published_unix"), (int, float))
            or not math.isfinite(float(payload["published_unix"]))
            or float(payload["published_unix"]) <= 0.0
            or not isinstance(checkpoint_value, dict)
            or set(checkpoint_value) != checkpoint_keys
            or checkpoint_value.get("relative_path")
            != f"candidates/base_official_adapt_epoch_{epoch:02d}.bin"
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
        expected_entry_epochs = list(contract.CANDIDATE_EPOCHS[: position + 1])
        if (
            set(manifest) != manifest_keys
            or manifest.get("format") != contract.MANIFEST_FORMAT
            or manifest.get("status") != "running"
            or manifest.get("candidate_epochs")
            != list(contract.CANDIDATE_EPOCHS)
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
            or entry.get("trajectory_anchor_match") is not True
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
            or not isinstance(throughput.get("topology_selection"), dict)
            or set(throughput["topology_selection"])
            != topology_selection_keys
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
            selection_report, selection_path, selection_sha = (
                contract._load_json_receipt(
                    Path(
                        str(throughput["topology_selection"]["path"])
                    ),
                    str(throughput["topology_selection"]["sha256"]),
                    f"{label} topology selection",
                )
            )
            normalized_selection = contract.validate_topology_selection(
                SimpleNamespace(
                    topology_selection_report=Path(
                        str(throughput["topology_selection"]["path"])
                    ),
                    expected_topology_selection_sha256=str(
                        throughput["topology_selection"]["sha256"]
                    ),
                    expected_topology_gate_spec_sha256=(
                        topology_gate_spec_sha256
                    ),
                    topology_mode=mode,
                ),
                throughput_gate=throughput,
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
            != payload["frozen_receipt_sha256"]
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
            or normalized_selection != throughput["topology_selection"]
            or selection_path
            != Path(str(throughput["topology_selection"]["path"]))
            or selection_sha != throughput["topology_selection"]["sha256"]
            or selection_report.get("quality_gate_spec_sha256")
            != quality_gate_spec_sha256
            or selection_report.get("topology_independent_input_sha256")
            != full_probe.get("topology_independent_input_sha256")
            or normalized_selection["probe_report_sha256"].get(mode)
            != throughput["sha256"]
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
            "topology_selection_sha256": normalized_selection["sha256"],
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
    required = {
        "format",
        "status",
        "topology_mode",
        "topology_gate_spec_sha256",
        "quality_gate_spec_sha256",
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
        or payload.get("candidate_epochs") != list(QUALITY_EPOCHS)
        or payload.get("split") != "val"
        or payload.get("test_visible") is not False
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(payload.get("topology_independent_input_sha256")),
        )
        is None
        or not isinstance(candidates, list)
        or len(candidates) != len(QUALITY_EPOCHS)
        or not isinstance(candidate_ready_receipts, list)
        or len(candidate_ready_receipts) != len(QUALITY_EPOCHS)
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
    expected_updates = int(contract.TOPOLOGY_SPECS[mode]["updates_per_epoch"])
    normalized: dict[int, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for expected_epoch, row in zip(QUALITY_EPOCHS, candidates):
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


def validate_quality_candidate_provenance(
    mode: str,
    epoch: int,
    *,
    topology_independent_input_sha256: str,
    short_trajectory_receipt: Mapping[str, Any],
    expected_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Any,
    prediction_manifest: Any,
    distribution_receipt: Any,
    primary_screen_receipt: Any,
    primary_replay_receipt: Any,
    canonical_manifest: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
) -> dict[str, Any]:
    """Freshly bind one e1/e2/e4/e8 quality result end to end.

    The metric receipts already prove raw released2 replay.  This additional
    layer closes the authority gap they intentionally do not own: which short
    training checkpoint produced the validation manifest whose distribution,
    screen, and independent replay are being selected.
    """

    label = f"{mode} e{epoch}"
    checkpoint, _ = _artifact(
        candidate_checkpoint,
        f"{label} checkpoint",
    )
    if checkpoint != dict(expected_checkpoint):
        raise TopologySelectionError(
            f"{label} checkpoint differs from short trajectory"
        )

    prediction, _ = _artifact(
        prediction_manifest,
        f"{label} prediction",
    )
    prediction_rows = _strict_jsonl_rows(
        prediction,
        f"{label} prediction",
    )
    expected_prediction_keys = {
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
    canonical_ids: set[str] = set()
    global_indices: list[int] = []
    for index, row in enumerate(prediction_rows):
        clip_id = row.get("canonical_clip_id")
        global_index = row.get("global_index")
        if (
            set(row) != expected_prediction_keys
            or row.get("split") != "val"
            or row.get("epoch") != epoch
            or row.get("candidate_checkpoint_sha256")
            != checkpoint["sha256"]
            or not isinstance(clip_id, str)
            or not clip_id
            or clip_id in canonical_ids
            or type(global_index) is not int
            or global_index < 0
            or type(row.get("frames")) is not int
            or row["frames"] <= 0
        ):
            raise TopologySelectionError(
                f"{label} prediction row {index} authority changed"
            )
        for role in ("prediction", "ground_truth"):
            artifact = row.get(role)
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"path", "sha256", "bytes"}
                or not isinstance(artifact.get("path"), str)
                or not Path(artifact["path"]).is_absolute()
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(artifact.get("sha256"))
                )
                is None
                or type(artifact.get("bytes")) is not int
                or artifact["bytes"] <= 0
            ):
                raise TopologySelectionError(
                    f"{label} prediction row {index} {role} changed"
                )
        canonical_ids.add(clip_id)
        global_indices.append(global_index)
    if (
        len(prediction_rows) != EXPECTED_VAL_CLIPS
        or len(canonical_ids) != EXPECTED_VAL_CLIPS
        or len(set(global_indices)) != EXPECTED_VAL_CLIPS
    ):
        raise TopologySelectionError(
            f"{label} prediction coverage changed"
        )

    distribution, distribution_payload = _artifact(
        distribution_receipt,
        f"{label} distribution",
        payload=True,
    )
    assert distribution_payload is not None
    validation_gate_value = distribution_payload.get("validation_gate")
    if (
        distribution_payload.get("prediction_manifest") != prediction
        or not isinstance(validation_gate_value, dict)
    ):
        raise TopologySelectionError(
            f"{label} distribution authority changed"
        )
    validation_gate, _ = _artifact(
        validation_gate_value,
        f"{label} validation gate",
        payload=True,
    )
    replication = _replication_module()
    try:
        replayed_gate, gate_payload = replication.load_gate(
            Path(validation_gate["path"]),
            validation_gate["sha256"],
            expected_scope="validation_candidate_family",
        )
    except Exception as error:
        raise TopologySelectionError(
            f"{label} validation gate fresh replay failed"
        ) from error
    base_checkpoint = (
        gate_payload.get("model_bundle", {})
        .get("checkpoints", {})
        .get("base")
    )
    if (
        replayed_gate != validation_gate
        or gate_payload.get("status") != "pass"
        or gate_payload.get("split") != "val"
        or gate_payload.get("test_visible") is not False
        or base_checkpoint != checkpoint
    ):
        raise TopologySelectionError(
            f"{label} checkpoint/gate authority changed"
        )

    screen, screen_payload = _artifact(
        primary_screen_receipt,
        f"{label} primary screen",
        payload=True,
    )
    replay, replay_payload = _artifact(
        primary_replay_receipt,
        f"{label} raw replay",
        payload=True,
    )
    assert screen_payload is not None and replay_payload is not None
    if (
        screen_payload.get("split") != "val"
        or screen_payload.get("test_visible") is not False
        or screen_payload.get("clip_count") != EXPECTED_VAL_CLIPS
        or screen_payload.get("canonical_manifest")
        != dict(canonical_manifest)
        or screen_payload.get("prediction_manifest") != prediction
        or screen_payload.get("distribution_receipt") != distribution
        or screen_payload.get("real_feature_cache")
        != dict(real_feature_cache)
        or replay_payload.get("split") != "val"
        or replay_payload.get("clip_count") != EXPECTED_VAL_CLIPS
        or replay_payload.get("report_payload_sha256")
        != screen_payload.get("receipt_payload_sha256")
        or replay_payload.get("canonical_manifest")
        != dict(canonical_manifest)
        or replay_payload.get("prediction_manifest") != prediction
        or replay_payload.get("distribution_receipt_payload_sha256")
        != distribution_payload.get("receipt_payload_sha256")
        or replay_payload.get("real_feature_cache")
        != dict(real_feature_cache)
    ):
        raise TopologySelectionError(
            f"{label} prediction/distribution/screen/replay chain changed"
        )

    metrics = _metrics_module()
    try:
        screen_validation = (
            metrics.validate_released2_primary_screen_receipt(
                screen,
                expected_prediction_manifest=prediction,
                expected_distribution_receipt=distribution,
                expected_real_feature_cache=real_feature_cache,
                expected_canonical_manifest=canonical_manifest,
                expected_selection_protocol=SELECTION_PROTOCOL,
                expected_split="val",
                expected_clip_count=EXPECTED_VAL_CLIPS,
            )
        )
        replay_validation = (
            metrics.validate_released2_primary_screen_replay_receipt(
                replay,
                expected_screen_artifact=screen,
                expected_screen=screen_payload,
                screen_validation=screen_validation,
                expected_prediction_manifest=prediction,
                expected_distribution_receipt=distribution,
                expected_selection_protocol=SELECTION_PROTOCOL,
                expected_split="val",
                expected_clip_count=EXPECTED_VAL_CLIPS,
            )
        )
        value = float(replay_validation["primary_metric"])
    except Exception as error:
        raise TopologySelectionError(
            f"{label} raw released2 replay validation failed"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise TopologySelectionError(f"{label} replay FGD is invalid")

    chain = {
        "short_trajectory_receipt": dict(short_trajectory_receipt),
        "candidate_checkpoint": checkpoint,
        "prediction_manifest": prediction,
        "validation_gate": validation_gate,
        "distribution_receipt": distribution,
        "primary_screen_receipt": screen,
        "primary_replay_receipt": replay,
    }
    provenance = {
        "format": QUALITY_PROVENANCE_FORMAT,
        "topology_mode": mode,
        "epoch": epoch,
        "split": "val",
        "test_visible": False,
        "clip_count": EXPECTED_VAL_CLIPS,
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
        "prediction_manifest_sha256": prediction["sha256"],
        "prediction_rows": len(prediction_rows),
        "prediction_checkpoint_sha256": checkpoint["sha256"],
        "validation_gate_sha256": validation_gate["sha256"],
        "validation_gate_payload_sha256": validation_gate[
            "receipt_payload_sha256"
        ],
        "distribution_receipt_sha256": distribution["sha256"],
        "distribution_receipt_payload_sha256": distribution[
            "receipt_payload_sha256"
        ],
        "primary_screen_receipt_sha256": screen["sha256"],
        "primary_screen_receipt_payload_sha256": screen[
            "receipt_payload_sha256"
        ],
        "primary_replay_receipt_sha256": replay["sha256"],
        "primary_replay_receipt_payload_sha256": replay[
            "receipt_payload_sha256"
        ],
        "chain_sha256": contract.canonical_json_sha256(chain),
    }
    return {
        "candidate_checkpoint": checkpoint,
        "prediction_manifest": prediction,
        "distribution_receipt": distribution,
        "primary_screen_receipt": screen,
        "primary_replay_receipt": replay,
        "provenance": provenance,
        "body_released2_fgd": value,
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
        "scope": "SemTalk Base on SHOW validation only",
        "split": "val",
        "test_visible": False,
        "trajectory_epochs": list(QUALITY_EPOCHS),
        "primary_metric": PRIMARY_METRIC_PATH,
        "raw_prediction_replay_required": True,
        "comparison_reference": contract.OFFICIAL_W1_REFERENCE_MODE,
        "measured_eta_constraint": {
            "modes": list(contract.TOPOLOGY_SPECS),
            "maximum_estimated_training_seconds": MAX_TRAINING_SECONDS,
            "finite_probe_required": True,
            "policy": (
                "all_quality_safe_finite_modes_compete_by_measured_eta_"
                "under_24h"
            ),
        },
        "candidate_quality_gate": {
            "modes": list(contract.TOPOLOGY_SPECS),
            "per_epoch_comparison": (
                "candidate_fgd_lte_reference_fgd_plus_max_of_"
                "absolute_or_relative_margin"
            ),
            "maximum_absolute_fgd_regression": MAX_ABSOLUTE_FGD_REGRESSION,
            "maximum_relative_fgd_regression": MAX_RELATIVE_FGD_REGRESSION,
            "all_trajectory_epochs_must_pass": True,
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
    required = {
        "format",
        "status",
        "mode",
        "quality_gate_spec_sha256",
        "split",
        "test_visible",
        "trajectory_epochs",
        "topology_independent_input_sha256",
        "short_trajectory_receipt",
        "canonical_manifest",
        "real_feature_cache",
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
        or report.get("split") != "val"
        or report.get("test_visible") is not False
        or report.get("trajectory_epochs") != list(QUALITY_EPOCHS)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(report.get("topology_independent_input_sha256")),
        )
        is None
        or not isinstance(candidates, list)
        or len(candidates) != len(QUALITY_EPOCHS)
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
    ):
        raise TopologySelectionError(f"{mode} short trajectory changed")
    canonical = _canonical_manifest_artifact(
        report["canonical_manifest"], f"{mode} canonical manifest"
    )
    cache, _cache_payload = _artifact(
        report["real_feature_cache"], f"{mode} real feature cache", payload=True
    )
    values: dict[int, float] = {}
    normalized_rows: list[dict[str, Any]] = []
    for expected_epoch, row in zip(QUALITY_EPOCHS, candidates):
        row_keys = {
            "epoch",
            "candidate_checkpoint",
            "prediction_manifest",
            "distribution_receipt",
            "primary_screen_receipt",
            "primary_replay_receipt",
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
            prediction_manifest=row["prediction_manifest"],
            distribution_receipt=row["distribution_receipt"],
            primary_screen_receipt=row["primary_screen_receipt"],
            primary_replay_receipt=row["primary_replay_receipt"],
            canonical_manifest=canonical,
            real_feature_cache=cache,
        )
        if row["provenance"] != validated["provenance"]:
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} provenance changed"
            )
        value = validated["body_released2_fgd"]
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
        "canonical_manifest": canonical,
        "real_feature_cache": cache,
        "candidate_fgd": {str(epoch): values[epoch] for epoch in QUALITY_EPOCHS},
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
        or sample_inventory.get("probe_unique_samples")
        != sample_inventory.get("probe_samples")
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


def select_topology(
    probes: Sequence[Mapping[str, Any]],
    quality_reports: Sequence[Mapping[str, Any]],
    *,
    gate_spec_sha256: str,
    quality_gate_spec_sha256: str,
) -> dict[str, Any]:
    if [probe.get("mode") for probe in probes] != list(
        contract.TOPOLOGY_SPECS
    ):
        raise TopologySelectionError("all five probes must be ordered exactly")
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
    if [report.get("mode") for report in quality_reports] != list(
        contract.TOPOLOGY_SPECS
    ):
        raise TopologySelectionError(
            "all five quality reports must be ordered exactly"
        )
    quality_semantic_hashes = {
        report["topology_independent_input_sha256"]
        for report in quality_reports
    }
    canonical_manifests = {
        (
            report["canonical_manifest"]["sha256"],
            report["canonical_manifest"]["bytes"],
        )
        for report in quality_reports
    }
    feature_caches = {
        (
            report["real_feature_cache"]["sha256"],
            report["real_feature_cache"]["receipt_payload_sha256"],
        )
        for report in quality_reports
    }
    if (
        quality_semantic_hashes != semantic_hashes
        or len(canonical_manifests) != 1
        or len(feature_caches) != 1
    ):
        raise TopologySelectionError(
            "quality reports do not share the measured five-stage/val authority"
        )
    by_mode = {report["mode"]: report for report in quality_reports}
    reference = by_mode[contract.OFFICIAL_W1_REFERENCE_MODE]
    quality_decisions: dict[str, dict[str, Any]] = {}
    for probe in eligible:
        mode = probe["mode"]
        report = by_mode[mode]
        comparisons = []
        for epoch in QUALITY_EPOCHS:
            reference_fgd = float(reference["candidate_fgd"][str(epoch)])
            candidate_fgd = float(report["candidate_fgd"][str(epoch)])
            allowed = maximum_allowed_fgd(reference_fgd)
            comparisons.append(
                {
                    "epoch": epoch,
                    "reference_fgd": reference_fgd,
                    "candidate_fgd": candidate_fgd,
                    "maximum_allowed_fgd": allowed,
                    "pass": candidate_fgd <= allowed,
                }
            )
        quality_decisions[mode] = {
            "report_path": report["report_path"],
            "report_sha256": report["report_sha256"],
            "comparisons": comparisons,
            "all_trajectory_epochs_pass": all(
                item["pass"] for item in comparisons
            ),
        }

    order = list(contract.TOPOLOGY_SPECS)
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
        and quality_decisions[probe["mode"]][
            "all_trajectory_epochs_pass"
        ]
    ]
    if not safe_under_budget:
        raise TopologySelectionError(
            "no quality-safe finite topology meets the 24-hour measured "
            "ETA limit"
        )
    selected = min(safe_under_budget, key=rank_key)
    decision_branch = "fastest_quality_safe_finite_under_24h"
    payload = {
        "format": contract.TOPOLOGY_SELECTION_FORMAT,
        "status": "pass",
        "topology_gate_spec_sha256": gate_spec_sha256,
        "quality_gate_spec_sha256": quality_gate_spec_sha256,
        "reference_mode": contract.OFFICIAL_W1_REFERENCE_MODE,
        "candidate_modes": list(contract.TOPOLOGY_SPECS),
        "topology_independent_input_sha256": next(iter(semantic_hashes)),
        "selection_policy": (
            "fastest_quality_safe_finite_under_24h_all_measured_"
            "topologies_v2"
        ),
        "selection_decision_branch": decision_branch,
        "quality_gate_policy": {
            "trajectory_epochs": list(QUALITY_EPOCHS),
            "primary_metric": PRIMARY_METRIC_PATH,
            "raw_prediction_replay_required": True,
            "maximum_training_seconds": MAX_TRAINING_SECONDS,
            "maximum_absolute_fgd_regression": (
                MAX_ABSOLUTE_FGD_REGRESSION
            ),
            "maximum_relative_fgd_regression": (
                MAX_RELATIVE_FGD_REGRESSION
            ),
        },
        "w1_trajectory_equivalence_claimed_for_selected": False,
        "probes": [dict(probe) for probe in probes],
        "quality_reports": [dict(report) for report in quality_reports],
        "quality_decisions": quality_decisions,
        "selected": dict(selected),
    }
    payload["receipt_sha256"] = contract.canonical_json_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal the five-mode SemTalk Base topology gate"
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
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = [probe[0] for probe in args.probe]
    if modes != list(contract.TOPOLOGY_SPECS):
        raise TopologySelectionError(
            "--probe must name all five measured modes exactly once"
        )
    quality_modes = [report[0] for report in args.quality_report]
    if quality_modes != list(contract.TOPOLOGY_SPECS):
        raise TopologySelectionError(
            "--quality-report must name all five measured modes exactly once"
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
    payload = select_topology(
        probes,
        quality_reports,
        gate_spec_sha256=gate_spec["sha256"],
        quality_gate_spec_sha256=quality_gate["sha256"],
    )
    contract._write_new_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
