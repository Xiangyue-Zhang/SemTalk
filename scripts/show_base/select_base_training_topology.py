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
QUALITY_EPOCHS = (1, 2, 4, 8)
PRIMARY_METRIC_PATH = "body.released2.metrics.FGD"
MAX_G64_TRAINING_SECONDS = 24 * 60 * 60
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


def _artifact(
    value: Any,
    label: str,
    *,
    payload: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TopologySelectionError(f"{label} artifact is absent") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        raise TopologySelectionError(
            f"{label} artifact must be one canonical regular file"
        )
    data = path.read_bytes()
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise TopologySelectionError(f"{label} artifact changed")
    normalized = dict(value)
    if not payload:
        return normalized, None
    try:
        parsed = json.loads(
            data.decode("utf-8"),
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
        "g64_preference": {
            "modes": [contract.W8_GLOBAL64_MODE, contract.W16_GLOBAL64_MODE],
            "maximum_estimated_training_seconds": MAX_G64_TRAINING_SECONDS,
            "policy": "fastest_safe_g64_under_24h_before_any_global512_mode",
        },
        "candidate_quality_gate": {
            "modes": [
                contract.W8_GLOBAL64_MODE,
                contract.W16_GLOBAL64_MODE,
                contract.W8_GLOBAL512_MODE,
                contract.W16_GLOBAL512_MODE,
            ],
            "per_epoch_comparison": "candidate_fgd_lte_reference_fgd_plus_max_of_absolute_or_relative_margin",
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
) -> dict[str, Any]:
    metrics = _metrics_module()
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
    short_trajectory, short_payload = _artifact(
        report["short_trajectory_receipt"],
        f"{mode} short trajectory",
        payload=True,
    )
    if (
        short_payload.get("status") != "complete"
        or short_payload.get("topology_mode") != mode
        or short_payload.get("candidate_epochs") != list(QUALITY_EPOCHS)
        or short_payload.get("split") != "val"
        or short_payload.get("test_visible") is not False
        or short_payload.get("topology_independent_input_sha256")
        != report["topology_independent_input_sha256"]
    ):
        raise TopologySelectionError(f"{mode} short trajectory changed")
    canonical, _ = _artifact(
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
        }
        if not isinstance(row, dict) or set(row) != row_keys:
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} quality row schema mismatch"
            )
        checkpoint, _ = _artifact(
            row["candidate_checkpoint"], f"{mode} e{expected_epoch} checkpoint"
        )
        prediction, _ = _artifact(
            row["prediction_manifest"], f"{mode} e{expected_epoch} prediction"
        )
        distribution, distribution_payload = _artifact(
            row["distribution_receipt"],
            f"{mode} e{expected_epoch} distribution",
            payload=True,
        )
        screen, screen_payload = _artifact(
            row["primary_screen_receipt"],
            f"{mode} e{expected_epoch} primary screen",
            payload=True,
        )
        replay, _replay_payload = _artifact(
            row["primary_replay_receipt"],
            f"{mode} e{expected_epoch} raw replay",
            payload=True,
        )
        if (
            row.get("epoch") != expected_epoch
            or distribution_payload.get("split") != "val"
            or distribution_payload.get("test_visible") is not False
        ):
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} validation identity changed"
            )
        screen_validation = metrics.validate_released2_primary_screen_receipt(
            screen,
            expected_prediction_manifest=prediction,
            expected_distribution_receipt=distribution,
            expected_real_feature_cache=cache,
            expected_canonical_manifest=canonical,
            expected_selection_protocol=SELECTION_PROTOCOL,
            expected_split="val",
            expected_clip_count=1715,
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
                expected_clip_count=1715,
            )
        )
        value = float(replay_validation["primary_metric"])
        if not math.isfinite(value) or value < 0:
            raise TopologySelectionError(
                f"{mode} e{expected_epoch} replay FGD is invalid"
            )
        values[expected_epoch] = value
        normalized_rows.append(
            {
                "epoch": expected_epoch,
                "candidate_checkpoint": checkpoint,
                "prediction_manifest": prediction,
                "distribution_receipt": distribution,
                "primary_screen_receipt": screen,
                "primary_replay_receipt": replay,
                "body_released2_fgd": value,
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
    if len(eligible) != 4:
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
    g64 = [
        probe
        for probe in eligible
        if probe["mode"]
        in {contract.W8_GLOBAL64_MODE, contract.W16_GLOBAL64_MODE}
        and probe["estimated_training_seconds"] <= MAX_G64_TRAINING_SECONDS
        and quality_decisions[probe["mode"]][
            "all_trajectory_epochs_pass"
        ]
    ]
    if g64:
        selected = min(g64, key=rank_key)
        decision_branch = "fastest_safe_g64_under_24h"
    else:
        gated = [
            probe
            for probe in eligible
            if probe["mode"]
            in {contract.W8_GLOBAL512_MODE, contract.W16_GLOBAL512_MODE}
            and quality_decisions[probe["mode"]][
                "all_trajectory_epochs_pass"
            ]
        ]
        if not gated:
            raise TopologySelectionError(
                "no g64 topology meets 24h and no accelerated topology passes "
                "the preregistered raw-replay quality gate"
            )
        selected = min(gated, key=rank_key)
        decision_branch = "fastest_quality_gated_global512"
    payload = {
        "format": contract.TOPOLOGY_SELECTION_FORMAT,
        "status": "pass",
        "topology_gate_spec_sha256": gate_spec_sha256,
        "quality_gate_spec_sha256": quality_gate_spec_sha256,
        "reference_mode": contract.OFFICIAL_W1_REFERENCE_MODE,
        "candidate_modes": list(contract.TOPOLOGY_SPECS)[1:],
        "topology_independent_input_sha256": next(iter(semantic_hashes)),
        "selection_policy": (
            "g64_under_24h_else_raw_replay_quality_gated_acceleration_v1"
        ),
        "selection_decision_branch": decision_branch,
        "quality_gate_policy": {
            "trajectory_epochs": list(QUALITY_EPOCHS),
            "primary_metric": PRIMARY_METRIC_PATH,
            "raw_prediction_replay_required": True,
            "maximum_g64_training_seconds": MAX_G64_TRAINING_SECONDS,
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
            "--probe must name W1 then the four candidate modes exactly once"
        )
    quality_modes = [report[0] for report in args.quality_report]
    if quality_modes != list(contract.TOPOLOGY_SPECS):
        raise TopologySelectionError(
            "--quality-report must name W1 then the four candidate modes "
            "exactly once"
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
