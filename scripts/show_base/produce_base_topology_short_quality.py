#!/usr/bin/env python3
"""Seal one formal SemTalk Base short-trajectory SHOW quality report.

This CPU-only producer accepts immutable mode-specific v3 checkpoints (W1
reference e1/e2/e4/e8; candidates e1/e2/e4/e8/e16/e32) and their
already-computed full-SHOW validation artifacts.  It freshly reuses the formal
long-run validators for checkpoint -> val inference lineage -> pinned DiffSHEG
FGD, then publishes one immutable report consumable by
``select_base_training_topology.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import select_base_training_topology as selector
from scripts.show_base import train_base_official_adapt_long as contract


def _positive_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise selector.TopologySelectionError(
            f"{label} must be an integer"
        ) from error
    if parsed <= 0:
        raise selector.TopologySelectionError(f"{label} must be positive")
    return parsed


def _sha256(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise selector.TopologySelectionError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _artifact3(values: Sequence[str], label: str) -> dict[str, Any]:
    path, digest, size = values
    return {
        # Preserve the caller's spelling.  The shared validator rejects
        # relative, symlinked, or otherwise non-canonical paths rather than
        # silently laundering them through resolve().
        "path": str(Path(path)),
        "sha256": _sha256(digest, f"{label} SHA-256"),
        "bytes": _positive_int(size, f"{label} bytes"),
    }


def _artifact4(values: Sequence[str], label: str) -> dict[str, Any]:
    path, digest, size, payload_digest = values
    return {
        **_artifact3((path, digest, size), label),
        "receipt_payload_sha256": _sha256(
            payload_digest,
            f"{label} payload SHA-256",
        ),
    }


def _epoch_artifacts(
    values: Sequence[Sequence[str]],
    *,
    label: str,
    payload: bool,
    expected_epochs: Sequence[int],
) -> dict[int, dict[str, Any]]:
    epochs: list[int] = []
    normalized: dict[int, dict[str, Any]] = {}
    for raw in values:
        try:
            epoch = int(raw[0])
        except ValueError as error:
            raise selector.TopologySelectionError(
                f"{label} epoch must be an integer"
            ) from error
        epochs.append(epoch)
        if epoch in normalized:
            raise selector.TopologySelectionError(
                f"duplicate {label} epoch e{epoch}"
            )
        artifact_values = raw[1:]
        normalized[epoch] = (
            _artifact4(artifact_values, f"e{epoch} {label}")
            if payload
            else _artifact3(artifact_values, f"e{epoch} {label}")
        )
    if epochs != list(expected_epochs):
        raise selector.TopologySelectionError(
            f"{label} must name "
            + "/".join(f"e{epoch}" for epoch in expected_epochs)
            + " exactly in order"
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=list(contract.TOPOLOGY_SPECS),
        required=True,
    )
    parser.add_argument("--topology-gate-spec", type=Path, required=True)
    parser.add_argument(
        "--expected-topology-gate-spec-sha256",
        required=True,
    )
    parser.add_argument("--quality-gate-spec", type=Path, required=True)
    parser.add_argument(
        "--expected-quality-gate-spec-sha256",
        required=True,
    )
    parser.add_argument(
        "--candidate-ready-receipt",
        nargs=5,
        action="append",
        metavar=("EPOCH", "PATH", "SHA256", "BYTES", "PAYLOAD_SHA256"),
        required=True,
    )
    parser.add_argument(
        "--short-quality-status",
        nargs=4,
        metavar=("PATH", "SHA256", "BYTES", "PAYLOAD_SHA256"),
        required=True,
    )
    parser.add_argument(
        "--short-trajectory-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--val-inputs-receipt",
        nargs=4,
        metavar=("PATH", "SHA256", "BYTES", "PAYLOAD_SHA256"),
        required=True,
    )
    parser.add_argument(
        "--pipeline-receipt",
        nargs=4,
        metavar=("PATH", "SHA256", "BYTES", "PAYLOAD_SHA256"),
        required=True,
    )
    parser.add_argument(
        "--inference-lineage",
        nargs=5,
        action="append",
        metavar=("EPOCH", "PATH", "SHA256", "BYTES", "PAYLOAD_SHA256"),
        required=True,
    )
    parser.add_argument(
        "--diffsheg-report",
        nargs=4,
        action="append",
        metavar=("EPOCH", "PATH", "SHA256", "BYTES"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def produce(args: argparse.Namespace) -> dict[str, Any]:
    quality_epochs = selector.quality_epochs_for_mode(args.mode)
    quality_role = selector.quality_role_for_mode(args.mode)
    reference_only = args.mode == contract.OFFICIAL_W1_REFERENCE_MODE
    output = args.output.expanduser().resolve()
    short_output = args.short_trajectory_output.expanduser().resolve()
    if output == short_output:
        raise selector.TopologySelectionError(
            "quality and short-trajectory outputs must differ"
        )
    for path, label in (
        (output, "quality output"),
        (short_output, "short-trajectory output"),
    ):
        if path.is_symlink() or os.path.lexists(path):
            raise FileExistsError(
                f"refusing to overwrite immutable {label}: {path}"
            )
    topology_gate = contract.validate_topology_gate_spec(
        SimpleNamespace(
            topology_gate_spec=args.topology_gate_spec,
            expected_topology_gate_spec_sha256=(
                args.expected_topology_gate_spec_sha256
            ),
            topology_mode=args.mode,
        )
    )
    quality_gate = selector.validate_quality_gate_spec(
        args.quality_gate_spec,
        args.expected_quality_gate_spec_sha256,
    )
    candidate_ready = _epoch_artifacts(
        args.candidate_ready_receipt,
        label="candidate-ready receipt",
        payload=True,
        expected_epochs=quality_epochs,
    )
    status_artifact, ready, semantic_sha, short_checkpoints = (
        selector.validate_short_quality_training_bundle(
            args.mode,
            _artifact4(
                args.short_quality_status,
                "short-quality status",
            ),
            [candidate_ready[epoch] for epoch in quality_epochs],
            topology_gate_spec_sha256=topology_gate["sha256"],
            quality_gate_spec_sha256=quality_gate["sha256"],
        )
    )
    _status_identity, status_payload = selector._artifact(
        status_artifact,
        f"{args.mode} validated short-quality status root authority",
        payload=True,
    )
    assert status_payload is not None
    artifact_root = status_payload["artifact_root"]
    short_body: dict[str, Any] = {
        "format": selector.SHORT_TRAJECTORY_FORMAT,
        "status": "complete",
        "topology_mode": args.mode,
        "topology_gate_spec_sha256": topology_gate["sha256"],
        "quality_gate_spec_sha256": quality_gate["sha256"],
        "quality_protocol_version": selector.QUALITY_PROTOCOL_VERSION,
        "artifact_root_namespace": (
            selector.QUALITY_ARTIFACT_ROOT_NAMESPACE
        ),
        "artifact_root": artifact_root,
        "quality_role": quality_role,
        "reference_only": reference_only,
        "late_w1_status": "not_measured",
        "w1_tail_equivalence_claimed": False,
        "candidate_epochs": list(quality_epochs),
        "split": "val",
        "test_visible": False,
        "topology_independent_input_sha256": semantic_sha,
        "candidate_ready_receipts": ready,
        "candidates": [
            {
                "epoch": epoch,
                "optimizer_updates": epoch
                * int(
                    contract.TOPOLOGY_SPECS[args.mode][
                        "updates_per_epoch"
                    ]
                ),
                "candidate_checkpoint": short_checkpoints[epoch],
            }
            for epoch in quality_epochs
        ],
    }
    short_body["receipt_payload_sha256"] = (
        contract.canonical_json_sha256(short_body)
    )
    contract._write_new_json(short_output, short_body)
    short_value = {
        "path": str(short_output),
        "sha256": hashlib.sha256(short_output.read_bytes()).hexdigest(),
        "bytes": short_output.stat().st_size,
        "receipt_payload_sha256": short_body[
            "receipt_payload_sha256"
        ],
    }
    short, short_payload = selector._artifact(
        short_value,
        "constructed short trajectory",
        payload=True,
    )
    assert short_payload is not None
    if short_payload != short_body:
        raise selector.TopologySelectionError(
            "constructed short-trajectory receipt changed during publish"
        )
    (
        val_inputs_source,
        val_inputs_artifact,
        coverage,
        pipeline_source,
        pipeline_artifact,
        pipeline_payload,
    ) = selector.validate_quality_common_authority(
        _artifact4(args.val_inputs_receipt, "val-input receipt"),
        _artifact4(args.pipeline_receipt, "pipeline receipt"),
    )
    lineages = _epoch_artifacts(
        args.inference_lineage,
        label="inference lineage",
        payload=True,
        expected_epochs=quality_epochs,
    )
    reports = _epoch_artifacts(
        args.diffsheg_report,
        label="DiffSHEG report",
        payload=False,
        expected_epochs=quality_epochs,
    )

    candidates: list[dict[str, Any]] = []
    semantic_sha = short_payload["topology_independent_input_sha256"]
    for epoch in quality_epochs:
        validated = selector.validate_quality_candidate_provenance(
            args.mode,
            epoch,
            topology_independent_input_sha256=semantic_sha,
            short_trajectory_receipt=short,
            expected_checkpoint=short_checkpoints[epoch],
            candidate_checkpoint=short_checkpoints[epoch],
            val_inputs_receipt=val_inputs_source,
            val_inputs_artifact=val_inputs_artifact,
            pipeline_receipt=pipeline_source,
            pipeline_artifact=pipeline_artifact,
            pipeline_payload=pipeline_payload,
            expected_coverage=coverage,
            inference_lineage=lineages[epoch],
            diffsheg_report=reports[epoch],
        )
        candidates.append(
            {
                "epoch": epoch,
                "candidate_checkpoint": validated[
                    "candidate_checkpoint"
                ],
                "inference_lineage": validated["inference_lineage"],
                "diffsheg_report": validated["diffsheg_report"],
                "provenance": validated["provenance"],
            }
        )
    report: dict[str, Any] = {
        "format": selector.QUALITY_REPORT_FORMAT,
        "status": "pass",
        "mode": args.mode,
        "quality_gate_spec_sha256": quality_gate["sha256"],
        "quality_protocol_version": selector.QUALITY_PROTOCOL_VERSION,
        "artifact_root_namespace": (
            selector.QUALITY_ARTIFACT_ROOT_NAMESPACE
        ),
        "artifact_root": artifact_root,
        "quality_role": quality_role,
        "reference_only": reference_only,
        "late_w1_status": "not_measured",
        "w1_tail_equivalence_claimed": False,
        "split": "val",
        "test_visible": False,
        "trajectory_epochs": list(quality_epochs),
        "topology_independent_input_sha256": semantic_sha,
        "short_trajectory_receipt": short,
        "val_inputs_receipt": val_inputs_source,
        "pipeline_receipt": pipeline_source,
        "candidates": candidates,
    }
    report["receipt_sha256"] = contract.canonical_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = produce(args)
    contract._write_new_json(args.output.expanduser().resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
