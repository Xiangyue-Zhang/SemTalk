#!/usr/bin/env python3
"""Produce one selector-ready Base validation DiffSHEG-FGD measurement.

Every input is externally SHA-pinned and freshly replayed before the receipt
is published.  The command accepts only one of the 22 registered validation
epochs, has no test input, and creates its output with new-only semantics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.show_base import base_long_val_contract as contract
from scripts.show_base import select_base_official_adapt as legacy
from scripts.show_base import select_base_official_adapt_long as long_selector


def _epoch(value: str) -> int:
    try:
        epoch = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch must be an integer") from error
    if epoch not in contract.EXPECTED_CANDIDATE_EPOCHS:
        raise argparse.ArgumentTypeError(
            "epoch must be one of "
            + ",".join(str(item) for item in contract.EXPECTED_CANDIDATE_EPOCHS)
            + "; e30 is withdrawn"
        )
    return epoch


def _payload_artifact(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    resolved, value, observed = legacy._verified_json(
        path,
        expected_sha256,
        label,
    )
    claimed = legacy.require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256", None)
    if legacy.canonical_json_sha256(unsigned) != claimed:
        raise legacy.SelectionContractError(f"{label} payload SHA mismatch")
    return {
        "path": str(resolved),
        "sha256": observed,
        "receipt_payload_sha256": claimed,
    }, value


def build_measurement(
    *,
    epoch: int,
    candidate_bundle: Mapping[str, Any],
    val_inputs_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
    inference_lineage_artifact: Mapping[str, Any],
    diffsheg_report_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_bundle["candidates"][epoch]
    measurement = {
        "format": legacy.MEASUREMENT_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "epoch": epoch,
        "candidate_checkpoint": {
            "path": candidate["path"],
            "sha256": candidate["sha256"],
        },
        "val_inputs_receipt": dict(val_inputs_artifact),
        "pipeline_receipt": dict(pipeline_artifact),
        "inference_lineage": dict(inference_lineage_artifact),
        "diffsheg_report": dict(diffsheg_report_artifact),
    }
    measurement["receipt_payload_sha256"] = (
        legacy.canonical_json_sha256(measurement)
    )
    return measurement


def produce(args: argparse.Namespace) -> dict[str, Any]:
    legacy.reject_forbidden_source_labels(*vars(args).values())
    output = args.output_json.expanduser()
    if not output.is_absolute():
        raise legacy.SelectionContractError(
            "measurement output must be absolute"
        )
    output = output.parent.resolve() / output.name
    legacy.reject_test_path(output, "Base DiffSHEG measurement output")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")

    val_artifact, val_coverage = contract.validate_val_inputs(
        args.val_inputs_json,
        args.expected_val_inputs_sha256,
    )
    pipeline_artifact, pipeline = contract.validate_pipeline(
        args.pipeline_json,
        args.expected_pipeline_sha256,
    )
    expected_selected = {
        stage: pipeline["fixed_checkpoints"][stage]["sha256"]
        for stage in ("face", "hands", "upper", "lower", "global")
    }
    candidate_bundle = contract.validate_candidate_bundle(
        manifest_path=args.base_candidate_manifest,
        expected_manifest_sha256=(
            args.expected_base_candidate_manifest_sha256
        ),
        status_path=args.base_status_json,
        expected_status_sha256=args.expected_base_formal_status_sha256,
        frozen_inputs_path=args.base_frozen_inputs_json,
        expected_frozen_inputs_sha256=(
            args.expected_base_frozen_inputs_sha256
        ),
        expected_selected_prerequisite_sha256=expected_selected,
    )
    candidate = candidate_bundle["candidates"][args.epoch]
    lineage_artifact, lineage = _payload_artifact(
        args.inference_lineage_json,
        args.expected_inference_lineage_sha256,
        f"Base e{args.epoch} DiffSHEG validation lineage",
    )
    audited_lineage, inference = contract.validate_val_inference_lineage(
        Path(lineage_artifact["path"]),
        lineage_artifact["sha256"],
        epoch=args.epoch,
        expected_candidate=candidate,
        val_inputs_artifact=val_artifact,
        pipeline_artifact=pipeline_artifact,
        expected_coverage=val_coverage,
    )
    if audited_lineage != lineage_artifact:
        raise legacy.SelectionContractError(
            "validation lineage artifact changed during replay"
        )

    report_artifact, report_path, report_bytes = legacy._verify_artifact(
        {
            "path": str(args.diffsheg_report_json.resolve()),
            "sha256": args.expected_diffsheg_report_sha256,
        },
        f"Base e{args.epoch} DiffSHEG validation report",
    )
    report = legacy._strict_json_bytes(
        report_bytes,
        f"Base e{args.epoch} DiffSHEG validation report {report_path}",
    )
    legacy.validate_diffsheg_report(
        report,
        expected_coverage=val_coverage,
        inference_lineage=inference,
    )
    measurement = build_measurement(
        epoch=args.epoch,
        candidate_bundle=candidate_bundle,
        val_inputs_artifact=val_artifact,
        pipeline_artifact=pipeline_artifact,
        inference_lineage_artifact=lineage_artifact,
        diffsheg_report_artifact=report_artifact,
    )
    legacy._receipt.atomic_json_new(output, measurement)
    try:
        output_sha = legacy.sha256_file(output)
        audited, row = long_selector.validate_measurement(
            measurement_path=output,
            expected_measurement_sha256=output_sha,
            candidate_bundle=candidate_bundle,
        )
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "epoch": args.epoch,
        "fgd": row["metrics"]["fgd"],
        "measurement": audited,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--epoch", type=_epoch, required=True)
    parser.add_argument("--base-candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-base-candidate-manifest-sha256", required=True
    )
    parser.add_argument("--base-status-json", type=Path, required=True)
    parser.add_argument("--expected-base-formal-status-sha256", required=True)
    parser.add_argument("--base-frozen-inputs-json", type=Path, required=True)
    parser.add_argument("--expected-base-frozen-inputs-sha256", required=True)
    parser.add_argument("--val-inputs-json", type=Path, required=True)
    parser.add_argument("--expected-val-inputs-sha256", required=True)
    parser.add_argument("--pipeline-json", type=Path, required=True)
    parser.add_argument("--expected-pipeline-sha256", required=True)
    parser.add_argument("--inference-lineage-json", type=Path, required=True)
    parser.add_argument("--expected-inference-lineage-sha256", required=True)
    parser.add_argument("--diffsheg-report-json", type=Path, required=True)
    parser.add_argument("--expected-diffsheg-report-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = produce(args)
    except (
        contract.SelectionContractError,
        legacy.SelectionContractError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
