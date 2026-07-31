#!/usr/bin/env python3
"""Select the unique validation-FGD winner from all 22 long Base candidates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.show_base import base_long_val_contract as long_contract
from scripts.show_base import select_base_official_adapt as legacy


def _profile_values() -> dict[str, Any]:
    return {
        "EXPECTED_CANDIDATE_EPOCHS": (
            long_contract.EXPECTED_CANDIDATE_EPOCHS
        ),
        "EXPECTED_UPDATES_PER_EPOCH": (
            long_contract.EXPECTED_UPDATES_PER_EPOCH
        ),
        "CANDIDATE_MANIFEST_FORMAT": (
            long_contract.CANDIDATE_MANIFEST_FORMAT
        ),
        "CANDIDATE_STATUS_FORMAT": long_contract.CANDIDATE_STATUS_FORMAT,
        "THROUGHPUT_GATE_FORMAT": (
            long_contract.THROUGHPUT_GATE_FORMAT
        ),
        "SELECTION_FORMAT": long_contract.SELECTION_FORMAT,
        "validate_candidate_bundle": (
            long_contract.validate_candidate_bundle
        ),
        # The audited decision math lives in the legacy DiffSHEG selector,
        # but the long producer uses the fresh five-prerequisite pipeline.
        # Patch the complete producer/consumer ABI, not just candidate
        # constants, so TalkSHOW-v2 lineage can never be mixed with a
        # DiffSHEG measurement receipt.
        "validate_val_inputs": long_contract.validate_val_inputs,
        "validate_pipeline": long_contract.validate_pipeline,
        "validate_val_inference_lineage": (
            long_contract.validate_val_inference_lineage
        ),
        "public_val_coverage": long_contract.public_val_coverage,
        "VAL_INFERENCE_LINEAGE_FORMAT": (
            long_contract.VAL_INFERENCE_LINEAGE_FORMAT
        ),
    }


def _activate_long_profile() -> None:
    """Install the long constants for compatibility fixture construction.

    Production selection uses :func:`_long_profile`, which always restores
    the audited legacy module.  This explicit compatibility hook exists for
    callers that must build a complete long fixture through the legacy helper
    before calling :func:`build_selection`; such callers are responsible for
    restoring their captured values.
    """

    for name, value in _profile_values().items():
        setattr(legacy, name, value)


@contextmanager
def _long_profile() -> Any:
    """Temporarily install the long envelope around audited decision math."""

    names = _profile_values()
    previous = {name: getattr(legacy, name) for name in names}
    try:
        for name, value in names.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(legacy, name, value)


def build_selection(
    *,
    candidate_bundle: Mapping[str, Any],
    measurement_paths: Sequence[Path],
    expected_measurement_sha256: Sequence[str],
) -> dict[str, Any]:
    with _long_profile():
        selection = legacy.build_selection(
            candidate_bundle=candidate_bundle,
            measurement_paths=measurement_paths,
            expected_measurement_sha256=expected_measurement_sha256,
        )
    selection["source_roles"]["base_candidate_producer"] = dict(
        candidate_bundle["producer_source"]
    )
    unsigned = dict(selection)
    unsigned.pop("receipt_payload_sha256", None)
    selection["receipt_payload_sha256"] = legacy.canonical_json_sha256(
        unsigned
    )
    return selection


def validate_measurement(
    *,
    measurement_path: Path,
    expected_measurement_sha256: str,
    candidate_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fresh-validate one production 22-way DiffSHEG measurement receipt."""

    with _long_profile():
        artifact, row, _val_inputs, _pipeline = legacy._measurement_artifact(
            measurement_path,
            expected_measurement_sha256,
            candidates=candidate_bundle["candidates"],
            common_val_inputs=None,
            common_pipeline=None,
        )
    return artifact, row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select long official-initialized SemTalk Base from all 22 "
            "validation FGD measurements"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--base-candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-base-candidate-manifest-sha256",
        required=True,
    )
    parser.add_argument("--base-status-json", type=Path, required=True)
    parser.add_argument(
        "--expected-base-formal-status-sha256",
        required=True,
    )
    parser.add_argument("--base-frozen-inputs-json", type=Path, required=True)
    parser.add_argument(
        "--expected-base-frozen-inputs-sha256",
        required=True,
    )
    parser.add_argument(
        "--measurement-json",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly 22 times in the registered epoch order",
    )
    parser.add_argument(
        "--expected-measurement-sha256",
        action="append",
        required=True,
        help="repeat exactly 22 times in matching epoch order",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    legacy.reject_forbidden_source_labels(args.output_json)
    output = args.output_json.resolve()
    legacy.reject_forbidden_source_labels(output)
    legacy.reject_test_path(output, "long Base selection output")
    if args.output_json.is_symlink() or args.output_json.exists():
        raise legacy.SelectionContractError(
            f"refusing to overwrite existing selection: {args.output_json}"
        )
    candidate_bundle = long_contract.validate_candidate_bundle(
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
    )
    selection = build_selection(
        candidate_bundle=candidate_bundle,
        measurement_paths=args.measurement_json,
        expected_measurement_sha256=args.expected_measurement_sha256,
    )
    try:
        legacy._receipt.atomic_json_new(args.output_json, selection)
    except (OSError, ValueError, RuntimeError) as error:
        raise legacy.SelectionContractError(str(error)) from error
    print(
        json.dumps(
            {
                "status": "selected",
                "split": "val",
                "test_visible": False,
                "candidate_count": len(
                    long_contract.EXPECTED_CANDIDATE_EPOCHS
                ),
                "selected_epoch": selection["selected"]["epoch"],
                "selected_fgd": selection["selected"]["fgd"],
                "output": str(args.output_json.resolve()),
                "receipt_payload_sha256": selection[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
