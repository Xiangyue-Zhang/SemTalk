#!/usr/bin/env python3
"""Publish one immutable five-stage prerequisite continuation authority.

The heavy validation remains in :mod:`prerequisite_continuation_wave`.  This
small CLI gives the formal operator an atomic, hash-bound publication path so
that an authorized ``eN -> eN+20`` wave never has to be assembled by hand or
written with an overwrite-capable command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract


FORMAT = "semtalk_show_prerequisite_continuation_stage_plans_v1"
TOP_KEYS = {
    "format",
    "status",
    "test_visible",
    "stages",
    "receipt_payload_sha256",
}


class PublishContinuationWaveError(RuntimeError):
    """Raised when stage plans or publication evidence are not exact."""


def _load_stage_plans(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    try:
        resolved, payload, observed_sha = contract.read_verified_file(
            path,
            expected_sha256,
            "continuation stage plans",
            val_only=False,
        )
        value = contract.strict_json_bytes(payload, str(resolved))
    except (contract.ContractError, OSError) as error:
        raise PublishContinuationWaveError(str(error)) from error
    if observed_sha != expected_sha256:
        raise PublishContinuationWaveError(
            "continuation stage-plan file SHA-256 mismatch"
        )
    if not isinstance(value, dict) or set(value) != TOP_KEYS:
        raise PublishContinuationWaveError(
            "continuation stage-plan schema mismatch"
        )
    if (
        value["format"] != FORMAT
        or value["status"] != "complete"
        or value["test_visible"] is not False
    ):
        raise PublishContinuationWaveError(
            "continuation stage-plan protocol mismatch"
        )
    stages = value["stages"]
    if (
        not isinstance(stages, dict)
        or set(stages) != set(wave.STAGES)
        or any(not isinstance(stages[stage], dict) for stage in wave.STAGES)
    ):
        raise PublishContinuationWaveError(
            "continuation stage plans must cover exactly five stages"
        )
    claimed_payload_sha = value["receipt_payload_sha256"]
    try:
        contract.require_sha256(
            claimed_payload_sha,
            "continuation stage-plan payload SHA-256",
        )
        unsigned = dict(value)
        unsigned.pop("receipt_payload_sha256")
        observed_payload_sha = wave.canonical_json_sha256(unsigned)
    except (contract.ContractError, wave.ContinuationWaveError) as error:
        raise PublishContinuationWaveError(str(error)) from error
    if observed_payload_sha != claimed_payload_sha:
        raise PublishContinuationWaveError(
            "continuation stage-plan payload SHA-256 mismatch"
        )
    return (
        {stage: dict(stages[stage]) for stage in wave.STAGES},
        {
            "path": str(resolved),
            "sha256": observed_sha,
            "receipt_payload_sha256": claimed_payload_sha,
        },
    )


def publish_wave(
    *,
    decision_path: Path,
    expected_decision_sha256: str,
    stage_plans_path: Path,
    expected_stage_plans_sha256: str,
    output_json: Path,
) -> tuple[dict[str, Any], str]:
    stage_plans, _ = _load_stage_plans(
        stage_plans_path,
        expected_stage_plans_sha256,
    )
    try:
        receipt = wave.authorize_wave_from_replayed_inputs(
            decision_path=decision_path,
            expected_decision_sha256=expected_decision_sha256,
            stage_plans=stage_plans,
        )
        file_sha = contract.atomic_json_new(output_json, receipt)
        replayed = wave.replay_wave_file(output_json, file_sha)
    except (
        contract.ContractError,
        wave.ContinuationWaveError,
        OSError,
        ValueError,
    ) as error:
        raise PublishContinuationWaveError(
            f"cannot publish continuation wave: {error}"
        ) from error
    if wave.canonical_json_bytes(replayed) != wave.canonical_json_bytes(
        receipt
    ):
        raise PublishContinuationWaveError(
            "published continuation wave differs from fresh replay"
        )
    try:
        _, _, observed_file_sha = contract.read_verified_file(
            output_json,
            file_sha,
            "published continuation wave",
            val_only=False,
        )
    except (contract.ContractError, OSError) as error:
        raise PublishContinuationWaveError(
            "published continuation wave changed or disappeared"
        ) from error
    if observed_file_sha != file_sha:
        raise PublishContinuationWaveError(
            "published continuation wave changed after replay"
        )
    return receipt, file_sha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one synchronized SHOW prerequisite +20 wave",
        allow_abbrev=False,
    )
    parser.add_argument("--decision-json", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--stage-plans-json", type=Path, required=True)
    parser.add_argument("--expected-stage-plans-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt, file_sha = publish_wave(
        decision_path=args.decision_json,
        expected_decision_sha256=args.expected_decision_sha256,
        stage_plans_path=args.stage_plans_json,
        expected_stage_plans_sha256=args.expected_stage_plans_sha256,
        output_json=args.output_json,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "boundary_epoch": receipt["boundary_epoch"],
                "target_epoch": receipt["target_epoch"],
                "file_sha256": file_sha,
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
