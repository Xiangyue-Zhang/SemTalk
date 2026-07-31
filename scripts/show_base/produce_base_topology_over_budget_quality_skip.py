#!/usr/bin/env python3
"""Seal one immutable SemTalk Base over-ETA quality-skip receipt.

All five real throughput probes remain mandatory.  This CPU-only producer is
usable only for a non-W1 topology whose validated 400-epoch ETA is strictly
greater than 24 hours.  It binds the skip to the frozen source authority, both
gate specifications, the exact probe artifact, and the shared five-stage SHOW
input identity.  W1 and every within-budget topology still require the formal
e1/e2/e4/e8 quality trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import select_base_training_topology as selector
from scripts.show_base import train_base_official_adapt_long as contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=[
            mode
            for mode in contract.TOPOLOGY_SPECS
            if mode != contract.OFFICIAL_W1_REFERENCE_MODE
        ],
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
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--expected-probe-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def produce(args: argparse.Namespace) -> dict[str, object]:
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
    return selector.build_quality_skip_receipt(
        args.mode,
        args.probe_report,
        args.expected_probe_report_sha256,
        topology_gate_spec_sha256=topology_gate["sha256"],
        quality_gate_spec_sha256=quality_gate["sha256"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = produce(args)
    contract._write_new_json(args.output.expanduser().resolve(), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
