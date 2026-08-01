#!/usr/bin/env python3
"""Build the frozen DiffSHEG-primary fresh Base validation pipeline.

This CPU-only entry point is intentionally separate from
``build_base_val_receipts.py``.  That module owns the TalkSHOW compatibility
pipeline, while long Base checkpoint selection is owned by
``select_base_official_adapt`` and its DiffSHEG-primary source closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import build_base_val_receipts as common
from scripts.show_base import select_base_official_adapt as primary


def build_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    payload = primary.build_fresh_pipeline_payload(
        source_root=args.source_root,
        prerequisite_selection=args.prerequisite_selection,
        expected_prerequisite_selection_sha256=(
            args.expected_prerequisite_selection_sha256
        ),
    )
    output, output_sha = common._atomic_new_json(
        args.output,
        payload,
        label="DiffSHEG-primary fresh validation pipeline receipt",
        validator=primary.validate_fresh_pipeline,
    )
    return {
        "status": "complete",
        "kind": "diffsheg-fresh-pipeline",
        "split": "val",
        "test_visible": False,
        "path": str(output),
        "sha256": output_sha,
        "receipt_payload_sha256": payload["receipt_payload_sha256"],
        "source": payload["source"],
        "prerequisite_selection": payload["prerequisite_selection"],
        "fixed_checkpoint_sha256": {
            stage: payload["fixed_checkpoints"][stage]["sha256"]
            for stage in ("face", "hands", "upper", "lower", "global")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the validation-only DiffSHEG-primary fresh Base pipeline."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--prerequisite-selection",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-prerequisite-selection-sha256",
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_pipeline(args)
    except (
        common.ReceiptBuildError,
        primary.SelectionContractError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
        TypeError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
