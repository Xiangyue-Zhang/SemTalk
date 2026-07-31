#!/usr/bin/env python3
"""Freshly replay one external SHOW prerequisite selected-five receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.show_base import select_prerequisite_candidates as selector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rehash every external prerequisite artifact and recompute all "
            "five validation winners"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-sha256",
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = selector.replay_selection(
        selection_path=args.selection_json,
        expected_selection_sha256=args.expected_selection_sha256,
    )
    print(
        json.dumps(
            {
                "status": "fresh-replay-pass",
                "format": result["format"],
                "candidate_epochs": result["protocol"][
                    "candidate_epochs"
                ],
                "candidate_epochs_by_stage": result["protocol"].get(
                    "candidate_epochs_by_stage",
                    {
                        item["stage"]: result["protocol"]["candidate_epochs"]
                        for item in result["stages"]
                    },
                ),
                "selected": {
                    item["stage"]: {
                        "epoch": item["epoch"],
                        "optimizer_updates": item["optimizer_updates"],
                        "checkpoint_sha256": item[
                            "candidate_checkpoint"
                        ]["sha256"],
                    }
                    for item in result["stages"]
                },
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
