#!/usr/bin/env python3
"""Build the frozen DiffSHEG-FGD validation input authority for Base.

The canonical SHOW validation rows and eight audio shards are identical to
the fresh Base inference inputs.  This receipt additionally freezes the
DiffSHEG 88-frame window manifest digest consumed by the formal 22-way
selector.  It has no split switch and cannot materialize test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.show_base import build_base_val_receipts as common
from scripts.show_base import select_base_official_adapt as selector


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    canonical, canonical_path, canonical_bytes = common._artifact(
        args.canonical_manifest,
        "DiffSHEG validation canonical manifest",
    )
    canonical_ids, coverage = selector._canonical_coverage(
        canonical_bytes,
        str(canonical_path),
    )
    canonical_summary, canonical_lineage = (
        selector._validate_val_canonical_receipts(
            summary_value=common._artifact(
                args.canonical_summary,
                "DiffSHEG validation canonical summary",
            )[0],
            lineage_value=common._artifact(
                args.canonical_lineage,
                "DiffSHEG validation canonical lineage",
            )[0],
            canonical_manifest_sha256=canonical["sha256"],
        )
    )
    audio = selector._audio_coverage(
        [
            common._artifact(path, "DiffSHEG validation audio manifest")[0]
            for path in args.audio_manifest
        ],
        [
            common._artifact(path, "DiffSHEG validation audio summary")[0]
            for path in args.audio_summary
        ],
        [
            common._artifact(path, "DiffSHEG validation audio lineage")[0]
            for path in args.audio_lineage
        ],
        canonical_ids,
    )
    receipt = common._payload_receipt(
        {
            "format": selector.VAL_INPUTS_FORMAT,
            "status": "frozen",
            "split": "val",
            "test_visible": False,
            "expected_clip_count": selector.EXPECTED_VAL_CLIPS,
            "canonical_manifest": canonical,
            "canonical_summary": canonical_summary,
            "canonical_lineage": canonical_lineage,
            "audio_manifests": audio["manifests"],
            "audio_summaries": audio["summaries"],
            "audio_lineages": audio["lineages"],
            "clip_ids_sha256": coverage["clip_ids_sha256"],
            "diffsheg_clip_manifest_sha256": coverage[
                "diffsheg_clip_manifest_sha256"
            ],
        }
    )
    output, output_sha = common._atomic_new_json(
        args.output,
        receipt,
        label="DiffSHEG validation inputs receipt",
        validator=selector.validate_val_inputs,
    )
    return {
        "status": "complete",
        "kind": "diffsheg-inputs",
        "split": "val",
        "test_visible": False,
        "path": str(output),
        "sha256": output_sha,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "clip_count": coverage["clip_count"],
        "audio_shards": audio["num_shards"],
        "diffsheg_clip_manifest_sha256": coverage[
            "diffsheg_clip_manifest_sha256"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the exact 1,715-clip DiffSHEG val input receipt.",
        allow_abbrev=False,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument(
        "--audio-manifest", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--audio-summary", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--audio-lineage", type=Path, action="append", required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("audio_manifest", "audio_summary", "audio_lineage"):
        if len(getattr(args, name)) != selector.EXPECTED_AUDIO_SHARDS:
            parser.error(f"--{name.replace('_', '-')} must appear 8 times")
    try:
        report = build_inputs(args)
    except (
        common.ReceiptBuildError,
        selector.SelectionContractError,
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
