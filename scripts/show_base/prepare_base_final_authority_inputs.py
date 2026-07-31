#!/usr/bin/env python3
"""Deterministically assemble the sole SemTalk Base final-authority inputs.

All file receipts are computed from regular, canonical, non-symlink inputs.
The producer derives the detached SemTalk source receipt itself, fixes the
continuation-wave inventory to the audited empty list, and writes one
exclusive canonical JSON document for ``base_final_authority.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_final_authority as authority


class AuthorityInputsError(RuntimeError):
    """Raised when a component cannot be frozen without ambiguity."""


def _regular_artifact(value: Path, label: str) -> dict[str, Any]:
    path = value.expanduser()
    if not path.is_absolute():
        raise AuthorityInputsError(f"{label} must be absolute")
    resolved = path.resolve(strict=True)
    if path != resolved:
        raise AuthorityInputsError(f"{label} must be canonical")
    observed = os.lstat(resolved)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise AuthorityInputsError(f"{label} must be a regular non-symlink file")
    payload = resolved.read_bytes()
    if not payload:
        raise AuthorityInputsError(f"{label} must be non-empty")
    after = os.lstat(resolved)
    if (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise AuthorityInputsError(f"{label} changed while being frozen")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuthorityInputsError(f"cannot attest source root {root}") from error


def _source_receipt(root_value: Path) -> dict[str, Any]:
    root = root_value.expanduser()
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise AuthorityInputsError("inference source root must be canonical")
    if root.is_symlink() or not root.is_dir():
        raise AuthorityInputsError("inference source root must be a directory")
    entrypoint_relative = Path(
        "scripts/show_base/semtalk_base_inference_core.py"
    )
    entrypoint = _regular_artifact(
        root / entrypoint_relative,
        "SemTalk Base inference entrypoint",
    )
    detached = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).returncode != 0
    local_branches = _git(
        root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/heads",
    ).splitlines()
    if (
        _git(root, "remote", "get-url", "origin") != authority.ORIGIN
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        or not detached
        or local_branches
    ):
        raise AuthorityInputsError(
            "inference source must be clean, detached, exact-origin, and branchless"
        )
    return {
        "source_root": str(root),
        "origin": authority.ORIGIN,
        "commit": _git(root, "rev-parse", "HEAD^{commit}"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "clean": True,
        "entrypoint": entrypoint_relative.as_posix(),
        "entrypoint_sha256": entrypoint["sha256"],
    }


def _indexed_groups(
    values: Sequence[Sequence[str]],
    *,
    expected: Sequence[str],
    label: str,
) -> dict[str, Sequence[str]]:
    result: dict[str, Sequence[str]] = {}
    for value in values:
        key = value[0]
        if key in result:
            raise AuthorityInputsError(f"duplicate {label} {key}")
        result[key] = value[1:]
    if tuple(sorted(result)) != tuple(sorted(expected)):
        raise AuthorityInputsError(f"{label} coverage mismatch")
    return result


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_receipt(args.inference_source_root)
    canonical_manifest = _regular_artifact(
        args.canonical_manifest, "canonical manifest"
    )
    canonical_summary = _regular_artifact(
        args.canonical_summary, "canonical summary"
    )
    canonical_lineage = _regular_artifact(
        args.canonical_lineage, "canonical lineage"
    )
    audio_groups = _indexed_groups(
        args.audio_shard,
        expected=tuple(str(index) for index in range(authority.NUM_SHARDS)),
        label="audio shard",
    )
    audio_authorities = []
    for shard_id in range(authority.NUM_SHARDS):
        manifest, summary, lineage = audio_groups[str(shard_id)]
        audio_authorities.append(
            {
                "shard_id": shard_id,
                "manifest": _regular_artifact(Path(manifest), f"audio {shard_id} manifest"),
                "summary": _regular_artifact(Path(summary), f"audio {shard_id} summary"),
                "lineage": _regular_artifact(Path(lineage), f"audio {shard_id} lineage"),
            }
        )
    checkpoint_groups = _indexed_groups(
        args.checkpoint,
        expected=authority.CHECKPOINT_STAGES,
        label="checkpoint stage",
    )
    checkpoints = {
        stage: _regular_artifact(
            Path(checkpoint_groups[stage][0]), f"{stage} checkpoint"
        )
        for stage in authority.CHECKPOINT_STAGES
    }
    output_root = args.expected_output_root.expanduser()
    if not output_root.is_absolute():
        raise AuthorityInputsError("expected output root must be absolute")
    unsigned: dict[str, Any] = {
        "format": authority.INPUTS_FORMAT,
        "status": "ready",
        "selection_protocol": authority.BASE_SELECTION_PROTOCOL,
        "test_policy": {
            "test_evaluations": 1,
            "test_feedback_into_selection": False,
        },
        "expected_output_root": str(output_root.resolve()),
        "canonical_manifest": canonical_manifest,
        "canonical_summary": canonical_summary,
        "canonical_lineage": canonical_lineage,
        "canonical_root_receipt": {
            "manifest": canonical_manifest,
            "summary": canonical_summary,
            "lineage": canonical_lineage,
            "source_commit": source["commit"],
            "source_tree": source["tree"],
        },
        "audio_authorities": audio_authorities,
        "base_long_candidate_artifacts": {
            "manifest": _regular_artifact(
                args.base_long_manifest, "Base-long manifest"
            ),
            "status": _regular_artifact(args.base_long_status, "Base-long status"),
            "frozen_inputs": _regular_artifact(
                args.base_long_frozen_inputs, "Base-long frozen inputs"
            ),
        },
        "winner_selection": _regular_artifact(
            args.winner_selection, "validation winner selection"
        ),
        "continuation_decision": _regular_artifact(
            args.continuation_decision, "continuation stop decision"
        ),
        "continuation_waves": [],
        "winner_validation_metric_closure": _regular_artifact(
            args.winner_validation_metric_closure,
            "selected DiffSHEG validation report",
        ),
        "test_claim": _regular_artifact(args.test_claim, "one-shot test claim"),
        "inference_source": source,
        "checkpoints": checkpoints,
    }
    return {
        **unsigned,
        "receipt_payload_sha256": authority.canonical_json_sha256(unsigned),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--expected-output-root", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument(
        "--audio-shard",
        nargs=4,
        action="append",
        metavar=("ID", "MANIFEST", "SUMMARY", "LINEAGE"),
        required=True,
    )
    parser.add_argument("--base-long-manifest", type=Path, required=True)
    parser.add_argument("--base-long-status", type=Path, required=True)
    parser.add_argument("--base-long-frozen-inputs", type=Path, required=True)
    parser.add_argument("--winner-selection", type=Path, required=True)
    parser.add_argument("--continuation-decision", type=Path, required=True)
    parser.add_argument(
        "--winner-validation-metric-closure", type=Path, required=True
    )
    parser.add_argument("--test-claim", type=Path, required=True)
    parser.add_argument("--inference-source-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("STAGE", "PATH"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_json.expanduser()
    if not output.is_absolute() or output != output.parent.resolve() / output.name:
        raise AuthorityInputsError("output JSON must be a canonical absolute path")
    value = build_inputs(args)
    authority.atomic_write_new(output, value)
    payload = output.read_bytes()
    print(
        json.dumps(
            {
                "status": "ready",
                "output": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "receipt_payload_sha256": value["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
