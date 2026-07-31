#!/usr/bin/env python3
"""Authorize one Base test checkpoint only after 22-way val selection.

This module has no test-data argument.  It validates the complete long
training transaction and validation-only selection, recomputes the strict
``min(FGD, epoch)`` winner, and proves that the requested checkpoint is that
winner.  A downstream test launcher can consume the returned receipt without
feeding any test result back into selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.show_base import base_long_val_contract as long_contract
from scripts.show_base import select_base_official_adapt as legacy
from scripts.show_base import select_base_official_adapt_long as long_selector


AUTHORIZATION_FORMAT = "semtalk_show_base_long_test_winner_authorization_v1"


class TestWinnerContractError(RuntimeError):
    """Raised when a test checkpoint is not the unique val-only winner."""


def _reject_selection_paths(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_selection_paths(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_selection_paths(child, f"{label}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        legacy.reject_forbidden_source_labels(value)
        legacy.reject_test_path(Path(value), label)


def _artifact(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise TestWinnerContractError(f"{label} artifact schema mismatch")
    path = legacy.require_absolute_path(value["path"], label)
    resolved = legacy._regular_file(path, label)
    digest = legacy.require_sha256(value["sha256"], f"{label} SHA-256")
    if legacy.sha256_file(resolved) != digest:
        raise TestWinnerContractError(f"{label} artifact changed")
    return resolved, digest


def validate_test_winner(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    candidate_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        selection_resolved, selection, selection_sha = legacy._verified_json(
            selection_path,
            expected_selection_sha256,
            "long Base validation selection",
        )
        legacy.reject_test_path(
            selection_resolved,
            "long Base validation selection",
        )
        _reject_selection_paths(
            selection,
            "long Base validation selection",
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        legacy.SelectionContractError,
    ) as error:
        raise TestWinnerContractError(
            f"long Base selection input is invalid: {error}"
        ) from error
    measurement_receipts = selection.get("measurement_receipts")
    if (
        not isinstance(measurement_receipts, list)
        or len(measurement_receipts)
        != len(long_contract.EXPECTED_CANDIDATE_EPOCHS)
    ):
        raise TestWinnerContractError(
            "long Base selection does not bind exactly 22 measurements"
        )
    try:
        rebuilt = long_selector.build_selection(
            candidate_bundle=candidate_bundle,
            measurement_paths=[
                Path(receipt["path"])
                for receipt in measurement_receipts
            ],
            expected_measurement_sha256=[
                receipt["sha256"]
                for receipt in measurement_receipts
            ],
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        legacy.SelectionContractError,
    ) as error:
        raise TestWinnerContractError(
            f"long Base selection replay failed: {error}"
        ) from error
    if rebuilt != selection:
        raise TestWinnerContractError(
            "long Base selection differs from replayed DiffSHEG reports"
        )
    claimed_payload_sha = legacy.require_sha256(
        selection.get("receipt_payload_sha256"),
        "long Base selection payload SHA-256",
    )
    unsigned = dict(selection)
    unsigned.pop("receipt_payload_sha256", None)
    if (
        legacy.canonical_json_sha256(unsigned) != claimed_payload_sha
        or selection.get("format") != long_contract.SELECTION_FORMAT
        or selection.get("status") != "selected"
        or selection.get("split") != "val"
        or selection.get("test_visible") is not False
        or selection.get("selection_eligible") is not True
    ):
        raise TestWinnerContractError(
            "long Base selection is not canonical val-only evidence"
        )
    expected_policy = {
        "candidate_epochs": list(
            long_contract.EXPECTED_CANDIDATE_EPOCHS
        ),
        "metric": "FGD",
        "metric_report_key": "fgd",
        "operator": "min",
        "tie_break": "lowest_epoch",
        "ordering": ["fgd", "epoch"],
        "test_feedback_into_selection": False,
    }
    if selection.get("selection_policy") != expected_policy:
        raise TestWinnerContractError(
            "long Base selection policy is not strict min(FGD, epoch)"
        )
    if selection.get("test_policy") != {
        "authorized_evaluations": 1,
        "one_shot_claim_required": True,
        "selection_feedback": False,
    }:
        raise TestWinnerContractError(
            "long Base selection does not enforce one-shot test semantics"
        )
    expected_bundle = {
        key: candidate_bundle[key]
        for key in ("manifest", "status", "frozen_inputs")
    }
    if selection.get("candidate_bundle") != expected_bundle:
        raise TestWinnerContractError(
            "long Base selection does not bind the complete training bundle"
        )
    rows = selection.get("candidate_metrics")
    if not isinstance(rows, list) or len(rows) != len(
        long_contract.EXPECTED_CANDIDATE_EPOCHS
    ):
        raise TestWinnerContractError(
            "long Base selection does not cover exactly 22 candidates"
        )
    validated: list[tuple[float, int, Mapping[str, Any]]] = []
    for expected_epoch, row in zip(
        long_contract.EXPECTED_CANDIDATE_EPOCHS,
        rows,
    ):
        if not isinstance(row, dict):
            raise TestWinnerContractError(
                f"long Base selection row e{expected_epoch} is invalid"
            )
        epoch = legacy.require_exact_int(
            row.get("epoch"),
            f"long Base selection row e{expected_epoch}",
        )
        metrics = row.get("metrics")
        fgd = metrics.get("fgd") if isinstance(metrics, dict) else None
        if (
            epoch != expected_epoch
            or epoch == 30
            or set(metrics or {}) != {"fgd"}
            or isinstance(fgd, bool)
            or type(fgd) not in {int, float}
            or not math.isfinite(float(fgd))
            or float(fgd) < 0.0
            or row.get("candidate_checkpoint")
            != candidate_bundle["candidates"][epoch]
        ):
            raise TestWinnerContractError(
                f"long Base selection row e{expected_epoch} is unbound"
            )
        for label in ("inference_lineage", "diffsheg_report"):
            _artifact(row.get(label), f"e{epoch} {label}")
        validated.append((float(fgd), epoch, row))
    winner_fgd, winner_epoch, winner_row = min(
        validated,
        key=lambda item: (item[0], item[1]),
    )
    selected = selection.get("selected")
    expected_selected = {
        "epoch": winner_epoch,
        "candidate_checkpoint": winner_row["candidate_checkpoint"],
        "fgd": winner_fgd,
        "inference_lineage": winner_row["inference_lineage"],
        "diffsheg_report": winner_row["diffsheg_report"],
    }
    if selected != expected_selected:
        raise TestWinnerContractError(
            "declared Base winner differs from recomputed min(FGD, epoch)"
        )
    checkpoint = legacy._regular_file(
        checkpoint_path,
        "requested long Base test checkpoint",
    )
    checkpoint_sha = legacy.require_sha256(
        expected_checkpoint_sha256,
        "requested long Base test checkpoint SHA-256",
    )
    expected_candidate = candidate_bundle["candidates"][winner_epoch]
    if (
        legacy.sha256_file(checkpoint) != checkpoint_sha
        or str(checkpoint) != expected_candidate["path"]
        or checkpoint_sha != expected_candidate["sha256"]
        or checkpoint.stat().st_size != expected_candidate["bytes"]
    ):
        raise TestWinnerContractError(
            "requested test checkpoint is not the unique validation winner"
        )
    return {
        "format": AUTHORIZATION_FORMAT,
        "status": "validated",
        "selection": {
            "path": str(selection_resolved),
            "sha256": selection_sha,
            "receipt_payload_sha256": claimed_payload_sha,
        },
        "selected_epoch": winner_epoch,
        "selected_fgd": winner_fgd,
        "selected_checkpoint": dict(expected_candidate),
        "candidate_bundle": expected_bundle,
        "candidate_count": len(long_contract.EXPECTED_CANDIDATE_EPOCHS),
        "selection_split": "val",
        "test_visible_during_selection": False,
        "authorized_test_evaluations": 0,
        "one_shot_claim_required": True,
        "test_feedback_into_selection": False,
    }


def _claim_path(selection_path: Path) -> Path:
    return selection_path.with_name(
        f"{selection_path.name}.test-winner-claim.json"
    )


def publish_test_winner_claim(
    validation: Mapping[str, Any],
    *,
    selection_path: Path,
) -> dict[str, Any]:
    """Atomically consume the validation selection's sole test allowance."""

    if set(validation) != {
        "format",
        "status",
        "selection",
        "selected_epoch",
        "selected_fgd",
        "selected_checkpoint",
        "candidate_bundle",
        "candidate_count",
        "selection_split",
        "test_visible_during_selection",
        "authorized_test_evaluations",
        "one_shot_claim_required",
        "test_feedback_into_selection",
    } or (
        validation.get("format") != AUTHORIZATION_FORMAT
        or validation.get("status") != "validated"
        or validation.get("authorized_test_evaluations") != 0
        or validation.get("one_shot_claim_required") is not True
        or validation.get("test_feedback_into_selection") is not False
    ):
        raise TestWinnerContractError(
            "test-winner validation evidence is not claimable"
        )
    selection_resolved = selection_path.resolve(strict=True)
    expected_selection = validation.get("selection")
    if (
        not isinstance(expected_selection, dict)
        or expected_selection.get("path") != str(selection_resolved)
        or legacy.sha256_file(selection_resolved)
        != expected_selection.get("sha256")
    ):
        raise TestWinnerContractError(
            "test-winner selection changed before one-shot claim"
        )
    claim_path = _claim_path(selection_resolved)
    body = {
        **dict(validation),
        "status": "authorized",
        "authorized_test_evaluations": 1,
    }
    body["receipt_payload_sha256"] = legacy.canonical_json_sha256(body)
    encoded = (
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as error:
        raise TestWinnerContractError(
            "the selected Base winner's one-shot test allowance is already "
            "claimed"
        ) from error
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing test claim")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        claim_path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    directory = os.open(claim_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "claim": {
            "path": str(claim_path),
            "sha256": legacy.sha256_file(claim_path),
            "receipt_payload_sha256": body["receipt_payload_sha256"],
        },
        **body,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authorize only the long Base validation winner for test",
        allow_abbrev=False,
    )
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = long_contract.validate_candidate_bundle(
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
    validation = validate_test_winner(
        selection_path=args.selection_json,
        expected_selection_sha256=args.expected_selection_sha256,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        candidate_bundle=bundle,
    )
    authorization = publish_test_winner_claim(
        validation,
        selection_path=args.selection_json,
    )
    print(json.dumps(authorization, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
