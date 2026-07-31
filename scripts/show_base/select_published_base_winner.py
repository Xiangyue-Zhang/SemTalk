#!/usr/bin/env python3
"""Select the one SemTalk SHOW Base winner from frozen TalkSHOW reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.show_base import published_test_winner_claim as authority


CANDIDATE_EVIDENCE_FORMAT = (
    "semtalk_show_base_talkshow_candidate_evidence_v1"
)
FRESH_CANDIDATE_EVIDENCE_FORMAT = (
    "semtalk_show_base_fresh_val_candidate_evidence_v2"
)


class PublishedBaseSelectionError(RuntimeError):
    """Raised when Base candidate evidence cannot produce a formal winner."""


def _fail_closed(label: str, operation):
    try:
        return operation()
    except PublishedBaseSelectionError:
        raise
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise PublishedBaseSelectionError(f"{label}: {error}") from error


def _validate_candidate_rows(
    rows_value: Any,
    *,
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
    expected_real_feature_cache: Mapping[str, Any],
    require_fresh_transaction: bool = False,
    expected_updates_per_epoch: int | None = None,
) -> list[dict[str, Any]]:
    if expected_updates_per_epoch is None:
        expected_updates_per_epoch = authority.BASE_UPDATES_PER_EPOCH
    if expected_updates_per_epoch not in {248, 1988}:
        raise PublishedBaseSelectionError(
            "selected Base topology updates per epoch changed"
        )
    if not isinstance(rows_value, list) or len(rows_value) != len(
        authority.BASE_CANDIDATE_EPOCHS
    ):
        raise PublishedBaseSelectionError(
            "candidate evidence must cover exactly 22 boundaries"
        )
    common_input_keys = {
        "epoch",
        "optimizer_updates",
        "candidate_checkpoint",
        "prediction_manifest",
        "inference_lineage",
        "distribution_receipt",
    }
    selected_rows: list[dict[str, Any]] = []
    checkpoint_paths: set[str] = set()
    evidence_paths: set[str] = set()
    for expected_epoch, raw_row in zip(
        authority.BASE_CANDIDATE_EPOCHS, rows_value
    ):
        input_keys = set(common_input_keys)
        has_transaction = isinstance(raw_row, dict) and (
            "candidate_transaction" in raw_row
        )
        if require_fresh_transaction and not has_transaction:
            raise PublishedBaseSelectionError(
                f"Base e{expected_epoch} fresh candidate transaction is mandatory"
            )
        if has_transaction:
            input_keys.add("candidate_transaction")
        if require_fresh_transaction:
            input_keys.update(
                {"primary_screen_receipt", "primary_replay_receipt"}
            )
        else:
            input_keys.update(
                {"talkshow_metric_report", "primary_replay_receipt"}
            )
        row = authority._exact_mapping(
            raw_row,
            input_keys,
            f"Base e{expected_epoch} candidate evidence",
        )
        epoch = authority._require_int(
            row["epoch"], f"Base e{expected_epoch} epoch", minimum=1
        )
        updates = authority._require_int(
            row["optimizer_updates"],
            f"Base e{expected_epoch} optimizer updates",
            minimum=1,
        )
        if (
            epoch != expected_epoch
            or updates != epoch * expected_updates_per_epoch
        ):
            raise PublishedBaseSelectionError(
                f"Base e{expected_epoch} boundary changed"
            )
        checkpoint = authority._validate_checkpoint(
            row["candidate_checkpoint"], f"Base e{epoch} checkpoint"
        )
        if checkpoint["path"] in checkpoint_paths:
            raise PublishedBaseSelectionError(
                "candidate checkpoint path was reused"
            )
        checkpoint_paths.add(checkpoint["path"])
        prediction_manifest, _manifest_payload = authority._normalize_artifact(
            row["prediction_manifest"],
            f"Base e{epoch} prediction manifest",
            with_payload=False,
        )
        distribution_artifact, distribution = authority._validate_distribution(
            row["distribution_receipt"],
            prediction_manifest=prediction_manifest,
        )
        transaction_artifact: dict[str, Any] | None = None
        expected_canonical_manifest: dict[str, Any] | None = None
        if has_transaction:
            (
                transaction_artifact,
                _transaction,
                lineage_artifact,
                expected_canonical_manifest,
            ) = authority._validate_candidate_transaction(
                row["candidate_transaction"],
                epoch=epoch,
                updates=updates,
                checkpoint=checkpoint,
                prediction_manifest=prediction_manifest,
                distribution_artifact=distribution_artifact,
                distribution=distribution,
                prerequisite_artifact=prerequisite_artifact,
                continuation_artifact=continuation_artifact,
            )
            if lineage_artifact != row["inference_lineage"]:
                raise PublishedBaseSelectionError(
                    f"Base e{epoch} transaction lineage differs from row"
                )
        else:
            lineage_artifact, _lineage = authority._validate_lineage(
                row["inference_lineage"],
                epoch=epoch,
                updates=updates,
                checkpoint=checkpoint,
                prediction_manifest=prediction_manifest,
                distribution_artifact=distribution_artifact,
                prerequisite_artifact=prerequisite_artifact,
                continuation_artifact=continuation_artifact,
            )
        normalized_input = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "candidate_checkpoint": checkpoint,
            "prediction_manifest": prediction_manifest,
            "inference_lineage": lineage_artifact,
            "distribution_receipt": distribution_artifact,
        }
        if transaction_artifact is not None:
            normalized_input["candidate_transaction"] = transaction_artifact
        if require_fresh_transaction:
            if expected_canonical_manifest is None:
                raise PublishedBaseSelectionError(
                    f"Base e{epoch} fresh canonical authority is missing"
                )
            screen_artifact, screen_fgd, screen_validation = (
                authority._validate_primary_screen_receipt(
                    row["primary_screen_receipt"],
                    prediction_manifest=prediction_manifest,
                    distribution_artifact=distribution_artifact,
                    expected_real_feature_cache=expected_real_feature_cache,
                    expected_canonical_manifest=expected_canonical_manifest,
                )
            )
            replay_artifact, fgd = (
                authority._validate_primary_screen_replay_receipt(
                    row["primary_replay_receipt"],
                    screen_artifact=screen_artifact,
                    screen_validation=screen_validation,
                    prediction_manifest=prediction_manifest,
                    distribution_artifact=distribution_artifact,
                )
            )
            if fgd != screen_fgd:
                raise PublishedBaseSelectionError(
                    f"Base e{epoch} raw replay differs from screen FGD"
                )
            normalized_input["primary_screen_receipt"] = screen_artifact
            normalized_input["primary_replay_receipt"] = replay_artifact
        else:
            report_artifact, report, report_validation = (
                authority._validate_metric_report(
                    row["talkshow_metric_report"],
                    prediction_manifest=prediction_manifest,
                    lineage_artifact=lineage_artifact,
                    distribution=distribution,
                    expected_canonical_manifest=expected_canonical_manifest,
                )
            )
            replay_artifact, fgd = authority._validate_primary_replay_receipt(
                row["primary_replay_receipt"],
                report=report,
                report_validation=report_validation,
                prediction_manifest=prediction_manifest,
                distribution=distribution,
                expected_real_feature_cache=expected_real_feature_cache,
                expected_canonical_manifest=expected_canonical_manifest,
            )
            normalized_input["talkshow_metric_report"] = report_artifact
            normalized_input["primary_replay_receipt"] = replay_artifact
        if normalized_input != row:
            raise PublishedBaseSelectionError(
                f"Base e{epoch} evidence row is not canonical"
            )
        evidence_roles = [
            "prediction_manifest",
            "inference_lineage",
            "distribution_receipt",
        ]
        evidence_roles.extend(
            ["primary_screen_receipt", "primary_replay_receipt"]
            if require_fresh_transaction
            else ["talkshow_metric_report", "primary_replay_receipt"]
        )
        if has_transaction:
            evidence_roles.insert(0, "candidate_transaction")
        for role in evidence_roles:
            path = normalized_input[role]["path"]
            if path in evidence_paths:
                raise PublishedBaseSelectionError(
                    f"candidate {role} path was reused"
                )
            evidence_paths.add(path)
        selected_rows.append(
            {
                **normalized_input,
                "body_released2_fgd": fgd,
            }
        )
    return selected_rows


def build_published_base_winner_selection(
    candidate_evidence: Mapping[str, Any],
    *,
    prerequisite_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Fresh-replay all 22 val reports and return the canonical winner.

    The fresh evidence artifact contains no full TalkSHOW report.  Ranking FGD
    is freshly recomputed from the byte-pinned primary-screen moments and the
    externally pinned real-feature cache.  paper16/face/SMPL-X metrics are
    deferred until after this immutable winner has been selected.
    """

    prerequisite_artifact, prerequisite_payload, _fixed = _fail_closed(
        "prerequisite selection replay failed",
        lambda: authority._validate_prerequisite_selection(
            prerequisite_selection
        ),
    )
    continuation_artifact, _decision = _fail_closed(
        "continuation replay failed",
        lambda: authority._validate_continuation_decision(
            continuation_decision,
            prerequisite_artifact=prerequisite_artifact,
            prerequisite_selection=prerequisite_payload,
        ),
    )
    evidence_artifact, evidence = _fail_closed(
        "candidate evidence replay failed",
        lambda: authority._verify_compact_receipt(
            candidate_evidence, "Base candidate evidence"
        ),
    )
    del evidence_artifact
    authority._exact_mapping(
        evidence,
        {
            "format",
            "payload_hash_algorithm",
            "status",
            "generator",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "candidate_epochs",
            "updates_per_epoch",
            "prerequisite_selection",
            "continuation_decision",
            "real_feature_cache",
            "candidates",
            "receipt_payload_sha256",
        },
        "Base candidate evidence",
    )
    fresh_protocol = evidence["format"] == FRESH_CANDIDATE_EVIDENCE_FORMAT
    updates_per_epoch = evidence.get("updates_per_epoch")
    if (
        evidence["format"]
        not in {CANDIDATE_EVIDENCE_FORMAT, FRESH_CANDIDATE_EVIDENCE_FORMAT}
        or evidence["payload_hash_algorithm"]
        != authority.PAYLOAD_HASH_ALGORITHM
        or evidence["status"] != "complete"
        or evidence["generator"] != "SemTalk Base Motion Generation"
        or evidence["dataset"] != "SHOW"
        or evidence["target_speaker_scope"] != authority.EXPECTED_SCOPE
        or evidence["split"] != "val"
        or evidence["test_visible"] is not False
        or evidence["candidate_epochs"]
        != list(authority.BASE_CANDIDATE_EPOCHS)
        or updates_per_epoch not in {248, 1988}
        or evidence["prerequisite_selection"] != prerequisite_artifact
        or evidence["continuation_decision"] != continuation_artifact
    ):
        raise PublishedBaseSelectionError(
            "Base candidate evidence identity changed"
        )
    real_feature_cache, _cache_payload = authority._normalize_artifact(
        evidence["real_feature_cache"],
        "Base released2 real-feature cache",
        with_payload=True,
    )
    authority._reject_forbidden_tree(evidence, "Base candidate evidence")
    rows = _fail_closed(
        "candidate row replay failed",
        lambda: _validate_candidate_rows(
            evidence["candidates"],
            prerequisite_artifact=prerequisite_artifact,
            continuation_artifact=continuation_artifact,
            expected_real_feature_cache=real_feature_cache,
            require_fresh_transaction=fresh_protocol,
            expected_updates_per_epoch=updates_per_epoch,
        ),
    )
    winner = min(
        rows,
        key=lambda row: (
            row["body_released2_fgd"],
            row["epoch"],
            row["optimizer_updates"],
        ),
    )
    result: dict[str, Any] = {
        "format": (
            authority.FRESH_WINNER_SELECTION_FORMAT
            if fresh_protocol
            else authority.WINNER_SELECTION_FORMAT
        ),
        "payload_hash_algorithm": authority.PAYLOAD_HASH_ALGORITHM,
        "status": "selected",
        "generator": "SemTalk Base Motion Generation",
        "dataset": "SHOW",
        "target_speaker_scope": authority.EXPECTED_SCOPE,
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "primary_metric": authority.PRIMARY_METRIC,
        "selection_policy": {
            "candidate_epochs": list(authority.BASE_CANDIDATE_EPOCHS),
            "updates_per_epoch": updates_per_epoch,
            "operator": "min",
            "ordering": [
                authority.PRIMARY_METRIC,
                "epoch",
                "optimizer_updates",
            ],
            "test_feedback_into_selection": False,
        },
        "prerequisite_selection": prerequisite_artifact,
        "continuation_decision": continuation_artifact,
        "real_feature_cache": real_feature_cache,
        "candidates": rows,
        "selected": winner,
        "test_policy": dict(authority.TEST_POLICY),
    }
    result["receipt_payload_sha256"] = authority.canonical_json_sha256(
        result
    )
    return result


def _atomic_write_new(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise PublishedBaseSelectionError("output path must be absolute")
    if path.exists() or path.is_symlink():
        raise PublishedBaseSelectionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _artifact_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-path", type=Path, required=True)
    parser.add_argument(f"--{prefix}-sha256", required=True)
    parser.add_argument(f"--{prefix}-bytes", type=int, required=True)
    parser.add_argument(f"--{prefix}-payload-sha256", required=True)


def _artifact_from_args(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    attribute = prefix.replace("-", "_")
    return {
        "path": str(getattr(args, f"{attribute}_path").resolve()),
        "sha256": getattr(args, f"{attribute}_sha256"),
        "bytes": getattr(args, f"{attribute}_bytes"),
        "receipt_payload_sha256": getattr(
            args, f"{attribute}_payload_sha256"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _artifact_args(parser, "candidate-evidence")
    _artifact_args(parser, "prerequisite-selection")
    _artifact_args(parser, "continuation-decision")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selection = build_published_base_winner_selection(
        _artifact_from_args(args, "candidate-evidence"),
        prerequisite_selection=_artifact_from_args(
            args, "prerequisite-selection"
        ),
        continuation_decision=_artifact_from_args(
            args, "continuation-decision"
        ),
    )
    _atomic_write_new(args.output.resolve(), selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
