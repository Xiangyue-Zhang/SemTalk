#!/usr/bin/env python3
"""Decide whether SHOW prerequisite training needs another boundary.

The decision is deliberately downstream of the strict five-prerequisite
validation replay.  It never looks at training loss or test data.  A stage
requests continuation only when the newest validation boundary is its
freshly replayed winner and improves on the best of the preceding two recent
boundaries by at least the fixed relative threshold below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.show_base import prerequisite_val_contract as contract
from scripts.show_base import selected_prerequisites as selected_contract


FORMAT = "semtalk_show_prerequisite_continuation_decision_v2"
RECENT_CANDIDATES = 3
MIN_RELATIVE_IMPROVEMENT = 0.005
SCORE_DIRECTION = "lower_is_better"
INTERVAL_EPOCHS = 20
STAGE_CAP_EPOCHS = {
    "face": 600,
    "hands": 500,
    "upper": 500,
    "lower": 600,
    "global": 1700,
}
DECISION_KEYS = {
    "format",
    "status",
    "decision",
    "test_visible",
    "protocol",
    "inputs",
    "stages",
    "receipt_payload_sha256",
}
PROTOCOL_KEYS = {
    "name",
    "score_direction",
    "recent_candidates",
    "relative_improvement_reference",
    "minimum_relative_improvement",
    "interval_epochs",
    "stage_cap_epochs",
    "continue_rule",
    "terminal_rule",
}
INPUT_KEYS = {"selection", "measurement_index", "stage_measurements"}
BINDING_KEYS = {"path", "sha256", "receipt_payload_sha256"}
STAGE_DECISION_KEYS = {
    "stage",
    "recent_candidate_epochs",
    "recent_selection_scores",
    "winner_epoch",
    "latest_epoch",
    "previous_best_score",
    "latest_score",
    "relative_improvement",
    "latest_is_winner",
    "meets_relative_improvement_threshold",
    "cap_epoch",
    "action",
    "target_epoch",
    "frozen_winner_epoch",
    "requests_continuation",
}


class ContinuationDecisionError(RuntimeError):
    """Raised when continuation evidence is incomplete or mutable."""


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuationDecisionError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContinuationDecisionError(f"{label} must be an exact integer")
    return value


def _require_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ContinuationDecisionError(f"{label} must be a JSON number")
    score = float(value)
    if not math.isfinite(score) or score < 0.0:
        raise ContinuationDecisionError(
            f"{label} must be finite and nonnegative"
        )
    return score


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ContinuationDecisionError(f"{label} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ContinuationDecisionError(f"{label} must be finite")
    return number


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ContinuationDecisionError(f"{label} must be a boolean")
    return value


def _exact_keys(
    value: Any,
    keys: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContinuationDecisionError(f"{label} schema mismatch")
    return value


def _decision_protocol() -> dict[str, Any]:
    return {
        "name": "fresh_replayed_independent_stage_val_improvement_v2",
        "score_direction": SCORE_DIRECTION,
        "recent_candidates": RECENT_CANDIDATES,
        "relative_improvement_reference": (
            "best_of_preceding_two_recent_candidates"
        ),
        "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
        "interval_epochs": INTERVAL_EPOCHS,
        "stage_cap_epochs": dict(STAGE_CAP_EPOCHS),
        "continue_rule": (
            "each_stage_latest_boundary_is_global_val_winner_and_relative_"
            "improvement_gte_threshold_and_below_stage_cap"
        ),
        "terminal_rule": (
            "otherwise_freeze_global_val_winner;at_cap_mark_capped;"
            "terminal_stages_never_reenter"
        ),
    }


def _validate_binding_schema(value: Any, label: str) -> dict[str, Any]:
    binding = _exact_keys(value, BINDING_KEYS, label)
    path = binding["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ContinuationDecisionError(f"{label} path must be absolute")
    _require_sha256(binding["sha256"], f"{label} SHA-256")
    _require_sha256(
        binding["receipt_payload_sha256"],
        f"{label} payload SHA-256",
    )
    return binding


def _validate_decision_schema(value: Any) -> dict[str, Any]:
    receipt = _exact_keys(value, DECISION_KEYS, "continuation decision")
    if (
        receipt["format"] != FORMAT
        or receipt["status"] != "complete"
        or not isinstance(receipt["decision"], str)
        or receipt["decision"] not in ("continue", "stop")
        or receipt["test_visible"] is not False
    ):
        raise ContinuationDecisionError(
            "continuation decision protocol mismatch"
        )
    protocol = _exact_keys(
        receipt["protocol"],
        PROTOCOL_KEYS,
        "continuation decision protocol",
    )
    if protocol != _decision_protocol():
        raise ContinuationDecisionError(
            "continuation decision rule changed"
        )
    inputs = _exact_keys(
        receipt["inputs"],
        INPUT_KEYS,
        "continuation decision inputs",
    )
    selection = _validate_binding_schema(
        inputs["selection"],
        "continuation decision selection",
    )
    _validate_binding_schema(
        inputs["measurement_index"],
        "continuation decision measurement index",
    )
    stage_measurements = _exact_keys(
        inputs["stage_measurements"],
        set(selected_contract.STAGES),
        "continuation decision stage measurements",
    )
    for stage in selected_contract.STAGES:
        _validate_binding_schema(
            stage_measurements[stage],
            f"continuation decision {stage} measurement",
        )

    stages = receipt["stages"]
    if not isinstance(stages, list) or len(stages) != len(
        selected_contract.STAGES
    ):
        raise ContinuationDecisionError(
            "continuation decision stages schema mismatch"
        )
    for expected_stage, stage_value in zip(
        selected_contract.STAGES,
        stages,
    ):
        stage = _exact_keys(
            stage_value,
            STAGE_DECISION_KEYS,
            f"continuation decision {expected_stage} stage",
        )
        if stage["stage"] != expected_stage:
            raise ContinuationDecisionError(
                "continuation decision stage order changed"
            )
        recent_epochs = stage["recent_candidate_epochs"]
        recent_scores = stage["recent_selection_scores"]
        if (
            not isinstance(recent_epochs, list)
            or len(recent_epochs) != RECENT_CANDIDATES
            or not isinstance(recent_scores, list)
            or len(recent_scores) != RECENT_CANDIDATES
        ):
            raise ContinuationDecisionError(
                f"continuation decision {expected_stage} recent schema "
                "mismatch"
            )
        for index, epoch in enumerate(recent_epochs):
            _require_int(
                epoch,
                f"continuation decision {expected_stage} recent epoch "
                f"{index}",
            )
        for index, score in enumerate(recent_scores):
            _require_score(
                score,
                f"continuation decision {expected_stage} recent score "
                f"{index}",
            )
        _require_int(
            stage["winner_epoch"],
            f"continuation decision {expected_stage} winner epoch",
        )
        _require_int(
            stage["latest_epoch"],
            f"continuation decision {expected_stage} latest epoch",
        )
        cap_epoch = _require_int(
            stage["cap_epoch"],
            f"continuation decision {expected_stage} cap epoch",
        )
        if cap_epoch != STAGE_CAP_EPOCHS[expected_stage]:
            raise ContinuationDecisionError(
                f"continuation decision {expected_stage} cap changed"
            )
        _require_score(
            stage["previous_best_score"],
            f"continuation decision {expected_stage} previous best score",
        )
        _require_score(
            stage["latest_score"],
            f"continuation decision {expected_stage} latest score",
        )
        _require_finite_number(
            stage["relative_improvement"],
            f"continuation decision {expected_stage} relative improvement",
        )
        for key in (
            "latest_is_winner",
            "meets_relative_improvement_threshold",
            "requests_continuation",
        ):
            _require_bool(
                stage[key],
                f"continuation decision {expected_stage} {key}",
            )
        action = stage["action"]
        if action not in {"continue", "freeze", "capped"}:
            raise ContinuationDecisionError(
                f"continuation decision {expected_stage} action changed"
            )
        target_epoch = stage["target_epoch"]
        if target_epoch is not None:
            target_epoch = _require_int(
                target_epoch,
                f"continuation decision {expected_stage} target epoch",
            )
        frozen_winner_epoch = _require_int(
            stage["frozen_winner_epoch"],
            (
                f"continuation decision {expected_stage} frozen winner "
                "epoch"
            ),
        )
        latest_epoch = stage["latest_epoch"]
        expected_request = action == "continue"
        if (
            stage["requests_continuation"] is not expected_request
            or frozen_winner_epoch != stage["winner_epoch"]
            or latest_epoch > cap_epoch
            or (
                action == "continue"
                and (
                    target_epoch != latest_epoch + INTERVAL_EPOCHS
                    or target_epoch > cap_epoch
                    or not stage["latest_is_winner"]
                    or not stage[
                        "meets_relative_improvement_threshold"
                    ]
                )
            )
            or (action != "continue" and target_epoch is not None)
            or (action == "capped" and latest_epoch != cap_epoch)
            or (action == "freeze" and latest_epoch >= cap_epoch)
        ):
            raise ContinuationDecisionError(
                f"continuation decision {expected_stage} action mismatch"
            )
    _require_sha256(
        receipt["receipt_payload_sha256"],
        "continuation decision payload SHA-256",
    )
    return selection


def _load_receipt(
    binding_value: Any,
    *,
    binding_keys: set[str],
    label: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    binding = _exact_keys(binding_value, binding_keys, f"{label} binding")
    path_value = binding["path"]
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ContinuationDecisionError(f"{label} path must be absolute")
    try:
        path, payload = selected_contract._safe_file_snapshot(
            path_value,
            label,
            val_only=False,
        )
    except selected_contract.SelectedPrerequisiteError as error:
        raise ContinuationDecisionError(str(error)) from error
    observed_sha = hashlib.sha256(payload).hexdigest()
    expected_sha = _require_sha256(binding["sha256"], f"{label} SHA-256")
    if observed_sha != expected_sha:
        raise ContinuationDecisionError(f"{label} SHA-256 mismatch")
    try:
        value = selected_contract.strict_json_bytes(payload, str(path))
    except selected_contract.SelectedPrerequisiteError as error:
        raise ContinuationDecisionError(str(error)) from error
    claimed_payload_sha = _require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256", None)
    if (
        selected_contract.canonical_json_sha256(unsigned)
        != claimed_payload_sha
    ):
        raise ContinuationDecisionError(f"{label} payload SHA-256 mismatch")
    if "receipt_payload_sha256" in binding_keys:
        bound_payload_sha = _require_sha256(
            binding["receipt_payload_sha256"],
            f"{label} bound payload SHA-256",
        )
        if claimed_payload_sha != bound_payload_sha:
            raise ContinuationDecisionError(
                f"{label} bound payload SHA-256 mismatch"
            )
    return path, value, dict(binding)


def _stage_decision(
    *,
    stage: str,
    candidates_value: Any,
    selected_stage: Mapping[str, Any],
    expected_candidate_epochs: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(candidates_value, list):
        raise ContinuationDecisionError(
            f"{stage} candidates must be a list"
        )
    if len(candidates_value) < RECENT_CANDIDATES:
        raise ContinuationDecisionError(
            f"{stage} requires at least {RECENT_CANDIDATES} candidates"
        )
    candidates: list[dict[str, Any]] = []
    epochs: list[int] = []
    for index, raw_candidate in enumerate(candidates_value):
        if not isinstance(raw_candidate, dict):
            raise ContinuationDecisionError(
                f"{stage} candidate {index} must be an object"
            )
        epoch = _require_int(
            raw_candidate.get("epoch"),
            f"{stage} candidate {index} epoch",
        )
        updates = _require_int(
            raw_candidate.get("optimizer_updates"),
            f"{stage} candidate {index} optimizer updates",
        )
        score = _require_score(
            raw_candidate.get("selection_score"),
            f"{stage} candidate {index} selection score",
        )
        checkpoint = raw_candidate.get("candidate_checkpoint")
        if not isinstance(checkpoint, dict):
            raise ContinuationDecisionError(
                f"{stage} candidate {index} checkpoint is missing"
            )
        checkpoint_sha = _require_sha256(
            checkpoint.get("sha256"),
            f"{stage} candidate {index} checkpoint SHA-256",
        )
        candidates.append(
            {
                "epoch": epoch,
                "optimizer_updates": updates,
                "selection_score": score,
                "checkpoint_sha256": checkpoint_sha,
            }
        )
        epochs.append(epoch)
    expected_epochs = [
        _require_int(epoch, "expected candidate epoch")
        for epoch in expected_candidate_epochs
    ]
    if (
        epochs != expected_epochs
        or len(set(epochs)) != len(epochs)
        or epochs != sorted(epochs)
    ):
        raise ContinuationDecisionError(
            f"{stage} candidate boundary inventory changed"
        )

    winner = min(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["epoch"],
            candidate["optimizer_updates"],
            candidate["checkpoint_sha256"],
        ),
    )
    winner_epoch = _require_int(
        selected_stage.get("epoch"),
        f"{stage} replayed winner epoch",
    )
    if (
        winner_epoch != winner["epoch"]
        or _require_score(
            selected_stage.get("selection_score"),
            f"{stage} replayed winner score",
        )
        != winner["selection_score"]
    ):
        raise ContinuationDecisionError(
            f"{stage} replayed winner differs from measurements"
        )

    recent = candidates[-RECENT_CANDIDATES:]
    latest = recent[-1]
    previous_best = min(
        candidate["selection_score"] for candidate in recent[:-1]
    )
    if previous_best == 0.0:
        relative_improvement = 0.0
    else:
        relative_improvement = (
            previous_best - latest["selection_score"]
        ) / previous_best
    if not math.isfinite(relative_improvement):
        raise ContinuationDecisionError(
            f"{stage} relative improvement is non-finite"
        )
    latest_is_winner = latest["epoch"] == winner_epoch
    meets_threshold = (
        relative_improvement >= MIN_RELATIVE_IMPROVEMENT
    )
    cap_epoch = STAGE_CAP_EPOCHS[stage]
    latest_epoch = latest["epoch"]
    if latest_epoch > cap_epoch:
        raise ContinuationDecisionError(
            f"{stage} latest boundary exceeds the formal cap"
        )
    if latest_epoch == cap_epoch:
        action = "capped"
        target_epoch = None
    elif latest_is_winner and meets_threshold:
        action = "continue"
        target_epoch = latest_epoch + INTERVAL_EPOCHS
        if target_epoch > cap_epoch:
            raise ContinuationDecisionError(
                f"{stage} continuation target exceeds the formal cap"
            )
    else:
        action = "freeze"
        target_epoch = None
    requests_continuation = action == "continue"
    return {
        "stage": stage,
        "recent_candidate_epochs": [
            candidate["epoch"] for candidate in recent
        ],
        "recent_selection_scores": [
            candidate["selection_score"] for candidate in recent
        ],
        "winner_epoch": winner_epoch,
        "latest_epoch": latest["epoch"],
        "previous_best_score": previous_best,
        "latest_score": latest["selection_score"],
        "relative_improvement": relative_improvement,
        "latest_is_winner": latest_is_winner,
        "meets_relative_improvement_threshold": meets_threshold,
        "cap_epoch": cap_epoch,
        "action": action,
        "target_epoch": target_epoch,
        "frozen_winner_epoch": winner_epoch,
        "requests_continuation": requests_continuation,
    }


def decide(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    output_json: Path | None,
) -> dict[str, Any]:
    """Fresh-replay evidence and optionally publish one immutable decision."""

    try:
        bridge = selected_contract.load_selected_prerequisites(
            selection_path,
            expected_selection_sha256,
        )
    except selected_contract.SelectedPrerequisiteError as error:
        raise ContinuationDecisionError(
            f"fresh prerequisite replay failed: {error}"
        ) from error
    if (
        bridge.get("format")
        != "semtalk_show_selected_prerequisite_bridge_v1"
        or bridge.get("test_visible") is not False
        or bridge.get("global_verified_not_consumed") is not True
    ):
        raise ContinuationDecisionError(
            "fresh prerequisite replay bridge is incomplete"
        )
    selection_binding = _exact_keys(
        bridge.get("selection"),
        {"path", "sha256", "receipt_payload_sha256"},
        "selection input",
    )
    if (
        selection_binding["sha256"]
        != _require_sha256(
            expected_selection_sha256,
            "selection expected SHA-256",
        )
        or Path(selection_binding["path"]).resolve(strict=True)
        != Path(selection_path).resolve(strict=True)
    ):
        raise ContinuationDecisionError(
            "fresh replay selection binding changed"
        )

    measurement_path, measurement, measurement_binding = _load_receipt(
        bridge.get("measurement_index_receipt"),
        binding_keys={"path", "sha256", "receipt_payload_sha256"},
        label="measurement index",
    )
    if measurement.get("test_visible") is not False:
        raise ContinuationDecisionError("measurement index exposes test data")
    protocol = measurement.get("protocol")
    if not isinstance(protocol, dict):
        raise ContinuationDecisionError(
            "measurement index protocol is missing"
        )
    common_candidate_epochs = protocol.get("candidate_epochs")
    candidate_epochs_by_stage = protocol.get(
        "candidate_epochs_by_stage"
    )
    if common_candidate_epochs is not None and not isinstance(
        common_candidate_epochs, list
    ):
        raise ContinuationDecisionError(
            "measurement candidate epoch inventory is invalid"
        )
    if candidate_epochs_by_stage is not None and (
        not isinstance(candidate_epochs_by_stage, dict)
        or set(candidate_epochs_by_stage) != set(selected_contract.STAGES)
    ):
        raise ContinuationDecisionError(
            "measurement per-stage candidate inventory is invalid"
        )
    if common_candidate_epochs is None and candidate_epochs_by_stage is None:
        raise ContinuationDecisionError(
            "measurement candidate epoch inventory is missing"
        )
    stages_value = measurement.get("stages")
    selected_stages = bridge.get("selected")
    if (
        not isinstance(stages_value, dict)
        or not isinstance(selected_stages, dict)
        or set(stages_value) != set(selected_stages)
        or set(selected_stages) != set(selected_contract.STAGES)
    ):
        raise ContinuationDecisionError(
            "measurement/selection stage coverage mismatch"
        )

    stage_measurement_bindings: dict[str, Any] = {}
    stage_decisions = []
    for stage in selected_contract.STAGES:
        stage_index_binding = _exact_keys(
            stages_value[stage],
            {
                "stage",
                "path",
                "sha256",
                "receipt_payload_sha256",
            },
            f"{stage} measurement index entry",
        )
        if stage_index_binding["stage"] != stage:
            raise ContinuationDecisionError(
                f"{stage} measurement index entry changed"
            )
        _, stage_measurement, stage_binding = _load_receipt(
            {
                key: stage_index_binding[key]
                for key in ("path", "sha256", "receipt_payload_sha256")
            },
            binding_keys={"path", "sha256", "receipt_payload_sha256"},
            label=f"{stage} measurement",
        )
        replayed_measurement = selected_stages[stage].get(
            "measurement_receipt"
        )
        if stage_binding != replayed_measurement:
            raise ContinuationDecisionError(
                f"{stage} measurement binding differs from fresh replay"
            )
        stage_protocol = stage_measurement.get("protocol")
        if not isinstance(stage_protocol, dict):
            raise ContinuationDecisionError(
                f"{stage} measurement protocol is missing"
            )
        stage_candidate_epochs = stage_protocol.get("candidate_epochs")
        indexed_candidate_epochs = (
            candidate_epochs_by_stage[stage]
            if candidate_epochs_by_stage is not None
            else common_candidate_epochs
        )
        if (
            stage_measurement.get("stage") != stage
            or stage_measurement.get("split") != "val"
            or stage_measurement.get("test_visible") is not False
            or not isinstance(stage_candidate_epochs, list)
            or stage_candidate_epochs != indexed_candidate_epochs
        ):
            raise ContinuationDecisionError(
                f"{stage} measurement protocol changed"
            )
        stage_measurement_bindings[stage] = stage_binding
        stage_decisions.append(
            _stage_decision(
                stage=stage,
                candidates_value=stage_measurement.get("candidates"),
                selected_stage=selected_stages[stage],
                expected_candidate_epochs=stage_candidate_epochs,
            )
        )

    decision = (
        "continue"
        if any(
            stage["requests_continuation"]
            for stage in stage_decisions
        )
        else "stop"
    )
    result = {
        "format": FORMAT,
        "status": "complete",
        "decision": decision,
        "test_visible": False,
        "protocol": _decision_protocol(),
        "inputs": {
            "selection": dict(selection_binding),
            "measurement_index": {
                **measurement_binding,
                "path": str(measurement_path),
            },
            "stage_measurements": stage_measurement_bindings,
        },
        "stages": stage_decisions,
    }
    result["receipt_payload_sha256"] = (
        selected_contract.canonical_json_sha256(result)
    )
    if output_json is not None:
        try:
            contract.atomic_json_new(output_json, result)
        except (contract.ContractError, OSError, ValueError) as error:
            raise ContinuationDecisionError(
                f"cannot publish continuation decision: {error}"
            ) from error
    return result


def replay_decision(
    decision_path: Path,
    expected_decision_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless an external decision exactly matches fresh replay."""

    expected_sha = _require_sha256(
        expected_decision_sha256,
        "continuation decision expected SHA-256",
    )
    try:
        path, payload = selected_contract._safe_file_snapshot(
            decision_path,
            "continuation decision",
            val_only=False,
        )
    except (
        selected_contract.SelectedPrerequisiteError,
        OSError,
    ) as error:
        raise ContinuationDecisionError(str(error)) from error
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ContinuationDecisionError(
            "continuation decision file SHA-256 mismatch"
        )
    try:
        receipt = selected_contract.strict_json_bytes(payload, str(path))
    except selected_contract.SelectedPrerequisiteError as error:
        raise ContinuationDecisionError(str(error)) from error
    selection = _validate_decision_schema(receipt)
    claimed_payload_sha = receipt["receipt_payload_sha256"]
    unsigned = dict(receipt)
    unsigned.pop("receipt_payload_sha256")
    try:
        observed_payload_sha = selected_contract.canonical_json_sha256(
            unsigned
        )
    except selected_contract.SelectedPrerequisiteError as error:
        raise ContinuationDecisionError(str(error)) from error
    if observed_payload_sha != claimed_payload_sha:
        raise ContinuationDecisionError(
            "continuation decision payload SHA-256 mismatch"
        )
    recomputed = decide(
        selection_path=Path(selection["path"]),
        expected_selection_sha256=selection["sha256"],
        output_json=None,
    )
    try:
        exact_match = (
            selected_contract.canonical_json_bytes(receipt)
            == selected_contract.canonical_json_bytes(recomputed)
        )
    except selected_contract.SelectedPrerequisiteError as error:
        raise ContinuationDecisionError(str(error)) from error
    if not exact_match:
        raise ContinuationDecisionError(
            "continuation decision differs from fresh replay"
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide whether SHOW prerequisite training continues",
        allow_abbrev=False,
    )
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = decide(
        selection_path=args.selection_json,
        expected_selection_sha256=args.expected_selection_sha256,
        output_json=args.output_json,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "stage_actions": {
                    stage["stage"]: {
                        "action": stage["action"],
                        "winner_epoch": stage["winner_epoch"],
                        "latest_epoch": stage["latest_epoch"],
                        "target_epoch": stage["target_epoch"],
                        "cap_epoch": stage["cap_epoch"],
                    }
                    for stage in result["stages"]
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
