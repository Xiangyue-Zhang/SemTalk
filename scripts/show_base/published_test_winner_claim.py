#!/usr/bin/env python3
"""Fresh validation for one published SemTalk SHOW Base test winner.

This module is intentionally self-contained and standard-library only.  It
does not trust caller-supplied checkpoint mappings.  The Base checkpoint is
recomputed from the frozen validation rows and the five representation
checkpoints are recovered from the separately pinned prerequisite selection.
Only then is the claim allowed to authorize one test evaluation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping


CLAIM_FORMAT = "semtalk_show_base_published_test_winner_claim_v1"
AUTHORIZATION_FORMAT = (
    "semtalk_show_base_published_test_winner_authorization_v1"
)
WINNER_SELECTION_FORMAT = (
    "semtalk_show_base_talkshow_released2_fgd_selection_v1"
)
PREREQUISITE_SELECTION_FORMAT = (
    "semtalk_show_prerequisite_val_selection_v1"
)
CONTINUATION_DECISION_FORMAT = (
    "semtalk_show_prerequisite_continuation_decision_v1"
)
VAL_LINEAGE_FORMAT = "semtalk_show_base_val_inference_lineage_v2"
METRIC_REPORT_FORMAT = "semtalk_show_talkshow_metrics_v1"
METRIC_REPORT_HASH_ALGORITHM = (
    "canonical_json_utf8_sorted_compact_newline_v1"
)
DISTRIBUTION_FORMAT = (
    "semtalk_show_deterministic_distribution_receipt_v2"
)
PAYLOAD_HASH_ALGORITHM = "canonical_json_utf8_sorted_compact_v1"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_SCOPE = "all_speakers_0_1_2_3"
EXPECTED_DATASET = {
    "name": "SHOW",
    "release": "ReleaseV1.0",
    "selection": ">2s",
    "speakers": {
        "oliver": 0,
        "chemistry": 1,
        "seth": 2,
        "conan": 3,
    },
}
STAGES = ("face", "hands", "upper", "lower", "global")
STAGE_SELECTION_METRICS = {
    "face": "face_geometry_expression_objective_v1",
    "hands": "hands_rotation_geometry_objective_v1",
    "upper": "upper_rotation_geometry_objective_v1",
    "lower": "lower_rotation_contact_objective_v1",
    "global": "global_root_contact_objective_v1",
}
PREREQUISITE_CANDIDATE_EPOCHS = tuple(range(20, 201, 20))
PREREQUISITE_UPDATES_PER_EPOCH = 497
BASE_CANDIDATE_EPOCHS = (
    1,
    2,
    4,
    8,
    16,
    32,
    40,
    50,
    60,
    70,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    240,
    280,
    320,
    360,
    400,
)
BASE_UPDATES_PER_EPOCH = 248
EXPECTED_VAL_CLIPS = 1_715
EXPECTED_SHARDS = 8
EXPECTED_TEST_CLIPS = 1_708
PRIMARY_METRIC = "body.released2.metrics.FGD"
ARTIFACT_KEYS = {"path", "sha256", "bytes"}
PAYLOAD_ARTIFACT_KEYS = {
    "path",
    "sha256",
    "bytes",
    "receipt_payload_sha256",
}
TEST_POLICY = {
    "authorized_evaluations": 1,
    "one_shot_claim_required": True,
    "selection_feedback": False,
    "num_shards": EXPECTED_SHARDS,
    "canonical_test_clips": EXPECTED_TEST_CLIPS,
}

_FORBIDDEN_COMPACT_TOKENS = (
    "speaker2",
    "semgate",
    "sparse",
    "diff" + "sheg",
)
_GLOBAL_MODEL_TOKEN = "globaldiff"


class PublishedWinnerClaimError(RuntimeError):
    """Raised when a published winner claim is not self-contained and exact."""


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            "receipt is not canonical finite JSON"
        ) from error
    return (payload + ("\n" if newline else "")).encode("utf-8")


def canonical_json_sha256(value: Any, *, newline: bool = False) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(value, newline=newline)
    ).hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublishedWinnerClaimError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except PublishedWinnerClaimError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            f"{label} is not strict JSON: {error}"
        ) from error


def _exact_mapping(
    value: Any,
    keys: Iterable[str],
    label: str,
) -> dict[str, Any]:
    expected = set(keys)
    if not isinstance(value, dict) or set(value) != expected:
        raise PublishedWinnerClaimError(f"{label} schema mismatch")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublishedWinnerClaimError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PublishedWinnerClaimError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise PublishedWinnerClaimError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PublishedWinnerClaimError(
            f"{label} must be finite and nonnegative"
        )
    return result


def _regular_file(path_value: Any, label: str) -> Path:
    if not isinstance(path_value, (str, os.PathLike)):
        raise PublishedWinnerClaimError(f"{label} path must be path-like")
    path = Path(path_value)
    if not path.is_absolute():
        raise PublishedWinnerClaimError(f"{label} path must be absolute")
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise PublishedWinnerClaimError(
            f"cannot stat {label}: {path}"
        ) from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise PublishedWinnerClaimError(
            f"{label} must be a regular non-symlink file"
        )
    return path.resolve(strict=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_artifact(
    value: Any,
    label: str,
    *,
    with_payload: bool,
) -> tuple[dict[str, Any], bytes]:
    keys = PAYLOAD_ARTIFACT_KEYS if with_payload else ARTIFACT_KEYS
    artifact = _exact_mapping(value, keys, f"{label} artifact")
    path = _regular_file(artifact["path"], label)
    payload = path.read_bytes()
    expected_sha = _require_sha256(
        artifact["sha256"], f"{label} file SHA-256"
    )
    expected_bytes = _require_int(
        artifact["bytes"], f"{label} bytes", minimum=1
    )
    if _sha256_bytes(payload) != expected_sha or len(payload) != expected_bytes:
        raise PublishedWinnerClaimError(f"{label} artifact changed")
    normalized = {
        "path": str(path),
        "sha256": expected_sha,
        "bytes": expected_bytes,
    }
    if with_payload:
        normalized["receipt_payload_sha256"] = _require_sha256(
            artifact["receipt_payload_sha256"],
            f"{label} payload SHA-256",
        )
    if normalized != artifact:
        raise PublishedWinnerClaimError(
            f"{label} artifact path is not canonical"
        )
    return normalized, payload


def _verify_compact_receipt(
    artifact: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized, payload = _normalize_artifact(
        artifact, label, with_payload=True
    )
    value = _strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise PublishedWinnerClaimError(f"{label} must be a JSON object")
    claimed = _require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} embedded payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256")
    if (
        canonical_json_sha256(unsigned) != claimed
        or normalized["receipt_payload_sha256"] != claimed
    ):
        raise PublishedWinnerClaimError(f"{label} payload hash mismatch")
    return normalized, value


def _verify_compact_reference(
    artifact_value: Any,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _exact_mapping(
        artifact_value,
        {"path", "sha256", "receipt_payload_sha256"},
        f"{label} artifact",
    )
    path = _regular_file(artifact["path"], label)
    payload = path.read_bytes()
    file_sha = _require_sha256(artifact["sha256"], f"{label} file SHA-256")
    payload_sha = _require_sha256(
        artifact["receipt_payload_sha256"], f"{label} payload SHA-256"
    )
    if _sha256_bytes(payload) != file_sha:
        raise PublishedWinnerClaimError(f"{label} artifact changed")
    value = _strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise PublishedWinnerClaimError(f"{label} must be a JSON object")
    embedded = _require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} embedded payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256")
    normalized = {
        "path": str(path),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
    }
    if (
        normalized != artifact
        or embedded != payload_sha
        or canonical_json_sha256(unsigned) != embedded
    ):
        raise PublishedWinnerClaimError(f"{label} payload hash mismatch")
    return normalized, value


def _compact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _reject_forbidden_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "origin" and child != EXPECTED_ORIGIN:
                raise PublishedWinnerClaimError(
                    f"{label}.{key} has a foreign repository origin"
                )
            _reject_forbidden_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_tree(child, f"{label}[{index}]")
    elif isinstance(value, str):
        compact = _compact_token(value)
        if any(token in compact for token in _FORBIDDEN_COMPACT_TOKENS):
            raise PublishedWinnerClaimError(
                f"{label} contains a forbidden generator dependency"
            )


def _validate_checkpoint(value: Any, label: str) -> dict[str, Any]:
    artifact, _payload = _normalize_artifact(
        value, label, with_payload=False
    )
    compact_path = _compact_token(artifact["path"])
    if (
        any(token in compact_path for token in _FORBIDDEN_COMPACT_TOKENS)
        or _GLOBAL_MODEL_TOKEN in compact_path
    ):
        raise PublishedWinnerClaimError(
            f"{label} is not a SemTalk checkpoint dependency"
        )
    return artifact


def _validate_prerequisite_selection(
    artifact_value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    artifact, selection = _verify_compact_receipt(
        artifact_value, "prerequisite selection"
    )
    _exact_mapping(
        selection,
        {
            "format",
            "status",
            "target_dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "protocol",
            "selection_policy",
            "canonical_view",
            "producer_sources",
            "training_sources",
            "config_sha256",
            "candidate_index_receipt",
            "measurement_index_receipt",
            "stages",
            "receipt_payload_sha256",
        },
        "prerequisite selection",
    )
    if (
        selection["format"] != PREREQUISITE_SELECTION_FORMAT
        or selection["status"] != "selected"
        or selection["target_dataset"] != "SHOW"
        or selection["target_speaker_scope"] != EXPECTED_SCOPE
        or selection["split"] != "val"
        or selection["test_visible"] is not False
    ):
        raise PublishedWinnerClaimError(
            "prerequisite selection identity mismatch"
        )
    if selection["protocol"] != {
        "name": "five_independent_show_prerequisite_validation_v1",
        "candidate_epochs": list(PREREQUISITE_CANDIDATE_EPOCHS),
        "candidates_per_stage": len(PREREQUISITE_CANDIDATE_EPOCHS),
        "clips_per_candidate": EXPECTED_VAL_CLIPS,
        "shards_per_candidate": EXPECTED_SHARDS,
        "window_length": 64,
        "window_stride": 20,
        "full_base_fgd_used": False,
    }:
        raise PublishedWinnerClaimError(
            "prerequisite selection protocol mismatch"
        )
    if selection["selection_policy"] != {
        "per_stage_independent": True,
        "ordering": [
            "selection_score",
            "epoch",
            "optimizer_updates",
            "checkpoint_sha256",
        ],
        "test_feedback_into_selection": False,
    }:
        raise PublishedWinnerClaimError(
            "prerequisite selection policy mismatch"
        )
    _require_sha256(selection["config_sha256"], "prerequisite config SHA")
    _reject_forbidden_tree(selection, "prerequisite selection")
    for role in ("candidate_index_receipt", "measurement_index_receipt"):
        _verify_compact_reference(selection[role], f"prerequisite {role}")
    stages = selection["stages"]
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise PublishedWinnerClaimError(
            "prerequisite selection must contain exactly five stages"
        )
    fixed: dict[str, dict[str, Any]] = {}
    for expected_stage, raw_stage in zip(STAGES, stages):
        stage = _exact_mapping(
            raw_stage,
            {
                "stage",
                "selection_metric",
                "epoch",
                "optimizer_updates",
                "candidate_index",
                "selection_score",
                "candidate_checkpoint",
                "measurement_receipt",
                "coverage",
            },
            f"prerequisite {expected_stage}",
        )
        epoch = _require_int(
            stage["epoch"], f"prerequisite {expected_stage} epoch", minimum=1
        )
        if (
            stage["stage"] != expected_stage
            or stage["selection_metric"]
            != STAGE_SELECTION_METRICS[expected_stage]
            or epoch not in PREREQUISITE_CANDIDATE_EPOCHS
            or stage["optimizer_updates"]
            != epoch * PREREQUISITE_UPDATES_PER_EPOCH
            or not isinstance(stage["candidate_index"], int)
            or isinstance(stage["candidate_index"], bool)
            or not 0 <= stage["candidate_index"]
            < len(PREREQUISITE_CANDIDATE_EPOCHS)
            or _require_number(
                stage["selection_score"],
                f"prerequisite {expected_stage} selection score",
            )
            < 0.0
            or stage["coverage"]
            != {
                "split": "val",
                "test_visible": False,
                "clips": EXPECTED_VAL_CLIPS,
                "shards": EXPECTED_SHARDS,
                "exact_once": True,
                "all_finite": True,
            }
        ):
            raise PublishedWinnerClaimError(
                f"prerequisite {expected_stage} selection changed"
            )
        _verify_compact_reference(
            stage["measurement_receipt"],
            f"prerequisite {expected_stage} measurement",
        )
        fixed[expected_stage] = _validate_checkpoint(
            stage["candidate_checkpoint"],
            f"prerequisite {expected_stage} checkpoint",
        )
    return artifact, selection, fixed


def _validate_continuation_decision(
    artifact_value: Any,
    *,
    prerequisite_artifact: Mapping[str, Any],
    prerequisite_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, decision = _verify_compact_receipt(
        artifact_value, "prerequisite continuation decision"
    )
    _exact_mapping(
        decision,
        {
            "format",
            "status",
            "decision",
            "test_visible",
            "protocol",
            "inputs",
            "stages",
            "receipt_payload_sha256",
        },
        "prerequisite continuation decision",
    )
    if (
        decision["format"] != CONTINUATION_DECISION_FORMAT
        or decision["status"] != "complete"
        or decision["decision"] != "stop"
        or decision["test_visible"] is not False
    ):
        raise PublishedWinnerClaimError(
            "prerequisite continuation is not a final stop decision"
        )
    if decision["protocol"] != {
        "name": "fresh_replayed_recent_val_improvement_v1",
        "score_direction": "lower_is_better",
        "recent_candidates": 3,
        "relative_improvement_reference": (
            "best_of_preceding_two_recent_candidates"
        ),
        "minimum_relative_improvement": 0.005,
        "continue_rule": (
            "any_stage_latest_boundary_is_winner_and_relative_"
            "improvement_gte_threshold"
        ),
    }:
        raise PublishedWinnerClaimError(
            "prerequisite continuation policy mismatch"
        )
    inputs = _exact_mapping(
        decision["inputs"],
        {"selection", "measurement_index", "stage_measurements"},
        "prerequisite continuation inputs",
    )
    expected_selection_binding = {
        key: prerequisite_artifact[key]
        for key in ("path", "sha256", "receipt_payload_sha256")
    }
    if inputs["selection"] != expected_selection_binding:
        raise PublishedWinnerClaimError(
            "continuation decision does not bind the pinned selection"
        )
    stage_rows = prerequisite_selection["stages"]
    decision_stages = decision["stages"]
    if not isinstance(decision_stages, list) or len(decision_stages) != len(
        STAGES
    ):
        raise PublishedWinnerClaimError(
            "continuation decision stage coverage mismatch"
        )
    for expected_stage, selected, raw_decision in zip(
        STAGES, stage_rows, decision_stages
    ):
        stage = _exact_mapping(
            raw_decision,
            {
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
                "requests_continuation",
            },
            f"continuation {expected_stage}",
        )
        if (
            stage["stage"] != expected_stage
            or stage["winner_epoch"] != selected["epoch"]
            or stage["requests_continuation"] is not False
        ):
            raise PublishedWinnerClaimError(
                f"continuation {expected_stage} contradicts selection"
            )
        recent_epochs = stage["recent_candidate_epochs"]
        recent_scores = stage["recent_selection_scores"]
        if (
            not isinstance(recent_epochs, list)
            or len(recent_epochs) != 3
            or not all(
                isinstance(epoch, int) and not isinstance(epoch, bool)
                for epoch in recent_epochs
            )
            or not isinstance(recent_scores, list)
            or len(recent_scores) != 3
        ):
            raise PublishedWinnerClaimError(
                f"continuation {expected_stage} recent evidence changed"
            )
        for index, score in enumerate(recent_scores):
            _require_number(
                score, f"continuation {expected_stage} score {index}"
            )
        _require_number(
            stage["previous_best_score"],
            f"continuation {expected_stage} previous score",
        )
        _require_number(
            stage["latest_score"],
            f"continuation {expected_stage} latest score",
        )
        relative = stage["relative_improvement"]
        if (
            isinstance(relative, bool)
            or type(relative) not in {int, float}
            or not math.isfinite(float(relative))
        ):
            raise PublishedWinnerClaimError(
                f"continuation {expected_stage} improvement is invalid"
            )
        for key in (
            "latest_is_winner",
            "meets_relative_improvement_threshold",
        ):
            if not isinstance(stage[key], bool):
                raise PublishedWinnerClaimError(
                    f"continuation {expected_stage} {key} is invalid"
                )
    _reject_forbidden_tree(decision, "prerequisite continuation")
    return artifact, decision


def _validate_distribution(
    artifact_value: Any,
    *,
    prediction_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, receipt = _verify_compact_receipt(
        artifact_value, "candidate distribution receipt"
    )
    if (
        receipt.get("format") != DISTRIBUTION_FORMAT
        or receipt.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM
        or receipt.get("protocol")
        != "deterministic_replication_of_single_prediction_v1"
        or receipt.get("physical_samples_per_clip") != 1
        or receipt.get("independent_samples") is not False
        or receipt.get("deterministic_delta_distribution") is not True
        or receipt.get("seed_consumed") is not False
        or receipt.get("logical_slots") != list(range(16))
        or receipt.get("released2_slots") != [0, 1]
        or receipt.get("paper16_slots") != list(range(16))
        or receipt.get("face_slot") != 0
        or receipt.get("prediction_manifest") != prediction_manifest
    ):
        raise PublishedWinnerClaimError(
            "candidate distribution receipt changed"
        )
    variation = receipt.get("variation_policy")
    if (
        not isinstance(variation, dict)
        or variation.get("reported_statistic")
        != "raw_metric_primitive_v1"
        or variation.get("exact_zero_claim") is not False
        or variation.get("public_value_transform")
        != "identity_no_clamp_no_round_v1"
    ):
        raise PublishedWinnerClaimError(
            "candidate Variation policy does not expose the raw primitive"
        )
    return artifact, receipt


def _validate_lineage(
    artifact_value: Any,
    *,
    epoch: int,
    updates: int,
    checkpoint: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    distribution_artifact: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, lineage = _verify_compact_receipt(
        artifact_value, f"Base e{epoch} inference lineage"
    )
    _exact_mapping(
        lineage,
        {
            "format",
            "status",
            "generator",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "epoch",
            "optimizer_updates",
            "candidate_checkpoint",
            "prediction_manifest",
            "distribution_receipt",
            "prerequisite_selection",
            "continuation_decision",
            "receipt_payload_sha256",
        },
        f"Base e{epoch} inference lineage",
    )
    if lineage != {
        "format": VAL_LINEAGE_FORMAT,
        "status": "complete",
        "generator": "SemTalk Base Motion Generation",
        "dataset": "SHOW",
        "target_speaker_scope": EXPECTED_SCOPE,
        "split": "val",
        "test_visible": False,
        "epoch": epoch,
        "optimizer_updates": updates,
        "candidate_checkpoint": checkpoint,
        "prediction_manifest": prediction_manifest,
        "distribution_receipt": distribution_artifact,
        "prerequisite_selection": prerequisite_artifact,
        "continuation_decision": continuation_artifact,
        "receipt_payload_sha256": lineage["receipt_payload_sha256"],
    }:
        raise PublishedWinnerClaimError(
            f"Base e{epoch} inference lineage changed"
        )
    _reject_forbidden_tree(lineage, f"Base e{epoch} inference lineage")
    return artifact, lineage


def _validate_metric_report(
    artifact_value: Any,
    *,
    prediction_manifest: Mapping[str, Any],
    lineage_artifact: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    artifact, payload = _normalize_artifact(
        artifact_value, "TalkSHOW validation metric report", with_payload=False
    )
    report = _strict_json_bytes(payload, "TalkSHOW validation metric report")
    _exact_mapping(
        report,
        {
            "format",
            "report_payload_hash_algorithm",
            "status",
            "generator",
            "dataset",
            "split",
            "selection_protocol",
            "distribution_receipt",
            "inputs",
            "metric_assets",
            "counts",
            "body",
            "face",
            "rs",
            "runtime",
            "formal_mode",
            "test_only_mode",
            "report_payload_sha256",
        },
        "TalkSHOW validation metric report",
    )
    claimed = _require_sha256(
        report["report_payload_sha256"],
        "TalkSHOW report payload SHA-256",
    )
    unsigned = dict(report)
    unsigned.pop("report_payload_sha256")
    if (
        canonical_json_sha256(unsigned, newline=True) != claimed
        or report["report_payload_hash_algorithm"]
        != METRIC_REPORT_HASH_ALGORITHM
    ):
        raise PublishedWinnerClaimError(
            "TalkSHOW validation report payload hash mismatch"
        )
    expected_protocol = {
        "primary_metric": PRIMARY_METRIC,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 0,
    }
    if (
        report["format"] != METRIC_REPORT_FORMAT
        or report["status"] != "complete"
        or report["generator"] != "SemTalk Base-only"
        or report["dataset"] != EXPECTED_DATASET
        or report["split"] != "val"
        or report["selection_protocol"] != expected_protocol
        or report["distribution_receipt"] != distribution
        or report["formal_mode"] is not True
        or report["test_only_mode"] is not False
    ):
        raise PublishedWinnerClaimError(
            "TalkSHOW validation report identity changed"
        )
    inputs = report["inputs"]
    if not isinstance(inputs, dict):
        raise PublishedWinnerClaimError("TalkSHOW report inputs are invalid")
    prediction_input = inputs.get("prediction_manifest")
    lineage_input = inputs.get("prediction_lineage")
    if (
        not isinstance(prediction_input, dict)
        or {
            key: prediction_input.get(key) for key in ARTIFACT_KEYS
        }
        != prediction_manifest
        or not isinstance(lineage_input, dict)
        or {
            key: lineage_input.get(key) for key in ARTIFACT_KEYS
        }
        != {
            key: lineage_artifact[key] for key in ARTIFACT_KEYS
        }
        or lineage_input.get("payload_sha256")
        != lineage_artifact["receipt_payload_sha256"]
    ):
        raise PublishedWinnerClaimError(
            "TalkSHOW report input bindings changed"
        )
    try:
        metric_adapter = importlib.import_module(
            "scripts.show_base.evaluate_talkshow_show_metrics"
        )
        validation = metric_adapter.validate_report(
            report,
            expected_split="val",
            expected_clip_count=EXPECTED_VAL_CLIPS,
            expected_prediction_manifest=prediction_manifest,
            expected_distribution_receipt=distribution,
            expected_selection_protocol=expected_protocol,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            "neutral TalkSHOW report replay failed"
        ) from error
    body = report["body"]
    released = body.get("released2") if isinstance(body, dict) else None
    metrics = released.get("metrics") if isinstance(released, dict) else None
    if not isinstance(metrics, dict) or set(metrics) != {
        "FGD",
        "Variation",
        "BC",
    }:
        raise PublishedWinnerClaimError(
            "TalkSHOW released2 metric schema changed"
        )
    fgd = _require_number(metrics["FGD"], PRIMARY_METRIC)
    if (
        not isinstance(validation, dict)
        or validation.get("status") != "pass"
        or validation.get("split") != "val"
        or validation.get("clips") != EXPECTED_VAL_CLIPS
        or validation.get("primary_metric_path") != PRIMARY_METRIC
        or _require_number(
            validation.get("primary_metric"),
            "fresh TalkSHOW primary metric",
        )
        != fgd
    ):
        raise PublishedWinnerClaimError(
            "neutral TalkSHOW report replay changed the primary metric"
        )
    return artifact, fgd


def _validate_winner_selection(
    artifact_value: Any,
    *,
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact, selection = _verify_compact_receipt(
        artifact_value, "Base validation winner selection"
    )
    _exact_mapping(
        selection,
        {
            "format",
            "payload_hash_algorithm",
            "status",
            "generator",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "selection_eligible",
            "primary_metric",
            "selection_policy",
            "prerequisite_selection",
            "continuation_decision",
            "candidates",
            "selected",
            "test_policy",
            "receipt_payload_sha256",
        },
        "Base validation winner selection",
    )
    if (
        selection["format"] != WINNER_SELECTION_FORMAT
        or selection["payload_hash_algorithm"] != PAYLOAD_HASH_ALGORITHM
        or selection["status"] != "selected"
        or selection["generator"] != "SemTalk Base Motion Generation"
        or selection["dataset"] != "SHOW"
        or selection["target_speaker_scope"] != EXPECTED_SCOPE
        or selection["split"] != "val"
        or selection["test_visible"] is not False
        or selection["selection_eligible"] is not True
        or selection["primary_metric"] != PRIMARY_METRIC
        or selection["prerequisite_selection"] != prerequisite_artifact
        or selection["continuation_decision"] != continuation_artifact
        or selection["test_policy"] != TEST_POLICY
    ):
        raise PublishedWinnerClaimError(
            "Base validation winner selection identity changed"
        )
    expected_policy = {
        "candidate_epochs": list(BASE_CANDIDATE_EPOCHS),
        "updates_per_epoch": BASE_UPDATES_PER_EPOCH,
        "operator": "min",
        "ordering": [PRIMARY_METRIC, "epoch", "optimizer_updates"],
        "test_feedback_into_selection": False,
    }
    if selection["selection_policy"] != expected_policy:
        raise PublishedWinnerClaimError(
            "Base validation selection policy changed"
        )
    candidates = selection["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(
        BASE_CANDIDATE_EPOCHS
    ):
        raise PublishedWinnerClaimError(
            "Base validation selection candidate coverage mismatch"
        )
    row_keys = {
        "epoch",
        "optimizer_updates",
        "candidate_checkpoint",
        "prediction_manifest",
        "inference_lineage",
        "distribution_receipt",
        "talkshow_metric_report",
        "body_released2_fgd",
    }
    replayed: list[tuple[float, int, int, dict[str, Any]]] = []
    checkpoint_paths: set[str] = set()
    for expected_epoch, raw_row in zip(BASE_CANDIDATE_EPOCHS, candidates):
        row = _exact_mapping(
            raw_row, row_keys, f"Base e{expected_epoch} selection row"
        )
        epoch = _require_int(row["epoch"], "Base candidate epoch", minimum=1)
        updates = _require_int(
            row["optimizer_updates"],
            f"Base e{expected_epoch} optimizer updates",
            minimum=1,
        )
        if (
            epoch != expected_epoch
            or updates != epoch * BASE_UPDATES_PER_EPOCH
        ):
            raise PublishedWinnerClaimError(
                f"Base e{expected_epoch} candidate boundary changed"
            )
        checkpoint = _validate_checkpoint(
            row["candidate_checkpoint"], f"Base e{epoch} checkpoint"
        )
        if checkpoint["path"] in checkpoint_paths:
            raise PublishedWinnerClaimError(
                "Base candidate checkpoint path was reused"
            )
        checkpoint_paths.add(checkpoint["path"])
        prediction_manifest, _manifest_payload = _normalize_artifact(
            row["prediction_manifest"],
            f"Base e{epoch} prediction manifest",
            with_payload=False,
        )
        distribution_artifact, distribution = _validate_distribution(
            row["distribution_receipt"],
            prediction_manifest=prediction_manifest,
        )
        lineage_artifact, _lineage = _validate_lineage(
            row["inference_lineage"],
            epoch=epoch,
            updates=updates,
            checkpoint=checkpoint,
            prediction_manifest=prediction_manifest,
            distribution_artifact=distribution_artifact,
            prerequisite_artifact=prerequisite_artifact,
            continuation_artifact=continuation_artifact,
        )
        report_artifact, fgd = _validate_metric_report(
            row["talkshow_metric_report"],
            prediction_manifest=prediction_manifest,
            lineage_artifact=lineage_artifact,
            distribution=distribution,
        )
        if _require_number(
            row["body_released2_fgd"],
            f"Base e{epoch} declared released2 FGD",
        ) != fgd:
            raise PublishedWinnerClaimError(
                f"Base e{epoch} declared FGD differs from report"
            )
        normalized_row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "candidate_checkpoint": checkpoint,
            "prediction_manifest": prediction_manifest,
            "inference_lineage": lineage_artifact,
            "distribution_receipt": distribution_artifact,
            "talkshow_metric_report": report_artifact,
            "body_released2_fgd": fgd,
        }
        if normalized_row != row:
            raise PublishedWinnerClaimError(
                f"Base e{epoch} selection row is not canonical"
            )
        replayed.append((fgd, epoch, updates, normalized_row))
    winner = min(replayed, key=lambda item: (item[0], item[1], item[2]))[3]
    if selection["selected"] != winner:
        raise PublishedWinnerClaimError(
            "published Base winner differs from fresh released2 FGD replay"
        )
    _reject_forbidden_tree(selection, "Base winner selection")
    return artifact, selection, winner


def _expected_output_path(value: Any, label: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise PublishedWinnerClaimError(f"{label} must be path-like")
    path = Path(value)
    if not path.is_absolute():
        raise PublishedWinnerClaimError(f"{label} must be absolute")
    return str(path.resolve(strict=False))


def validate_published_test_winner_claim(
    claim_path: str | os.PathLike[str],
    *,
    expected_claim_sha256: str,
    expected_claim_bytes: int,
    expected_claim_payload_sha256: str,
    expected_output_root: str | os.PathLike[str],
    prerequisite_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return one immutable, one-shot SHOW test authorization.

    ``prerequisite_selection`` and ``continuation_decision`` are externally
    pinned artifact mappings.  No checkpoint mapping is accepted from the
    caller: all six checkpoints are recovered from fresh receipt replay.
    """

    claim_file = _regular_file(claim_path, "published winner claim")
    claim_payload = claim_file.read_bytes()
    expected_file_sha = _require_sha256(
        expected_claim_sha256, "published claim file SHA-256"
    )
    expected_size = _require_int(
        expected_claim_bytes, "published claim bytes", minimum=1
    )
    expected_payload_sha = _require_sha256(
        expected_claim_payload_sha256,
        "published claim payload SHA-256",
    )
    if (
        _sha256_bytes(claim_payload) != expected_file_sha
        or len(claim_payload) != expected_size
    ):
        raise PublishedWinnerClaimError("published claim artifact changed")
    claim = _strict_json_bytes(claim_payload, "published winner claim")
    _exact_mapping(
        claim,
        {
            "format",
            "payload_hash_algorithm",
            "status",
            "generator",
            "dataset",
            "target_speaker_scope",
            "winner_selection",
            "prerequisite_selection",
            "continuation_decision",
            "selected_base_checkpoint",
            "fixed_checkpoints",
            "expected_output_root",
            "test_policy",
            "test_visible_during_selection",
            "receipt_payload_sha256",
        },
        "published winner claim",
    )
    embedded_payload_sha = _require_sha256(
        claim["receipt_payload_sha256"],
        "published claim embedded payload SHA-256",
    )
    unsigned_claim = dict(claim)
    unsigned_claim.pop("receipt_payload_sha256")
    if (
        canonical_json_sha256(unsigned_claim) != embedded_payload_sha
        or embedded_payload_sha != expected_payload_sha
    ):
        raise PublishedWinnerClaimError("published claim payload hash mismatch")
    expected_root = _expected_output_path(
        expected_output_root, "expected output root"
    )
    if (
        claim["format"] != CLAIM_FORMAT
        or claim["payload_hash_algorithm"] != PAYLOAD_HASH_ALGORITHM
        or claim["status"] != "authorized"
        or claim["generator"] != "SemTalk Base Motion Generation"
        or claim["dataset"] != "SHOW"
        or claim["target_speaker_scope"] != EXPECTED_SCOPE
        or claim["expected_output_root"] != expected_root
        or claim["test_policy"] != TEST_POLICY
        or claim["test_visible_during_selection"] is not False
    ):
        raise PublishedWinnerClaimError("published claim identity changed")
    prerequisite_artifact, prerequisite_payload, fixed_checkpoints = (
        _validate_prerequisite_selection(prerequisite_selection)
    )
    continuation_artifact, _continuation_payload = (
        _validate_continuation_decision(
            continuation_decision,
            prerequisite_artifact=prerequisite_artifact,
            prerequisite_selection=prerequisite_payload,
        )
    )
    winner_artifact, _winner_payload, winner = _validate_winner_selection(
        claim["winner_selection"],
        prerequisite_artifact=prerequisite_artifact,
        continuation_artifact=continuation_artifact,
    )
    if (
        claim["prerequisite_selection"] != prerequisite_artifact
        or claim["continuation_decision"] != continuation_artifact
        or claim["winner_selection"] != winner_artifact
        or claim["selected_base_checkpoint"]
        != winner["candidate_checkpoint"]
        or claim["fixed_checkpoints"] != fixed_checkpoints
    ):
        raise PublishedWinnerClaimError(
            "published claim differs from freshly derived checkpoint authority"
        )
    _reject_forbidden_tree(claim, "published winner claim")
    claim_artifact = {
        "path": str(claim_file),
        "sha256": expected_file_sha,
        "bytes": expected_size,
    }
    return {
        "claim_artifact": claim_artifact,
        "receipt_payload_sha256": embedded_payload_sha,
        "winner_selection": winner_artifact,
        "selected_base_checkpoint": dict(winner["candidate_checkpoint"]),
        "fixed_checkpoints": {
            stage: dict(fixed_checkpoints[stage]) for stage in STAGES
        },
        "expected_output_root": expected_root,
        "test_policy": dict(TEST_POLICY),
    }


__all__ = [
    "AUTHORIZATION_FORMAT",
    "BASE_CANDIDATE_EPOCHS",
    "BASE_UPDATES_PER_EPOCH",
    "CLAIM_FORMAT",
    "CONTINUATION_DECISION_FORMAT",
    "DISTRIBUTION_FORMAT",
    "EXPECTED_SCOPE",
    "METRIC_REPORT_FORMAT",
    "PAYLOAD_HASH_ALGORITHM",
    "PREREQUISITE_SELECTION_FORMAT",
    "PublishedWinnerClaimError",
    "STAGES",
    "TEST_POLICY",
    "VAL_LINEAGE_FORMAT",
    "WINNER_SELECTION_FORMAT",
    "canonical_json_sha256",
    "validate_published_test_winner_claim",
]
