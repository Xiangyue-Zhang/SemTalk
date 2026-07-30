#!/usr/bin/env python3
"""SemTalk SHOW task-space gate v2.

This is a new protocol.  It does not reinterpret, replace, or mutate any v1
measurement or decision receipt.

The protocol has four deliberately separate phases:

``measure``
    Measure exactly one canonical SHOW split.  Validation may measure any
    candidate.  Test requires an immutable validation selection lock and an
    atomically claimed one-shot test token.  Measurements never authorize
    inference and contain no thresholds.

``decide``
    Apply an independently frozen threshold file to one immutable measurement.
    Only validation decisions can authorize ``lock``.  A test decision is
    evaluation-only and can never select a candidate.

``lock``
    Lock one candidate from an accepted validation decision.  The resulting
    receipt authorizes exactly one test measurement of exactly those weights.

``inspect-weights``
    CPU-only validation of a candidate weight manifest.  This is useful before
    a guarded GPU run and emits the same lineage receipt used by ``measure``.

Only task-space quantities used by the released models are decision metrics:

* face jaw rotation geodesic error and face expression error;
* hands, upper-body, and lower-body rotation geodesic error;
* lower-body contact error;
* global/root channel error and integrated translation error.

Raw rotation-6D errors, the untrained lower translation channels, and global
rotation errors are intentionally absent.  The integrated zero-velocity
baseline uses the same ground-truth frame-0 x/z anchor as the model output.
The Base checkpoint is cryptographically bound but explicitly
``bound_not_evaluated``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import gate_released_all_speakers_on_show as _v1


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
MEASUREMENT_FORMAT = "semtalk_show_task_space_measurement_v2"
THRESHOLD_FORMAT = "semtalk_show_task_space_thresholds_v2"
DECISION_FORMAT = "semtalk_show_task_space_decision_v2"
LOCK_FORMAT = "semtalk_show_task_space_selection_lock_v2"
WEIGHTS_FORMAT = "semtalk_show_task_space_candidate_weights_v2"
LINEAGE_FORMAT = "semtalk_show_task_space_candidate_lineage_v2"
TEST_CLAIM_FORMAT = "semtalk_show_task_space_test_once_claim_v2"
DEPENDENCY_SCRIPT = PROJECT_ROOT / "scripts/show_base/gate_released_all_speakers_on_show.py"

SPLITS = ("val", "test")
EXPECTED_SPLIT_COUNTS = {"val": 1_715, "test": 1_708}
STAGE_NAMES = ("face", "hands", "upper", "lower")
WEIGHT_NAMES = ("base", "face", "hands", "upper", "lower", "global")
EVALUATED_WEIGHT_NAMES = ("face", "hands", "upper", "lower", "global")
BASE_STATUS = "bound_not_evaluated"

METRIC_PROTOCOLS: dict[str, dict[str, str]] = {
    "face_jaw_geodesic": {
        "units": "radians",
        "baseline": "identity_rotation",
    },
    "face_expression": {
        "units": "canonical_expression_units",
        "baseline": "zero_expression",
    },
    "hands_rotation_geodesic": {
        "units": "radians",
        "baseline": "identity_rotation",
    },
    "upper_rotation_geodesic": {
        "units": "radians",
        "baseline": "identity_rotation",
    },
    "lower_rotation_geodesic": {
        "units": "radians",
        "baseline": "identity_rotation",
    },
    "lower_contact": {
        "units": "contact_probability",
        "baseline": "zero_contact",
    },
    "global_root_channels": {
        "units": "velocity_x_height_y_velocity_z",
        "baseline": "zero_root_channels",
    },
    "global_integrated_translation": {
        "units": "canonical_translation_units",
        "baseline": (
            "zero_root_channels_integrated_with_same_gt_frame0_xz_anchor"
        ),
    },
}
FORBIDDEN_DECISION_METRICS = (
    "raw_rotation6d",
    "full_error",
    "lower_translation",
    "global_rotation",
)
THRESHOLD_PATHS = tuple(
    f"metrics.{metric}.baseline.model_to_baseline_ratio.{statistic}"
    for metric in METRIC_PROTOCOLS
    for statistic in ("mae", "rmse")
)
METRIC_SET_SHA256 = hashlib.sha256(
    _v1.canonical_json_bytes(list(THRESHOLD_PATHS))
).hexdigest()

ARCHITECTURES: dict[str, dict[str, Any]] = {
    name: {
        key: value
        for key, value in _v1.WEIGHT_SPECS[name].items()
        if key in {"model", "dimension", "vae_layer"}
    }
    for name in EVALUATED_WEIGHT_NAMES
}


def canonical_json_bytes(value: Any) -> bytes:
    return _v1.canonical_json_bytes(value)


def sha256_file(path: Path) -> str:
    return _v1.sha256_file(path)


def require_sha256(value: Any, label: str) -> str:
    return _v1.require_sha256(value, label)


def require_git_oid(value: Any, label: str) -> str:
    return _v1.require_git_oid(value, label)


def require_exact_int(value: Any, label: str) -> int:
    return _v1.require_exact_int(value, label)


def require_finite_number(value: Any, label: str) -> float:
    return _v1.require_finite_number(value, label)


def add_receipt_payload_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _v1.add_receipt_payload_hash(payload)


def verify_receipt_payload_hash(payload: Mapping[str, Any], label: str) -> None:
    _v1.verify_receipt_payload_hash(payload, label)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def require_finite_tree(value: Any, label: str = "JSON") -> None:
    """Reject NaN/Inf and unsupported JSON values recursively."""
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains NaN/Inf")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} has a non-string object key")
            require_finite_tree(item, f"{label}.{key}")
        return
    raise TypeError(f"{label} contains unsupported JSON type {type(value).__name__}")


def strict_json_loads(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error
    require_finite_tree(value, label)
    return value


def read_verified_artifact(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    return _v1.read_verified_bytes(path, expected_sha256, label)


def atomic_json_new(path: Path, value: Any) -> None:
    require_finite_tree(value, "output JSON")
    _v1.atomic_json_new(path, value)


def _artifact_receipt(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, str], bytes]:
    resolved, payload = read_verified_artifact(path, expected_sha256, label)
    return {"path": str(resolved), "sha256": expected_sha256}, payload


def _validate_candidate_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value)
    ):
        raise ValueError(
            "candidate_id must use 1..160 ASCII letters, digits, dot, underscore, or dash"
        )
    return value


def _validate_weight_entry(value: Any, name: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise RuntimeError(f"candidate weights.{name} schema mismatch")
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"candidate weights.{name}.path must be nonempty")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"candidate weights.{name}.path must be absolute")
    return path, require_sha256(
        value["sha256"],
        f"candidate weights.{name}.sha256",
    )


def _validate_optional_provenance_artifact(
    value: Any,
    label: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    path, digest = _validate_weight_entry(value, label)
    receipt, _ = _artifact_receipt(path, digest, label)
    return receipt


def load_candidate_lineage(
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify an externally supplied candidate and every weight/provenance byte."""
    manifest_receipt, payload = _artifact_receipt(
        manifest_path,
        expected_manifest_sha256,
        "candidate weight manifest",
    )
    manifest = strict_json_loads(payload, "candidate weight manifest")
    if not isinstance(manifest, dict) or set(manifest) != {
        "format",
        "candidate_id",
        "weights",
        "provenance",
    }:
        raise RuntimeError("candidate weight manifest schema mismatch")
    if manifest["format"] != WEIGHTS_FORMAT:
        raise RuntimeError("unexpected candidate weight manifest format")
    candidate_id = _validate_candidate_id(manifest["candidate_id"])
    weights = manifest["weights"]
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_NAMES):
        raise RuntimeError("candidate weight coverage must be exactly six weights")

    verified_weights: dict[str, dict[str, Any]] = {}
    for name in WEIGHT_NAMES:
        path, digest = _validate_weight_entry(weights[name], name)
        artifact, _ = _artifact_receipt(path, digest, f"candidate {name} weight")
        verified_weights[name] = {
            **artifact,
            "role": BASE_STATUS if name == "base" else "task_space_evaluated",
            "architecture": None if name == "base" else ARCHITECTURES[name],
        }

    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "kind",
        "training_receipt",
        "parent_receipt",
        "notes",
    }:
        raise RuntimeError("candidate provenance schema mismatch")
    kind = provenance["kind"]
    if kind not in {"official_release", "show_adaptation"}:
        raise RuntimeError("candidate provenance kind is invalid")
    notes = provenance["notes"]
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("candidate provenance notes must be nonempty")
    training_receipt = _validate_optional_provenance_artifact(
        provenance["training_receipt"],
        "candidate training receipt",
    )
    parent_receipt = _validate_optional_provenance_artifact(
        provenance["parent_receipt"],
        "candidate parent receipt",
    )
    if kind == "show_adaptation" and training_receipt is None:
        raise RuntimeError("SHOW-adapted candidate requires a training receipt")
    if kind == "official_release" and training_receipt is not None:
        raise RuntimeError("official-release candidate must not claim adaptation training")

    lineage = add_receipt_payload_hash(
        {
            "format": LINEAGE_FORMAT,
            "candidate_id": candidate_id,
            "candidate_manifest": manifest_receipt,
            "provenance": {
                "kind": kind,
                "training_receipt": training_receipt,
                "parent_receipt": parent_receipt,
                "notes": notes,
            },
            "weights": verified_weights,
            "evaluated_weights": list(EVALUATED_WEIGHT_NAMES),
            "bound_not_evaluated": ["base"],
        }
    )
    return manifest, lineage


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError(f"{label} schema mismatch")
    return value


def _error_summary_schema(value: Any, label: str) -> dict[str, Any]:
    value = _require_exact_keys(
        value,
        {"count", "mae", "rmse", "max_abs"},
        label,
    )
    count = require_exact_int(value["count"], f"{label}.count")
    if count <= 0:
        raise RuntimeError(f"{label}.count must be positive")
    for key in ("mae", "rmse", "max_abs"):
        number = require_finite_number(value[key], f"{label}.{key}")
        if number < 0:
            raise RuntimeError(f"{label}.{key} must not be negative")
    return value


def metric_receipt(
    model_error: Mapping[str, Any],
    baseline_error: Mapping[str, Any],
    metric: str,
) -> dict[str, Any]:
    protocol = METRIC_PROTOCOLS[metric]
    return {
        "units": protocol["units"],
        "model_error": dict(model_error),
        "baseline": _v1.baseline_comparison(
            model_error,
            baseline_error,
            protocol=protocol["baseline"],
        ),
    }


def validate_metric_receipt(value: Any, metric: str) -> None:
    value = _require_exact_keys(
        value,
        {"units", "model_error", "baseline"},
        f"metric {metric}",
    )
    protocol = METRIC_PROTOCOLS[metric]
    if value["units"] != protocol["units"]:
        raise RuntimeError(f"metric {metric} units mismatch")
    model_error = _error_summary_schema(
        value["model_error"],
        f"metric {metric}.model_error",
    )
    _v1.validate_baseline_comparison(
        value["baseline"],
        model_error,
        protocol=protocol["baseline"],
        label=f"metric {metric}",
    )


def _extract_numeric_path(payload: Mapping[str, Any], dotted_path: str) -> float:
    current: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise RuntimeError(f"missing threshold metric {dotted_path!r}")
        current = current[component]
    return require_finite_number(current, dotted_path)


def validate_thresholds(value: Any, split: str) -> list[dict[str, Any]]:
    value = _require_exact_keys(
        value,
        {"format", "split", "metric_set_sha256", "provenance", "rules"},
        "threshold file",
    )
    if value["format"] != THRESHOLD_FORMAT or value["split"] != split:
        raise RuntimeError("threshold format/split mismatch")
    if require_sha256(
        value["metric_set_sha256"],
        "threshold metric_set_sha256",
    ) != METRIC_SET_SHA256:
        raise RuntimeError("threshold metric set binding mismatch")
    provenance = _require_exact_keys(
        value["provenance"],
        {"frozen_by", "frozen_at_utc", "rationale"},
        "threshold provenance",
    )
    for field, item in provenance.items():
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"threshold provenance.{field} must be nonempty")
    rules = value["rules"]
    if not isinstance(rules, list) or len(rules) != len(THRESHOLD_PATHS):
        raise RuntimeError("threshold rules must cover the exact task-space metric set")
    seen: set[str] = set()
    for index, rule in enumerate(rules):
        rule = _require_exact_keys(
            rule,
            {"metric", "operator", "value"},
            f"threshold rule {index}",
        )
        metric = rule["metric"]
        if metric not in THRESHOLD_PATHS or metric in seen:
            raise RuntimeError(f"threshold rule {index} metric is invalid/duplicate")
        seen.add(metric)
        if any(fragment in metric for fragment in FORBIDDEN_DECISION_METRICS):
            raise RuntimeError(f"forbidden task-space metric in rule {index}")
        if rule["operator"] not in {"<", "<="}:
            raise RuntimeError(f"threshold rule {index} operator is invalid")
        value_number = require_finite_number(
            rule["value"],
            f"threshold rule {index}.value",
        )
        if value_number < 0:
            raise RuntimeError(f"threshold rule {index}.value must not be negative")
    if seen != set(THRESHOLD_PATHS):
        raise RuntimeError("threshold metric coverage mismatch")
    return rules


def evaluate_thresholds(
    measurement: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    split = measurement["split"]
    rules = validate_thresholds(thresholds, split)
    operators = {
        "<": lambda actual, expected: actual < expected,
        "<=": lambda actual, expected: actual <= expected,
    }
    results = []
    for rule in rules:
        actual = _extract_numeric_path(measurement, rule["metric"])
        expected = require_finite_number(rule["value"], "threshold")
        results.append(
            {
                "metric": rule["metric"],
                "operator": rule["operator"],
                "threshold": expected,
                "actual": actual,
                "passed": bool(operators[rule["operator"]](actual, expected)),
            }
        )
    return results


def _anchor_zero_velocity_reference(
    *,
    frames: int,
    anchor_x: float,
    anchor_z: float,
) -> list[list[float]]:
    """Pure-stdlib statement of the testable integrated baseline contract."""
    frames = require_exact_int(frames, "frames")
    anchor_x = require_finite_number(anchor_x, "anchor_x")
    anchor_z = require_finite_number(anchor_z, "anchor_z")
    if frames <= 0:
        raise ValueError("frames must be positive")
    return [[anchor_x, 0.0, anchor_z] for _ in range(frames)]


def _integrate_xz(torch: Any, channels: Any, anchor: Any) -> Any:
    """Use model and baseline channels with the exact same GT frame-0 x/z anchor."""
    if channels.shape[-1] != 3 or anchor.shape[-1] != 3:
        raise ValueError("root channels/anchor must end in dimension 3")
    result = torch.zeros_like(channels)
    result[..., 1] = channels[..., 1]
    result[:, 0, 0] = anchor[:, 0]
    result[:, 0, 2] = anchor[:, 2]
    dt = 1.0 / _v1.FPS
    for frame in range(1, channels.shape[1]):
        result[:, frame, 0] = (
            result[:, frame - 1, 0] + channels[:, frame - 1, 0] * dt
        )
        result[:, frame, 2] = (
            result[:, frame - 1, 2] + channels[:, frame - 1, 2] * dt
        )
    return result


def _dependency_receipt() -> dict[str, str]:
    path = DEPENDENCY_SCRIPT.resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("v1 utility dependency is missing/unsafe")
    return {
        "path": str(path),
        "relative": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
    }


def _source_receipt(args: argparse.Namespace) -> dict[str, Any]:
    return _v1.git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
        script=Path(__file__).resolve(),
    )


def _validate_source_receipt(
    receipt: Any,
    expected_commit: str,
    expected_tree: str,
) -> None:
    _v1.validate_entrypoint_source_receipt(
        receipt,
        label="task-space gate v2",
        expected_commit=expected_commit,
        expected_tree=expected_tree,
        expected_script_relative="scripts/show_base/gate_task_space_on_show_v2.py",
    )


def _validate_lineage_shape(lineage: Any) -> dict[str, Any]:
    lineage = _require_exact_keys(
        lineage,
        {
            "format",
            "candidate_id",
            "candidate_manifest",
            "provenance",
            "weights",
            "evaluated_weights",
            "bound_not_evaluated",
            "receipt_payload_sha256",
        },
        "candidate lineage",
    )
    verify_receipt_payload_hash(lineage, "candidate lineage")
    if (
        lineage["format"] != LINEAGE_FORMAT
        or lineage["evaluated_weights"] != list(EVALUATED_WEIGHT_NAMES)
        or lineage["bound_not_evaluated"] != ["base"]
    ):
        raise RuntimeError("candidate lineage protocol mismatch")
    _validate_candidate_id(lineage["candidate_id"])
    if not isinstance(lineage["weights"], dict) or set(lineage["weights"]) != set(
        WEIGHT_NAMES
    ):
        raise RuntimeError("candidate lineage weight coverage mismatch")
    if lineage["weights"]["base"].get("role") != BASE_STATUS:
        raise RuntimeError("Base must be bound_not_evaluated")
    return lineage


def _rehash_candidate_lineage(lineage: Mapping[str, Any]) -> None:
    """Re-open the manifest, weights, and provenance artifacts fail-closed."""
    checked = _validate_lineage_shape(lineage)
    manifest = checked["candidate_manifest"]
    if not isinstance(manifest, dict) or set(manifest) != {"path", "sha256"}:
        raise RuntimeError("candidate lineage manifest artifact schema mismatch")
    _, rebuilt = load_candidate_lineage(
        Path(manifest["path"]),
        manifest["sha256"],
    )
    if rebuilt != checked:
        raise RuntimeError("candidate lineage no longer matches its bound artifacts")


def _validate_canonical_coverage(
    receipt: Any,
    split: str,
    *,
    expected_split_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Re-hash canonical artifacts and independently recompute split coverage."""
    receipt = _require_exact_keys(
        receipt,
        {
            "manifest",
            "manifest_sha256",
            "summary",
            "summary_sha256",
            "lineage",
            "lineage_sha256",
            "lineage_contract_sha256",
        },
        "canonical receipt",
    )
    artifacts = {}
    for name in ("manifest", "summary", "lineage"):
        path = receipt[name]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise RuntimeError(f"canonical receipt {name} path must be absolute")
        artifacts[name] = _artifact_receipt(
            Path(path),
            receipt[f"{name}_sha256"],
            f"canonical {name}",
        )
    require_sha256(
        receipt["lineage_contract_sha256"],
        "canonical lineage_contract_sha256",
    )
    if expected_split_counts is None:
        expected_split_counts = _v1.EXPECTED_SPLIT_COUNTS
    manifest_payload = artifacts["manifest"][1]
    rows = []
    for line_number, raw_line in enumerate(manifest_payload.splitlines(), 1):
        if not raw_line.strip():
            raise RuntimeError(f"canonical manifest has blank line {line_number}")
        row = strict_json_loads(
            raw_line,
            f"canonical manifest line {line_number}",
        )
        if not isinstance(row, dict):
            raise TypeError(f"canonical manifest line {line_number} is not an object")
        rows.append(row)
    _v1.validate_manifest_rows(rows, expected_split_counts)
    split_rows = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: row["global_index"],
    )
    if len(split_rows) != expected_split_counts[split]:
        raise RuntimeError("canonical receipt split coverage mismatch")
    window_digest = hashlib.sha256()
    expected_windows = 0
    for row in split_rows:
        records = list(_v1.window_records(row))
        expected_windows += len(records)
        for record in records:
            window_digest.update(canonical_json_bytes(record))
    return {
        "split": split,
        "clip_count": len(split_rows),
        "expected_windows": expected_windows,
        "window_records_sha256": window_digest.hexdigest(),
    }


def validate_measurement(
    measurement: Any,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
) -> dict[str, Any]:
    measurement = _require_exact_keys(
        measurement,
        {
            "format",
            "status",
            "authorization",
            "split",
            "selection_eligible",
            "finite",
            "exact_once",
            "deterministic",
            "source_receipt",
            "implementation_dependency",
            "candidate_lineage",
            "canonical_receipt",
            "input_source_receipt",
            "protocol",
            "coverage",
            "determinism",
            "runtime",
            "test_lock",
            "test_once_claim",
            "metrics",
            "diagnostics",
            "receipt_payload_sha256",
        },
        "measurement",
    )
    verify_receipt_payload_hash(measurement, "measurement")
    split = measurement["split"]
    if (
        measurement["format"] != MEASUREMENT_FORMAT
        or measurement["status"] != "measured"
        or measurement["authorization"] is not False
        or split not in SPLITS
        or measurement["selection_eligible"] is not (split == "val")
        or measurement["finite"] is not True
        or measurement["exact_once"] is not True
        or measurement["deterministic"] is not True
    ):
        raise RuntimeError("measurement top-level protocol mismatch")
    _validate_source_receipt(
        measurement["source_receipt"],
        expected_source_commit,
        expected_source_tree,
    )
    dependency = _require_exact_keys(
        measurement["implementation_dependency"],
        {"path", "relative", "sha256"},
        "implementation dependency",
    )
    if (
        dependency["relative"]
        != "scripts/show_base/gate_released_all_speakers_on_show.py"
        or not isinstance(dependency["path"], str)
        or not Path(dependency["path"]).is_absolute()
        or require_sha256(
            dependency["sha256"],
            "implementation dependency SHA-256",
        )
        != sha256_file(Path(dependency["path"]))
    ):
        raise RuntimeError("implementation dependency binding mismatch")
    lineage = _validate_lineage_shape(measurement["candidate_lineage"])
    _rehash_candidate_lineage(lineage)
    protocol = _require_exact_keys(
        measurement["protocol"],
        {
            "task_space_version",
            "split",
            "candidate_selection",
            "test_policy",
            "base",
            "evaluated_weights",
            "metric_protocols",
            "forbidden_decision_metrics",
            "window_length",
            "window_stride",
            "fps",
            "rvq_operation",
            "global_input",
            "integrated_anchor",
            "thresholds",
        },
        "measurement protocol",
    )
    if (
        protocol["task_space_version"] != 2
        or protocol["split"] != split
        or protocol["candidate_selection"] != "validation_only"
        or protocol["test_policy"] != "locked_checkpoint_exactly_once"
        or protocol["base"] != BASE_STATUS
        or protocol["evaluated_weights"] != list(EVALUATED_WEIGHT_NAMES)
        or protocol["metric_protocols"] != METRIC_PROTOCOLS
        or protocol["forbidden_decision_metrics"]
        != list(FORBIDDEN_DECISION_METRICS)
        or protocol["window_length"] != _v1.WINDOW_LENGTH
        or protocol["window_stride"] != _v1.WINDOW_STRIDE
        or protocol["fps"] != _v1.FPS
        or protocol["rvq_operation"] != "map2index_then_decode"
        or protocol["global_input"] != "decoded_lower_projected_rotation_zero_translation"
        or protocol["integrated_anchor"] != "ground_truth_frame0_xz"
        or protocol["thresholds"] is not None
    ):
        raise RuntimeError("measurement protocol content mismatch")
    if lineage["weights"]["base"]["role"] != protocol["base"]:
        raise RuntimeError("measurement Base role mismatch")
    coverage = _require_exact_keys(
        measurement["coverage"],
        {
            "split",
            "clip_count",
            "expected_clip_count",
            "expected_windows",
            "observed_windows",
            "window_exact_once",
            "window_records_sha256",
        },
        "measurement coverage",
    )
    expected_clips = EXPECTED_SPLIT_COUNTS[split]
    canonical_coverage = _validate_canonical_coverage(
        measurement["canonical_receipt"],
        split,
    )
    if (
        coverage["split"] != split
        or coverage["clip_count"] != expected_clips
        or coverage["expected_clip_count"] != expected_clips
        or coverage["clip_count"] != canonical_coverage["clip_count"]
        or coverage["expected_windows"] != canonical_coverage["expected_windows"]
        or coverage["expected_windows"] != coverage["observed_windows"]
        or require_exact_int(
            coverage["observed_windows"],
            "coverage.observed_windows",
        )
        <= 0
        or coverage["window_exact_once"] is not True
        or coverage["window_records_sha256"]
        != canonical_coverage["window_records_sha256"]
    ):
        raise RuntimeError("measurement exact coverage mismatch")
    require_sha256(
        coverage["window_records_sha256"],
        "coverage.window_records_sha256",
    )
    metrics = measurement["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_PROTOCOLS):
        raise RuntimeError("measurement task-space metric coverage mismatch")
    for metric in METRIC_PROTOCOLS:
        validate_metric_receipt(metrics[metric], metric)
    diagnostics = _require_exact_keys(
        measurement["diagnostics"],
        {
            "codebooks_not_decision_rules",
            "forbidden_metrics_absent",
            "base",
        },
        "measurement diagnostics",
    )
    if (
        diagnostics["forbidden_metrics_absent"]
        != list(FORBIDDEN_DECISION_METRICS)
        or diagnostics["base"] != BASE_STATUS
        or not isinstance(diagnostics["codebooks_not_decision_rules"], dict)
        or set(diagnostics["codebooks_not_decision_rules"]) != set(STAGE_NAMES)
    ):
        raise RuntimeError("measurement diagnostics protocol mismatch")
    if split == "val":
        if measurement["test_lock"] is not None or measurement["test_once_claim"] is not None:
            raise RuntimeError("validation measurement must not contain test authorization")
    else:
        if not isinstance(measurement["test_lock"], dict) or not isinstance(
            measurement["test_once_claim"], dict
        ):
            raise RuntimeError("test measurement lacks lock/one-shot claim")
    require_finite_tree(measurement, "measurement")
    return measurement


def _validate_decision(value: Any) -> dict[str, Any]:
    value = _require_exact_keys(
        value,
        {
            "format",
            "status",
            "authorization",
            "evaluation_pass",
            "split",
            "selection_eligible",
            "candidate_lineage",
            "measurement_receipt",
            "threshold_receipt",
            "results",
            "protocol",
            "receipt_payload_sha256",
        },
        "decision",
    )
    verify_receipt_payload_hash(value, "decision")
    split = value["split"]
    passed = value["evaluation_pass"]
    if (
        value["format"] != DECISION_FORMAT
        or split not in SPLITS
        or type(passed) is not bool
        or value["status"] != ("pass" if passed else "reject")
        or value["selection_eligible"] is not (split == "val")
        or value["authorization"] is not (passed and split == "val")
    ):
        raise RuntimeError("decision protocol mismatch")
    measurement_receipt = _require_exact_keys(
        value["measurement_receipt"],
        {"path", "sha256", "receipt_payload_sha256"},
        "decision measurement receipt",
    )
    threshold_receipt = _require_exact_keys(
        value["threshold_receipt"],
        {"path", "sha256"},
        "decision threshold receipt",
    )
    for label, receipt in (
        ("measurement", measurement_receipt),
        ("threshold", threshold_receipt),
    ):
        if not isinstance(receipt["path"], str) or not Path(receipt["path"]).is_absolute():
            raise RuntimeError(f"decision {label} receipt path must be absolute")
        require_sha256(receipt["sha256"], f"decision {label} receipt SHA-256")
    require_sha256(
        measurement_receipt["receipt_payload_sha256"],
        "decision measurement payload SHA-256",
    )
    results = value["results"]
    if not isinstance(results, list) or len(results) != len(THRESHOLD_PATHS):
        raise RuntimeError("decision result coverage mismatch")
    seen: set[str] = set()
    recomputed_passes = []
    for index, result in enumerate(results):
        result = _require_exact_keys(
            result,
            {"metric", "operator", "threshold", "actual", "passed"},
            f"decision result {index}",
        )
        metric = result["metric"]
        operator = result["operator"]
        if metric not in THRESHOLD_PATHS or metric in seen:
            raise RuntimeError("decision result metric coverage mismatch")
        seen.add(metric)
        if operator not in {"<", "<="} or type(result["passed"]) is not bool:
            raise RuntimeError("decision result operator/pass type mismatch")
        actual = require_finite_number(result["actual"], "decision result actual")
        threshold = require_finite_number(
            result["threshold"],
            "decision result threshold",
        )
        recomputed = actual < threshold if operator == "<" else actual <= threshold
        if result["passed"] is not recomputed:
            raise RuntimeError("decision result pass flag is inconsistent")
        recomputed_passes.append(recomputed)
    if seen != set(THRESHOLD_PATHS) or passed is not all(recomputed_passes):
        raise RuntimeError("decision pass status/result coverage mismatch")
    protocol = _require_exact_keys(
        value["protocol"],
        {
            "candidate_selection",
            "test_decision",
            "base",
            "thresholds",
            "metric_set_sha256",
        },
        "decision protocol",
    )
    if protocol != {
        "candidate_selection": "validation_only",
        "test_decision": "evaluation_only_never_selection_authority",
        "base": BASE_STATUS,
        "thresholds": "external_frozen_file",
        "metric_set_sha256": METRIC_SET_SHA256,
    }:
        raise RuntimeError("decision protocol content mismatch")
    _rehash_candidate_lineage(value["candidate_lineage"])
    return value


def _validate_lock(
    value: Any,
    current_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    value = _require_exact_keys(
        value,
        {
            "format",
            "status",
            "candidate_id",
            "candidate_lineage",
            "selected_from",
            "test_measurements_authorized",
            "test_once_claim_path",
            "selection_policy",
            "receipt_payload_sha256",
        },
        "selection lock",
    )
    verify_receipt_payload_hash(value, "selection lock")
    if (
        value["format"] != LOCK_FORMAT
        or value["status"] != "locked"
        or value["test_measurements_authorized"] != 1
        or not isinstance(value["test_once_claim_path"], str)
        or not Path(value["test_once_claim_path"]).is_absolute()
        or value["selection_policy"] != "validation_only"
        or value["candidate_id"] != current_lineage["candidate_id"]
        or value["candidate_lineage"] != current_lineage
    ):
        raise RuntimeError("selection lock does not bind this candidate")
    selected_from = _require_exact_keys(
        value["selected_from"],
        {"split", "decision"},
        "selection lock selected_from",
    )
    if selected_from["split"] != "val":
        raise RuntimeError("test lock was not selected from validation")
    return value


def claim_test_once(
    claim_path: Path,
    *,
    lock_receipt: Mapping[str, Any],
    lock_file_sha256: str,
    candidate_lineage: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Atomically consume the single test attempt before any GPU model forward."""
    claim = add_receipt_payload_hash(
        {
            "format": TEST_CLAIM_FORMAT,
            "status": "claimed",
            "selection_lock_sha256": require_sha256(
                lock_file_sha256,
                "selection lock file SHA-256",
            ),
            "selection_lock_receipt_sha256": require_sha256(
                lock_receipt["receipt_payload_sha256"],
                "selection lock receipt SHA-256",
            ),
            "candidate_id": candidate_lineage["candidate_id"],
            "candidate_lineage_receipt_sha256": candidate_lineage[
                "receipt_payload_sha256"
            ],
            "output_path": str(output_path.resolve()),
            "attempt_limit": 1,
        }
    )
    atomic_json_new(claim_path, claim)
    return claim


def _load_models(
    torch: Any,
    lineage: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    from models.motion_representation import VAEConvZero
    from models.rvq import RVQVAE

    models: dict[str, Any] = {}
    for name in EVALUATED_WEIGHT_NAMES:
        spec = ARCHITECTURES[name]
        weight = lineage["weights"][name]
        _, payload = read_verified_artifact(
            Path(weight["path"]),
            weight["sha256"],
            f"candidate {name} weight",
        )
        namespace = SimpleNamespace(
            vae_test_dim=spec["dimension"],
            vae_layer=spec["vae_layer"],
            vae_length=256,
        )
        model = (
            RVQVAE(namespace)
            if spec["model"] == "RVQVAE"
            else VAEConvZero(namespace)
        )
        state = _v1._state_dict_from_payload(torch, payload, f"candidate {name}")
        if set(state) != set(model.state_dict()):
            raise RuntimeError(f"candidate {name} state schema mismatch")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
    return models


def _fresh_error_pairs() -> dict[str, tuple[Any, Any]]:
    return {
        metric: (_v1.ErrorAccumulator(), _v1.ErrorAccumulator())
        for metric in METRIC_PROTOCOLS
    }


def _update_pair(
    pair: tuple[Any, Any],
    model_difference: Any,
    baseline_difference: Any,
) -> None:
    _v1._torch_error_update(pair[0], model_difference)
    _v1._torch_error_update(pair[1], baseline_difference)


def _measurement_protocol(split: str) -> dict[str, Any]:
    return {
        "task_space_version": 2,
        "split": split,
        "candidate_selection": "validation_only",
        "test_policy": "locked_checkpoint_exactly_once",
        "base": BASE_STATUS,
        "evaluated_weights": list(EVALUATED_WEIGHT_NAMES),
        "metric_protocols": METRIC_PROTOCOLS,
        "forbidden_decision_metrics": list(FORBIDDEN_DECISION_METRICS),
        "window_length": _v1.WINDOW_LENGTH,
        "window_stride": _v1.WINDOW_STRIDE,
        "fps": _v1.FPS,
        "rvq_operation": "map2index_then_decode",
        "global_input": "decoded_lower_projected_rotation_zero_translation",
        "integrated_anchor": "ground_truth_frame0_xz",
        "thresholds": None,
    }


def _prepare_test_authorization(
    args: argparse.Namespace,
    lineage: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if args.split == "val":
        if (
            args.selection_lock_json is not None
            or args.expected_selection_lock_sha256 is not None
            or args.test_once_claim is not None
        ):
            raise RuntimeError("validation measurement forbids test lock/claim arguments")
        return None, None
    if (
        args.selection_lock_json is None
        or args.expected_selection_lock_sha256 is None
        or args.test_once_claim is None
    ):
        raise RuntimeError(
            "test measurement requires selection lock, exact SHA, and one-shot claim"
        )
    lock_artifact, lock_payload = _artifact_receipt(
        args.selection_lock_json,
        args.expected_selection_lock_sha256,
        "selection lock",
    )
    lock = strict_json_loads(lock_payload, "selection lock")
    if not isinstance(lock, dict):
        raise TypeError("selection lock must be a JSON object")
    _validate_lock(lock, lineage)
    expected_claim_path = Path(lock["test_once_claim_path"])
    if args.test_once_claim.resolve() != expected_claim_path.resolve():
        raise RuntimeError("test claim path does not match the immutable selection lock")
    if args.output_json.exists() or args.output_json.is_symlink():
        raise FileExistsError("test measurement output already exists")
    claim = claim_test_once(
        args.test_once_claim,
        lock_receipt=lock,
        lock_file_sha256=args.expected_selection_lock_sha256,
        candidate_lineage=lineage,
        output_path=args.output_json,
    )
    return {
        **lock_artifact,
        "receipt_payload_sha256": lock["receipt_payload_sha256"],
    }, {
        "path": str(args.test_once_claim.resolve()),
        "sha256": sha256_file(args.test_once_claim.resolve()),
        "receipt_payload_sha256": claim["receipt_payload_sha256"],
    }


def _measure(args: argparse.Namespace) -> int:
    # Heavy imports remain isolated to the GPU measurement phase.
    import numpy as np
    import torch
    from utils import rotation_conversions as rc

    if args.split not in SPLITS:
        raise RuntimeError("measurement split must be val or test")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    source = _source_receipt(args)
    _, lineage = load_candidate_lineage(
        args.candidate_weights_json,
        args.expected_candidate_weights_sha256,
    )
    test_lock, test_claim = _prepare_test_authorization(args, lineage)
    rows, canonical = _v1.load_canonical_receipt(
        manifest_path=args.canonical_manifest,
        summary_path=args.canonical_summary,
        lineage_path=args.canonical_lineage,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_summary_sha256=args.expected_summary_sha256,
        expected_lineage_sha256=args.expected_lineage_sha256,
        expected_canonical_commit=args.expected_canonical_commit,
        expected_canonical_tree=args.expected_canonical_tree,
    )
    split_rows = sorted(
        (row for row in rows if row["split"] == args.split),
        key=lambda row: row["global_index"],
    )
    if len(split_rows) != EXPECTED_SPLIT_COUNTS[args.split]:
        raise RuntimeError("canonical split clip coverage mismatch")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("task-space measurement requires CUDA")
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError("unsupported deterministic CUBLAS workspace config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.index is not None:
        torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    models = _load_models(torch, lineage, device)
    metric_pairs = _fresh_error_pairs()
    codebooks = {
        name: _v1.CodebookAccumulator()
        for name in STAGE_NAMES
    }
    expected_windows = sum(_v1.window_count(row["frames"]) for row in split_rows)
    seen_windows: set[tuple[int, int]] = set()
    window_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    batches = 0
    replayed_batches = 0

    with torch.inference_mode():
        for records, batch in _v1._batch_iterator(
            np,
            split_rows,
            args.batch_size,
        ):
            batches += 1
            for record in records:
                identity = (record["global_index"], record["start"])
                if identity in seen_windows:
                    raise RuntimeError(f"duplicate {args.split} window {identity}")
                seen_windows.add(identity)
                window_digest.update(canonical_json_bytes(record))
            features = _v1._build_features(torch, rc, batch, device)
            first: dict[str, tuple[Any, Any]] = {}
            for name in STAGE_NAMES:
                if not bool(features[name].isfinite().all().item()):
                    raise RuntimeError(f"{name} input contains NaN/Inf")
                first[name] = _v1._decode_rvq_checked(
                    torch,
                    models[name],
                    features[name],
                    name,
                )
                replay = _v1._decode_rvq_checked(
                    torch,
                    models[name],
                    features[name],
                    name,
                )
                if not torch.equal(first[name][0], replay[0]) or not torch.equal(
                    first[name][1], replay[1]
                ):
                    raise RuntimeError(f"{name} reconstruction is non-deterministic")
                indices = first[name][0]
                index_digest.update(
                    indices.detach().cpu().contiguous().numpy().tobytes()
                )
                codebooks[name].update_rows(
                    indices.detach().cpu().reshape(-1, _v1.RVQ_LEVELS).tolist()
                )

            face = first["face"][1]
            target_jaw = features["face"][..., :6]
            identity_jaw = _v1._identity_rotation6d_like(torch, target_jaw)
            _update_pair(
                metric_pairs["face_jaw_geodesic"],
                _v1._rotation_geodesic_error(
                    torch,
                    rc,
                    target_jaw,
                    face[..., :6],
                ),
                _v1._rotation_geodesic_error(
                    torch,
                    rc,
                    target_jaw,
                    identity_jaw,
                ),
            )
            _update_pair(
                metric_pairs["face_expression"],
                face[..., 6:] - features["face"][..., 6:],
                -features["face"][..., 6:],
            )

            rotation_specs = {
                "hands_rotation_geodesic": ("hands", 180),
                "upper_rotation_geodesic": ("upper", 78),
                "lower_rotation_geodesic": ("lower", 54),
            }
            for metric, (name, width) in rotation_specs.items():
                target = features[name][..., :width].reshape(
                    *features[name].shape[:2],
                    -1,
                    6,
                )
                reconstructed = first[name][1][..., :width].reshape(
                    *features[name].shape[:2],
                    -1,
                    6,
                )
                identity = _v1._identity_rotation6d_like(torch, target)
                _update_pair(
                    metric_pairs[metric],
                    _v1._rotation_geodesic_error(
                        torch,
                        rc,
                        target,
                        reconstructed,
                    ),
                    _v1._rotation_geodesic_error(
                        torch,
                        rc,
                        target,
                        identity,
                    ),
                )

            lower = first["lower"][1]
            _update_pair(
                metric_pairs["lower_contact"],
                lower[..., 57:61] - features["contact"],
                -features["contact"],
            )
            projected_lower = lower.clone()
            projected_lower[..., :54] = rc.matrix_to_rotation_6d(
                rc.rotation_6d_to_matrix(
                    lower[..., :54].reshape(*lower.shape[:2], 9, 6)
                )
            ).reshape(*lower.shape[:2], 54)
            projected_lower[..., 54:57] = 0.0
            if not bool(projected_lower.isfinite().all().item()):
                raise RuntimeError("global input contains NaN/Inf")
            global_first = models["global"](projected_lower).get("rec_pose")
            global_replay = models["global"](projected_lower).get("rec_pose")
            if (
                global_first is None
                or global_replay is None
                or tuple(global_first.shape) != tuple(projected_lower.shape)
                or not bool(global_first.isfinite().all().item())
                or not torch.equal(global_first, global_replay)
            ):
                raise RuntimeError("global reconstruction is invalid/non-deterministic")
            replayed_batches += 1
            target_velocity = _v1._central_velocity(torch, features["translation"])
            target_root = torch.stack(
                [
                    target_velocity[..., 0],
                    features["translation"][..., 1],
                    target_velocity[..., 2],
                ],
                dim=-1,
            )
            zero_root = torch.zeros_like(target_root)
            _update_pair(
                metric_pairs["global_root_channels"],
                global_first[..., 54:57] - target_root,
                zero_root - target_root,
            )
            anchor = features["translation"][:, 0]
            model_translation = _integrate_xz(
                torch,
                global_first[..., 54:57],
                anchor,
            )
            # Critical v2 fix: baseline and model share the exact GT frame-0
            # x/z anchor and the exact same recurrence.
            baseline_translation = _integrate_xz(torch, zero_root, anchor)
            _update_pair(
                metric_pairs["global_integrated_translation"],
                model_translation - features["translation"],
                baseline_translation - features["translation"],
            )

    if len(seen_windows) != expected_windows or expected_windows <= 0:
        raise RuntimeError("split window coverage mismatch")
    if batches <= 0 or replayed_batches != batches:
        raise RuntimeError("determinism replay coverage mismatch")
    metrics = {
        metric: metric_receipt(
            pair[0].finalize(),
            pair[1].finalize(),
            metric,
        )
        for metric, pair in metric_pairs.items()
    }
    input_source = {
        "format": "semtalk_show_input_artifact_source_v1",
        "origin": EXPECTED_ORIGIN,
        "commit": require_git_oid(
            args.expected_input_source_commit,
            "expected input source commit",
        ),
        "tree": require_git_oid(
            args.expected_input_source_tree,
            "expected input source tree",
        ),
    }
    measurement = add_receipt_payload_hash(
        {
            "format": MEASUREMENT_FORMAT,
            "status": "measured",
            "authorization": False,
            "split": args.split,
            "selection_eligible": args.split == "val",
            "finite": True,
            "exact_once": True,
            "deterministic": True,
            "source_receipt": source,
            "implementation_dependency": _dependency_receipt(),
            "candidate_lineage": lineage,
            "canonical_receipt": {
                key: canonical[key]
                for key in (
                    "manifest",
                    "manifest_sha256",
                    "summary",
                    "summary_sha256",
                    "lineage",
                    "lineage_sha256",
                    "lineage_contract_sha256",
                )
            },
            "input_source_receipt": input_source,
            "protocol": _measurement_protocol(args.split),
            "coverage": {
                "split": args.split,
                "clip_count": len(split_rows),
                "expected_clip_count": EXPECTED_SPLIT_COUNTS[args.split],
                "expected_windows": expected_windows,
                "observed_windows": len(seen_windows),
                "window_exact_once": True,
                "window_records_sha256": window_digest.hexdigest(),
            },
            "determinism": {
                "seed": args.seed,
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "tf32": False,
                "cublas_workspace_config": workspace,
                "batches": batches,
                "replayed_batches": replayed_batches,
                "full_batch_exact_replay": True,
                "rvq_indices_sha256": index_digest.hexdigest(),
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "batch_size": args.batch_size,
            },
            "test_lock": test_lock,
            "test_once_claim": test_claim,
            "metrics": metrics,
            "diagnostics": {
                "codebooks_not_decision_rules": {
                    name: codebooks[name].finalize()
                    for name in STAGE_NAMES
                },
                "forbidden_metrics_absent": list(FORBIDDEN_DECISION_METRICS),
                "base": BASE_STATUS,
            },
        }
    )
    validate_measurement(
        measurement,
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
    )
    atomic_json_new(args.output_json, measurement)
    print(
        json.dumps(
            {
                "status": "measured",
                "authorization": False,
                "split": args.split,
                "selection_eligible": args.split == "val",
                "candidate_id": lineage["candidate_id"],
                "output": str(args.output_json.resolve()),
                "receipt_payload_sha256": measurement["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _decide(args: argparse.Namespace) -> int:
    measurement_artifact, measurement_payload = _artifact_receipt(
        args.measurement_json,
        args.expected_measurement_sha256,
        "task-space measurement",
    )
    threshold_artifact, threshold_payload = _artifact_receipt(
        args.thresholds_json,
        args.expected_thresholds_sha256,
        "externally frozen thresholds",
    )
    measurement = strict_json_loads(
        measurement_payload,
        "task-space measurement",
    )
    thresholds = strict_json_loads(
        threshold_payload,
        "externally frozen thresholds",
    )
    if not isinstance(measurement, dict) or not isinstance(thresholds, dict):
        raise TypeError("measurement/thresholds must be JSON objects")
    validate_measurement(
        measurement,
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
    )
    results = evaluate_thresholds(measurement, thresholds)
    passed = all(result["passed"] for result in results)
    split = measurement["split"]
    decision = add_receipt_payload_hash(
        {
            "format": DECISION_FORMAT,
            "status": "pass" if passed else "reject",
            "authorization": passed and split == "val",
            "evaluation_pass": passed,
            "split": split,
            "selection_eligible": split == "val",
            "candidate_lineage": measurement["candidate_lineage"],
            "measurement_receipt": {
                **measurement_artifact,
                "receipt_payload_sha256": measurement[
                    "receipt_payload_sha256"
                ],
            },
            "threshold_receipt": threshold_artifact,
            "results": results,
            "protocol": {
                "candidate_selection": "validation_only",
                "test_decision": "evaluation_only_never_selection_authority",
                "base": BASE_STATUS,
                "thresholds": "external_frozen_file",
                "metric_set_sha256": METRIC_SET_SHA256,
            },
        }
    )
    _validate_decision(decision)
    atomic_json_new(args.output_json, decision)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "authorization": decision["authorization"],
                "evaluation_pass": passed,
                "split": split,
                "output": str(args.output_json.resolve()),
                "receipt_payload_sha256": decision["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


def _lock(args: argparse.Namespace) -> int:
    decision_artifact, payload = _artifact_receipt(
        args.val_decision_json,
        args.expected_val_decision_sha256,
        "validation decision",
    )
    decision = strict_json_loads(payload, "validation decision")
    if not isinstance(decision, dict):
        raise TypeError("validation decision must be a JSON object")
    _validate_decision(decision)
    if (
        decision["split"] != "val"
        or decision["status"] != "pass"
        or decision["authorization"] is not True
        or decision["selection_eligible"] is not True
    ):
        raise RuntimeError("only an accepted validation decision can be locked")
    lineage = decision["candidate_lineage"]
    claim_path = args.test_once_claim_path.resolve()
    if claim_path.exists() or claim_path.is_symlink():
        raise FileExistsError(f"test one-shot claim already exists: {claim_path}")
    if claim_path == args.output_json.resolve():
        raise RuntimeError("selection lock and test one-shot claim must be distinct")
    lock = add_receipt_payload_hash(
        {
            "format": LOCK_FORMAT,
            "status": "locked",
            "candidate_id": lineage["candidate_id"],
            "candidate_lineage": lineage,
            "selected_from": {
                "split": "val",
                "decision": {
                    **decision_artifact,
                    "receipt_payload_sha256": decision[
                        "receipt_payload_sha256"
                    ],
                },
            },
            "test_measurements_authorized": 1,
            "test_once_claim_path": str(claim_path),
            "selection_policy": "validation_only",
        }
    )
    _validate_lock(lock, lineage)
    atomic_json_new(args.output_json, lock)
    print(
        json.dumps(
            {
                "status": "locked",
                "candidate_id": lineage["candidate_id"],
                "test_measurements_authorized": 1,
                "test_once_claim_path": lock["test_once_claim_path"],
                "output": str(args.output_json.resolve()),
                "receipt_payload_sha256": lock["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _inspect_weights(args: argparse.Namespace) -> int:
    _, lineage = load_candidate_lineage(
        args.candidate_weights_json,
        args.expected_candidate_weights_sha256,
    )
    atomic_json_new(args.output_json, lineage)
    print(
        json.dumps(
            {
                "status": "verified",
                "candidate_id": lineage["candidate_id"],
                "base": BASE_STATUS,
                "output": str(args.output_json.resolve()),
                "receipt_payload_sha256": lineage["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-weights-json", type=Path, required=True)
    parser.add_argument("--expected-candidate-weights-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SemTalk canonical SHOW task-space gate v2"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_weights = subparsers.add_parser(
        "inspect-weights",
        help="CPU-only exact weight and lineage validation",
    )
    _add_candidate_arguments(inspect_weights)
    inspect_weights.add_argument("--output-json", type=Path, required=True)
    inspect_weights.set_defaults(handler=_inspect_weights)

    measure = subparsers.add_parser(
        "measure",
        help="GPU measurement for exactly one split; no thresholds/authorization",
    )
    measure.add_argument("--split", choices=SPLITS, required=True)
    measure.add_argument("--canonical-manifest", type=Path, required=True)
    measure.add_argument("--canonical-summary", type=Path, required=True)
    measure.add_argument("--canonical-lineage", type=Path, required=True)
    measure.add_argument("--expected-manifest-sha256", required=True)
    measure.add_argument("--expected-summary-sha256", required=True)
    measure.add_argument("--expected-lineage-sha256", required=True)
    measure.add_argument("--expected-canonical-commit", required=True)
    measure.add_argument("--expected-canonical-tree", required=True)
    _add_candidate_arguments(measure)
    measure.add_argument("--selection-lock-json", type=Path)
    measure.add_argument("--expected-selection-lock-sha256")
    measure.add_argument("--test-once-claim", type=Path)
    measure.add_argument("--device", required=True)
    measure.add_argument("--batch-size", type=int, default=8)
    measure.add_argument("--seed", type=int, default=20260730)
    measure.add_argument("--expected-source-commit", required=True)
    measure.add_argument("--expected-source-tree", required=True)
    measure.add_argument("--expected-input-source-commit", required=True)
    measure.add_argument("--expected-input-source-tree", required=True)
    measure.add_argument("--output-json", type=Path, required=True)
    measure.set_defaults(handler=_measure)

    decide = subparsers.add_parser(
        "decide",
        help="CPU-only decision using an externally frozen threshold file",
    )
    decide.add_argument("--measurement-json", type=Path, required=True)
    decide.add_argument("--expected-measurement-sha256", required=True)
    decide.add_argument("--thresholds-json", type=Path, required=True)
    decide.add_argument("--expected-thresholds-sha256", required=True)
    decide.add_argument("--expected-source-commit", required=True)
    decide.add_argument("--expected-source-tree", required=True)
    decide.add_argument("--output-json", type=Path, required=True)
    decide.set_defaults(handler=_decide)

    lock = subparsers.add_parser(
        "lock",
        help="lock exactly one candidate from an accepted validation decision",
    )
    lock.add_argument("--val-decision-json", type=Path, required=True)
    lock.add_argument("--expected-val-decision-sha256", required=True)
    lock.add_argument("--test-once-claim-path", type=Path, required=True)
    lock.add_argument("--output-json", type=Path, required=True)
    lock.set_defaults(handler=_lock)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
