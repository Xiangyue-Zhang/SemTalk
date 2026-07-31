#!/usr/bin/env python3
"""Select one immutable SHOW validation winner for each prerequisite stage."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import gate_released_all_speakers_on_show as gate
from scripts.show_base import merge_prerequisite_val_shards as merger
from scripts.show_base import prerequisite_val_contract as contract


MEASUREMENT_INDEX_KEYS = {
    "format",
    "status",
    "target_dataset",
    "target_speaker_scope",
    "split",
    "test_visible",
    "protocol",
    "canonical_view",
    "producer_sources",
    "training_sources",
    "config_sha256",
    "candidate_index_receipt",
    "stages",
    "coverage",
    "receipt_payload_sha256",
}
STAGE_MEASUREMENT_KEYS = {
    "format",
    "status",
    "target_dataset",
    "target_speaker_scope",
    "split",
    "test_visible",
    "stage",
    "selection_metric",
    "protocol",
    "canonical_view",
    "producer_sources",
    "training_source",
    "config_sha256",
    "candidate_index_receipt",
    "candidates",
    "coverage",
    "receipt_payload_sha256",
}
CANDIDATE_KEYS = {
    "candidate_index",
    "epoch",
    "optimizer_updates",
    "selection_score",
    "selection_metric",
    "candidate_checkpoint",
    "metrics",
    "codebook_histograms",
    "coverage",
    "shard_receipts",
}


def _load_receipt_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    resolved, payload, observed_sha = contract.read_verified_file(
        path,
        expected_sha256,
        label,
    )
    value = contract.verify_receipt_payload(
        contract.strict_json_bytes(payload, str(resolved)),
        label,
    )
    return resolved, value, observed_sha


def _validate_metric_payload(
    stage: str,
    value: Any,
    expected_score: Any,
) -> None:
    if not isinstance(value, dict) or set(value) != set(
        contract.ACCUMULATOR_KEYS[stage]
    ):
        raise contract.ContractError(f"{stage} metric coverage mismatch")
    raw = {}
    for name in contract.ACCUMULATOR_KEYS[stage]:
        metric = contract.exact_keys(
            value[name],
            (
                "count",
                "sum_abs",
                "sum_squared",
                "max_abs",
                "mae",
                "mse",
                "rmse",
            ),
            f"{stage}.{name}",
        )
        accumulator = contract.validate_accumulator(
            {
                key: metric[key]
                for key in ("count", "sum_abs", "sum_squared", "max_abs")
            },
            f"{stage}.{name}",
        )
        expected = contract.finalize_accumulator(
            accumulator,
            f"{stage}.{name}",
        )
        for key in ("mae", "mse", "rmse"):
            observed = contract.require_finite(
                metric[key],
                f"{stage}.{name}.{key}",
                nonnegative=True,
            )
            if observed != expected[key]:
                raise contract.ContractError(
                    f"{stage}.{name}.{key} was not derived exactly"
                )
        raw[name] = accumulator
    recomputed_metrics, recomputed_score = contract.stage_metrics(stage, raw)
    if recomputed_metrics != value:
        raise contract.ContractError(f"{stage} finalized metrics changed")
    score = contract.require_finite(
        expected_score,
        f"{stage} selection score",
        nonnegative=True,
    )
    if score != recomputed_score:
        raise contract.ContractError(f"{stage} selection score changed")


def _validate_codebook_summary(
    stage: str,
    value: Any,
    *,
    expected_windows: int,
) -> None:
    if stage == "global":
        if value is not None:
            raise contract.ContractError("Global codebook evidence must be null")
        return
    if not isinstance(value, list) or len(value) != contract.RVQ_LEVELS:
        raise contract.ContractError(f"{stage} codebook level mismatch")
    # ``expected_windows`` is already the merged all-shard window count.
    expected_tokens = expected_windows * (contract.WINDOW_LENGTH // 4)
    for level, item in enumerate(value):
        item = contract.exact_keys(
            item,
            (
                "level",
                "tokens",
                "occupied_codes",
                "occupancy_fraction",
                "dead_fraction",
                "entropy_nats",
                "normalized_entropy",
                "histogram",
            ),
            f"{stage} codebook level {level}",
        )
        if item["level"] != level:
            raise contract.ContractError(f"{stage} codebook order changed")
        histogram = item["histogram"]
        if (
            not isinstance(histogram, list)
            or len(histogram) != contract.CODEBOOK_SIZE
        ):
            raise contract.ContractError(f"{stage} histogram schema mismatch")
        counts = [
            contract.require_exact_int(
                count,
                f"{stage} codebook level {level} count",
            )
            for count in histogram
        ]
        if any(count < 0 for count in counts) or sum(counts) != expected_tokens:
            raise contract.ContractError(f"{stage} histogram tokens mismatch")
        tokens = sum(counts)
        occupied = sum(count > 0 for count in counts)
        entropy = 0.0
        for count in counts:
            if count:
                probability = count / tokens
                entropy -= probability * math.log(probability)
        expected = {
            "tokens": tokens,
            "occupied_codes": occupied,
            "occupancy_fraction": occupied / contract.CODEBOOK_SIZE,
            "dead_fraction": 1.0 - occupied / contract.CODEBOOK_SIZE,
            "entropy_nats": entropy,
            "normalized_entropy": entropy / math.log(contract.CODEBOOK_SIZE),
        }
        for key, expected_value in expected.items():
            observed = (
                contract.require_exact_int(item[key], f"{stage}.{key}")
                if key in {"tokens", "occupied_codes"}
                else contract.require_finite(
                    item[key],
                    f"{stage}.{key}",
                    nonnegative=True,
                )
            )
            if observed != expected_value:
                raise contract.ContractError(
                    f"{stage} codebook {key} changed"
                )


def _validate_shard_receipts(
    value: Any,
    *,
    stage: str,
    epoch: int,
) -> None:
    if not isinstance(value, list) or len(value) != contract.EXPECTED_SHARDS:
        raise contract.ContractError("measurement shard receipt coverage mismatch")
    seen = set()
    for expected_index, receipt in enumerate(value):
        receipt = contract.exact_keys(
            receipt,
            (
                "path",
                "sha256",
                "receipt_payload_sha256",
                "shard_index",
                "clips",
                "windows",
            ),
            "measurement shard receipt",
        )
        if receipt["shard_index"] != expected_index:
            raise contract.ContractError("measurement shard order changed")
        path, payload, observed_sha = contract.read_verified_file(
            receipt["path"],
            receipt["sha256"],
            f"{stage} epoch {epoch} shard receipt",
        )
        if observed_sha in seen:
            raise contract.ContractError("measurement shard receipt is reused")
        seen.add(observed_sha)
        shard = contract.verify_receipt_payload(
            contract.strict_json_bytes(payload, str(path)),
            f"{stage} epoch {epoch} shard receipt",
        )
        if (
            shard.get("format") != contract.SHARD_FORMAT
            or shard.get("stage") != stage
            or shard.get("epoch") != epoch
            or shard.get("split") != "val"
            or shard.get("test_visible") is not False
            or shard.get("finite") is not True
            or shard.get("receipt_payload_sha256")
            != receipt["receipt_payload_sha256"]
            or shard.get("shard", {}).get("index") != expected_index
            or shard.get("shard", {}).get("clip_count") != receipt["clips"]
            or shard.get("shard", {}).get("window_count")
            != receipt["windows"]
        ):
            raise contract.ContractError("measurement shard receipt changed")


def _validate_stage_measurement(
    *,
    stage: str,
    payload: Mapping[str, Any],
    measurement_index: Mapping[str, Any],
    candidate_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate_epochs = contract.candidate_epochs(candidate_index)
    if set(payload) != STAGE_MEASUREMENT_KEYS:
        raise contract.ContractError(f"{stage} measurement schema mismatch")
    if (
        payload["format"] != contract.STAGE_MEASUREMENT_FORMAT
        or payload["status"] != "complete"
        or payload["target_dataset"] != "SHOW"
        or payload["target_speaker_scope"] != contract.TARGET_SPEAKER_SCOPE
        or payload["split"] != "val"
        or payload["test_visible"] is not False
        or payload["stage"] != stage
        or payload["selection_metric"] != contract.SELECTION_METRICS[stage]
        or payload["canonical_view"] != measurement_index["canonical_view"]
        or payload["producer_sources"] != measurement_index["producer_sources"]
        or payload["training_source"]
        != candidate_index["source_receipts"][stage]
        or payload["config_sha256"]
        != candidate_index["config_sha256"][stage]
        or payload["candidate_index_receipt"]
        != measurement_index["candidate_index_receipt"]
    ):
        raise contract.ContractError(f"{stage} measurement binding mismatch")
    protocol = contract.exact_keys(
        payload["protocol"],
        (
            "per_stage_independent",
            "candidate_variable_only",
            "candidate_epochs",
            "window_length",
            "window_stride",
            "full_base_fgd_used",
            "test_feedback_into_selection",
        ),
        f"{stage} measurement protocol",
    )
    if protocol != {
        "per_stage_independent": True,
        "candidate_variable_only": True,
        "candidate_epochs": list(candidate_epochs),
        "window_length": contract.WINDOW_LENGTH,
        "window_stride": contract.WINDOW_STRIDE,
        "full_base_fgd_used": False,
        "test_feedback_into_selection": False,
    }:
        raise contract.ContractError(f"{stage} measurement protocol changed")
    coverage = contract.exact_keys(
        payload["coverage"],
        (
            "split",
            "test_visible",
            "clips_per_candidate",
            "candidates",
            "shards_per_candidate",
            "shard_jobs",
            "windows_per_candidate",
            "exact_once_per_candidate",
            "all_finite",
        ),
        f"{stage} measurement coverage",
    )
    expected_windows = contract.require_exact_int(
        coverage["windows_per_candidate"],
        f"{stage} windows per candidate",
    )
    if (
        coverage["split"] != "val"
        or coverage["test_visible"] is not False
        or coverage["clips_per_candidate"] != contract.EXPECTED_VAL_CLIPS
        or coverage["candidates"] != len(candidate_epochs)
        or coverage["shards_per_candidate"] != contract.EXPECTED_SHARDS
        or coverage["shard_jobs"]
        != len(candidate_epochs)
        * contract.EXPECTED_SHARDS
        or expected_windows <= 0
        or coverage["exact_once_per_candidate"] is not True
        or coverage["all_finite"] is not True
    ):
        raise contract.ContractError(f"{stage} measurement coverage changed")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(
        candidate_epochs
    ):
        raise contract.ContractError(f"{stage} candidate count mismatch")
    validated = []
    for expected_index, (expected_epoch, candidate) in enumerate(
        zip(candidate_epochs, candidates)
    ):
        if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_KEYS:
            raise contract.ContractError(f"{stage} candidate schema mismatch")
        expected = contract.candidate_lookup(
            candidate_index,
            stage,
            expected_epoch,
        )
        checkpoint = contract.exact_keys(
            candidate["candidate_checkpoint"],
            ("path", "sha256", "bytes"),
            f"{stage} candidate checkpoint",
        )
        if (
            candidate["candidate_index"] != expected_index
            or candidate["epoch"] != expected_epoch
            or candidate["optimizer_updates"] != expected["optimizer_updates"]
            or candidate["selection_metric"]
            != contract.SELECTION_METRICS[stage]
            or checkpoint
            != {
                "path": expected["checkpoint"],
                "sha256": expected["checkpoint_sha256"],
                "bytes": expected["checkpoint_bytes"],
            }
        ):
            raise contract.ContractError(f"{stage} candidate binding mismatch")
        item_coverage = contract.exact_keys(
            candidate["coverage"],
            (
                "split",
                "test_visible",
                "clips",
                "shards",
                "windows",
                "exact_once",
                "all_finite",
            ),
            f"{stage} candidate coverage",
        )
        if item_coverage != {
            "split": "val",
            "test_visible": False,
            "clips": contract.EXPECTED_VAL_CLIPS,
            "shards": contract.EXPECTED_SHARDS,
            "windows": expected_windows,
            "exact_once": True,
            "all_finite": True,
        }:
            raise contract.ContractError(f"{stage} candidate coverage changed")
        _validate_metric_payload(
            stage,
            candidate["metrics"],
            candidate["selection_score"],
        )
        _validate_codebook_summary(
            stage,
            candidate["codebook_histograms"],
            expected_windows=expected_windows,
        )
        _validate_shard_receipts(
            candidate["shard_receipts"],
            stage=stage,
            epoch=expected_epoch,
        )
        validated.append(dict(candidate))
    return validated


def select(
    *,
    candidate_index_path: Path,
    candidate_index_sha256: str,
    measurement_index_path: Path,
    measurement_index_sha256: str,
    output_json: Path | None,
    selector_source: Mapping[str, Any],
    reprove_selector_source: bool = True,
) -> dict[str, Any]:
    if output_json is not None:
        if not output_json.is_absolute():
            raise contract.ContractError("selection output must be absolute")
        output_json = (
            output_json.parent.resolve(strict=True) / output_json.name
        )
    selector_source = merger._validate_source_receipt(
        selector_source,
        "selector source",
        reprove_local=reprove_selector_source,
    )
    candidate_index, candidate_artifact = contract.load_candidate_index(
        candidate_index_path,
        candidate_index_sha256,
    )
    candidate_epochs = contract.candidate_epochs(candidate_index)
    measurement_path, measurement_index, measurement_sha = (
        _load_receipt_json(
            measurement_index_path,
            measurement_index_sha256,
            "measurement index",
        )
    )
    if set(measurement_index) != MEASUREMENT_INDEX_KEYS:
        raise contract.ContractError("measurement index schema mismatch")
    expected_candidate_receipt = {
        **candidate_artifact,
        "receipt_payload_sha256": candidate_index[
            "receipt_payload_sha256"
        ],
    }
    if (
        measurement_index["format"] != contract.MEASUREMENT_FORMAT
        or measurement_index["status"] != "complete"
        or measurement_index["target_dataset"] != "SHOW"
        or measurement_index["target_speaker_scope"]
        != contract.TARGET_SPEAKER_SCOPE
        or measurement_index["split"] != "val"
        or measurement_index["test_visible"] is not False
        or measurement_index["candidate_index_receipt"]
        != expected_candidate_receipt
        or measurement_index["training_sources"]
        != candidate_index["source_receipts"]
        or measurement_index["config_sha256"]
        != candidate_index["config_sha256"]
    ):
        raise contract.ContractError("measurement index binding mismatch")
    index_protocol = contract.exact_keys(
        measurement_index["protocol"],
        (
            "per_stage_independent",
            "candidate_epochs",
            "candidates_per_stage",
            "shards_per_candidate",
            "full_base_fgd_used",
            "test_feedback_into_selection",
        ),
        "measurement index protocol",
    )
    if index_protocol != {
        "per_stage_independent": True,
        "candidate_epochs": list(candidate_epochs),
        "candidates_per_stage": len(candidate_epochs),
        "shards_per_candidate": contract.EXPECTED_SHARDS,
        "full_base_fgd_used": False,
        "test_feedback_into_selection": False,
    }:
        raise contract.ContractError("measurement index protocol changed")
    canonical = contract.exact_keys(
        measurement_index["canonical_view"],
        (
            "manifest",
            "summary",
            "lineage",
            "clip_count",
            "clip_ids_sha256",
            "split",
            "test_visible",
        ),
        "measurement canonical view",
    )
    for name in ("manifest", "summary", "lineage"):
        contract.exact_keys(
            canonical[name],
            ("path", "sha256"),
            f"measurement canonical {name}",
        )
    canonical_rows, observed_canonical = contract.load_val_canonical(
        manifest_path=Path(canonical["manifest"]["path"]),
        manifest_sha256=canonical["manifest"]["sha256"],
        summary_path=Path(canonical["summary"]["path"]),
        summary_sha256=canonical["summary"]["sha256"],
        lineage_path=Path(canonical["lineage"]["path"]),
        lineage_sha256=canonical["lineage"]["sha256"],
    )
    if observed_canonical != canonical:
        raise contract.ContractError("measurement canonical view changed")
    expected_windows = sum(
        contract.window_count(row["frames"]) for row in canonical_rows
    )
    index_coverage = contract.exact_keys(
        measurement_index["coverage"],
        (
            "stages",
            "candidates_per_stage",
            "shards_per_candidate",
            "total_shard_jobs",
            "clips_per_candidate",
            "windows_per_candidate",
            "exact_once",
            "all_finite",
        ),
        "measurement index coverage",
    )
    if index_coverage != {
        "stages": len(contract.STAGES),
        "candidates_per_stage": len(candidate_epochs),
        "shards_per_candidate": contract.EXPECTED_SHARDS,
        "total_shard_jobs": len(contract.STAGES)
        * len(candidate_epochs)
        * contract.EXPECTED_SHARDS,
        "clips_per_candidate": contract.EXPECTED_VAL_CLIPS,
        "windows_per_candidate": expected_windows,
        "exact_once": True,
        "all_finite": True,
    }:
        raise contract.ContractError("measurement index coverage changed")
    producer_sources = measurement_index["producer_sources"]
    if (
        not isinstance(producer_sources, dict)
        or set(producer_sources) != {"evaluator", "merge"}
    ):
        raise contract.ContractError("measurement producer roles mismatch")
    validated_producers = {
        role: merger._validate_source_receipt(
            source,
            f"{role} source",
            reprove_local=False,
        )
        for role, source in producer_sources.items()
    }
    expected_repository = merger._repository_identity(
        selector_source,
        "selector source",
    )
    for role, source in validated_producers.items():
        if (
            merger._repository_identity(source, f"{role} source")
            != expected_repository
        ):
            raise contract.ContractError(
                f"{role} and selector repository identities differ"
            )
    stage_receipts = measurement_index["stages"]
    if not isinstance(stage_receipts, dict) or set(stage_receipts) != set(
        contract.STAGES
    ):
        raise contract.ContractError("measurement stage coverage mismatch")
    selected_stages = []
    for stage in contract.STAGES:
        receipt = contract.exact_keys(
            stage_receipts[stage],
            ("stage", "path", "sha256", "receipt_payload_sha256"),
            f"{stage} measurement receipt",
        )
        if receipt["stage"] != stage:
            raise contract.ContractError("measurement stage receipt changed")
        stage_path, stage_payload, stage_sha = _load_receipt_json(
            Path(receipt["path"]),
            receipt["sha256"],
            f"{stage} measurement",
        )
        if (
            stage_sha != receipt["sha256"]
            or stage_payload["receipt_payload_sha256"]
            != receipt["receipt_payload_sha256"]
        ):
            raise contract.ContractError(
                f"{stage} measurement receipt binding mismatch"
            )
        candidates = _validate_stage_measurement(
            stage=stage,
            payload=stage_payload,
            measurement_index=measurement_index,
            candidate_index=candidate_index,
        )
        winner = min(
            candidates,
            key=lambda candidate: (
                candidate["selection_score"],
                candidate["epoch"],
                candidate["optimizer_updates"],
                candidate["candidate_checkpoint"]["sha256"],
            ),
        )
        selected_stages.append(
            {
                "stage": stage,
                "selection_metric": contract.SELECTION_METRICS[stage],
                "epoch": winner["epoch"],
                "optimizer_updates": winner["optimizer_updates"],
                "candidate_index": winner["candidate_index"],
                "selection_score": winner["selection_score"],
                "candidate_checkpoint": winner["candidate_checkpoint"],
                "measurement_receipt": {
                    "path": str(stage_path),
                    "sha256": stage_sha,
                    "receipt_payload_sha256": stage_payload[
                        "receipt_payload_sha256"
                    ],
                },
                "coverage": {
                    "split": "val",
                    "test_visible": False,
                    "clips": contract.EXPECTED_VAL_CLIPS,
                    "shards": contract.EXPECTED_SHARDS,
                    "exact_once": True,
                    "all_finite": True,
                },
            }
        )
    measurement_receipt = {
        "path": str(measurement_path),
        "sha256": measurement_sha,
        "receipt_payload_sha256": measurement_index[
            "receipt_payload_sha256"
        ],
    }
    result = contract.receipt_payload(
        {
            "format": contract.SELECTION_FORMAT,
            "status": "selected",
            "target_dataset": "SHOW",
            "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
            "split": "val",
            "test_visible": False,
            "protocol": {
                "name": "five_independent_show_prerequisite_validation_v1",
                "candidate_epochs": list(candidate_epochs),
                "candidates_per_stage": len(candidate_epochs),
                "clips_per_candidate": contract.EXPECTED_VAL_CLIPS,
                "shards_per_candidate": contract.EXPECTED_SHARDS,
                "window_length": contract.WINDOW_LENGTH,
                "window_stride": contract.WINDOW_STRIDE,
                "full_base_fgd_used": False,
            },
            "selection_policy": {
                "per_stage_independent": True,
                "ordering": [
                    "selection_score",
                    "epoch",
                    "optimizer_updates",
                    "checkpoint_sha256",
                ],
                "test_feedback_into_selection": False,
            },
            "canonical_view": measurement_index["canonical_view"],
            "producer_sources": {
                **measurement_index["producer_sources"],
                "selector": selector_source,
            },
            "training_sources": candidate_index["source_receipts"],
            "config_sha256": candidate_index["config_sha256"],
            "candidate_index_receipt": expected_candidate_receipt,
            "measurement_index_receipt": measurement_receipt,
            "stages": selected_stages,
        }
    )
    if set(result) != {
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
    }:
        raise AssertionError("selection bridge top-level schema drift")
    expected_stage_keys = {
        "stage",
        "selection_metric",
        "epoch",
        "optimizer_updates",
        "candidate_index",
        "selection_score",
        "candidate_checkpoint",
        "measurement_receipt",
        "coverage",
    }
    for stage_result in result["stages"]:
        if (
            set(stage_result) != expected_stage_keys
            or set(stage_result["candidate_checkpoint"])
            != {"path", "sha256", "bytes"}
            or set(stage_result["measurement_receipt"])
            != {"path", "sha256", "receipt_payload_sha256"}
            or set(stage_result["coverage"])
            != {
                "split",
                "test_visible",
                "clips",
                "shards",
                "exact_once",
                "all_finite",
            }
        ):
            raise AssertionError("selection bridge nested schema drift")
    contract.validate_selection_receipt(result)
    if output_json is not None:
        contract.atomic_json_new(output_json, result)
    return result


def replay_selection(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
) -> dict[str, Any]:
    """Freshly recompute an externally supplied selected-five receipt.

    Merely replacing ``receipt_payload_sha256`` is not sufficient: replay
    opens and rehashes the frozen candidate index, measurement index, every
    per-stage measurement, all shard receipts, and candidate checkpoints,
    then reruns the exact winner ordering before accepting the external file.
    """

    resolved, payload_bytes, observed_sha = contract.read_verified_file(
        selection_path,
        expected_selection_sha256,
        "selected-five external receipt",
    )
    observed = contract.verify_receipt_payload(
        contract.strict_json_bytes(payload_bytes, str(resolved)),
        "selected-five external receipt",
    )
    contract.validate_selection_receipt(observed)
    candidate_receipt = contract.exact_keys(
        observed["candidate_index_receipt"],
        ("path", "sha256", "receipt_payload_sha256"),
        "selected-five candidate index receipt",
    )
    measurement_receipt = contract.exact_keys(
        observed["measurement_index_receipt"],
        ("path", "sha256", "receipt_payload_sha256"),
        "selected-five measurement index receipt",
    )
    producer_sources = observed.get("producer_sources")
    if (
        not isinstance(producer_sources, dict)
        or set(producer_sources) != {"evaluator", "merge", "selector"}
    ):
        raise contract.ContractError(
            "selected-five producer source coverage mismatch"
        )
    recomputed = select(
        candidate_index_path=Path(candidate_receipt["path"]),
        candidate_index_sha256=candidate_receipt["sha256"],
        measurement_index_path=Path(measurement_receipt["path"]),
        measurement_index_sha256=measurement_receipt["sha256"],
        output_json=None,
        selector_source=producer_sources["selector"],
        reprove_selector_source=False,
    )
    if recomputed != observed:
        raise contract.ContractError(
            "selected-five external receipt does not equal fresh replay"
        )
    if observed_sha != expected_selection_sha256:
        raise AssertionError("selected-five external SHA drift")
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select five SHOW prerequisite validation winners",
        allow_abbrev=False,
    )
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--expected-candidate-index-sha256", required=True)
    parser.add_argument("--measurement-index", type=Path, required=True)
    parser.add_argument("--expected-measurement-index-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selector_source = gate.git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
        script=Path(__file__).resolve(),
    )
    result = select(
        candidate_index_path=args.candidate_index,
        candidate_index_sha256=args.expected_candidate_index_sha256,
        measurement_index_path=args.measurement_index,
        measurement_index_sha256=args.expected_measurement_index_sha256,
        output_json=args.output_json,
        selector_source=selector_source,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "winners": {
                    stage["stage"]: stage["epoch"]
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
