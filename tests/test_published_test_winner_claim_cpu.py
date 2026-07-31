from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts.show_base import prerequisite_val_contract as PREREQUISITE_CONTRACT
import scripts.show_base as SHOW_BASE_PACKAGE

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "show_base" / "published_test_winner_claim.py"
SPEC = importlib.util.spec_from_file_location(
    "published_test_winner_claim_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
CLAIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLAIM)


def _write_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _write_json(path: Path, value: object) -> dict[str, object]:
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
    return _write_bytes(path, payload)


def _write_receipt(path: Path, value: dict[str, object]) -> dict[str, object]:
    receipt = copy.deepcopy(value)
    receipt.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = CLAIM.canonical_json_sha256(receipt)
    artifact = _write_json(path, receipt)
    artifact["receipt_payload_sha256"] = receipt["receipt_payload_sha256"]
    return artifact


def _write_wave(
    path: Path,
    *,
    boundary: int,
    predecessor: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    receipt: dict[str, object] = {
        "format": "semtalk_show_prerequisite_continuation_wave_v2",
        "status": "authorized",
        "test_visible": False,
        "decision": {
            "path": str((path.parent / f"continue-e{boundary}.json").resolve()),
            "sha256": hashlib.sha256(
                f"continue-file-{boundary}".encode()
            ).hexdigest(),
            "receipt_payload_sha256": hashlib.sha256(
                f"continue-payload-{boundary}".encode()
            ).hexdigest(),
        },
        "trigger_stages": list(CLAIM.STAGES),
        "stages": [
            {
                "stage": stage,
                "boundary_epoch": boundary,
                "target_epoch": boundary + 20,
                "cap_epoch": {
                    "face": 600,
                    "hands": 500,
                    "upper": 500,
                    "lower": 600,
                    "global": 1700,
                }[stage],
                "old_segment": {
                    "predecessor_wave": copy.deepcopy(predecessor)
                },
            }
            for stage in CLAIM.STAGES
        ],
    }
    artifact = _write_receipt(path, receipt)
    receipt["receipt_payload_sha256"] = artifact[
        "receipt_payload_sha256"
    ]
    return artifact, receipt


def _reference(artifact: dict[str, object]) -> dict[str, object]:
    return {
        key: artifact[key]
        for key in ("path", "sha256", "receipt_payload_sha256")
    }


class ClaimFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.output_root = (root / "formal-output").resolve()
        self.adapter_calls = 0
        self.aux = root / "aux"
        self.aux.mkdir()
        self.candidate_index = _write_receipt(
            self.aux / "candidate-index.json",
            {"format": "fixture-candidate-index", "status": "complete"},
        )
        self.measurement_index = _write_receipt(
            self.aux / "measurement-index.json",
            {"format": "fixture-measurement-index", "status": "complete"},
        )
        self.stage_measurements: dict[str, dict[str, object]] = {}
        self.fixed_checkpoints: dict[str, dict[str, object]] = {}
        prerequisite_stages = []
        for index, stage in enumerate(CLAIM.STAGES):
            measurement = _write_receipt(
                self.aux / f"{stage}-measurement.json",
                {
                    "format": "fixture-stage-measurement",
                    "status": "complete",
                    "stage": stage,
                },
            )
            self.stage_measurements[stage] = measurement
            checkpoint = _write_bytes(
                root / f"{stage}-selected.pth",
                f"{stage}-selected-checkpoint".encode("utf-8"),
            )
            self.fixed_checkpoints[stage] = checkpoint
            epoch = CLAIM.PREREQUISITE_CANDIDATE_EPOCHS[index]
            prerequisite_stages.append(
                {
                    "stage": stage,
                    "selection_metric": CLAIM.STAGE_SELECTION_METRICS[stage],
                    "epoch": epoch,
                    "optimizer_updates": (
                        epoch
                        * PREREQUISITE_CONTRACT.updates_per_epoch(stage)
                    ),
                    "candidate_index": index,
                    "selection_score": 1.0 + index,
                    "candidate_checkpoint": checkpoint,
                    "measurement_receipt": _reference(measurement),
                    "coverage": {
                        "split": "val",
                        "test_visible": False,
                        "clips": CLAIM.EXPECTED_VAL_CLIPS,
                        "shards": CLAIM.EXPECTED_SHARDS,
                        "exact_once": True,
                        "all_finite": True,
                    },
                }
            )
        self.prerequisite_payload = {
            "format": CLAIM.PREREQUISITE_SELECTION_FORMAT,
            "status": "selected",
            "target_dataset": "SHOW",
            "target_speaker_scope": CLAIM.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "protocol": {
                "name": "five_independent_show_prerequisite_validation_v2",
                "candidate_epochs": list(
                    CLAIM.PREREQUISITE_CANDIDATE_EPOCHS
                ),
                "candidate_epochs_by_stage": {
                    stage: list(CLAIM.PREREQUISITE_CANDIDATE_EPOCHS)
                    for stage in CLAIM.STAGES
                },
                "candidates_per_stage": {
                    stage: len(CLAIM.PREREQUISITE_CANDIDATE_EPOCHS)
                    for stage in CLAIM.STAGES
                },
                "clips_per_candidate": CLAIM.EXPECTED_VAL_CLIPS,
                "shards_per_candidate": CLAIM.EXPECTED_SHARDS,
                "window_length": 64,
                "window_stride": 20,
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
            "canonical_view": {"fixture": "frozen-val"},
            "producer_sources": {
                "selector": {"origin": CLAIM.EXPECTED_ORIGIN}
            },
            "training_sources": {
                stage: {"origin": CLAIM.EXPECTED_ORIGIN}
                for stage in CLAIM.STAGES
            },
            "config_sha256": "a" * 64,
            "candidate_index_receipt": _reference(self.candidate_index),
            "measurement_index_receipt": _reference(self.measurement_index),
            "stages": prerequisite_stages,
        }
        self.prerequisite_artifact = _write_receipt(
            root / "prerequisite-selection.json",
            self.prerequisite_payload,
        )
        self.continuation_payload = self._continuation_payload("stop")
        self.continuation_artifact = _write_receipt(
            root / "continuation-decision.json",
            self.continuation_payload,
        )
        self.continuation_waves: list[dict[str, object]] = []
        self.real_feature_cache = _write_receipt(
            root / "released2-real-feature-cache.json",
            {
                "format": "semtalk_show_released2_real_feature_cache_v1",
                "status": "complete",
                "feature_count": 1715,
            },
        )
        self.rows = self._candidate_rows()
        self.winner_payload = self._winner_payload()
        self.winner_artifact = _write_receipt(
            root / "winner-selection.json", self.winner_payload
        )
        self.claim_payload = self._claim_payload()
        self.claim_artifact = _write_receipt(
            root / "published-claim.json", self.claim_payload
        )

    def _continuation_payload(self, decision: str) -> dict[str, object]:
        rows = []
        for stage, selected in zip(
            CLAIM.STAGES, self.prerequisite_payload["stages"]
        ):
            rows.append(
                {
                    "stage": stage,
                    "recent_candidate_epochs": [160, 180, 200],
                    "recent_selection_scores": [3.0, 2.0, 1.0],
                    "winner_epoch": selected["epoch"],
                    "latest_epoch": 200,
                    "previous_best_score": 2.0,
                    "latest_score": 1.0,
                    "relative_improvement": 0.5,
                    "latest_is_winner": False,
                    "meets_relative_improvement_threshold": True,
                    "cap_epoch": {
                        "face": 600,
                        "hands": 500,
                        "upper": 500,
                        "lower": 600,
                        "global": 1700,
                    }[stage],
                    "action": "freeze",
                    "target_epoch": None,
                    "frozen_winner_epoch": selected["epoch"],
                    "requests_continuation": decision == "continue",
                }
            )
        return {
            "format": CLAIM.CONTINUATION_DECISION_FORMAT,
            "status": "complete",
            "decision": decision,
            "test_visible": False,
            "protocol": {
                "name": "fresh_replayed_independent_stage_val_improvement_v2",
                "score_direction": "lower_is_better",
                "recent_candidates": 3,
                "relative_improvement_reference": (
                    "best_of_preceding_two_recent_candidates"
                ),
                "minimum_relative_improvement": 0.005,
                "interval_epochs": 20,
                "stage_cap_epochs": {
                    "face": 600,
                    "hands": 500,
                    "upper": 500,
                    "lower": 600,
                    "global": 1700,
                },
                "continue_rule": (
                    "each_stage_latest_boundary_is_global_val_winner_and_"
                    "relative_improvement_gte_threshold_and_below_stage_cap"
                ),
                "terminal_rule": (
                    "otherwise_freeze_global_val_winner;at_cap_mark_capped;"
                    "terminal_stages_never_reenter"
                ),
            },
            "inputs": {
                "selection": _reference(self.prerequisite_artifact),
                "measurement_index": _reference(self.measurement_index),
                "stage_measurements": {
                    stage: _reference(self.stage_measurements[stage])
                    for stage in CLAIM.STAGES
                },
            },
            "stages": rows,
        }

    @staticmethod
    def _variation_policy() -> dict[str, object]:
        return {
            "format": "raw_primitive_with_deterministic_delta_integrity_v1",
            "reported_statistic": "raw_metric_primitive_v1",
            "reported_value_path_template": "body.<protocol>.metrics.Variation",
            "exact_zero_claim": False,
            "delta_integrity_check": (
                "variation_sum_lte_integrity_tolerance_sum_v1"
            ),
            "integrity_tolerance_source": (
                "metric_report_float64_roundoff_bound_v1"
            ),
            "public_value_transform": "identity_no_clamp_no_round_v1",
        }

    def _candidate_rows(self) -> list[dict[str, object]]:
        rows = []
        for index, epoch in enumerate(CLAIM.BASE_CANDIDATE_EPOCHS):
            candidate_root = self.root / f"candidate-{epoch}"
            candidate_root.mkdir()
            checkpoint = _write_bytes(
                candidate_root / "base.pth",
                f"base-checkpoint-{epoch}".encode("utf-8"),
            )
            manifest = _write_bytes(
                candidate_root / "predictions.jsonl",
                (f'{{"epoch":{epoch}}}\n').encode("utf-8"),
            )
            distribution = {
                "format": CLAIM.DISTRIBUTION_FORMAT,
                "payload_hash_algorithm": CLAIM.PAYLOAD_HASH_ALGORITHM,
                "protocol": "deterministic_replication_of_single_prediction_v1",
                "physical_samples_per_clip": 1,
                "independent_samples": False,
                "deterministic_delta_distribution": True,
                "seed_consumed": False,
                "logical_slots": list(range(16)),
                "released2_slots": [0, 1],
                "paper16_slots": list(range(16)),
                "face_slot": 0,
                "prediction_manifest": manifest,
                "variation_policy": self._variation_policy(),
            }
            distribution_artifact = _write_receipt(
                candidate_root / "distribution.json", distribution
            )
            updates = epoch * CLAIM.BASE_UPDATES_PER_EPOCH
            lineage = {
                "format": CLAIM.VAL_LINEAGE_FORMAT,
                "status": "complete",
                "generator": "SemTalk Base Motion Generation",
                "dataset": "SHOW",
                "target_speaker_scope": CLAIM.EXPECTED_SCOPE,
                "split": "val",
                "test_visible": False,
                "epoch": epoch,
                "optimizer_updates": updates,
                "candidate_checkpoint": checkpoint,
                "prediction_manifest": manifest,
                "distribution_receipt": distribution_artifact,
                "prerequisite_selection": self.prerequisite_artifact,
                "continuation_decision": self.continuation_artifact,
            }
            lineage_artifact = _write_receipt(
                candidate_root / "lineage.json", lineage
            )
            fgd = 0.25 if index == 5 else float(10 + index)
            report = {
                "format": CLAIM.METRIC_REPORT_FORMAT,
                "report_payload_hash_algorithm": (
                    CLAIM.METRIC_REPORT_HASH_ALGORITHM
                ),
                "status": "complete",
                "generator": "SemTalk Base-only",
                "dataset": CLAIM.EXPECTED_DATASET,
                "split": "val",
                "selection_protocol": {
                    "primary_metric": CLAIM.PRIMARY_METRIC,
                    "mode": "min",
                    "validation_only_for_selection": True,
                    "test_evaluations": 0,
                },
                "distribution_receipt": {
                    **distribution,
                    "receipt_payload_sha256": distribution_artifact[
                        "receipt_payload_sha256"
                    ],
                },
                "inputs": {
                    "canonical_manifest": {"fixture": True},
                    "prediction_manifest": {**manifest, "rows": 1715},
                    "prediction_lineage": {
                        "path": lineage_artifact["path"],
                        "sha256": lineage_artifact["sha256"],
                        "bytes": lineage_artifact["bytes"],
                        "payload_sha256": lineage_artifact[
                            "receipt_payload_sha256"
                        ],
                    },
                },
                "metric_assets": {"fixture": True},
                "counts": {"clips": 1715, "exact_once": True},
                "body": {
                    "released2": {
                        "metrics": {
                            "FGD": fgd,
                            "Variation": 1e-14,
                            "BC": 0.5,
                        }
                    }
                },
                "face": {"fixture": True},
                "rs": {"status": "N/A/unreleased", "value": None},
                "runtime": {"epoch": epoch},
                "formal_mode": True,
                "test_only_mode": False,
            }
            report["report_payload_sha256"] = CLAIM.canonical_json_sha256(
                report, newline=True
            )
            report_artifact = _write_json(
                candidate_root / "metrics.json", report
            )
            primary_replay = {
                "format": "semtalk_show_released2_primary_fresh_replay_v1",
                "status": "complete",
                "primary_metric_path": CLAIM.PRIMARY_METRIC,
                "primary_metric": fgd,
                "report_payload_sha256": report[
                    "report_payload_sha256"
                ],
                "prediction_manifest": manifest,
                "real_feature_cache": self.real_feature_cache,
                "metric_assets": report["metric_assets"],
                "runtime": {"device": "cuda:0", "epoch": epoch},
            }
            primary_replay_artifact = _write_receipt(
                candidate_root / "released2-primary-replay.json",
                primary_replay,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "optimizer_updates": updates,
                    "candidate_checkpoint": checkpoint,
                    "prediction_manifest": manifest,
                    "inference_lineage": lineage_artifact,
                    "distribution_receipt": distribution_artifact,
                    "talkshow_metric_report": report_artifact,
                    "primary_replay_receipt": primary_replay_artifact,
                    "body_released2_fgd": fgd,
                }
            )
        return rows

    def _winner_payload(self) -> dict[str, object]:
        selected = min(
            self.rows,
            key=lambda row: (
                row["body_released2_fgd"],
                row["epoch"],
                row["optimizer_updates"],
            ),
        )
        return {
            "format": CLAIM.WINNER_SELECTION_FORMAT,
            "payload_hash_algorithm": CLAIM.PAYLOAD_HASH_ALGORITHM,
            "status": "selected",
            "generator": "SemTalk Base Motion Generation",
            "dataset": "SHOW",
            "target_speaker_scope": CLAIM.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "selection_eligible": True,
            "primary_metric": CLAIM.PRIMARY_METRIC,
            "selection_policy": {
                "candidate_epochs": list(CLAIM.BASE_CANDIDATE_EPOCHS),
                "updates_per_epoch": CLAIM.BASE_UPDATES_PER_EPOCH,
                "operator": "min",
                "ordering": [
                    CLAIM.PRIMARY_METRIC,
                    "epoch",
                    "optimizer_updates",
                ],
                "test_feedback_into_selection": False,
            },
            "prerequisite_selection": self.prerequisite_artifact,
            "continuation_decision": self.continuation_artifact,
            "real_feature_cache": self.real_feature_cache,
            "candidates": self.rows,
            "selected": selected,
            "test_policy": CLAIM.TEST_POLICY,
        }

    def _claim_payload(self) -> dict[str, object]:
        return {
            "format": CLAIM.CLAIM_FORMAT,
            "payload_hash_algorithm": CLAIM.PAYLOAD_HASH_ALGORITHM,
            "status": "authorized",
            "generator": "SemTalk Base Motion Generation",
            "dataset": "SHOW",
            "target_speaker_scope": CLAIM.EXPECTED_SCOPE,
            "winner_selection": self.winner_artifact,
            "prerequisite_selection": self.prerequisite_artifact,
            "continuation_decision": self.continuation_artifact,
            "continuation_waves": self.continuation_waves,
            "real_feature_cache": self.real_feature_cache,
            "selected_base_checkpoint": self.winner_payload["selected"][
                "candidate_checkpoint"
            ],
            "fixed_checkpoints": self.fixed_checkpoints,
            "expected_output_root": str(self.output_root),
            "test_policy": CLAIM.TEST_POLICY,
            "test_visible_during_selection": False,
        }

    def fake_validate_report(self, report: dict[str, object], **kwargs: object):
        self.adapter_calls += 1
        self.assert_adapter_call(report, kwargs)
        return {
            "status": "pass",
            "split": "val",
            "clips": CLAIM.EXPECTED_VAL_CLIPS,
            "primary_metric_path": CLAIM.PRIMARY_METRIC,
            "primary_metric": report["body"]["released2"]["metrics"]["FGD"],
            "report_payload_sha256": report["report_payload_sha256"],
        }

    def fake_validate_primary_replay(
        self,
        artifact: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        payload = json.loads(
            Path(artifact["path"]).read_text(encoding="utf-8")
        )
        report = kwargs["expected_report"]
        if (
            kwargs["expected_split"] != "val"
            or kwargs["expected_clip_count"] != CLAIM.EXPECTED_VAL_CLIPS
            or kwargs["expected_selection_protocol"]
            != {
                "primary_metric": CLAIM.PRIMARY_METRIC,
                "mode": "min",
                "validation_only_for_selection": True,
                "test_evaluations": 0,
            }
            or kwargs["expected_prediction_manifest"]
            != payload["prediction_manifest"]
            or kwargs["expected_distribution_receipt"]
            != report["distribution_receipt"]
            or payload["report_payload_sha256"]
            != report["report_payload_sha256"]
        ):
            raise RuntimeError("fresh replay expectation changed")
        return {
            "artifact": artifact,
            "receipt_payload_sha256": artifact[
                "receipt_payload_sha256"
            ],
            "primary_metric_path": CLAIM.PRIMARY_METRIC,
            "primary_metric": payload["primary_metric"],
            "report_payload_sha256": report[
                "report_payload_sha256"
            ],
            "prediction_manifest": payload["prediction_manifest"],
            "real_feature_cache": payload["real_feature_cache"],
            "metric_assets": payload["metric_assets"],
            "runtime": payload["runtime"],
        }

    def adapter(self) -> SimpleNamespace:
        return SimpleNamespace(
            validate_report=self.fake_validate_report,
            validate_released2_primary_replay_receipt=(
                self.fake_validate_primary_replay
            ),
        )

    @staticmethod
    def assert_adapter_call(
        report: dict[str, object], kwargs: dict[str, object]
    ) -> None:
        if kwargs != {
            "expected_split": "val",
            "expected_clip_count": CLAIM.EXPECTED_VAL_CLIPS,
            "expected_prediction_manifest": kwargs[
                "expected_prediction_manifest"
            ],
            "expected_distribution_receipt": kwargs[
                "expected_distribution_receipt"
            ],
            "expected_selection_protocol": {
                "primary_metric": CLAIM.PRIMARY_METRIC,
                "mode": "min",
                "validation_only_for_selection": True,
                "test_evaluations": 0,
            },
        }:
            raise AssertionError("metric adapter call changed")

    def validate(self) -> dict[str, object]:
        adapter = self.adapter()
        with mock.patch.object(
            CLAIM,
            "_fresh_local_module",
            side_effect=lambda name: self.source_module(name, adapter),
        ):
            return CLAIM.validate_published_test_winner_claim(
                self.claim_artifact["path"],
                expected_claim_sha256=self.claim_artifact["sha256"],
                expected_claim_bytes=self.claim_artifact["bytes"],
                expected_claim_payload_sha256=self.claim_artifact[
                    "receipt_payload_sha256"
                ],
                expected_output_root=self.output_root,
                prerequisite_selection=self.prerequisite_artifact,
                continuation_decision=self.continuation_artifact,
                continuation_waves=self.continuation_waves,
            )

    @staticmethod
    def source_module(name: str, adapter: object) -> object:
        if name == "evaluate_talkshow_show_metrics":
            return adapter
        if name == "prerequisite_val_contract":
            return PREREQUISITE_CONTRACT
        raise AssertionError(f"unexpected local module {name}")

    def rewrite_claim(self, mutate) -> None:
        payload = copy.deepcopy(self.claim_payload)
        mutate(payload)
        self.claim_payload = payload
        self.claim_artifact = _write_receipt(
            Path(self.claim_artifact["path"]), payload
        )


class PublishedWinnerClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ClaimFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fresh_replay_derives_base_and_five_fixed_checkpoints(self) -> None:
        result = self.fixture.validate()
        self.assertEqual(
            set(result),
            {
                "claim_artifact",
                "receipt_payload_sha256",
                "winner_selection",
                "selected_base_checkpoint",
                "fixed_checkpoints",
                "continuation_waves",
                "expected_output_root",
                "test_policy",
            },
        )
        self.assertEqual(set(result["claim_artifact"]), CLAIM.ARTIFACT_KEYS)
        self.assertEqual(result["fixed_checkpoints"], self.fixture.fixed_checkpoints)
        self.assertEqual(
            result["selected_base_checkpoint"],
            self.fixture.winner_payload["selected"]["candidate_checkpoint"],
        )
        self.assertEqual(self.fixture.adapter_calls, len(CLAIM.BASE_CANDIDATE_EPOCHS))

    def test_tampered_claim_file_fails(self) -> None:
        Path(self.fixture.claim_artifact["path"]).write_bytes(b"{}\n")
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError, "claim artifact changed"
        ):
            self.fixture.validate()

    def test_wrong_output_root_fails(self) -> None:
        adapter = self.fixture.adapter()
        with mock.patch.object(
            CLAIM,
            "_fresh_local_module",
            side_effect=lambda name: self.fixture.source_module(
                name, adapter
            ),
        ), self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError, "claim identity changed"
        ):
            CLAIM.validate_published_test_winner_claim(
                self.fixture.claim_artifact["path"],
                expected_claim_sha256=self.fixture.claim_artifact["sha256"],
                expected_claim_bytes=self.fixture.claim_artifact["bytes"],
                expected_claim_payload_sha256=self.fixture.claim_artifact[
                    "receipt_payload_sha256"
                ],
                expected_output_root=(
                    Path(self.temporary.name) / "different-output"
                ).resolve(),
                prerequisite_selection=self.fixture.prerequisite_artifact,
                continuation_decision=self.fixture.continuation_artifact,
                continuation_waves=self.fixture.continuation_waves,
            )

    def test_more_than_one_test_evaluation_fails(self) -> None:
        self.fixture.rewrite_claim(
            lambda claim: claim["test_policy"].update(
                {"authorized_evaluations": 2}
            )
        )
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError, "claim identity changed"
        ):
            self.fixture.validate()

    def test_foreign_generator_fails(self) -> None:
        self.fixture.rewrite_claim(
            lambda claim: claim.update({"generator": "Foreign Base"})
        )
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError, "claim identity changed"
        ):
            self.fixture.validate()

    def test_forbidden_dependency_labels_fail(self) -> None:
        labels = ("Speaker2", "SemGate", "sparse", "diff" + "sheg")
        for label in labels:
            with self.subTest(label=label):
                fixture = ClaimFixture(Path(self.temporary.name) / label)
                fixture.rewrite_claim(
                    lambda claim, name=label: claim.update(
                        {
                            "expected_output_root": str(
                                (Path(fixture.root) / name / "output").resolve()
                            )
                        }
                    )
                )
                fixture.output_root = Path(
                    fixture.claim_payload["expected_output_root"]
                )
                with self.assertRaisesRegex(
                    CLAIM.PublishedWinnerClaimError,
                    "forbidden generator dependency|not a SemTalk",
                ):
                    fixture.validate()

    def test_continuation_must_be_stop(self) -> None:
        fixture = ClaimFixture(Path(self.temporary.name) / "continue-case")
        fixture.continuation_payload = fixture._continuation_payload("continue")
        fixture.continuation_artifact = _write_receipt(
            Path(fixture.continuation_artifact["path"]),
            fixture.continuation_payload,
        )
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError, "not a final stop decision"
        ):
            fixture.validate()

    def test_initial_v1_stop_without_continuation_waves_is_accepted(
        self,
    ) -> None:
        fixture = ClaimFixture(Path(self.temporary.name) / "initial-v1-stop")
        fixture.prerequisite_payload["format"] = (
            CLAIM.INITIAL_PREREQUISITE_SELECTION_FORMAT
        )
        fixture.prerequisite_payload["protocol"] = {
            "name": "five_independent_show_prerequisite_validation_v1",
            "candidate_epochs": list(
                CLAIM.PREREQUISITE_CANDIDATE_EPOCHS
            ),
            "candidates_per_stage": len(
                CLAIM.PREREQUISITE_CANDIDATE_EPOCHS
            ),
            "clips_per_candidate": CLAIM.EXPECTED_VAL_CLIPS,
            "shards_per_candidate": CLAIM.EXPECTED_SHARDS,
            "window_length": 64,
            "window_stride": 20,
            "full_base_fgd_used": False,
        }
        fixture.prerequisite_artifact = _write_receipt(
            Path(fixture.prerequisite_artifact["path"]),
            fixture.prerequisite_payload,
        )
        prerequisite_artifact, prerequisite, _fixed = (
            CLAIM._validate_prerequisite_selection(
                fixture.prerequisite_artifact
            )
        )
        fixture.continuation_payload = fixture._continuation_payload("stop")
        fixture.continuation_artifact = _write_receipt(
            Path(fixture.continuation_artifact["path"]),
            fixture.continuation_payload,
        )
        _continuation_artifact, decision = (
            CLAIM._validate_continuation_decision(
                fixture.continuation_artifact,
                prerequisite_artifact=prerequisite_artifact,
                prerequisite_selection=prerequisite,
            )
        )
        self.assertEqual(decision["decision"], "stop")
        self.assertEqual(
            CLAIM._validate_continuation_waves(
                [],
                prerequisite_selection=prerequisite,
            ),
            [],
        )

    def test_initial_v1_cannot_claim_an_appended_schedule(self) -> None:
        fixture = ClaimFixture(
            Path(self.temporary.name) / "initial-v1-appended"
        )
        fixture.prerequisite_payload["format"] = (
            CLAIM.INITIAL_PREREQUISITE_SELECTION_FORMAT
        )
        appended = [*CLAIM.PREREQUISITE_CANDIDATE_EPOCHS, 220]
        fixture.prerequisite_payload["protocol"] = {
            "name": "five_independent_show_prerequisite_validation_v1",
            "candidate_epochs": appended,
            "candidates_per_stage": len(appended),
            "clips_per_candidate": CLAIM.EXPECTED_VAL_CLIPS,
            "shards_per_candidate": CLAIM.EXPECTED_SHARDS,
            "window_length": 64,
            "window_stride": 20,
            "full_base_fgd_used": False,
        }
        fixture.prerequisite_artifact = _write_receipt(
            Path(fixture.prerequisite_artifact["path"]),
            fixture.prerequisite_payload,
        )
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError,
            "initial-v1 schedule must be the mandatory prefix",
        ):
            CLAIM._validate_prerequisite_selection(
                fixture.prerequisite_artifact
            )

    def test_appended_e220_schedule_can_publish_a_fresh_stop(self) -> None:
        fixture = ClaimFixture(Path(self.temporary.name) / "e220-stop")
        fixture.prerequisite_payload["protocol"]["candidate_epochs"].append(
            220
        )
        fixture.prerequisite_payload["protocol"]["candidate_epochs_by_stage"] = {
            stage: [*CLAIM.PREREQUISITE_CANDIDATE_EPOCHS, 220]
            for stage in CLAIM.STAGES
        }
        fixture.prerequisite_payload["protocol"]["candidates_per_stage"] = {
            stage: 11 for stage in CLAIM.STAGES
        }
        fixture.prerequisite_artifact = _write_receipt(
            Path(fixture.prerequisite_artifact["path"]),
            fixture.prerequisite_payload,
        )
        prerequisite_artifact, prerequisite, _fixed = (
            CLAIM._validate_prerequisite_selection(
                fixture.prerequisite_artifact
            )
        )
        fixture.continuation_payload = fixture._continuation_payload("stop")
        for stage in fixture.continuation_payload["stages"]:
            stage["recent_candidate_epochs"] = [180, 200, 220]
            stage["latest_epoch"] = 220
        fixture.continuation_artifact = _write_receipt(
            Path(fixture.continuation_artifact["path"]),
            fixture.continuation_payload,
        )
        continuation_artifact, decision = (
            CLAIM._validate_continuation_decision(
                fixture.continuation_artifact,
                prerequisite_artifact=prerequisite_artifact,
                prerequisite_selection=prerequisite,
            )
        )
        self.assertEqual(
            prerequisite["protocol"]["candidate_epochs"][-1],
            220,
        )
        self.assertEqual(decision["decision"], "stop")
        self.assertEqual(
            continuation_artifact,
            fixture.continuation_artifact,
        )
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError,
            "stage-local boundary",
        ):
            CLAIM._validate_continuation_waves(
                [],
                prerequisite_selection=prerequisite,
            )
        wave, receipt = _write_wave(
            fixture.root / "wave-e200-e220.json",
            boundary=200,
            predecessor=None,
        )
        with mock.patch.object(
            CLAIM,
            "_replay_continuation_wave_file",
            return_value=receipt,
        ):
            self.assertEqual(
                CLAIM._validate_continuation_waves(
                    [wave],
                    prerequisite_selection=prerequisite,
                ),
                [wave],
            )

    def test_self_signed_extension_cannot_forge_candidate_index(self) -> None:
        fixture = ClaimFixture(
            Path(self.temporary.name) / "forged-e220-selection"
        )
        fixture.prerequisite_payload["protocol"]["candidate_epochs"].append(
            220
        )
        fixture.prerequisite_payload["protocol"]["candidate_epochs_by_stage"] = {
            stage: [*CLAIM.PREREQUISITE_CANDIDATE_EPOCHS, 220]
            for stage in CLAIM.STAGES
        }
        fixture.prerequisite_payload["protocol"]["candidates_per_stage"] = {
            stage: 11 for stage in CLAIM.STAGES
        }
        face = fixture.prerequisite_payload["stages"][0]
        face["epoch"] = 220
        face["optimizer_updates"] = (
            220 * CLAIM.PREREQUISITE_UPDATES_PER_EPOCH
        )
        # The forged receipt is rehashed but still claims candidate slot zero.
        fixture.prerequisite_artifact = _write_receipt(
            Path(fixture.prerequisite_artifact["path"]),
            fixture.prerequisite_payload,
        )
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError,
            "face selection changed",
        ):
            CLAIM._validate_prerequisite_selection(
                fixture.prerequisite_artifact
            )

    def test_continuation_wave_intermediate_tamper_is_rejected(self) -> None:
        root = Path(self.temporary.name) / "wave-chain"
        root.mkdir()
        first, first_receipt = _write_wave(
            root / "wave-e200-e220.json",
            boundary=200,
            predecessor=None,
        )
        predecessor = {
            key: first[key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        }
        second, second_receipt = _write_wave(
            root / "wave-e220-e240.json",
            boundary=220,
            predecessor=predecessor,
        )
        receipts = {
            first["path"]: first_receipt,
            second["path"]: second_receipt,
        }
        extended = [*range(20, 201, 20), 220, 240]
        selection = {
            "format": CLAIM.PREREQUISITE_SELECTION_FORMAT,
            "protocol": {
                "name": "five_independent_show_prerequisite_validation_v2",
                "candidate_epochs": extended,
                "candidate_epochs_by_stage": {
                    stage: list(extended)
                    for stage in CLAIM.STAGES
                },
                "candidates_per_stage": {
                    stage: len(extended) for stage in CLAIM.STAGES
                },
                "clips_per_candidate": CLAIM.EXPECTED_VAL_CLIPS,
                "shards_per_candidate": CLAIM.EXPECTED_SHARDS,
                "window_length": 64,
                "window_stride": 20,
                "full_base_fgd_used": False,
            }
        }
        with (
            mock.patch.object(
                CLAIM,
                "_fresh_local_module",
                return_value=PREREQUISITE_CONTRACT,
            ),
            mock.patch.object(
                CLAIM,
                "_replay_continuation_wave_file",
                side_effect=lambda artifact: copy.deepcopy(
                    receipts[artifact["path"]]
                ),
            ),
        ):
            self.assertEqual(
                CLAIM._validate_continuation_waves(
                    [first, second],
                    prerequisite_selection=selection,
                ),
                [first, second],
            )
            receipts[second["path"]]["stages"][2]["old_segment"][
                "predecessor_wave"
            ] = None
            with self.assertRaisesRegex(
                CLAIM.PublishedWinnerClaimError,
                "predecessor chain mismatch",
            ):
                CLAIM._validate_continuation_waves(
                    [first, second],
                    prerequisite_selection=selection,
                )

    def test_metric_adapter_failure_has_no_fallback(self) -> None:
        adapter = SimpleNamespace(
            validate_report=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("primitive replay rejected")
            )
        )
        with mock.patch.object(
            CLAIM,
            "_fresh_local_module",
            side_effect=lambda name: self.fixture.source_module(
                name, adapter
            ),
        ), self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError,
            "neutral TalkSHOW report replay failed",
        ):
            CLAIM.validate_published_test_winner_claim(
                self.fixture.claim_artifact["path"],
                expected_claim_sha256=self.fixture.claim_artifact["sha256"],
                expected_claim_bytes=self.fixture.claim_artifact["bytes"],
                expected_claim_payload_sha256=self.fixture.claim_artifact[
                    "receipt_payload_sha256"
                ],
                expected_output_root=self.fixture.output_root,
                prerequisite_selection=self.fixture.prerequisite_artifact,
                continuation_decision=self.fixture.continuation_artifact,
                continuation_waves=self.fixture.continuation_waves,
            )

    def test_primary_replay_failure_has_no_report_fgd_fallback(self) -> None:
        adapter = self.fixture.adapter()
        adapter.validate_released2_primary_replay_receipt = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("fresh primary replay rejected")
            )
        )
        with mock.patch.object(
            CLAIM,
            "_fresh_local_module",
            side_effect=lambda name: self.fixture.source_module(
                name, adapter
            ),
        ), self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError,
            "released2 primary fresh replay verification failed",
        ):
            CLAIM.validate_published_test_winner_claim(
                self.fixture.claim_artifact["path"],
                expected_claim_sha256=self.fixture.claim_artifact["sha256"],
                expected_claim_bytes=self.fixture.claim_artifact["bytes"],
                expected_claim_payload_sha256=self.fixture.claim_artifact[
                    "receipt_payload_sha256"
                ],
                expected_output_root=self.fixture.output_root,
                prerequisite_selection=self.fixture.prerequisite_artifact,
                continuation_decision=self.fixture.continuation_artifact,
                continuation_waves=self.fixture.continuation_waves,
            )

    def test_external_prerequisite_artifact_swap_fails(self) -> None:
        swapped = dict(self.fixture.prerequisite_artifact)
        swapped["sha256"] = "0" * 64
        adapter = self.fixture.adapter()
        with mock.patch.object(
            CLAIM,
            "_fresh_local_module",
            side_effect=lambda name: self.fixture.source_module(
                name, adapter
            ),
        ), self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError, "artifact changed"
        ):
            CLAIM.validate_published_test_winner_claim(
                self.fixture.claim_artifact["path"],
                expected_claim_sha256=self.fixture.claim_artifact["sha256"],
                expected_claim_bytes=self.fixture.claim_artifact["bytes"],
                expected_claim_payload_sha256=self.fixture.claim_artifact[
                    "receipt_payload_sha256"
                ],
                expected_output_root=self.fixture.output_root,
                prerequisite_selection=swapped,
                continuation_decision=self.fixture.continuation_artifact,
                continuation_waves=self.fixture.continuation_waves,
            )

    def test_claim_real_feature_cache_is_an_independent_pin(self) -> None:
        replacement = _write_receipt(
            Path(self.temporary.name) / "replacement-real-cache.json",
            {
                "format": "semtalk_show_released2_real_feature_cache_v1",
                "status": "complete",
                "feature_count": 1715,
            },
        )
        self.fixture.rewrite_claim(
            lambda claim: claim.update({"real_feature_cache": replacement})
        )
        with self.assertRaisesRegex(
            CLAIM.PublishedWinnerClaimError,
            "winner selection identity changed",
        ):
            self.fixture.validate()

    def test_same_path_cached_schedule_validator_is_ignored(self) -> None:
        expected = (
            Path(CLAIM.__file__).resolve().parent
            / "prerequisite_val_contract.py"
        )
        poisoned = types.SimpleNamespace(
            __file__=str(expected),
            REQUIRED_CANDIDATE_EPOCHS=(999,),
            EXPECTED_UPDATES_PER_EPOCH=1,
            validate_candidate_epochs=lambda _value: (999,),
        )
        full_name = "scripts.show_base.prerequisite_val_contract"
        with (
            mock.patch.dict(sys.modules, {full_name: poisoned}),
            mock.patch.object(
                SHOW_BASE_PACKAGE,
                "prerequisite_val_contract",
                poisoned,
            ),
        ):
            observed = CLAIM._fresh_local_module(
                "prerequisite_val_contract"
            )
        self.assertIsNot(observed, poisoned)
        self.assertEqual(
            observed.REQUIRED_CANDIDATE_EPOCHS,
            tuple(range(20, 201, 20)),
        )


if __name__ == "__main__":
    unittest.main()
