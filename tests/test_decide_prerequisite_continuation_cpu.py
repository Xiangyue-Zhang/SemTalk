from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.show_base import decide_prerequisite_continuation as decision
from scripts.show_base import prerequisite_val_contract as contract
from tests import test_prerequisite_val_selection_cpu as fixture_helpers
from tests import test_selected_prerequisite_bridge_cpu as bridge_fixture


class PrerequisiteContinuationDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bridge_fixture.SelectedPrerequisiteBridgeTests.setUpClass()
        cls.fixture = bridge_fixture.SelectedPrerequisiteBridgeTests
        cls.root = cls.fixture.root
        cls.selection = cls.fixture.selection
        cls.selection_sha = cls.fixture.selection_sha

    @classmethod
    def tearDownClass(cls) -> None:
        bridge_fixture.SelectedPrerequisiteBridgeTests.tearDownClass()

    @staticmethod
    def _candidates(
        scores: list[float],
        *,
        epochs: list[int] | None = None,
    ) -> list[dict[str, object]]:
        epochs = epochs or [160, 180, 200]
        return [
            {
                "epoch": epoch,
                "optimizer_updates": epoch * 497,
                "selection_score": score,
                "candidate_checkpoint": {
                    "sha256": hashlib.sha256(
                        f"checkpoint:{epoch}".encode()
                    ).hexdigest()
                },
            }
            for epoch, score in zip(epochs, scores)
        ]

    @staticmethod
    def _selected(
        epoch: int,
        score: float,
    ) -> dict[str, object]:
        return {"epoch": epoch, "selection_score": score}

    def test_fixed_rule_continues_only_for_improving_latest_winner(
        self,
    ) -> None:
        continued = decision._stage_decision(
            stage="face",
            candidates_value=self._candidates([1.0, 0.9, 0.8]),
            selected_stage=self._selected(200, 0.8),
            expected_candidate_epochs=[160, 180, 200],
        )
        self.assertTrue(continued["latest_is_winner"])
        self.assertTrue(
            continued["meets_relative_improvement_threshold"]
        )
        self.assertTrue(continued["requests_continuation"])
        self.assertAlmostEqual(
            continued["relative_improvement"],
            (0.9 - 0.8) / 0.9,
        )

        plateau = decision._stage_decision(
            stage="face",
            candidates_value=self._candidates([1.0, 0.8, 0.8]),
            selected_stage=self._selected(180, 0.8),
            expected_candidate_epochs=[160, 180, 200],
        )
        self.assertFalse(plateau["latest_is_winner"])
        self.assertFalse(plateau["requests_continuation"])
        self.assertEqual(
            plateau["relative_improvement"],
            0.0,
        )

        below_threshold = decision._stage_decision(
            stage="face",
            candidates_value=self._candidates([1.0, 1.0, 0.996]),
            selected_stage=self._selected(200, 0.996),
            expected_candidate_epochs=[160, 180, 200],
        )
        self.assertTrue(below_threshold["latest_is_winner"])
        self.assertFalse(
            below_threshold["meets_relative_improvement_threshold"]
        )
        self.assertFalse(below_threshold["requests_continuation"])
        self.assertEqual(
            decision.MIN_RELATIVE_IMPROVEMENT,
            0.005,
        )

    def test_boundary_inventory_and_nonfinite_scores_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "at least 3",
        ):
            decision._stage_decision(
                stage="face",
                candidates_value=self._candidates(
                    [1.0, 0.9],
                    epochs=[160, 180],
                ),
                selected_stage=self._selected(180, 0.9),
                expected_candidate_epochs=[160, 180, 200],
            )

    def test_stage_actions_obey_independent_caps_and_freeze_rule(self) -> None:
        continued = decision._stage_decision(
            stage="upper",
            candidates_value=self._candidates(
                [1.0, 0.9, 0.8],
                epochs=[440, 460, 480],
            ),
            selected_stage=self._selected(480, 0.8),
            expected_candidate_epochs=[440, 460, 480],
        )
        self.assertEqual(continued["action"], "continue")
        self.assertEqual(continued["target_epoch"], 500)
        self.assertEqual(continued["cap_epoch"], 500)

        capped = decision._stage_decision(
            stage="upper",
            candidates_value=self._candidates(
                [1.0, 0.9, 0.8],
                epochs=[460, 480, 500],
            ),
            selected_stage=self._selected(500, 0.8),
            expected_candidate_epochs=[460, 480, 500],
        )
        self.assertEqual(capped["action"], "capped")
        self.assertIsNone(capped["target_epoch"])
        self.assertFalse(capped["requests_continuation"])

        frozen = decision._stage_decision(
            stage="global",
            candidates_value=self._candidates(
                [0.8, 0.7, 0.9],
                epochs=[200, 220, 240],
            ),
            selected_stage=self._selected(220, 0.7),
            expected_candidate_epochs=[200, 220, 240],
        )
        self.assertEqual(frozen["action"], "freeze")
        self.assertEqual(frozen["frozen_winner_epoch"], 220)
        self.assertIsNone(frozen["target_epoch"])

        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "exceeds the formal cap",
        ):
            decision._stage_decision(
                stage="hands",
                candidates_value=self._candidates(
                    [1.0, 0.9, 0.8],
                    epochs=[480, 500, 520],
                ),
                selected_stage=self._selected(520, 0.8),
                expected_candidate_epochs=[480, 500, 520],
            )
        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "finite",
        ):
            decision._stage_decision(
                stage="face",
                candidates_value=self._candidates(
                    [1.0, 0.9, float("nan")]
                ),
                selected_stage=self._selected(200, 0.8),
                expected_candidate_epochs=[160, 180, 200],
            )

    def test_real_fresh_replay_publishes_hash_bound_stop_receipt(
        self,
    ) -> None:
        output = self.root / "continuation-stop.json"
        result = decision.decide(
            selection_path=self.selection,
            expected_selection_sha256=self.selection_sha,
            output_json=output,
        )
        self.assertEqual(result["decision"], "stop")
        self.assertFalse(result["test_visible"])
        self.assertEqual(
            result["inputs"]["selection"]["sha256"],
            self.selection_sha,
        )
        self.assertEqual(
            set(result["inputs"]["stage_measurements"]),
            set(contract.STAGES),
        )
        self.assertEqual(
            result["receipt_payload_sha256"],
            hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in result.items()
                        if key != "receipt_payload_sha256"
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(
            hashlib.sha256(output.read_bytes()).hexdigest(),
            fixture_helpers.write_json(
                self.root / "continuation-stop-copy.json",
                result,
            ),
        )
        self.assertEqual(
            decision.replay_decision(
                output,
                hashlib.sha256(output.read_bytes()).hexdigest(),
            ),
            result,
        )

    def _resigned_decision_attack(
        self,
        name: str,
        original: Path,
        mutate: object,
    ) -> tuple[Path, str]:
        payload = json.loads(original.read_text(encoding="utf-8"))
        mutate(payload)
        payload.pop("receipt_payload_sha256", None)
        payload["receipt_payload_sha256"] = (
            decision.selected_contract.canonical_json_sha256(payload)
        )
        path = self.root / f"{name}-continuation-decision.json"
        return path, fixture_helpers.write_json(path, payload)

    def test_replay_rejects_external_self_hash_and_schema_attacks(
        self,
    ) -> None:
        output = self.root / "continuation-replay-attacks.json"
        result = decision.decide(
            selection_path=self.selection,
            expected_selection_sha256=self.selection_sha,
            output_json=output,
        )
        output_sha = hashlib.sha256(output.read_bytes()).hexdigest()

        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "file SHA-256 mismatch",
        ):
            decision.replay_decision(output, "0" * 64)

        bad_self_hash = dict(result)
        bad_self_hash["receipt_payload_sha256"] = "0" * 64
        bad_self_hash_path = self.root / "bad-self-hash-decision.json"
        bad_self_hash_sha = fixture_helpers.write_json(
            bad_self_hash_path,
            bad_self_hash,
        )
        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "payload SHA-256 mismatch",
        ):
            decision.replay_decision(
                bad_self_hash_path,
                bad_self_hash_sha,
            )

        bad_schema = dict(result)
        bad_schema["unexpected"] = True
        bad_schema.pop("receipt_payload_sha256")
        bad_schema["receipt_payload_sha256"] = (
            decision.selected_contract.canonical_json_sha256(bad_schema)
        )
        bad_schema_path = self.root / "bad-schema-decision.json"
        bad_schema_sha = fixture_helpers.write_json(
            bad_schema_path,
            bad_schema,
        )
        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "schema mismatch",
        ):
            decision.replay_decision(bad_schema_path, bad_schema_sha)

        self.assertEqual(
            decision.replay_decision(output, output_sha),
            result,
        )

    def test_replay_rejects_resigned_decision_and_selection_binding_attacks(
        self,
    ) -> None:
        output = self.root / "continuation-replay-source.json"
        result = decision.decide(
            selection_path=self.selection,
            expected_selection_sha256=self.selection_sha,
            output_json=output,
        )

        def change_decision(value: dict[str, object]) -> None:
            value["decision"] = (
                "continue" if result["decision"] == "stop" else "stop"
            )

        forged, forged_sha = self._resigned_decision_attack(
            "resigned-top-level",
            output,
            change_decision,
        )
        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "differs from fresh replay",
        ):
            decision.replay_decision(forged, forged_sha)

        def change_selection_sha(value: dict[str, object]) -> None:
            value["inputs"]["selection"]["sha256"] = "0" * 64

        forged, forged_sha = self._resigned_decision_attack(
            "resigned-selection-binding",
            output,
            change_selection_sha,
        )
        with self.assertRaisesRegex(
            decision.ContinuationDecisionError,
            "fresh prerequisite replay failed",
        ):
            decision.replay_decision(forged, forged_sha)

    def _resigned_measurement_attack(
        self,
        name: str,
        mutate: object,
        *,
        allow_nan: bool = False,
    ) -> tuple[Path, str]:
        selection = json.loads(
            self.selection.read_text(encoding="utf-8")
        )
        index_path = Path(
            selection["measurement_index_receipt"]["path"]
        )
        measurement_index = json.loads(
            index_path.read_text(encoding="utf-8")
        )
        face_receipt = measurement_index["stages"]["face"]
        face = json.loads(
            Path(face_receipt["path"]).read_text(encoding="utf-8")
        )
        mutate(face)
        face_path = self.root / f"{name}-face.json"
        if allow_nan:
            face["receipt_payload_sha256"] = "0" * 64
            face_path.write_text(
                json.dumps(
                    face,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=True,
                )
                + "\n",
                encoding="utf-8",
            )
            face_sha = hashlib.sha256(face_path.read_bytes()).hexdigest()
        else:
            face.pop("receipt_payload_sha256", None)
            face = contract.receipt_payload(face)
            face_sha = fixture_helpers.write_json(face_path, face)

        measurement_index["stages"]["face"] = {
            "stage": "face",
            "path": str(face_path),
            "sha256": face_sha,
            "receipt_payload_sha256": face[
                "receipt_payload_sha256"
            ],
        }
        measurement_index.pop("receipt_payload_sha256", None)
        measurement_index = contract.receipt_payload(measurement_index)
        forged_index = self.root / f"{name}-measurement-index.json"
        forged_index_sha = fixture_helpers.write_json(
            forged_index,
            measurement_index,
        )

        selection["measurement_index_receipt"] = {
            "path": str(forged_index),
            "sha256": forged_index_sha,
            "receipt_payload_sha256": measurement_index[
                "receipt_payload_sha256"
            ],
        }
        face_selection = next(
            stage
            for stage in selection["stages"]
            if stage["stage"] == "face"
        )
        face_selection["measurement_receipt"] = {
            "path": str(face_path),
            "sha256": face_sha,
            "receipt_payload_sha256": face[
                "receipt_payload_sha256"
            ],
        }
        selection.pop("receipt_payload_sha256", None)
        selection = contract.receipt_payload(selection)
        forged_selection = self.root / f"{name}-selection.json"
        forged_selection_sha = fixture_helpers.write_json(
            forged_selection,
            selection,
        )
        return forged_selection, forged_selection_sha

    def test_resigned_plateau_deleted_latest_and_nonfinite_attacks_fail(
        self,
    ) -> None:
        def plateau(value: dict[str, object]) -> None:
            value["candidates"][-1]["selection_score"] = value[
                "candidates"
            ][-2]["selection_score"]

        def delete_latest(value: dict[str, object]) -> None:
            value["candidates"].pop()
            value["coverage"]["candidates"] -= 1
            value["coverage"]["shard_jobs"] -= contract.EXPECTED_SHARDS

        def nonfinite(value: dict[str, object]) -> None:
            value["candidates"][-1]["selection_score"] = float("nan")

        attacks = (
            ("resigned-plateau", plateau, False),
            ("resigned-delete-latest", delete_latest, False),
            ("resigned-nonfinite", nonfinite, True),
        )
        for name, mutate, allow_nan in attacks:
            with self.subTest(name=name):
                forged, forged_sha = self._resigned_measurement_attack(
                    name,
                    mutate,
                    allow_nan=allow_nan,
                )
                output = self.root / f"{name}-decision.json"
                with self.assertRaises(
                    decision.ContinuationDecisionError
                ):
                    decision.decide(
                        selection_path=forged,
                        expected_selection_sha256=forged_sha,
                        output_json=output,
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
