from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/show_base/gate_task_space_on_show_v2.py"
SPEC = importlib.util.spec_from_file_location("show_task_space_gate_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def error_summary(value: float, count: int = 12) -> dict[str, object]:
    return {
        "count": count,
        "mae": value,
        "rmse": value,
        "max_abs": value,
    }


def complete_metrics(
    model_value: float = 0.25,
    baseline_value: float = 1.0,
) -> dict[str, object]:
    return {
        metric: gate.metric_receipt(
            error_summary(model_value),
            error_summary(baseline_value),
            metric,
        )
        for metric in gate.METRIC_PROTOCOLS
    }


def complete_thresholds(
    split: str,
    value: float = 0.5,
) -> dict[str, object]:
    return {
        "format": gate.THRESHOLD_FORMAT,
        "split": split,
        "metric_set_sha256": gate.METRIC_SET_SHA256,
        "provenance": {
            "frozen_by": "unit-test",
            "frozen_at_utc": "2026-07-30T00:00:00Z",
            "rationale": "predeclared task-space thresholds",
        },
        "rules": [
            {"metric": metric, "operator": "<", "value": value}
            for metric in gate.THRESHOLD_PATHS
        ],
    }


class StrictJsonTest(unittest.TestCase):
    def test_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            gate.strict_json_loads(b'{"x":1,"x":2}', "duplicate")
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "strict JSON"):
                    gate.strict_json_loads(b'{"x":' + token + b"}", "nonfinite")

    def test_recursive_finite_check_rejects_non_json_and_nan(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            gate.require_finite_tree({"nested": [float("nan")]})
        with self.assertRaisesRegex(TypeError, "unsupported JSON type"):
            gate.require_finite_tree({"bad": {1, 2}})


class CandidateLineageTest(unittest.TestCase):
    def _write_candidate(
        self,
        root: Path,
        *,
        kind: str = "show_adaptation",
    ) -> tuple[Path, str, dict[str, bytes]]:
        payloads: dict[str, bytes] = {}
        weights = {}
        for index, name in enumerate(gate.WEIGHT_NAMES):
            payload = f"independent-test-weight-{index}-{name}".encode()
            payloads[name] = payload
            path = root / f"{name}.bin"
            path.write_bytes(payload)
            weights[name] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        training = root / "training-receipt.json"
        training.write_bytes(b'{"status":"complete"}\n')
        manifest = {
            "format": gate.WEIGHTS_FORMAT,
            "candidate_id": "show-adapt-candidate-001",
            "weights": weights,
            "provenance": {
                "kind": kind,
                "training_receipt": (
                    {
                        "path": str(training.resolve()),
                        "sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                    }
                    if kind == "show_adaptation"
                    else None
                ),
                "parent_receipt": None,
                "notes": "arbitrary externally supplied exact weights",
            },
        }
        path = root / "candidate.json"
        path.write_bytes(gate.canonical_json_bytes(manifest))
        return path, hashlib.sha256(path.read_bytes()).hexdigest(), payloads

    def test_adapted_weights_are_external_exact_and_lineage_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, digest, payloads = self._write_candidate(root)
            manifest, lineage = gate.load_candidate_lineage(path, digest)
            self.assertEqual(manifest["candidate_id"], lineage["candidate_id"])
            self.assertEqual(set(lineage["weights"]), set(gate.WEIGHT_NAMES))
            self.assertEqual(lineage["bound_not_evaluated"], ["base"])
            self.assertEqual(
                lineage["weights"]["base"]["role"],
                gate.BASE_STATUS,
            )
            self.assertEqual(
                lineage["evaluated_weights"],
                list(gate.EVALUATED_WEIGHT_NAMES),
            )
            self.assertEqual(
                lineage["weights"]["face"]["sha256"],
                hashlib.sha256(payloads["face"]).hexdigest(),
            )
            gate.verify_receipt_payload_hash(lineage, "candidate lineage")

    def test_weight_bytes_are_rehashed_and_adaptation_requires_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, digest, _ = self._write_candidate(root)
            (root / "face.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                gate.load_candidate_lineage(path, digest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self._write_candidate(root)
            manifest = json.loads(path.read_text())
            manifest["provenance"]["training_receipt"] = None
            path.write_bytes(gate.canonical_json_bytes(manifest))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "requires a training receipt"):
                gate.load_candidate_lineage(path, digest)

    def test_candidate_manifest_requires_all_six_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self._write_candidate(root)
            manifest = json.loads(path.read_text())
            del manifest["weights"]["base"]
            path.write_bytes(gate.canonical_json_bytes(manifest))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "exactly six"):
                gate.load_candidate_lineage(path, digest)


class TaskSpaceMetricContractTest(unittest.TestCase):
    def test_exact_metric_set_excludes_invalid_representation_rules(self) -> None:
        expected = {
            "face_jaw_geodesic",
            "face_expression",
            "hands_rotation_geodesic",
            "upper_rotation_geodesic",
            "lower_rotation_geodesic",
            "lower_contact",
            "global_root_channels",
            "global_integrated_translation",
        }
        self.assertEqual(set(gate.METRIC_PROTOCOLS), expected)
        serialized = "\n".join(gate.THRESHOLD_PATHS)
        for forbidden in gate.FORBIDDEN_DECISION_METRICS:
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(len(gate.THRESHOLD_PATHS), 16)

    def test_thresholds_require_exact_external_coverage(self) -> None:
        thresholds = complete_thresholds("val")
        gate.validate_thresholds(thresholds, "val")

        missing = json.loads(json.dumps(thresholds))
        missing["rules"].pop()
        with self.assertRaisesRegex(RuntimeError, "exact task-space metric set"):
            gate.validate_thresholds(missing, "val")

        wrong_split = json.loads(json.dumps(thresholds))
        with self.assertRaisesRegex(RuntimeError, "format/split"):
            gate.validate_thresholds(wrong_split, "test")

        invalid = json.loads(json.dumps(thresholds))
        invalid["rules"][0]["metric"] = "metrics.lower_translation.mae"
        with self.assertRaisesRegex(RuntimeError, "invalid/duplicate"):
            gate.validate_thresholds(invalid, "val")

    def test_threshold_evaluation_is_finite_and_deterministic(self) -> None:
        measurement = {"split": "val", "metrics": complete_metrics()}
        thresholds = complete_thresholds("val")
        results = gate.evaluate_thresholds(measurement, thresholds)
        self.assertEqual(len(results), len(gate.THRESHOLD_PATHS))
        self.assertTrue(all(result["passed"] for result in results))
        self.assertEqual(
            [result["metric"] for result in results],
            list(gate.THRESHOLD_PATHS),
        )

    def test_integrated_zero_velocity_baseline_keeps_gt_frame0_xz_anchor(self) -> None:
        baseline = gate._anchor_zero_velocity_reference(
            frames=4,
            anchor_x=12.5,
            anchor_z=-3.25,
        )
        self.assertEqual(
            baseline,
            [
                [12.5, 0.0, -3.25],
                [12.5, 0.0, -3.25],
                [12.5, 0.0, -3.25],
                [12.5, 0.0, -3.25],
            ],
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "baseline_translation = _integrate_xz(torch, zero_root, anchor)",
            source,
        )
        self.assertIn(
            "model_translation = _integrate_xz(",
            source,
        )

    def test_canonical_coverage_is_recomputed_exactly_per_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = ("train", "val", "val", "test")
            rows = []
            speakers = list(gate._v1.SHOW_SPEAKERS.items())
            for index, split in enumerate(splits):
                speaker, speaker_id = speakers[index]
                rows.append(
                    {
                        "global_index": index,
                        "clip_id": f"clip-{index}",
                        "split": split,
                        "speaker": speaker,
                        "speaker_id": speaker_id,
                        "frames": 90,
                        "canonical_npz": str((root / f"{index}.npz").resolve()),
                        "canonical_npz_sha256": f"{index + 1:x}" * 64,
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_bytes(
                b"".join(gate.canonical_json_bytes(row) for row in rows)
            )
            summary = root / "summary.json"
            summary.write_bytes(b"{}\n")
            lineage = root / "lineage.json"
            lineage.write_bytes(b"{}\n")
            receipt = {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "summary": str(summary.resolve()),
                "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "lineage": str(lineage.resolve()),
                "lineage_sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
                "lineage_contract_sha256": "a" * 64,
            }
            coverage = gate._validate_canonical_coverage(
                receipt,
                "val",
                expected_split_counts={"train": 1, "val": 2, "test": 1},
            )
            self.assertEqual(coverage["clip_count"], 2)
            self.assertEqual(coverage["expected_windows"], 4)
            self.assertEqual(len(coverage["window_records_sha256"]), 64)


class SplitAndOneShotTest(unittest.TestCase):
    def _lineage(self, root: Path, candidate: str = "candidate-a") -> dict[str, object]:
        candidate_root = root / candidate
        candidate_root.mkdir()
        weights = {}
        for name in gate.WEIGHT_NAMES:
            path = candidate_root / f"{name}.bin"
            path.write_bytes(f"{candidate}-{name}".encode())
            weights[name] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        training = candidate_root / "train.json"
        training.write_bytes(b'{"status":"complete"}\n')
        manifest = {
            "format": gate.WEIGHTS_FORMAT,
            "candidate_id": candidate,
            "weights": weights,
            "provenance": {
                "kind": "show_adaptation",
                "training_receipt": {
                    "path": str(training.resolve()),
                    "sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                },
                "parent_receipt": None,
                "notes": "unit test",
            },
        }
        manifest_path = candidate_root / "candidate.json"
        manifest_path.write_bytes(gate.canonical_json_bytes(manifest))
        _, lineage = gate.load_candidate_lineage(
            manifest_path,
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        return lineage

    def _decision(
        self,
        root: Path,
        lineage: dict[str, object],
        split: str,
        passed: bool,
    ) -> dict[str, object]:
        results = []
        for index, metric in enumerate(gate.THRESHOLD_PATHS):
            actual = 0.25 if passed or index else 0.75
            results.append(
                {
                    "metric": metric,
                    "operator": "<",
                    "threshold": 0.5,
                    "actual": actual,
                    "passed": actual < 0.5,
                }
            )
        return gate.add_receipt_payload_hash(
            {
                "format": gate.DECISION_FORMAT,
                "status": "pass" if passed else "reject",
                "authorization": passed and split == "val",
                "evaluation_pass": passed,
                "split": split,
                "selection_eligible": split == "val",
                "candidate_lineage": lineage,
                "measurement_receipt": {
                    "path": str((root / "measurement.json").resolve()),
                    "sha256": "4" * 64,
                    "receipt_payload_sha256": "5" * 64,
                },
                "threshold_receipt": {
                    "path": str((root / "thresholds.json").resolve()),
                    "sha256": "6" * 64,
                },
                "results": results,
                "protocol": {
                    "candidate_selection": "validation_only",
                    "test_decision": "evaluation_only_never_selection_authority",
                    "base": gate.BASE_STATUS,
                    "thresholds": "external_frozen_file",
                    "metric_set_sha256": gate.METRIC_SET_SHA256,
                },
            }
        )

    def test_only_passing_validation_decision_can_be_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lineage = self._lineage(root)
            for split, passed in (("test", True), ("val", False)):
                with self.subTest(split=split, passed=passed):
                    decision = self._decision(root, lineage, split, passed)
                    decision_path = root / f"decision-{split}-{passed}.json"
                    decision_path.write_bytes(gate.canonical_json_bytes(decision))
                    output = root / f"lock-{split}-{passed}.json"
                    args = Namespace(
                        val_decision_json=decision_path,
                        expected_val_decision_sha256=hashlib.sha256(
                            decision_path.read_bytes()
                        ).hexdigest(),
                        test_once_claim_path=root / f"claim-{split}-{passed}.json",
                        output_json=output,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "only an accepted validation decision",
                    ):
                        gate._lock(args)

            accepted = self._decision(root, lineage, "val", True)
            decision_path = root / "decision-val-pass.json"
            decision_path.write_bytes(gate.canonical_json_bytes(accepted))
            lock_path = root / "selection-lock.json"
            args = Namespace(
                val_decision_json=decision_path,
                expected_val_decision_sha256=hashlib.sha256(
                    decision_path.read_bytes()
                ).hexdigest(),
                test_once_claim_path=root / "formal-test-once-claim.json",
                output_json=lock_path,
            )
            self.assertEqual(gate._lock(args), 0)
            lock = gate.strict_json_loads(lock_path.read_bytes(), "lock")
            self.assertEqual(lock["selected_from"]["split"], "val")
            self.assertEqual(lock["test_measurements_authorized"], 1)

    def test_test_claim_is_atomic_one_shot_and_candidate_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lineage = self._lineage(root)
            claim_path = root / "test-once-claim.json"
            lock = gate.add_receipt_payload_hash(
                {
                    "format": gate.LOCK_FORMAT,
                    "status": "locked",
                    "candidate_id": lineage["candidate_id"],
                    "candidate_lineage": lineage,
                    "selected_from": {
                        "split": "val",
                        "decision": {
                            "path": str((root / "decision.json").resolve()),
                            "sha256": "7" * 64,
                            "receipt_payload_sha256": "8" * 64,
                        },
                    },
                    "test_measurements_authorized": 1,
                    "test_once_claim_path": str(claim_path.resolve()),
                    "selection_policy": "validation_only",
                }
            )
            gate._validate_lock(lock, lineage)
            claim = gate.claim_test_once(
                claim_path,
                lock_receipt=lock,
                lock_file_sha256="9" * 64,
                candidate_lineage=lineage,
                output_path=root / "test-measurement.json",
            )
            self.assertEqual(claim["attempt_limit"], 1)
            with self.assertRaises(FileExistsError):
                gate.claim_test_once(
                    claim_path,
                    lock_receipt=lock,
                    lock_file_sha256="9" * 64,
                    candidate_lineage=lineage,
                    output_path=root / "test-measurement-2.json",
                )

            other = self._lineage(root, "candidate-b")
            with self.assertRaisesRegex(RuntimeError, "does not bind this candidate"):
                gate._validate_lock(lock, other)

    def test_validation_forbids_lock_and_test_requires_all_authorization_args(self) -> None:
        val_args = Namespace(
            split="val",
            selection_lock_json=Path("/not/allowed"),
            expected_selection_lock_sha256=None,
            test_once_claim=None,
        )
        with self.assertRaisesRegex(RuntimeError, "validation measurement forbids"):
            gate._prepare_test_authorization(val_args, {})
        test_args = Namespace(
            split="test",
            selection_lock_json=None,
            expected_selection_lock_sha256=None,
            test_once_claim=None,
        )
        with self.assertRaisesRegex(RuntimeError, "test measurement requires"):
            gate._prepare_test_authorization(test_args, {})


if __name__ == "__main__":
    unittest.main()
