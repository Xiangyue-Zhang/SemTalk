from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "show_base" / "gate_released_all_speakers_on_show.py"
INFERENCE_SCRIPT = ROOT / "scripts" / "show_base" / "run_base_inference.py"
SPEC = importlib.util.spec_from_file_location("released_show_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


SHA = "1" * 64
COMMIT = "2" * 40
TREE = "3" * 40


def error_summary(value: float, *, count: int = 10) -> dict[str, object]:
    return {
        "count": count,
        "mae": value,
        "rmse": value,
        "max_abs": value,
    }


def comparison(
    model_error: dict[str, object],
    protocol: str,
    *,
    baseline_value: float = 1.0,
) -> dict[str, object]:
    return gate.baseline_comparison(
        model_error,
        error_summary(baseline_value, count=int(model_error["count"])),
        protocol=protocol,
    )


def manifest_row(index: int, split: str) -> dict[str, object]:
    speakers = list(gate.SHOW_SPEAKERS.items())
    speaker, speaker_id = speakers[index % len(speakers)]
    return {
        "global_index": index,
        "clip_id": f"clip-{index}",
        "split": split,
        "speaker": speaker,
        "speaker_id": speaker_id,
        "frames": 90,
        "canonical_npz": f"/canonical/{index}.npz",
        "canonical_npz_sha256": SHA,
    }


def measurement_payload(mae: float = 0.25) -> dict[str, object]:
    stages = {}
    for name in ("face", "hands", "upper", "lower"):
        stage: dict[str, object] = {
            "input_finite": True,
            "reconstruction_finite": True,
            "codebook": [
                {
                    "level": level,
                    "dead_fraction": 0.5,
                }
                for level in range(gate.RVQ_LEVELS)
            ],
        }
        stage["baselines"] = {}
        for metric, protocol in gate.STAGE_BASELINE_PROTOCOLS[name].items():
            model_error = error_summary(mae)
            stage[metric] = model_error
            stage["baselines"][metric] = comparison(model_error, protocol)
        stages[name] = stage
    verified_weights = {
        name: {
            "filename": spec["filename"],
            "sha256": spec["sha256"],
            "model": spec["model"],
            "dimension": spec["dimension"],
            "vae_layer": spec["vae_layer"],
            "vae_length": 256,
        }
        for name, spec in gate.WEIGHT_SPECS.items()
    }
    gate_script_receipt = {
        "source_root": str(ROOT),
        "origin": gate.EXPECTED_ORIGIN,
        "commit": COMMIT,
        "tree": TREE,
        "clean": True,
        "script": str(SCRIPT),
        "script_relative": str(SCRIPT.relative_to(ROOT)),
        "script_sha256": gate.sha256_file(SCRIPT),
    }
    inference_script_receipt = {
        **{
            key: value
            for key, value in gate_script_receipt.items()
            if key not in {"script", "script_relative", "script_sha256"}
        },
        "script": str(INFERENCE_SCRIPT),
        "script_relative": str(INFERENCE_SCRIPT.relative_to(ROOT)),
        "script_sha256": gate.sha256_file(INFERENCE_SCRIPT),
    }
    canonical_source = {
        "origin": gate.EXPECTED_ORIGIN,
        "commit": "9" * 40,
        "tree": "a" * 40,
    }
    input_source = {
        "format": "semtalk_show_input_artifact_source_v1",
        "origin": gate.EXPECTED_ORIGIN,
        "commit": "b" * 40,
        "tree": "c" * 40,
    }
    return gate.add_receipt_payload_hash(
        {
            "format": gate.GATE_FORMAT,
            "status": "measured",
            "authorization": False,
            "finite": True,
            "exact_once": True,
            "deterministic": True,
            "source_receipt": gate_script_receipt,
            "release_trust_root": gate.OFFICIAL_RELEASE_TRUST_ROOT,
            "official_weights": gate.OFFICIAL_WEIGHT_RECEIPTS,
            "verified_prerequisite_weights": verified_weights,
            "canonical_receipt": {
                "manifest": "/canonical/manifest.jsonl",
                "manifest_sha256": "6" * 64,
                "summary": "/canonical/summary.json",
                "summary_sha256": "7" * 64,
                "lineage": "/canonical/lineage.json",
                "lineage_sha256": "8" * 64,
                "lineage_contract_sha256": "d" * 64,
            },
            "canonical_source_receipt": canonical_source,
            "source_roles": {
                "current": inference_script_receipt,
                "gate": gate_script_receipt,
                "canonical": canonical_source,
                "input_artifact": input_source,
            },
            "protocol": {
                "splits": list(gate.HELD_OUT_SPLITS),
                "window_length": gate.WINDOW_LENGTH,
                "window_stride": gate.WINDOW_STRIDE,
                "fps": gate.FPS,
                "rvq_operation": "real map2index -> decode",
                "global_input": "decoded lower",
                "trivial_baselines": {
                    "stage": gate.STAGE_BASELINE_PROTOCOLS,
                    "global": gate.GLOBAL_BASELINE_PROTOCOLS,
                    "extra_model_forwards": 0,
                },
                "thresholds": None,
            },
            "coverage": {
                "clip_count": sum(
                    gate.EXPECTED_SPLIT_COUNTS[split]
                    for split in gate.HELD_OUT_SPLITS
                ),
                "split_clip_counts": {
                    split: gate.EXPECTED_SPLIT_COUNTS[split]
                    for split in gate.HELD_OUT_SPLITS
                },
                "expected_windows": 10,
                "observed_windows": 10,
                "window_exact_once": True,
                "window_records_sha256": "4" * 64,
            },
            "determinism": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "tf32": False,
                "cublas_workspace_config": ":4096:8",
                "batches": 2,
                "replayed_batches": 2,
                "full_batch_exact_replay": True,
                "rvq_indices_sha256": "5" * 64,
            },
            "runtime": {
                "python": "test",
                "torch": "test",
                "numpy": "test",
                "device": "cuda:0",
                "batch_size": 1,
            },
            "stages": stages,
            "global": {
                "input_finite": True,
                "output_finite": True,
                "input_semantics": "decoded lower; projected and root-zeroed",
                **{
                    metric: error_summary(mae)
                    for metric in gate.GLOBAL_BASELINE_PROTOCOLS
                },
                "baselines": {
                    metric: comparison(error_summary(mae), protocol)
                    for metric, protocol in (
                        gate.GLOBAL_BASELINE_PROTOCOLS.items()
                    )
                },
            },
        }
    )


class ErrorAccumulatorTest(unittest.TestCase):
    def test_aggregate(self) -> None:
        accumulator = gate.ErrorAccumulator()
        accumulator.update_differences([-1.0, 2.0, -3.0])
        accumulator.update_summary(
            count=1,
            sum_abs=4.0,
            sum_squared=16.0,
            max_abs=4.0,
        )
        result = accumulator.finalize()
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mae"], 2.5)
        self.assertAlmostEqual(result["rmse"], math.sqrt(7.5))
        self.assertEqual(result["max_abs"], 4.0)

    def test_rejects_empty_or_nonfinite(self) -> None:
        with self.assertRaises(RuntimeError):
            gate.ErrorAccumulator().finalize()
        with self.assertRaises(ValueError):
            gate.ErrorAccumulator().update_differences([float("nan")])

    def test_self_consistent_trivial_baseline_ratio(self) -> None:
        model_error = error_summary(0.25)
        baseline_error = error_summary(1.0)
        receipt = gate.baseline_comparison(
            model_error,
            baseline_error,
            protocol="zero",
        )
        self.assertEqual(
            receipt["model_to_baseline_ratio"],
            {"mae": 0.25, "rmse": 0.25},
        )
        gate.validate_baseline_comparison(
            receipt,
            model_error,
            protocol="zero",
            label="test",
        )
        receipt["model_to_baseline_ratio"]["mae"] = 0.24
        with self.assertRaises(RuntimeError):
            gate.validate_baseline_comparison(
                receipt,
                model_error,
                protocol="zero",
                label="test",
            )
        with self.assertRaises(RuntimeError):
            gate.baseline_comparison(
                model_error,
                error_summary(0.0),
                protocol="zero",
            )


class CodebookAccumulatorTest(unittest.TestCase):
    def test_occupancy_entropy_and_dead_fraction(self) -> None:
        accumulator = gate.CodebookAccumulator(levels=2, codebook_size=4)
        accumulator.update_rows([[0, 1], [0, 2], [1, 2], [1, 2]])
        levels = accumulator.finalize()
        self.assertEqual(levels[0]["tokens"], 4)
        self.assertEqual(levels[0]["occupied_codes"], 2)
        self.assertEqual(levels[0]["occupancy_fraction"], 0.5)
        self.assertEqual(levels[0]["dead_fraction"], 0.5)
        self.assertAlmostEqual(levels[0]["entropy_nats"], math.log(2.0))
        self.assertEqual(levels[1]["histogram"], [0, 1, 3, 0])

    def test_rejects_bad_level_or_index(self) -> None:
        accumulator = gate.CodebookAccumulator(levels=2, codebook_size=4)
        with self.assertRaises(ValueError):
            accumulator.update_rows([[0]])
        with self.assertRaises(ValueError):
            accumulator.update_rows([[0, 4]])


class ManifestAndWindowTest(unittest.TestCase):
    def test_small_manifest_exact_once(self) -> None:
        rows = [
            manifest_row(0, "train"),
            manifest_row(1, "val"),
            manifest_row(2, "test"),
        ]
        gate.validate_manifest_rows(
            rows,
            expected_split_counts={"train": 1, "val": 1, "test": 1},
        )
        duplicate = [dict(row) for row in rows]
        duplicate[2]["global_index"] = 1
        with self.assertRaises(RuntimeError):
            gate.validate_manifest_rows(
                duplicate,
                expected_split_counts={"train": 1, "val": 1, "test": 1},
            )

    def test_window_protocol(self) -> None:
        self.assertEqual(gate.window_count(29), 0)
        self.assertEqual(gate.window_count(64), 0)  # usable prefix is only 60
        self.assertEqual(gate.window_count(90), 2)
        self.assertEqual(gate.window_count(120), 3)
        records = list(gate.window_records(manifest_row(8, "test")))
        self.assertEqual(
            [(record["start"], record["end"]) for record in records],
            [(0, 64), (20, 84)],
        )
        self.assertTrue(all(record["global_index"] == 8 for record in records))


class ReceiptAndAtomicWriteTest(unittest.TestCase):
    def test_self_hash_detects_mutation(self) -> None:
        payload = measurement_payload()
        gate.verify_receipt_payload_hash(payload, "measurement")
        payload["status"] = "mutated"
        with self.assertRaises(RuntimeError):
            gate.verify_receipt_payload_hash(payload, "measurement")

    def test_atomic_write_is_new_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            gate.atomic_json_new(output, {"b": 2, "a": 1})
            self.assertEqual(json.loads(output.read_text()), {"a": 1, "b": 2})
            with self.assertRaises(FileExistsError):
                gate.atomic_json_new(output, {"changed": True})

    def test_measurement_requires_distinct_real_entrypoint_roles(self) -> None:
        payload = measurement_payload()
        payload["source_roles"]["current"] = dict(
            payload["source_roles"]["gate"]
        )
        unsigned = dict(payload)
        del unsigned["receipt_payload_sha256"]
        payload["receipt_payload_sha256"] = gate.canonical_payload_sha256(
            unsigned
        )
        with self.assertRaises(RuntimeError):
            gate.validate_measurement_contract(
                payload,
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
            )


class ThresholdTest(unittest.TestCase):
    def thresholds(
        self,
        measurement_sha: str,
        *,
        operator: str = "<=",
        value: float = 0.5,
    ) -> dict[str, object]:
        return {
            "format": gate.THRESHOLD_FORMAT,
            "measurement_format": gate.GATE_FORMAT,
            "measurement_sha256": measurement_sha,
            "provenance": {
                "basis": "frozen empirical acceptance study",
                "authorized_by": "human-reviewer",
            },
            "rules": [
                {
                    "metric": (
                        "stages.face.baselines.full_error."
                        "model_to_baseline_ratio.mae"
                    ),
                    "operator": operator,
                    "value": value,
                }
            ],
        }

    def test_pass_and_reject(self) -> None:
        measurement = measurement_payload(0.25)
        rules = gate.evaluate_thresholds(
            measurement,
            self.thresholds(SHA, value=0.5),
            SHA,
        )
        self.assertTrue(rules[0]["passed"])
        rules = gate.evaluate_thresholds(
            measurement,
            self.thresholds(SHA, value=0.1),
            SHA,
        )
        self.assertFalse(rules[0]["passed"])
        list_thresholds = self.thresholds(SHA, value=0.6)
        list_thresholds["rules"][0]["metric"] = (
            "stages.face.codebook.0.dead_fraction"
        )
        self.assertTrue(
            gate.evaluate_thresholds(measurement, list_thresholds, SHA)[0]["passed"]
        )

    def test_unknown_duplicate_empty_and_nonfinite_fail_closed(self) -> None:
        measurement = measurement_payload()
        thresholds = self.thresholds(SHA)
        thresholds["rules"][0]["metric"] = "stages.face.missing.mae"
        with self.assertRaises(KeyError):
            gate.evaluate_thresholds(measurement, thresholds, SHA)

        thresholds = self.thresholds(SHA)
        thresholds["rules"].append(dict(thresholds["rules"][0]))
        with self.assertRaises(RuntimeError):
            gate.evaluate_thresholds(measurement, thresholds, SHA)

        thresholds = self.thresholds(SHA)
        thresholds["rules"] = []
        with self.assertRaises(RuntimeError):
            gate.evaluate_thresholds(measurement, thresholds, SHA)

        thresholds = self.thresholds(SHA)
        thresholds["rules"][0]["value"] = float("inf")
        with self.assertRaises(ValueError):
            gate.evaluate_thresholds(measurement, thresholds, SHA)

    def test_decide_cli_writes_separate_pass_or_reject_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            measurement = measurement_payload(0.25)
            measurement_path = root / "measurement.json"
            measurement_path.write_bytes(gate.canonical_json_bytes(measurement))
            measurement_sha = gate.sha256_file(measurement_path)

            thresholds = self.thresholds(measurement_sha, value=0.5)
            threshold_path = root / "thresholds.json"
            threshold_path.write_bytes(gate.canonical_json_bytes(thresholds))
            threshold_sha = gate.sha256_file(threshold_path)
            output = root / "decision.json"
            result = gate.main(
                [
                    "decide",
                    "--measurement-json",
                    str(measurement_path),
                    "--expected-measurement-sha256",
                    measurement_sha,
                    "--thresholds-json",
                    str(threshold_path),
                    "--expected-thresholds-sha256",
                    threshold_sha,
                    "--expected-source-commit",
                    COMMIT,
                    "--expected-source-tree",
                    TREE,
                    "--output-json",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            decision = json.loads(output.read_text())
            self.assertEqual(decision["status"], "pass")
            self.assertIs(decision["authorization"], True)
            self.assertEqual(decision["format"], gate.GATE_FORMAT)
            self.assertEqual(
                decision["release_trust_root"],
                gate.OFFICIAL_RELEASE_TRUST_ROOT,
            )
            self.assertEqual(
                decision["official_weights"],
                gate.OFFICIAL_WEIGHT_RECEIPTS,
            )
            self.assertEqual(
                decision["canonical_receipt"],
                measurement["canonical_receipt"],
            )
            self.assertEqual(
                decision["threshold_receipt"]["sha256"],
                threshold_sha,
            )
            self.assertEqual(
                decision["gate_script"],
                {
                    "path": decision["source_roles"]["gate"]["script"],
                    "sha256": (
                        decision["source_roles"]["gate"]["script_sha256"]
                    ),
                },
            )
            self.assertNotEqual(
                decision["source_roles"]["current"]["script"],
                decision["source_roles"]["gate"]["script"],
            )
            self.assertEqual(set(decision), gate.FORMAL_GATE_KEYS)
            unsigned = dict(decision)
            receipt_sha = unsigned.pop("receipt_sha256")
            self.assertEqual(
                receipt_sha,
                gate.canonical_payload_sha256(unsigned),
            )
            self.assertTrue(
                all(decision["decisions"].values())
            )

            rejecting_thresholds = self.thresholds(measurement_sha, value=0.1)
            rejecting_path = root / "rejecting-thresholds.json"
            rejecting_path.write_bytes(
                gate.canonical_json_bytes(rejecting_thresholds)
            )
            rejecting_output = root / "reject.json"
            result = gate.main(
                [
                    "decide",
                    "--measurement-json",
                    str(measurement_path),
                    "--expected-measurement-sha256",
                    measurement_sha,
                    "--thresholds-json",
                    str(rejecting_path),
                    "--expected-thresholds-sha256",
                    gate.sha256_file(rejecting_path),
                    "--expected-source-commit",
                    COMMIT,
                    "--expected-source-tree",
                    TREE,
                    "--output-json",
                    str(rejecting_output),
                ]
            )
            self.assertEqual(result, 2)
            rejection = json.loads(rejecting_output.read_text())
            self.assertEqual(rejection["status"], "reject")
            self.assertIs(rejection["authorization"], False)


class ParserContractTest(unittest.TestCase):
    def test_measure_and_decide_are_disjoint_subcommands(self) -> None:
        parser = gate.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        help_text = parser.format_help()
        self.assertIn("measure", help_text)
        self.assertIn("decide", help_text)
        self.assertNotIn("--threshold", parser._subparsers._group_actions[0].choices["measure"].format_help())


try:
    import torch
    from utils import rotation_conversions as rc
except Exception:  # pragma: no cover - exercised only in a provisioned runtime.
    torch = None
    rc = None


@unittest.skipUnless(torch is not None and rc is not None, "torch CPU runtime unavailable")
class TorchCpuSemanticsTest(unittest.TestCase):
    def test_velocity_and_integration_match_public_recurrence(self) -> None:
        translation = torch.tensor(
            [
                [
                    [1.0, 0.1, 2.0],
                    [2.0, 0.2, 4.0],
                    [3.0, 0.3, 6.0],
                ]
            ]
        )
        velocity = gate._central_velocity(torch, translation)
        self.assertTrue(
            torch.equal(
                velocity[..., (0, 2)],
                torch.tensor([[[30.0, 60.0], [30.0, 60.0], [30.0, 60.0]]]),
            )
        )
        channels = torch.stack(
            [velocity[..., 0], translation[..., 1], velocity[..., 2]],
            dim=-1,
        )
        integrated = gate._integrate_xz(torch, channels, translation[:, 0])
        self.assertTrue(torch.allclose(integrated, translation))

    def test_identical_rotation_has_zero_geodesic(self) -> None:
        identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
        error = gate._rotation_geodesic_error(torch, rc, identity, identity)
        self.assertTrue(torch.equal(error, torch.zeros_like(error)))


if __name__ == "__main__":
    unittest.main()
