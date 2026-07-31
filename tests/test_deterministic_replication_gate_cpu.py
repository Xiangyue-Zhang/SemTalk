from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "show_base"
    / "deterministic_replication_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deterministic_replication_gate_under_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _write_json(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, *, payload_sha: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    if payload_sha is not None:
        result["receipt_payload_sha256"] = payload_sha
    return result


def _with_payload_sha(value: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(value)
    result["receipt_payload_sha256"] = GATE.canonical_json_sha256(result)
    return result


def _npz_bytes(offset: float = 0.0) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        betas=np.arange(300, dtype=np.float32) + offset,
        poses=np.arange(5 * 165, dtype=np.float32).reshape(5, 165) + offset,
        expressions=(
            np.arange(5 * 100, dtype=np.float32).reshape(5, 100) + offset
        ),
        trans=np.arange(15, dtype=np.float32).reshape(5, 3) + offset,
        model=np.asarray("smplx"),
        gender=np.asarray("neutral"),
        mocap_frame_rate=np.asarray(30, dtype=np.int64),
    )
    return buffer.getvalue()


class GateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_files: dict[str, Path] = {}
        for stage in GATE.MODEL_STAGES:
            path = root / f"{stage}.bin"
            path.write_bytes(f"{stage}-checkpoint".encode("ascii"))
            self.model_files[stage] = path
        self.model_bundle = self._model_bundle()
        self.source_closure = self._source_closure()
        self.subset_path = root / "gate-subset.json"
        self.subset_rows = self._subset_rows()
        _write_json(self.subset_path, self.subset_rows)
        self.subset_artifact = _artifact(self.subset_path)
        self.seed_paths: list[Path] = []
        self.seed_shas: list[str] = []
        canonical_bytes = [_npz_bytes(float(index)) for index in range(4)]
        for seed in GATE.EXPECTED_SEEDS:
            output = root / f"seed-{seed}"
            output.mkdir()
            prediction_paths = []
            for index, payload in enumerate(canonical_bytes):
                path = output / f"prediction-{index}.npz"
                path.write_bytes(payload)
                prediction_paths.append(path)
            receipt = self._seed_receipt(seed, prediction_paths)
            path = root / f"seed-{seed}.json"
            digest = _write_json(path, receipt)
            self.seed_paths.append(path)
            self.seed_shas.append(digest)

    def _source_closure(self) -> dict[str, object]:
        callables = [
            {
                "qualified_name": name,
                "source_sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            for name in (
                "_infer_clip",
                "_rvq_indices",
                "_decode_checked",
                "_decode_body_axis_angle",
                "_translation_from_channels",
            )
        ]
        return {
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "commit": "1" * 40,
            "tree": "2" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
            "entrypoint": {
                "relative_path": (
                    "scripts/show_base/run_base_val_inference.py"
                ),
                "sha256": "3" * 64,
                "git_blob_sha1": "4" * 40,
            },
            "inference_helper": {
                "relative_path": (
                    "scripts/show_base/run_base_inference.py"
                ),
                "sha256": "5" * 64,
                "git_blob_sha1": "6" * 40,
            },
            "callables": callables,
            "static_randomness_scan": {
                "algorithm": "python_ast_call_symbol_scan_v1",
                "forbidden_symbols": list(GATE.FORBIDDEN_RANDOM_SYMBOLS),
                "matches": [],
                "call_closure_sha256": (
                    GATE.canonical_json_sha256(callables)
                ),
            },
        }

    def _model_bundle(self) -> dict[str, object]:
        checkpoints = {
            stage: _artifact(path)
            for stage, path in self.model_files.items()
        }
        return {
            "checkpoints": checkpoints,
            "all_state_tensors_finite": True,
            "models_eval": True,
            "requires_grad_false": True,
            "bundle_sha256": GATE.canonical_json_sha256(checkpoints),
        }

    def _subset_rows(self) -> list[dict[str, object]]:
        lengths = (64, 88, 121, 88)
        return [
            {
                "gate_position": index,
                "canonical_position": index * 10,
                "global_index": 100 + index,
                "source_clip_id": f"{speaker}/video/clip-{index}",
                "canonical_clip_id": f"{speaker}__video__clip-{index}",
                "speaker": speaker,
                "frames": lengths[index],
                "canonical_row_sha256": str(index + 1) * 64,
                "audio_row_sha256": str(index + 5) * 64,
            }
            for index, speaker in enumerate(GATE.EXPECTED_SPEAKERS)
        ]

    @staticmethod
    def _rng_state(seed: int, position: int) -> dict[str, str]:
        values = {
            field: hashlib.sha256(
                f"{seed}:{position}:{field}".encode()
            ).hexdigest()
            for field in GATE.RNG_FIELDS
        }
        return {
            **values,
            "rng_state_sha256": GATE.canonical_json_sha256(values),
        }

    def _seed_receipt(
        self,
        seed: int,
        predictions: list[Path],
    ) -> dict[str, object]:
        clips = []
        for row, path in zip(self.subset_rows, predictions):
            rng = self._rng_state(seed, int(row["gate_position"]))
            clips.append(
                {
                    "gate_position": row["gate_position"],
                    "canonical_clip_id": row["canonical_clip_id"],
                    "frames": row["frames"],
                    "input_binding": {
                        "canonical_row_sha256": row[
                            "canonical_row_sha256"
                        ],
                        "audio_row_sha256": row["audio_row_sha256"],
                    },
                    "prediction": _artifact(path),
                    "rng": {
                        "before": rng,
                        "after": copy.deepcopy(rng),
                        "seed_consumed": False,
                    },
                }
            )
        return _with_payload_sha(
            {
                "format": GATE.SEED_RUN_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "seed": seed,
                "run_nonce": f"independent-seed-{seed}-0123456789abcdef",
                "process_identity": f"host-a:pid-{1000 + seed}",
                "independent_process": True,
                "subset_manifest": self.subset_artifact,
                "source_closure": self.source_closure,
                "model_bundle": self.model_bundle,
                "runtime": {
                    "torch_inference_mode": True,
                    "deterministic_algorithms": True,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                    "models_eval": True,
                    "requires_grad_false": True,
                    "discrete_decoding": "logits.argmax(dim=2)",
                    "device": "cuda:0",
                },
                "clips": clips,
            }
        )

    def rewrite_seed(
        self,
        index: int,
        mutate: object,
    ) -> None:
        path = self.seed_paths[index]
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        value.pop("receipt_payload_sha256", None)
        value["receipt_payload_sha256"] = (
            GATE.canonical_json_sha256(value)
        )
        self.seed_shas[index] = _write_json(path, value)


class DeterministicReplicationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = GateFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self) -> dict[str, object]:
        return GATE.build_gate(
            seed_run_paths=self.fixture.seed_paths,
            expected_seed_run_sha256=self.fixture.seed_shas,
            scope="validation_candidate_family",
        )

    def test_gate_passes_and_fresh_replay_matches(self) -> None:
        gate = self.build()
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["seeds"], [0, 15])
        self.assertTrue(gate["proof"]["byte_exact"])
        self.assertTrue(gate["proof"]["array_exact"])
        authorization = gate["replication_authorization"]
        self.assertFalse(authorization["independent_samples"])
        self.assertTrue(
            authorization["deterministic_delta_distribution"]
        )
        self.assertFalse(authorization["seed_consumed"])
        self.assertEqual(authorization["logical_slots"], list(range(16)))
        path = self.root / "gate.json"
        digest = _write_json(path, gate)
        artifact, replayed = GATE.load_gate(
            path,
            digest,
            expected_scope="validation_candidate_family",
        )
        self.assertEqual(replayed, gate)
        self.assertEqual(artifact["sha256"], digest)

    def test_distribution_binds_all_slots_to_one_artifact_manifest(self) -> None:
        gate = self.build()
        gate_path = self.root / "gate.json"
        gate_sha = _write_json(gate_path, gate)
        manifest_path = self.root / "predictions.jsonl"
        manifest_path.write_bytes(b'{"clip":"a"}\n')
        manifest_artifact = _artifact(manifest_path)
        records = [
            {
                "canonical_clip_id": f"clip-{index}",
                "prediction_sha256": hashlib.sha256(
                    f"prediction-{index}".encode()
                ).hexdigest(),
                "prediction_bytes": 100 + index,
            }
            for index in range(3)
        ]
        receipt = GATE.build_distribution_receipt(
            gate_path=gate_path,
            expected_gate_sha256=gate_sha,
            prediction_manifest_artifact=manifest_artifact,
            prediction_records=records,
            expected_scope="validation_candidate_family",
        )
        digest = receipt["prediction_artifact_manifest_sha256"]
        self.assertEqual(
            receipt["logical_slot_bindings"],
            [
                {
                    "slot": slot,
                    "prediction_artifact_manifest_sha256": digest,
                }
                for slot in range(16)
            ],
        )
        self.assertEqual(
            GATE.validate_distribution_receipt(
                receipt,
                expected_gate_artifact=receipt["validation_gate"],
                expected_prediction_manifest=manifest_artifact,
                expected_prediction_records=records,
            ),
            receipt,
        )

    def test_rng_consumption_fails_closed(self) -> None:
        def mutate(value: dict[str, object]) -> None:
            rng = value["clips"][0]["rng"]
            rng["seed_consumed"] = True

        self.fixture.rewrite_seed(1, mutate)
        with self.assertRaisesRegex(
            GATE.ReplicationGateError,
            "consumed RNG",
        ):
            self.build()

    def test_seed_outputs_must_be_byte_identical(self) -> None:
        prediction_path = self.root / "seed-15" / "prediction-0.npz"
        prediction_path.write_bytes(_npz_bytes(99.0))

        def mutate(value: dict[str, object]) -> None:
            value["clips"][0]["prediction"] = _artifact(prediction_path)

        self.fixture.rewrite_seed(1, mutate)
        with self.assertRaisesRegex(
            GATE.ReplicationGateError,
            "byte-identical",
        ):
            self.build()

    def test_seed_runs_must_be_independent_processes(self) -> None:
        first = json.loads(
            self.fixture.seed_paths[0].read_text(encoding="utf-8")
        )

        def mutate(value: dict[str, object]) -> None:
            value["run_nonce"] = first["run_nonce"]
            value["process_identity"] = first["process_identity"]

        self.fixture.rewrite_seed(1, mutate)
        with self.assertRaisesRegex(
            GATE.ReplicationGateError,
            "independent processes",
        ):
            self.build()

    def test_static_random_symbol_match_fails_closed(self) -> None:
        def mutate(value: dict[str, object]) -> None:
            value["source_closure"]["static_randomness_scan"][
                "matches"
            ] = ["torch.randn"]

        self.fixture.rewrite_seed(0, mutate)
        self.fixture.rewrite_seed(1, mutate)
        with self.assertRaisesRegex(
            GATE.ReplicationGateError,
            "random ops",
        ):
            self.build()

    def test_gate_requires_four_speakers_and_multiple_lengths(self) -> None:
        rows = copy.deepcopy(self.fixture.subset_rows)
        rows[-1]["speaker"] = "oliver"
        _write_json(self.fixture.subset_path, rows)
        self.fixture.subset_artifact = _artifact(self.fixture.subset_path)

        def mutate(value: dict[str, object]) -> None:
            value["subset_manifest"] = self.fixture.subset_artifact

        self.fixture.rewrite_seed(0, mutate)
        self.fixture.rewrite_seed(1, mutate)
        with self.assertRaisesRegex(
            GATE.ReplicationGateError,
            "gate subset row",
        ):
            self.build()

    def test_model_bundle_change_fails_before_comparison(self) -> None:
        def mutate(value: dict[str, object]) -> None:
            value["model_bundle"]["models_eval"] = False

        self.fixture.rewrite_seed(1, mutate)
        with self.assertRaisesRegex(
            GATE.ReplicationGateError,
            "finite/frozen/eval",
        ):
            self.build()

    def test_distribution_slot_tampering_is_rejected(self) -> None:
        gate = self.build()
        gate_path = self.root / "gate.json"
        gate_sha = _write_json(gate_path, gate)
        manifest_path = self.root / "predictions.jsonl"
        manifest_path.write_bytes(b'{"clip":"a"}\n')
        manifest_artifact = _artifact(manifest_path)
        records = [
            {
                "canonical_clip_id": "clip-a",
                "prediction_sha256": "a" * 64,
                "prediction_bytes": 1,
            }
        ]
        receipt = GATE.build_distribution_receipt(
            gate_path=gate_path,
            expected_gate_sha256=gate_sha,
            prediction_manifest_artifact=manifest_artifact,
            prediction_records=records,
            expected_scope="validation_candidate_family",
        )
        receipt["logical_slot_bindings"][15][
            "prediction_artifact_manifest_sha256"
        ] = "b" * 64
        receipt["receipt_payload_sha256"] = GATE.canonical_json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_payload_sha256"
            }
        )
        with self.assertRaisesRegex(
            GATE.ReplicationGateError,
            "differs from exact",
        ):
            GATE.validate_distribution_receipt(
                receipt,
                expected_gate_artifact=receipt["validation_gate"],
                expected_prediction_manifest=manifest_artifact,
                expected_prediction_records=records,
            )


if __name__ == "__main__":
    unittest.main()
