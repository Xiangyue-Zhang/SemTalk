from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np


METRICS = importlib.import_module(
    "scripts.show_base.evaluate_talkshow_show_metrics"
)
ROOT = Path(__file__).resolve().parents[1]


def load_external_gate_test_module() -> object:
    gate_source = Path(
        os.environ.get(
            METRICS.REPLICATION_GATE_MODULE_ENV,
            ROOT
            / "scripts"
            / "show_base"
            / "deterministic_replication_gate.py",
        )
    ).resolve()
    source = (
        gate_source.parents[2]
        / "tests"
        / "test_deterministic_replication_gate_cpu.py"
    )
    if not source.is_file():
        raise RuntimeError(
            "deterministic replication gate CPU fixture is required for "
            f"cross-module tests: {source}"
        )
    spec = importlib.util.spec_from_file_location(
        "_external_replication_gate_test_fixture",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import cross-module fixture {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_json_write(path: Path, value: object) -> None:
    path.write_bytes(METRICS.canonical_json_bytes(value))


def canonical_jsonl_write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(METRICS.canonical_json_bytes(row) for row in rows)
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_npz(path: Path, fields: tuple[str, ...], arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **{name: arrays[name] for name in fields})


class SyntheticBackend:
    @property
    def asset_receipt(self) -> dict[str, object]:
        return {
            "format": "semtalk_show_talkshow_metric_assets_v1",
            "status": "pass",
            "execution_device": "cpu",
            "talkshow": {
                "commit": METRICS.TALKSHOW_METRIC_COMMIT,
            },
            "feature_extractor": {
                "sha256": METRICS.FEATURE_EXTRACTOR_SHA256,
            },
            "smplx": {"sha256": METRICS.SMPLX_SHA256},
        }

    @property
    def runtime_receipt(self) -> dict[str, str]:
        return {
            "python": "3.12-test",
            "numpy": np.__version__,
            "torch": "synthetic-test",
            "smplx": "synthetic-test",
            "librosa": "synthetic-test",
            "device": "cpu",
        }

    def extract_body_features(self, parameters_265: np.ndarray) -> np.ndarray:
        array = np.asarray(parameters_265)
        return np.stack(
            (
                array[..., 9].reshape(-1),
                array[..., 12].reshape(-1),
                array[..., 165].reshape(-1),
            ),
            axis=-1,
        )

    def joints(
        self,
        parameters_265: np.ndarray,
        betas_300: np.ndarray,
    ) -> np.ndarray:
        parameters = np.asarray(parameters_265)
        batch, frames, _channels = parameters.shape
        time = np.arange(frames, dtype=np.float64)[None, :, None]
        joint = np.arange(80, dtype=np.float64)[None, None, :]
        result = np.empty((batch, frames, 80, 3), dtype=np.float64)
        result[..., 0] = np.sin((time + 1.0) * (joint + 1.0) * 0.113)
        result[..., 1] = np.cos((time + 2.0) * (joint + 1.0) * 0.071)
        result[..., 2] = np.sin((time + joint + 1.0) * 0.193)
        result[:, :, 22:25, 0] += parameters[:, :, 0, None]
        result[:, :, 74:, 1] += parameters[:, :, 165, None]
        return result


class ShapeRecordingCudaBackend(SyntheticBackend):
    def __init__(self) -> None:
        self.feature_call_shapes: list[tuple[int, ...]] = []
        self.joint_call_shapes: list[tuple[int, ...]] = []

    @property
    def asset_receipt(self) -> dict[str, object]:
        receipt = super().asset_receipt
        receipt["execution_device"] = "cuda:0"
        receipt["feature_extractor"]["runtime_dtype"] = "float32"
        receipt["smplx"]["runtime_dtype"] = "float64"
        return receipt

    @property
    def runtime_receipt(self) -> dict[str, object]:
        return {
            "python": "3.12-test",
            "numpy": np.__version__,
            "torch": "2.test",
            "smplx": "synthetic-test",
            "librosa": "synthetic-test",
            "scipy": "synthetic-test",
            "cuda": "12.test",
            "cudnn": "9.test",
            "device": "cuda:0",
            "device_type": "cuda",
            "device_index": 0,
            "device_name": "Synthetic H200",
        }

    def extract_body_features(self, parameters_265: np.ndarray) -> np.ndarray:
        self.feature_call_shapes.append(tuple(parameters_265.shape))
        return super().extract_body_features(parameters_265)

    def joints(
        self,
        parameters_265: np.ndarray,
        betas_300: np.ndarray,
    ) -> np.ndarray:
        self.joint_call_shapes.append(tuple(parameters_265.shape))
        return super().joints(parameters_265, betas_300)


class SyntheticBundle:
    def __init__(self, root: Path, *, clips: int = 4, frames: int = 12):
        if clips != 4:
            raise ValueError("fixture uses one clip per SHOW speaker")
        self.root = root
        self.frames = frames
        self.canonical_manifest = root / "canonical_manifest.jsonl"
        self.prediction_manifest = root / "prediction_manifest.jsonl"
        self.prediction_lineage = root / "prediction_lineage.json"
        self.validation_gate_path = root / "validation_gate.json"
        self.validation_gate_artifact: dict[str, object]
        self.canonical_rows: list[dict[str, object]] = []
        self.prediction_rows: list[dict[str, object]] = []
        self._build()

    @staticmethod
    def _wav(path: Path, frames: int = 6400) -> None:
        samples = (
            np.sin(np.arange(frames) * (2.0 * np.pi * 220.0 / 16000.0))
            * 12000.0
        ).astype("<i2")
        payload = io.BytesIO()
        with wave.open(payload, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(samples.tobytes())
        path.write_bytes(payload.getvalue())

    def _build(self) -> None:
        for index, speaker in enumerate(METRICS.SPEAKER_NAMES):
            clip_id = f"{speaker}/video/clip{index:02d}"
            output_id = METRICS.canonical_clip_id(clip_id)
            pose = np.zeros((self.frames, 165), dtype=np.float32)
            facial = np.zeros((self.frames, 100), dtype=np.float32)
            pose[:, 0] = np.linspace(
                index * 0.1,
                index * 0.1 + 0.4,
                self.frames,
                dtype=np.float32,
            )
            pose[:, 3] = np.linspace(
                0.0,
                0.2 + index * 0.03,
                self.frames,
                dtype=np.float32,
            )
            facial[:, 0] = np.linspace(
                index * 0.02,
                0.1 + index * 0.02,
                self.frames,
                dtype=np.float32,
            )
            beta = np.full(
                (self.frames, 300),
                index * 0.001,
                dtype=np.float32,
            )
            canonical_arrays = {
                "pose": pose,
                "contact": np.zeros(
                    (self.frames, 4),
                    dtype=np.float32,
                ),
                "facial": facial,
                "beta": beta,
                "trans": np.zeros(
                    (self.frames, 3),
                    dtype=np.float32,
                ),
                "speaker_id": np.full(
                    (self.frames, 1),
                    METRICS.SHOW_SPEAKER_IDS[speaker],
                    dtype=np.int64,
                ),
            }
            canonical_npz = self.root / f"canonical_{output_id}.npz"
            write_npz(
                canonical_npz,
                METRICS.CANONICAL_FIELDS,
                canonical_arrays,
            )
            wav = self.root / f"{output_id}.wav"
            self._wav(wav)
            dummy_sha = f"{index + 1:x}" * 64
            canonical_row: dict[str, object] = {
                "global_index": 100 + index,
                "clip_id": clip_id,
                "split": "val",
                "speaker": speaker,
                "speaker_id": METRICS.SHOW_SPEAKER_IDS[speaker],
                "video": "video",
                "sequence": f"clip{index:02d}",
                "source_pkl": str(self.root / f"{output_id}.pkl"),
                "source_wav": str(wav),
                "canonical_npz": str(canonical_npz),
                "canonical_npz_relative": canonical_npz.name,
                "global_foot_fastpath_contract": {
                    "format": "synthetic"
                },
                "lower_foot_local": str(
                    self.root / f"{output_id}.lower.npy"
                ),
                "lower_foot_local_relative": f"{output_id}.lower.npy",
                "frames": self.frames,
                "pose_fps": 30,
                "wav_channels": 1,
                "wav_sample_width": 2,
                "wav_sample_rate": 16000,
                "wav_frames": 6400,
                "wav_mono_policy": "synthetic_mono",
                "source_pkl_sha256": dummy_sha,
                "source_wav_sha256": sha256_file(wav),
                "canonical_npz_sha256": sha256_file(canonical_npz),
                "lower_foot_local_sha256": dummy_sha,
                "lineage_contract_sha256": dummy_sha,
            }
            self.canonical_rows.append(canonical_row)
            prediction_pose = pose.copy()
            prediction_pose[:, 0] += np.float32(0.03 + index * 0.002)
            prediction_expression = facial.copy()
            prediction_expression[:, 0] += np.float32(0.02)
            prediction_arrays = {
                "betas": beta[0],
                "poses": prediction_pose,
                "expressions": prediction_expression,
                "trans": canonical_arrays["trans"],
                "model": np.asarray("smplx2020"),
                "gender": np.asarray("neutral"),
                "mocap_frame_rate": np.asarray(30, dtype=np.int64),
            }
            ground_truth_arrays = {
                "betas": beta[0],
                "poses": pose,
                "expressions": facial,
                "trans": canonical_arrays["trans"],
                "model": np.asarray("smplx2020"),
                "gender": np.asarray("neutral"),
                "mocap_frame_rate": np.asarray(30, dtype=np.int64),
            }
            prediction_path = self.root / f"res_{output_id}.npz"
            ground_truth_path = self.root / f"gt_{output_id}.npz"
            write_npz(
                prediction_path,
                METRICS.OUTPUT_FIELDS,
                prediction_arrays,
            )
            write_npz(
                ground_truth_path,
                METRICS.OUTPUT_FIELDS,
                ground_truth_arrays,
            )
            self.prediction_rows.append(
                {
                    "global_index": 100 + index,
                    "split": "val",
                    "source_clip_id": clip_id,
                    "canonical_clip_id": output_id,
                    "frames": self.frames,
                    "epoch": 200,
                    "candidate_checkpoint_sha256": "a" * 64,
                    "prediction": {
                        "path": str(prediction_path),
                        "sha256": sha256_file(prediction_path),
                        "bytes": prediction_path.stat().st_size,
                    },
                    "ground_truth": {
                        "path": str(ground_truth_path),
                        "sha256": sha256_file(ground_truth_path),
                        "bytes": ground_truth_path.stat().st_size,
                    },
                }
            )
        canonical_jsonl_write(
            self.canonical_manifest,
            self.canonical_rows,
        )
        canonical_jsonl_write(
            self.prediction_manifest,
            self.prediction_rows,
        )
        lineage: dict[str, object] = {
            "format": (
                "semtalk_show_base_official_adapt_"
                "val_inference_lineage_v1"
            ),
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "final_manifest": {
                "path": str(self.prediction_manifest),
                "sha256": sha256_file(self.prediction_manifest),
            },
            "clip_count": 4,
            "exact_once": True,
            "finite": True,
        }
        lineage["receipt_payload_sha256"] = (
            METRICS.canonical_json_sha256(lineage)
        )
        canonical_json_write(self.prediction_lineage, lineage)
        external = load_external_gate_test_module()
        gate_root = self.root / "external_gate_fixture"
        gate_root.mkdir()
        fixture = external.GateFixture(gate_root)
        gate = external.GATE.build_gate(
            seed_run_paths=fixture.seed_paths,
            expected_seed_run_sha256=fixture.seed_shas,
            scope="validation_candidate_family",
            test_only_allow_four_clip_subset=True,
        )
        self.validation_gate_path.write_text(
            json.dumps(
                gate,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact, replayed = external.GATE.load_gate(
            self.validation_gate_path,
            sha256_file(self.validation_gate_path),
            expected_scope="validation_candidate_family",
            test_only_allow_four_clip_subset=True,
        )
        if replayed != gate:
            raise AssertionError("external gate fixture did not replay")
        self.validation_gate_artifact = artifact

    def validation_gate(self) -> dict[str, object]:
        return copy.deepcopy(self.validation_gate_artifact)

    def distribution(self) -> dict[str, object]:
        return METRICS.build_distribution_receipt(
            prediction_manifest_artifact={
                "path": str(self.prediction_manifest.resolve()),
                "sha256": sha256_file(self.prediction_manifest),
                "bytes": self.prediction_manifest.stat().st_size,
            },
            prediction_artifacts=[
                {
                    "canonical_clip_id": row["canonical_clip_id"],
                    "prediction_sha256": row["prediction"]["sha256"],
                    "prediction_bytes": row["prediction"]["bytes"],
                }
                for row in self.prediction_rows
            ],
            validation_gate=self.validation_gate(),
            test_only_allow_four_clip_subset=True,
        )

    def evaluate(
        self,
        *,
        declaration: dict[str, object] | None = None,
        prediction_manifest_sha: str | None = None,
        backend: object | None = None,
        formal_mode: bool = False,
    ) -> dict[str, object]:
        return METRICS.evaluate_canonical_bundle(
            canonical_manifest=self.canonical_manifest,
            expected_canonical_manifest_sha256=sha256_file(
                self.canonical_manifest
            ),
            prediction_manifest=self.prediction_manifest,
            expected_prediction_manifest_sha256=(
                prediction_manifest_sha
                or sha256_file(self.prediction_manifest)
            ),
            prediction_lineage=self.prediction_lineage,
            expected_prediction_lineage_sha256=sha256_file(
                self.prediction_lineage
            ),
            validation_gate=self.validation_gate(),
            distribution_declaration=declaration or self.distribution(),
            backend=backend or SyntheticBackend(),
            split="val",
            expected_clip_count=4,
            audio_beat_extractor=lambda _waveform: np.asarray(
                [0.1, 0.2],
                dtype=np.float64,
            ),
            formal_mode=formal_mode,
            test_only_allow_four_clip_gate=True,
        )


class FeatureMomentsTest(unittest.TestCase):
    def test_unbiased_covariance_matches_numpy(self) -> None:
        values = np.asarray(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]],
            dtype=np.float64,
        )
        moments = METRICS.FeatureMoments()
        moments.update(values)
        mean, covariance = moments.mean_and_covariance()
        np.testing.assert_array_equal(mean, values.mean(axis=0))
        np.testing.assert_allclose(
            covariance,
            np.cov(values, rowvar=False, ddof=1),
            rtol=0.0,
            atol=1e-14,
        )

    def test_logical_repeat_is_physical_repeat_equivalent(self) -> None:
        values = np.asarray(
            [[0.0, 1.0], [1.5, 2.0], [3.0, 5.0]],
            dtype=np.float64,
        )
        for repeats in (2, 16):
            logical = METRICS.FeatureMoments()
            logical.update(values, repeat=repeats)
            physical = METRICS.FeatureMoments()
            physical.update(np.repeat(values, repeats, axis=0))
            self.assertEqual(logical.count, physical.count)
            np.testing.assert_array_equal(
                logical.feature_sum,
                physical.feature_sum,
            )
            np.testing.assert_array_equal(
                logical.feature_outer_sum,
                physical.feature_outer_sum,
            )
            logical_mean, logical_covariance = logical.mean_and_covariance()
            physical_mean, physical_covariance = (
                physical.mean_and_covariance()
            )
            np.testing.assert_array_equal(logical_mean, physical_mean)
            np.testing.assert_array_equal(
                logical_covariance,
                physical_covariance,
            )


class FormulaTest(unittest.TestCase):
    def test_face_and_body_formulas(self) -> None:
        ground_truth = np.zeros((4, 80, 3), dtype=np.float64)
        prediction = ground_truth.copy()
        prediction[:, 22:25, 0] = 1.0
        prediction[:, 74:, 1] = 2.0
        face = METRICS.released_face_metrics(
            ground_truth,
            prediction,
        )
        self.assertEqual(face["jaw_l1"], 3.0)
        self.assertEqual(face["landmark_l1"], 12.0)
        self.assertEqual(face["LVD"], 0.0)
        self.assertEqual(face["face_l2_combined"], 15.0)
        poses = np.arange(2 * 165, dtype=np.float32).reshape(2, 165)
        expressions = np.arange(
            2 * 100,
            dtype=np.float32,
        ).reshape(2, 100)
        parameters = METRICS.reorder_to_talkshow(poses, expressions)
        fixed = METRICS.released_body_parameters(parameters)
        self.assertEqual(fixed.shape, parameters.shape)
        np.testing.assert_array_equal(fixed[:, 0:3], 0.0)
        np.testing.assert_array_equal(fixed[:, 165:265], 0.0)

    def test_frozen_globaldiff_primitive_oracle(self) -> None:
        """Differential oracle frozen from globaldiff_show_metrics.py.

        The constants were generated by the evaluator-only GlobalDiff
        primitive at commit d7cee4d, not by this adapter.
        """

        real = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 0.5],
                [2.0, -1.0, 1.5],
                [4.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )
        generated = np.asarray(
            [
                [0.25, 1.25, 2.1],
                [1.5, 1.5, 0.25],
                [1.75, -0.5, 1.0],
                [3.5, 0.25, 2.75],
            ],
            dtype=np.float64,
        )
        real_moments = METRICS.FeatureMoments()
        real_moments.update(real)
        expected_fgd = {
            2: 0.5678648236822035,
            16: 0.6834738940238569,
        }
        for repeats, expected in expected_fgd.items():
            generated_moments = METRICS.FeatureMoments()
            generated_moments.update(generated, repeat=repeats)
            self.assertAlmostEqual(
                METRICS.frechet_distance(
                    real_moments,
                    generated_moments,
                ),
                expected,
                delta=1e-12,
            )
        rng = np.random.default_rng(20260731)
        joints = rng.normal(size=(12, 80, 3))
        bc_numerator, bc_denominator = (
            METRICS.bc_components_for_sequence(
                joints,
                np.asarray([0.05, 0.15, 0.25], dtype=np.float64),
            )
        )
        self.assertAlmostEqual(
            bc_numerator,
            6.052863086728709,
            delta=1e-12,
        )
        self.assertEqual(bc_denominator, 9)
        self.assertEqual(
            METRICS.variation_from_joints(
                np.repeat(joints[None], 2, axis=0)
            ),
            0.0,
        )
        face_gt = np.zeros((4, 80, 3), dtype=np.float64)
        face_prediction = face_gt.copy()
        face_prediction[:, 22:25, 0] = 1.0
        face_prediction[:, 74:, 1] = 2.0
        self.assertEqual(
            METRICS.released_face_metrics(face_gt, face_prediction),
            {
                "jaw_l1": 3.0,
                "landmark_l1": 12.0,
                "LVD": 0.0,
                "face_l2_combined": 15.0,
            },
        )


class OfflineAdapterTest(unittest.TestCase):
    def test_validate_report_returns_released2_primary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticBundle(root)
            report = fixture.evaluate(
                backend=ShapeRecordingCudaBackend(),
                formal_mode=True,
            )
            talkshow_root = root / "talkshow-metric-root"
            talkshow_root.mkdir()
            asset_path = root / "metric-asset.bin"
            asset_path.write_bytes(b"asset")
            talkshow_receipt = {
                "path": str(talkshow_root),
                "commit": METRICS.TALKSHOW_METRIC_COMMIT,
            }
            report["metric_assets"]["talkshow"] = talkshow_receipt
            report["metric_assets"]["feature_extractor"] = {
                "path": str(asset_path),
                "sha256": METRICS.FEATURE_EXTRACTOR_SHA256,
                "bytes": 5,
                "runtime_dtype": "float32",
            }
            report["metric_assets"]["smplx"] = {
                "path": str(asset_path),
                "sha256": METRICS.SMPLX_SHA256,
                "bytes": 5,
                "runtime_dtype": "float64",
            }
            report["test_only_mode"] = False
            report.pop("report_payload_sha256")
            report["report_payload_sha256"] = (
                METRICS.canonical_json_sha256(report)
            )
            original_snapshot = METRICS._verified_file_snapshot

            def snapshot(
                path: object,
                expected_sha: str,
                label: str,
            ) -> tuple[Path, bytes]:
                if label in {
                    "metric report feature_extractor",
                    "metric report smplx",
                }:
                    return Path(path), b"asset"
                return original_snapshot(path, expected_sha, label)

            with (
                mock.patch.object(
                    METRICS,
                    "validate_talkshow_metric_root",
                    return_value=talkshow_receipt,
                ),
                mock.patch.object(
                    METRICS,
                    "_verified_file_snapshot",
                    side_effect=snapshot,
                ),
            ):
                validation = METRICS.validate_report(
                    report,
                    expected_split="val",
                    expected_clip_count=4,
                    expected_prediction_manifest=(
                        report["distribution_receipt"][
                            "prediction_manifest"
                        ]
                    ),
                    expected_distribution_receipt=(
                        report["distribution_receipt"]
                    ),
                    expected_selection_protocol=(
                        report["selection_protocol"]
                    ),
                )
            self.assertEqual(validation["status"], "pass")
            self.assertEqual(
                validation["primary_metric_path"],
                "body.released2.metrics.FGD",
            )
            self.assertEqual(
                validation["primary_metric"],
                report["body"]["released2"]["metrics"]["FGD"],
            )

    def test_formal_cuda_device_and_metric_batch_plumbing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            backend = ShapeRecordingCudaBackend()
            report = fixture.evaluate(
                backend=backend,
                formal_mode=True,
            )
        self.assertTrue(report["formal_mode"])
        self.assertEqual(report["runtime"]["device"], "cuda:0")
        self.assertEqual(
            backend.feature_call_shapes,
            [
                shape
                for _clip in range(4)
                for shape in (
                    (1, 12, 265),
                    (2, 12, 265),
                    (16, 12, 265),
                )
            ],
        )
        self.assertEqual(
            backend.joint_call_shapes,
            [
                shape
                for _clip in range(4)
                for shape in (
                    (16, 12, 265),
                    (2, 12, 265),
                )
            ],
        )
        self.assertEqual(
            report["distribution_receipt"][
                "metric_input_materialization"
            ],
            METRICS.METRIC_INPUT_MATERIALIZATION,
        )

    def test_formal_mode_rejects_cpu_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "CUDA runtime attestation",
            ):
                fixture.evaluate(formal_mode=True)

    def test_complete_delta_report_and_original_repeat_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            report = fixture.evaluate()
        self.assertEqual(report["format"], METRICS.REPORT_FORMAT)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["counts"]["clips"], 4)
        distribution = report["distribution_receipt"]
        self.assertFalse(distribution["independent_samples"])
        self.assertTrue(distribution["deterministic_delta_distribution"])
        self.assertEqual(
            [row["slot"] for row in distribution["logical_slot_bindings"]],
            list(range(16)),
        )
        slot_hashes = {
            row["prediction_artifact_manifest_sha256"]
            for row in distribution["logical_slot_bindings"]
        }
        self.assertEqual(len(slot_hashes), 1)
        released = report["body"]["released2"]
        paper = report["body"]["paper16"]
        self.assertEqual(released["metrics"]["Variation"], 0.0)
        self.assertEqual(paper["metrics"]["Variation"], 0.0)
        self.assertEqual(
            released["counts"]["generated_features"],
            2 * released["counts"]["real_features"],
        )
        self.assertEqual(
            paper["counts"]["generated_features"],
            16 * paper["counts"]["real_features"],
        )
        self.assertEqual(
            released["counts"]["bc_sample_evaluations"],
            4,
        )
        self.assertEqual(
            paper["counts"]["bc_sample_evaluations"],
            64,
        )
        self.assertEqual(
            set(report["face"]["metrics"]),
            {"released", "derived"},
        )
        self.assertEqual(
            report["rs"],
            {"status": "N/A/unreleased", "value": None},
        )
        unsigned = dict(report)
        report_sha = unsigned.pop("report_payload_sha256")
        self.assertEqual(
            report_sha,
            METRICS.canonical_json_sha256(unsigned),
        )

    def test_slot_tamper_is_rejected_even_with_rehashed_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            declaration = copy.deepcopy(fixture.distribution())
            declaration["released2_slots"] = [0, 2]
            declaration.pop("receipt_payload_sha256")
            declaration["receipt_payload_sha256"] = (
                METRICS.compact_canonical_json_sha256(declaration)
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "distribution declaration",
            ):
                fixture.evaluate(declaration=declaration)

    def test_randomness_declaration_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            declaration = copy.deepcopy(fixture.distribution())
            declaration["independent_samples"] = True
            declaration["deterministic_delta_distribution"] = False
            declaration.pop("receipt_payload_sha256")
            declaration["receipt_payload_sha256"] = (
                METRICS.compact_canonical_json_sha256(declaration)
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "distribution declaration",
            ):
                fixture.evaluate(declaration=declaration)

    def test_cross_module_receipt_fixture_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            declaration = fixture.distribution()
            external = load_external_gate_test_module()
            unsigned = dict(declaration)
            claimed = unsigned.pop("receipt_payload_sha256")
            self.assertEqual(
                claimed,
                external.GATE.canonical_json_sha256(unsigned),
            )
            self.assertEqual(
                claimed,
                METRICS.compact_canonical_json_sha256(unsigned),
            )
            records = [
                {
                    "canonical_clip_id": row["canonical_clip_id"],
                    "prediction_sha256": row["prediction"]["sha256"],
                    "prediction_bytes": row["prediction"]["bytes"],
                }
                for row in fixture.prediction_rows
            ]
            manifest = {
                "path": str(fixture.prediction_manifest.resolve()),
                "sha256": sha256_file(fixture.prediction_manifest),
                "bytes": fixture.prediction_manifest.stat().st_size,
            }
            self.assertEqual(
                external.GATE.validate_distribution_receipt(
                    declaration,
                    expected_gate_artifact=fixture.validation_gate(),
                    expected_prediction_manifest=manifest,
                    expected_prediction_records=records,
                ),
                declaration,
            )

    def test_slot_binding_tamper_is_rejected_after_valid_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            declaration = copy.deepcopy(fixture.distribution())
            declaration["logical_slot_bindings"][5][
                "prediction_artifact_manifest_sha256"
            ] = "f" * 64
            declaration.pop("receipt_payload_sha256")
            declaration["receipt_payload_sha256"] = (
                METRICS.compact_canonical_json_sha256(declaration)
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "deterministic delta contract",
            ):
                fixture.evaluate(declaration=declaration)

    def test_metric_batch_contract_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            declaration = copy.deepcopy(fixture.distribution())
            declaration["metric_input_materialization"][
                "feature_extractor_batches"
            ] = [1, 16]
            declaration.pop("receipt_payload_sha256")
            declaration["receipt_payload_sha256"] = (
                METRICS.compact_canonical_json_sha256(declaration)
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "deterministic delta contract",
            ):
                fixture.evaluate(declaration=declaration)

    def test_receipt_payload_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            declaration = copy.deepcopy(fixture.distribution())
            declaration["receipt_payload_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "payload SHA-256 mismatch",
            ):
                fixture.evaluate(declaration=declaration)

    def test_receipt_schema_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            declaration = copy.deepcopy(fixture.distribution())
            declaration["unexpected"] = True
            declaration.pop("receipt_payload_sha256")
            declaration["receipt_payload_sha256"] = (
                METRICS.compact_canonical_json_sha256(declaration)
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "schema mismatch",
            ):
                fixture.evaluate(declaration=declaration)

    def test_lineage_randomness_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            lineage = json.loads(fixture.prediction_lineage.read_text())
            lineage["randomness"] = {
                "independent_samples": True,
                "stochastic": True,
            }
            lineage.pop("receipt_payload_sha256")
            lineage["receipt_payload_sha256"] = (
                METRICS.canonical_json_sha256(lineage)
            )
            canonical_json_write(fixture.prediction_lineage, lineage)
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "conflicts with deterministic delta",
            ):
                fixture.evaluate()

    def test_manifest_sha_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            expected_sha = sha256_file(fixture.prediction_manifest)
            with fixture.prediction_manifest.open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "SHA-256",
            ):
                fixture.evaluate(prediction_manifest_sha=expected_sha)

    def test_canonical_speaker_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            fixture.canonical_rows[0]["speaker"] = "chemistry"
            fixture.canonical_rows[0]["speaker_id"] = (
                METRICS.SHOW_SPEAKER_IDS["chemistry"]
            )
            canonical_jsonl_write(
                fixture.canonical_manifest,
                fixture.canonical_rows,
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "speaker/frame-rate mismatch",
            ):
                fixture.evaluate()

    def test_canonical_split_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            fixture.canonical_rows[0]["split"] = "test"
            canonical_jsonl_write(
                fixture.canonical_manifest,
                fixture.canonical_rows,
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "canonical val clips 3 != 4",
            ):
                fixture.evaluate()

    def test_canonical_npz_path_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            first = fixture.canonical_rows[0]
            second = fixture.canonical_rows[1]
            for field in (
                "canonical_npz",
                "canonical_npz_relative",
                "canonical_npz_sha256",
            ):
                first[field], second[field] = second[field], first[field]
            canonical_jsonl_write(
                fixture.canonical_manifest,
                fixture.canonical_rows,
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "speaker_id",
            ):
                fixture.evaluate()

    def test_prediction_lineage_path_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            alternate = fixture.root / "alternate_prediction_manifest.jsonl"
            alternate.write_bytes(fixture.prediction_manifest.read_bytes())
            lineage = json.loads(fixture.prediction_lineage.read_text())
            lineage["final_manifest"]["path"] = str(alternate)
            lineage.pop("receipt_payload_sha256")
            lineage["receipt_payload_sha256"] = (
                METRICS.canonical_json_sha256(lineage)
            )
            canonical_json_write(fixture.prediction_lineage, lineage)
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "lineage/manifest mismatch",
            ):
                fixture.evaluate()

    def test_prediction_artifact_path_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            first = fixture.prediction_rows[0]["prediction"]
            second = fixture.prediction_rows[1]["prediction"]
            first_copy = dict(first)
            first.update(second)
            second.update(first_copy)
            canonical_jsonl_write(
                fixture.prediction_manifest,
                fixture.prediction_rows,
            )
            lineage = json.loads(fixture.prediction_lineage.read_text())
            lineage["final_manifest"]["sha256"] = sha256_file(
                fixture.prediction_manifest
            )
            lineage.pop("receipt_payload_sha256")
            lineage["receipt_payload_sha256"] = (
                METRICS.canonical_json_sha256(lineage)
            )
            canonical_json_write(fixture.prediction_lineage, lineage)
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "output NPZ basename",
            ):
                fixture.evaluate()

    def test_missing_prediction_sample_is_rejected_before_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            canonical_jsonl_write(
                fixture.prediction_manifest,
                fixture.prediction_rows[:-1],
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "cover canonical split exactly once",
            ):
                fixture.evaluate(
                    prediction_manifest_sha=sha256_file(
                        fixture.prediction_manifest
                    )
                )

    def test_nonfinite_prediction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            row = fixture.prediction_rows[0]
            path = Path(row["prediction"]["path"])
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name]).copy()
                    for name in METRICS.OUTPUT_FIELDS
                }
            arrays["poses"][0, 0] = np.nan
            write_npz(path, METRICS.OUTPUT_FIELDS, arrays)
            row["prediction"]["sha256"] = sha256_file(path)
            row["prediction"]["bytes"] = path.stat().st_size
            canonical_jsonl_write(
                fixture.prediction_manifest,
                fixture.prediction_rows,
            )
            lineage = json.loads(fixture.prediction_lineage.read_text())
            lineage["final_manifest"]["sha256"] = sha256_file(
                fixture.prediction_manifest
            )
            lineage.pop("receipt_payload_sha256")
            lineage["receipt_payload_sha256"] = (
                METRICS.canonical_json_sha256(lineage)
            )
            canonical_json_write(fixture.prediction_lineage, lineage)
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "non-finite",
            ):
                fixture.evaluate()


if __name__ == "__main__":
    unittest.main()
