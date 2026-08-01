from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np

import scripts.show_base as SHOW_BASE_PACKAGE

METRICS = importlib.import_module(
    "scripts.show_base.evaluate_talkshow_show_metrics"
)
PRIMARY_CLI = importlib.import_module(
    "scripts.show_base.replay_released2_primary"
)
ROOT = Path(__file__).resolve().parents[1]
INTERNAL_RELEASED2_MODULES = frozenset(
    {
        "scripts/show_base/evaluate_talkshow_show_metrics.py",
        "scripts/show_base/replay_released2_primary.py",
    }
)


def source_bound_cache_authority() -> tuple[dict[str, object], dict[str, object]]:
    canonical: dict[str, object] = {
        "path": "/authority/canonical/manifest.jsonl",
        "sha256": "1" * 64,
        "bytes": 100,
        "rows": 1715,
        "selected_rows": 1715,
    }
    inputs: dict[str, object] = {
        "format": "semtalk_show_base_talkshow_val_inputs_v2",
        "status": "frozen",
        "split": "val",
        "test_visible": False,
        "expected_clip_count": 1715,
        "canonical_manifest": {
            "path": canonical["path"],
            "sha256": canonical["sha256"],
        },
        "canonical_summary": {
            "path": "/authority/canonical/summary.json",
            "sha256": "2" * 64,
        },
        "canonical_lineage": {
            "path": "/authority/canonical/lineage.json",
            "sha256": "3" * 64,
        },
        "audio_manifests": [
            {
                "path": f"/authority/audio/shard_{index}/manifest.jsonl",
                "sha256": "4" * 64,
            }
            for index in range(8)
        ],
        "audio_summaries": [
            {
                "path": f"/authority/audio/shard_{index}/summary.json",
                "sha256": "5" * 64,
            }
            for index in range(8)
        ],
        "audio_lineages": [
            {
                "path": f"/authority/audio/shard_{index}/lineage.json",
                "sha256": "6" * 64,
            }
            for index in range(8)
        ],
        "clip_ids_sha256": "7" * 64,
        "talkshow_window_manifest_sha256": "8" * 64,
    }
    inputs["receipt_payload_sha256"] = (
        METRICS.compact_canonical_json_sha256(inputs)
    )
    authority: dict[str, object] = {
        "format": (
            METRICS.PRIMARY_REAL_FEATURE_CACHE_PRODUCTION_AUTHORITY_FORMAT
        ),
        "status": "frozen",
        "source": {
            "origin": METRICS.SEMTALK_OFFICIAL_ORIGIN,
            "source_root": "/authority/source",
            "commit": "9" * 40,
            "tree": "a" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
            "entrypoint": {
                "path": (
                    "/authority/source/"
                    + METRICS.PRIMARY_REAL_FEATURE_CACHE_ENTRYPOINT
                ),
                "relative": METRICS.PRIMARY_REAL_FEATURE_CACHE_ENTRYPOINT,
                "sha256": "b" * 64,
                "bytes": 1234,
                "git_mode": "100755",
                "git_blob_sha1": "c" * 40,
            },
        },
        "validation_inputs": inputs,
    }
    authority["receipt_payload_sha256"] = (
        METRICS.compact_canonical_json_sha256(authority)
    )
    return authority, canonical


def local_show_base_imports(relative: str) -> set[str]:
    tree = ast.parse(
        (ROOT / relative).read_text(encoding="utf-8"),
        filename=relative,
    )
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module == "scripts.show_base":
                names = [
                    f"scripts.show_base.{alias.name}"
                    for alias in node.names
                ]
            elif (node.module or "").startswith("scripts.show_base."):
                names = [node.module or ""]
            else:
                names = []
        else:
            continue
        for name in names:
            prefix = "scripts.show_base."
            if not name.startswith(prefix):
                continue
            module = name[len(prefix):].split(".", 1)[0]
            candidate = f"scripts/show_base/{module}.py"
            if (ROOT / candidate).is_file():
                dependencies.add(candidate)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        loader = node.func
        loader_name = (
            loader.id
            if isinstance(loader, ast.Name)
            else loader.attr
            if isinstance(loader, ast.Attribute)
            else ""
        )
        argument = node.args[0]
        if (
            loader_name
            not in {"import_module", "_fresh_local_module", "_control_module"}
            or not isinstance(argument, ast.Constant)
            or not isinstance(argument.value, str)
        ):
            continue
        module = argument.value.removeprefix("scripts.show_base.")
        candidate = f"scripts/show_base/{module}.py"
        if (ROOT / candidate).is_file():
            dependencies.add(candidate)
    return dependencies


def local_show_base_import_closure(roots: set[str]) -> set[str]:
    pending = list(roots)
    closure: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        closure.add(relative)
        pending.extend(local_show_base_imports(relative) - closure)
    return closure


def load_external_gate_test_module() -> object:
    gate_source = (
        ROOT
        / "scripts"
        / "show_base"
        / "deterministic_replication_gate.py"
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
            "soundfile": "synthetic-test",
            "soxr": "synthetic-test",
            "device": "cpu",
        }

    def decode_audio_16k(self, snapshot: bytes) -> np.ndarray:
        with wave.open(io.BytesIO(snapshot), "rb") as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = handle.getnframes()
            samples = np.frombuffer(
                handle.readframes(frames),
                dtype="<i2",
            ).reshape(frames, channels)
        mono = samples.astype(np.float32).mean(axis=1) / np.float32(32768.0)
        output_frames = int(np.ceil(mono.size * 16000 / rate))
        positions = np.arange(output_frames, dtype=np.float64) * rate / 16000
        return np.interp(
            positions,
            np.arange(mono.size, dtype=np.float64),
            mono,
        ).astype(np.float32)

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
    def __init__(self, root: Path, *, clips: int = 4, frames: int = 64):
        if clips != 4:
            raise ValueError("fixture uses one clip per SHOW speaker")
        root = root.resolve()
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
    def _wav(path: Path, frames: int = 48_000) -> None:
        time = np.arange(frames)
        left = (
            np.sin(time * (2.0 * np.pi * 220.0 / 22_000.0))
            * 12000.0
        ).astype("<i2")
        right = (
            np.cos(time * (2.0 * np.pi * 330.0 / 22_000.0))
            * 9000.0
        ).astype("<i2")
        samples = np.stack((left, right), axis=1)
        payload = io.BytesIO()
        with wave.open(payload, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(22000)
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
                "wav_channels": 2,
                "wav_sample_width": 2,
                "wav_sample_rate": 22000,
                "wav_frames": 48000,
                "wav_mono_policy": METRICS.CANONICAL_WAV_MONO_POLICY,
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
            logical.update(np.repeat(values, repeats, axis=0))
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
            generated_moments.update(
                np.repeat(generated, repeats, axis=0)
            )
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
    def test_primary_cli_supports_direct_script_execution(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "show_base"
                    / "replay_released2_primary.py"
                ),
                "--help",
            ],
            cwd="/",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("build-cache", completed.stdout)
        self.assertIn("replay", completed.stdout)

    def test_primary_cli_is_wired_and_create_new_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "cache.json"
            canonical = root / "canonical.jsonl"
            canonical.write_bytes(b"fixture\n")
            result = {
                "format": METRICS.PRIMARY_REAL_FEATURE_CACHE_FORMAT,
                "receipt_payload_sha256": "1" * 64,
            }
            argv = [
                "build-cache",
                "--talkshow-metric-root",
                str(root),
                "--feature-extractor",
                str(root / "feature.pth"),
                "--smplx-asset",
                str(root / "smplx.npz"),
                "--device",
                "cuda:0",
                "--split",
                "val",
                "--expected-clip-count",
                "1715",
                "--output-json",
                str(output),
                "--canonical-manifest",
                str(canonical),
                "--expected-canonical-manifest-sha256",
                "2" * 64,
                "--canonical-summary",
                str(root / "summary.json"),
                "--expected-canonical-summary-sha256",
                "3" * 64,
                "--canonical-lineage",
                str(root / "lineage.json"),
                "--expected-canonical-lineage-sha256",
                "4" * 64,
                "--source-root",
                str(ROOT),
                "--expected-source-commit",
                "5" * 40,
                "--expected-source-tree",
                "6" * 40,
                "--expected-entrypoint-sha256",
                "7" * 64,
            ]
            for name, digit in (
                ("manifest", "8"),
                ("summary", "9"),
                ("lineage", "a"),
            ):
                for index in range(8):
                    argv.extend(
                        [
                            f"--audio-{name}",
                            str(root / f"shard-{index}-{name}.json"),
                            f"--expected-audio-{name}-sha256",
                            digit * 64,
                        ]
                    )
            authority = {"source_bound": True}
            with (
                mock.patch.object(
                    PRIMARY_CLI,
                    "_backend",
                    return_value=object(),
                ),
                mock.patch.object(
                    PRIMARY_CLI,
                    "_production_authority",
                    return_value=authority,
                ) as authority_builder,
                mock.patch.object(
                    METRICS,
                    "build_released2_real_feature_cache",
                    return_value=result,
                ) as builder,
                mock.patch.object(
                    PRIMARY_CLI,
                    "_validate_written_cache",
                ) as validator,
            ):
                self.assertEqual(PRIMARY_CLI.main(argv), 0)
                self.assertEqual(
                    json.loads(output.read_bytes()),
                    result,
                )
                self.assertEqual(authority_builder.call_count, 2)
                builder.assert_called_once_with(
                    canonical_manifest=canonical,
                    expected_canonical_manifest_sha256="2" * 64,
                    backend=mock.ANY,
                    split="val",
                    expected_clip_count=1715,
                    formal_mode=True,
                    test_only_allow_four_clip_subset=False,
                    production_authority=authority,
                )
                validator.assert_called_once()
                with self.assertRaises(FileExistsError):
                    PRIMARY_CLI.main(argv)

    def test_formal_cache_authority_binds_source_and_all_audio_receipts(
        self,
    ) -> None:
        authority, canonical = source_bound_cache_authority()
        with self.assertRaisesRegex(
            METRICS.MetricAdapterContractError,
            "production authority schema",
        ):
            METRICS._validate_real_feature_cache_production_authority(
                None,
                canonical_manifest=canonical,
                split="val",
                clip_count=1715,
            )
        validated = (
            METRICS._validate_real_feature_cache_production_authority(
                authority,
                canonical_manifest=canonical,
                split="val",
                clip_count=1715,
            )
        )
        self.assertEqual(validated, authority)

        missing_shard = copy.deepcopy(authority)
        missing_shard["validation_inputs"]["audio_lineages"].pop()
        inputs = missing_shard["validation_inputs"]
        inputs.pop("receipt_payload_sha256")
        inputs["receipt_payload_sha256"] = (
            METRICS.compact_canonical_json_sha256(inputs)
        )
        missing_shard.pop("receipt_payload_sha256")
        missing_shard["receipt_payload_sha256"] = (
            METRICS.compact_canonical_json_sha256(missing_shard)
        )
        with self.assertRaisesRegex(
            METRICS.MetricAdapterContractError,
            "eight shards",
        ):
            METRICS._validate_real_feature_cache_production_authority(
                missing_shard,
                canonical_manifest=canonical,
                split="val",
                clip_count=1715,
            )

    def test_cache_source_authority_rejects_commit_tree_substitution(
        self,
    ) -> None:
        entrypoint = {
            "path": str(ROOT / PRIMARY_CLI.ENTRYPOINT_RELATIVE),
            "sha256": "3" * 64,
            "bytes": 123,
            "git_mode": "100755",
            "git_blob_sha1": "4" * 40,
        }
        source = {
            "origin": METRICS.SEMTALK_OFFICIAL_ORIGIN,
            "source_root": str(ROOT),
            "commit": "1" * 40,
            "tree": "2" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
            "files": {PRIMARY_CLI.ENTRYPOINT_RELATIVE: entrypoint},
        }
        args = types.SimpleNamespace(
            source_root=ROOT,
            expected_source_commit="1" * 40,
            expected_source_tree="5" * 40,
            expected_entrypoint_sha256="3" * 64,
        )
        with mock.patch.object(
            PRIMARY_CLI.val_contract,
            "build_fresh_pipeline_source_receipt",
            return_value=source,
        ):
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "commit/tree mismatch",
            ):
                PRIMARY_CLI._source_authority(args)

    def test_final_npz_provenance_rejects_relocation_and_unrelated_shard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            npz_root = root / "npz" / "test"
            npz_root.mkdir(parents=True)
            output_id = "oliver_clip00"
            source_clip_id = "oliver/video/clip00"
            common = {
                "global_index": 15402,
                "source_clip_id": source_clip_id,
                "canonical_clip_id": output_id,
                "speaker": "oliver",
                "speaker_id": 0,
                "frames": 64,
                "canonical_npz": str((root / "canonical.npz").resolve()),
                "canonical_npz_sha256": "1" * 64,
                "audio_feature_npz": str((root / "audio.npz").resolve()),
                "audio_feature_npz_sha256": "2" * 64,
            }
            receipts = {}
            shard_receipts = {}
            shard_npz_root = (
                Path(directory)
                / "shards"
                / "shard-00002-of-00008"
                / "npz"
                / "test"
            )
            shard_npz_root.mkdir(parents=True)
            for role, prefix in (
                ("prediction", "res"),
                ("ground_truth", "gt"),
            ):
                path = npz_root / f"{prefix}_{output_id}.npz"
                path.write_bytes(role.encode())
                receipts[role] = {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                shard_path = shard_npz_root / path.name
                shutil.copyfile(path, shard_path)
                shard_receipts[role] = {
                    "path": str(shard_path.resolve()),
                    "sha256": sha256_file(shard_path),
                    "bytes": shard_path.stat().st_size,
                }
            shard = {**common, **copy.deepcopy(shard_receipts)}
            final = {
                **common,
                **copy.deepcopy(receipts),
                "evaluation_index": 0,
            }
            METRICS._validate_final_npz_provenance(
                root=root.resolve(),
                ordered_predictions=[final],
                shard_rows_by_id={output_id: shard},
            )
            relocated = Path(directory) / f"res_{output_id}.npz"
            shutil.copyfile(receipts["prediction"]["path"], relocated)
            relocated_row = copy.deepcopy(final)
            relocated_row["prediction"]["path"] = str(relocated.resolve())
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "exact artifact path",
            ):
                METRICS._validate_final_npz_provenance(
                    root=root.resolve(),
                    ordered_predictions=[relocated_row],
                    shard_rows_by_id={output_id: shard},
                )
            unrelated = copy.deepcopy(shard)
            unrelated["source_clip_id"] = "oliver/video/other"
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "identity receipts",
            ):
                METRICS._validate_final_npz_provenance(
                    root=root.resolve(),
                    ordered_predictions=[final],
                    shard_rows_by_id={output_id: unrelated},
                )
            missing_source = Path(shard["prediction"]["path"])
            missing_source.unlink()
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "regular file",
            ):
                METRICS._validate_final_npz_provenance(
                    root=root.resolve(),
                    ordered_predictions=[final],
                    shard_rows_by_id={output_id: shard},
                )
            missing_source.write_bytes(b"different")
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "SHA-256",
            ):
                METRICS._validate_final_npz_provenance(
                    root=root.resolve(),
                    ordered_predictions=[final],
                    shard_rows_by_id={output_id: shard},
                )

    def test_final_order_is_canonical_id_not_global_index_order(self) -> None:
        canonical = {}
        rows = []
        definitions = (
            ("oliver/video/z", 100, "oliver"),
            ("chemistry/video/a", 101, "chemistry"),
        )
        for clip_id, global_index, speaker in definitions:
            output_id = METRICS.canonical_clip_id(clip_id)
            canonical[output_id] = {
                "clip_id": clip_id,
                "global_index": global_index,
                "frames": 64,
                "speaker": speaker,
                "speaker_id": METRICS.SHOW_SPEAKER_IDS[speaker],
                "canonical_npz": f"/fixture/{output_id}.npz",
                "canonical_npz_sha256": "1" * 64,
            }
        for evaluation_index, output_id in enumerate(sorted(canonical)):
            value = canonical[output_id]
            rows.append(
                {
                    "global_index": value["global_index"],
                    "source_clip_id": value["clip_id"],
                    "canonical_clip_id": output_id,
                    "speaker": value["speaker"],
                    "speaker_id": value["speaker_id"],
                    "frames": 64,
                    "canonical_npz": value["canonical_npz"],
                    "canonical_npz_sha256": "1" * 64,
                    "audio_feature_npz": f"/fixture/{output_id}-audio.npz",
                    "audio_feature_npz_sha256": "2" * 64,
                    "prediction": {
                        "path": f"/fixture/res_{output_id}.npz",
                        "sha256": "3" * 64,
                        "bytes": 1,
                    },
                    "ground_truth": {
                        "path": f"/fixture/gt_{output_id}.npz",
                        "sha256": "4" * 64,
                        "bytes": 1,
                    },
                    "evaluation_index": evaluation_index,
                }
            )
        validated = METRICS._validate_prediction_rows(
            rows,
            canonical=canonical,
            split="test",
        )
        self.assertEqual(
            [row["canonical_clip_id"] for row in validated],
            sorted(canonical),
        )
        self.assertNotEqual(
            [row["global_index"] for row in validated],
            sorted(row["global_index"] for row in validated),
        )

    def test_output_audio_must_match_authorized_audio_manifest(self) -> None:
        row = {
            "source_clip_id": "oliver/video/clip00",
            "canonical_clip_id": "oliver_clip00",
            "audio_feature_npz": "/authorized/audio.npz",
            "audio_feature_npz_sha256": "1" * 64,
        }
        authority = {
            row["source_clip_id"]: (
                row["audio_feature_npz"],
                row["audio_feature_npz_sha256"],
            )
        }
        METRICS._validate_output_audio_authority(
            ordered_predictions=[row],
            authorized_audio_by_clip=authority,
        )
        tampered = dict(row)
        tampered["audio_feature_npz"] = "/other/audio.npz"
        tampered["audio_feature_npz_sha256"] = "2" * 64
        with self.assertRaisesRegex(
            METRICS.MetricAdapterContractError,
            "authorized audio",
        ):
            METRICS._validate_output_audio_authority(
                ordered_predictions=[tampered],
                authorized_audio_by_clip=authority,
            )

    def test_final_inference_evidence_uses_external_checkpoint_projection(
        self,
    ) -> None:
        stages = ("base", "face", "hands", "upper", "lower", "global")
        authority = {
            "checkpoints": {
                stage: {
                    "path": f"/fixture/{stage}.bin",
                    "sha256": hashlib.sha256(stage.encode()).hexdigest(),
                    "bytes": len(stage),
                }
                for stage in stages
            }
        }
        contract = {
            "checkpoints": {
                stage: {
                    "path": authority["checkpoints"][stage]["path"],
                    "expected_sha256": authority["checkpoints"][stage][
                        "sha256"
                    ],
                }
                for stage in stages
            },
            "untrusted_training_summary": "telemetry-only",
        }
        receipts = {
            stage: {
                **authority["checkpoints"][stage],
                "formal_stage": stage,
                "audit": {"telemetry_only": True},
            }
            for stage in stages
        }
        METRICS._validate_frozen_inference_evidence(
            authority=authority,
            contract=contract,
            checkpoint_receipts=receipts,
        )
        changed_contract = copy.deepcopy(contract)
        changed_contract["checkpoints"]["base"]["expected_sha256"] = "1" * 64
        with self.assertRaisesRegex(
            METRICS.MetricAdapterContractError,
            "changed the base checkpoint",
        ):
            METRICS._validate_frozen_inference_evidence(
                authority=authority,
                contract=changed_contract,
            )
        changed_receipts = copy.deepcopy(receipts)
        changed_receipts["face"]["bytes"] += 1
        with self.assertRaisesRegex(
            METRICS.MetricAdapterContractError,
            "changed the face checkpoint",
        ):
            METRICS._validate_frozen_inference_evidence(
                authority=authority,
                checkpoint_receipts=changed_receipts,
            )

    def test_pinned_talkshow_import_ignores_poisoned_module_caches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nets").mkdir()
            (root / "data_utils").mkdir()
            sources = {
                "nets/__init__.py": "",
                "nets/body_ae.py": (
                    "from data_utils.consts import VALUE\n"
                    "class TrainWrapper:\n"
                    "    source = VALUE\n"
                ),
                "data_utils/__init__.py": "",
                "data_utils/consts.py": "VALUE = 'pinned'\n",
            }
            files = {}
            for relative, source in sources.items():
                path = root / relative
                path.write_text(source)
                payload = path.read_bytes()
                files[relative] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            fake_nets = types.ModuleType("nets")
            fake_body = types.ModuleType("nets.body_ae")
            fake_body.TrainWrapper = type(
                "Poisoned",
                (),
                {"source": "poisoned"},
            )
            fake_data = types.ModuleType("data_utils")
            fake_consts = types.ModuleType("data_utils.consts")
            fake_consts.VALUE = "poisoned"
            poisoned = {
                "nets": fake_nets,
                "nets.body_ae": fake_body,
                "data_utils": fake_data,
                "data_utils.consts": fake_consts,
            }
            with mock.patch.dict(sys.modules, poisoned, clear=False):
                wrapper = METRICS._import_pinned_body_feature_extractor(
                    {"path": str(root.resolve()), "files": files}
                )
                self.assertEqual(wrapper.source, "pinned")
                self.assertIs(sys.modules["nets.body_ae"], fake_body)
                self.assertIs(sys.modules["data_utils.consts"], fake_consts)

    def test_pinned_talkshow_import_rejects_timestamp_valid_bytecode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nets").mkdir()
            (root / "data_utils").mkdir()
            pinned_body = (
                "class TrainWrapper:\n"
                "    source = 'PINNED'\n"
            )
            stale_body = (
                "class TrainWrapper:\n"
                "    source = 'STALED'\n"
            )
            self.assertEqual(len(pinned_body), len(stale_body))
            sources = {
                "nets/__init__.py": "",
                "nets/body_ae.py": pinned_body,
                "data_utils/__init__.py": "",
            }
            files = {}
            for relative, source in sources.items():
                path = root / relative
                path.write_text(source)
                payload = path.read_bytes()
                files[relative] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            body_path = root / "nets" / "body_ae.py"
            fixed_timestamp = 1_700_000_000
            body_path.write_text(stale_body)
            os.utime(
                body_path,
                (fixed_timestamp, fixed_timestamp),
            )
            bytecode_path = (
                body_path.parent
                / "__pycache__"
                / f"body_ae.{sys.implementation.cache_tag}.pyc"
            )
            bytecode_path.parent.mkdir()
            py_compile.compile(
                str(body_path),
                cfile=str(bytecode_path),
                doraise=True,
            )
            body_path.write_text(pinned_body)
            os.utime(
                body_path,
                (fixed_timestamp, fixed_timestamp),
            )
            self.assertEqual(
                hashlib.sha256(body_path.read_bytes()).hexdigest(),
                files["nets/body_ae.py"]["sha256"],
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "cached bytecode",
            ):
                METRICS._import_pinned_body_feature_extractor(
                    {"path": str(root.resolve()), "files": files}
                )

    def test_pinned_talkshow_import_executes_namespace_from_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "nets" / "spg").mkdir(parents=True)
            (root / "data_utils").mkdir()
            sources = {
                "nets/__init__.py": "",
                "nets/body_ae.py": (
                    "from nets.spg.helper import VALUE\n"
                    "class TrainWrapper:\n"
                    "    source = VALUE\n"
                ),
                "nets/spg/helper.py": "VALUE = 'snapshot-namespace'\n",
                "data_utils/__init__.py": "",
            }
            files = {}
            for relative, source in sources.items():
                path = root / relative
                path.write_text(source, encoding="utf-8")
                payload = path.read_bytes()
                files[relative] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            wrapper = METRICS._import_pinned_body_feature_extractor(
                {"path": str(root), "files": files}
            )
            self.assertEqual(wrapper.source, "snapshot-namespace")
            self.assertNotIn("nets.spg", sys.modules)

    def test_pinned_talkshow_import_executes_attested_bytes_during_swap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "nets").mkdir()
            (root / "data_utils").mkdir()
            pinned = (
                "class TrainWrapper:\n"
                "    source = 'pinned-bytes'\n"
            )
            transient = (
                "class TrainWrapper:\n"
                "    source = 'swapped-code'\n"
            )
            sources = {
                "nets/__init__.py": "",
                "nets/body_ae.py": pinned,
                "data_utils/__init__.py": "",
            }
            files = {}
            for relative, source in sources.items():
                path = root / relative
                path.write_text(source, encoding="utf-8")
                payload = path.read_bytes()
                files[relative] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            body_path = root / "nets" / "body_ae.py"
            real_import = importlib.import_module

            def swap_around_import(name: str, package: str | None = None):
                body_path.write_text(transient, encoding="utf-8")
                try:
                    return real_import(name, package)
                finally:
                    body_path.write_text(pinned, encoding="utf-8")

            with mock.patch.object(
                METRICS.importlib,
                "import_module",
                side_effect=swap_around_import,
            ):
                wrapper = METRICS._import_pinned_body_feature_extractor(
                    {"path": str(root), "files": files}
                )
            self.assertEqual(wrapper.source, "pinned-bytes")

    def test_fresh_primary_replay_defeats_coordinated_report_forgery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            backend = SyntheticBackend()
            report = fixture.evaluate(backend=backend)
            cache = METRICS.build_released2_real_feature_cache(
                canonical_manifest=fixture.canonical_manifest,
                expected_canonical_manifest_sha256=sha256_file(
                    fixture.canonical_manifest
                ),
                backend=backend,
                split="val",
                expected_clip_count=4,
                formal_mode=False,
                test_only_allow_four_clip_subset=True,
            )
            cache_path = Path(directory) / "real-cache.json"
            canonical_json_write(cache_path, cache)
            cache_artifact = {
                "path": str(cache_path.resolve()),
                "sha256": sha256_file(cache_path),
                "bytes": cache_path.stat().st_size,
                "receipt_payload_sha256": cache[
                    "receipt_payload_sha256"
                ],
            }
            prediction_artifact = report["distribution_receipt"][
                "prediction_manifest"
            ]
            replay = METRICS.fresh_replay_released2_primary(
                report,
                backend,
                real_feature_cache=cache,
                expected_real_feature_cache_artifact=cache_artifact,
                expected_prediction_manifest=prediction_artifact,
                expected_distribution_receipt=report[
                    "distribution_receipt"
                ],
                expected_selection_protocol=report[
                    "selection_protocol"
                ],
                expected_split="val",
                expected_clip_count=4,
                test_only_allow_four_clip_subset=True,
            )
            replay_path = Path(directory) / "replay.json"
            canonical_json_write(replay_path, replay)
            replay_artifact = {
                "path": str(replay_path.resolve()),
                "sha256": sha256_file(replay_path),
                "bytes": replay_path.stat().st_size,
                "receipt_payload_sha256": replay[
                    "receipt_payload_sha256"
                ],
            }
            validated = (
                METRICS.validate_released2_primary_replay_receipt(
                    replay_artifact,
                    expected_report=report,
                    expected_prediction_manifest=prediction_artifact,
                    expected_distribution_receipt=report[
                        "distribution_receipt"
                    ],
                    expected_selection_protocol=report[
                        "selection_protocol"
                    ],
                    expected_split="val",
                    expected_clip_count=4,
                    test_only_allow_four_clip_subset=True,
                )
            )
            self.assertEqual(
                validated["primary_metric"],
                replay["primary_metric"],
            )
            incompatible_cache = copy.deepcopy(cache)
            incompatible_cache["runtime"]["torch"] = "different-runtime"
            incompatible_cache.pop("receipt_payload_sha256")
            incompatible_cache["receipt_payload_sha256"] = (
                METRICS.canonical_json_sha256(incompatible_cache)
            )
            incompatible_path = Path(directory) / "incompatible-cache.json"
            canonical_json_write(incompatible_path, incompatible_cache)
            incompatible_artifact = {
                "path": str(incompatible_path.resolve()),
                "sha256": sha256_file(incompatible_path),
                "bytes": incompatible_path.stat().st_size,
                "receipt_payload_sha256": incompatible_cache[
                    "receipt_payload_sha256"
                ],
            }
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "assets/runtime",
            ):
                METRICS.fresh_replay_released2_primary(
                    report,
                    backend,
                    real_feature_cache=incompatible_cache,
                    expected_real_feature_cache_artifact=(
                        incompatible_artifact
                    ),
                    expected_prediction_manifest=prediction_artifact,
                    expected_distribution_receipt=report[
                        "distribution_receipt"
                    ],
                    expected_selection_protocol=report[
                        "selection_protocol"
                    ],
                    expected_split="val",
                    expected_clip_count=4,
                    test_only_allow_four_clip_subset=True,
                )

    def test_primary_screen_is_exactly_equivalent_to_full_released2(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticBundle(root)
            backend = SyntheticBackend()
            report = fixture.evaluate(backend=backend)
            cache = METRICS.build_released2_real_feature_cache(
                canonical_manifest=fixture.canonical_manifest,
                expected_canonical_manifest_sha256=sha256_file(
                    fixture.canonical_manifest
                ),
                backend=backend,
                split="val",
                expected_clip_count=4,
                formal_mode=False,
                test_only_allow_four_clip_subset=True,
            )
            cache_path = root / "screen-real-cache.json"
            canonical_json_write(cache_path, cache)
            cache_artifact = {
                "path": str(cache_path.resolve()),
                "sha256": sha256_file(cache_path),
                "bytes": cache_path.stat().st_size,
                "receipt_payload_sha256": cache[
                    "receipt_payload_sha256"
                ],
            }
            distribution = fixture.distribution()
            distribution_path = root / "screen-distribution.json"
            canonical_json_write(distribution_path, distribution)
            distribution_artifact = {
                "path": str(distribution_path.resolve()),
                "sha256": sha256_file(distribution_path),
                "bytes": distribution_path.stat().st_size,
                "receipt_payload_sha256": distribution[
                    "receipt_payload_sha256"
                ],
            }
            prediction_artifact = distribution["prediction_manifest"]
            screen = METRICS.build_released2_primary_screen(
                backend,
                canonical_manifest={
                    "path": str(fixture.canonical_manifest.resolve()),
                    "sha256": sha256_file(fixture.canonical_manifest),
                    "bytes": fixture.canonical_manifest.stat().st_size,
                },
                prediction_manifest=prediction_artifact,
                distribution_receipt=distribution_artifact,
                real_feature_cache=cache,
                expected_real_feature_cache_artifact=cache_artifact,
                expected_selection_protocol=report["selection_protocol"],
                expected_split="val",
                expected_clip_count=4,
                test_only_allow_four_clip_subset=True,
            )
            screen_path = root / "primary-screen.json"
            canonical_json_write(screen_path, screen)
            screen_artifact = {
                "path": str(screen_path.resolve()),
                "sha256": sha256_file(screen_path),
                "bytes": screen_path.stat().st_size,
                "receipt_payload_sha256": screen[
                    "receipt_payload_sha256"
                ],
            }
            validated = METRICS.validate_released2_primary_screen_receipt(
                screen_artifact,
                expected_prediction_manifest=prediction_artifact,
                expected_distribution_receipt=distribution_artifact,
                expected_real_feature_cache=cache_artifact,
                expected_canonical_manifest=screen["canonical_manifest"],
                expected_selection_protocol=report["selection_protocol"],
                expected_split="val",
                expected_clip_count=4,
                test_only_allow_four_clip_subset=True,
            )
            released = report["body"]["released2"]
            self.assertEqual(
                screen["generated_feature_statistics"],
                released["feature_statistics"]["generated"],
            )
            self.assertEqual(
                screen["primary_metric"], released["metrics"]["FGD"]
            )
            self.assertEqual(
                validated["generated_feature_statistics"],
                screen["generated_feature_statistics"],
            )

            # Even a one-ULP perturbation can reverse a near-tied checkpoint
            # ordering.  Rehashing the receipt must not turn an approximate
            # comparison into selection authority.
            near_tie = copy.deepcopy(screen)
            near_tie["primary_metric"] = float(
                np.nextafter(
                    np.float64(screen["primary_metric"]),
                    np.float64(np.inf),
                )
            )
            near_tie.pop("receipt_payload_sha256")
            near_tie["receipt_payload_sha256"] = (
                METRICS.canonical_json_sha256(near_tie)
            )
            near_tie_path = root / "near-tie-primary-screen.json"
            canonical_json_write(near_tie_path, near_tie)
            near_tie_artifact = {
                "path": str(near_tie_path.resolve()),
                "sha256": sha256_file(near_tie_path),
                "bytes": near_tie_path.stat().st_size,
                "receipt_payload_sha256": near_tie[
                    "receipt_payload_sha256"
                ],
            }
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "metric derivation mismatch",
            ):
                METRICS.validate_released2_primary_screen_receipt(
                    near_tie_artifact,
                    expected_prediction_manifest=prediction_artifact,
                    expected_distribution_receipt=distribution_artifact,
                    expected_real_feature_cache=cache_artifact,
                    expected_canonical_manifest=screen[
                        "canonical_manifest"
                    ],
                    expected_selection_protocol=report[
                        "selection_protocol"
                    ],
                    expected_split="val",
                    expected_clip_count=4,
                    test_only_allow_four_clip_subset=True,
                )

            forged = copy.deepcopy(report)
            released = forged["body"]["released2"]
            real = METRICS.FeatureMoments.from_json(
                released["feature_statistics"]["real"],
                expected_count=released["counts"]["real_features"],
                label="fixture real",
            )
            dimensions = real.dimension
            fake = METRICS.FeatureMoments()
            fake.update(
                np.full(
                    (
                        released["counts"]["generated_features"],
                        dimensions,
                    ),
                    7.0,
                    dtype=np.float64,
                )
            )
            released["feature_statistics"]["generated"] = fake.to_json()
            released["metrics"]["FGD"] = METRICS.frechet_distance(
                real,
                fake,
            )
            forged.pop("report_payload_sha256")
            forged["report_payload_sha256"] = (
                METRICS.canonical_json_sha256(forged)
            )
            METRICS.validate_report(
                forged,
                expected_split="val",
                expected_clip_count=4,
                expected_prediction_manifest=prediction_artifact,
                expected_distribution_receipt=forged[
                    "distribution_receipt"
                ],
                expected_selection_protocol=forged[
                    "selection_protocol"
                ],
                test_only_allow_four_clip_subset=True,
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "Fresh|fresh",
            ):
                METRICS.fresh_replay_released2_primary(
                    forged,
                    backend,
                    real_feature_cache=cache,
                    expected_real_feature_cache_artifact=cache_artifact,
                    expected_prediction_manifest=prediction_artifact,
                    expected_distribution_receipt=forged[
                        "distribution_receipt"
                    ],
                    expected_selection_protocol=forged[
                        "selection_protocol"
                    ],
                    expected_split="val",
                    expected_clip_count=4,
                    test_only_allow_four_clip_subset=True,
                )

    def test_formal_mode_forbids_injected_audio_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticBundle(Path(directory))
            with (
                mock.patch.object(
                    METRICS,
                    "_require_concrete_formal_backend",
                ),
                mock.patch.dict(
                    METRICS.FORMAL_SPLITS,
                    {
                        "val": {
                            "count": 4,
                            "global_start": 100,
                            "global_stop": 104,
                        }
                    },
                ),
                self.assertRaisesRegex(
                    METRICS.MetricAdapterContractError,
                    "forbids injected audio",
                ),
            ):
                METRICS.evaluate_canonical_bundle(
                    canonical_manifest=fixture.canonical_manifest,
                    expected_canonical_manifest_sha256=sha256_file(
                        fixture.canonical_manifest
                    ),
                    prediction_manifest=fixture.prediction_manifest,
                    expected_prediction_manifest_sha256=sha256_file(
                        fixture.prediction_manifest
                    ),
                    prediction_lineage=fixture.prediction_lineage,
                    expected_prediction_lineage_sha256=sha256_file(
                        fixture.prediction_lineage
                    ),
                    validation_gate=fixture.validation_gate(),
                    distribution_declaration=fixture.distribution(),
                    backend=SyntheticBackend(),
                    split="val",
                    expected_clip_count=4,
                    audio_beat_extractor=lambda _waveform: np.asarray([0.1]),
                    formal_mode=True,
                    test_only_allow_four_clip_gate=False,
                )

    def test_same_path_cached_replication_gate_is_ignored(self) -> None:
        expected = (
            Path(METRICS.__file__).resolve().parent
            / "deterministic_replication_gate.py"
        )
        fake = types.SimpleNamespace(
            __file__=str(expected),
            PAYLOAD_HASH_ALGORITHM=METRICS.PAYLOAD_HASH_ALGORITHM,
            DISTRIBUTION_FORMAT=METRICS.DISTRIBUTION_RECEIPT_FORMAT,
            PROTOCOL=METRICS.DISTRIBUTION_PROTOCOL,
            variation_policy_receipt=lambda: {"poisoned": True},
            validate_distribution_receipt=lambda *_args, **_kwargs: {
                "poisoned": True
            },
            build_distribution_receipt_from_validated_artifacts=(
                lambda *_args, **_kwargs: {"poisoned": True}
            ),
        )
        with mock.patch.dict(
            sys.modules,
            {"scripts.show_base.deterministic_replication_gate": fake},
        ):
            observed = METRICS._replication_gate_module()
        self.assertIsNot(observed, fake)
        self.assertEqual(
            observed.DISTRIBUTION_FORMAT,
            "semtalk_show_deterministic_distribution_receipt_v2",
        )

    def test_standalone_test_cli_is_fail_closed(self) -> None:
        argv = [
            "--canonical-manifest", "/canonical.jsonl",
            "--expected-canonical-manifest-sha256", "1" * 64,
            "--prediction-manifest", "/predictions.jsonl",
            "--expected-prediction-manifest-sha256", "2" * 64,
            "--prediction-lineage", "/lineage.json",
            "--expected-prediction-lineage-sha256", "3" * 64,
            "--validation-gate-json", "/gate.json",
            "--expected-validation-gate-sha256", "4" * 64,
            "--expected-validation-gate-receipt-payload-sha256", "5" * 64,
            "--distribution-declaration-json", "/distribution.json",
            "--expected-distribution-declaration-sha256", "6" * 64,
            "--test-authority-json", "/authority.json",
            "--expected-test-authority-sha256", "7" * 64,
            "--expected-test-authority-bytes", "1",
            "--expected-test-authority-receipt-payload-sha256", "8" * 64,
            "--talkshow-metric-root", "/metric-root",
            "--feature-extractor", "/feature.bin",
            "--smplx-asset", "/SMPLX_NEUTRAL.npz",
            "--device", "cuda:0",
            "--split", "test",
            "--expected-clip-count", "1708",
            "--output-json", "/report.json",
        ]
        with self.assertRaises(SystemExit) as raised:
            METRICS.parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_formal_test_programmatic_api_requires_consumed_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with (
                mock.patch.object(
                    METRICS,
                    "_require_concrete_formal_backend",
                ),
                mock.patch.object(
                    METRICS,
                    "_validated_test_authority",
                    return_value={
                        "expected_output_root": str(root / "inference"),
                        "contract": {"final_metric_event": {}},
                    },
                ),
                self.assertRaisesRegex(
                    METRICS.MetricAdapterContractError,
                    "pre-consumed combined one-shot claim",
                ),
            ):
                METRICS.evaluate_canonical_bundle(
                    canonical_manifest=root / "canonical.jsonl",
                    expected_canonical_manifest_sha256="1" * 64,
                    prediction_manifest=root / "predictions.jsonl",
                    expected_prediction_manifest_sha256="2" * 64,
                    prediction_lineage=root / "lineage.json",
                    expected_prediction_lineage_sha256="3" * 64,
                    validation_gate={},
                    distribution_declaration={},
                    backend=object(),
                    split="test",
                    expected_clip_count=1_708,
                    test_authority={},
                )

    def test_combined_talkshow_suite_start_is_non_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inference_root = root / "inference"
            output_root = root / "metrics"
            inference_root.mkdir()
            output_root.mkdir()
            manifest_path = inference_root / "manifest.jsonl"
            lineage_path = inference_root / "lineage.json"
            paspa_path = output_root / "paspa_diffsheg_show_metrics.json"
            manifest_path.write_bytes(b'{"fixture":true}\n')
            lineage_path.write_bytes(b'{"fixture":true}\n')
            paspa_path.write_bytes(b'{"status":"complete"}\n')
            manifest = {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "bytes": manifest_path.stat().st_size,
            }
            lineage = {
                "path": str(lineage_path),
                "sha256": sha256_file(lineage_path),
                "bytes": lineage_path.stat().st_size,
            }
            authority_receipt = {
                "path": str(root / "authority.json"),
                "sha256": "4" * 64,
                "bytes": 1,
                "receipt_payload_sha256": "5" * 64,
            }
            event = {
                "authorized_events": 1,
                "generation_passes": 1,
                "shared_prediction_bundle": True,
                "single_claim_required": True,
                "all_suites_required": True,
            }
            claim = {
                "format": METRICS.COMBINED_CLAIM_FORMAT,
                "status": "claimed",
                "authority": {"fresh_test_authority": authority_receipt},
                "preflight": {},
                "output_root": str(output_root),
                "test_evaluations": 1,
                "test_feedback_into_selection": False,
                "final_metric_event": event,
                "shared_predictions": {
                    "manifest": manifest,
                    "lineage": lineage,
                },
                "claim_consumed_before_metrics": True,
                "failure_consumes_claim": True,
                "retry_allowed": False,
                "input_set_sha256": "6" * 64,
            }
            claim_path = inference_root / METRICS.COMBINED_CLAIM_NAME
            claim_path.write_bytes(METRICS.canonical_json_bytes(claim))
            claim_path.chmod(0o600)
            claim_receipt = {
                "path": str(claim_path),
                "sha256": sha256_file(claim_path),
                "bytes": claim_path.stat().st_size,
            }
            validated_claim = METRICS._validated_combined_claim(
                claim_receipt,
                test_authority_receipt=authority_receipt,
                validated_test_authority={
                    "expected_output_root": str(inference_root),
                    "contract": {"final_metric_event": event},
                },
                prediction_manifest=manifest_path,
                expected_prediction_manifest_sha256=manifest["sha256"],
                prediction_lineage=lineage_path,
                expected_prediction_lineage_sha256=lineage["sha256"],
            )
            paspa = {
                "path": str(paspa_path),
                "sha256": sha256_file(paspa_path),
                "bytes": paspa_path.stat().st_size,
            }
            kwargs = {
                "claim_artifact": validated_claim[0],
                "claim": validated_claim[1],
                "paspa_report": paspa,
                "test_authority_receipt": authority_receipt,
                "prediction_manifest": manifest_path,
                "expected_prediction_manifest_sha256": manifest["sha256"],
                "prediction_lineage": lineage_path,
                "expected_prediction_lineage_sha256": lineage["sha256"],
            }
            evidence = METRICS._consume_combined_talkshow_suite_start(
                **kwargs
            )
            self.assertEqual(evidence["claim"], claim_receipt)
            self.assertEqual(
                (output_root / METRICS.COMBINED_TALKSHOW_START_NAME)
                .stat()
                .st_mode
                & 0o777,
                0o600,
            )
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "already started; replay refused",
            ):
                METRICS._consume_combined_talkshow_suite_start(**kwargs)

    def test_same_path_cached_final_authority_is_ignored(self) -> None:
        expected = (
            Path(METRICS.__file__).resolve().parent
            / "base_final_authority.py"
        )
        fake = types.SimpleNamespace(
            __file__=str(expected),
            FORMAT="semtalk_show_base_final_test_authority_v1",
            validate_test_authority=lambda *_args, **_kwargs: {
                "poisoned": True
            },
        )
        with mock.patch.dict(
            sys.modules,
            {"scripts.show_base.base_final_authority": fake},
        ):
            observed = METRICS._base_final_authority_module()
        self.assertIsNot(observed, fake)
        self.assertEqual(
            observed.FORMAT,
            "semtalk_show_base_final_test_authority_v2",
        )

    def test_same_path_cached_talkshow_contract_is_ignored(self) -> None:
        expected = (
            Path(METRICS.__file__).resolve().parent
            / "talkshow_base_val_contract.py"
        )
        fake = types.SimpleNamespace(
            __file__=str(expected),
            validate_val_inputs=lambda *_args, **_kwargs: {
                "poisoned": True
            },
            validate_pipeline=lambda *_args, **_kwargs: {
                "poisoned": True
            },
            validate_val_inference_lineage=lambda *_args, **_kwargs: {
                "poisoned": True
            },
        )
        full_name = "scripts.show_base.talkshow_base_val_contract"
        with (
            mock.patch.dict(sys.modules, {full_name: fake}),
            mock.patch.object(
                SHOW_BASE_PACKAGE,
                "talkshow_base_val_contract",
                fake,
                create=True,
            ),
        ):
            observed = METRICS._fresh_base_selector_module()
        self.assertIsNot(observed, fake)
        self.assertTrue(callable(observed.validate_val_inputs))

    def test_verified_snapshot_rejects_symlinked_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            real = root / "real"
            real.mkdir()
            payload = b"immutable"
            target = real / "artifact.json"
            target.write_bytes(payload)
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "canonical and contain no symlink",
            ):
                METRICS._verified_file_snapshot(
                    alias / target.name,
                    hashlib.sha256(payload).hexdigest(),
                    "symlinked fixture",
                )

    def test_verified_snapshot_rejects_post_open_rename_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "artifact.json"
            archived = root / "artifact.opened.json"
            original = b"immutable-original"
            replacement = b"immutable-replaced"
            target.write_bytes(original)
            real_open = os.open
            replaced = False

            def open_then_replace(
                path: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal replaced
                descriptor = real_open(
                    path,
                    flags,
                    mode,
                    dir_fd=dir_fd,
                )
                if (
                    not replaced
                    and path == target.name
                    and dir_fd is not None
                ):
                    target.rename(archived)
                    target.write_bytes(replacement)
                    replaced = True
                return descriptor

            with mock.patch.object(
                METRICS.os,
                "open",
                side_effect=open_then_replace,
            ), self.assertRaisesRegex(
                METRICS.MetricAdapterContractError,
                "changed",
            ):
                METRICS._verified_file_snapshot(
                    target,
                    hashlib.sha256(original).hexdigest(),
                    "rename-raced fixture",
                )
            self.assertTrue(replaced)

    def test_git_porcelain_leading_status_column_is_preserved(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout=" M nets/__init__.py\n?? marker.json\n",
            stderr="",
        )
        with mock.patch.object(
            METRICS.subprocess,
            "run",
            return_value=completed,
        ):
            observed = METRICS._git_output(Path("/metric-root"), "status")
        self.assertEqual(
            observed.splitlines(),
            [" M nets/__init__.py", "?? marker.json"],
        )

    def test_production_dependency_closure_uses_one_combined_evaluator(self) -> None:
        validation = importlib.import_module(
            "scripts.show_base.evaluate_diffsheg_val_fgd"
        )
        final = importlib.import_module(
            "scripts.show_base.evaluate_diffsheg_final_test"
        )
        authority = importlib.import_module(
            "scripts.show_base.base_final_authority"
        )

        # released2 remains callable for validation and is also required in
        # the same final event as PASPA DiffSHEG.  The replay-only CLI stays
        # outside the production closure.
        self.assertEqual(
            METRICS.PRIMARY_METRIC_PATH,
            "body.released2.metrics.FGD",
        )
        self.assertIs(PRIMARY_CLI.metrics, METRICS)

        control_modules = set(authority._CONTROL_DEPENDENCIES)
        for dependencies in authority._CONTROL_DEPENDENCIES.values():
            control_modules.update(dependencies)
        formal_roots = {
            "scripts/show_base/base_final_authority.py",
            "scripts/show_base/base_long_val_contract.py",
            "scripts/show_base/produce_base_val_measurement.py",
            "scripts/show_base/run_base_final_test.py",
            "scripts/show_base/run_base_val_inference.py",
            "scripts/show_base/evaluate_diffsheg_val_fgd.py",
            "scripts/show_base/evaluate_diffsheg_final_test.py",
            *(f"scripts/show_base/{name}.py" for name in control_modules),
        }
        formal_closure = local_show_base_import_closure(formal_roots)
        self.assertEqual(
            INTERNAL_RELEASED2_MODULES & formal_closure,
            {"scripts/show_base/evaluate_talkshow_show_metrics.py"},
        )
        self.assertEqual(
            {
                relative
                for relative in formal_closure
                if Path(relative).name.startswith("evaluate_")
            },
            {
                "scripts/show_base/evaluate_diffsheg_val_fgd.py",
                "scripts/show_base/evaluate_diffsheg_final_test.py",
                "scripts/show_base/evaluate_talkshow_show_metrics.py",
            },
        )
        for launcher_name in (
            "run_base_final_test.sh",
            "run_diffsheg_final_test_eval.sh",
        ):
            launcher = (
                ROOT / "scripts" / "show_base" / launcher_name
            ).read_text(encoding="utf-8")
            self.assertIn("evaluate_diffsheg_final_test.py", launcher)
            # Neither launcher starts a second TalkSHOW CLI process.  The
            # combined producer imports the module from verified source.
            for internal in INTERNAL_RELEASED2_MODULES:
                self.assertNotIn(Path(internal).name, launcher)

        # Validation and final evaluation share one audited PASPA entrypoint
        # and one DiffSHEG statistics/FGD asset lineage.  The combined final
        # event extends that lineage with both the remaining published AEs
        # and the TalkSHOW body/face evaluator over the same predictions.
        self.assertEqual(
            validation.PASPA_EVALUATOR_RELATIVE,
            Path("scripts/diffsheg_show_eval.py"),
        )
        self.assertEqual(
            final.PASPA_EVALUATOR_RELATIVE,
            validation.PASPA_EVALUATOR_RELATIVE,
        )
        self.assertEqual(
            final.PASPA_EVALUATOR_SHA256,
            validation.PASPA_EVALUATOR_SHA256,
        )
        self.assertEqual(
            final.DIFFSHEG_COMMIT,
            validation.DIFFSHEG_REFERENCE_COMMIT,
        )
        self.assertEqual(
            final.DIFFSHEG_STATS_SHA256,
            validation.DIFFSHEG_STATS_SHA256,
        )
        self.assertEqual(
            final.DIFFSHEG_AE_PINS["fgd"]["sha256"],
            validation.DIFFSHEG_GESTURE_AE_SHA256,
        )
        self.assertEqual(set(final.DIFFSHEG_AE_PINS), {"fmd", "fed", "fgd"})
        self.assertEqual(
            final.EXPECTED_METRICS,
            (
                "fmd",
                "fed",
                "expression_diversity",
                "fgd",
                "ba",
                "pcm",
                "gesture_diversity",
            ),
        )
        self.assertEqual(final.TALKSHOW_COMMIT, METRICS.TALKSHOW_METRIC_COMMIT)
        self.assertRegex(final.SMPLX_NEUTRAL_SHA256, r"^[0-9a-f]{64}$")

    def test_generator_identity_filter_is_semantic_not_path_global(self) -> None:
        METRICS._reject_forbidden_generator_identity(
            {
                "metric_assets": {
                    "smplx_asset": (
                        "/frozen/globaldiff-show-evaluator/"
                        "SMPLX_NEUTRAL.npz"
                    ),
                    "talkshow_asset": (
                        "/frozen/globaldiff-show-evaluator/body_ae.pth"
                    ),
                }
            },
            "frozen evaluator",
        )
        with self.assertRaisesRegex(
            METRICS.MetricAdapterContractError,
            "forbidden generator identity",
        ):
            METRICS._reject_forbidden_generator_identity(
                {
                    "checkpoints": {
                        "base": {
                            "path": "/models/globaldiff/speaker2/base.pth"
                        }
                    }
                },
                "generator contract",
            )

    def test_import_closure_matches_frozen_globaldiff_evaluator(self) -> None:
        frozen_evaluator = Path(
            "/private/tmp/globaldiff_show_eval_twostage_20260729/"
            "scripts/show_eval.py"
        )
        upstream_talkshow = Path(
            "/private/tmp/talkshow_upstream_9aef82d_20260731"
        )
        if not frozen_evaluator.is_file() or not upstream_talkshow.is_dir():
            self.skipTest("frozen differential oracle is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            metric_root = Path(directory).resolve() / "TalkSHOW"
            shutil.copytree(upstream_talkshow, metric_root)
            (metric_root / "nets/__init__.py").write_text(
                '"""Metric-only TalkSHOW package."""\n',
                encoding="utf-8",
            )
            specification = importlib.util.spec_from_file_location(
                "frozen_globaldiff_show_eval_for_test",
                frozen_evaluator,
            )
            if specification is None or specification.loader is None:
                self.fail("cannot load frozen GlobalDiff evaluator oracle")
            frozen = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(frozen)
            expected = frozen._talkshow_fgd_import_closure(metric_root)
            observed = METRICS._talkshow_fgd_import_closure(metric_root)
            self.assertEqual(observed, expected)
            self.assertEqual(observed, METRICS.TALKSHOW_FGD_SOURCE_FILES)

    def test_atomic_write_preserves_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                METRICS._atomic_write_new(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"existing")

    def test_validate_report_returns_released2_primary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticBundle(root)
            report = fixture.evaluate(
                backend=ShapeRecordingCudaBackend(),
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
            report["metric_assets"]["execution_device"] = "cuda:0"
            report["runtime"] = {
                "python": "3.12-test",
                "numpy": np.__version__,
                "torch": "2.test",
                "smplx": "synthetic-test",
                "librosa": "synthetic-test",
                "soundfile": "synthetic-test",
                "soxr": "synthetic-test",
                "scipy": "synthetic-test",
                "cuda": "12.test",
                "cudnn": "9.test",
                "device": "cuda:0",
                "device_type": "cuda",
                "device_index": 0,
                "device_name": "Synthetic H200",
            }
            report["formal_mode"] = True
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
                mock.patch.dict(
                    METRICS.FORMAL_SPLITS,
                    {
                        "val": {
                            "count": 4,
                            "global_start": 100,
                            "global_stop": 104,
                        }
                    },
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
                tampered_reports = []
                fgd_tamper = copy.deepcopy(report)
                fgd_tamper["body"]["released2"]["feature_statistics"][
                    "generated"
                ]["sum"][0] += 1.0
                tampered_reports.append(fgd_tamper)
                variation_tamper = copy.deepcopy(report)
                variation_tamper["body"]["released2"][
                    "primitive_receipt"
                ]["variation_sum"] += 1.0
                tampered_reports.append(variation_tamper)
                bc_tamper = copy.deepcopy(report)
                bc_tamper["body"]["released2"]["primitive_receipt"][
                    "bc_numerator"
                ] += 1.0
                tampered_reports.append(bc_tamper)
                for tampered in tampered_reports:
                    tampered.pop("report_payload_sha256")
                    tampered["report_payload_sha256"] = (
                        METRICS.canonical_json_sha256(tampered)
                    )
                    with self.assertRaises(
                        METRICS.MetricAdapterContractError
                    ):
                        METRICS.validate_report(
                            tampered,
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
            )
        self.assertFalse(report["formal_mode"])
        self.assertEqual(report["runtime"]["device"], "cpu")
        self.assertEqual(
            backend.feature_call_shapes,
            [
                shape
                for _clip in range(4)
                for shape in (
                    (1, 64, 265),
                    (2, 64, 265),
                    (16, 64, 265),
                )
            ],
        )
        self.assertEqual(
            backend.joint_call_shapes,
            [
                shape
                for _clip in range(4)
                for shape in (
                    (16, 64, 265),
                    (2, 64, 265),
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
                "formal evaluation clip count",
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
        for value in (released, paper):
            self.assertLessEqual(
                value["primitive_receipt"]["variation_sum"],
                value["primitive_receipt"][
                    "variation_integrity_tolerance_sum"
                ],
            )
            self.assertEqual(
                value["metrics"]["Variation"],
                value["primitive_receipt"]["variation_sum"] / 4,
            )
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
