from __future__ import annotations

import hashlib
import importlib.util
import io
import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "show_base"
    / "run_base_inference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_base_inference_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "show_base"
    / "build_base_features.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_base_features_under_test",
    BUILDER_PATH,
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


class OutputNpzValidationTest(unittest.TestCase):
    def arrays(self, *, frames: int = 88) -> dict[str, np.ndarray]:
        return MODULE._output_arrays(
            betas=np.zeros((MODULE.BETA_DIM,), dtype=np.float32),
            poses=np.zeros((frames, MODULE.POSE_DIM), dtype=np.float32),
            expressions=np.zeros(
                (frames, MODULE.EXPRESSION_DIM),
                dtype=np.float32,
            ),
            trans=np.zeros((frames, 3), dtype=np.float32),
        )

    def test_valid_prediction_reopens_with_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "res_clip.npz"
            path.write_bytes(MODULE.deterministic_npz_bytes(self.arrays()))
            arrays = MODULE._load_and_validate_output_npz(
                path,
                frames=88,
                prediction=True,
            )
            self.assertEqual(tuple(arrays), MODULE.OUTPUT_FIELDS)

    def test_prediction_rejects_nonzero_eye_pose(self) -> None:
        arrays = self.arrays()
        arrays["poses"][0, 69] = np.float32(1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "res_clip.npz"
            path.write_bytes(MODULE.deterministic_npz_bytes(arrays))
            with self.assertRaises(MODULE.InferenceContractError):
                MODULE._load_and_validate_output_npz(
                    path,
                    frames=88,
                    prediction=True,
                )

    def test_output_rejects_wrong_float_dtype(self) -> None:
        arrays = self.arrays()
        arrays["poses"] = arrays["poses"].astype(np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt_clip.npz"
            path.write_bytes(MODULE.deterministic_npz_bytes(arrays))
            with self.assertRaises(MODULE.InferenceContractError):
                MODULE._load_and_validate_output_npz(
                    path,
                    frames=88,
                    prediction=False,
                )


class PublicationPrimitiveTest(unittest.TestCase):
    def test_copy_is_fsynced_independent_inode(self) -> None:
        payload = b"frozen-shard-output" * 1024
        expected_sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            destination = root / "destination.npz"
            source.write_bytes(payload)
            receipt = MODULE._copy_file_fsync_new(
                source,
                destination,
                expected_sha256=expected_sha,
                expected_bytes=len(payload),
            )
            self.assertEqual(receipt["sha256"], expected_sha)
            self.assertNotEqual(
                (source.stat().st_dev, source.stat().st_ino),
                (destination.stat().st_dev, destination.stat().st_ino),
            )
            source.write_bytes(b"changed")
            self.assertEqual(destination.read_bytes(), payload)

    def test_finalize_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with MODULE._finalize_lock(output_root):
                with self.assertRaises(MODULE.InferenceContractError):
                    with MODULE._finalize_lock(output_root):
                        pass


class CrossStageReceiptTest(unittest.TestCase):
    def test_audio_contract_hash_matches_inference_canonical_json(self) -> None:
        payload = {
            "format": "semtalk_show_audio_lineage_contract_v1",
            "nested": {"speaker_map": {"oliver": 0, "conan": 3}},
            "values": [1, 2, 3],
        }
        self.assertEqual(
            BUILDER.canonical_file_payload_sha256(payload),
            MODULE.canonical_json_sha256(payload),
        )

    def test_training_receipt_hash_matches_inference_compact_json(self) -> None:
        payload = {
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "commit": "a" * 40,
            "tree": "b" * 40,
            "entrypoint": "/immutable/show_base_train.py",
        }
        self.assertEqual(
            BUILDER.compact_payload_sha256(payload),
            MODULE.compact_json_sha256(payload),
        )


class CanonicalAudioMetadataTest(unittest.TestCase):
    @staticmethod
    def valid_row() -> dict[str, object]:
        return {
            "wav_channels": 2,
            "wav_sample_width": 2,
            "wav_sample_rate": 22_000,
            "wav_frames": 132_000,
            "wav_mono_policy": (
                "librosa.load(sr=None,mono=True):arithmetic_channel_mean"
            ),
        }

    def test_feature_and_inference_consumers_accept_exact_protocol(self) -> None:
        row = self.valid_row()
        BUILDER.validate_canonical_audio_metadata(row, "builder")
        MODULE._validate_canonical_audio_metadata(row, "inference")

    def test_feature_and_inference_consumers_reject_tampering(self) -> None:
        for key, value in (
            ("wav_channels", 3),
            ("wav_sample_width", 3),
            ("wav_sample_rate", 16_000),
            ("wav_mono_policy", "different"),
        ):
            row = self.valid_row()
            row[key] = value
            with self.subTest(consumer="builder", key=key):
                with self.assertRaises(RuntimeError):
                    BUILDER.validate_canonical_audio_metadata(
                        row,
                        "builder",
                    )
            with self.subTest(consumer="inference", key=key):
                with self.assertRaises(MODULE.InferenceContractError):
                    MODULE._validate_canonical_audio_metadata(
                        row,
                        "inference",
                    )


class OptionalRenderingDependencyTest(unittest.TestCase):
    def test_fast_render_is_lazy_in_training_utility_modules(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        expected_render_functions = {
            "utils/other_tools.py": {"render_one_sequence"},
            "utils/other_tools_hf.py": {
                "render_one_sequence",
                "render_one_sequence_no_gt",
            },
        }
        for relative, function_names in expected_render_functions.items():
            tree = ast.parse(
                (repository / relative).read_text(encoding="utf-8"),
                filename=relative,
            )
            top_level_fast_render = [
                node
                for node in tree.body
                if isinstance(node, ast.Import)
                and any(
                    alias.name == "utils.fast_render"
                    for alias in node.names
                )
            ]
            self.assertEqual(top_level_fast_render, [], relative)
            functions = {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            for function_name in function_names:
                imports = [
                    alias.name
                    for node in ast.walk(functions[function_name])
                    if isinstance(node, ast.Import)
                    for alias in node.names
                ]
                self.assertIn(
                    "utils.fast_render",
                    imports,
                    f"{relative}:{function_name}",
                )


class OptionalTextDependencyTest(unittest.TestCase):
    def test_core_motion_layers_do_not_import_unused_vocab_stack(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        relative = "models/utils/layer.py"
        tree = ast.parse(
            (repository / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        build_vocab_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "build_vocab"
        ]
        self.assertEqual(build_vocab_imports, [])
        vocab_references = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "Vocab"
        ]
        self.assertEqual(vocab_references, [])


class VerifiedInputSnapshotTest(unittest.TestCase):
    @staticmethod
    def canonical_payload(frames: int = 8) -> bytes:
        arrays = {
            "pose": np.zeros((frames, 165), dtype=np.float32),
            "contact": np.zeros((frames, 4), dtype=np.float32),
            "facial": np.zeros((frames, 100), dtype=np.float32),
            "beta": np.zeros((frames, 300), dtype=np.float32),
            "trans": np.zeros((frames, 3), dtype=np.float32),
            "speaker_id": np.zeros((frames, 1), dtype=np.int64),
        }
        with io.BytesIO() as handle:
            np.savez(handle, **arrays)
            return handle.getvalue()

    def test_canonical_decode_uses_the_verified_byte_snapshot(self) -> None:
        payload = self.canonical_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.npz"
            path.write_bytes(payload)
            row = {
                "canonical_npz": str(path),
                "canonical_npz_sha256": hashlib.sha256(payload).hexdigest(),
                "frames": 8,
            }
            original_read_bytes = Path.read_bytes

            def read_then_replace(target: Path) -> bytes:
                observed = original_read_bytes(target)
                target.write_bytes(b"changed-after-the-single-read")
                return observed

            with mock.patch.object(
                Path,
                "read_bytes",
                new=read_then_replace,
            ):
                arrays, frames = BUILDER.load_canonical_clip(row)
            self.assertEqual(frames, 8)
            self.assertEqual(arrays["pose"].shape, (8, 165))
            self.assertEqual(
                path.read_bytes(),
                b"changed-after-the-single-read",
            )

    def test_manifest_symlink_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "manifest.jsonl"
            target.write_text("{}\n")
            symlink = root / "manifest-link.jsonl"
            symlink.symlink_to(target)
            with self.assertRaises(RuntimeError):
                BUILDER.load_jsonl([symlink])


if __name__ == "__main__":
    unittest.main()
