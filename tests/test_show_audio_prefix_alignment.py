from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load(
    "scripts/show_base/build_base_features.py",
    "show_audio_prefix_builder_under_test",
)
INFERENCE = _load(
    "scripts/show_base/run_base_inference.py",
    "show_audio_prefix_inference_under_test",
)


class PublicAudioPrefixAlignmentTest(unittest.TestCase):
    @staticmethod
    def _audio_row() -> dict[str, object]:
        return {
            "format": "semtalk_show_audio_clip_v1",
            "split": "train",
            "clip_id": "oliver/example",
            "canonical_npz": "/frozen/canonical.npz",
            "canonical_npz_sha256": "a" * 64,
            "source_wav": "/frozen/source.wav",
            "source_wav_sha256": "b" * 64,
            "audio_feature_npz": "/frozen/audio.npz",
            "audio_feature_npz_sha256": "c" * 64,
            "frames": 62,
            "lineage_contract_sha256": "d" * 64,
            "audio_lineage_contract_sha256": "e" * 64,
            "shard_id": 0,
            "num_shards": 8,
            "source_audio_field": "source_wav",
            "audio_samples_16k": 160_000,
            "native_30fps_frames": 300,
            "canonical_usable_frames": 60,
            "discarded_source_30fps_frames": 238,
            "edge_padded_tail_frames": 0,
            "source_sample_rate": 48_000,
            "hubert_native_frames": 499,
            "beat_shape": [62, 3],
            "hubert_shape": [62, 1024],
        }

    def test_long_audio_is_exact_leading_prefix(self) -> None:
        native = np.arange(300 * 3, dtype=np.float32).reshape(300, 3)
        for target in (62, 163, 230):
            with self.subTest(target=target):
                aligned, timing = BUILDER.align_public_audio_prefix(
                    native,
                    target_frames=target,
                    max_frame_mismatch=1,
                )
                self.assertTrue(np.array_equal(aligned, native[:target]))
                self.assertEqual(aligned.dtype, native.dtype)
                self.assertEqual(
                    timing,
                    {
                        "canonical_frames": target,
                        "canonical_usable_frames": (target // 30) * 30,
                        "native_30fps_frames": 300,
                        "discarded_source_30fps_frames": 300 - target,
                        "edge_padded_tail_frames": 0,
                    },
                )

    def test_exact_length_is_value_and_dtype_exact(self) -> None:
        native = np.arange(64 * 2, dtype=np.float64).reshape(64, 2)
        aligned, timing = BUILDER.align_public_audio_prefix(
            native,
            target_frames=64,
            max_frame_mismatch=1,
        )
        self.assertTrue(np.array_equal(aligned, native))
        self.assertEqual(aligned.dtype, native.dtype)
        self.assertEqual(timing["discarded_source_30fps_frames"], 0)
        self.assertEqual(timing["edge_padded_tail_frames"], 0)

    def test_one_frame_shortfall_within_same_whole_second_edge_pads(self) -> None:
        native = np.arange(162 * 2, dtype=np.float32).reshape(162, 2)
        aligned, timing = BUILDER.align_public_audio_prefix(
            native,
            target_frames=163,
            max_frame_mismatch=1,
        )
        self.assertTrue(np.array_equal(aligned[:162], native))
        self.assertTrue(np.array_equal(aligned[162], native[-1]))
        self.assertEqual(timing["edge_padded_tail_frames"], 1)
        self.assertEqual(timing["discarded_source_30fps_frames"], 0)

    def test_material_shortfall_is_rejected(self) -> None:
        native = np.zeros((149, 3), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "prefix mismatch"):
            BUILDER.align_public_audio_prefix(
                native,
                target_frames=163,
                max_frame_mismatch=1,
            )

    def test_one_frame_shortfall_cannot_drop_a_whole_second(self) -> None:
        native = np.zeros((299, 3), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "prefix mismatch"):
            BUILDER.align_public_audio_prefix(
                native,
                target_frames=300,
                max_frame_mismatch=1,
            )

    def test_nonfinite_native_feature_is_rejected(self) -> None:
        native = np.zeros((30, 3), dtype=np.float32)
        native[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            BUILDER.align_public_audio_prefix(
                native,
                target_frames=30,
                max_frame_mismatch=1,
            )

    def test_builder_and_inference_freeze_identical_protocol(self) -> None:
        self.assertEqual(
            BUILDER.AUDIO_ALIGNMENT_PROTOCOL,
            INFERENCE.AUDIO_ALIGNMENT_PROTOCOL,
        )
        self.assertEqual(
            BUILDER.AUDIO_ALIGNMENT_PROTOCOL["canonical_alignment"],
            "leading_prefix",
        )

    def test_consumers_reject_sample_count_timing_tamper(self) -> None:
        row = self._audio_row()
        row["audio_samples_16k"] = 159_999
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "invalid public-prefix timing receipt",
            ):
                BUILDER.load_audio_rows([manifest])

            inference_row = dict(row)
            inference_row["split"] = "test"
            canonical = {
                "clip_id": inference_row["clip_id"],
                "canonical_npz": inference_row["canonical_npz"],
                "canonical_npz_sha256": inference_row[
                    "canonical_npz_sha256"
                ],
                "source_wav": inference_row["source_wav"],
                "source_wav_sha256": inference_row["source_wav_sha256"],
                "frames": inference_row["frames"],
                "lineage_contract_sha256": inference_row[
                    "lineage_contract_sha256"
                ],
            }
            manifest.write_text(
                json.dumps(inference_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                INFERENCE.InferenceContractError,
                "invalid public-prefix timing receipt",
            ):
                INFERENCE._audio_feature_rows(
                    [manifest],
                    {str(inference_row["clip_id"]): canonical},
                )

    def test_hubert_is_first_aligned_to_full_audio_duration(self) -> None:
        source = inspect.getsource(BUILDER.audio_mode)
        native_alignment = (
            "hubert_native_30fps = align_hubert(\n"
            "                    native_hubert,\n"
            '                    timing["native_30fps_frames"],'
        )
        self.assertIn(native_alignment, source)
        self.assertNotIn("align_hubert(native_hubert, frames)", source)
        self.assertLess(
            source.index("hubert_native_30fps = align_hubert("),
            source.index(
                "hubert, hubert_timing = align_public_audio_prefix("
            ),
        )

    def test_rhythm_long_audio_matches_full_feature_prefix(self) -> None:
        try:
            import librosa  # noqa: F401
        except ImportError:
            self.skipTest("optional librosa test dependency unavailable")
        samples = 33_600
        axis = np.arange(samples, dtype=np.float32)
        speech = (
            0.1 * np.sin(axis * np.float32(0.013))
            + 0.03 * np.sin(axis * np.float32(0.031))
        ).astype(np.float32)
        full, full_timing = BUILDER.semtalk_rhythm_features(
            speech,
            target_frames=63,
            max_frame_mismatch=1,
        )
        prefix, prefix_timing = BUILDER.semtalk_rhythm_features(
            speech,
            target_frames=32,
            max_frame_mismatch=1,
        )
        self.assertEqual(full_timing["native_30fps_frames"], 63)
        self.assertEqual(prefix_timing["native_30fps_frames"], 63)
        self.assertTrue(np.array_equal(prefix, full[:32]))
        self.assertEqual(
            prefix_timing["discarded_source_30fps_frames"],
            31,
        )


if __name__ == "__main__":
    unittest.main()
