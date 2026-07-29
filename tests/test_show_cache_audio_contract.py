from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "show_base"
    / "build_show_cache.py"
)
MODULE_NAME = "build_show_cache_audio_contract_under_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)


def write_pcm_wav(
    path: Path,
    channels: int,
    *,
    rate: int = MODULE.SOURCE_AUDIO_SAMPLE_RATE,
    sample_width: int = MODULE.SOURCE_AUDIO_SAMPLE_WIDTH,
) -> None:
    values = tuple(
        ((channel + 1) * 1000) * (-1 if channel % 2 else 1)
        for channel in range(channels)
    )
    frame = struct.pack("<" + ("h" * channels), *values)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(rate)
        handle.writeframes(frame * 32)


class ShowAudioContractTest(unittest.TestCase):
    def test_stereo_source_is_accepted_with_explicit_mono_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            write_pcm_wav(path, channels=2)
            metadata = MODULE.inspect_wav(
                path,
                expected_rate=MODULE.SOURCE_AUDIO_SAMPLE_RATE,
            )
        self.assertEqual(metadata["wav_channels"], 2)
        self.assertEqual(
            metadata["wav_mono_policy"],
            MODULE.AUDIO_MONO_POLICY,
        )

    def test_mono_source_remains_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mono.wav"
            write_pcm_wav(path, channels=1)
            metadata = MODULE.inspect_wav(
                path,
                expected_rate=MODULE.SOURCE_AUDIO_SAMPLE_RATE,
            )
        self.assertEqual(metadata["wav_channels"], 1)

    def test_more_than_two_channels_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "three-channel.wav"
            write_pcm_wav(path, channels=3)
            with self.assertRaises(MODULE.ShowCacheError):
                MODULE.inspect_wav(
                    path,
                    expected_rate=MODULE.SOURCE_AUDIO_SAMPLE_RATE,
                )

    def test_noncanonical_sample_width_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eight-bit.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(1)
                handle.setframerate(MODULE.SOURCE_AUDIO_SAMPLE_RATE)
                handle.writeframes(b"\x80\x80" * 32)
            with self.assertRaises(MODULE.ShowCacheError):
                MODULE.inspect_wav(
                    path,
                    expected_rate=MODULE.SOURCE_AUDIO_SAMPLE_RATE,
                )

    def test_librosa_mono_true_is_bit_exact_channel_mean(self) -> None:
        try:
            import librosa
        except ImportError as exc:
            self.skipTest(f"librosa is unavailable: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            write_pcm_wav(path, channels=2)
            payload = path.read_bytes()
            mono, mono_rate = librosa.load(
                io.BytesIO(payload),
                sr=None,
                mono=True,
            )
            multi, multi_rate = librosa.load(
                io.BytesIO(payload),
                sr=None,
                mono=False,
            )
        self.assertEqual(
            mono_rate,
            MODULE.SOURCE_AUDIO_SAMPLE_RATE,
        )
        self.assertEqual(multi_rate, mono_rate)
        self.assertTrue(
            np.array_equal(mono, np.mean(multi, axis=0))
        )

    def test_lineage_records_source_and_target_audio_protocols(self) -> None:
        contract = MODULE.lineage_contract(
            split_root=Path("/frozen/split"),
            receipt={
                "split_sha256": "a" * 64,
                "counts": dict(MODULE.EXPECTED_SPLIT_COUNTS),
                "missing_count": MODULE.EXPECTED_MISSING_COUNT,
            },
            receipt_path=Path("/frozen/split/receipt.json"),
            receipt_sha256="b" * 64,
            hand_component_path=Path("/frozen/hand.json"),
            hand_component_sha256="c" * 64,
            smplx_asset_path=Path("/frozen/SMPLX_NEUTRAL_2020.npz"),
            smplx_sha256="d" * 64,
            expected_source_audio_rate=MODULE.SOURCE_AUDIO_SAMPLE_RATE,
            source={"origin": MODULE.EXPECTED_ORIGIN},
        )
        self.assertEqual(
            contract["source_audio_sample_rate"],
            MODULE.SOURCE_AUDIO_SAMPLE_RATE,
        )
        self.assertEqual(
            contract["hubert_target_sample_rate"],
            MODULE.HUBERT_AUDIO_SAMPLE_RATE,
        )
        self.assertEqual(
            contract["audio_channel_protocol"]["manifest_policy"],
            MODULE.AUDIO_MONO_POLICY,
        )

    def test_manifest_audio_metadata_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            write_pcm_wav(path, channels=2)
            observed = MODULE.inspect_wav(
                path,
                expected_rate=MODULE.SOURCE_AUDIO_SAMPLE_RATE,
            )
        MODULE.validate_wav_manifest_metadata(
            dict(observed),
            observed,
            "valid row",
        )
        for key, bad_value in (
            ("wav_channels", 1),
            ("wav_mono_policy", "untracked_policy"),
        ):
            tampered = dict(observed)
            tampered[key] = bad_value
            with self.subTest(key=key):
                with self.assertRaises(MODULE.ShowCacheError):
                    MODULE.validate_wav_manifest_metadata(
                        tampered,
                        observed,
                        "tampered row",
                    )


if __name__ == "__main__":
    unittest.main()
