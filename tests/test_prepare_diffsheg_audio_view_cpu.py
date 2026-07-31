from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import wave

from scripts.show_base import evaluate_diffsheg_final_test as closure
from scripts.show_base import prepare_diffsheg_audio_view as prepare


class PrepareDiffSHEGAudioViewCpuTests(unittest.TestCase):
    def _wav(self, path: Path, value: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(bytes([value, 0]) * 160)

    def _fixture(self, root: Path) -> tuple[SimpleNamespace, dict, dict, list[dict]]:
        source_root = (root / "source").resolve()
        source_root.mkdir()
        source_ids = ("chemistry/video0/seq_a", "conan/video1/seq_b")
        clip_ids = ("chemistry__seq_a", "conan__seq_b")
        canonical_rows = []
        inference_rows = []
        for index, (source_id, clip_id) in enumerate(zip(source_ids, clip_ids)):
            speaker, video, sequence = source_id.split("/")
            self._wav(
                source_root / speaker / video / sequence / f"{sequence}.wav",
                index + 1,
            )
            canonical_rows.append(
                {
                    "global_index": index,
                    "clip_id": source_id,
                    "frames": 90,
                }
            )
            inference_rows.append(
                {
                    "canonical_clip_id": clip_id,
                    "source_clip_id": source_id,
                }
            )
        inference_manifest = root / "final_manifest.jsonl"
        inference_manifest.write_bytes(
            b"".join(closure.canonical_json_bytes(row) for row in inference_rows)
        )
        clip_manifest = root / "diffsheg_eval_clip_ids.txt"
        clip_manifest.write_text(
            "".join(f"{clip_id}\n" for clip_id in clip_ids),
            encoding="utf-8",
        )
        inference = {
            "manifest": {
                "path": str(inference_manifest.resolve()),
                "bytes": inference_manifest.stat().st_size,
                "sha256": closure.sha256_file(inference_manifest),
            },
            "clip_manifest": {
                "path": str(clip_manifest.resolve()),
                "bytes": clip_manifest.stat().st_size,
                "sha256": closure.sha256_file(clip_manifest),
            },
        }
        authority = {
            "canonical": {
                "manifest": {
                    "path": "/frozen/canonical_test_manifest.jsonl",
                    "bytes": 123,
                    "sha256": "c" * 64,
                }
            }
        }
        args = SimpleNamespace(
            inference_final_root=root / "inference",
            source_audio_root=source_root,
            output_root=(root / "prepared_view").resolve(),
        )
        return args, authority, inference, canonical_rows

    def test_builds_exact_paspa_test_layout_and_seals_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args, authority, inference, canonical_rows = self._fixture(root)
            clip_ids = ("chemistry__seq_a", "conan__seq_b")
            with (
                mock.patch.object(closure, "EXPECTED_TEST_CLIPS", 2),
                mock.patch.object(
                    closure,
                    "_load_current_authority",
                    return_value=(authority, {}),
                ),
                mock.patch.object(
                    closure,
                    "_validate_inference_bundle",
                    return_value=(inference, list(clip_ids)),
                ),
                mock.patch.object(
                    prepare.final_test,
                    "_load_inputs",
                    return_value=(canonical_rows, {}, {}),
                ),
            ):
                result = prepare.build_view(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["clip_count"], 2)
            for source_id, clip_id in zip(
                ("chemistry/video0/seq_a", "conan/video1/seq_b"),
                clip_ids,
            ):
                speaker, video, sequence = source_id.split("/")
                source = (
                    args.source_audio_root
                    / speaker
                    / video
                    / sequence
                    / f"{sequence}.wav"
                )
                view = (
                    args.output_root
                    / speaker
                    / video
                    / "test"
                    / sequence
                    / f"{sequence}.wav"
                )
                self.assertTrue(view.is_file(), clip_id)
                self.assertEqual(source.read_bytes(), view.read_bytes())
            receipt = json.loads(
                (args.output_root / closure.AUDIO_VIEW_RECEIPT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["clip_count"], 2)
            self.assertTrue(receipt["exact_once"])
            self.assertEqual(receipt["padding"], "forbidden")
            self.assertRegex(receipt["source_ordered_set_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_symlinked_original_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args, authority, inference, canonical_rows = self._fixture(root)
            source = args.source_audio_root / "chemistry/video0/seq_a/seq_a.wav"
            replacement = root / "replacement.wav"
            replacement.write_bytes(source.read_bytes())
            source.unlink()
            source.symlink_to(replacement)
            with (
                mock.patch.object(closure, "EXPECTED_TEST_CLIPS", 2),
                mock.patch.object(
                    closure,
                    "_load_current_authority",
                    return_value=(authority, {}),
                ),
                mock.patch.object(
                    closure,
                    "_validate_inference_bundle",
                    return_value=(
                        inference,
                        ["chemistry__seq_a", "conan__seq_b"],
                    ),
                ),
                mock.patch.object(
                    prepare.final_test,
                    "_load_inputs",
                    return_value=(canonical_rows, {}, {}),
                ),
                self.assertRaisesRegex(prepare.AudioViewError, "symlink"),
            ):
                prepare.build_view(args)


if __name__ == "__main__":
    unittest.main()
