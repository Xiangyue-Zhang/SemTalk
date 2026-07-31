from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import wave


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "show_base" / "evaluate_diffsheg_final_test.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_diffsheg_final_test_cpu_contract_v2",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FinalDiffSHEGConstantsTests(unittest.TestCase):
    def test_protocol_pins_exact_public_authorities(self) -> None:
        self.assertEqual(BRIDGE.EXPECTED_TEST_CLIPS, 1_708)
        self.assertEqual(BRIDGE.EXPECTED_NUM_SHARDS, 8)
        self.assertEqual((BRIDGE.WINDOW_LENGTH, BRIDGE.WINDOW_STRIDE), (88, 88))
        self.assertEqual(
            BRIDGE.EXPECTED_METRICS,
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
        self.assertEqual(
            set(BRIDGE.DIFFSHEG_AE_PINS),
            {"fmd", "fed", "fgd"},
        )
        self.assertEqual(
            BRIDGE.PASPA_COMMIT,
            "0df27e6cab4b5ced19cc923afe352f77d547924b",
        )
        self.assertEqual(
            BRIDGE.PASPA_EVALUATOR_SHA256,
            "21fa84fdb9c3f64eb2920714e1d27a685210a225bcffa78503452c4018a53f8c",
        )
        self.assertEqual(
            BRIDGE.PASPA_PROTOCOL_DOC_SHA256,
            "c6ec216581a519e9f9ef45e8d280b796ece289164af770086b8a319fa22541ce",
        )
        self.assertEqual(
            BRIDGE.SMPLX_NEUTRAL_SHA256,
            "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74",
        )

    def test_guarded_launcher_excludes_compatibility_protocols(self) -> None:
        source = (
            ROOT
            / "scripts"
            / "show_base"
            / "run_diffsheg_final_test_eval.sh"
        ).read_text(encoding="utf-8")
        for token in (
            "--skip-ba",
            "--window-stride",
            "released2",
            "paper16",
            "speaker2",
            "/tmp/globaldiff_guarded_runner.py",
        ):
            self.assertIn(token, source)

    def test_manifest_name_is_uniform(self) -> None:
        producer = (
            ROOT / "scripts" / "show_base" / "run_base_val_inference.py"
        ).read_text(encoding="utf-8")
        contract = (
            ROOT / "scripts" / "show_base" / "talkshow_base_val_contract.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("talkshow_eval_clip_ids.txt", producer)
        self.assertNotIn("talkshow_eval_clip_ids.txt", contract)
        self.assertIn("diffsheg_eval_clip_ids.txt", producer)
        self.assertIn("diffsheg_eval_clip_ids.txt", contract)


class CurrentAuthorityTests(unittest.TestCase):
    def test_replays_current_authority_and_binds_selected_six(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            final_root = root / "final"
            final_root.mkdir()
            checkpoint = root / "base.pth"
            checkpoint.write_bytes(b"base winner")
            selected = {
                "path": str(checkpoint),
                "sha256": BRIDGE.sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
            checkpoints = {
                stage: {
                    "path": str(root / f"{stage}.pth"),
                    "sha256": hashlib.sha256(stage.encode()).hexdigest(),
                    "bytes": 1,
                }
                for stage in BRIDGE.final_test.CHECKPOINT_STAGES
            }
            checkpoints["base"] = selected
            winner = {
                "path": str(root / "selection.json"),
                "sha256": "1" * 64,
                "bytes": 1,
                "receipt_payload_sha256": "2" * 64,
                "canonical_payload_sha256": "3" * 64,
                "selected_epoch": 400,
                "selected_optimizer_updates": 99_200,
                "selected_checkpoint": selected,
            }
            source = {
                "source_root": str(ROOT),
                "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                "commit": "4" * 40,
                "tree": "5" * 40,
                "entrypoint": "scripts/show_base/semtalk_base_inference_core.py",
                "entrypoint_sha256": "6" * 64,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            }
            policy = {
                "authorized_evaluations": 1,
                "one_shot_claim_required": True,
                "selection_feedback": False,
                "num_shards": 8,
                "canonical_test_clips": 1_708,
            }
            authority = {
                "winner_selection": winner,
                "test_claim": {"test_policy": policy},
                "checkpoints": checkpoints,
                "inference_source": source,
            }
            args = SimpleNamespace(inference_final_root=final_root)
            artifact = {
                "path": str(root / "authority.json"),
                "sha256": "7" * 64,
                "bytes": 1,
                "receipt_payload_sha256": "8" * 64,
            }
            with (
                mock.patch.object(
                    BRIDGE.final_test,
                    "_validated_authority",
                    return_value=authority,
                ),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_output_roots",
                    return_value=(
                        final_root.resolve(),
                        (root / "shards").resolve(),
                    ),
                ),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_authority_artifact",
                    return_value=artifact,
                ),
            ):
                observed, receipt = BRIDGE._load_current_authority(args)
            self.assertIs(observed, authority)
            self.assertEqual(receipt["fresh_test_authority"], artifact)
            self.assertEqual(
                receipt["selected_checkpoint"]["sha256"],
                selected["sha256"],
            )
            self.assertEqual(receipt["selection"]["split"], "val")
            self.assertTrue(
                receipt["selection"]["validation_only_for_selection"]
            )
            self.assertFalse(
                receipt["selection"]["test_feedback_into_selection"]
            )
            self.assertEqual(
                set(receipt["selected_prerequisite_sha256"]),
                set(BRIDGE.final_test.REPRESENTATION_STAGES),
            )


class FinalInferencePreflightTests(unittest.TestCase):
    def _bundle(
        self,
        root: Path,
        *,
        bad_frames: bool = False,
    ) -> tuple[SimpleNamespace, dict, dict, list[dict]]:
        final_root = root / "final"
        npz_root = final_root / "npz" / "test"
        npz_root.mkdir(parents=True)
        ids = (
            "chemistry__0000",
            "conan__0000",
            "oliver__0000",
            "seth__0000",
        )
        rows: list[dict] = []
        for index, clip_id in enumerate(ids):
            speaker = clip_id.split("__", 1)[0]
            rows.append(
                {
                    "global_index": 15_402 + index,
                    "source_clip_id": f"{speaker}/video/clip",
                    "canonical_clip_id": clip_id,
                    "speaker": speaker,
                    "speaker_id": index,
                    "frames": 87 if bad_frames and index == 0 else 88,
                    "canonical_npz": f"/canonical/{clip_id}.npz",
                    "canonical_npz_sha256": "a" * 64,
                    "audio_feature_npz": f"/audio/{clip_id}.npz",
                    "audio_feature_npz_sha256": "b" * 64,
                    "prediction": {"path": "/p", "sha256": "c" * 64, "bytes": 1},
                    "ground_truth": {"path": "/g", "sha256": "d" * 64, "bytes": 1},
                    "evaluation_index": index,
                }
            )
        manifest_path = final_root / "final_manifest.jsonl"
        manifest_path.write_bytes(
            b"".join(BRIDGE.canonical_json_bytes(row) for row in rows)
        )
        clip_path = final_root / "diffsheg_eval_clip_ids.txt"
        clip_path.write_text(
            "".join(f"{clip_id}\n" for clip_id in ids),
            encoding="utf-8",
        )
        runtime = {"torch": "fixture"}
        selected = {
            "path": str(root / "base.pth"),
            "sha256": "e" * 64,
            "bytes": 1,
        }
        contract = {
            "selection_policy": {
                "primary_metric": BRIDGE.VALIDATION_PRIMARY_METRIC,
                "mode": "min",
                "validation_only_for_selection": True,
                "test_evaluations": 1,
                "test_feedback_into_selection": False,
            },
            "checkpoints": {
                "base": {
                    "path": selected["path"],
                    "expected_sha256": selected["sha256"],
                }
            },
            "test_clips": 4,
            "num_shards": 2,
            "split": "test",
            "exact_once": True,
            "physical_predictions_per_clip": 1,
        }
        lineage = {
            "format": BRIDGE.INFERENCE_LINEAGE_FORMAT,
            "status": "complete",
            "contract": contract,
            "contract_sha256": BRIDGE.final_test._canonical_json_sha256(contract),
            "runtime": runtime,
            "runtime_sha256": BRIDGE.final_test._canonical_json_sha256(runtime),
            "shards": [{"shard_id": 0}, {"shard_id": 1}],
            "final_manifest_sha256": BRIDGE.sha256_file(manifest_path),
            "clip_manifest_sha256": BRIDGE.sha256_file(clip_path),
        }
        lineage_path = final_root / "final_lineage.json"
        lineage_path.write_bytes(BRIDGE.canonical_json_bytes(lineage))
        summary = {
            "format": BRIDGE.INFERENCE_SUMMARY_FORMAT,
            "status": "complete",
            "generator": "SemTalk Base-only",
            "test_clips": 4,
            "prediction_files": 4,
            "ground_truth_files": 4,
            "num_shards": 2,
            "npz_root": str(npz_root),
            "manifest_sha256": BRIDGE.sha256_file(manifest_path),
            "clip_manifest_sha256": BRIDGE.sha256_file(clip_path),
            "lineage_sha256": BRIDGE.final_test._canonical_json_sha256(lineage),
            "contract_sha256": lineage["contract_sha256"],
            "runtime_sha256": lineage["runtime_sha256"],
            "finite": True,
            "exact_once": True,
            "split_disjoint": True,
            "test_evaluations": 0,
            "test_feedback_into_selection": False,
        }
        (final_root / "final_summary.json").write_bytes(
            BRIDGE.canonical_json_bytes(summary)
        )
        manifest = {
            "path": str(manifest_path),
            "sha256": BRIDGE.sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        }
        authority = {
            "winner_selection": {"selected_checkpoint": selected},
        }
        args = SimpleNamespace()
        return args, authority, contract, rows

    def test_accepts_exact_pairs_at_88_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, authority, contract, rows = self._bundle(root)
            final_root = root / "final"
            manifest = {
                "path": str(final_root / "final_manifest.jsonl"),
                "sha256": BRIDGE.sha256_file(final_root / "final_manifest.jsonl"),
                "bytes": (final_root / "final_manifest.jsonl").stat().st_size,
            }
            lineage = json.loads(
                (final_root / "final_lineage.json").read_text(encoding="utf-8")
            )
            with (
                mock.patch.object(BRIDGE, "EXPECTED_TEST_CLIPS", 4),
                mock.patch.object(BRIDGE, "EXPECTED_NUM_SHARDS", 2),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_load_final_rows",
                    return_value=(manifest, rows, lineage),
                ),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_contract",
                    return_value=contract,
                ),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_output_roots",
                    return_value=(final_root, root / "shards"),
                ),
            ):
                receipt, clip_ids = BRIDGE._validate_inference_bundle(
                    args,
                    authority,
                )
            self.assertEqual(len(clip_ids), 4)
            self.assertEqual(receipt["minimum_frames"], 88)
            self.assertEqual(receipt["prediction_files"], 4)
            self.assertRegex(receipt["input_set_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_any_subwindow_clip_without_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, authority, contract, rows = self._bundle(
                root,
                bad_frames=True,
            )
            final_root = root / "final"
            manifest = {
                "path": str(final_root / "final_manifest.jsonl"),
                "sha256": BRIDGE.sha256_file(final_root / "final_manifest.jsonl"),
                "bytes": (final_root / "final_manifest.jsonl").stat().st_size,
            }
            lineage = json.loads(
                (final_root / "final_lineage.json").read_text(encoding="utf-8")
            )
            with (
                mock.patch.object(BRIDGE, "EXPECTED_TEST_CLIPS", 4),
                mock.patch.object(BRIDGE, "EXPECTED_NUM_SHARDS", 2),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_load_final_rows",
                    return_value=(manifest, rows, lineage),
                ),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_contract",
                    return_value=contract,
                ),
                mock.patch.object(
                    BRIDGE.final_test,
                    "_output_roots",
                    return_value=(final_root, root / "shards"),
                ),
                self.assertRaisesRegex(
                    BRIDGE.FinalDiffSHEGError,
                    "sub-88-frame",
                ),
            ):
                BRIDGE._validate_inference_bundle(args, authority)


class WavCoverageTests(unittest.TestCase):
    class ProtocolError(RuntimeError):
        pass

    def _wav(self, path: Path) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 160)

    def test_requires_exact_readable_wav_for_every_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ids = ("chemistry__a", "conan__b")
            source_root = root / "source"
            source_root.mkdir()
            audio_root = root / "audio_view"
            audio_root.mkdir()
            audio_paths = {}
            source_ids = ("chemistry/v0/a", "conan/v1/b")
            audio_rows = []
            inference_rows = []
            for index, (clip_id, source_id) in enumerate(zip(ids, source_ids)):
                speaker, video, sequence = source_id.split("/")
                source = source_root / speaker / video / sequence / f"{sequence}.wav"
                source.parent.mkdir(parents=True)
                self._wav(source)
                view = (
                    audio_root
                    / speaker
                    / video
                    / "test"
                    / sequence
                    / f"{sequence}.wav"
                )
                view.parent.mkdir(parents=True)
                view.write_bytes(source.read_bytes())
                audio_paths[clip_id] = view.resolve()
                wav = BRIDGE._wav_metadata(source.read_bytes(), clip_id)
                digest = BRIDGE.sha256_file(source)
                audio_rows.append(
                    {
                        "evaluation_index": index,
                        "clip_id": clip_id,
                        "source_clip_id": source_id,
                        "source": {
                            "path": str(source.resolve()),
                            "relative_path": f"{speaker}/{video}/{sequence}/{sequence}.wav",
                            "bytes": source.stat().st_size,
                            "sha256": digest,
                        },
                        "view": {
                            "path": str(view.resolve()),
                            "relative_path": (
                                f"{speaker}/{video}/test/{sequence}/{sequence}.wav"
                            ),
                            "bytes": view.stat().st_size,
                            "sha256": digest,
                            "materialization": "copy",
                        },
                        "wav": wav,
                    }
                )
                inference_rows.append(
                    {
                        "canonical_clip_id": clip_id,
                        "source_clip_id": source_id,
                    }
                )
            audio_manifest = audio_root / BRIDGE.AUDIO_VIEW_MANIFEST_NAME
            audio_manifest.write_bytes(
                b"".join(BRIDGE.canonical_json_bytes(row) for row in audio_rows)
            )
            inference_manifest = root / "final_manifest.jsonl"
            inference_manifest.write_bytes(
                b"".join(
                    BRIDGE.canonical_json_bytes(row) for row in inference_rows
                )
            )
            clip_manifest = root / "diffsheg_eval_clip_ids.txt"
            clip_manifest.write_text(
                "".join(f"{clip_id}\n" for clip_id in ids),
                encoding="utf-8",
            )
            validation = SimpleNamespace(
                clips=tuple(
                    SimpleNamespace(clip_id=clip_id) for clip_id in ids
                ),
                total_frames=176,
                total_windows=2,
                uncovered_tail_frames=0,
                clip_manifest_sha256="f" * 64,
                clip_order=f"explicit_manifest:{clip_manifest}",
            )
            evaluator = SimpleNamespace(
                ProtocolError=self.ProtocolError,
                validate_inputs=mock.Mock(return_value=validation),
                load_normalization_stats=mock.Mock(return_value=None),
                _resolve_audio_paths=mock.Mock(
                    return_value=(audio_paths, "talkshow_original_source")
                ),
            )
            inference = {
                "npz_root": str(root),
                "manifest": {
                    "path": str(inference_manifest),
                    "sha256": BRIDGE.sha256_file(inference_manifest),
                },
                "clip_manifest": {
                    "path": str(clip_manifest),
                    "sha256": BRIDGE.sha256_file(clip_manifest),
                },
            }
            audio_receipt = {
                "format": BRIDGE.AUDIO_VIEW_FORMAT,
                "status": "complete",
                "source_root": str(source_root.resolve()),
                "output_root": str(audio_root.resolve()),
                "canonical_manifest": {
                    "path": "/frozen/manifest.jsonl",
                    "bytes": 1,
                    "sha256": "a" * 64,
                },
                "inference_manifest_sha256": inference["manifest"]["sha256"],
                "clip_manifest_sha256": inference["clip_manifest"]["sha256"],
                "clip_count": 2,
                "exact_once": True,
                "symlinks": "forbidden",
                "padding": "forbidden",
                "truncation": "forbidden",
                "fabrication": "forbidden",
                "manifest": {
                    "path": str(audio_manifest.resolve()),
                    "bytes": audio_manifest.stat().st_size,
                    "sha256": BRIDGE.sha256_file(audio_manifest),
                },
                "source_ordered_set_sha256": BRIDGE._audio_set_sha(
                    audio_rows, "source"
                ),
                "view_ordered_set_sha256": BRIDGE._audio_set_sha(
                    audio_rows, "view"
                ),
            }
            audio_receipt["receipt_payload_sha256"] = (
                BRIDGE.canonical_json_sha256(audio_receipt)
            )
            (audio_root / BRIDGE.AUDIO_VIEW_RECEIPT_NAME).write_bytes(
                BRIDGE.canonical_json_bytes(audio_receipt)
            )
            assets = {
                "diffsheg": {"stats": {"path": str(root / "stats.npy")}}
            }
            with mock.patch.object(BRIDGE, "EXPECTED_TEST_CLIPS", 2):
                observed, receipt = BRIDGE._validate_protocol_inputs(
                    evaluator,
                    inference,
                    ids,
                    assets,
                    audio_root,
                    None,
                )
            self.assertIs(observed, validation)
            self.assertEqual(receipt["clip_count"], 2)
            self.assertRegex(
                receipt["ordered_manifest_sha256"],
                r"^[0-9a-f]{64}$",
            )
            audio_paths[ids[0]].write_bytes(b"broken")
            with (
                mock.patch.object(BRIDGE, "EXPECTED_TEST_CLIPS", 2),
                self.assertRaisesRegex(
                    BRIDGE.FinalDiffSHEGError,
                    "audio-view bytes changed",
                ),
            ):
                BRIDGE._validate_protocol_inputs(
                    evaluator,
                    inference,
                    ids,
                    assets,
                    audio_root,
                    None,
                )


class PaspaReportTests(unittest.TestCase):
    def _preflight(self) -> dict:
        return {
            "inference": {
                "window_count": 123,
                "frame_count": 456,
                "uncovered_tail_frames": 7,
                "paspa_clip_manifest_sha256": "d" * 64,
            },
            "assets": {
                "talkshow": {"commit": BRIDGE.TALKSHOW_COMMIT},
                "smplx": {"sha256": BRIDGE.SMPLX_NEUTRAL_SHA256},
                "diffsheg": {
                    "autoencoders": {
                        metric: {"path": f"/assets/{pin['filename']}"}
                        for metric, pin in BRIDGE.DIFFSHEG_AE_PINS.items()
                    }
                },
            },
        }

    def _report(self) -> dict:
        metrics = {
            "pcm": 0.1,
            "gesture_diversity": 0.2,
            "expression_diversity": 0.3,
            "fmd": 0.4,
            "fed": 0.5,
            "fgd": 0.6,
            "ba": 0.7,
        }
        return {
            "status": "ok",
            "protocol": {
                "name": "diffsheg_show_reconstructed",
                "version": 1,
                "status": "reconstructed_from_public_components",
                "diffsheg_reference_commit": BRIDGE.DIFFSHEG_COMMIT,
                "window_length": 88,
                "window_stride": 88,
                "overlap_length": 0,
                "precision": "float32 AE inference; no autocast",
            },
            "inputs": {
                "clip_count": 1_708,
                "window_count": 123,
                "frame_count": 456,
                "uncovered_tail_frames": 7,
                "clip_manifest_sha256": "d" * 64,
                "audio_protocol": "talkshow_original_source",
                "stats": {"sha256": BRIDGE.DIFFSHEG_STATS_SHA256},
                "checkpoint_paths": {
                    metric: f"/assets/{pin['filename']}"
                    for metric, pin in BRIDGE.DIFFSHEG_AE_PINS.items()
                },
            },
            "metrics": metrics,
            "provenance": {
                "evaluator": {
                    "sha256": BRIDGE.PASPA_EVALUATOR_SHA256,
                    "repository_git_head": BRIDGE.PASPA_COMMIT,
                },
                "autoencoders": {
                    metric: {
                        "path": f"/assets/{pin['filename']}",
                        "sha256": pin["sha256"],
                        "input_dim": pin["input_dim"],
                        "feature_count": 123,
                    }
                    for metric, pin in BRIDGE.DIFFSHEG_AE_PINS.items()
                },
                "ba": {
                    "talkshow_git_head": BRIDGE.TALKSHOW_COMMIT,
                    "smplx_neutral_asset_sha256": (
                        BRIDGE.SMPLX_NEUTRAL_SHA256
                    ),
                },
            },
        }

    def test_accepts_exact_finite_seven_metric_report(self) -> None:
        metrics = BRIDGE._validate_paspa_report(
            self._report(),
            self._preflight(),
        )
        self.assertEqual(tuple(metrics), BRIDGE.EXPECTED_METRICS)

    def test_rejects_extra_or_nonfinite_metric(self) -> None:
        report = self._report()
        report["metrics"]["body_released2_fgd"] = 1.0
        with self.assertRaisesRegex(
            BRIDGE.FinalDiffSHEGError,
            "exactly the seven",
        ):
            BRIDGE._validate_paspa_report(report, self._preflight())
        report = self._report()
        report["metrics"]["fgd"] = float("nan")
        with self.assertRaisesRegex(
            BRIDGE.FinalDiffSHEGError,
            "not finite",
        ):
            BRIDGE._validate_paspa_report(report, self._preflight())


class AtomicOutputTests(unittest.TestCase):
    def test_claim_is_final_namespace_scoped_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                BRIDGE._claim_path(root),
                root / "diffsheg-full-test-one-shot.claim.json",
            )
            path = root / "receipt.json"
            BRIDGE._atomic_new(path, {"status": "first"}, "fixture")
            with self.assertRaisesRegex(
                BRIDGE.FinalDiffSHEGError,
                "refusing to overwrite",
            ):
                BRIDGE._atomic_new(path, {"status": "second"}, "fixture")


if __name__ == "__main__":
    unittest.main()
