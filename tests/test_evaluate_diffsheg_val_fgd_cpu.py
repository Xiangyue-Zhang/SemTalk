from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / "scripts" / "show_base" / "evaluate_diffsheg_val_fgd.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_diffsheg_val_fgd", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT_PATH}")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeProtocolError(RuntimeError):
    pass


class FakeTorch:
    __version__ = "fake-cpu"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            raise AssertionError("CPU evaluation must not touch CUDA")

    @staticmethod
    def device(value: str) -> SimpleNamespace:
        if value != "cpu":
            raise AssertionError(value)
        return SimpleNamespace(type="cpu", index=None)


class FakePASPAEvaluator:
    ProtocolError = FakeProtocolError

    def __init__(self, clip_manifest: Path, clip_ids: tuple[str, ...]) -> None:
        self.clip_manifest = clip_manifest
        self.clip_ids = clip_ids
        self.loaded_checkpoints: list[tuple[str, int]] = []

    def load_normalization_stats(self, path: Path) -> object:
        return {"stats": str(path)}

    def validate_inputs(self, **kwargs: object) -> SimpleNamespace:
        if kwargs["clip_manifest"] != self.clip_manifest:
            raise AssertionError(kwargs)
        if kwargs["window_stride"] != 88:
            raise AssertionError(kwargs)
        if kwargs["require_betas"] is not False:
            raise AssertionError(kwargs)
        return SimpleNamespace(
            clips=tuple(self.clip_ids),
            total_frames=176,
            total_windows=2,
            uncovered_tail_frames=0,
            clip_manifest_sha256="a" * 64,
            clip_order=f"explicit_manifest:{self.clip_manifest}",
        )

    def load_embedding_model(self, **kwargs: object) -> tuple[object, dict]:
        checkpoint = Path(str(kwargs["checkpoint_path"]))
        input_dim = int(kwargs["input_dim"])
        self.loaded_checkpoints.append((checkpoint.name, input_dim))
        return object(), {
            "path": str(checkpoint),
            "sha256": EVALUATOR.DIFFSHEG_GESTURE_AE_SHA256,
            "input_dim": 129,
            "state_container": "model_state",
            "load_mode": "encoder_only",
        }

    def extract_features(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        if kwargs["component"] != "gesture":
            raise AssertionError(kwargs)
        predicted = np.zeros((2, 300), dtype=np.float32)
        ground_truth = np.ones((2, 300), dtype=np.float32)
        return predicted, ground_truth

    def frechet_distance(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        if predicted.dtype != np.float32 or ground_truth.dtype != np.float32:
            raise AssertionError("features are not float32")
        return 0.25


class EvaluateDiffSHEGValFGDCPUTest(unittest.TestCase):
    def _fixture(
        self, root: Path
    ) -> tuple[argparse.Namespace, FakePASPAEvaluator]:
        prediction_dir = root / "predictions" / "val"
        ground_truth_dir = root / "ground-truth" / "val"
        output_dir = root / "reports" / "val"
        prediction_dir.mkdir(parents=True)
        ground_truth_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        clip_ids = ("oliver__0001", "seth__0002")
        for clip_id in clip_ids:
            (prediction_dir / f"res_{clip_id}.npz").write_bytes(b"pred")
            (ground_truth_dir / f"gt_{clip_id}.npz").write_bytes(b"gt")
        clip_manifest = root / "manifest" / "val" / "clips.txt"
        clip_manifest.parent.mkdir(parents=True)
        clip_manifest.write_text(
            "".join(f"{clip_id}\n" for clip_id in clip_ids),
            encoding="utf-8",
        )

        paspa_root = root / "paspa"
        diffsheg_root = root / "diffsheg"
        paspa_root.mkdir()
        (diffsheg_root / "data" / "SHOW" / "ae_weights").mkdir(
            parents=True
        )
        stats_path = (
            diffsheg_root / EVALUATOR.DIFFSHEG_STATS_RELATIVE
        )
        gesture_path = (
            diffsheg_root / EVALUATOR.DIFFSHEG_GESTURE_AE_RELATIVE
        )
        stats_path.write_bytes(b"stats")
        gesture_path.write_bytes(b"gesture")
        args = argparse.Namespace(
            pred_dir=str(prediction_dir),
            gt_dir=str(ground_truth_dir),
            clip_manifest=str(clip_manifest),
            clip_manifest_sha256=_sha256(clip_manifest),
            paspa_root=str(paspa_root),
            diffsheg_root=str(diffsheg_root),
            device="cpu",
            batch_size=8,
            output=str(output_dir / "fgd.json"),
        )
        return args, FakePASPAEvaluator(clip_manifest.resolve(), clip_ids)

    def _asset_receipt(self, args: argparse.Namespace) -> tuple:
        root = Path(args.diffsheg_root).resolve()
        stats = root / EVALUATOR.DIFFSHEG_STATS_RELATIVE
        gesture = root / EVALUATOR.DIFFSHEG_GESTURE_AE_RELATIVE
        return (
            root,
            stats,
            gesture,
            {
                "diffsheg_root": {
                    "path": str(root),
                    "git_head": EVALUATOR.DIFFSHEG_REFERENCE_COMMIT,
                },
                "stats": {
                    "path": str(stats),
                    "sha256": EVALUATOR.DIFFSHEG_STATS_SHA256,
                },
                "weights_dir": str(gesture.parent),
                "gesture_autoencoder": {
                    "path": str(gesture),
                    "sha256": EVALUATOR.DIFFSHEG_GESTURE_AE_SHA256,
                    "input_dim": 129,
                },
            },
        )

    def test_pins_match_selector_contract(self) -> None:
        selector_path = (
            ROOT
            / "scripts"
            / "show_base"
            / "select_base_official_adapt.py"
        )
        selector_spec = importlib.util.spec_from_file_location(
            "selector_for_val_fgd_test", selector_path
        )
        if selector_spec is None or selector_spec.loader is None:
            raise RuntimeError(selector_path)
        selector = importlib.util.module_from_spec(selector_spec)
        selector_spec.loader.exec_module(selector)
        pins = selector.DIFFSHEG_PINNED_RECEIPT
        self.assertEqual(EVALUATOR.PASPA_COMMIT, pins["paspa"]["commit"])
        self.assertEqual(EVALUATOR.PASPA_TREE, pins["paspa"]["tree"])
        self.assertEqual(
            EVALUATOR.PASPA_EVALUATOR_SHA256,
            pins["paspa"]["evaluator_sha256"],
        )
        self.assertEqual(
            EVALUATOR.DIFFSHEG_REFERENCE_COMMIT,
            pins["diffsheg_reference_commit"],
        )
        self.assertEqual(
            EVALUATOR.DIFFSHEG_STATS_SHA256, pins["stats_sha256"]
        )
        self.assertEqual(
            EVALUATOR.DIFFSHEG_GESTURE_AE_SHA256,
            pins["autoencoders"]["fgd"]["sha256"],
        )
        self.assertEqual(selector.VAL_METRIC_KEYS, ("fgd",))

    def test_evaluate_calls_only_gesture_ae_and_emits_fgd_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, fake = self._fixture(Path(temporary))
            evaluator_receipt = {
                "path": "/pinned/PASPA/scripts/diffsheg_show_eval.py",
                "sha256": EVALUATOR.PASPA_EVALUATOR_SHA256,
                "repository_root": "/pinned/PASPA",
                "repository_git_head": EVALUATOR.PASPA_COMMIT,
                "repository_git_tree": EVALUATOR.PASPA_TREE,
                "repository_origin": EVALUATOR.PASPA_ORIGIN,
            }
            with (
                mock.patch.object(EVALUATOR, "EXPECTED_VAL_CLIPS", 2),
                mock.patch.object(
                    EVALUATOR,
                    "_load_pinned_paspa_evaluator",
                    return_value=(fake, evaluator_receipt),
                ),
                mock.patch.object(
                    EVALUATOR,
                    "_verify_diffsheg_assets",
                    return_value=self._asset_receipt(args),
                ),
                mock.patch.dict(sys.modules, {"torch": FakeTorch}),
            ):
                report = EVALUATOR.evaluate(args)
            self.assertEqual(report["metrics"], {"fgd": 0.25})
            self.assertEqual(fake.loaded_checkpoints, [("gesture.pth.tar", 129)])
            self.assertEqual(set(report["provenance"]["autoencoders"]), {"fgd"})
            self.assertIsNone(report["protocol"]["ba"])
            self.assertEqual(report["protocol"]["window_length"], 88)
            self.assertEqual(report["protocol"]["window_stride"], 88)
            self.assertEqual(
                report["protocol"]["precision"],
                "float32 AE inference; no autocast",
            )
            self.assertEqual(report["inputs"]["clip_count"], 2)
            self.assertFalse(report["protocol"]["test_visible"])

    def test_manifest_hash_count_and_forbidden_labels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "manifest" / "val"
            root.mkdir(parents=True)
            manifest = root / "clips.txt"
            manifest.write_text("oliver__0001\n", encoding="utf-8")
            with (
                mock.patch.object(EVALUATOR, "EXPECTED_VAL_CLIPS", 1),
                self.assertRaisesRegex(
                    EVALUATOR.ValidationFGDError, "SHA-256 mismatch"
                ),
            ):
                EVALUATOR._read_exact_val_manifest(manifest, "0" * 64)

            for clip_id, message in (
                ("oliver__test_sequence", "test-labelled"),
                ("oliver__e30", "withdrawn e30"),
                ("speaker2__0001", "forbidden Speaker2"),
            ):
                manifest.write_text(f"{clip_id}\n", encoding="utf-8")
                with (
                    mock.patch.object(EVALUATOR, "EXPECTED_VAL_CLIPS", 1),
                    self.subTest(clip_id=clip_id),
                    self.assertRaisesRegex(
                        EVALUATOR.ValidationFGDError, message
                    ),
                ):
                    EVALUATOR._read_exact_val_manifest(
                        manifest, _sha256(manifest)
                    )

    def test_symlink_and_nonexact_directory_cover_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "predictions" / "val"
            directory.mkdir(parents=True)
            target = root / "target.npz"
            target.write_bytes(b"x")
            os.symlink(target, directory / "res_oliver__0001.npz")
            with self.assertRaisesRegex(
                EVALUATOR.ValidationFGDError, "symlink"
            ):
                EVALUATOR._validate_exact_directory_cover(
                    directory,
                    ("oliver__0001",),
                    role="prediction",
                    prefix="res_",
                )

            (directory / "res_oliver__0001.npz").unlink()
            (directory / "res_oliver__0001.npz").write_bytes(b"x")
            (directory / "unexpected.bin").write_bytes(b"x")
            with self.assertRaisesRegex(
                EVALUATOR.ValidationFGDError, "not the exact"
            ):
                EVALUATOR._validate_exact_directory_cover(
                    directory,
                    ("oliver__0001",),
                    role="prediction",
                    prefix="res_",
                )

    def test_atomic_output_is_new_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "reports" / "val"
            output_dir.mkdir(parents=True)
            output = output_dir / "fgd.json"
            result = EVALUATOR._atomic_write_new_json(
                output, {"metrics": {"fgd": 0.1}}
            )
            self.assertEqual(result, output.resolve())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"metrics": {"fgd": 0.1}},
            )
            with self.assertRaisesRegex(
                EVALUATOR.ValidationFGDError, "overwrite"
            ):
                EVALUATOR._atomic_write_new_json(
                    output, {"metrics": {"fgd": 0.2}}
                )

    def test_negative_or_nonfinite_fgd_is_rejected(self) -> None:
        for value in (-0.1, float("inf"), float("nan")):
            with tempfile.TemporaryDirectory() as temporary:
                args, fake = self._fixture(Path(temporary))
                fake.frechet_distance = mock.Mock(return_value=value)
                evaluator_receipt = {
                    "path": "/pinned/evaluator.py",
                    "sha256": EVALUATOR.PASPA_EVALUATOR_SHA256,
                    "repository_root": "/pinned/PASPA",
                    "repository_git_head": EVALUATOR.PASPA_COMMIT,
                    "repository_git_tree": EVALUATOR.PASPA_TREE,
                    "repository_origin": EVALUATOR.PASPA_ORIGIN,
                }
                with (
                    mock.patch.object(EVALUATOR, "EXPECTED_VAL_CLIPS", 2),
                    mock.patch.object(
                        EVALUATOR,
                        "_load_pinned_paspa_evaluator",
                        return_value=(fake, evaluator_receipt),
                    ),
                    mock.patch.object(
                        EVALUATOR,
                        "_verify_diffsheg_assets",
                        return_value=self._asset_receipt(args),
                    ),
                    mock.patch.dict(sys.modules, {"torch": FakeTorch}),
                    self.subTest(value=value),
                    self.assertRaisesRegex(
                        EVALUATOR.ValidationFGDError,
                        "finite and nonnegative",
                    ),
                ):
                    EVALUATOR.evaluate(args)


if __name__ == "__main__":
    unittest.main()
