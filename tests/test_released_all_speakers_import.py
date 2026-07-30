from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - minimal local environments
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/show_base/build_base_features.py"
SPEC = importlib.util.spec_from_file_location(
    "released_all_speakers_build_base_features",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
FEATURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FEATURES)


@unittest.skipIf(torch is None, "torch is unavailable")
class ReleasedAllSpeakersImportTests(unittest.TestCase):
    class TinyModel(torch.nn.Module if torch is not None else object):
        def __init__(self, _args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    def _write_suite(
        self,
        root: Path,
        *,
        outer_extra: bool = False,
        nonfinite_stage: str | None = None,
        wrong_key_stage: str | None = None,
    ):
        specifications = {
            name: dict(value)
            for name, value in FEATURES.RELEASED_ALL_SPEAKERS_MODELS.items()
        }
        paths = {}
        for index, (name, specification) in enumerate(
            specifications.items()
        ):
            value = (
                torch.tensor([float("nan")])
                if name == nonfinite_stage
                else torch.tensor([float(index + 1)])
            )
            key = (
                "module.other"
                if name == wrong_key_stage
                else "module.weight"
            )
            payload = {"model_state": {key: value}}
            if outer_extra and name == "face":
                payload["audit"] = {}
            path = root / specification["filename"]
            torch.save(payload, path)
            specification["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            paths[name] = path
        return specifications, paths

    @staticmethod
    def _args(paths, *, source=None, status=None):
        values = {
            "device": "cpu",
            "prerequisite_source": (
                source
                or FEATURES.RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE
            ),
        }
        for name, path in paths.items():
            values[f"{name}_checkpoint"] = str(path)
            values[f"{name}_status_json"] = status
        return argparse.Namespace(**values)

    def _fake_model_modules(self):
        rvq = ModuleType("models.rvq")
        rvq.RVQVAE = self.TinyModel
        motion = ModuleType("models.motion_representation")
        motion.VAEConvZero = self.TinyModel
        return mock.patch.dict(
            sys.modules,
            {
                "models.rvq": rvq,
                "models.motion_representation": motion,
            },
        )

    @staticmethod
    def _base_cli(
        *,
        expected_input_commit: str | None = None,
        expected_input_tree: str | None = None,
    ) -> list[str]:
        sha256 = "a" * 64
        builder_commit = "1" * 40
        builder_tree = "2" * 40
        argv = [
            "base",
            "--canonical-manifest",
            "/frozen/canonical.jsonl",
            "--canonical-summary",
            "/frozen/canonical-summary.json",
            "--canonical-lineage",
            "/frozen/canonical-lineage.json",
            "--audio-manifest",
            "/frozen/audio.jsonl",
            "--audio-lineage-json",
            "/frozen/audio-lineage.json",
            "--representation-training-lineage-manifest",
            "/frozen/representation-lineage.json",
            "--prerequisite-source",
            FEATURES.RELEASED_ALL_SPEAKERS_PREREQUISITE_SOURCE,
        ]
        for name, specification in (
            FEATURES.RELEASED_ALL_SPEAKERS_MODELS.items()
        ):
            argv.extend(
                [
                    f"--{name}-checkpoint",
                    f"/weights/{specification['filename']}",
                ]
            )
        argv.extend(
            [
                "--output-lmdb",
                "/output/base.lmdb",
                "--summary-json",
                "/output/base-summary.json",
                "--lineage-json",
                "/output/base-lineage.json",
                "--device",
                "cpu",
                "--expected-hubert-tree-sha256",
                sha256,
                "--expected-source-commit",
                builder_commit,
                "--expected-source-tree",
                builder_tree,
                "--expected-canonical-source-commit",
                "3" * 40,
                "--expected-canonical-source-tree",
                "4" * 40,
                "--expected-canonical-manifest-sha256",
                sha256,
                "--expected-canonical-summary-sha256",
                sha256,
                "--expected-canonical-lineage-sha256",
                sha256,
            ]
        )
        if expected_input_commit is not None:
            argv.extend(
                [
                    "--expected-input-source-commit",
                    expected_input_commit,
                ]
            )
        if expected_input_tree is not None:
            argv.extend(
                [
                    "--expected-input-source-tree",
                    expected_input_tree,
                ]
            )
        return argv

    def test_production_release_manifest_is_exact(self):
        self.assertEqual(
            FEATURES.RELEASED_ALL_SPEAKERS_MODELS,
            {
                "face": {
                    "filename": "rvq_face_600.bin",
                    "sha256": (
                        "31b04c88456a25f4d57841c0cb507b4c856daccb"
                        "3875878d06545110a6152127"
                    ),
                    "model_class": "RVQVAE",
                    "vae_test_dim": 106,
                    "vae_layer": 2,
                },
                "hands": {
                    "filename": "rvq_hands_500.bin",
                    "sha256": (
                        "08f887aac60d5a2102dce7c57559a6b3d9b7f56e"
                        "3d4a38055ca47a539b03e436"
                    ),
                    "model_class": "RVQVAE",
                    "vae_test_dim": 180,
                    "vae_layer": 2,
                },
                "upper": {
                    "filename": "rvq_upper_500.bin",
                    "sha256": (
                        "05101461e75b4e9b687ef30437585d56969c6a13"
                        "d0047b91000b31d88d08ac17"
                    ),
                    "model_class": "RVQVAE",
                    "vae_test_dim": 78,
                    "vae_layer": 2,
                },
                "lower": {
                    "filename": "rvq_lower_600.bin",
                    "sha256": (
                        "2bb43d10e5f32d13d21e6b85580a1b70d36e407"
                        "c8552a7e62f99c171ae4efce8"
                    ),
                    "model_class": "RVQVAE",
                    "vae_test_dim": 61,
                    "vae_layer": 4,
                },
                "global": {
                    "filename": "last_1700_foot.bin",
                    "sha256": (
                        "6e6f88abd98ccbe2c52102b937067f4ade0aa307"
                        "d6e1dac8e127e19e0144ee12"
                    ),
                    "model_class": "VAEConvZero",
                    "vae_test_dim": 61,
                    "vae_layer": 4,
                },
            },
        )

    def test_exact_suite_strict_loads_freezes_and_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            specifications, paths = self._write_suite(Path(temporary))
            with (
                mock.patch.object(
                    FEATURES,
                    "RELEASED_ALL_SPEAKERS_MODELS",
                    specifications,
                ),
                self._fake_model_modules(),
            ):
                models, records, receipt = (
                    FEATURES.load_released_all_speakers_models(
                        self._args(paths)
                    )
                )
                FEATURES.revalidate_checkpoint_records(records)
            self.assertEqual(set(models), set(FEATURES.RVQ_NAMES))
            self.assertEqual(set(records), {*FEATURES.RVQ_NAMES, "global"})
            self.assertEqual(
                receipt["prerequisite_source"],
                "released_all_speakers_v1",
            )
            self.assertEqual(
                receipt["classification"],
                FEATURES.RELEASED_ALL_SPEAKERS_CLASSIFICATION,
            )
            self.assertEqual(receipt["training_dataset"], "BEAT2")
            self.assertEqual(receipt["speaker_scope"], "All-Speakers")
            self.assertIs(receipt["show_trained"], False)
            self.assertEqual(
                receipt["source_receipt"]["release_trust_root"],
                FEATURES.RELEASED_ALL_SPEAKERS_RELEASE_TRUST_ROOT,
            )
            self.assertEqual(
                receipt["receipt_sha256"],
                FEATURES.canonical_file_payload_sha256(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_sha256"
                    }
                ),
            )
            for model in models.values():
                self.assertFalse(model.training)
                self.assertTrue(
                    all(
                        not parameter.requires_grad
                        for parameter in model.parameters()
                    )
                )
            for record in records.values():
                self.assertIs(record["strict_state_dict_load"], True)
                self.assertIs(
                    record["all_model_state_tensors_finite"],
                    True,
                )
                self.assertIs(record["frozen_eval"], True)
                self.assertIs(record["show_trained"], False)

    def test_rejects_symlink_even_when_target_bytes_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specifications, paths = self._write_suite(root)
            face = paths["face"]
            target = root / "face-target.bin"
            face.replace(target)
            face.symlink_to(target)
            with (
                mock.patch.object(
                    FEATURES,
                    "RELEASED_ALL_SPEAKERS_MODELS",
                    specifications,
                ),
                self._fake_model_modules(),
                self.assertRaisesRegex(RuntimeError, "non-symlink"),
            ):
                FEATURES.load_released_all_speakers_models(
                    self._args(paths)
                )

    def test_rejects_wrong_filename_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specifications, paths = self._write_suite(root)
            renamed = root / "face.bin"
            paths["face"].replace(renamed)
            paths["face"] = renamed
            with (
                mock.patch.object(
                    FEATURES,
                    "RELEASED_ALL_SPEAKERS_MODELS",
                    specifications,
                ),
                self._fake_model_modules(),
                self.assertRaisesRegex(RuntimeError, "filename"),
            ):
                FEATURES.load_released_all_speakers_models(
                    self._args(paths)
                )
            specifications["face"]["filename"] = "face.bin"
            specifications["face"]["sha256"] = "0" * 64
            with (
                mock.patch.object(
                    FEATURES,
                    "RELEASED_ALL_SPEAKERS_MODELS",
                    specifications,
                ),
                self._fake_model_modules(),
                self.assertRaisesRegex(RuntimeError, "SHA-256"),
            ):
                FEATURES.load_released_all_speakers_models(
                    self._args(paths)
                )

    def test_rejects_extra_container_key_nonfinite_and_schema_mismatch(self):
        cases = (
            ({"outer_extra": True}, "only model_state"),
            ({"nonfinite_stage": "face"}, "non-finite"),
            ({"wrong_key_stage": "face"}, "state_dict"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with tempfile.TemporaryDirectory() as temporary:
                    specifications, paths = self._write_suite(
                        Path(temporary),
                        **kwargs,
                    )
                    with (
                        mock.patch.object(
                            FEATURES,
                            "RELEASED_ALL_SPEAKERS_MODELS",
                            specifications,
                        ),
                        self._fake_model_modules(),
                        self.assertRaisesRegex(RuntimeError, message),
                    ):
                        FEATURES.load_released_all_speakers_models(
                            self._args(paths)
                        )

    def test_sources_have_mutually_exclusive_status_contracts(self):
        paths = {
            name: Path(f"/weights/{specification['filename']}")
            for name, specification in (
                FEATURES.RELEASED_ALL_SPEAKERS_MODELS.items()
            )
        }
        with self.assertRaisesRegex(RuntimeError, "forbids SHOW"):
            FEATURES.load_prerequisite_models(
                self._args(paths, status="/status.json"),
                "a" * 64,
                {},
            )
        with self.assertRaisesRegex(RuntimeError, "requires status JSON"):
            FEATURES.load_prerequisite_models(
                self._args(
                    paths,
                    source=FEATURES.SHOW_TRAINED_PREREQUISITE_SOURCE,
                ),
                "a" * 64,
                {},
            )

    def test_input_artifact_source_defaults_to_builder_source(self):
        with mock.patch.object(
            sys,
            "argv",
            [str(MODULE_PATH), *self._base_cli()],
        ):
            args = FEATURES.parse_args()
        self.assertIsNone(args.expected_input_source_commit)
        self.assertIsNone(args.expected_input_source_tree)
        FEATURES.normalize_input_artifact_source_args(args)
        self.assertEqual(args.expected_input_source_commit, "1" * 40)
        self.assertEqual(args.expected_input_source_tree, "2" * 40)
        self.assertEqual(
            FEATURES.input_artifact_source_receipt(
                args.expected_input_source_commit,
                args.expected_input_source_tree,
            ),
            {
                "format": "semtalk_show_input_artifact_source_v1",
                "origin": FEATURES.EXPECTED_ORIGIN,
                "commit": "1" * 40,
                "tree": "2" * 40,
            },
        )

    def test_explicit_input_artifact_source_is_independent(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                str(MODULE_PATH),
                *self._base_cli(
                    expected_input_commit="5" * 40,
                    expected_input_tree="6" * 40,
                ),
            ],
        ):
            args = FEATURES.parse_args()
        FEATURES.normalize_input_artifact_source_args(args)
        self.assertEqual(args.expected_source_commit, "1" * 40)
        self.assertEqual(args.expected_source_tree, "2" * 40)
        self.assertEqual(args.expected_input_source_commit, "5" * 40)
        self.assertEqual(args.expected_input_source_tree, "6" * 40)
        receipt = FEATURES.input_artifact_source_receipt(
            args.expected_input_source_commit,
            args.expected_input_source_tree,
        )
        self.assertEqual(receipt["commit"], "5" * 40)
        self.assertEqual(receipt["tree"], "6" * 40)
        self.assertEqual(
            len(FEATURES.canonical_file_payload_sha256(receipt)),
            64,
        )

    def test_input_artifact_source_pair_is_fail_closed(self):
        for commit, tree in (("5" * 40, None), (None, "6" * 40)):
            with self.subTest(commit=commit, tree=tree):
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            str(MODULE_PATH),
                            *self._base_cli(
                                expected_input_commit=commit,
                                expected_input_tree=tree,
                            ),
                        ],
                    ),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    FEATURES.parse_args()
                self.assertEqual(raised.exception.code, 2)

    def test_input_artifact_source_oid_is_strict(self):
        args = argparse.Namespace(
            mode="base",
            expected_source_commit="1" * 40,
            expected_source_tree="2" * 40,
            expected_input_source_commit="not-an-oid",
            expected_input_source_tree="6" * 40,
        )
        with self.assertRaisesRegex(ValueError, "lowercase Git object ID"):
            FEATURES.normalize_input_artifact_source_args(args)


if __name__ == "__main__":
    unittest.main()
