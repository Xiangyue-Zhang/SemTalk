from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRODUCER_COMMIT = "8" * 40
PRODUCER_TREE = "9" * 40
CANONICAL_COMMIT = "a" * 40
CANONICAL_TREE = "b" * 40
SHA256 = "c" * 64


def load_module(relative: str, name: str, *, stub_lmdb: bool = False):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous_lmdb = sys.modules.get("lmdb", missing)
    if stub_lmdb:
        sys.modules["lmdb"] = ModuleType("lmdb")
    try:
        spec.loader.exec_module(module)
    finally:
        if stub_lmdb:
            if previous_lmdb is missing:
                sys.modules.pop("lmdb", None)
            else:
                sys.modules["lmdb"] = previous_lmdb
    return module


FEATURES = load_module(
    "scripts/show_base/build_base_features.py",
    "dual_receipt_build_base_features",
)
REPRESENTATION = load_module(
    "scripts/show_base/build_representation_lmdb.py",
    "dual_receipt_build_representation",
    stub_lmdb=True,
)
PARITY = load_module(
    "scripts/show_base/run_global_foot_fastpath_parity_suite.py",
    "dual_receipt_global_parity",
)


def parse_with(module, argv: list[str]):
    with mock.patch.object(sys, "argv", [str(module.__file__), *argv]):
        return module.parse_args()


def producer_args() -> list[str]:
    return [
        "--expected-source-commit",
        PRODUCER_COMMIT,
        "--expected-source-tree",
        PRODUCER_TREE,
        "--expected-canonical-source-commit",
        CANONICAL_COMMIT,
        "--expected-canonical-source-tree",
        CANONICAL_TREE,
    ]


class FrozenRepresentationLedgerTest(unittest.TestCase):
    def test_public_loader_window_ledger_is_exact(self) -> None:
        self.assertEqual(
            REPRESENTATION.EXPECTED_NON_WHOLE_SECOND_FRAME_LENGTHS,
            (62, 163, 230),
        )
        self.assertEqual(
            REPRESENTATION.EXPECTED_SPEAKER_CLIP_COUNTS,
            {
                "oliver": 5_246,
                "chemistry": 1_949,
                "seth": 1_984,
                "conan": 4_508,
            },
        )
        self.assertEqual(
            REPRESENTATION.EXPECTED_SPEAKER_WINDOW_COUNTS,
            {
                "oliver": 50_285,
                "chemistry": 14_374,
                "seth": 19_310,
                "conan": 43_317,
            },
        )
        self.assertEqual(
            sum(REPRESENTATION.EXPECTED_SPEAKER_CLIP_COUNTS.values()),
            13_687,
        )
        self.assertEqual(
            sum(REPRESENTATION.EXPECTED_SPEAKER_WINDOW_COUNTS.values()),
            127_286,
        )
        anomalous_counts = {
            frames: (
                REPRESENTATION.window_count_for_frames(
                    frames,
                    floor_to_whole_seconds=True,
                ),
                REPRESENTATION.window_count_for_frames(
                    frames,
                    floor_to_whole_seconds=False,
                ),
            )
            for frames in (62, 163, 230)
        }
        self.assertEqual(
            anomalous_counts,
            {
                62: (0, 0),
                163: (5, 5),
                230: (8, 9),
            },
        )
        self.assertEqual(REPRESENTATION.EXPECTED_ENTRIES, 127_286)
        self.assertEqual(REPRESENTATION.EXPECTED_RAW_ENTRIES, 127_287)
        self.assertEqual(127_286 // 64, 1_988)
        self.assertEqual(127_286 % 64, 54)
        self.assertEqual((127_286 + 63) // 64, 1_989)
        self.assertEqual(64 - (127_286 % 64), 10)
        self.assertEqual(
            {
                "face": 600 * 1_988,
                "hands": 500 * 1_988,
                "upper": 500 * 1_988,
                "lower": 600 * 1_988,
                "global": 1_700 * 1_988,
                "base": 400 * 1_988,
            },
            {
                "face": 1_192_800,
                "hands": 994_000,
                "upper": 994_000,
                "lower": 1_192_800,
                "global": 3_379_600,
                "base": 795_200,
            },
        )


class DualSourceCliTest(unittest.TestCase):
    def assert_each_source_root_is_required(
        self,
        module,
        argv: list[str],
    ) -> None:
        parsed = parse_with(module, argv)
        self.assertEqual(parsed.expected_source_commit, PRODUCER_COMMIT)
        self.assertEqual(parsed.expected_source_tree, PRODUCER_TREE)
        self.assertEqual(
            parsed.expected_canonical_source_commit,
            CANONICAL_COMMIT,
        )
        self.assertEqual(
            parsed.expected_canonical_source_tree,
            CANONICAL_TREE,
        )
        for option in (
            "--expected-source-commit",
            "--expected-source-tree",
            "--expected-canonical-source-commit",
            "--expected-canonical-source-tree",
        ):
            with self.subTest(module=module.__name__, option=option):
                incomplete = list(argv)
                index = incomplete.index(option)
                del incomplete[index : index + 2]
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_with(module, incomplete)
                self.assertEqual(raised.exception.code, 2)
        abbreviated = list(argv)
        index = abbreviated.index("--expected-canonical-source-commit")
        abbreviated[index] = "--expected-canonical-source-comm"
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_with(module, abbreviated)
        self.assertEqual(raised.exception.code, 2)

    def test_representation_requires_independent_roots(self) -> None:
        self.assert_each_source_root_is_required(
            REPRESENTATION,
            [
                "--canonical-manifest",
                "/frozen/manifest.jsonl",
                "--canonical-summary",
                "/frozen/summary.json",
                "--canonical-lineage",
                "/frozen/lineage.json",
                "--output-lmdb",
                "/output/representation.lmdb",
                "--summary-json",
                "/output/representation.json",
                "--expected-train-clips",
                "13687",
                "--expected-entries",
                "127286",
                *producer_args(),
            ],
        )

    def test_parity_requires_independent_roots(self) -> None:
        self.assert_each_source_root_is_required(
            PARITY,
            [
                "--canonical-manifest",
                "/frozen/manifest.jsonl",
                "--canonical-summary",
                "/frozen/summary.json",
                "--canonical-lineage",
                "/frozen/lineage.json",
                "--smplx-model",
                "/frozen/SMPLX_NEUTRAL_2020.npz",
                "--expected-smplx-sha256",
                SHA256,
                "--output-root",
                "/output/parity",
                *producer_args(),
            ],
        )

    def test_feature_audio_mode_requires_independent_roots(self) -> None:
        self.assert_each_source_root_is_required(
            FEATURES,
            [
                "audio",
                "--canonical-manifest",
                "/frozen/manifest.jsonl",
                "--canonical-summary",
                "/frozen/summary.json",
                "--canonical-lineage",
                "/frozen/lineage.json",
                "--split",
                "train",
                "--hubert-model",
                "/frozen/hubert",
                "--output-dir",
                "/output/features",
                "--output-manifest",
                "/output/manifest.jsonl",
                "--summary-json",
                "/output/summary.json",
                "--lineage-json",
                "/output/lineage.json",
                "--device",
                "cuda:0",
                "--shard-id",
                "0",
                "--num-shards",
                "8",
                "--expected-hubert-tree-sha256",
                SHA256,
                *producer_args(),
            ],
        )

    def test_feature_base_mode_requires_independent_roots(self) -> None:
        argv = [
            "base",
            "--canonical-manifest",
            "/frozen/manifest.jsonl",
            "--canonical-summary",
            "/frozen/summary.json",
            "--canonical-lineage",
            "/frozen/lineage.json",
            "--audio-manifest",
            "/frozen/audio.jsonl",
            "--audio-lineage-json",
            "/frozen/audio-lineage.json",
            "--representation-training-lineage-manifest",
            "/frozen/representation.json",
        ]
        for stage in (*FEATURES.RVQ_NAMES, "global"):
            argv.extend(
                [
                    f"--{stage}-checkpoint",
                    f"/frozen/{stage}.bin",
                    f"--{stage}-status-json",
                    f"/frozen/{stage}.json",
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
                "cuda:0",
                "--expected-hubert-tree-sha256",
                SHA256,
                *producer_args(),
            ]
        )
        self.assert_each_source_root_is_required(FEATURES, argv)

    def test_audio_launcher_forwards_both_receipts(self) -> None:
        source = (
            ROOT / "scripts/show_base/run_audio_cache_8shard.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ $# -ne 14 ]]', source)
        self.assertIn(
            '--expected-source-commit "$source_commit"',
            source,
        )
        self.assertIn(
            '--expected-source-tree "$source_tree"',
            source,
        )
        self.assertIn(
            '--expected-canonical-source-commit "$canonical_source_commit"',
            source,
        )
        self.assertIn(
            '--expected-canonical-source-tree "$canonical_source_tree"',
            source,
        )


class CanonicalReceiptIsolationTest(unittest.TestCase):
    def test_canonical_integer_boundaries_reject_coercible_values(self) -> None:
        for module in (FEATURES, REPRESENTATION, PARITY):
            for invalid in (True, False, "1", 1.0, None):
                with self.subTest(module=module.__name__, invalid=invalid):
                    with self.assertRaises(RuntimeError):
                        module.require_exact_int(invalid, "fixture")
            self.assertEqual(module.require_exact_int(1, "fixture"), 1)

    def write_canonical_receipts(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path]:
        manifest = root / "manifest.jsonl"
        summary_path = root / "summary.json"
        lineage_path = root / "lineage.json"
        manifest.write_text("{}\n", encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        source = {
            "origin": FEATURES.EXPECTED_ORIGIN,
            "commit": CANONICAL_COMMIT,
            "tree": CANONICAL_TREE,
            "entrypoint": "/frozen/build_show_cache.py",
            "entrypoint_sha256": SHA256,
        }
        contract = {
            "source_receipt": source,
            "source_audio_sample_rate": FEATURES.CANONICAL_SOURCE_AUDIO_RATE,
            "hubert_target_sample_rate": FEATURES.CANONICAL_HUBERT_TARGET_RATE,
            "audio_channel_protocol": FEATURES.CANONICAL_AUDIO_CHANNEL_PROTOCOL,
        }
        contract_sha = FEATURES.canonical_file_payload_sha256(contract)
        lineage = {
            "final_manifest_sha256": manifest_sha,
            "lineage_contract": contract,
            "lineage_contract_sha256": contract_sha,
        }
        summary = {
            "status": "complete",
            "schema_name": "semtalk-show-canonical-motion",
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "split_counts": {
                "train": 13_687,
                "val": 1_715,
                "test": 1_708,
            },
            "clip_count": 17_110,
            "exact_once": True,
            "finite": True,
            "split_disjoint": True,
            "lineage_sha256": FEATURES.canonical_file_payload_sha256(
                lineage
            ),
            "lineage_contract_sha256": contract_sha,
            "source_receipt_sha256": (
                FEATURES.canonical_file_payload_sha256(source)
            ),
        }
        lineage_path.write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest, summary_path, lineage_path

    def test_frozen_canonical_source_need_not_equal_current_producer(
        self,
    ) -> None:
        self.assertNotEqual(CANONICAL_COMMIT, PRODUCER_COMMIT)
        self.assertNotEqual(CANONICAL_TREE, PRODUCER_TREE)
        with tempfile.TemporaryDirectory() as directory:
            manifest, summary, lineage = self.write_canonical_receipts(
                Path(directory)
            )
            receipt = FEATURES.load_canonical_receipt(
                manifest_paths=[manifest],
                manifest_hashes={
                    str(manifest.resolve()): hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest()
                },
                summary_path=summary,
                lineage_path=lineage,
                expected_canonical_source_commit=CANONICAL_COMMIT,
                expected_canonical_source_tree=CANONICAL_TREE,
            )
            self.assertEqual(
                receipt["source_receipt"]["commit"],
                CANONICAL_COMMIT,
            )
            representation_receipt = (
                REPRESENTATION.load_canonical_receipts(
                    manifest=manifest,
                    manifest_sha256=hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    summary_path=summary,
                    lineage_path=lineage,
                    expected_canonical_source_commit=CANONICAL_COMMIT,
                    expected_canonical_source_tree=CANONICAL_TREE,
                    expected_train_clips=13_687,
                )
            )
            self.assertEqual(
                representation_receipt["source_receipt"]["commit"],
                CANONICAL_COMMIT,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "canonical source receipt mismatch",
            ):
                FEATURES.load_canonical_receipt(
                    manifest_paths=[manifest],
                    manifest_hashes={
                        str(manifest.resolve()): hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest()
                    },
                    summary_path=summary,
                    lineage_path=lineage,
                    expected_canonical_source_commit=PRODUCER_COMMIT,
                    expected_canonical_source_tree=PRODUCER_TREE,
                )
            with self.assertRaisesRegex(
                RuntimeError,
                "canonical source receipt mismatch",
            ):
                REPRESENTATION.load_canonical_receipts(
                    manifest=manifest,
                    manifest_sha256=hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    summary_path=summary,
                    lineage_path=lineage,
                    expected_canonical_source_commit=PRODUCER_COMMIT,
                    expected_canonical_source_tree=PRODUCER_TREE,
                    expected_train_clips=13_687,
                )

    def test_split_counts_reject_coercible_non_integers(self) -> None:
        loaders = (
            (
                "features",
                lambda manifest, summary, lineage: (
                    FEATURES.load_canonical_receipt(
                        manifest_paths=[manifest],
                        manifest_hashes={
                            str(manifest.resolve()): hashlib.sha256(
                                manifest.read_bytes()
                            ).hexdigest()
                        },
                        summary_path=summary,
                        lineage_path=lineage,
                        expected_canonical_source_commit=CANONICAL_COMMIT,
                        expected_canonical_source_tree=CANONICAL_TREE,
                    )
                ),
            ),
            (
                "representation",
                lambda manifest, summary, lineage: (
                    REPRESENTATION.load_canonical_receipts(
                        manifest=manifest,
                        manifest_sha256=hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                        summary_path=summary,
                        lineage_path=lineage,
                        expected_canonical_source_commit=CANONICAL_COMMIT,
                        expected_canonical_source_tree=CANONICAL_TREE,
                        expected_train_clips=13_687,
                    )
                ),
            ),
        )
        for loader_name, loader in loaders:
            for key, valid in (("val", 1_715), ("test", 1_708)):
                for invalid in (float(valid), str(valid), True, False):
                    with self.subTest(
                        loader=loader_name,
                        key=key,
                        invalid=invalid,
                    ):
                        with tempfile.TemporaryDirectory() as directory:
                            manifest, summary, lineage = (
                                self.write_canonical_receipts(Path(directory))
                            )
                            payload = json.loads(summary.read_text())
                            payload["split_counts"][key] = invalid
                            summary.write_text(
                                json.dumps(payload, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                            with self.assertRaises(RuntimeError):
                                loader(manifest, summary, lineage)


class CanonicalSpeakerBoundaryTest(unittest.TestCase):
    @staticmethod
    def write_npz(
        root: Path,
        *,
        speaker_id: float | int,
        dtype,
    ) -> tuple[Path, dict[str, object]]:
        frames = 4
        path = root / "clip.npz"
        np.savez(
            path,
            pose=np.zeros((frames, 165), dtype=np.float32),
            contact=np.zeros((frames, 4), dtype=np.float32),
            facial=np.zeros((frames, 100), dtype=np.float32),
            beta=np.zeros((frames, 300), dtype=np.float32),
            trans=np.zeros((frames, 3), dtype=np.float32),
            speaker_id=np.full((frames, 1), speaker_id, dtype=dtype),
        )
        row: dict[str, object] = {
            "clip_id": "oliver/clip",
            "speaker": "oliver",
            "speaker_id": 0,
            "frames": frames,
            "canonical_npz": str(path),
            "canonical_npz_sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        }
        return path, row

    def assert_both_loaders_reject(
        self,
        path: Path,
        row: dict[str, object],
    ) -> None:
        with self.assertRaises(RuntimeError):
            REPRESENTATION.validate_clip(
                path,
                row,
                enable_global_foot_fastpath=False,
            )
        with self.assertRaises(RuntimeError):
            FEATURES.load_canonical_clip(row)

    def test_fractional_npz_speaker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_npz(
                Path(directory),
                speaker_id=0.5,
                dtype=np.float32,
            )
            self.assert_both_loaders_reject(path, row)

    def test_npz_and_manifest_speaker_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_npz(
                Path(directory),
                speaker_id=1,
                dtype=np.int64,
            )
            self.assert_both_loaders_reject(path, row)


class ParityFinalRevalidationTest(unittest.TestCase):
    def fixture(
        self,
        root: Path,
    ) -> tuple[list[str], dict[str, str], Path]:
        rows = []
        for speaker, speaker_id in PARITY.SPEAKERS.items():
            canonical = root / f"{speaker}.npz"
            lower_foot = root / f"{speaker}.lower_foot.npy"
            canonical.write_bytes(f"canonical:{speaker}".encode())
            lower_foot.write_bytes(f"lower-foot:{speaker}".encode())
            rows.append(
                {
                    "clip_id": f"{speaker}_clip",
                    "split": "train",
                    "speaker": speaker,
                    "speaker_id": speaker_id,
                    "frames": 64,
                    "global_index": len(rows),
                    "canonical_npz": str(canonical),
                    "canonical_npz_sha256": hashlib.sha256(
                        canonical.read_bytes()
                    ).hexdigest(),
                    "lower_foot_local": str(lower_foot),
                    "lower_foot_local_sha256": hashlib.sha256(
                        lower_foot.read_bytes()
                    ).hexdigest(),
                }
            )
        speaker_names = tuple(PARITY.SPEAKERS)
        for global_index in range(
            len(rows),
            sum(PARITY.EXPECTED_SPLIT_COUNTS.values()),
        ):
            if global_index < PARITY.EXPECTED_SPLIT_COUNTS["train"]:
                split = "train"
            elif global_index < (
                PARITY.EXPECTED_SPLIT_COUNTS["train"]
                + PARITY.EXPECTED_SPLIT_COUNTS["val"]
            ):
                split = "val"
            else:
                split = "test"
            speaker = speaker_names[global_index % len(speaker_names)]
            rows.append(
                {
                    "clip_id": f"{speaker}/filler_{global_index:05d}",
                    "split": split,
                    "speaker": speaker,
                    "speaker_id": PARITY.SPEAKERS[speaker],
                    "frames": 64,
                    "global_index": global_index,
                }
            )
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        canonical_source = {
            "origin": PARITY.EXPECTED_ORIGIN,
            "commit": CANONICAL_COMMIT,
            "tree": CANONICAL_TREE,
            "entrypoint": "/frozen/build_show_cache.py",
            "entrypoint_sha256": SHA256,
        }
        contract = {"source_receipt": canonical_source}
        contract_sha = PARITY.canonical_payload_sha256(contract)
        lineage = {
            "final_manifest_sha256": manifest_sha,
            "lineage_contract": contract,
            "lineage_contract_sha256": contract_sha,
        }
        summary = {
            "status": "complete",
            "schema_name": "semtalk-show-canonical-motion",
            "schema_version": 1,
            "clip_count": 17_110,
            "manifest_sha256": manifest_sha,
            "split_counts": {
                "train": 13_687,
                "val": 1_715,
                "test": 1_708,
            },
            "split_disjoint": True,
            "exact_once": True,
            "finite": True,
            "lineage_sha256": PARITY.canonical_payload_sha256(lineage),
            "lineage_contract_sha256": contract_sha,
            "source_receipt_sha256": (
                PARITY.canonical_payload_sha256(canonical_source)
            ),
        }
        lineage_path = root / "lineage.json"
        summary_path = root / "summary.json"
        lineage_path.write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        smplx = root / "SMPLX_NEUTRAL_2020.npz"
        smplx.write_bytes(b"synthetic-smplx")
        smplx_sha = hashlib.sha256(smplx.read_bytes()).hexdigest()
        output = root / "parity"
        argv = [
            "--canonical-manifest",
            str(manifest),
            "--canonical-summary",
            str(summary_path),
            "--canonical-lineage",
            str(lineage_path),
            "--smplx-model",
            str(smplx),
            "--expected-smplx-sha256",
            smplx_sha,
            "--output-root",
            str(output),
            *producer_args(),
        ]
        producer_source = {
            "origin": PARITY.EXPECTED_ORIGIN,
            "commit": PRODUCER_COMMIT,
            "tree": PRODUCER_TREE,
            "entrypoint": str(PARITY.__file__),
            "entrypoint_sha256": hashlib.sha256(
                Path(PARITY.__file__).read_bytes()
            ).hexdigest(),
        }
        return argv, producer_source, output

    @staticmethod
    def fake_checker(command, *, cwd, check):
        del cwd, check
        canonical = Path(
            command[command.index("--canonical-npz") + 1]
        )
        lower_foot = Path(
            command[command.index("--lower-foot-local") + 1]
        )
        smplx = Path(command[command.index("--smplx-model") + 1])
        report = Path(command[command.index("--report-json") + 1])
        payload = {
            "status": "pass",
            "contract": PARITY.CONTRACT,
            "canonical_npz_sha256": hashlib.sha256(
                canonical.read_bytes()
            ).hexdigest(),
            "lower_foot_local_sha256": hashlib.sha256(
                lower_foot.read_bytes()
            ).hexdigest(),
            "smplx_asset_sha256": hashlib.sha256(
                smplx.read_bytes()
            ).hexdigest(),
        }
        report.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return mock.Mock(returncode=0)

    def run_parity(
        self,
        argv: list[str],
        source_effect,
    ) -> None:
        with (
            mock.patch.object(sys, "argv", [str(PARITY.__file__), *argv]),
            mock.patch.object(
                PARITY,
                "source_receipt",
                side_effect=source_effect,
            ),
            mock.patch.object(
                PARITY.subprocess,
                "run",
                side_effect=self.fake_checker,
            ),
        ):
            PARITY.main()

    def test_success_revalidates_and_publishes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv, source, output = self.fixture(Path(directory))
            self.run_parity(argv, [source, source, source])
            bundle = json.loads((output / "bundle.json").read_text())
            self.assertEqual(bundle["status"], "pass")
            self.assertEqual(bundle["source_receipt"], source)
            self.assertEqual(
                bundle["canonical_receipt"]["source_receipt"]["commit"],
                CANONICAL_COMMIT,
            )

    def test_changed_current_source_prevents_bundle_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv, source, output = self.fixture(Path(directory))
            changed = {**source, "tree": "d" * 40}
            with self.assertRaisesRegex(
                RuntimeError,
                "inputs changed during parity execution",
            ):
                self.run_parity(argv, [source, changed])
            self.assertFalse((output / "bundle.json").exists())

    def test_frozen_canonical_source_is_independent_in_parity_loader(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv, _, _ = self.fixture(Path(directory))
            parsed = parse_with(PARITY, argv)
            _, receipt = PARITY.load_canonical_receipt(
                parsed.canonical_manifest,
                parsed.canonical_summary,
                parsed.canonical_lineage,
                CANONICAL_COMMIT,
                CANONICAL_TREE,
            )
            self.assertEqual(
                receipt["source_receipt"]["commit"],
                CANONICAL_COMMIT,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "canonical source receipt mismatch",
            ):
                PARITY.load_canonical_receipt(
                    parsed.canonical_manifest,
                    parsed.canonical_summary,
                    parsed.canonical_lineage,
                    PRODUCER_COMMIT,
                    PRODUCER_TREE,
                )

    def test_manifest_counts_must_match_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv, _, _ = self.fixture(Path(directory))
            parsed = parse_with(PARITY, argv)
            lines = parsed.canonical_manifest.read_text().splitlines()
            parsed.canonical_manifest.write_text(
                "\n".join(lines[:-1]) + "\n",
                encoding="utf-8",
            )
            manifest_sha = hashlib.sha256(
                parsed.canonical_manifest.read_bytes()
            ).hexdigest()
            lineage_payload = json.loads(
                parsed.canonical_lineage.read_text()
            )
            lineage_payload["final_manifest_sha256"] = manifest_sha
            parsed.canonical_lineage.write_text(
                json.dumps(lineage_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary_payload = json.loads(
                parsed.canonical_summary.read_text()
            )
            summary_payload["manifest_sha256"] = manifest_sha
            summary_payload["lineage_sha256"] = (
                PARITY.canonical_payload_sha256(lineage_payload)
            )
            parsed.canonical_summary.write_text(
                json.dumps(summary_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "canonical manifest has",
            ):
                PARITY.load_canonical_receipt(
                    parsed.canonical_manifest,
                    parsed.canonical_summary,
                    parsed.canonical_lineage,
                    CANONICAL_COMMIT,
                    CANONICAL_TREE,
                )

    def test_changed_report_prevents_bundle_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv, source, output = self.fixture(Path(directory))
            calls = 0

            def source_effect(*_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    (output / "oliver.json").write_text(
                        "{}\n",
                        encoding="utf-8",
                    )
                return source

            with self.assertRaisesRegex(
                RuntimeError,
                "parity report changed during execution",
            ):
                self.run_parity(argv, source_effect)
            self.assertFalse((output / "bundle.json").exists())

    def test_terminal_mutation_prevents_bundle_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv, source, output = self.fixture(Path(directory))
            original_atomic = PARITY.atomic_json_new

            def mutate_after_temp_fsync(
                path,
                payload,
                *,
                before_replace,
            ):
                def mutate_then_revalidate():
                    (output / "oliver.json").write_text(
                        "{}\n",
                        encoding="utf-8",
                    )
                    before_replace()

                return original_atomic(
                    path,
                    payload,
                    before_replace=mutate_then_revalidate,
                )

            with (
                mock.patch.object(
                    PARITY,
                    "atomic_json_new",
                    side_effect=mutate_after_temp_fsync,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "parity report changed during execution",
                ),
            ):
                self.run_parity(argv, [source, source, source])
            self.assertFalse((output / "bundle.json").exists())
            self.assertEqual(
                list(output.glob(".bundle.json.tmp.*")),
                [],
            )

class FormalConfigIsolationTest(unittest.TestCase):
    def test_effective_formal_configs_have_no_legacy_weight_or_data_paths(
        self,
    ) -> None:
        try:
            from utils import config
        except ImportError as exc:
            self.skipTest(f"formal config runtime is unavailable: {exc}")
        configs = (
            "configs/cnn_vqvae_face_30.yaml",
            "configs/cnn_vqvae_hands_30.yaml",
            "configs/cnn_vqvae_upper_30.yaml",
            "configs/cnn_vqvae_lower_30.yaml",
            "configs/cnn_vqvae_lower_foot_30.yaml",
            "configs/semtalk_base.yaml",
        )
        for relative in configs:
            with self.subTest(config=relative):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "show_base_train.py",
                        "--config",
                        str(ROOT / relative),
                        "--data_path_1",
                        "/frozen/eval-assets/",
                        "--run_name",
                        "formal_config_test",
                    ],
                ):
                    args = config.parse_args()
                self.assertEqual(args.training_speakers, [0, 1, 2, 3])
                self.assertEqual(args.dataset, "show_base")
                self.assertEqual(args.data_path, "")
                self.assertEqual(args.cache_path, "")
                self.assertEqual(args.e_path, "")
                self.assertIsNone(args.e_name)
                self.assertEqual(args.test_ckpt, "")
                self.assertEqual(args.test_path, "")
                self.assertEqual(args.data_path_1, "/frozen/eval-assets/")
                self.assertEqual(args.hubert_mean_path, "")
                self.assertEqual(args.hubert_std_path, "")
                self.assertEqual(args.audio_infer_path, "")
                self.assertEqual(args.base_ckpt, "")
        with mock.patch.object(
            sys,
            "argv",
            [
                "show_base_train.py",
                "--config",
                str(ROOT / "configs/semtalk_base.yaml"),
                "--run_name",
                "formal_config_test",
            ],
        ):
            base = config.parse_args()
        self.assertEqual(base.load_ckpt, "")
        self.assertEqual(base.train_path, "")
        self.assertEqual(base.word_rep, "disabled_zero_placeholder")
        self.assertEqual(base.word_index_num, 0)
        self.assertEqual(base.word_dims, 0)
        self.assertEqual(base.word_f, 0)
        self.assertEqual(base.t_pre_encoder, "disabled")
        self.assertTrue(base.freeze_wordembed)


if __name__ == "__main__":
    unittest.main()
