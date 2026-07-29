from __future__ import annotations

import hashlib
import importlib.util
import io
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
    import show_base_train as FORMAL
except ImportError:  # pragma: no cover - minimal local environments
    torch = None
    FORMAL = None


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


@unittest.skipIf(torch is None, "torch runtime is unavailable")
class CheckpointSnapshotPrimitiveTest(unittest.TestCase):
    def test_ordinary_checkpoint_deserializes_the_verified_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordinary.bin"
            torch.save(
                {
                    "model_state": torch.nn.Linear(3, 2).state_dict(),
                    "audit": {
                        "format": "semtalk_show_model_v2",
                        "formal_stage": "face",
                    },
                },
                path,
            )
            expected_bytes = path.read_bytes()
            expected_sha = hashlib.sha256(expected_bytes).hexdigest()
            original_read_bytes = Path.read_bytes

            def read_then_replace(target: Path) -> bytes:
                payload = original_read_bytes(target)
                target.write_bytes(b"changed-after-checkpoint-snapshot")
                return payload

            with mock.patch.object(
                Path,
                "read_bytes",
                new=read_then_replace,
            ):
                resolved, snapshot, observed_sha = (
                    MODULE._read_verified_checkpoint_snapshot(
                        path,
                        expected_sha,
                        "ordinary checkpoint",
                    )
                )
                loaded = MODULE._torch_load_checkpoint(snapshot, resolved)
                builder_loaded = BUILDER._torch_load(snapshot, resolved)
            self.assertEqual(observed_sha, expected_sha)
            self.assertEqual(snapshot, expected_bytes)
            self.assertEqual(
                set(loaded["model_state"]),
                set(torch.nn.Linear(3, 2).state_dict()),
            )
            self.assertEqual(
                set(builder_loaded["model_state"]),
                set(loaded["model_state"]),
            )


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


class FormalCliContractTest(unittest.TestCase):
    @staticmethod
    def valid_argv() -> list[str]:
        sha = "a" * 64
        argv = [
            "--canonical-manifest",
            "/frozen/canonical.jsonl",
            "--canonical-summary-json",
            "/frozen/canonical-summary.json",
            "--canonical-lineage-json",
            "/frozen/canonical-lineage.json",
            "--expected-canonical-manifest-sha256",
            sha,
            "--audio-manifest",
            *[f"/frozen/audio-{index}.jsonl" for index in range(8)],
            "--audio-summary-json",
            *[f"/frozen/audio-{index}.summary.json" for index in range(8)],
            "--audio-lineage-json",
            *[f"/frozen/audio-{index}.lineage.json" for index in range(8)],
            "--base-training-lineage-manifest",
            "/frozen/base-lineage.json",
            "--base-training-summary-json",
            "/frozen/base-summary.json",
            "--representation-training-lineage-manifest",
            "/frozen/representation-lineage.json",
            "--expected-inference-script-sha256",
            sha,
            "--expected-source-commit",
            "b" * 40,
            "--expected-source-tree",
            "c" * 40,
            "--expected-canonical-source-commit",
            "d" * 40,
            "--expected-canonical-source-tree",
            "e" * 40,
            "--expected-hubert-tree-sha256",
            sha,
        ]
        for stage in MODULE.CHECKPOINT_STAGES:
            argv.extend(
                [
                    f"--{stage}-checkpoint",
                    f"/frozen/{stage}.bin",
                    f"--expected-{stage}-sha256",
                    sha,
                    f"--{stage}-status-json",
                    f"/frozen/{stage}-status.json",
                ]
            )
        argv.extend(
            [
                "--base-candidate-manifest",
                "/frozen/base_candidate_manifest.json",
                "--expected-base-candidate-manifest-sha256",
                sha,
                "--expected-base-formal-status-sha256",
                sha,
                "--expected-base-final-checkpoint-sha256",
                sha,
                "--output-root",
                "/output",
            ]
        )
        return argv

    def test_all_candidate_trust_roots_are_cli_required(self) -> None:
        required = (
            "--base-candidate-manifest",
            "--expected-base-candidate-manifest-sha256",
            "--expected-base-formal-status-sha256",
            "--expected-base-final-checkpoint-sha256",
        )
        MODULE.parse_args(self.valid_argv())
        for option in required:
            with self.subTest(option=option):
                argv = self.valid_argv()
                index = argv.index(option)
                del argv[index : index + 2]
                with self.assertRaises(SystemExit) as raised:
                    MODULE.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_canonical_and_producer_source_roots_are_independently_required(
        self,
    ) -> None:
        argv = self.valid_argv()
        parsed = MODULE.parse_args(argv)
        self.assertNotEqual(
            parsed.expected_source_commit,
            parsed.expected_canonical_source_commit,
        )
        self.assertNotEqual(
            parsed.expected_source_tree,
            parsed.expected_canonical_source_tree,
        )
        for option in (
            "--expected-source-commit",
            "--expected-source-tree",
            "--expected-canonical-source-commit",
            "--expected-canonical-source-tree",
        ):
            with self.subTest(option=option):
                incomplete = self.valid_argv()
                index = incomplete.index(option)
                del incomplete[index : index + 2]
                with self.assertRaises(SystemExit) as raised:
                    MODULE.parse_args(incomplete)
                self.assertEqual(raised.exception.code, 2)

    def test_formal_launcher_owns_all_base_trust_root_arguments(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "show_base"
            / "run_base_formal_inference.sh"
        ).read_text(encoding="utf-8")
        for option in (
            "--base-checkpoint",
            "--expected-base-sha256",
            "--base-status-json",
            "--base-candidate-manifest",
            "--expected-base-candidate-manifest-sha256",
            "--expected-base-formal-status-sha256",
            "--expected-base-final-checkpoint-sha256",
        ):
            self.assertGreaterEqual(launcher.count(option), 2, option)

    def test_argparse_abbreviation_cannot_override_launcher_checkpoint(
        self,
    ) -> None:
        argv = self.valid_argv()
        checkpoint_index = argv.index("--base-checkpoint")
        argv[checkpoint_index] = "--base-checkp"
        argv[checkpoint_index + 1] = "/attacker/override.bin"
        with self.assertRaises(SystemExit) as raised:
            MODULE.parse_args(argv)
        self.assertEqual(raised.exception.code, 2)


class ExactIntegerBoundaryTest(unittest.TestCase):
    def test_helpers_reject_bool_string_and_float(self) -> None:
        for invalid in (True, False, "1", 1.0):
            with self.subTest(consumer="builder", invalid=invalid):
                with self.assertRaises(RuntimeError):
                    BUILDER.require_exact_int(invalid, "fixture")
            with self.subTest(consumer="inference", invalid=invalid):
                with self.assertRaises(MODULE.InferenceContractError):
                    MODULE._require_exact_int(invalid, "fixture")
        self.assertEqual(BUILDER.require_exact_int(1, "fixture"), 1)
        self.assertEqual(MODULE._require_exact_int(1, "fixture"), 1)

    def test_json_consumers_do_not_use_permissive_int_coercion(self) -> None:
        untrusted_names = (
            "summary",
            "split_counts",
            "row",
            "record",
            "lineage",
            "protocol",
            "canonical",
            "value",
        )
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "scripts/show_base/build_base_features.py",
            "scripts/show_base/run_base_inference.py",
        ):
            source = (root / relative).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
            violations = []
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "int"
                    and node.args
                ):
                    continue
                names = {
                    child.id
                    for child in ast.walk(node.args[0])
                    if isinstance(child, ast.Name)
                }
                if names.intersection(untrusted_names):
                    violations.append(
                        ast.get_source_segment(source, node)
                    )
            self.assertEqual(violations, [], relative)

        for relative in (
            "scripts/show_base/run_five_prerequisites.sh",
            "scripts/show_base/run_base_training.sh",
        ):
            source = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn('entries = int(summary["entries"])', source)
            self.assertIn("entries = require_exact_int(", source)
            self.assertIn("type(value) is not int", source)


@unittest.skipIf(torch is None, "torch/formal runtime is unavailable")
class FormalCheckpointConsumerCompatibilityTest(unittest.TestCase):
    @staticmethod
    def source_receipt() -> dict[str, object]:
        return {
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "commit": "1" * 40,
            "tree": "2" * 40,
            "entrypoint": "/immutable/show_base_train.py",
            "entrypoint_sha256": "3" * 64,
        }

    @staticmethod
    def dataset_receipt(
        *,
        summary_sha: str,
        lineage_sha: str,
        data_sha: str,
        smplx_asset: object,
    ) -> dict[str, object]:
        return {
            "summary_sha256": summary_sha,
            "lineage_sha256": lineage_sha,
            "data_mdb_sha256": data_sha,
            "entries": 127_309,
            "train_clips": 13_687,
            "smplx_asset": smplx_asset,
        }

    @staticmethod
    def write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_producer_model_v2_is_strictly_consumed_by_both_readers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "rvq_face_600.bin"
            status_path = root / "formal_status.json"
            source = self.source_receipt()
            lineage_sha = "4" * 64
            summary_sha = "5" * 64
            data_sha = "6" * 64
            config_sha = "7" * 64
            smplx_asset = {
                "format": "semtalk_show_smplx_asset_v1",
                "sha256": "8" * 64,
            }
            dataset = self.dataset_receipt(
                summary_sha=summary_sha,
                lineage_sha=lineage_sha,
                data_sha=data_sha,
                smplx_asset=smplx_asset,
            )
            trainer = SimpleNamespace(model=torch.nn.Linear(3, 2))
            optimizer_updates = 600 * 1_989
            payload = FORMAL._model_payload(
                trainer,
                formal_stage="face",
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_receipt=dataset,
                source_receipt=source,
                optimizer_updates=optimizer_updates,
                candidate_manifest_receipt=None,
            )
            torch.save(payload, checkpoint)
            checkpoint_sha = MODULE.sha256_file(checkpoint)
            status = {
                "status": "complete",
                "formal_stage": "face",
                "world_size": 1,
                "epochs": 600,
                "completed_epochs": 600,
                "train_samples": 127_309,
                "updates_per_epoch": 1_989,
                "optimizer_updates": optimizer_updates,
                "lineage_manifest_sha256": lineage_sha,
                "config_sha256": config_sha,
                "dataset_receipt": dataset,
                "smplx_asset_receipt": smplx_asset,
                "base_candidate_manifest": None,
                "source_receipt": source,
                "source_receipt_sha256": MODULE.compact_json_sha256(
                    source
                ),
                "final_checkpoint": str(checkpoint),
                "final_checkpoint_sha256": checkpoint_sha,
            }
            self.write_json(status_path, status)

            _, builder_receipt = BUILDER.checkpoint_record(
                checkpoint,
                formal_stage="face",
                expected_lineage_sha256=lineage_sha,
                status_path=status_path,
                expected_source_receipt=source,
            )
            self.assertEqual(
                builder_receipt["audit"]["format"],
                "semtalk_show_model_v2",
            )
            _, inference_receipt = MODULE._checkpoint_payload_and_receipt(
                checkpoint,
                formal_stage="face",
                expected_sha256=checkpoint_sha,
                expected_training_lineage_sha256=lineage_sha,
                status_path=status_path,
                expected_source_receipt=source,
                expected_dataset_summary_sha256=summary_sha,
                expected_data_mdb_sha256=data_sha,
            )
            self.assertEqual(
                inference_receipt["audit"]["format"],
                "semtalk_show_model_v2",
            )

            for invalid_updates in (
                optimizer_updates - 1,
                str(optimizer_updates),
                float(optimizer_updates),
                True,
            ):
                with self.subTest(invalid_updates=invalid_updates):
                    tampered = {
                        "model_state": payload["model_state"],
                        "audit": dict(payload["audit"]),
                    }
                    tampered["audit"]["optimizer_updates"] = invalid_updates
                    torch.save(tampered, checkpoint)
                    status["final_checkpoint_sha256"] = MODULE.sha256_file(
                        checkpoint
                    )
                    self.write_json(status_path, status)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "model_v2 dataset/SMPL-X/update audit",
                    ):
                        BUILDER.checkpoint_record(
                            checkpoint,
                            formal_stage="face",
                            expected_lineage_sha256=lineage_sha,
                            status_path=status_path,
                            expected_source_receipt=source,
                        )
                    with self.assertRaises(
                        MODULE.InferenceContractError
                    ):
                        MODULE._checkpoint_payload_and_receipt(
                            checkpoint,
                            formal_stage="face",
                            expected_sha256=(
                                status["final_checkpoint_sha256"]
                            ),
                            expected_training_lineage_sha256=lineage_sha,
                            status_path=status_path,
                            expected_source_receipt=source,
                            expected_dataset_summary_sha256=summary_sha,
                            expected_data_mdb_sha256=data_sha,
                        )

            torch.save(payload, checkpoint)
            checkpoint_sha = MODULE.sha256_file(checkpoint)
            status["final_checkpoint_sha256"] = checkpoint_sha
            for invalid_accounting in (
                str(optimizer_updates),
                float(optimizer_updates),
                True,
            ):
                with self.subTest(invalid_accounting=invalid_accounting):
                    status["optimizer_updates"] = invalid_accounting
                    self.write_json(status_path, status)
                    with self.assertRaises(RuntimeError):
                        BUILDER.checkpoint_record(
                            checkpoint,
                            formal_stage="face",
                            expected_lineage_sha256=lineage_sha,
                            status_path=status_path,
                            expected_source_receipt=source,
                        )
                    with self.assertRaises(
                        MODULE.InferenceContractError
                    ):
                        MODULE._checkpoint_payload_and_receipt(
                            checkpoint,
                            formal_stage="face",
                            expected_sha256=checkpoint_sha,
                            expected_training_lineage_sha256=lineage_sha,
                            status_path=status_path,
                            expected_source_receipt=source,
                            expected_dataset_summary_sha256=summary_sha,
                            expected_data_mdb_sha256=data_sha,
                        )
            status["optimizer_updates"] = optimizer_updates
            checkpoint_link = root / "face-final-link.bin"
            checkpoint_link.symlink_to(checkpoint)
            status["final_checkpoint"] = str(checkpoint_link)
            self.write_json(status_path, status)
            with self.assertRaises(RuntimeError):
                BUILDER.checkpoint_record(
                    checkpoint,
                    formal_stage="face",
                    expected_lineage_sha256=lineage_sha,
                    status_path=status_path,
                    expected_source_receipt=source,
                )
            with self.assertRaises(MODULE.InferenceContractError):
                MODULE._checkpoint_payload_and_receipt(
                    checkpoint,
                    formal_stage="face",
                    expected_sha256=checkpoint_sha,
                    expected_training_lineage_sha256=lineage_sha,
                    status_path=status_path,
                    expected_source_receipt=source,
                    expected_dataset_summary_sha256=summary_sha,
                    expected_data_mdb_sha256=data_sha,
                )

    def test_legacy_model_v1_is_rejected_before_status_is_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "legacy-face.bin"
            torch.save(
                {
                    "model_state": torch.nn.Linear(3, 2).state_dict(),
                    "audit": {
                        "format": "semtalk_show_model_v1",
                        "formal_stage": "face",
                    },
                },
                checkpoint,
            )
            checkpoint_sha = MODULE.sha256_file(checkpoint)
            with self.assertRaisesRegex(RuntimeError, "model_v2"):
                BUILDER.checkpoint_record(
                    checkpoint,
                    formal_stage="face",
                    expected_lineage_sha256="1" * 64,
                    status_path=root / "untrusted-status.json",
                    expected_source_receipt=self.source_receipt(),
                )
            with self.assertRaisesRegex(
                MODULE.InferenceContractError,
                "model_v2",
            ):
                MODULE._checkpoint_payload_and_receipt(
                    checkpoint,
                    formal_stage="face",
                    expected_sha256=checkpoint_sha,
                    expected_training_lineage_sha256="1" * 64,
                    status_path=root / "untrusted-status.json",
                    expected_source_receipt=self.source_receipt(),
                    expected_dataset_summary_sha256="2" * 64,
                    expected_data_mdb_sha256="3" * 64,
                )

    def test_completed_base_final_cannot_bypass_candidate_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "semtalk_base_epoch_400.bin"
            torch.save(
                {
                    "model_state": torch.nn.Linear(3, 2).state_dict(),
                    "audit": {
                        "format": "semtalk_show_model_v2",
                        "formal_stage": "base",
                    },
                },
                checkpoint,
            )
            with self.assertRaisesRegex(
                MODULE.InferenceContractError,
                "must select one immutable checkpoint",
            ):
                MODULE._checkpoint_payload_and_receipt(
                    checkpoint,
                    formal_stage="base",
                    expected_sha256=MODULE.sha256_file(checkpoint),
                    expected_training_lineage_sha256="1" * 64,
                    status_path=root / "untrusted-status.json",
                    expected_source_receipt=self.source_receipt(),
                    expected_dataset_summary_sha256="2" * 64,
                    expected_data_mdb_sha256="3" * 64,
                )

    def build_candidate_fixture(
        self,
        root: Path,
    ) -> dict[str, object]:
        source = self.source_receipt()
        lineage_sha = "9" * 64
        summary_sha = "a" * 64
        data_sha = "b" * 64
        config_sha = "c" * 64
        dataset = self.dataset_receipt(
            summary_sha=summary_sha,
            lineage_sha=lineage_sha,
            data_sha=data_sha,
            smplx_asset=None,
        )
        manifest_path = root / "base_candidate_manifest.json"
        candidate_dir = root / "base_candidates"
        candidate_dir.mkdir()
        model = torch.nn.Linear(3, 2)
        entries: list[dict[str, object]] = []
        previous_manifest_sha: str | None = None
        selected_path: Path | None = None
        selected_sha: str | None = None
        for index in range(40):
            epoch = (index + 1) * 10
            updates = epoch * 1_989
            audit = FORMAL._base_candidate_audit(
                epoch=epoch,
                optimizer_updates=updates,
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_receipt=dataset,
                source_receipt=source,
            )
            core = FORMAL._base_candidate_transaction_core(
                epoch=epoch,
                optimizer_updates=updates,
                previous_manifest_sha256=previous_manifest_sha,
                previous_entries=entries,
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_receipt=dataset,
                source_receipt=source,
            )
            relative_path = FORMAL._candidate_relative_path(epoch, updates)
            candidate_path = root / relative_path
            torch.save(
                {"model_state": model.state_dict(), "audit": audit},
                candidate_path,
            )
            checkpoint_sha = MODULE.sha256_file(candidate_path)
            record = {
                "epoch": epoch,
                "optimizer_updates": updates,
                "checkpoint": relative_path,
                "checkpoint_sha256": checkpoint_sha,
                "model_audit_sha256": FORMAL._payload_sha256(audit),
                "transaction_core": core,
                "transaction_core_sha256": FORMAL._payload_sha256(core),
            }
            entries.append(record)
            manifest_payload = FORMAL._base_candidate_manifest_payload(
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_receipt=dataset,
                source_receipt=source,
                entries=entries,
            )
            previous_manifest_sha = FORMAL._json_document_sha256(
                manifest_payload
            )
            if epoch == 10:
                selected_path = candidate_path
                selected_sha = checkpoint_sha

        FORMAL._atomic_json(manifest_path, manifest_payload)
        manifest_sha = MODULE.sha256_file(manifest_path)
        manifest_receipt = {
            "path": manifest_path.name,
            "sha256": manifest_sha,
            "entries": 40,
            "last_epoch": 400,
            "last_optimizer_updates": 400 * 1_989,
        }
        final_path = root / "semtalk_base_epoch_400.bin"
        final_payload = FORMAL._model_payload(
            SimpleNamespace(model=model),
            formal_stage="base",
            config_sha256=config_sha,
            lineage_sha256=lineage_sha,
            dataset_receipt=dataset,
            source_receipt=source,
            optimizer_updates=400 * 1_989,
            candidate_manifest_receipt=manifest_receipt,
        )
        torch.save(final_payload, final_path)
        status_path = root / "formal_status.json"
        status = {
            "status": "complete",
            "formal_stage": "base",
            "world_size": 1,
            "epochs": 400,
            "completed_epochs": 400,
            "train_samples": 127_309,
            "updates_per_epoch": 1_989,
            "optimizer_updates": 400 * 1_989,
            "lineage_manifest_sha256": lineage_sha,
            "config_sha256": config_sha,
            "dataset_receipt": dataset,
            "smplx_asset_receipt": None,
            "base_candidate_manifest": manifest_receipt,
            "source_receipt": source,
            "source_receipt_sha256": MODULE.compact_json_sha256(source),
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": MODULE.sha256_file(final_path),
        }
        self.write_json(status_path, status)
        assert selected_path is not None and selected_sha is not None
        return {
            "source": source,
            "lineage_sha": lineage_sha,
            "summary_sha": summary_sha,
            "data_sha": data_sha,
            "manifest_path": manifest_path,
            "manifest_sha": manifest_sha,
            "status_path": status_path,
            "status_sha": MODULE.sha256_file(status_path),
            "selected_path": selected_path,
            "selected_sha": selected_sha,
            "final_path": final_path,
            "final_sha": MODULE.sha256_file(final_path),
        }

    @staticmethod
    def candidate_kwargs(fixture: dict[str, object]) -> dict[str, object]:
        return {
            "formal_stage": "base",
            "expected_sha256": fixture["selected_sha"],
            "expected_training_lineage_sha256": fixture["lineage_sha"],
            "status_path": fixture["status_path"],
            "expected_source_receipt": fixture["source"],
            "expected_dataset_summary_sha256": fixture["summary_sha"],
            "expected_data_mdb_sha256": fixture["data_sha"],
            "base_candidate_manifest_path": fixture["manifest_path"],
            "expected_base_candidate_manifest_sha256": fixture[
                "manifest_sha"
            ],
            "expected_base_formal_status_sha256": fixture["status_sha"],
            "expected_base_final_checkpoint_sha256": fixture["final_sha"],
        }

    def consume_candidate(
        self,
        fixture: dict[str, object],
        *,
        kwargs: dict[str, object] | None = None,
    ):
        with mock.patch.object(
            MODULE,
            "_validate_base_model_state_schema",
        ):
            return MODULE._checkpoint_payload_and_receipt(
                fixture["selected_path"],
                **(
                    self.candidate_kwargs(fixture)
                    if kwargs is None
                    else kwargs
                ),
            )

    def test_candidate_is_consumed_from_manifest_without_pretending_final(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_candidate_fixture(Path(directory))
            _, receipt = self.consume_candidate(fixture)
            self.assertEqual(receipt["checkpoint_kind"], "immutable_candidate")
            self.assertEqual(receipt["candidate_epoch"], 10)
            self.assertNotEqual(
                receipt["path"],
                receipt["completed_final_checkpoint"]["path"],
            )

            with self.assertRaisesRegex(
                MODULE.InferenceContractError,
                "requires the immutable",
            ):
                MODULE._checkpoint_payload_and_receipt(
                    fixture["selected_path"],
                    formal_stage="base",
                    expected_sha256=fixture["selected_sha"],
                    expected_training_lineage_sha256=fixture["lineage_sha"],
                    status_path=fixture["status_path"],
                    expected_source_receipt=fixture["source"],
                    expected_dataset_summary_sha256=fixture["summary_sha"],
                    expected_data_mdb_sha256=fixture["data_sha"],
                )

    def test_candidate_hashes_every_unselected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_candidate_fixture(Path(directory))
            unselected = (
                Path(directory)
                / FORMAL._candidate_relative_path(20, 20 * 1_989)
            )
            unselected.write_bytes(unselected.read_bytes() + b"tampered")
            with self.assertRaisesRegex(
                MODULE.InferenceContractError,
                "candidate checkpoint SHA differs",
            ):
                self.consume_candidate(fixture)

    def test_candidate_requires_external_status_and_final_trust_roots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_candidate_fixture(Path(directory))
            kwargs = self.candidate_kwargs(fixture)
            kwargs["expected_base_formal_status_sha256"] = "d" * 64
            with self.assertRaisesRegex(
                MODULE.InferenceContractError,
                "formal training status SHA mismatch",
            ):
                self.consume_candidate(fixture, kwargs=kwargs)
            kwargs = self.candidate_kwargs(fixture)
            kwargs["expected_base_final_checkpoint_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                MODULE.InferenceContractError,
                "invalid complete Base candidate training receipt",
            ):
                self.consume_candidate(fixture, kwargs=kwargs)

    def test_candidate_rejects_invalid_completed_final_envelope_and_schema(
        self,
    ) -> None:
        for case in ("empty_model_state", "extra_envelope", "missing_state_key"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_candidate_fixture(Path(directory))
                final_path = fixture["final_path"]
                final_payload = torch.load(
                    final_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if case == "empty_model_state":
                    final_payload["model_state"] = {}
                elif case == "extra_envelope":
                    final_payload["extra"] = "forbidden"
                else:
                    final_payload["model_state"].pop(
                        next(iter(final_payload["model_state"]))
                    )
                torch.save(final_payload, final_path)
                final_sha = MODULE.sha256_file(final_path)
                status = json.loads(
                    fixture["status_path"].read_text(encoding="utf-8")
                )
                status["final_checkpoint_sha256"] = final_sha
                self.write_json(fixture["status_path"], status)
                fixture["final_sha"] = final_sha
                fixture["status_sha"] = MODULE.sha256_file(
                    fixture["status_path"]
                )
                with self.assertRaises(MODULE.InferenceContractError):
                    self.consume_candidate(fixture)

    def test_candidate_transaction_integer_is_exact(self) -> None:
        for invalid in ("0", 0.0, False):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_candidate_fixture(Path(directory))
                manifest = json.loads(
                    fixture["manifest_path"].read_text(encoding="utf-8")
                )
                core = manifest["entries"][0]["transaction_core"]
                core["previous_entry_count"] = invalid
                manifest["entries"][0][
                    "transaction_core_sha256"
                ] = FORMAL._payload_sha256(core)
                FORMAL._atomic_json(fixture["manifest_path"], manifest)
                manifest_sha = MODULE.sha256_file(fixture["manifest_path"])
                status = json.loads(
                    fixture["status_path"].read_text(encoding="utf-8")
                )
                status["base_candidate_manifest"]["sha256"] = manifest_sha
                self.write_json(fixture["status_path"], status)
                fixture["manifest_sha"] = manifest_sha
                fixture["status_sha"] = MODULE.sha256_file(
                    fixture["status_path"]
                )
                with self.assertRaises(MODULE.InferenceContractError):
                    self.consume_candidate(fixture)

    def test_selected_candidate_loads_the_verified_byte_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_candidate_fixture(Path(directory))
            selected = Path(fixture["selected_path"]).resolve()
            original_bytes = selected.read_bytes()
            original_read_bytes = Path.read_bytes

            def read_then_replace(path: Path) -> bytes:
                payload = original_read_bytes(path)
                if path.resolve() == selected:
                    path.write_bytes(b"changed-after-selected-snapshot")
                return payload

            with mock.patch.object(
                Path,
                "read_bytes",
                new=read_then_replace,
            ):
                _, receipt = self.consume_candidate(fixture)
            self.assertEqual(receipt["sha256"], fixture["selected_sha"])
            self.assertEqual(receipt["bytes"], len(original_bytes))

    def test_completed_final_loads_the_verified_byte_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_candidate_fixture(Path(directory))
            final_path = Path(fixture["final_path"]).resolve()
            original_bytes = final_path.read_bytes()
            original_read_bytes = Path.read_bytes

            def read_then_replace(path: Path) -> bytes:
                payload = original_read_bytes(path)
                if path.resolve() == final_path:
                    path.write_bytes(b"changed-after-final-snapshot")
                return payload

            with mock.patch.object(
                Path,
                "read_bytes",
                new=read_then_replace,
            ):
                _, receipt = self.consume_candidate(fixture)
            self.assertEqual(
                receipt["completed_final_checkpoint"]["sha256"],
                hashlib.sha256(original_bytes).hexdigest(),
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
                "clip_id": "oliver/snapshot",
                "speaker": "oliver",
                "speaker_id": 0,
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
