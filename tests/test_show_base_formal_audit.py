from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

try:
    import torch
    import show_base_train as formal
    import train
except ImportError:  # pragma: no cover - minimal local environments
    torch = None
    formal = None
    train = None


def make_trainer(*, updates: int = 0, train_length: int = 2):
    trainer = train.BaseTrainer.__new__(train.BaseTrainer)
    trainer.args = SimpleNamespace(
        train_only=True,
        formal_stage="base",
        batch_size=64,
    )
    trainer._formal_optimizer_updates = updates
    trainer.model = torch.nn.Linear(3, 2)
    trainer.opt = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    trainer.opt_s = torch.optim.lr_scheduler.StepLR(
        trainer.opt,
        step_size=1,
    )
    trainer.train_length = train_length
    trainer.train_data = list(range(3))
    return trainer


def dataset_receipt():
    return {
        "summary_sha256": "a" * 64,
        "data_mdb_sha256": "b" * 64,
        "smplx_asset": None,
    }


def source_receipt():
    return {
        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "commit": "c" * 40,
        "tree": "d" * 40,
        "entrypoint": "/formal/show_base_train.py",
        "entrypoint_sha256": "e" * 64,
    }


@unittest.skipIf(torch is None, "torch/formal runtime is unavailable")
class FormalOptimizerUpdateTests(unittest.TestCase):
    def test_counts_only_after_successful_optimizer_step(self):
        trainer = make_trainer()
        result = trainer._formal_optimizer_step()
        self.assertIsNone(result)
        self.assertEqual(trainer.formal_optimizer_updates, 1)

        trainer.opt.step = mock.Mock(side_effect=RuntimeError("failed step"))
        with self.assertRaisesRegex(RuntimeError, "failed step"):
            trainer._formal_optimizer_step()
        self.assertEqual(trainer.formal_optimizer_updates, 1)

    def test_nonformal_call_preserves_step_without_counting(self):
        trainer = make_trainer()
        trainer.args.train_only = False
        trainer._formal_optimizer_step()
        self.assertEqual(trainer.formal_optimizer_updates, 0)

    def test_restore_rejects_negative_count(self):
        trainer = make_trainer()
        with self.assertRaises(ValueError):
            trainer._restore_formal_optimizer_updates(-1)


@unittest.skipIf(torch is None, "torch/formal runtime is unavailable")
class FormalSmplxReceiptTests(unittest.TestCase):
    def args(self, root: Path, *, expected: str | None = None):
        return SimpleNamespace(
            formal_stage="face",
            data_path_1=str(root),
            expected_smplx_asset_sha256=(
                formal.FORMAL_SMPLX_SHA256 if expected is None else expected
            ),
        )

    def make_asset(self, root: Path) -> Path:
        path = (
            root
            / "smplx_models"
            / "smplx"
            / formal.FORMAL_SMPLX_FILENAME
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(b"frozen-smplx-test-fixture")
        return path

    def test_accepts_exact_regular_asset_and_records_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = self.make_asset(root)
            with mock.patch.object(
                formal,
                "_sha256",
                return_value=formal.FORMAL_SMPLX_SHA256,
            ):
                receipt = formal._formal_smplx_asset_receipt(self.args(root))
        assert receipt is not None
        self.assertEqual(receipt["path"], str(asset.resolve()))
        self.assertEqual(receipt["sha256"], formal.FORMAL_SMPLX_SHA256)
        self.assertTrue(receipt["regular_file"])
        self.assertFalse(receipt["symlink"])

    def test_rejects_missing_required_sha_and_observed_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_asset(root)
            with self.assertRaisesRegex(RuntimeError, "requires the frozen"):
                formal._formal_smplx_asset_receipt(
                    self.args(root, expected="")
                )
            with mock.patch.object(
                formal,
                "_sha256",
                return_value="f" * 64,
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                    formal._formal_smplx_asset_receipt(self.args(root))

    def test_rejects_symlink_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.npz"
            real.write_bytes(b"fixture")
            asset = (
                root
                / "smplx_models"
                / "smplx"
                / formal.FORMAL_SMPLX_FILENAME
            )
            asset.parent.mkdir(parents=True)
            asset.symlink_to(real)
            with self.assertRaisesRegex(RuntimeError, "non-symlink"):
                formal._formal_smplx_asset_receipt(self.args(root))

    def test_forbids_smplx_argument_outside_four_rvq_stages(self):
        args = SimpleNamespace(
            formal_stage="base",
            data_path_1="/unused",
            expected_smplx_asset_sha256=formal.FORMAL_SMPLX_SHA256,
        )
        with self.assertRaisesRegex(RuntimeError, "restricted"):
            formal._formal_smplx_asset_receipt(args)


@unittest.skipIf(torch is None, "torch/formal runtime is unavailable")
class BaseCandidateAuditTests(unittest.TestCase):
    def bindings(self):
        return {
            "updates_per_epoch": 2,
            "config_sha256": "1" * 64,
            "lineage_sha256": "2" * 64,
            "dataset_receipt": dataset_receipt(),
            "source_receipt": source_receipt(),
        }

    def save_epoch_ten(self, root: Path):
        trainer = make_trainer(updates=20)
        manifest = root / "base_candidate_manifest.json"
        receipt = formal._save_base_candidate(
            trainer,
            manifest,
            completed_epochs=10,
            **self.bindings(),
        )
        return trainer, manifest, receipt

    def candidate_path(self, manifest: Path, index: int = -1) -> Path:
        record = formal._base_candidate_entries(manifest)[index]
        return manifest.parent / record["checkpoint"]

    def save_resume(
        self,
        trainer,
        resume: Path,
        *,
        completed_epochs: int,
        candidate_manifest_sha256: str | None,
        candidate_entries: list[dict],
    ):
        bindings = self.bindings()
        formal._save_resume(
            trainer,
            resume,
            completed_epochs=completed_epochs,
            rng_states=[{"fixture": True}],
            world_size=1,
            train_samples=len(trainer.train_data),
            updates_per_epoch=trainer.train_length,
            config_sha256=bindings["config_sha256"],
            lineage_sha256=bindings["lineage_sha256"],
            dataset_summary_sha256=bindings[
                "dataset_receipt"
            ]["summary_sha256"],
            data_mdb_sha256=bindings[
                "dataset_receipt"
            ]["data_mdb_sha256"],
            dataset_receipt=bindings["dataset_receipt"],
            source_receipt_sha256=formal._payload_sha256(
                bindings["source_receipt"]
            ),
            optimizer_updates=trainer.formal_optimizer_updates,
            candidate_manifest_sha256=candidate_manifest_sha256,
            candidate_manifest_entry_count=len(candidate_entries),
            candidate_manifest_entries_sha256=(
                formal._base_candidate_entries_sha256(candidate_entries)
            ),
            last_metrics={},
            started_unix=1.0,
        )

    def load_resume(self, trainer, resume: Path, manifest: Path):
        bindings = self.bindings()
        with mock.patch.object(formal, "_restore_rng_state"):
            return formal._load_resume(
                trainer,
                resume,
                rank=0,
                world_size=1,
                config_sha256=bindings["config_sha256"],
                lineage_sha256=bindings["lineage_sha256"],
                dataset_summary_sha256=bindings[
                    "dataset_receipt"
                ]["summary_sha256"],
                data_mdb_sha256=bindings[
                    "dataset_receipt"
                ]["data_mdb_sha256"],
                dataset_receipt=bindings["dataset_receipt"],
                source_receipt=bindings["source_receipt"],
                source_receipt_sha256=formal._payload_sha256(
                    bindings["source_receipt"]
                ),
                candidate_manifest_path=manifest,
            )

    def save_epoch_five_resume(self, root: Path) -> Path:
        resume = root / "latest_resume.pt"
        trainer = make_trainer(updates=10)
        self.save_resume(
            trainer,
            resume,
            completed_epochs=5,
            candidate_manifest_sha256=None,
            candidate_entries=[],
        )
        return resume

    def replay_epoch_ten(
        self,
        restored,
        target_trainer,
        manifest: Path,
        expected_manifest_sha256: str | None,
    ):
        restored.model.load_state_dict(target_trainer.model.state_dict())
        restored._restore_formal_optimizer_updates(20)
        return formal._save_base_candidate(
            restored,
            manifest,
            completed_epochs=10,
            expected_manifest_sha256=expected_manifest_sha256,
            **self.bindings(),
        )

    def validate_epoch_ten(self, manifest: Path):
        bindings = self.bindings()
        return formal._validate_base_candidate_manifest(
            manifest,
            formal_stage="base",
            completed_epochs=10,
            updates_per_epoch=bindings["updates_per_epoch"],
            config_sha256=bindings["config_sha256"],
            lineage_sha256=bindings["lineage_sha256"],
            dataset_receipt=bindings["dataset_receipt"],
            source_receipt_sha256=formal._payload_sha256(
                bindings["source_receipt"]
            ),
        )

    def test_candidate_is_immutable_and_manifest_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, manifest, receipt = self.save_epoch_ten(root)
            record = formal._base_candidate_entries(manifest)[0]
            candidate = self.candidate_path(manifest)
            original_sha = formal._sha256(candidate)
            self.assertFalse(Path(record["checkpoint"]).is_absolute())
            self.assertEqual(receipt["entries"], 1)
            self.assertEqual(receipt["path"], manifest.name)
            self.assertEqual(
                self.validate_epoch_ten(manifest),
                receipt["sha256"],
            )
            replay_receipt = formal._save_base_candidate(
                trainer,
                manifest,
                completed_epochs=10,
                expected_manifest_sha256=receipt["sha256"],
                **self.bindings(),
            )
            self.assertEqual(replay_receipt["sha256"], receipt["sha256"])
            self.assertEqual(formal._sha256(candidate), original_sha)

            mismatched = make_trainer(updates=20)
            with self.assertRaisesRegex(RuntimeError, "differs from immutable"):
                formal._save_base_candidate(
                    mismatched,
                    manifest,
                    completed_epochs=10,
                    expected_manifest_sha256=receipt["sha256"],
                    **self.bindings(),
                )

    def test_normal_candidate_append_does_not_reload_full_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, first_receipt = self.save_epoch_ten(root)
            formal._clear_committed_base_candidate_transaction(
                manifest,
                completed_epochs=10,
                manifest_sha256=first_receipt["sha256"],
            )
            trainer = make_trainer(updates=40)
            with mock.patch.object(
                formal,
                "_validate_base_candidate_manifest",
                side_effect=AssertionError(
                    "normal append must not full-load historical checkpoints"
                ),
            ):
                second_receipt = formal._save_base_candidate(
                    trainer,
                    manifest,
                    completed_epochs=20,
                    expected_manifest_sha256=first_receipt["sha256"],
                    **self.bindings(),
                )
            self.assertEqual(second_receipt["entries"], 2)
            bindings = self.bindings()
            self.assertEqual(
                formal._validate_base_candidate_manifest(
                    manifest,
                    formal_stage="base",
                    completed_epochs=20,
                    updates_per_epoch=bindings["updates_per_epoch"],
                    config_sha256=bindings["config_sha256"],
                    lineage_sha256=bindings["lineage_sha256"],
                    dataset_receipt=bindings["dataset_receipt"],
                    source_receipt_sha256=formal._payload_sha256(
                        bindings["source_receipt"]
                    ),
                ),
                second_receipt["sha256"],
            )

    def test_checkpoint_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, _ = self.save_epoch_ten(root)
            payload = formal._load_regular_json(
                manifest,
                "candidate manifest",
            )
            checkpoint = manifest.parent / payload["entries"][0]["checkpoint"]
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                self.validate_epoch_ten(manifest)

    def test_candidate_hash_and_load_share_one_byte_snapshot(self):
        for consumer in ("manifest", "pending"):
            with self.subTest(consumer=consumer), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                _, manifest, _ = self.save_epoch_ten(root)
                candidate = self.candidate_path(manifest)
                snapshot = candidate.read_bytes()
                expected_sha = hashlib.sha256(snapshot).hexdigest()
                expected_payload = torch.load(
                    io.BytesIO(snapshot),
                    map_location="cpu",
                    weights_only=True,
                )
                real_loader = formal._torch_load_candidate_snapshot

                def mutate_path_then_load(
                    received_snapshot,
                    *,
                    checkpoint_path,
                ):
                    self.assertEqual(received_snapshot, snapshot)
                    self.assertEqual(checkpoint_path, candidate)
                    candidate.write_bytes(b"changed-after-candidate-snapshot")
                    return real_loader(
                        received_snapshot,
                        checkpoint_path=checkpoint_path,
                    )

                with mock.patch.object(
                    formal,
                    "_torch_load_candidate_snapshot",
                    side_effect=mutate_path_then_load,
                ):
                    if consumer == "manifest":
                        self.validate_epoch_ten(manifest)
                    else:
                        payload, observed_sha = (
                            formal._load_and_validate_base_candidate_checkpoint(
                                candidate,
                                expected_audit=expected_payload["audit"],
                            )
                        )
                        self.assertEqual(observed_sha, expected_sha)
                        self.assertEqual(
                            payload["audit"],
                            expected_payload["audit"],
                        )

    def test_manifest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, _ = self.save_epoch_ten(root)
            payload = formal._load_regular_json(
                manifest,
                "candidate manifest",
            )
            payload["config_sha256"] = "9" * 64
            formal._atomic_json(manifest, payload)
            with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                self.validate_epoch_ten(manifest)

    def test_lightweight_snapshot_rejects_manifest_sha_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, receipt = self.save_epoch_ten(root)
            payload = formal._load_regular_json(manifest, "manifest")
            payload["interval_epochs"] = 20
            formal._atomic_json(manifest, payload)
            with self.assertRaisesRegex(RuntimeError, "snapshot changed"):
                formal._validate_base_candidate_snapshot(
                    manifest,
                    expected_sha256=receipt["sha256"],
                    expected_entry_count=1,
                )

    def test_lightweight_snapshot_rejects_missing_extra_and_symlink(self):
        for failure in ("missing", "extra", "symlink"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                _, manifest, receipt = self.save_epoch_ten(root)
                checkpoint = self.candidate_path(manifest)
                if failure == "missing":
                    checkpoint.unlink()
                elif failure == "extra":
                    (checkpoint.parent / "extra.bin").write_bytes(b"extra")
                else:
                    real = root / "replacement.bin"
                    real.write_bytes(checkpoint.read_bytes())
                    checkpoint.unlink()
                    checkpoint.symlink_to(real)
                with self.assertRaises(RuntimeError):
                    formal._validate_base_candidate_snapshot(
                        manifest,
                        expected_sha256=receipt["sha256"],
                        expected_entry_count=1,
                    )

    def test_resume_restores_actual_count_and_revalidates_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, manifest, candidate_receipt = self.save_epoch_ten(root)
            resume = root / "latest_resume.pt"
            self.save_resume(
                trainer,
                resume,
                completed_epochs=10,
                candidate_manifest_sha256=candidate_receipt["sha256"],
                candidate_entries=formal._base_candidate_entries(manifest),
            )

            restored = make_trainer()
            completed, metrics, started, sha, count, transaction = (
                self.load_resume(restored, resume, manifest)
            )
            self.assertEqual((completed, metrics, started), (10, {}, 1.0))
            self.assertEqual((sha, count), (candidate_receipt["sha256"], 1))
            self.assertIsNotNone(transaction)
            self.assertEqual(restored.formal_optimizer_updates, 20)

            tampered = torch.load(
                resume,
                map_location="cpu",
                weights_only=False,
            )
            tampered["optimizer_updates"] = 19
            formal._atomic_torch_save(resume, tampered)
            with self.assertRaisesRegex(
                RuntimeError,
                "actual optimizer update count",
            ):
                self.load_resume(make_trainer(), resume, manifest)

    def test_resume_audit_integer_rejects_bool_string_and_float(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, manifest, candidate_receipt = self.save_epoch_ten(root)
            resume = root / "latest_resume.pt"
            entries = formal._base_candidate_entries(manifest)
            self.save_resume(
                trainer,
                resume,
                completed_epochs=10,
                candidate_manifest_sha256=candidate_receipt["sha256"],
                candidate_entries=entries,
            )
            original = torch.load(
                resume,
                map_location="cpu",
                weights_only=False,
            )
            for invalid in (True, "10", 10.0):
                with self.subTest(invalid=invalid):
                    tampered = dict(original)
                    tampered["completed_epochs"] = invalid
                    formal._atomic_torch_save(resume, tampered)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "exact integer",
                    ):
                        self.load_resume(make_trainer(), resume, manifest)

    def test_resume_metric_average_must_be_real_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, manifest, candidate_receipt = self.save_epoch_ten(root)
            resume = root / "latest_resume.pt"
            entries = formal._base_candidate_entries(manifest)
            self.save_resume(
                trainer,
                resume,
                completed_epochs=10,
                candidate_manifest_sha256=candidate_receipt["sha256"],
                candidate_entries=entries,
            )
            original = torch.load(
                resume,
                map_location="cpu",
                weights_only=False,
            )
            for invalid in (
                True,
                "1.0",
                complex(1.0, 0.0),
                float("nan"),
                float("inf"),
                float("-inf"),
            ):
                with self.subTest(invalid=invalid):
                    tampered = dict(original)
                    tampered["last_metrics"] = {
                        "loss": {"avg": invalid, "count": 1}
                    }
                    formal._atomic_torch_save(resume, tampered)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "invalid metric receipt",
                    ):
                        self.load_resume(make_trainer(), resume, manifest)

    def test_rvq_resume_init_must_be_an_exact_boolean(self):
        class QuantizeEMAReset(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.codebook = torch.nn.Parameter(torch.zeros(2, 3))
                self.init = False
                self.code_sum = None
                self.code_count = None

        model = torch.nn.Sequential(QuantizeEMAReset())
        for invalid in (0, 1, "false", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "exact boolean",
                ):
                    formal._restore_rvq_ema_state(
                        model,
                        {"0": {"init": invalid}},
                    )

    def test_candidate_manifest_audit_integer_is_exact(self):
        for invalid in (True, "10", 10.0):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                _, manifest, _ = self.save_epoch_ten(root)
                payload = formal._load_regular_json(manifest, "manifest")
                payload["interval_epochs"] = invalid
                formal._atomic_json(manifest, payload)
                with self.assertRaisesRegex(RuntimeError, "exact integer"):
                    self.validate_epoch_ten(manifest)

    def test_candidate_transaction_audit_integer_is_exact(self):
        for invalid in (True, "0", 0.0):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                _, manifest, _ = self.save_epoch_ten(root)
                transaction_path = formal._candidate_transaction_path(
                    manifest
                )
                payload = formal._load_regular_json(
                    transaction_path,
                    "transaction",
                )
                payload["core"]["previous_entry_count"] = invalid
                payload["core_sha256"] = formal._payload_sha256(
                    payload["core"]
                )
                formal._atomic_json(transaction_path, payload)
                with self.assertRaisesRegex(RuntimeError, "exact integer"):
                    formal._inspect_base_candidate_transaction(
                        manifest,
                        formal_stage="base",
                        updates_per_epoch=2,
                        config_sha256="1" * 64,
                        lineage_sha256="2" * 64,
                        dataset_receipt=dataset_receipt(),
                        source_receipt=source_receipt(),
                    )

    def test_crash_before_intent_has_no_candidate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "base_candidate_manifest.json"
            resume = self.save_epoch_five_resume(root)
            loaded = self.load_resume(make_trainer(), resume, manifest)
            self.assertEqual(loaded[0], 5)
            self.assertIsNone(loaded[3])
            self.assertEqual(loaded[4], 0)
            self.assertIsNone(loaded[5])

    def test_crash_after_intent_before_checkpoint_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "base_candidate_manifest.json"
            target = make_trainer(updates=20)
            with mock.patch.object(
                formal,
                "_atomic_torch_save",
                side_effect=RuntimeError("injected checkpoint crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    formal._save_base_candidate(
                        target,
                        manifest,
                        completed_epochs=10,
                        **self.bindings(),
                    )
            transaction = formal._load_regular_json(
                formal._candidate_transaction_path(manifest),
                "transaction",
            )
            self.assertEqual(transaction["state"], "prepared")
            self.assertFalse(self.candidate_path_from_target(manifest).exists())

            resume = self.save_epoch_five_resume(root)
            restored = make_trainer()
            loaded = self.load_resume(restored, resume, manifest)
            self.assertEqual(loaded[0], 5)
            self.assertEqual(loaded[5]["target_epoch"], 10)
            replay = self.replay_epoch_ten(
                restored,
                target,
                manifest,
                loaded[3],
            )
            self.assertEqual(replay["entries"], 1)

    def candidate_path_from_target(self, manifest: Path) -> Path:
        return (
            manifest.parent
            / formal._candidate_relative_path(10, 20)
        )

    def test_crash_after_checkpoint_before_manifest_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "base_candidate_manifest.json"
            target = make_trainer(updates=20)
            original_atomic_json = formal._atomic_json

            def fail_manifest(path, payload):
                if path == manifest:
                    raise RuntimeError("injected manifest crash")
                return original_atomic_json(path, payload)

            with mock.patch.object(
                formal,
                "_atomic_json",
                side_effect=fail_manifest,
            ):
                with self.assertRaisesRegex(RuntimeError, "manifest crash"):
                    formal._save_base_candidate(
                        target,
                        manifest,
                        completed_epochs=10,
                        **self.bindings(),
                    )
            checkpoint = self.candidate_path_from_target(manifest)
            checkpoint_sha = formal._sha256(checkpoint)
            self.assertFalse(manifest.exists())
            transaction = formal._load_regular_json(
                formal._candidate_transaction_path(manifest),
                "transaction",
            )
            self.assertEqual(transaction["state"], "checkpoint_committed")

            resume = self.save_epoch_five_resume(root)
            restored = make_trainer()
            loaded = self.load_resume(restored, resume, manifest)
            self.assertEqual((loaded[0], loaded[4]), (5, 0))
            replay = self.replay_epoch_ten(
                restored,
                target,
                manifest,
                loaded[3],
            )
            self.assertEqual(formal._sha256(checkpoint), checkpoint_sha)
            self.assertEqual(replay["entries"], 1)

    def test_crash_after_manifest_before_resume_replays_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_trainer, manifest, receipt = self.save_epoch_ten(root)
            checkpoint = self.candidate_path(manifest)
            checkpoint_sha = formal._sha256(checkpoint)

            resume = self.save_epoch_five_resume(root)
            restored = make_trainer()
            loaded = self.load_resume(restored, resume, manifest)
            self.assertEqual((loaded[0], loaded[4]), (5, 1))
            replay = self.replay_epoch_ten(
                restored,
                candidate_trainer,
                manifest,
                loaded[3],
            )
            self.assertEqual(replay["sha256"], receipt["sha256"])
            self.assertEqual(formal._sha256(checkpoint), checkpoint_sha)

    def test_crash_after_resume_before_intent_clear_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, manifest, receipt = self.save_epoch_ten(root)
            resume = root / "latest_resume.pt"
            entries = formal._base_candidate_entries(manifest)
            self.save_resume(
                trainer,
                resume,
                completed_epochs=10,
                candidate_manifest_sha256=receipt["sha256"],
                candidate_entries=entries,
            )
            loaded = self.load_resume(make_trainer(), resume, manifest)
            transaction = loaded[5]
            self.assertIsNotNone(transaction)
            self.assertEqual(transaction["target_epoch"], 10)
            formal._clear_committed_base_candidate_transaction(
                manifest,
                completed_epochs=10,
                manifest_sha256=receipt["sha256"],
            )
            self.assertFalse(
                formal._candidate_transaction_path(manifest).exists()
            )

    def test_transaction_core_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, _ = self.save_epoch_ten(root)
            transaction_path = formal._candidate_transaction_path(manifest)
            payload = formal._load_regular_json(transaction_path, "transaction")
            payload["core"]["config_sha256"] = "9" * 64
            payload["core_sha256"] = formal._payload_sha256(payload["core"])
            formal._atomic_json(transaction_path, payload)
            with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                formal._inspect_base_candidate_transaction(
                    manifest,
                    formal_stage="base",
                    updates_per_epoch=2,
                    config_sha256="1" * 64,
                    lineage_sha256="2" * 64,
                    dataset_receipt=dataset_receipt(),
                    source_receipt=source_receipt(),
                )

    def test_transaction_checkpoint_sha_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, _ = self.save_epoch_ten(root)
            transaction_path = formal._candidate_transaction_path(manifest)
            payload = formal._load_regular_json(transaction_path, "transaction")
            payload["checkpoint_sha256"] = "9" * 64
            formal._atomic_json(transaction_path, payload)
            with self.assertRaisesRegex(
                RuntimeError,
                "manifest_committed candidate transaction mismatch",
            ):
                formal._inspect_base_candidate_transaction(
                    manifest,
                    formal_stage="base",
                    updates_per_epoch=2,
                    config_sha256="1" * 64,
                    lineage_sha256="2" * 64,
                    dataset_receipt=dataset_receipt(),
                    source_receipt=source_receipt(),
                )

    def test_staging_symlink_without_transaction_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "base_candidate_manifest.json"
            target = root / "target"
            target.write_bytes(b"target")
            formal._candidate_staging_path(manifest).symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "without a transaction"):
                formal._inspect_base_candidate_transaction(
                    manifest,
                    formal_stage="base",
                    updates_per_epoch=2,
                    config_sha256="1" * 64,
                    lineage_sha256="2" * 64,
                    dataset_receipt=dataset_receipt(),
                    source_receipt=source_receipt(),
                )

    def test_resume_rejects_more_than_one_future_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, first_receipt = self.save_epoch_ten(root)
            formal._clear_committed_base_candidate_transaction(
                manifest,
                completed_epochs=10,
                manifest_sha256=first_receipt["sha256"],
            )
            second = make_trainer(updates=40)
            formal._save_base_candidate(
                second,
                manifest,
                completed_epochs=20,
                expected_manifest_sha256=first_receipt["sha256"],
                **self.bindings(),
            )
            resume = self.save_epoch_five_resume(root)
            with self.assertRaisesRegex(
                RuntimeError,
                "more than one uncommitted",
            ):
                self.load_resume(make_trainer(), resume, manifest)

    def test_final_observed_manifest_sha_must_match_trusted_snapshot(self):
        trusted = "1" * 64
        self.assertEqual(
            formal._require_trusted_candidate_manifest_sha(trusted, trusted),
            trusted,
        )
        with self.assertRaisesRegex(RuntimeError, "trusted training snapshot"):
            formal._require_trusted_candidate_manifest_sha(
                "2" * 64,
                trusted,
            )

    def test_final_model_audit_binds_actual_updates_and_dataset(self):
        trainer = make_trainer(updates=20)
        candidate_receipt = {
            "path": "base_candidate_manifest.json",
            "sha256": "3" * 64,
            "entries": 1,
            "last_epoch": 10,
            "last_optimizer_updates": 20,
        }
        payload = formal._model_payload(
            trainer,
            formal_stage="base",
            config_sha256="1" * 64,
            lineage_sha256="2" * 64,
            dataset_receipt=dataset_receipt(),
            source_receipt=source_receipt(),
            optimizer_updates=trainer.formal_optimizer_updates,
            candidate_manifest_receipt=candidate_receipt,
        )
        self.assertEqual(payload["audit"]["optimizer_updates"], 20)
        self.assertEqual(
            payload["audit"]["dataset_receipt_sha256"],
            formal._payload_sha256(dataset_receipt()),
        )
        self.assertEqual(
            payload["audit"]["base_candidate_manifest"],
            candidate_receipt,
        )


@unittest.skipIf(torch is None, "torch/formal runtime is unavailable")
class LowerTargetCacheFormalBindingTests(unittest.TestCase):
    def cache_receipt(self):
        return {
            "format": "semtalk_show_lower_target_joints_raw_lmdb_v1",
            "receipt_sha256": "7" * 64,
            "speaker_scope": "All",
        }

    def lower_dataset_receipt(self):
        receipt = dataset_receipt()
        receipt["lower_target_joints_cache"] = self.cache_receipt()
        return receipt

    def save_lower_resume(self, root: Path) -> tuple[Path, dict]:
        trainer = make_trainer(updates=10)
        trainer.args.formal_stage = "lower"
        resume = root / "latest_resume.pt"
        receipt = self.lower_dataset_receipt()
        formal._save_resume(
            trainer,
            resume,
            completed_epochs=5,
            rng_states=[{"fixture": True}],
            world_size=1,
            train_samples=len(trainer.train_data),
            updates_per_epoch=trainer.train_length,
            config_sha256="1" * 64,
            lineage_sha256="2" * 64,
            dataset_summary_sha256=receipt["summary_sha256"],
            data_mdb_sha256=receipt["data_mdb_sha256"],
            dataset_receipt=receipt,
            source_receipt_sha256=formal._payload_sha256(source_receipt()),
            optimizer_updates=trainer.formal_optimizer_updates,
            candidate_manifest_sha256=None,
            candidate_manifest_entry_count=0,
            candidate_manifest_entries_sha256=(
                formal._base_candidate_entries_sha256([])
            ),
            last_metrics={},
            started_unix=1.0,
        )
        return resume, receipt

    def load_lower_resume(
        self,
        resume: Path,
        receipt: dict,
        candidate_manifest: Path,
    ):
        restored = make_trainer()
        restored.args.formal_stage = "lower"
        with mock.patch.object(formal, "_restore_rng_state"):
            return formal._load_resume(
                restored,
                resume,
                rank=0,
                world_size=1,
                config_sha256="1" * 64,
                lineage_sha256="2" * 64,
                dataset_summary_sha256=receipt["summary_sha256"],
                data_mdb_sha256=receipt["data_mdb_sha256"],
                dataset_receipt=receipt,
                source_receipt=source_receipt(),
                source_receipt_sha256=formal._payload_sha256(
                    source_receipt()
                ),
                candidate_manifest_path=candidate_manifest,
            )

    def test_resume_roundtrip_requires_exact_cache_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resume, receipt = self.save_lower_resume(root)
            original = torch.load(
                resume,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                original["lower_target_joints_cache"],
                self.cache_receipt(),
            )
            loaded = self.load_lower_resume(
                resume,
                receipt,
                root / "base_candidate_manifest.json",
            )
            self.assertEqual(loaded[0], 5)

            for mutation in ("missing", "mismatch", "null"):
                with self.subTest(mutation=mutation):
                    tampered = dict(original)
                    if mutation == "missing":
                        tampered.pop("lower_target_joints_cache")
                    elif mutation == "mismatch":
                        tampered["lower_target_joints_cache"] = {
                            **self.cache_receipt(),
                            "receipt_sha256": "8" * 64,
                        }
                    else:
                        tampered["lower_target_joints_cache"] = None
                    formal._atomic_torch_save(resume, tampered)
                    with self.assertRaises(RuntimeError):
                        self.load_lower_resume(
                            resume,
                            receipt,
                            root / "base_candidate_manifest.json",
                        )

    def test_final_model_audit_binds_cache_receipt_exactly(self):
        trainer = make_trainer(updates=20)
        receipt = self.lower_dataset_receipt()
        payload = formal._model_payload(
            trainer,
            formal_stage="lower",
            config_sha256="1" * 64,
            lineage_sha256="2" * 64,
            dataset_receipt=receipt,
            source_receipt=source_receipt(),
            optimizer_updates=trainer.formal_optimizer_updates,
            candidate_manifest_receipt=None,
        )
        self.assertEqual(
            payload["audit"]["lower_target_joints_cache"],
            self.cache_receipt(),
        )
        baseline = formal._model_payload(
            trainer,
            formal_stage="base",
            config_sha256="1" * 64,
            lineage_sha256="2" * 64,
            dataset_receipt=dataset_receipt(),
            source_receipt=source_receipt(),
            optimizer_updates=trainer.formal_optimizer_updates,
            candidate_manifest_receipt=None,
        )
        self.assertNotIn(
            "lower_target_joints_cache",
            baseline["audit"],
        )


if __name__ == "__main__":
    unittest.main()
