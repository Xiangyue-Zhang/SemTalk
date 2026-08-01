from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
TRAINER_PATH = (
    REPOSITORY / "scripts" / "show_base" / "train_base_official_adapt_long.py"
)
BASELINE_COMMIT = "40424f4766eeb771edae75fdb0a5aaddf03719ea"
def _load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


TRAINER = _load("deferred_e1_trainer", TRAINER_PATH)


class _FakeBoolean:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self):
        return self

    def item(self) -> bool:
        return self.value


class _FakeTensor:
    def __init__(self, values=(1.0, 2.0)) -> None:
        self.array = np.asarray(values, dtype=np.float32)
        self.shape = self.array.shape
        self.dtype = self.array.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, **_kwargs):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.array

    def isfinite(self):
        return _FakeBoolean(bool(np.isfinite(self.array).all()))

    def is_floating_point(self) -> bool:
        return True

    def is_complex(self) -> bool:
        return False


class _FakeModel:
    def __init__(self, value: float = 1.0) -> None:
        self._state = {"weight": _FakeTensor((value, value + 1.0))}

    def state_dict(self):
        return self._state

    def parameters(self):
        return []


class _FakeOptimizer:
    def __init__(self) -> None:
        self.state = {"parameter": {"exp_avg": _FakeTensor((3.0, 4.0))}}


def _fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")

    def save(value, target) -> None:
        if hasattr(target, "write"):
            pickle.dump(value, target, protocol=4)
            return
        with Path(target).open("wb") as handle:
            pickle.dump(value, handle, protocol=4)

    def load(source, *, map_location=None, weights_only=None):
        del map_location, weights_only
        if hasattr(source, "read"):
            return pickle.load(source)
        with Path(source).open("rb") as handle:
            return pickle.load(handle)

    module.save = save
    module.load = load
    module.isfinite = lambda value: value.isfinite()
    module.is_tensor = lambda value: isinstance(value, _FakeTensor)
    module._cpu_rng_state = b"cpu-rng-state"
    module._cuda_rng_states = [b"cuda-0", b"cuda-1"]
    module.get_rng_state = lambda: bytes(module._cpu_rng_state)
    module.set_rng_state = lambda value: setattr(
        module, "_cpu_rng_state", bytes(value)
    )
    module.cuda = types.SimpleNamespace(
        get_rng_state_all=lambda: list(module._cuda_rng_states),
        set_rng_state_all=lambda values: setattr(
            module, "_cuda_rng_states", [bytes(value) for value in values]
        ),
    )
    return module


def _context() -> tuple[dict[str, object], dict[str, object]]:
    frozen = {
        "protocol": {"format": TRAINER.PROTOCOL_FORMAT},
    }
    frozen["receipt_sha256"] = TRAINER.canonical_json_sha256(frozen)
    contracts = {
        "schedule": {"sha256": "b" * 64},
        "trajectory_anchor": {
            "mode": TRAINER.FRESH_TRAJECTORY_MODE,
            "sha256": "c" * 64,
            "entries": {},
        },
    }
    return frozen, contracts


def _manifest(module, frozen, contracts, *, root: Path, provisional=False, verified=True):
    expected_probe = {"optimizer_updates": 70}
    candidates = (
        list(module.short_quality_epochs(module.W8_GLOBAL2048_MODE))
        if provisional
        else list(module.CANDIDATE_EPOCHS)
    )
    manifest = {
        "format": (
            module.SHORT_QUALITY_MANIFEST_FORMAT
            if provisional
            else module.MANIFEST_FORMAT
        ),
        "status": "running",
        "candidate_epochs": candidates,
        "frozen_receipt_sha256": frozen["receipt_sha256"],
        "schedule_sha256": contracts["schedule"]["sha256"],
        "trajectory_anchor_sha256": contracts["trajectory_anchor"]["sha256"],
        "throughput_gate": {
            "sha256": "d" * 64,
            "trajectory_probe": dict(expected_probe),
        },
        "trajectory_mode": module.FRESH_TRAJECTORY_MODE,
        "trajectory_probe_verified": verified,
        "trajectory_probe": dict(expected_probe) if verified else None,
        "entries": [],
        "entries_sha256": module.canonical_json_sha256([]),
    }
    if provisional:
        manifest.update(
            {
                "run_purpose": module.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": candidates,
                "quality_protocol_version": module.SHORT_QUALITY_PROTOCOL_VERSION,
                "artifact_root_namespace": module.SHORT_QUALITY_ARTIFACT_ROOT_NAMESPACE,
                "artifact_root": str(root),
                "quality_role": module.short_quality_role(
                    module.W8_GLOBAL2048_MODE
                ),
                "reference_only": False,
                "late_w1_status": "not_measured",
                "w1_tail_equivalence_claimed": False,
            }
        )
    return manifest


class DeferredE1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="semtalk-deferred-e1-")
        self.root = Path(self.temporary.name).resolve()
        self.frozen, self.contracts = _context()
        (self.root / "frozen_inputs.json").write_text(
            json.dumps(self.frozen, sort_keys=True) + "\n"
        )
        self.model = _FakeModel()
        self.optimizer = _FakeOptimizer()
        self.fake_torch = _fake_torch()
        self.torch_patch = mock.patch.dict(
            sys.modules, {"torch": self.fake_torch}
        )
        self.torch_patch.start()
        self.update_patch = mock.patch.object(
            TRAINER, "EXPECTED_UPDATES_PER_EPOCH", 62
        )
        self.update_patch.start()

    def tearDown(self) -> None:
        self.update_patch.stop()
        self.torch_patch.stop()
        self.temporary.cleanup()

    def _stage(self, *, provisional: bool = False) -> Path:
        self.manifest = _manifest(
            TRAINER,
            self.frozen,
            self.contracts,
            root=self.root,
            provisional=provisional,
            verified=False,
        )
        return TRAINER._stage_deferred_candidate(
            model=self.model,
            optimizer=self.optimizer,
            run_dir=self.root,
            epoch=1,
            optimizer_updates=62,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=self.manifest,
            provisional=provisional,
        )

    def test_w8g2048_stages_private_then_publishes_exact_e1(self) -> None:
        staged = self._stage()
        sidecar = TRAINER._deferred_candidate_receipt_path(self.root)
        self.assertEqual(oct(os.stat(staged).st_mode & 0o777), "0o400")
        self.assertEqual(oct(os.stat(sidecar).st_mode & 0o777), "0o400")
        self.assertEqual(oct(os.stat(staged.parent).st_mode & 0o777), "0o700")
        self.assertFalse((self.root / "candidate_receipts/epoch-0001.json").exists())
        expected_gate_sha = TRAINER.canonical_json_sha256(
            self.manifest["throughput_gate"]
        )
        expected_probe = self.manifest["throughput_gate"]["trajectory_probe"]
        frozen_path = self.root / "frozen_inputs.json"
        private_receipt = json.loads(sidecar.read_text())
        self.assertEqual(
            private_receipt["throughput_gate_canonical_sha256"],
            expected_gate_sha,
        )
        self.assertEqual(
            private_receipt["expected_trajectory_probe"], expected_probe
        )
        frozen_seal = private_receipt["frozen_inputs_seal"]
        self.assertEqual(
            set(frozen_seal),
            {
                "path",
                "sha256",
                "bytes",
                "identity",
                "receipt_payload_sha256",
            },
        )
        self.assertEqual(frozen_seal["path"], str(frozen_path))
        self.assertEqual(
            frozen_seal["sha256"], TRAINER.sha256_file(frozen_path)
        )
        self.assertEqual(frozen_seal["bytes"], frozen_path.stat().st_size)
        self.assertEqual(
            set(frozen_seal["identity"]),
            {"device", "inode", "size", "mtime_ns", "ctime_ns"},
        )
        self.assertEqual(
            frozen_seal["receipt_payload_sha256"],
            self.frozen["receipt_sha256"],
        )
        _path, private_payload = TRAINER._load_deferred_candidate(
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=self.manifest,
            provisional=False,
        )
        self.assertEqual(
            private_payload["throughput_gate_canonical_sha256"],
            expected_gate_sha,
        )
        self.assertEqual(
            private_payload["expected_trajectory_probe"], expected_probe
        )
        self.assertEqual(private_payload["frozen_inputs_seal"], frozen_seal)

        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        TRAINER._publish_deferred_candidate(
            model=_FakeModel(99.0),
            optimizer=self.optimizer,
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=manifest,
            provisional=False,
        )
        self.assertFalse(staged.parent.exists())
        self.assertEqual([row["epoch"] for row in manifest["entries"]], [1])
        self.assertEqual(
            manifest["entries"][0]["model_state_semantic_sha256"],
            TRAINER._model_state_semantic_sha256(self.model.state_dict()),
        )
        ready = json.loads(
            (self.root / "candidate_receipts/epoch-0001.json").read_text()
        )
        self.assertFalse(ready["selection_eligible"])
        self.assertTrue(ready["candidate_checkpoint"]["path"].endswith("epoch_01.bin"))

    def test_w8g2048_short_quality_uses_its_own_exact_public_schema(self) -> None:
        staged = self._stage(provisional=True)
        self.assertFalse(
            (
                self.root
                / "short_quality_candidate_receipts/epoch-0001.json"
            ).exists()
        )
        manifest = _manifest(
            TRAINER,
            self.frozen,
            self.contracts,
            root=self.root,
            provisional=True,
        )
        TRAINER._publish_deferred_candidate(
            model=_FakeModel(99.0),
            optimizer=self.optimizer,
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=manifest,
            provisional=True,
        )
        self.assertFalse(staged.parent.exists())
        self.assertFalse((self.root / "candidate_receipts").exists())
        ready_path = (
            self.root
            / "short_quality_candidate_receipts/epoch-0001.json"
        )
        ready = json.loads(ready_path.read_text())
        self.assertEqual(ready["format"], TRAINER.SHORT_QUALITY_READY_RECEIPT_FORMAT)
        self.assertEqual(ready["run_purpose"], TRAINER.RUN_PURPOSE_SHORT_QUALITY)
        self.assertEqual(
            ready["quality_protocol_version"],
            TRAINER.SHORT_QUALITY_PROTOCOL_VERSION,
        )
        self.assertEqual(ready["artifact_root"], str(self.root))
        self.assertEqual(
            manifest["entries"][0]["checkpoint"],
            "provisional_candidates/"
            "base_official_adapt_short_quality_epoch_01.bin",
        )
        checkpoint = self.fake_torch.load(
            self.root
            / "provisional_candidates/"
            "base_official_adapt_short_quality_epoch_01.bin",
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(
            checkpoint["audit"]["format"],
            TRAINER.SHORT_QUALITY_CHECKPOINT_FORMAT,
        )
        self.assertEqual(
            checkpoint["audit"]["run_purpose"],
            TRAINER.RUN_PURPOSE_SHORT_QUALITY,
        )

    def test_staged_byte_tamper_is_rejected_by_external_sidecar(self) -> None:
        staged = self._stage()
        os.chmod(staged, 0o600)
        staged.write_bytes(staged.read_bytes() + b"tamper")
        os.chmod(staged, 0o400)
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "sidecar receipt or staged bytes"
        ):
            TRAINER._load_deferred_candidate(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=self.manifest,
                provisional=False,
            )

    def test_stage_rejects_frozen_file_not_equal_to_self_hashed_preflight(
        self,
    ) -> None:
        frozen_path = self.root / "frozen_inputs.json"
        forged = dict(self.frozen)
        forged["forged_after_preflight"] = {"same_type": True}
        frozen_path.write_text(json.dumps(forged, sort_keys=True) + "\n")
        with self.assertRaises(TRAINER.AdaptationContractError):
            self._stage()
        self.assertFalse(
            TRAINER._deferred_candidate_path(self.root).parent.exists()
        )

    def test_same_bytes_frozen_inode_replacement_blocks_publication(self) -> None:
        self._stage()
        frozen_path = self.root / "frozen_inputs.json"
        original_bytes = frozen_path.read_bytes()
        replacement = self.root / "replacement_frozen_inputs.json"
        replacement.write_bytes(original_bytes)
        os.replace(replacement, frozen_path)
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError,
            "sidecar receipt or staged bytes",
        ):
            TRAINER._load_deferred_candidate(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=self.manifest,
                provisional=False,
            )
        self.assertTrue(TRAINER._deferred_candidate_path(self.root).is_file())

    def test_frozen_symlink_replacement_blocks_publication(self) -> None:
        self._stage()
        frozen_path = self.root / "frozen_inputs.json"
        target = self.root / "frozen_inputs_target.json"
        target.write_bytes(frozen_path.read_bytes())
        frozen_path.unlink()
        frozen_path.symlink_to(target)
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError,
            "could not safely open deferred e1 frozen inputs",
        ):
            TRAINER._load_deferred_candidate(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=self.manifest,
                provisional=False,
            )
        self.assertTrue(TRAINER._deferred_candidate_path(self.root).is_file())

    def test_staged_e1_is_still_strictly_compared_to_trajectory_anchor(self) -> None:
        semantic = TRAINER._model_state_semantic_sha256(
            self.model.state_dict()
        )
        self.contracts["trajectory_anchor"]["entries"]["1"] = {
            "tensor_count": len(self.model.state_dict()),
            "model_state_semantic_sha256": semantic,
        }
        self._stage()
        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        TRAINER._publish_deferred_candidate(
            model=_FakeModel(99.0),
            optimizer=self.optimizer,
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=manifest,
            provisional=False,
        )
        self.assertIs(manifest["entries"][0]["trajectory_anchor_match"], True)

    def test_staged_e1_anchor_mismatch_fails_before_public_checkpoint(self) -> None:
        self.contracts["trajectory_anchor"]["entries"]["1"] = {
            "tensor_count": len(self.model.state_dict()),
            "model_state_semantic_sha256": "f" * 64,
        }
        self._stage()
        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "trajectory anchor failed"
        ):
            TRAINER._publish_deferred_candidate(
                model=_FakeModel(99.0),
                optimizer=self.optimizer,
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=manifest,
                provisional=False,
            )
        self.assertFalse(
            (
                self.root
                / "candidates/base_official_adapt_epoch_01.bin"
            ).exists()
        )
        self.assertTrue(TRAINER._deferred_candidate_path(self.root).is_file())

    def test_stale_extra_file_and_restart_are_fail_closed(self) -> None:
        staged = self._stage()
        extra = staged.parent / "unexpected"
        extra.write_bytes(b"stale")
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "absent, stale, or unsafe"
        ):
            TRAINER._load_deferred_candidate(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=self.manifest,
                provisional=False,
            )
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "stale, restarted"
        ):
            self._stage()

    def test_duplicate_publication_and_restaging_are_rejected(self) -> None:
        self._stage()
        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        TRAINER._publish_deferred_candidate(
            model=self.model,
            optimizer=self.optimizer,
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=manifest,
            provisional=False,
        )
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "verified empty-prefix"
        ):
            TRAINER._publish_deferred_candidate(
                model=self.model,
                optimizer=self.optimizer,
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=manifest,
                provisional=False,
            )
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "already-public"
        ):
            self._stage()

    def test_stage_and_publish_do_not_change_model_optimizer_or_any_rng(self) -> None:
        random.seed(9173)
        np.random.seed(9173)

        def snapshot(model):
            return {
                "model": TRAINER._model_state_semantic_sha256(
                    model.state_dict()
                ),
                "optimizer": hashlib.sha256(
                    pickle.dumps(self.optimizer.state, protocol=4)
                ).hexdigest(),
                "python": random.getstate(),
                "numpy": hashlib.sha256(
                    pickle.dumps(np.random.get_state(), protocol=4)
                ).hexdigest(),
                "torch_cpu": self.fake_torch.get_rng_state(),
                "torch_cuda": self.fake_torch.cuda.get_rng_state_all(),
            }

        before_stage = snapshot(self.model)
        self._stage()
        self.assertEqual(snapshot(self.model), before_stage)

        current_model = _FakeModel(99.0)
        before_publish = snapshot(current_model)
        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        TRAINER._publish_deferred_candidate(
            model=current_model,
            optimizer=self.optimizer,
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=manifest,
            provisional=False,
        )
        self.assertEqual(snapshot(current_model), before_publish)
        self.assertEqual(
            manifest["entries"][0]["model_state_semantic_sha256"],
            before_stage["model"],
        )
        self.assertNotEqual(
            manifest["entries"][0]["model_state_semantic_sha256"],
            before_publish["model"],
        )

    def test_crash_after_public_ready_can_only_cleanup_exact_closure(self) -> None:
        self._stage()
        TRAINER._load_deferred_candidate(
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            manifest=self.manifest,
            provisional=False,
        )
        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        with (
            mock.patch.object(
                TRAINER,
                "_cleanup_deferred_candidate_after_publication",
                side_effect=RuntimeError("simulated crash"),
            ),
            self.assertRaisesRegex(RuntimeError, "simulated crash"),
        ):
            TRAINER._publish_deferred_candidate(
                model=self.model,
                optimizer=self.optimizer,
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=manifest,
                provisional=False,
            )
        self.assertTrue(
            (self.root / "candidate_receipts/epoch-0001.json").is_file()
        )
        self.assertTrue(TRAINER._deferred_candidate_path(self.root).is_file())

        ready_path = self.root / "candidate_receipts/epoch-0001.json"
        snapshot_path = (
            self.root / "candidate_manifest_snapshots/epoch-0001.json"
        )
        checkpoint_path = (
            self.root / "candidates/base_official_adapt_epoch_01.bin"
        )
        original_ready = ready_path.read_bytes()
        original_snapshot = snapshot_path.read_bytes()
        original_checkpoint = checkpoint_path.read_bytes()

        ready = json.loads(original_ready)
        ready["unexpected"] = "schema extension"
        unsigned = dict(ready)
        unsigned.pop("receipt_payload_sha256")
        ready["receipt_payload_sha256"] = TRAINER.canonical_json_sha256(
            unsigned
        )
        ready_path.write_text(json.dumps(ready, sort_keys=True) + "\n")
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "public deferred e1 closure"
        ):
            TRAINER._cleanup_deferred_candidate_after_publication(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                provisional=False,
            )
        self.assertTrue(TRAINER._deferred_candidate_path(self.root).is_file())
        ready_path.write_bytes(original_ready)

        snapshot = json.loads(original_snapshot)
        snapshot["unexpected"] = "schema extension"
        snapshot_path.write_text(json.dumps(snapshot, sort_keys=True) + "\n")
        ready = json.loads(original_ready)
        ready["candidate_manifest"]["sha256_at_ready"] = TRAINER.sha256_file(
            snapshot_path
        )
        unsigned = dict(ready)
        unsigned.pop("receipt_payload_sha256")
        ready["receipt_payload_sha256"] = TRAINER.canonical_json_sha256(
            unsigned
        )
        ready_path.write_text(json.dumps(ready, sort_keys=True) + "\n")
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "public deferred e1 closure"
        ):
            TRAINER._cleanup_deferred_candidate_after_publication(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                provisional=False,
            )
        snapshot_path.write_bytes(original_snapshot)
        ready_path.write_bytes(original_ready)

        checkpoint_path.write_bytes(original_checkpoint + b"tamper")
        with self.assertRaisesRegex(
            TRAINER.AdaptationContractError, "public deferred e1 closure"
        ):
            TRAINER._cleanup_deferred_candidate_after_publication(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                provisional=False,
            )
        checkpoint_path.write_bytes(original_checkpoint)

        TRAINER._cleanup_deferred_candidate_after_publication(
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            provisional=False,
        )
        self.assertFalse(
            TRAINER._deferred_candidate_path(self.root).parent.exists()
        )

    def test_forged_same_type_gate_and_probe_cannot_cleanup_private_e1(self) -> None:
        self._stage()
        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        with (
            mock.patch.object(
                TRAINER,
                "_cleanup_deferred_candidate_after_publication",
                side_effect=RuntimeError("simulated crash"),
            ),
            self.assertRaisesRegex(RuntimeError, "simulated crash"),
        ):
            TRAINER._publish_deferred_candidate(
                model=self.model,
                optimizer=self.optimizer,
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=manifest,
                provisional=False,
            )

        snapshot_path = (
            self.root / "candidate_manifest_snapshots/epoch-0001.json"
        )
        ready_path = self.root / "candidate_receipts/epoch-0001.json"
        original_snapshot = snapshot_path.read_bytes()
        original_ready = ready_path.read_bytes()

        snapshot = json.loads(original_snapshot)
        forged_probe = {"optimizer_updates": 70, "forged_count": 999}
        forged_gate = dict(snapshot["throughput_gate"])
        forged_gate["trajectory_probe"] = dict(forged_probe)
        forged_gate["forged_same_type"] = {"accepted": True}
        snapshot["throughput_gate"] = forged_gate
        snapshot["trajectory_probe"] = dict(forged_probe)
        snapshot_path.write_text(json.dumps(snapshot, sort_keys=True) + "\n")

        ready = json.loads(original_ready)
        ready["candidate_manifest"]["sha256_at_ready"] = TRAINER.sha256_file(
            snapshot_path
        )
        unsigned = dict(ready)
        unsigned.pop("receipt_payload_sha256")
        ready["receipt_payload_sha256"] = TRAINER.canonical_json_sha256(
            unsigned
        )
        ready_path.write_text(json.dumps(ready, sort_keys=True) + "\n")

        with self.assertRaises(TRAINER.AdaptationContractError):
            TRAINER._cleanup_deferred_candidate_after_publication(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                provisional=False,
            )
        self.assertTrue(TRAINER._deferred_candidate_path(self.root).is_file())

        snapshot_path.write_bytes(original_snapshot)
        ready_path.write_bytes(original_ready)
        TRAINER._cleanup_deferred_candidate_after_publication(
            run_dir=self.root,
            frozen_receipt=self.frozen,
            contract_receipts=self.contracts,
            provisional=False,
        )
        self.assertFalse(
            TRAINER._deferred_candidate_path(self.root).parent.exists()
        )

    def test_forged_frozen_file_and_ready_sha_cannot_cleanup_private_e1(
        self,
    ) -> None:
        self._stage()
        manifest = _manifest(
            TRAINER, self.frozen, self.contracts, root=self.root
        )
        with (
            mock.patch.object(
                TRAINER,
                "_cleanup_deferred_candidate_after_publication",
                side_effect=RuntimeError("simulated crash"),
            ),
            self.assertRaisesRegex(RuntimeError, "simulated crash"),
        ):
            TRAINER._publish_deferred_candidate(
                model=self.model,
                optimizer=self.optimizer,
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                manifest=manifest,
                provisional=False,
            )

        frozen_path = self.root / "frozen_inputs.json"
        forged_frozen = json.loads(frozen_path.read_text())
        forged_frozen["forged_after_ready"] = {"same_type": True}
        frozen_path.write_text(
            json.dumps(forged_frozen, sort_keys=True) + "\n"
        )

        ready_path = self.root / "candidate_receipts/epoch-0001.json"
        ready = json.loads(ready_path.read_text())
        ready["frozen_inputs"]["sha256"] = TRAINER.sha256_file(
            frozen_path
        )
        unsigned = dict(ready)
        unsigned.pop("receipt_payload_sha256")
        ready["receipt_payload_sha256"] = TRAINER.canonical_json_sha256(
            unsigned
        )
        ready_path.write_text(json.dumps(ready, sort_keys=True) + "\n")

        with self.assertRaises(TRAINER.AdaptationContractError):
            TRAINER._cleanup_deferred_candidate_after_publication(
                run_dir=self.root,
                frozen_receipt=self.frozen,
                contract_receipts=self.contracts,
                provisional=False,
            )
        self.assertTrue(TRAINER._deferred_candidate_path(self.root).is_file())

    def test_only_w8g2048_fresh_paths_are_deferred(self) -> None:
        for provisional in (False, True):
            self.assertTrue(
                TRAINER._deferred_candidate_required(
                    provisional=provisional,
                    trajectory_mode=TRAINER.FRESH_TRAJECTORY_MODE,
                    updates_per_epoch=62,
                )
            )
            for updates in (124, 248, 1988):
                self.assertFalse(
                    TRAINER._deferred_candidate_required(
                        provisional=provisional,
                        trajectory_mode=TRAINER.FRESH_TRAJECTORY_MODE,
                        updates_per_epoch=updates,
                    )
                )

    def test_training_loop_stages_at_e1_and_publishes_once_after_probe(self) -> None:
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(TRAINER._run_training))
        )
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        def named_calls(name):
            return [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
            ]

        stage_calls = named_calls("_stage_deferred_candidate")
        publish_calls = named_calls("_publish_deferred_candidate")
        self.assertEqual(len(stage_calls), 1)
        self.assertEqual(len(publish_calls), 1)

        stage_expression = parents[stage_calls[0]]
        stage_if = parents[stage_expression]
        self.assertIsInstance(stage_if, ast.If)
        self.assertEqual(
            ast.unparse(stage_if.test),
            "deferred_e1 and completed_epoch == 1",
        )
        self.assertEqual(
            [
                node.func.id
                for node in ast.walk(ast.Module(body=stage_if.orelse))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_save_candidate"
            ],
            ["_save_candidate"],
        )

        publish_expression = parents[publish_calls[0]]
        publish_if = parents[publish_expression]
        self.assertIsInstance(publish_if, ast.If)
        self.assertEqual(ast.unparse(publish_if.test), "deferred_e1")
        rank_if = parents[publish_if]
        self.assertIsInstance(rank_if, ast.If)
        self.assertEqual(ast.unparse(rank_if.test), "rank == 0")
        update_if = parents[rank_if]
        self.assertIsInstance(update_if, ast.If)
        self.assertEqual(
            ast.unparse(update_if.test),
            "fresh_trajectory and optimizer_updates == TRAJECTORY_PROBE_UPDATES",
        )
        self.assertEqual(
            [ast.unparse(statement) for statement in rank_if.body[:3]],
            [
                "manifest['trajectory_probe_verified'] = True",
                "manifest['trajectory_probe'] = matched_probe",
                "_atomic_json(run_dir / manifest_name, manifest)",
            ],
        )
        self.assertIs(rank_if.body[3], publish_if)

    def test_private_writers_use_exclusive_nofollow_inode_chain(self) -> None:
        for writer in (
            TRAINER._write_new_torch_save,
            TRAINER._write_new_private_json,
        ):
            source = inspect.getsource(writer)
            for token in (
                "os.O_CREAT",
                "os.O_EXCL",
                'getattr(os, "O_NOFOLLOW", 0)',
                "os.fstat",
                "follow_symlinks=False",
                "os.link",
            ):
                self.assertIn(token, source)

    def test_direct_candidate_path_is_baseline_ast_and_byte_stable(self) -> None:
        # Normalize only the new optional source-selection seam.  The result
        # must equal the frozen pre-change `_save_candidate` AST digest, so the
        # default branch cannot drift without updating an explicit golden.
        class _NormalizeOverride(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                pairs = [
                    (argument, default)
                    for argument, default in zip(
                        node.args.kwonlyargs,
                        node.args.kw_defaults,
                    )
                    if argument.arg != "model_state_override"
                ]
                node.args.kwonlyargs = [argument for argument, _ in pairs]
                node.args.kw_defaults = [default for _, default in pairs]
                node.body = [
                    statement
                    for statement in node.body
                    if not (
                        isinstance(statement, ast.Assign)
                        and any(
                            isinstance(target, ast.Name)
                            and target.id == "state_source"
                            for target in statement.targets
                        )
                    )
                ]
                return self.generic_visit(node)

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "items"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "state_source"
                ):
                    return ast.parse(
                        "_unwrap_model(model).state_dict().items()",
                        mode="eval",
                    ).body
                return node

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(TRAINER._save_candidate))
        )
        normalized = _NormalizeOverride().visit(tree)
        ast.fix_missing_locations(normalized)
        baseline_source = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY),
                "show",
                (
                    f"{BASELINE_COMMIT}:scripts/show_base/"
                    "train_base_official_adapt_long.py"
                ),
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        baseline_tree = ast.parse(baseline_source)
        baseline_function = next(
            node
            for node in baseline_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_save_candidate"
        )
        baseline = ast.Module(body=[baseline_function], type_ignores=[])
        ast.fix_missing_locations(baseline)
        self.assertEqual(
            ast.dump(normalized, include_attributes=False),
            ast.dump(baseline, include_attributes=False),
        )

        # The default and explicit-same-state paths must also publish exactly
        # identical bytes when wall time and absolute artifact root are fixed.
        with tempfile.TemporaryDirectory(prefix="semtalk-direct-") as directory:
            root = Path(directory).resolve()
            (root / "frozen_inputs.json").write_text(
                json.dumps(self.frozen, sort_keys=True) + "\n"
            )
            outputs: list[dict[Path, bytes]] = []
            for explicit_override in (False, True):
                manifest = _manifest(
                    TRAINER,
                    self.frozen,
                    self.contracts,
                    root=root,
                )
                with mock.patch.object(
                    TRAINER.time, "time", return_value=1234.5
                ):
                    keyword = (
                        {"model_state_override": self.model.state_dict()}
                        if explicit_override
                        else {}
                    )
                    TRAINER._save_candidate(
                        model=self.model,
                        optimizer=self.optimizer,
                        run_dir=root,
                        epoch=1,
                        optimizer_updates=124,
                        frozen_receipt=self.frozen,
                        contract_receipts=self.contracts,
                        manifest=manifest,
                        provisional=False,
                        **keyword,
                    )
                outputs.append(
                    {
                        path.relative_to(root): path.read_bytes()
                        for path in root.rglob("*")
                        if path.is_file()
                        and path.name != "frozen_inputs.json"
                    }
                )
                for generated in (
                    "candidates",
                    "candidate_manifest_snapshots",
                    "candidate_receipts",
                ):
                    shutil.rmtree(root / generated)
            self.assertEqual(set(outputs[0]), set(outputs[1]))
            for relative in sorted(outputs[0]):
                self.assertEqual(
                    hashlib.sha256(outputs[0][relative]).hexdigest(),
                    hashlib.sha256(outputs[1][relative]).hexdigest(),
                    str(relative),
                )


if __name__ == "__main__":
    unittest.main()
