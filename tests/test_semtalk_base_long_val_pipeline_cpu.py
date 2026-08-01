from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
WATCHER_PATH = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "semtalk_base_long_val_ready_watcher_20260731.py"
)
HARDLINK_PATH = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "semtalk_base_long_val_hardlink_20260731.py"
)
SCHEDULE_PATH = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_long_schedule_20260731.json"
)
ANCHOR_PATH = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_long_trajectory_anchor_20260731.json"
)
TRAINER_PATH = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "train_base_official_adapt_long.py"
)


def _load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


WATCHER = _load("semtalk_base_long_val_ready_watcher", WATCHER_PATH)
HARDLINK = _load("semtalk_base_long_val_hardlink", HARDLINK_PATH)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


class LongValidationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.anchor = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
        (root / "candidates").mkdir(parents=True)
        (root / "candidate_receipts").mkdir()
        (root / "candidate_manifest_snapshots").mkdir()
        self.frozen = self._write_frozen_inputs()
        self.entries: list[dict[str, object]] = []
        self._write_wave_three_candidates()

    def _write_frozen_inputs(self) -> dict[str, object]:
        protocol = {
            "format": WATCHER.PROTOCOL_FORMAT,
            "scope": "SemTalk Base only",
            "initialization": {
                "source": "released_all_speakers_v1",
                "classification": "official_released_all_speakers",
                "filename": "best_semtalk_base.bin",
                "sha256": (
                    "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
                ),
                "speaker_scope": "All-Speakers",
                "forbidden_sources": ["e30", "Speaker2"],
            },
            "target_dataset": "SHOW",
            "target_speaker_scope": "All",
            "world_size": 8,
            "local_batch_size": 64,
            "global_batch_size": 512,
            "expected_updates_per_epoch": 248,
            "epochs": 400,
            "candidate_epochs": list(WATCHER.CANDIDATE_EPOCHS),
            "trajectory_anchor_epochs": sorted(WATCHER.ANCHOR_EPOCHS),
            "precision": "bf16",
            "vq_models_in_training_graph": False,
            "forward_contract": {
                "forwards_per_optimizer_step": 1,
                "audio_conditioned_main_forward": True,
                "masked_self_forward": False,
                "word_auxiliary_forward": False,
            },
            "schedule": {
                "path": "/frozen/schedule.json",
                "sha256": WATCHER.SCHEDULE_SHA256,
            },
            "trajectory_anchor": {
                "path": "/frozen/anchor.json",
                "sha256": WATCHER.TRAJECTORY_ANCHOR_SHA256,
            },
        }
        frozen_body = {
            "format": WATCHER.FROZEN_INPUTS_FORMAT,
            "run_purpose": "formal_training",
            "target_epochs": list(WATCHER.CANDIDATE_EPOCHS),
            "source": {
                "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                "commit": "7d7a8d5b80009ae8e5da56a93e97e9249600a496",
                "tree": "1957c03d1aac7226ad4c697d47ee89e10b23948e",
                "branch": None,
                "clean": True,
                "entrypoint": (
                    "/source/scripts/show_base/"
                    "train_base_official_adapt_long.py"
                ),
                "entrypoint_sha256": WATCHER.TRAINER_ENTRYPOINT_SHA256,
            },
            "official_base": {
                "source": "released_all_speakers_v1",
                "path": "/weights/all_speakers/best_semtalk_base.bin",
                "sha256": (
                    "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
                ),
                "speaker_scope": "All-Speakers",
                "training_dataset": "BEAT2",
                "model_state_tensors": 1790,
                "model_state_schema_sha256": (
                    WATCHER.MODEL_STATE_SCHEMA_SHA256
                ),
                "all_model_state_tensors_finite": True,
                "strict_state_dict_load": True,
            },
            "speaker_initialization": {},
            "dataset": {},
            "protocol": protocol,
            "long_contract": {
                "format": "semtalk_show_base_long_contract_receipts_v1",
                "schedule": {"sha256": WATCHER.SCHEDULE_SHA256},
                "trajectory_anchor": {
                    "sha256": WATCHER.TRAJECTORY_ANCHOR_SHA256
                },
            },
        }
        frozen = {
            **frozen_body,
            "receipt_sha256": WATCHER.canonical_json_sha256(frozen_body),
        }
        _write_json(self.root / "frozen_inputs.json", frozen)
        return frozen

    def _manifest_entry(
        self,
        epoch: int,
        *,
        checkpoint_sha: str | None = None,
        checkpoint_bytes: int = 1,
    ) -> dict[str, object]:
        anchor_entry = self.anchor["entries"].get(str(epoch))
        semantic_sha = (
            anchor_entry["model_state_semantic_sha256"]
            if anchor_entry is not None
            else _sha_bytes(f"semantic-{epoch}".encode())
        )
        return {
            "epoch": epoch,
            "optimizer_updates": epoch * 248,
            "checkpoint": (
                f"candidates/base_official_adapt_epoch_{epoch:02d}.bin"
            ),
            "checkpoint_sha256": (
                checkpoint_sha
                if checkpoint_sha is not None
                else (
                    anchor_entry["checkpoint_sha256"]
                    if anchor_entry is not None
                    else _sha_bytes(f"checkpoint-{epoch}".encode())
                )
            ),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_container_schema": ["audit", "model_state"],
            "model_state_tensors": 1790,
            "model_state_schema_sha256": (
                WATCHER.MODEL_STATE_SCHEMA_SHA256
            ),
            "model_state_semantic_sha256": semantic_sha,
            "all_model_state_tensors_finite": True,
            "frozen_receipt_sha256": self.frozen["receipt_sha256"],
            "trajectory_anchor_match": (
                True if anchor_entry is not None else None
            ),
        }

    def _write_wave_three_candidates(self) -> None:
        for epoch in WATCHER.CANDIDATE_EPOCHS:
            if epoch > 100:
                break
            checkpoint_sha = None
            checkpoint_bytes = 1
            checkpoint_path = (
                self.root
                / "candidates"
                / f"base_official_adapt_epoch_{epoch:02d}.bin"
            )
            if epoch in WATCHER.VALIDATION_WAVES[2]:
                payload = f"candidate-{epoch}".encode()
                checkpoint_path.write_bytes(payload)
                checkpoint_sha = _sha_bytes(payload)
                checkpoint_bytes = len(payload)
            entry = self._manifest_entry(
                epoch,
                checkpoint_sha=checkpoint_sha,
                checkpoint_bytes=checkpoint_bytes,
            )
            self.entries.append(entry)
            if epoch not in WATCHER.VALIDATION_WAVES[2]:
                continue
            manifest = {
                "format": WATCHER.MANIFEST_FORMAT,
                "status": "running",
                "candidate_epochs": list(WATCHER.CANDIDATE_EPOCHS),
                "frozen_receipt_sha256": self.frozen["receipt_sha256"],
                "schedule_sha256": WATCHER.SCHEDULE_SHA256,
                "trajectory_anchor_sha256": (
                    WATCHER.TRAJECTORY_ANCHOR_SHA256
                ),
                "throughput_gate": {
                    "path": "/efs/long-run/throughput_gate.json",
                    "sha256": _sha_bytes(b"gate"),
                    "samples_per_second": 512.0,
                    "seconds_per_update": 1.0,
                },
                "entries": list(self.entries),
                "entries_sha256": WATCHER.canonical_json_sha256(
                    self.entries
                ),
            }
            snapshot = (
                self.root
                / "candidate_manifest_snapshots"
                / f"epoch-{epoch:04d}.json"
            )
            _write_json(snapshot, manifest)
            _write_json(self.root / "candidate_manifest.json", manifest)
            checkpoint = {
                "path": str(checkpoint_path),
                "relative_path": (
                    f"candidates/base_official_adapt_epoch_{epoch:02d}.bin"
                ),
                "sha256": checkpoint_sha,
                "bytes": checkpoint_bytes,
                "model_state_tensors": 1790,
                "model_state_schema_sha256": (
                    WATCHER.MODEL_STATE_SCHEMA_SHA256
                ),
                "model_state_semantic_sha256": (
                    entry["model_state_semantic_sha256"]
                ),
            }
            ready_body = {
                "format": WATCHER.READY_FORMAT,
                "status": "ready",
                "selection_eligible": False,
                "test_visible": False,
                "epoch": epoch,
                "optimizer_updates": epoch * 248,
                "candidate_checkpoint": checkpoint,
                "candidate_manifest": {
                    "path": str(snapshot),
                    "sha256_at_ready": HARDLINK.sha256_file(snapshot),
                    "entries_sha256_at_ready": manifest["entries_sha256"],
                    "immutable_snapshot": True,
                    "live_path": str(self.root / "candidate_manifest.json"),
                },
                "frozen_inputs": {
                    "path": str(self.root / "frozen_inputs.json"),
                    "sha256": HARDLINK.sha256_file(
                        self.root / "frozen_inputs.json"
                    ),
                    "receipt_payload_sha256": self.frozen["receipt_sha256"],
                },
                "protocol": {
                    "format": WATCHER.PROTOCOL_FORMAT,
                    "payload_sha256": WATCHER.canonical_json_sha256(
                        self.frozen["protocol"]
                    ),
                },
                "frozen_receipt_sha256": self.frozen["receipt_sha256"],
                "schedule_sha256": WATCHER.SCHEDULE_SHA256,
                "trajectory_anchor_sha256": (
                    WATCHER.TRAJECTORY_ANCHOR_SHA256
                ),
                "trajectory_anchor_match": None,
                "published_unix": float(epoch),
            }
            ready = {
                **ready_body,
                "receipt_payload_sha256": WATCHER.canonical_json_sha256(
                    ready_body
                ),
            }
            _write_json(
                self.root
                / "candidate_receipts"
                / f"epoch-{epoch:04d}.json",
                ready,
            )


class ReadyWatcherContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="semtalk-fixture-"
        )
        self.root = Path(self.temporary.name)
        self.fixture = LongValidationFixture(self.root)
        self.anchor = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _validate_epoch(self, epoch: int) -> dict[str, object]:
        return WATCHER.validate_ready(
            train_root=self.root,
            epoch=epoch,
            schedule_path=SCHEDULE_PATH,
            anchor_path=ANCHOR_PATH,
            anchor=self.anchor,
        )

    def test_exact_ready_schema_and_min_width_checkpoint_name(self) -> None:
        candidate = self._validate_epoch(60)
        self.assertTrue(
            candidate["checkpoint"]["path"].endswith(
                "base_official_adapt_epoch_60.bin"
            )
        )
        self.assertFalse(candidate["trajectory_anchor_exact"])
        self.assertEqual(
            candidate["source"]["trainer_entrypoint_sha256"],
            WATCHER.TRAINER_ENTRYPOINT_SHA256,
        )
        self.assertEqual(candidate["candidate_manifest"]["entry_count"], 9)

    def test_wait_wave_publishes_selection_ineligible_receipt(self) -> None:
        output = self.root / "wave-0003-ready.json"
        result = WATCHER.wait_wave(
            argparse.Namespace(
                schedule=SCHEDULE_PATH,
                expected_schedule_sha256=WATCHER.SCHEDULE_SHA256,
                trajectory_anchor=ANCHOR_PATH,
                expected_trajectory_anchor_sha256=(
                    WATCHER.TRAJECTORY_ANCHOR_SHA256
                ),
                train_root=self.root,
                wave_index=3,
                output=output,
                poll_seconds=0.01,
                timeout_seconds=1.0,
            )
        )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["epochs"], [60, 70, 80, 100])
        self.assertFalse(receipt["selection_eligible"])
        self.assertFalse(receipt["test_visible"])
        self.assertEqual(
            [candidate["epoch"] for candidate in receipt["candidates"]],
            [60, 70, 80, 100],
        )
        with self.assertRaises(FileExistsError):
            WATCHER.wait_wave(
                argparse.Namespace(
                    schedule=SCHEDULE_PATH,
                    expected_schedule_sha256=WATCHER.SCHEDULE_SHA256,
                    trajectory_anchor=ANCHOR_PATH,
                    expected_trajectory_anchor_sha256=(
                        WATCHER.TRAJECTORY_ANCHOR_SHA256
                    ),
                    train_root=self.root,
                    wave_index=3,
                    output=output,
                    poll_seconds=0.01,
                    timeout_seconds=1.0,
                )
            )

    def test_ready_extra_key_is_rejected_even_with_recomputed_payload(self) -> None:
        path = self.root / "candidate_receipts" / "epoch-0060.json"
        ready = json.loads(path.read_text(encoding="utf-8"))
        ready["unexpected"] = "forbidden"
        ready["receipt_payload_sha256"] = WATCHER.payload_sha256(ready)
        _write_json(path, ready)
        with self.assertRaises(WATCHER.WatchContractError):
            self._validate_epoch(60)

    def test_manifest_snapshot_mutation_is_rejected(self) -> None:
        path = (
            self.root
            / "candidate_manifest_snapshots"
            / "epoch-0060.json"
        )
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(WATCHER.WatchContractError):
            self._validate_epoch(60)

    def test_unpinned_trainer_entrypoint_is_rejected(self) -> None:
        frozen_path = self.root / "frozen_inputs.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen["source"]["entrypoint_sha256"] = "0" * 64
        body = dict(frozen)
        body.pop("receipt_sha256")
        frozen["receipt_sha256"] = WATCHER.canonical_json_sha256(body)
        _write_json(frozen_path, frozen)
        reference = {
            "path": str(frozen_path),
            "sha256": HARDLINK.sha256_file(frozen_path),
            "receipt_payload_sha256": frozen["receipt_sha256"],
        }
        with self.assertRaises(WATCHER.WatchContractError):
            WATCHER.validate_frozen_inputs(
                train_root=self.root,
                value=reference,
                expected_receipt_sha=frozen["receipt_sha256"],
            )


class TrainerWatcherCompatibilityContracts(unittest.TestCase):
    def test_legacy_watcher_pin_stays_frozen_and_rejects_current_trainer(
        self,
    ) -> None:
        self.assertEqual(
            WATCHER.TRAINER_ENTRYPOINT_SHA256,
            "73d2139bf289000e4909becb319e430cebde67b09ff7a32c101389b482f270aa",
        )
        current_trainer_sha256 = _sha_bytes(TRAINER_PATH.read_bytes())
        self.assertNotEqual(
            current_trainer_sha256,
            WATCHER.TRAINER_ENTRYPOINT_SHA256,
        )

        with tempfile.TemporaryDirectory(
            prefix="semtalk-legacy-watcher-current-trainer-"
        ) as temporary:
            root = Path(temporary)
            fixture = LongValidationFixture(root)
            frozen_path = root / "frozen_inputs.json"
            frozen = dict(fixture.frozen)
            frozen["source"] = dict(frozen["source"])
            frozen["source"]["entrypoint_sha256"] = current_trainer_sha256
            body = dict(frozen)
            body.pop("receipt_sha256")
            frozen["receipt_sha256"] = WATCHER.canonical_json_sha256(body)
            _write_json(frozen_path, frozen)
            reference = {
                "path": str(frozen_path),
                "sha256": HARDLINK.sha256_file(frozen_path),
                "receipt_payload_sha256": frozen["receipt_sha256"],
            }
            with self.assertRaises(WATCHER.WatchContractError):
                WATCHER.validate_frozen_inputs(
                    train_root=root,
                    value=reference,
                    expected_receipt_sha=frozen["receipt_sha256"],
                )

    def test_formal_diffsheg_pipeline_excludes_legacy_and_compatibility(
        self,
    ) -> None:
        from scripts.show_base import base_final_authority
        from scripts.show_base import base_long_val_contract as primary
        from scripts.show_base import select_base_official_adapt as diffsheg
        from scripts.show_base import (
            talkshow_base_val_contract as compatibility,
        )

        primary_roots = set(primary.PRIMARY_SELECTION_ENTRYPOINTS)
        primary_roots.update(primary.PRIMARY_AUTHORITY_FINAL_ENTRYPOINTS)
        compatibility_roots = set(primary.COMPATIBILITY_ONLY_ENTRYPOINTS)
        self.assertFalse(primary_roots & compatibility_roots)
        primary_source_roots = set(
            diffsheg.DIFFSHEG_PRIMARY_PIPELINE_SOURCE_FILES
        )
        self.assertIn(
            "scripts/show_base/semtalk_base_inference_core.py",
            primary_source_roots,
        )
        self.assertNotIn(
            "scripts/show_base/run_base_inference.py",
            primary_source_roots,
        )
        self.assertFalse(primary_source_roots & compatibility_roots)
        self.assertEqual(
            primary_source_roots,
            set(diffsheg.FRESH_PIPELINE_SOURCE_FILES),
        )
        self.assertEqual(
            primary.PRIMARY_SELECTION_PROTOCOL,
            "diffsheg_show_validation_fgd_v1",
        )
        self.assertEqual(
            primary.PRIMARY_SELECTION_METRIC_PATH,
            "validation.diffsheg.metrics.fgd",
        )
        self.assertEqual(primary.PRIMARY_SELECTION_REPORT_KEY, "fgd")
        self.assertEqual(
            base_final_authority.BASE_SELECTION_PROTOCOL,
            primary.PRIMARY_SELECTION_PROTOCOL,
        )
        self.assertEqual(
            base_final_authority.BASE_SELECTION_METRIC,
            primary.PRIMARY_SELECTION_METRIC_PATH,
        )

        required_roots = {
            "scripts/show_base/run_base_diffsheg_val_8shard.sh",
            "scripts/show_base/finalize_base_diffsheg_val_partitions.sh",
            "scripts/show_base/select_base_official_adapt_long.py",
            "scripts/show_base/validate_base_long_test_winner.py",
            "scripts/show_base/base_final_authority.py",
            "scripts/show_base/run_base_final_test.sh",
            "scripts/show_base/evaluate_diffsheg_final_test.py",
        }
        self.assertTrue(required_roots.issubset(primary_roots))

        legacy_tokens = {WATCHER_PATH.name, WATCHER_PATH.stem}
        for relative in primary_roots:
            source = (REPOSITORY / relative).read_text(encoding="utf-8")
            for token in legacy_tokens:
                self.assertNotIn(token, source, relative)

        pending = [
            relative
            for relative in primary_roots
            if relative.endswith(".py")
        ]
        python_closure: set[str] = set()
        while pending:
            relative = pending.pop()
            if relative in python_closure:
                continue
            python_closure.add(relative)
            path = REPOSITORY / relative
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            imported_modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imported_modules.add(module)
                    if module == "scripts.show_base":
                        imported_modules.update(
                            f"{module}.{alias.name}" for alias in node.names
                        )
            for module in imported_modules:
                if not module.startswith("scripts.show_base."):
                    continue
                imported = module.replace(".", "/") + ".py"
                if (REPOSITORY / imported).is_file():
                    pending.append(imported)

        forbidden_modules = {
            Path(relative).stem
            for relative in compatibility_roots
            if relative.endswith(".py")
        }
        observed_modules = {Path(relative).stem for relative in python_closure}
        self.assertFalse(forbidden_modules & observed_modules)
        self.assertIn("evaluate_talkshow_show_metrics", observed_modules)
        self.assertNotIn("replay_released2_primary", observed_modules)
        self.assertNotIn("talkshow_base_val_contract", observed_modules)
        self.assertNotIn(WATCHER_PATH.stem, observed_modules)

        expected_pipeline = {"authority": "diffsheg-primary"}
        expected_source = {"commit": "1" * 40}
        validated = ({"sha256": "2" * 64}, expected_pipeline)
        with (
            mock.patch.object(
                diffsheg,
                "validate_fresh_pipeline",
                return_value=validated,
            ) as primary_validator,
            mock.patch.object(
                compatibility,
                "validate_fresh_pipeline",
                side_effect=AssertionError(
                    "primary validation entered compatibility authority"
                ),
            ) as compatibility_validator,
        ):
            result = primary.validate_pipeline(
                Path("/private/tmp/diffsheg-primary-pipeline.json"),
                "3" * 64,
                expected_prerequisite_selection=expected_pipeline,
                expected_source=expected_source,
            )
        self.assertEqual(result, validated)
        primary_validator.assert_called_once_with(
            Path("/private/tmp/diffsheg-primary-pipeline.json"),
            "3" * 64,
            expected_prerequisite_selection=expected_pipeline,
            expected_source=expected_source,
        )
        compatibility_validator.assert_not_called()

        compatibility_relative = (
            "scripts/show_base/replay_released2_primary.py"
        )
        self.assertIn(
            compatibility_relative,
            compatibility.TALKSHOW_COMPATIBILITY_PIPELINE_SOURCE_FILES,
        )
        self.assertNotIn(compatibility_relative, primary_source_roots)
        with tempfile.TemporaryDirectory(
            prefix="semtalk-primary-source-",
            dir="/private/tmp",
        ) as temporary:
            source_root = (Path(temporary) / "source").resolve()
            for relative in sorted(primary_source_roots):
                target = source_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((REPOSITORY / relative).read_bytes())
            compatibility_path = source_root / compatibility_relative
            compatibility_path.parent.mkdir(parents=True, exist_ok=True)
            compatibility_path.write_bytes(
                (REPOSITORY / compatibility_relative).read_bytes()
            )

            def git(*arguments: str) -> str:
                process = subprocess.run(
                    ["git", "-C", str(source_root), *arguments],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return process.stdout.strip()

            git("init", "--quiet")
            git(
                "remote",
                "add",
                "origin",
                "git@github.com:Xiangyue-Zhang/SemTalk.git",
            )

            def commit_fixture(message: str) -> None:
                git(
                    "-c",
                    "user.name=Xiangyue-Zhang",
                    "-c",
                    (
                        "user.email="
                        "85532891+Xiangyue-Zhang@users.noreply.github.com"
                    ),
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "-m",
                    message,
                )

            git(
                "add",
                "--",
                *sorted(primary_source_roots),
                compatibility_relative,
            )
            commit_fixture("primary source receipt fixture")
            branch = git("symbolic-ref", "--short", "HEAD")
            commit = git("rev-parse", "HEAD")
            git("checkout", "--quiet", "--detach", commit)
            git("branch", "-D", branch)

            before = diffsheg.build_fresh_pipeline_source_receipt(source_root)
            compatibility_path.write_bytes(
                compatibility_path.read_bytes()
                + b"\n# compatibility-only revision\n"
            )
            git("add", "--", compatibility_relative)
            commit_fixture("change compatibility-only source")
            after = diffsheg.build_fresh_pipeline_source_receipt(source_root)

        # commit/tree identify the whole repository; files is the formal
        # DiffSHEG-primary source_closure authority.
        self.assertNotEqual(before, after)
        self.assertNotEqual(before["commit"], after["commit"])
        self.assertNotEqual(before["tree"], after["tree"])
        self.assertEqual(before["files"], after["files"])
        self.assertEqual(set(before["files"]), primary_source_roots)
        self.assertFalse(set(before["files"]) & compatibility_roots)

    def test_fresh_diffsheg_pipeline_pins_the_neutral_inference_core(
        self,
    ) -> None:
        from scripts.show_base import select_base_official_adapt as diffsheg

        entrypoint_relative = "scripts/show_base/run_base_val_inference.py"
        helper_relative = (
            "scripts/show_base/semtalk_base_inference_core.py"
        )
        entrypoint = {
            "path": f"/source/{entrypoint_relative}",
            "sha256": "1" * 64,
            "bytes": 1,
            "git_mode": "100644",
            "git_blob_sha1": "1" * 40,
        }
        helper = {
            "path": f"/source/{helper_relative}",
            "sha256": "2" * 64,
            "bytes": 2,
            "git_mode": "100644",
            "git_blob_sha1": "2" * 40,
        }
        source = {
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "source_root": "/source",
            "commit": "3" * 40,
            "tree": "4" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
            "files": {
                entrypoint_relative: entrypoint,
                helper_relative: helper,
            },
        }
        selected = {}
        for stage, character in zip(
            ("face", "hands", "upper", "lower", "global"),
            "56789",
        ):
            selected[stage] = {
                "candidate_checkpoint": {
                    "path": f"/selected/{stage}.bin",
                    "sha256": character * 64,
                    "bytes": 10,
                },
                "epoch": 1,
                "optimizer_updates": 2,
                "updates_per_epoch": 2,
                "candidate_audit_sha256": "a" * 64,
                "selection_metric": {"name": "validation"},
                "measurement_receipt": {"path": f"/{stage}.json"},
            }
        bridge = {
            "selection": {"receipt_payload_sha256": "b" * 64},
            "selected": selected,
        }
        with (
            mock.patch.object(
                diffsheg._selected_prerequisites,
                "load_selected_prerequisites",
                return_value=bridge,
            ),
            mock.patch.object(
                diffsheg,
                "_safe_file_snapshot",
                return_value=(Path("/selection.json"), b"{}"),
            ),
            mock.patch.object(
                diffsheg,
                "build_fresh_pipeline_source_receipt",
                return_value=source,
            ),
        ):
            pipeline = diffsheg.build_fresh_pipeline_payload(
                source_root=Path("/source"),
                prerequisite_selection=Path("/selection.json"),
                expected_prerequisite_selection_sha256="c" * 64,
            )
        self.assertEqual(pipeline["inference_entrypoint"], entrypoint)
        self.assertEqual(pipeline["inference_helper"], helper)
        self.assertNotEqual(pipeline["inference_helper"], entrypoint)

    def test_watcher_ready_schema_matches_trainer_literal(self) -> None:
        tree = ast.parse(
            TRAINER_PATH.read_text(encoding="utf-8"), str(TRAINER_PATH)
        )
        save_candidate = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_save_candidate"
        )
        ready_assignment = next(
            node
            for node in ast.walk(save_candidate)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "ready_body"
                for target in node.targets
            )
        )
        self.assertIsInstance(ready_assignment.value, ast.Dict)
        ready_dict = ready_assignment.value
        ready_fields = {
            key.value: value
            for key, value in zip(ready_dict.keys, ready_dict.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertEqual(
            set(ready_fields) | {"receipt_payload_sha256"},
            set(WATCHER.READY_KEYS),
        )

        def nested_keys(field: str) -> set[str]:
            value = ready_fields[field]
            self.assertIsInstance(value, ast.Dict)
            return {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            }

        self.assertEqual(
            nested_keys("candidate_checkpoint"),
            set(WATCHER.CANDIDATE_CHECKPOINT_KEYS),
        )
        self.assertEqual(
            nested_keys("candidate_manifest"),
            set(WATCHER.CANDIDATE_MANIFEST_KEYS),
        )
        self.assertEqual(
            nested_keys("frozen_inputs"), set(WATCHER.FROZEN_INPUT_KEYS)
        )
        self.assertEqual(nested_keys("protocol"), set(WATCHER.PROTOCOL_KEYS))


class HardlinkHelperContracts(unittest.TestCase):
    def test_link_regular_preserves_exact_inode_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk-link-fixture-"
        ) as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            destination = root / "destination.npz"
            source.write_bytes(b"immutable-npz-fixture")
            HARDLINK.link_regular(source, destination, "fixture")
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(os.stat(source).st_dev, os.stat(destination).st_dev)
            self.assertEqual(os.stat(source).st_ino, os.stat(destination).st_ino)

    def test_helpers_pin_the_same_frozen_schedule(self) -> None:
        self.assertEqual(WATCHER.SCHEDULE_SHA256, HARDLINK.SCHEDULE_SHA256)
        self.assertEqual(WATCHER.CANDIDATE_EPOCHS, HARDLINK.CANDIDATE_EPOCHS)
        self.assertEqual(WATCHER.VALIDATION_WAVES, HARDLINK.VALIDATION_WAVES)


if __name__ == "__main__":
    unittest.main()
