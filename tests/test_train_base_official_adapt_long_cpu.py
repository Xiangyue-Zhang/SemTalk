from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "train_base_official_adapt_long.py"
)
FRESH_SCHEDULE = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_fresh_lineage_schedule_20260731.json"
)
SPEC = importlib.util.spec_from_file_location(
    "train_base_official_adapt_long", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ADAPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OfficialBaseAdaptStaticContracts(unittest.TestCase):
    def test_scratch_entrypoint_is_untouched_by_the_new_entrypoint(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "This is intentionally separate from ``show_base_train.py``",
            source,
        )
        self.assertNotIn("load_pretrained_vq_suite", source)
        self.assertNotIn("from models.rvq", source)
        self.assertNotIn("import models.rvq", source)

    def test_objective_contains_exactly_one_model_forward(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT))
        objective = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "audio_conditioned_objective"
        )
        calls = [
            node
            for node in ast.walk(objective)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "model"
        ]
        self.assertEqual(len(calls), 1)

    def test_fixed_parallelism_candidates_and_gate_lengths(self) -> None:
        self.assertEqual(ADAPT.WORLD_SIZE, 8)
        self.assertEqual(ADAPT.LOCAL_BATCH_SIZE, 64)
        self.assertEqual(ADAPT.GLOBAL_BATCH_SIZE, 512)
        self.assertEqual(
            ADAPT.CANDIDATE_EPOCHS,
            (
                1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80, 100, 120,
                140, 160, 180, 200, 240, 280, 320, 360, 400,
            ),
        )
        self.assertEqual(ADAPT.TOTAL_EPOCHS, 400)
        self.assertEqual(
            ADAPT.RESUME_EPOCHS,
            (40, 80, 120, 160, 200, 240, 280, 320, 360, 400),
        )
        self.assertEqual(ADAPT.THROUGHPUT_WARMUP_UPDATES, 20)
        self.assertEqual(ADAPT.THROUGHPUT_TIMED_UPDATES, 50)
        self.assertEqual(ADAPT.EXPECTED_UPDATES_PER_EPOCH, 248)
        self.assertEqual(
            ADAPT.CHECKPOINT_FORMAT,
            "semtalk_show_base_official_adapt_checkpoint_v1",
        )

    def test_candidate_ready_publication_is_atomic_and_manifest_is_immutable(
        self,
    ) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("os.link(temporary, path)", source)
        self.assertIn("candidate_manifest_snapshots", source)
        self.assertIn('"immutable_snapshot": True', source)
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "candidate_receipts" / "epoch-0001.json"
            ADAPT._write_new_json(receipt, {"epoch": 1, "status": "ready"})
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8")),
                {"epoch": 1, "status": "ready"},
            )
            self.assertEqual(
                [path.name for path in receipt.parent.iterdir()],
                [receipt.name],
            )
            with self.assertRaises(FileExistsError):
                ADAPT._write_new_json(receipt, {"epoch": 2})
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8"))["epoch"],
                1,
            )

    def test_committed_schedules_and_legacy_anchor_are_pinned(self) -> None:
        schedule = (
            REPOSITORY
            / "configs/show_base/semtalk_base_long_schedule_20260731.json"
        )
        anchor = (
            REPOSITORY
            / "configs/show_base/"
            "semtalk_base_long_trajectory_anchor_20260731.json"
        )
        self.assertEqual(
            _sha(schedule),
            "013f8ade256f20579f9d545ac44da681238ed56af1cb687fc6fe9356c0bbefa3",
        )
        self.assertEqual(
            _sha(anchor),
            "e27c27a0da2793f44608b618d08356df0b60f55ae039434a228c50ff73028cf2",
        )
        self.assertEqual(
            _sha(FRESH_SCHEDULE),
            "87ffea4de0b28cbd9b41eb0e935b74d93355e1c1b9a21fb570c98c5ba7cfbbd8",
        )

    def test_protocol_is_main_forward_only_and_vq_free(self) -> None:
        args = argparse.Namespace(
            learning_rate=3e-5,
            precision="bf16",
            schedule_json="/frozen/schedule.json",
            expected_schedule_sha256="a" * 64,
            trajectory_anchor_json="/frozen/anchor.json",
            expected_trajectory_anchor_sha256="b" * 64,
            trajectory_mode=ADAPT.LEGACY_TRAJECTORY_MODE,
            loader_workers=4,
        )
        protocol = ADAPT.protocol_receipt(
            args,
            contract_receipts={
                "trajectory_anchor": {
                    "mode": ADAPT.LEGACY_TRAJECTORY_MODE,
                    "path": "/frozen/anchor.json",
                    "sha256": "b" * 64,
                }
            },
        )
        self.assertEqual(
            protocol["forward_contract"]["forwards_per_optimizer_step"], 1
        )
        self.assertTrue(
            protocol["forward_contract"]["audio_conditioned_main_forward"]
        )
        self.assertFalse(
            protocol["forward_contract"]["masked_self_forward"]
        )
        self.assertFalse(
            protocol["forward_contract"]["word_auxiliary_forward"]
        )
        self.assertFalse(protocol["vq_models_in_training_graph"])
        self.assertEqual(
            protocol["initialization"]["sha256"],
            "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603",
        )

    def test_e30_and_speaker2_are_hard_rejected(self) -> None:
        for label in (
            "/weights/e30/best_semtalk_base.bin",
            "/weights/E_30/best_semtalk_base.bin",
            "/weights/speaker2/best_semtalk_base.bin",
            "SPEAKER-2-adaptation",
        ):
            with self.subTest(label=label):
                with self.assertRaises(ADAPT.AdaptationContractError):
                    ADAPT.reject_forbidden_source_labels(label)
        ADAPT.reject_forbidden_source_labels(
            "/weights/all_speakers/best_semtalk_base.bin"
        )
        ADAPT.reject_forbidden_source_labels(
            "/runs/all_speakers/base_epoch_e300.bin"
        )

    def test_argument_contract_rejects_wrong_batch_or_epochs(self) -> None:
        parser = ADAPT.build_parser()
        base = [
            "--mode",
            "throughput_gate",
            "--official-base-checkpoint",
            "/weights/all/best_semtalk_base.bin",
            "--train-lmdb",
            "/cache/base.lmdb",
            "--dataset-summary",
            "/cache/summary.json",
            "--expected-dataset-summary-sha256",
            "a" * 64,
            "--lineage-manifest",
            "/cache/lineage.json",
            "--expected-lineage-sha256",
            "b" * 64,
            "--schedule-json",
            "/frozen/schedule.json",
            "--expected-schedule-sha256",
            "c" * 64,
            "--trajectory-mode",
            ADAPT.LEGACY_TRAJECTORY_MODE,
            "--trajectory-anchor-json",
            "/frozen/anchor.json",
            "--expected-trajectory-anchor-sha256",
            "d" * 64,
            "--output-root",
            "/runs/base",
            "--run-name",
            "official_all_show",
        ]
        args = parser.parse_args(base)
        ADAPT.validate_args(args)
        args.local_batch_size = 32
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT.validate_args(args)
        args.local_batch_size = 64
        args.epochs = 200
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT.validate_args(args)

    def test_fresh_selected_vq_args_forbid_external_anchor(self) -> None:
        parser = ADAPT.build_parser()
        base = [
            "--mode", "throughput_gate",
            "--official-base-checkpoint", "/weights/all/best_semtalk_base.bin",
            "--train-lmdb", "/cache/current-selected/base.lmdb",
            "--dataset-summary", "/cache/current-selected/summary.json",
            "--expected-dataset-summary-sha256", "a" * 64,
            "--lineage-manifest", "/cache/current-selected/lineage.json",
            "--expected-lineage-sha256", "b" * 64,
            "--prerequisite-selection-json", "/selection/current-five.json",
            "--expected-prerequisite-selection-sha256", "c" * 64,
            "--schedule-json", str(FRESH_SCHEDULE),
            "--expected-schedule-sha256", _sha(FRESH_SCHEDULE),
            "--trajectory-mode", ADAPT.FRESH_TRAJECTORY_MODE,
            "--output-root", "/runs/base",
            "--run-name", "selected_all_show",
        ]
        args = parser.parse_args(base)
        ADAPT.validate_args(args)
        args.trajectory_anchor_json = "/frozen/old-anchor.json"
        args.expected_trajectory_anchor_sha256 = "d" * 64
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "forbids an external old trajectory anchor",
        ):
            ADAPT.validate_args(args)


class OfficialBaseAdaptReceiptContracts(unittest.TestCase):
    class _FakeTensor:
        shape = (2, 3)
        dtype = "torch.float32"

        def is_floating_point(self) -> bool:
            return False

        def is_complex(self) -> bool:
            return False

    class _FakeTorch:
        envelope: object = None

        @classmethod
        def load(cls, *_: object, **__: object) -> object:
            return cls.envelope

        @staticmethod
        def is_tensor(value: object) -> bool:
            return isinstance(
                value, OfficialBaseAdaptReceiptContracts._FakeTensor
            )

    def _fresh_dataset_receipt(self) -> dict[str, object]:
        return {
            "format": (
                "semtalk_show_base_selected_feature_dataset_receipt_v1"
            ),
            "lmdb": "/cache/current-selected/base.lmdb",
            "entries": ADAPT.EXPECTED_TRAIN_SAMPLES,
            "train_clips": ADAPT.EXPECTED_TRAIN_CLIPS,
            "split": "train",
            "test_visible": False,
            "data_mdb_sha256": "1" * 64,
            "lock_mdb_sha256": "2" * 64,
            "summary": "/cache/current-selected/summary.json",
            "summary_sha256": "3" * 64,
            "lineage": "/cache/current-selected/lineage.json",
            "lineage_sha256": "4" * 64,
            "prerequisite_source": ADAPT.SHOW_VAL_SELECTED_SOURCE,
            "formal_checkpoints": {},
            "vq_models_in_training_graph": False,
            "vq_targets": "precomputed_frozen_lmdb_tensors",
            "prerequisite_selection": {
                "path": "/selection/current-five.json",
                "sha256": "5" * 64,
                "receipt_payload_sha256": "6" * 64,
            },
            "selected_prerequisite_sha256": {
                stage: str(index + 10) * 32
                for index, stage in enumerate(ADAPT.selected_contract.STAGES)
            },
            "global_verified_not_consumed": True,
        }

    def _fresh_long_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            schedule_json=str(FRESH_SCHEDULE),
            expected_schedule_sha256=_sha(FRESH_SCHEDULE),
            trajectory_mode=ADAPT.FRESH_TRAJECTORY_MODE,
            trajectory_anchor_json=None,
            expected_trajectory_anchor_sha256=None,
            expected_prerequisite_selection_sha256="5" * 64,
            expected_dataset_summary_sha256="3" * 64,
            expected_lineage_sha256="4" * 64,
            precision="bf16",
            learning_rate=3e-5,
            seed=43,
            loader_workers=4,
        )

    def test_fresh_trajectory_binds_exact_selected_feature_lineage(self) -> None:
        dataset = self._fresh_dataset_receipt()
        receipt = ADAPT.validate_long_contract_receipts(
            self._fresh_long_args(),
            dataset_receipt=dataset,
        )
        trajectory = receipt["trajectory_anchor"]
        self.assertEqual(
            trajectory["mode"],
            ADAPT.FRESH_TRAJECTORY_MODE,
        )
        self.assertIsNone(trajectory["path"])
        self.assertEqual(trajectory["entries"], {})
        self.assertEqual(
            trajectory["feature_lineage_sha256"],
            dataset["lineage_sha256"],
        )
        self.assertEqual(
            trajectory["prerequisite_selection_sha256"],
            dataset["prerequisite_selection"]["sha256"],
        )
        self.assertEqual(
            trajectory["probe_optimizer_updates"],
            ADAPT.TRAJECTORY_PROBE_UPDATES,
        )

    def test_fresh_trajectory_rejects_feature_lineage_mismatch(self) -> None:
        dataset = self._fresh_dataset_receipt()
        dataset["lineage_sha256"] = "7" * 64
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "exact selected SHOW prerequisite/feature lineage",
        ):
            ADAPT.validate_long_contract_receipts(
                self._fresh_long_args(),
                dataset_receipt=dataset,
            )

    def test_fresh_trajectory_rejects_test_visible_features(self) -> None:
        dataset = self._fresh_dataset_receipt()
        dataset["split"] = "test"
        dataset["test_visible"] = True
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "exact selected SHOW prerequisite/feature lineage",
        ):
            ADAPT.validate_long_contract_receipts(
                self._fresh_long_args(),
                dataset_receipt=dataset,
            )

    def test_legacy_anchor_is_rejected_for_selected_show_vqs(self) -> None:
        schedule = (
            REPOSITORY
            / "configs"
            / "show_base"
            / "semtalk_base_long_schedule_20260731.json"
        )
        anchor = (
            REPOSITORY
            / "configs"
            / "show_base"
            / "semtalk_base_long_trajectory_anchor_20260731.json"
        )
        args = self._fresh_long_args()
        args.schedule_json = str(schedule)
        args.expected_schedule_sha256 = _sha(schedule)
        args.trajectory_mode = ADAPT.LEGACY_TRAJECTORY_MODE
        args.trajectory_anchor_json = str(anchor)
        args.expected_trajectory_anchor_sha256 = _sha(anchor)
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "cannot be used with freshly selected SHOW prerequisites",
        ):
            ADAPT.validate_long_contract_receipts(
                args,
                dataset_receipt=self._fresh_dataset_receipt(),
            )

    def test_official_checkpoint_filename_sha_envelope_and_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best_semtalk_base.bin"
            path.write_bytes(b"fixture-official-base")
            specification = {
                **ADAPT.OFFICIAL_BASE_SPEC,
                "sha256": _sha(path),
            }
            self._FakeTorch.envelope = {
                "epoch": ADAPT.OFFICIAL_BASE_EPOCH,
                "lrs": dict(ADAPT.OFFICIAL_BASE_LRS),
                "model_state": {
                    "module.weight": self._FakeTensor(),
                },
                "opt_state": {
                    "state": {},
                    "param_groups": [
                        {
                            **ADAPT.OFFICIAL_BASE_OPTIMIZER_GROUP,
                            "params": [],
                        }
                    ],
                },
            }
            with (
                mock.patch.object(
                    ADAPT, "OFFICIAL_BASE_OPTIMIZER_STATE_ENTRIES", 0
                ),
                mock.patch.object(
                    ADAPT, "OFFICIAL_BASE_OPTIMIZER_PARAMETERS", 0
                ),
            ):
                state, receipt = ADAPT.read_official_base_checkpoint(
                    path,
                    torch_module=self._FakeTorch,
                    specification=specification,
                )
            self.assertEqual(set(state), {"weight"})
            self.assertEqual(
                receipt["checkpoint_container_schema"],
                ["epoch", "lrs", "model_state", "opt_state"],
            )
            self.assertTrue(receipt["strict_state_dict_load"])
            self._FakeTorch.envelope = {
                "epoch": ADAPT.OFFICIAL_BASE_EPOCH,
                "lrs": dict(ADAPT.OFFICIAL_BASE_LRS),
                "model_state": {"weight": self._FakeTensor()},
                "opt_state": {
                    "state": {},
                    "param_groups": [
                        {
                            **ADAPT.OFFICIAL_BASE_OPTIMIZER_GROUP,
                            "params": [],
                        }
                    ],
                },
                "unexpected": True,
            }
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT.read_official_base_checkpoint(
                    path,
                    torch_module=self._FakeTorch,
                    specification=specification,
                )

    def _dataset_fixture(
        self, root: Path
    ) -> tuple[argparse.Namespace, dict[str, object], dict[str, object]]:
        lmdb = root / "base.lmdb"
        lmdb.mkdir()
        (lmdb / "data.mdb").write_bytes(b"official-base-data")
        (lmdb / "lock.mdb").write_bytes(b"official-base-lock")
        lineage_path = root / "lineage.json"
        records = {}
        for stage, specification in ADAPT.OFFICIAL_PREREQUISITE_SPECS.items():
            records[stage] = {
                "path": f"/weights/all/{specification['filename']}",
                "filename": specification["filename"],
                "sha256": specification["sha256"],
                "formal_stage": stage,
                "prerequisite_source": ADAPT.OFFICIAL_BASE_SOURCE,
                "classification": ADAPT.OFFICIAL_BASE_CLASSIFICATION,
                "training_dataset": "BEAT2",
                "speaker_scope": "All-Speakers",
                "show_trained": False,
                "checkpoint_container_schema": ["model_state"],
                "strict_state_dict_load": True,
                "all_model_state_tensors_finite": True,
                "frozen_eval": True,
            }
        lineage: dict[str, object] = {
            "format": "semtalk_show_base_feature_lineage_v1",
            "status": "complete",
            "entries": ADAPT.EXPECTED_TRAIN_SAMPLES,
            "train_clips": ADAPT.EXPECTED_TRAIN_CLIPS,
            "protocol": {
                "scope": "SemTalk Base only",
                "split": "train",
                "speakers": ADAPT.SHOW_SPEAKERS,
                "window_length": 64,
                "stride": 20,
                "in_word": "int64_all_zero_unused_placeholder",
                "forbidden_components": [
                    "ASR",
                    "TextGrid",
                    "vocabulary",
                    "CLIP",
                    "emotion",
                    "semantic",
                    "SemGate",
                    "Sparse",
                ],
                "prerequisite_source": ADAPT.OFFICIAL_BASE_SOURCE,
            },
            "formal_checkpoints": records,
        }
        lineage_path.write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary: dict[str, object] = {
            "format": "semtalk_show_base_lmdb_summary_v1",
            "status": "complete",
            "scope": "SemTalk Base only",
            "entries": ADAPT.EXPECTED_TRAIN_SAMPLES,
            "train_clips": ADAPT.EXPECTED_TRAIN_CLIPS,
            "lmdb": str(lmdb),
            "data_mdb_sha256": _sha(lmdb / "data.mdb"),
            "lock_mdb_sha256": _sha(lmdb / "lock.mdb"),
            "lineage_json": str(lineage_path),
            "lineage_json_sha256": _sha(lineage_path),
        }
        summary_path = root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            dataset_summary=str(summary_path),
            expected_dataset_summary_sha256=_sha(summary_path),
            lineage_manifest=str(lineage_path),
            expected_lineage_sha256=_sha(lineage_path),
            train_lmdb=str(lmdb),
        )
        return args, summary, lineage

    def test_official_all_speakers_dataset_lineage_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = self._dataset_fixture(Path(temporary))
            receipt = ADAPT.validate_dataset_receipts(args)
            self.assertFalse(receipt["vq_models_in_training_graph"])
            self.assertEqual(
                receipt["vq_targets"],
                "precomputed_frozen_lmdb_tensors",
            )
            self.assertEqual(
                set(receipt["formal_checkpoints"]),
                {"face", "hands", "upper", "lower", "global"},
            )

    def test_speaker2_or_nonofficial_vq_lineage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, summary, lineage = self._dataset_fixture(root)
            lineage["formal_checkpoints"]["face"]["path"] = (
                "/weights/speaker2/rvq_face_600.bin"
            )
            lineage_path = Path(args.lineage_manifest)
            lineage_path.write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary["lineage_json_sha256"] = _sha(lineage_path)
            summary_path = Path(args.dataset_summary)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.expected_lineage_sha256 = _sha(lineage_path)
            args.expected_dataset_summary_sha256 = _sha(summary_path)
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT.validate_dataset_receipts(args)

    def test_throughput_gate_must_bind_exact_frozen_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "format": ADAPT.GATE_FORMAT,
                "status": "pass",
                "frozen_receipt_sha256": "a" * 64,
                "world_size": 8,
                "local_batch_size": 64,
                "global_batch_size": 512,
                "warmup_updates": 20,
                "timed_updates": 50,
                "precision": "bf16",
                "learning_rate": 3e-5,
                "all_losses_finite": True,
                "samples_per_second": 512.0,
                "seconds_per_update": 1.0,
                "optimizer_updates": ADAPT.TRAJECTORY_PROBE_UPDATES,
                "trajectory_mode": ADAPT.LEGACY_TRAJECTORY_MODE,
                "trajectory_probe": None,
            }
            path = root / "gate.json"
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                throughput_gate_report=str(path),
                expected_throughput_gate_sha256=_sha(path),
                precision="bf16",
                learning_rate=3e-5,
            )
            receipt = ADAPT.validate_throughput_gate(
                args,
                frozen_receipt={
                    "receipt_sha256": "a" * 64,
                    "long_contract": {
                        "trajectory_anchor": {
                            "mode": ADAPT.LEGACY_TRAJECTORY_MODE,
                        }
                    },
                },
            )
            self.assertEqual(receipt["samples_per_second"], 512.0)
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT.validate_throughput_gate(
                    args,
                    frozen_receipt={
                        "receipt_sha256": "b" * 64,
                        "long_contract": {
                            "trajectory_anchor": {
                                "mode": ADAPT.LEGACY_TRAJECTORY_MODE,
                            }
                        },
                    },
                )

    def test_fresh_trajectory_probe_is_byte_exact(self) -> None:
        expected = {
            "format": ADAPT.TRAJECTORY_PROBE_FORMAT,
            "optimizer_updates": ADAPT.TRAJECTORY_PROBE_UPDATES,
            "model_state_tensors": 1790,
            "model_state_schema_sha256": "a" * 64,
            "model_state_semantic_sha256": "b" * 64,
            "optimizer_state_semantic_sha256": "e" * 64,
        }
        self.assertEqual(
            ADAPT._require_matching_trajectory_probe(expected, dict(expected)),
            expected,
        )
        changed = dict(expected)
        changed["model_state_semantic_sha256"] = "c" * 64
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "does not reproduce",
        ):
            ADAPT._require_matching_trajectory_probe(expected, changed)

    def test_fresh_throughput_gate_carries_lineage_bound_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            probe = {
                "format": ADAPT.TRAJECTORY_PROBE_FORMAT,
                "optimizer_updates": ADAPT.TRAJECTORY_PROBE_UPDATES,
                "model_state_tensors": 1790,
                "model_state_schema_sha256": "a" * 64,
                "model_state_semantic_sha256": "b" * 64,
                "optimizer_state_semantic_sha256": "e" * 64,
            }
            report = {
                "format": ADAPT.GATE_FORMAT,
                "status": "pass",
                "frozen_receipt_sha256": "c" * 64,
                "world_size": ADAPT.WORLD_SIZE,
                "local_batch_size": ADAPT.LOCAL_BATCH_SIZE,
                "global_batch_size": ADAPT.GLOBAL_BATCH_SIZE,
                "warmup_updates": ADAPT.THROUGHPUT_WARMUP_UPDATES,
                "timed_updates": ADAPT.THROUGHPUT_TIMED_UPDATES,
                "optimizer_updates": ADAPT.TRAJECTORY_PROBE_UPDATES,
                "trajectory_mode": ADAPT.FRESH_TRAJECTORY_MODE,
                "trajectory_probe": probe,
                "precision": "bf16",
                "learning_rate": 3e-5,
                "all_losses_finite": True,
                "samples_per_second": 1024.0,
                "seconds_per_update": 0.5,
            }
            path = Path(temporary) / "fresh-gate.json"
            path.write_text(
                json.dumps(report, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                throughput_gate_report=str(path),
                expected_throughput_gate_sha256=_sha(path),
                precision="bf16",
                learning_rate=3e-5,
            )
            receipt = ADAPT.validate_throughput_gate(
                args,
                frozen_receipt={
                    "receipt_sha256": "c" * 64,
                    "long_contract": {
                        "trajectory_anchor": {
                            "mode": ADAPT.FRESH_TRAJECTORY_MODE,
                        }
                    },
                },
            )
            self.assertEqual(receipt["trajectory_probe"], probe)
            report["trajectory_probe"][
                "model_state_semantic_sha256"
            ] = "d" * 64
            changed_path = Path(temporary) / "changed-gate.json"
            changed_path.write_text(
                json.dumps(report, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.throughput_gate_report = str(changed_path)
            args.expected_throughput_gate_sha256 = _sha(changed_path)
            changed = ADAPT.validate_throughput_gate(
                args,
                frozen_receipt={
                    "receipt_sha256": "c" * 64,
                    "long_contract": {
                        "trajectory_anchor": {
                            "mode": ADAPT.FRESH_TRAJECTORY_MODE,
                        }
                    },
                },
            )
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT._require_matching_trajectory_probe(
                    receipt["trajectory_probe"],
                    changed["trajectory_probe"],
                )

    def test_source_receipt_supports_detached_head(self) -> None:
        scripted = [
            subprocess.CompletedProcess([], 0, ADAPT.EXPECTED_ORIGIN + "\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", ""),
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
            subprocess.CompletedProcess([], 0, "b" * 40 + "\n", ""),
        ]
        with mock.patch.object(
            ADAPT.subprocess,
            "run",
            side_effect=scripted,
        ):
            receipt = ADAPT.source_receipt()
        self.assertIsNone(receipt["branch"])
        self.assertTrue(receipt["clean"])


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is optional in the local CPU contract environment",
)
class OfficialBaseAdaptTorchContracts(unittest.TestCase):
    def test_strict_load_and_mean_initializes_only_four_rows(self) -> None:
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spearker_encoder_face = torch.nn.Embedding(25, 768)
                self.spearker_encoder_body = torch.nn.Embedding(25, 768)
                self.other = torch.nn.Linear(3, 2)

        official = Tiny().state_dict()
        official = {
            key: torch.arange(
                value.numel(), dtype=value.dtype
            ).reshape_as(value)
            for key, value in official.items()
        }
        model = Tiny()
        receipt = ADAPT.strict_load_and_initialize_show_speakers(
            model,
            official,
            torch_module=torch,
        )
        state = model.state_dict()
        for key in ADAPT.SPEAKER_EMBEDDING_KEYS:
            mean = official[key].mean(dim=0)
            self.assertTrue(torch.equal(state[key][:4], mean.expand(4, -1)))
            self.assertTrue(torch.equal(state[key][4:], official[key][4:]))
        self.assertTrue(torch.equal(state["other.weight"], official["other.weight"]))
        self.assertTrue(torch.equal(state["other.bias"], official["other.bias"]))
        self.assertTrue(receipt["other_state_unchanged"])

    def test_audio_objective_calls_model_once_and_is_finite(self) -> None:
        import torch

        class Fake:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, *_: object, **__: object) -> dict[str, object]:
                self.calls += 1
                output = {}
                for stage in ("face", "upper", "hands", "lower"):
                    output[f"rec_{stage}"] = torch.zeros(1, 6, 1, 16, 256)
                    output[f"cls_{stage}"] = torch.zeros(1, 16, 256, 6)
                output["hubert_cons_loss"] = torch.tensor(0.25)
                output["beat_cons_loss"] = torch.tensor(0.5)
                return output

        batch = {
            "beat": torch.zeros(1, 64, 3),
            "in_word": torch.zeros(1, 64, dtype=torch.int64),
            "tar_id": torch.zeros(1, 64, 1, dtype=torch.int64),
            "latent_all": torch.zeros(1, 64, 337),
            "hubert": torch.zeros(1, 64, 1024),
        }
        for stage in ("face", "upper", "hands", "lower"):
            batch[f"zq_{stage}"] = torch.zeros(1, 6, 1, 16, 256)
            batch[f"tar_index_value_{stage}_top"] = torch.zeros(
                1, 16, 6, dtype=torch.int64
            )
        model = Fake()
        loss, metrics = ADAPT.audio_conditioned_objective(
            model, batch, torch_module=torch
        )
        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(ADAPT.LOSS_COMPONENTS) - set(metrics), set())


if __name__ == "__main__":
    unittest.main()
