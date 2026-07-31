from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts.show_base import build_base_features as feature_builder
from scripts.show_base import merge_prerequisite_val_shards as merger
from scripts.show_base import prerequisite_val_contract as producer_contract
from scripts.show_base import select_prerequisite_candidates as selector
from scripts.show_base import selected_prerequisites as consumer
from scripts.show_base import train_base_official_adapt_long as base_trainer
from tests import test_prerequisite_val_selection_cpu as selector_fixture


class SelectedPrerequisiteBridgeTests(unittest.TestCase):
    """Exercise the consumer against output made by the real selector."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="semtalk-selected-prereq-"
        )
        cls.root = Path(cls.temporary.name).resolve()
        fixture_builder = selector_fixture.PrerequisiteValidationSelectionTest(
            methodName="test_full_merge_and_exact_selection_bridge"
        )
        cls.fixture = fixture_builder.build_full_fixture(cls.root)

        # The selector's own CPU fixture intentionally does not dereference
        # formal status receipts.  The Base consumer does, so complete those
        # real candidate-index bindings before invoking merge/select.
        candidate_index = cls.fixture["candidate_index"]
        for stage in producer_contract.STAGES:
            status_path = Path(
                candidate_index["formal_training_status"][stage]["path"]
            )
            status_sha = selector_fixture.write_json(
                status_path,
                {
                    "status": "complete",
                    "formal_stage": stage,
                    "completed_epochs": 200,
                    "updates_per_epoch": (
                        producer_contract.EXPECTED_UPDATES_PER_EPOCH
                    ),
                    "optimizer_updates": (
                        200 * producer_contract.EXPECTED_UPDATES_PER_EPOCH
                    ),
                    "all_training_state_finite": True,
                },
            )
            candidate_index["formal_training_status"][stage]["sha256"] = (
                status_sha
            )
        unsigned_index = dict(candidate_index)
        unsigned_index.pop("receipt_payload_sha256")
        candidate_index = producer_contract.receipt_payload(unsigned_index)
        cls.fixture["candidate_index"] = candidate_index
        cls.fixture["candidate_sha"] = selector_fixture.write_json(
            cls.fixture["candidate_path"],
            candidate_index,
        )

        measurement_root = cls.root / "measurements"
        merger.merge(
            candidate_index_path=cls.fixture["candidate_path"],
            candidate_index_sha256=cls.fixture["candidate_sha"],
            canonical_manifest=cls.fixture["manifest"],
            canonical_manifest_sha256=cls.fixture["manifest_sha"],
            canonical_summary=cls.fixture["summary"],
            canonical_summary_sha256=cls.fixture["summary_sha"],
            canonical_lineage=cls.fixture["lineage"],
            canonical_lineage_sha256=cls.fixture["lineage_sha"],
            shard_roots=cls.fixture["shard_roots"],
            output_root=measurement_root,
            merge_source=cls.fixture["merge_source"],
        )
        measurement_index = measurement_root / "measurement_index.json"
        selection_path = cls.root / "selection.json"
        selector.select(
            candidate_index_path=cls.fixture["candidate_path"],
            candidate_index_sha256=cls.fixture["candidate_sha"],
            measurement_index_path=measurement_index,
            measurement_index_sha256=hashlib.sha256(
                measurement_index.read_bytes()
            ).hexdigest(),
            output_json=selection_path,
            selector_source=cls.fixture["selector_source"],
        )
        cls.selection = selection_path
        cls.selection_sha = hashlib.sha256(
            selection_path.read_bytes()
        ).hexdigest()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _mutated_selection(
        self,
        name: str,
        mutate: object,
    ) -> Path:
        value = json.loads(self.selection.read_text(encoding="utf-8"))
        mutate(value)
        unsigned = dict(value)
        unsigned.pop("receipt_payload_sha256")
        value = producer_contract.receipt_payload(unsigned)
        path = self.root / name
        selector_fixture.write_json(path, value)
        return path

    def test_accepts_real_selector_five_stage_val_selection(self) -> None:
        result = consumer.load_selected_prerequisites(
            self.selection,
            self.selection_sha,
        )
        self.assertEqual(set(result["selected"]), set(consumer.STAGES))
        self.assertEqual(
            [result["selected"][stage]["epoch"] for stage in consumer.STAGES],
            [40, 80, 100, 120, 140],
        )
        self.assertEqual(
            set(result["producer_sources"]),
            {"evaluator", "merge", "selector"},
        )
        self.assertTrue(result["global_verified_not_consumed"])
        self.assertFalse(result["test_visible"])
        consumer.revalidate_selected_prerequisites(result)

    def test_missing_global_fails_closed(self) -> None:
        def mutate(value: dict[str, object]) -> None:
            value["stages"] = value["stages"][:-1]

        path = self._mutated_selection("missing-global.json", mutate)
        with self.assertRaises(consumer.SelectedPrerequisiteError):
            consumer.load_selected_prerequisites(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_selected_checkpoint_must_match_candidate_index(self) -> None:
        def mutate(value: dict[str, object]) -> None:
            value["stages"][0]["candidate_checkpoint"] = value["stages"][1][
                "candidate_checkpoint"
            ]

        path = self._mutated_selection("wrong-checkpoint.json", mutate)
        with self.assertRaises(consumer.SelectedPrerequisiteError):
            consumer.load_selected_prerequisites(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_non_val_or_nonfinite_selection_fails_closed(self) -> None:
        def mutate(value: dict[str, object]) -> None:
            value["stages"][3]["coverage"]["exact_once"] = False

        path = self._mutated_selection("invalid-coverage.json", mutate)
        with self.assertRaises(consumer.SelectedPrerequisiteError):
            consumer.load_selected_prerequisites(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_forged_merged_score_is_rejected_by_raw_shard_replay(
        self,
    ) -> None:
        selection = json.loads(
            self.selection.read_text(encoding="utf-8")
        )
        measurement_index_path = Path(
            selection["measurement_index_receipt"]["path"]
        )
        measurement_index = json.loads(
            measurement_index_path.read_text(encoding="utf-8")
        )
        face_receipt = measurement_index["stages"]["face"]
        face_path = Path(face_receipt["path"])
        face = json.loads(face_path.read_text(encoding="utf-8"))
        winner_epoch = selection["stages"][0]["epoch"]
        winner_row = next(
            row
            for row in face["candidates"]
            if row["epoch"] == winner_epoch
        )
        forged_score = float(winner_row["selection_score"]) + 123.0
        winner_row["selection_score"] = forged_score
        face.pop("receipt_payload_sha256")
        face = producer_contract.receipt_payload(face)
        forged_face_path = self.root / "forged-face-measurements.json"
        forged_face_sha = selector_fixture.write_json(
            forged_face_path,
            face,
        )

        measurement_index["stages"]["face"] = {
            **face_receipt,
            "path": str(forged_face_path),
            "sha256": forged_face_sha,
            "receipt_payload_sha256": face["receipt_payload_sha256"],
        }
        measurement_index.pop("receipt_payload_sha256")
        measurement_index = producer_contract.receipt_payload(
            measurement_index
        )
        forged_index_path = self.root / "forged-measurement-index.json"
        forged_index_sha = selector_fixture.write_json(
            forged_index_path,
            measurement_index,
        )

        selection["measurement_index_receipt"] = {
            "path": str(forged_index_path),
            "sha256": forged_index_sha,
            "receipt_payload_sha256": measurement_index[
                "receipt_payload_sha256"
            ],
        }
        selection["stages"][0]["selection_score"] = forged_score
        selection["stages"][0]["measurement_receipt"] = {
            "path": str(forged_face_path),
            "sha256": forged_face_sha,
            "receipt_payload_sha256": face["receipt_payload_sha256"],
        }
        selection.pop("receipt_payload_sha256")
        selection = producer_contract.receipt_payload(selection)
        forged_selection = self.root / "forged-score-selection.json"
        forged_selection_sha = selector_fixture.write_json(
            forged_selection,
            selection,
        )
        with self.assertRaisesRegex(
            consumer.SelectedPrerequisiteError,
            "raw shards",
        ):
            consumer.load_selected_prerequisites(
                forged_selection,
                forged_selection_sha,
            )

    def test_long_base_trainer_accepts_exact_selected_feature_lineage(
        self,
    ) -> None:
        bridge = consumer.load_selected_prerequisites(
            self.selection,
            self.selection_sha,
        )
        fixture_root = self.root / "selected-base-dataset"
        lmdb = fixture_root / "lmdb"
        lmdb.mkdir(parents=True)
        (lmdb / "data.mdb").write_bytes(b"selected-base-data")
        (lmdb / "lock.mdb").write_bytes(b"selected-base-lock")
        records = {}
        for stage in consumer.STAGES:
            selected = bridge["selected"][stage]
            checkpoint = selected["candidate_checkpoint"]
            records[stage] = {
                "formal_stage": stage,
                "path": checkpoint["path"],
                "sha256": checkpoint["sha256"],
                "bytes": checkpoint["bytes"],
                "prerequisite_source": "show_val_selected_v1",
                "training_dataset": "SHOW",
                "speaker_scope": "All",
                "show_trained": True,
                "selection_split": "val",
                "test_visible": False,
                "selected_epoch": selected["epoch"],
                "selected_optimizer_updates": selected[
                    "optimizer_updates"
                ],
                "selection_metric": selected["selection_metric"],
                "selection_score": selected["selection_score"],
                "candidate_audit_sha256": selected[
                    "candidate_audit_sha256"
                ],
                "measurement_receipt": selected["measurement_receipt"],
                "checkpoint_container_schema": ["audit", "model_state"],
                "strict_state_dict_load": True,
                "all_model_state_tensors_finite": True,
                "frozen_eval": True,
            }
        lineage = {
            "format": "semtalk_show_base_feature_lineage_v1",
            "status": "complete",
            "entries": base_trainer.EXPECTED_TRAIN_SAMPLES,
            "train_clips": base_trainer.EXPECTED_TRAIN_CLIPS,
            "protocol": {
                "scope": "SemTalk Base only",
                "split": "train",
                "speakers": base_trainer.SHOW_SPEAKERS,
                "window_length": base_trainer.POSE_LENGTH,
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
                "prerequisite_source": "show_val_selected_v1",
            },
            "formal_checkpoints": records,
            "prerequisite_source_receipt": bridge,
        }
        lineage_path = fixture_root / "lineage.json"
        selector_fixture.write_json(lineage_path, lineage)
        summary = {
            "format": "semtalk_show_base_lmdb_summary_v1",
            "status": "complete",
            "scope": "SemTalk Base only",
            "entries": base_trainer.EXPECTED_TRAIN_SAMPLES,
            "train_clips": base_trainer.EXPECTED_TRAIN_CLIPS,
            "lmdb": str(lmdb.resolve()),
            "data_mdb_sha256": hashlib.sha256(
                (lmdb / "data.mdb").read_bytes()
            ).hexdigest(),
            "lock_mdb_sha256": hashlib.sha256(
                (lmdb / "lock.mdb").read_bytes()
            ).hexdigest(),
            "lineage_json": str(lineage_path.resolve()),
            "lineage_json_sha256": hashlib.sha256(
                lineage_path.read_bytes()
            ).hexdigest(),
        }
        summary_path = fixture_root / "summary.json"
        selector_fixture.write_json(summary_path, summary)
        receipt = base_trainer.validate_dataset_receipts(
            argparse.Namespace(
                dataset_summary=str(summary_path),
                expected_dataset_summary_sha256=hashlib.sha256(
                    summary_path.read_bytes()
                ).hexdigest(),
                lineage_manifest=str(lineage_path),
                expected_lineage_sha256=hashlib.sha256(
                    lineage_path.read_bytes()
                ).hexdigest(),
                train_lmdb=str(lmdb),
                prerequisite_selection_json=str(self.selection),
                expected_prerequisite_selection_sha256=self.selection_sha,
            )
        )
        self.assertEqual(
            receipt["format"],
            "semtalk_show_base_selected_feature_dataset_receipt_v1",
        )
        self.assertEqual(
            set(receipt["selected_prerequisite_sha256"]),
            set(consumer.STAGES),
        )
        self.assertTrue(receipt["global_verified_not_consumed"])

    def test_global_is_strict_loaded_as_real_vae_schema(self) -> None:
        class TinyTensor:
            shape = (1,)
            dtype = "float32"

            def is_floating_point(self) -> bool:
                return True

            def is_complex(self) -> bool:
                return False

        class TinyFinite:
            def all(self) -> "TinyFinite":
                return self

            def item(self) -> bool:
                return True

        class TinyRVQ:
            instances: list["TinyRVQ"] = []

            def __init__(self, args: object) -> None:
                self.args = args
                self.training = True
                self.loaded = False
                self.__class__.instances.append(self)

            def to(self, device: str) -> "TinyRVQ":
                return self

            def load_state_dict(
                self,
                state: object,
                *,
                strict: bool,
            ) -> object:
                self.loaded = strict
                return types.SimpleNamespace(
                    missing_keys=[],
                    unexpected_keys=[],
                )

            def eval(self) -> None:
                self.training = False

            def requires_grad_(self, value: bool) -> None:
                self.requires_grad_value = value

            def parameters(self) -> list[object]:
                return []

        class TinyGlobal(TinyRVQ):
            instances: list["TinyGlobal"] = []

            def load_state_dict(
                self,
                state: object,
                *,
                strict: bool,
            ) -> object:
                if set(state) != {"global.expected"}:
                    raise RuntimeError("global schema mismatch")
                return super().load_state_dict(state, strict=strict)

        checkpoint_root = self.root / "strict-global"
        checkpoint_root.mkdir(exist_ok=True)
        training_sources = {}
        config_sha256 = {}
        selected = {}
        audits = {}
        for stage in consumer.STAGES:
            checkpoint = checkpoint_root / f"{stage}.bin"
            checkpoint.write_bytes(stage.encode("ascii"))
            source = {"stage": stage, "frozen": True}
            config = hashlib.sha256(
                f"config:{stage}".encode("ascii")
            ).hexdigest()
            audit = {
                "format": "semtalk_show_representation_candidate_v1",
                "formal_stage": stage,
                "completed_epochs": 20,
                "optimizer_updates": 20 * 497,
                "selection_status": "offline_validation_pending",
                "source_receipt": source,
                "config_sha256": config,
            }
            audits[stage] = audit
            training_sources[stage] = {"training_audit": source}
            config_sha256[stage] = config
            selected[stage] = {
                "epoch": 20,
                "optimizer_updates": 20 * 497,
                "selection_metric": consumer.EXPECTED_METRICS[stage],
                "selection_score": 0.0,
                "candidate_audit_sha256": (
                    consumer.canonical_json_sha256(audit)
                ),
                "candidate_checkpoint": {
                    "path": str(checkpoint),
                    "sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "bytes": checkpoint.stat().st_size,
                },
                "measurement_receipt": {
                    "path": str(checkpoint),
                    "sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "receipt_payload_sha256": "0" * 64,
                },
            }
        bridge = {
            "selection": {
                "path": "selection.json",
                "sha256": "1" * 64,
                "receipt_payload_sha256": "2" * 64,
            },
            "training_sources": training_sources,
            "config_sha256": config_sha256,
            "selected": selected,
        }
        fake_torch = types.ModuleType("torch")
        fake_torch.is_tensor = lambda value: isinstance(value, TinyTensor)
        fake_torch.isfinite = lambda value: TinyFinite()
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
        fake_rvq = types.ModuleType("models.rvq")
        fake_rvq.RVQVAE = TinyRVQ
        fake_motion = types.ModuleType("models.motion_representation")
        fake_motion.VAEConvZero = TinyGlobal
        invalid_global = False

        def fake_load(payload: bytes, path: Path) -> dict[str, object]:
            stage = path.stem
            state_key = (
                "global.invalid"
                if stage == "global" and invalid_global
                else (
                    "global.expected"
                    if stage == "global"
                    else f"{stage}.expected"
                )
            )
            return {
                "model_state": {state_key: TinyTensor()},
                "audit": audits[stage],
            }

        args = argparse.Namespace(
            prerequisite_selection_json="selection.json",
            expected_prerequisite_selection_sha256="1" * 64,
            device="cpu",
        )
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "models.rvq": fake_rvq,
                    "models.motion_representation": fake_motion,
                },
            ),
            mock.patch.object(
                feature_builder.selected_contract,
                "load_selected_prerequisites",
                return_value=bridge,
            ),
            mock.patch.object(
                feature_builder,
                "_torch_load",
                side_effect=fake_load,
            ),
        ):
            models, records, observed = (
                feature_builder.load_val_selected_models(args)
            )
            self.assertEqual(set(models), set(consumer.RVQ_STAGES))
            self.assertEqual(set(records), set(consumer.STAGES))
            self.assertIs(observed, bridge)
            self.assertEqual(len(TinyGlobal.instances), 1)
            self.assertTrue(TinyGlobal.instances[0].loaded)
            self.assertFalse(TinyGlobal.instances[0].training)

            invalid_global = True
            with self.assertRaisesRegex(
                RuntimeError,
                "global schema mismatch",
            ):
                feature_builder.load_val_selected_models(args)


if __name__ == "__main__":
    unittest.main()
