from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_long_val_contract as contract
from scripts.show_base import build_base_diffsheg_val_inputs as inputs
from scripts.show_base import produce_base_val_measurement as producer
from scripts.show_base import select_base_official_adapt as legacy
from scripts.show_base import select_base_official_adapt_long as selector


def _frozen_for_topology(
    mode: str,
    *,
    optimizer_learning_rate: float | None = None,
) -> dict[str, object]:
    topology = contract.TOPOLOGY_SPECS[mode]
    optimizer_learning_rate = (
        topology["learning_rate"]
        if optimizer_learning_rate is None
        else optimizer_learning_rate
    )
    selected = {
        stage: character * 64
        for stage, character in zip(
            ("face", "hands", "upper", "lower", "global"),
            "12345",
        )
    }
    identity = {
        "device": 1,
        "inode": 2,
        "size": 3,
        "mtime_ns": 4,
        "ctime_ns": 5,
    }
    node_bindings = []
    topology_nodes = []
    for node_rank in range(topology["node_count"]):
        host_slot = node_rank
        hostname = contract.FORMAL_HOST_BY_SLOT[host_slot]
        topology_nodes.append(
            {
                "node_rank": node_rank,
                "host_slot": host_slot,
                "hostname": hostname,
                "rank_range": list(
                    range(
                        node_rank * topology["local_world_size"],
                        (node_rank + 1) * topology["local_world_size"],
                    )
                ),
            }
        )
        node_bindings.append(
            {
                "node_rank": node_rank,
                "host_slot": host_slot,
                "hostname": hostname,
                "binding": {
                    "format": "semtalk_show_base_lmdb_inode_binding_v1",
                    "directory_identity": dict(identity),
                    "files": {
                        "data.mdb": {
                            "sha256": "6" * 64,
                            "identity": dict(identity),
                        },
                        "lock.mdb": {
                            "sha256": "7" * 64,
                            "identity": dict(identity),
                        },
                    },
                },
            }
        )
    topology_receipt = {
        "format": "semtalk_show_base_topology_receipt_v1",
        "topology_mode": mode,
        "classification": topology["classification"],
        "node_count": topology["node_count"],
        "local_world_size": topology["local_world_size"],
        "world_size": topology["world_size"],
        "local_batch_size": topology["local_batch_size"],
        "global_batch_size": topology["global_batch_size"],
        "updates_per_epoch": topology["updates_per_epoch"],
        "unique_samples_per_epoch": topology["unique_samples_per_epoch"],
    }
    topology_receipt["receipt_sha256"] = contract.canonical_json_sha256(
        topology_receipt
    )
    protocol = {
        "format": contract.PROTOCOL_FORMAT,
        "target_dataset": "SHOW",
        "target_speaker_scope": "All",
        "candidate_epochs": list(contract.EXPECTED_CANDIDATE_EPOCHS),
        "epochs": contract.TOTAL_EPOCHS,
        "expected_updates_per_epoch": topology["updates_per_epoch"],
        "expected_unique_samples_per_epoch": topology[
            "unique_samples_per_epoch"
        ],
        "node_count": topology["node_count"],
        "local_world_size": topology["local_world_size"],
        "world_size": topology["world_size"],
        "local_batch_size": topology["local_batch_size"],
        "global_batch_size": topology["global_batch_size"],
        "precision": topology["precision"],
        "distributed_topology": {
            "mode": mode,
            "nodes": topology_nodes,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": optimizer_learning_rate,
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.99,
            "scheduler": "constant",
        },
        "vq_models_in_training_graph": False,
    }
    dataset = {
        "format": "semtalk_show_base_selected_feature_dataset_receipt_v1",
        "entries": 127_286,
        "train_clips": 13_687,
        "prerequisite_source": "show_val_selected_v1",
        "global_verified_not_consumed": True,
        "selected_prerequisite_sha256": selected,
        "lmdb": "/local/base-features.lmdb",
        "summary": "/local/base-summary.json",
        "summary_sha256": "8" * 64,
        "lineage": "/local/base-lineage.json",
        "lineage_sha256": "9" * 64,
        "split": "train",
        "test_visible": False,
        "data_mdb_sha256": "6" * 64,
        "lock_mdb_sha256": "7" * 64,
        "prerequisite_selection": {
            "path": "/efs/prerequisite-selection.json",
            "sha256": "a" * 64,
            "receipt_payload_sha256": "b" * 64,
        },
        "lmdb_binding_scope": (
            "ordered_node_local_inode_bindings_with_global_content_sha256"
        ),
        "node_lmdb_inode_bindings": node_bindings,
    }
    frozen = {
        "format": next(iter(contract.FROZEN_INPUTS_FORMATS)),
        "protocol": protocol,
        "dataset": dataset,
        "source": {
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "branch": None,
            "clean": True,
        },
        "official_base": {
            "sha256": contract.OFFICIAL_BASE_SHA256,
            "speaker_scope": "All-Speakers",
        },
        "topology": topology_receipt,
    }
    frozen["receipt_sha256"] = contract.canonical_json_sha256(frozen)
    return frozen


def _reseal_frozen(frozen: dict[str, object]) -> dict[str, object]:
    frozen = copy.deepcopy(frozen)
    frozen.pop("receipt_sha256", None)
    frozen["receipt_sha256"] = contract.canonical_json_sha256(frozen)
    return frozen


class BaseDiffSHEGLongClosureTests(unittest.TestCase):
    def test_frozen_node_bindings_require_exact_host_slot_authority(self) -> None:
        mode = "validation_gated_w16_l64_g1024_empirical_acceleration"
        accepted = _frozen_for_topology(mode)
        _claimed, _selected, _topology, dataset = contract._validate_frozen(
            accepted
        )
        self.assertEqual(
            [
                (row["node_rank"], row["host_slot"], row["hostname"])
                for row in dataset["node_lmdb_inode_bindings"]
            ],
            [
                (0, 0, contract.FORMAL_HOST_BY_SLOT[0]),
                (1, 1, contract.FORMAL_HOST_BY_SLOT[1]),
            ],
        )

        mutations = {}

        missing = copy.deepcopy(accepted)
        del missing["dataset"]["node_lmdb_inode_bindings"][0]["host_slot"]
        mutations["missing host_slot"] = missing

        wrong = copy.deepcopy(accepted)
        wrong["dataset"]["node_lmdb_inode_bindings"][0]["host_slot"] = 9
        mutations["wrong host_slot"] = wrong

        duplicate = copy.deepcopy(accepted)
        duplicate["dataset"]["node_lmdb_inode_bindings"][1]["host_slot"] = 0
        duplicate["dataset"]["node_lmdb_inode_bindings"][1]["hostname"] = (
            contract.FORMAL_HOST_BY_SLOT[0]
        )
        mutations["duplicate host_slot"] = duplicate

        swapped = copy.deepcopy(accepted)
        bindings = swapped["dataset"]["node_lmdb_inode_bindings"]
        bindings[0]["host_slot"], bindings[1]["host_slot"] = (
            bindings[1]["host_slot"],
            bindings[0]["host_slot"],
        )
        bindings[0]["hostname"], bindings[1]["hostname"] = (
            bindings[1]["hostname"],
            bindings[0]["hostname"],
        )
        mutations["swapped binding hosts"] = swapped

        wrong_hostname = copy.deepcopy(accepted)
        wrong_hostname["dataset"]["node_lmdb_inode_bindings"][0][
            "hostname"
        ] = contract.FORMAL_HOST_BY_SLOT[1]
        mutations["wrong hostname"] = wrong_hostname

        topology_swapped = copy.deepcopy(accepted)
        nodes = topology_swapped["protocol"]["distributed_topology"]["nodes"]
        nodes[0]["host_slot"], nodes[1]["host_slot"] = (
            nodes[1]["host_slot"],
            nodes[0]["host_slot"],
        )
        nodes[0]["hostname"], nodes[1]["hostname"] = (
            nodes[1]["hostname"],
            nodes[0]["hostname"],
        )
        mutations["swapped topology hosts"] = topology_swapped

        bool_topology_rank = copy.deepcopy(accepted)
        bool_topology_rank["protocol"]["distributed_topology"]["nodes"][0][
            "node_rank"
        ] = False
        mutations["boolean topology node rank"] = bool_topology_rank

        bool_binding_rank = copy.deepcopy(accepted)
        bool_binding_rank["dataset"]["node_lmdb_inode_bindings"][0][
            "node_rank"
        ] = False
        mutations["boolean dataset node rank"] = bool_binding_rank

        bool_rank_range = copy.deepcopy(accepted)
        bool_rank_range["protocol"]["distributed_topology"]["nodes"][0][
            "rank_range"
        ][0] = False
        mutations["boolean topology rank range"] = bool_rank_range

        for label, forged in mutations.items():
            with self.subTest(label=label), self.assertRaises(
                contract.LongCandidateContractError
            ):
                contract._validate_frozen(_reseal_frozen(forged))

    def test_frozen_topology_binds_the_exact_adam_learning_rate(self) -> None:
        modes = (
            "validation_gated_w16_l64_g1024_empirical_acceleration",
            "validation_gated_w16_l64_g1024_lr6e5_empirical_acceleration",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                accepted = _frozen_for_topology(mode)
                _claimed, _selected, topology, _dataset = (
                    contract._validate_frozen(accepted)
                )
                self.assertEqual(topology["mode"], mode)
                wrong_learning_rate = (
                    6e-5
                    if contract.TOPOLOGY_SPECS[mode]["learning_rate"] == 3e-5
                    else 3e-5
                )
                forged = _frozen_for_topology(
                    mode,
                    optimizer_learning_rate=wrong_learning_rate,
                )
                with self.assertRaisesRegex(
                    contract.LongCandidateContractError,
                    "frozen inputs do not bind",
                ):
                    contract._validate_frozen(forged)

    def test_long_profile_is_one_coherent_diffsheg_producer_abi(self) -> None:
        profile = selector._profile_values()
        self.assertIs(profile["validate_val_inputs"], contract.validate_val_inputs)
        self.assertIs(profile["validate_pipeline"], contract.validate_pipeline)
        self.assertIs(
            profile["validate_val_inference_lineage"],
            contract.validate_val_inference_lineage,
        )
        self.assertEqual(
            profile["VAL_INFERENCE_LINEAGE_FORMAT"],
            legacy.VAL_INFERENCE_LINEAGE_FORMAT,
        )

    def test_measurement_producer_is_val_only_and_binds_one_candidate(
        self,
    ) -> None:
        epoch = 200
        checkpoint = {
            "path": "/validation/base-e0200.bin",
            "sha256": "a" * 64,
            "bytes": 17,
        }
        artifacts = {
            role: {
                "path": f"/validation/{role}.json",
                "sha256": character * 64,
                "receipt_payload_sha256": character.upper().lower() * 64,
            }
            for role, character in (
                ("val", "b"),
                ("pipeline", "c"),
                ("lineage", "d"),
                ("report", "e"),
            )
        }
        artifacts["report"].pop("receipt_payload_sha256")
        measurement = producer.build_measurement(
            epoch=epoch,
            candidate_bundle={"candidates": {epoch: checkpoint}},
            val_inputs_artifact=artifacts["val"],
            pipeline_artifact=artifacts["pipeline"],
            inference_lineage_artifact=artifacts["lineage"],
            diffsheg_report_artifact=artifacts["report"],
        )
        claimed = measurement.pop("receipt_payload_sha256")
        self.assertEqual(legacy.canonical_json_sha256(measurement), claimed)
        self.assertEqual(measurement["split"], "val")
        self.assertFalse(measurement["test_visible"])
        self.assertTrue(measurement["selection_eligible"])
        self.assertEqual(
            measurement["candidate_checkpoint"],
            {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
        )
        self.assertNotIn("released2", str(measurement).casefold())

    def test_withdrawn_e30_is_not_a_producer_epoch(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            producer._epoch("30")

    def test_diffsheg_input_builder_freezes_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            canonical = root / "canonical.jsonl"
            canonical.write_bytes(b"canonical")
            args = argparse.Namespace(
                canonical_manifest=canonical,
                canonical_summary=root / "summary.json",
                canonical_lineage=root / "lineage.json",
                audio_manifest=[root / f"audio-{i}.jsonl" for i in range(8)],
                audio_summary=[root / f"audio-{i}-summary.json" for i in range(8)],
                audio_lineage=[root / f"audio-{i}-lineage.json" for i in range(8)],
                output=root / "diffsheg-val-inputs.json",
            )
            artifact_counter = 0

            def fake_artifact(path: Path, _label: str):
                nonlocal artifact_counter
                artifact_counter += 1
                return (
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(str(path).encode()).hexdigest(),
                    },
                    path,
                    b"canonical" if path == canonical else b"receipt",
                )

            captured: dict[str, object] = {}

            def fake_atomic(path, receipt, *, label, validator):
                captured.update(receipt)
                self.assertEqual(label, "DiffSHEG validation inputs receipt")
                self.assertIs(validator, legacy.validate_val_inputs)
                return path.resolve(), "f" * 64

            with (
                mock.patch.object(inputs.common, "_artifact", side_effect=fake_artifact),
                mock.patch.object(
                    legacy,
                    "_canonical_coverage",
                    return_value=(
                        {f"clip-{i}" for i in range(legacy.EXPECTED_VAL_CLIPS)},
                        {
                            "clip_count": legacy.EXPECTED_VAL_CLIPS,
                            "clip_ids_sha256": "1" * 64,
                            "diffsheg_clip_manifest_sha256": "2" * 64,
                        },
                    ),
                ),
                mock.patch.object(
                    legacy,
                    "_validate_val_canonical_receipts",
                    return_value=({"summary": True}, {"lineage": True}),
                ),
                mock.patch.object(
                    legacy,
                    "_audio_coverage",
                    return_value={
                        "manifests": [{"shard": i} for i in range(8)],
                        "summaries": [{"shard": i} for i in range(8)],
                        "lineages": [{"shard": i} for i in range(8)],
                        "num_shards": 8,
                    },
                ),
                mock.patch.object(inputs.common, "_atomic_new_json", side_effect=fake_atomic),
            ):
                result = inputs.build_inputs(args)
            self.assertEqual(artifact_counter, 27)
            self.assertEqual(captured["format"], legacy.VAL_INPUTS_FORMAT)
            self.assertEqual(captured["split"], "val")
            self.assertFalse(captured["test_visible"])
            self.assertEqual(
                captured["diffsheg_clip_manifest_sha256"],
                "2" * 64,
            )
            self.assertEqual(result["audio_shards"], 8)


if __name__ == "__main__":
    unittest.main()
