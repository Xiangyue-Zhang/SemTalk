#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "base_v14_live_validation_authority.py"
)
SPEC = importlib.util.spec_from_file_location("semtalk_v14_live_adapter", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ADAPTER.canonical_json_bytes(value, newline=True))
    return ADAPTER.safe_regular_hash(path, "fixture")[1]


def self_hash(value: dict[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = ADAPTER.canonical_json_sha256(result)
    return result


class FakeHooks:
    def __init__(self) -> None:
        self.require_exact_v14_schedule_bytes = False
        self.schema_sha = "a" * 64
        self.semantic_sha = "b" * 64
        self.topology_specs = copy.deepcopy(ADAPTER.SUPPORTED_TOPOLOGIES)
        self.pipeline_source_commit = None
        self.validation_evidence_source = {
            "origin": ADAPTER.EXPECTED_ORIGIN,
            "source_root": "/verified/validation/4066f20",
            "commit": ADAPTER.VALIDATION_EVIDENCE_SOURCE_COMMIT,
            "tree": ADAPTER.VALIDATION_EVIDENCE_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        self.runtime_validation_source = {
            "origin": ADAPTER.EXPECTED_ORIGIN,
            "source_root": "/verified/validation/70a70f4",
            "commit": ADAPTER.RUNTIME_VALIDATION_SOURCE_COMMIT,
            "tree": ADAPTER.RUNTIME_VALIDATION_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        self.runtime_validation_proof = {
            "format": "semtalk_show_base_runtime_validation_successor_proof_v1",
            "evidence_source": {
                "commit": ADAPTER.VALIDATION_EVIDENCE_SOURCE_COMMIT,
                "tree": ADAPTER.VALIDATION_EVIDENCE_SOURCE_TREE,
            },
            "runtime_source": {
                "commit": ADAPTER.RUNTIME_VALIDATION_SOURCE_COMMIT,
                "tree": ADAPTER.RUNTIME_VALIDATION_SOURCE_TREE,
            },
            "ancestry_verified": True,
            "file_projection": {"fixture": True},
        }

    def validate_training_source(self, frozen_source):
        if (
            frozen_source.get("origin") != ADAPTER.EXPECTED_ORIGIN
            or frozen_source.get("commit")
            != ADAPTER.PRODUCER_SOURCE_COMMIT
            or frozen_source.get("tree") != ADAPTER.PRODUCER_SOURCE_TREE
            or frozen_source.get("entrypoint_sha256")
            != ADAPTER.PRODUCER_TRAINER_SHA256
        ):
            raise ADAPTER.LiveValidationContractError(
                "training source changed"
            )
        return {
            **dict(frozen_source),
            "training_semantics_proof": {
                "format": "semtalk_show_base_v14_training_semantics_ast_proof_v1",
                "runtime_producer": {
                    "commit": ADAPTER.PRODUCER_SOURCE_COMMIT,
                    "tree": ADAPTER.PRODUCER_SOURCE_TREE,
                    "trainer_sha256": ADAPTER.PRODUCER_TRAINER_SHA256,
                },
                "training_semantics_source": {
                    "commit": ADAPTER.TRAINING_SEMANTICS_SOURCE_COMMIT,
                    "tree": ADAPTER.TRAINING_SEMANTICS_SOURCE_TREE,
                    "trainer_sha256": ADAPTER.TRAINING_SEMANTICS_TRAINER_SHA256,
                },
                "allowed_changed_definitions": sorted(
                    ADAPTER.TRAINING_SEMANTICS_ALLOWED_CHANGED_DEFS
                ),
                "unchanged_definition_count": 98,
                "unchanged_definitions_sha256": (
                    ADAPTER.TRAINING_SEMANTICS_UNCHANGED_DEFS_SHA256
                ),
            },
        }

    def validate_validation_sources(self):
        return (
            copy.deepcopy(self.validation_evidence_source),
            copy.deepcopy(self.runtime_validation_source),
            copy.deepcopy(self.runtime_validation_proof),
        )

    def validate_trajectory_probe(self, probe, topology_mode):
        del topology_mode
        if (
            not isinstance(probe, dict)
            or probe.get("format") != ADAPTER.TRAJECTORY_PROBE_FORMAT
            or probe.get("optimizer_updates") != 70
            or probe.get("world_size") != 8
        ):
            raise ADAPTER.LiveValidationContractError("probe changed")
        return copy.deepcopy(probe)

    def validate_throughput(self, *, reference, frozen, topology_mode):
        del frozen
        if reference.get("topology_mode") != topology_mode:
            raise ADAPTER.LiveValidationContractError("throughput topology changed")
        return copy.deepcopy(reference)

    def validate_long_contract(
        self, *, frozen, config, topology_mode
    ):
        del config, topology_mode
        return copy.deepcopy(frozen["long_contract"])

    def verify_checkpoint(
        self,
        *,
        path,
        expected_sha256,
        expected_bytes,
        epoch,
        updates_per_epoch,
        frozen_receipt_sha256,
    ):
        del expected_sha256, expected_bytes, epoch, updates_per_epoch, frozen_receipt_sha256
        payload = json.loads(path.read_text())
        return {
            "model_state_tensors": payload["model_state_tensors"],
            "model_state_schema_sha256": payload["model_state_schema_sha256"],
            "model_state_semantic_sha256": payload["model_state_semantic_sha256"],
            "audit": {"fixture": True},
        }

    def validate_val_inputs(self, path, expected_sha256):
        resolved, digest, size = ADAPTER.safe_regular_hash(path, "fake val inputs")
        if digest != expected_sha256:
            raise ADAPTER.LiveValidationContractError("val inputs changed")
        return (
            {"path": str(resolved), "sha256": digest, "bytes": size},
            {
                "split": "val",
                "test_visible": False,
                "clips": 1715,
                "coverage_sha256": "c" * 64,
            },
        )

    def validate_pipeline(
        self,
        path,
        expected_sha256,
        *,
        frozen,
        validation_evidence_source,
    ):
        resolved, digest, size = ADAPTER.safe_regular_hash(path, "fake pipeline")
        if digest != expected_sha256:
            raise ADAPTER.LiveValidationContractError("pipeline changed")
        fixed = {
            stage: {"sha256": frozen["dataset"]["selected_prerequisite_sha256"][stage]}
            for stage in ("face", "hands", "upper", "lower", "global")
        }
        return (
            {"path": str(resolved), "sha256": digest, "bytes": size},
            {
                "source": {
                    **copy.deepcopy(validation_evidence_source),
                    "commit": self.pipeline_source_commit
                    or validation_evidence_source["commit"],
                },
                "inference_entrypoint": {"path": "/verified/val/inference.py", "sha256": "d" * 64},
                "fixed_checkpoints": fixed,
            },
        )


class Fixture:
    def __init__(self, root: Path, mode: str = ADAPTER.W8G1024_MODE) -> None:
        self.root = root
        self.mode = mode
        self.specification = ADAPTER.SUPPORTED_TOPOLOGIES[mode]
        self.hooks = FakeHooks()
        self.train_root = root / "formal-run"
        self.authority_dir = root / "val-authorities"
        self.reconcile_dir = root / "val-reconciliation"
        self.source_root = root / "source"
        for directory in (
            self.train_root / "candidates",
            self.train_root / "candidate_manifest_snapshots",
            self.train_root / "candidate_receipts",
            self.authority_dir,
            self.reconcile_dir,
            self.source_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.schedule_path = root / "fresh-schedule.json"
        self.val_inputs = root / "val-inputs.json"
        self.pipeline = root / "val-pipeline.json"
        self.val_inputs_sha = write_json(self.val_inputs, {"split": "val", "clips": 1715})
        self.pipeline_sha = write_json(self.pipeline, {"split": "val", "pipeline": "fixed-five"})
        self.schedule = self._schedule()
        self.schedule_sha = write_json(self.schedule_path, self.schedule)
        self.frozen = self._frozen()
        self.frozen_path = self.train_root / "frozen_inputs.json"
        self.frozen_sha = write_json(self.frozen_path, self.frozen)
        self.entries: list[dict[str, object]] = []

    def _schedule(self):
        return {
            "format": ADAPTER.FRESH_SCHEDULE_FORMAT,
            "scope": "SemTalk Base only",
            "target_dataset": "SHOW",
            "target_speaker_scope": "All",
            "initialization": {
                "source": "released_all_speakers_v1",
                "checkpoint_sha256": ADAPTER.OFFICIAL_BASE_SHA256,
                "forbidden_epoch": 30,
                "forbidden_sources": ["Speaker2", "SemGate", "Sparse"],
            },
            "training": {
                "total_epochs": 400,
                "topology_source": ADAPTER.V14_TOPOLOGY_SOURCE,
                "topology_matrix": {
                    mode: copy.deepcopy(ADAPTER.SUPPORTED_TOPOLOGIES[mode])
                    for mode in ADAPTER.V14_MODES
                },
                "loader_workers": 4,
                "precision_source": "selected_topology_matrix_entry",
                "learning_rate_source": "selected_topology_matrix_entry",
                "optimizer": "Adam",
                "betas": [0.5, 0.999],
                "weight_decay": 0.0,
                "gradient_clip_norm": 0.99,
                "scheduler": "constant",
                "seed": 43,
                "vq_models_in_training_graph": False,
            },
            "topology_selection_contract": {
                "format": "semtalk_show_base_v14_formal_selection_binding_v1",
                "training_semantics_source_commit": (
                    ADAPTER.TRAINING_SEMANTICS_SOURCE_COMMIT
                ),
                "training_semantics_source_tree": (
                    ADAPTER.TRAINING_SEMANTICS_SOURCE_TREE
                ),
                "validation_source_commit": (
                    ADAPTER.VALIDATION_EVIDENCE_SOURCE_COMMIT
                ),
                "validation_source_tree": (
                    ADAPTER.VALIDATION_EVIDENCE_SOURCE_TREE
                ),
                "selection_protocol_sha256": (
                    ADAPTER.V14_SELECTION_PROTOCOL_SHA256
                ),
                "candidate_modes": list(ADAPTER.V14_MODES),
                "candidate_epochs": list(
                    ADAPTER.V14_TOPOLOGY_SELECTION_EPOCHS
                ),
                "validation_measurements": 12,
                "selection_tuple": [
                    "validation_diffsheg_fgd_ascending",
                    "epoch_ascending",
                    "topology_mode_lexicographic_ascending",
                ],
                "split": "val",
                "test_visible": False,
                "native_nine_mode_role": (
                    "optional_audit_evidence_never_v14_decision_authority"
                ),
                "fresh_selected_mode_throughput_gate_required": True,
            },
            "candidate_epochs": list(ADAPTER.CANDIDATE_EPOCHS),
            "validation_waves": [
                list(wave) for wave in ADAPTER.VALIDATION_WAVES
            ],
            "trajectory_contract": {
                "mode": ADAPTER.FRESH_TRAJECTORY_MODE,
                "external_anchor": False,
                "probe_optimizer_updates": 70,
                "probe_source": "matching_frozen_receipt_throughput_gate",
                "comparison": "byte_exact_model_state_semantic_sha256",
                "restart_from_official_initialization": True,
            },
            "selection": {
                "split": "val",
                "test_visible": False,
                "ordering": ["FGD", "epoch"],
                "direction": ["min", "min"],
                "test_runs": 1,
            },
        }

    def _identity(self, seed: int):
        return {
            "device": seed,
            "inode": seed + 1,
            "size": seed + 2,
            "mtime_ns": seed + 3,
            "ctime_ns": seed + 4,
        }

    def _frozen(self):
        spec = self.specification
        host_slot = 0
        hostname = ADAPTER.FORMAL_HOST_BY_SLOT[host_slot]
        formal_run_id = "formal-run-v14"
        master_addr = "10.0.0.1"
        master_port = 29500
        ranks = [
            {
                "rank": rank,
                "local_rank": rank,
                "node_rank": 0,
                "host_slot": host_slot,
                "hostname": hostname,
                "master_addr": master_addr,
                "master_port": master_port,
                "formal_run_id": formal_run_id,
            }
            for rank in range(8)
        ]
        topology = {
            "format": "semtalk_show_base_topology_receipt_v1",
            "topology_mode": self.mode,
            "classification": spec["classification"],
            "backend": "nccl",
            "node_count": 1,
            "local_world_size": 8,
            "world_size": 8,
            "global_batch_size": spec["global_batch_size"],
            "local_batch_size": spec["local_batch_size"],
            "updates_per_epoch": spec["updates_per_epoch"],
            "unique_samples_per_epoch": spec["unique_samples_per_epoch"],
            "master_addr": master_addr,
            "master_port": master_port,
            "formal_run_id": formal_run_id,
            "ranks": ranks,
        }
        topology = self_hash(topology, "receipt_sha256")
        selected = {
            "face": "1" * 64,
            "hands": "2" * 64,
            "upper": "3" * 64,
            "lower": "4" * 64,
            "global": "5" * 64,
        }
        selection = {
            "path": str(self.root / "selected-prerequisites.json"),
            "sha256": "6" * 64,
            "receipt_payload_sha256": "7" * 64,
        }
        dataset = {
            "format": "semtalk_show_base_selected_feature_dataset_receipt_v1",
            "lmdb": str(self.root / "train-lmdb"),
            "summary": str(self.root / "summary.json"),
            "summary_sha256": "8" * 64,
            "lineage": str(self.root / "lineage.json"),
            "lineage_sha256": "9" * 64,
            "entries": 127286,
            "train_clips": 13687,
            "split": "train",
            "test_visible": False,
            "prerequisite_source": "show_val_selected_v1",
            "global_verified_not_consumed": True,
            "data_mdb_sha256": "a" * 64,
            "lock_mdb_sha256": "b" * 64,
            "prerequisite_selection": selection,
            "selected_prerequisite_sha256": selected,
            "lmdb_binding_scope": "ordered_node_local_inode_bindings_with_global_content_sha256",
            "canonical_dataset_evidence": {"exact_once": True},
            "node_lmdb_inode_bindings": [
                {
                    "node_rank": 0,
                    "host_slot": host_slot,
                    "hostname": hostname,
                    "binding": {
                        "format": "semtalk_show_base_lmdb_inode_binding_v1",
                        "directory_identity": self._identity(10),
                        "files": {
                            "data.mdb": {"sha256": "a" * 64, "identity": self._identity(20)},
                            "lock.mdb": {"sha256": "b" * 64, "identity": self._identity(30)},
                        },
                    },
                }
            ],
        }
        trajectory_body = {
            "format": ADAPTER.FRESH_TRAJECTORY_FORMAT,
            "mode": ADAPTER.FRESH_TRAJECTORY_MODE,
            "schedule_sha256": self.schedule_sha,
            "official_base_checkpoint_sha256": ADAPTER.OFFICIAL_BASE_SHA256,
            "dataset_receipt_payload_sha256": "c" * 64,
            "dataset_split": "train",
            "test_visible": False,
            "dataset_summary_sha256": dataset["summary_sha256"],
            "feature_lineage_sha256": dataset["lineage_sha256"],
            "data_mdb_sha256": dataset["data_mdb_sha256"],
            "lock_mdb_sha256": dataset["lock_mdb_sha256"],
            "lmdb_binding_scope": "node_local_inode_content_global_sha256",
            "canonical_dataset_evidence": dataset["canonical_dataset_evidence"],
            "prerequisite_selection_sha256": selection["sha256"],
            "selected_prerequisite_sha256": {stage: selected[stage] for stage in sorted(selected)},
            "probe_optimizer_updates": 70,
            "probe_source": "matching_frozen_receipt_throughput_gate",
            "comparison": "byte_exact_rank_local_model_buffers_all_rank_parameters_adam_rng_and_sample_order_v3",
            "seed": 43,
            "precision": spec["precision"],
            "learning_rate": spec["learning_rate"],
            "topology_mode": self.mode,
            "topology_classification": spec["classification"],
            "world_size": 8,
            "node_count": 1,
            "local_world_size": 8,
            "local_batch_size": spec["local_batch_size"],
            "global_batch_size": spec["global_batch_size"],
            "updates_per_epoch": spec["updates_per_epoch"],
            "unique_samples_per_epoch": spec["unique_samples_per_epoch"],
            "loader_workers": 4,
        }
        trajectory_sha = ADAPTER.canonical_json_sha256(trajectory_body)
        trajectory = {
            **trajectory_body,
            "path": None,
            "sha256": trajectory_sha,
            "payload_sha256": trajectory_sha,
            "entries": {},
        }
        protocol = {
            "format": ADAPTER.PROTOCOL_FORMAT,
            "scope": "SemTalk Base only",
            "target_dataset": "SHOW",
            "target_speaker_scope": "All",
            "node_count": 1,
            "local_world_size": 8,
            "world_size": 8,
            "local_batch_size": spec["local_batch_size"],
            "global_batch_size": spec["global_batch_size"],
            "expected_updates_per_epoch": spec["updates_per_epoch"],
            "expected_unique_samples_per_epoch": spec["unique_samples_per_epoch"],
            "epochs": 400,
            "candidate_epochs": list(ADAPTER.CANDIDATE_EPOCHS),
            "trajectory_anchor_epochs": [],
            "distributed_topology": {
                "mode": self.mode,
                "classification": spec["classification"],
                "node_count": 1,
                "local_world_size": 8,
                "world_size": 8,
                "nodes": [
                    {
                        "node_rank": 0,
                        "host_slot": host_slot,
                        "hostname": hostname,
                        "rank_range": list(range(8)),
                    }
                ],
                "master_addr": master_addr,
                "master_port": master_port,
                "formal_run_id": formal_run_id,
            },
            "optimizer": {
                "name": "Adam",
                "learning_rate": spec["learning_rate"],
                "betas": [0.5, 0.999],
                "weight_decay": 0.0,
                "gradient_clip_norm": 0.99,
                "scheduler": "constant",
            },
            "precision": spec["precision"],
            "vq_models_in_training_graph": False,
            "schedule": {"path": str(self.schedule_path), "sha256": self.schedule_sha},
            "trajectory_anchor": {
                "mode": ADAPTER.FRESH_TRAJECTORY_MODE,
                "path": None,
                "sha256": trajectory_sha,
                "external": False,
            },
            "trajectory_gate": {
                "required": True,
                "probe_optimizer_updates": 70,
                "probe_source": "matching_frozen_receipt_throughput_gate",
                "comparison": "byte_exact_rank_local_model_buffers_all_rank_parameters_adam_rng_and_sample_order_v3",
            },
            "topology_gate_spec": {
                "path": str(self.root / "topology-gate.json"),
                "sha256": "d" * 64,
                "payload_sha256": "e" * 64,
                "selected_probe_mode": self.mode,
            },
        }
        frozen = {
            "format": ADAPTER.FROZEN_FORMAT,
            "run_purpose": "formal_training",
            "target_epochs": list(ADAPTER.CANDIDATE_EPOCHS),
            "source": {
                "origin": ADAPTER.EXPECTED_ORIGIN,
                "commit": ADAPTER.PRODUCER_SOURCE_COMMIT,
                "tree": ADAPTER.PRODUCER_SOURCE_TREE,
                "clean": True,
                "entrypoint_sha256": ADAPTER.PRODUCER_TRAINER_SHA256,
                "node_local_clones": [
                    {
                        "node_rank": 0,
                        "host_slot": host_slot,
                        "hostname": hostname,
                        "entrypoint": str(
                            self.source_root
                            / "scripts/show_base/train_base_official_adapt_long.py"
                        ),
                        "branch": None,
                    }
                ],
            },
            "official_base": {
                "source": "released_all_speakers_v1",
                "sha256": ADAPTER.OFFICIAL_BASE_SHA256,
                "speaker_scope": "All-Speakers",
                "training_dataset": "BEAT2",
                "all_model_state_tensors_finite": True,
                "strict_state_dict_load": True,
            },
            "speaker_initialization": {
                "format": "semtalk_show_official_speaker_mean_init_v1",
                "source_rows": 25,
                "target_show_rows": [0, 1, 2, 3],
                "other_state_unchanged": True,
            },
            "dataset": dataset,
            "protocol": protocol,
            "long_contract": {
                "format": "semtalk_show_base_long_contract_receipts_v1",
                "schedule": {
                    "path": str(self.schedule_path),
                    "sha256": self.schedule_sha,
                    "payload_sha256": ADAPTER.canonical_json_sha256(self.schedule),
                    "format": ADAPTER.FRESH_SCHEDULE_FORMAT,
                    "topology_source": ADAPTER.V14_TOPOLOGY_SOURCE,
                },
                "trajectory_anchor": trajectory,
            },
            "topology": topology,
        }
        return self_hash(frozen, "receipt_sha256")

    def probe(self):
        return {
            "format": ADAPTER.TRAJECTORY_PROBE_FORMAT,
            "optimizer_updates": 70,
            "world_size": 8,
            "rank_order": list(range(8)),
            "all_rank_model_state_identical": True,
            "all_rank_parameter_state_identical": True,
            "all_rank_buffer_state_identical": True,
            "all_rank_optimizer_state_identical": True,
            "ranks": [],
        }

    def throughput(self):
        probe = self.probe()
        return {
            "path": str(self.root / "throughput.json"),
            "sha256": "0" * 64,
            "topology_mode": self.mode,
            "samples_per_second": 1000.0,
            "seconds_per_update": 1.0,
            "median_seconds": 1.0,
            "p90_seconds": 1.1,
            "p99_seconds": 1.2,
            "estimated_training_seconds": 100.0,
            "trajectory_mode": ADAPTER.FRESH_TRAJECTORY_MODE,
            "trajectory_probe": probe,
            "gate_frozen_receipt_sha256": "1" * 64,
            "frozen_gate_compatibility_sha256": "2" * 64,
        }

    def publish_candidate(self, epoch: int):
        checkpoint_path = self.train_root / "candidates" / f"base_official_adapt_epoch_{epoch:02d}.bin"
        write_json(
            checkpoint_path,
            {
                "model_state_tensors": 1790,
                "model_state_schema_sha256": self.hooks.schema_sha,
                "model_state_semantic_sha256": self.hooks.semantic_sha,
            },
        )
        _, checkpoint_sha, checkpoint_bytes = ADAPTER.safe_regular_hash(checkpoint_path, "checkpoint")
        entry = {
            "epoch": epoch,
            "optimizer_updates": epoch * self.specification["updates_per_epoch"],
            "checkpoint": f"candidates/base_official_adapt_epoch_{epoch:02d}.bin",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_container_schema": ["audit", "model_state"],
            "model_state_tensors": 1790,
            "model_state_schema_sha256": self.hooks.schema_sha,
            "model_state_semantic_sha256": self.hooks.semantic_sha,
            "all_model_state_tensors_finite": True,
            "frozen_receipt_sha256": self.frozen["receipt_sha256"],
            "trajectory_anchor_match": None,
            "trajectory_probe_verified": True,
        }
        self.entries.append(entry)
        trajectory = self.frozen["long_contract"]["trajectory_anchor"]
        manifest = {
            "format": ADAPTER.MANIFEST_FORMAT,
            "status": "running",
            "candidate_epochs": list(ADAPTER.CANDIDATE_EPOCHS),
            "frozen_receipt_sha256": self.frozen["receipt_sha256"],
            "schedule_sha256": self.schedule_sha,
            "trajectory_anchor_sha256": trajectory["sha256"],
            "throughput_gate": self.throughput(),
            "trajectory_mode": ADAPTER.FRESH_TRAJECTORY_MODE,
            "trajectory_probe_verified": True,
            "trajectory_probe": self.probe(),
            "entries": copy.deepcopy(self.entries),
            "entries_sha256": ADAPTER.canonical_json_sha256(self.entries),
        }
        snapshot_path = self.train_root / "candidate_manifest_snapshots" / f"epoch-{epoch:04d}.json"
        snapshot_sha = write_json(snapshot_path, manifest)
        protocol = self.frozen["protocol"]
        ready = {
            "format": ADAPTER.READY_FORMAT,
            "status": "ready",
            "selection_eligible": False,
            "test_visible": False,
            "epoch": epoch,
            "optimizer_updates": entry["optimizer_updates"],
            "candidate_checkpoint": {
                "path": str(checkpoint_path),
                "relative_path": entry["checkpoint"],
                "sha256": checkpoint_sha,
                "bytes": checkpoint_bytes,
                "model_state_tensors": 1790,
                "model_state_schema_sha256": self.hooks.schema_sha,
                "model_state_semantic_sha256": self.hooks.semantic_sha,
            },
            "candidate_manifest": {
                "path": str(snapshot_path),
                "sha256_at_ready": snapshot_sha,
                "entries_sha256_at_ready": manifest["entries_sha256"],
                "immutable_snapshot": True,
                "live_path": str(self.train_root / "candidate_manifest.json"),
            },
            "frozen_inputs": {
                "path": str(self.frozen_path),
                "sha256": self.frozen_sha,
                "receipt_payload_sha256": self.frozen["receipt_sha256"],
            },
            "protocol": {
                "format": ADAPTER.PROTOCOL_FORMAT,
                "payload_sha256": ADAPTER.canonical_json_sha256(protocol),
            },
            "frozen_receipt_sha256": self.frozen["receipt_sha256"],
            "schedule_sha256": self.schedule_sha,
            "trajectory_anchor_sha256": trajectory["sha256"],
            "trajectory_anchor_match": None,
            "published_unix": float(epoch + 1000),
        }
        ready = self_hash(ready, "receipt_payload_sha256")
        ready_path = self.train_root / "candidate_receipts" / f"epoch-{epoch:04d}.json"
        write_json(ready_path, ready)
        return ready, manifest

    def config(self, epoch: int, *, output: Path | None = None):
        return ADAPTER.AuthorizeConfig(
            train_root=self.train_root,
            epoch=epoch,
            topology_mode=self.mode,
            producer_source_root=self.source_root,
            expected_producer_trainer_sha256=(
                ADAPTER.PRODUCER_TRAINER_SHA256
            ),
            expected_producer_contract_sha256=(
                ADAPTER.PRODUCER_V14_CONTRACT_SHA256
            ),
            validation_evidence_source_root=self.source_root,
            expected_validation_evidence_contract_sha256=(
                ADAPTER.VALIDATION_EVIDENCE_CONTRACT_SHA256
            ),
            expected_validation_evidence_selector_sha256=(
                ADAPTER.VALIDATION_EVIDENCE_SELECTOR_SHA256
            ),
            runtime_validation_source_root=self.source_root,
            expected_runtime_validation_contract_sha256=(
                ADAPTER.RUNTIME_VALIDATION_CONTRACT_SHA256
            ),
            expected_runtime_validation_selector_sha256=(
                ADAPTER.RUNTIME_VALIDATION_SELECTOR_SHA256
            ),
            schedule=self.schedule_path,
            expected_schedule_sha256=self.schedule_sha,
            expected_frozen_inputs_sha256=self.frozen_sha,
            val_inputs=self.val_inputs,
            expected_val_inputs_sha256=self.val_inputs_sha,
            pipeline=self.pipeline,
            expected_pipeline_sha256=self.pipeline_sha,
            output=output or self.authority_dir / f"epoch-{epoch:04d}.json",
        )

    def rewrite_ready(self, epoch: int, mutate):
        path = self.train_root / "candidate_receipts" / f"epoch-{epoch:04d}.json"
        value = json.loads(path.read_text())
        mutate(value)
        value = self_hash(value, "receipt_payload_sha256")
        write_json(path, value)

    def rewrite_frozen(self, mutate):
        value = json.loads(self.frozen_path.read_text())
        mutate(value)
        value = self_hash(value, "receipt_sha256")
        self.frozen = value
        self.frozen_sha = write_json(self.frozen_path, value)
        for ready_path in (self.train_root / "candidate_receipts").glob("*.json"):
            ready = json.loads(ready_path.read_text())
            ready["frozen_inputs"]["sha256"] = self.frozen_sha
            ready["frozen_inputs"]["receipt_payload_sha256"] = value["receipt_sha256"]
            ready["frozen_receipt_sha256"] = value["receipt_sha256"]
            ready["protocol"]["payload_sha256"] = ADAPTER.canonical_json_sha256(value["protocol"])
            write_json(ready_path, self_hash(ready, "receipt_payload_sha256"))

    def finalize(self):
        trajectory_sha = self.frozen["long_contract"]["trajectory_anchor"]["sha256"]
        manifest = {
            "format": ADAPTER.MANIFEST_FORMAT,
            "status": "complete",
            "candidate_epochs": list(ADAPTER.CANDIDATE_EPOCHS),
            "frozen_receipt_sha256": self.frozen["receipt_sha256"],
            "schedule_sha256": self.schedule_sha,
            "trajectory_anchor_sha256": trajectory_sha,
            "throughput_gate": self.throughput(),
            "trajectory_mode": ADAPTER.FRESH_TRAJECTORY_MODE,
            "trajectory_probe_verified": True,
            "trajectory_probe": self.probe(),
            "entries": copy.deepcopy(self.entries),
            "entries_sha256": ADAPTER.canonical_json_sha256(self.entries),
            "completed_epochs": 400,
            "optimizer_updates": 400 * self.specification["updates_per_epoch"],
        }
        manifest_path = self.train_root / "candidate_manifest.json"
        manifest_sha = write_json(manifest_path, manifest)
        metrics_path = self.train_root / "epoch_metrics.jsonl"
        metric_lines = []
        for epoch in range(1, 401):
            metric_lines.append(
                ADAPTER.canonical_json_bytes(
                    {
                        "format": "semtalk_show_base_long_epoch_metric_v1",
                        "epoch": epoch,
                        "optimizer_updates": epoch
                        * self.specification["updates_per_epoch"],
                        "updates_per_epoch": self.specification["updates_per_epoch"],
                        "learning_rate": self.specification["learning_rate"],
                        "metrics": {"loss": 1.0 / epoch},
                        "all_finite": True,
                        "completed_unix": float(1000 + epoch),
                    },
                    newline=True,
                )
            )
        metrics_path.write_bytes(b"".join(metric_lines))
        metrics_sha = ADAPTER.safe_regular_hash(
            metrics_path, "fixture epoch metrics"
        )[1]
        resume_dir = self.train_root / "resume"
        resume_dir.mkdir()
        resume_path = resume_dir / "latest_resume.bin"
        resume_path.write_bytes(b"synthetic-resume-payload")
        _, resume_sha, resume_bytes = ADAPTER.safe_regular_hash(
            resume_path, "fixture resume payload"
        )
        final_entry = self.entries[-1]
        resume_receipt_path = resume_dir / "latest_resume.json"
        resume_receipt = {
            "format": "semtalk_show_base_official_adapt_long_resume_receipt_v1",
            "status": "complete",
            "completed_epochs": 400,
            "optimizer_updates": 400 * self.specification["updates_per_epoch"],
            "path": str(resume_path),
            "sha256": resume_sha,
            "bytes": resume_bytes,
            "candidate_checkpoint": {
                "path": str(self.train_root / final_entry["checkpoint"]),
                "sha256": final_entry["checkpoint_sha256"],
                "model_state_semantic_sha256": final_entry[
                    "model_state_semantic_sha256"
                ],
            },
            "rank_rng_states": 8,
            "frozen_receipt_sha256": self.frozen["receipt_sha256"],
            "schedule_sha256": self.schedule_sha,
            "trajectory_anchor_sha256": trajectory_sha,
            "trajectory_mode": ADAPTER.FRESH_TRAJECTORY_MODE,
            "trajectory_probe_verified": True,
            "completed_unix": 2000.0,
        }
        resume_receipt_sha = write_json(resume_receipt_path, resume_receipt)
        status = {
            "format": ADAPTER.STATUS_FORMAT,
            "status": "complete",
            "completed_epochs": 400,
            "optimizer_updates": 400 * self.specification["updates_per_epoch"],
            "updates_per_epoch": self.specification["updates_per_epoch"],
            "candidate_manifest_sha256": manifest_sha,
            "frozen_receipt_sha256": self.frozen["receipt_sha256"],
            "world_size": 8,
            "local_batch_size": self.specification["local_batch_size"],
            "global_batch_size": self.specification["global_batch_size"],
            "all_training_state_finite": True,
            "epoch_metrics_jsonl": str(metrics_path),
            "epoch_metrics_sha256": metrics_sha,
            "epoch_metrics_records": 400,
            "schedule_sha256": self.schedule_sha,
            "trajectory_anchor_sha256": trajectory_sha,
            "trajectory_mode": ADAPTER.FRESH_TRAJECTORY_MODE,
            "trajectory_probe_verified": True,
            "throughput_gate": self.throughput(),
            "trajectory_probe": self.probe(),
            "started_unix": 900.0,
            "completed_unix": 2000.0,
            "resume_receipt": str(resume_receipt_path),
            "resume_receipt_sha256": resume_receipt_sha,
        }
        status_path = self.train_root / "status.json"
        status_sha = write_json(status_path, status)
        return manifest_path, manifest_sha, status_path, status_sha


class LiveValidationAdapterTests(unittest.TestCase):
    def fixture(self, mode=ADAPTER.W8G1024_MODE):
        temporary = tempfile.TemporaryDirectory(
            prefix="semtalk-live-val-", dir="/private/tmp"
        )
        self.addCleanup(temporary.cleanup)
        return Fixture(Path(temporary.name).resolve(), mode)

    def test_default_hooks_do_not_execute_before_source_validation(self):
        validation_root = Path(
            "/private/tmp/semtalk_final_integration_20260801"
        ).resolve(strict=True)
        producer_root = REPOSITORY.resolve(strict=True)
        with mock.patch.object(
            ADAPTER.ValidationHooks, "_load_verified_modules"
        ) as loader:
            ADAPTER.ValidationHooks(
                producer_source_root=producer_root,
                expected_producer_trainer_sha256=(
                    ADAPTER.PRODUCER_TRAINER_SHA256
                ),
                expected_producer_contract_sha256=(
                    ADAPTER.PRODUCER_V14_CONTRACT_SHA256
                ),
                validation_evidence_source_root=validation_root,
                expected_validation_evidence_contract_sha256=(
                    ADAPTER.VALIDATION_EVIDENCE_CONTRACT_SHA256
                ),
                expected_validation_evidence_selector_sha256=(
                    ADAPTER.VALIDATION_EVIDENCE_SELECTOR_SHA256
                ),
                runtime_validation_source_root=Path(
                    "/private/tmp/semtalk_diffsheg_provenance_fix_20260802"
                ).resolve(strict=True),
                expected_runtime_validation_contract_sha256=(
                    ADAPTER.RUNTIME_VALIDATION_CONTRACT_SHA256
                ),
                expected_runtime_validation_selector_sha256=(
                    ADAPTER.RUNTIME_VALIDATION_SELECTOR_SHA256
                ),
            )
        loader.assert_not_called()

    def test_producer_pin_is_exact_8f_and_control_successors_do_not_chase_it(self):
        self.assertEqual(
            ADAPTER.PRODUCER_SOURCE_COMMIT,
            "8f1fa7b85ed8253600a4c571e98eb9927edeb073",
        )
        self.assertEqual(
            ADAPTER.PRODUCER_SOURCE_TREE,
            "d5e5eb74acdc6d9ca330d5731e718e33b67cf626",
        )
        expected = {
            "scripts/show_base/train_base_official_adapt_long.py": (
                ADAPTER.PRODUCER_TRAINER_SHA256
            ),
            "scripts/show_base/base_v14_formal_contract.py": (
                ADAPTER.PRODUCER_V14_CONTRACT_SHA256
            ),
            "configs/show_base/semtalk_base_v14_formal_schedule_20260802.json": (
                ADAPTER.V14_SCHEDULE_SHA256
            ),
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                blob = subprocess.run(
                    [
                        "git", "-C", str(REPOSITORY), "show",
                        f"{ADAPTER.PRODUCER_SOURCE_COMMIT}:{relative}",
                    ],
                    check=True,
                    capture_output=True,
                ).stdout
                self.assertEqual(hashlib.sha256(blob).hexdigest(), digest)

    def test_default_training_source_accepts_only_portable_8f_clone_schema(self):
        hooks = ADAPTER.ValidationHooks.__new__(ADAPTER.ValidationHooks)
        hooks.producer_source_root = REPOSITORY.resolve(strict=True)
        hooks._git_checkout_authority = mock.Mock(
            return_value={
                "origin": ADAPTER.EXPECTED_ORIGIN,
                "commit": ADAPTER.PRODUCER_SOURCE_COMMIT,
                "tree": ADAPTER.PRODUCER_SOURCE_TREE,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            }
        )
        hooks._load_verified_modules = mock.Mock()
        entrypoint = (
            REPOSITORY
            / "scripts/show_base/train_base_official_adapt_long.py"
        ).resolve(strict=True)
        source = {
            "origin": ADAPTER.EXPECTED_ORIGIN,
            "commit": ADAPTER.PRODUCER_SOURCE_COMMIT,
            "tree": ADAPTER.PRODUCER_SOURCE_TREE,
            "clean": True,
            "entrypoint_sha256": ADAPTER.PRODUCER_TRAINER_SHA256,
            "node_local_clones": [
                {
                    "node_rank": 0,
                    "host_slot": 0,
                    "hostname": ADAPTER.FORMAL_HOST_BY_SLOT[0],
                    "entrypoint": str(entrypoint),
                    "branch": None,
                }
            ],
        }
        with mock.patch.object(
            ADAPTER,
            "_prove_training_semantics",
            return_value={"format": "fixture-proof"},
        ):
            accepted = hooks.validate_training_source(source)
            self.assertEqual(
                accepted["training_semantics_proof"],
                {"format": "fixture-proof"},
            )
            with self.assertRaises(ADAPTER.LiveValidationContractError):
                hooks.validate_training_source(None)
            for label, mutate in (
                (
                    "old-top-level-entrypoint-schema",
                    lambda value: value.update(
                        {"entrypoint": str(entrypoint), "branch": None}
                    ),
                ),
                (
                    "missing-clones",
                    lambda value: value.pop("node_local_clones"),
                ),
                (
                    "bool-node-rank",
                    lambda value: value["node_local_clones"][0].update(
                        {"node_rank": True}
                    ),
                ),
                (
                    "bool-host-slot",
                    lambda value: value["node_local_clones"][0].update(
                        {"host_slot": True}
                    ),
                ),
                (
                    "clone-branch",
                    lambda value: value["node_local_clones"][0].update(
                        {"branch": "main"}
                    ),
                ),
                (
                    "clone-hostname",
                    lambda value: value["node_local_clones"][0].update(
                        {"hostname": "forged-host"}
                    ),
                ),
                (
                    "clone-entrypoint",
                    lambda value: value["node_local_clones"][0].update(
                        {
                            "entrypoint": (
                                "/private/tmp/forged-source/scripts/show_base/"
                                "train_base_official_adapt_long.py"
                            )
                        }
                    ),
                ),
            ):
                attacked = copy.deepcopy(source)
                mutate(attacked)
                with self.subTest(label=label), self.assertRaises(
                    ADAPTER.LiveValidationContractError
                ):
                    hooks.validate_training_source(attacked)

    def test_authorize_both_supported_topologies(self):
        for mode in (ADAPTER.W8G1024_MODE, ADAPTER.W8G2048_MODE):
            fixture = self.fixture(mode)
            fixture.publish_candidate(1)
            artifact = ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)
            payload = json.loads(Path(artifact["path"]).read_text())
            self.assertEqual(payload["split"], "val")
            self.assertIs(payload["test_visible"], False)
            self.assertIs(payload["selection_eligible"], False)
            self.assertEqual(payload["selected_topology"]["mode"], mode)
            self.assertEqual(
                payload["producer_source"]["commit"],
                ADAPTER.PRODUCER_SOURCE_COMMIT,
            )
            self.assertEqual(
                payload["producer_source"]["training_semantics_proof"][
                    "training_semantics_source"
                ]["commit"],
                ADAPTER.TRAINING_SEMANTICS_SOURCE_COMMIT,
            )
            self.assertEqual(
                payload["validation_evidence_source"]["commit"],
                ADAPTER.VALIDATION_EVIDENCE_SOURCE_COMMIT,
            )
            self.assertEqual(
                payload["pipeline_source"],
                payload["validation_evidence_source"],
            )
            self.assertEqual(
                payload["runtime_validation_source"]["commit"],
                ADAPTER.RUNTIME_VALIDATION_SOURCE_COMMIT,
            )
            self.assertIs(
                payload["runtime_validation_proof"]["ancestry_verified"],
                True,
            )

    def test_checkpoint_without_ready_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        (fixture.train_root / "candidate_receipts/epoch-0001.json").unlink()
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_duplicate_json_key_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        path = fixture.train_root / "candidate_receipts/epoch-0001.json"
        value = path.read_text().rstrip()
        path.write_text(value[:-1] + ',"status":"ready"}\n')
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_nonfinite_json_token_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        path = fixture.train_root / "candidate_receipts/epoch-0001.json"
        encoded = path.read_text().replace(
            '"published_unix":1001.0', '"published_unix":NaN'
        )
        path.write_text(encoded)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_symlinked_validation_inputs_are_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        alias = fixture.root / "val-inputs-alias.json"
        alias.symlink_to(fixture.val_inputs)
        config = ADAPTER.AuthorizeConfig(
            **{**fixture.config(1).__dict__, "val_inputs": alias}
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(config, hooks=fixture.hooks)

    def test_rehashed_test_visible_attack_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        fixture.rewrite_ready(1, lambda value: value.__setitem__("test_visible", True))
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_bool_integer_attack_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        fixture.rewrite_ready(
            1, lambda value: value.__setitem__("epoch", True)
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_schedule_bool_numeric_alias_attack_is_rejected(self):
        class BoolScheduleFixture(Fixture):
            def _schedule(self):
                value = super()._schedule()
                value["training"]["weight_decay"] = False
                value["selection"]["test_runs"] = True
                return value

        temporary = tempfile.TemporaryDirectory(
            prefix="semtalk-live-val-bool-schedule-", dir="/private/tmp"
        )
        self.addCleanup(temporary.cleanup)
        fixture = BoolScheduleFixture(Path(temporary.name).resolve())
        fixture.publish_candidate(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_schedule_forbidden_source_and_wave_contradiction_is_rejected(self):
        class ContradictoryScheduleFixture(Fixture):
            def _schedule(self):
                value = super()._schedule()
                value["initialization"]["source"] = "Speaker2"
                value["initialization"]["forbidden_epoch"] = 30.0
                value["validation_waves"][-1] = [30, 400]
                return value

        temporary = tempfile.TemporaryDirectory(
            prefix="semtalk-live-val-bad-schedule-", dir="/private/tmp"
        )
        self.addCleanup(temporary.cleanup)
        fixture = ContradictoryScheduleFixture(Path(temporary.name).resolve())
        fixture.publish_candidate(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_legacy_v1_schedule_is_rejected_by_v14_authority(self):
        class LegacyScheduleFixture(Fixture):
            def _schedule(self):
                value = super()._schedule()
                value["format"] = "semtalk_show_base_fresh_lineage_schedule_v1"
                value["training"]["topology_source"] = (
                    "sealed_nine_mode_topology_gate_v1"
                )
                value.pop("topology_selection_contract")
                value["trajectory_anchor_epochs"] = None
                return value

        temporary = tempfile.TemporaryDirectory(
            prefix="semtalk-live-val-legacy-schedule-", dir="/private/tmp"
        )
        self.addCleanup(temporary.cleanup)
        fixture = LegacyScheduleFixture(Path(temporary.name).resolve())
        fixture.publish_candidate(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_hardlinked_ready_receipt_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        ready = fixture.train_root / "candidate_receipts/epoch-0001.json"
        os.link(ready, fixture.root / "ready-alias.json")
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_checkpoint_append_attack_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        checkpoint = fixture.train_root / "candidates/base_official_adapt_epoch_01.bin"
        with checkpoint.open("ab") as handle:
            handle.write(b"forgery")
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_checkpoint_tensor_count_float_alias_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        fixture.rewrite_ready(
            1,
            lambda value: value["candidate_checkpoint"].__setitem__(
                "model_state_tensors",
                float(value["candidate_checkpoint"]["model_state_tensors"]),
            ),
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_semantic_forgery_with_rehashed_json_is_rejected(self):
        fixture = self.fixture()
        ready, manifest = fixture.publish_candidate(1)
        forged = "f" * 64
        manifest["entries"][-1]["model_state_semantic_sha256"] = forged
        manifest["entries_sha256"] = ADAPTER.canonical_json_sha256(manifest["entries"])
        snapshot = fixture.train_root / "candidate_manifest_snapshots/epoch-0001.json"
        snapshot_sha = write_json(snapshot, manifest)
        ready["candidate_checkpoint"]["model_state_semantic_sha256"] = forged
        ready["candidate_manifest"]["sha256_at_ready"] = snapshot_sha
        ready["candidate_manifest"]["entries_sha256_at_ready"] = manifest["entries_sha256"]
        write_json(
            fixture.train_root / "candidate_receipts/epoch-0001.json",
            self_hash(ready, "receipt_payload_sha256"),
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_manifest_entries_hash_attack_is_rejected(self):
        fixture = self.fixture()
        ready, manifest = fixture.publish_candidate(1)
        manifest["entries_sha256"] = "f" * 64
        snapshot = fixture.train_root / "candidate_manifest_snapshots/epoch-0001.json"
        snapshot_sha = write_json(snapshot, manifest)
        ready["candidate_manifest"]["sha256_at_ready"] = snapshot_sha
        ready["candidate_manifest"]["entries_sha256_at_ready"] = "f" * 64
        write_json(
            fixture.train_root / "candidate_receipts/epoch-0001.json",
            self_hash(ready, "receipt_payload_sha256"),
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_topology_swap_is_rejected(self):
        fixture = self.fixture(ADAPTER.W8G1024_MODE)
        fixture.publish_candidate(1)
        config = fixture.config(1)
        config = ADAPTER.AuthorizeConfig(
            **{**config.__dict__, "topology_mode": ADAPTER.W8G2048_MODE}
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(config, hooks=fixture.hooks)

    def test_host_slot_hostname_swap_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        fixture.rewrite_frozen(
            lambda value: value["dataset"]["node_lmdb_inode_bindings"][0].update(
                {"host_slot": 1}
            )
        )
        config = fixture.config(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(config, hooks=fixture.hooks)

    def test_source_clone_and_topology_host_binding_cannot_diverge(self):
        fixture = self.fixture()
        fixture.rewrite_frozen(
            lambda value: value["source"]["node_local_clones"][0].update(
                {
                    "host_slot": 1,
                    "hostname": ADAPTER.FORMAL_HOST_BY_SLOT[1],
                }
            )
        )
        fixture.publish_candidate(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(
                fixture.config(1), hooks=fixture.hooks
            )

    def test_self_consistent_unaudited_host_slot_is_rejected(self):
        fixture = self.fixture()

        def mutate(value):
            node = value["protocol"]["distributed_topology"]["nodes"][0]
            node["host_slot"] = 2
            node["hostname"] = None
            for rank in value["topology"]["ranks"]:
                rank["host_slot"] = 2
                rank["hostname"] = None
            binding = value["dataset"]["node_lmdb_inode_bindings"][0]
            binding["host_slot"] = 2
            binding["hostname"] = None
            value["topology"] = self_hash(
                value["topology"], "receipt_sha256"
            )

        fixture.rewrite_frozen(mutate)
        fixture.publish_candidate(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_pipeline_source_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        fixture.hooks.pipeline_source_commit = "3" * 40
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_runtime_source_cannot_masquerade_as_pipeline_evidence(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        fixture.hooks.pipeline_source_commit = (
            ADAPTER.RUNTIME_VALIDATION_SOURCE_COMMIT
        )
        with self.assertRaisesRegex(
            ADAPTER.LiveValidationContractError,
            "pipeline source differs",
        ):
            ADAPTER.authorize_candidate(
                fixture.config(1), hooks=fixture.hooks
            )

    def test_training_source_cannot_be_relabelled_as_validation_source(self):
        fixture = self.fixture()
        fixture.rewrite_frozen(
            lambda value: value["source"].update(
                {
                    "commit": ADAPTER.VALIDATION_EVIDENCE_SOURCE_COMMIT,
                    "tree": ADAPTER.VALIDATION_EVIDENCE_SOURCE_TREE,
                    "entrypoint_sha256": (
                        ADAPTER.VALIDATION_EVIDENCE_CONTRACT_SHA256
                    ),
                }
            )
        )
        fixture.publish_candidate(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(
                fixture.config(1), hooks=fixture.hooks
            )

    def test_training_semantics_commit_cannot_masquerade_as_runtime_producer(self):
        fixture = self.fixture()
        fixture.rewrite_frozen(
            lambda value: value["source"].update(
                {
                    "commit": ADAPTER.TRAINING_SEMANTICS_SOURCE_COMMIT,
                    "tree": ADAPTER.TRAINING_SEMANTICS_SOURCE_TREE,
                    "entrypoint_sha256": (
                        ADAPTER.TRAINING_SEMANTICS_TRAINER_SHA256
                    ),
                }
            )
        )
        fixture.publish_candidate(1)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(
                fixture.config(1), hooks=fixture.hooks
            )

    def test_real_runtime_producer_ast_proof(self):
        root = REPOSITORY
        proof = ADAPTER._prove_training_semantics(root.resolve(strict=True))
        self.assertEqual(
            proof["runtime_producer"]["commit"],
            ADAPTER.PRODUCER_SOURCE_COMMIT,
        )
        self.assertEqual(
            proof["training_semantics_source"]["commit"],
            ADAPTER.TRAINING_SEMANTICS_SOURCE_COMMIT,
        )
        self.assertEqual(
            proof["unchanged_definitions_sha256"],
            ADAPTER.TRAINING_SEMANTICS_UNCHANGED_DEFS_SHA256,
        )

    def test_real_runtime_validation_successor_proof(self):
        root = Path(
            "/private/tmp/semtalk_diffsheg_provenance_fix_20260802"
        )
        if not root.is_dir():
            self.skipTest("pinned runtime validation repository is absent")
        proof = ADAPTER._prove_runtime_validation_successor(
            root.resolve(strict=True)
        )
        self.assertEqual(
            proof["evidence_source"]["commit"],
            ADAPTER.VALIDATION_EVIDENCE_SOURCE_COMMIT,
        )
        self.assertEqual(
            proof["runtime_source"]["commit"],
            ADAPTER.RUNTIME_VALIDATION_SOURCE_COMMIT,
        )
        self.assertIs(proof["ancestry_verified"], True)
        self.assertEqual(
            proof["file_projection"][
                "scripts/show_base/select_base_official_adapt.py"
            ]["runtime_sha256"],
            ADAPTER.RUNTIME_VALIDATION_SELECTOR_SHA256,
        )

    def test_runtime_validation_pins_official_gesture_checkpoint_truth(self):
        self.assertEqual(
            ADAPTER.EXPECTED_DIFFSHEG_FGD_PROVENANCE,
            {
                "filename": "gesture.pth.tar",
                "sha256": (
                    "5eaf9b882a5ccd5f6eb4385aaadf3d28f3ee4382360ecb13c12f4904b3c3216e"
                ),
                "input_dim": 129,
                "latent_dim": 300,
                "state_container": "state_dict",
                "load_mode": "full_half_embedding_net",
            },
        )

    def test_fresh_trajectory_match_true_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        fixture.rewrite_ready(
            1, lambda value: value.__setitem__("trajectory_anchor_match", True)
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_missing_trajectory_probe_is_rejected(self):
        fixture = self.fixture()
        ready, manifest = fixture.publish_candidate(1)
        manifest["trajectory_probe_verified"] = False
        snapshot = fixture.train_root / "candidate_manifest_snapshots/epoch-0001.json"
        snapshot_sha = write_json(snapshot, manifest)
        ready["candidate_manifest"]["sha256_at_ready"] = snapshot_sha
        write_json(
            fixture.train_root / "candidate_receipts/epoch-0001.json",
            self_hash(ready, "receipt_payload_sha256"),
        )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(fixture.config(1), hooks=fixture.hooks)

    def test_test_labelled_output_is_rejected(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        forbidden = fixture.root / "test" / "epoch-0001.json"
        forbidden.parent.mkdir()
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.authorize_candidate(
                fixture.config(1, output=forbidden), hooks=fixture.hooks
            )

    def test_authority_is_create_new(self):
        fixture = self.fixture()
        fixture.publish_candidate(1)
        config = fixture.config(1)
        ADAPTER.authorize_candidate(config, hooks=fixture.hooks)
        with self.assertRaises(FileExistsError):
            ADAPTER.authorize_candidate(config, hooks=fixture.hooks)

    def test_reconciliation_before_e400_finalize_is_rejected(self):
        fixture = self.fixture()
        for epoch in ADAPTER.CANDIDATE_EPOCHS:
            fixture.publish_candidate(epoch)
            ADAPTER.authorize_candidate(
                fixture.config(epoch), hooks=fixture.hooks
            )
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.reconcile(
                ADAPTER.ReconcileConfig(
                    common=fixture.config(
                        1,
                        output=(
                            fixture.reconcile_dir / "reconciliation.json"
                        ),
                    ),
                    authority_dir=fixture.authority_dir,
                    final_manifest=fixture.train_root / "candidate_manifest.json",
                    expected_final_manifest_sha256="d" * 64,
                    final_status=fixture.train_root / "run_status.json",
                    expected_final_status_sha256="e" * 64,
                ),
                hooks=fixture.hooks,
            )

    def test_full_22_way_reconciliation_and_missing_attack(self):
        fixture = self.fixture()
        for epoch in ADAPTER.CANDIDATE_EPOCHS:
            fixture.publish_candidate(epoch)
            ADAPTER.authorize_candidate(fixture.config(epoch), hooks=fixture.hooks)
        manifest, manifest_sha, status, status_sha = fixture.finalize()
        output = fixture.reconcile_dir / "reconciliation.json"
        common = fixture.config(1, output=output)
        result = ADAPTER.reconcile(
            ADAPTER.ReconcileConfig(
                common=common,
                authority_dir=fixture.authority_dir,
                final_manifest=manifest,
                expected_final_manifest_sha256=manifest_sha,
                final_status=status,
                expected_final_status_sha256=status_sha,
            ),
            hooks=fixture.hooks,
        )
        payload = json.loads(Path(result["path"]).read_text())
        self.assertTrue(payload["all_exact"])
        self.assertTrue(payload["selection_eligible"])
        self.assertEqual(len(payload["work_authorities"]), 22)

        second = self.fixture()
        for epoch in ADAPTER.CANDIDATE_EPOCHS:
            second.publish_candidate(epoch)
            ADAPTER.authorize_candidate(second.config(epoch), hooks=second.hooks)
        manifest, manifest_sha, status, status_sha = second.finalize()
        (second.authority_dir / "epoch-0400.json").unlink()
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.reconcile(
                ADAPTER.ReconcileConfig(
                    common=second.config(
                        1, output=second.reconcile_dir / "reconciliation.json"
                    ),
                    authority_dir=second.authority_dir,
                    final_manifest=manifest,
                    expected_final_manifest_sha256=manifest_sha,
                    final_status=status,
                    expected_final_status_sha256=status_sha,
                ),
                hooks=second.hooks,
            )

    def test_rehashed_authority_field_attack_is_rejected_at_reconciliation(self):
        fixture = self.fixture()
        for epoch in ADAPTER.CANDIDATE_EPOCHS:
            fixture.publish_candidate(epoch)
            ADAPTER.authorize_candidate(fixture.config(epoch), hooks=fixture.hooks)
        manifest, manifest_sha, status, status_sha = fixture.finalize()
        authority = fixture.authority_dir / "epoch-0001.json"
        payload = json.loads(authority.read_text())
        payload["execution_contract"]["may_influence_training"] = True
        payload = self_hash(payload, "receipt_payload_sha256")
        authority.chmod(0o644)
        write_json(authority, payload)
        authority.chmod(0o444)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.reconcile(
                ADAPTER.ReconcileConfig(
                    common=fixture.config(
                        1, output=fixture.reconcile_dir / "reconciliation.json"
                    ),
                    authority_dir=fixture.authority_dir,
                    final_manifest=manifest,
                    expected_final_manifest_sha256=manifest_sha,
                    final_status=status,
                    expected_final_status_sha256=status_sha,
                ),
                hooks=fixture.hooks,
            )

    def test_rehashed_final_probe_closure_attack_is_rejected(self):
        fixture = self.fixture()
        for epoch in ADAPTER.CANDIDATE_EPOCHS:
            fixture.publish_candidate(epoch)
            ADAPTER.authorize_candidate(fixture.config(epoch), hooks=fixture.hooks)
        manifest_path, _, status_path, _ = fixture.finalize()
        manifest = json.loads(manifest_path.read_text())
        forged_probe = copy.deepcopy(manifest["trajectory_probe"])
        forged_probe["optimizer_updates"] = 71
        manifest["trajectory_probe"] = forged_probe
        manifest["throughput_gate"]["trajectory_probe"] = forged_probe
        manifest_sha = write_json(manifest_path, manifest)
        status = json.loads(status_path.read_text())
        status["candidate_manifest_sha256"] = manifest_sha
        status["trajectory_probe"] = forged_probe
        status["throughput_gate"]["trajectory_probe"] = forged_probe
        status_sha = write_json(status_path, status)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.reconcile(
                ADAPTER.ReconcileConfig(
                    common=fixture.config(
                        1, output=fixture.reconcile_dir / "reconciliation.json"
                    ),
                    authority_dir=fixture.authority_dir,
                    final_manifest=manifest_path,
                    expected_final_manifest_sha256=manifest_sha,
                    final_status=status_path,
                    expected_final_status_sha256=status_sha,
                ),
                hooks=fixture.hooks,
            )

    def test_rehashed_final_status_test_field_attack_is_rejected(self):
        fixture = self.fixture()
        for epoch in ADAPTER.CANDIDATE_EPOCHS:
            fixture.publish_candidate(epoch)
            ADAPTER.authorize_candidate(fixture.config(epoch), hooks=fixture.hooks)
        manifest_path, manifest_sha, status_path, _ = fixture.finalize()
        status = json.loads(status_path.read_text())
        status["test_visible"] = True
        status["test_metrics"] = {"FGD": 0.0}
        status_sha = write_json(status_path, status)
        with self.assertRaises(ADAPTER.LiveValidationContractError):
            ADAPTER.reconcile(
                ADAPTER.ReconcileConfig(
                    common=fixture.config(
                        1, output=fixture.reconcile_dir / "reconciliation.json"
                    ),
                    authority_dir=fixture.authority_dir,
                    final_manifest=manifest_path,
                    expected_final_manifest_sha256=manifest_sha,
                    final_status=status_path,
                    expected_final_status_sha256=status_sha,
                ),
                hooks=fixture.hooks,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
