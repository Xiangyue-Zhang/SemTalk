#!/usr/bin/env python3
"""Twenty-two-candidate DiffSHEG validation contract for Base adaptation.

The Base checkpoint family and the five-prerequisite inference pipeline stay
under the fresh SemTalk/DiffSHEG-primary source authority.  Selection is a
different concern: all 22 candidates are compared only by the reconstructed
DiffSHEG SHOW validation FGD protocol.  The explicit
``diffsheg_eval_clip_ids.txt`` manifest is therefore the selection coverage
authority; TalkSHOW released2 metrics never participate in Base selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scripts.show_base import select_base_official_adapt as diffsheg


OFFICIAL_BASE_SHA256 = (
    "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
)
FORMAL_HOST_BY_SLOT = {
    0: (
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0"
    ),
    1: (
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0"
    ),
}
TOPOLOGY_SPECS = {
    "official_w1_b64_reference": {
        "classification": "exact_official_runtime_topology_reference",
        "node_count": 1,
        "local_world_size": 1,
        "world_size": 1,
        "local_batch_size": 64,
        "global_batch_size": 64,
        "updates_per_epoch": 1_988,
        "unique_samples_per_epoch": 127_232,
        "learning_rate": 5e-5,
        "precision": "fp32",
        "formal_training_eligible": True,
    },
    "official_objective_w8_l8_g64_ddp_adaptation": {
        "classification": (
            "official_objective_ddp_adaptation_not_trajectory_equivalent"
        ),
        "node_count": 1,
        "local_world_size": 8,
        "world_size": 8,
        "local_batch_size": 8,
        "global_batch_size": 64,
        "updates_per_epoch": 1_988,
        "unique_samples_per_epoch": 127_232,
        "learning_rate": 5e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    "official_objective_w16_l4_g64_ddp_adaptation": {
        "classification": (
            "official_objective_ddp_adaptation_not_trajectory_equivalent"
        ),
        "node_count": 2,
        "local_world_size": 8,
        "world_size": 16,
        "local_batch_size": 4,
        "global_batch_size": 64,
        "updates_per_epoch": 1_988,
        "unique_samples_per_epoch": 127_232,
        "learning_rate": 5e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    "validation_gated_w8_l64_g512_empirical_acceleration": {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 1,
        "local_world_size": 8,
        "world_size": 8,
        "local_batch_size": 64,
        "global_batch_size": 512,
        "updates_per_epoch": 248,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 3e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    "validation_gated_w16_l32_g512_empirical_acceleration": {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 2,
        "local_world_size": 8,
        "world_size": 16,
        "local_batch_size": 32,
        "global_batch_size": 512,
        "updates_per_epoch": 248,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 3e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    "validation_gated_w8_l128_g1024_empirical_acceleration": {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 1,
        "local_world_size": 8,
        "world_size": 8,
        "local_batch_size": 128,
        "global_batch_size": 1_024,
        "updates_per_epoch": 124,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 3e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    "validation_gated_w8_l256_g2048_empirical_acceleration": {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 1,
        "local_world_size": 8,
        "world_size": 8,
        "local_batch_size": 256,
        "global_batch_size": 2_048,
        "updates_per_epoch": 62,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 3e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    "validation_gated_w16_l64_g1024_empirical_acceleration": {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 2,
        "local_world_size": 8,
        "world_size": 16,
        "local_batch_size": 64,
        "global_batch_size": 1_024,
        "updates_per_epoch": 124,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 3e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
    "validation_gated_w16_l64_g1024_lr6e5_empirical_acceleration": {
        "classification": "validation_gated_empirical_acceleration",
        "node_count": 2,
        "local_world_size": 8,
        "world_size": 16,
        "local_batch_size": 64,
        "global_batch_size": 1_024,
        "updates_per_epoch": 124,
        "unique_samples_per_epoch": 126_976,
        "learning_rate": 6e-5,
        "precision": "bf16",
        "formal_training_eligible": True,
    },
}


EXPECTED_CANDIDATE_EPOCHS = (
    1,
    2,
    4,
    8,
    16,
    32,
    40,
    50,
    60,
    70,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    240,
    280,
    320,
    360,
    400,
)
# Historical public name retained only for callers that validate archived
# global512 runs.  Fresh formal validation derives the selected topology from
# the hash-pinned frozen receipt and never assumes this value.
EXPECTED_UPDATES_PER_EPOCH = 248
TOTAL_EPOCHS = 400
CANDIDATE_MANIFEST_FORMAT = (
    "semtalk_show_base_official_adapt_long_manifest_v1"
)
CANDIDATE_STATUS_FORMAT = (
    "semtalk_show_base_official_adapt_long_status_v1"
)
THROUGHPUT_GATE_FORMAT = (
    "semtalk_show_base_official_adapt_long_throughput_gate_v1"
)
FROZEN_INPUTS_FORMATS = {
    "semtalk_show_base_official_adapt_frozen_inputs_v1",
    "semtalk_show_base_official_adapt_long_frozen_inputs_v1",
}
PROTOCOL_FORMAT = "semtalk_show_base_official_adapt_long_protocol_v1"
READY_FORMAT = (
    "semtalk_show_base_official_adapt_long_candidate_ready_v1"
)
SELECTION_FORMAT = "semtalk_show_base_official_adapt_long_selection_v1"
PRIMARY_SELECTION_PROTOCOL = diffsheg.PRIMARY_SELECTION_PROTOCOL
PRIMARY_SELECTION_METRIC_PATH = diffsheg.PRIMARY_SELECTION_METRIC_PATH
PRIMARY_SELECTION_REPORT_KEY = diffsheg.PRIMARY_SELECTION_REPORT_KEY
PRIMARY_SELECTION_ENTRYPOINTS = (
    "scripts/show_base/run_base_diffsheg_val_8shard.sh",
    "scripts/show_base/base_diffsheg_val_partition_contract.py",
    "scripts/show_base/run_base_val_inference.py",
    "scripts/show_base/evaluate_diffsheg_val_fgd.py",
    "scripts/show_base/produce_base_val_measurement.py",
    "scripts/show_base/finalize_base_diffsheg_val_partitions.sh",
    "scripts/show_base/select_base_official_adapt_long.py",
)
PRIMARY_AUTHORITY_FINAL_ENTRYPOINTS = (
    "scripts/show_base/validate_base_long_test_winner.py",
    "scripts/show_base/prepare_base_final_authority_inputs.py",
    "scripts/show_base/build_base_final_authority.sh",
    "scripts/show_base/base_final_authority.py",
    "scripts/show_base/run_base_final_test.sh",
    "scripts/show_base/run_base_final_test.py",
    "scripts/show_base/prepare_diffsheg_audio_view.py",
    "scripts/show_base/evaluate_diffsheg_final_test.py",
    "scripts/show_base/validate_diffsheg_final_result.py",
)
COMPATIBILITY_ONLY_ENTRYPOINTS = (
    "scripts/show_base/run_base_fresh_val_8shard.sh",
    "scripts/show_base/finalize_base_fresh_val_partitions.sh",
    "scripts/show_base/base_fresh_val_orchestrator.py",
    "scripts/show_base/select_published_base_winner.py",
    "scripts/show_base/published_test_winner_claim.py",
    "scripts/show_base/replay_released2_primary.py",
    "scripts/show_base/evaluate_talkshow_show_metrics.py",
)

# Re-export the exact formal validation ABI used by the 22-way producer.  The
# input/lineage, coverage, and fresh five-prerequisite pipeline validators are
# all pinned to the DiffSHEG-primary source closure.  Compatibility code is
# not a runtime dependency below this boundary.
EXPECTED_VAL_CLIPS = diffsheg.EXPECTED_VAL_CLIPS
INFERENCE_HELPERS = diffsheg.INFERENCE_HELPERS
VAL_INFERENCE_LINEAGE_FORMAT = diffsheg.VAL_INFERENCE_LINEAGE_FORMAT
VAL_INFERENCE_SOURCE = diffsheg.VAL_INFERENCE_SOURCE
canonical_json_sha256 = diffsheg.canonical_json_sha256
sha256_file = diffsheg.sha256_file
require_sha256 = diffsheg.require_sha256
require_exact_int = diffsheg.require_exact_int
canonical_clip_id = diffsheg.canonical_clip_id
public_val_coverage = diffsheg.public_val_coverage
validate_val_inputs = diffsheg.validate_val_inputs
validate_val_inference_lineage = diffsheg.validate_val_inference_lineage
_strict_json_bytes = diffsheg._strict_json_bytes
_strict_jsonl = diffsheg._strict_jsonl
SelectionContractError = diffsheg.SelectionContractError


def validate_diffsheg_report(
    report: Any,
    *,
    expected_coverage: Mapping[str, Any],
    inference_lineage: Mapping[str, Any] | None = None,
    expected_pipeline: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, int]]:
    """Validate FGD while mandatorily binding the fresh source closure."""

    return diffsheg.validate_diffsheg_report(
        report,
        expected_coverage=expected_coverage,
        inference_lineage=inference_lineage,
        expected_pipeline=expected_pipeline,
    )


def reject_test_path(path: Path, label: str) -> None:
    """Expose one error ABI while retaining the strict validation path gate."""

    try:
        diffsheg.reject_test_path(path, label)
    except diffsheg.SelectionContractError as exc:
        raise SelectionContractError(str(exc)) from exc


def validate_pipeline(
    path: Path,
    expected_sha256: str,
    *,
    expected_prerequisite_selection: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the fresh selected-five DiffSHEG-primary pipeline."""

    try:
        return diffsheg.validate_fresh_pipeline(
            path,
            expected_sha256,
            expected_prerequisite_selection=(
                expected_prerequisite_selection
            ),
            expected_source=expected_source,
        )
    except diffsheg.SelectionContractError as exc:
        raise SelectionContractError(str(exc)) from exc


class LongCandidateContractError(SelectionContractError):
    """Raised when the complete long Base candidate transaction is absent."""


def _verified_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    return diffsheg._verified_json(path, expected_sha256, label)


def _artifact(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256}


def _validate_frozen(
    frozen: dict[str, Any],
    *,
    expected_selected_prerequisite_sha256: dict[str, str] | None = None,
) -> tuple[str, dict[str, str], dict[str, Any], dict[str, Any]]:
    claimed = require_sha256(
        frozen.get("receipt_sha256"),
        "long Base frozen-input payload SHA-256",
    )
    unsigned = dict(frozen)
    unsigned.pop("receipt_sha256", None)
    if canonical_json_sha256(unsigned) != claimed:
        raise LongCandidateContractError(
            "long Base frozen-input payload SHA-256 mismatch"
        )
    protocol = frozen.get("protocol")
    dataset = frozen.get("dataset")
    source = frozen.get("source")
    official = frozen.get("official_base")
    topology_receipt = frozen.get("topology")
    distributed_topology = (
        protocol.get("distributed_topology")
        if isinstance(protocol, dict)
        else None
    )
    topology_mode = (
        distributed_topology.get("mode")
        if isinstance(distributed_topology, dict)
        else None
    )
    topology = TOPOLOGY_SPECS.get(topology_mode)
    topology_nodes = (
        distributed_topology.get("nodes")
        if isinstance(distributed_topology, dict)
        else None
    )
    expected_optimizer = (
        {
            "name": "Adam",
            "learning_rate": topology["learning_rate"],
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.99,
            "scheduler": "constant",
        }
        if isinstance(topology, dict)
        else None
    )
    if (
        frozen.get("format") not in FROZEN_INPUTS_FORMATS
        or not isinstance(protocol, dict)
        or protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("target_dataset") != "SHOW"
        or protocol.get("target_speaker_scope") != "All"
        or protocol.get("candidate_epochs")
        != list(EXPECTED_CANDIDATE_EPOCHS)
        or protocol.get("epochs") != TOTAL_EPOCHS
        or not isinstance(topology, dict)
        or protocol.get("expected_updates_per_epoch")
        != topology["updates_per_epoch"]
        or protocol.get("expected_unique_samples_per_epoch")
        != topology["unique_samples_per_epoch"]
        or protocol.get("node_count") != topology["node_count"]
        or protocol.get("local_world_size") != topology["local_world_size"]
        or protocol.get("world_size") != topology["world_size"]
        or protocol.get("local_batch_size") != topology["local_batch_size"]
        or protocol.get("global_batch_size") != topology["global_batch_size"]
        or protocol.get("precision") != topology["precision"]
        or protocol.get("optimizer") != expected_optimizer
        or not isinstance(topology_nodes, list)
        or len(topology_nodes) != topology["node_count"]
        or not isinstance(topology_receipt, dict)
        or topology_receipt.get("topology_mode") != topology_mode
        or topology_receipt.get("classification")
        != topology["classification"]
        or topology_receipt.get("node_count") != topology["node_count"]
        or topology_receipt.get("local_world_size")
        != topology["local_world_size"]
        or topology_receipt.get("world_size") != topology["world_size"]
        or topology_receipt.get("local_batch_size")
        != topology["local_batch_size"]
        or topology_receipt.get("global_batch_size")
        != topology["global_batch_size"]
        or topology_receipt.get("updates_per_epoch")
        != topology["updates_per_epoch"]
        or topology_receipt.get("unique_samples_per_epoch")
        != topology["unique_samples_per_epoch"]
        or topology_receipt.get("receipt_sha256")
        != canonical_json_sha256(
            {
                key: value
                for key, value in topology_receipt.items()
                if key != "receipt_sha256"
            }
        )
        or protocol.get("vq_models_in_training_graph") is not False
        or not isinstance(dataset, dict)
        or dataset.get("entries") != 127_286
        or dataset.get("train_clips") != 13_687
        or dataset.get("prerequisite_source") != "show_val_selected_v1"
        or dataset.get("global_verified_not_consumed") is not True
        or set(dataset.get("selected_prerequisite_sha256", {}))
        != {"face", "hands", "upper", "lower", "global"}
        or not isinstance(source, dict)
        or source.get("origin")
        != "git@github.com:Xiangyue-Zhang/SemTalk.git"
        or source.get("branch") is not None
        or source.get("clean") is not True
        or not isinstance(official, dict)
        or official.get("sha256")
        != OFFICIAL_BASE_SHA256
        or official.get("speaker_scope") != "All-Speakers"
    ):
        raise LongCandidateContractError(
            "long Base frozen inputs do not bind selected SHOW prerequisites "
            "and the official All-Speakers Base warm start"
        )
    observed_topology_host_slots: set[int] = set()
    observed_topology_hostnames: set[str] = set()
    for expected_rank, node in enumerate(topology_nodes):
        expected_rank_range = list(
            range(
                expected_rank * topology["local_world_size"],
                (expected_rank + 1) * topology["local_world_size"],
            )
        )
        if not isinstance(node, dict) or set(node) != {
            "node_rank",
            "host_slot",
            "hostname",
            "rank_range",
        }:
            raise LongCandidateContractError(
                "long Base frozen topology node schema changed"
            )
        host_slot = node["host_slot"]
        hostname = node["hostname"]
        if (
            type(node["node_rank"]) is not int
            or node["node_rank"] != expected_rank
            or type(host_slot) is not int
            or host_slot not in FORMAL_HOST_BY_SLOT
            or host_slot in observed_topology_host_slots
            or hostname != FORMAL_HOST_BY_SLOT[host_slot]
            or hostname in observed_topology_hostnames
            or not isinstance(node["rank_range"], list)
            or any(type(rank) is not int for rank in node["rank_range"])
            or node["rank_range"] != expected_rank_range
        ):
            raise LongCandidateContractError(
                "long Base frozen topology node/host mapping changed"
            )
        observed_topology_host_slots.add(host_slot)
        observed_topology_hostnames.add(hostname)
    for stage, digest in dataset["selected_prerequisite_sha256"].items():
        require_sha256(digest, f"selected {stage} checkpoint SHA-256")
    selection = dataset.get("prerequisite_selection")
    node_bindings = dataset.get("node_lmdb_inode_bindings")
    expected_identity_keys = {
        "device",
        "inode",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
    dataset_paths = {
        role: dataset.get(role)
        for role in ("lmdb", "summary", "lineage")
    }
    if (
        dataset.get("format")
        != "semtalk_show_base_selected_feature_dataset_receipt_v1"
        or dataset.get("split") != "train"
        or dataset.get("test_visible") is not False
        or any(
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in dataset_paths.values()
        )
        or require_sha256(
            dataset.get("summary_sha256"), "selected dataset summary SHA-256"
        )
        != dataset.get("summary_sha256")
        or require_sha256(
            dataset.get("lineage_sha256"), "selected dataset lineage SHA-256"
        )
        != dataset.get("lineage_sha256")
        or require_sha256(
            dataset.get("data_mdb_sha256"), "selected dataset data.mdb SHA-256"
        )
        != dataset.get("data_mdb_sha256")
        or require_sha256(
            dataset.get("lock_mdb_sha256"), "selected dataset lock.mdb SHA-256"
        )
        != dataset.get("lock_mdb_sha256")
        or not isinstance(selection, dict)
        or set(selection)
        != {"path", "sha256", "receipt_payload_sha256"}
        or not isinstance(selection.get("path"), str)
        or not Path(selection["path"]).is_absolute()
        or require_sha256(
            selection.get("sha256"), "selected prerequisite receipt SHA-256"
        )
        != selection.get("sha256")
        or require_sha256(
            selection.get("receipt_payload_sha256"),
            "selected prerequisite payload SHA-256",
        )
        != selection.get("receipt_payload_sha256")
        or dataset.get("lmdb_binding_scope")
        != "ordered_node_local_inode_bindings_with_global_content_sha256"
        or not isinstance(node_bindings, list)
        or len(node_bindings) != topology["node_count"]
    ):
        raise LongCandidateContractError(
            "long Base frozen dataset provenance is incomplete"
        )
    observed_host_slots: set[int] = set()
    observed_hostnames: set[str] = set()
    for expected_rank, node in enumerate(node_bindings):
        if not isinstance(node, dict) or set(node) != {
            "node_rank",
            "host_slot",
            "hostname",
            "binding",
        }:
            raise LongCandidateContractError(
                "long Base frozen dataset node binding schema changed"
            )
        binding = node["binding"]
        files = binding.get("files") if isinstance(binding, dict) else None
        host_slot = node["host_slot"]
        hostname = node["hostname"]
        topology_node = topology_nodes[expected_rank]
        if (
            type(node["node_rank"]) is not int
            or node["node_rank"] != expected_rank
            or type(host_slot) is not int
            or host_slot not in FORMAL_HOST_BY_SLOT
            or host_slot in observed_host_slots
            or hostname != FORMAL_HOST_BY_SLOT[host_slot]
            or host_slot != topology_node["host_slot"]
            or hostname != topology_node["hostname"]
            or hostname in observed_hostnames
            or not isinstance(binding, dict)
            or set(binding) != {"format", "directory_identity", "files"}
            or binding.get("format")
            != "semtalk_show_base_lmdb_inode_binding_v1"
            or not isinstance(binding.get("directory_identity"), dict)
            or set(binding["directory_identity"]) != expected_identity_keys
            or any(
                type(value) is not int or value < 0
                for value in binding["directory_identity"].values()
            )
            or not isinstance(files, dict)
            or set(files) != {"data.mdb", "lock.mdb"}
        ):
            raise LongCandidateContractError(
                "long Base frozen dataset node binding changed"
            )
        observed_host_slots.add(host_slot)
        observed_hostnames.add(hostname)
        for filename, expected_sha in (
            ("data.mdb", dataset["data_mdb_sha256"]),
            ("lock.mdb", dataset["lock_mdb_sha256"]),
        ):
            file_receipt = files[filename]
            if (
                not isinstance(file_receipt, dict)
                or set(file_receipt) != {"sha256", "identity"}
                or file_receipt.get("sha256") != expected_sha
                or not isinstance(file_receipt.get("identity"), dict)
                or set(file_receipt["identity"]) != expected_identity_keys
                or any(
                    type(value) is not int or value < 0
                    for value in file_receipt["identity"].values()
                )
            ):
                raise LongCandidateContractError(
                    "long Base frozen dataset file binding changed"
                )
    selected = dict(dataset["selected_prerequisite_sha256"])
    if (
        expected_selected_prerequisite_sha256 is not None
        and selected != expected_selected_prerequisite_sha256
    ):
        raise LongCandidateContractError(
            "long Base frozen inputs use different selected SHOW prerequisites"
        )
    return (
        claimed,
        selected,
        {
            "mode": topology_mode,
            **dict(topology),
            "topology_receipt_sha256": topology_receipt["receipt_sha256"],
        },
        {
            "format": dataset["format"],
            "lmdb": dataset["lmdb"],
            "summary": dataset["summary"],
            "summary_sha256": dataset["summary_sha256"],
            "lineage": dataset["lineage"],
            "lineage_sha256": dataset["lineage_sha256"],
            "entries": dataset["entries"],
            "train_clips": dataset["train_clips"],
            "split": dataset["split"],
            "test_visible": dataset["test_visible"],
            "data_mdb_sha256": dataset["data_mdb_sha256"],
            "lock_mdb_sha256": dataset["lock_mdb_sha256"],
            "prerequisite_selection": dict(selection),
            "selected_prerequisite_sha256": selected,
            "lmdb_binding_scope": dataset["lmdb_binding_scope"],
            "node_lmdb_inode_bindings": [dict(row) for row in node_bindings],
        },
    )


def validate_candidate_bundle(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    status_path: Path,
    expected_status_sha256: str,
    frozen_inputs_path: Path,
    expected_frozen_inputs_sha256: str,
    expected_selected_prerequisite_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify the exact complete 22-candidate/e400 producer transaction."""

    frozen_path, frozen, frozen_sha = _verified_json(
        frozen_inputs_path,
        expected_frozen_inputs_sha256,
        "long Base frozen inputs",
    )
    (
        frozen_receipt_sha,
        selected_prerequisite_sha256,
        selected_topology,
        selected_dataset,
    ) = _validate_frozen(
        frozen,
        expected_selected_prerequisite_sha256=(
            expected_selected_prerequisite_sha256
        ),
    )
    manifest_resolved, manifest, manifest_sha = _verified_json(
        manifest_path,
        expected_manifest_sha256,
        "long Base candidate manifest",
    )
    status_resolved, status, status_sha = _verified_json(
        status_path,
        expected_status_sha256,
        "long Base training status",
    )
    if not (
        frozen_path.parent
        == manifest_resolved.parent
        == status_resolved.parent
    ):
        raise LongCandidateContractError(
            "long Base manifest/status/frozen inputs must share one run root"
        )
    entries = manifest.get("entries")
    updates_per_epoch = int(selected_topology["updates_per_epoch"])
    if (
        manifest.get("format") != CANDIDATE_MANIFEST_FORMAT
        or manifest.get("status") != "complete"
        or manifest.get("candidate_epochs")
        != list(EXPECTED_CANDIDATE_EPOCHS)
        or manifest.get("completed_epochs") != TOTAL_EPOCHS
        or manifest.get("optimizer_updates")
        != TOTAL_EPOCHS * updates_per_epoch
        or manifest.get("frozen_receipt_sha256") != frozen_receipt_sha
        or not isinstance(entries, list)
        or len(entries) != len(EXPECTED_CANDIDATE_EPOCHS)
        or manifest.get("entries_sha256")
        != canonical_json_sha256(entries)
    ):
        raise LongCandidateContractError(
            "long Base candidate manifest is not exact and complete"
        )
    candidates: dict[int, dict[str, Any]] = {}
    expected_paths: set[Path] = set()
    for expected_epoch, entry in zip(EXPECTED_CANDIDATE_EPOCHS, entries):
        if not isinstance(entry, dict):
            raise LongCandidateContractError(
                f"long Base candidate e{expected_epoch} is not an object"
            )
        epoch = require_exact_int(
            entry.get("epoch"),
            f"long Base candidate e{expected_epoch} epoch",
        )
        updates = require_exact_int(
            entry.get("optimizer_updates"),
            f"long Base candidate e{expected_epoch} updates",
        )
        relative = entry.get("checkpoint")
        if (
            epoch != expected_epoch
            or updates != epoch * updates_per_epoch
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or entry.get("checkpoint_container_schema")
            != ["audit", "model_state"]
            or entry.get("all_model_state_tensors_finite") is not True
            or entry.get("frozen_receipt_sha256") != frozen_receipt_sha
        ):
            raise LongCandidateContractError(
                f"long Base candidate e{expected_epoch} protocol mismatch"
            )
        checkpoint = diffsheg._regular_file(
            manifest_resolved.parent / relative,
            f"long Base candidate e{expected_epoch}",
        )
        try:
            checkpoint.relative_to(manifest_resolved.parent)
        except ValueError as error:
            raise LongCandidateContractError(
                "long Base candidate escapes the immutable run root"
            ) from error
        digest = require_sha256(
            entry.get("checkpoint_sha256"),
            f"long Base candidate e{expected_epoch} SHA-256",
        )
        size = require_exact_int(
            entry.get("checkpoint_bytes"),
            f"long Base candidate e{expected_epoch} bytes",
        )
        if checkpoint.stat().st_size != size or sha256_file(checkpoint) != digest:
            raise LongCandidateContractError(
                f"long Base candidate e{expected_epoch} changed"
            )
        expected_paths.add(checkpoint)
        candidates[epoch] = {
            "path": str(checkpoint),
            "sha256": digest,
            "bytes": size,
        }
    candidate_dir = manifest_resolved.parent / "candidates"
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise LongCandidateContractError("long Base candidate directory unsafe")
    children = list(candidate_dir.iterdir())
    if (
        len(children) != len(EXPECTED_CANDIDATE_EPOCHS)
        or {path.resolve() for path in children} != expected_paths
        or any(path.is_symlink() or not path.is_file() for path in children)
    ):
        raise LongCandidateContractError(
            "long Base candidate directory is not exact-once"
        )

    if (
        status.get("format") != CANDIDATE_STATUS_FORMAT
        or status.get("status") != "complete"
        or status.get("completed_epochs") != TOTAL_EPOCHS
        or status.get("optimizer_updates")
        != TOTAL_EPOCHS * updates_per_epoch
        or status.get("updates_per_epoch") != updates_per_epoch
        or status.get("candidate_manifest_sha256") != manifest_sha
        or status.get("frozen_receipt_sha256") != frozen_receipt_sha
        or status.get("world_size") != selected_topology["world_size"]
        or status.get("local_batch_size")
        != selected_topology["local_batch_size"]
        or status.get("global_batch_size")
        != selected_topology["global_batch_size"]
        or status.get("all_training_state_finite") is not True
    ):
        raise LongCandidateContractError(
            "long Base training did not finalize exact e400 state"
        )
    return {
        "manifest": _artifact(manifest_resolved, manifest_sha),
        "status": _artifact(status_resolved, status_sha),
        "frozen_inputs": {
            "path": str(frozen_path),
            "sha256": frozen_sha,
            "receipt_sha256": frozen_receipt_sha,
        },
        "producer_source": dict(frozen["source"]),
        "selected_prerequisite_sha256": selected_prerequisite_sha256,
        "selected_topology": selected_topology,
        "selected_dataset": selected_dataset,
        "updates_per_epoch": updates_per_epoch,
        "candidates": candidates,
    }
