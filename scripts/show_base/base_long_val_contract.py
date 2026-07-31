#!/usr/bin/env python3
"""Twenty-two-candidate TalkSHOW validation contract for Base adaptation.

The validation input, five-prerequisite pipeline, and inference-lineage
helpers come exclusively from the TalkSHOW released2 authority.  Historical
evaluator selectors are intentionally outside this formal control closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.show_base import talkshow_base_val_contract as talkshow


OFFICIAL_BASE_SHA256 = (
    "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
)
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

# Re-export the audited TalkSHOW-only validation helpers used by inference.
EXPECTED_VAL_CLIPS = talkshow.EXPECTED_VAL_CLIPS
INFERENCE_HELPERS = talkshow.INFERENCE_HELPERS
VAL_INFERENCE_LINEAGE_FORMAT = talkshow.VAL_INFERENCE_LINEAGE_FORMAT
VAL_INFERENCE_SOURCE = talkshow.VAL_INFERENCE_SOURCE
canonical_json_sha256 = talkshow.canonical_json_sha256
sha256_file = talkshow.sha256_file
require_sha256 = talkshow.require_sha256
require_exact_int = talkshow.require_exact_int
reject_test_path = talkshow.reject_test_path
canonical_clip_id = talkshow.canonical_clip_id
public_val_coverage = talkshow.public_val_coverage
validate_val_inputs = talkshow.validate_val_inputs
validate_pipeline = talkshow.validate_fresh_pipeline
validate_val_inference_lineage = talkshow.validate_val_inference_lineage
_strict_json_bytes = talkshow._strict_json_bytes
_strict_jsonl = talkshow._strict_jsonl
SelectionContractError = talkshow.SelectionContractError


class LongCandidateContractError(talkshow.SelectionContractError):
    """Raised when the complete long Base candidate transaction is absent."""


def _verified_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    return talkshow._verified_json(path, expected_sha256, label)


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
    observed_hostnames: set[str] = set()
    for expected_rank, node in enumerate(node_bindings):
        if not isinstance(node, dict) or set(node) != {
            "node_rank",
            "hostname",
            "binding",
        }:
            raise LongCandidateContractError(
                "long Base frozen dataset node binding schema changed"
            )
        binding = node["binding"]
        files = binding.get("files") if isinstance(binding, dict) else None
        hostname = node["hostname"]
        if (
            node["node_rank"] != expected_rank
            or not isinstance(hostname, str)
            or not hostname
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
        checkpoint = talkshow._regular_file(
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
