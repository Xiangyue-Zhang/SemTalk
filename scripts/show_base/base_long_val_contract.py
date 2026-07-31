#!/usr/bin/env python3
"""Twenty-two-candidate validation contract for long SHOW Base adaptation.

All validation input, pipeline and inference-lineage helpers remain the
already-audited implementation in :mod:`select_base_official_adapt`.  This
module changes only the candidate transaction envelope from the historical
seven-candidate/e40 run to the complete long e400 trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.show_base import select_base_official_adapt as legacy


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

# Re-export the audited validation-only helpers used by the inference producer.
EXPECTED_VAL_CLIPS = legacy.EXPECTED_VAL_CLIPS
INFERENCE_HELPERS = legacy.INFERENCE_HELPERS
VAL_INFERENCE_LINEAGE_FORMAT = legacy.VAL_INFERENCE_LINEAGE_FORMAT
VAL_INFERENCE_SOURCE = legacy.VAL_INFERENCE_SOURCE
canonical_json_sha256 = legacy.canonical_json_sha256
sha256_file = legacy.sha256_file
require_sha256 = legacy.require_sha256
require_exact_int = legacy.require_exact_int
reject_test_path = legacy.reject_test_path
canonical_clip_id = legacy.canonical_clip_id
public_val_coverage = legacy.public_val_coverage
validate_val_inputs = legacy.validate_val_inputs
validate_pipeline = legacy.validate_pipeline
validate_val_inference_lineage = legacy.validate_val_inference_lineage
_strict_json_bytes = legacy._strict_json_bytes
_strict_jsonl = legacy._strict_jsonl
SelectionContractError = legacy.SelectionContractError


class LongCandidateContractError(legacy.SelectionContractError):
    """Raised when the complete long Base candidate transaction is absent."""


def _verified_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    return legacy._verified_json(path, expected_sha256, label)


def _artifact(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256}


def _validate_frozen(
    frozen: dict[str, Any],
) -> str:
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
    if (
        frozen.get("format") not in FROZEN_INPUTS_FORMATS
        or not isinstance(protocol, dict)
        or protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("target_dataset") != "SHOW"
        or protocol.get("target_speaker_scope") != "All"
        or protocol.get("candidate_epochs")
        != list(EXPECTED_CANDIDATE_EPOCHS)
        or protocol.get("epochs") != TOTAL_EPOCHS
        or protocol.get("expected_updates_per_epoch")
        != EXPECTED_UPDATES_PER_EPOCH
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
        != legacy.OFFICIAL_BASE_CHECKPOINT["sha256"]
        or official.get("speaker_scope") != "All-Speakers"
    ):
        raise LongCandidateContractError(
            "long Base frozen inputs do not bind selected SHOW prerequisites "
            "and the official All-Speakers Base warm start"
        )
    for stage, digest in dataset["selected_prerequisite_sha256"].items():
        require_sha256(digest, f"selected {stage} checkpoint SHA-256")
    return claimed


def validate_candidate_bundle(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    status_path: Path,
    expected_status_sha256: str,
    frozen_inputs_path: Path,
    expected_frozen_inputs_sha256: str,
) -> dict[str, Any]:
    """Verify the exact complete 22-candidate/e400 producer transaction."""

    frozen_path, frozen, frozen_sha = _verified_json(
        frozen_inputs_path,
        expected_frozen_inputs_sha256,
        "long Base frozen inputs",
    )
    frozen_receipt_sha = _validate_frozen(frozen)
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
    if (
        manifest.get("format") != CANDIDATE_MANIFEST_FORMAT
        or manifest.get("status") != "complete"
        or manifest.get("candidate_epochs")
        != list(EXPECTED_CANDIDATE_EPOCHS)
        or manifest.get("completed_epochs") != TOTAL_EPOCHS
        or manifest.get("optimizer_updates")
        != TOTAL_EPOCHS * EXPECTED_UPDATES_PER_EPOCH
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
            or updates != epoch * EXPECTED_UPDATES_PER_EPOCH
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
        checkpoint = legacy._regular_file(
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
        != TOTAL_EPOCHS * EXPECTED_UPDATES_PER_EPOCH
        or status.get("updates_per_epoch") != EXPECTED_UPDATES_PER_EPOCH
        or status.get("candidate_manifest_sha256") != manifest_sha
        or status.get("frozen_receipt_sha256") != frozen_receipt_sha
        or status.get("world_size") != 8
        or status.get("local_batch_size") != 64
        or status.get("global_batch_size") != 512
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
        "candidates": candidates,
    }
