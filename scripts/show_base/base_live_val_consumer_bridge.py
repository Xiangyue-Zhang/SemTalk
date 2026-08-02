#!/usr/bin/env python3
"""Consume one live SemTalk SHOW Base candidate without exposing test data.

The producer-side live authority is deliberately *not* a complete 22-way
validation preflight.  This bridge therefore gives it a separate execution
ABI.  One externally SHA-pinned, create-new work authority may produce one
eight-shard SHOW ``val`` inference, one DiffSHEG FGD report, and one
selection-ineligible live measurement.  A deterministic claim prevents the
same authority from being consumed by a second run root.

Only after the producer publishes the e400 reconciliation receipt may the
``reconcile`` command replay all 22 live measurements against the final
candidate bundle and call the audited 22-way selector.  No command accepts a
test split, a test path, or a test result.

The immutable pipeline evidence remains bound to the clean detached 4066f20
source.  GPU-facing commands and DiffSHEG report validation dynamically import
only the clean detached 70a70f4 provenance-fix successor.  The work authority
keeps these roles separate; the bridge itself contains no model implementation.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
import uuid


sys.dont_write_bytecode = True

EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
VALIDATION_EVIDENCE_SOURCE_COMMIT = "4066f2096e1675f9c19d725894007ff25f3e9b4b"
VALIDATION_EVIDENCE_SOURCE_TREE = "0b66e3aa1fb23732e76e51492737c4ab1f4db2d0"
RUNTIME_VALIDATION_SOURCE_COMMIT = "70a70f452bdf743e317b583a7770980f0ce744c3"
RUNTIME_VALIDATION_SOURCE_TREE = "bdf7680f56f9f53c92e7ab0bf6c6a84ef4f83d69"
VALIDATION_EVIDENCE_CONTRACT_SHA256 = (
    "3983eb7adf2f8cbfcf738fca8cc1cf8f3b34acab274bb5dfd2e23964550f8daa"
)
VALIDATION_EVIDENCE_SELECTOR_SHA256 = (
    "1b76579d54efc99ac0d0d63f3d1a68ff671216e693ef9ada4714c9fcda004121"
)
RUNTIME_VALIDATION_CONTRACT_SHA256 = (
    "0168e7f9ab2b9a122656ed98c36dc36777f577816ef08b0186ffc892813d2c70"
)
RUNTIME_VALIDATION_SELECTOR_SHA256 = (
    "1857460c188096fcd4b24a9f8f5a0b0390a80eacd20744a46098195ac0f39ddf"
)
RUNTIME_VALIDATION_INFERENCE_SHA256 = (
    "ec3de79ce1e45ff5385db40797bbcfa0dc91ab3e61d265e8f975180515190b56"
)
RUNTIME_VALIDATION_MEASUREMENT_SHA256 = (
    "e985056c889c9803212a236597066877726826badfcd1fa91f329e5b00e56550"
)
RUNTIME_VALIDATION_LONG_SELECTOR_SHA256 = (
    "d813864a33a5ff2109096453f956329a96480dd11f26c8a9144228df34d67269"
)
EXPECTED_DIFFSHEG_FGD_PROVENANCE = {
    "filename": "gesture.pth.tar",
    "sha256": "5eaf9b882a5ccd5f6eb4385aaadf3d28f3ee4382360ecb13c12f4904b3c3216e",
    "input_dim": 129,
    "latent_dim": 300,
    "state_container": "state_dict",
    "load_mode": "full_half_embedding_net",
}
WORK_FORMAT = "semtalk_show_base_live_val_candidate_work_authority_v2"
RECONCILIATION_FORMAT = "semtalk_show_base_live_val_reconciliation_v2"
PREFLIGHT_FORMAT = "semtalk_show_base_live_val_execution_preflight_v2"
CLAIM_FORMAT = "semtalk_show_base_live_val_consumer_claim_v2"
RECOVERY_REQUEST_FORMAT = (
    "semtalk_show_base_v14_failed_consumer_recovery_request_v1"
)
RECOVERY_CLAIM_FORMAT = (
    "semtalk_show_base_live_val_consumer_recovery_claim_v1"
)
RECOVERY_AUTHORITY_FORMAT = (
    "semtalk_show_base_live_val_consumer_recovery_authority_v1"
)
FAILED_RUN_INVENTORY_FORMAT = (
    "semtalk_show_base_failed_consumer_run_inventory_v1"
)
MEASUREMENT_FORMAT = "semtalk_show_base_live_val_measurement_v2"
SELECTION_FORMAT = "semtalk_show_base_live_val_22way_selection_v2"

CANDIDATE_EPOCHS = (
    1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80,
    100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400,
)
RUNTIME_EXECUTION_SOURCE_FILES = (
    "scripts/show_base/__init__.py",
    "scripts/show_base/run_base_val_inference.py",
    "scripts/show_base/semtalk_base_inference_core.py",
    "scripts/show_base/evaluate_diffsheg_val_fgd.py",
    "scripts/show_base/base_long_val_contract.py",
    "scripts/show_base/select_base_official_adapt.py",
    "scripts/show_base/build_base_features.py",
    "scripts/show_base/selected_prerequisites.py",
    "scripts/show_base/prerequisite_val_contract.py",
    "scripts/show_base/merge_prerequisite_val_shards.py",
    "scripts/show_base/gate_task_space_on_show_v2.py",
    "scripts/show_base/gate_released_all_speakers_on_show.py",
    "utils/show_base_joints.py",
    "utils/rotation_conversions.py",
    "utils/__init__.py",
    "dataloaders/__init__.py",
    "dataloaders/data_tools.py",
    "models/__init__.py",
    "models/semtalk.py",
    "models/motion_encoder.py",
    "models/motion_representation.py",
    "models/rvq.py",
    "models/encdec.py",
    "models/residual_vq.py",
    "models/quantizer.py",
    "models/resnet.py",
    "models/utils/__init__.py",
    "models/utils/layer.py",
    "models/utils/skeleton.py",
)
RUNTIME_EXECUTION_CHANGED_FILES = frozenset(
    {
        "scripts/show_base/base_long_val_contract.py",
        "scripts/show_base/select_base_official_adapt.py",
    }
)
RUNTIME_EXECUTION_CHANGED_SHA256 = {
    "scripts/show_base/base_long_val_contract.py": (
        VALIDATION_EVIDENCE_CONTRACT_SHA256,
        RUNTIME_VALIDATION_CONTRACT_SHA256,
    ),
    "scripts/show_base/select_base_official_adapt.py": (
        VALIDATION_EVIDENCE_SELECTOR_SHA256,
        RUNTIME_VALIDATION_SELECTOR_SHA256,
    ),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_OID_RE = re.compile(r"[0-9a-f]{40}")
FORBIDDEN_SOURCE_RE = re.compile(
    r"(^|[^a-z0-9])(?:e(?:poch)?[-_]?30|speaker[-_]?2|semgate|sparse)"
    r"([^a-z0-9]|$)",
    re.IGNORECASE,
)

WORK_KEYS = frozenset(
    {
        "format", "status", "split", "test_visible", "selection_eligible",
        "candidate_epoch", "optimizer_updates", "candidate_epochs",
        "producer_ready_receipt", "producer_manifest_snapshot",
        "producer_manifest_entry", "candidate_checkpoint", "frozen_inputs",
        "protocol", "schedule", "trajectory_contract", "throughput_gate",
        "producer_source", "validation_evidence_source",
        "runtime_validation_source", "runtime_validation_proof",
        "selected_topology",
        "selected_prerequisite_sha256", "val_inputs_receipt", "coverage",
        "pipeline_receipt", "pipeline_source", "inference_entrypoint",
        "execution_contract", "published_unix", "receipt_payload_sha256",
    }
)
RECONCILIATION_KEYS = frozenset(
    {
        "format", "status", "split", "test_visible", "selection_eligible",
        "candidate_epochs", "all_exact", "producer_manifest",
        "producer_status", "frozen_inputs", "producer_source",
        "validation_evidence_source", "runtime_validation_source",
        "runtime_validation_proof", "selected_topology", "schedule",
        "trajectory_contract", "val_inputs_receipt", "pipeline_receipt",
        "pipeline_source", "work_authorities", "test_evaluations_observed",
        "completed_unix", "receipt_payload_sha256",
    }
)
EXECUTION_KEYS = frozenset(
    {
        "purpose", "split", "test_visible", "expected_shards",
        "candidate_epochs_in_work_item",
        "may_publish_standard_22_candidate_measurement_before_e400",
        "may_influence_training", "requires_guarded_runner",
    }
)
SOURCE_KEYS = frozenset(
    {
        "origin", "source_root", "commit", "tree", "clean", "detached",
        "local_branches_at_commit",
    }
)
SOURCE_RECEIPT_KEYS = SOURCE_KEYS | frozenset({"files"})
SOURCE_FILE_KEYS = frozenset(
    {"path", "sha256", "bytes", "git_mode", "git_blob_sha1"}
)
COVERAGE_KEYS = frozenset(
    {
        "split", "clip_count", "frame_count", "window_count",
        "uncovered_tail_frames", "clip_ids_sha256",
        "diffsheg_clip_manifest_sha256",
    }
)
SNAPSHOT_DOCUMENT_KEYS = frozenset(
    {
        "format", "status", "candidate_epochs", "frozen_receipt_sha256",
        "schedule_sha256", "trajectory_anchor_sha256", "throughput_gate",
        "trajectory_mode", "trajectory_probe_verified", "trajectory_probe",
        "entries", "entries_sha256",
    }
)
MANIFEST_ENTRY_KEYS = frozenset(
    {
        "epoch", "optimizer_updates", "checkpoint", "checkpoint_sha256",
        "checkpoint_bytes", "checkpoint_container_schema",
        "model_state_tensors", "model_state_schema_sha256",
        "model_state_semantic_sha256", "all_model_state_tensors_finite",
        "frozen_receipt_sha256", "trajectory_anchor_match",
        "trajectory_probe_verified",
    }
)
CLAIM_KEYS = frozenset(
    {
        "format", "status", "split", "test_visible", "selection_eligible",
        "candidate_epoch", "expected_shards", "work_authority", "run_root",
        "receipt_payload_sha256",
    }
)
ARTIFACT_CORE_KEYS = frozenset({"path", "sha256", "bytes"})
RECOVERY_REQUEST_KEYS = frozenset(
    {
        "format", "status", "split", "test_visible", "selection_eligible",
        "candidate_epoch", "failed_campaign", "failed_job_claim",
        "failed_active_claim", "failed_authorization", "failed_work_authority",
        "failed_consumer_claim", "failed_runner_status", "failed_runner_log",
        "failed_run_root", "failed_run_inventory", "guard_proof",
        "failed_process_proof",
        "new_campaign", "new_control_source", "new_work_authority",
        "new_run_root", "created_unix", "receipt_payload_sha256",
    }
)
RECOVERY_CLAIM_KEYS = frozenset(
    {
        "format", "status", "split", "test_visible", "selection_eligible",
        "candidate_epoch", "expected_shards", "recovery_request",
        "failed_consumer_claim", "failed_runner_status", "new_campaign",
        "new_control_source", "new_work_authority", "new_run_root",
        "recovery_authority", "receipt_payload_sha256",
    }
)
RECOVERY_AUTHORITY_KEYS = frozenset(
    {
        "format", "status", "split", "test_visible", "selection_eligible",
        "candidate_epoch", "expected_shards", "recovery_request",
        "recovery_claim_path", "failed_consumer_claim", "failed_runner_status",
        "new_campaign", "new_control_source", "new_work_authority",
        "new_run_root", "receipt_payload_sha256",
    }
)
FAILED_RUN_INVENTORY_KEYS = frozenset(
    {
        "format", "root", "directories", "files", "shard_log_sha256",
        "shard_failure_marker", "semantic_outputs",
    }
)
FAILED_RUN_FILE_KEYS = frozenset({"relative_path", "sha256", "bytes"})
GUARD_PROOF_KEYS = frozenset(
    {"verifier", "argv", "stdout", "restored_guards", "verified_unix"}
)
FAILED_PROCESS_PROOF_KEYS = frozenset(
    {
        "wrapper_pid", "child_pid", "wrapper_proc_state",
        "child_proc_state", "runner_command", "checked_unix",
    }
)
FAILED_CAMPAIGN_KEYS = frozenset(
    {
        "format", "status", "split", "test_visible",
        "test_measurements_authorized", "candidate_epochs", "state_root",
        "campaign_claim_path", "summary_path", "control_source",
        "runtime_validation_source", "authority_adapter_config",
        "final_manifest_path", "final_status_path", "paspa_root",
        "diffsheg_root", "seed", "diffsheg_batch_size", "formal_python",
        "formal_python_runtime_contract", "guarded_runner", "guard_verifier",
        "reconciliation_path", "reconcile_selection_root", "jobs",
        "campaign_payload_sha256",
    }
)
CAMPAIGN_JOB_KEYS = frozenset(
    {
        "epoch", "candidate_receipt_path", "authority_path",
        "authorization_path", "run_root", "measurement_path",
        "completion_path", "runner_status_path", "runner_log_path",
    }
)
FAILED_JOB_CLAIM_KEYS = frozenset(
    {
        "format", "status", "candidate_epoch", "campaign",
        "candidate_receipt", "authority_path", "run_root",
        "measurement_path", "authorize_argv_sha256", "created_unix",
        "claim_payload_sha256",
    }
)
FAILED_ACTIVE_CLAIM_KEYS = frozenset(
    {
        "format", "status", "operation", "campaign", "created_unix",
        "claim_payload_sha256",
    }
)
FAILED_AUTHORIZATION_KEYS = frozenset(
    {
        "format", "status", "candidate_epoch", "campaign",
        "candidate_receipt", "work_authority", "adapter", "adapter_argv",
        "adapter_stdout_sha256", "runner_argv", "completed_unix",
        "receipt_payload_sha256",
    }
)
FAILED_RUNNER_STATUS_KEYS = frozenset(
    {
        "updated_at", "state", "wrapper_pid", "child_pid", "return_code",
        "received_signal", "error", "cleanup_error", "restored_guards",
        "restore_error", "command",
    }
)
CONTROL_SOURCE_KEYS = frozenset(
    {"root", "origin", "commit", "tree", "supervisor", "launcher", "bridge", "authority_adapter"}
)
FORMAL_EPOCH_ONE = 1
FAILED_RUN_DIRECTORIES = (
    ".", "candidates", "candidates/e1", "candidates/e1/shards", "logs",
)
FAILED_RUN_FILES = (
    "diffsheg-evaluator-bundle.json",
    "logs/e1-shard0.log", "logs/e1-shard1.log",
    "logs/e1-shard2.log", "logs/e1-shard3.log",
    "logs/e1-shard4.log", "logs/e1-shard5.log",
    "logs/e1-shard6.log", "logs/e1-shard7.log",
    "logs/evaluator-preflight.log", "logs/work-inspect.json",
    "logs/work-preflight.log", "work-preflight.json",
)
FAILED_RUN_FILE_PINS = {
    "diffsheg-evaluator-bundle.json": (
        "de641ffb88c5393c5df7327f21b704588f18ece21bb1b0c4cabe4556c7550bd6",
        1347,
    ),
    "logs/e1-shard0.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/e1-shard1.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/e1-shard2.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/e1-shard3.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/e1-shard4.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/e1-shard5.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/e1-shard6.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/e1-shard7.log": (
        "eea9c651b4777b8ad068a8530655385dd7c0bc4c450c3dd493d0ed8c8673a2a8",
        1521,
    ),
    "logs/evaluator-preflight.log": (
        "71ca30c56e0e8db4d7de2b59e419e24d7f0bdd9589f24b60e9bd27b8a7ef814c",
        1402,
    ),
    "logs/work-inspect.json": (
        "dc27d291061f4157ba73704e8323a55472b3484a0a60c123d13204565293734f",
        433,
    ),
    "logs/work-preflight.log": (
        "12d1b367af40a8135348230c04b2ec2d40f76950fd391c7b0d226bff185db875",
        818,
    ),
    "work-preflight.json": (
        "5c42928d3af37cce915ada37bd98b90b4da28e82869d6f383f4fa5621ca9c6c0",
        6294,
    ),
}
FAILED_SHARD_LOGS = tuple(f"logs/e1-shard{index}.log" for index in range(8))
FAILED_SHARD_FAILURE_MARKER = (
    "ValInferenceContractError: scripts.show_base.build_base_features was "
    "imported from another checkout"
)
FAILED_RUNNER_LOG = b"eight-shard live validation failed for e1\n"
FAILED_CONTROL_COMMIT = "58407ed3207fdd76dbf9a6e480de8af579bdb747"
FAILED_CONTROL_TREE = "b71f20161677fc68b4f9220a71d25e40632b17a0"
MINIMUM_RECOVERY_CONTROL_COMMIT = "8532272130e14098fdef3f54c7733d4ba6033f4c"
REJECTED_RECOVERY_CONTROL_COMMIT = "7d9967c9c5124d6f1ba041c2cb315dee88554a76"
REJECTED_RECOVERY_CONTROL_TREE = "a965cf1ccff214bd8922692603b33832b7357543"
FAILED_WRAPPER_PID = 261736
FAILED_CHILD_PID = 261737
FAILED_INCIDENT_ARTIFACTS = {
    "campaign": ("9f299db9dc874826808142bfec4a69e403102a90293a42e2aa6c9ad52e3fab0a", 33838),
    "job_claim": ("195994203d9f762b29c612c00aa1f185a9b407b1cfca687baa0fbc535d4c3b9b", 1231),
    "active_claim": ("0f4349f368b532334f6540252d2d7609d0f5ca13d7ed1d5307a74a5432f25f4a", 456),
    "authorization": ("c122131f1198a431e927fc45e6ea529c21fc6dbf49e0cdfe0f3ca166e3647207", 5193),
    "work_authority": ("31c2930acf1bc608687351132b0ce135ddddc7d8d75655857ca9334fb51142d2", 23578),
    "runner_status": ("b0c3f8a9a7ffd203621ac6d5bbcc29fdc2ff008ce9c6a2825b934139cee6fd2d", 1517),
    "runner_log": ("dd2a04e4989df737ca48fd55c3df02c027d0e5c9a2e3b2540d3908c864628ad8", 42),
    "candidate_receipt": ("16b2fe66a7f8ec0c38c7ab326b540a6a14043c74cd85fe013e7ca6cf10b24d8a", 2396),
    "consumer_claim": ("b8449bf107ca4c97a543676e8883ad4218231524241e3ff685fbcc1cbff940b3", 720),
}
GUARD_PASS_RE = re.compile(
    r"^PASS GPU0=PID([1-9][0-9]*) GPU1=PID([1-9][0-9]*) "
    r"GPU2=PID([1-9][0-9]*) GPU3=PID([1-9][0-9]*) "
    r"GPU4=PID([1-9][0-9]*) GPU5=PID([1-9][0-9]*) "
    r"GPU6=PID([1-9][0-9]*) GPU7=PID([1-9][0-9]*)\n$"
)

MANIFEST_FORMAT = "semtalk_show_base_official_adapt_long_manifest_v1"
FRESH_TRAJECTORY_MODE = "fresh_lineage_gate_v1"


class LiveConsumerError(RuntimeError):
    """The live authority cannot safely authorize this operation."""


class DuplicateConsumptionError(LiveConsumerError):
    """A different run root already owns the deterministic work claim."""


class PublicationLinkCountError(LiveConsumerError):
    """A create-new hardlink publication has not dropped its temp link yet."""


def _canonical_json(value: Any, *, newline: bool = False) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise LiveConsumerError("value is not strict JSON") from error
    if newline:
        text += "\n"
    return text.encode("utf-8")


def _payload_sha(value: Mapping[str, Any], key: str = "receipt_payload_sha256") -> str:
    body = dict(value)
    body.pop(key, None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _supervisor_payload_sha(value: Mapping[str, Any], key: str) -> str:
    """Reproduce the V14 supervisor's distinct self-hash ABI exactly."""

    body = dict(value)
    body.pop(key, None)
    try:
        encoded = (
            json.dumps(
                body,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise LiveConsumerError(
            "value is not strict supervisor canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _with_payload_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_payload_sha256"] = _payload_sha(result)
    return result


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LiveConsumerError(f"{label} must be an object")
    if set(value) != expected:
        raise LiveConsumerError(
            f"{label} keys changed; missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise LiveConsumerError(f"{label} is not a lowercase SHA-256")
    return value


def _oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_OID_RE.fullmatch(value) is None:
        raise LiveConsumerError(f"{label} is not a lowercase Git object ID")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LiveConsumerError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveConsumerError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise LiveConsumerError(f"{label} is out of range")
    return result


def _reject_path_text(value: Any, label: str) -> None:
    text = str(value).replace("\\", "/")
    for component in text.split("/"):
        lowered = component.casefold()
        if "test" in lowered or FORBIDDEN_SOURCE_RE.search(lowered):
            raise LiveConsumerError(f"{label} contains a forbidden component: {value}")


def _reject_absolute_paths(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_absolute_paths(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_paths(child, f"{label}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        _reject_path_text(value, label)


def _canonical_path(path_value: Any, label: str, *, must_exist: bool = True) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise LiveConsumerError(f"{label} must be an absolute normalized path")
    _reject_path_text(path, label)
    if must_exist:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise LiveConsumerError(f"{label} is absent: {path}") from error
        if resolved != path:
            raise LiveConsumerError(f"{label} contains a symlink: {path}")
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise LiveConsumerError(f"{label} contains a symlink: {path}")
    return path


def _safe_file(
    path_value: Any,
    label: str,
    *,
    expected_sha: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[Path, bytes, str]:
    path = _canonical_path(path_value, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LiveConsumerError(f"cannot open {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LiveConsumerError(f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise PublicationLinkCountError(
                f"{label} must be a single-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_mode, row.st_nlink, row.st_size,
            row.st_mtime_ns, row.st_ctime_ns,
        )
        current = os.stat(path, follow_symlinks=False)
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise LiveConsumerError(f"{label} changed while read")
        payload = b"".join(chunks)
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha is not None and digest != _sha(expected_sha, f"{label} SHA"):
            raise LiveConsumerError(f"{label} SHA mismatch")
        if expected_bytes is not None and len(payload) != _integer(
            expected_bytes, f"{label} bytes", minimum=1
        ):
            raise LiveConsumerError(f"{label} byte count mismatch")
        return path, payload, digest
    finally:
        os.close(descriptor)


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise LiveConsumerError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LiveConsumerError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except LiveConsumerError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LiveConsumerError(f"invalid strict JSON: {label}") from error
    if not isinstance(value, dict):
        raise LiveConsumerError(f"{label} must contain an object")
    return value


def _json_file(
    path_value: Any,
    expected_sha: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, payload, digest = _safe_file(
        path_value, label, expected_sha=expected_sha
    )
    value = _strict_json_bytes(payload, label)
    artifact = {"path": str(path), "sha256": digest, "bytes": len(payload)}
    claimed = value.get("receipt_payload_sha256")
    if claimed is not None:
        claimed = _sha(claimed, f"{label} payload SHA")
        if _payload_sha(value) != claimed:
            raise LiveConsumerError(f"{label} payload SHA mismatch")
        artifact["receipt_payload_sha256"] = claimed
    return artifact, value


def _artifact_core(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the producer reconciliation's path/SHA/byte artifact ABI."""

    return {
        "path": value["path"],
        "sha256": value["sha256"],
        "bytes": value["bytes"],
    }


def _core_artifact(
    value: Any,
    label: str,
    *,
    executable: bool = False,
) -> tuple[dict[str, Any], bytes]:
    """Read one exact path/SHA/byte artifact without extending its ABI."""

    item = _exact_keys(value, ARTIFACT_CORE_KEYS, label)
    path, payload, digest = _safe_file(
        item["path"],
        label,
        expected_sha=_sha(item["sha256"], f"{label} SHA"),
        expected_bytes=_integer(item["bytes"], f"{label} bytes", minimum=1),
    )
    if executable and path.stat().st_mode & 0o111 == 0:
        raise LiveConsumerError(f"{label} is not executable")
    artifact = {"path": str(path), "sha256": digest, "bytes": len(payload)}
    if not _strict_equal(artifact, item):
        raise LiveConsumerError(f"{label} artifact changed")
    return artifact, payload


def _self_hashed_document(
    artifact_value: Any,
    *,
    keys: frozenset[str],
    payload_key: str,
    label: str,
    supervisor_payload_abi: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, raw = _core_artifact(artifact_value, label)
    value = _strict_json_bytes(raw, label)
    _exact_keys(value, keys, label)
    claimed = _sha(value.get(payload_key), f"{label} payload SHA")
    observed = (
        _supervisor_payload_sha(value, payload_key)
        if supervisor_payload_abi
        else _payload_sha(value, payload_key)
    )
    if observed != claimed:
        raise LiveConsumerError(f"{label} payload SHA mismatch")
    return artifact, value


def _relative_inventory(root: Path) -> dict[str, Any]:
    """Recompute the one accepted failed-e1 tree without following links."""

    canonical_root = _canonical_path(root, "failed run root")
    if not canonical_root.is_dir():
        raise LiveConsumerError("failed run root must be a directory")
    directories = ["."]
    files: list[dict[str, Any]] = []
    stack = [canonical_root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda row: row.name)
        except OSError as error:
            raise LiveConsumerError("cannot inventory failed run root") from error
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                relative = entry_path.relative_to(canonical_root).as_posix()
            except ValueError as error:  # pragma: no cover - scandir invariant
                raise LiveConsumerError("failed run entry escaped root") from error
            observed = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode):
                raise LiveConsumerError("failed run inventory contains a symlink")
            if stat.S_ISDIR(observed.st_mode):
                directories.append(relative)
                stack.append(entry_path)
            elif stat.S_ISREG(observed.st_mode):
                if observed.st_nlink != 1:
                    raise LiveConsumerError(
                        "failed run inventory contains a multi-link file"
                    )
                _path, payload, digest = _safe_file(
                    entry_path, f"failed run file {relative}"
                )
                files.append(
                    {
                        "relative_path": relative,
                        "sha256": digest,
                        "bytes": len(payload),
                    }
                )
            else:
                raise LiveConsumerError(
                    "failed run inventory contains a special entry"
                )
    directories.sort()
    files.sort(key=lambda row: row["relative_path"])
    if tuple(directories) != tuple(sorted(FAILED_RUN_DIRECTORIES)):
        raise LiveConsumerError("failed run directory inventory changed")
    if tuple(row["relative_path"] for row in files) != tuple(
        sorted(FAILED_RUN_FILES)
    ):
        raise LiveConsumerError(
            "failed run contains a semantic or unknown output"
        )
    observed_pins = {
        row["relative_path"]: (row["sha256"], row["bytes"])
        for row in files
    }
    if observed_pins != FAILED_RUN_FILE_PINS:
        raise LiveConsumerError("failed run artifact SHA/byte pins changed")
    file_by_path = {row["relative_path"]: row for row in files}
    shard_shas = {file_by_path[path]["sha256"] for path in FAILED_SHARD_LOGS}
    if len(shard_shas) != 1:
        raise LiveConsumerError("failed shard logs are not byte-identical")
    for relative in FAILED_SHARD_LOGS:
        _path, payload, _digest = _safe_file(
            canonical_root / relative, f"failed shard log {relative}"
        )
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise LiveConsumerError("failed shard log is not UTF-8") from error
        if text.count(FAILED_SHARD_FAILURE_MARKER) != 1:
            raise LiveConsumerError("failed shard log error marker changed")
    return {
        "format": FAILED_RUN_INVENTORY_FORMAT,
        "root": str(canonical_root),
        "directories": directories,
        "files": files,
        "shard_log_sha256": next(iter(shard_shas)),
        "shard_failure_marker": FAILED_SHARD_FAILURE_MARKER,
        "semantic_outputs": [],
    }


def _artifact(
    value: Any,
    keys: frozenset[str],
    label: str,
    *,
    artifact_payload_key: str | None = None,
    document_payload_key: str | None = None,
    canonical_full_payload: bool = False,
) -> tuple[dict[str, Any], bytes]:
    item = _exact_keys(value, keys, label)
    path, payload, digest = _safe_file(
        item["path"],
        label,
        expected_sha=_sha(item["sha256"], f"{label} SHA"),
        expected_bytes=(item.get("bytes") if "bytes" in item else None),
    )
    if artifact_payload_key is not None:
        parsed = _strict_json_bytes(payload, label)
        artifact_claim = _sha(
            item[artifact_payload_key], f"{label} artifact payload SHA"
        )
        if document_payload_key is not None:
            document_claim = parsed.get(document_payload_key)
            valid = (
                document_claim == artifact_claim
                and _payload_sha(parsed, document_payload_key)
                == _sha(document_claim, f"{label} document payload SHA")
            )
        elif canonical_full_payload:
            valid = hashlib.sha256(_canonical_json(parsed)).hexdigest() == artifact_claim
        else:
            raise AssertionError(
                "artifact payload binding requires a document key or canonical mode"
            )
        if not valid:
            raise LiveConsumerError(f"{label} payload binding mismatch")
    return dict(item), payload


def _write_new_or_identical(path_value: Any, value: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    path = _canonical_path(path_value, "output", must_exist=False)
    parent = _canonical_path(path.parent, "output parent")
    if not parent.is_dir() or path.parent != parent:
        raise LiveConsumerError("output parent is unsafe")
    encoded = _canonical_json(dict(value), newline=True)
    if os.path.lexists(path):
        existing, payload, digest = _safe_published_file(path, "existing output")
        if payload != encoded:
            raise DuplicateConsumptionError(f"existing output differs: {path}")
        return {"path": str(existing), "sha256": digest, "bytes": len(payload)}, False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", suffix=f"-{uuid.uuid4().hex}", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing, payload, digest = _safe_published_file(path, "racing output")
            if payload != encoded:
                raise DuplicateConsumptionError(
                    f"racing output differs: {path}"
                ) from None
            return {"path": str(existing), "sha256": digest, "bytes": len(payload)}, False
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    final, payload, digest = _safe_file(path, "published output")
    return {"path": str(final), "sha256": digest, "bytes": len(payload)}, True


def _safe_published_file(
    path: Path,
    label: str,
) -> tuple[Path, bytes, str]:
    """Read a create-new publication after its temporary hardlink is gone."""

    for attempt in range(500):
        try:
            return _safe_file(path, label)
        except PublicationLinkCountError:
            try:
                observed = os.lstat(path)
            except OSError:
                raise
            if (
                not stat.S_ISREG(observed.st_mode)
                or stat.S_ISLNK(observed.st_mode)
                or attempt == 499
            ):
                raise
            time.sleep(0.001)
    raise AssertionError("unreachable")


def _git(root: Path, *arguments: str, allow_failure: bool = False) -> tuple[int, str]:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if process.returncode != 0 and not allow_failure:
        raise LiveConsumerError(
            f"Git source audit failed: {process.stderr.strip()}"
        )
    return process.returncode, process.stdout.strip()


def _validate_source(
    value: Any,
    *,
    expected_commit: str,
    expected_tree: str,
    role: str,
) -> Path:
    source = _exact_keys(value, SOURCE_KEYS, role)
    root = _canonical_path(source["source_root"], f"{role} root")
    if not root.is_dir():
        raise LiveConsumerError(f"{role} root is not a directory")
    remotes = _git(root, "remote")[1].splitlines()
    origin = _git(root, "remote", "get-url", "origin")[1]
    push = _git(root, "remote", "get-url", "--push", "origin")[1]
    commit = _git(root, "rev-parse", "HEAD")[1]
    tree = _git(root, "rev-parse", "HEAD^{tree}")[1]
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")[1]
    symbolic_rc, symbolic = _git(root, "symbolic-ref", "-q", "HEAD", allow_failure=True)
    heads = _git(root, "for-each-ref", "--format=%(refname)", "refs/heads")[1]
    expected = {
        "origin": EXPECTED_ORIGIN,
        "source_root": str(root),
        "commit": expected_commit,
        "tree": expected_tree,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
    }
    if (
        not _strict_equal(source, expected)
        or remotes != ["origin"]
        or origin != EXPECTED_ORIGIN
        or push != EXPECTED_ORIGIN
        or commit != expected_commit
        or tree != expected_tree
        or status
        or symbolic_rc == 0
        or symbolic
        or heads
    ):
        raise LiveConsumerError(
            f"{role} must be the exact clean detached branchless SemTalk source"
        )
    return root


def _validate_recovery_control_source(value: Any) -> dict[str, Any]:
    """Audit the new clean detached control checkout and this exact bridge."""

    source = _exact_keys(value, CONTROL_SOURCE_KEYS, "new control source")
    root = _canonical_path(source["root"], "new control source root")
    if not root.is_dir():
        raise LiveConsumerError("new control source root is not a directory")
    commit = _oid(source["commit"], "new control source commit")
    tree = _oid(source["tree"], "new control source tree")
    if source["origin"] != EXPECTED_ORIGIN:
        raise LiveConsumerError("new control source origin changed")
    ancestry_rc, _ancestry_output = _git(
        root,
        "merge-base",
        "--is-ancestor",
        MINIMUM_RECOVERY_CONTROL_COMMIT,
        commit,
        allow_failure=True,
    )
    if (
        (
            commit == REJECTED_RECOVERY_CONTROL_COMMIT
            and tree == REJECTED_RECOVERY_CONTROL_TREE
        )
        or ancestry_rc != 0
        or _git(root, "remote")[1].splitlines() != ["origin"]
        or _git(root, "remote", "get-url", "origin")[1] != EXPECTED_ORIGIN
        or _git(root, "remote", "get-url", "--push", "origin")[1]
        != EXPECTED_ORIGIN
        or _git(root, "rev-parse", "HEAD")[1] != commit
        or _git(root, "rev-parse", "HEAD^{tree}")[1] != tree
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")[1]
        or _git(root, "symbolic-ref", "-q", "HEAD", allow_failure=True)[0] == 0
        or _git(root, "for-each-ref", "--format=%(refname)", "refs/heads")[1]
    ):
        raise LiveConsumerError(
            "new control source must be exact clean detached branchless SemTalk"
        )
    expected_relatives = {
        "supervisor": "scripts/show_base/supervise_base_v14_live_validation.py",
        "launcher": "scripts/show_base/run_base_live_val_8shard.sh",
        "bridge": "scripts/show_base/base_live_val_consumer_bridge.py",
        "authority_adapter": "scripts/show_base/base_v14_live_validation_authority.py",
    }
    validated = dict(source)
    for role, relative in expected_relatives.items():
        artifact, payload = _core_artifact(source[role], f"new control {role}")
        expected_path = root / relative
        if Path(artifact["path"]) != expected_path:
            raise LiveConsumerError(f"new control {role} path changed")
        tracked = _git(root, "ls-files", "--error-unmatch", relative)[1]
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            capture_output=True,
            check=False,
        )
        if tracked != relative or committed.returncode != 0 or committed.stdout != payload:
            raise LiveConsumerError(f"new control {role} is not exact tracked content")
        validated[role] = artifact
    if Path(validated["bridge"]["path"]).resolve(strict=True) != Path(
        __file__
    ).resolve(strict=True):
        raise LiveConsumerError("recovery must run through the new pinned bridge")
    return validated


def _validate_evidence_source(value: Any) -> Path:
    return _validate_source(
        value,
        expected_commit=VALIDATION_EVIDENCE_SOURCE_COMMIT,
        expected_tree=VALIDATION_EVIDENCE_SOURCE_TREE,
        role="validation evidence source",
    )


def _validate_runtime_source(value: Any) -> Path:
    return _validate_source(
        value,
        expected_commit=RUNTIME_VALIDATION_SOURCE_COMMIT,
        expected_tree=RUNTIME_VALIDATION_SOURCE_TREE,
        role="runtime validation source",
    )


def _expected_runtime_validation_proof() -> dict[str, Any]:
    unchanged = {
        "scripts/show_base/run_base_val_inference.py": (
            RUNTIME_VALIDATION_INFERENCE_SHA256,
            RUNTIME_VALIDATION_INFERENCE_SHA256,
        ),
        "scripts/show_base/produce_base_val_measurement.py": (
            RUNTIME_VALIDATION_MEASUREMENT_SHA256,
            RUNTIME_VALIDATION_MEASUREMENT_SHA256,
        ),
        "scripts/show_base/select_base_official_adapt_long.py": (
            RUNTIME_VALIDATION_LONG_SELECTOR_SHA256,
            RUNTIME_VALIDATION_LONG_SELECTOR_SHA256,
        ),
        "scripts/show_base/base_long_val_contract.py": (
            VALIDATION_EVIDENCE_CONTRACT_SHA256,
            RUNTIME_VALIDATION_CONTRACT_SHA256,
        ),
        "scripts/show_base/select_base_official_adapt.py": (
            VALIDATION_EVIDENCE_SELECTOR_SHA256,
            RUNTIME_VALIDATION_SELECTOR_SHA256,
        ),
    }
    return {
        "format": "semtalk_show_base_runtime_validation_successor_proof_v1",
        "evidence_source": {
            "commit": VALIDATION_EVIDENCE_SOURCE_COMMIT,
            "tree": VALIDATION_EVIDENCE_SOURCE_TREE,
        },
        "runtime_source": {
            "commit": RUNTIME_VALIDATION_SOURCE_COMMIT,
            "tree": RUNTIME_VALIDATION_SOURCE_TREE,
        },
        "ancestry_verified": True,
        "file_projection": {
            relative: {
                "evidence_sha256": evidence_sha,
                "runtime_sha256": runtime_sha,
            }
            for relative, (evidence_sha, runtime_sha) in unchanged.items()
        },
    }


def _validate_runtime_validation_roles(authority: Mapping[str, Any]) -> tuple[Path, Path]:
    evidence_root = _validate_evidence_source(
        authority["validation_evidence_source"]
    )
    runtime_root = _validate_runtime_source(
        authority["runtime_validation_source"]
    )
    if evidence_root == runtime_root:
        raise LiveConsumerError(
            "validation evidence and runtime source roots must be distinct"
        )
    if not _strict_equal(
        authority["runtime_validation_proof"],
        _expected_runtime_validation_proof(),
    ):
        raise LiveConsumerError("runtime validation successor proof changed")
    ancestry_rc, _ancestry_output = _git(
        runtime_root,
        "merge-base",
        "--is-ancestor",
        VALIDATION_EVIDENCE_SOURCE_COMMIT,
        RUNTIME_VALIDATION_SOURCE_COMMIT,
        allow_failure=True,
    )
    if ancestry_rc != 0:
        raise LiveConsumerError(
            "runtime validation no longer descends from evidence source"
        )
    return evidence_root, runtime_root


def _runtime_execution_pipeline(
    authority: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Project validated evidence paths onto the pinned runtime checkout.

    The immutable pipeline receipt is intentionally authored by, and
    revalidated against, the 4066 evidence source.  Runtime imports are
    intentionally bound to its 70a provenance-fix successor.  Passing the
    evidence receipt's absolute paths to the 70a engine therefore creates a
    false mixed-checkout failure even when an unchanged tracked file has the
    same bytes.  Build a non-persisted execution view only after replaying the
    complete tracked source closure in both roots.
    """

    evidence_root, runtime_root = _validate_runtime_validation_roles(authority)
    contract = modules.get("contract")
    legacy = modules.get("legacy")
    validate_pipeline = getattr(contract, "validate_pipeline", None)
    build_runtime_source = getattr(
        legacy, "build_fresh_pipeline_source_receipt", None
    )
    runtime_source_files = getattr(
        legacy, "DIFFSHEG_PRIMARY_PIPELINE_SOURCE_FILES", None
    )
    if (
        not callable(validate_pipeline)
        or not callable(build_runtime_source)
        or tuple(runtime_source_files or ()) != RUNTIME_EXECUTION_SOURCE_FILES
    ):
        raise LiveConsumerError(
            "runtime validation source-closure implementation changed"
        )

    pipeline_reference = authority["pipeline_receipt"]
    try:
        pipeline_artifact, evidence_pipeline = validate_pipeline(
            Path(pipeline_reference["path"]),
            pipeline_reference["sha256"],
            expected_source=authority["validation_evidence_source"],
        )
    except Exception as error:
        raise LiveConsumerError(
            "cannot replay the immutable evidence validation pipeline"
        ) from error
    if (
        not isinstance(evidence_pipeline, dict)
        or not _strict_equal(pipeline_artifact, pipeline_reference)
        or not _strict_equal(
            evidence_pipeline.get("source"),
            authority["validation_evidence_source"],
        )
    ):
        raise LiveConsumerError(
            "immutable evidence validation pipeline changed during replay"
        )
    evidence_snapshot = copy.deepcopy(evidence_pipeline)
    evidence_source = _exact_keys(
        evidence_pipeline.get("source"),
        SOURCE_KEYS,
        "evidence pipeline source",
    )
    evidence_files = evidence_pipeline.get("source_closure")
    if (
        not isinstance(evidence_files, dict)
        or set(evidence_files) != set(RUNTIME_EXECUTION_SOURCE_FILES)
    ):
        raise LiveConsumerError(
            "evidence pipeline source closure is not the exact pinned 29 files"
        )
    if evidence_source["source_root"] != str(evidence_root):
        raise LiveConsumerError("evidence pipeline source root changed")

    try:
        runtime_source_receipt = build_runtime_source(runtime_root)
    except Exception as error:
        raise LiveConsumerError(
            "cannot build the pinned runtime source closure"
        ) from error
    runtime_source_receipt = _exact_keys(
        runtime_source_receipt,
        SOURCE_RECEIPT_KEYS,
        "runtime execution source receipt",
    )
    runtime_source = {
        key: runtime_source_receipt[key] for key in SOURCE_KEYS
    }
    if not _strict_equal(runtime_source, authority["runtime_validation_source"]):
        raise LiveConsumerError(
            "runtime execution source differs from its clean pinned authority"
        )
    runtime_files = runtime_source_receipt["files"]
    if (
        not isinstance(runtime_files, dict)
        or set(runtime_files) != set(RUNTIME_EXECUTION_SOURCE_FILES)
    ):
        raise LiveConsumerError(
            "runtime execution source closure is not the exact pinned 29 files"
        )

    metadata_differences: set[str] = set()
    for relative in RUNTIME_EXECUTION_SOURCE_FILES:
        evidence_entry = _exact_keys(
            evidence_files[relative],
            SOURCE_FILE_KEYS,
            f"evidence source closure {relative}",
        )
        runtime_entry = _exact_keys(
            runtime_files[relative],
            SOURCE_FILE_KEYS,
            f"runtime source closure {relative}",
        )
        if (
            Path(str(evidence_entry["path"])) != evidence_root / relative
            or Path(str(runtime_entry["path"])) != runtime_root / relative
        ):
            raise LiveConsumerError(
                f"source closure path projection escaped its checkout: {relative}"
            )
        for label, entry in (
            ("evidence", evidence_entry),
            ("runtime", runtime_entry),
        ):
            _sha(entry["sha256"], f"{label} source closure {relative} SHA")
            _integer(
                entry["bytes"],
                f"{label} source closure {relative} bytes",
                minimum=0,
            )
            if entry["git_mode"] not in {"100644", "100755"}:
                raise LiveConsumerError(
                    f"{label} source closure {relative} Git mode changed"
                )
            _oid(
                entry["git_blob_sha1"],
                f"{label} source closure {relative} Git blob",
            )
        evidence_metadata = {
            key: evidence_entry[key] for key in SOURCE_FILE_KEYS if key != "path"
        }
        runtime_metadata = {
            key: runtime_entry[key] for key in SOURCE_FILE_KEYS if key != "path"
        }
        if not _strict_equal(evidence_metadata, runtime_metadata):
            metadata_differences.add(relative)
        if relative in RUNTIME_EXECUTION_CHANGED_FILES:
            expected_evidence_sha, expected_runtime_sha = (
                RUNTIME_EXECUTION_CHANGED_SHA256[relative]
            )
            if (
                evidence_entry["sha256"] != expected_evidence_sha
                or runtime_entry["sha256"] != expected_runtime_sha
                or evidence_entry["git_mode"] != runtime_entry["git_mode"]
            ):
                raise LiveConsumerError(
                    f"pinned runtime successor projection changed: {relative}"
                )
        elif not _strict_equal(evidence_metadata, runtime_metadata):
            raise LiveConsumerError(
                f"unchanged source differs across validation roots: {relative}"
            )
    if metadata_differences != set(RUNTIME_EXECUTION_CHANGED_FILES):
        raise LiveConsumerError(
            "runtime successor must differ at exactly the two pinned proof files"
        )

    helper_relative = "scripts/show_base/semtalk_base_inference_core.py"
    engine_relative = "scripts/show_base/run_base_val_inference.py"
    if (
        not _strict_equal(
            evidence_pipeline.get("inference_helper"),
            evidence_files[helper_relative],
        )
        or not _strict_equal(
            evidence_pipeline.get("inference_entrypoint"),
            evidence_files[engine_relative],
        )
    ):
        raise LiveConsumerError(
            "evidence pipeline entrypoints differ from its source closure"
        )

    runtime_pipeline = copy.deepcopy(evidence_pipeline)
    runtime_pipeline["source"] = copy.deepcopy(runtime_source)
    runtime_pipeline["source_closure"] = copy.deepcopy(runtime_files)
    runtime_pipeline["inference_helper"] = copy.deepcopy(
        runtime_files[helper_relative]
    )
    runtime_pipeline["inference_entrypoint"] = copy.deepcopy(
        runtime_files[engine_relative]
    )
    runtime_pipeline["receipt_payload_sha256"] = _payload_sha(runtime_pipeline)
    if not _strict_equal(evidence_pipeline, evidence_snapshot):
        raise LiveConsumerError(
            "runtime execution projection mutated immutable evidence"
        )
    return evidence_pipeline, runtime_pipeline, runtime_root


def _validate_execution_contract(value: Any, epoch: int) -> dict[str, Any]:
    execution = _exact_keys(value, EXECUTION_KEYS, "execution contract")
    expected = {
        "purpose": "single_candidate_diffsheg_validation_only",
        "split": "val",
        "test_visible": False,
        "expected_shards": 8,
        "candidate_epochs_in_work_item": [epoch],
        "may_publish_standard_22_candidate_measurement_before_e400": False,
        "may_influence_training": False,
        "requires_guarded_runner": True,
    }
    if not _strict_equal(execution, expected):
        raise LiveConsumerError("live execution contract changed")
    return execution


def _project_modules_are_from(root: Path) -> None:
    prefixes = ("scripts", "models", "dataloaders", "utils")
    for name, module in tuple(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        file_value = getattr(module, "__file__", None)
        if file_value is None:
            paths = getattr(module, "__path__", ())
            if not paths:
                raise LiveConsumerError(f"unrooted project module is loaded: {name}")
            values = [Path(str(path)).resolve(strict=True) for path in paths]
        else:
            values = [Path(str(file_value)).resolve(strict=True)]
        for path in values:
            try:
                path.relative_to(root)
            except ValueError as error:
                raise LiveConsumerError(
                    f"project module escaped validation source: {name}"
                ) from error


def _bind_exact_scripts_namespace(source_root: Path) -> None:
    """Bind SemTalk's namespace-package ``scripts`` to one checkout only.

    The repository deliberately has no ``scripts/__init__.py``.  Letting the
    default namespace-package finder construct ``scripts`` can therefore merge
    the pinned runtime checkout with the launcher's checkout or an unrelated
    site-packages directory.  Install a one-path namespace before importing
    any runtime module so the post-import provenance audit remains exact.
    """

    scripts_root = _canonical_path(
        source_root / "scripts", "runtime validation scripts namespace"
    )
    if not scripts_root.is_dir():
        raise LiveConsumerError(
            "runtime validation scripts namespace is not a directory"
        )
    existing = sys.modules.get("scripts")
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        module_paths = list(getattr(existing, "__path__", []) or [])
        if module_file is not None or module_paths != [str(scripts_root)]:
            raise LiveConsumerError(
                "preloaded scripts namespace is not exactly runtime-bound"
            )
        return
    namespace = ModuleType("scripts")
    namespace.__file__ = None
    namespace.__package__ = "scripts"
    namespace.__path__ = [str(scripts_root)]
    specification = importlib.util.spec_from_loader(
        "scripts", loader=None, is_package=True
    )
    if specification is None:
        raise LiveConsumerError(
            "cannot construct runtime validation scripts namespace"
        )
    specification.submodule_search_locations = [str(scripts_root)]
    namespace.__spec__ = specification
    sys.modules["scripts"] = namespace


def _load_validation_modules(source_root: Path) -> dict[str, ModuleType]:
    _project_modules_are_from(source_root)
    root_text = str(source_root)
    sys.path = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    try:
        _bind_exact_scripts_namespace(source_root)
        modules = {
            "contract": importlib.import_module(
                "scripts.show_base.base_long_val_contract"
            ),
            "engine": importlib.import_module(
                "scripts.show_base.run_base_val_inference"
            ),
            "producer": importlib.import_module(
                "scripts.show_base.produce_base_val_measurement"
            ),
            "selector": importlib.import_module(
                "scripts.show_base.select_base_official_adapt_long"
            ),
            "legacy": importlib.import_module(
                "scripts.show_base.select_base_official_adapt"
            ),
        }
        _project_modules_are_from(source_root)
        _validate_runtime_source(
            {
                "origin": EXPECTED_ORIGIN,
                "source_root": str(source_root),
                "commit": RUNTIME_VALIDATION_SOURCE_COMMIT,
                "tree": RUNTIME_VALIDATION_SOURCE_TREE,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            }
        )
        contract_path = source_root / "scripts/show_base/base_long_val_contract.py"
        selector_path = source_root / "scripts/show_base/select_base_official_adapt.py"
        if (
            _safe_file(contract_path, "runtime validation contract")[2]
            != RUNTIME_VALIDATION_CONTRACT_SHA256
            or _safe_file(selector_path, "runtime validation selector")[2]
            != RUNTIME_VALIDATION_SELECTOR_SHA256
        ):
            raise LiveConsumerError("runtime validation entrypoint bytes changed")
        pins = getattr(modules["legacy"], "DIFFSHEG_PINNED_RECEIPT", None)
        fgd_specification = (
            pins.get("autoencoders", {}).get("fgd")
            if isinstance(pins, dict)
            else None
        )
        if not _strict_equal(
            fgd_specification, EXPECTED_DIFFSHEG_FGD_PROVENANCE
        ):
            raise LiveConsumerError(
                "runtime validation lacks exact official DiffSHEG provenance"
            )
        if not sys.path or sys.path[0] != root_text:
            raise LiveConsumerError(
                "runtime validation source lost import-path precedence"
            )
        return modules
    except BaseException:
        if sys.path and sys.path[0] == root_text:
            sys.path.pop(0)
        raise


def _validate_work_authority(
    path: Path,
    expected_sha: str,
    *,
    load_modules: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, ModuleType] | None]:
    authority_artifact, authority = _json_file(
        path, _sha(expected_sha, "work authority external SHA"), "work authority"
    )
    _exact_keys(authority, WORK_KEYS, "work authority")
    epoch = _integer(authority["candidate_epoch"], "candidate epoch", minimum=1)
    if epoch not in CANDIDATE_EPOCHS:
        raise LiveConsumerError("candidate epoch is outside the formal 22")
    if (
        authority["format"] != WORK_FORMAT
        or authority["status"] != "ready"
        or authority["split"] != "val"
        or authority["test_visible"] is not False
        or authority["selection_eligible"] is not False
        or not _strict_equal(authority["candidate_epochs"], list(CANDIDATE_EPOCHS))
        or _payload_sha(authority) != authority["receipt_payload_sha256"]
    ):
        raise LiveConsumerError("work authority is not strict val-only live work")
    _validate_execution_contract(authority["execution_contract"], epoch)
    topology = authority["selected_topology"]
    if not isinstance(topology, dict):
        raise LiveConsumerError("selected topology is absent")
    updates = _integer(topology.get("updates_per_epoch"), "updates per epoch", minimum=1)
    if _integer(authority["optimizer_updates"], "optimizer updates") != epoch * updates:
        raise LiveConsumerError("optimizer update count does not bind the epoch")
    _finite(authority["published_unix"], "authority publication time", positive=True)
    _reject_absolute_paths(authority, "work authority")

    ready, _ = _artifact(
        authority["producer_ready_receipt"],
        frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
        "producer ready receipt",
        artifact_payload_key="receipt_payload_sha256",
        document_payload_key="receipt_payload_sha256",
    )
    snapshot, snapshot_bytes = _artifact(
        authority["producer_manifest_snapshot"],
        frozenset({"path", "sha256", "bytes", "entries_sha256", "entry_count"}),
        "producer manifest snapshot",
    )
    checkpoint, _ = _artifact(
        authority["candidate_checkpoint"],
        frozenset(
            {
                "path", "relative_path", "sha256", "bytes", "model_state_tensors",
                "model_state_schema_sha256", "model_state_semantic_sha256",
            }
        ),
        "candidate checkpoint",
    )
    frozen, _ = _artifact(
        authority["frozen_inputs"],
        frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
        "frozen inputs",
        artifact_payload_key="receipt_payload_sha256",
        document_payload_key="receipt_sha256",
    )
    schedule, _ = _artifact(
        authority["schedule"],
        frozenset({"path", "sha256", "bytes", "payload_sha256"}),
        "schedule",
        artifact_payload_key="payload_sha256",
        canonical_full_payload=True,
    )
    for digest_key in ("entries_sha256",):
        _sha(snapshot[digest_key], f"snapshot {digest_key}")
    for digest_key in (
        "sha256", "model_state_schema_sha256", "model_state_semantic_sha256"
    ):
        _sha(checkpoint[digest_key], f"checkpoint {digest_key}")
    _integer(checkpoint["model_state_tensors"], "model state tensors", minimum=1)
    snapshot_payload = _exact_keys(
        _strict_json_bytes(snapshot_bytes, "producer manifest snapshot"),
        SNAPSHOT_DOCUMENT_KEYS,
        "producer manifest snapshot",
    )
    entry = _exact_keys(
        authority["producer_manifest_entry"],
        MANIFEST_ENTRY_KEYS,
        "producer manifest entry",
    )
    entries = snapshot_payload["entries"]
    prefix_length = CANDIDATE_EPOCHS.index(epoch) + 1
    snapshot_entry_count = _integer(
        snapshot["entry_count"], "snapshot entry count", minimum=1
    )
    if (
        not isinstance(entries, list)
        or len(entries) != prefix_length
        or len(entries) != snapshot_entry_count
        or snapshot_payload["format"] != MANIFEST_FORMAT
        or snapshot_payload["status"] != "running"
        or not _strict_equal(
            snapshot_payload["candidate_epochs"], list(CANDIDATE_EPOCHS)
        )
        or snapshot_payload["trajectory_mode"] != FRESH_TRAJECTORY_MODE
        or snapshot_payload["trajectory_probe_verified"] is not True
        or any(
            not isinstance(candidate_entry, dict)
            or set(candidate_entry) != MANIFEST_ENTRY_KEYS
            or type(candidate_entry.get("epoch")) is not int
            or candidate_entry["epoch"] != expected_epoch
            for candidate_entry, expected_epoch in zip(
                entries, CANDIDATE_EPOCHS[:prefix_length]
            )
        )
        or hashlib.sha256(_canonical_json(entries)).hexdigest()
        != snapshot["entries_sha256"]
        or snapshot_payload["entries_sha256"] != snapshot["entries_sha256"]
        or not _strict_equal(entries[-1], entry)
        or _integer(entry["epoch"], "manifest entry epoch", minimum=1) != epoch
        or _integer(
            entry["optimizer_updates"], "manifest entry optimizer updates"
        )
        != authority["optimizer_updates"]
        or entry["checkpoint_sha256"] != checkpoint["sha256"]
        or _integer(
            entry["checkpoint_bytes"], "manifest entry checkpoint bytes", minimum=1
        )
        != checkpoint["bytes"]
        or entry["checkpoint"] != checkpoint["relative_path"]
        or not _strict_equal(
            entry["checkpoint_container_schema"], ["audit", "model_state"]
        )
        or _integer(
            entry["model_state_tensors"], "manifest entry model tensors", minimum=1
        )
        != checkpoint["model_state_tensors"]
        or entry["model_state_schema_sha256"]
        != checkpoint["model_state_schema_sha256"]
        or entry["model_state_semantic_sha256"]
        != checkpoint["model_state_semantic_sha256"]
        or entry["all_model_state_tensors_finite"] is not True
        or entry["trajectory_anchor_match"] is not None
        or entry["trajectory_probe_verified"] is not True
    ):
        raise LiveConsumerError("manifest snapshot does not bind the candidate")
    if Path(checkpoint["path"]).name != Path(checkpoint["relative_path"]).name:
        raise LiveConsumerError("candidate relative path changed")

    for value, keys, label in (
        (
            authority["val_inputs_receipt"],
            frozenset({"path", "sha256", "receipt_payload_sha256"}),
            "validation inputs receipt",
        ),
        (
            authority["pipeline_receipt"],
            frozenset({"path", "sha256", "receipt_payload_sha256"}),
            "validation pipeline receipt",
        ),
    ):
        _artifact(
            value,
            keys,
            label,
            artifact_payload_key="receipt_payload_sha256",
            document_payload_key="receipt_payload_sha256",
        )
    coverage = _exact_keys(authority["coverage"], COVERAGE_KEYS, "val coverage")
    if (
        coverage["split"] != "val"
        or _integer(coverage["clip_count"], "val clips", minimum=1) != 1715
        or any(
            _integer(coverage[key], f"coverage {key}", minimum=(0 if key == "uncovered_tail_frames" else 1)) < 0
            for key in ("frame_count", "window_count", "uncovered_tail_frames")
        )
    ):
        raise LiveConsumerError("validation coverage changed")
    _sha(coverage["clip_ids_sha256"], "clip IDs SHA")
    _sha(coverage["diffsheg_clip_manifest_sha256"], "DiffSHEG manifest SHA")

    if not _strict_equal(
        authority["pipeline_source"],
        authority["validation_evidence_source"],
    ):
        raise LiveConsumerError("pipeline and validation evidence source differ")
    evidence_root, runtime_root = _validate_runtime_validation_roles(authority)
    entrypoint = _exact_keys(
        authority["inference_entrypoint"],
        frozenset({"path", "sha256", "bytes", "git_mode", "git_blob_sha1"}),
        "inference entrypoint",
    )
    entry_path, entry_bytes, _entry_sha = _safe_file(
        entrypoint["path"],
        "inference entrypoint",
        expected_sha=entrypoint["sha256"],
        expected_bytes=entrypoint["bytes"],
    )
    try:
        relative = entry_path.relative_to(evidence_root)
    except ValueError as error:
        raise LiveConsumerError(
            "inference entrypoint escaped validation evidence source"
        ) from error
    if entrypoint["git_mode"] not in {"100644", "100755"}:
        raise LiveConsumerError("inference entrypoint Git mode changed")
    blob = hashlib.sha1(f"blob {len(entry_bytes)}\0".encode() + entry_bytes).hexdigest()
    if blob != entrypoint["git_blob_sha1"]:
        raise LiveConsumerError("inference entrypoint Git blob changed")
    committed = subprocess.run(
        [
            "git",
            "-C",
            str(evidence_root),
            "show",
            f"{VALIDATION_EVIDENCE_SOURCE_COMMIT}:{relative.as_posix()}",
        ],
        capture_output=True,
        check=False,
    )
    if committed.returncode != 0 or committed.stdout != entry_bytes:
        raise LiveConsumerError("inference entrypoint differs from 4066f20")
    runtime_committed = subprocess.run(
        [
            "git",
            "-C",
            str(runtime_root),
            "show",
            f"{RUNTIME_VALIDATION_SOURCE_COMMIT}:{relative.as_posix()}",
        ],
        capture_output=True,
        check=False,
    )
    if runtime_committed.returncode != 0 or runtime_committed.stdout != entry_bytes:
        raise LiveConsumerError(
            "runtime inference entrypoint differs from pipeline evidence"
        )

    modules: dict[str, ModuleType] | None = None
    if load_modules:
        modules = _load_validation_modules(runtime_root)
        contract = modules["contract"]
        val_artifact, full_coverage = contract.validate_val_inputs(
            Path(authority["val_inputs_receipt"]["path"]),
            authority["val_inputs_receipt"]["sha256"],
        )
        pipeline_artifact, pipeline = contract.validate_pipeline(
            Path(authority["pipeline_receipt"]["path"]),
            authority["pipeline_receipt"]["sha256"],
            expected_source=authority["validation_evidence_source"],
        )
        if (
            not _strict_equal(val_artifact, authority["val_inputs_receipt"])
            or not _strict_equal(
                pipeline_artifact, authority["pipeline_receipt"]
            )
            or not _strict_equal(contract.public_val_coverage(full_coverage), coverage)
            or not _strict_equal(
                pipeline.get("source"),
                authority["validation_evidence_source"],
            )
            or not _strict_equal(pipeline.get("inference_entrypoint"), entrypoint)
            or any(
                pipeline.get("fixed_checkpoints", {}).get(stage, {}).get("sha256")
                != authority["selected_prerequisite_sha256"].get(stage)
                for stage in ("face", "hands", "upper", "lower", "global")
            )
        ):
            raise LiveConsumerError("authority differs from fresh validation replay")
    return authority_artifact, authority, modules


def _training_root_for_candidate(
    *,
    producer_ready_receipt: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    epoch: int,
) -> Path:
    """Return the immutable training root anchoring one producer candidate.

    The work-authority file path is not part of its signed JSON body.  A
    byte-identical copy may therefore live at another path.  Deriving the
    claim namespace from that copy would give the same producer candidate a
    second GPU allowance.  The producer-ready receipt *is* embedded in the
    authority and is SHA-verified before this helper is reached, so anchor the
    claim next to its immutable training-run inventory instead.
    """

    ready = _exact_keys(
        producer_ready_receipt,
        frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
        "producer ready receipt for consumer claim",
    )
    checkpoint = _exact_keys(
        candidate_checkpoint,
        frozenset(
            {
                "path", "relative_path", "sha256", "bytes",
                "model_state_tensors", "model_state_schema_sha256",
                "model_state_semantic_sha256",
            }
        ),
        "candidate checkpoint for consumer claim",
    )
    ready_path = _canonical_path(
        ready["path"], "producer ready receipt for consumer claim"
    )
    checkpoint_path = _canonical_path(
        checkpoint["path"], "candidate checkpoint for consumer claim"
    )
    expected_name = f"epoch-{epoch:04d}.json"
    if checkpoint_path.parent.name != "candidates":
        raise LiveConsumerError(
            "candidate checkpoint is outside its fixed producer inventory"
        )
    training_root = checkpoint_path.parent.parent
    expected_ready_path = training_root / "candidate_receipts" / expected_name
    if (
        ready_path != expected_ready_path
        or ready_path.name != expected_name
    ):
        raise LiveConsumerError(
            "producer ready receipt is outside its fixed candidate inventory"
        )
    return training_root


def _claim_slot(
    training_root: Path,
    *,
    directory_name: str,
    epoch: int,
    create_directory: bool,
) -> Path:
    claim_dir = training_root / directory_name
    created = False
    if os.path.lexists(claim_dir):
        resolved = _canonical_path(claim_dir, f"{directory_name} directory")
        if not resolved.is_dir():
            raise LiveConsumerError(f"{directory_name} path is not a directory")
    elif create_directory:
        try:
            claim_dir.mkdir(mode=0o755)
            created = True
        except FileExistsError:
            pass
        resolved = _canonical_path(claim_dir, f"{directory_name} directory")
        if not resolved.is_dir():
            raise LiveConsumerError(f"{directory_name} directory race was unsafe")
    else:
        resolved = claim_dir
    if created:
        parent_fd = os.open(claim_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    return resolved / f"epoch-{epoch:04d}.json"


def _claim_path(
    *,
    producer_ready_receipt: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    epoch: int,
) -> Path:
    """Return/create the original one-shot consumer claim slot."""

    training_root = _training_root_for_candidate(
        producer_ready_receipt=producer_ready_receipt,
        candidate_checkpoint=candidate_checkpoint,
        epoch=epoch,
    )
    return _claim_slot(
        training_root,
        directory_name="live_val_consumer_claims",
        epoch=epoch,
        create_directory=True,
    )


def _recovery_claim_path(
    authority: Mapping[str, Any], *, create_directory: bool
) -> Path:
    epoch = _integer(
        authority.get("candidate_epoch"), "recovery candidate epoch", minimum=1
    )
    training_root = _training_root_for_candidate(
        producer_ready_receipt=authority["producer_ready_receipt"],
        candidate_checkpoint=authority["candidate_checkpoint"],
        epoch=epoch,
    )
    return _claim_slot(
        training_root,
        directory_name="live_val_consumer_recovery_claims",
        epoch=epoch,
        create_directory=create_directory,
    )


def _campaign_document(
    artifact_value: Any, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _self_hashed_document(
        artifact_value,
        keys=FAILED_CAMPAIGN_KEYS,
        payload_key="campaign_payload_sha256",
        label=label,
        supervisor_payload_abi=True,
    )
    if (
        value["format"]
        != "semtalk_show_base_v14_live_validation_campaign_v3"
        or value["status"] != "frozen_before_execution"
        or value["split"] != "val"
        or value["test_visible"] is not False
        or type(value["test_measurements_authorized"]) is not int
        or value["test_measurements_authorized"] != 0
        or value["candidate_epochs"] != list(CANDIDATE_EPOCHS)
    ):
        raise LiveConsumerError(f"{label} is not the frozen val-only campaign")
    state_root = _canonical_path(value["state_root"], f"{label} state root")
    if not state_root.is_dir():
        raise LiveConsumerError(f"{label} state root is not a directory")
    if Path(artifact["path"]) != state_root / "campaign.json":
        raise LiveConsumerError(f"{label} path is outside its state root")
    jobs = value["jobs"]
    if not isinstance(jobs, list) or len(jobs) != len(CANDIDATE_EPOCHS):
        raise LiveConsumerError(f"{label} jobs changed")
    for epoch, job_value in zip(CANDIDATE_EPOCHS, jobs):
        job = _exact_keys(job_value, CAMPAIGN_JOB_KEYS, f"{label} e{epoch} job")
        if type(job["epoch"]) is not int or job["epoch"] != epoch:
            raise LiveConsumerError(f"{label} job order changed")
    return artifact, value


def _campaign_job(campaign: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    index = CANDIDATE_EPOCHS.index(epoch)
    return dict(campaign["jobs"][index])


def _argv_option(argv: Any, option: str, label: str) -> str:
    if (
        not isinstance(argv, list)
        or not all(isinstance(token, str) and token and "\0" not in token for token in argv)
        or argv.count(option) != 1
    ):
        raise LiveConsumerError(f"{label} changed")
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise LiveConsumerError(f"{label} is missing a value")
    return argv[index + 1]


def _artifact_reference(
    artifact: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(artifact)
    result["receipt_payload_sha256"] = value["receipt_payload_sha256"]
    return result


def _require_failed_incident_artifact(
    artifact: Mapping[str, Any], role: str
) -> None:
    expected_sha, expected_bytes = FAILED_INCIDENT_ARTIFACTS[role]
    if (
        artifact.get("sha256") != expected_sha
        or type(artifact.get("bytes")) is not int
        or artifact["bytes"] != expected_bytes
    ):
        raise LiveConsumerError(f"failed incident {role} is not the pinned artifact")


def _validate_guard_proof(
    value: Any,
    *,
    failed_campaign: Mapping[str, Any],
    failed_status: Mapping[str, Any],
) -> dict[str, Any]:
    proof = _exact_keys(value, GUARD_PROOF_KEYS, "guard proof")
    verifier, _verifier_payload = _core_artifact(
        proof["verifier"], "guard verifier"
    )
    if not _strict_equal(verifier, failed_campaign["guard_verifier"]):
        raise LiveConsumerError("guard proof verifier differs from failed campaign")
    restored = proof["restored_guards"]
    expected_keys = {str(index) for index in range(8)}
    if (
        not isinstance(restored, dict)
        or set(restored) != expected_keys
        or any(
            type(restored[str(index)]) is not int
            or restored[str(index)] <= 1
            for index in range(8)
        )
        or len(set(restored.values())) != 8
        or not _strict_equal(restored, failed_status["restored_guards"])
    ):
        raise LiveConsumerError("guard proof PID set changed")
    formal = failed_campaign["formal_python"]
    if not isinstance(formal, dict) or not isinstance(formal.get("argv0"), str):
        raise LiveConsumerError("failed campaign formal Python changed")
    expected_argv = [
        formal["argv0"], verifier["path"],
        *[str(restored[str(index)]) for index in range(8)],
    ]
    if proof["argv"] != expected_argv:
        raise LiveConsumerError("guard proof argv changed")
    match = GUARD_PASS_RE.fullmatch(proof["stdout"])
    if (
        match is None
        or {str(index): int(match.group(index + 1)) for index in range(8)}
        != restored
    ):
        raise LiveConsumerError("guard proof stdout changed")
    _finite(proof["verified_unix"], "guard proof time", positive=True)
    result = dict(proof)
    result["verifier"] = verifier
    return result


def _validate_failed_process_proof(
    value: Any, failed_status: Mapping[str, Any]
) -> dict[str, Any]:
    proof = _exact_keys(
        value, FAILED_PROCESS_PROOF_KEYS, "failed process proof"
    )
    wrapper_pid = _integer(
        proof["wrapper_pid"], "failed wrapper PID", minimum=2
    )
    child_pid = _integer(proof["child_pid"], "failed child PID", minimum=2)
    if (
        wrapper_pid != failed_status["wrapper_pid"]
        or child_pid != failed_status["child_pid"]
        or wrapper_pid == child_pid
        or proof["wrapper_proc_state"] != "absent"
        or proof["child_proc_state"] != "absent"
        or not _strict_equal(proof["runner_command"], failed_status["command"])
    ):
        raise LiveConsumerError("failed process proof changed")
    _finite(proof["checked_unix"], "failed process proof time", positive=True)
    for pid in (wrapper_pid, child_pid):
        if os.path.lexists(Path("/proc") / str(pid)):
            raise LiveConsumerError(f"failed process PID {pid} is no longer absent")
    return dict(proof)


def _validate_recovery_work_authority_replay(
    request: Mapping[str, Any],
    failed_work_artifact: Mapping[str, Any],
    failed_work: Mapping[str, Any],
    new_work_artifact: Mapping[str, Any],
    new_work: Mapping[str, Any],
) -> None:
    """Bind distinct publications to one exact work-authority semantics."""

    failed_work_semantics = dict(failed_work)
    new_work_semantics = dict(new_work)
    for projection in (failed_work_semantics, new_work_semantics):
        projection.pop("published_unix", None)
        projection.pop("receipt_payload_sha256", None)
    if (
        not _strict_equal(
            _artifact_core(failed_work_artifact),
            request["failed_work_authority"],
        )
        or not _strict_equal(
            _artifact_core(new_work_artifact),
            request["new_work_authority"],
        )
        or not _strict_equal(failed_work_semantics, new_work_semantics)
    ):
        raise LiveConsumerError(
            "new work authority does not semantically replay failed work"
        )


def _validate_recovery_request(
    request_path: Path,
    expected_sha: str,
    *,
    new_run_must_exist: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    request_artifact, request = _json_file(
        request_path, expected_sha, "consumer recovery request"
    )
    _exact_keys(request, RECOVERY_REQUEST_KEYS, "consumer recovery request")
    if (
        request["format"] != RECOVERY_REQUEST_FORMAT
        or request["status"] != "ready_for_single_recovery"
        or request["split"] != "val"
        or request["test_visible"] is not False
        or request["selection_eligible"] is not False
        or type(request["candidate_epoch"]) is not int
        or request["candidate_epoch"] != FORMAL_EPOCH_ONE
        or _payload_sha(request) != request["receipt_payload_sha256"]
    ):
        raise LiveConsumerError("consumer recovery request state changed")
    _finite(request["created_unix"], "consumer recovery request time", positive=True)
    request_artifact["receipt_payload_sha256"] = request[
        "receipt_payload_sha256"
    ]

    failed_campaign_artifact, failed_campaign = _campaign_document(
        request["failed_campaign"], "failed campaign"
    )
    _require_failed_incident_artifact(failed_campaign_artifact, "campaign")
    failed_control = _exact_keys(
        failed_campaign["control_source"],
        CONTROL_SOURCE_KEYS,
        "failed campaign control source",
    )
    if (
        failed_control["commit"] != FAILED_CONTROL_COMMIT
        or failed_control["tree"] != FAILED_CONTROL_TREE
    ):
        raise LiveConsumerError("failed campaign is not the pinned 58407ed incident")
    new_campaign_artifact, new_campaign = _campaign_document(
        request["new_campaign"], "new campaign"
    )
    if failed_campaign_artifact == new_campaign_artifact:
        raise LiveConsumerError("failed and new campaigns are not distinct")
    failed_job = _campaign_job(failed_campaign, FORMAL_EPOCH_ONE)
    new_job = _campaign_job(new_campaign, FORMAL_EPOCH_ONE)

    failed_work_artifact, failed_work, _ = _validate_work_authority(
        Path(request["failed_work_authority"]["path"]),
        request["failed_work_authority"]["sha256"],
        load_modules=False,
    )
    new_work_artifact, new_work, _ = _validate_work_authority(
        Path(request["new_work_authority"]["path"]),
        request["new_work_authority"]["sha256"],
        load_modules=False,
    )
    _validate_recovery_work_authority_replay(
        request,
        failed_work_artifact,
        failed_work,
        new_work_artifact,
        new_work,
    )
    _require_failed_incident_artifact(
        _artifact_core(failed_work_artifact), "work_authority"
    )
    _require_failed_incident_artifact(
        _artifact_core(failed_work["producer_ready_receipt"]),
        "candidate_receipt",
    )
    epoch = failed_work["candidate_epoch"]
    if epoch != FORMAL_EPOCH_ONE:
        raise LiveConsumerError("only the failed formal e1 is recoverable")

    failed_run_root = _canonical_path(
        request["failed_run_root"], "failed run root"
    )
    new_run_root = _canonical_path(
        request["new_run_root"], "new run root", must_exist=False
    )
    if new_run_must_exist:
        new_run_root = _canonical_path(new_run_root, "new run root")
        if not new_run_root.is_dir():
            raise LiveConsumerError("new recovery run root is not a directory")
    elif os.path.lexists(new_run_root):
        raise LiveConsumerError("new recovery run root is not new")
    if failed_run_root == new_run_root:
        raise LiveConsumerError("recovery cannot reuse the failed run root")
    if (
        failed_job["authority_path"] != failed_work_artifact["path"]
        or failed_job["run_root"] != str(failed_run_root)
        or new_job["authority_path"] != new_work_artifact["path"]
        or new_job["run_root"] != str(new_run_root)
    ):
        raise LiveConsumerError("campaign job paths differ from recovery request")

    failed_job_artifact, failed_job_claim = _self_hashed_document(
        request["failed_job_claim"],
        keys=FAILED_JOB_CLAIM_KEYS,
        payload_key="claim_payload_sha256",
        label="failed job claim",
        supervisor_payload_abi=True,
    )
    _require_failed_incident_artifact(failed_job_artifact, "job_claim")
    failed_state = _canonical_path(
        failed_campaign["state_root"], "failed campaign state root"
    )
    if (
        Path(failed_job_artifact["path"])
        != failed_state / "job_claims" / "epoch-0001.json"
        or failed_job_claim["format"]
        != "semtalk_show_base_v14_live_validation_job_claim_v2"
        or failed_job_claim["status"] != "claimed"
        or failed_job_claim["candidate_epoch"] != FORMAL_EPOCH_ONE
        or not _strict_equal(failed_job_claim["campaign"], failed_campaign_artifact)
        or failed_job_claim["authority_path"] != failed_work_artifact["path"]
        or failed_job_claim["run_root"] != str(failed_run_root)
        or failed_job_claim["measurement_path"] != failed_job["measurement_path"]
        or not _strict_equal(
            failed_job_claim["candidate_receipt"],
            _artifact_core(failed_work["producer_ready_receipt"]),
        )
    ):
        raise LiveConsumerError("failed job claim binding changed")
    _sha(failed_job_claim["authorize_argv_sha256"], "failed authorize argv SHA")
    _finite(failed_job_claim["created_unix"], "failed job claim time", positive=True)

    failed_active_artifact, failed_active = _self_hashed_document(
        request["failed_active_claim"],
        keys=FAILED_ACTIVE_CLAIM_KEYS,
        payload_key="claim_payload_sha256",
        label="failed active claim",
        supervisor_payload_abi=True,
    )
    _require_failed_incident_artifact(failed_active_artifact, "active_claim")
    if (
        Path(failed_active_artifact["path"])
        != failed_state / "active_invocation.claim.json"
        or failed_active["format"]
        != "semtalk_show_base_v14_live_validation_active_claim_v1"
        or failed_active["status"] != "active"
        or failed_active["operation"] != "run-next"
        or not _strict_equal(failed_active["campaign"], failed_campaign_artifact)
    ):
        raise LiveConsumerError("failed active claim binding changed")
    _finite(failed_active["created_unix"], "failed active claim time", positive=True)

    failed_authorization_artifact, failed_authorization = _self_hashed_document(
        request["failed_authorization"],
        keys=FAILED_AUTHORIZATION_KEYS,
        payload_key="receipt_payload_sha256",
        label="failed authorization",
        supervisor_payload_abi=True,
    )
    _require_failed_incident_artifact(
        failed_authorization_artifact, "authorization"
    )
    if (
        Path(failed_authorization_artifact["path"])
        != Path(failed_job["authorization_path"])
        or failed_authorization["format"]
        != "semtalk_show_base_v14_live_authorization_v1"
        or failed_authorization["status"] != "complete"
        or failed_authorization["candidate_epoch"] != FORMAL_EPOCH_ONE
        or not _strict_equal(
            failed_authorization["campaign"], failed_campaign_artifact
        )
        or not _strict_equal(
            failed_authorization["candidate_receipt"],
            _artifact_core(failed_work["producer_ready_receipt"]),
        )
        or not _strict_equal(
            failed_authorization["work_authority"],
            _artifact_core(failed_work_artifact),
        )
    ):
        raise LiveConsumerError("failed authorization binding changed")
    _sha(
        failed_authorization["adapter_stdout_sha256"],
        "failed adapter stdout SHA",
    )
    _finite(
        failed_authorization["completed_unix"],
        "failed authorization time",
        positive=True,
    )

    failed_status_artifact, failed_status_raw = _core_artifact(
        request["failed_runner_status"], "failed runner status"
    )
    _require_failed_incident_artifact(failed_status_artifact, "runner_status")
    failed_status = _strict_json_bytes(failed_status_raw, "failed runner status")
    _exact_keys(failed_status, FAILED_RUNNER_STATUS_KEYS, "failed runner status")
    restored = failed_status["restored_guards"]
    if (
        Path(failed_status_artifact["path"]) != Path(failed_job["runner_status_path"])
        or failed_status["state"] != "failed"
        or type(failed_status["return_code"]) is not int
        or failed_status["return_code"] != 1
        or any(
            failed_status[key] is not None
            for key in ("received_signal", "error", "cleanup_error", "restore_error")
        )
        or failed_status["wrapper_pid"] != FAILED_WRAPPER_PID
        or failed_status["child_pid"] != FAILED_CHILD_PID
        or not isinstance(restored, dict)
    ):
        raise LiveConsumerError("failed runner was not the exact clean rc1 exit")
    runner_argv = failed_authorization["runner_argv"]
    if not isinstance(runner_argv, list) or runner_argv.count("--") != 1:
        raise LiveConsumerError("failed guarded runner argv changed")
    separator = runner_argv.index("--")
    if (
        failed_status["command"] != runner_argv[separator + 1 :]
        or _argv_option(runner_argv, "--status", "failed runner status option")
        != failed_status_artifact["path"]
        or _argv_option(runner_argv, "--log", "failed runner log option")
        != failed_job["runner_log_path"]
        or _argv_option(
            failed_status["command"], "--work-authority", "failed work option"
        )
        != failed_work_artifact["path"]
        or _argv_option(failed_status["command"], "--run-root", "failed run option")
        != str(failed_run_root)
    ):
        raise LiveConsumerError("failed runner command binding changed")

    failed_log_artifact, failed_log = _core_artifact(
        request["failed_runner_log"], "failed runner log"
    )
    _require_failed_incident_artifact(failed_log_artifact, "runner_log")
    if (
        Path(failed_log_artifact["path"]) != Path(failed_job["runner_log_path"])
        or failed_log != FAILED_RUNNER_LOG
    ):
        raise LiveConsumerError("failed runner log changed")

    failed_claim_artifact, failed_claim = _json_file(
        Path(request["failed_consumer_claim"]["path"]),
        request["failed_consumer_claim"]["sha256"],
        "failed global consumer claim",
    )
    _exact_keys(failed_claim, CLAIM_KEYS, "failed global consumer claim")
    failed_claim_artifact["receipt_payload_sha256"] = failed_claim[
        "receipt_payload_sha256"
    ]
    _require_failed_incident_artifact(
        _artifact_core(failed_claim_artifact), "consumer_claim"
    )
    failed_training_root = _training_root_for_candidate(
        producer_ready_receipt=failed_work["producer_ready_receipt"],
        candidate_checkpoint=failed_work["candidate_checkpoint"],
        epoch=FORMAL_EPOCH_ONE,
    )
    expected_failed_claim_path = _claim_slot(
        failed_training_root,
        directory_name="live_val_consumer_claims",
        epoch=FORMAL_EPOCH_ONE,
        create_directory=False,
    )
    if (
        not _strict_equal(
            _artifact_core(failed_claim_artifact),
            request["failed_consumer_claim"],
        )
        or Path(failed_claim_artifact["path"]) != expected_failed_claim_path
        or failed_claim["format"] != CLAIM_FORMAT
        or failed_claim["status"] != "claimed"
        or failed_claim["split"] != "val"
        or failed_claim["test_visible"] is not False
        or failed_claim["selection_eligible"] is not False
        or failed_claim["candidate_epoch"] != FORMAL_EPOCH_ONE
        or failed_claim["expected_shards"] != 8
        or not _strict_equal(
            failed_claim["work_authority"], failed_work_artifact
        )
        or failed_claim["run_root"] != str(failed_run_root)
    ):
        raise LiveConsumerError("failed global consumer claim changed")

    inventory = _exact_keys(
        request["failed_run_inventory"],
        FAILED_RUN_INVENTORY_KEYS,
        "failed run inventory",
    )
    for index, file_value in enumerate(inventory["files"]):
        _exact_keys(file_value, FAILED_RUN_FILE_KEYS, f"failed run file {index}")
    if not _strict_equal(inventory, _relative_inventory(failed_run_root)):
        raise LiveConsumerError("failed run inventory changed")
    _validate_guard_proof(
        request["guard_proof"],
        failed_campaign=failed_campaign,
        failed_status=failed_status,
    )
    _validate_failed_process_proof(request["failed_process_proof"], failed_status)

    new_control = _validate_recovery_control_source(request["new_control_source"])
    if not _strict_equal(new_control, new_campaign["control_source"]):
        raise LiveConsumerError("new campaign/control source binding changed")
    new_state = _canonical_path(new_campaign["state_root"], "new campaign state root")
    recovery_authority_path = new_state / "recovery-authority.epoch-0001.json"
    recovery_claim_path = _recovery_claim_path(
        new_work, create_directory=False
    )
    return (
        request_artifact,
        request,
        {
            "failed_campaign": failed_campaign_artifact,
            "failed_consumer_claim": failed_claim_artifact,
            "failed_runner_status": failed_status_artifact,
            "new_campaign": new_campaign_artifact,
            "new_control_source": new_control,
            "new_work_authority": new_work_artifact,
            "new_work_authority_value": new_work,
            "new_run_root": str(new_run_root),
        },
        recovery_claim_path,
        recovery_authority_path,
    )


def _recovery_documents(
    *,
    request_artifact: Mapping[str, Any],
    request: Mapping[str, Any],
    bindings: Mapping[str, Any],
    recovery_claim_path: Path,
    recovery_authority_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_reference = _artifact_reference(request_artifact, request)
    authority_body = _with_payload_sha(
        {
            "format": RECOVERY_AUTHORITY_FORMAT,
            "status": "authorized",
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "candidate_epoch": FORMAL_EPOCH_ONE,
            "expected_shards": 8,
            "recovery_request": request_reference,
            "recovery_claim_path": str(recovery_claim_path),
            "failed_consumer_claim": _artifact_core(
                bindings["failed_consumer_claim"]
            ),
            "failed_runner_status": dict(bindings["failed_runner_status"]),
            "new_campaign": dict(bindings["new_campaign"]),
            "new_control_source": copy.deepcopy(bindings["new_control_source"]),
            "new_work_authority": _artifact_core(
                bindings["new_work_authority"]
            ),
            "new_run_root": bindings["new_run_root"],
        }
    )
    authority_encoded = _canonical_json(authority_body, newline=True)
    authority_preview = {
        "path": str(recovery_authority_path),
        "sha256": hashlib.sha256(authority_encoded).hexdigest(),
        "bytes": len(authority_encoded),
    }
    claim_body = _with_payload_sha(
        {
            "format": RECOVERY_CLAIM_FORMAT,
            "status": "claimed",
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "candidate_epoch": FORMAL_EPOCH_ONE,
            "expected_shards": 8,
            "recovery_request": request_reference,
            "failed_consumer_claim": _artifact_core(
                bindings["failed_consumer_claim"]
            ),
            "failed_runner_status": dict(bindings["failed_runner_status"]),
            "new_campaign": dict(bindings["new_campaign"]),
            "new_control_source": copy.deepcopy(bindings["new_control_source"]),
            "new_work_authority": _artifact_core(
                bindings["new_work_authority"]
            ),
            "new_run_root": bindings["new_run_root"],
            "recovery_authority": authority_preview,
        }
    )
    return authority_body, claim_body, authority_preview


def _recovery_preview(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path
]:
    (
        request_artifact,
        request,
        bindings,
        recovery_claim_path,
        recovery_authority_path,
    ) = _validate_recovery_request(
        args.request,
        args.expected_request_sha256,
        new_run_must_exist=False,
    )
    output = _canonical_path(
        args.output_authority, "recovery authority output", must_exist=False
    )
    if output != recovery_authority_path:
        raise LiveConsumerError(
            "recovery authority output is not the deterministic new-state slot"
        )
    parent = _canonical_path(output.parent, "recovery authority output parent")
    if not parent.is_dir():
        raise LiveConsumerError("recovery authority output parent is absent")
    authority_body, claim_body, authority_preview = _recovery_documents(
        request_artifact=request_artifact,
        request=request,
        bindings=bindings,
        recovery_claim_path=recovery_claim_path,
        recovery_authority_path=recovery_authority_path,
    )
    if os.path.lexists(recovery_claim_path) or os.path.lexists(output):
        raise DuplicateConsumptionError("consumer recovery is already reserved")
    return (
        authority_body,
        claim_body,
        authority_preview,
        bindings,
        recovery_claim_path,
    )


def inspect_recovery(args: argparse.Namespace) -> dict[str, Any]:
    _authority, _claim, preview, _bindings, claim_path = _recovery_preview(args)
    return {
        "status": "ready",
        "recovery_authority": preview,
        "recovery_claim_path": str(claim_path),
    }


def _write_strict_new(
    path: Path, value: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if os.path.lexists(path):
        raise DuplicateConsumptionError(f"{label} already exists: {path}")
    artifact, created = _write_new_or_identical(path, value)
    if not created:
        raise DuplicateConsumptionError(f"{label} was not create-new: {path}")
    return artifact


def reserve_recovery(args: argparse.Namespace) -> dict[str, Any]:
    authority_body, claim_body, preview, bindings, claim_path = (
        _recovery_preview(args)
    )
    created_claim_path = _recovery_claim_path(
        bindings["new_work_authority_value"],
        create_directory=True,
    )
    if created_claim_path != claim_path:
        raise LiveConsumerError("recovery claim slot changed during reservation")
    claim_artifact = _write_strict_new(
        claim_path, claim_body, "consumer recovery claim"
    )
    # The claim is intentionally durable before the authority.  If authority
    # publication fails, the one recovery allowance remains consumed.
    authority_artifact = _write_strict_new(
        Path(preview["path"]), authority_body, "consumer recovery authority"
    )
    if not _strict_equal(authority_artifact, preview):
        raise LiveConsumerError("published recovery authority differs from preview")
    return {
        "status": "reserved",
        "recovery_authority": authority_artifact,
        "recovery_claim": claim_artifact,
    }


def _validate_recovery_binding(
    *,
    recovery_authority_path: Path,
    expected_recovery_authority_sha: str,
    recovery_claim_path: Path,
    expected_recovery_claim_sha: str,
    work_authority_artifact: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    authority_artifact, authority = _json_file(
        recovery_authority_path,
        expected_recovery_authority_sha,
        "consumer recovery authority",
    )
    _exact_keys(authority, RECOVERY_AUTHORITY_KEYS, "consumer recovery authority")
    claim_artifact, claim = _json_file(
        recovery_claim_path,
        expected_recovery_claim_sha,
        "consumer recovery claim",
    )
    _exact_keys(claim, RECOVERY_CLAIM_KEYS, "consumer recovery claim")
    authority_artifact["receipt_payload_sha256"] = authority[
        "receipt_payload_sha256"
    ]
    claim_artifact["receipt_payload_sha256"] = claim["receipt_payload_sha256"]
    if (
        authority["format"] != RECOVERY_AUTHORITY_FORMAT
        or authority["status"] != "authorized"
        or claim["format"] != RECOVERY_CLAIM_FORMAT
        or claim["status"] != "claimed"
        or any(
            document["split"] != "val"
            or document["test_visible"] is not False
            or document["selection_eligible"] is not False
            or document["candidate_epoch"] != FORMAL_EPOCH_ONE
            or document["expected_shards"] != 8
            or _payload_sha(document) != document["receipt_payload_sha256"]
            for document in (authority, claim)
        )
    ):
        raise LiveConsumerError("consumer recovery authority/claim state changed")
    request_value = authority["recovery_request"]
    _exact_keys(
        request_value,
        frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
        "recovery request reference",
    )
    (
        request_artifact,
        request,
        bindings,
        expected_claim_path,
        expected_authority_path,
    ) = _validate_recovery_request(
        Path(request_value["path"]),
        request_value["sha256"],
        new_run_must_exist=True,
    )
    if (
        not _strict_equal(_artifact_reference(request_artifact, request), request_value)
        or Path(authority_artifact["path"]) != expected_authority_path
        or Path(claim_artifact["path"]) != expected_claim_path
        or run_root != Path(bindings["new_run_root"])
        or not _strict_equal(
            _artifact_core(work_authority_artifact),
            _artifact_core(bindings["new_work_authority"]),
        )
    ):
        raise LiveConsumerError("consumer recovery execution binding changed")
    expected_authority, expected_claim, authority_preview = _recovery_documents(
        request_artifact=request_artifact,
        request=request,
        bindings=bindings,
        recovery_claim_path=expected_claim_path,
        recovery_authority_path=expected_authority_path,
    )
    if (
        not _strict_equal(authority, expected_authority)
        or not _strict_equal(claim, expected_claim)
        or not _strict_equal(_artifact_core(authority_artifact), authority_preview)
        or not _strict_equal(claim["recovery_authority"], authority_preview)
        or authority["recovery_claim_path"] != claim_artifact["path"]
    ):
        raise LiveConsumerError("consumer recovery replay changed")
    return claim_artifact


def _preflight_body(
    *,
    authority_artifact: Mapping[str, Any],
    authority: Mapping[str, Any],
    claim_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    epoch = authority["candidate_epoch"]
    checkpoint = authority["candidate_checkpoint"]
    return {
        "format": PREFLIGHT_FORMAT,
        "status": "ready",
        "split": "val",
        "test_visible": False,
        "selection_eligible": False,
        "candidate_epochs": [epoch],
        "candidate_bundle": {
            "manifest": {
                key: authority["producer_manifest_snapshot"][key]
                for key in ("path", "sha256")
            },
            "status": {
                key: authority["producer_ready_receipt"][key]
                for key in ("path", "sha256")
            },
            "frozen_inputs": {
                "path": authority["frozen_inputs"]["path"],
                "sha256": authority["frozen_inputs"]["sha256"],
                "receipt_sha256": authority["frozen_inputs"]["receipt_payload_sha256"],
            },
            "producer_source": authority["producer_source"],
            "selected_prerequisite_sha256": authority["selected_prerequisite_sha256"],
            "selected_topology": authority["selected_topology"],
            "updates_per_epoch": authority["selected_topology"]["updates_per_epoch"],
            "candidates": {
                str(epoch): {
                    key: checkpoint[key] for key in ("path", "sha256", "bytes")
                }
            },
        },
        "val_inputs_receipt": authority["val_inputs_receipt"],
        "pipeline_receipt": authority["pipeline_receipt"],
        "pipeline_source": authority["pipeline_source"],
        "inference_entrypoint": authority["inference_entrypoint"],
        "coverage": authority["coverage"],
        "work_authority": dict(authority_artifact),
        "consumer_claim": dict(claim_artifact),
        "execution_contract": dict(authority["execution_contract"]),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    run_root = _canonical_path(args.run_root, "run root")
    if not run_root.is_dir():
        raise LiveConsumerError("run root must already be a new directory")
    output = _canonical_path(args.output, "preflight output", must_exist=False)
    if output != run_root / "work-preflight.json":
        raise LiveConsumerError("preflight output must be RUN_ROOT/work-preflight.json")
    authority_artifact, authority, _modules = _validate_work_authority(
        args.work_authority, args.expected_work_authority_sha256
    )
    epoch = authority["candidate_epoch"]
    recovery_values = (
        getattr(args, "recovery_authority", None),
        getattr(args, "expected_recovery_authority_sha256", None),
        getattr(args, "recovery_claim", None),
        getattr(args, "expected_recovery_claim_sha256", None),
    )
    if any(value is not None for value in recovery_values) and not all(
        value is not None for value in recovery_values
    ):
        raise LiveConsumerError(
            "prepare recovery arguments must be supplied all-or-none"
        )
    if all(value is not None for value in recovery_values):
        claim_artifact = _validate_recovery_binding(
            recovery_authority_path=Path(recovery_values[0]),
            expected_recovery_authority_sha=str(recovery_values[1]),
            recovery_claim_path=Path(recovery_values[2]),
            expected_recovery_claim_sha=str(recovery_values[3]),
            work_authority_artifact=authority_artifact,
            run_root=run_root,
        )
    else:
        claim_body = _with_payload_sha(
            {
                "format": CLAIM_FORMAT,
                "status": "claimed",
                "split": "val",
                "test_visible": False,
                "selection_eligible": False,
                "candidate_epoch": epoch,
                "expected_shards": 8,
                "work_authority": authority_artifact,
                "run_root": str(run_root),
            }
        )
        claim_path = _claim_path(
            producer_ready_receipt=authority["producer_ready_receipt"],
            candidate_checkpoint=authority["candidate_checkpoint"],
            epoch=epoch,
        )
        claim_artifact, _created = _write_new_or_identical(claim_path, claim_body)
        claim_artifact["receipt_payload_sha256"] = claim_body[
            "receipt_payload_sha256"
        ]
    preflight = _with_payload_sha(
        _preflight_body(
            authority_artifact=authority_artifact,
            authority=authority,
            claim_artifact=claim_artifact,
        )
    )
    artifact, created = _write_new_or_identical(output, preflight)
    artifact["receipt_payload_sha256"] = preflight["receipt_payload_sha256"]
    return {
        "status": "ready",
        "split": "val",
        "test_visible": False,
        "selection_eligible": False,
        "epoch": epoch,
        "created": created,
        "preflight": artifact,
        "claim": claim_artifact,
    }


def _preflight_artifact(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, payload = _json_file(path, expected_sha, "live validation preflight")
    expected_keys = frozenset(
        {
            "format", "status", "split", "test_visible", "selection_eligible",
            "candidate_epochs", "candidate_bundle", "val_inputs_receipt",
            "pipeline_receipt", "pipeline_source", "inference_entrypoint",
            "coverage", "work_authority", "consumer_claim", "execution_contract",
            "receipt_payload_sha256",
        }
    )
    _exact_keys(payload, expected_keys, "live validation preflight")
    epochs = payload["candidate_epochs"]
    if (
        payload["format"] != PREFLIGHT_FORMAT
        or payload["status"] != "ready"
        or payload["split"] != "val"
        or payload["test_visible"] is not False
        or payload["selection_eligible"] is not False
        or not isinstance(epochs, list)
        or len(epochs) != 1
        or type(epochs[0]) is not int
        or epochs[0] not in CANDIDATE_EPOCHS
        or _payload_sha(payload) != payload["receipt_payload_sha256"]
    ):
        raise LiveConsumerError("live preflight is not one val-only candidate")
    authority_value = _exact_keys(
        payload["work_authority"],
        frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
        "preflight work authority",
    )
    authority_artifact, authority, _ = _validate_work_authority(
        Path(authority_value["path"]), authority_value["sha256"], load_modules=False
    )
    authority_artifact["receipt_payload_sha256"] = authority["receipt_payload_sha256"]
    claim_value = _exact_keys(
        payload["consumer_claim"],
        frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
        "consumer claim",
    )
    claim_artifact, claim = _json_file(
        Path(claim_value["path"]), claim_value["sha256"], "consumer claim"
    )
    claim_artifact["receipt_payload_sha256"] = claim["receipt_payload_sha256"]
    if claim.get("format") == CLAIM_FORMAT:
        _exact_keys(claim, CLAIM_KEYS, "consumer claim")
        if (
            claim.get("status") != "claimed"
            or claim.get("split") != "val"
            or claim.get("test_visible") is not False
            or claim.get("selection_eligible") is not False
            or _integer(
                claim.get("candidate_epoch"),
                "claim candidate epoch",
                minimum=1,
            )
            != epochs[0]
            or _integer(
                claim.get("expected_shards"),
                "claim expected shards",
                minimum=1,
            )
            != 8
            or not _strict_equal(claim.get("work_authority"), authority_artifact)
            or not _strict_equal(claim_artifact, claim_value)
        ):
            raise LiveConsumerError("consumer claim changed")
    elif claim.get("format") == RECOVERY_CLAIM_FORMAT:
        _exact_keys(claim, RECOVERY_CLAIM_KEYS, "consumer recovery claim")
        recovery_authority = _exact_keys(
            claim["recovery_authority"],
            ARTIFACT_CORE_KEYS,
            "consumer recovery authority reference",
        )
        replayed_claim = _validate_recovery_binding(
            recovery_authority_path=Path(recovery_authority["path"]),
            expected_recovery_authority_sha=recovery_authority["sha256"],
            recovery_claim_path=Path(claim_artifact["path"]),
            expected_recovery_claim_sha=claim_artifact["sha256"],
            work_authority_artifact=authority_artifact,
            run_root=Path(claim["new_run_root"]),
        )
        if (
            epochs[0] != FORMAL_EPOCH_ONE
            or not _strict_equal(replayed_claim, claim_artifact)
            or not _strict_equal(claim_artifact, claim_value)
        ):
            raise LiveConsumerError("consumer recovery claim changed")
    else:
        raise LiveConsumerError("consumer claim format changed")
    expected = _with_payload_sha(
        _preflight_body(
            authority_artifact=authority_artifact,
            authority=authority,
            claim_artifact=claim_artifact,
        )
    )
    if not _strict_equal(payload, expected):
        raise LiveConsumerError("live preflight differs from authority replay")
    artifact["receipt_payload_sha256"] = payload["receipt_payload_sha256"]
    return artifact, payload


@contextmanager
def _engine_adapter(
    engine: ModuleType,
    *,
    evidence_pipeline: Mapping[str, Any],
    runtime_pipeline: Mapping[str, Any],
) -> Iterable[None]:
    names = (
        "_preflight_artifact",
        "_load_pinned_helper",
        "_pinned_joint_mask_arrays",
        "_load_models",
    )
    previous = {name: getattr(engine, name, None) for name in names}
    if any(not callable(previous[name]) for name in names):
        raise LiveConsumerError("runtime validation engine adapter ABI changed")
    evidence_snapshot = copy.deepcopy(evidence_pipeline)

    def execution_view(pipeline: Mapping[str, Any]) -> dict[str, Any]:
        if (
            not _strict_equal(pipeline, evidence_pipeline)
            or not _strict_equal(evidence_pipeline, evidence_snapshot)
        ):
            raise LiveConsumerError(
                "engine pipeline differs from validated immutable evidence"
            )
        return copy.deepcopy(runtime_pipeline)

    def load_pinned_helper(pipeline: Mapping[str, Any]) -> ModuleType:
        return previous["_load_pinned_helper"](execution_view(pipeline))

    def pinned_joint_mask_arrays(
        helper: ModuleType,
        pipeline: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return previous["_pinned_joint_mask_arrays"](
            helper,
            execution_view(pipeline),
        )

    def load_models(
        helper: ModuleType,
        *,
        epoch: int,
        preflight: Mapping[str, Any],
        pipeline: Mapping[str, Any],
        device: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return previous["_load_models"](
            helper,
            epoch=epoch,
            preflight=preflight,
            pipeline=execution_view(pipeline),
            device=device,
        )

    try:
        engine._preflight_artifact = _preflight_artifact
        engine._load_pinned_helper = load_pinned_helper
        engine._pinned_joint_mask_arrays = pinned_joint_mask_arrays
        engine._load_models = load_models
        yield
    finally:
        for name in names:
            setattr(engine, name, previous[name])


def _engine_command(args: argparse.Namespace, command: str) -> dict[str, Any]:
    _artifact_value, preflight = _preflight_artifact(
        args.preflight, args.expected_preflight_sha256
    )
    epoch = preflight["candidate_epochs"][0]
    if args.epoch != epoch or args.num_shards != 8:
        raise LiveConsumerError("engine command differs from one-candidate eight-shard work")
    authority = preflight["work_authority"]
    _authority_artifact, validated_authority, modules = _validate_work_authority(
        Path(authority["path"]), authority["sha256"]
    )
    assert modules is not None
    evidence_pipeline, runtime_pipeline, runtime_root = (
        _runtime_execution_pipeline(validated_authority, modules)
    )
    engine = modules["engine"]
    with _engine_adapter(
        engine,
        evidence_pipeline=evidence_pipeline,
        runtime_pipeline=runtime_pipeline,
    ):
        if command == "shard":
            result = engine.run_shard(args)
        else:
            result = engine.finalize(args)
    _project_modules_are_from(runtime_root)
    return result


def _measurement_value(
    *,
    preflight_artifact: Mapping[str, Any],
    preflight: Mapping[str, Any],
    lineage_artifact: Mapping[str, Any],
    report_artifact: Mapping[str, Any],
    fgd: float,
) -> dict[str, Any]:
    epoch = preflight["candidate_epochs"][0]
    candidate = preflight["candidate_bundle"]["candidates"][str(epoch)]
    return _with_payload_sha(
        {
            "format": MEASUREMENT_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "candidate_epoch": epoch,
            "candidate_checkpoint": {
                "path": candidate["path"], "sha256": candidate["sha256"]
            },
            "work_authority": preflight["work_authority"],
            "consumer_claim": preflight["consumer_claim"],
            "execution_preflight": dict(preflight_artifact),
            "val_inputs_receipt": preflight["val_inputs_receipt"],
            "pipeline_receipt": preflight["pipeline_receipt"],
            "inference_lineage": dict(lineage_artifact),
            "diffsheg_report": dict(report_artifact),
            "metrics": {"fgd": fgd},
            "execution_contract": {
                "expected_shards": 8,
                "exact_once": True,
                "finite": True,
                "may_influence_training": False,
                "requires_e400_reconciliation_for_selection": True,
            },
        }
    )


def complete(args: argparse.Namespace) -> dict[str, Any]:
    preflight_artifact, preflight = _preflight_artifact(
        args.preflight, args.expected_preflight_sha256
    )
    authority = preflight["work_authority"]
    _authority_artifact, _authority, modules = _validate_work_authority(
        Path(authority["path"]), authority["sha256"]
    )
    assert modules is not None
    contract = modules["contract"]
    epoch = preflight["candidate_epochs"][0]
    candidate = preflight["candidate_bundle"]["candidates"][str(epoch)]
    lineage_artifact, lineage = _json_file(
        args.inference_lineage,
        args.expected_inference_lineage_sha256,
        "live inference lineage",
    )
    audited_lineage, lineage_payload = contract.validate_val_inference_lineage(
        Path(lineage_artifact["path"]),
        lineage_artifact["sha256"],
        epoch=epoch,
        expected_candidate=candidate,
        val_inputs_artifact=preflight["val_inputs_receipt"],
        pipeline_artifact=preflight["pipeline_receipt"],
        expected_coverage=contract.validate_val_inputs(
            Path(preflight["val_inputs_receipt"]["path"]),
            preflight["val_inputs_receipt"]["sha256"],
        )[1],
    )
    if audited_lineage["receipt_payload_sha256"] != lineage.get("receipt_payload_sha256"):
        raise LiveConsumerError("live inference lineage payload binding changed")
    report_snapshot, report = _json_file(
        args.diffsheg_report,
        args.expected_diffsheg_report_sha256,
        "live DiffSHEG report",
    )
    report_artifact = {
        "path": report_snapshot["path"],
        "sha256": report_snapshot["sha256"],
    }
    _pipeline_artifact, pipeline = contract.validate_pipeline(
        Path(preflight["pipeline_receipt"]["path"]),
        preflight["pipeline_receipt"]["sha256"],
        expected_source=preflight["pipeline_source"],
    )
    metrics, _coverage = contract.validate_diffsheg_report(
        report,
        expected_coverage=contract.validate_val_inputs(
            Path(preflight["val_inputs_receipt"]["path"]),
            preflight["val_inputs_receipt"]["sha256"],
        )[1],
        inference_lineage=lineage_payload,
        expected_pipeline=pipeline,
    )
    fgd = _finite(metrics.get("fgd"), "live DiffSHEG FGD")
    value = _measurement_value(
        preflight_artifact=preflight_artifact,
        preflight=preflight,
        lineage_artifact=audited_lineage,
        report_artifact=report_artifact,
        fgd=fgd,
    )
    artifact, created = _write_new_or_identical(args.output, value)
    artifact["receipt_payload_sha256"] = value["receipt_payload_sha256"]
    return {
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "selection_eligible": False,
        "epoch": epoch,
        "fgd": fgd,
        "created": created,
        "measurement": artifact,
    }


def _validate_live_measurement(
    path: Path,
    expected_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _json_file(path, expected_sha, "live measurement")
    expected = frozenset(
        {
            "format", "status", "split", "test_visible", "selection_eligible",
            "candidate_epoch", "candidate_checkpoint", "work_authority",
            "consumer_claim", "execution_preflight", "val_inputs_receipt",
            "pipeline_receipt", "inference_lineage", "diffsheg_report",
            "metrics", "execution_contract", "receipt_payload_sha256",
        }
    )
    _exact_keys(value, expected, "live measurement")
    epoch = _integer(value["candidate_epoch"], "live measurement epoch", minimum=1)
    candidate = _exact_keys(
        value["candidate_checkpoint"],
        frozenset({"path", "sha256"}),
        "live candidate checkpoint",
    )
    _canonical_path(candidate["path"], "live candidate checkpoint")
    _sha(candidate["sha256"], "live candidate checkpoint SHA")
    for key, keys, label in (
        (
            "work_authority",
            frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
            "live work authority",
        ),
        (
            "consumer_claim",
            frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
            "live consumer claim",
        ),
        (
            "execution_preflight",
            frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
            "live execution preflight",
        ),
        (
            "val_inputs_receipt",
            frozenset({"path", "sha256", "receipt_payload_sha256"}),
            "live validation inputs receipt",
        ),
        (
            "pipeline_receipt",
            frozenset({"path", "sha256", "receipt_payload_sha256"}),
            "live pipeline receipt",
        ),
        (
            "inference_lineage",
            frozenset({"path", "sha256", "receipt_payload_sha256"}),
            "live inference lineage",
        ),
        (
            "diffsheg_report",
            frozenset({"path", "sha256"}),
            "live DiffSHEG report",
        ),
    ):
        item = _exact_keys(value[key], keys, label)
        _canonical_path(item["path"], label)
        _sha(item["sha256"], f"{label} SHA")
        if "bytes" in item:
            _integer(item["bytes"], f"{label} bytes", minimum=1)
        if "receipt_payload_sha256" in item:
            _sha(item["receipt_payload_sha256"], f"{label} payload SHA")
    metrics = _exact_keys(
        value["metrics"], frozenset({"fgd"}), "live measurement metrics"
    )
    fgd = _finite(metrics["fgd"], "live measurement FGD")
    if (
        value["format"] != MEASUREMENT_FORMAT
        or value["status"] != "complete"
        or value["split"] != "val"
        or value["test_visible"] is not False
        or value["selection_eligible"] is not False
        or epoch not in CANDIDATE_EPOCHS
        or fgd < 0.0
        or not _strict_equal(
            value["execution_contract"],
            {
                "expected_shards": 8,
                "exact_once": True,
                "finite": True,
                "may_influence_training": False,
                "requires_e400_reconciliation_for_selection": True,
            },
        )
    ):
        raise LiveConsumerError("live measurement is not selection-ineligible val work")
    preflight_artifact, preflight = _preflight_artifact(
        Path(value["execution_preflight"]["path"]),
        value["execution_preflight"]["sha256"],
    )
    if (
        not _strict_equal(preflight_artifact, value["execution_preflight"])
        or not _strict_equal(preflight["candidate_epochs"], [epoch])
        or preflight["candidate_bundle"]["candidates"][str(epoch)]["path"]
        != candidate["path"]
        or preflight["candidate_bundle"]["candidates"][str(epoch)]["sha256"]
        != candidate["sha256"]
        or not _strict_equal(preflight["work_authority"], value["work_authority"])
        or not _strict_equal(preflight["consumer_claim"], value["consumer_claim"])
        or not _strict_equal(
            preflight["val_inputs_receipt"], value["val_inputs_receipt"]
        )
        or not _strict_equal(
            preflight["pipeline_receipt"], value["pipeline_receipt"]
        )
    ):
        raise LiveConsumerError("live measurement differs from execution replay")
    artifact["receipt_payload_sha256"] = value["receipt_payload_sha256"]
    return artifact, value


def _validate_reconciliation(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _json_file(path, expected_sha, "live reconciliation")
    _exact_keys(value, RECONCILIATION_KEYS, "live reconciliation")
    if (
        value["format"] != RECONCILIATION_FORMAT
        or value["status"] != "complete"
        or value["split"] != "val"
        or value["test_visible"] is not False
        or value["selection_eligible"] is not True
        or value["all_exact"] is not True
        or not _strict_equal(value["candidate_epochs"], list(CANDIDATE_EPOCHS))
        or type(value["test_evaluations_observed"]) is not int
        or value["test_evaluations_observed"] != 0
        or _payload_sha(value) != value["receipt_payload_sha256"]
        or not isinstance(value["work_authorities"], list)
        or len(value["work_authorities"]) != len(CANDIDATE_EPOCHS)
    ):
        raise LiveConsumerError("reconciliation is not exact e400 val-only authority")
    _finite(value["completed_unix"], "reconciliation completion time", positive=True)
    if not _strict_equal(
        value["pipeline_source"], value["validation_evidence_source"]
    ):
        raise LiveConsumerError("reconciliation evidence source roles differ")
    _validate_runtime_validation_roles(value)
    _artifact(
        value["producer_manifest"],
        frozenset({"path", "sha256", "bytes"}),
        "reconciled producer manifest",
    )
    _artifact(
        value["producer_status"],
        frozenset({"path", "sha256", "bytes"}),
        "reconciled producer status",
    )
    _artifact(
        value["frozen_inputs"],
        frozenset({"path", "sha256", "bytes", "receipt_payload_sha256"}),
        "reconciled frozen inputs",
        artifact_payload_key="receipt_payload_sha256",
        document_payload_key="receipt_sha256",
    )
    _artifact(
        value["schedule"],
        frozenset({"path", "sha256", "bytes", "payload_sha256"}),
        "reconciled schedule",
        artifact_payload_key="payload_sha256",
        canonical_full_payload=True,
    )
    trajectory = _exact_keys(
        value["trajectory_contract"],
        frozenset(
            {
                "format", "mode", "sha256", "payload_sha256",
                "probe_optimizer_updates",
            }
        ),
        "reconciled trajectory contract",
    )
    if (
        trajectory["format"]
        != "semtalk_show_base_fresh_lineage_trajectory_contract_v1"
        or trajectory["mode"] != "fresh_lineage_gate_v1"
        or _sha(trajectory["sha256"], "trajectory SHA")
        != _sha(trajectory["payload_sha256"], "trajectory payload SHA")
        or _integer(
            trajectory["probe_optimizer_updates"],
            "trajectory probe updates",
            minimum=1,
        )
        != 70
    ):
        raise LiveConsumerError("reconciled trajectory contract changed")
    for child, label in (
        (value["val_inputs_receipt"], "reconciled validation inputs"),
        (value["pipeline_receipt"], "reconciled validation pipeline"),
    ):
        _artifact(
            child,
            frozenset({"path", "sha256", "receipt_payload_sha256"}),
            label,
            artifact_payload_key="receipt_payload_sha256",
            document_payload_key="receipt_payload_sha256",
        )
    authority_paths: set[str] = set()
    authority_shas: set[str] = set()
    common_fields = (
        "frozen_inputs", "producer_source", "validation_evidence_source",
        "runtime_validation_source", "runtime_validation_proof",
        "selected_topology", "schedule", "trajectory_contract",
        "val_inputs_receipt", "pipeline_receipt", "pipeline_source",
    )
    for epoch, authority_value in zip(CANDIDATE_EPOCHS, value["work_authorities"]):
        reference = _exact_keys(
            authority_value,
            frozenset({"path", "sha256", "bytes"}),
            f"reconciled work authority e{epoch}",
        )
        authority_artifact, authority, _modules = _validate_work_authority(
            Path(reference["path"]), reference["sha256"], load_modules=False
        )
        if (
            not _strict_equal(_artifact_core(authority_artifact), reference)
            or authority["candidate_epoch"] != epoch
            or any(
                not _strict_equal(authority[field], value[field])
                for field in common_fields
            )
        ):
            raise LiveConsumerError(
                f"reconciliation does not replay work authority e{epoch}"
            )
        authority_paths.add(reference["path"])
        authority_shas.add(reference["sha256"])
    if (
        len(authority_paths) != len(CANDIDATE_EPOCHS)
        or len(authority_shas) != len(CANDIDATE_EPOCHS)
    ):
        raise LiveConsumerError("reconciled work authorities are not exact-once")
    artifact["receipt_payload_sha256"] = value["receipt_payload_sha256"]
    return artifact, value


def reconcile(args: argparse.Namespace) -> dict[str, Any]:
    reconciliation_artifact, reconciliation = _validate_reconciliation(
        args.reconciliation, args.expected_reconciliation_sha256
    )
    triples = args.live_measurement
    if len(triples) != len(CANDIDATE_EPOCHS):
        raise LiveConsumerError("reconcile requires exactly 22 live measurements")
    observed_epochs = []
    live_artifacts = []
    live_values = []
    for raw_epoch, raw_path, raw_sha in triples:
        if not raw_epoch.isdecimal():
            raise LiveConsumerError("live measurement epoch must be an integer")
        epoch = int(raw_epoch)
        artifact, value = _validate_live_measurement(Path(raw_path), raw_sha)
        if epoch != value["candidate_epoch"]:
            raise LiveConsumerError("live measurement label/receipt epoch mismatch")
        observed_epochs.append(epoch)
        live_artifacts.append(artifact)
        live_values.append(value)
    if observed_epochs != list(CANDIDATE_EPOCHS):
        raise LiveConsumerError("live measurements must cover the 22 epochs in order")
    if len({item["path"] for item in live_artifacts}) != len(CANDIDATE_EPOCHS):
        raise LiveConsumerError("live measurement paths are not distinct")
    if len({item["sha256"] for item in live_artifacts}) != len(CANDIDATE_EPOCHS):
        raise LiveConsumerError("live measurement SHA roots are not distinct")
    if not _strict_equal(
        [_artifact_core(value["work_authority"]) for value in live_values],
        reconciliation["work_authorities"],
    ):
        raise LiveConsumerError("live measurements do not bind reconciled work authorities")
    common_keys = ("val_inputs_receipt", "pipeline_receipt")
    if any(
        not _strict_equal(value[key], live_values[0][key])
        for value in live_values[1:]
        for key in common_keys
    ):
        raise LiveConsumerError("live measurements mix validation pipelines")
    if (
        not _strict_equal(
            live_values[0]["val_inputs_receipt"],
            reconciliation["val_inputs_receipt"],
        )
        or not _strict_equal(
            live_values[0]["pipeline_receipt"],
            reconciliation["pipeline_receipt"],
        )
    ):
        raise LiveConsumerError("live measurements differ from reconciliation inputs")

    _evidence_root, runtime_root = _validate_runtime_validation_roles(
        reconciliation
    )
    modules = _load_validation_modules(runtime_root)
    contract = modules["contract"]
    pipeline_artifact, pipeline = contract.validate_pipeline(
        Path(reconciliation["pipeline_receipt"]["path"]),
        reconciliation["pipeline_receipt"]["sha256"],
        expected_source=reconciliation["validation_evidence_source"],
    )
    if not _strict_equal(
        pipeline_artifact, reconciliation["pipeline_receipt"]
    ):
        raise LiveConsumerError("fresh pipeline replay differs from reconciliation")
    expected_selected = {
        stage: pipeline["fixed_checkpoints"][stage]["sha256"]
        for stage in ("face", "hands", "upper", "lower", "global")
    }
    candidate_bundle = contract.validate_candidate_bundle(
        manifest_path=Path(reconciliation["producer_manifest"]["path"]),
        expected_manifest_sha256=reconciliation["producer_manifest"]["sha256"],
        status_path=Path(reconciliation["producer_status"]["path"]),
        expected_status_sha256=reconciliation["producer_status"]["sha256"],
        frozen_inputs_path=Path(reconciliation["frozen_inputs"]["path"]),
        expected_frozen_inputs_sha256=reconciliation["frozen_inputs"]["sha256"],
        expected_selected_prerequisite_sha256=expected_selected,
    )
    for epoch, live in zip(CANDIDATE_EPOCHS, live_values):
        candidate = candidate_bundle["candidates"][epoch]
        if not _strict_equal(
            live["candidate_checkpoint"],
            {"path": candidate["path"], "sha256": candidate["sha256"]},
        ):
            raise LiveConsumerError(f"live e{epoch} candidate differs from final bundle")

    output_root = _canonical_path(args.output_root, "selection output root", must_exist=False)
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite selection root: {output_root}")
    parent = _canonical_path(output_root.parent, "selection output parent")
    output_root.mkdir(mode=0o755)
    measurements_root = output_root / "reconciled-measurements"
    measurements_root.mkdir()
    standard_paths = []
    standard_shas = []
    standard_artifacts = []
    try:
        producer = modules["producer"]
        selector = modules["selector"]
        for epoch, live in zip(CANDIDATE_EPOCHS, live_values):
            standard = producer.build_measurement(
                epoch=epoch,
                candidate_bundle=candidate_bundle,
                val_inputs_artifact=live["val_inputs_receipt"],
                pipeline_artifact=live["pipeline_receipt"],
                inference_lineage_artifact=live["inference_lineage"],
                diffsheg_report_artifact=live["diffsheg_report"],
            )
            path = measurements_root / f"epoch-{epoch:04d}.json"
            artifact, created = _write_new_or_identical(path, standard)
            if not created:
                raise LiveConsumerError("new selection root contained a measurement")
            standard_paths.append(path)
            standard_shas.append(artifact["sha256"])
            standard_artifacts.append(artifact)
        formal_selection = selector.build_selection(
            candidate_bundle=candidate_bundle,
            measurement_paths=standard_paths,
            expected_measurement_sha256=standard_shas,
        )
        value = _with_payload_sha(
            {
                "format": SELECTION_FORMAT,
                "status": "selected",
                "split": "val",
                "test_visible": False,
                "selection_eligible": True,
                "candidate_epochs": list(CANDIDATE_EPOCHS),
                "reconciliation_receipt": reconciliation_artifact,
                "live_measurements": live_artifacts,
                "reconciled_measurements": standard_artifacts,
                "formal_selection": formal_selection,
                "selected": formal_selection["selected"],
                "test_evaluations_observed": 0,
            }
        )
        output = output_root / "selection.json"
        selection_artifact, created = _write_new_or_identical(output, value)
        if not created:
            raise LiveConsumerError("new selection root contained a selection")
        selection_artifact["receipt_payload_sha256"] = value["receipt_payload_sha256"]
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Keep partial evidence.  The output root is never reused or overwritten.
        raise
    return {
        "status": "selected",
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "candidate_count": len(CANDIDATE_EPOCHS),
        "selected_epoch": value["selected"]["epoch"],
        "selected_fgd": value["selected"]["fgd"],
        "selection": selection_artifact,
    }


def _epoch_arg(value: str) -> int:
    if not value.isdecimal():
        raise argparse.ArgumentTypeError("epoch must be an integer")
    epoch = int(value)
    if epoch not in CANDIDATE_EPOCHS:
        raise argparse.ArgumentTypeError("epoch is outside the formal 22")
    return epoch


def _add_preflight(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("val",), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--epoch", type=_epoch_arg, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, choices=(8,), required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", allow_abbrev=False)
    prepare_parser.add_argument("--work-authority", type=Path, required=True)
    prepare_parser.add_argument("--expected-work-authority-sha256", required=True)
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--recovery-authority", type=Path)
    prepare_parser.add_argument("--expected-recovery-authority-sha256")
    prepare_parser.add_argument("--recovery-claim", type=Path)
    prepare_parser.add_argument("--expected-recovery-claim-sha256")

    for command in ("inspect-recovery", "reserve-recovery"):
        recovery_parser = commands.add_parser(command, allow_abbrev=False)
        recovery_parser.add_argument("--request", type=Path, required=True)
        recovery_parser.add_argument("--expected-request-sha256", required=True)
        recovery_parser.add_argument(
            "--output-authority", type=Path, required=True
        )

    inspect_parser = commands.add_parser("inspect", allow_abbrev=False)
    inspect_parser.add_argument("--preflight", type=Path, required=True)
    inspect_parser.add_argument("--expected-preflight-sha256", required=True)

    shard_parser = commands.add_parser("shard", allow_abbrev=False)
    _add_preflight(shard_parser)
    shard_parser.add_argument("--shard-id", type=int, choices=range(8), required=True)
    shard_parser.add_argument("--device", required=True)
    shard_parser.add_argument("--seed", type=int, default=20260731)
    shard_parser.add_argument("--progress-every", type=int, default=20)

    finalize_parser = commands.add_parser("finalize", allow_abbrev=False)
    _add_preflight(finalize_parser)

    complete_parser = commands.add_parser("complete", allow_abbrev=False)
    complete_parser.add_argument("--preflight", type=Path, required=True)
    complete_parser.add_argument("--expected-preflight-sha256", required=True)
    complete_parser.add_argument("--inference-lineage", type=Path, required=True)
    complete_parser.add_argument("--expected-inference-lineage-sha256", required=True)
    complete_parser.add_argument("--diffsheg-report", type=Path, required=True)
    complete_parser.add_argument("--expected-diffsheg-report-sha256", required=True)
    complete_parser.add_argument("--output", type=Path, required=True)

    reconcile_parser = commands.add_parser("reconcile", allow_abbrev=False)
    reconcile_parser.add_argument("--reconciliation", type=Path, required=True)
    reconcile_parser.add_argument("--expected-reconciliation-sha256", required=True)
    reconcile_parser.add_argument(
        "--live-measurement", nargs=3, action="append", required=True,
        metavar=("EPOCH", "PATH", "SHA256"),
    )
    reconcile_parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "inspect-recovery":
            result = inspect_recovery(args)
        elif args.command == "reserve-recovery":
            result = reserve_recovery(args)
        elif args.command == "inspect":
            artifact, payload = _preflight_artifact(
                args.preflight, args.expected_preflight_sha256
            )
            result = {
                "status": "ready", "split": "val", "test_visible": False,
                "selection_eligible": False,
                "epoch": payload["candidate_epochs"][0], "preflight": artifact,
            }
        elif args.command in {"shard", "finalize"}:
            result = _engine_command(args, args.command)
        elif args.command == "complete":
            result = complete(args)
        elif args.command == "reconcile":
            result = reconcile(args)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (
        DuplicateConsumptionError,
        LiveConsumerError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
