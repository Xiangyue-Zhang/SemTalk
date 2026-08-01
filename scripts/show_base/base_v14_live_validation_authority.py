#!/usr/bin/env python3
"""Fail-closed live validation authority for fresh SemTalk SHOW Base runs.

This program is deliberately external to the producer checkout.  The runtime
producer is the exact V14 formal-training source at 8f1fa7b.  Its numerical
training semantics are proved against 5b84075.  The immutable pipeline evidence
remains pinned to 4066f20, while validation code executes only from the minimal
70a70f4 provenance-fix successor.  Those roles are intentionally distinct.  It never
infers readiness from a checkpoint or from the mutable live manifest.  The
only producer event it consumes is the create-new candidate-ready receipt at
``candidate_receipts/epoch-NNNN.json``.  A successful ``authorize`` command
publishes a create-new, validation-only work authority.  The authority is not
a standard 22-candidate preflight and is never selection eligible.

After the producer has atomically finalized e400, ``reconcile`` replays all
22 producer closures and checks them against the complete manifest/status.
Only the reconciliation receipt is selection eligible.  It contains no test
path or test result and cannot authorize a test evaluation.

GPU launch, inference, DiffSHEG scoring, and selection are intentionally out
of scope.  A launcher may consume the work authority only through a separate
guarded validation-only runner.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
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
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
PRODUCER_SOURCE_COMMIT = "8f1fa7b85ed8253600a4c571e98eb9927edeb073"
PRODUCER_SOURCE_TREE = "d5e5eb74acdc6d9ca330d5731e718e33b67cf626"
PRODUCER_TRAINER_SHA256 = (
    "29fdd5d3e9bdfc61904f649b71d4dae1766b42a4a6a5a40b2bbd37a4c6b33173"
)
PRODUCER_V14_CONTRACT_SHA256 = (
    "3526ca896f23e7849545e3f81553dd242dc1ca3049eabdeeb89fc38357616323"
)
TRAINING_SEMANTICS_SOURCE_COMMIT = (
    "5b84075bb5bc9577a891a1f5ff72e93c39bab2e8"
)
TRAINING_SEMANTICS_SOURCE_TREE = (
    "05372163157d24cc8a72c073ac172b7c1859bba0"
)
TRAINING_SEMANTICS_TRAINER_SHA256 = (
    "65cb565ceaf70f5c744b5c85db50e41d728f507d12cac28b06a5cc3756b65dba"
)
# The unchanged top-level numerical objective/data/checkpoint definitions in
# 5b84075 and 8f1fa7b have this canonical name->AST hash projection.  The exact
# changed-name set below is limited to V14 control-plane, schedule, and
# throughput-gate plumbing; the entire runtime producer file is independently
# byte pinned above.
TRAINING_SEMANTICS_UNCHANGED_DEFS_SHA256 = (
    "ca62d6297e80c2e71bb2c6a92305b44fe746e161fa17f8fddf22ea10ef223bab"
)
TRAINING_SEMANTICS_ALLOWED_CHANGED_DEFS = frozenset(
    {
        "_frozen_gate_cross_run_compatibility_sha256",
        "_load_throughput_gate_frozen_receipt",
        "build_parser",
        "main",
        "validate_args",
        "validate_long_contract_receipts",
        "validate_throughput_gate",
    }
)
VALIDATION_EVIDENCE_SOURCE_COMMIT = (
    "4066f2096e1675f9c19d725894007ff25f3e9b4b"
)
VALIDATION_EVIDENCE_SOURCE_TREE = (
    "0b66e3aa1fb23732e76e51492737c4ab1f4db2d0"
)
VALIDATION_EVIDENCE_CONTRACT_SHA256 = (
    "3983eb7adf2f8cbfcf738fca8cc1cf8f3b34acab274bb5dfd2e23964550f8daa"
)
VALIDATION_EVIDENCE_SELECTOR_SHA256 = (
    "1b76579d54efc99ac0d0d63f3d1a68ff671216e693ef9ada4714c9fcda004121"
)
RUNTIME_VALIDATION_SOURCE_COMMIT = (
    "70a70f452bdf743e317b583a7770980f0ce744c3"
)
RUNTIME_VALIDATION_SOURCE_TREE = (
    "bdf7680f56f9f53c92e7ab0bf6c6a84ef4f83d69"
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
OFFICIAL_BASE_SHA256 = (
    "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
)
READY_FORMAT = "semtalk_show_base_official_adapt_long_candidate_ready_v1"
MANIFEST_FORMAT = "semtalk_show_base_official_adapt_long_manifest_v1"
STATUS_FORMAT = "semtalk_show_base_official_adapt_long_status_v1"
CHECKPOINT_FORMAT = "semtalk_show_base_official_adapt_checkpoint_v1"
FROZEN_FORMAT = "semtalk_show_base_official_adapt_frozen_inputs_v1"
PROTOCOL_FORMAT = "semtalk_show_base_official_adapt_long_protocol_v1"
FRESH_SCHEDULE_FORMAT = "semtalk_show_base_fresh_lineage_schedule_v14_v1"
V14_SCHEDULE_SHA256 = (
    "1c87b74907d8e69f3713c659d5448a8f3355b92599fb83660e54d4896a0c48db"
)
V14_TOPOLOGY_SOURCE = (
    "sealed_v14_two_topology_12_validation_fgd_min_tuple_v1"
)
V14_SELECTION_PROTOCOL_SHA256 = (
    "ad556b191f580b21e589a0f4800ccd6bce38acff6183d1194223fe592ef0bc69"
)
FRESH_TRAJECTORY_FORMAT = (
    "semtalk_show_base_fresh_lineage_trajectory_contract_v1"
)
FRESH_TRAJECTORY_MODE = "fresh_lineage_gate_v1"
TRAJECTORY_PROBE_FORMAT = "semtalk_show_base_trajectory_probe_v3"
THROUGHPUT_FORMAT = "semtalk_show_base_official_adapt_long_throughput_gate_v1"
WORK_FORMAT = "semtalk_show_base_live_val_candidate_work_authority_v2"
RECONCILIATION_FORMAT = "semtalk_show_base_live_val_reconciliation_v2"
VALIDATION_HELPER_MAX_INPUT_BYTES = 8 * 1024 * 1024
VALIDATION_HELPER_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
VALIDATION_HELPER_TIMEOUT_SECONDS = 300

CANDIDATE_EPOCHS = (
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
VALIDATION_WAVES = (
    (1, 2, 4, 8),
    (16, 32, 40, 50),
    (60, 70, 80, 100),
    (120, 140, 160, 180),
    (200, 240, 280, 320),
    (360, 400),
)

W8G1024_MODE = "validation_gated_w8_l128_g1024_empirical_acceleration"
W8G2048_MODE = "validation_gated_w8_l256_g2048_empirical_acceleration"
V14_MODES = (W8G2048_MODE, W8G1024_MODE)
V14_TOPOLOGY_SELECTION_EPOCHS = (1, 2, 4, 8, 16, 32)
SUPPORTED_TOPOLOGIES: dict[str, dict[str, Any]] = {
    W8G1024_MODE: {
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
    W8G2048_MODE: {
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
}

FORMAL_HOST_BY_SLOT = {
    0: "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0",
    1: "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_ID_RE = re.compile(r"[0-9a-f]{40}")
TEST_PATH_TOKENS = frozenset({"test", "tests", "testset", "testsets"})
FORBIDDEN_SOURCE_RE = re.compile(
    r"(^|[^a-z0-9])(?:e(?:poch)?[-_]?30|speaker[-_]?2|semgate|sparse)"
    r"([^a-z0-9]|$)",
    re.IGNORECASE,
)

READY_KEYS = frozenset(
    {
        "format",
        "status",
        "selection_eligible",
        "test_visible",
        "epoch",
        "optimizer_updates",
        "candidate_checkpoint",
        "candidate_manifest",
        "frozen_inputs",
        "protocol",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "trajectory_anchor_match",
        "published_unix",
        "receipt_payload_sha256",
    }
)
CHECKPOINT_REFERENCE_KEYS = frozenset(
    {
        "path",
        "relative_path",
        "sha256",
        "bytes",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
    }
)
MANIFEST_REFERENCE_KEYS = frozenset(
    {
        "path",
        "sha256_at_ready",
        "entries_sha256_at_ready",
        "immutable_snapshot",
        "live_path",
    }
)
FROZEN_REFERENCE_KEYS = frozenset(
    {"path", "sha256", "receipt_payload_sha256"}
)
PROTOCOL_REFERENCE_KEYS = frozenset({"format", "payload_sha256"})
SNAPSHOT_KEYS = frozenset(
    {
        "format",
        "status",
        "candidate_epochs",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "throughput_gate",
        "trajectory_mode",
        "trajectory_probe_verified",
        "trajectory_probe",
        "entries",
        "entries_sha256",
    }
)
ENTRY_KEYS = frozenset(
    {
        "epoch",
        "optimizer_updates",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_container_schema",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
        "all_model_state_tensors_finite",
        "frozen_receipt_sha256",
        "trajectory_anchor_match",
        "trajectory_probe_verified",
    }
)
FROZEN_KEYS = frozenset(
    {
        "format",
        "run_purpose",
        "target_epochs",
        "source",
        "official_base",
        "speaker_initialization",
        "dataset",
        "protocol",
        "long_contract",
        "topology",
        "receipt_sha256",
    }
)
AUDIT_KEYS = frozenset(
    {
        "format",
        "completed_epochs",
        "optimizer_updates",
        "frozen_receipt_sha256",
        "official_base_checkpoint_sha256",
        "speaker_scope",
        "speaker_rows",
        "vq_models_in_training_graph",
        "all_model_state_tensors_finite",
        "model_state_semantic_sha256",
        "trajectory_anchor_match",
        "trajectory_probe_verified",
    }
)
THROUGHPUT_REFERENCE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "topology_mode",
        "samples_per_second",
        "seconds_per_update",
        "median_seconds",
        "p90_seconds",
        "p99_seconds",
        "estimated_training_seconds",
        "trajectory_mode",
        "trajectory_probe",
        "gate_frozen_receipt_sha256",
        "frozen_gate_compatibility_sha256",
    }
)
FINAL_STATUS_KEYS = frozenset(
    {
        "format",
        "status",
        "completed_epochs",
        "optimizer_updates",
        "updates_per_epoch",
        "candidate_manifest_sha256",
        "frozen_receipt_sha256",
        "throughput_gate",
        "world_size",
        "local_batch_size",
        "global_batch_size",
        "all_training_state_finite",
        "epoch_metrics_jsonl",
        "epoch_metrics_sha256",
        "epoch_metrics_records",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "trajectory_mode",
        "trajectory_probe_verified",
        "trajectory_probe",
        "started_unix",
        "completed_unix",
        "resume_receipt",
        "resume_receipt_sha256",
    }
)
EPOCH_METRIC_KEYS = frozenset(
    {
        "format",
        "epoch",
        "optimizer_updates",
        "updates_per_epoch",
        "learning_rate",
        "metrics",
        "all_finite",
        "completed_unix",
    }
)
RESUME_RECEIPT_KEYS = frozenset(
    {
        "format",
        "status",
        "completed_epochs",
        "optimizer_updates",
        "path",
        "sha256",
        "bytes",
        "candidate_checkpoint",
        "rank_rng_states",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "trajectory_mode",
        "trajectory_probe_verified",
        "completed_unix",
    }
)


class LiveValidationContractError(RuntimeError):
    """Raised before publication when any live-validation input is unsafe."""


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise LiveValidationContractError("value is not strict JSON") from error
    if newline:
        encoded += "\n"
    return encoded.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON semantics without Python's bool/int or int/float aliases."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            strict_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise LiveValidationContractError(f"{label} is not a canonical SHA-256")
    return value


def exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LiveValidationContractError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def exact_int_equal(value: Any, expected: int) -> bool:
    """JSON booleans/floats must never compare equal to integer counters."""

    return type(value) is int and value == expected


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveValidationContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise LiveValidationContractError(f"{label} is out of range")
    return result


def exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LiveValidationContractError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        raise LiveValidationContractError(
            f"{label} keys changed; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    return value


def reject_test_path(path: Path, label: str) -> None:
    for part in path.parts:
        tokens = set(re.findall(r"[a-z0-9]+", part.casefold()))
        if tokens & TEST_PATH_TOKENS:
            raise LiveValidationContractError(
                f"{label} must not expose a test-labelled path: {path}"
            )


def reject_forbidden_source(value: Any, label: str) -> None:
    if FORBIDDEN_SOURCE_RE.search(str(value)):
        raise LiveValidationContractError(
            f"{label} contains a forbidden source marker: {value}"
        )


def canonical_existing_directory(value: Any, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise LiveValidationContractError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LiveValidationContractError(f"{label} is absent: {path}") from error
    if resolved != path or path.is_symlink() or not path.is_dir():
        raise LiveValidationContractError(
            f"{label} must be a canonical non-symlink directory: {path}"
        )
    return path


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def safe_regular_bytes(path_value: Any, label: str) -> tuple[Path, bytes, str, int]:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        raise LiveValidationContractError(f"{label} must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LiveValidationContractError(
            f"cannot safely open {label}: {path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise LiveValidationContractError(
                f"{label} must be a single-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise LiveValidationContractError(f"{label} changed while read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise LiveValidationContractError(f"{label} size changed while read")
        resolved = path.resolve(strict=True)
        current = os.stat(path, follow_symlinks=False)
        if resolved != path or _identity(current) != _identity(before):
            raise LiveValidationContractError(
                f"{label} pathname changed after descriptor verification"
            )
        return path, payload, hashlib.sha256(payload).hexdigest(), len(payload)
    finally:
        os.close(descriptor)


def safe_regular_hash(path_value: Any, label: str) -> tuple[Path, str, int]:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        raise LiveValidationContractError(f"{label} must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LiveValidationContractError(
            f"cannot safely open {label}: {path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise LiveValidationContractError(
                f"{label} must be a single-link regular file"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise LiveValidationContractError(f"{label} changed while hashed")
        resolved = path.resolve(strict=True)
        current = os.stat(path, follow_symlinks=False)
        if resolved != path or _identity(current) != _identity(before):
            raise LiveValidationContractError(
                f"{label} pathname changed after descriptor verification"
            )
        return path, digest.hexdigest(), int(before.st_size)
    finally:
        os.close(descriptor)


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LiveValidationContractError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LiveValidationContractError(
                    f"non-finite JSON token in {label}: {token}"
                )
            ),
        )
    except LiveValidationContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LiveValidationContractError(f"invalid strict JSON: {label}") from error
    if not isinstance(value, dict):
        raise LiveValidationContractError(f"{label} must contain an object")
    return value


def verified_json(
    path_value: Any,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str, int]:
    expected = require_sha256(expected_sha256, f"{label} expected SHA-256")
    path, payload, observed, size = safe_regular_bytes(path_value, label)
    if observed != expected:
        raise LiveValidationContractError(
            f"{label} SHA-256 {observed} != {expected}"
        )
    return path, strict_json_bytes(payload, label), observed, size


def artifact(path: Path, sha256: str, size: int) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256, "bytes": size}


def write_new_json(path_value: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        raise LiveValidationContractError("authority output must be absolute")
    reject_test_path(path, "authority output")
    parent = canonical_existing_directory(path.parent, "authority output parent")
    output = parent / path.name
    if output != path or os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite authority: {output}")
    encoded = canonical_json_bytes(dict(payload), newline=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.partial-", suffix=".json", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite authority: {output}") from None
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    _, digest, size = safe_regular_hash(output, "published authority")
    return artifact(output, digest, size)


@dataclass(frozen=True)
class AuthorizeConfig:
    train_root: Path
    epoch: int
    topology_mode: str
    producer_source_root: Path
    expected_producer_trainer_sha256: str
    expected_producer_contract_sha256: str
    validation_evidence_source_root: Path
    expected_validation_evidence_contract_sha256: str
    expected_validation_evidence_selector_sha256: str
    runtime_validation_source_root: Path
    expected_runtime_validation_contract_sha256: str
    expected_runtime_validation_selector_sha256: str
    schedule: Path
    expected_schedule_sha256: str
    expected_frozen_inputs_sha256: str
    val_inputs: Path
    expected_val_inputs_sha256: str
    pipeline: Path
    expected_pipeline_sha256: str
    output: Path


@dataclass(frozen=True)
class ReconcileConfig:
    common: AuthorizeConfig
    authority_dir: Path
    final_manifest: Path
    expected_final_manifest_sha256: str
    final_status: Path
    expected_final_status_sha256: str


def _top_level_definition_hashes(
    encoded: bytes, *, label: str
) -> dict[str, str]:
    try:
        source = encoded.decode("utf-8", errors="strict")
        module = ast.parse(source, filename=label)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise LiveValidationContractError(
            f"cannot parse {label} for the training-semantics proof"
        ) from error
    definitions: dict[str, str] = {}
    for node in module.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if node.name in definitions:
            raise LiveValidationContractError(
                f"duplicate top-level definition in {label}: {node.name}"
            )
        normalized = ast.dump(node, include_attributes=False).encode("utf-8")
        definitions[node.name] = hashlib.sha256(normalized).hexdigest()
    if not definitions:
        raise LiveValidationContractError(
            f"{label} has no top-level definitions"
        )
    return definitions


def _git_file_bytes(root: Path, commit: str, relative: str, label: str) -> bytes:
    try:
        process = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LiveValidationContractError(
            f"cannot read the pinned {label} Git blob"
        ) from error
    if not process.stdout:
        raise LiveValidationContractError(f"pinned {label} Git blob is empty")
    return process.stdout


def _prove_training_semantics(root: Path) -> dict[str, Any]:
    relative = "scripts/show_base/train_base_official_adapt_long.py"
    try:
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                TRAINING_SEMANTICS_SOURCE_COMMIT,
                PRODUCER_SOURCE_COMMIT,
            ],
            check=False,
            capture_output=True,
        )
        semantic_tree = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                f"{TRAINING_SEMANTICS_SOURCE_COMMIT}^{{tree}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise LiveValidationContractError(
            "cannot prove the producer/training-semantics ancestry"
        ) from error
    if ancestor.returncode != 0 or semantic_tree != TRAINING_SEMANTICS_SOURCE_TREE:
        raise LiveValidationContractError(
            "producer does not descend from the pinned training semantics"
        )
    semantic_bytes = _git_file_bytes(
        root,
        TRAINING_SEMANTICS_SOURCE_COMMIT,
        relative,
        "training-semantics trainer",
    )
    producer_bytes = _git_file_bytes(
        root, PRODUCER_SOURCE_COMMIT, relative, "runtime producer trainer"
    )
    if (
        hashlib.sha256(semantic_bytes).hexdigest()
        != TRAINING_SEMANTICS_TRAINER_SHA256
        or hashlib.sha256(producer_bytes).hexdigest() != PRODUCER_TRAINER_SHA256
    ):
        raise LiveValidationContractError(
            "training-semantics or runtime-producer trainer blob changed"
        )
    semantic_defs = _top_level_definition_hashes(
        semantic_bytes, label="5b84075 trainer"
    )
    producer_defs = _top_level_definition_hashes(
        producer_bytes, label="8f1fa7b trainer"
    )
    changed = {
        name
        for name in set(semantic_defs) | set(producer_defs)
        if semantic_defs.get(name) != producer_defs.get(name)
    }
    semantic_projection = {
        name: semantic_defs[name]
        for name in sorted(semantic_defs)
        if name not in TRAINING_SEMANTICS_ALLOWED_CHANGED_DEFS
    }
    producer_projection = {
        name: producer_defs[name]
        for name in sorted(producer_defs)
        if name not in TRAINING_SEMANTICS_ALLOWED_CHANGED_DEFS
    }
    projection_sha = canonical_json_sha256(semantic_projection)
    if (
        changed != set(TRAINING_SEMANTICS_ALLOWED_CHANGED_DEFS)
        or not strict_json_equal(semantic_projection, producer_projection)
        or projection_sha != TRAINING_SEMANTICS_UNCHANGED_DEFS_SHA256
    ):
        raise LiveValidationContractError(
            "runtime producer changed outside the sealed V14 control-plane definitions"
        )
    return {
        "format": "semtalk_show_base_v14_training_semantics_ast_proof_v1",
        "runtime_producer": {
            "commit": PRODUCER_SOURCE_COMMIT,
            "tree": PRODUCER_SOURCE_TREE,
            "trainer_sha256": PRODUCER_TRAINER_SHA256,
        },
        "training_semantics_source": {
            "commit": TRAINING_SEMANTICS_SOURCE_COMMIT,
            "tree": TRAINING_SEMANTICS_SOURCE_TREE,
            "trainer_sha256": TRAINING_SEMANTICS_TRAINER_SHA256,
        },
        "allowed_changed_definitions": sorted(
            TRAINING_SEMANTICS_ALLOWED_CHANGED_DEFS
        ),
        "unchanged_definition_count": len(semantic_projection),
        "unchanged_definitions_sha256": projection_sha,
    }


def _prove_runtime_validation_successor(root: Path) -> dict[str, Any]:
    """Prove the 70a runtime is the exact reviewed successor of 4066 evidence.

    The full commits and trees are independently pinned.  This additional
    projection makes the role split auditable: inference, measurement
    production, and the long selector stay byte-identical, while the runtime
    contract and legacy DiffSHEG validator move to their exact reviewed bytes.
    """

    relative_hashes = {
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
    try:
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                VALIDATION_EVIDENCE_SOURCE_COMMIT,
                RUNTIME_VALIDATION_SOURCE_COMMIT,
            ],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise LiveValidationContractError(
            "cannot prove runtime-validation ancestry"
        ) from error
    if ancestor.returncode != 0:
        raise LiveValidationContractError(
            "runtime validation does not descend from evidence source"
        )
    observed: dict[str, dict[str, str]] = {}
    for relative, (evidence_sha, runtime_sha) in relative_hashes.items():
        evidence_bytes = _git_file_bytes(
            root,
            VALIDATION_EVIDENCE_SOURCE_COMMIT,
            relative,
            f"validation evidence {relative}",
        )
        runtime_bytes = _git_file_bytes(
            root,
            RUNTIME_VALIDATION_SOURCE_COMMIT,
            relative,
            f"runtime validation {relative}",
        )
        evidence_observed = hashlib.sha256(evidence_bytes).hexdigest()
        runtime_observed = hashlib.sha256(runtime_bytes).hexdigest()
        if evidence_observed != evidence_sha or runtime_observed != runtime_sha:
            raise LiveValidationContractError(
                f"runtime-validation file projection changed: {relative}"
            )
        observed[relative] = {
            "evidence_sha256": evidence_observed,
            "runtime_sha256": runtime_observed,
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
        "file_projection": observed,
    }


class ValidationHooks:
    """Default bridge to separately pinned producer, evidence, and runtime code."""

    def __init__(
        self,
        *,
        producer_source_root: Path,
        expected_producer_trainer_sha256: str,
        expected_producer_contract_sha256: str,
        validation_evidence_source_root: Path,
        expected_validation_evidence_contract_sha256: str,
        expected_validation_evidence_selector_sha256: str,
        runtime_validation_source_root: Path,
        expected_runtime_validation_contract_sha256: str,
        expected_runtime_validation_selector_sha256: str,
    ) -> None:
        self.producer_source_root = canonical_existing_directory(
            producer_source_root, "producer source root"
        )
        self.validation_evidence_source_root = canonical_existing_directory(
            validation_evidence_source_root, "validation evidence source root"
        )
        self.runtime_validation_source_root = canonical_existing_directory(
            runtime_validation_source_root, "runtime validation source root"
        )
        self.trainer_path = (
            self.producer_source_root
            / "scripts/show_base/train_base_official_adapt_long.py"
        )
        self.producer_contract_path = (
            self.producer_source_root
            / "scripts/show_base/base_v14_formal_contract.py"
        )
        self.validation_evidence_contract_path = (
            self.validation_evidence_source_root
            / "scripts/show_base/base_long_val_contract.py"
        )
        self.validation_evidence_selector_path = (
            self.validation_evidence_source_root
            / "scripts/show_base/select_base_official_adapt.py"
        )
        self.runtime_validation_contract_path = (
            self.runtime_validation_source_root
            / "scripts/show_base/base_long_val_contract.py"
        )
        self.runtime_validation_selector_path = (
            self.runtime_validation_source_root
            / "scripts/show_base/select_base_official_adapt.py"
        )
        _, trainer_sha, _ = safe_regular_hash(self.trainer_path, "trainer entrypoint")
        _, producer_contract_sha, _ = safe_regular_hash(
            self.producer_contract_path, "producer V14 contract entrypoint"
        )
        _, evidence_contract_sha, _ = safe_regular_hash(
            self.validation_evidence_contract_path,
            "validation evidence contract entrypoint",
        )
        _, evidence_selector_sha, _ = safe_regular_hash(
            self.validation_evidence_selector_path,
            "validation evidence selector entrypoint",
        )
        _, runtime_contract_sha, _ = safe_regular_hash(
            self.runtime_validation_contract_path,
            "runtime validation contract entrypoint",
        )
        _, runtime_selector_sha, _ = safe_regular_hash(
            self.runtime_validation_selector_path,
            "runtime validation selector entrypoint",
        )
        expected_trainer_sha = require_sha256(
            expected_producer_trainer_sha256,
            "expected producer trainer entrypoint SHA-256",
        )
        expected_producer_contract_sha = require_sha256(
            expected_producer_contract_sha256,
            "expected producer V14 contract SHA-256",
        )
        expected_evidence_contract_sha = require_sha256(
            expected_validation_evidence_contract_sha256,
            "expected validation evidence contract SHA-256",
        )
        expected_evidence_selector_sha = require_sha256(
            expected_validation_evidence_selector_sha256,
            "expected validation evidence selector SHA-256",
        )
        expected_runtime_contract_sha = require_sha256(
            expected_runtime_validation_contract_sha256,
            "expected runtime validation contract SHA-256",
        )
        expected_runtime_selector_sha = require_sha256(
            expected_runtime_validation_selector_sha256,
            "expected runtime validation selector SHA-256",
        )
        if (
            trainer_sha != PRODUCER_TRAINER_SHA256
            or trainer_sha != expected_trainer_sha
        ):
            raise LiveValidationContractError(
                "producer trainer entrypoint SHA-256 changed"
            )
        if (
            producer_contract_sha != PRODUCER_V14_CONTRACT_SHA256
            or producer_contract_sha != expected_producer_contract_sha
        ):
            raise LiveValidationContractError(
                "producer V14 contract entrypoint SHA-256 changed"
            )
        if (
            evidence_contract_sha != VALIDATION_EVIDENCE_CONTRACT_SHA256
            or evidence_contract_sha != expected_evidence_contract_sha
        ):
            raise LiveValidationContractError(
                "validation evidence contract entrypoint SHA-256 changed"
            )
        if (
            evidence_selector_sha != VALIDATION_EVIDENCE_SELECTOR_SHA256
            or evidence_selector_sha != expected_evidence_selector_sha
        ):
            raise LiveValidationContractError(
                "validation evidence selector entrypoint SHA-256 changed"
            )
        if (
            runtime_contract_sha != RUNTIME_VALIDATION_CONTRACT_SHA256
            or runtime_contract_sha != expected_runtime_contract_sha
        ):
            raise LiveValidationContractError(
                "runtime validation contract entrypoint SHA-256 changed"
            )
        if (
            runtime_selector_sha != RUNTIME_VALIDATION_SELECTOR_SHA256
            or runtime_selector_sha != expected_runtime_selector_sha
        ):
            raise LiveValidationContractError(
                "runtime validation selector entrypoint SHA-256 changed"
            )
        self.trainer_sha256 = trainer_sha
        self.producer_contract_sha256 = producer_contract_sha
        self.validation_evidence_contract_sha256 = evidence_contract_sha
        self.validation_evidence_selector_sha256 = evidence_selector_sha
        self.runtime_validation_contract_sha256 = runtime_contract_sha
        self.runtime_validation_selector_sha256 = runtime_selector_sha
        self.require_exact_v14_schedule_bytes = True
        self.trainer: ModuleType | None = None
        self.topology_specs: dict[str, Any] | None = None

    def _path_within_producer_source(self, path: Path) -> bool:
        try:
            path.resolve(strict=True).relative_to(self.producer_source_root)
        except (OSError, ValueError):
            return False
        return True

    def _verify_project_module_closure(self) -> None:
        project_prefixes = ("scripts", "models", "dataloaders")
        for name, module in tuple(sys.modules.items()):
            if not any(name == prefix or name.startswith(prefix + ".") for prefix in project_prefixes):
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is not None:
                if not self._path_within_producer_source(Path(str(module_file))):
                    raise LiveValidationContractError(
                        f"project module escaped verified source root: {name}"
                    )
                continue
            module_paths = getattr(module, "__path__", None)
            if module_paths is None or any(
                not self._path_within_producer_source(Path(str(path)))
                for path in module_paths
            ):
                raise LiveValidationContractError(
                    f"project namespace escaped verified source root: {name}"
                )

    def _load_verified_modules(self) -> None:
        if self.trainer is not None:
            return
        # Reject a preloaded project namespace before executing either pinned
        # entrypoint.  Git cleanliness has already been proved by the caller.
        self._verify_project_module_closure()
        self.trainer = self._load_module(
            self.trainer_path, f"_live_val_trainer_{self.trainer_sha256}"
        )
        self._verify_project_module_closure()
        self.topology_specs = dict(self.trainer.TOPOLOGY_SPECS)

    def _load_module(self, path: Path, name: str) -> ModuleType:
        root_text = str(self.producer_source_root)
        if root_text in sys.path:
            sys.path.remove(root_text)
        sys.path.insert(0, root_text)
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:
            raise LiveValidationContractError(f"cannot load verified module {path}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        if Path(str(module.__file__)).resolve(strict=True) != path:
            raise LiveValidationContractError(f"verified module escaped source root: {path}")
        return module

    @staticmethod
    def _git_checkout_authority(
        root: Path,
        *,
        expected_commit: str,
        expected_tree: str,
        label: str,
    ) -> dict[str, Any]:
        commands = {
            "commit": ["git", "-C", str(root), "rev-parse", "HEAD"],
            "tree": ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
            "origin": ["git", "-C", str(root), "remote", "get-url", "origin"],
        }
        observed: dict[str, str] = {}
        try:
            for key, command in commands.items():
                process = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                observed[key] = process.stdout.strip()
            status_result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            branch_result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "symbolic-ref",
                    "-q",
                    "--short",
                    "HEAD",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            heads_result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "for-each-ref",
                    "--format=%(refname)",
                    "refs/heads",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise LiveValidationContractError(
                f"could not prove {label} Git source authority"
            ) from error
        if (
            observed
            != {
                "commit": expected_commit,
                "tree": expected_tree,
                "origin": EXPECTED_ORIGIN,
            }
            or status_result.stdout
            or branch_result.returncode != 1
            or branch_result.stdout.strip()
            or heads_result.stdout.strip()
        ):
            raise LiveValidationContractError(
                f"{label} is not the pinned clean detached branchless checkout"
            )
        return {
            "origin": EXPECTED_ORIGIN,
            "commit": expected_commit,
            "tree": expected_tree,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }

    def validate_training_source(
        self, frozen_source: Mapping[str, Any]
    ) -> dict[str, Any]:
        expected_keys = {
            "origin",
            "commit",
            "tree",
            "clean",
            "entrypoint_sha256",
            "node_local_clones",
        }
        if not isinstance(frozen_source, dict):
            raise LiveValidationContractError(
                "frozen runtime producer source changed"
            )
        clones = frozen_source.get("node_local_clones")
        if (
            set(frozen_source) != expected_keys
            or frozen_source.get("origin") != EXPECTED_ORIGIN
            or frozen_source.get("commit") != PRODUCER_SOURCE_COMMIT
            or frozen_source.get("tree") != PRODUCER_SOURCE_TREE
            or frozen_source.get("clean") is not True
            or frozen_source.get("entrypoint_sha256")
            != PRODUCER_TRAINER_SHA256
            or not isinstance(clones, list)
            or len(clones) != 1
        ):
            raise LiveValidationContractError("frozen runtime producer source changed")
        observed_slots: set[int] = set()
        for node_rank, clone in enumerate(clones):
            if not isinstance(clone, dict) or set(clone) != {
                "node_rank", "host_slot", "hostname", "entrypoint", "branch"
            }:
                raise LiveValidationContractError(
                    "frozen node-local producer source schema changed"
                )
            slot = clone.get("host_slot")
            entrypoint = Path(str(clone.get("entrypoint", "")))
            if (
                type(clone.get("node_rank")) is not int
                or clone.get("node_rank") != node_rank
                or type(slot) is not int
                or slot in observed_slots
                or slot not in FORMAL_HOST_BY_SLOT
                or clone.get("hostname") != FORMAL_HOST_BY_SLOT[slot]
                or clone.get("branch") is not None
                or not entrypoint.is_absolute()
                or ".." in entrypoint.parts
                or entrypoint.name != "train_base_official_adapt_long.py"
            ):
                raise LiveValidationContractError(
                    "frozen node-local producer source changed"
                )
            resolved_entrypoint, entrypoint_sha, _entrypoint_bytes = (
                safe_regular_hash(
                    entrypoint,
                    f"frozen node {node_rank} training entrypoint",
                )
            )
            training_root = canonical_existing_directory(
                entrypoint.parents[2],
                f"frozen node {node_rank} training source root",
            )
            if (
                resolved_entrypoint != entrypoint
                or entrypoint_sha != PRODUCER_TRAINER_SHA256
                or training_root != self.producer_source_root
            ):
                raise LiveValidationContractError(
                    "frozen runtime producer differs from the configured producer root"
                )
            observed_slots.add(slot)
        self._git_checkout_authority(
            self.producer_source_root,
            expected_commit=PRODUCER_SOURCE_COMMIT,
            expected_tree=PRODUCER_SOURCE_TREE,
            label="runtime producer source",
        )
        proof = _prove_training_semantics(self.producer_source_root)
        self._load_verified_modules()
        return {
            **dict(frozen_source),
            "training_semantics_proof": proof,
        }

    def validate_validation_sources(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self.validation_evidence_source_root == self.runtime_validation_source_root:
            raise LiveValidationContractError(
                "validation evidence and runtime source roots must be distinct"
            )
        evidence = self._git_checkout_authority(
            self.validation_evidence_source_root,
            expected_commit=VALIDATION_EVIDENCE_SOURCE_COMMIT,
            expected_tree=VALIDATION_EVIDENCE_SOURCE_TREE,
            label="validation evidence source",
        )
        runtime = self._git_checkout_authority(
            self.runtime_validation_source_root,
            expected_commit=RUNTIME_VALIDATION_SOURCE_COMMIT,
            expected_tree=RUNTIME_VALIDATION_SOURCE_TREE,
            label="runtime validation source",
        )
        proof = _prove_runtime_validation_successor(
            self.runtime_validation_source_root
        )

        def public_source(observed: Mapping[str, Any], root: Path) -> dict[str, Any]:
            return {
                "origin": observed["origin"],
                "source_root": str(root),
                "commit": observed["commit"],
                "tree": observed["tree"],
                "clean": observed["clean"],
                "detached": observed["detached"],
                "local_branches_at_commit": observed[
                    "local_branches_at_commit"
                ],
            }

        return (
            public_source(evidence, self.validation_evidence_source_root),
            public_source(runtime, self.runtime_validation_source_root),
            proof,
        )

    def validate_throughput(
        self,
        *,
        reference: Mapping[str, Any],
        frozen: Mapping[str, Any],
        topology_mode: str,
    ) -> dict[str, Any]:
        protocol = frozen["protocol"]
        topology_gate_spec = protocol.get("topology_gate_spec")
        if not isinstance(topology_gate_spec, dict):
            raise LiveValidationContractError("topology gate specification is absent")
        args = SimpleNamespace(
            topology_mode=topology_mode,
            topology_gate_spec=topology_gate_spec.get("path"),
            expected_topology_gate_spec_sha256=topology_gate_spec.get("sha256"),
            throughput_gate_report=reference.get("path"),
            expected_throughput_gate_sha256=reference.get("sha256"),
            precision=protocol.get("precision"),
            learning_rate=protocol.get("optimizer", {}).get("learning_rate"),
        )
        try:
            self.trainer._activate_topology(args)
            gate_spec = self.trainer.validate_topology_gate_spec(args)
            normalized = self.trainer.validate_throughput_gate(
                args, frozen_receipt=frozen
            )
        except Exception as error:
            raise LiveValidationContractError(
                "existing producer throughput/trajectory contract rejected the candidate"
            ) from error
        if gate_spec != topology_gate_spec or normalized != dict(reference):
            raise LiveValidationContractError(
                "manifest throughput closure differs from producer replay"
            )
        return dict(normalized)

    def validate_trajectory_probe(
        self, probe: Any, topology_mode: str
    ) -> dict[str, Any]:
        try:
            self.trainer._activate_topology(SimpleNamespace(topology_mode=topology_mode))
            return self.trainer._validate_trajectory_probe(
                probe, label="live validation trajectory probe"
            )
        except Exception as error:
            raise LiveValidationContractError("trajectory probe replay failed") from error

    def validate_long_contract(
        self,
        *,
        frozen: Mapping[str, Any],
        config: AuthorizeConfig,
        topology_mode: str,
    ) -> dict[str, Any]:
        """Replay the exact producer schedule/fresh-lineage validator.

        The formal frozen receipt stores an ordered global node binding.  The
        producer validator consumed the corresponding node-local receipt
        before aggregation, so reconstruct that exact local shape without
        changing any portable dataset semantics.
        """

        dataset = dict(frozen["dataset"])
        bindings = dataset.pop("node_lmdb_inode_bindings", None)
        dataset.pop("lmdb_binding_scope", None)
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise LiveValidationContractError(
                "single-node live validation requires exactly one LMDB binding"
            )
        row = bindings[0]
        if not isinstance(row, dict) or not isinstance(row.get("binding"), dict):
            raise LiveValidationContractError("node-local LMDB binding is malformed")
        dataset["lmdb_inode_binding"] = dict(row["binding"])
        long_contract = frozen.get("long_contract")
        trajectory = (
            long_contract.get("trajectory_anchor")
            if isinstance(long_contract, dict)
            else None
        )
        selected = dataset.get("prerequisite_selection")
        if not isinstance(trajectory, dict) or not isinstance(selected, dict):
            raise LiveValidationContractError(
                "schedule/dataset cannot be replayed by the producer"
            )
        args = SimpleNamespace(
            topology_mode=topology_mode,
            schedule_json=str(config.schedule),
            expected_schedule_sha256=config.expected_schedule_sha256,
            trajectory_mode=FRESH_TRAJECTORY_MODE,
            precision=SUPPORTED_TOPOLOGIES[topology_mode]["precision"],
            learning_rate=SUPPORTED_TOPOLOGIES[topology_mode]["learning_rate"],
            seed=trajectory.get("seed"),
            loader_workers=trajectory.get("loader_workers"),
            expected_dataset_summary_sha256=dataset.get("summary_sha256"),
            expected_lineage_sha256=dataset.get("lineage_sha256"),
            expected_prerequisite_selection_sha256=selected.get("sha256"),
        )
        try:
            self.trainer._activate_topology(args)
            replayed = self.trainer.validate_long_contract_receipts(
                args, dataset_receipt=dataset
            )
        except Exception as error:
            raise LiveValidationContractError(
                "exact producer schedule/fresh-lineage replay failed"
            ) from error
        if not strict_json_equal(replayed, frozen.get("long_contract")):
            raise LiveValidationContractError(
                "frozen long contract differs from exact producer replay"
            )
        return dict(replayed)

    def verify_checkpoint(
        self,
        *,
        path: Path,
        expected_sha256: str,
        expected_bytes: int,
        epoch: int,
        updates_per_epoch: int,
        frozen_receipt_sha256: str,
    ) -> dict[str, Any]:
        # Hash and torch.load the same open descriptor.  A pathname replacement
        # during deserialization therefore cannot create a hash/load split.
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise LiveValidationContractError(
                f"cannot open candidate checkpoint: {path}"
            ) from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_bytes
            ):
                raise LiveValidationContractError("candidate checkpoint identity changed")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise LiveValidationContractError("candidate checkpoint SHA-256 changed")
            try:
                import torch
            except ModuleNotFoundError as error:
                raise LiveValidationContractError(
                    "torch is mandatory for checkpoint semantic verification"
                ) from error
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                try:
                    payload = torch.load(
                        handle, map_location="cpu", weights_only=True
                    )
                except TypeError as error:
                    raise LiveValidationContractError(
                        "formal runtime must support torch.load(weights_only=True)"
                    ) from error
            after = os.fstat(descriptor)
            current = os.stat(path, follow_symlinks=False)
            if _identity(before) != _identity(after) or _identity(before) != _identity(current):
                raise LiveValidationContractError(
                    "candidate checkpoint changed during semantic verification"
                )
        finally:
            os.close(descriptor)
        if not isinstance(payload, dict) or set(payload) != {"audit", "model_state"}:
            raise LiveValidationContractError("candidate checkpoint envelope changed")
        state = payload.get("model_state")
        audit = exact_keys(payload.get("audit"), AUDIT_KEYS, "checkpoint audit")
        if not isinstance(state, Mapping) or not state:
            raise LiveValidationContractError("candidate model_state is empty")
        for key, tensor in state.items():
            if not isinstance(key, str) or not key or not torch.is_tensor(tensor):
                raise LiveValidationContractError("candidate model_state schema changed")
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                tensor.isfinite().all().item()
            ):
                raise LiveValidationContractError("candidate model_state is non-finite")
        schema_sha = self.trainer._state_schema_sha256(state)
        semantic_sha = self.trainer._model_state_semantic_sha256(state)
        if (
            audit.get("format") != CHECKPOINT_FORMAT
            or not exact_int_equal(audit.get("completed_epochs"), epoch)
            or not exact_int_equal(
                audit.get("optimizer_updates"), epoch * updates_per_epoch
            )
            or audit.get("frozen_receipt_sha256") != frozen_receipt_sha256
            or audit.get("official_base_checkpoint_sha256") != OFFICIAL_BASE_SHA256
            or audit.get("speaker_scope") != "SHOW_All"
            or not strict_json_equal(audit.get("speaker_rows"), [0, 1, 2, 3])
            or audit.get("vq_models_in_training_graph") is not False
            or audit.get("all_model_state_tensors_finite") is not True
            or audit.get("model_state_semantic_sha256") != semantic_sha
            or audit.get("trajectory_anchor_match") is not None
            or audit.get("trajectory_probe_verified") is not True
        ):
            raise LiveValidationContractError("candidate checkpoint audit changed")
        return {
            "model_state_tensors": len(state),
            "model_state_schema_sha256": schema_sha,
            "model_state_semantic_sha256": semantic_sha,
            "audit": dict(audit),
        }

    def _run_validation_helper(
        self, operation: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        request = canonical_json_bytes(
            {"operation": operation, "payload": dict(payload)}
        )
        if len(request) > VALIDATION_HELPER_MAX_INPUT_BYTES:
            raise LiveValidationContractError(
                "isolated validation-helper request exceeds the byte bound"
            )
        command = [
            sys.executable,
            str(Path(__file__).resolve(strict=True)),
            "_validation-helper",
            "--runtime-validation-source-root",
            str(self.runtime_validation_source_root),
            "--expected-runtime-validation-contract-sha256",
            self.runtime_validation_contract_sha256,
            "--expected-runtime-validation-selector-sha256",
            self.runtime_validation_selector_sha256,
        ]
        try:
            process = subprocess.run(
                command,
                input=request,
                capture_output=True,
                timeout=VALIDATION_HELPER_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LiveValidationContractError(
                "isolated validation-helper execution failed"
            ) from error
        if (
            process.returncode != 0
            or process.stderr
            or not process.stdout
            or len(process.stdout) > VALIDATION_HELPER_MAX_OUTPUT_BYTES
        ):
            raise LiveValidationContractError(
                "isolated validation-helper did not return one bounded clean result"
            )
        result = strict_json_bytes(
            process.stdout, "isolated validation-helper result"
        )
        if not isinstance(result, dict) or set(result) != {"result"}:
            raise LiveValidationContractError(
                "isolated validation-helper result schema changed"
            )
        return result["result"]

    def validate_val_inputs(
        self, path: Path, expected_sha256: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            result = self._run_validation_helper(
                "validate_val_inputs",
                {
                    "path": str(path),
                    "expected_sha256": expected_sha256,
                },
            )
        except Exception as error:
            raise LiveValidationContractError("validation inputs were rejected") from error
        if not isinstance(result, dict) or set(result) != {"receipt", "coverage"}:
            raise LiveValidationContractError(
                "isolated validation-input result schema changed"
            )
        return result["receipt"], result["coverage"]

    def validate_pipeline(
        self,
        path: Path,
        expected_sha256: str,
        *,
        frozen: Mapping[str, Any],
        validation_evidence_source: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            result = self._run_validation_helper(
                "validate_pipeline",
                {
                    "path": str(path),
                    "expected_sha256": expected_sha256,
                    "expected_prerequisite_selection": frozen["dataset"][
                        "prerequisite_selection"
                    ],
                    "expected_source": dict(validation_evidence_source),
                },
            )
        except Exception as error:
            raise LiveValidationContractError("validation pipeline was rejected") from error
        if not isinstance(result, dict) or set(result) != {"receipt", "pipeline"}:
            raise LiveValidationContractError(
                "isolated validation-pipeline result schema changed"
            )
        receipt = result["receipt"]
        pipeline = result["pipeline"]
        selected = frozen["dataset"]["selected_prerequisite_sha256"]
        fixed = pipeline.get("fixed_checkpoints")
        if not isinstance(fixed, dict) or any(
            fixed.get(stage, {}).get("sha256") != selected[stage]
            for stage in ("face", "hands", "upper", "lower", "global")
        ):
            raise LiveValidationContractError(
                "validation pipeline differs from frozen five-stage prerequisites"
            )
        return receipt, pipeline


def _validate_topology_and_dataset(
    frozen: Mapping[str, Any], topology_mode: str
) -> dict[str, Any]:
    specification = SUPPORTED_TOPOLOGIES.get(topology_mode)
    if specification is None:
        raise LiveValidationContractError(
            "live adapter accepts only the W8G1024 or W8G2048 winner"
        )
    protocol = frozen.get("protocol")
    topology = frozen.get("topology")
    dataset = frozen.get("dataset")
    source = frozen.get("source")
    if not isinstance(protocol, dict) or not isinstance(topology, dict) or not isinstance(dataset, dict) or not isinstance(source, dict):
        raise LiveValidationContractError("frozen topology/dataset is malformed")
    distributed = protocol.get("distributed_topology")
    protocol_nodes = distributed.get("nodes") if isinstance(distributed, dict) else None
    source_clones = source.get("node_local_clones")
    expected_optimizer = {
        "name": "Adam",
        "learning_rate": specification["learning_rate"],
        "betas": [0.5, 0.999],
        "weight_decay": 0.0,
        "gradient_clip_norm": 0.99,
        "scheduler": "constant",
    }
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("scope") != "SemTalk Base only"
        or protocol.get("target_dataset") != "SHOW"
        or protocol.get("target_speaker_scope") != "All"
        or any(
            not exact_int_equal(protocol.get(key), expected)
            for key, expected in (
                ("node_count", specification["node_count"]),
                ("local_world_size", specification["local_world_size"]),
                ("world_size", specification["world_size"]),
                ("local_batch_size", specification["local_batch_size"]),
                ("global_batch_size", specification["global_batch_size"]),
                ("expected_updates_per_epoch", specification["updates_per_epoch"]),
                (
                    "expected_unique_samples_per_epoch",
                    specification["unique_samples_per_epoch"],
                ),
                ("epochs", 400),
            )
        )
        or not strict_json_equal(
            protocol.get("candidate_epochs"), list(CANDIDATE_EPOCHS)
        )
        or not strict_json_equal(protocol.get("trajectory_anchor_epochs"), [])
        or not strict_json_equal(protocol.get("optimizer"), expected_optimizer)
        or protocol.get("precision") != specification["precision"]
        or protocol.get("vq_models_in_training_graph") is not False
        or not isinstance(distributed, dict)
        or distributed.get("mode") != topology_mode
        or distributed.get("classification") != specification["classification"]
        or any(
            not exact_int_equal(distributed.get(key), specification[key])
            for key in ("node_count", "local_world_size", "world_size")
        )
        or not isinstance(protocol_nodes, list)
        or len(protocol_nodes) != specification["node_count"]
        or not isinstance(source_clones, list)
        or len(source_clones) != specification["node_count"]
        or type(distributed.get("master_port")) is not int
        or not (1 <= distributed["master_port"] <= 65_535)
        or not isinstance(distributed.get("master_addr"), str)
        or not distributed["master_addr"]
        or not isinstance(distributed.get("formal_run_id"), str)
        or not distributed["formal_run_id"]
    ):
        raise LiveValidationContractError("frozen selected topology protocol changed")

    topology_keys = {
        "format",
        "topology_mode",
        "classification",
        "backend",
        "node_count",
        "local_world_size",
        "world_size",
        "global_batch_size",
        "local_batch_size",
        "updates_per_epoch",
        "unique_samples_per_epoch",
        "master_addr",
        "master_port",
        "formal_run_id",
        "ranks",
        "receipt_sha256",
    }
    topology_unsigned = dict(topology)
    topology_claimed = topology_unsigned.pop("receipt_sha256", None)
    if (
        set(topology) != topology_keys
        or require_sha256(topology_claimed, "topology self-hash")
        != canonical_json_sha256(topology_unsigned)
        or topology.get("format") != "semtalk_show_base_topology_receipt_v1"
        or topology.get("topology_mode") != topology_mode
        or topology.get("classification") != specification["classification"]
        or any(
            not exact_int_equal(topology.get(key), specification[key])
            for key in (
                "node_count",
                "local_world_size",
                "world_size",
                "global_batch_size",
                "local_batch_size",
                "updates_per_epoch",
                "unique_samples_per_epoch",
            )
        )
        or topology.get("backend") != "nccl"
        or topology.get("master_addr") != distributed.get("master_addr")
        or topology.get("master_port") != distributed.get("master_port")
        or topology.get("formal_run_id") != distributed.get("formal_run_id")
        or not isinstance(topology.get("ranks"), list)
        or len(topology["ranks"]) != specification["world_size"]
    ):
        raise LiveValidationContractError("frozen topology receipt changed")
    expected_ranks: list[dict[str, Any]] = []
    for node_rank, node in enumerate(protocol_nodes):
        clone = source_clones[node_rank]
        if not isinstance(node, dict) or not isinstance(clone, dict):
            raise LiveValidationContractError("protocol node is malformed")
        slot = exact_int(node.get("host_slot"), "protocol host slot")
        if slot not in FORMAL_HOST_BY_SLOT:
            raise LiveValidationContractError("protocol host slot is not audited")
        hostname = FORMAL_HOST_BY_SLOT.get(slot)
        if (
            set(node) != {"node_rank", "host_slot", "hostname", "rank_range"}
            or set(clone) != {
                "node_rank", "host_slot", "hostname", "entrypoint", "branch"
            }
            or not exact_int_equal(node.get("node_rank"), node_rank)
            or node.get("hostname") != hostname
            or not exact_int_equal(clone.get("node_rank"), node_rank)
            or not exact_int_equal(clone.get("host_slot"), slot)
            or clone.get("hostname") != hostname
            or clone.get("branch") is not None
            or not strict_json_equal(
                node.get("rank_range"),
                list(
                    range(
                        node_rank * specification["local_world_size"],
                        (node_rank + 1) * specification["local_world_size"],
                    )
                ),
            )
        ):
            raise LiveValidationContractError("protocol host-slot binding changed")
        for local_rank in range(specification["local_world_size"]):
            expected_ranks.append(
                {
                    "rank": node_rank * specification["local_world_size"] + local_rank,
                    "local_rank": local_rank,
                    "node_rank": node_rank,
                    "host_slot": slot,
                    "hostname": hostname,
                    "master_addr": distributed["master_addr"],
                    "master_port": distributed["master_port"],
                    "formal_run_id": distributed["formal_run_id"],
                }
            )
    if not strict_json_equal(topology["ranks"], expected_ranks):
        raise LiveValidationContractError("frozen rank inventory changed")

    selection = dataset.get("prerequisite_selection")
    selected = dataset.get("selected_prerequisite_sha256")
    bindings = dataset.get("node_lmdb_inode_bindings")
    if (
        dataset.get("format")
        != "semtalk_show_base_selected_feature_dataset_receipt_v1"
        or not exact_int_equal(dataset.get("entries"), 127_286)
        or not exact_int_equal(dataset.get("train_clips"), 13_687)
        or dataset.get("split") != "train"
        or dataset.get("test_visible") is not False
        or dataset.get("prerequisite_source") != "show_val_selected_v1"
        or dataset.get("global_verified_not_consumed") is not True
        or not isinstance(selection, dict)
        or set(selection) != {"path", "sha256", "receipt_payload_sha256"}
        or not isinstance(selected, dict)
        or set(selected) != {"face", "hands", "upper", "lower", "global"}
        or dataset.get("lmdb_binding_scope")
        != "ordered_node_local_inode_bindings_with_global_content_sha256"
        or not isinstance(bindings, list)
        or len(bindings) != specification["node_count"]
    ):
        raise LiveValidationContractError("frozen selected dataset changed")
    for label, value in {
        "dataset summary": dataset.get("summary_sha256"),
        "dataset lineage": dataset.get("lineage_sha256"),
        "data.mdb": dataset.get("data_mdb_sha256"),
        "lock.mdb": dataset.get("lock_mdb_sha256"),
        "selection file": selection.get("sha256"),
        "selection payload": selection.get("receipt_payload_sha256"),
        **{f"selected {stage}": digest for stage, digest in selected.items()},
    }.items():
        require_sha256(value, f"{label} SHA-256")
    identity_keys = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
    for node_rank, row in enumerate(bindings):
        expected_node = protocol_nodes[node_rank]
        binding = row.get("binding") if isinstance(row, dict) else None
        files = binding.get("files") if isinstance(binding, dict) else None
        if (
            not isinstance(row, dict)
            or set(row) != {"node_rank", "host_slot", "hostname", "binding"}
            or not exact_int_equal(row.get("node_rank"), node_rank)
            or not exact_int_equal(
                row.get("host_slot"), expected_node["host_slot"]
            )
            or row.get("hostname") != expected_node["hostname"]
            or not isinstance(binding, dict)
            or set(binding) != {"format", "directory_identity", "files"}
            or binding.get("format") != "semtalk_show_base_lmdb_inode_binding_v1"
            or not isinstance(binding.get("directory_identity"), dict)
            or set(binding["directory_identity"]) != identity_keys
            or any(type(value) is not int or value < 0 for value in binding["directory_identity"].values())
            or not isinstance(files, dict)
            or set(files) != {"data.mdb", "lock.mdb"}
        ):
            raise LiveValidationContractError("dataset node host-slot binding changed")
        for filename, digest in (
            ("data.mdb", dataset["data_mdb_sha256"]),
            ("lock.mdb", dataset["lock_mdb_sha256"]),
        ):
            entry = files[filename]
            if (
                not isinstance(entry, dict)
                or set(entry) != {"sha256", "identity"}
                or entry.get("sha256") != digest
                or not isinstance(entry.get("identity"), dict)
                or set(entry["identity"]) != identity_keys
                or any(type(value) is not int or value < 0 for value in entry["identity"].values())
            ):
                raise LiveValidationContractError("dataset file binding changed")
    return {"mode": topology_mode, **specification, "topology_receipt_sha256": topology_claimed}


def _validate_schedule_and_trajectory(
    *,
    config: AuthorizeConfig,
    frozen: Mapping[str, Any],
    topology: Mapping[str, Any],
    hooks: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule_path, schedule, schedule_sha, schedule_bytes = verified_json(
        config.schedule,
        config.expected_schedule_sha256,
        "fresh Base schedule",
    )
    protocol = frozen["protocol"]
    long_contract = frozen.get("long_contract")
    protocol_schedule = protocol.get("schedule")
    protocol_trajectory = protocol.get("trajectory_anchor")
    trajectory_gate = protocol.get("trajectory_gate")
    if not isinstance(long_contract, dict) or long_contract.get("format") != "semtalk_show_base_long_contract_receipts_v1":
        raise LiveValidationContractError("frozen long contract changed")
    long_schedule = long_contract.get("schedule")
    trajectory = long_contract.get("trajectory_anchor")
    if (
        not isinstance(protocol_schedule, dict)
        or protocol_schedule.get("path") != str(schedule_path)
        or protocol_schedule.get("sha256") != schedule_sha
        or not isinstance(long_schedule, dict)
        or set(long_schedule)
        != {"path", "sha256", "payload_sha256", "format", "topology_source"}
        or long_schedule.get("path") != str(schedule_path)
        or long_schedule.get("sha256") != schedule_sha
        or long_schedule.get("payload_sha256") != canonical_json_sha256(schedule)
        or long_schedule.get("format") != FRESH_SCHEDULE_FORMAT
        or long_schedule.get("topology_source") != V14_TOPOLOGY_SOURCE
    ):
        raise LiveValidationContractError("schedule file/receipt lineage changed")
    training = schedule.get("training")
    selection = schedule.get("selection")
    initialization = schedule.get("initialization")
    topology_selection = schedule.get("topology_selection_contract")
    expected_trajectory_contract = {
        "mode": FRESH_TRAJECTORY_MODE,
        "external_anchor": False,
        "probe_optimizer_updates": 70,
        "probe_source": "matching_frozen_receipt_throughput_gate",
        "comparison": "byte_exact_model_state_semantic_sha256",
        "restart_from_official_initialization": True,
    }
    exact_keys(
        schedule,
        frozenset(
            {
                "format",
                "scope",
                "target_dataset",
                "target_speaker_scope",
                "initialization",
                "training",
                "topology_selection_contract",
                "candidate_epochs",
                "validation_waves",
                "selection",
                "trajectory_contract",
            }
        ),
        "fresh Base schedule",
    )
    exact_keys(
        initialization,
        frozenset(
            {
                "source",
                "checkpoint_sha256",
                "forbidden_epoch",
                "forbidden_sources",
            }
        ),
        "fresh Base schedule initialization",
    )
    exact_keys(
        training,
        frozenset(
            {
                "total_epochs",
                "topology_source",
                "topology_matrix",
                "loader_workers",
                "precision_source",
                "optimizer",
                "learning_rate_source",
                "betas",
                "weight_decay",
                "gradient_clip_norm",
                "scheduler",
                "seed",
                "vq_models_in_training_graph",
            }
        ),
        "fresh Base schedule training",
    )
    exact_keys(
        selection,
        frozenset({"split", "test_visible", "ordering", "direction", "test_runs"}),
        "fresh Base schedule selection",
    )
    exact_keys(
        topology_selection,
        frozenset(
            {
                "format",
                "training_semantics_source_commit",
                "training_semantics_source_tree",
                "validation_source_commit",
                "validation_source_tree",
                "selection_protocol_sha256",
                "candidate_modes",
                "candidate_epochs",
                "validation_measurements",
                "selection_tuple",
                "split",
                "test_visible",
                "native_nine_mode_role",
                "fresh_selected_mode_throughput_gate_required",
            }
        ),
        "V14 topology-selection binding",
    )
    if (
        schedule.get("format") != FRESH_SCHEDULE_FORMAT
        or (
            getattr(hooks, "require_exact_v14_schedule_bytes", True)
            and (
                schedule_sha != V14_SCHEDULE_SHA256
                or config.expected_schedule_sha256 != V14_SCHEDULE_SHA256
            )
        )
        or schedule.get("scope") != "SemTalk Base only"
        or schedule.get("target_dataset") != "SHOW"
        or schedule.get("target_speaker_scope") != "All"
        or not strict_json_equal(
            schedule.get("candidate_epochs"), list(CANDIDATE_EPOCHS)
        )
        or not strict_json_equal(
            schedule.get("validation_waves"),
            [list(wave) for wave in VALIDATION_WAVES],
        )
        or not strict_json_equal(
            schedule.get("trajectory_contract"), expected_trajectory_contract
        )
        or "trajectory_anchor_epochs" in schedule
        or not isinstance(training, dict)
        or not exact_int_equal(training.get("total_epochs"), 400)
        or training.get("topology_source") != V14_TOPOLOGY_SOURCE
        or not strict_json_equal(
            training.get("topology_matrix"),
            {mode: SUPPORTED_TOPOLOGIES[mode] for mode in V14_MODES},
        )
        or any(
            not strict_json_equal(
                hooks.topology_specs.get(mode), SUPPORTED_TOPOLOGIES[mode]
            )
            for mode in V14_MODES
        )
        or training.get("precision_source") != "selected_topology_matrix_entry"
        or training.get("learning_rate_source") != "selected_topology_matrix_entry"
        or training.get("optimizer") != "Adam"
        or not strict_json_equal(training.get("betas"), [0.5, 0.999])
        or not strict_json_equal(training.get("weight_decay"), 0.0)
        or training.get("gradient_clip_norm") != 0.99
        or training.get("scheduler") != "constant"
        or type(training.get("seed")) is not int
        or type(training.get("loader_workers")) is not int
        or training.get("loader_workers") < 0
        or training.get("vq_models_in_training_graph") is not False
        or not isinstance(initialization, dict)
        or initialization.get("source") != "released_all_speakers_v1"
        or initialization.get("checkpoint_sha256") != OFFICIAL_BASE_SHA256
        or not exact_int_equal(initialization.get("forbidden_epoch"), 30)
        or not strict_json_equal(
            initialization.get("forbidden_sources"),
            ["Speaker2", "SemGate", "Sparse"],
        )
        or not isinstance(selection, dict)
        or selection.get("split") != "val"
        or selection.get("test_visible") is not False
        or not strict_json_equal(selection.get("ordering"), ["FGD", "epoch"])
        or not strict_json_equal(selection.get("direction"), ["min", "min"])
        or not exact_int_equal(selection.get("test_runs"), 1)
        or not isinstance(topology_selection, dict)
        or topology_selection.get("format")
        != "semtalk_show_base_v14_formal_selection_binding_v1"
        or topology_selection.get("training_semantics_source_commit")
        != TRAINING_SEMANTICS_SOURCE_COMMIT
        or topology_selection.get("training_semantics_source_tree")
        != TRAINING_SEMANTICS_SOURCE_TREE
        or topology_selection.get("validation_source_commit")
        != VALIDATION_EVIDENCE_SOURCE_COMMIT
        or topology_selection.get("validation_source_tree")
        != VALIDATION_EVIDENCE_SOURCE_TREE
        or topology_selection.get("selection_protocol_sha256")
        != V14_SELECTION_PROTOCOL_SHA256
        or not strict_json_equal(
            topology_selection.get("candidate_modes"), list(V14_MODES)
        )
        or not strict_json_equal(
            topology_selection.get("candidate_epochs"),
            list(V14_TOPOLOGY_SELECTION_EPOCHS),
        )
        or not exact_int_equal(
            topology_selection.get("validation_measurements"), 12
        )
        or not strict_json_equal(
            topology_selection.get("selection_tuple"),
            [
                "validation_diffsheg_fgd_ascending",
                "epoch_ascending",
                "topology_mode_lexicographic_ascending",
            ],
        )
        or topology_selection.get("split") != "val"
        or topology_selection.get("test_visible") is not False
        or topology_selection.get("native_nine_mode_role")
        != "optional_audit_evidence_never_v14_decision_authority"
        or topology_selection.get(
            "fresh_selected_mode_throughput_gate_required"
        )
        is not True
    ):
        raise LiveValidationContractError("fresh Base schedule semantics changed")

    if not isinstance(trajectory, dict):
        raise LiveValidationContractError("fresh trajectory contract is absent")
    trajectory_unsigned = {
        key: value
        for key, value in trajectory.items()
        if key not in {"path", "sha256", "payload_sha256", "entries"}
    }
    trajectory_sha = canonical_json_sha256(trajectory_unsigned)
    dataset = frozen["dataset"]
    expected_binding = {
        "format": FRESH_TRAJECTORY_FORMAT,
        "mode": FRESH_TRAJECTORY_MODE,
        "schedule_sha256": schedule_sha,
        "official_base_checkpoint_sha256": OFFICIAL_BASE_SHA256,
        "dataset_split": "train",
        "test_visible": False,
        "dataset_summary_sha256": dataset["summary_sha256"],
        "feature_lineage_sha256": dataset["lineage_sha256"],
        "data_mdb_sha256": dataset["data_mdb_sha256"],
        "lock_mdb_sha256": dataset["lock_mdb_sha256"],
        "prerequisite_selection_sha256": dataset["prerequisite_selection"]["sha256"],
        "selected_prerequisite_sha256": {
            stage: dataset["selected_prerequisite_sha256"][stage]
            for stage in sorted(dataset["selected_prerequisite_sha256"])
        },
        "probe_optimizer_updates": 70,
        "probe_source": "matching_frozen_receipt_throughput_gate",
        "comparison": "byte_exact_rank_local_model_buffers_all_rank_parameters_adam_rng_and_sample_order_v3",
        "precision": topology["precision"],
        "learning_rate": topology["learning_rate"],
        "topology_mode": topology["mode"],
        "topology_classification": topology["classification"],
        "world_size": topology["world_size"],
        "node_count": topology["node_count"],
        "local_world_size": topology["local_world_size"],
        "local_batch_size": topology["local_batch_size"],
        "global_batch_size": topology["global_batch_size"],
        "updates_per_epoch": topology["updates_per_epoch"],
        "unique_samples_per_epoch": topology["unique_samples_per_epoch"],
    }
    # These fields are content-dependent and remain mandatory; compare every
    # result-affecting fixed field above and require the remaining values to be
    # present in the exact producer binding before checking its canonical SHA.
    for key, expected in expected_binding.items():
        if trajectory_unsigned.get(key) != expected:
            raise LiveValidationContractError(f"fresh trajectory {key} changed")
    for key in (
        "dataset_receipt_payload_sha256",
        "lmdb_binding_scope",
        "canonical_dataset_evidence",
        "seed",
        "loader_workers",
    ):
        if key not in trajectory_unsigned:
            raise LiveValidationContractError(f"fresh trajectory lacks {key}")
    require_sha256(
        trajectory_unsigned["dataset_receipt_payload_sha256"],
        "fresh trajectory dataset payload SHA-256",
    )
    exact_int(trajectory_unsigned["seed"], "fresh trajectory seed")
    exact_int(trajectory_unsigned["loader_workers"], "fresh trajectory loader workers")
    if (
        trajectory_unsigned["seed"] != training["seed"]
        or trajectory_unsigned["loader_workers"] != training["loader_workers"]
        or trajectory_unsigned["loader_workers"] > 16
        or
        trajectory.get("path") is not None
        or trajectory.get("entries") != {}
        or trajectory.get("sha256") != trajectory_sha
        or trajectory.get("payload_sha256") != trajectory_sha
        or not isinstance(protocol_trajectory, dict)
        or protocol_trajectory
        != {
            "mode": FRESH_TRAJECTORY_MODE,
            "path": None,
            "sha256": trajectory_sha,
            "external": False,
        }
        or trajectory_gate
        != {
            "required": True,
            "probe_optimizer_updates": 70,
            "probe_source": "matching_frozen_receipt_throughput_gate",
            "comparison": "byte_exact_rank_local_model_buffers_all_rank_parameters_adam_rng_and_sample_order_v3",
        }
    ):
        raise LiveValidationContractError("fresh trajectory closure changed")
    return (
        {
            "path": str(schedule_path),
            "sha256": schedule_sha,
            "bytes": schedule_bytes,
            "payload_sha256": canonical_json_sha256(schedule),
        },
        {
            "format": FRESH_TRAJECTORY_FORMAT,
            "mode": FRESH_TRAJECTORY_MODE,
            "sha256": trajectory_sha,
            "payload_sha256": trajectory_sha,
            "probe_optimizer_updates": 70,
        },
    )


def _validate_frozen(
    *,
    config: AuthorizeConfig,
    ready: Mapping[str, Any],
    hooks: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    reference = exact_keys(ready.get("frozen_inputs"), FROZEN_REFERENCE_KEYS, "ready frozen-input reference")
    expected_path = config.train_root / "frozen_inputs.json"
    if Path(str(reference.get("path"))) != expected_path:
        raise LiveValidationContractError("ready frozen-input path changed")
    if reference.get("sha256") != config.expected_frozen_inputs_sha256:
        raise LiveValidationContractError("ready frozen-input external SHA changed")
    path, frozen, file_sha, size = verified_json(
        expected_path,
        config.expected_frozen_inputs_sha256,
        "formal frozen inputs",
    )
    exact_keys(frozen, FROZEN_KEYS, "formal frozen inputs")
    unsigned = dict(frozen)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        frozen.get("format") != FROZEN_FORMAT
        or frozen.get("run_purpose") != "formal_training"
        or not strict_json_equal(
            frozen.get("target_epochs"), list(CANDIDATE_EPOCHS)
        )
        or require_sha256(claimed, "frozen-input self-hash")
        != canonical_json_sha256(unsigned)
        or reference.get("receipt_payload_sha256") != claimed
        or ready.get("frozen_receipt_sha256") != claimed
    ):
        raise LiveValidationContractError("formal frozen-input receipt changed")
    training_source = hooks.validate_training_source(frozen.get("source"))
    (
        validation_evidence_source,
        runtime_validation_source,
        runtime_validation_proof,
    ) = hooks.validate_validation_sources()
    topology = _validate_topology_and_dataset(frozen, config.topology_mode)
    schedule, trajectory = _validate_schedule_and_trajectory(
        config=config,
        frozen=frozen,
        topology=topology,
        hooks=hooks,
    )
    hooks.validate_long_contract(
        frozen=frozen,
        config=config,
        topology_mode=config.topology_mode,
    )
    protocol_reference = exact_keys(
        ready.get("protocol"), PROTOCOL_REFERENCE_KEYS, "ready protocol reference"
    )
    if (
        protocol_reference.get("format") != PROTOCOL_FORMAT
        or protocol_reference.get("payload_sha256")
        != canonical_json_sha256(frozen["protocol"])
        or ready.get("schedule_sha256") != schedule["sha256"]
        or ready.get("trajectory_anchor_sha256") != trajectory["sha256"]
    ):
        raise LiveValidationContractError("ready frozen protocol lineage changed")
    official = frozen.get("official_base")
    speaker = frozen.get("speaker_initialization")
    if (
        not isinstance(official, dict)
        or official.get("sha256") != OFFICIAL_BASE_SHA256
        or official.get("source") != "released_all_speakers_v1"
        or official.get("speaker_scope") != "All-Speakers"
        or official.get("training_dataset") != "BEAT2"
        or official.get("all_model_state_tensors_finite") is not True
        or official.get("strict_state_dict_load") is not True
        or not isinstance(speaker, dict)
        or speaker.get("format") != "semtalk_show_official_speaker_mean_init_v1"
        or speaker.get("source_rows") != 25
        or not strict_json_equal(
            speaker.get("target_show_rows"), [0, 1, 2, 3]
        )
        or speaker.get("other_state_unchanged") is not True
    ):
        raise LiveValidationContractError("official All-Speakers initialization changed")
    return (
        {
            "path": str(path),
            "sha256": file_sha,
            "bytes": size,
            "receipt_payload_sha256": claimed,
        },
        training_source,
        validation_evidence_source,
        runtime_validation_source,
        runtime_validation_proof,
        topology,
        {"schedule": schedule, "trajectory": trajectory, "payload": frozen},
    )


def _validate_snapshot(
    *,
    config: AuthorizeConfig,
    ready: Mapping[str, Any],
    frozen_receipt_sha256: str,
    topology: Mapping[str, Any],
    lineage: Mapping[str, Any],
    hooks: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference = exact_keys(
        ready.get("candidate_manifest"),
        MANIFEST_REFERENCE_KEYS,
        "ready manifest reference",
    )
    expected_path = (
        config.train_root
        / "candidate_manifest_snapshots"
        / f"epoch-{config.epoch:04d}.json"
    )
    if (
        reference.get("immutable_snapshot") is not True
        or Path(str(reference.get("path"))) != expected_path
        or Path(str(reference.get("live_path")))
        != config.train_root / "candidate_manifest.json"
    ):
        raise LiveValidationContractError("immutable snapshot reference changed")
    path, snapshot, snapshot_sha, snapshot_size = verified_json(
        expected_path,
        str(reference.get("sha256_at_ready")),
        "candidate manifest snapshot",
    )
    exact_keys(snapshot, SNAPSHOT_KEYS, "candidate manifest snapshot")
    prefix_length = CANDIDATE_EPOCHS.index(config.epoch) + 1
    expected_epochs = list(CANDIDATE_EPOCHS[:prefix_length])
    entries = snapshot.get("entries")
    if (
        snapshot.get("format") != MANIFEST_FORMAT
        or snapshot.get("status") != "running"
        or not strict_json_equal(
            snapshot.get("candidate_epochs"), list(CANDIDATE_EPOCHS)
        )
        or snapshot.get("frozen_receipt_sha256") != frozen_receipt_sha256
        or snapshot.get("schedule_sha256") != lineage["schedule"]["sha256"]
        or snapshot.get("trajectory_anchor_sha256")
        != lineage["trajectory"]["sha256"]
        or snapshot.get("trajectory_mode") != FRESH_TRAJECTORY_MODE
        or snapshot.get("trajectory_probe_verified") is not True
        or not isinstance(entries, list)
        or len(entries) != prefix_length
        or any(
            not isinstance(entry, dict)
            or not exact_int_equal(entry.get("epoch"), expected_epoch)
            for entry, expected_epoch in zip(entries, expected_epochs)
        )
        or snapshot.get("entries_sha256") != canonical_json_sha256(entries)
        or reference.get("entries_sha256_at_ready")
        != snapshot.get("entries_sha256")
    ):
        raise LiveValidationContractError("candidate manifest prefix changed")
    probe = hooks.validate_trajectory_probe(
        snapshot.get("trajectory_probe"), config.topology_mode
    )
    throughput = exact_keys(
        snapshot.get("throughput_gate"),
        THROUGHPUT_REFERENCE_KEYS,
        "manifest throughput reference",
    )
    if (
        throughput.get("topology_mode") != config.topology_mode
        or throughput.get("trajectory_mode") != FRESH_TRAJECTORY_MODE
        or throughput.get("trajectory_probe") != probe
    ):
        raise LiveValidationContractError("manifest trajectory probe changed")
    normalized_throughput = hooks.validate_throughput(
        reference=throughput,
        frozen=lineage["payload"],
        topology_mode=config.topology_mode,
    )
    for entry_epoch, entry_value in zip(expected_epochs, entries):
        entry = exact_keys(
            entry_value, ENTRY_KEYS, f"manifest entry e{entry_epoch}"
        )
        expected_relative = (
            f"candidates/base_official_adapt_epoch_{entry_epoch:02d}.bin"
        )
        if (
            not exact_int_equal(entry.get("epoch"), entry_epoch)
            or not exact_int_equal(
                entry.get("optimizer_updates"),
                entry_epoch * topology["updates_per_epoch"],
            )
            or entry.get("checkpoint") != expected_relative
            or type(entry.get("checkpoint_bytes")) is not int
            or entry["checkpoint_bytes"] <= 0
            or not strict_json_equal(
                entry.get("checkpoint_container_schema"), ["audit", "model_state"]
            )
            or type(entry.get("model_state_tensors")) is not int
            or entry["model_state_tensors"] <= 0
            or not isinstance(entry.get("model_state_schema_sha256"), str)
            or SHA256_RE.fullmatch(entry["model_state_schema_sha256"]) is None
            or not isinstance(entry.get("model_state_semantic_sha256"), str)
            or SHA256_RE.fullmatch(entry["model_state_semantic_sha256"]) is None
            or not isinstance(entry.get("checkpoint_sha256"), str)
            or SHA256_RE.fullmatch(entry["checkpoint_sha256"]) is None
            or entry.get("all_model_state_tensors_finite") is not True
            or entry.get("frozen_receipt_sha256") != frozen_receipt_sha256
            or entry.get("trajectory_anchor_match") is not None
            or entry.get("trajectory_probe_verified") is not True
        ):
            raise LiveValidationContractError(f"manifest entry e{entry_epoch} changed")
    return (
        {
            "path": str(path),
            "sha256": snapshot_sha,
            "bytes": snapshot_size,
            "entries_sha256": snapshot["entries_sha256"],
            "entry_count": len(entries),
        },
        dict(entries[-1]),
        normalized_throughput,
    )


def _validate_ready(
    config: AuthorizeConfig,
    hooks: Any,
) -> dict[str, Any]:
    train_root = canonical_existing_directory(config.train_root, "training root")
    if train_root != config.train_root:
        raise LiveValidationContractError("training root is not canonical")
    if type(config.epoch) is not int or config.epoch not in CANDIDATE_EPOCHS:
        raise LiveValidationContractError("epoch is outside the formal 22 candidates")
    if config.topology_mode not in SUPPORTED_TOPOLOGIES:
        raise LiveValidationContractError("unsupported live-validation topology")
    ready_path = train_root / "candidate_receipts" / f"epoch-{config.epoch:04d}.json"
    ready_path, ready_bytes, ready_file_sha, ready_size = safe_regular_bytes(
        ready_path, f"candidate e{config.epoch} ready receipt"
    )
    ready = strict_json_bytes(ready_bytes, f"candidate e{config.epoch} ready receipt")
    exact_keys(ready, READY_KEYS, "candidate-ready receipt")
    unsigned_ready = dict(ready)
    claimed_ready_payload = unsigned_ready.pop("receipt_payload_sha256", None)
    updates_per_epoch = SUPPORTED_TOPOLOGIES[config.topology_mode]["updates_per_epoch"]
    if (
        ready.get("format") != READY_FORMAT
        or ready.get("status") != "ready"
        or ready.get("selection_eligible") is not False
        or ready.get("test_visible") is not False
        or not exact_int_equal(ready.get("epoch"), config.epoch)
        or not exact_int_equal(
            ready.get("optimizer_updates"), config.epoch * updates_per_epoch
        )
        or ready.get("trajectory_anchor_match") is not None
        or require_sha256(claimed_ready_payload, "candidate-ready self-hash")
        != canonical_json_sha256(unsigned_ready)
    ):
        raise LiveValidationContractError("candidate-ready receipt changed")
    finite_number(ready.get("published_unix"), "ready publication time", positive=True)

    (
        frozen_artifact,
        training_source,
        validation_evidence_source,
        runtime_validation_source,
        runtime_validation_proof,
        topology,
        lineage,
    ) = _validate_frozen(
        config=config, ready=ready, hooks=hooks
    )
    manifest_artifact, current_entry, throughput = _validate_snapshot(
        config=config,
        ready=ready,
        frozen_receipt_sha256=frozen_artifact["receipt_payload_sha256"],
        topology=topology,
        lineage=lineage,
        hooks=hooks,
    )
    checkpoint_reference = exact_keys(
        ready.get("candidate_checkpoint"),
        CHECKPOINT_REFERENCE_KEYS,
        "ready checkpoint reference",
    )
    relative = f"candidates/base_official_adapt_epoch_{config.epoch:02d}.bin"
    checkpoint_path = train_root / relative
    if (
        checkpoint_reference.get("relative_path") != relative
        or Path(str(checkpoint_reference.get("path"))) != checkpoint_path
        or type(checkpoint_reference.get("bytes")) is not int
        or checkpoint_reference["bytes"] <= 0
        or type(checkpoint_reference.get("model_state_tensors")) is not int
        or checkpoint_reference["model_state_tensors"] <= 0
    ):
        raise LiveValidationContractError("ready checkpoint path/size changed")
    for key in (
        "sha256",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
    ):
        require_sha256(checkpoint_reference.get(key), f"ready checkpoint {key}")
    checkpoint_path, checkpoint_sha, checkpoint_bytes = safe_regular_hash(
        checkpoint_path, "candidate checkpoint"
    )
    if (
        checkpoint_sha != checkpoint_reference["sha256"]
        or checkpoint_bytes != checkpoint_reference["bytes"]
        or current_entry.get("checkpoint") != relative
        or current_entry.get("checkpoint_sha256") != checkpoint_sha
        or current_entry.get("checkpoint_bytes") != checkpoint_bytes
        or not exact_int_equal(
            current_entry.get("model_state_tensors"),
            checkpoint_reference["model_state_tensors"],
        )
        or current_entry.get("model_state_schema_sha256")
        != checkpoint_reference.get("model_state_schema_sha256")
        or current_entry.get("model_state_semantic_sha256")
        != checkpoint_reference.get("model_state_semantic_sha256")
    ):
        raise LiveValidationContractError("ready/snapshot checkpoint closure changed")
    verified_checkpoint = hooks.verify_checkpoint(
        path=checkpoint_path,
        expected_sha256=checkpoint_sha,
        expected_bytes=checkpoint_bytes,
        epoch=config.epoch,
        updates_per_epoch=updates_per_epoch,
        frozen_receipt_sha256=frozen_artifact["receipt_payload_sha256"],
    )
    if (
        not exact_int_equal(
            verified_checkpoint.get("model_state_tensors"),
            checkpoint_reference["model_state_tensors"],
        )
        or verified_checkpoint.get("model_state_schema_sha256")
        != checkpoint_reference.get("model_state_schema_sha256")
        or verified_checkpoint.get("model_state_semantic_sha256")
        != checkpoint_reference.get("model_state_semantic_sha256")
    ):
        raise LiveValidationContractError("checkpoint semantic replay changed")

    val_inputs_receipt, coverage = hooks.validate_val_inputs(
        config.val_inputs, config.expected_val_inputs_sha256
    )
    pipeline_receipt, pipeline = hooks.validate_pipeline(
        config.pipeline,
        config.expected_pipeline_sha256,
        frozen=lineage["payload"],
        validation_evidence_source=validation_evidence_source,
    )
    pipeline_source = pipeline.get("source")
    if (
        not isinstance(pipeline_source, dict)
        or not strict_json_equal(
            pipeline_source, validation_evidence_source
        )
    ):
        raise LiveValidationContractError(
            "validation pipeline source differs from pinned 4066f20 evidence"
        )
    for value, label in (
        (Path(val_inputs_receipt["path"]), "validation inputs"),
        (Path(pipeline_receipt["path"]), "validation pipeline"),
        (config.output, "work authority output"),
    ):
        reject_test_path(value, label)
    expected_output_name = f"epoch-{config.epoch:04d}.json"
    if config.output.name != expected_output_name:
        raise LiveValidationContractError(
            f"work authority output must be named {expected_output_name}"
        )
    return {
        "ready_receipt": {
            "path": str(ready_path),
            "sha256": ready_file_sha,
            "bytes": ready_size,
            "receipt_payload_sha256": claimed_ready_payload,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "relative_path": relative,
            "sha256": checkpoint_sha,
            "bytes": checkpoint_bytes,
            "model_state_tensors": checkpoint_reference["model_state_tensors"],
            "model_state_schema_sha256": checkpoint_reference[
                "model_state_schema_sha256"
            ],
            "model_state_semantic_sha256": checkpoint_reference[
                "model_state_semantic_sha256"
            ],
        },
        "manifest_snapshot": manifest_artifact,
        "manifest_entry": current_entry,
        "frozen_inputs": frozen_artifact,
        "producer_source": training_source,
        "validation_evidence_source": validation_evidence_source,
        "runtime_validation_source": runtime_validation_source,
        "runtime_validation_proof": runtime_validation_proof,
        "selected_topology": topology,
        "selected_prerequisite_sha256": dict(
            lineage["payload"]["dataset"]["selected_prerequisite_sha256"]
        ),
        "protocol": {
            "format": PROTOCOL_FORMAT,
            "payload_sha256": canonical_json_sha256(lineage["payload"]["protocol"]),
        },
        "schedule": lineage["schedule"],
        "trajectory_contract": lineage["trajectory"],
        "throughput_gate": throughput,
        "val_inputs_receipt": val_inputs_receipt,
        "coverage": coverage,
        "pipeline_receipt": pipeline_receipt,
        "pipeline_source": pipeline_source,
        "inference_entrypoint": pipeline["inference_entrypoint"],
    }


def authorize_candidate(config: AuthorizeConfig, *, hooks: Any | None = None) -> dict[str, Any]:
    if hooks is None:
        hooks = ValidationHooks(
            producer_source_root=config.producer_source_root,
            expected_producer_trainer_sha256=(
                config.expected_producer_trainer_sha256
            ),
            expected_producer_contract_sha256=(
                config.expected_producer_contract_sha256
            ),
            validation_evidence_source_root=(
                config.validation_evidence_source_root
            ),
            expected_validation_evidence_contract_sha256=(
                config.expected_validation_evidence_contract_sha256
            ),
            expected_validation_evidence_selector_sha256=(
                config.expected_validation_evidence_selector_sha256
            ),
            runtime_validation_source_root=(
                config.runtime_validation_source_root
            ),
            expected_runtime_validation_contract_sha256=(
                config.expected_runtime_validation_contract_sha256
            ),
            expected_runtime_validation_selector_sha256=(
                config.expected_runtime_validation_selector_sha256
            ),
        )
    closure = _validate_ready(config, hooks)
    body = _work_authority_body(config, closure, published_unix=time.time())
    return write_new_json(config.output, body)


def _work_authority_body(
    config: AuthorizeConfig,
    closure: Mapping[str, Any],
    *,
    published_unix: Any,
) -> dict[str, Any]:
    published = finite_number(
        published_unix, "work authority publication time", positive=True
    )
    body = {
        "format": WORK_FORMAT,
        "status": "ready",
        "split": "val",
        "test_visible": False,
        "selection_eligible": False,
        "candidate_epoch": config.epoch,
        "optimizer_updates": closure["manifest_entry"]["optimizer_updates"],
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "producer_ready_receipt": closure["ready_receipt"],
        "producer_manifest_snapshot": closure["manifest_snapshot"],
        "producer_manifest_entry": closure["manifest_entry"],
        "candidate_checkpoint": closure["checkpoint"],
        "frozen_inputs": closure["frozen_inputs"],
        "protocol": closure["protocol"],
        "schedule": closure["schedule"],
        "trajectory_contract": closure["trajectory_contract"],
        "throughput_gate": closure["throughput_gate"],
        "producer_source": closure["producer_source"],
        "validation_evidence_source": closure[
            "validation_evidence_source"
        ],
        "runtime_validation_source": closure[
            "runtime_validation_source"
        ],
        "runtime_validation_proof": closure[
            "runtime_validation_proof"
        ],
        "selected_topology": closure["selected_topology"],
        "selected_prerequisite_sha256": closure[
            "selected_prerequisite_sha256"
        ],
        "val_inputs_receipt": closure["val_inputs_receipt"],
        "coverage": closure["coverage"],
        "pipeline_receipt": closure["pipeline_receipt"],
        "pipeline_source": closure["pipeline_source"],
        "inference_entrypoint": closure["inference_entrypoint"],
        "execution_contract": {
            "purpose": "single_candidate_diffsheg_validation_only",
            "split": "val",
            "test_visible": False,
            "expected_shards": 8,
            "candidate_epochs_in_work_item": [config.epoch],
            "may_publish_standard_22_candidate_measurement_before_e400": False,
            "may_influence_training": False,
            "requires_guarded_runner": True,
        },
        "published_unix": published,
    }
    body["receipt_payload_sha256"] = canonical_json_sha256(body)
    return body


def _load_work_authority(path: Path, expected_epoch: int) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, encoded, digest, size = safe_regular_bytes(
        path, "work authority"
    )
    payload = strict_json_bytes(encoded, "work authority")
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_payload_sha256", None)
    if (
        payload.get("format") != WORK_FORMAT
        or payload.get("status") != "ready"
        or payload.get("split") != "val"
        or payload.get("test_visible") is not False
        or payload.get("selection_eligible") is not False
        or not exact_int_equal(payload.get("candidate_epoch"), expected_epoch)
        or not strict_json_equal(
            payload.get("candidate_epochs"), list(CANDIDATE_EPOCHS)
        )
        or require_sha256(claimed, "work authority self-hash")
        != canonical_json_sha256(unsigned)
    ):
        raise LiveValidationContractError(
            f"work authority e{expected_epoch} changed"
        )
    return artifact(resolved, digest, size), payload


def _validate_epoch_metrics(
    *,
    train_root: Path,
    status: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    expected_path = train_root / "epoch_metrics.jsonl"
    if Path(str(status.get("epoch_metrics_jsonl"))) != expected_path:
        raise LiveValidationContractError("final epoch-metrics path changed")
    path, encoded, digest, size = safe_regular_bytes(
        expected_path, "final epoch metrics"
    )
    if digest != require_sha256(
        status.get("epoch_metrics_sha256"), "final epoch-metrics SHA-256"
    ):
        raise LiveValidationContractError("final epoch-metrics SHA-256 changed")
    if not encoded.endswith(b"\n"):
        raise LiveValidationContractError("final epoch metrics is not newline sealed")
    lines = encoded.splitlines()
    if len(lines) != 400 or not exact_int_equal(
        status.get("epoch_metrics_records"), 400
    ):
        raise LiveValidationContractError("final epoch metrics is not exact 400-way")
    for epoch, line in enumerate(lines, 1):
        record = strict_json_bytes(line, f"epoch metric e{epoch}")
        exact_keys(record, EPOCH_METRIC_KEYS, f"epoch metric e{epoch}")
        metrics = record.get("metrics")
        if (
            record.get("format") != "semtalk_show_base_long_epoch_metric_v1"
            or not exact_int_equal(record.get("epoch"), epoch)
            or not exact_int_equal(
                record.get("optimizer_updates"),
                epoch * topology["updates_per_epoch"],
            )
            or not exact_int_equal(
                record.get("updates_per_epoch"), topology["updates_per_epoch"]
            )
            or not strict_json_equal(
                record.get("learning_rate"), topology["learning_rate"]
            )
            or record.get("all_finite") is not True
            or not isinstance(metrics, dict)
            or not metrics
        ):
            raise LiveValidationContractError(f"epoch metric e{epoch} changed")
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise LiveValidationContractError("epoch metric key changed")
            finite_number(value, f"epoch metric e{epoch} {key}")
        finite_number(
            record.get("completed_unix"),
            f"epoch metric e{epoch} completion time",
            positive=True,
        )
    return artifact(path, digest, size)


def _validate_resume_receipt(
    *,
    train_root: Path,
    status: Mapping[str, Any],
    topology: Mapping[str, Any],
    frozen_receipt_sha256: str,
    schedule_sha256: str,
    trajectory_sha256: str,
    final_entry: Mapping[str, Any],
) -> dict[str, Any]:
    expected_receipt_path = train_root / "resume" / "latest_resume.json"
    if Path(str(status.get("resume_receipt"))) != expected_receipt_path:
        raise LiveValidationContractError("final resume receipt path changed")
    receipt_path, receipt, receipt_sha, receipt_size = verified_json(
        expected_receipt_path,
        str(status.get("resume_receipt_sha256")),
        "final resume receipt",
    )
    exact_keys(receipt, RESUME_RECEIPT_KEYS, "final resume receipt")
    resume_path = train_root / "resume" / "latest_resume.bin"
    candidate_path = train_root / str(final_entry["checkpoint"])
    candidate = receipt.get("candidate_checkpoint")
    if (
        receipt.get("format")
        != "semtalk_show_base_official_adapt_long_resume_receipt_v1"
        or receipt.get("status") != "complete"
        or not exact_int_equal(receipt.get("completed_epochs"), 400)
        or not exact_int_equal(
            receipt.get("optimizer_updates"),
            400 * topology["updates_per_epoch"],
        )
        or Path(str(receipt.get("path"))) != resume_path
        or type(receipt.get("bytes")) is not int
        or receipt["bytes"] <= 0
        or not isinstance(candidate, dict)
        or set(candidate) != {"path", "sha256", "model_state_semantic_sha256"}
        or Path(str(candidate.get("path"))) != candidate_path
        or candidate.get("sha256") != final_entry.get("checkpoint_sha256")
        or candidate.get("model_state_semantic_sha256")
        != final_entry.get("model_state_semantic_sha256")
        or not exact_int_equal(
            receipt.get("rank_rng_states"), topology["world_size"]
        )
        or receipt.get("frozen_receipt_sha256") != frozen_receipt_sha256
        or receipt.get("schedule_sha256") != schedule_sha256
        or receipt.get("trajectory_anchor_sha256") != trajectory_sha256
        or receipt.get("trajectory_mode") != FRESH_TRAJECTORY_MODE
        or receipt.get("trajectory_probe_verified") is not True
    ):
        raise LiveValidationContractError("final resume receipt changed")
    resume_resolved, resume_sha, resume_size = safe_regular_hash(
        resume_path, "final resume payload"
    )
    if (
        resume_sha != require_sha256(receipt.get("sha256"), "resume payload SHA-256")
        or resume_size != receipt["bytes"]
    ):
        raise LiveValidationContractError("final resume payload changed")
    finite_number(
        receipt.get("completed_unix"), "resume completion time", positive=True
    )
    return {
        "receipt": artifact(receipt_path, receipt_sha, receipt_size),
        "payload": artifact(resume_resolved, resume_sha, resume_size),
    }


def _validate_complete_manifest_status(
    config: ReconcileConfig,
    *,
    frozen_receipt_sha256: str,
    topology: Mapping[str, Any],
    schedule_sha256: str,
    trajectory_sha256: str,
    throughput_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest_path, manifest, manifest_sha, manifest_bytes = verified_json(
        config.final_manifest,
        config.expected_final_manifest_sha256,
        "final candidate manifest",
    )
    final_manifest_keys = SNAPSHOT_KEYS | {"completed_epochs", "optimizer_updates"}
    exact_keys(manifest, frozenset(final_manifest_keys), "final candidate manifest")
    entries = manifest.get("entries")
    expected_updates = 400 * topology["updates_per_epoch"]
    if (
        manifest_path != config.common.train_root / "candidate_manifest.json"
        or manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("status") != "complete"
        or not strict_json_equal(
            manifest.get("candidate_epochs"), list(CANDIDATE_EPOCHS)
        )
        or not exact_int_equal(manifest.get("completed_epochs"), 400)
        or not exact_int_equal(manifest.get("optimizer_updates"), expected_updates)
        or manifest.get("frozen_receipt_sha256") != frozen_receipt_sha256
        or manifest.get("schedule_sha256") != schedule_sha256
        or manifest.get("trajectory_anchor_sha256") != trajectory_sha256
        or manifest.get("trajectory_mode") != FRESH_TRAJECTORY_MODE
        or manifest.get("trajectory_probe_verified") is not True
        or not strict_json_equal(
            manifest.get("throughput_gate"), throughput_gate
        )
        or not strict_json_equal(
            manifest.get("trajectory_probe"),
            throughput_gate.get("trajectory_probe"),
        )
        or not isinstance(entries, list)
        or len(entries) != len(CANDIDATE_EPOCHS)
        or any(
            not isinstance(entry, dict)
            or not exact_int_equal(entry.get("epoch"), expected_epoch)
            for entry, expected_epoch in zip(entries, CANDIDATE_EPOCHS)
        )
        or manifest.get("entries_sha256") != canonical_json_sha256(entries)
    ):
        raise LiveValidationContractError("final candidate manifest is incomplete")
    status_path, status, status_sha, status_bytes = verified_json(
        config.final_status,
        config.expected_final_status_sha256,
        "final training status",
    )
    exact_keys(status, FINAL_STATUS_KEYS, "final training status")
    if (
        status_path != config.common.train_root / "status.json"
        or status.get("format") != STATUS_FORMAT
        or status.get("status") != "complete"
        or not exact_int_equal(status.get("completed_epochs"), 400)
        or not exact_int_equal(status.get("optimizer_updates"), expected_updates)
        or not exact_int_equal(
            status.get("updates_per_epoch"), topology["updates_per_epoch"]
        )
        or status.get("candidate_manifest_sha256") != manifest_sha
        or status.get("frozen_receipt_sha256") != frozen_receipt_sha256
        or not exact_int_equal(status.get("world_size"), topology["world_size"])
        or not exact_int_equal(
            status.get("local_batch_size"), topology["local_batch_size"]
        )
        or not exact_int_equal(
            status.get("global_batch_size"), topology["global_batch_size"]
        )
        or status.get("all_training_state_finite") is not True
        or status.get("schedule_sha256") != schedule_sha256
        or status.get("trajectory_anchor_sha256") != trajectory_sha256
        or status.get("trajectory_mode") != FRESH_TRAJECTORY_MODE
        or status.get("trajectory_probe_verified") is not True
        or not strict_json_equal(status.get("throughput_gate"), throughput_gate)
        or not strict_json_equal(
            status.get("trajectory_probe"),
            throughput_gate.get("trajectory_probe"),
        )
        or not exact_int_equal(status.get("epoch_metrics_records"), 400)
    ):
        raise LiveValidationContractError("final training status is incomplete")
    started = finite_number(
        status.get("started_unix"), "training start time", positive=True
    )
    completed = finite_number(
        status.get("completed_unix"), "training completion time", positive=True
    )
    if completed < started:
        raise LiveValidationContractError("training completion precedes start")
    _validate_epoch_metrics(
        train_root=config.common.train_root,
        status=status,
        topology=topology,
    )
    _validate_resume_receipt(
        train_root=config.common.train_root,
        status=status,
        topology=topology,
        frozen_receipt_sha256=frozen_receipt_sha256,
        schedule_sha256=schedule_sha256,
        trajectory_sha256=trajectory_sha256,
        final_entry=entries[-1],
    )
    candidate_directory = canonical_existing_directory(
        config.common.train_root / "candidates", "final candidate directory"
    )
    expected_paths = {
        candidate_directory / f"base_official_adapt_epoch_{epoch:02d}.bin"
        for epoch in CANDIDATE_EPOCHS
    }
    observed_paths = set(candidate_directory.iterdir())
    if observed_paths != expected_paths or any(
        path.is_symlink() or not path.is_file() for path in observed_paths
    ):
        raise LiveValidationContractError("final candidate directory is not exact-once")
    return (
        artifact(manifest_path, manifest_sha, manifest_bytes),
        artifact(status_path, status_sha, status_bytes),
        [dict(entry) for entry in entries],
    )


def reconcile(config: ReconcileConfig, *, hooks: Any | None = None) -> dict[str, Any]:
    authority_dir = canonical_existing_directory(config.authority_dir, "authority directory")
    reject_test_path(authority_dir, "authority directory")
    expected_authority_paths = {
        authority_dir / f"epoch-{epoch:04d}.json" for epoch in CANDIDATE_EPOCHS
    }
    observed_authority_paths = set(authority_dir.iterdir())
    if observed_authority_paths != expected_authority_paths:
        raise LiveValidationContractError("live authority set is not exact 22-way coverage")
    if hooks is None:
        hooks = ValidationHooks(
            producer_source_root=config.common.producer_source_root,
            expected_producer_trainer_sha256=(
                config.common.expected_producer_trainer_sha256
            ),
            expected_producer_contract_sha256=(
                config.common.expected_producer_contract_sha256
            ),
            validation_evidence_source_root=(
                config.common.validation_evidence_source_root
            ),
            expected_validation_evidence_contract_sha256=(
                config.common.expected_validation_evidence_contract_sha256
            ),
            expected_validation_evidence_selector_sha256=(
                config.common.expected_validation_evidence_selector_sha256
            ),
            runtime_validation_source_root=(
                config.common.runtime_validation_source_root
            ),
            expected_runtime_validation_contract_sha256=(
                config.common.expected_runtime_validation_contract_sha256
            ),
            expected_runtime_validation_selector_sha256=(
                config.common.expected_runtime_validation_selector_sha256
            ),
        )
    authority_artifacts: list[dict[str, Any]] = []
    authority_payloads: list[dict[str, Any]] = []
    replayed: list[dict[str, Any]] = []
    for epoch in CANDIDATE_EPOCHS:
        path = authority_dir / f"epoch-{epoch:04d}.json"
        authority_artifact, authority_payload = _load_work_authority(path, epoch)
        per_epoch = AuthorizeConfig(
            **{
                **config.common.__dict__,
                "epoch": epoch,
                "output": path,
            }
        )
        replay = _validate_ready(per_epoch, hooks)
        expected_authority = _work_authority_body(
            per_epoch,
            replay,
            published_unix=authority_payload.get("published_unix"),
        )
        if not strict_json_equal(authority_payload, expected_authority):
            raise LiveValidationContractError(
                f"live authority e{epoch} differs from producer replay"
            )
        authority_artifacts.append(authority_artifact)
        authority_payloads.append(authority_payload)
        replayed.append(replay)
    first = replayed[0]
    common_fields = (
        "frozen_inputs",
        "producer_source",
        "validation_evidence_source",
        "runtime_validation_source",
        "runtime_validation_proof",
        "selected_topology",
        "selected_prerequisite_sha256",
        "protocol",
        "schedule",
        "trajectory_contract",
        "throughput_gate",
        "val_inputs_receipt",
        "coverage",
        "pipeline_receipt",
        "pipeline_source",
        "inference_entrypoint",
    )
    if any(
        not strict_json_equal(candidate[field], first[field])
        for candidate in replayed[1:]
        for field in common_fields
    ):
        raise LiveValidationContractError("live authorities mix producer/validation lineages")
    final_manifest, final_status, final_entries = _validate_complete_manifest_status(
        config,
        frozen_receipt_sha256=first["frozen_inputs"]["receipt_payload_sha256"],
        topology=first["selected_topology"],
        schedule_sha256=first["schedule"]["sha256"],
        trajectory_sha256=first["trajectory_contract"]["sha256"],
        throughput_gate=first["throughput_gate"],
    )
    for epoch, final_entry, replay in zip(CANDIDATE_EPOCHS, final_entries, replayed):
        if not strict_json_equal(final_entry, replay["manifest_entry"]):
            raise LiveValidationContractError(
                f"final manifest entry e{epoch} differs from live authority"
            )
    # Prefix snapshots must be monotonically append-only, not merely
    # individually self-consistent.
    previous_entries: list[Any] = []
    for replay in replayed:
        snapshot_path = Path(replay["manifest_snapshot"]["path"])
        _, snapshot, _, _ = verified_json(
            snapshot_path,
            replay["manifest_snapshot"]["sha256"],
            "reconciliation manifest snapshot",
        )
        entries = snapshot["entries"]
        if not strict_json_equal(entries[: len(previous_entries)], previous_entries):
            raise LiveValidationContractError("manifest snapshots are not append-only")
        previous_entries = entries
    body = {
        "format": RECONCILIATION_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "all_exact": True,
        "producer_manifest": final_manifest,
        "producer_status": final_status,
        "frozen_inputs": first["frozen_inputs"],
        "producer_source": first["producer_source"],
        "validation_evidence_source": first[
            "validation_evidence_source"
        ],
        "runtime_validation_source": first[
            "runtime_validation_source"
        ],
        "runtime_validation_proof": first[
            "runtime_validation_proof"
        ],
        "selected_topology": first["selected_topology"],
        "schedule": first["schedule"],
        "trajectory_contract": first["trajectory_contract"],
        "val_inputs_receipt": first["val_inputs_receipt"],
        "pipeline_receipt": first["pipeline_receipt"],
        "pipeline_source": first["pipeline_source"],
        "work_authorities": authority_artifacts,
        "test_evaluations_observed": 0,
        "completed_unix": time.time(),
    }
    body["receipt_payload_sha256"] = canonical_json_sha256(body)
    return write_new_json(config.common.output, body)


def _authorize_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("authorize")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--epoch", type=int, choices=CANDIDATE_EPOCHS, required=True)
    parser.add_argument(
        "--topology-mode", choices=tuple(SUPPORTED_TOPOLOGIES), required=True
    )
    parser.add_argument("--producer-source-root", type=Path, required=True)
    parser.add_argument("--expected-producer-trainer-sha256", required=True)
    parser.add_argument("--expected-producer-contract-sha256", required=True)
    parser.add_argument(
        "--validation-evidence-source-root", type=Path, required=True
    )
    parser.add_argument(
        "--expected-validation-evidence-contract-sha256", required=True
    )
    parser.add_argument(
        "--expected-validation-evidence-selector-sha256", required=True
    )
    parser.add_argument(
        "--runtime-validation-source-root", type=Path, required=True
    )
    parser.add_argument(
        "--expected-runtime-validation-contract-sha256", required=True
    )
    parser.add_argument(
        "--expected-runtime-validation-selector-sha256", required=True
    )
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--expected-frozen-inputs-sha256", required=True)
    parser.add_argument("--val-inputs", type=Path, required=True)
    parser.add_argument("--expected-val-inputs-sha256", required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--expected-pipeline-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _authorize_parser(subparsers)
    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("--train-root", type=Path, required=True)
    reconcile_parser.add_argument(
        "--topology-mode", choices=tuple(SUPPORTED_TOPOLOGIES), required=True
    )
    reconcile_parser.add_argument("--producer-source-root", type=Path, required=True)
    reconcile_parser.add_argument("--expected-producer-trainer-sha256", required=True)
    reconcile_parser.add_argument("--expected-producer-contract-sha256", required=True)
    reconcile_parser.add_argument(
        "--validation-evidence-source-root", type=Path, required=True
    )
    reconcile_parser.add_argument(
        "--expected-validation-evidence-contract-sha256", required=True
    )
    reconcile_parser.add_argument(
        "--expected-validation-evidence-selector-sha256", required=True
    )
    reconcile_parser.add_argument(
        "--runtime-validation-source-root", type=Path, required=True
    )
    reconcile_parser.add_argument(
        "--expected-runtime-validation-contract-sha256", required=True
    )
    reconcile_parser.add_argument(
        "--expected-runtime-validation-selector-sha256", required=True
    )
    reconcile_parser.add_argument("--schedule", type=Path, required=True)
    reconcile_parser.add_argument("--expected-schedule-sha256", required=True)
    reconcile_parser.add_argument("--expected-frozen-inputs-sha256", required=True)
    reconcile_parser.add_argument("--val-inputs", type=Path, required=True)
    reconcile_parser.add_argument("--expected-val-inputs-sha256", required=True)
    reconcile_parser.add_argument("--pipeline", type=Path, required=True)
    reconcile_parser.add_argument("--expected-pipeline-sha256", required=True)
    reconcile_parser.add_argument("--authority-dir", type=Path, required=True)
    reconcile_parser.add_argument("--final-manifest", type=Path, required=True)
    reconcile_parser.add_argument("--expected-final-manifest-sha256", required=True)
    reconcile_parser.add_argument("--final-status", type=Path, required=True)
    reconcile_parser.add_argument("--expected-final-status-sha256", required=True)
    reconcile_parser.add_argument("--output", type=Path, required=True)
    return parser


def _config_from_args(args: argparse.Namespace, *, epoch: int) -> AuthorizeConfig:
    return AuthorizeConfig(
        train_root=args.train_root,
        epoch=epoch,
        topology_mode=args.topology_mode,
        producer_source_root=args.producer_source_root,
        expected_producer_trainer_sha256=(
            args.expected_producer_trainer_sha256
        ),
        expected_producer_contract_sha256=(
            args.expected_producer_contract_sha256
        ),
        validation_evidence_source_root=(
            args.validation_evidence_source_root
        ),
        expected_validation_evidence_contract_sha256=(
            args.expected_validation_evidence_contract_sha256
        ),
        expected_validation_evidence_selector_sha256=(
            args.expected_validation_evidence_selector_sha256
        ),
        runtime_validation_source_root=args.runtime_validation_source_root,
        expected_runtime_validation_contract_sha256=(
            args.expected_runtime_validation_contract_sha256
        ),
        expected_runtime_validation_selector_sha256=(
            args.expected_runtime_validation_selector_sha256
        ),
        schedule=args.schedule,
        expected_schedule_sha256=args.expected_schedule_sha256,
        expected_frozen_inputs_sha256=args.expected_frozen_inputs_sha256,
        val_inputs=args.val_inputs,
        expected_val_inputs_sha256=args.expected_val_inputs_sha256,
        pipeline=args.pipeline,
        expected_pipeline_sha256=args.expected_pipeline_sha256,
        output=args.output,
    )


def _run_internal_validation_helper(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--runtime-validation-source-root", type=Path, required=True
    )
    parser.add_argument(
        "--expected-runtime-validation-contract-sha256", required=True
    )
    parser.add_argument(
        "--expected-runtime-validation-selector-sha256", required=True
    )
    args = parser.parse_args(argv)
    root = canonical_existing_directory(
        args.runtime_validation_source_root,
        "isolated runtime validation source root",
    )
    if root != args.runtime_validation_source_root:
        raise LiveValidationContractError(
            "isolated validation source root is not canonical"
        )
    ValidationHooks._git_checkout_authority(
        root,
        expected_commit=RUNTIME_VALIDATION_SOURCE_COMMIT,
        expected_tree=RUNTIME_VALIDATION_SOURCE_TREE,
        label="isolated runtime validation source",
    )
    contract_path = root / "scripts/show_base/base_long_val_contract.py"
    selector_path = root / "scripts/show_base/select_base_official_adapt.py"
    _, contract_sha, _ = safe_regular_hash(
        contract_path, "isolated validation contract"
    )
    _, selector_sha, _ = safe_regular_hash(
        selector_path, "isolated validation selector"
    )
    if (
        contract_sha != RUNTIME_VALIDATION_CONTRACT_SHA256
        or contract_sha
        != require_sha256(
            args.expected_runtime_validation_contract_sha256,
            "isolated expected runtime validation contract SHA-256",
        )
        or selector_sha != RUNTIME_VALIDATION_SELECTOR_SHA256
        or selector_sha
        != require_sha256(
            args.expected_runtime_validation_selector_sha256,
            "isolated expected runtime validation selector SHA-256",
        )
    ):
        raise LiveValidationContractError(
            "isolated validation source entrypoint changed"
        )
    request_bytes = sys.stdin.buffer.read(
        VALIDATION_HELPER_MAX_INPUT_BYTES + 1
    )
    if not request_bytes or len(request_bytes) > VALIDATION_HELPER_MAX_INPUT_BYTES:
        raise LiveValidationContractError(
            "isolated validation-helper input is empty or over the byte bound"
        )
    request = strict_json_bytes(
        request_bytes, "isolated validation-helper request"
    )
    if not isinstance(request, dict) or set(request) != {"operation", "payload"}:
        raise LiveValidationContractError(
            "isolated validation-helper request schema changed"
        )
    payload = request["payload"]
    if not isinstance(payload, dict):
        raise LiveValidationContractError(
            "isolated validation-helper payload is not an object"
        )
    for name, module in tuple(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts."):
            raise LiveValidationContractError(
                "project module was preloaded before isolated validation"
            )
    root_text = str(root)
    sys.path = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    specification = importlib.util.spec_from_file_location(
        f"_isolated_validation_contract_{contract_sha}", contract_path
    )
    if specification is None or specification.loader is None:
        raise LiveValidationContractError(
            "cannot load the isolated validation contract"
        )
    contract = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(contract)
    diffsheg = getattr(contract, "diffsheg", None)
    pinned = getattr(diffsheg, "DIFFSHEG_PINNED_RECEIPT", None)
    fgd_specification = (
        pinned.get("autoencoders", {}).get("fgd")
        if isinstance(pinned, dict)
        else None
    )
    if not strict_json_equal(
        fgd_specification, EXPECTED_DIFFSHEG_FGD_PROVENANCE
    ):
        raise LiveValidationContractError(
            "runtime validation lacks the exact official DiffSHEG FGD provenance fix"
        )
    for name, module in tuple(sys.modules.items()):
        if not (
            name in {"scripts", "models", "dataloaders"}
            or name.startswith(("scripts.", "models.", "dataloaders."))
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            module_paths = getattr(module, "__path__", None)
            if module_paths is None:
                raise LiveValidationContractError(
                    f"isolated project namespace has no source path: {name}"
                )
            try:
                candidates = [
                    Path(str(value)).resolve(strict=True)
                    for value in module_paths
                ]
            except OSError as error:
                raise LiveValidationContractError(
                    f"isolated project namespace path is invalid: {name}"
                ) from error
        else:
            try:
                candidates = [Path(str(module_file)).resolve(strict=True)]
            except OSError as error:
                raise LiveValidationContractError(
                    f"isolated project module path is invalid: {name}"
                ) from error
        try:
            for candidate in candidates:
                candidate.relative_to(root)
        except ValueError as error:
            raise LiveValidationContractError(
                f"isolated project module escaped validation source: {name}"
            ) from error
    operation = request["operation"]
    if operation == "validate_val_inputs":
        if set(payload) != {"path", "expected_sha256"}:
            raise LiveValidationContractError(
                "isolated validation-input request schema changed"
            )
        receipt, coverage = contract.validate_val_inputs(
            Path(payload["path"]), payload["expected_sha256"]
        )
        result = {
            "receipt": receipt,
            "coverage": contract.public_val_coverage(coverage),
        }
    elif operation == "validate_pipeline":
        if set(payload) != {
            "path",
            "expected_sha256",
            "expected_prerequisite_selection",
            "expected_source",
        }:
            raise LiveValidationContractError(
                "isolated validation-pipeline request schema changed"
            )
        receipt, pipeline = contract.validate_pipeline(
            Path(payload["path"]),
            payload["expected_sha256"],
            expected_prerequisite_selection=payload[
                "expected_prerequisite_selection"
            ],
            expected_source=payload["expected_source"],
        )
        result = {"receipt": receipt, "pipeline": pipeline}
    else:
        raise LiveValidationContractError(
            "unknown isolated validation-helper operation"
        )
    encoded = canonical_json_bytes({"result": result})
    if len(encoded) > VALIDATION_HELPER_MAX_OUTPUT_BYTES:
        raise LiveValidationContractError(
            "isolated validation-helper output exceeds the byte bound"
        )
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] == "_validation-helper":
        return _run_internal_validation_helper(effective_argv[1:])
    args = build_parser().parse_args(effective_argv)
    if args.command == "authorize":
        result = authorize_candidate(_config_from_args(args, epoch=args.epoch))
    else:
        common = _config_from_args(args, epoch=CANDIDATE_EPOCHS[0])
        result = reconcile(
            ReconcileConfig(
                common=common,
                authority_dir=args.authority_dir,
                final_manifest=args.final_manifest,
                expected_final_manifest_sha256=args.expected_final_manifest_sha256,
                final_status=args.final_status,
                expected_final_status_sha256=args.expected_final_status_sha256,
            )
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
