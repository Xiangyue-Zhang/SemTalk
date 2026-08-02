#!/usr/bin/env python3
"""Fail-closed, single-flight supervisor for V14 live SHOW validation.

The supervisor is intentionally CPU-only.  It never inventories GPUs, searches
for processes, or sends signals.  Its only side effect outside its private
state directory is one synchronous invocation of the *exact* guarded-runner
argv frozen in the campaign.  Success is established solely by fresh,
create-new completion receipts and post-run guard-restoration evidence.

Crash policy is deliberately conservative: a job claim without its matching
completion, or an active-invocation claim left by a crashed supervisor, is a
terminal/manual state.  Nothing is retried automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


sys.dont_write_bytecode = True

CANDIDATE_EPOCHS: Tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80, 100, 120, 140, 160,
    180, 200, 240, 280, 320, 360, 400,
)
GPU_INDICES: Tuple[int, ...] = tuple(range(8))
HEX64 = re.compile(r"^[0-9a-f]{64}$")

CAMPAIGN_FORMAT = "semtalk_show_base_v14_live_validation_campaign_v3"
CAMPAIGN_CLAIM_FORMAT = "semtalk_show_base_v14_live_validation_campaign_claim_v1"
ACTIVE_CLAIM_FORMAT = "semtalk_show_base_v14_live_validation_active_claim_v1"
JOB_CLAIM_FORMAT = "semtalk_show_base_v14_live_validation_job_claim_v2"
COMPLETION_FORMAT = "semtalk_show_base_v14_live_validation_completion_v4"
SUMMARY_FORMAT = "semtalk_show_base_v14_live_validation_summary_v3"
MEASUREMENT_FORMAT = "semtalk_show_base_live_val_measurement_v2"
SELECTION_FORMAT = "semtalk_show_base_live_val_22way_selection_v2"
RECONCILIATION_FORMAT = "semtalk_show_base_live_val_reconciliation_v2"
VALIDATION_SEMANTICS_COMMIT = "4066f2096e1675f9c19d725894007ff25f3e9b4b"
VALIDATION_SEMANTICS_TREE = "0b66e3aa1fb23732e76e51492737c4ab1f4db2d0"
PRODUCER_SOURCE_COMMIT = "8f1fa7b85ed8253600a4c571e98eb9927edeb073"
PRODUCER_SOURCE_TREE = "d5e5eb74acdc6d9ca330d5731e718e33b67cf626"
PRODUCER_TRAINER_SHA256 = "29fdd5d3e9bdfc61904f649b71d4dae1766b42a4a6a5a40b2bbd37a4c6b33173"
PRODUCER_CONTRACT_SHA256 = "3526ca896f23e7849545e3f81553dd242dc1ca3049eabdeeb89fc38357616323"
VALIDATION_SEMANTICS_ROOT = "/local-ssd/xiangyuezhang/semtalk_final_4066f20_20260802"
VALIDATION_SEMANTICS_CONTRACT_SHA256 = "3983eb7adf2f8cbfcf738fca8cc1cf8f3b34acab274bb5dfd2e23964550f8daa"
VALIDATION_SEMANTICS_SELECTOR_SHA256 = "1b76579d54efc99ac0d0d63f3d1a68ff671216e693ef9ada4714c9fcda004121"
RUNTIME_VALIDATION_COMMIT = "70a70f452bdf743e317b583a7770980f0ce744c3"
RUNTIME_VALIDATION_TREE = "bdf7680f56f9f53c92e7ab0bf6c6a84ef4f83d69"
RUNTIME_VALIDATION_ROOT = "/local-ssd/xiangyuezhang/semtalk_diffsheg_validation_70a70f4_20260802"
RUNTIME_VALIDATION_CONTRACT_SHA256 = "0168e7f9ab2b9a122656ed98c36dc36777f577816ef08b0186ffc892813d2c70"
RUNTIME_VALIDATION_SELECTOR_SHA256 = "1857460c188096fcd4b24a9f8f5a0b0390a80eacd20744a46098195ac0f39ddf"
RUNTIME_VALIDATION_EVALUATOR_SHA256 = "18b7dfcc2989a8c2136b8e530e471cfa9a088b29600e084522f0312f1e8e661f"
RUNTIME_VALIDATION_INFERENCE_SHA256 = "ec3de79ce1e45ff5385db40797bbcfa0dc91ab3e61d265e8f975180515190b56"
RUNTIME_VALIDATION_MEASUREMENT_SHA256 = "e985056c889c9803212a236597066877726826badfcd1fa91f329e5b00e56550"
RUNTIME_VALIDATION_LONG_SELECTOR_SHA256 = "d813864a33a5ff2109096453f956329a96480dd11f26c8a9144228df34d67269"
RUNTIME_VALIDATION_SOURCE_FORMAT = "semtalk_show_base_runtime_validation_source_v1"
AUTHORIZATION_FORMAT = "semtalk_show_base_v14_live_authorization_v1"
RECOVERY_AUTHORIZATION_FORMAT = "semtalk_show_base_v14_live_recovery_authorization_v1"
RECOVERY_REQUEST_FORMAT = "semtalk_show_base_v14_failed_consumer_recovery_request_v1"
RECOVERY_CONTROL_MIN_COMMIT = "8532272130e14098fdef3f54c7733d4ba6033f4c"
FAILED_CAMPAIGN_SHA256 = "9f299db9dc874826808142bfec4a69e403102a90293a42e2aa6c9ad52e3fab0a"
FAILED_CAMPAIGN_BYTES = 33838
FAILED_CONTROL_COMMIT = "58407ed3207fdd76dbf9a6e480de8af579bdb747"
FAILED_CONTROL_TREE = "b71f20161677fc68b4f9220a71d25e40632b17a0"
FAILED_JOB_CLAIM_SHA256 = "195994203d9f762b29c612c00aa1f185a9b407b1cfca687baa0fbc535d4c3b9b"
FAILED_ACTIVE_CLAIM_SHA256 = "0f4349f368b532334f6540252d2d7609d0f5ca13d7ed1d5307a74a5432f25f4a"
FAILED_AUTHORIZATION_SHA256 = "c122131f1198a431e927fc45e6ea529c21fc6dbf49e0cdfe0f3ca166e3647207"
FAILED_WORK_AUTHORITY_SHA256 = "31c2930acf1bc608687351132b0ce135ddddc7d8d75655857ca9334fb51142d2"
FAILED_CONSUMER_CLAIM_SHA256 = "b8449bf107ca4c97a543676e8883ad4218231524241e3ff685fbcc1cbff940b3"
FAILED_RUNNER_STATUS_SHA256 = "b0c3f8a9a7ffd203621ac6d5bbcc29fdc2ff008ce9c6a2825b934139cee6fd2d"
FAILED_RUNNER_LOG_SHA256 = "dd2a04e4989df737ca48fd55c3df02c027d0e5c9a2e3b2540d3908c864628ad8"
FAILED_CANDIDATE_RECEIPT_SHA256 = "16b2fe66a7f8ec0c38c7ab326b540a6a14043c74cd85fe013e7ca6cf10b24d8a"
FAILED_WRAPPER_PID = 261736
FAILED_CHILD_PID = 261737
PRODUCER_RECONCILE_CLAIM_FORMAT = "semtalk_show_base_v14_producer_reconcile_claim_v1"
BRIDGE_RECONCILE_CLAIM_FORMAT = "semtalk_show_base_v14_bridge_reconcile_claim_v1"

ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
CAMPAIGN_KEYS = frozenset({
    "format", "status", "split", "test_visible",
    "test_measurements_authorized", "candidate_epochs", "state_root",
    "campaign_claim_path", "summary_path", "control_source",
    "runtime_validation_source", "authority_adapter_config",
    "final_manifest_path", "final_status_path",
    "paspa_root", "diffsheg_root", "seed", "diffsheg_batch_size",
    "formal_python", "formal_python_runtime_contract", "guarded_runner",
    "guard_verifier", "reconciliation_path", "reconcile_selection_root",
    "jobs", "campaign_payload_sha256",
})
JOB_KEYS = frozenset({
    "epoch", "candidate_receipt_path", "authority_path",
    "authorization_path", "run_root", "measurement_path",
    "completion_path", "runner_status_path", "runner_log_path",
})
COMPLETION_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "test_measurements_authorized", "candidate_epoch", "campaign",
    "candidate_receipt", "work_authority", "authorization",
    "measurement", "runner_status", "runner_log",
    "bridge_replay_argv", "bridge_replay_stdout",
    "guard_verifier", "guard_verifier_argv", "guard_verifier_stdout",
    "restored_guards", "validation_diffsheg_fgd", "completed_unix",
    "consumer_recovery",
    "receipt_payload_sha256",
})
RUNNER_STATUS_KEYS = frozenset({
    "updated_at", "state", "wrapper_pid", "child_pid", "return_code",
    "received_signal", "error", "cleanup_error", "restored_guards",
    "restore_error", "command",
})
MEASUREMENT_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "candidate_epoch", "candidate_checkpoint", "work_authority",
    "consumer_claim", "execution_preflight", "val_inputs_receipt",
    "pipeline_receipt", "inference_lineage", "diffsheg_report", "metrics",
    "execution_contract", "receipt_payload_sha256",
})
SELECTION_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "candidate_epochs", "reconciliation_receipt", "live_measurements",
    "reconciled_measurements", "formal_selection", "selected",
    "test_evaluations_observed", "receipt_payload_sha256",
})
SELECTION_SELECTED_KEYS = frozenset({
    "epoch", "candidate_checkpoint", "fgd", "inference_lineage",
    "diffsheg_report",
})
SUMMARY_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "test_measurements_authorized", "campaign", "candidate_epochs",
    "completion_count", "producer_reconcile_claim", "bridge_reconcile_claim",
    "official_selection",
    "selected", "completed_unix", "receipt_payload_sha256",
})
FORMAL_PYTHON_KEYS = frozenset({
    "format", "argv0", "venv_root", "symlink_chain", "resolved_target",
    "pyvenv_cfg",
})
CONTROL_SOURCE_KEYS = frozenset({
    "root", "origin", "commit", "tree", "supervisor", "launcher", "bridge",
    "authority_adapter",
})
RUNTIME_VALIDATION_SOURCE_KEYS = frozenset({
    "format", "origin", "source_root", "commit", "tree",
    "validation_semantics_source", "semantics_ancestry_verified",
    "autoencoder_provenance", "contract", "evaluator", "selector",
})
VALIDATION_SEMANTICS_SOURCE_KEYS = frozenset({
    "origin", "source_root", "commit", "tree", "contract", "selector",
})
AUTHORITY_ADAPTER_CONFIG_KEYS = frozenset({
    "train_root", "topology_mode", "producer_source_root",
    "expected_producer_trainer_sha256", "expected_producer_contract_sha256",
    "schedule", "expected_frozen_inputs_sha256", "val_inputs", "pipeline",
})
AUTHORIZATION_KEYS = frozenset({
    "format", "status", "candidate_epoch", "campaign", "candidate_receipt",
    "work_authority", "adapter", "adapter_argv", "adapter_stdout_sha256",
    "runner_argv",
    "completed_unix", "receipt_payload_sha256",
})
RECOVERY_AUTHORIZATION_KEYS = AUTHORIZATION_KEYS | frozenset({
    "recovery_request", "recovery_authority", "recovery_claim",
})
RECOVERY_REQUEST_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "candidate_epoch", "failed_campaign", "failed_job_claim",
    "failed_active_claim", "failed_authorization", "failed_work_authority",
    "failed_consumer_claim", "failed_runner_status", "failed_runner_log",
    "failed_run_root", "failed_run_inventory", "failed_process_proof",
    "guard_proof",
    "new_campaign", "new_control_source", "new_work_authority",
    "new_run_root", "created_unix", "receipt_payload_sha256",
})
ACTIVE_CLAIM_KEYS = frozenset({
    "format", "status", "operation", "campaign", "created_unix",
    "claim_payload_sha256",
})
JOB_CLAIM_KEYS = frozenset({
    "format", "status", "candidate_epoch", "campaign", "candidate_receipt",
    "authority_path", "run_root", "measurement_path",
    "authorize_argv_sha256", "created_unix", "claim_payload_sha256",
})
PRODUCER_RECONCILE_CLAIM_KEYS = frozenset({
    "format", "status", "campaign", "final_manifest", "final_status",
    "argv_sha256", "created_unix", "claim_payload_sha256",
})
BRIDGE_RECONCILE_CLAIM_KEYS = frozenset({
    "format", "status", "campaign", "reconciliation", "argv_sha256",
    "created_unix", "claim_payload_sha256",
})
SOURCE_REFERENCE_KEYS = frozenset({
    "origin", "source_root", "commit", "tree", "clean", "detached",
    "local_branches_at_commit",
})


class SupervisorError(RuntimeError):
    """A terminal, fail-closed supervisor contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SupervisorError(message)


def _exact_int(value: Any, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, "%s must be an exact integer >= %d" % (label, minimum))
    return value


def _finite_number(value: Any, label: str, minimum: float = 0.0) -> float:
    require(type(value) in (int, float), "%s must be a number" % label)
    result = float(value)
    require(math.isfinite(result) and result >= minimum, "%s is not finite/in-range" % label)
    return result


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise SupervisorError("value is not strict canonical JSON") from error


def payload_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(canonical_json_bytes(dict(value)))


def strict_json_equal(left: Any, right: Any) -> bool:
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


def _strict_object_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        require(isinstance(key, str) and key not in result, "duplicate/non-string JSON key")
        result[key] = value
    return result


def strict_json(raw: bytes, label: str) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SupervisorError("%s contains non-finite JSON number %s" % (label, value))

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupervisorError("%s is not strict JSON" % label) from error
    require(isinstance(value, dict), "%s must be a JSON object" % label)
    require(raw == canonical_json_bytes(value), "%s is not canonical newline JSON" % label)
    return value


def _identity(metadata: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_size),
        int(metadata.st_mtime_ns), int(metadata.st_ctime_ns),
    )


def _lexical_absolute(path_text: Any, label: str) -> Path:
    if isinstance(path_text, os.PathLike):
        path_text = os.fspath(path_text)
    require(isinstance(path_text, str) and path_text == path_text.strip(), "%s path is not lexical" % label)
    require(path_text.startswith("/"), "%s path must be absolute" % label)
    require(not any(ch.isspace() or ord(ch) < 32 for ch in path_text), "%s path contains whitespace/control" % label)
    path = Path(path_text)
    require(str(path) == path_text and "." not in path.parts and ".." not in path.parts, "%s path is not canonical" % label)
    return path


def _no_symlink_components(path: Path, label: str, include_leaf: bool = True) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.is_absolute() else path.parts
    limit = len(parts) if include_leaf else max(0, len(parts) - 1)
    for part in parts[:limit]:
        current = current / part
        metadata = os.lstat(current)
        require(not stat.S_ISLNK(metadata.st_mode), "%s contains a symlink component" % label)


def canonical_directory(path_text: Any, label: str) -> Path:
    path = _lexical_absolute(path_text, label)
    _no_symlink_components(path, label)
    metadata = path.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and path.resolve(strict=True) == path, "%s is not a canonical directory" % label)
    return path


def canonical_output_path(path_text: Any, label: str) -> Path:
    path = _lexical_absolute(path_text, label)
    _no_symlink_components(path, label, include_leaf=False)
    require(path.parent.resolve(strict=True) == path.parent, "%s parent is not canonical" % label)
    return path


def safe_regular_bytes(
    path_text: Any,
    label: str,
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> Tuple[Path, bytes, str, int]:
    path = _lexical_absolute(path_text, label)
    _no_symlink_components(path, label, include_leaf=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SupervisorError("cannot safely open %s: %s" % (label, error)) from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "%s must be a single-link regular file" % label)
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        require(_identity(before) == _identity(after), "%s changed during read" % label)
        raw = b"".join(chunks)
        require(len(raw) == before.st_size, "%s length changed during read" % label)
        current = os.stat(path, follow_symlinks=False)
        require(_identity(current) == _identity(before), "%s pathname changed after read" % label)
        require(path.resolve(strict=True) == path, "%s path resolves elsewhere" % label)
    finally:
        os.close(descriptor)
    digest = _sha256(raw)
    if expected_sha256 is not None:
        require(isinstance(expected_sha256, str) and HEX64.fullmatch(expected_sha256) is not None, "%s expected SHA is invalid" % label)
        require(digest == expected_sha256, "%s SHA changed" % label)
    if expected_bytes is not None:
        _exact_int(expected_bytes, "%s expected bytes" % label, 1)
        require(len(raw) == expected_bytes, "%s byte length changed" % label)
    return path, raw, digest, len(raw)


def _artifact(value: Any, label: str, verify: bool = True) -> Dict[str, Any]:
    require(isinstance(value, dict) and set(value) == ARTIFACT_KEYS, "%s artifact schema changed" % label)
    path_text = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None, "%s SHA is invalid" % label)
    _exact_int(size, "%s bytes" % label, 1)
    if verify:
        path, _raw, observed_sha, observed_size = safe_regular_bytes(path_text, label, digest, size)
        return {"path": str(path), "sha256": observed_sha, "bytes": observed_size}
    path = _lexical_absolute(path_text, label)
    return {"path": str(path), "sha256": digest, "bytes": size}


def _read_existing_json(path: Path, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved, raw, digest, size = safe_regular_bytes(str(path), label)
    return {"path": str(resolved), "sha256": digest, "bytes": size}, strict_json(raw, label)


def write_new_json(path_text: Any, payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    path = canonical_output_path(path_text, label)
    require(not os.path.lexists(path), "%s already exists" % label)
    raw = canonical_json_bytes(dict(payload))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            require(count > 0, "%s short write" % label)
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and stat.S_IMODE(metadata.st_mode) == 0o400, "%s did not seal" % label)
    reread_path, reread, digest, size = safe_regular_bytes(path, label)
    require(reread == raw, "%s changed after write" % label)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {"path": str(reread_path), "sha256": digest, "bytes": size}


def _self_hashed(payload: Mapping[str, Any], field: str, label: str) -> Dict[str, Any]:
    result = dict(payload)
    claimed = result.pop(field, None)
    require(isinstance(claimed, str) and HEX64.fullmatch(claimed) is not None, "%s self-hash is invalid" % label)
    require(payload_sha256(result) == claimed, "%s self-hash changed" % label)
    return dict(payload)


def _add_self_hash(payload: Mapping[str, Any], field: str) -> Dict[str, Any]:
    result = dict(payload)
    require(field not in result, "self-hash field already present")
    result[field] = payload_sha256(result)
    return result


# ---------------------------------------------------------------------------
# Live-launch contract.  Descriptor-safe primitives above are shared by the
# single V3 supervisor implementation below.

MAX_RUNNER_LOG_BYTES = 4096
GUARD_PASS_RE = re.compile(
    r"\APASS "
    + " ".join("GPU%d=PID([2-9]|[1-9][0-9]+)" % gpu for gpu in GPU_INDICES)
    + r"\n\Z"
)
BRIDGE_REPLAY_CODE = r'''
import importlib.util
from pathlib import Path
import sys

bridge, measurement, digest, payload_digest, epoch_text, authority_path, authority_sha, authority_bytes = sys.argv[1:]
spec = importlib.util.spec_from_file_location("semtalk_live_bridge_replay", bridge)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load pinned bridge")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
artifact, value = module._validate_live_measurement(Path(measurement), digest)
authority = value.get("work_authority")
if (
    artifact.get("receipt_payload_sha256") != payload_digest
    or value.get("candidate_epoch") != int(epoch_text)
    or not isinstance(authority, dict)
    or authority.get("path") != authority_path
    or authority.get("sha256") != authority_sha
    or authority.get("bytes") != int(authority_bytes)
):
    raise SystemExit("fresh live measurement replay differs")
sys.stdout.write("PASS\n")
'''.strip()


def _strict_json_document(raw: bytes, label: str) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SupervisorError("%s contains non-finite JSON number %s" % (label, value))
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupervisorError("%s is not strict JSON" % label) from error
    require(isinstance(value, dict), "%s must be a JSON object" % label)
    return value


def _bridge_payload_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_payload_sha256", None)
    raw = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(raw)


def _add_bridge_self_hash(
    payload: Mapping[str, Any], field: str,
) -> Dict[str, Any]:
    """Publish a document using the bridge's no-newline payload ABI."""

    result = dict(payload)
    require(field not in result, "bridge self-hash field already present")
    result[field] = _bridge_payload_sha(result)
    return result


def _plain_artifact(value: Any, label: str, *, executable: bool = False) -> Dict[str, Any]:
    artifact = _artifact(value, label)
    if executable:
        require(Path(artifact["path"]).stat().st_mode & 0o111 != 0, "%s is not executable" % label)
    return artifact


def _normalize_link_target(link: Path, target: str) -> Path:
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = link.parent / candidate
    normalized = Path(os.path.normpath(str(candidate)))
    require(normalized.is_absolute(), "formal Python symlink target is not absolute")
    return normalized


def _formal_python(value: Any, *, running_executable: Optional[str] = None) -> Dict[str, Any]:
    require(isinstance(value, dict) and set(value) == FORMAL_PYTHON_KEYS, "formal Python schema changed")
    binding = dict(value)
    require(binding.get("format") == "semtalk.formal_venv_python_binding.v1", "formal Python format changed")
    argv0 = _lexical_absolute(binding.get("argv0"), "formal Python argv0")
    require(re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", argv0.name) is not None, "formal Python leaf changed")
    venv_root = canonical_directory(binding.get("venv_root"), "formal Python venv root")
    require(argv0.parent == venv_root / "bin" and argv0.parent.resolve(strict=True) == argv0.parent, "formal Python is not under canonical venv/bin")
    expected_running = sys.executable if running_executable is None else running_executable
    require(os.fsencode(expected_running) == os.fsencode(str(argv0)), "supervisor is not running through the frozen venv launcher")
    chain = binding.get("symlink_chain")
    require(isinstance(chain, list) and 1 <= len(chain) <= 16, "formal Python symlink chain changed")
    current = argv0
    seen: set = set()
    normalized_chain: List[Dict[str, str]] = []
    for ordinal, row in enumerate(chain):
        require(isinstance(row, dict) and set(row) == {"path", "target"}, "formal Python symlink row changed")
        require(row.get("path") == str(current) and str(current) not in seen, "formal Python symlink order/loop changed")
        seen.add(str(current))
        before = current.lstat()
        require(stat.S_ISLNK(before.st_mode), "formal Python chain hop is not a symlink")
        target = os.readlink(current)
        after = current.lstat()
        require(_identity(before) == _identity(after) and target == row.get("target"), "formal Python symlink hop changed")
        normalized_chain.append({"path": str(current), "target": target})
        current = _normalize_link_target(current, target)
    require(not current.is_symlink() and argv0.resolve(strict=True) == current, "formal Python chain does not end at one ELF")
    target = _plain_artifact(binding.get("resolved_target"), "formal Python final ELF", executable=True)
    require(target["path"] == str(current), "formal Python final ELF path changed")
    cfg = _plain_artifact(binding.get("pyvenv_cfg"), "formal Python pyvenv.cfg")
    require(cfg["path"] == str(venv_root / "pyvenv.cfg"), "formal Python pyvenv.cfg path changed")
    _cfg_path, cfg_raw, _cfg_sha, _cfg_size = safe_regular_bytes(cfg["path"], "formal Python pyvenv.cfg replay", cfg["sha256"], cfg["bytes"])
    configuration: Dict[str, str] = {}
    try:
        for raw_line in cfg_raw.decode("utf-8").splitlines():
            if not raw_line.strip():
                continue
            require("=" in raw_line, "formal Python pyvenv.cfg record changed")
            key, item = raw_line.split("=", 1)
            key = key.strip().casefold()
            require(bool(key) and key not in configuration, "formal Python pyvenv.cfg key changed")
            configuration[key] = item.strip()
    except UnicodeDecodeError as error:
        raise SupervisorError("formal Python pyvenv.cfg is not UTF-8") from error
    require({"home", "include-system-site-packages", "version"}.issubset(configuration), "formal Python pyvenv.cfg is incomplete")
    require(configuration["include-system-site-packages"].casefold() in {"true", "false"}, "formal Python site policy changed")
    home = Path(configuration["home"])
    require(home.is_absolute() and (home / current.name).resolve(strict=True) == current, "formal Python home/ELF binding changed")
    configured_executable = configuration.get("executable")
    if configured_executable is not None:
        require(Path(configured_executable).is_absolute() and Path(configured_executable).resolve(strict=True) == current, "formal Python configured executable changed")
    binding["argv0"] = str(argv0)
    binding["venv_root"] = str(venv_root)
    binding["symlink_chain"] = normalized_chain
    binding["resolved_target"] = target
    binding["pyvenv_cfg"] = cfg
    return binding


def _control_source(value: Any, running_script: str) -> Dict[str, Any]:
    require(isinstance(value, dict) and set(value) == CONTROL_SOURCE_KEYS, "control source schema changed")
    source = dict(value)
    root = canonical_directory(source.get("root"), "control source root")
    require(source.get("origin") == "git@github.com:Xiangyue-Zhang/SemTalk.git", "control source origin changed")
    require(isinstance(source.get("commit"), str) and re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is not None, "control source commit changed")
    require(isinstance(source.get("tree"), str) and re.fullmatch(r"[0-9a-f]{40}", source["tree"]) is not None, "control source tree changed")
    for key in ("supervisor", "launcher", "bridge", "authority_adapter"):
        source[key] = _plain_artifact(source.get(key), "control %s" % key, executable=(key == "launcher"))
        require(Path(source[key]["path"]).is_relative_to(root), "control %s escapes source root" % key)
    require(source["supervisor"]["path"] == running_script, "running supervisor differs from frozen control source")
    require(source["launcher"]["path"] == str(root / "scripts/show_base/run_base_live_val_8shard.sh"), "tracked live launcher path changed")
    require(source["bridge"]["path"] == str(root / "scripts/show_base/base_live_val_consumer_bridge.py"), "tracked live bridge path changed")
    require(source["authority_adapter"]["path"] == str(root / "scripts/show_base/base_v14_live_validation_authority.py"), "tracked authority adapter path changed")
    require(_git_stdout(root, "remote") == "origin", "control source remotes changed")
    require(
        _git_stdout(root, "remote", "get-url", "origin") == source["origin"]
        and _git_stdout(root, "remote", "get-url", "--push", "origin") == source["origin"],
        "control source origin changed",
    )
    require(
        _git_stdout(root, "rev-parse", "HEAD") == source["commit"]
        and _git_stdout(root, "rev-parse", "HEAD^{tree}") == source["tree"],
        "control source commit/tree changed",
    )
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "--short", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    require(symbolic.returncode == 1 and symbolic.stdout == "" and symbolic.stderr == "", "control source is not detached")
    require(
        _git_stdout(root, "status", "--porcelain=v1", "--untracked-files=all") == ""
        and _git_stdout(root, "for-each-ref", "--format=%(refname)", "refs/heads") == "",
        "control source is not clean zero-branch",
    )
    for key in ("supervisor", "launcher", "bridge", "authority_adapter"):
        relative = str(Path(source[key]["path"]).relative_to(root))
        require(_git_stdout(root, "ls-files", "--error-unmatch", relative) == relative, "control source file is not tracked: %s" % relative)
    source["root"] = str(root)
    return source


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False, capture_output=True, text=True,
    )
    require(
        completed.returncode == 0 and completed.stdout == "" and completed.stderr == "",
        "runtime validation source is not a proved descendant of validation semantics 4066",
    )


def _git_authority(root: Path, commit: str, tree: str, label: str) -> None:
    require(_git_stdout(root, "remote") == "origin", "%s remotes changed" % label)
    require(_git_stdout(root, "remote", "get-url", "origin") == "git@github.com:Xiangyue-Zhang/SemTalk.git" and _git_stdout(root, "remote", "get-url", "--push", "origin") == "git@github.com:Xiangyue-Zhang/SemTalk.git", "%s origin changed" % label)
    require(_git_stdout(root, "rev-parse", "HEAD") == commit and _git_stdout(root, "rev-parse", "HEAD^{tree}") == tree, "%s commit/tree changed" % label)
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "--short", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    require(symbolic.returncode == 1 and symbolic.stdout == "" and symbolic.stderr == "", "%s is not detached" % label)
    require(_git_stdout(root, "status", "--porcelain=v1", "--untracked-files=all") == "" and _git_stdout(root, "for-each-ref", "--format=%(refname)", "refs/heads") == "", "%s is not clean zero-branch" % label)


def _validation_semantics_source(value: Any) -> Dict[str, Any]:
    require(isinstance(value, dict) and set(value) == VALIDATION_SEMANTICS_SOURCE_KEYS, "validation semantics source schema changed")
    source = dict(value)
    root = canonical_directory(source.get("source_root"), "validation semantics source root")
    require(str(root) == VALIDATION_SEMANTICS_ROOT, "validation semantics source root changed")
    require(source.get("origin") == "git@github.com:Xiangyue-Zhang/SemTalk.git", "validation semantics source origin changed")
    require(source.get("commit") == VALIDATION_SEMANTICS_COMMIT and source.get("tree") == VALIDATION_SEMANTICS_TREE, "validation semantics source commit/tree changed")
    require(_git_stdout(root, "remote") == "origin", "validation semantics source remotes changed")
    require(_git_stdout(root, "remote", "get-url", "origin") == source["origin"] and _git_stdout(root, "remote", "get-url", "--push", "origin") == source["origin"], "validation semantics source origin changed")
    require(_git_stdout(root, "rev-parse", "HEAD") == VALIDATION_SEMANTICS_COMMIT and _git_stdout(root, "rev-parse", "HEAD^{tree}") == VALIDATION_SEMANTICS_TREE, "validation semantics checkout changed")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "--short", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    require(symbolic.returncode == 1 and symbolic.stdout == "" and symbolic.stderr == "", "validation semantics source is not detached")
    require(_git_stdout(root, "status", "--porcelain=v1", "--untracked-files=all") == "" and _git_stdout(root, "for-each-ref", "--format=%(refname)", "refs/heads") == "", "validation semantics source is not clean zero-branch")
    expected_paths = {
        "contract": root / "scripts/show_base/base_long_val_contract.py",
        "selector": root / "scripts/show_base/select_base_official_adapt.py",
    }
    for key, expected in expected_paths.items():
        source[key] = _plain_artifact(source.get(key), "validation semantics %s" % key)
        require(source[key]["path"] == str(expected), "validation semantics %s path changed" % key)
        relative = str(expected.relative_to(root))
        require(_git_stdout(root, "ls-files", "--error-unmatch", relative) == relative, "validation semantics %s is not tracked" % key)
    require(source["contract"]["sha256"] == VALIDATION_SEMANTICS_CONTRACT_SHA256 and source["selector"]["sha256"] == VALIDATION_SEMANTICS_SELECTOR_SHA256, "validation semantics source hashes changed")
    source["source_root"] = str(root)
    return source


def _runtime_validation_source(value: Any) -> Dict[str, Any]:
    require(isinstance(value, dict) and set(value) == RUNTIME_VALIDATION_SOURCE_KEYS, "runtime validation source schema changed")
    binding = dict(value)
    require(binding.get("format") == RUNTIME_VALIDATION_SOURCE_FORMAT, "runtime validation source format changed")
    root = canonical_directory(binding.get("source_root"), "runtime validation source root")
    require(str(root) == RUNTIME_VALIDATION_ROOT, "runtime validation source root changed")
    require(binding.get("origin") == "git@github.com:Xiangyue-Zhang/SemTalk.git", "runtime validation source origin changed")
    commit, tree = binding.get("commit"), binding.get("tree")
    require(commit == RUNTIME_VALIDATION_COMMIT and tree == RUNTIME_VALIDATION_TREE, "runtime validation source commit/tree changed")
    semantics = _validation_semantics_source(binding.get("validation_semantics_source"))
    require(binding.get("semantics_ancestry_verified") is True, "runtime validation ancestry proof changed")
    require(strict_json_equal(binding.get("autoencoder_provenance"), {"state_container": "state_dict", "load_mode": "full_half_embedding_net"}), "runtime autoencoder provenance contract changed")
    require(_git_stdout(root, "remote") == "origin", "runtime validation source remotes changed")
    require(_git_stdout(root, "remote", "get-url", "origin") == binding["origin"] and _git_stdout(root, "remote", "get-url", "--push", "origin") == binding["origin"], "runtime validation source origin changed")
    require(_git_stdout(root, "rev-parse", "HEAD") == commit and _git_stdout(root, "rev-parse", "HEAD^{tree}") == tree, "runtime validation source commit/tree changed")
    require(_git_stdout(root, "rev-parse", VALIDATION_SEMANTICS_COMMIT + "^{tree}") == VALIDATION_SEMANTICS_TREE, "validation semantics tree changed")
    _git_is_ancestor(root, VALIDATION_SEMANTICS_COMMIT, commit)
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "--short", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    require(symbolic.returncode == 1 and symbolic.stdout == "" and symbolic.stderr == "", "runtime validation source is not detached")
    require(_git_stdout(root, "status", "--porcelain=v1", "--untracked-files=all") == "" and _git_stdout(root, "for-each-ref", "--format=%(refname)", "refs/heads") == "", "runtime validation source is not clean zero-branch")
    expected_paths = {
        "contract": root / "scripts/show_base/base_long_val_contract.py",
        "evaluator": root / "scripts/show_base/evaluate_diffsheg_val_fgd.py",
        "selector": root / "scripts/show_base/select_base_official_adapt.py",
    }
    for key, expected in expected_paths.items():
        binding[key] = _plain_artifact(binding.get(key), "runtime validation %s" % key)
        require(binding[key]["path"] == str(expected), "runtime validation %s path changed" % key)
        relative = str(expected.relative_to(root))
        require(_git_stdout(root, "ls-files", "--error-unmatch", relative) == relative, "runtime validation %s is not tracked" % key)
    require(binding["contract"]["sha256"] == RUNTIME_VALIDATION_CONTRACT_SHA256 and binding["evaluator"]["sha256"] == RUNTIME_VALIDATION_EVALUATOR_SHA256 and binding["selector"]["sha256"] == RUNTIME_VALIDATION_SELECTOR_SHA256, "runtime validation source hashes changed")
    binding["source_root"] = str(root)
    binding["validation_semantics_source"] = semantics
    return binding


def _runtime_source_reference(binding: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "origin": binding["origin"], "source_root": binding["source_root"],
        "commit": binding["commit"], "tree": binding["tree"],
        "clean": True, "detached": True, "local_branches_at_commit": [],
    }


def _semantics_source_reference(binding: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "origin": binding["origin"], "source_root": binding["source_root"],
        "commit": binding["commit"], "tree": binding["tree"],
        "clean": True, "detached": True, "local_branches_at_commit": [],
    }


def _runtime_validation_proof_expected() -> Dict[str, Any]:
    unchanged = {
        "scripts/show_base/run_base_val_inference.py": RUNTIME_VALIDATION_INFERENCE_SHA256,
        "scripts/show_base/produce_base_val_measurement.py": RUNTIME_VALIDATION_MEASUREMENT_SHA256,
        "scripts/show_base/select_base_official_adapt_long.py": RUNTIME_VALIDATION_LONG_SELECTOR_SHA256,
    }
    projection = {
        path: {"evidence_sha256": digest, "runtime_sha256": digest}
        for path, digest in unchanged.items()
    }
    projection["scripts/show_base/base_long_val_contract.py"] = {
        "evidence_sha256": VALIDATION_SEMANTICS_CONTRACT_SHA256,
        "runtime_sha256": RUNTIME_VALIDATION_CONTRACT_SHA256,
    }
    projection["scripts/show_base/select_base_official_adapt.py"] = {
        "evidence_sha256": VALIDATION_SEMANTICS_SELECTOR_SHA256,
        "runtime_sha256": RUNTIME_VALIDATION_SELECTOR_SHA256,
    }
    return {
        "format": "semtalk_show_base_runtime_validation_successor_proof_v1",
        "evidence_source": {"commit": VALIDATION_SEMANTICS_COMMIT, "tree": VALIDATION_SEMANTICS_TREE},
        "runtime_source": {"commit": RUNTIME_VALIDATION_COMMIT, "tree": RUNTIME_VALIDATION_TREE},
        "ancestry_verified": True, "file_projection": projection,
    }


def _work_authority_runtime_binding(artifact: Mapping[str, Any], runtime: Mapping[str, Any], epoch: int) -> None:
    _path, raw, _digest, _size = safe_regular_bytes(artifact["path"], "e%d work authority runtime binding" % epoch, artifact["sha256"], artifact["bytes"])
    value = _strict_json_document(raw, "e%d work authority runtime binding" % epoch)
    expected_runtime = _runtime_source_reference(runtime)
    expected_evidence = _semantics_source_reference(runtime["validation_semantics_source"])
    require(isinstance(value.get("runtime_validation_source"), dict) and set(value["runtime_validation_source"]) == SOURCE_REFERENCE_KEYS and strict_json_equal(value["runtime_validation_source"], expected_runtime), "e%d work authority runtime validation source changed" % epoch)
    require(isinstance(value.get("validation_evidence_source"), dict) and set(value["validation_evidence_source"]) == SOURCE_REFERENCE_KEYS and strict_json_equal(value["validation_evidence_source"], expected_evidence), "e%d work authority validation evidence source changed" % epoch)
    require(isinstance(value.get("pipeline_source"), dict) and set(value["pipeline_source"]) == SOURCE_REFERENCE_KEYS and strict_json_equal(value["pipeline_source"], expected_evidence), "e%d work authority pipeline evidence source changed" % epoch)
    require(strict_json_equal(value.get("runtime_validation_proof"), _runtime_validation_proof_expected()), "e%d work authority runtime validation proof changed" % epoch)
    require(type(value.get("candidate_epoch")) is int and value["candidate_epoch"] == epoch, "e%d work authority epoch changed" % epoch)


def _work_authority_candidate_binding(artifact: Mapping[str, Any], candidate_receipt: Mapping[str, Any], epoch: int) -> None:
    _path, raw, _digest, _size = safe_regular_bytes(artifact["path"], "e%d work authority candidate binding" % epoch, artifact["sha256"], artifact["bytes"])
    value = _strict_json_document(raw, "e%d work authority candidate binding" % epoch)
    ready = value.get("producer_ready_receipt")
    require(isinstance(ready, dict), "e%d work authority producer receipt changed" % epoch)
    for key in ARTIFACT_KEYS:
        require(ready.get(key) == candidate_receipt[key] and type(ready.get(key)) is type(candidate_receipt[key]), "e%d work authority producer receipt %s changed" % (epoch, key))


def _authority_adapter_config(value: Any, runtime: Mapping[str, Any]) -> Dict[str, Any]:
    require(isinstance(value, dict) and set(value) == AUTHORITY_ADAPTER_CONFIG_KEYS, "authority adapter config schema changed")
    config = dict(value)
    config["train_root"] = str(canonical_directory(config.get("train_root"), "producer train root"))
    require(config.get("topology_mode") in {
        "validation_gated_w8_l128_g1024_empirical_acceleration",
        "validation_gated_w8_l256_g2048_empirical_acceleration",
    }, "authority adapter topology changed")
    producer_root = canonical_directory(config.get("producer_source_root"), "producer source root")
    config["producer_source_root"] = str(producer_root)
    require(
        config.get("expected_producer_trainer_sha256") == PRODUCER_TRAINER_SHA256
        and config.get("expected_producer_contract_sha256") == PRODUCER_CONTRACT_SHA256,
        "authority adapter producer source hashes changed",
    )
    require(
        isinstance(config.get("expected_frozen_inputs_sha256"), str)
        and HEX64.fullmatch(config["expected_frozen_inputs_sha256"]) is not None,
        "authority adapter expected_frozen_inputs_sha256 changed",
    )
    _git_authority(
        producer_root, PRODUCER_SOURCE_COMMIT, PRODUCER_SOURCE_TREE,
        "producer source",
    )
    producer_paths = {
        "trainer": producer_root / "scripts/show_base/train_base_official_adapt_long.py",
        "contract": producer_root / "scripts/show_base/base_v14_formal_contract.py",
    }
    producer_artifacts = {
        key: _artifact_from_path(str(path), "producer %s" % key)
        for key, path in producer_paths.items()
    }
    require(
        producer_artifacts["trainer"]["sha256"] == PRODUCER_TRAINER_SHA256
        and producer_artifacts["contract"]["sha256"] == PRODUCER_CONTRACT_SHA256,
        "producer source file hashes changed",
    )
    for key, path in producer_paths.items():
        relative = str(path.relative_to(producer_root))
        require(
            _git_stdout(producer_root, "ls-files", "--error-unmatch", relative) == relative,
            "producer %s is not tracked" % key,
        )
    config["schedule"] = _plain_artifact(config.get("schedule"), "authority schedule")
    config["val_inputs"] = _plain_artifact(config.get("val_inputs"), "authority validation inputs")
    config["pipeline"] = _plain_artifact(config.get("pipeline"), "authority validation pipeline")
    require(runtime["contract"]["path"] == str(Path(runtime["source_root"]) / "scripts/show_base/base_long_val_contract.py"), "runtime validation contract path changed")
    return config


def _authorize_argv(campaign: Mapping[str, Any], job: Mapping[str, Any]) -> List[str]:
    config = campaign["_adapter_config"]
    runtime = campaign["_runtime_validation_source"]
    semantics = runtime["validation_semantics_source"]
    return [
        campaign["_formal_python"]["argv0"], "-I",
        campaign["_control_source"]["authority_adapter"]["path"], "authorize",
        "--train-root", config["train_root"], "--epoch", str(job["epoch"]),
        "--topology-mode", config["topology_mode"],
        "--producer-source-root", config["producer_source_root"],
        "--expected-producer-trainer-sha256", config["expected_producer_trainer_sha256"],
        "--expected-producer-contract-sha256", config["expected_producer_contract_sha256"],
        "--validation-evidence-source-root", semantics["source_root"],
        "--expected-validation-evidence-contract-sha256", semantics["contract"]["sha256"],
        "--expected-validation-evidence-selector-sha256", semantics["selector"]["sha256"],
        "--runtime-validation-source-root", runtime["source_root"],
        "--expected-runtime-validation-contract-sha256", runtime["contract"]["sha256"],
        "--expected-runtime-validation-selector-sha256", runtime["selector"]["sha256"],
        "--schedule", config["schedule"]["path"],
        "--expected-schedule-sha256", config["schedule"]["sha256"],
        "--expected-frozen-inputs-sha256", config["expected_frozen_inputs_sha256"],
        "--val-inputs", config["val_inputs"]["path"],
        "--expected-val-inputs-sha256", config["val_inputs"]["sha256"],
        "--pipeline", config["pipeline"]["path"],
        "--expected-pipeline-sha256", config["pipeline"]["sha256"],
        "--output", job["authority_path"],
    ]


def _adapter_stdout_artifact(stdout: bytes, expected: Mapping[str, Any], label: str) -> str:
    require(0 < len(stdout) <= 4096, "%s stdout is empty/oversize" % label)
    value = _strict_json_document(stdout, "%s stdout" % label)
    require(set(value) == ARTIFACT_KEYS and strict_json_equal(value, expected), "%s stdout artifact changed" % label)
    canonical = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    require(stdout == canonical, "%s stdout encoding changed" % label)
    return _sha256(stdout)


def _candidate_receipt_artifact(job: Mapping[str, Any]) -> Dict[str, Any]:
    path, _raw, digest, size = safe_regular_bytes(
        job["candidate_receipt_path"], "e%d candidate-ready receipt" % job["epoch"]
    )
    require(path == Path(job["candidate_receipt_path"]), "candidate-ready receipt path changed")
    return {"path": str(path), "sha256": digest, "bytes": size}


def _runner_workload(job: Mapping[str, Any], campaign: Mapping[str, Any]) -> List[str]:
    source = campaign["_control_source"]
    formal = campaign["_formal_python"]
    argv = [
        "/bin/bash", source["launcher"]["path"],
        "--repo-root", source["root"],
        "--python", formal["argv0"],
        "--work-authority", job["work_authority"]["path"],
        "--expected-work-authority-sha256", job["work_authority"]["sha256"],
        "--run-root", job["run_root"],
        "--source-commit", source["commit"],
        "--source-tree", source["tree"],
        "--paspa-root", campaign["_paspa_root"],
        "--diffsheg-root", campaign["_diffsheg_root"],
        "--seed", str(campaign["_seed"]),
        "--diffsheg-batch-size", str(campaign["_diffsheg_batch_size"]),
    ]
    recovery = job.get("consumer_recovery")
    if recovery is not None:
        require(
            isinstance(recovery, dict)
            and set(recovery) == {
                "request", "authority", "claim",
            },
            "consumer recovery runner binding changed",
        )
        normalized = {
            label: _artifact(recovery[label], "consumer recovery %s" % label)
            for label in ("request", "authority", "claim")
        }
        argv.extend([
            "--recovery-authority", normalized["authority"]["path"],
            "--expected-recovery-authority-sha256", normalized["authority"]["sha256"],
            "--recovery-claim", normalized["claim"]["path"],
            "--expected-recovery-claim-sha256", normalized["claim"]["sha256"],
        ])
    return argv


def _runner_argv_v2(job: Mapping[str, Any], campaign: Mapping[str, Any]) -> List[str]:
    return [
        campaign["_formal_python"]["argv0"], campaign["_guarded_runner"]["path"],
        "--gpus", "0,1,2,3,4,5,6,7",
        "--cwd", campaign["_control_source"]["root"],
        "--status", job["runner_status_path"],
        "--log", job["runner_log_path"], "--", *_runner_workload(job, campaign),
    ]


def _validate_runner_argv_v2(argv: Any, job: Mapping[str, Any], campaign: Mapping[str, Any]) -> List[str]:
    require(isinstance(argv, list) and all(isinstance(token, str) and token and "\0" not in token for token in argv), "runner argv changed")
    expected = _runner_argv_v2(job, campaign)
    require(argv == expected, "runner/shell argv differs from the exact tracked live launcher command")
    return list(argv)


def load_campaign(
    path_text: Any, expected_sha256: str, expected_bytes: int,
    *, running_script: Optional[str] = None, running_executable: Optional[str] = None,
) -> Dict[str, Any]:
    campaign_path, raw, digest, size = safe_regular_bytes(path_text, "campaign", expected_sha256, expected_bytes)
    payload = strict_json(raw, "campaign")
    require(set(payload) == CAMPAIGN_KEYS, "campaign schema changed")
    _self_hashed(payload, "campaign_payload_sha256", "campaign")
    require(payload.get("format") == CAMPAIGN_FORMAT and payload.get("status") == "frozen_before_execution", "campaign state changed")
    require(payload.get("split") == "val" and payload.get("test_visible") is False and type(payload.get("test_measurements_authorized")) is int and payload["test_measurements_authorized"] == 0, "campaign is not val-only")
    require(payload.get("candidate_epochs") == list(CANDIDATE_EPOCHS) and all(type(x) is int for x in payload["candidate_epochs"]), "campaign epoch queue changed")
    state_root = canonical_directory(payload.get("state_root"), "state root")
    claims = canonical_directory(str(state_root / "job_claims"), "job claims")
    completions = canonical_directory(str(state_root / "completions"), "completions")
    statuses = canonical_directory(str(state_root / "runner_status"), "runner statuses")
    logs = canonical_directory(str(state_root / "runner_logs"), "runner logs")
    authorities = canonical_directory(str(state_root / "authorities"), "work authorities")
    authorizations = canonical_directory(str(state_root / "authorizations"), "authorizations")
    require(payload.get("campaign_claim_path") == str(state_root / "campaign.claim.json") and payload.get("summary_path") == str(state_root / "final_summary.json"), "campaign output paths changed")
    actual_script = str(Path(__file__).resolve(strict=True)) if running_script is None else running_script
    control = _control_source(payload.get("control_source"), actual_script)
    runtime_validation = _runtime_validation_source(payload.get("runtime_validation_source"))
    adapter_config = _authority_adapter_config(payload.get("authority_adapter_config"), runtime_validation)
    formal = _formal_python(payload.get("formal_python"), running_executable=running_executable)
    runtime_contract = _plain_artifact(payload.get("formal_python_runtime_contract"), "formal Python runtime contract")
    require(runtime_contract["path"] == str(Path(control["root"]) / "scripts/show_base/formal_python_runtime_contract.sh"), "formal Python runtime contract path changed")
    runtime_relative = str(Path(runtime_contract["path"]).relative_to(Path(control["root"])))
    require(_git_stdout(Path(control["root"]), "ls-files", "--error-unmatch", runtime_relative) == runtime_relative, "formal Python runtime contract is not tracked")
    runner = _plain_artifact(payload.get("guarded_runner"), "guarded runner", executable=True)
    require(runner["path"] == "/tmp/globaldiff_guarded_runner.py", "guarded runner path changed")
    verifier = _plain_artifact(payload.get("guard_verifier"), "guard verifier")
    require(verifier["path"] == "/tmp/verify_globaldiff_guards.py", "guard verifier path changed")
    reconciliation = canonical_output_path(payload.get("reconciliation_path"), "e400 reconciliation")
    selection_root = canonical_output_path(payload.get("reconcile_selection_root"), "reconcile selection root")
    final_manifest = _lexical_absolute(payload.get("final_manifest_path"), "final producer manifest")
    final_status = _lexical_absolute(payload.get("final_status_path"), "final producer status")
    require(final_manifest == Path(adapter_config["train_root"]) / "candidate_manifest.json" and final_status == Path(adapter_config["train_root"]) / "status.json", "final producer paths changed")
    require(reconciliation == state_root / "reconciliation.json", "reconciliation output path changed")
    require(not str(selection_root).startswith(str(state_root) + "/"), "selection output overlaps private queue state")
    jobs_value = payload.get("jobs")
    require(isinstance(jobs_value, list) and len(jobs_value) == len(CANDIDATE_EPOCHS), "campaign jobs changed")
    result: Dict[str, Any] = dict(payload)
    result.update({
        "_artifact": {"path": str(campaign_path), "sha256": digest, "bytes": size},
        "_state_root": state_root, "_claims_dir": claims,
        "_completions_dir": completions, "_runner_status_dir": statuses,
        "_runner_logs_dir": logs, "_control_source": control,
        "_authorities_dir": authorities, "_authorizations_dir": authorizations,
        "_runtime_validation_source": runtime_validation,
        "_adapter_config": adapter_config,
        "_formal_python": formal, "_runtime_contract": runtime_contract,
        "_guarded_runner": runner, "_guard_verifier": verifier,
        "_reconciliation_path": reconciliation, "_selection_root": selection_root,
        "_final_manifest_path": final_manifest, "_final_status_path": final_status,
    })
    # These launcher values are frozen once at campaign publication.
    result["_paspa_root"] = str(canonical_directory(payload.get("paspa_root"), "PASPA root"))
    result["_diffsheg_root"] = str(canonical_directory(payload.get("diffsheg_root"), "DiffSHEG root"))
    result["_seed"] = _exact_int(payload.get("seed"), "seed", 0)
    result["_diffsheg_batch_size"] = _exact_int(payload.get("diffsheg_batch_size"), "DiffSHEG batch size", 1)
    jobs: List[Dict[str, Any]] = []
    run_roots: set = set()
    for epoch, value in zip(CANDIDATE_EPOCHS, jobs_value):
        require(isinstance(value, dict) and set(value) == JOB_KEYS and type(value.get("epoch")) is int and value["epoch"] == epoch, "job order/schema changed")
        job = dict(value)
        require(job.get("candidate_receipt_path") == str(Path(adapter_config["train_root"]) / "candidate_receipts" / ("epoch-%04d.json" % epoch)), "candidate receipt path changed")
        require(job.get("authority_path") == str(authorities / ("epoch-%04d.json" % epoch)), "work authority output path changed")
        require(job.get("authorization_path") == str(authorizations / ("epoch-%04d.json" % epoch)), "authorization receipt path changed")
        run_root = canonical_output_path(job.get("run_root"), "e%d run root" % epoch)
        require(str(run_root) not in run_roots, "run roots are not unique")
        run_roots.add(str(run_root))
        job["run_root"] = str(run_root)
        require(job.get("measurement_path") == str(run_root / "candidates" / ("e%d" % epoch) / "live-measurement.json"), "measurement path changed")
        require(job.get("completion_path") == str(completions / ("epoch-%04d.json" % epoch)), "completion path changed")
        require(job.get("runner_status_path") == str(statuses / ("epoch-%04d.json" % epoch)), "runner status path changed")
        require(job.get("runner_log_path") == str(logs / ("epoch-%04d.log" % epoch)), "runner log path changed")
        jobs.append(job)
    result["_jobs"] = jobs
    return result


def _runner_status(job: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact, raw, digest, size = safe_regular_bytes(job["runner_status_path"], "runner status")
    value = _strict_json_document(raw, "runner status")
    require(set(value) == RUNNER_STATUS_KEYS, "runner status schema changed")
    require(value.get("state") == "finished" and type(value.get("return_code")) is int and value["return_code"] == 0, "runner did not finish successfully")
    for key in ("error", "cleanup_error", "restore_error", "received_signal"):
        require(key in value and value[key] is None, "runner status %s is not explicit null" % key)
    require(value.get("command") == _runner_workload(job, job["_campaign"]), "runner status command changed")
    _exact_int(value.get("wrapper_pid"), "runner wrapper PID", 2)
    _exact_int(value.get("child_pid"), "runner child PID", 2)
    require(value["wrapper_pid"] != value["child_pid"], "runner wrapper/child PIDs collide")
    restored = value.get("restored_guards")
    require(isinstance(restored, dict) and set(restored) == {str(i) for i in GPU_INDICES}, "runner did not restore GPU0..7")
    require(all(type(restored[str(i)]) is int and restored[str(i)] > 1 for i in GPU_INDICES) and len(set(restored.values())) == 8, "runner restored guard PIDs changed")
    updated = value.get("updated_at")
    require(
        (isinstance(updated, str) and bool(updated.strip()))
        or (type(updated) in (int, float) and math.isfinite(float(updated))),
        "runner updated_at changed",
    )
    require(value["wrapper_pid"] not in restored.values() and value["child_pid"] not in restored.values(), "runner/child PID collides with guard")
    return {"path": str(artifact), "sha256": digest, "bytes": size}, value


def _failed_runner_status(
    job: Mapping[str, Any], expected_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact, raw, digest, size = safe_regular_bytes(
        job["runner_status_path"], "failed runner status", expected_sha256
    )
    value = _strict_json_document(raw, "failed runner status")
    require(set(value) == RUNNER_STATUS_KEYS, "failed runner status schema changed")
    require(
        value.get("state") == "failed"
        and type(value.get("return_code")) is int
        and value["return_code"] == 1,
        "failed runner is not the exact rc1 incident",
    )
    for key in ("error", "cleanup_error", "restore_error", "received_signal"):
        require(
            key in value and value[key] is None,
            "failed runner status %s is not explicit null" % key,
        )
    require(
        value.get("command") == _runner_workload(job, job["_campaign"]),
        "failed runner command changed",
    )
    _exact_int(value.get("wrapper_pid"), "failed runner wrapper PID", 2)
    _exact_int(value.get("child_pid"), "failed runner child PID", 2)
    require(
        value["wrapper_pid"] != value["child_pid"],
        "failed runner wrapper/child PIDs collide",
    )
    require(
        value["wrapper_pid"] == FAILED_WRAPPER_PID
        and value["child_pid"] == FAILED_CHILD_PID,
        "failed runner is not the pinned wrapper/child process pair",
    )
    restored = value.get("restored_guards")
    require(
        isinstance(restored, dict)
        and set(restored) == {str(i) for i in GPU_INDICES}
        and all(type(restored[str(i)]) is int and restored[str(i)] > 1 for i in GPU_INDICES)
        and len(set(restored.values())) == 8,
        "failed runner did not restore eight distinct guards",
    )
    require(
        value["wrapper_pid"] not in restored.values()
        and value["child_pid"] not in restored.values(),
        "failed runner PID collides with restored guard",
    )
    return {"path": str(artifact), "sha256": digest, "bytes": size}, value


FAILED_RUN_INVENTORY_FORMAT = "semtalk_show_base_failed_consumer_run_inventory_v1"
FAILED_SHARD_MARKER = (
    "ValInferenceContractError: scripts.show_base.build_base_features "
    "was imported from another checkout"
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


def _failed_run_inventory(path_text: Any) -> Dict[str, Any]:
    root = canonical_directory(path_text, "failed e1 run root")
    expected_directories = [
        ".", "candidates", "candidates/e1", "candidates/e1/shards", "logs",
    ]
    expected_files = [
        "diffsheg-evaluator-bundle.json", "work-preflight.json",
        "logs/evaluator-preflight.log", "logs/work-inspect.json",
        "logs/work-preflight.log",
        *["logs/e1-shard%d.log" % shard for shard in GPU_INDICES],
    ]
    observed_directories = ["."]
    observed_files: List[str] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root).as_posix()
            metadata = child.lstat()
            require(not stat.S_ISLNK(metadata.st_mode), "failed run contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                observed_directories.append(relative)
                pending.append(child)
            else:
                require(
                    stat.S_ISREG(metadata.st_mode),
                    "failed run contains a non-regular artifact",
                )
                observed_files.append(relative)
    observed_directories.sort()
    observed_files.sort()
    require(
        observed_directories == sorted(expected_directories)
        and observed_files == sorted(expected_files),
        "failed run is not the exact zero-semantic-output tree",
    )
    files: List[Dict[str, Any]] = []
    shard_payloads: List[bytes] = []
    shard_shas: List[str] = []
    for relative in observed_files:
        _path, raw, digest, size = safe_regular_bytes(
            root / relative, "failed run artifact %s" % relative
        )
        files.append({"relative_path": relative, "sha256": digest, "bytes": size})
        if re.fullmatch(r"logs/e1-shard[0-7]\.log", relative):
            shard_payloads.append(raw)
            shard_shas.append(digest)
    observed_pins = {
        row["relative_path"]: (row["sha256"], row["bytes"])
        for row in files
    }
    require(
        observed_pins == FAILED_RUN_FILE_PINS,
        "failed run artifact SHA/byte pins changed",
    )
    require(
        len(shard_payloads) == 8
        and len(set(shard_shas)) == 1
        and len(set(shard_payloads)) == 1,
        "failed shard logs are not eight identical pre-inference failures",
    )
    try:
        shard_text = shard_payloads[0].decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise SupervisorError("failed shard log is not UTF-8") from error
    require(
        shard_text.count(FAILED_SHARD_MARKER) == 1,
        "failed shard log does not contain the exact pinned failure marker",
    )
    return {
        "format": FAILED_RUN_INVENTORY_FORMAT,
        "root": str(root),
        "directories": sorted(expected_directories),
        "files": files,
        "shard_log_sha256": shard_shas[0],
        "shard_failure_marker": FAILED_SHARD_MARKER,
        "semantic_outputs": [],
    }


def _failed_process_proof(
    status: Mapping[str, Any], now: float,
) -> Dict[str, Any]:
    wrapper_pid = _exact_int(
        status.get("wrapper_pid"), "failed runner wrapper PID", 2
    )
    child_pid = _exact_int(
        status.get("child_pid"), "failed runner child PID", 2
    )
    require(wrapper_pid != child_pid, "failed runner PIDs collide")
    for pid, role in ((wrapper_pid, "wrapper"), (child_pid, "child")):
        require(
            not os.path.lexists("/proc/%d" % pid),
            "failed runner %s PID %d is still live or has been reused" % (role, pid),
        )
    command = status.get("command")
    require(
        isinstance(command, list)
        and all(isinstance(token, str) and token and "\0" not in token for token in command),
        "failed runner command proof changed",
    )
    return {
        "wrapper_pid": wrapper_pid,
        "child_pid": child_pid,
        "wrapper_proc_state": "absent",
        "child_proc_state": "absent",
        "runner_command": list(command),
        "checked_unix": _finite_number(
            now, "failed runner process proof time", 0.000001
        ),
    }


def _measurement(job: Mapping[str, Any], expected_sha: str, expected_payload_sha: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path, raw, digest, size = safe_regular_bytes(job["measurement_path"], "live measurement", expected_sha)
    value = _strict_json_document(raw, "live measurement")
    require(set(value) == MEASUREMENT_KEYS, "live measurement schema changed")
    require(value.get("receipt_payload_sha256") == expected_payload_sha == _bridge_payload_sha(value), "live measurement payload SHA changed")
    require(value.get("format") == MEASUREMENT_FORMAT and value.get("status") == "complete" and value.get("split") == "val" and value.get("test_visible") is False and value.get("selection_eligible") is False, "live measurement is not selection-ineligible val work")
    require(type(value.get("candidate_epoch")) is int and value["candidate_epoch"] == job["epoch"], "live measurement epoch changed")
    authority = value.get("work_authority")
    require(isinstance(authority, dict) and authority.get("path") == job["work_authority"]["path"] and authority.get("sha256") == job["work_authority"]["sha256"] and authority.get("bytes") == job["work_authority"]["bytes"], "live measurement work authority changed")
    require(strict_json_equal(value.get("execution_contract"), {"expected_shards": 8, "exact_once": True, "finite": True, "may_influence_training": False, "requires_e400_reconciliation_for_selection": True}), "live measurement execution contract changed")
    metrics = value.get("metrics")
    require(isinstance(metrics, dict) and set(metrics) == {"fgd"}, "live measurement metrics changed")
    _finite_number(metrics["fgd"], "live DiffSHEG FGD", 0.0)
    return {"path": str(path), "sha256": digest, "bytes": size, "receipt_payload_sha256": expected_payload_sha}, value


def _verify_runner_log(job: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    path, raw, digest, size = safe_regular_bytes(job["runner_log_path"], "runner exact stdout")
    require(size <= MAX_RUNNER_LOG_BYTES, "runner stdout exceeds bound")
    fields = raw.split(b"\0")
    require(len(fields) == 4 and fields[-1] == b"" and all(fields[:3]), "runner stdout is not one exact trailing-NUL triple")
    try:
        measurement_path, measurement_sha, payload_sha = (field.decode("ascii") for field in fields[:3])
    except UnicodeDecodeError as error:
        raise SupervisorError("runner stdout triple is not ASCII") from error
    require(measurement_path == job["measurement_path"] and HEX64.fullmatch(measurement_sha) is not None and HEX64.fullmatch(payload_sha) is not None, "runner stdout triple changed")
    measurement, value = _measurement(job, measurement_sha, payload_sha)
    return {"path": str(path), "sha256": digest, "bytes": size}, measurement, value


def _run_process(argv: Sequence[str]) -> int:
    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return int(subprocess.run(list(argv), shell=False, check=False, env=environment).returncode)


def _run_capture(argv: Sequence[str]) -> Tuple[int, bytes, bytes]:
    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(list(argv), shell=False, check=False, env=environment, capture_output=True)
    return int(completed.returncode), completed.stdout, completed.stderr


def _guard_verify(campaign: Mapping[str, Any], status: Mapping[str, Any], capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]]) -> Tuple[List[str], str, Dict[str, int]]:
    restored = status["restored_guards"]
    argv = [campaign["_formal_python"]["argv0"], campaign["_guard_verifier"]["path"], *[str(restored[str(i)]) for i in GPU_INDICES]]
    rc, stdout, stderr = capture(argv)
    require(type(rc) is int and rc == 0 and stderr == b"" and len(stdout) <= 1024, "guard verifier failed or emitted stderr/oversize output")
    try:
        text = stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise SupervisorError("guard verifier output is not ASCII") from error
    match = GUARD_PASS_RE.fullmatch(text)
    require(match is not None, "guard verifier output changed")
    observed = {str(i): int(match.group(i + 1)) for i in GPU_INDICES}
    require(observed == restored, "guard verifier PIDs differ from runner restoration")
    return argv, text, observed


def _bridge_replay(campaign: Mapping[str, Any], job: Mapping[str, Any], measurement: Mapping[str, Any], capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]]) -> List[str]:
    authority = job["work_authority"]
    argv = [
        campaign["_formal_python"]["argv0"], "-I", "-c", BRIDGE_REPLAY_CODE,
        campaign["_control_source"]["bridge"]["path"], measurement["path"],
        measurement["sha256"], measurement["receipt_payload_sha256"],
        str(job["epoch"]), authority["path"], authority["sha256"],
        str(authority["bytes"]),
    ]
    rc, stdout, stderr = capture(argv)
    require(type(rc) is int and rc == 0 and stdout == b"PASS\n" and stderr == b"", "fresh isolated bridge measurement replay failed")
    return argv


def _recovery_bridge_argv(
    campaign: Mapping[str, Any], command: str,
    request: Mapping[str, Any],
) -> List[str]:
    require(
        command in {"inspect-recovery", "reserve-recovery"},
        "unknown recovery bridge command",
    )
    request_artifact = _artifact(request, "consumer recovery request")
    return [
        campaign["_formal_python"]["argv0"], "-I",
        campaign["_control_source"]["bridge"]["path"], command,
        "--request", request_artifact["path"],
        "--expected-request-sha256", request_artifact["sha256"],
        "--output-authority",
        str(campaign["_state_root"] / "recovery-authority.epoch-0001.json"),
    ]


def _recovery_stdout_document(
    raw: bytes, label: str,
) -> Dict[str, Any]:
    require(0 < len(raw) <= 4096, "%s stdout is empty/oversize" % label)
    value = _strict_json_document(raw, "%s stdout" % label)
    require(
        raw == (
            json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        "%s stdout is not one canonical JSON line" % label,
    )
    return value


def _recovery_core_preview(
    value: Any, expected_path: Path, label: str,
) -> Dict[str, Any]:
    require(isinstance(value, dict) and set(value) == ARTIFACT_KEYS, "%s schema changed" % label)
    path = canonical_output_path(value.get("path"), "%s path" % label)
    require(path == expected_path, "%s path changed" % label)
    digest = value.get("sha256")
    size = value.get("bytes")
    require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None, "%s SHA changed" % label)
    _exact_int(size, "%s bytes" % label, 1)
    return {"path": str(path), "sha256": digest, "bytes": size}


def _inspect_recovery_stdout(
    campaign: Mapping[str, Any], raw: bytes,
) -> Tuple[Dict[str, Any], str]:
    value = _recovery_stdout_document(raw, "inspect-recovery")
    require(
        set(value) == {"status", "recovery_authority", "recovery_claim_path"}
        and value.get("status") == "ready",
        "inspect-recovery stdout schema/state changed",
    )
    expected_authority = campaign["_state_root"] / "recovery-authority.epoch-0001.json"
    authority = _recovery_core_preview(
        value["recovery_authority"], expected_authority,
        "inspect-recovery authority preview",
    )
    expected_claim = (
        Path(campaign["_adapter_config"]["train_root"])
        / "live_val_consumer_recovery_claims" / "epoch-0001.json"
    )
    claim_path = _lexical_absolute(
        value.get("recovery_claim_path"), "inspect-recovery claim path"
    )
    require(claim_path == expected_claim, "inspect-recovery claim path changed")
    return authority, str(claim_path)


def _reserve_recovery_stdout(
    campaign: Mapping[str, Any], raw: bytes,
    preview_authority: Mapping[str, Any], preview_claim_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    value = _recovery_stdout_document(raw, "reserve-recovery")
    require(
        set(value) == {"status", "recovery_authority", "recovery_claim"}
        and value.get("status") == "reserved",
        "reserve-recovery stdout schema/state changed",
    )
    authority = _artifact(
        value["recovery_authority"], "reserved recovery authority"
    )
    claim = _artifact(value["recovery_claim"], "reserved recovery claim")
    require(
        strict_json_equal(authority, preview_authority),
        "reserved recovery authority differs from inspected preview",
    )
    require(
        claim["path"] == preview_claim_path,
        "reserved recovery claim path differs from inspected preview",
    )
    return authority, claim


def _revalidate_control(campaign: Mapping[str, Any]) -> None:
    _control_source(campaign["control_source"], campaign["_control_source"]["supervisor"]["path"])
    runtime = _runtime_validation_source(campaign["runtime_validation_source"])
    _authority_adapter_config(campaign["authority_adapter_config"], runtime)
    _formal_python(campaign["formal_python"], running_executable=campaign["_formal_python"]["argv0"])
    _plain_artifact(campaign["formal_python_runtime_contract"], "formal Python runtime contract")
    _plain_artifact(campaign["guarded_runner"], "guarded runner", executable=True)
    _plain_artifact(campaign["guard_verifier"], "guard verifier")


def _runtime_contract_check(campaign: Mapping[str, Any], capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]]) -> None:
    argv = ["/bin/bash", campaign["_runtime_contract"]["path"], campaign["_formal_python"]["argv0"], "semtalk"]
    rc, stdout, stderr = capture(argv)
    require(type(rc) is int and rc == 0 and stdout == b"" and stderr == b"", "formal Python runtime contract failed")


def _completion_body(campaign: Mapping[str, Any], job: Mapping[str, Any], status_artifact: Mapping[str, Any], status: Mapping[str, Any], log_artifact: Mapping[str, Any], measurement: Mapping[str, Any], measurement_value: Mapping[str, Any], bridge_replay_argv: Sequence[str], verifier_argv: Sequence[str], verifier_stdout: str, guards: Mapping[str, int], now: float) -> Dict[str, Any]:
    recovery = job.get("consumer_recovery")
    normalized_recovery = None
    if recovery is not None:
        require(
            isinstance(recovery, dict)
            and set(recovery) == {"request", "authority", "claim"},
            "completion recovery binding changed",
        )
        normalized_recovery = {
            label: _artifact(recovery[label], "completion recovery %s" % label)
            for label in ("request", "authority", "claim")
        }
        observed_claim = measurement_value.get("consumer_claim")
        require(
            isinstance(observed_claim, dict)
            and all(
                observed_claim.get(key) == normalized_recovery["claim"][key]
                and type(observed_claim.get(key)) is type(normalized_recovery["claim"][key])
                for key in ARTIFACT_KEYS
            ),
            "recovered measurement does not bind the reserved recovery claim",
        )
    return _add_self_hash({
        "format": COMPLETION_FORMAT, "status": "complete", "split": "val",
        "test_visible": False, "selection_eligible": False,
        "test_measurements_authorized": 0, "candidate_epoch": job["epoch"],
        "campaign": campaign["_artifact"], "candidate_receipt": job["candidate_receipt"],
        "work_authority": job["work_authority"], "authorization": job["authorization"],
        "measurement": dict(measurement), "runner_status": dict(status_artifact),
        "runner_log": dict(log_artifact), "guard_verifier": campaign["_guard_verifier"],
        "bridge_replay_argv": list(bridge_replay_argv),
        "bridge_replay_stdout": "PASS\n",
        "guard_verifier_argv": list(verifier_argv),
        "guard_verifier_stdout": verifier_stdout, "restored_guards": dict(guards),
        "validation_diffsheg_fgd": measurement_value["metrics"]["fgd"],
        "consumer_recovery": normalized_recovery,
        "completed_unix": _finite_number(now, "completion time", 0.000001),
    }, "receipt_payload_sha256")


def _job_claim_path_v2(campaign: Mapping[str, Any], epoch: int) -> Path:
    return campaign["_claims_dir"] / ("epoch-%04d.json" % epoch)


def _completion_path_v2(campaign: Mapping[str, Any], epoch: int) -> Path:
    return campaign["_completions_dir"] / ("epoch-%04d.json" % epoch)


def _job_claim_v2(campaign: Mapping[str, Any], job: Mapping[str, Any], candidate_receipt: Mapping[str, Any], authorize_argv: Sequence[str], now: float) -> Dict[str, Any]:
    return _add_self_hash({
        "format": JOB_CLAIM_FORMAT, "status": "claimed", "candidate_epoch": job["epoch"],
        "campaign": campaign["_artifact"], "candidate_receipt": dict(candidate_receipt),
        "authority_path": job["authority_path"],
        "run_root": job["run_root"], "measurement_path": job["measurement_path"],
        "authorize_argv_sha256": _sha256(b"\0".join(os.fsencode(x) for x in authorize_argv) + b"\0"),
        "created_unix": _finite_number(now, "job claim time", 0.000001),
    }, "claim_payload_sha256")


def _load_job_claim_v2(campaign: Mapping[str, Any], job: Mapping[str, Any], candidate_receipt: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact, value = _read_existing_json(_job_claim_path_v2(campaign, job["epoch"]), "job claim")
    require(set(value) == JOB_CLAIM_KEYS, "job claim schema changed")
    expected = _job_claim_v2(campaign, job, candidate_receipt, _authorize_argv(campaign, job), value.get("created_unix"))
    require(strict_json_equal(value, expected), "e%d job claim changed" % job["epoch"])
    return artifact, value


def _authorization_body(campaign: Mapping[str, Any], job: Mapping[str, Any], authorize_argv: Sequence[str], adapter_stdout_sha: str, runner_argv: Sequence[str], now: float) -> Dict[str, Any]:
    recovery = job.get("consumer_recovery")
    body = {
        "format": (
            RECOVERY_AUTHORIZATION_FORMAT if recovery is not None
            else AUTHORIZATION_FORMAT
        ),
        "status": "complete",
        "candidate_epoch": job["epoch"], "campaign": campaign["_artifact"],
        "candidate_receipt": job["candidate_receipt"],
        "work_authority": job["work_authority"],
        "adapter": campaign["_control_source"]["authority_adapter"],
        "adapter_argv": list(authorize_argv), "adapter_stdout_sha256": adapter_stdout_sha,
        "runner_argv": list(runner_argv),
        "completed_unix": _finite_number(now, "authorization time", 0.000001),
    }
    if recovery is not None:
        require(
            isinstance(recovery, dict)
            and set(recovery) == {"request", "authority", "claim"},
            "recovery authorization binding changed",
        )
        body.update({
            "recovery_request": _artifact(
                recovery["request"], "recovery authorization request"
            ),
            "recovery_authority": _artifact(
                recovery["authority"], "recovery authorization authority"
            ),
            "recovery_claim": _artifact(
                recovery["claim"], "recovery authorization claim"
            ),
        })
    return _add_self_hash(body, "receipt_payload_sha256")


def _load_authorization(campaign: Mapping[str, Any], job: Mapping[str, Any], candidate_receipt: Mapping[str, Any], authority: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact, value = _read_existing_json(Path(job["authorization_path"]), "authorization receipt")
    recovery = job.get("consumer_recovery")
    expected_keys = (
        RECOVERY_AUTHORIZATION_KEYS if recovery is not None else AUTHORIZATION_KEYS
    )
    expected_format = (
        RECOVERY_AUTHORIZATION_FORMAT if recovery is not None
        else AUTHORIZATION_FORMAT
    )
    require(set(value) == expected_keys and value.get("format") == expected_format and value.get("status") == "complete" and value.get("candidate_epoch") == job["epoch"], "authorization receipt schema/state changed")
    _self_hashed(value, "receipt_payload_sha256", "authorization receipt")
    require(strict_json_equal(value.get("campaign"), campaign["_artifact"]) and strict_json_equal(value.get("candidate_receipt"), candidate_receipt) and strict_json_equal(value.get("work_authority"), authority), "authorization receipt authority changed")
    require(value.get("adapter") == campaign["_control_source"]["authority_adapter"] and value.get("adapter_argv") == _authorize_argv(campaign, job), "authorization adapter command changed")
    expected_runner = _validate_runner_argv_v2(value.get("runner_argv"), {**job, "work_authority": authority}, campaign)
    require(value["runner_argv"] == expected_runner and isinstance(value.get("adapter_stdout_sha256"), str) and HEX64.fullmatch(value["adapter_stdout_sha256"]) is not None, "authorization runner/stdout binding changed")
    if recovery is not None:
        require(
            isinstance(recovery, dict)
            and set(recovery) == {"request", "authority", "claim"}
            and all(
                strict_json_equal(
                    value["recovery_%s" % label],
                    _artifact(recovery[label], "recovery authorization %s" % label),
                )
                for label in ("request", "authority", "claim")
            ),
            "recovery authorization artifacts changed",
        )
    _finite_number(value.get("completed_unix"), "authorization time", 0.000001)
    return artifact, value


def _load_completion_v2(campaign: Mapping[str, Any], job: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact, value = _read_existing_json(_completion_path_v2(campaign, job["epoch"]), "completion")
    require(set(value) == COMPLETION_KEYS and value.get("format") == COMPLETION_FORMAT and value.get("status") == "complete", "completion schema/state changed")
    _self_hashed(value, "receipt_payload_sha256", "completion")
    require(value.get("candidate_epoch") == job["epoch"] and strict_json_equal(value.get("campaign"), campaign["_artifact"]), "completion authority changed")
    candidate_receipt = _artifact(value.get("candidate_receipt"), "completed candidate receipt")
    require(candidate_receipt["path"] == job["candidate_receipt_path"], "completed candidate receipt path changed")
    _load_job_claim_v2(campaign, job, candidate_receipt)
    authority = _artifact(value.get("work_authority"), "completed work authority")
    require(authority["path"] == job["authority_path"], "completed work authority path changed")
    _work_authority_runtime_binding(authority, campaign["_runtime_validation_source"], job["epoch"])
    _work_authority_candidate_binding(authority, candidate_receipt, job["epoch"])
    recovery_value = value.get("consumer_recovery")
    recovery = None
    if recovery_value is not None:
        require(
            isinstance(recovery_value, dict)
            and set(recovery_value) == {"request", "authority", "claim"},
            "completion recovery schema changed",
        )
        recovery = {
            label: _artifact(recovery_value[label], "completed recovery %s" % label)
            for label in ("request", "authority", "claim")
        }
        require(
            strict_json_equal(recovery, recovery_value),
            "completion recovery artifacts changed",
        )
    replay_job = {
        **job, "candidate_receipt": candidate_receipt,
        "work_authority": authority, "consumer_recovery": recovery,
    }
    authorization_artifact, authorization = _load_authorization(campaign, replay_job, candidate_receipt, authority)
    require(strict_json_equal(value.get("authorization"), authorization_artifact), "completion authorization receipt changed")
    replay_job["authorization"] = authorization_artifact
    measurement, measurement_value = _measurement(replay_job, value["measurement"]["sha256"], value["measurement"]["receipt_payload_sha256"])
    require(strict_json_equal(measurement, value["measurement"]), "completion measurement changed")
    if recovery is not None:
        observed_claim = measurement_value.get("consumer_claim")
        require(
            isinstance(observed_claim, dict)
            and all(
                observed_claim.get(key) == recovery["claim"][key]
                and type(observed_claim.get(key)) is type(recovery["claim"][key])
                for key in ARTIFACT_KEYS
            ),
            "completed recovered measurement claim changed",
        )
    replay_job["_campaign"] = campaign
    status_artifact, status = _runner_status(replay_job)
    require(strict_json_equal(status_artifact, value["runner_status"]), "completion runner status changed")
    log_artifact, log_measurement, _log_value = _verify_runner_log(replay_job)
    require(strict_json_equal(log_artifact, value["runner_log"]) and strict_json_equal(log_measurement, measurement), "completion runner log changed")
    restored = status["restored_guards"]
    expected_guard_argv = [campaign["_formal_python"]["argv0"], campaign["_guard_verifier"]["path"], *[str(restored[str(i)]) for i in GPU_INDICES]]
    require(value.get("guard_verifier") == campaign["_guard_verifier"] and value.get("guard_verifier_argv") == expected_guard_argv, "completion guard verifier authority changed")
    match = GUARD_PASS_RE.fullmatch(value.get("guard_verifier_stdout", ""))
    require(match is not None and {str(i): int(match.group(i + 1)) for i in GPU_INDICES} == restored == value.get("restored_guards"), "completion guard proof changed")
    expected_bridge = [
        campaign["_formal_python"]["argv0"], "-I", "-c", BRIDGE_REPLAY_CODE,
        campaign["_control_source"]["bridge"]["path"], measurement["path"],
        measurement["sha256"], measurement["receipt_payload_sha256"], str(job["epoch"]),
        authority["path"], authority["sha256"], str(authority["bytes"]),
    ]
    require(value.get("bridge_replay_argv") == expected_bridge and value.get("bridge_replay_stdout") == "PASS\n", "completion bridge replay proof changed")
    _finite_number(value.get("validation_diffsheg_fgd"), "completion FGD", 0.0)
    require(float(value["validation_diffsheg_fgd"]) == float(measurement_value["metrics"]["fgd"]), "completion FGD changed")
    return artifact, value


def _scan_v2(campaign: Mapping[str, Any]) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], Optional[Dict[str, Any]]]:
    allowed_epoch_names = {"epoch-%04d.json" % epoch for epoch in CANDIDATE_EPOCHS}
    allowed_log_names = {"epoch-%04d.log" % epoch for epoch in CANDIDATE_EPOCHS}
    for directory, allowed, label in (
        (campaign["_claims_dir"], allowed_epoch_names, "job-claims directory"),
        (campaign["_completions_dir"], allowed_epoch_names, "completions directory"),
        (campaign["_runner_status_dir"], allowed_epoch_names, "runner-status directory"),
        (campaign["_runner_logs_dir"], allowed_log_names, "runner-log directory"),
        (campaign["_authorities_dir"], allowed_epoch_names, "authorities directory"),
        (campaign["_authorizations_dir"], allowed_epoch_names, "authorizations directory"),
    ):
        names = {entry.name for entry in directory.iterdir()}
        require(names.issubset(allowed), "%s contains an unknown entry" % label)
    state_allowed = {
        "campaign.json", "campaign.claim.json", "final_summary.json",
        "active_invocation.claim.json", "producer_reconcile.claim.json",
        "bridge_reconcile.claim.json", "reconciliation.json",
        "recovery-request.epoch-0001.json",
        "recovery-authority.epoch-0001.json",
        "job_claims", "completions", "runner_status", "runner_logs",
        "authorities", "authorizations",
    }
    require({entry.name for entry in campaign["_state_root"].iterdir()}.issubset(state_allowed), "state root contains an unknown entry")
    completed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    head: Optional[Dict[str, Any]] = None
    gap = False
    for job in campaign["_jobs"]:
        claim = os.path.lexists(_job_claim_path_v2(campaign, job["epoch"]))
        done = os.path.lexists(_completion_path_v2(campaign, job["epoch"]))
        if done:
            require(claim and not gap, "queue skipped ahead")
            completed.append(_load_completion_v2(campaign, job))
        elif claim:
            raise SupervisorError("e%d claim exists without completion; terminal/manual recovery required" % job["epoch"])
        else:
            for path, label in (
                (job["authority_path"], "work authority"),
                (job["authorization_path"], "authorization"),
                (job["run_root"], "run root"),
                (job["runner_status_path"], "runner status"),
                (job["runner_log_path"], "runner log"),
            ):
                require(not os.path.lexists(path), "unclaimed e%d %s already exists" % (job["epoch"], label))
            if head is None:
                head, gap = job, True
        require(not (gap and done), "queue skipped ahead")
    if len(completed) < len(CANDIDATE_EPOCHS):
        for path, label in (
            (campaign["summary_path"], "final summary"),
            (campaign["_reconciliation_path"], "producer reconciliation"),
            (campaign["_selection_root"], "selection root"),
            (campaign["_state_root"] / "producer_reconcile.claim.json", "producer reconcile claim"),
            (campaign["_state_root"] / "bridge_reconcile.claim.json", "bridge reconcile claim"),
        ):
            require(not os.path.lexists(path), "%s exists before all 22 completions" % label)
    return completed, head


FAILED_GLOBAL_CLAIM_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "candidate_epoch", "expected_shards", "work_authority", "run_root",
    "receipt_payload_sha256",
})


def _load_failed_campaign(
    path_text: Any, expected_sha256: str, expected_bytes: int,
) -> Dict[str, Any]:
    require(
        expected_sha256 == FAILED_CAMPAIGN_SHA256
        and expected_bytes == FAILED_CAMPAIGN_BYTES,
        "recovery may reference only the pinned 58407ed e1 failure campaign",
    )
    path, raw, _digest, _size = safe_regular_bytes(
        path_text, "failed recovery campaign", expected_sha256, expected_bytes
    )
    preview = strict_json(raw, "failed recovery campaign")
    control = preview.get("control_source")
    formal = preview.get("formal_python")
    require(
        isinstance(control, dict)
        and isinstance(control.get("supervisor"), dict)
        and isinstance(formal, dict)
        and isinstance(formal.get("argv0"), str),
        "failed recovery campaign lacks source/runtime bindings",
    )
    failed = load_campaign(
        str(path), expected_sha256, expected_bytes,
        running_script=control["supervisor"].get("path"),
        running_executable=formal["argv0"],
    )
    require(
        failed["_control_source"]["commit"] == FAILED_CONTROL_COMMIT
        and failed["_control_source"]["tree"] == FAILED_CONTROL_TREE,
        "recovery source is not the pinned 58407ed incident",
    )
    return failed


def _failed_global_claim(
    failed: Mapping[str, Any], old_job: Mapping[str, Any],
    old_authority: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    claim_path = (
        Path(failed["_adapter_config"]["train_root"])
        / "live_val_consumer_claims" / "epoch-0001.json"
    )
    artifact, value = _read_existing_json(claim_path, "failed global consumer claim")
    require(
        artifact["sha256"] == FAILED_CONSUMER_CLAIM_SHA256
        and set(value) == FAILED_GLOBAL_CLAIM_KEYS
        and value.get("format") == "semtalk_show_base_live_val_consumer_claim_v2"
        and value.get("status") == "claimed"
        and value.get("split") == "val"
        and value.get("test_visible") is False
        and value.get("selection_eligible") is False
        and value.get("candidate_epoch") == 1
        and value.get("expected_shards") == 8
        and value.get("run_root") == old_job["run_root"]
        and value.get("receipt_payload_sha256") == _bridge_payload_sha(value),
        "failed global consumer claim changed",
    )
    claimed_authority = value.get("work_authority")
    require(
        isinstance(claimed_authority, dict)
        and all(
            claimed_authority.get(key) == old_authority[key]
            for key in ARTIFACT_KEYS
        ),
        "failed global claim work authority changed",
    )
    return artifact, value


def _recovery_request_value(
    campaign: Mapping[str, Any], job: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
    failed: Mapping[str, Any], capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]],
    clock: Callable[[], float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    require(job["epoch"] == 1, "only e1 is eligible for failed-claim recovery")
    require(
        _git_stdout(
            Path(campaign["_control_source"]["root"]), "merge-base",
            "--is-ancestor", RECOVERY_CONTROL_MIN_COMMIT,
            campaign["_control_source"]["commit"],
        ) == "",
        "recovery control is not a descendant of the proven 8532272 fix",
    )
    old_job = dict(failed["_jobs"][0])
    old_job["_campaign"] = failed
    old_candidate = _candidate_receipt_artifact(old_job)
    require(
        old_candidate["sha256"] == FAILED_CANDIDATE_RECEIPT_SHA256
        and strict_json_equal(old_candidate, candidate_receipt),
        "recovery candidate is not the exact failed e1 candidate",
    )
    old_claim, old_claim_value = _load_job_claim_v2(
        failed, old_job, old_candidate
    )
    require(
        old_claim["sha256"] == FAILED_JOB_CLAIM_SHA256,
        "failed private job claim changed",
    )
    active_artifact, active_value = _read_existing_json(
        failed["_state_root"] / "active_invocation.claim.json",
        "failed active invocation claim",
    )
    require(
        active_artifact["sha256"] == FAILED_ACTIVE_CLAIM_SHA256
        and set(active_value) == ACTIVE_CLAIM_KEYS
        and active_value.get("format") == ACTIVE_CLAIM_FORMAT
        and active_value.get("status") == "active"
        and active_value.get("operation") == "run-next"
        and strict_json_equal(active_value.get("campaign"), failed["_artifact"]),
        "failed active invocation claim changed",
    )
    _self_hashed(active_value, "claim_payload_sha256", "failed active invocation claim")
    old_authority = _artifact_from_path(
        old_job["authority_path"], "failed work authority"
    )
    require(
        old_authority["sha256"] == FAILED_WORK_AUTHORITY_SHA256,
        "failed e1 work authority changed",
    )
    new_authority = {
        "path": job["authority_path"],
        "sha256": old_authority["sha256"],
        "bytes": old_authority["bytes"],
    }
    _work_authority_runtime_binding(
        old_authority, failed["_runtime_validation_source"], 1
    )
    _work_authority_candidate_binding(old_authority, old_candidate, 1)
    old_dynamic = {
        **old_job, "candidate_receipt": old_candidate,
        "work_authority": old_authority,
    }
    authorization_artifact, _authorization = _load_authorization(
        failed, old_dynamic, old_candidate, old_authority
    )
    require(
        authorization_artifact["sha256"] == FAILED_AUTHORIZATION_SHA256,
        "failed authorization changed",
    )
    old_dynamic["_campaign"] = failed
    status_artifact, status = _failed_runner_status(
        old_dynamic, FAILED_RUNNER_STATUS_SHA256
    )
    log_path, log_raw, log_sha, log_size = safe_regular_bytes(
        old_job["runner_log_path"], "failed runner log", FAILED_RUNNER_LOG_SHA256
    )
    require(
        log_raw == b"eight-shard live validation failed for e1\n",
        "failed runner log changed",
    )
    log_artifact = {"path": str(log_path), "sha256": log_sha, "bytes": log_size}
    global_claim, _global_claim_value = _failed_global_claim(
        failed, old_job, old_authority
    )
    inventory = _failed_run_inventory(old_job["run_root"])
    _failed_process_proof(status, clock())
    require(
        strict_json_equal(
            campaign["_guard_verifier"], failed["_guard_verifier"]
        ),
        "new campaign guard verifier differs from the failed incident verifier",
    )
    verifier_argv, verifier_stdout, guards = _guard_verify(
        failed, status, capture
    )
    # The runner status and immutable failed tree are read again after the
    # independent verifier, closing the admission-time TOCTOU window.
    status_artifact_2, status_2 = _failed_runner_status(
        old_dynamic, FAILED_RUNNER_STATUS_SHA256
    )
    require(
        strict_json_equal(status_artifact_2, status_artifact)
        and strict_json_equal(status_2, status)
        and strict_json_equal(_failed_run_inventory(old_job["run_root"]), inventory),
        "failed incident changed during recovery admission",
    )
    process_proof = _failed_process_proof(status_2, clock())
    request_value = _add_bridge_self_hash({
        "format": RECOVERY_REQUEST_FORMAT,
        "status": "ready_for_single_recovery",
        "split": "val", "test_visible": False,
        "selection_eligible": False, "candidate_epoch": 1,
        "failed_campaign": failed["_artifact"],
        "failed_job_claim": old_claim,
        "failed_active_claim": active_artifact,
        "failed_authorization": authorization_artifact,
        "failed_work_authority": old_authority,
        "failed_consumer_claim": global_claim,
        "failed_runner_status": status_artifact,
        "failed_runner_log": log_artifact,
        "failed_run_root": old_job["run_root"],
        "failed_run_inventory": inventory,
        "failed_process_proof": process_proof,
        "guard_proof": {
            "verifier": campaign["_guard_verifier"],
            "argv": verifier_argv, "stdout": verifier_stdout,
            "restored_guards": guards,
            "verified_unix": _finite_number(
                clock(), "failed guard verification time", 0.000001
            ),
        },
        "new_campaign": campaign["_artifact"],
        "new_control_source": campaign["control_source"],
        "new_work_authority": new_authority,
        "new_run_root": job["run_root"],
        "created_unix": _finite_number(
            clock(), "recovery request time", 0.000001
        ),
    }, "receipt_payload_sha256")
    return request_value, new_authority


def _campaign_claim_v2(campaign: Mapping[str, Any], clock: Callable[[], float]) -> Dict[str, Any]:
    path = Path(campaign["campaign_claim_path"])
    if not os.path.lexists(path):
        return write_new_json(path, _add_self_hash({
            "format": CAMPAIGN_CLAIM_FORMAT, "status": "claimed",
            "campaign": campaign["_artifact"], "candidate_epochs": list(CANDIDATE_EPOCHS),
            "created_unix": _finite_number(clock(), "campaign claim time", 0.000001),
        }, "claim_payload_sha256"), "campaign claim")
    artifact, value = _read_existing_json(path, "campaign claim")
    _self_hashed(value, "claim_payload_sha256", "campaign claim")
    require(value.get("format") == CAMPAIGN_CLAIM_FORMAT and value.get("campaign") == campaign["_artifact"], "campaign claim changed")
    return artifact


def _active_v2(campaign: Mapping[str, Any], operation: str, clock: Callable[[], float]) -> Dict[str, Any]:
    path = campaign["_state_root"] / "active_invocation.claim.json"
    require(not os.path.lexists(path), "active invocation claim exists; terminal/manual recovery required")
    return write_new_json(path, _add_self_hash({
        "format": ACTIVE_CLAIM_FORMAT, "status": "active", "operation": operation,
        "campaign": campaign["_artifact"], "created_unix": _finite_number(clock(), "active time", 0.000001),
    }, "claim_payload_sha256"), "active invocation claim")


def _release_active_v2(campaign: Mapping[str, Any], artifact: Mapping[str, Any]) -> None:
    path, _raw, _sha, _size = safe_regular_bytes(artifact["path"], "active invocation", artifact["sha256"], artifact["bytes"])
    require(path == campaign["_state_root"] / "active_invocation.claim.json", "active path changed")
    path.unlink()


def _reconcile_argv(campaign: Mapping[str, Any], completed: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], reconciliation_sha: str) -> List[str]:
    argv = [
        campaign["_formal_python"]["argv0"], "-I",
        campaign["_control_source"]["bridge"]["path"], "reconcile",
        "--reconciliation", str(campaign["_reconciliation_path"]),
        "--expected-reconciliation-sha256", reconciliation_sha,
    ]
    for epoch, (_artifact_value, value) in zip(CANDIDATE_EPOCHS, completed):
        argv.extend(["--live-measurement", str(epoch), value["measurement"]["path"], value["measurement"]["sha256"]])
    argv.extend(["--output-root", str(campaign["_selection_root"])])
    return argv


def _authority_reconcile_argv(
    campaign: Mapping[str, Any], final_manifest: Mapping[str, Any],
    final_status: Mapping[str, Any],
) -> List[str]:
    config = campaign["_adapter_config"]
    runtime = campaign["_runtime_validation_source"]
    semantics = runtime["validation_semantics_source"]
    return [
        campaign["_formal_python"]["argv0"], "-I",
        campaign["_control_source"]["authority_adapter"]["path"], "reconcile",
        "--train-root", config["train_root"],
        "--topology-mode", config["topology_mode"],
        "--producer-source-root", config["producer_source_root"],
        "--expected-producer-trainer-sha256", config["expected_producer_trainer_sha256"],
        "--expected-producer-contract-sha256", config["expected_producer_contract_sha256"],
        "--validation-evidence-source-root", semantics["source_root"],
        "--expected-validation-evidence-contract-sha256", semantics["contract"]["sha256"],
        "--expected-validation-evidence-selector-sha256", semantics["selector"]["sha256"],
        "--runtime-validation-source-root", runtime["source_root"],
        "--expected-runtime-validation-contract-sha256", runtime["contract"]["sha256"],
        "--expected-runtime-validation-selector-sha256", runtime["selector"]["sha256"],
        "--schedule", config["schedule"]["path"],
        "--expected-schedule-sha256", config["schedule"]["sha256"],
        "--expected-frozen-inputs-sha256", config["expected_frozen_inputs_sha256"],
        "--val-inputs", config["val_inputs"]["path"],
        "--expected-val-inputs-sha256", config["val_inputs"]["sha256"],
        "--pipeline", config["pipeline"]["path"],
        "--expected-pipeline-sha256", config["pipeline"]["sha256"],
        "--authority-dir", str(campaign["_authorities_dir"]),
        "--final-manifest", final_manifest["path"],
        "--expected-final-manifest-sha256", final_manifest["sha256"],
        "--final-status", final_status["path"],
        "--expected-final-status-sha256", final_status["sha256"],
        "--output", str(campaign["_reconciliation_path"]),
    ]


def _reconciliation_artifact(campaign: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reconciliation_path, reconciliation_raw, reconciliation_sha, reconciliation_size = safe_regular_bytes(campaign["_reconciliation_path"], "e400 reconciliation")
    reconciliation_value = _strict_json_document(reconciliation_raw, "e400 reconciliation")
    payload_sha = reconciliation_value.get("receipt_payload_sha256")
    require(isinstance(payload_sha, str) and HEX64.fullmatch(payload_sha) is not None and payload_sha == _bridge_payload_sha(reconciliation_value), "e400 reconciliation payload SHA changed")
    require(
        reconciliation_value.get("format") == RECONCILIATION_FORMAT
        and reconciliation_value.get("status") == "complete"
        and reconciliation_value.get("split") == "val"
        and reconciliation_value.get("test_visible") is False
        and reconciliation_value.get("selection_eligible") is True
        and reconciliation_value.get("candidate_epochs") == list(CANDIDATE_EPOCHS)
        and type(reconciliation_value.get("test_evaluations_observed")) is int
        and reconciliation_value["test_evaluations_observed"] == 0,
        "e400 reconciliation is not exact val-only authority",
    )
    return ({
        "path": str(reconciliation_path), "sha256": reconciliation_sha,
        "bytes": reconciliation_size, "receipt_payload_sha256": payload_sha,
    }, reconciliation_value)


def _load_producer_reconcile_claim(campaign: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = campaign["_state_root"] / "producer_reconcile.claim.json"
    artifact, value = _read_existing_json(path, "producer reconcile claim")
    require(set(value) == PRODUCER_RECONCILE_CLAIM_KEYS and value.get("format") == PRODUCER_RECONCILE_CLAIM_FORMAT and value.get("status") == "claimed", "producer reconcile claim schema/state changed")
    _self_hashed(value, "claim_payload_sha256", "producer reconcile claim")
    require(strict_json_equal(value.get("campaign"), campaign["_artifact"]), "producer reconcile claim campaign changed")
    final_manifest = _artifact(value.get("final_manifest"), "claimed final producer manifest")
    final_status = _artifact(value.get("final_status"), "claimed final producer status")
    require(final_manifest["path"] == str(campaign["_final_manifest_path"]) and final_status["path"] == str(campaign["_final_status_path"]), "producer reconcile final paths changed")
    argv = _authority_reconcile_argv(campaign, final_manifest, final_status)
    require(value.get("argv_sha256") == _sha256(b"\0".join(os.fsencode(x) for x in argv) + b"\0"), "producer reconcile argv changed")
    _finite_number(value.get("created_unix"), "producer reconcile claim time", 0.000001)
    return artifact, value


def _load_bridge_reconcile_claim(
    campaign: Mapping[str, Any], completed: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    reconciliation: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = campaign["_state_root"] / "bridge_reconcile.claim.json"
    artifact, value = _read_existing_json(path, "bridge reconcile claim")
    require(set(value) == BRIDGE_RECONCILE_CLAIM_KEYS and value.get("format") == BRIDGE_RECONCILE_CLAIM_FORMAT and value.get("status") == "claimed", "bridge reconcile claim schema/state changed")
    _self_hashed(value, "claim_payload_sha256", "bridge reconcile claim")
    require(strict_json_equal(value.get("campaign"), campaign["_artifact"]) and strict_json_equal(value.get("reconciliation"), reconciliation), "bridge reconcile claim authority changed")
    argv = _reconcile_argv(campaign, completed, reconciliation["sha256"])
    require(value.get("argv_sha256") == _sha256(b"\0".join(os.fsencode(x) for x in argv) + b"\0"), "bridge reconcile argv changed")
    _finite_number(value.get("created_unix"), "bridge reconcile claim time", 0.000001)
    return artifact, value


def _selection_v2(
    campaign: Mapping[str, Any],
    completed: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    reconciliation: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    selection_path = campaign["_selection_root"] / "selection.json"
    path, raw, digest, size = safe_regular_bytes(str(selection_path), "official bridge selection")
    selection = _strict_json_document(raw, "official bridge selection")
    expected_raw = (
        json.dumps(selection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    require(raw == expected_raw, "official bridge selection is not canonical newline JSON")
    require(set(selection) == SELECTION_KEYS, "official bridge selection schema changed")
    require(
        selection.get("format") == SELECTION_FORMAT
        and selection.get("status") == "selected"
        and selection.get("split") == "val"
        and selection.get("test_visible") is False
        and selection.get("selection_eligible") is True,
        "bridge selection is not authoritative val selection",
    )
    require(
        isinstance(selection.get("receipt_payload_sha256"), str)
        and HEX64.fullmatch(selection["receipt_payload_sha256"]) is not None
        and selection["receipt_payload_sha256"] == _bridge_payload_sha(selection),
        "official bridge selection payload SHA changed",
    )
    require(
        selection.get("candidate_epochs") == list(CANDIDATE_EPOCHS)
        and type(selection.get("test_evaluations_observed")) is int
        and selection["test_evaluations_observed"] == 0,
        "bridge selection coverage changed",
    )
    require(strict_json_equal(selection.get("reconciliation_receipt"), reconciliation), "bridge selection reconciliation root changed")
    live = [value["measurement"] for _artifact, value in completed]
    require(strict_json_equal(selection.get("live_measurements"), live), "bridge selection live measurements changed")
    require(isinstance(selection.get("reconciled_measurements"), list) and len(selection["reconciled_measurements"]) == len(CANDIDATE_EPOCHS), "bridge selection reconciled measurement coverage changed")
    selected = selection.get("selected")
    require(isinstance(selected, dict) and set(selected) == SELECTION_SELECTED_KEYS, "bridge selected row schema changed")
    require(type(selected.get("epoch")) is int and selected["epoch"] in CANDIDATE_EPOCHS, "bridge selected epoch changed")
    _finite_number(selected.get("fgd"), "selected FGD", 0.0)
    formal = selection.get("formal_selection")
    require(isinstance(formal, dict) and strict_json_equal(formal.get("selected"), selected), "bridge formal/outer selection differ")
    artifact = {
        "path": str(path), "sha256": digest, "bytes": size,
        "receipt_payload_sha256": selection["receipt_payload_sha256"],
    }
    return artifact, selection


def _validate_bridge_reconcile_stdout(
    raw: bytes, selection_artifact: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    require(0 < len(raw) <= 4096, "bridge reconcile stdout is empty/oversize")
    result = _strict_json_document(raw, "bridge reconcile stdout")
    require(
        set(result) == {
            "status", "split", "test_visible", "selection_eligible",
            "candidate_count", "selected_epoch", "selected_fgd", "selection",
        }
        and result.get("status") == "selected"
        and result.get("split") == "val"
        and result.get("test_visible") is False
        and result.get("selection_eligible") is True
        and type(result.get("candidate_count")) is int
        and result["candidate_count"] == len(CANDIDATE_EPOCHS)
        and type(result.get("selected_epoch")) is int
        and result["selected_epoch"] == selection["selected"]["epoch"]
        and type(result.get("selected_fgd")) in (int, float)
        and float(result["selected_fgd"]) == float(selection["selected"]["fgd"])
        and strict_json_equal(result.get("selection"), selection_artifact),
        "bridge reconcile stdout differs from the authoritative selection",
    )
    require(
        raw == (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
        "bridge reconcile stdout is not one canonical JSON document",
    )
    return result


def _finalize_v2(
    campaign: Mapping[str, Any],
    completed: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    runner: Callable[[Sequence[str]], int],
    capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]],
    clock: Callable[[], float],
) -> Dict[str, Any]:
    require(len(completed) == 22, "cannot reconcile before all 22 completions")
    final_manifest_present = os.path.lexists(campaign["_final_manifest_path"])
    final_status_present = os.path.lexists(campaign["_final_status_path"])
    if not (final_manifest_present and final_status_present):
        for path, label in (
            (campaign["_state_root"] / "active_invocation.claim.json", "active claim"),
            (campaign["_state_root"] / "producer_reconcile.claim.json", "producer reconcile claim"),
            (campaign["_state_root"] / "bridge_reconcile.claim.json", "bridge reconcile claim"),
            (campaign["_reconciliation_path"], "producer reconciliation"),
            (campaign["_selection_root"], "selection root"),
            (campaign["summary_path"], "final summary"),
        ):
            require(not os.path.lexists(path), "%s exists before e400 final manifest/status" % label)
        return {
            "status": "waiting_for_e400_final", "completion_count": 22,
            "final_manifest_present": final_manifest_present,
            "final_status_present": final_status_present,
            "producer_reconcile_claim_created": False,
            "bridge_reconcile_claim_created": False,
            "selection_eligible": False,
        }
    require(not os.path.lexists(campaign["_reconciliation_path"]) and not os.path.lexists(campaign["_selection_root"]), "reconciliation/selection output already exists without final summary")
    require(not os.path.lexists(campaign["_state_root"] / "producer_reconcile.claim.json") and not os.path.lexists(campaign["_state_root"] / "bridge_reconcile.claim.json"), "reconcile claim already exists without final summary")
    final_manifest = _artifact_from_path(campaign["_final_manifest_path"], "final producer manifest")
    final_status = _artifact_from_path(campaign["_final_status_path"], "final producer status")
    active = _active_v2(campaign, "reconcile", clock)
    _revalidate_control(campaign)
    _runtime_contract_check(campaign, capture)
    producer_argv = _authority_reconcile_argv(campaign, final_manifest, final_status)
    producer_claim = write_new_json(
        campaign["_state_root"] / "producer_reconcile.claim.json",
        _add_self_hash({
            "format": PRODUCER_RECONCILE_CLAIM_FORMAT, "status": "claimed",
            "campaign": campaign["_artifact"], "final_manifest": final_manifest,
            "final_status": final_status,
            "argv_sha256": _sha256(b"\0".join(os.fsencode(x) for x in producer_argv) + b"\0"),
            "created_unix": _finite_number(clock(), "producer reconcile claim time", 0.000001),
        }, "claim_payload_sha256"),
        "producer reconcile claim",
    )
    producer_rc, producer_stdout, producer_stderr = capture(producer_argv)
    require(type(producer_rc) is int and producer_rc == 0 and producer_stderr == b"", "authority adapter reconcile failed; terminal/manual recovery required")
    reconciliation_artifact, _reconciliation_value = _reconciliation_artifact(campaign)
    _adapter_stdout_artifact(producer_stdout, {key: reconciliation_artifact[key] for key in ARTIFACT_KEYS}, "authority adapter reconcile")
    require(strict_json_equal(_artifact_from_path(campaign["_final_manifest_path"], "final producer manifest"), final_manifest) and strict_json_equal(_artifact_from_path(campaign["_final_status_path"], "final producer status"), final_status), "final producer manifest/status changed during reconcile")
    bridge_argv = _reconcile_argv(campaign, completed, reconciliation_artifact["sha256"])
    bridge_claim = write_new_json(
        campaign["_state_root"] / "bridge_reconcile.claim.json",
        _add_self_hash({
            "format": BRIDGE_RECONCILE_CLAIM_FORMAT, "status": "claimed",
            "campaign": campaign["_artifact"],
            "reconciliation": reconciliation_artifact,
            "argv_sha256": _sha256(b"\0".join(os.fsencode(x) for x in bridge_argv) + b"\0"),
            "created_unix": _finite_number(clock(), "bridge reconcile claim time", 0.000001),
        }, "claim_payload_sha256"),
        "bridge reconcile claim",
    )
    bridge_rc, bridge_stdout, bridge_stderr = capture(bridge_argv)
    require(
        type(bridge_rc) is int and bridge_rc == 0 and bridge_stderr == b""
        and 0 < len(bridge_stdout) <= 4096,
        "bridge reconcile failed or emitted stderr/oversize output; terminal/manual recovery required",
    )
    reconciliation_artifact_2, _reconciliation_value_2 = _reconciliation_artifact(campaign)
    require(strict_json_equal(reconciliation_artifact_2, reconciliation_artifact), "e400 reconciliation changed during bridge reconcile")
    producer_claim_2, _producer_claim_value = _load_producer_reconcile_claim(campaign)
    bridge_claim_2, _bridge_claim_value = _load_bridge_reconcile_claim(campaign, completed, reconciliation_artifact)
    require(strict_json_equal(producer_claim_2, producer_claim) and strict_json_equal(bridge_claim_2, bridge_claim), "reconcile claims changed during selection")
    selection_artifact, selection = _selection_v2(campaign, completed, reconciliation_artifact)
    _validate_bridge_reconcile_stdout(bridge_stdout, selection_artifact, selection)
    selected = selection.get("selected")
    summary = write_new_json(campaign["summary_path"], _add_self_hash({
        "format": SUMMARY_FORMAT, "status": "complete", "split": "val",
        "test_visible": False, "selection_eligible": True,
        "test_measurements_authorized": 0, "campaign": campaign["_artifact"],
        "candidate_epochs": list(CANDIDATE_EPOCHS), "completion_count": 22,
        "producer_reconcile_claim": producer_claim,
        "bridge_reconcile_claim": bridge_claim,
        "official_selection": selection_artifact,
        "selected": selected, "completed_unix": _finite_number(clock(), "summary time", 0.000001),
    }, "receipt_payload_sha256"), "final summary")
    _release_active_v2(campaign, active)
    return {"status": "complete", "selection_eligible": True, "summary": summary, "selected": selected}


def finalize_campaign(
    campaign: Mapping[str, Any], runner: Callable[[Sequence[str]], int] = _run_process,
    capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]] = _run_capture,
    clock: Callable[[], float] = time.time,
) -> Dict[str, Any]:
    # The read-only admission above consumes no durable slot.  From this first
    # private write onward the new campaign attempt is terminal on every
    # failure, but the one global recovery slot remains unconsumed until the
    # bridge's reserve-recovery operation publishes its global claim.
    _campaign_claim_v2(campaign, clock)
    completed, head = _scan_v2(campaign)
    require(head is None and len(completed) == 22, "cannot finalize before all 22 completions")
    if os.path.lexists(campaign["summary_path"]):
        artifact, value = _read_existing_json(Path(campaign["summary_path"]), "final summary")
        require(set(value) == SUMMARY_KEYS and value.get("format") == SUMMARY_FORMAT and value.get("status") == "complete" and value.get("split") == "val" and value.get("test_visible") is False and value.get("selection_eligible") is True and type(value.get("test_measurements_authorized")) is int and value["test_measurements_authorized"] == 0, "final summary changed")
        _self_hashed(value, "receipt_payload_sha256", "final summary")
        require(strict_json_equal(value.get("campaign"), campaign["_artifact"]) and value.get("candidate_epochs") == list(CANDIDATE_EPOCHS) and type(value.get("completion_count")) is int and value["completion_count"] == 22, "final summary campaign/coverage changed")
        require(not os.path.lexists(campaign["_state_root"] / "active_invocation.claim.json"), "active claim exists after final summary")
        reconciliation_artifact, _reconciliation_value = _reconciliation_artifact(campaign)
        producer_claim, _producer_claim_value = _load_producer_reconcile_claim(campaign)
        bridge_claim, _bridge_claim_value = _load_bridge_reconcile_claim(campaign, completed, reconciliation_artifact)
        require(strict_json_equal(value.get("producer_reconcile_claim"), producer_claim) and strict_json_equal(value.get("bridge_reconcile_claim"), bridge_claim), "final summary reconcile claim artifacts changed")
        selection_artifact, selection = _selection_v2(campaign, completed, reconciliation_artifact)
        require(strict_json_equal(value.get("official_selection"), selection_artifact) and strict_json_equal(value.get("selected"), selection["selected"]), "final summary selection changed")
        _finite_number(value.get("completed_unix"), "summary time", 0.000001)
        return {"status": "complete", "selection_eligible": True, "summary": artifact, "selected": value["selected"]}
    return _finalize_v2(campaign, completed, runner, capture, clock)


def run_next(
    campaign: Mapping[str, Any], runner: Callable[[Sequence[str]], int] = _run_process,
    capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]] = _run_capture,
    clock: Callable[[], float] = time.time,
) -> Dict[str, Any]:
    _campaign_claim_v2(campaign, clock)
    completed, head = _scan_v2(campaign)
    if head is None:
        return finalize_campaign(campaign, runner=runner, capture=capture, clock=clock)
    active_path = campaign["_state_root"] / "active_invocation.claim.json"
    if not os.path.lexists(head["candidate_receipt_path"]):
        require(not os.path.lexists(active_path), "active invocation claim exists while awaiting candidate receipt; terminal/manual recovery required")
        return {
            "status": "waiting_for_candidate_receipt", "candidate_epoch": head["epoch"],
            "completion_count": len(completed), "selection_eligible": False,
            "job_claim_created": False, "active_claim_created": False,
        }
    if head["epoch"] == 1:
        original_claim_path = (
            Path(campaign["_adapter_config"]["train_root"])
            / "live_val_consumer_claims" / "epoch-0001.json"
        )
        require(
            not os.path.lexists(original_claim_path),
            "the global e1 consumer claim is already occupied; use the pinned recover-e1 command",
        )
    candidate_receipt = _candidate_receipt_artifact(head)
    active = _active_v2(campaign, "run-next", clock)
    require(not os.path.lexists(head["authority_path"]) and not os.path.lexists(head["authorization_path"]) and not os.path.lexists(head["run_root"]) and not os.path.lexists(head["runner_status_path"]) and not os.path.lexists(head["runner_log_path"]), "queue head outputs already exist")
    _revalidate_control(campaign)
    _runtime_contract_check(campaign, capture)
    authorize_argv = _authorize_argv(campaign, head)
    claim = write_new_json(
        _job_claim_path_v2(campaign, head["epoch"]),
        _job_claim_v2(campaign, head, candidate_receipt, authorize_argv, clock()),
        "job claim",
    )
    authorize_rc, authorize_stdout, authorize_stderr = capture(authorize_argv)
    require(type(authorize_rc) is int and authorize_rc == 0 and authorize_stderr == b"", "authority adapter authorize failed; terminal/manual recovery required")
    authority = _artifact_from_path(head["authority_path"], "queue-head work authority")
    adapter_stdout_sha = _adapter_stdout_artifact(authorize_stdout, authority, "authority adapter authorize")
    _work_authority_runtime_binding(authority, campaign["_runtime_validation_source"], head["epoch"])
    _work_authority_candidate_binding(authority, candidate_receipt, head["epoch"])
    require(strict_json_equal(_candidate_receipt_artifact(head), candidate_receipt), "candidate-ready receipt changed during authorization")
    dynamic = dict(head)
    dynamic.update({"candidate_receipt": candidate_receipt, "work_authority": authority})
    runner_argv = _runner_argv_v2(dynamic, campaign)
    _validate_runner_argv_v2(runner_argv, dynamic, campaign)
    authorization = write_new_json(
        dynamic["authorization_path"],
        _authorization_body(campaign, dynamic, authorize_argv, adapter_stdout_sha, runner_argv, clock()),
        "authorization receipt",
    )
    dynamic["authorization"] = authorization
    rc = runner(runner_argv)
    require(type(rc) is int and rc == 0, "guarded runner returned nonzero; terminal/manual recovery required")
    dynamic["_campaign"] = campaign
    status_artifact, status = _runner_status(dynamic)
    log_artifact, measurement, measurement_value = _verify_runner_log(dynamic)
    # Replay the pinned bridge under the original venv launcher.  This is
    # independent of the launcher process that created the measurement.
    _revalidate_control(campaign)
    _runtime_contract_check(campaign, capture)
    bridge_replay_argv = _bridge_replay(campaign, dynamic, measurement, capture)
    verifier_argv, verifier_stdout, guards = _guard_verify(campaign, status, capture)
    status_artifact_2, status_2 = _runner_status(dynamic)
    require(strict_json_equal(status_artifact, status_artifact_2) and strict_json_equal(status, status_2), "runner status changed during guard verification")
    _measurement(dynamic, measurement["sha256"], measurement["receipt_payload_sha256"])
    require(strict_json_equal(_candidate_receipt_artifact(dynamic), candidate_receipt), "candidate-ready receipt changed during validation")
    require(strict_json_equal(_artifact_from_path(dynamic["authority_path"], "queue-head work authority"), authority), "work authority changed during validation")
    completion_value = _completion_body(campaign, dynamic, status_artifact, status, log_artifact, measurement, measurement_value, bridge_replay_argv, verifier_argv, verifier_stdout, guards, clock())
    completion = write_new_json(dynamic["completion_path"], completion_value, "completion")
    _release_active_v2(campaign, active)
    return {"status": "job_complete", "candidate_epoch": dynamic["epoch"], "selection_eligible": False, "job_claim": claim, "authorization": authorization, "completion": completion, "validation_diffsheg_fgd": completion_value["validation_diffsheg_fgd"], "remaining": 21 - len(completed)}


def recover_e1(
    campaign: Mapping[str, Any], failed_campaign_path: Any,
    expected_failed_campaign_sha256: str,
    expected_failed_campaign_bytes: int,
    runner: Callable[[Sequence[str]], int] = _run_process,
    capture: Callable[[Sequence[str]], Tuple[int, bytes, bytes]] = _run_capture,
    clock: Callable[[], float] = time.time,
) -> Dict[str, Any]:
    """Consume the one globally reserved recovery slot for the pinned e1 incident."""

    failed = _load_failed_campaign(
        failed_campaign_path, expected_failed_campaign_sha256,
        expected_failed_campaign_bytes,
    )
    completed, head = _scan_v2(campaign)
    require(
        not completed and head is not None and head["epoch"] == 1,
        "recover-e1 requires a fresh campaign whose exact queue head is e1",
    )
    require(
        os.path.lexists(head["candidate_receipt_path"]),
        "recover-e1 requires the exact e1 candidate-ready receipt",
    )
    original_claim_path = (
        Path(campaign["_adapter_config"]["train_root"])
        / "live_val_consumer_claims" / "epoch-0001.json"
    )
    recovery_claim_path = (
        Path(campaign["_adapter_config"]["train_root"])
        / "live_val_consumer_recovery_claims" / "epoch-0001.json"
    )
    request_path = campaign["_state_root"] / "recovery-request.epoch-0001.json"
    recovery_authority_path = (
        campaign["_state_root"] / "recovery-authority.epoch-0001.json"
    )
    require(
        os.path.lexists(original_claim_path),
        "the pinned failed e1 global consumer claim is absent",
    )
    for path, label in (
        (Path(campaign["campaign_claim_path"]), "new campaign claim"),
        (campaign["_state_root"] / "active_invocation.claim.json", "new active claim"),
        (recovery_claim_path, "global e1 recovery claim"),
        (request_path, "e1 recovery request"),
        (recovery_authority_path, "e1 recovery authority"),
    ):
        require(
            not os.path.lexists(path),
            "%s already exists; the one-time recovery is consumed or terminal" % label,
        )
    candidate_receipt = _candidate_receipt_artifact(head)
    require(
        not os.path.lexists(head["authority_path"])
        and not os.path.lexists(head["authorization_path"])
        and not os.path.lexists(head["run_root"])
        and not os.path.lexists(head["runner_status_path"])
        and not os.path.lexists(head["runner_log_path"]),
        "recovery queue-head outputs already exist",
    )
    # Everything through the request preview is a read-only admission phase.
    # In particular, the pinned guard verifier runs before any campaign,
    # active, private-job, work-authority, request, or recovery-slot write.
    _revalidate_control(campaign)
    _runtime_contract_check(campaign, capture)
    request_value, authority_preview = _recovery_request_value(
        campaign, head, candidate_receipt, failed, capture, clock,
    )

    _campaign_claim_v2(campaign, clock)
    active = _active_v2(campaign, "recover-e1", clock)
    authorize_argv = _authorize_argv(campaign, head)
    claim = write_new_json(
        _job_claim_path_v2(campaign, 1),
        _job_claim_v2(
            campaign, head, candidate_receipt, authorize_argv, clock()
        ),
        "recovery job claim",
    )
    authorize_rc, authorize_stdout, authorize_stderr = capture(authorize_argv)
    require(
        type(authorize_rc) is int and authorize_rc == 0
        and authorize_stderr == b"",
        "recovery authority adapter authorize failed; recovery is terminal",
    )
    authority = _artifact_from_path(
        head["authority_path"], "recovery queue-head work authority"
    )
    require(
        strict_json_equal(authority, authority_preview),
        "new work authority is not byte-identical to the read-only recovery preview",
    )
    adapter_stdout_sha = _adapter_stdout_artifact(
        authorize_stdout, authority, "recovery authority adapter authorize"
    )
    _work_authority_runtime_binding(
        authority, campaign["_runtime_validation_source"], 1
    )
    _work_authority_candidate_binding(authority, candidate_receipt, 1)
    require(
        strict_json_equal(
            _candidate_receipt_artifact(head), candidate_receipt
        ),
        "candidate-ready receipt changed during recovery authorization",
    )
    dynamic = dict(head)
    dynamic.update({
        "candidate_receipt": candidate_receipt,
        "work_authority": authority,
    })
    require(
        strict_json_equal(
            request_value.get("new_work_authority"), authority
        ),
        "recovery request work-authority preview changed",
    )
    request = write_new_json(
        request_path, request_value,
        "failed consumer recovery request",
    )
    inspect_argv = _recovery_bridge_argv(
        campaign, "inspect-recovery", request
    )
    inspect_rc, inspect_stdout, inspect_stderr = capture(inspect_argv)
    require(
        type(inspect_rc) is int and inspect_rc == 0
        and inspect_stderr == b"",
        "inspect-recovery failed; no recovery reservation was authorized",
    )
    preview_authority, preview_claim_path = _inspect_recovery_stdout(
        campaign, inspect_stdout
    )
    require(
        not os.path.lexists(recovery_claim_path)
        and not os.path.lexists(recovery_authority_path),
        "inspect-recovery changed durable recovery state",
    )
    # inspect-recovery is still read-only with respect to the global slot.  A
    # failure through inspect leaves this private campaign terminal; reserve is
    # the exact boundary after which the global recovery can never be reused.
    reserve_argv = _recovery_bridge_argv(
        campaign, "reserve-recovery", request
    )
    reserve_rc, reserve_stdout, reserve_stderr = capture(reserve_argv)
    require(
        type(reserve_rc) is int and reserve_rc == 0
        and reserve_stderr == b"",
        "reserve-recovery failed; the recovery slot is terminal and must not be retried",
    )
    recovery_authority, recovery_claim = _reserve_recovery_stdout(
        campaign, reserve_stdout, preview_authority, preview_claim_path
    )
    recovery = {
        "request": request,
        "authority": recovery_authority,
        "claim": recovery_claim,
    }
    dynamic["consumer_recovery"] = recovery
    runner_argv = _runner_argv_v2(dynamic, campaign)
    _validate_runner_argv_v2(runner_argv, dynamic, campaign)
    authorization = write_new_json(
        dynamic["authorization_path"],
        _authorization_body(
            campaign, dynamic, authorize_argv, adapter_stdout_sha,
            runner_argv, clock(),
        ),
        "recovery authorization receipt",
    )
    dynamic["authorization"] = authorization
    # Reservation is deliberately durable before this only GPU-capable call.
    # Any failure below is terminal: no code path removes or reuses the slot.
    rc = runner(runner_argv)
    require(
        type(rc) is int and rc == 0,
        "recovery guarded runner returned nonzero; recovery is terminal",
    )
    dynamic["_campaign"] = campaign
    status_artifact, status = _runner_status(dynamic)
    log_artifact, measurement, measurement_value = _verify_runner_log(dynamic)
    _revalidate_control(campaign)
    _runtime_contract_check(campaign, capture)
    bridge_replay_argv = _bridge_replay(
        campaign, dynamic, measurement, capture
    )
    verifier_argv, verifier_stdout, guards = _guard_verify(
        campaign, status, capture
    )
    status_artifact_2, status_2 = _runner_status(dynamic)
    require(
        strict_json_equal(status_artifact, status_artifact_2)
        and strict_json_equal(status, status_2),
        "recovery runner status changed during guard verification",
    )
    _measurement(
        dynamic, measurement["sha256"],
        measurement["receipt_payload_sha256"],
    )
    require(
        strict_json_equal(
            _candidate_receipt_artifact(dynamic), candidate_receipt
        ),
        "candidate-ready receipt changed during recovered validation",
    )
    require(
        strict_json_equal(
            _artifact_from_path(
                dynamic["authority_path"],
                "recovery queue-head work authority",
            ),
            authority,
        )
        and strict_json_equal(
            _artifact(request, "consumer recovery request"), request
        )
        and strict_json_equal(
            _artifact(recovery_authority, "consumer recovery authority"),
            recovery_authority,
        )
        and strict_json_equal(
            _artifact(recovery_claim, "consumer recovery claim"),
            recovery_claim,
        ),
        "recovery authority chain changed during validation",
    )
    completion_value = _completion_body(
        campaign, dynamic, status_artifact, status, log_artifact,
        measurement, measurement_value, bridge_replay_argv,
        verifier_argv, verifier_stdout, guards, clock(),
    )
    completion = write_new_json(
        dynamic["completion_path"], completion_value,
        "recovery completion",
    )
    _release_active_v2(campaign, active)
    return {
        "status": "job_complete", "candidate_epoch": 1,
        "selection_eligible": False, "job_claim": claim,
        "recovery_request": request,
        "recovery_authority": recovery_authority,
        "recovery_claim": recovery_claim,
        "authorization": authorization, "completion": completion,
        "validation_diffsheg_fgd": completion_value["validation_diffsheg_fgd"],
        "remaining": 21,
    }


def _artifact_from_path(path_text: Any, label: str, *, executable: bool = False) -> Dict[str, Any]:
    path, _raw, digest, size = safe_regular_bytes(path_text, label)
    if executable:
        require(path.stat().st_mode & 0o111 != 0, "%s is not executable" % label)
    return {"path": str(path), "sha256": digest, "bytes": size}


def _capture_formal_python_binding(argv0_text: Any) -> Dict[str, Any]:
    argv0 = _lexical_absolute(argv0_text, "formal Python argv0")
    require(os.fsencode(sys.executable) == os.fsencode(str(argv0)), "build-campaign must run through the supplied formal Python launcher")
    require(argv0.parent.name == "bin" and argv0.parent.resolve(strict=True) == argv0.parent, "formal Python parent changed")
    venv_root = argv0.parent.parent
    canonical_directory(str(venv_root), "formal Python venv root")
    chain: List[Dict[str, str]] = []
    current = argv0
    seen: set = set()
    for _ordinal in range(16):
        require(str(current) not in seen, "formal Python symlink loop")
        seen.add(str(current))
        metadata = current.lstat()
        if not stat.S_ISLNK(metadata.st_mode):
            break
        target = os.readlink(current)
        chain.append({"path": str(current), "target": target})
        current = _normalize_link_target(current, target)
    else:
        raise SupervisorError("formal Python symlink chain is too deep")
    require(bool(chain) and argv0.resolve(strict=True) == current and not current.is_symlink(), "formal Python is not a frozen venv symlink chain")
    value = {
        "format": "semtalk.formal_venv_python_binding.v1", "argv0": str(argv0),
        "venv_root": str(venv_root), "symlink_chain": chain,
        "resolved_target": _artifact_from_path(str(current), "formal Python final ELF", executable=True),
        "pyvenv_cfg": _artifact_from_path(str(venv_root / "pyvenv.cfg"), "formal Python pyvenv.cfg"),
    }
    return _formal_python(value, running_executable=str(argv0))


def _git_stdout(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)
    require(completed.returncode == 0 and completed.stderr == "", "control source Git query failed")
    return completed.stdout.rstrip("\n")


def _freeze_control_source(root_text: Any, commit: str, tree: str) -> Dict[str, Any]:
    root = canonical_directory(root_text, "control source root")
    require(re.fullmatch(r"[0-9a-f]{40}", commit or "") is not None and re.fullmatch(r"[0-9a-f]{40}", tree or "") is not None, "control source OID changed")
    require(_git_stdout(root, "remote") == "origin", "control source remotes changed")
    require(_git_stdout(root, "remote", "get-url", "origin") == "git@github.com:Xiangyue-Zhang/SemTalk.git" and _git_stdout(root, "remote", "get-url", "--push", "origin") == "git@github.com:Xiangyue-Zhang/SemTalk.git", "control source origin changed")
    require(_git_stdout(root, "rev-parse", "HEAD") == commit and _git_stdout(root, "rev-parse", "HEAD^{tree}") == tree, "control source commit/tree changed")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "--short", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    require(symbolic.returncode == 1 and symbolic.stdout == "" and symbolic.stderr == "", "control source is not detached")
    require(_git_stdout(root, "status", "--porcelain=v1", "--untracked-files=all") == "" and _git_stdout(root, "for-each-ref", "--format=%(refname)", "refs/heads") == "", "control source is not clean zero-branch")
    supervisor = Path(__file__).resolve(strict=True)
    launcher = root / "scripts/show_base/run_base_live_val_8shard.sh"
    bridge = root / "scripts/show_base/base_live_val_consumer_bridge.py"
    authority_adapter = root / "scripts/show_base/base_v14_live_validation_authority.py"
    runtime_contract = root / "scripts/show_base/formal_python_runtime_contract.sh"
    require(supervisor.is_relative_to(root), "supervisor is not the tracked integrated control source")
    for path in (supervisor, launcher, bridge, authority_adapter, runtime_contract):
        relative = str(path.relative_to(root))
        require(_git_stdout(root, "ls-files", "--error-unmatch", relative) == relative, "control source file is not tracked: %s" % relative)
    return {
        "root": str(root), "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "commit": commit, "tree": tree,
        "supervisor": _artifact_from_path(str(supervisor), "supervisor control source"),
        "launcher": _artifact_from_path(str(launcher), "live launcher", executable=True),
        "bridge": _artifact_from_path(str(bridge), "live bridge"),
        "authority_adapter": _artifact_from_path(str(authority_adapter), "authority adapter"),
    }


def _freeze_validation_semantics_source(
    root_text: Any, contract_sha256: str, selector_sha256: str,
) -> Dict[str, Any]:
    require(contract_sha256 == VALIDATION_SEMANTICS_CONTRACT_SHA256 and selector_sha256 == VALIDATION_SEMANTICS_SELECTOR_SHA256, "validation evidence expected SHA changed")
    root = canonical_directory(root_text, "validation semantics source root")
    require(str(root) == VALIDATION_SEMANTICS_ROOT, "validation semantics source root changed")
    value = {
        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "source_root": str(root), "commit": VALIDATION_SEMANTICS_COMMIT,
        "tree": VALIDATION_SEMANTICS_TREE,
        "contract": _artifact_from_path(str(root / "scripts/show_base/base_long_val_contract.py"), "validation semantics contract"),
        "selector": _artifact_from_path(str(root / "scripts/show_base/select_base_official_adapt.py"), "validation semantics selector"),
    }
    require(value["contract"]["sha256"] == contract_sha256 and value["selector"]["sha256"] == selector_sha256, "validation evidence source SHA differs from explicit builder pin")
    return _validation_semantics_source(value)


def _freeze_runtime_validation_source(
    root_text: Any, commit: str, tree: str,
    contract_sha256: str, evaluator_sha256: str, selector_sha256: str,
    validation_semantics_root: Any, validation_semantics_contract_sha256: str,
    validation_semantics_selector_sha256: str,
) -> Dict[str, Any]:
    require(contract_sha256 == RUNTIME_VALIDATION_CONTRACT_SHA256 and evaluator_sha256 == RUNTIME_VALIDATION_EVALUATOR_SHA256 and selector_sha256 == RUNTIME_VALIDATION_SELECTOR_SHA256, "runtime validator expected SHA changed")
    root = canonical_directory(root_text, "runtime validation source root")
    require(str(root) == RUNTIME_VALIDATION_ROOT and commit == RUNTIME_VALIDATION_COMMIT and tree == RUNTIME_VALIDATION_TREE, "runtime validation source identity changed")
    value = {
        "format": RUNTIME_VALIDATION_SOURCE_FORMAT,
        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "source_root": str(root), "commit": commit, "tree": tree,
        "validation_semantics_source": _freeze_validation_semantics_source(
            validation_semantics_root, validation_semantics_contract_sha256,
            validation_semantics_selector_sha256,
        ),
        "semantics_ancestry_verified": True,
        "autoencoder_provenance": {"state_container": "state_dict", "load_mode": "full_half_embedding_net"},
        "contract": _artifact_from_path(str(root / "scripts/show_base/base_long_val_contract.py"), "runtime validation contract"),
        "evaluator": _artifact_from_path(str(root / "scripts/show_base/evaluate_diffsheg_val_fgd.py"), "runtime validation evaluator"),
        "selector": _artifact_from_path(str(root / "scripts/show_base/select_base_official_adapt.py"), "runtime validation selector"),
    }
    require(value["contract"]["sha256"] == contract_sha256 and value["evaluator"]["sha256"] == evaluator_sha256 and value["selector"]["sha256"] == selector_sha256, "runtime validator source SHA differs from explicit builder pin")
    return _runtime_validation_source(value)


def build_campaign(args: argparse.Namespace) -> Dict[str, Any]:
    require(type(args.seed) is int and args.seed >= 0 and type(args.diffsheg_batch_size) is int and args.diffsheg_batch_size > 0, "seed/batch size changed")
    for name in (
        "expected_producer_trainer_sha256", "expected_producer_contract_sha256",
        "expected_schedule_sha256", "expected_frozen_inputs_sha256",
        "expected_val_inputs_sha256", "expected_pipeline_sha256",
        "expected_guarded_runner_sha256", "expected_guard_verifier_sha256",
    ):
        require(isinstance(getattr(args, name), str) and HEX64.fullmatch(getattr(args, name)) is not None, "%s changed" % name)
    control = _freeze_control_source(args.control_root, args.control_commit, args.control_tree)
    runtime_validation = _freeze_runtime_validation_source(
        args.runtime_validation_root, args.runtime_validation_commit,
        args.runtime_validation_tree, args.runtime_contract_sha256,
        args.runtime_evaluator_sha256, args.runtime_selector_sha256,
        args.validation_evidence_root, args.validation_evidence_contract_sha256,
        args.validation_evidence_selector_sha256,
    )
    adapter_config = _authority_adapter_config({
        "train_root": args.train_root, "topology_mode": args.topology_mode,
        "producer_source_root": args.producer_source_root,
        "expected_producer_trainer_sha256": args.expected_producer_trainer_sha256,
        "expected_producer_contract_sha256": args.expected_producer_contract_sha256,
        "schedule": _artifact_from_path(args.schedule, "authority schedule"),
        "expected_frozen_inputs_sha256": args.expected_frozen_inputs_sha256,
        "val_inputs": _artifact_from_path(args.val_inputs, "authority validation inputs"),
        "pipeline": _artifact_from_path(args.pipeline, "authority validation pipeline"),
    }, runtime_validation)
    require(adapter_config["schedule"]["sha256"] == args.expected_schedule_sha256, "schedule SHA differs from explicit builder pin")
    require(adapter_config["val_inputs"]["sha256"] == args.expected_val_inputs_sha256, "validation inputs SHA differs from explicit builder pin")
    require(adapter_config["pipeline"]["sha256"] == args.expected_pipeline_sha256, "validation pipeline SHA differs from explicit builder pin")
    formal = _capture_formal_python_binding(args.formal_python)
    runtime_contract = _artifact_from_path(str(Path(control["root"]) / "scripts/show_base/formal_python_runtime_contract.sh"), "formal runtime contract")
    runner = _artifact_from_path(args.guarded_runner, "guarded runner", executable=True)
    verifier = _artifact_from_path(args.guard_verifier, "guard verifier")
    require(runner["path"] == "/tmp/globaldiff_guarded_runner.py" and verifier["path"] == "/tmp/verify_globaldiff_guards.py", "formal runner/verifier path changed")
    require(
        runner["sha256"] == args.expected_guarded_runner_sha256
        and runner["bytes"] == _exact_int(
            args.expected_guarded_runner_bytes, "expected guarded-runner bytes", 1,
        )
        and verifier["sha256"] == args.expected_guard_verifier_sha256
        and verifier["bytes"] == _exact_int(
            args.expected_guard_verifier_bytes, "expected guard-verifier bytes", 1,
        ),
        "runner/verifier differs from the explicit builder pin",
    )
    require(
        len({
            control["root"], adapter_config["producer_source_root"],
            runtime_validation["source_root"],
            runtime_validation["validation_semantics_source"]["source_root"],
        }) == 4,
        "control, producer, validation evidence, and runtime roots must be distinct",
    )
    state = _lexical_absolute(args.state_root, "state root")
    require(not os.path.lexists(state) and state.parent.resolve(strict=True) == state.parent, "state root must be create-new")
    output = _lexical_absolute(args.output, "campaign output")
    require(output == state / "campaign.json", "campaign output must be STATE_ROOT/campaign.json")
    run_base = canonical_directory(args.run_root_base, "run-root base")
    selection_root = canonical_output_path(args.reconcile_selection_root, "reconcile selection root")
    require(not os.path.lexists(selection_root), "future selection output already exists")
    paspa_root = canonical_directory(args.paspa_root, "PASPA root")
    diffsheg_root = canonical_directory(args.diffsheg_root, "DiffSHEG root")
    reconciliation = state / "reconciliation.json"
    jobs: List[Dict[str, Any]] = []
    for epoch in CANDIDATE_EPOCHS:
        run_root = run_base / ("e%d" % epoch)
        require(not os.path.lexists(run_root), "candidate run root already exists")
        status = state / "runner_status" / ("epoch-%04d.json" % epoch)
        log = state / "runner_logs" / ("epoch-%04d.log" % epoch)
        completion = state / "completions" / ("epoch-%04d.json" % epoch)
        jobs.append({
            "epoch": epoch,
            "candidate_receipt_path": str(Path(adapter_config["train_root"]) / "candidate_receipts" / ("epoch-%04d.json" % epoch)),
            "authority_path": str(state / "authorities" / ("epoch-%04d.json" % epoch)),
            "authorization_path": str(state / "authorizations" / ("epoch-%04d.json" % epoch)),
            "run_root": str(run_root),
            "measurement_path": str(run_root / "candidates" / ("e%d" % epoch) / "live-measurement.json"),
            "completion_path": str(completion), "runner_status_path": str(status),
            "runner_log_path": str(log),
        })
    os.mkdir(state, 0o700)
    for name in ("job_claims", "completions", "runner_status", "runner_logs", "authorities", "authorizations"):
        os.mkdir(state / name, 0o700)
    body = _add_self_hash({
        "format": CAMPAIGN_FORMAT, "status": "frozen_before_execution", "split": "val",
        "test_visible": False, "test_measurements_authorized": 0,
        "candidate_epochs": list(CANDIDATE_EPOCHS), "state_root": str(state),
        "campaign_claim_path": str(state / "campaign.claim.json"),
        "summary_path": str(state / "final_summary.json"), "control_source": control,
        "runtime_validation_source": runtime_validation,
        "authority_adapter_config": adapter_config,
        "final_manifest_path": str(Path(adapter_config["train_root"]) / "candidate_manifest.json"),
        "final_status_path": str(Path(adapter_config["train_root"]) / "status.json"),
        "paspa_root": str(paspa_root), "diffsheg_root": str(diffsheg_root),
        "seed": args.seed, "diffsheg_batch_size": args.diffsheg_batch_size,
        "formal_python": formal, "formal_python_runtime_contract": runtime_contract,
        "guarded_runner": runner, "guard_verifier": verifier,
        "reconciliation_path": str(reconciliation), "reconcile_selection_root": str(selection_root),
        "jobs": jobs,
    }, "campaign_payload_sha256")
    artifact = write_new_json(str(output), body, "campaign")
    # Full replay is the last builder step; no campaign can be returned unless
    # the same parser used by run-next accepts it.
    load_campaign(artifact["path"], artifact["sha256"], artifact["bytes"])
    return {"status": "frozen", "campaign": artifact, "candidate_count": 22}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run-next", "recover-e1", "finalize"):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument("--campaign", required=True)
        sub.add_argument("--expected-campaign-sha256", required=True)
        sub.add_argument("--expected-campaign-bytes", type=int, required=True)
        if name == "recover-e1":
            sub.add_argument("--failed-campaign", required=True)
            sub.add_argument(
                "--expected-failed-campaign-sha256", required=True
            )
            sub.add_argument(
                "--expected-failed-campaign-bytes", type=int, required=True
            )
    build = commands.add_parser("build-campaign", allow_abbrev=False)
    build.add_argument("--output", required=True)
    build.add_argument("--state-root", required=True)
    build.add_argument("--control-root", required=True)
    build.add_argument("--control-commit", required=True)
    build.add_argument("--control-tree", required=True)
    build.add_argument("--runtime-validation-root", required=True)
    build.add_argument("--runtime-validation-commit", required=True)
    build.add_argument("--runtime-validation-tree", required=True)
    build.add_argument("--runtime-contract-sha256", required=True)
    build.add_argument("--runtime-evaluator-sha256", required=True)
    build.add_argument("--runtime-selector-sha256", required=True)
    build.add_argument("--validation-evidence-root", required=True)
    build.add_argument("--validation-evidence-contract-sha256", required=True)
    build.add_argument("--validation-evidence-selector-sha256", required=True)
    build.add_argument("--train-root", required=True)
    build.add_argument("--topology-mode", required=True, choices=(
        "validation_gated_w8_l128_g1024_empirical_acceleration",
        "validation_gated_w8_l256_g2048_empirical_acceleration",
    ))
    build.add_argument("--producer-source-root", required=True)
    build.add_argument("--expected-producer-trainer-sha256", required=True)
    build.add_argument("--expected-producer-contract-sha256", required=True)
    build.add_argument("--schedule", required=True)
    build.add_argument("--expected-schedule-sha256", required=True)
    build.add_argument("--expected-frozen-inputs-sha256", required=True)
    build.add_argument("--val-inputs", required=True)
    build.add_argument("--expected-val-inputs-sha256", required=True)
    build.add_argument("--pipeline", required=True)
    build.add_argument("--expected-pipeline-sha256", required=True)
    build.add_argument("--formal-python", required=True)
    build.add_argument("--guarded-runner", default="/tmp/globaldiff_guarded_runner.py")
    build.add_argument("--expected-guarded-runner-sha256", required=True)
    build.add_argument("--expected-guarded-runner-bytes", type=int, required=True)
    build.add_argument("--guard-verifier", default="/tmp/verify_globaldiff_guards.py")
    build.add_argument("--expected-guard-verifier-sha256", required=True)
    build.add_argument("--expected-guard-verifier-bytes", type=int, required=True)
    build.add_argument("--run-root-base", required=True)
    build.add_argument("--reconcile-selection-root", required=True)
    build.add_argument("--paspa-root", required=True)
    build.add_argument("--diffsheg-root", required=True)
    build.add_argument("--seed", type=int, default=20260731)
    build.add_argument("--diffsheg-batch-size", type=int, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-campaign":
        result = build_campaign(args)
        sys.stdout.buffer.write(canonical_json_bytes(result))
        return 0
    campaign = load_campaign(
        args.campaign, args.expected_campaign_sha256, args.expected_campaign_bytes
    )
    if args.command == "run-next":
        result = run_next(campaign)
    elif args.command == "recover-e1":
        result = recover_e1(
            campaign, args.failed_campaign,
            args.expected_failed_campaign_sha256,
            args.expected_failed_campaign_bytes,
        )
    else:
        result = finalize_campaign(campaign)
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupervisorError as error:
        sys.stderr.write("supervisor-error: %s\n" % error)
        raise SystemExit(2)
