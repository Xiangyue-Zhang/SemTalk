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
from contextlib import contextmanager
import hashlib
import importlib
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
MEASUREMENT_FORMAT = "semtalk_show_base_live_val_measurement_v2"
SELECTION_FORMAT = "semtalk_show_base_live_val_22way_selection_v2"

CANDIDATE_EPOCHS = (
    1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80,
    100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400,
)
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


def _load_validation_modules(source_root: Path) -> dict[str, ModuleType]:
    _project_modules_are_from(source_root)
    sys.path.insert(0, str(source_root))
    try:
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
        return modules
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)


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


def _claim_path(
    *,
    producer_ready_receipt: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    epoch: int,
) -> Path:
    """Return the one claim slot for the producer candidate itself.

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
    claim_dir = training_root / "live_val_consumer_claims"
    created = False
    if os.path.lexists(claim_dir):
        resolved = _canonical_path(claim_dir, "consumer claim directory")
        if not resolved.is_dir():
            raise LiveConsumerError("consumer claim path is not a directory")
    else:
        try:
            claim_dir.mkdir(mode=0o755)
            created = True
        except FileExistsError:
            pass
        resolved = _canonical_path(claim_dir, "consumer claim directory")
        if not resolved.is_dir():
            raise LiveConsumerError("consumer claim directory race was unsafe")
    if created:
        parent_fd = os.open(claim_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    return resolved / expected_name


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
    claim_artifact["receipt_payload_sha256"] = claim_body["receipt_payload_sha256"]
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
    _exact_keys(claim, CLAIM_KEYS, "consumer claim")
    claim_artifact["receipt_payload_sha256"] = claim["receipt_payload_sha256"]
    if (
        claim.get("format") != CLAIM_FORMAT
        or claim.get("status") != "claimed"
        or claim.get("split") != "val"
        or claim.get("test_visible") is not False
        or claim.get("selection_eligible") is not False
        or _integer(claim.get("candidate_epoch"), "claim candidate epoch", minimum=1)
        != epochs[0]
        or _integer(claim.get("expected_shards"), "claim expected shards", minimum=1)
        != 8
        or not _strict_equal(claim.get("work_authority"), authority_artifact)
        or not _strict_equal(claim_artifact, claim_value)
    ):
        raise LiveConsumerError("consumer claim changed")
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
def _engine_adapter(engine: ModuleType) -> Iterable[None]:
    previous = engine._preflight_artifact
    engine._preflight_artifact = _preflight_artifact
    try:
        yield
    finally:
        engine._preflight_artifact = previous


def _engine_command(args: argparse.Namespace, command: str) -> dict[str, Any]:
    _artifact_value, preflight = _preflight_artifact(
        args.preflight, args.expected_preflight_sha256
    )
    epoch = preflight["candidate_epochs"][0]
    if args.epoch != epoch or args.num_shards != 8:
        raise LiveConsumerError("engine command differs from one-candidate eight-shard work")
    authority = preflight["work_authority"]
    _authority_artifact, _authority, modules = _validate_work_authority(
        Path(authority["path"]), authority["sha256"]
    )
    assert modules is not None
    engine = modules["engine"]
    with _engine_adapter(engine):
        if command == "shard":
            return engine.run_shard(args)
        return engine.finalize(args)


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
