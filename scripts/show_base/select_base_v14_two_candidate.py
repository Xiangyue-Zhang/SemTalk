#!/usr/bin/env python3
"""Select the SemTalk SHOW V14 winner from exactly two validated reports.

This is a CPU-only, fail-closed adapter around the corrected 70a70f4 official
``validate_quality_report`` runtime replay.  The already-produced validation
reports remain bound to their frozen 4066f20 evidence pipeline, while the
short-quality training artifacts remain bound to the earlier 5b84075 source.
Keeping those three authorities explicit prevents an evaluator-only source
update from being misrepresented as a training-source change or an old report
from being relabelled as a new inference.  A legacy nine-mode decision may
be replayed as independent audit evidence when it exists.  Its absence is
recorded explicitly (and may itself be evidenced by the nine probes plus the
failed W1 short-quality receipt), but neither audit state can decide or block
the V14 winner.  This adapter never opens a test-set artifact.
"""

from __future__ import annotations

import argparse
import copy
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
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True


class SelectionError(RuntimeError):
    """Raised when frozen V14 selection authority is incomplete or stale."""


PROTOCOL_SHA256 = (
    "ad556b191f580b21e589a0f4800ccd6bce38acff6183d1194223fe592ef0bc69"
)
SOURCE_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
TRAINING_SOURCE_COMMIT = "5b84075bb5bc9577a891a1f5ff72e93c39bab2e8"
TRAINING_SOURCE_TREE = "05372163157d24cc8a72c073ac172b7c1859bba0"
RUNTIME_VALIDATION_SOURCE_COMMIT = "70a70f452bdf743e317b583a7770980f0ce744c3"
RUNTIME_VALIDATION_SOURCE_TREE = "bdf7680f56f9f53c92e7ab0bf6c6a84ef4f83d69"
PIPELINE_EVIDENCE_SOURCE_COMMIT = "4066f2096e1675f9c19d725894007ff25f3e9b4b"
PIPELINE_EVIDENCE_SOURCE_TREE = "0b66e3aa1fb23732e76e51492737c4ab1f4db2d0"
OFFICIAL_SELECTOR_SHA256 = (
    "2510956db237d5f622517e9828ee4704e75b98cd47ddbc8ce888b56f37f70769"
)
OFFICIAL_TRAIN_CONTRACT_SHA256 = (
    "29fdd5d3e9bdfc61904f649b71d4dae1766b42a4a6a5a40b2bbd37a4c6b33173"
)
OFFICIAL_VALIDATION_CONTRACT_SHA256 = (
    "0168e7f9ab2b9a122656ed98c36dc36777f577816ef08b0186ffc892813d2c70"
)
OFFICIAL_DIFFSHEG_ADAPTER_SHA256 = (
    "1857460c188096fcd4b24a9f8f5a0b0390a80eacd20744a46098195ac0f39ddf"
)
TOPOLOGY_GATE_SHA256 = (
    "1ee9ae31e2ca735265972022c26a82ba2789f7538a128631168af4e86f7808c4"
)
QUALITY_GATE_SHA256 = (
    "bd5286d8845b04c20f2e954334ac4d4907c6837d60ca5101ca5ec735291af78f"
)
CANDIDATE_EPOCHS = (1, 2, 4, 8, 16, 32)
MODE_P1 = "validation_gated_w8_l256_g2048_empirical_acceleration"
MODE_P2 = "validation_gated_w8_l128_g1024_empirical_acceleration"
MODES = (MODE_P1, MODE_P2)
TRAINER_NATIVE_UNAVAILABLE_FORMAT = (
    "semtalk_show_base_trainer_native_unavailable_audit_v1"
)
TRAINER_NATIVE_UNAVAILABLE_REASON = (
    "w1_reference_quality_report_unavailable"
)
TRAINER_NATIVE_NOT_PROVIDED_REASON = "trainer_native_audit_not_provided"
W1_FAILURE_ERROR = "throughput gate does not bind the exact training protocol"
MODE_PROTOCOL = {
    MODE_P1: {
        "id": "quality_p1_w8g2048_worker",
        "world_size": 8,
        "local_batch_size": 256,
        "global_batch_size": 2048,
    },
    MODE_P2: {
        "id": "quality_p2_w8g1024_master",
        "world_size": 8,
        "local_batch_size": 128,
        "global_batch_size": 1024,
    },
}
TOPOLOGY_VARYING_PROTOCOL_KEYS = {
    "node_count",
    "local_world_size",
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "expected_updates_per_epoch",
    "expected_unique_samples_per_epoch",
    "distributed_topology",
}
TOPOLOGY_VARYING_TRAJECTORY_KEYS = {
    "topology_mode",
    "topology_classification",
    "world_size",
    "node_count",
    "local_world_size",
    "local_batch_size",
    "global_batch_size",
    "updates_per_epoch",
    "unique_samples_per_epoch",
    "sha256",
    "payload_sha256",
}
EXPECTED_PROTOCOL = {
    "candidate_epochs": list(CANDIDATE_EPOCHS),
    "candidate_topologies": [
        {
            "global_batch_size": 2048,
            "id": "quality_p1_w8g2048_worker",
            "local_batch_size": 256,
            "mode": MODE_P1,
            "world_size": 8,
        },
        {
            "global_batch_size": 1024,
            "id": "quality_p2_w8g1024_master",
            "local_batch_size": 128,
            "mode": MODE_P2,
            "world_size": 8,
        },
    ],
    "created_cst": "2026-08-02T01:38:32+0800",
    "decision": {
        "final_test_runs_after_formal_selection": 1,
        "formal_base_training_restarts_from_the_same_frozen_initialization": True,
        "formal_checkpoint_selection_split": "val",
        "formal_target_epochs": 400,
        "select_exactly_one_topology": True,
        "selected_quality_checkpoint_is_evidence_only": True,
    },
    "format": "semtalk_show_base_v14_fast_selection_protocol_v1",
    "quality_campaign_root": (
        "/efs/xiangyuezhang/"
        "semtalk_show_base_topology_quality_5b84075_20260802_v14_fast"
    ),
    "scope": {
        "base_motion_only": True,
        "dataset": "SHOW",
        "repository": SOURCE_ORIGIN,
        "semgate": False,
        "source_commit": TRAINING_SOURCE_COMMIT,
        "source_tree": TRAINING_SOURCE_TREE,
        "sparse_motion_generation": False,
        "speaker_scope": "All",
        "speakers": ["oliver", "chemistry", "seth", "conan"],
    },
    "selection_tuple": [
        "validation_diffsheg_fgd_ascending",
        "epoch_ascending",
        "topology_mode_lexicographic_ascending",
    ],
    "status": "frozen_before_validation_results",
    "validation": {
        "expected_reports": 12,
        "metric_direction": "minimize",
        "primary_metric": "DiffSHEG FGD",
        "require_all_metrics_finite": True,
        "require_all_reports_complete": True,
        "require_exact_source_data_vq_init_and_runtime_lineage": True,
        "split": "val",
        "test_measurements_authorized": 0,
        "test_visible": False,
    },
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise SelectionError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise SelectionError(f"{label} must be one JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelectionError("value is not strict canonical JSON") from error


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SelectionError(f"{label} is not one lowercase SHA-256")
    return value


def _read_regular(path: Path, label: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise SelectionError(f"{label} path is not absolute/canonical")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise SelectionError(f"could not safely open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SelectionError(f"{label} is not a regular file")
        if before.st_nlink != 1:
            raise SelectionError(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(candidate, follow_symlinks=False)
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise SelectionError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise SelectionError(f"{label} disappeared") from error
    if resolved != candidate or candidate.is_symlink():
        raise SelectionError(f"{label} path is not canonical")
    current_path = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current_path /= part
        if stat.S_ISLNK(os.lstat(current_path).st_mode):
            raise SelectionError(f"{label} path contains a symlink")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise SelectionError(f"{label} size changed")
    return resolved, data


def _verified_json_file(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_bytes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_sha(expected_sha256, f"{label} expected file SHA-256")
    resolved, data = _read_regular(path, label)
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected_sha256:
        raise SelectionError(f"{label} file SHA-256 mismatch")
    if expected_bytes is not None and (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or len(data) != expected_bytes
    ):
        raise SelectionError(f"{label} byte count mismatch")
    return _strict_json(data, label), {
        "path": str(resolved),
        "sha256": observed,
        "bytes": len(data),
    }


def load_protocol(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _require_sha(
        expected_sha256, "V14 frozen selection protocol external SHA-256"
    )
    if expected != PROTOCOL_SHA256:
        raise SelectionError(
            "V14 frozen selection protocol external SHA-256 changed"
        )
    payload, artifact = _verified_json_file(
        path, expected, "V14 frozen selection protocol"
    )
    if payload != EXPECTED_PROTOCOL:
        raise SelectionError("V14 frozen selection protocol semantic payload changed")
    return payload, artifact


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise SelectionError(f"git authority query failed: {' '.join(args)}") from error


def _git_result(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise SelectionError(f"git authority query failed: {' '.join(args)}") from error


def _verified_code_file(path: Path, expected_sha256: str, label: str) -> None:
    _require_sha(expected_sha256, f"{label} expected SHA-256")
    _resolved, data = _read_regular(path, label)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise SelectionError(f"{label} SHA-256 mismatch")


def _module_file(module: Any, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise SelectionError(f"{label} has no concrete source file")
    try:
        return Path(raw).resolve(strict=True)
    except OSError as error:
        raise SelectionError(f"{label} source file is absent") from error


def load_official_modules(root: Path) -> tuple[ModuleType, ModuleType]:
    project = Path(root).resolve(strict=True)
    if (
        _git(project, "rev-parse", "HEAD")
        != RUNTIME_VALIDATION_SOURCE_COMMIT
        or _git(project, "rev-parse", "HEAD^{tree}")
        != RUNTIME_VALIDATION_SOURCE_TREE
        or _git(project, "remote", "get-url", "origin") != SOURCE_ORIGIN
        or _git(project, "status", "--porcelain=v1", "--untracked-files=all")
        != ""
        or _git(project, "for-each-ref", "--format=%(refname)", "refs/heads")
        != ""
    ):
        raise SelectionError("official project Git source authority changed")
    symbolic = _git_result(project, "symbolic-ref", "-q", "HEAD")
    if symbolic.returncode == 0 or symbolic.stdout.strip():
        raise SelectionError("official project must have detached HEAD")
    if symbolic.returncode != 1:
        raise SelectionError("could not prove official project has detached HEAD")
    selector_path = project / "scripts/show_base/select_base_training_topology.py"
    train_path = project / "scripts/show_base/train_base_official_adapt_long.py"
    validation_path = project / "scripts/show_base/base_long_val_contract.py"
    diffsheg_path = project / "scripts/show_base/select_base_official_adapt.py"
    topology_gate = project / "configs/show_base/semtalk_base_topology_gate_spec_20260731.json"
    quality_gate = project / "configs/show_base/semtalk_base_topology_quality_gate_spec_v4_20260801.json"
    _verified_code_file(
        selector_path, OFFICIAL_SELECTOR_SHA256, "official selector source"
    )
    _verified_code_file(
        train_path, OFFICIAL_TRAIN_CONTRACT_SHA256, "official training contract"
    )
    _verified_code_file(
        validation_path,
        OFFICIAL_VALIDATION_CONTRACT_SHA256,
        "official validation contract",
    )
    _verified_code_file(
        diffsheg_path,
        OFFICIAL_DIFFSHEG_ADAPTER_SHA256,
        "official DiffSHEG adapter",
    )
    _verified_json_file(
        topology_gate, TOPOLOGY_GATE_SHA256, "topology gate specification"
    )
    _verified_json_file(
        quality_gate, QUALITY_GATE_SHA256, "quality gate specification"
    )
    expected_imports = {
        "scripts.show_base.train_base_official_adapt_long": train_path,
        "scripts.show_base.base_long_val_contract": validation_path,
        "scripts.show_base.select_base_official_adapt": diffsheg_path,
    }
    for name, expected_path in expected_imports.items():
        existing = sys.modules.get(name)
        if existing is not None and _module_file(existing, name) != expected_path:
            raise SelectionError(f"pre-existing {name} would pollute official imports")
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location(
        "semtalk_v14_official_selector", selector_path
    )
    if spec is None or spec.loader is None:
        raise SelectionError("could not load official selector module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = module.contract
    if (
        _module_file(module, "official selector") != selector_path
        or _module_file(contract, "imported official training contract")
        != train_path
        or _module_file(module.formal_validation, "imported formal validation")
        != validation_path
        or _module_file(
            module.formal_validation.diffsheg,
            "imported official DiffSHEG adapter",
        )
        != diffsheg_path
    ):
        raise SelectionError("official selector imported from the wrong path")
    return module, contract


def _validation_source_authority(root: Path) -> dict[str, Any]:
    """Return the independently pinned evaluator/validation source."""

    project = Path(root).resolve(strict=True)
    # Reuse the complete loader gate rather than accepting a caller-provided
    # commit label.  This also proves clean, detached, branchless Git state and
    # the exact selector/trainer/validation blobs before the authority is
    # serialized into the final decision receipt.
    load_official_modules(project)
    return {
        "origin": SOURCE_ORIGIN,
        "commit": RUNTIME_VALIDATION_SOURCE_COMMIT,
        "tree": RUNTIME_VALIDATION_SOURCE_TREE,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "selector_sha256": OFFICIAL_SELECTOR_SHA256,
        "training_contract_sha256": OFFICIAL_TRAIN_CONTRACT_SHA256,
        "validation_contract_sha256": OFFICIAL_VALIDATION_CONTRACT_SHA256,
        "diffsheg_adapter_sha256": OFFICIAL_DIFFSHEG_ADAPTER_SHA256,
    }


def _prevalidate_report(
    mode: str,
    path: Path,
    expected_file_sha256: str,
    expected_bytes: int,
    expected_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, artifact = _verified_json_file(
        path,
        expected_file_sha256,
        f"{mode} quality report v3",
        expected_bytes=expected_bytes,
    )
    expected_payload_sha256 = _require_sha(
        expected_payload_sha256, f"{mode} expected report payload SHA-256"
    )
    claimed = _require_sha(payload.get("receipt_sha256"), f"{mode} report payload SHA-256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    if claimed != expected_payload_sha256 or _canonical_sha(unsigned) != claimed:
        raise SelectionError(f"{mode} quality report payload SHA-256 mismatch")
    if payload.get("format") != "semtalk_show_base_topology_quality_report_v3":
        raise SelectionError(f"{mode} is not quality-report v3")
    return payload, {**artifact, "payload_sha256": claimed}


def _artifact_payload(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(str(value.get("path", "")))
    digest = _require_sha(value.get("sha256"), f"{label} SHA-256")
    expected_bytes = value.get("bytes")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise SelectionError(f"{label} byte count is invalid")
    payload, _artifact = _verified_json_file(
        path,
        digest,
        label,
        expected_bytes=expected_bytes,
    )
    claimed_key = "receipt_payload_sha256"
    claimed = value.get(claimed_key)
    if claimed is not None:
        claimed = _require_sha(claimed, f"{label} payload SHA-256")
        observed = payload.get(claimed_key)
        if observed != claimed:
            raise SelectionError(f"{label} payload identity changed")
        unsigned = dict(payload)
        unsigned.pop(claimed_key, None)
        if _canonical_sha(unsigned) != claimed:
            raise SelectionError(f"{label} payload self-hash mismatch")
    return payload


def _validation_pipeline_authority(
    validated: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    """Prove that one validation report used the 4066f20 pipeline."""

    pipeline = _artifact_payload(
        validated["pipeline_receipt"], f"{mode} validation pipeline"
    )
    source = pipeline.get("source")
    expected_keys = {
        "origin",
        "source_root",
        "commit",
        "tree",
        "clean",
        "detached",
        "local_branches_at_commit",
    }
    source_root = (
        Path(str(source.get("source_root", "")))
        if isinstance(source, dict)
        else Path("")
    )
    if (
        pipeline.get("split") != "val"
        or pipeline.get("test_visible") is not False
        or not isinstance(source, dict)
        or set(source) != expected_keys
        or source.get("origin") != SOURCE_ORIGIN
        or source.get("commit") != PIPELINE_EVIDENCE_SOURCE_COMMIT
        or source.get("tree") != PIPELINE_EVIDENCE_SOURCE_TREE
        or source.get("clean") is not True
        or source.get("detached") is not True
        or source.get("local_branches_at_commit") != []
        or not source_root.is_absolute()
        or ".." in source_root.parts
    ):
        raise SelectionError(
            f"{mode} validation pipeline is not the pinned 4066f20 val-only "
            "evidence source"
        )
    return copy.deepcopy(source)


def _load_mode_frozen(validated: Mapping[str, Any], mode: str) -> dict[str, Any]:
    short = _artifact_payload(validated["short_trajectory_receipt"], f"{mode} short trajectory")
    ready_values = short.get("candidate_ready_receipts")
    if not isinstance(ready_values, list) or len(ready_values) != len(CANDIDATE_EPOCHS):
        raise SelectionError(f"{mode} candidate-ready coverage changed")
    ready = _artifact_payload(ready_values[0], f"{mode} first candidate-ready receipt")
    frozen_value = ready.get("frozen_inputs")
    if not isinstance(frozen_value, dict) or set(frozen_value) != {
        "path", "sha256", "receipt_payload_sha256"
    }:
        raise SelectionError(f"{mode} frozen-input authority changed")
    frozen, _artifact = _verified_json_file(
        Path(frozen_value["path"]),
        _require_sha(frozen_value["sha256"], f"{mode} frozen file SHA-256"),
        f"{mode} frozen inputs",
    )
    claimed = _require_sha(frozen.get("receipt_sha256"), f"{mode} frozen payload SHA-256")
    unsigned = dict(frozen)
    unsigned.pop("receipt_sha256")
    if (
        claimed != frozen_value["receipt_payload_sha256"]
        or _canonical_sha(unsigned) != claimed
    ):
        raise SelectionError(f"{mode} frozen inputs self-hash mismatch")
    return frozen


def _without_keys(value: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in value.items() if key not in keys}


def _authority_projection(frozen: Mapping[str, Any], mode: str, contract: ModuleType) -> dict[str, str]:
    source = frozen.get("source")
    dataset = frozen.get("dataset")
    official = frozen.get("official_base")
    initialization = frozen.get("speaker_initialization")
    protocol = frozen.get("protocol")
    long_contract = frozen.get("long_contract")
    topology = frozen.get("topology")
    if not all(isinstance(value, dict) for value in (
        source, dataset, official, initialization, protocol, long_contract, topology
    )):
        raise SelectionError(f"{mode} frozen authority schema changed")
    if source != {
        **source,
        "origin": SOURCE_ORIGIN,
        "commit": TRAINING_SOURCE_COMMIT,
        "tree": TRAINING_SOURCE_TREE,
        "clean": True,
    }:
        raise SelectionError(f"{mode} source authority changed")
    expected = contract.TOPOLOGY_SPECS.get(mode)
    fixed = MODE_PROTOCOL[mode]
    if not isinstance(expected, dict) or any(
        expected.get(key) != fixed[key]
        for key in ("world_size", "local_batch_size", "global_batch_size")
    ):
        raise SelectionError(f"{mode} registered topology changed")
    if any(protocol.get(key) != expected[key] for key in (
        "node_count", "local_world_size", "world_size", "local_batch_size",
        "global_batch_size", "precision"
    )):
        raise SelectionError(f"{mode} frozen runtime topology changed")
    distributed = protocol.get("distributed_topology")
    if not isinstance(distributed, dict) or distributed.get("mode") != mode:
        raise SelectionError(f"{mode} distributed topology identity changed")
    if any(topology.get(key) != expected[key] for key in (
        "node_count", "local_world_size", "world_size", "local_batch_size",
        "global_batch_size", "updates_per_epoch", "unique_samples_per_epoch"
    )) or topology.get("topology_mode") != mode:
        raise SelectionError(f"{mode} topology receipt changed")

    portable_dataset = _without_keys(
        dataset, {"lmdb_binding_scope", "node_lmdb_inode_bindings"}
    )
    portable_official = _without_keys(
        official, {"file_binding_scope", "node_local_files"}
    )
    runtime = _without_keys(protocol, TOPOLOGY_VARYING_PROTOCOL_KEYS)
    normalized_long = copy.deepcopy(long_contract)
    trajectory = normalized_long.get("trajectory_anchor")
    if isinstance(trajectory, dict):
        normalized_long["trajectory_anchor"] = _without_keys(
            trajectory, TOPOLOGY_VARYING_TRAJECTORY_KEYS
        )
    return {
        "source_sha256": _canonical_sha(source),
        "data_vq_sha256": _canonical_sha({
            "dataset": portable_dataset,
            "official_base": portable_official,
        }),
        "initialization_sha256": _canonical_sha(initialization),
        "runtime_objective_sha256": _canonical_sha({
            "protocol": runtime,
            "long_contract": normalized_long,
        }),
    }


def _validate_gate_specs(root: Path, selector: ModuleType, contract: ModuleType) -> None:
    topology_path = root / "configs/show_base/semtalk_base_topology_gate_spec_20260731.json"
    quality_path = root / "configs/show_base/semtalk_base_topology_quality_gate_spec_v4_20260801.json"
    for mode in MODES:
        receipt = contract.validate_topology_gate_spec(SimpleNamespace(
            topology_gate_spec=topology_path,
            expected_topology_gate_spec_sha256=TOPOLOGY_GATE_SHA256,
            topology_mode=mode,
        ))
        if receipt.get("sha256") != TOPOLOGY_GATE_SHA256:
            raise SelectionError("topology gate validation identity changed")
    quality = selector.validate_quality_gate_spec(quality_path, QUALITY_GATE_SHA256)
    if quality.get("sha256") != QUALITY_GATE_SHA256:
        raise SelectionError("quality gate validation identity changed")


def _validate_trainer_native_selection(
    path: Path,
    expected_sha256: str,
    *,
    selector: ModuleType,
    contract: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freshly replay the complete legacy nine-mode safety authority.

    The legacy selector remains useful as an independent quality-safety
    audit, but its fastest-safe policy is not the frozen V14 min-FGD policy.
    Consequently its ``selected`` row is recorded without influencing the
    V14 winner below.
    """

    value, artifact = _verified_json_file(
        path, expected_sha256, "trainer-native nine-mode selection"
    )
    probes_raw = value.get("probes")
    reports_raw = value.get("quality_reports")
    skips_raw = value.get("quality_skips")
    if (
        value.get("format") != contract.TOPOLOGY_SELECTION_FORMAT
        or value.get("status") != "pass"
        or value.get("topology_gate_spec_sha256") != TOPOLOGY_GATE_SHA256
        or value.get("quality_gate_spec_sha256") != QUALITY_GATE_SHA256
        or not isinstance(probes_raw, list)
        or not isinstance(reports_raw, list)
        or not isinstance(skips_raw, list)
    ):
        raise SelectionError("trainer-native selection envelope changed")
    try:
        probes = [
            selector.validate_probe(
                str(row["mode"]),
                Path(str(row["report_path"])),
                str(row["report_sha256"]),
                gate_spec_sha256=TOPOLOGY_GATE_SHA256,
            )
            for row in probes_raw
            if isinstance(row, dict)
        ]
        reports = [
            selector.validate_quality_report(
                str(row["mode"]),
                Path(str(row["report_path"])),
                str(row["report_sha256"]),
                quality_gate_spec_sha256=QUALITY_GATE_SHA256,
                topology_gate_spec_sha256=TOPOLOGY_GATE_SHA256,
            )
            for row in reports_raw
            if isinstance(row, dict)
        ]
        skips = [
            selector.validate_quality_skip(
                str(row["mode"]),
                Path(str(row["receipt_path"])),
                str(row["receipt_sha256"]),
                topology_gate_spec_sha256=TOPOLOGY_GATE_SHA256,
                quality_gate_spec_sha256=QUALITY_GATE_SHA256,
            )
            for row in skips_raw
            if isinstance(row, dict)
        ]
        replayed = selector.select_topology(
            probes,
            reports,
            gate_spec_sha256=TOPOLOGY_GATE_SHA256,
            quality_gate_spec_sha256=QUALITY_GATE_SHA256,
            quality_skips=skips,
        )
    except Exception as error:
        raise SelectionError(
            "trainer-native nine-mode selection replay failed"
        ) from error
    selected = value.get("selected")
    if (
        len(probes) != len(probes_raw)
        or len(reports) != len(reports_raw)
        or len(skips) != len(skips_raw)
        or value != replayed
        or not isinstance(selected, dict)
        or selected.get("mode") not in contract.TOPOLOGY_SPECS
        or value.get("receipt_sha256") != _canonical_sha(
            {
                key: item
                for key, item in value.items()
                if key != "receipt_sha256"
            }
        )
    ):
        raise SelectionError(
            "trainer-native nine-mode selection differs from fresh replay"
        )
    return value, artifact


def _canonical_directory(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise SelectionError(f"{label} must be one absolute directory path")
    candidate = Path(value)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise SelectionError(f"{label} is not absolute/canonical")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise SelectionError(f"{label} is absent") from error
    if resolved != candidate or candidate.is_symlink() or not candidate.is_dir():
        raise SelectionError(f"{label} is not one canonical directory")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise SelectionError(f"{label} contains a symlink")
    return candidate


def _validate_trainer_native_unavailable_receipt(
    path: Path,
    expected_sha256: str,
    *,
    selector: ModuleType,
    contract: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the evidence explaining why native nine-mode audit is absent."""

    value, artifact = _verified_json_file(
        path,
        expected_sha256,
        "trainer-native unavailable audit receipt",
    )
    required = {
        "format",
        "status",
        "role",
        "authoritative_for_winner",
        "reason_code",
        "topology_gate_spec_sha256",
        "quality_gate_spec_sha256",
        "probe_campaign_root",
        "quality_campaign_root",
        "probes",
        "blocking_failure",
        "receipt_sha256",
    }
    probes_raw = value.get("probes")
    failure_raw = value.get("blocking_failure")
    expected_modes = list(contract.TOPOLOGY_SPECS)
    if (
        set(value) != required
        or value.get("format") != TRAINER_NATIVE_UNAVAILABLE_FORMAT
        or value.get("status") != "unavailable_not_authoritative"
        or value.get("role") != "audit_only"
        or value.get("authoritative_for_winner") is not False
        or value.get("reason_code") != TRAINER_NATIVE_UNAVAILABLE_REASON
        or value.get("topology_gate_spec_sha256") != TOPOLOGY_GATE_SHA256
        or value.get("quality_gate_spec_sha256") != QUALITY_GATE_SHA256
        or not isinstance(probes_raw, list)
        or len(probes_raw) != len(expected_modes)
        or not isinstance(failure_raw, dict)
        or value.get("receipt_sha256")
        != _canonical_sha(
            {key: item for key, item in value.items() if key != "receipt_sha256"}
        )
    ):
        raise SelectionError("trainer-native unavailable audit envelope changed")

    probe_root = _canonical_directory(
        value["probe_campaign_root"], "trainer-native probe campaign root"
    )
    quality_root = _canonical_directory(
        value["quality_campaign_root"], "trainer-native quality campaign root"
    )
    validated_modes: list[str] = []
    for expected_mode, row in zip(expected_modes, probes_raw):
        if not isinstance(row, dict) or set(row) != {
            "mode",
            "path",
            "sha256",
            "bytes",
        }:
            raise SelectionError("trainer-native probe artifact schema changed")
        mode = row.get("mode")
        probe_path = Path(str(row.get("path")))
        if (
            mode != expected_mode
            or probe_path.name != "throughput_gate.json"
            or probe_path.parent.parent != probe_root / "probes"
        ):
            raise SelectionError("trainer-native probe coverage/order changed")
        _payload, verified = _verified_json_file(
            probe_path,
            str(row.get("sha256")),
            f"trainer-native probe {mode}",
            expected_bytes=row.get("bytes"),
        )
        try:
            replayed = selector.validate_probe(
                mode,
                Path(verified["path"]),
                verified["sha256"],
                gate_spec_sha256=TOPOLOGY_GATE_SHA256,
            )
        except Exception as error:
            raise SelectionError(
                f"trainer-native probe replay failed for {mode}"
            ) from error
        if (
            not isinstance(replayed, dict)
            or replayed.get("mode") != mode
            or replayed.get("report_path") != verified["path"]
            or replayed.get("report_sha256") != verified["sha256"]
        ):
            raise SelectionError(f"trainer-native probe replay changed for {mode}")
        validated_modes.append(mode)

    if set(failure_raw) != {"mode", "path", "sha256", "bytes"}:
        raise SelectionError("trainer-native blocking failure schema changed")
    failure_mode = failure_raw.get("mode")
    failure_path = Path(str(failure_raw.get("path")))
    if (
        failure_mode != contract.OFFICIAL_W1_REFERENCE_MODE
        or failure_path.name != "failure.json"
        or failure_path.parent.parent != quality_root / "training"
    ):
        raise SelectionError("trainer-native blocking failure path/mode changed")
    failure, _failure_artifact = _verified_json_file(
        failure_path,
        str(failure_raw.get("sha256")),
        "trainer-native W1 blocking failure",
        expected_bytes=failure_raw.get("bytes"),
    )
    failed_unix = failure.get("failed_unix")
    if (
        set(failure)
        != {
            "format",
            "status",
            "error_type",
            "error",
            "failed_unix",
            "run_purpose",
            "target_epochs",
        }
        or failure.get("format") != contract.SHORT_QUALITY_STATUS_FORMAT
        or failure.get("status") != "failed"
        or failure.get("error_type") != "AdaptationContractError"
        or failure.get("error") != W1_FAILURE_ERROR
        or isinstance(failed_unix, bool)
        or not isinstance(failed_unix, (int, float))
        or not math.isfinite(float(failed_unix))
        or float(failed_unix) <= 0.0
        or failure.get("run_purpose") != contract.RUN_PURPOSE_SHORT_QUALITY
        or failure.get("target_epochs") != list(selector.W1_REFERENCE_EPOCHS)
    ):
        raise SelectionError("trainer-native W1 blocking failure changed")
    return {
        "role": "audit_only",
        "authoritative_for_winner": False,
        "status": "unavailable_not_authoritative",
        "reason_code": TRAINER_NATIVE_UNAVAILABLE_REASON,
        "evidence_receipt": {
            **artifact,
            "receipt_sha256": value["receipt_sha256"],
        },
        "probe_modes": validated_modes,
        "blocking_mode": failure_mode,
    }, artifact


def _trainer_native_audit(
    *,
    trainer_native_selection: Path | None,
    expected_trainer_native_selection_sha256: str | None,
    trainer_native_unavailable_receipt: Path | None,
    expected_trainer_native_unavailable_receipt_sha256: str | None,
    selector: ModuleType,
    contract: ModuleType,
    validate_specs: bool,
) -> dict[str, Any]:
    available_pair = (
        trainer_native_selection is not None,
        expected_trainer_native_selection_sha256 is not None,
    )
    unavailable_pair = (
        trainer_native_unavailable_receipt is not None,
        expected_trainer_native_unavailable_receipt_sha256 is not None,
    )
    if len(set(available_pair)) != 1 or len(set(unavailable_pair)) != 1:
        raise SelectionError("trainer-native path and SHA must be supplied together")
    if all(available_pair) and all(unavailable_pair):
        raise SelectionError("trainer-native audit alternatives are mutually exclusive")
    if all(available_pair):
        assert trainer_native_selection is not None
        assert expected_trainer_native_selection_sha256 is not None
        if validate_specs:
            native, artifact = _validate_trainer_native_selection(
                trainer_native_selection,
                expected_trainer_native_selection_sha256,
                selector=selector,
                contract=contract,
            )
        else:
            native, artifact = _verified_json_file(
                trainer_native_selection,
                expected_trainer_native_selection_sha256,
                "trainer-native nine-mode selection test fixture",
            )
            if (
                native.get("format")
                != "semtalk_show_base_topology_selection_v2"
                or native.get("status") != "pass"
                or not isinstance(native.get("selected"), dict)
            ):
                raise SelectionError("trainer-native test fixture is invalid")
        return {
            "role": "audit_only",
            "authoritative_for_winner": False,
            "status": "available_not_authoritative",
            "path": artifact["path"],
            "format": native["format"],
            "receipt_sha256": native["receipt_sha256"],
            "selected_mode": native["selected"]["mode"],
        }
    if all(unavailable_pair):
        assert trainer_native_unavailable_receipt is not None
        assert expected_trainer_native_unavailable_receipt_sha256 is not None
        audit, _artifact = _validate_trainer_native_unavailable_receipt(
            trainer_native_unavailable_receipt,
            expected_trainer_native_unavailable_receipt_sha256,
            selector=selector,
            contract=contract,
        )
        return audit
    return {
        "role": "audit_only",
        "authoritative_for_winner": False,
        "status": "unavailable_not_authoritative",
        "reason_code": TRAINER_NATIVE_NOT_PROVIDED_REASON,
        "evidence_receipt": None,
        "probe_modes": [],
        "blocking_mode": None,
    }


def select_two_reports(
    report_specs: Sequence[tuple[str, Path, str, int, str]],
    *,
    output: Path,
    project_root: Path,
    selection_protocol: Path,
    expected_selection_protocol_sha256: str,
    trainer_native_selection: Path | None = None,
    expected_trainer_native_selection_sha256: str | None = None,
    trainer_native_unavailable_receipt: Path | None = None,
    expected_trainer_native_unavailable_receipt_sha256: str | None = None,
    selector: ModuleType | None = None,
    contract: ModuleType | None = None,
    validate_specs: bool = True,
) -> dict[str, Any]:
    protocol, protocol_artifact = load_protocol(
        selection_protocol, expected_selection_protocol_sha256
    )
    if selector is None or contract is None:
        selector, contract = load_official_modules(project_root)
    root = Path(project_root).resolve(strict=True)
    validation_source_authority = _validation_source_authority(root)
    if validate_specs:
        _validate_gate_specs(root, selector, contract)
    trainer_native_audit = _trainer_native_audit(
        trainer_native_selection=trainer_native_selection,
        expected_trainer_native_selection_sha256=(
            expected_trainer_native_selection_sha256
        ),
        trainer_native_unavailable_receipt=(
            trainer_native_unavailable_receipt
        ),
        expected_trainer_native_unavailable_receipt_sha256=(
            expected_trainer_native_unavailable_receipt_sha256
        ),
        selector=selector,
        contract=contract,
        validate_specs=validate_specs,
    )
    if len(report_specs) != 2 or {item[0] for item in report_specs} != set(MODES):
        raise SelectionError("exactly one report for each frozen V14 topology is required")

    validated_by_mode: dict[str, dict[str, Any]] = {}
    input_artifacts: list[dict[str, Any]] = []
    frozen_by_mode: dict[str, dict[str, Any]] = {}
    projections: dict[str, dict[str, str]] = {}
    validation_pipeline_by_mode: dict[str, dict[str, Any]] = {}
    for mode, path, file_sha, byte_count, payload_sha in report_specs:
        if mode not in MODES:
            raise SelectionError(f"unexpected topology mode {mode!r}")
        raw_payload, artifact = _prevalidate_report(
            mode, path, file_sha, byte_count, payload_sha
        )
        raw_candidates = raw_payload.get("candidates")
        if (
            not isinstance(raw_candidates, list)
            or len(raw_candidates) != len(CANDIDATE_EPOCHS)
        ):
            raise SelectionError(f"{mode} raw candidate coverage changed")
        for expected_epoch, raw_row in zip(
            CANDIDATE_EPOCHS, raw_candidates
        ):
            raw_provenance = (
                raw_row.get("provenance")
                if isinstance(raw_row, dict)
                else None
            )
            if (
                not isinstance(raw_row, dict)
                or type(raw_row.get("epoch")) is not int
                or raw_row.get("epoch") != expected_epoch
                or not isinstance(raw_provenance, dict)
                or type(raw_provenance.get("epoch")) is not int
                or raw_provenance.get("epoch") != expected_epoch
            ):
                raise SelectionError(
                    f"{mode} raw e{expected_epoch} epoch is invalid"
                )
        try:
            validated = selector.validate_quality_report(
                mode,
                Path(artifact["path"]),
                artifact["sha256"],
                quality_gate_spec_sha256=QUALITY_GATE_SHA256,
                topology_gate_spec_sha256=TOPOLOGY_GATE_SHA256,
            )
        except Exception as error:
            raise SelectionError(f"official quality-report replay failed for {mode}") from error
        if (
            validated.get("mode") != mode
            or validated.get("quality_role") != "candidate_quality"
            or validated.get("reference_only") is not False
            or validated.get("report_sha256") != artifact["sha256"]
            or list(validated.get("candidate_fgd", {}))
            != [str(epoch) for epoch in CANDIDATE_EPOCHS]
        ):
            raise SelectionError(f"{mode} official replay coverage changed")
        validated_by_mode[mode] = validated
        if validate_specs:
            validation_pipeline_by_mode[mode] = (
                _validation_pipeline_authority(validated, mode)
            )
        input_artifacts.append({"mode": mode, **artifact})
        frozen = _load_mode_frozen(validated, mode)
        frozen_by_mode[mode] = frozen
        projections[mode] = _authority_projection(frozen, mode, contract)

    semantic = {
        validated_by_mode[mode]["topology_independent_input_sha256"]
        for mode in MODES
    }
    if len(semantic) != 1:
        raise SelectionError("two reports do not share topology-independent authority")
    if validated_by_mode[MODE_P1]["val_inputs_receipt"] != validated_by_mode[MODE_P2]["val_inputs_receipt"]:
        raise SelectionError("two reports do not share one validation-input authority")
    if validated_by_mode[MODE_P1]["pipeline_receipt"] != validated_by_mode[MODE_P2]["pipeline_receipt"]:
        raise SelectionError("two reports do not share one inference-pipeline authority")
    if projections[MODE_P1] != projections[MODE_P2]:
        raise SelectionError("source/data/VQ/init/runtime authority differs between reports")
    if validate_specs and (
        set(validation_pipeline_by_mode) != set(MODES)
        or validation_pipeline_by_mode[MODE_P1]
        != validation_pipeline_by_mode[MODE_P2]
    ):
        raise SelectionError(
            "two reports do not share the pinned 4066f20 validation source"
        )

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        validated = validated_by_mode[mode]
        candidates = validated.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(CANDIDATE_EPOCHS):
            raise SelectionError(f"{mode} candidate row coverage changed")
        for expected_epoch, row in zip(CANDIDATE_EPOCHS, candidates):
            value = row.get("diffsheg_fgd") if isinstance(row, dict) else None
            checkpoint = row.get("candidate_checkpoint") if isinstance(row, dict) else None
            provenance = row.get("provenance") if isinstance(row, dict) else None
            if (
                type(value) not in {int, float}
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or not isinstance(checkpoint, dict)
                or re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256"))) is None
                or not isinstance(provenance, dict)
                or type(row.get("epoch")) is not int
                or row.get("epoch") != expected_epoch
                or type(provenance.get("epoch")) is not int
                or provenance.get("epoch") != expected_epoch
                or provenance.get("topology_mode") != mode
                or provenance.get("split") != "val"
                or provenance.get("test_visible") is not False
            ):
                raise SelectionError(f"{mode} e{expected_epoch} result is invalid")
            rows.append({
                "topology_id": MODE_PROTOCOL[mode]["id"],
                "topology_mode": mode,
                "epoch": expected_epoch,
                "validation_diffsheg_fgd": float(value),
                "checkpoint_sha256": checkpoint["sha256"],
            })
    if len(rows) != 12:
        raise SelectionError("selection does not contain exactly 12 validation FGD values")
    ordered = sorted(
        rows,
        key=lambda row: (
            row["validation_diffsheg_fgd"],
            row["epoch"],
            row["topology_mode"],
        ),
    )
    winner = ordered[0]
    winner_tuple = (
        winner["validation_diffsheg_fgd"], winner["epoch"], winner["topology_mode"]
    )
    if sum(
        (
            row["validation_diffsheg_fgd"], row["epoch"], row["topology_mode"]
        ) == winner_tuple
        for row in rows
    ) != 1:
        raise SelectionError("selection tuple does not identify one unique winner")

    payload: dict[str, Any] = {
        "format": "semtalk_show_base_v14_two_candidate_selection_audit_v3",
        "status": "complete",
        "selection_protocol": protocol_artifact,
        "selection_tuple": list(protocol["selection_tuple"]),
        "validation_split": "val",
        "test_visible": False,
        "test_measurements_authorized": 0,
        "training_source_authority": {
            "origin": SOURCE_ORIGIN,
            "commit": TRAINING_SOURCE_COMMIT,
            "tree": TRAINING_SOURCE_TREE,
        },
        "validation_source_authority": validation_source_authority,
        "validation_pipeline_source": (
            validation_pipeline_by_mode.get(MODE_P1)
        ),
        "quality_reports": sorted(input_artifacts, key=lambda row: row["mode"]),
        "quality_report_sha256": {
            mode: validated_by_mode[mode]["report_sha256"] for mode in MODES
        },
        "topology_independent_input_sha256": next(iter(semantic)),
        "common_authority": projections[MODE_P1],
        "results": rows,
        "winner": {
            "topology_id": winner["topology_id"],
            "topology_mode": winner["topology_mode"],
            "epoch": winner["epoch"],
            "validation_diffsheg_fgd": winner["validation_diffsheg_fgd"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
        },
        "trainer_native_selection": trainer_native_audit,
        "formal_training": {
            "target_epochs": 400,
            "fresh": True,
            "restart_from_same_frozen_initialization": True,
            "selected_quality_checkpoint_is_evidence_only": True,
            "checkpoint_selection_split": "val",
            "final_test_runs_after_formal_selection": 1,
        },
    }
    payload["receipt_sha256"] = _canonical_sha(payload)
    _write_new_json(output, payload)
    return payload


def _validated_output_path(path: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise SelectionError(f"{label} path must be absolute and canonical")
    parent = candidate.parent.resolve(strict=True)
    if parent != candidate.parent or candidate.is_symlink():
        raise SelectionError(f"{label} parent/path is not canonical")
    return candidate


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    candidate = _validated_output_path(path, "output")
    parent = candidate.parent
    data = _canonical_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags, 0o600)
    except FileExistsError as error:
        raise SelectionError("refusing to overwrite selection output") from error
    except OSError as error:
        raise SelectionError("could not create selection output") from error
    try:
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-project-root", type=Path, required=True)
    parser.add_argument("--selection-protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-protocol-sha256",
        required=True,
    )
    parser.add_argument(
        "--trainer-native-selection",
        type=Path,
    )
    parser.add_argument(
        "--expected-trainer-native-selection-sha256",
    )
    parser.add_argument(
        "--trainer-native-unavailable-receipt",
        type=Path,
    )
    parser.add_argument(
        "--expected-trainer-native-unavailable-receipt-sha256",
    )
    parser.add_argument(
        "--quality-report",
        action="append",
        nargs=5,
        metavar=("MODE", "PATH", "FILE_SHA256", "BYTES", "PAYLOAD_SHA256"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        specs = [
            (mode, Path(path), file_sha, int(byte_count), payload_sha)
            for mode, path, file_sha, byte_count, payload_sha in args.quality_report
        ]
        payload = select_two_reports(
            specs,
            output=args.output,
            project_root=args.official_project_root,
            selection_protocol=args.selection_protocol,
            expected_selection_protocol_sha256=(
                args.expected_selection_protocol_sha256
            ),
            trainer_native_selection=args.trainer_native_selection,
            expected_trainer_native_selection_sha256=(
                args.expected_trainer_native_selection_sha256
            ),
            trainer_native_unavailable_receipt=(
                args.trainer_native_unavailable_receipt
            ),
            expected_trainer_native_unavailable_receipt_sha256=(
                args.expected_trainer_native_unavailable_receipt_sha256
            ),
        )
    except (SelectionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"output": str(args.output), "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
