#!/usr/bin/env python3
"""Fail-closed CPU control plane for the SHOW V14 Base formal run.

The V14 decision is deliberately separate from the retained nine-topology
gate.  The latter remains useful safety evidence, but only the twelve
validation DiffSHEG FGD measurements select the formal topology.  This module
freshly opens and replays every decision artifact before a 400-epoch run may
start.  It contains no model, loss, optimizer, or data-loop implementation.
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
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True


class V14ContractError(RuntimeError):
    """Raised when the V14 formal-training authority is incomplete or stale."""


ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
TRAINING_SOURCE_COMMIT = "5b84075bb5bc9577a891a1f5ff72e93c39bab2e8"
TRAINING_SOURCE_TREE = "05372163157d24cc8a72c073ac172b7c1859bba0"
VALIDATION_SOURCE_COMMIT = "4066f2096e1675f9c19d725894007ff25f3e9b4b"
VALIDATION_SOURCE_TREE = "0b66e3aa1fb23732e76e51492737c4ab1f4db2d0"
SELECTION_PROTOCOL_SHA256 = (
    "ad556b191f580b21e589a0f4800ccd6bce38acff6183d1194223fe592ef0bc69"
)
TOPOLOGY_GATE_SHA256 = (
    "1ee9ae31e2ca735265972022c26a82ba2789f7538a128631168af4e86f7808c4"
)
QUALITY_GATE_SHA256 = (
    "bd5286d8845b04c20f2e954334ac4d4907c6837d60ca5101ca5ec735291af78f"
)
PROTOCOL_FORMAT = "semtalk_show_base_v14_fast_selection_protocol_v1"
AUDIT_FORMAT = "semtalk_show_base_v14_two_candidate_selection_audit_v3"
SCHEDULE_FORMAT = "semtalk_show_base_fresh_lineage_schedule_v14_v1"
SCHEDULE_SHA256 = (
    "1c87b74907d8e69f3713c659d5448a8f3355b92599fb83660e54d4896a0c48db"
)
TOPOLOGY_SOURCE = "sealed_v14_two_topology_12_validation_fgd_min_tuple_v1"
CANDIDATE_EPOCHS = (1, 2, 4, 8, 16, 32)
MODE_P1 = "validation_gated_w8_l256_g2048_empirical_acceleration"
MODE_P2 = "validation_gated_w8_l128_g1024_empirical_acceleration"
MODES = (MODE_P1, MODE_P2)
MODE_IDS = {
    MODE_P1: "quality_p1_w8g2048_worker",
    MODE_P2: "quality_p2_w8g1024_master",
}
SELECTION_TUPLE = (
    "validation_diffsheg_fgd_ascending",
    "epoch_ascending",
    "topology_mode_lexicographic_ascending",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_json_sha256(value: Any) -> str:
    try:
        data = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V14ContractError("value is not strict canonical JSON") from error
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise V14ContractError(f"{label} is not one lowercase SHA-256")
    return value


def _strict_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise V14ContractError(f"{label} is not strict JSON") from error
    if not isinstance(payload, dict):
        raise V14ContractError(f"{label} must contain one JSON object")
    return payload


def _read_regular(path: Path, label: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise V14ContractError(f"{label} path is not absolute/canonical")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise V14ContractError(f"could not safely open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise V14ContractError(f"{label} is not a single-link regular file")
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
            raise V14ContractError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or candidate.is_symlink():
        raise V14ContractError(f"{label} path is not canonical")
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        if stat.S_ISLNK(os.lstat(cursor).st_mode):
            raise V14ContractError(f"{label} path contains a symlink")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise V14ContractError(f"{label} size changed")
    return resolved, data


def _load_json(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_bytes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    digest = _require_sha256(expected_sha256, f"{label} expected SHA-256")
    resolved, data = _read_regular(path, label)
    if hashlib.sha256(data).hexdigest() != digest:
        raise V14ContractError(f"{label} file SHA-256 mismatch")
    if expected_bytes is not None and (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or len(data) != expected_bytes
    ):
        raise V14ContractError(f"{label} byte count mismatch")
    return _strict_json(data, label), {
        "path": str(resolved),
        "sha256": digest,
        "bytes": len(data),
    }


def _self_hash(payload: Mapping[str, Any], key: str, label: str) -> str:
    claimed = _require_sha256(payload.get(key), f"{label} self-hash")
    unsigned = dict(payload)
    unsigned.pop(key)
    if canonical_json_sha256(unsigned) != claimed:
        raise V14ContractError(f"{label} self-hash mismatch")
    return claimed


def validate_schedule_payload(payload: Mapping[str, Any]) -> None:
    training = payload.get("training")
    binding = payload.get("topology_selection_contract")
    if (
        payload.get("format") != SCHEDULE_FORMAT
        or payload.get("scope") != "SemTalk Base only"
        or payload.get("target_dataset") != "SHOW"
        or payload.get("target_speaker_scope") != "All"
        or not isinstance(training, dict)
        or training.get("total_epochs") != 400
        or training.get("topology_source") != TOPOLOGY_SOURCE
        or not isinstance(binding, dict)
        or binding.get("format")
        != "semtalk_show_base_v14_formal_selection_binding_v1"
        or binding.get("training_semantics_source_commit")
        != TRAINING_SOURCE_COMMIT
        or binding.get("training_semantics_source_tree") != TRAINING_SOURCE_TREE
        or binding.get("validation_source_commit") != VALIDATION_SOURCE_COMMIT
        or binding.get("validation_source_tree") != VALIDATION_SOURCE_TREE
        or binding.get("selection_protocol_sha256") != SELECTION_PROTOCOL_SHA256
        or binding.get("candidate_modes") != list(MODES)
        or binding.get("candidate_epochs") != list(CANDIDATE_EPOCHS)
        or binding.get("validation_measurements") != 12
        or binding.get("selection_tuple") != list(SELECTION_TUPLE)
        or binding.get("split") != "val"
        or binding.get("test_visible") is not False
        or binding.get("native_nine_mode_role")
        != "optional_audit_evidence_never_v14_decision_authority"
        or binding.get("fresh_selected_mode_throughput_gate_required") is not True
    ):
        raise V14ContractError("V14 schedule topology authority is not truthful")


def validate_protocol(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if expected_sha256 != SELECTION_PROTOCOL_SHA256:
        raise V14ContractError("V14 selection protocol SHA-256 changed")
    payload, artifact = _load_json(path, expected_sha256, "V14 selection protocol")
    scope = payload.get("scope")
    validation = payload.get("validation")
    decision = payload.get("decision")
    topologies = payload.get("candidate_topologies")
    if (
        payload.get("format") != PROTOCOL_FORMAT
        or payload.get("status") != "frozen_before_validation_results"
        or payload.get("candidate_epochs") != list(CANDIDATE_EPOCHS)
        or payload.get("selection_tuple") != list(SELECTION_TUPLE)
        or not isinstance(scope, dict)
        or scope.get("repository") != ORIGIN
        or scope.get("source_commit") != TRAINING_SOURCE_COMMIT
        or scope.get("source_tree") != TRAINING_SOURCE_TREE
        or scope.get("dataset") != "SHOW"
        or scope.get("speaker_scope") != "All"
        or scope.get("speakers") != ["oliver", "chemistry", "seth", "conan"]
        or scope.get("base_motion_only") is not True
        or scope.get("semgate") is not False
        or scope.get("sparse_motion_generation") is not False
        or not isinstance(validation, dict)
        or validation.get("expected_reports") != 12
        or validation.get("primary_metric") != "DiffSHEG FGD"
        or validation.get("metric_direction") != "minimize"
        or validation.get("split") != "val"
        or validation.get("test_visible") is not False
        or validation.get("test_measurements_authorized") != 0
        or validation.get("require_all_reports_complete") is not True
        or validation.get("require_all_metrics_finite") is not True
        or not isinstance(decision, dict)
        or decision.get("formal_target_epochs") != 400
        or decision.get("select_exactly_one_topology") is not True
        or decision.get("formal_checkpoint_selection_split") != "val"
        or decision.get("final_test_runs_after_formal_selection") != 1
        or not isinstance(topologies, list)
        or [row.get("mode") for row in topologies if isinstance(row, dict)]
        != list(MODES)
    ):
        raise V14ContractError("V14 selection protocol semantics changed")
    return payload, artifact


def _validated_quality_reports(
    audit: Mapping[str, Any],
    *,
    selector: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifacts = audit.get("quality_reports")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != 2
        or [row.get("mode") for row in artifacts if isinstance(row, dict)]
        != sorted(MODES)
    ):
        raise V14ContractError("V14 audit must bind exactly two sorted quality reports")
    validated: list[dict[str, Any]] = []
    normalized_artifacts: list[dict[str, Any]] = []
    for row in artifacts:
        if not isinstance(row, dict) or set(row) != {
            "mode", "path", "sha256", "bytes", "payload_sha256"
        }:
            raise V14ContractError("V14 quality-report artifact schema changed")
        mode = row["mode"]
        if mode not in MODES:
            raise V14ContractError("unexpected V14 quality topology")
        raw, artifact = _load_json(
            Path(str(row["path"])),
            _require_sha256(row["sha256"], f"{mode} report SHA-256"),
            f"{mode} V14 quality report",
            expected_bytes=row["bytes"],
        )
        if _self_hash(raw, "receipt_sha256", f"{mode} quality report") != row["payload_sha256"]:
            raise V14ContractError(f"{mode} quality-report payload SHA-256 changed")
        try:
            replay = selector.validate_quality_report(
                mode,
                Path(artifact["path"]),
                artifact["sha256"],
                quality_gate_spec_sha256=QUALITY_GATE_SHA256,
                topology_gate_spec_sha256=TOPOLOGY_GATE_SHA256,
            )
        except Exception as error:
            raise V14ContractError(f"{mode} quality report fresh replay failed") from error
        if (
            replay.get("mode") != mode
            or replay.get("quality_role") != "candidate_quality"
            or replay.get("reference_only") is not False
            or replay.get("report_sha256") != artifact["sha256"]
            or list(replay.get("candidate_fgd", {}))
            != [str(epoch) for epoch in CANDIDATE_EPOCHS]
        ):
            raise V14ContractError(f"{mode} quality replay coverage changed")
        normalized_artifacts.append(
            {
                "mode": mode,
                **artifact,
                "payload_sha256": row["payload_sha256"],
            }
        )
        validated.append(replay)
    return validated, normalized_artifacts


def _training_source_from_frozen(
    frozen: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    source = frozen.get("source")
    if (
        not isinstance(source, dict)
        or source.get("origin") != ORIGIN
        or source.get("commit") != TRAINING_SOURCE_COMMIT
        or source.get("tree") != TRAINING_SOURCE_TREE
        or source.get("clean") is not True
        or frozen.get("run_purpose") != "topology_short_quality"
        or frozen.get("target_epochs") != list(CANDIDATE_EPOCHS)
    ):
        raise V14ContractError(f"{mode} quality training semantics are not frozen 5b84075")
    return source


def _quality_execution_identity(
    frozen: Mapping[str, Any], mode: str
) -> tuple[str, int]:
    protocol = frozen.get("protocol")
    distributed = (
        protocol.get("distributed_topology")
        if isinstance(protocol, dict)
        else None
    )
    if (
        not isinstance(distributed, dict)
        or distributed.get("mode") != mode
        or re.fullmatch(
            r"[A-Za-z0-9._-]{8,128}",
            str(distributed.get("formal_run_id", "")),
        )
        is None
        or type(distributed.get("master_port")) is not int
        or not 1024 <= distributed["master_port"] <= 65535
    ):
        raise V14ContractError(
            f"{mode} short-quality execution identity changed"
        )
    return distributed["formal_run_id"], distributed["master_port"]


def _validate_native_audit_claim(
    claim: Any,
    *,
    selector: Any,
    v14_selector: Any,
    training_contract: Any,
    topology_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    common = {
        "role": "audit_only",
        "authoritative_for_winner": False,
    }
    if not isinstance(claim, dict):
        raise V14ContractError("V14 audit native status is absent")
    if claim.get("status") == "available_not_authoritative":
        if (
            set(claim)
            != {
                *common,
                "status",
                "path",
                "format",
                "receipt_sha256",
                "selected_mode",
            }
            or claim.get("role") != common["role"]
            or claim.get("authoritative_for_winner") is not False
            or claim.get("format")
            != "semtalk_show_base_topology_selection_v2"
            or not isinstance(claim.get("path"), str)
            or not Path(claim["path"]).is_absolute()
            or ".." in Path(claim["path"]).parts
            or re.fullmatch(r"[0-9a-f]{64}", str(claim.get("receipt_sha256")))
            is None
            or claim.get("selected_mode") not in topology_specs
        ):
            raise V14ContractError("V14 available native audit schema changed")
        try:
            native_path, native_bytes = _read_regular(
                Path(claim["path"]), "available native nine-mode audit"
            )
            native, artifact = v14_selector._validate_trainer_native_selection(
                native_path,
                hashlib.sha256(native_bytes).hexdigest(),
                selector=selector,
                contract=training_contract,
            )
        except Exception as error:
            raise V14ContractError(
                "V14 available native audit replay failed"
            ) from error
        selected = native.get("selected")
        if claim != {
            **common,
            "status": "available_not_authoritative",
            "path": artifact["path"],
            "format": native.get("format"),
            "receipt_sha256": native.get("receipt_sha256"),
            "selected_mode": (
                selected.get("mode")
                if isinstance(selected, dict)
                else None
            ),
        }:
            raise V14ContractError("V14 available native audit changed")
        return dict(claim)
    if claim.get("status") != "unavailable_not_authoritative":
        raise V14ContractError("V14 native audit status changed")
    unavailable_keys = {
        *common,
        "status",
        "reason_code",
        "evidence_receipt",
        "probe_modes",
        "blocking_mode",
    }
    if (
        set(claim) != unavailable_keys
        or claim.get("role") != common["role"]
        or claim.get("authoritative_for_winner") is not False
    ):
        raise V14ContractError("V14 unavailable native audit schema changed")
    evidence = claim.get("evidence_receipt")
    if evidence is None:
        if claim != {
            **common,
            "status": "unavailable_not_authoritative",
            "reason_code": v14_selector.TRAINER_NATIVE_NOT_PROVIDED_REASON,
            "evidence_receipt": None,
            "probe_modes": [],
            "blocking_mode": None,
        }:
            raise V14ContractError("V14 no-native audit declaration changed")
        return dict(claim)
    if (
        claim.get("reason_code")
        != v14_selector.TRAINER_NATIVE_UNAVAILABLE_REASON
        or not isinstance(evidence, dict)
        or set(evidence) != {"path", "sha256", "bytes", "receipt_sha256"}
        or claim.get("probe_modes") != list(topology_specs)
        or claim.get("blocking_mode")
        != training_contract.OFFICIAL_W1_REFERENCE_MODE
    ):
        raise V14ContractError("V14 native-unavailable evidence schema changed")
    try:
        replay, _artifact = (
            v14_selector._validate_trainer_native_unavailable_receipt(
                Path(evidence["path"]),
                evidence["sha256"],
                selector=selector,
                contract=training_contract,
            )
        )
    except Exception as error:
        raise V14ContractError(
            "V14 native-unavailable evidence replay failed"
        ) from error
    if replay != claim:
        raise V14ContractError("V14 native-unavailable evidence changed")
    return dict(claim)


def _gate_source_and_identity(
    gate_path: Path,
    gate_sha256: str,
    *,
    selected_mode: str,
) -> tuple[dict[str, Any], str, int]:
    report, _artifact = _load_json(gate_path, gate_sha256, "fresh selected-mode throughput gate")
    if (
        report.get("status") != "pass"
        or report.get("topology_mode") != selected_mode
        or report.get("optimizer_updates") != 70
        or report.get("warmup_updates") != 20
        or report.get("timed_updates") != 50
    ):
        raise V14ContractError("fresh throughput gate protocol/mode changed")
    _self_hash(report, "receipt_sha256", "fresh selected-mode throughput gate")
    frozen_path = Path(_artifact["path"]).parent / "frozen_inputs.json"
    _frozen_resolved, frozen_bytes = _read_regular(
        frozen_path, "fresh throughput frozen inputs"
    )
    frozen = _strict_json(frozen_bytes, "fresh throughput frozen inputs")
    if _self_hash(
        frozen, "receipt_sha256", "fresh throughput frozen inputs"
    ) != report.get("frozen_receipt_sha256"):
        raise V14ContractError("fresh gate does not bind its frozen inputs")
    source = frozen.get("source")
    protocol = frozen.get("protocol")
    distributed = protocol.get("distributed_topology") if isinstance(protocol, dict) else None
    clones = source.get("node_local_clones") if isinstance(source, dict) else None
    if (
        frozen.get("run_purpose") != "topology_throughput_gate"
        or frozen.get("target_epochs") != []
        or not isinstance(source, dict)
        or source.get("origin") != ORIGIN
        or source.get("clean") is not True
        or not isinstance(clones, list)
        or not clones
        or any(not isinstance(row, dict) or row.get("branch") is not None for row in clones)
        or not isinstance(distributed, dict)
        or distributed.get("mode") != selected_mode
        or not isinstance(distributed.get("formal_run_id"), str)
        or type(distributed.get("master_port")) is not int
    ):
        raise V14ContractError("fresh gate source/execution identity is invalid")
    return source, distributed["formal_run_id"], distributed["master_port"]


def _git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise V14ContractError(f"Git authority query failed: {' '.join(args)}")
    return result


def current_project_authority(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    origin = _git(root, "remote", "get-url", "origin").stdout.strip()
    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    branches = _git(root, "for-each-ref", "--format=%(refname)", "refs/heads").stdout.splitlines()
    symbolic = _git(root, "symbolic-ref", "-q", "HEAD", check=False)
    ancestor = _git(
        root,
        "merge-base",
        "--is-ancestor",
        VALIDATION_SOURCE_COMMIT,
        commit,
        check=False,
    )
    if (
        origin != ORIGIN
        or status != ""
        or branches != []
        or symbolic.returncode != 1
        or symbolic.stdout.strip()
        or ancestor.returncode != 0
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", tree) is None
    ):
        raise V14ContractError(
            "formal V14 project must be clean, detached, branchless, and "
            "descend from 4066f20"
        )
    return {
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
    }


def validate_control_plane(
    *,
    project_root: Path,
    schedule_path: Path,
    expected_schedule_sha256: str,
    protocol_path: Path,
    expected_protocol_sha256: str,
    audit_path: Path,
    expected_audit_sha256: str,
    throughput_gate_path: Path,
    expected_throughput_gate_sha256: str,
    topology_mode: str,
    formal_run_id: str,
    formal_master_port: int,
    selector: Any,
    v14_selector: Any,
    training_contract: Any,
    topology_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if topology_mode not in MODES or topology_mode not in topology_specs:
        raise V14ContractError("formal V14 topology must be one of the two measured modes")
    if expected_schedule_sha256 != SCHEDULE_SHA256:
        raise V14ContractError("formal V14 schedule SHA-256 changed")
    schedule, schedule_artifact = _load_json(
        schedule_path,
        expected_schedule_sha256,
        "V14 formal schedule",
    )
    validate_schedule_payload(schedule)
    _protocol, protocol_artifact = validate_protocol(protocol_path, expected_protocol_sha256)
    audit, audit_artifact = _load_json(audit_path, expected_audit_sha256, "V14 selection audit")
    expected_audit_keys = {
        "format", "status", "selection_protocol", "selection_tuple",
        "validation_split", "test_visible", "test_measurements_authorized",
        "training_source_authority", "validation_source_authority",
        "validation_pipeline_source", "quality_reports", "quality_report_sha256",
        "topology_independent_input_sha256", "common_authority", "results",
        "winner", "trainer_native_selection", "formal_training", "receipt_sha256",
    }
    if (
        set(audit) != expected_audit_keys
        or audit.get("format") != AUDIT_FORMAT
        or audit.get("status") != "complete"
    ):
        raise V14ContractError("V14 selection audit schema/status changed")
    _self_hash(audit, "receipt_sha256", "V14 selection audit")
    if (
        audit.get("selection_protocol") != protocol_artifact
        or audit.get("selection_tuple") != list(SELECTION_TUPLE)
        or audit.get("validation_split") != "val"
        or audit.get("test_visible") is not False
        or audit.get("test_measurements_authorized") != 0
        or audit.get("training_source_authority")
        != {"origin": ORIGIN, "commit": TRAINING_SOURCE_COMMIT, "tree": TRAINING_SOURCE_TREE}
    ):
        raise V14ContractError("V14 audit protocol/training authority changed")
    validated, normalized_artifacts = _validated_quality_reports(audit, selector=selector)
    if normalized_artifacts != audit["quality_reports"]:
        raise V14ContractError("V14 quality artifact projection changed")
    try:
        frozen_by_mode = {
            row["mode"]: v14_selector._load_mode_frozen(
                row, row["mode"]
            )
            for row in validated
        }
        projections = {
            mode: v14_selector._authority_projection(
                frozen,
                mode,
                training_contract,
            )
            for mode, frozen in frozen_by_mode.items()
        }
        validation_sources = [
            v14_selector._validation_pipeline_authority(
                row, row["mode"]
            )
            for row in validated
        ]
    except Exception as error:
        raise V14ContractError(
            "V14 common source/data/initialization/runtime authority replay failed"
        ) from error
    training_sources = [
        _training_source_from_frozen(frozen_by_mode[mode], mode)
        for mode in MODES
    ]
    quality_execution_identities = {
        _quality_execution_identity(frozen_by_mode[mode], mode)
        for mode in MODES
    }
    for source in training_sources:
        if (
            source.get("origin") != ORIGIN
            or source.get("commit") != TRAINING_SOURCE_COMMIT
            or source.get("tree") != TRAINING_SOURCE_TREE
            or source.get("clean") is not True
        ):
            raise V14ContractError(
                "V14 quality reports do not share frozen training semantics"
            )
    validation_semantics = [
        {
            key: source.get(key)
            for key in (
                "origin",
                "commit",
                "tree",
                "clean",
                "detached",
                "local_branches_at_commit",
            )
        }
        for source in validation_sources
    ]
    if validation_semantics[0] != validation_semantics[1]:
        raise V14ContractError(
            "V14 quality reports do not share validation source semantics"
        )
    validation_authority = audit.get("validation_source_authority")
    quality_report_sha256 = audit.get("quality_report_sha256")
    topology_independent = {
        row.get("topology_independent_input_sha256") for row in validated
    }
    common_authority = audit.get("common_authority")
    expected_validation_authority = {
        "origin": ORIGIN,
        "commit": VALIDATION_SOURCE_COMMIT,
        "tree": VALIDATION_SOURCE_TREE,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "selector_sha256": v14_selector.OFFICIAL_SELECTOR_SHA256,
        "training_contract_sha256": (
            v14_selector.OFFICIAL_TRAIN_CONTRACT_SHA256
        ),
        "validation_contract_sha256": (
            v14_selector.OFFICIAL_VALIDATION_CONTRACT_SHA256
        ),
    }
    if (
        validation_authority != expected_validation_authority
        or audit.get("validation_pipeline_source") != validation_sources[0]
        or quality_report_sha256
        != {row["mode"]: row["report_sha256"] for row in validated}
        or len(topology_independent) != 1
        or audit.get("topology_independent_input_sha256")
        != next(iter(topology_independent))
        or projections.get(MODE_P1) != projections.get(MODE_P2)
        or common_authority != projections.get(MODE_P1)
    ):
        raise V14ContractError("V14 audit validation source authority changed")

    validated_by_mode = {row["mode"]: row for row in validated}
    expected_rows: list[dict[str, Any]] = []
    for mode in MODES:
        replay = validated_by_mode[mode]
        for candidate in replay["candidates"]:
            epoch = candidate["epoch"]
            value = candidate["diffsheg_fgd"]
            checkpoint = candidate["candidate_checkpoint"]
            if (
                epoch not in CANDIDATE_EPOCHS
                or type(value) not in {int, float}
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or not isinstance(checkpoint, dict)
            ):
                raise V14ContractError("V14 replay returned an invalid FGD row")
            expected_rows.append({
                "topology_id": MODE_IDS[mode],
                "topology_mode": mode,
                "epoch": epoch,
                "validation_diffsheg_fgd": float(value),
                "checkpoint_sha256": _require_sha256(
                    checkpoint.get("sha256"),
                    "candidate checkpoint SHA-256",
                ),
            })
    if audit.get("results") != expected_rows or len(expected_rows) != 12:
        raise V14ContractError("V14 audit does not exactly match twelve replayed FGD rows")
    ordered = sorted(
        expected_rows,
        key=lambda row: (
            row["validation_diffsheg_fgd"],
            row["epoch"],
            row["topology_mode"],
        ),
    )
    winner = ordered[0]
    key = (winner["validation_diffsheg_fgd"], winner["epoch"], winner["topology_mode"])
    if (
        sum(
            (
                row["validation_diffsheg_fgd"],
                row["epoch"],
                row["topology_mode"],
            )
            == key
            for row in expected_rows
        )
        != 1
        or audit.get("winner") != winner
    ):
        raise V14ContractError("V14 winner is not the unique minimum FGD/epoch/mode tuple")
    if winner["topology_mode"] != topology_mode:
        raise V14ContractError("requested formal topology is not the V14 winner")

    native_claim = audit.get("trainer_native_selection")
    normalized_native_claim = _validate_native_audit_claim(
        native_claim,
        selector=selector,
        v14_selector=v14_selector,
        training_contract=training_contract,
        topology_specs=topology_specs,
    )

    formal = audit.get("formal_training")
    if formal != {
        "target_epochs": 400,
        "fresh": True,
        "restart_from_same_frozen_initialization": True,
        "selected_quality_checkpoint_is_evidence_only": True,
        "checkpoint_selection_split": "val",
        "final_test_runs_after_formal_selection": 1,
    }:
        raise V14ContractError("V14 formal-training decision semantics changed")

    project = current_project_authority(project_root)
    try:
        gate_replay = selector.validate_probe(
            topology_mode,
            throughput_gate_path,
            expected_throughput_gate_sha256,
            gate_spec_sha256=TOPOLOGY_GATE_SHA256,
        )
    except Exception as error:
        raise V14ContractError(
            "fresh selected-mode throughput gate replay failed"
        ) from error
    if (
        gate_replay.get("mode") != topology_mode
        or gate_replay.get("status") != "pass"
        or gate_replay.get("report_sha256")
        != expected_throughput_gate_sha256
    ):
        raise V14ContractError(
            "fresh selected-mode throughput gate replay changed"
        )
    gate_source, gate_run_id, gate_port = _gate_source_and_identity(
        throughput_gate_path,
        expected_throughput_gate_sha256,
        selected_mode=topology_mode,
    )
    if (
        gate_source.get("commit") != project["commit"]
        or gate_source.get("tree") != project["tree"]
        or project["commit"]
        in {TRAINING_SOURCE_COMMIT, VALIDATION_SOURCE_COMMIT}
        or gate_run_id == formal_run_id
        or gate_port == formal_master_port
        or gate_run_id
        in {run_id for run_id, _port in quality_execution_identities}
        or formal_run_id
        in {run_id for run_id, _port in quality_execution_identities}
        or gate_port
        in {port for _run_id, port in quality_execution_identities}
        or formal_master_port
        in {port for _run_id, port in quality_execution_identities}
        or re.fullmatch(r"[A-Za-z0-9._-]{8,128}", formal_run_id) is None
        or type(formal_master_port) is not int
        or not 1024 <= formal_master_port <= 65535
    ):
        raise V14ContractError("fresh gate must use current source and a distinct run ID/port")
    return {
        "format": "semtalk_show_base_v14_formal_control_plane_receipt_v1",
        "status": "pass",
        "selected_mode": topology_mode,
        "winner_epoch": winner["epoch"],
        "winner_validation_diffsheg_fgd": winner["validation_diffsheg_fgd"],
        "schedule": schedule_artifact,
        "selection_protocol": protocol_artifact,
        "selection_audit": audit_artifact,
        "native_nine_mode_audit": normalized_native_claim,
        "fresh_throughput_gate": {
            "path": str(Path(throughput_gate_path).resolve(strict=True)),
            "sha256": expected_throughput_gate_sha256,
            "formal_run_id_distinct": True,
            "formal_master_port_distinct": True,
        },
        "training_semantics_source": {
            "commit": TRAINING_SOURCE_COMMIT,
            "tree": TRAINING_SOURCE_TREE,
        },
        "validation_source": {
            "commit": VALIDATION_SOURCE_COMMIT,
            "tree": VALIDATION_SOURCE_TREE,
        },
        "formal_control_plane_source": project,
        "native_nine_mode_is_not_v14_authority": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--schedule-json", type=Path, required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--selection-protocol", type=Path, required=True)
    parser.add_argument("--expected-selection-protocol-sha256", required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--expected-selection-audit-sha256", required=True)
    parser.add_argument("--throughput-gate-report", type=Path, required=True)
    parser.add_argument("--expected-throughput-gate-sha256", required=True)
    parser.add_argument("--topology-mode", choices=MODES, required=True)
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument("--formal-master-port", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from scripts.show_base import select_base_training_topology as selector
        from scripts.show_base import select_base_v14_two_candidate as v14_selector
        from scripts.show_base import train_base_official_adapt_long as training

        receipt = validate_control_plane(
            project_root=args.project_root,
            schedule_path=args.schedule_json,
            expected_schedule_sha256=args.expected_schedule_sha256,
            protocol_path=args.selection_protocol,
            expected_protocol_sha256=args.expected_selection_protocol_sha256,
            audit_path=args.selection_audit,
            expected_audit_sha256=args.expected_selection_audit_sha256,
            throughput_gate_path=args.throughput_gate_report,
            expected_throughput_gate_sha256=args.expected_throughput_gate_sha256,
            topology_mode=args.topology_mode,
            formal_run_id=args.formal_run_id,
            formal_master_port=args.formal_master_port,
            selector=selector,
            v14_selector=v14_selector,
            training_contract=training,
            topology_specs=training.TOPOLOGY_SPECS,
        )
    except (V14ContractError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
