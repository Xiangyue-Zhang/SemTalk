#!/usr/bin/env python3
"""Fail-closed two-host dispatcher for fresh SHOW Base validation candidates.

The dispatcher is deliberately subordinate to the frozen live-validation
campaign.  It does not invent a second queue or a second producer authority:
all candidate claims, work authorities, authorizations, measurements, and
completion receipts use the contracts in
``supervise_base_v14_live_validation.py``.  A persistent dispatcher claim
binds one master and one worker topology.  Each invocation may consume one
create-new two-candidate wave (one guarded runner per host), except for the
final odd candidate which is run on the master alone.

There is no resume or retry path.  A wave lock, job claim, endpoint ticket, or
launch record left without a committed wave is terminal evidence requiring
manual audit.  In particular, the dispatcher never deletes failure evidence
or republishes a candidate completion.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

sys.dont_write_bytecode = True
_ENDPOINT_PATH = (
    Path(__file__).resolve(strict=True).with_name("base_live_val_remote_endpoint.py")
)
_ENDPOINT_SPEC = importlib.util.spec_from_file_location(
    "_semtalk_frozen_remote_endpoint", _ENDPOINT_PATH
)
if _ENDPOINT_SPEC is None or _ENDPOINT_SPEC.loader is None:
    raise RuntimeError("cannot load the frozen sibling remote endpoint")
endpoint = importlib.util.module_from_spec(_ENDPOINT_SPEC)
_ENDPOINT_SPEC.loader.exec_module(endpoint)
supervisor = endpoint.supervisor

TOPOLOGY_FORMAT = "semtalk_show_base_live_val_dual_host_topology_v1"
DISPATCHER_CLAIM_FORMAT = "semtalk_show_base_live_val_dual_host_claim_v1"
WAVE_FORMAT = "semtalk_show_base_live_val_dual_host_wave_v1"
ACTIVE_LOCK_FORMAT = "semtalk_show_base_live_val_dual_host_active_lock_v1"
LAUNCH_FORMAT = "semtalk_show_base_live_val_dual_host_launch_v1"
VALIDATION_FORMAT = "semtalk_show_base_live_val_dual_host_validation_v1"
COMMIT_FORMAT = "semtalk_show_base_live_val_dual_host_commit_v1"

DUAL_DIR_NAME = "dual_dispatch"
TOPOLOGY_NAME = "topology.json"
CLAIM_NAME = "claim.json"
ACTIVE_LOCK_NAME = "active-wave.lock.json"
WAVES_DIR_NAME = "waves"
SSH_BINARY = "/usr/bin/ssh"
SSH_OPTIONS = (
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ClearAllForwardings=yes",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "LogLevel=ERROR",
    "-o", "RequestTTY=no",
    "--",
)
SAFE_REMOTE_TOKEN = re.compile(r"[A-Za-z0-9_./:=+,-]+\Z")
SAFE_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
ROLE_KEYS = frozenset({
    "role", "hostname", "machine_id", "host_identity", "source_root", "source_origin",
    "source_commit", "source_tree", "formal_python", "guarded_runner",
    "guard_verifier",
})
WORKER_ROLE_KEYS = ROLE_KEYS | frozenset({"ssh_host"})
TOPOLOGY_KEYS = frozenset({
    "format", "status", "campaign", "master", "worker", "ssh_binary",
    "ssh_options", "endpoint", "dispatcher", "created_unix",
    "topology_payload_sha256",
})
DISPATCHER_CLAIM_KEYS = frozenset({
    "format", "status", "campaign", "topology", "fresh_candidate_epochs",
    "waves", "policy", "created_unix", "claim_payload_sha256",
})
WAVE_KEYS = frozenset({
    "format", "status", "campaign", "dispatcher_claim", "topology",
    "wave_index", "candidate_epochs", "master_epoch", "worker_epoch",
    "supervisor_active_claim", "created_unix", "wave_payload_sha256",
})
ACTIVE_LOCK_KEYS = frozenset({
    "format", "status", "campaign", "dispatcher_claim", "topology", "wave",
    "created_unix", "lock_payload_sha256",
})
LAUNCH_KEYS = frozenset({
    "format", "status", "campaign", "dispatcher_claim", "topology", "wave",
    "active_lock", "candidate_epochs", "job_claims", "authorizations",
    "master_runner_argv", "master_runner_argv_sha256", "worker_ssh_argv",
    "worker_ssh_argv_sha256", "worker_ticket", "created_unix",
    "launch_payload_sha256",
})
VALIDATION_KEYS = frozenset({
    "format", "status", "campaign", "dispatcher_claim", "topology", "wave",
    "launch", "candidate_epochs", "runner_statuses", "runner_logs",
    "measurements", "bridge_replay_argvs", "guard_proofs",
    "worker_endpoint_receipt", "prepared_completions", "validated_unix",
    "validation_payload_sha256",
})
COMMIT_KEYS = frozenset({
    "format", "status", "campaign", "dispatcher_claim", "topology", "wave",
    "launch", "validation", "candidate_epochs", "job_claims",
    "authorizations", "worker_ticket", "worker_endpoint_receipt",
    "prepared_completions", "completions", "completed_unix",
    "commit_payload_sha256",
})
REMOTE_RECEIPT_KEYS = frozenset({
    "format", "status", "role", "hostname", "machine_id", "host_identity",
    "candidate_epoch", "ticket", "campaign", "wave", "active_claim",
    "job_claim", "authorization", "formal_python", "source_root",
    "source_origin", "source_commit", "source_tree", "guarded_runner",
    "guard_verifier", "runner_argv", "runner_argv_sha256", "runner_status",
    "runner_log", "guard_verifier_argv", "guard_verifier_stdout",
    "restored_guards", "completed_unix", "receipt_payload_sha256",
})
CENTRAL_TICKET_KEYS = endpoint.TICKET_KEYS | frozenset({"topology", "endpoint"})


class DualDispatchError(RuntimeError):
    """The dual-host authority or execution evidence is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DualDispatchError(message)


def _finite(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result) and result > 0, f"{label} is invalid")
    return result


def _add_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    require(field not in result, f"{field} already exists")
    result[field] = supervisor.payload_sha256(result)
    return result


def _validate_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    claimed = value.get(field)
    require(
        isinstance(claimed, str) and supervisor.HEX64.fullmatch(claimed) is not None,
        f"{label} self-hash is invalid",
    )
    body = dict(value)
    del body[field]
    require(supervisor.payload_sha256(body) == claimed, f"{label} self-hash changed")


def _artifact_from_path(path: Any, label: str, *, executable: bool = False) -> dict[str, Any]:
    return supervisor._artifact_from_path(path, label, executable=executable)


def _read_json_artifact(path: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = supervisor._read_existing_json(Path(path), label)
    return artifact, value


def fresh_waves() -> list[list[int]]:
    fresh = list(supervisor.CANDIDATE_EPOCHS[1:])
    return [fresh[index:index + 2] for index in range(0, len(fresh), 2)]


def _dual_root(campaign: Mapping[str, Any]) -> Path:
    return Path(campaign["_state_root"]) / DUAL_DIR_NAME


def _topology_path(campaign: Mapping[str, Any]) -> Path:
    return _dual_root(campaign) / TOPOLOGY_NAME


def _claim_path(campaign: Mapping[str, Any]) -> Path:
    return _dual_root(campaign) / CLAIM_NAME


def _active_lock_path(campaign: Mapping[str, Any]) -> Path:
    return _dual_root(campaign) / ACTIVE_LOCK_NAME


def _wave_dir(campaign: Mapping[str, Any], wave_index: int) -> Path:
    return _dual_root(campaign) / WAVES_DIR_NAME / ("wave-%04d" % wave_index)


def _machine_artifact(path: Any, label: str) -> dict[str, Any]:
    artifact = _artifact_from_path(path, label)
    _path, raw, _sha, _size = supervisor.safe_regular_bytes(
        artifact["path"], label, artifact["sha256"], artifact["bytes"]
    )
    require(endpoint.MACHINE_ID.fullmatch(raw) is not None, f"{label} changed")
    return artifact


def _worker_machine_artifact(path: Any, sha256: Any, size: Any) -> dict[str, Any]:
    require(path == endpoint.MACHINE_ID_PATH, "worker machine-id path changed")
    require(
        isinstance(sha256, str) and endpoint.HEX64.fullmatch(sha256) is not None,
        "worker machine-id SHA changed",
    )
    require(type(size) is int and 32 <= size <= 33, "worker machine-id size changed")
    return {"path": path, "sha256": sha256, "bytes": size}


def _host_identity_artifact(path: Any, label: str) -> dict[str, Any]:
    path_value, _raw, digest, size = endpoint.safe_host_identity_bytes(path, label)
    return {"path": str(path_value), "sha256": digest, "bytes": size}


def _worker_host_identity_artifact(
    path: Any, sha256: Any, size: Any,
) -> dict[str, Any]:
    require(path == endpoint.HOST_IDENTITY_PATH, "worker host identity path changed")
    require(
        isinstance(sha256, str) and endpoint.HEX64.fullmatch(sha256) is not None,
        "worker host identity SHA changed",
    )
    require(type(size) is int and 36 <= size <= 37, "worker host identity size changed")
    return {"path": path, "sha256": sha256, "bytes": size}


def _role_value(
    campaign: Mapping[str, Any], *, role: str, hostname: str,
    machine_id: Mapping[str, Any], host_identity: Mapping[str, Any],
    ssh_host: Optional[str] = None,
) -> dict[str, Any]:
    control = campaign["_control_source"]
    result: dict[str, Any] = {
        "role": role,
        "hostname": hostname,
        "machine_id": dict(machine_id),
        "host_identity": dict(host_identity),
        "source_root": control["root"],
        "source_origin": endpoint.OFFICIAL_ORIGIN,
        "source_commit": control["commit"],
        "source_tree": control["tree"],
        "formal_python": campaign["_formal_python"],
        "guarded_runner": campaign["_guarded_runner"],
        "guard_verifier": campaign["_guard_verifier"],
    }
    if ssh_host is not None:
        result["ssh_host"] = ssh_host
    return result


def _require_distinct_hosts(
    master: Mapping[str, Any], worker: Mapping[str, Any],
) -> None:
    """Reject aliases that could place both all-GPU runners on one machine."""

    require(
        master.get("hostname") != worker.get("hostname"),
        "master and worker hostnames must identify different machines",
    )
    master_identity = _artifact_descriptor(
        master.get("host_identity"), "master host identity"
    )
    worker_identity = _artifact_descriptor(
        worker.get("host_identity"), "worker host identity"
    )
    require(
        master_identity["sha256"] != worker_identity["sha256"],
        "master and worker host identity fingerprints must differ",
    )


def freeze_topology(
    campaign: Mapping[str, Any], *, output: Any, worker_ssh_host: str,
    worker_hostname: str, worker_machine_id_sha256: str,
    worker_machine_id_bytes: int, worker_host_identity_sha256: str,
    worker_host_identity_bytes: int,
    clock: Callable[[], float] = time.time,
    hostname_provider: Callable[[], str] = socket.gethostname,
) -> dict[str, Any]:
    """Freeze a create-new topology; the worker replays its live half later."""

    require(campaign.get("_adopted_e1") is not None, "dual dispatch requires adopted e1")
    require(SAFE_HOST.fullmatch(worker_ssh_host) is not None, "worker SSH host changed")
    require(SAFE_HOST.fullmatch(worker_hostname) is not None, "worker hostname changed")
    expected_output = _topology_path(campaign)
    require(Path(output) == expected_output, "topology output path changed")

    master_hostname = hostname_provider()
    require(
        isinstance(master_hostname, str)
        and SAFE_HOST.fullmatch(master_hostname) is not None,
        "master hostname changed",
    )
    master_machine = _machine_artifact(endpoint.MACHINE_ID_PATH, "master machine-id")
    worker_machine = _worker_machine_artifact(
        endpoint.MACHINE_ID_PATH,
        worker_machine_id_sha256,
        worker_machine_id_bytes,
    )
    master_host_identity = _host_identity_artifact(
        endpoint.HOST_IDENTITY_PATH, "master host identity"
    )
    worker_host_identity = _worker_host_identity_artifact(
        endpoint.HOST_IDENTITY_PATH,
        worker_host_identity_sha256,
        worker_host_identity_bytes,
    )
    master_role = _role_value(
        campaign,
        role="master",
        hostname=master_hostname,
        machine_id=master_machine,
        host_identity=master_host_identity,
    )
    worker_role = _role_value(
        campaign,
        role="worker",
        hostname=worker_hostname,
        machine_id=worker_machine,
        host_identity=worker_host_identity,
        ssh_host=worker_ssh_host,
    )
    _require_distinct_hosts(master_role, worker_role)
    control_root = Path(campaign["_control_source"]["root"])
    endpoint_artifact = _artifact_from_path(
        control_root / "scripts/show_base/base_live_val_remote_endpoint.py",
        "remote endpoint",
    )
    dispatcher_artifact = _artifact_from_path(Path(__file__).resolve(strict=True), "dual dispatcher")
    require(
        dispatcher_artifact["path"]
        == str(control_root / "scripts/show_base/dispatch_base_v14_live_validation_dual_host.py"),
        "dispatcher path is outside the frozen control source",
    )
    ssh_binary = _artifact_from_path(SSH_BINARY, "SSH binary", executable=True)

    # Complete every read-only admission check before the first mutation.  A
    # bad topology request must not leave a half-frozen dual-dispatch root.
    dual = _dual_root(campaign)
    if os.path.lexists(dual):
        require(dual.is_dir() and not dual.is_symlink(), "dual-dispatch root changed")
        require(not any(dual.iterdir()), "dual-dispatch root is not create-new")
    else:
        os.mkdir(dual, 0o700)
    waves = dual / WAVES_DIR_NAME
    os.mkdir(waves, 0o700)
    body = _add_hash({
        "format": TOPOLOGY_FORMAT,
        "status": "frozen_before_execution",
        "campaign": campaign["_artifact"],
        "master": master_role,
        "worker": worker_role,
        "ssh_binary": ssh_binary,
        "ssh_options": list(SSH_OPTIONS),
        "endpoint": endpoint_artifact,
        "dispatcher": dispatcher_artifact,
        "created_unix": _finite(clock(), "topology creation time"),
    }, "topology_payload_sha256")
    artifact = supervisor.write_new_json(expected_output, body, "dual-host topology")
    loaded_artifact, _loaded = load_topology(campaign, artifact)
    require(supervisor.strict_json_equal(artifact, loaded_artifact), "topology changed after publication")
    # Claim the queue in the same CPU-only transaction.  This prevents a
    # topology-only intermediate state from being mistaken for permission to
    # use the legacy single-host scheduler.
    dispatcher_claim, _dispatcher_value = load_or_create_dispatcher_claim(
        campaign, artifact, clock=clock, allow_create=True
    )
    return {
        "status": "frozen",
        "topology": artifact,
        "dispatcher_claim": dispatcher_claim,
        "fresh_candidate_count": 21,
        "wave_count": len(fresh_waves()),
    }


def _artifact_descriptor(value: Any, label: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == ARTIFACT_KEYS, f"{label} schema changed")
    path = value.get("path")
    sha = value.get("sha256")
    size = value.get("bytes")
    require(isinstance(path, str) and Path(path).is_absolute(), f"{label} path changed")
    require(isinstance(sha, str) and endpoint.HEX64.fullmatch(sha) is not None, f"{label} SHA changed")
    require(type(size) is int and size > 0, f"{label} size changed")
    return dict(value)


def _validate_role(
    campaign: Mapping[str, Any], value: Any, *, role: str, live_master: bool,
    hostname_provider: Callable[[], str],
) -> dict[str, Any]:
    expected_keys = ROLE_KEYS if role == "master" else WORKER_ROLE_KEYS
    require(type(value) is dict and set(value) == expected_keys, f"{role} topology schema changed")
    require(value.get("role") == role, f"{role} role changed")
    hostname = value.get("hostname")
    require(isinstance(hostname, str) and SAFE_HOST.fullmatch(hostname) is not None, f"{role} hostname changed")
    if live_master:
        require(hostname == hostname_provider(), "live master hostname differs from topology")
    machine = _artifact_descriptor(value.get("machine_id"), f"{role} machine-id")
    host_identity = _artifact_descriptor(
        value.get("host_identity"), f"{role} host identity"
    )
    require(machine["path"] == endpoint.MACHINE_ID_PATH, f"{role} machine-id path changed")
    require(
        host_identity["path"] == endpoint.HOST_IDENTITY_PATH,
        f"{role} host identity path changed",
    )
    if live_master:
        endpoint.artifact(machine, "master machine-id")
        _machine_artifact(machine["path"], "master machine-id replay")
        require(
            endpoint.strict_equal(
                endpoint.host_identity_artifact(
                    host_identity, "master host identity replay"
                ),
                host_identity,
            ),
            "master host identity changed",
        )
    else:
        require(32 <= machine["bytes"] <= 33, "worker machine-id size changed")
        require(
            36 <= host_identity["bytes"] <= 37,
            "worker host identity size changed",
        )
        ssh_host = value.get("ssh_host")
        require(isinstance(ssh_host, str) and SAFE_HOST.fullmatch(ssh_host) is not None, "worker SSH host changed")

    control = campaign["_control_source"]
    require(
        value.get("source_root") == control["root"]
        and value.get("source_origin") == endpoint.OFFICIAL_ORIGIN
        and value.get("source_commit") == control["commit"]
        and value.get("source_tree") == control["tree"],
        f"{role} source binding changed",
    )
    require(
        endpoint.strict_equal(value.get("formal_python"), campaign["_formal_python"])
        and endpoint.strict_equal(value.get("guarded_runner"), campaign["_guarded_runner"])
        and endpoint.strict_equal(value.get("guard_verifier"), campaign["_guard_verifier"]),
        f"{role} runtime binding changed",
    )
    if live_master:
        endpoint.validate_source(control["root"], control["commit"], control["tree"])
        endpoint.validate_formal_python(value["formal_python"])
        endpoint.artifact(value["guarded_runner"], "master guarded runner", executable=True)
        endpoint.artifact(value["guard_verifier"], "master guard verifier")
    return dict(value)


def load_topology(
    campaign: Mapping[str, Any], topology_artifact: Mapping[str, Any], *,
    hostname_provider: Callable[[], str] = socket.gethostname,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _artifact_descriptor(topology_artifact, "topology")
    require(descriptor["path"] == str(_topology_path(campaign)), "topology path changed")
    artifact, value = _read_json_artifact(descriptor["path"], "dual-host topology")
    require(supervisor.strict_json_equal(artifact, descriptor), "topology artifact changed")
    require(set(value) == TOPOLOGY_KEYS, "topology schema changed")
    _validate_hash(value, "topology_payload_sha256", "topology")
    require(
        value.get("format") == TOPOLOGY_FORMAT
        and value.get("status") == "frozen_before_execution"
        and supervisor.strict_json_equal(value.get("campaign"), campaign["_artifact"]),
        "topology state/campaign changed",
    )
    master = _validate_role(
        campaign, value.get("master"), role="master", live_master=True,
        hostname_provider=hostname_provider,
    )
    worker = _validate_role(
        campaign, value.get("worker"), role="worker", live_master=False,
        hostname_provider=hostname_provider,
    )
    _require_distinct_hosts(master, worker)
    ssh_binary = endpoint.artifact(value.get("ssh_binary"), "SSH binary", executable=True)
    require(ssh_binary["path"] == SSH_BINARY, "SSH binary path changed")
    require(value.get("ssh_options") == list(SSH_OPTIONS), "SSH options changed")
    endpoint_artifact = endpoint.artifact(value.get("endpoint"), "remote endpoint")
    dispatcher_artifact = endpoint.artifact(value.get("dispatcher"), "dual dispatcher")
    control_root = Path(campaign["_control_source"]["root"])
    require(
        endpoint_artifact["path"]
        == str(control_root / "scripts/show_base/base_live_val_remote_endpoint.py")
        and dispatcher_artifact["path"]
        == str(control_root / "scripts/show_base/dispatch_base_v14_live_validation_dual_host.py"),
        "dispatcher/endpoint path changed",
    )
    for artifact_value, label in (
        (endpoint_artifact, "remote endpoint"),
        (dispatcher_artifact, "dual dispatcher"),
    ):
        relative = str(Path(artifact_value["path"]).relative_to(control_root))
        require(
            supervisor._git_stdout(
                control_root, "ls-files", "--error-unmatch", relative
            ) == relative,
            f"{label} is not tracked by the frozen source",
        )
    _finite(value.get("created_unix"), "topology creation time")
    return artifact, value


def _dispatcher_claim_body(
    campaign: Mapping[str, Any], topology: Mapping[str, Any], now: float,
) -> dict[str, Any]:
    return _add_hash({
        "format": DISPATCHER_CLAIM_FORMAT,
        "status": "claimed",
        "campaign": campaign["_artifact"],
        "topology": dict(topology),
        "fresh_candidate_epochs": list(supervisor.CANDIDATE_EPOCHS[1:]),
        "waves": fresh_waves(),
        "policy": {
            "adopted_candidates": 1,
            "fresh_candidates": 21,
            "paired_fresh_candidates": 20,
            "terminal_singleton_candidates": 1,
            "max_active_waves": 1,
            "retry_authorized": False,
            "rerun_authorized": False,
            "publish_completions_after_full_wave_validation": True,
        },
        "created_unix": _finite(now, "dispatcher claim time"),
    }, "claim_payload_sha256")


def load_or_create_dispatcher_claim(
    campaign: Mapping[str, Any], topology: Mapping[str, Any], *,
    clock: Callable[[], float] = time.time, allow_create: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _claim_path(campaign)
    if not os.path.lexists(path):
        require(
            allow_create,
            "persistent dispatcher claim is missing; replacement authority is not authorized",
        )
        value = _dispatcher_claim_body(campaign, topology, clock())
        artifact = supervisor.write_new_json(path, value, "dual dispatcher claim")
        return artifact, value
    artifact, value = _read_json_artifact(path, "dual dispatcher claim")
    require(set(value) == DISPATCHER_CLAIM_KEYS, "dispatcher claim schema changed")
    _validate_hash(value, "claim_payload_sha256", "dispatcher claim")
    expected = _dispatcher_claim_body(campaign, topology, value.get("created_unix"))
    require(supervisor.strict_json_equal(value, expected), "dispatcher claim changed")
    return artifact, value


def _scan_dual(
    campaign: Mapping[str, Any],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], Optional[dict[str, Any]]]:
    """Replay the supervisor queue while admitting only our private evidence."""

    allowed_epoch_names = {
        "epoch-%04d.json" % epoch for epoch in supervisor.CANDIDATE_EPOCHS
    }
    allowed_log_names = {
        "epoch-%04d.log" % epoch for epoch in supervisor.CANDIDATE_EPOCHS
    }
    for directory, allowed, label in (
        (campaign["_claims_dir"], allowed_epoch_names, "job claims"),
        (campaign["_completions_dir"], allowed_epoch_names, "completions"),
        (campaign["_runner_status_dir"], allowed_epoch_names, "runner statuses"),
        (campaign["_runner_logs_dir"], allowed_log_names, "runner logs"),
        (campaign["_authorities_dir"], allowed_epoch_names, "authorities"),
        (campaign["_authorizations_dir"], allowed_epoch_names, "authorizations"),
    ):
        names = {entry.name for entry in directory.iterdir()}
        require(names.issubset(allowed), f"{label} contains an unknown entry")
    state_allowed = {
        "campaign.json", "campaign.claim.json", "final_summary.json",
        "active_invocation.claim.json", "producer_reconcile.claim.json",
        "bridge_reconcile.claim.json", "reconciliation.json",
        "recovery-request.epoch-0001.json", "recovery-authority.epoch-0001.json",
        "job_claims", "completions", "runner_status", "runner_logs",
        "authorities", "authorizations", DUAL_DIR_NAME,
    }
    require(
        {entry.name for entry in campaign["_state_root"].iterdir()}.issubset(state_allowed),
        "state root contains an unknown entry",
    )
    completed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    head: Optional[dict[str, Any]] = None
    gap = False
    for job in campaign["_jobs"]:
        claim_exists = os.path.lexists(supervisor._job_claim_path_v2(campaign, job["epoch"]))
        completion_exists = os.path.lexists(supervisor._completion_path_v2(campaign, job["epoch"]))
        if job["epoch"] == 1 and campaign.get("_adopted_e1") is not None:
            require(
                not claim_exists
                and not completion_exists
                and not os.path.lexists(job["authority_path"])
                and not os.path.lexists(job["authorization_path"])
                and not os.path.lexists(job["run_root"])
                and not os.path.lexists(job["runner_status_path"])
                and not os.path.lexists(job["runner_log_path"])
                and not os.path.lexists(
                    campaign["_state_root"] / "recovery-request.epoch-0001.json"
                )
                and not os.path.lexists(
                    campaign["_state_root"] / "recovery-authority.epoch-0001.json"
                ),
                "adopted e1 was republished or has successor execution output",
            )
            completed.append(supervisor._load_adopted_e1(campaign))
            continue
        if completion_exists:
            require(claim_exists and not gap, "queue skipped ahead")
            completed.append(supervisor._load_completion_v2(campaign, job))
        elif claim_exists:
            raise DualDispatchError(
                f"e{job['epoch']} claim exists without completion; no retry is authorized"
            )
        else:
            for output in (
                job["authority_path"], job["authorization_path"], job["run_root"],
                job["runner_status_path"], job["runner_log_path"],
            ):
                require(not os.path.lexists(output), f"unclaimed e{job['epoch']} output exists")
            if head is None:
                head = dict(job)
                gap = True
        require(not (gap and completion_exists), "queue skipped ahead")
    if len(completed) < len(supervisor.CANDIDATE_EPOCHS):
        for path, label in (
            (campaign["summary_path"], "final summary"),
            (campaign["_reconciliation_path"], "producer reconciliation"),
            (campaign["_selection_root"], "selection root"),
            (campaign["_state_root"] / "producer_reconcile.claim.json", "producer reconcile claim"),
            (campaign["_state_root"] / "bridge_reconcile.claim.json", "bridge reconcile claim"),
        ):
            require(not os.path.lexists(path), f"{label} exists before all 22 completions")
    return completed, head


def _expected_wave(
    campaign: Mapping[str, Any], completed: Sequence[Any], head: Optional[Mapping[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    require(completed and len(completed) <= 22, "adopted/fresh completion count changed")
    fresh_done = len(completed) - 1
    require(0 <= fresh_done <= 21, "fresh completion count changed")
    if fresh_done == 21:
        require(head is None, "queue head exists after 21 fresh completions")
        return 12, []
    require(fresh_done % 2 == 0, "partial prior wave is terminal")
    expected_epochs = fresh_waves()[fresh_done // 2]
    require(head is not None and head.get("epoch") == expected_epochs[0], "queue head differs from wave plan")
    by_epoch = {job["epoch"]: job for job in campaign["_jobs"]}
    return fresh_done // 2 + 1, [dict(by_epoch[epoch]) for epoch in expected_epochs]


def _load_self_hashed_document(
    path: Path, *, keys: frozenset[str], field: str, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _read_json_artifact(path, label)
    require(set(value) == keys, f"{label} schema changed")
    _validate_hash(value, field, label)
    return artifact, value


def _validate_committed_wave(
    campaign: Mapping[str, Any], dispatcher_claim: Mapping[str, Any],
    topology: Mapping[str, Any], topology_value: Mapping[str, Any], wave_index: int,
) -> None:
    """Replay permanent wave evidence before admitting a later wave."""

    expected_epochs = fresh_waves()[wave_index - 1]
    directory = _wave_dir(campaign, wave_index)
    require(directory.is_dir() and not directory.is_symlink(), "prior wave directory changed")
    paired = len(expected_epochs) == 2
    expected_names = {
        "wave.claim.json", "active.claim.json", "launch.claim.json",
        "validation.json", "commit.json",
        *{
            "prepared-completion-epoch-%04d.json" % epoch
            for epoch in expected_epochs
        },
    }
    if paired:
        expected_names.update({"worker-ticket.json", "worker-endpoint-receipt.json"})
    require(
        {entry.name for entry in directory.iterdir()} == expected_names,
        "prior wave inventory changed",
    )
    wave, wave_value = _load_self_hashed_document(
        directory / "wave.claim.json", keys=WAVE_KEYS,
        field="wave_payload_sha256", label="prior wave claim",
    )
    require(
        wave_value.get("format") == WAVE_FORMAT
        and wave_value.get("status") == "active"
        and wave_value.get("wave_index") == wave_index
        and wave_value.get("candidate_epochs") == expected_epochs
        and wave_value.get("master_epoch") == expected_epochs[0]
        and wave_value.get("worker_epoch")
        == (expected_epochs[1] if paired else None)
        and supervisor.strict_json_equal(wave_value.get("campaign"), campaign["_artifact"])
        and supervisor.strict_json_equal(wave_value.get("dispatcher_claim"), dispatcher_claim)
        and supervisor.strict_json_equal(wave_value.get("topology"), topology),
        "prior wave authority changed",
    )
    permanent_active, active_value = _load_self_hashed_document(
        directory / "active.claim.json", keys=ACTIVE_LOCK_KEYS,
        field="lock_payload_sha256", label="prior permanent active claim",
    )
    require(
        active_value.get("format") == ACTIVE_LOCK_FORMAT
        and active_value.get("status") == "active"
        and supervisor.strict_json_equal(active_value.get("campaign"), campaign["_artifact"])
        and supervisor.strict_json_equal(active_value.get("dispatcher_claim"), dispatcher_claim)
        and supervisor.strict_json_equal(active_value.get("topology"), topology)
        and supervisor.strict_json_equal(active_value.get("wave"), wave),
        "prior active claim changed",
    )
    launch, launch_value = _load_self_hashed_document(
        directory / "launch.claim.json", keys=LAUNCH_KEYS,
        field="launch_payload_sha256", label="prior launch claim",
    )
    require(
        launch_value.get("format") == LAUNCH_FORMAT
        and launch_value.get("status") == "authorized"
        and launch_value.get("candidate_epochs") == expected_epochs
        and supervisor.strict_json_equal(launch_value.get("campaign"), campaign["_artifact"])
        and supervisor.strict_json_equal(launch_value.get("dispatcher_claim"), dispatcher_claim)
        and supervisor.strict_json_equal(launch_value.get("topology"), topology)
        and supervisor.strict_json_equal(launch_value.get("wave"), wave),
        "prior launch authority changed",
    )
    require(
        type(launch_value.get("master_runner_argv")) is list
        and _argv_sha(launch_value["master_runner_argv"])
        == launch_value.get("master_runner_argv_sha256")
        and type(launch_value.get("worker_ssh_argv")) is list
        and _argv_sha(launch_value["worker_ssh_argv"])
        == launch_value.get("worker_ssh_argv_sha256"),
        "prior launch argv digest changed",
    )
    retired_active = _artifact_descriptor(
        launch_value.get("active_lock"), "prior retired active lock"
    )
    require(
        retired_active["path"] == str(_active_lock_path(campaign))
        and retired_active["sha256"] == permanent_active["sha256"]
        and retired_active["bytes"] == permanent_active["bytes"],
        "prior launch active-lock binding changed",
    )
    validation, validation_value = _load_self_hashed_document(
        directory / "validation.json", keys=VALIDATION_KEYS,
        field="validation_payload_sha256", label="prior validation",
    )
    require(
        validation_value.get("format") == VALIDATION_FORMAT
        and validation_value.get("status") == "complete"
        and validation_value.get("candidate_epochs") == expected_epochs
        and supervisor.strict_json_equal(validation_value.get("campaign"), campaign["_artifact"])
        and supervisor.strict_json_equal(validation_value.get("dispatcher_claim"), dispatcher_claim)
        and supervisor.strict_json_equal(validation_value.get("topology"), topology)
        and supervisor.strict_json_equal(validation_value.get("wave"), wave)
        and supervisor.strict_json_equal(validation_value.get("launch"), launch),
        "prior validation authority changed",
    )
    commit, commit_value = _load_self_hashed_document(
        directory / "commit.json", keys=COMMIT_KEYS,
        field="commit_payload_sha256", label="prior wave commit",
    )
    del commit
    require(
        commit_value.get("format") == COMMIT_FORMAT
        and commit_value.get("status") == "complete"
        and commit_value.get("candidate_epochs") == expected_epochs
        and supervisor.strict_json_equal(commit_value.get("campaign"), campaign["_artifact"])
        and supervisor.strict_json_equal(commit_value.get("dispatcher_claim"), dispatcher_claim)
        and supervisor.strict_json_equal(commit_value.get("topology"), topology)
        and supervisor.strict_json_equal(commit_value.get("wave"), wave)
        and supervisor.strict_json_equal(commit_value.get("launch"), launch)
        and supervisor.strict_json_equal(commit_value.get("validation"), validation),
        "prior wave commit authority changed",
    )
    cardinality = len(expected_epochs)
    for label, values in (
        ("launch job claims", launch_value.get("job_claims")),
        ("launch authorizations", launch_value.get("authorizations")),
        ("validation runner statuses", validation_value.get("runner_statuses")),
        ("validation runner logs", validation_value.get("runner_logs")),
        ("validation measurements", validation_value.get("measurements")),
        ("validation bridge replay argvs", validation_value.get("bridge_replay_argvs")),
        ("validation guard proofs", validation_value.get("guard_proofs")),
        ("validation prepared completions", validation_value.get("prepared_completions")),
        ("commit job claims", commit_value.get("job_claims")),
        ("commit authorizations", commit_value.get("authorizations")),
        ("commit prepared completions", commit_value.get("prepared_completions")),
        ("commit completions", commit_value.get("completions")),
    ):
        require(
            type(values) is list and len(values) == cardinality,
            f"prior {label} cardinality changed",
        )

    by_epoch = {job["epoch"]: job for job in campaign["_jobs"]}
    dynamics: list[dict[str, Any]] = []
    expected_claims: list[dict[str, Any]] = []
    expected_authorizations: list[dict[str, Any]] = []
    expected_prepared: list[dict[str, Any]] = []
    expected_completions: list[dict[str, Any]] = []
    completion_values: list[dict[str, Any]] = []
    status_values: list[dict[str, Any]] = []
    for position, epoch in enumerate(expected_epochs):
        job = by_epoch[epoch]
        completion, completion_value = supervisor._load_completion_v2(campaign, job)
        candidate = _artifact_descriptor(
            completion_value.get("candidate_receipt"),
            f"prior e{epoch} candidate receipt",
        )
        authority = _artifact_descriptor(
            completion_value.get("work_authority"),
            f"prior e{epoch} work authority",
        )
        require(
            completion_value.get("consumer_recovery") is None,
            f"prior fresh e{epoch} unexpectedly contains recovery authority",
        )
        dynamic = {
            **job,
            "candidate_receipt": candidate,
            "work_authority": authority,
            "consumer_recovery": None,
            "authorization": completion_value["authorization"],
        }
        claim, _claim_value = supervisor._load_job_claim_v2(
            campaign, dynamic, candidate
        )
        authorization, _authorization_value = supervisor._load_authorization(
            campaign, dynamic, candidate, authority
        )
        require(
            supervisor.strict_json_equal(
                completion_value.get("authorization"), authorization
            ),
            f"prior e{epoch} completion authorization binding changed",
        )
        prepared_path = directory / (
            "prepared-completion-epoch-%04d.json" % epoch
        )
        prepared, prepared_value = _read_json_artifact(
            prepared_path, f"prior e{epoch} prepared completion"
        )
        require(
            supervisor.strict_json_equal(
                prepared, commit_value["prepared_completions"][position]
            )
            and supervisor.strict_json_equal(
                prepared, validation_value["prepared_completions"][position]
            )
            and prepared["sha256"] == completion["sha256"]
            and prepared["bytes"] == completion["bytes"]
            and supervisor.strict_json_equal(prepared_value, completion_value),
            f"prior e{epoch} prepared/published completion changed",
        )
        replay = dict(dynamic)
        replay["_campaign"] = campaign
        status_artifact, status_value = supervisor._runner_status(replay)
        expected_guard = {
            "guard_verifier_argv": completion_value["guard_verifier_argv"],
            "guard_verifier_stdout": completion_value["guard_verifier_stdout"],
            "restored_guards": completion_value["restored_guards"],
        }
        require(
            supervisor.strict_json_equal(
                validation_value["runner_statuses"][position],
                completion_value["runner_status"],
            )
            and supervisor.strict_json_equal(
                validation_value["runner_statuses"][position], status_artifact
            )
            and supervisor.strict_json_equal(
                validation_value["runner_logs"][position],
                completion_value["runner_log"],
            )
            and supervisor.strict_json_equal(
                validation_value["measurements"][position],
                completion_value["measurement"],
            )
            and supervisor.strict_json_equal(
                validation_value["bridge_replay_argvs"][position],
                completion_value["bridge_replay_argv"],
            )
            and supervisor.strict_json_equal(
                validation_value["guard_proofs"][position], expected_guard
            ),
            f"prior e{epoch} validation evidence changed",
        )
        dynamics.append(dynamic)
        expected_claims.append(claim)
        expected_authorizations.append(authorization)
        expected_prepared.append(prepared)
        expected_completions.append(completion)
        completion_values.append(completion_value)
        status_values.append(status_value)

    require(
        supervisor.strict_json_equal(launch_value.get("job_claims"), expected_claims)
        and supervisor.strict_json_equal(
            launch_value.get("authorizations"), expected_authorizations
        )
        and supervisor.strict_json_equal(commit_value.get("job_claims"), expected_claims)
        and supervisor.strict_json_equal(
            commit_value.get("authorizations"), expected_authorizations
        )
        and supervisor.strict_json_equal(
            commit_value.get("prepared_completions"), expected_prepared
        )
        and supervisor.strict_json_equal(
            validation_value.get("prepared_completions"), expected_prepared
        )
        and supervisor.strict_json_equal(
            commit_value.get("completions"), expected_completions
        ),
        "prior wave job/completion cross-links changed",
    )
    require(
        launch_value.get("master_runner_argv")
        == supervisor._runner_argv_v2(dynamics[0], campaign),
        "prior master runner argv changed",
    )
    if paired:
        ticket, ticket_value = _read_json_artifact(
            directory / "worker-ticket.json", "prior worker ticket"
        )
        require(
            set(ticket_value) == CENTRAL_TICKET_KEYS,
            "prior worker ticket schema changed",
        )
        endpoint._validate_self_hash(
            ticket_value, "ticket_payload_sha256", "prior worker ticket"
        )
        worker = dynamics[1]
        require(
            ticket_value.get("format") == endpoint.FORMAT
            and ticket_value.get("status") == "authorized"
            and ticket_value.get("role") == "worker"
            and ticket_value.get("hostname") == topology_value["worker"]["hostname"]
            and supervisor.strict_json_equal(
                ticket_value.get("machine_id"), topology_value["worker"]["machine_id"]
            )
            and ticket_value.get("candidate_epoch") == worker["epoch"]
            and supervisor.strict_json_equal(
                ticket_value.get("campaign"), campaign["_artifact"]
            )
            and supervisor.strict_json_equal(ticket_value.get("topology"), topology)
            and supervisor.strict_json_equal(
                ticket_value.get("endpoint"), topology_value["endpoint"]
            )
            and supervisor.strict_json_equal(ticket_value.get("wave"), wave)
            and supervisor.strict_json_equal(
                ticket_value.get("active_claim"), permanent_active
            )
            and supervisor.strict_json_equal(
                ticket_value.get("job_claim"), expected_claims[1]
            )
            and supervisor.strict_json_equal(
                ticket_value.get("authorization"), expected_authorizations[1]
            )
            and supervisor.strict_json_equal(
                ticket_value.get("formal_python"), campaign["_formal_python"]
            )
            and ticket_value.get("source_root")
            == topology_value["worker"]["source_root"]
            and ticket_value.get("source_origin") == endpoint.OFFICIAL_ORIGIN
            and ticket_value.get("source_commit")
            == topology_value["worker"]["source_commit"]
            and ticket_value.get("source_tree")
            == topology_value["worker"]["source_tree"]
            and supervisor.strict_json_equal(
                ticket_value.get("guarded_runner"), campaign["_guarded_runner"]
            )
            and supervisor.strict_json_equal(
                ticket_value.get("guard_verifier"), campaign["_guard_verifier"]
            )
            and ticket_value.get("runner_argv")
            == supervisor._runner_argv_v2(worker, campaign)
            and ticket_value.get("runner_status_path") == worker["runner_status_path"]
            and ticket_value.get("runner_log_path") == worker["runner_log_path"]
            and ticket_value.get("receipt_path")
            == str(directory / "worker-endpoint-receipt.json"),
            "prior worker ticket authority changed",
        )
        remote = [
            campaign["_formal_python"]["argv0"], "-I",
            topology_value["endpoint"]["path"], "run",
            "--ticket", ticket["path"],
            "--expected-ticket-sha256", ticket["sha256"],
            "--expected-ticket-bytes", str(ticket["bytes"]),
        ]
        expected_ssh = [
            topology_value["ssh_binary"]["path"],
            *topology_value["ssh_options"],
            topology_value["worker"]["ssh_host"],
            *remote,
        ]
        require(
            launch_value.get("worker_ssh_argv") == expected_ssh
            and launch_value.get("worker_ssh_argv_sha256") == _argv_sha(expected_ssh)
            and supervisor.strict_json_equal(launch_value.get("worker_ticket"), ticket)
            and supervisor.strict_json_equal(commit_value.get("worker_ticket"), ticket),
            "prior worker launch binding changed",
        )
        receipt = endpoint.artifact(
            commit_value.get("worker_endpoint_receipt"),
            "prior worker endpoint receipt",
        )
        require(
            supervisor.strict_json_equal(
                validation_value.get("worker_endpoint_receipt"), receipt
            ),
            "prior worker receipt binding changed",
        )
        receipt_artifact, _receipt_value = _validate_worker_receipt(
            campaign,
            topology_value,
            ticket,
            wave,
            permanent_active,
            worker,
            expected_claims[1],
            expected_authorizations[1],
            ProcessResult(0, endpoint.canonical_json_bytes(receipt), b""),
            completion_values[1]["runner_status"],
            status_values[1],
            completion_values[1]["runner_log"],
        )
        require(
            supervisor.strict_json_equal(receipt_artifact, receipt),
            "prior worker receipt replay changed",
        )
    else:
        require(
            commit_value.get("worker_ticket") is None
            and commit_value.get("worker_endpoint_receipt") is None
            and launch_value.get("worker_ticket") is None
            and launch_value.get("worker_ssh_argv") == []
            and validation_value.get("worker_endpoint_receipt") is None,
            "prior singleton contains worker evidence",
        )
    _finite(commit_value.get("completed_unix"), "prior wave completion time")
    _finite(validation_value.get("validated_unix"), "prior wave validation time")


def _validate_prior_waves(
    campaign: Mapping[str, Any], dispatcher_claim: Mapping[str, Any],
    topology: Mapping[str, Any], topology_value: Mapping[str, Any],
    completed_wave_count: int,
) -> None:
    waves_root = _dual_root(campaign) / WAVES_DIR_NAME
    require(waves_root.is_dir() and not waves_root.is_symlink(), "waves directory changed")
    expected_names = {"wave-%04d" % index for index in range(1, completed_wave_count + 1)}
    require(
        {entry.name for entry in waves_root.iterdir()} == expected_names,
        "wave directory coverage changed",
    )
    for index in range(1, completed_wave_count + 1):
        _validate_committed_wave(
            campaign, dispatcher_claim, topology, topology_value, index
        )


def _prepare_job(
    campaign: Mapping[str, Any], job: Mapping[str, Any], *,
    capture: Callable[[Sequence[str]], tuple[int, bytes, bytes]],
    clock: Callable[[], float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    epoch = job["epoch"]
    for output in (
        supervisor._job_claim_path_v2(campaign, epoch), job["authority_path"],
        job["authorization_path"], job["run_root"], job["runner_status_path"],
        job["runner_log_path"], job["completion_path"],
    ):
        require(not os.path.lexists(output), f"e{epoch} output already exists")
    candidate = supervisor._candidate_receipt_artifact(job)
    authorize_argv = supervisor._authorize_argv(campaign, job)
    claim = supervisor.write_new_json(
        supervisor._job_claim_path_v2(campaign, epoch),
        supervisor._job_claim_v2(campaign, job, candidate, authorize_argv, clock()),
        f"e{epoch} job claim",
    )
    rc, stdout, stderr = capture(authorize_argv)
    require(
        type(rc) is int and rc == 0 and stderr == b"",
        f"e{epoch} authority creation failed; wave is terminal",
    )
    authority = _artifact_from_path(job["authority_path"], f"e{epoch} work authority")
    adapter_stdout_sha = supervisor._adapter_stdout_artifact(
        stdout, authority, f"e{epoch} authority adapter"
    )
    supervisor._work_authority_runtime_binding(
        authority, campaign["_runtime_validation_source"], epoch
    )
    supervisor._work_authority_candidate_binding(authority, candidate, epoch)
    require(
        supervisor.strict_json_equal(supervisor._candidate_receipt_artifact(job), candidate),
        f"e{epoch} candidate receipt changed during authorization",
    )
    dynamic = dict(job)
    dynamic.update({"candidate_receipt": candidate, "work_authority": authority})
    runner_argv = supervisor._runner_argv_v2(dynamic, campaign)
    supervisor._validate_runner_argv_v2(runner_argv, dynamic, campaign)
    authorization = supervisor.write_new_json(
        dynamic["authorization_path"],
        supervisor._authorization_body(
            campaign, dynamic, authorize_argv, adapter_stdout_sha,
            runner_argv, clock(),
        ),
        f"e{epoch} authorization",
    )
    dynamic["authorization"] = authorization
    loaded_authorization, _value = supervisor._load_authorization(
        campaign, dynamic, candidate, authority
    )
    require(
        supervisor.strict_json_equal(loaded_authorization, authorization),
        f"e{epoch} authorization changed after publication",
    )
    return dynamic, claim, authorization


def _wave_claim(
    campaign: Mapping[str, Any], dispatcher_claim: Mapping[str, Any],
    topology: Mapping[str, Any], wave_index: int, jobs: Sequence[Mapping[str, Any]],
    supervisor_active: Mapping[str, Any], clock: Callable[[], float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = _wave_dir(campaign, wave_index)
    require(not os.path.lexists(directory), "wave evidence already exists; no retry is authorized")
    os.mkdir(directory, 0o700)
    epochs = [job["epoch"] for job in jobs]
    wave_value = _add_hash({
        "format": WAVE_FORMAT,
        "status": "active",
        "campaign": campaign["_artifact"],
        "dispatcher_claim": dict(dispatcher_claim),
        "topology": dict(topology),
        "wave_index": wave_index,
        "candidate_epochs": epochs,
        "master_epoch": epochs[0],
        "worker_epoch": epochs[1] if len(epochs) == 2 else None,
        "supervisor_active_claim": dict(supervisor_active),
        "created_unix": _finite(clock(), "wave creation time"),
    }, "wave_payload_sha256")
    wave = supervisor.write_new_json(directory / "wave.claim.json", wave_value, "wave claim")
    active_value = _add_hash({
        "format": ACTIVE_LOCK_FORMAT,
        "status": "active",
        "campaign": campaign["_artifact"],
        "dispatcher_claim": dict(dispatcher_claim),
        "topology": dict(topology),
        "wave": wave,
        "created_unix": _finite(clock(), "wave lock time"),
    }, "lock_payload_sha256")
    active = supervisor.write_new_json(_active_lock_path(campaign), active_value, "wave lock")
    permanent_active = supervisor.write_new_json(
        directory / "active.claim.json", active_value, "permanent wave activity claim"
    )
    return wave, wave_value, active, permanent_active


def _worker_ticket(
    campaign: Mapping[str, Any], topology_value: Mapping[str, Any],
    topology: Mapping[str, Any], wave: Mapping[str, Any],
    permanent_active: Mapping[str, Any],
    dynamic: Mapping[str, Any], claim: Mapping[str, Any],
    authorization: Mapping[str, Any], directory: Path,
) -> tuple[dict[str, Any], list[str]]:
    ticket_path = directory / "worker-ticket.json"
    receipt_path = directory / "worker-endpoint-receipt.json"
    ticket_value = endpoint._add_self_hash({
        "format": endpoint.FORMAT,
        "status": "authorized",
        "role": "worker",
        "hostname": topology_value["worker"]["hostname"],
        "machine_id": topology_value["worker"]["machine_id"],
        "host_identity": topology_value["worker"]["host_identity"],
        "candidate_epoch": dynamic["epoch"],
        "campaign": campaign["_artifact"],
        "topology": dict(topology),
        "endpoint": dict(topology_value["endpoint"]),
        "wave": dict(wave),
        "active_claim": dict(permanent_active),
        "job_claim": dict(claim),
        "authorization": dict(authorization),
        "formal_python": campaign["_formal_python"],
        "source_root": topology_value["worker"]["source_root"],
        "source_origin": topology_value["worker"]["source_origin"],
        "source_commit": topology_value["worker"]["source_commit"],
        "source_tree": topology_value["worker"]["source_tree"],
        "guarded_runner": campaign["_guarded_runner"],
        "guard_verifier": campaign["_guard_verifier"],
        "runner_argv": supervisor._runner_argv_v2(dynamic, campaign),
        "runner_status_path": dynamic["runner_status_path"],
        "runner_log_path": dynamic["runner_log_path"],
        "receipt_path": str(receipt_path),
    }, "ticket_payload_sha256")
    ticket = supervisor.write_new_json(ticket_path, ticket_value, "worker ticket")
    remote = [
        campaign["_formal_python"]["argv0"], "-I",
        topology_value["endpoint"]["path"], "run",
        "--ticket", ticket["path"],
        "--expected-ticket-sha256", ticket["sha256"],
        "--expected-ticket-bytes", str(ticket["bytes"]),
    ]
    require(
        all(SAFE_REMOTE_TOKEN.fullmatch(token) is not None for token in remote),
        "remote endpoint argv contains a shell-sensitive token",
    )
    ssh_argv = [
        topology_value["ssh_binary"]["path"],
        *topology_value["ssh_options"],
        topology_value["worker"]["ssh_host"],
        *remote,
    ]
    return ticket, ssh_argv


def _argv_sha(argv: Sequence[str]) -> str:
    return supervisor._sha256(b"\0".join(os.fsencode(token) for token in argv) + b"\0")


def _launch_claim(
    campaign: Mapping[str, Any], dispatcher_claim: Mapping[str, Any],
    topology: Mapping[str, Any], wave: Mapping[str, Any],
    active: Mapping[str, Any], dynamics: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]], authorizations: Sequence[Mapping[str, Any]],
    worker_ticket: Optional[Mapping[str, Any]], worker_ssh_argv: Sequence[str],
    directory: Path, clock: Callable[[], float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    master_argv = supervisor._runner_argv_v2(dynamics[0], campaign)
    body = _add_hash({
        "format": LAUNCH_FORMAT,
        "status": "authorized",
        "campaign": campaign["_artifact"],
        "dispatcher_claim": dict(dispatcher_claim),
        "topology": dict(topology),
        "wave": dict(wave),
        "active_lock": dict(active),
        "candidate_epochs": [job["epoch"] for job in dynamics],
        "job_claims": [dict(item) for item in claims],
        "authorizations": [dict(item) for item in authorizations],
        "master_runner_argv": master_argv,
        "master_runner_argv_sha256": _argv_sha(master_argv),
        "worker_ssh_argv": list(worker_ssh_argv),
        "worker_ssh_argv_sha256": _argv_sha(worker_ssh_argv),
        "worker_ticket": dict(worker_ticket) if worker_ticket is not None else None,
        "created_unix": _finite(clock(), "launch claim time"),
    }, "launch_payload_sha256")
    return supervisor.write_new_json(directory / "launch.claim.json", body, "launch claim"), body


def _replay_prelaunch_authority(
    campaign: Mapping[str, Any], dispatcher_claim: Mapping[str, Any],
    topology: Mapping[str, Any], topology_value: Mapping[str, Any],
    wave: Mapping[str, Any], active_lock: Mapping[str, Any],
    permanent_active: Mapping[str, Any], dynamics: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]], authorizations: Sequence[Mapping[str, Any]],
    worker_ticket: Optional[Mapping[str, Any]], worker_ssh_argv: Sequence[str],
    launch: Mapping[str, Any], directory: Path,
) -> None:
    """Full schema/self-hash/cross-link replay at the mutation boundary."""

    live_topology, live_topology_value = load_topology(campaign, topology)
    require(
        supervisor.strict_json_equal(live_topology, topology)
        and supervisor.strict_json_equal(live_topology_value, topology_value),
        "prelaunch topology changed",
    )
    live_dispatcher_claim, _live_dispatcher_value = load_or_create_dispatcher_claim(
        campaign, topology
    )
    require(
        supervisor.strict_json_equal(live_dispatcher_claim, dispatcher_claim),
        "prelaunch dispatcher claim changed",
    )
    wave_artifact, wave_value = _load_self_hashed_document(
        directory / "wave.claim.json", keys=WAVE_KEYS,
        field="wave_payload_sha256", label="prelaunch wave claim",
    )
    require(supervisor.strict_json_equal(wave_artifact, wave), "prelaunch wave artifact changed")
    epochs = [dynamic["epoch"] for dynamic in dynamics]
    require(
        wave_value.get("format") == WAVE_FORMAT
        and wave_value.get("status") == "active"
        and wave_value.get("candidate_epochs") == epochs
        and wave_value.get("master_epoch") == epochs[0]
        and wave_value.get("worker_epoch") == (epochs[1] if len(epochs) == 2 else None)
        and supervisor.strict_json_equal(wave_value.get("campaign"), campaign["_artifact"])
        and supervisor.strict_json_equal(wave_value.get("dispatcher_claim"), dispatcher_claim)
        and supervisor.strict_json_equal(wave_value.get("topology"), topology),
        "prelaunch wave authority changed",
    )
    supervisor_active, supervisor_active_value = _read_json_artifact(
        wave_value["supervisor_active_claim"]["path"],
        "prelaunch supervisor active claim",
    )
    require(
        supervisor.strict_json_equal(
            supervisor_active, wave_value.get("supervisor_active_claim")
        )
        and supervisor_active["path"]
        == str(campaign["_state_root"] / "active_invocation.claim.json"),
        "prelaunch supervisor active-claim path changed",
    )
    require(
        set(supervisor_active_value) == supervisor.ACTIVE_CLAIM_KEYS,
        "prelaunch supervisor active-claim schema changed",
    )
    supervisor._self_hashed(
        supervisor_active_value, "claim_payload_sha256",
        "prelaunch supervisor active claim",
    )
    require(
        supervisor_active_value.get("format") == supervisor.ACTIVE_CLAIM_FORMAT
        and supervisor_active_value.get("status") == "active"
        and supervisor_active_value.get("operation")
        == "dual-host-wave-%04d" % wave_value["wave_index"]
        and supervisor.strict_json_equal(
            supervisor_active_value.get("campaign"), campaign["_artifact"]
        ),
        "prelaunch supervisor active-claim authority changed",
    )
    for path, expected, label in (
        (_active_lock_path(campaign), active_lock, "prelaunch active lock"),
        (directory / "active.claim.json", permanent_active, "prelaunch permanent active claim"),
    ):
        artifact, value = _load_self_hashed_document(
            path, keys=ACTIVE_LOCK_KEYS, field="lock_payload_sha256", label=label
        )
        require(supervisor.strict_json_equal(artifact, expected), f"{label} artifact changed")
        require(
            value.get("format") == ACTIVE_LOCK_FORMAT
            and value.get("status") == "active"
            and supervisor.strict_json_equal(value.get("campaign"), campaign["_artifact"])
            and supervisor.strict_json_equal(value.get("dispatcher_claim"), dispatcher_claim)
            and supervisor.strict_json_equal(value.get("topology"), topology)
            and supervisor.strict_json_equal(value.get("wave"), wave),
            f"{label} authority changed",
        )
    for dynamic, expected_claim, expected_authorization in zip(
        dynamics, claims, authorizations
    ):
        loaded_claim, _claim_value = supervisor._load_job_claim_v2(
            campaign, dynamic, dynamic["candidate_receipt"]
        )
        loaded_authorization, _authorization_value = supervisor._load_authorization(
            campaign, dynamic, dynamic["candidate_receipt"], dynamic["work_authority"]
        )
        require(
            supervisor.strict_json_equal(loaded_claim, expected_claim)
            and supervisor.strict_json_equal(loaded_authorization, expected_authorization),
            f"e{dynamic['epoch']} claim/authorization changed before launch",
        )
        supervisor._work_authority_runtime_binding(
            dynamic["work_authority"], campaign["_runtime_validation_source"],
            dynamic["epoch"],
        )
        supervisor._work_authority_candidate_binding(
            dynamic["work_authority"], dynamic["candidate_receipt"], dynamic["epoch"]
        )
    launch_artifact, launch_value = _load_self_hashed_document(
        directory / "launch.claim.json", keys=LAUNCH_KEYS,
        field="launch_payload_sha256", label="prelaunch launch claim",
    )
    require(
        supervisor.strict_json_equal(launch_artifact, launch)
        and launch_value.get("format") == LAUNCH_FORMAT
        and launch_value.get("status") == "authorized"
        and launch_value.get("candidate_epochs") == epochs
        and supervisor.strict_json_equal(launch_value.get("campaign"), campaign["_artifact"])
        and supervisor.strict_json_equal(launch_value.get("dispatcher_claim"), dispatcher_claim)
        and supervisor.strict_json_equal(launch_value.get("topology"), topology)
        and supervisor.strict_json_equal(launch_value.get("wave"), wave)
        and supervisor.strict_json_equal(launch_value.get("active_lock"), active_lock)
        and supervisor.strict_json_equal(launch_value.get("job_claims"), list(claims))
        and supervisor.strict_json_equal(
            launch_value.get("authorizations"), list(authorizations)
        ),
        "prelaunch launch authority changed",
    )
    require(
        launch_value.get("master_runner_argv")
        == supervisor._runner_argv_v2(dynamics[0], campaign)
        and launch_value.get("master_runner_argv_sha256")
        == _argv_sha(launch_value["master_runner_argv"])
        and launch_value.get("worker_ssh_argv_sha256")
        == _argv_sha(launch_value.get("worker_ssh_argv", [])),
        "prelaunch argv digest changed",
    )
    require(
        launch_value.get("worker_ssh_argv") == list(worker_ssh_argv),
        "prelaunch worker SSH argv changed",
    )
    endpoint.artifact(topology_value["endpoint"], "prelaunch endpoint")
    if len(dynamics) == 2:
        require(worker_ticket is not None, "prelaunch worker ticket is missing")
        ticket_artifact, ticket_value = _read_json_artifact(
            worker_ticket["path"], "prelaunch worker ticket"
        )
        require(
            supervisor.strict_json_equal(ticket_artifact, worker_ticket)
            and set(ticket_value) == CENTRAL_TICKET_KEYS,
            "prelaunch worker ticket schema/artifact changed",
        )
        endpoint._validate_self_hash(
            ticket_value, "ticket_payload_sha256", "prelaunch worker ticket"
        )
        require(
            ticket_value.get("format") == endpoint.FORMAT
            and ticket_value.get("status") == "authorized"
            and ticket_value.get("role") == "worker"
            and ticket_value.get("candidate_epoch") == dynamics[1]["epoch"]
            and supervisor.strict_json_equal(ticket_value.get("campaign"), campaign["_artifact"])
            and supervisor.strict_json_equal(ticket_value.get("topology"), topology)
            and supervisor.strict_json_equal(
                ticket_value.get("endpoint"), topology_value["endpoint"]
            )
            and supervisor.strict_json_equal(ticket_value.get("wave"), wave)
            and supervisor.strict_json_equal(
                ticket_value.get("active_claim"), permanent_active
            )
            and supervisor.strict_json_equal(ticket_value.get("job_claim"), claims[1])
            and supervisor.strict_json_equal(
                ticket_value.get("authorization"), authorizations[1]
            )
            and ticket_value.get("hostname") == topology_value["worker"]["hostname"]
            and supervisor.strict_json_equal(
                ticket_value.get("machine_id"), topology_value["worker"]["machine_id"]
            )
            and supervisor.strict_json_equal(
                ticket_value.get("host_identity"),
                topology_value["worker"]["host_identity"],
            )
            and ticket_value.get("formal_python") == campaign["_formal_python"]
            and ticket_value.get("source_root")
            == topology_value["worker"]["source_root"]
            and ticket_value.get("source_origin") == endpoint.OFFICIAL_ORIGIN
            and ticket_value.get("source_commit")
            == topology_value["worker"]["source_commit"]
            and ticket_value.get("source_tree")
            == topology_value["worker"]["source_tree"]
            and ticket_value.get("guarded_runner") == campaign["_guarded_runner"]
            and ticket_value.get("guard_verifier") == campaign["_guard_verifier"]
            and ticket_value.get("runner_argv")
            == supervisor._runner_argv_v2(dynamics[1], campaign)
            and ticket_value.get("runner_status_path")
            == dynamics[1]["runner_status_path"]
            and ticket_value.get("runner_log_path") == dynamics[1]["runner_log_path"]
            and ticket_value.get("receipt_path")
            == str(directory / "worker-endpoint-receipt.json")
            and launch_value.get("worker_ticket") == worker_ticket,
            "prelaunch worker ticket authority changed",
        )
    else:
        require(
            worker_ticket is None
            and launch_value.get("worker_ticket") is None
            and launch_value.get("worker_ssh_argv") == [],
            "singleton has worker launch authority",
        )


class ProcessResult(NamedTuple):
    returncode: int
    stdout: bytes
    stderr: bytes


def _default_launch(
    master_argv: Sequence[str], worker_ssh_argv: Sequence[str],
) -> tuple[ProcessResult, Optional[ProcessResult]]:
    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if not worker_ssh_argv:
        completed = subprocess.run(
            list(master_argv), shell=False, check=False, capture_output=True,
            env=environment,
        )
        return ProcessResult(completed.returncode, completed.stdout, completed.stderr), None

    # Start the remote endpoint first; it has SSH admission overhead before it
    # reaches the guarded runner.  Starting the local runner immediately after
    # keeps both hosts occupied without a shell or a second GPU authority.
    worker_process = subprocess.Popen(
        list(worker_ssh_argv), shell=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment,
    )
    try:
        master_process = subprocess.Popen(
            list(master_argv), shell=False, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=environment,
        )
    except BaseException:
        # Never kill or retry a worker whose mutation boundary may have been
        # crossed.  Wait for its immutable endpoint evidence, then fail closed.
        worker_process.communicate()
        raise
    master_stdout, master_stderr = master_process.communicate()
    worker_stdout, worker_stderr = worker_process.communicate()
    return (
        ProcessResult(master_process.returncode, master_stdout, master_stderr),
        ProcessResult(worker_process.returncode, worker_stdout, worker_stderr),
    )


def _validate_worker_receipt(
    campaign: Mapping[str, Any], topology_value: Mapping[str, Any],
    ticket: Mapping[str, Any], wave: Mapping[str, Any],
    permanent_active: Mapping[str, Any], dynamic: Mapping[str, Any],
    job_claim: Mapping[str, Any], authorization: Mapping[str, Any],
    process: ProcessResult, status_artifact: Mapping[str, Any],
    status_value: Mapping[str, Any], log_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        type(process.returncode) is int and process.returncode == 0
        and process.stderr == b"" and 0 < len(process.stdout) <= 4096,
        "worker endpoint failed or emitted stderr/oversize stdout",
    )
    stdout_value = endpoint.strict_json(process.stdout, "worker endpoint stdout", canonical=True)
    stdout_artifact = _artifact_descriptor(stdout_value, "worker endpoint stdout")
    ticket_artifact, ticket_value = _read_json_artifact(
        ticket["path"], "worker ticket receipt replay"
    )
    require(
        supervisor.strict_json_equal(ticket_artifact, ticket)
        and set(ticket_value) == CENTRAL_TICKET_KEYS,
        "worker ticket changed after launch",
    )
    endpoint._validate_self_hash(
        ticket_value, "ticket_payload_sha256", "worker ticket receipt replay"
    )
    require(
        stdout_artifact["path"] == ticket_value.get("receipt_path"),
        "worker receipt path differs from ticket",
    )
    artifact, value = _read_json_artifact(stdout_artifact["path"], "worker endpoint receipt")
    require(supervisor.strict_json_equal(artifact, stdout_artifact), "worker receipt stdout changed")
    require(set(value) == REMOTE_RECEIPT_KEYS, "worker endpoint receipt schema changed")
    endpoint._validate_self_hash(value, "receipt_payload_sha256", "worker endpoint receipt")
    require(
        value.get("format") == endpoint.RECEIPT_FORMAT
        and value.get("status") == "complete"
        and value.get("role") == "worker"
        and value.get("hostname") == topology_value["worker"]["hostname"]
        and value.get("candidate_epoch") == dynamic["epoch"],
        "worker receipt role/state changed",
    )
    require(
        endpoint.strict_equal(value.get("machine_id"), topology_value["worker"]["machine_id"])
        and endpoint.strict_equal(
            value.get("host_identity"), topology_value["worker"]["host_identity"]
        )
        and endpoint.strict_equal(value.get("ticket"), ticket_artifact)
        and endpoint.strict_equal(value.get("campaign"), campaign["_artifact"])
        and endpoint.strict_equal(value.get("wave"), wave)
        and endpoint.strict_equal(value.get("active_claim"), permanent_active)
        and endpoint.strict_equal(value.get("job_claim"), job_claim)
        and endpoint.strict_equal(value.get("authorization"), authorization),
        "worker receipt authority binding changed",
    )
    runner_argv = supervisor._runner_argv_v2(dynamic, campaign)
    require(
        endpoint.strict_equal(value.get("formal_python"), campaign["_formal_python"])
        and value.get("source_root") == topology_value["worker"]["source_root"]
        and value.get("source_origin") == endpoint.OFFICIAL_ORIGIN
        and value.get("source_commit") == topology_value["worker"]["source_commit"]
        and value.get("source_tree") == topology_value["worker"]["source_tree"]
        and endpoint.strict_equal(value.get("guarded_runner"), campaign["_guarded_runner"])
        and endpoint.strict_equal(value.get("guard_verifier"), campaign["_guard_verifier"])
        and endpoint.strict_equal(value.get("runner_argv"), runner_argv)
        and value.get("runner_argv_sha256") == endpoint.canonical_json_sha256(runner_argv),
        "worker receipt runtime binding changed",
    )
    require(
        endpoint.strict_equal(value.get("runner_status"), status_artifact)
        and endpoint.strict_equal(value.get("runner_log"), log_artifact)
        and endpoint.strict_equal(value.get("restored_guards"), status_value["restored_guards"]),
        "worker receipt runner evidence changed",
    )
    expected_guard_argv = [
        campaign["_formal_python"]["argv0"], campaign["_guard_verifier"]["path"],
        *[str(status_value["restored_guards"][str(index)]) for index in supervisor.GPU_INDICES],
    ]
    guard_stdout = value.get("guard_verifier_stdout")
    match = supervisor.GUARD_PASS_RE.fullmatch(guard_stdout if isinstance(guard_stdout, str) else "")
    require(
        value.get("guard_verifier_argv") == expected_guard_argv
        and match is not None
        and {str(index): int(match.group(index + 1)) for index in supervisor.GPU_INDICES}
        == status_value["restored_guards"],
        "worker guard proof changed",
    )
    _finite(value.get("completed_unix"), "worker endpoint completion time")
    return artifact, value


def _validate_job_after_launch(
    campaign: Mapping[str, Any], dynamic: Mapping[str, Any], *, role: str,
    capture: Callable[[Sequence[str]], tuple[int, bytes, bytes]],
    worker_receipt: Optional[tuple[Mapping[str, Any], Mapping[str, Any]]],
    clock: Callable[[], float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = dict(dynamic)
    replay["_campaign"] = campaign
    status_artifact, status = supervisor._runner_status(replay)
    log_artifact, measurement, measurement_value = supervisor._verify_runner_log(replay)
    bridge_argv = supervisor._bridge_replay(campaign, replay, measurement, capture)
    if role == "master":
        verifier_argv, verifier_stdout, guards = supervisor._guard_verify(
            campaign, status, capture
        )
        endpoint_receipt_artifact = None
    else:
        require(worker_receipt is not None, "worker endpoint receipt is missing")
        endpoint_receipt_artifact, endpoint_receipt_value = worker_receipt
        verifier_argv = list(endpoint_receipt_value["guard_verifier_argv"])
        verifier_stdout = endpoint_receipt_value["guard_verifier_stdout"]
        guards = dict(endpoint_receipt_value["restored_guards"])
    status_artifact_2, status_2 = supervisor._runner_status(replay)
    require(
        supervisor.strict_json_equal(status_artifact, status_artifact_2)
        and supervisor.strict_json_equal(status, status_2),
        f"{role} runner status changed during validation",
    )
    measurement_2, _measurement_value_2 = supervisor._measurement(
        replay, measurement["sha256"], measurement["receipt_payload_sha256"]
    )
    require(supervisor.strict_json_equal(measurement_2, measurement), f"{role} measurement changed")
    require(
        supervisor.strict_json_equal(
            supervisor._candidate_receipt_artifact(replay), replay["candidate_receipt"]
        )
        and supervisor.strict_json_equal(
            _artifact_from_path(replay["authority_path"], f"{role} work authority replay"),
            replay["work_authority"],
        ),
        f"{role} producer authority changed",
    )
    loaded_authorization, _authorization_value = supervisor._load_authorization(
        campaign, replay, replay["candidate_receipt"], replay["work_authority"]
    )
    require(
        supervisor.strict_json_equal(loaded_authorization, replay["authorization"]),
        f"{role} authorization changed",
    )
    completion = supervisor._completion_body(
        campaign, replay, status_artifact, status, log_artifact, measurement,
        measurement_value, bridge_argv, verifier_argv, verifier_stdout, guards,
        clock(),
    )
    evidence = {
        "runner_status": status_artifact,
        "runner_log": log_artifact,
        "measurement": measurement,
        "bridge_replay_argv": bridge_argv,
        "guard_proof": {
            "guard_verifier_argv": verifier_argv,
            "guard_verifier_stdout": verifier_stdout,
            "restored_guards": guards,
        },
        "worker_endpoint_receipt": endpoint_receipt_artifact,
    }
    return completion, evidence


def _replay_publication_inputs(
    campaign: Mapping[str, Any], dynamics: Sequence[Mapping[str, Any]],
    evidences: Sequence[Mapping[str, Any]],
    worker_receipt: Optional[Mapping[str, Any]],
) -> None:
    require(len(dynamics) == len(evidences), "publication evidence cardinality changed")
    for dynamic, evidence in zip(dynamics, evidences):
        replay = dict(dynamic)
        replay["_campaign"] = campaign
        require(
            supervisor.strict_json_equal(
                supervisor._candidate_receipt_artifact(replay),
                replay["candidate_receipt"],
            )
            and supervisor.strict_json_equal(
                _artifact_from_path(
                    replay["authority_path"],
                    f"e{replay['epoch']} publication authority replay",
                ),
                replay["work_authority"],
            ),
            f"e{replay['epoch']} producer input changed before publication",
        )
        authorization, _authorization_value = supervisor._load_authorization(
            campaign, replay, replay["candidate_receipt"], replay["work_authority"]
        )
        status_artifact, _status_value = supervisor._runner_status(replay)
        log_artifact, measurement, _measurement_value = supervisor._verify_runner_log(replay)
        require(
            supervisor.strict_json_equal(authorization, replay["authorization"])
            and supervisor.strict_json_equal(
                status_artifact, evidence["runner_status"]
            )
            and supervisor.strict_json_equal(log_artifact, evidence["runner_log"])
            and supervisor.strict_json_equal(measurement, evidence["measurement"]),
            f"e{replay['epoch']} execution evidence changed before publication",
        )
    if worker_receipt is not None:
        require(
            supervisor.strict_json_equal(
                endpoint.artifact(worker_receipt, "worker receipt publication replay"),
                worker_receipt,
            ),
            "worker endpoint receipt changed before publication",
        )


def _publish_wave(
    campaign: Mapping[str, Any], dispatcher_claim: Mapping[str, Any],
    topology: Mapping[str, Any], wave: Mapping[str, Any], launch: Mapping[str, Any],
    dynamics: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]],
    authorizations: Sequence[Mapping[str, Any]], worker_ticket: Optional[Mapping[str, Any]],
    worker_receipt: Optional[Mapping[str, Any]], completions: Sequence[Mapping[str, Any]],
    evidences: Sequence[Mapping[str, Any]], directory: Path,
    clock: Callable[[], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Create immutable prepared payloads before publishing either queue
    # completion.  Their bytes must match the two final receipts exactly.
    require(
        all(not os.path.lexists(dynamic["completion_path"]) for dynamic in dynamics),
        "a queue completion appeared before full-wave publication",
    )
    prepared: list[dict[str, Any]] = []
    for dynamic, value in zip(dynamics, completions):
        prepared.append(supervisor.write_new_json(
            directory / ("prepared-completion-epoch-%04d.json" % dynamic["epoch"]),
            value,
            f"e{dynamic['epoch']} prepared completion",
        ))
    validation_value = _add_hash({
        "format": VALIDATION_FORMAT,
        "status": "complete",
        "campaign": campaign["_artifact"],
        "dispatcher_claim": dict(dispatcher_claim),
        "topology": dict(topology),
        "wave": dict(wave),
        "launch": dict(launch),
        "candidate_epochs": [dynamic["epoch"] for dynamic in dynamics],
        "runner_statuses": [evidence["runner_status"] for evidence in evidences],
        "runner_logs": [evidence["runner_log"] for evidence in evidences],
        "measurements": [evidence["measurement"] for evidence in evidences],
        "bridge_replay_argvs": [evidence["bridge_replay_argv"] for evidence in evidences],
        "guard_proofs": [evidence["guard_proof"] for evidence in evidences],
        "worker_endpoint_receipt": dict(worker_receipt) if worker_receipt is not None else None,
        "prepared_completions": prepared,
        "validated_unix": _finite(clock(), "wave validation time"),
    }, "validation_payload_sha256")
    validation = supervisor.write_new_json(
        directory / "validation.json", validation_value, "wave validation"
    )

    _replay_publication_inputs(campaign, dynamics, evidences, worker_receipt)

    published: list[dict[str, Any]] = []
    for dynamic, value, staged in zip(dynamics, completions, prepared):
        artifact = supervisor.write_new_json(
            dynamic["completion_path"], value, f"e{dynamic['epoch']} completion"
        )
        require(
            artifact["sha256"] == staged["sha256"]
            and artifact["bytes"] == staged["bytes"],
            f"e{dynamic['epoch']} published completion differs from prepared bytes",
        )
        loaded_completion, _loaded_completion_value = supervisor._load_completion_v2(
            campaign, dynamic
        )
        require(
            supervisor.strict_json_equal(loaded_completion, artifact),
            f"e{dynamic['epoch']} completion failed post-publication replay",
        )
        published.append(artifact)
    commit_value = _add_hash({
        "format": COMMIT_FORMAT,
        "status": "complete",
        "campaign": campaign["_artifact"],
        "dispatcher_claim": dict(dispatcher_claim),
        "topology": dict(topology),
        "wave": dict(wave),
        "launch": dict(launch),
        "validation": validation,
        "candidate_epochs": [dynamic["epoch"] for dynamic in dynamics],
        "job_claims": [dict(item) for item in claims],
        "authorizations": [dict(item) for item in authorizations],
        "worker_ticket": dict(worker_ticket) if worker_ticket is not None else None,
        "worker_endpoint_receipt": dict(worker_receipt) if worker_receipt is not None else None,
        "prepared_completions": prepared,
        "completions": published,
        "completed_unix": _finite(clock(), "wave commit time"),
    }, "commit_payload_sha256")
    commit = supervisor.write_new_json(directory / "commit.json", commit_value, "wave commit")
    return commit, published


def _release_lock(path: Path, artifact: Mapping[str, Any], label: str) -> None:
    live = _artifact_from_path(path, label)
    require(supervisor.strict_json_equal(live, artifact), f"{label} changed")
    path.unlink()


def dispatch_next(
    campaign: Mapping[str, Any], topology_artifact: Mapping[str, Any], *,
    capture: Callable[[Sequence[str]], tuple[int, bytes, bytes]] = supervisor._run_capture,
    launch: Callable[[Sequence[str], Sequence[str]], tuple[ProcessResult, Optional[ProcessResult]]] = _default_launch,
    clock: Callable[[], float] = time.time,
    hostname_provider: Callable[[], str] = socket.gethostname,
) -> dict[str, Any]:
    require(
        endpoint.TICKET_KEYS == CENTRAL_TICKET_KEYS,
        "remote endpoint ticket schema lacks direct topology/endpoint binding",
    )
    topology, topology_value = load_topology(
        campaign, topology_artifact, hostname_provider=hostname_provider
    )
    dispatcher_claim, _dispatcher_value = load_or_create_dispatcher_claim(
        campaign, topology, clock=clock
    )
    require(
        not os.path.lexists(_active_lock_path(campaign)),
        "active dual-host wave exists; no retry/resume is authorized",
    )
    require(
        not os.path.lexists(campaign["_state_root"] / "active_invocation.claim.json"),
        "supervisor active claim exists; no concurrent operation is authorized",
    )
    supervisor._campaign_claim_v2(campaign, clock)
    completed, head = _scan_dual(campaign)
    wave_index, jobs = _expected_wave(campaign, completed, head)
    require(
        {entry.name for entry in _dual_root(campaign).iterdir()}
        == {TOPOLOGY_NAME, CLAIM_NAME, WAVES_DIR_NAME},
        "dual-dispatch root inventory changed",
    )
    _validate_prior_waves(
        campaign, dispatcher_claim, topology, topology_value,
        len(fresh_waves()) if not jobs else wave_index - 1,
    )
    if not jobs:
        return {
            "status": "ready_to_finalize",
            "completion_count": 22,
            "fresh_completion_count": 21,
            "dispatcher_claim": dispatcher_claim,
            "topology": topology,
        }
    for job in jobs:
        require(
            os.path.lexists(job["candidate_receipt_path"]),
            f"e{job['epoch']} candidate receipt is not ready",
        )

    supervisor_active = supervisor._active_v2(
        campaign, "dual-host-wave-%04d" % wave_index, clock
    )
    wave, _wave_value, active_lock, permanent_active = _wave_claim(
        campaign, dispatcher_claim, topology, wave_index, jobs,
        supervisor_active, clock,
    )
    supervisor._revalidate_control(campaign)
    supervisor._runtime_contract_check(campaign, capture)
    dynamics: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    authorizations: list[dict[str, Any]] = []
    for job in jobs:
        dynamic, claim, authorization = _prepare_job(
            campaign, job, capture=capture, clock=clock
        )
        dynamics.append(dynamic)
        claims.append(claim)
        authorizations.append(authorization)

    directory = _wave_dir(campaign, wave_index)
    worker_ticket: Optional[dict[str, Any]] = None
    worker_ssh_argv: list[str] = []
    if len(dynamics) == 2:
        worker_ticket, worker_ssh_argv = _worker_ticket(
            campaign, topology_value, topology, wave, permanent_active, dynamics[1],
            claims[1], authorizations[1], directory,
        )
    launch_artifact, _launch_value = _launch_claim(
        campaign, dispatcher_claim, topology, wave, active_lock, dynamics,
        claims, authorizations, worker_ticket, worker_ssh_argv, directory, clock,
    )
    _replay_prelaunch_authority(
        campaign, dispatcher_claim, topology, topology_value, wave,
        active_lock, permanent_active, dynamics, claims, authorizations,
        worker_ticket, worker_ssh_argv, launch_artifact, directory,
    )
    master_argv = supervisor._runner_argv_v2(dynamics[0], campaign)
    master_result, worker_result = launch(master_argv, worker_ssh_argv)
    require(
        type(master_result.returncode) is int and master_result.returncode == 0
        and master_result.stdout == b"" and master_result.stderr == b"",
        "master guarded runner failed or emitted process output",
    )
    if len(dynamics) == 2:
        require(worker_result is not None and worker_ticket is not None, "worker process evidence is missing")
    else:
        require(worker_result is None and worker_ticket is None, "singleton unexpectedly launched worker")

    supervisor._revalidate_control(campaign)
    supervisor._runtime_contract_check(campaign, capture)
    worker_receipt_pair: Optional[tuple[dict[str, Any], dict[str, Any]]] = None
    if len(dynamics) == 2:
        worker_replay = dict(dynamics[1])
        worker_replay["_campaign"] = campaign
        worker_status_artifact, worker_status = supervisor._runner_status(worker_replay)
        worker_log_artifact, _measurement, _measurement_value = supervisor._verify_runner_log(worker_replay)
        worker_receipt_pair = _validate_worker_receipt(
            campaign, topology_value, worker_ticket, wave, permanent_active,
            dynamics[1], claims[1], authorizations[1], worker_result,
            worker_status_artifact, worker_status, worker_log_artifact,
        )

    completion_values: list[dict[str, Any]] = []
    evidences: list[dict[str, Any]] = []
    for index, dynamic in enumerate(dynamics):
        completion_value, evidence = _validate_job_after_launch(
            campaign, dynamic,
            role="master" if index == 0 else "worker",
            capture=capture,
            worker_receipt=worker_receipt_pair if index == 1 else None,
            clock=clock,
        )
        completion_values.append(completion_value)
        evidences.append(evidence)
    worker_receipt_artifact = worker_receipt_pair[0] if worker_receipt_pair else None
    commit, published = _publish_wave(
        campaign, dispatcher_claim, topology, wave, launch_artifact, dynamics,
        claims, authorizations, worker_ticket, worker_receipt_artifact,
        completion_values, evidences, directory, clock,
    )
    supervisor._release_active_v2(campaign, supervisor_active)
    _release_lock(_active_lock_path(campaign), active_lock, "active wave lock")
    return {
        "status": "wave_complete",
        "wave_index": wave_index,
        "candidate_epochs": [dynamic["epoch"] for dynamic in dynamics],
        "completion_count": len(completed) + len(dynamics),
        "fresh_completion_count": len(completed) - 1 + len(dynamics),
        "commit": commit,
        "completions": published,
    }


def _load_campaign_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return supervisor.load_campaign(
        args.campaign, args.expected_campaign_sha256,
        args.expected_campaign_bytes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-topology", allow_abbrev=False)
    dispatch = commands.add_parser("dispatch-next", allow_abbrev=False)
    for command in (freeze, dispatch):
        command.add_argument("--campaign", required=True)
        command.add_argument("--expected-campaign-sha256", required=True)
        command.add_argument("--expected-campaign-bytes", required=True, type=int)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--worker-ssh-host", required=True)
    freeze.add_argument("--worker-hostname", required=True)
    freeze.add_argument("--worker-machine-id-sha256", required=True)
    freeze.add_argument("--worker-machine-id-bytes", required=True, type=int)
    freeze.add_argument("--worker-host-identity-sha256", required=True)
    freeze.add_argument("--worker-host-identity-bytes", required=True, type=int)
    dispatch.add_argument("--topology", required=True)
    dispatch.add_argument("--expected-topology-sha256", required=True)
    dispatch.add_argument("--expected-topology-bytes", required=True, type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        campaign = _load_campaign_from_args(args)
        if args.command == "freeze-topology":
            result = freeze_topology(
                campaign,
                output=args.output,
                worker_ssh_host=args.worker_ssh_host,
                worker_hostname=args.worker_hostname,
                worker_machine_id_sha256=args.worker_machine_id_sha256,
                worker_machine_id_bytes=args.worker_machine_id_bytes,
                worker_host_identity_sha256=args.worker_host_identity_sha256,
                worker_host_identity_bytes=args.worker_host_identity_bytes,
            )
        else:
            topology = {
                "path": args.topology,
                "sha256": args.expected_topology_sha256,
                "bytes": args.expected_topology_bytes,
            }
            result = dispatch_next(campaign, topology)
    except (
        DualDispatchError, supervisor.SupervisorError,
        endpoint.RemoteEndpointError, OSError, subprocess.SubprocessError,
    ) as error:
        sys.stderr.write(f"dual-host dispatcher rejected operation: {error}\n")
        return 1
    sys.stdout.buffer.write(supervisor.canonical_json_bytes(result))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
