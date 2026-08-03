#!/usr/bin/env python3
"""Execute one frozen SHOW Base live-validation ticket on the worker host.

The endpoint deliberately has no SSH or scheduling logic.  A controller
publishes one immutable, self-hashed ticket and invokes ``run`` with the
ticket's path, SHA-256, and byte length.  This process replays every binding,
runs exactly one all-GPU guarded runner with ``shell=False``, verifies the
restored guards on the same host, and creates one immutable receipt.

Nothing is written on an admission, runner, status, log, guard, or replay
failure.  (The guarded runner may of course have published its own terminal
status/log before a later verification failure.)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

sys.dont_write_bytecode = True
_SUPERVISOR_PATH = (
    Path(__file__).resolve(strict=True).with_name(
        "supervise_base_v14_live_validation.py"
    )
)
_SUPERVISOR_SPEC = importlib.util.spec_from_file_location(
    "_semtalk_frozen_live_validation_supervisor", _SUPERVISOR_PATH
)
if _SUPERVISOR_SPEC is None or _SUPERVISOR_SPEC.loader is None:
    raise RuntimeError("cannot load the frozen sibling supervisor")
supervisor = importlib.util.module_from_spec(_SUPERVISOR_SPEC)
_SUPERVISOR_SPEC.loader.exec_module(supervisor)


FORMAT = "semtalk_show_base_live_val_remote_ticket_v1"
RECEIPT_FORMAT = "semtalk_show_base_live_val_remote_receipt_v1"
TOPOLOGY_FORMAT = "semtalk_show_base_live_val_dual_host_topology_v1"
DISPATCHER_CLAIM_FORMAT = "semtalk_show_base_live_val_dual_host_claim_v1"
WAVE_FORMAT = "semtalk_show_base_live_val_dual_host_wave_v1"
ACTIVE_LOCK_FORMAT = "semtalk_show_base_live_val_dual_host_active_lock_v1"
OFFICIAL_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
GUARDED_RUNNER_PATH = "/tmp/globaldiff_guarded_runner.py"
GUARD_VERIFIER_PATH = "/tmp/verify_globaldiff_guards.py"
MACHINE_ID_PATH = "/etc/machine-id"
HOST_IDENTITY_PATH = "/sys/devices/virtual/dmi/id/product_uuid"
GPU_LIST = "0,1,2,3,4,5,6,7"
GPU_KEYS = frozenset(str(index) for index in range(8))
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MACHINE_ID = re.compile(rb"[0-9a-f]{32}\n?\Z")
HOST_IDENTITY = re.compile(
    rb"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    rb"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\n?\Z"
)
ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
FORMAL_PYTHON_KEYS = frozenset(
    {
        "format",
        "argv0",
        "venv_root",
        "symlink_chain",
        "resolved_target",
        "pyvenv_cfg",
    }
)
TICKET_KEYS = frozenset(
    {
        "format",
        "status",
        "role",
        "hostname",
        "machine_id",
        "host_identity",
        "candidate_epoch",
        "campaign",
        "topology",
        "endpoint",
        "wave",
        "active_claim",
        "job_claim",
        "authorization",
        "formal_python",
        "source_root",
        "source_origin",
        "source_commit",
        "source_tree",
        "guarded_runner",
        "guard_verifier",
        "runner_argv",
        "runner_status_path",
        "runner_log_path",
        "receipt_path",
        "ticket_payload_sha256",
    }
)
ROLE_KEYS = frozenset(
    {
        "role",
        "hostname",
        "machine_id",
        "host_identity",
        "source_root",
        "source_origin",
        "source_commit",
        "source_tree",
        "formal_python",
        "guarded_runner",
        "guard_verifier",
    }
)
WORKER_ROLE_KEYS = ROLE_KEYS | frozenset({"ssh_host"})
TOPOLOGY_KEYS = frozenset(
    {
        "format",
        "status",
        "campaign",
        "master",
        "worker",
        "ssh_binary",
        "ssh_options",
        "endpoint",
        "dispatcher",
        "created_unix",
        "topology_payload_sha256",
    }
)
DISPATCHER_CLAIM_KEYS = frozenset(
    {
        "format",
        "status",
        "campaign",
        "topology",
        "fresh_candidate_epochs",
        "waves",
        "policy",
        "created_unix",
        "claim_payload_sha256",
    }
)
WAVE_KEYS = frozenset(
    {
        "format",
        "status",
        "campaign",
        "dispatcher_claim",
        "topology",
        "wave_index",
        "candidate_epochs",
        "master_epoch",
        "worker_epoch",
        "supervisor_active_claim",
        "created_unix",
        "wave_payload_sha256",
    }
)
ACTIVE_LOCK_KEYS = frozenset(
    {
        "format",
        "status",
        "campaign",
        "dispatcher_claim",
        "topology",
        "wave",
        "created_unix",
        "lock_payload_sha256",
    }
)
RUNNER_STATUS_KEYS = frozenset(
    {
        "updated_at",
        "state",
        "wrapper_pid",
        "child_pid",
        "return_code",
        "received_signal",
        "error",
        "cleanup_error",
        "restored_guards",
        "restore_error",
        "command",
    }
)
GUARD_PASS_RE = re.compile(
    r"\APASS "
    + " ".join(
        r"GPU%d=PID([2-9]|[1-9][0-9]+)" % index for index in range(8)
    )
    + r"\n\Z"
)


class RemoteEndpointError(RuntimeError):
    """A frozen remote ticket or its execution evidence is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RemoteEndpointError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise RemoteEndpointError("value is not strict canonical JSON") from error


def canonical_json_sha256(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(type(key) is str and key not in value, "duplicate/non-string JSON key")
        value[key] = item
    return value


def strict_json(raw: bytes, label: str, *, canonical: bool) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RemoteEndpointError(f"{label} contains non-finite JSON {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteEndpointError(f"{label} is not strict JSON") from error
    require(type(value) is dict, f"{label} must be a JSON object")
    if canonical:
        require(raw == canonical_json_bytes(value), f"{label} is not canonical JSON")
    return value


def _exact_int(value: Any, label: str, minimum: int) -> int:
    require(type(value) is int and value >= minimum, f"{label} is not an exact integer")
    return value


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _lexical_absolute(path_value: Any, label: str) -> Path:
    if isinstance(path_value, os.PathLike):
        path_value = os.fspath(path_value)
    require(
        type(path_value) is str and path_value == path_value.strip(),
        f"{label} path is not lexical",
    )
    require(path_value.startswith("/"), f"{label} path must be absolute")
    require(
        not any(character.isspace() or ord(character) < 32 for character in path_value),
        f"{label} path contains whitespace/control",
    )
    path = Path(path_value)
    require(
        str(path) == path_value and "." not in path.parts and ".." not in path.parts,
        f"{label} path is not canonical",
    )
    return path


def _no_symlink_components(path: Path, label: str, *, include_leaf: bool) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:]
    limit = len(parts) if include_leaf else max(0, len(parts) - 1)
    for part in parts[:limit]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise RemoteEndpointError(f"cannot lstat {label}: {error}") from error
        require(not stat.S_ISLNK(metadata.st_mode), f"{label} contains a symlink component")


def canonical_directory(path_value: Any, label: str) -> Path:
    path = _lexical_absolute(path_value, label)
    _no_symlink_components(path, label, include_leaf=True)
    metadata = path.lstat()
    require(
        stat.S_ISDIR(metadata.st_mode) and path.resolve(strict=True) == path,
        f"{label} is not a canonical directory",
    )
    return path


def canonical_output_path(path_value: Any, label: str) -> Path:
    path = _lexical_absolute(path_value, label)
    _no_symlink_components(path, label, include_leaf=False)
    require(
        path.parent.resolve(strict=True) == path.parent,
        f"{label} parent is not canonical",
    )
    return path


def safe_regular_bytes(
    path_value: Any,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[Path, bytes, str, int]:
    path = _lexical_absolute(path_value, label)
    _no_symlink_components(path, label, include_leaf=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RemoteEndpointError(f"cannot safely open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"{label} must be a single-link regular file",
        )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        require(_identity(before) == _identity(after), f"{label} changed during read")
        require(len(raw) == before.st_size, f"{label} length changed during read")
        current = os.stat(path, follow_symlinks=False)
        require(_identity(current) == _identity(before), f"{label} pathname changed")
        require(path.resolve(strict=True) == path, f"{label} resolves elsewhere")
    finally:
        os.close(descriptor)
    digest = _sha256(raw)
    if expected_sha256 is not None:
        require(
            type(expected_sha256) is str and HEX64.fullmatch(expected_sha256) is not None,
            f"{label} expected SHA is invalid",
        )
        require(digest == expected_sha256, f"{label} SHA changed")
    if expected_bytes is not None:
        _exact_int(expected_bytes, f"{label} expected bytes", 1)
        require(len(raw) == expected_bytes, f"{label} byte length changed")
    return path, raw, digest, len(raw)


def safe_host_identity_bytes(
    path_value: Any,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[Path, bytes, str, int]:
    """Read the pinned DMI UUID without trusting sysfs ``st_size``.

    The canonical sysfs node reports a virtual size of 4096 while returning a
    36-character UUID plus an optional newline.  All pathname, fd identity,
    link-count, and content checks remain fail-closed; only the regular-file
    helper's ``len(raw) == st_size`` rule is intentionally inapplicable here.
    """

    path = _lexical_absolute(path_value, label)
    require(str(path) == HOST_IDENTITY_PATH, f"{label} path changed")
    _no_symlink_components(path, label, include_leaf=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RemoteEndpointError(f"cannot safely open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"{label} must be a single-link regular file",
        )
        raw = os.read(descriptor, 128)
        require(os.read(descriptor, 1) == b"", f"{label} is oversized")
        after = os.fstat(descriptor)
        require(_identity(before) == _identity(after), f"{label} changed during read")
        current = os.stat(path, follow_symlinks=False)
        require(_identity(current) == _identity(before), f"{label} pathname changed")
        require(path.resolve(strict=True) == path, f"{label} resolves elsewhere")
        require(HOST_IDENTITY.fullmatch(raw) is not None, f"{label} content changed")
    finally:
        os.close(descriptor)
    digest = _sha256(raw)
    if expected_sha256 is not None:
        require(
            type(expected_sha256) is str and HEX64.fullmatch(expected_sha256) is not None,
            f"{label} expected SHA is invalid",
        )
        require(digest == expected_sha256, f"{label} SHA changed")
    if expected_bytes is not None:
        _exact_int(expected_bytes, f"{label} expected bytes", 36)
        require(expected_bytes <= 37, f"{label} expected bytes changed")
        require(len(raw) == expected_bytes, f"{label} byte length changed")
    return path, raw, digest, len(raw)


def host_identity_artifact(value: Any, label: str) -> dict[str, Any]:
    require(
        type(value) is dict and set(value) == ARTIFACT_KEYS,
        f"{label} artifact schema changed",
    )
    digest = value.get("sha256")
    size = value.get("bytes")
    require(
        type(digest) is str and HEX64.fullmatch(digest) is not None,
        f"{label} SHA is invalid",
    )
    _exact_int(size, f"{label} bytes", 36)
    require(size <= 37, f"{label} bytes changed")
    path, _raw, observed_sha, observed_size = safe_host_identity_bytes(
        value.get("path"), label, digest, size
    )
    return {"path": str(path), "sha256": observed_sha, "bytes": observed_size}


def artifact(value: Any, label: str, *, executable: bool = False) -> dict[str, Any]:
    require(type(value) is dict and set(value) == ARTIFACT_KEYS, f"{label} artifact schema changed")
    digest = value.get("sha256")
    size = value.get("bytes")
    require(type(digest) is str and HEX64.fullmatch(digest) is not None, f"{label} SHA is invalid")
    _exact_int(size, f"{label} bytes", 1)
    path, _raw, observed_sha, observed_size = safe_regular_bytes(
        value.get("path"), label, digest, size
    )
    if executable:
        require(path.stat().st_mode & 0o111 != 0, f"{label} is not executable")
    return {"path": str(path), "sha256": observed_sha, "bytes": observed_size}


def _artifact_from_read(
    path: Path, raw: bytes, digest: str, size: int
) -> dict[str, Any]:
    del raw
    return {"path": str(path), "sha256": digest, "bytes": size}


def read_canonical_artifact(
    value: Any, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = artifact(value, label)
    _path, raw, _digest, _size = safe_regular_bytes(
        normalized["path"], label, normalized["sha256"], normalized["bytes"]
    )
    return normalized, strict_json(raw, label, canonical=True)


def _validate_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    require(
        type(claimed) is str and HEX64.fullmatch(claimed) is not None,
        f"{label} self-hash is invalid",
    )
    require(canonical_json_sha256(body) == claimed, f"{label} self-hash changed")


def _add_self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    require(field not in result, "self-hash field already exists")
    result[field] = canonical_json_sha256(result)
    return result


def _finite_positive(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    normalized = float(value)
    require(math.isfinite(normalized) and normalized > 0, f"{label} is invalid")
    return normalized


def _artifact_descriptor(value: Any, label: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == ARTIFACT_KEYS, f"{label} schema changed")
    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    require(type(path) is str and Path(path).is_absolute(), f"{label} path changed")
    require(type(digest) is str and HEX64.fullmatch(digest) is not None, f"{label} SHA changed")
    require(type(size) is int and size > 0, f"{label} size changed")
    return dict(value)


def _fresh_waves() -> list[list[int]]:
    fresh = list(supervisor.CANDIDATE_EPOCHS[1:])
    return [fresh[index : index + 2] for index in range(0, len(fresh), 2)]


def _validate_campaign_document(
    campaign_artifact: Mapping[str, Any], campaign: Mapping[str, Any],
    ticket: Mapping[str, Any], epoch: int,
) -> Mapping[str, Any]:
    require(set(campaign) == supervisor.ADOPTION_CAMPAIGN_KEYS, "campaign schema changed")
    _validate_self_hash(campaign, "campaign_payload_sha256", "campaign")
    require(
        campaign.get("format") == supervisor.ADOPTION_CAMPAIGN_FORMAT
        and campaign.get("status") == "frozen_before_execution"
        and campaign.get("split") == "val"
        and campaign.get("test_visible") is False
        and type(campaign.get("test_measurements_authorized")) is int
        and campaign["test_measurements_authorized"] == 0
        and campaign.get("candidate_epochs") == list(supervisor.CANDIDATE_EPOCHS),
        "campaign protocol changed",
    )
    jobs = campaign.get("jobs")
    require(
        type(jobs) is list
        and len(jobs) == len(supervisor.CANDIDATE_EPOCHS),
        "campaign jobs changed",
    )
    selected: Mapping[str, Any] | None = None
    for expected_epoch, job in zip(supervisor.CANDIDATE_EPOCHS, jobs):
        require(
            type(job) is dict
            and set(job) == supervisor.JOB_KEYS
            and type(job.get("epoch")) is int
            and job["epoch"] == expected_epoch,
            "campaign job order/schema changed",
        )
        if expected_epoch == epoch:
            selected = job
    require(selected is not None, "candidate epoch is outside the campaign")
    control = campaign.get("control_source")
    require(
        type(control) is dict
        and set(control) == supervisor.CONTROL_SOURCE_KEYS
        and control.get("root") == ticket.get("source_root")
        and control.get("origin") == OFFICIAL_ORIGIN
        and control.get("commit") == ticket.get("source_commit")
        and control.get("tree") == ticket.get("source_tree"),
        "campaign source binding changed",
    )
    supervisor_artifact = artifact(
        control.get("supervisor"), "campaign frozen supervisor"
    )
    require(
        supervisor_artifact["path"] == str(_SUPERVISOR_PATH)
        and _SUPERVISOR_PATH
        == Path(ticket["source_root"])
        / "scripts/show_base/supervise_base_v14_live_validation.py",
        "campaign frozen supervisor path changed",
    )
    require(
        strict_equal(campaign.get("formal_python"), ticket.get("formal_python"))
        and strict_equal(campaign.get("guarded_runner"), ticket.get("guarded_runner"))
        and strict_equal(campaign.get("guard_verifier"), ticket.get("guard_verifier")),
        "campaign runtime binding changed",
    )
    require(
        campaign.get("state_root") == str(Path(campaign_artifact["path"]).parent),
        "campaign state root changed",
    )
    artifact(campaign.get("adopted_e1"), "adopted e1 receipt")
    return selected


def _validate_dispatch_authority(
    *, ticket: Mapping[str, Any], inputs: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, Mapping[str, Any]], campaign_job: Mapping[str, Any],
    epoch: int, hostname: str, machine: Mapping[str, Any],
    host_identity: Mapping[str, Any],
) -> None:
    state_root = Path(values["campaign"]["state_root"])
    dual_root = state_root / "dual_dispatch"
    topology = values["topology"]
    require(set(topology) == TOPOLOGY_KEYS, "topology schema changed")
    _validate_self_hash(topology, "topology_payload_sha256", "topology")
    require(
        topology.get("format") == TOPOLOGY_FORMAT
        and topology.get("status") == "frozen_before_execution"
        and strict_equal(topology.get("campaign"), inputs["campaign"]),
        "topology state/campaign changed",
    )
    require(
        inputs["topology"]["path"] == str(dual_root / "topology.json"),
        "topology path changed",
    )
    endpoint_artifact = artifact(ticket.get("endpoint"), "ticket endpoint")
    live_endpoint = Path(__file__).resolve(strict=True)
    require(endpoint_artifact["path"] == str(live_endpoint), "ticket endpoint path changed")
    require(
        strict_equal(topology.get("endpoint"), endpoint_artifact),
        "topology endpoint binding changed",
    )
    source_root = Path(ticket["source_root"])
    require(
        live_endpoint == source_root / "scripts/show_base/base_live_val_remote_endpoint.py",
        "endpoint is outside the frozen source",
    )
    dispatcher = artifact(topology.get("dispatcher"), "topology dispatcher")
    require(
        dispatcher["path"]
        == str(source_root / "scripts/show_base/dispatch_base_v14_live_validation_dual_host.py"),
        "topology dispatcher path changed",
    )
    ssh_binary = artifact(topology.get("ssh_binary"), "topology SSH binary", executable=True)
    require(
        ssh_binary["path"] == "/usr/bin/ssh"
        and topology.get("ssh_options")
        == [
            "-T", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
            "-o", "ExitOnForwardFailure=yes", "-o", "LogLevel=ERROR",
            "-o", "RequestTTY=no", "--",
        ],
        "topology SSH contract changed",
    )
    roles: dict[str, Mapping[str, Any]] = {}
    for role in ("master", "worker"):
        value = topology.get(role)
        expected_keys = ROLE_KEYS if role == "master" else WORKER_ROLE_KEYS
        require(type(value) is dict and set(value) == expected_keys, f"{role} topology schema changed")
        require(value.get("role") == role, f"{role} topology role changed")
        role_machine = _artifact_descriptor(value.get("machine_id"), f"{role} machine-id")
        role_host_identity = _artifact_descriptor(
            value.get("host_identity"), f"{role} host identity"
        )
        require(
            role_machine["path"] == MACHINE_ID_PATH
            and role_host_identity["path"] == HOST_IDENTITY_PATH
            and value.get("source_root") == ticket.get("source_root")
            and value.get("source_origin") == OFFICIAL_ORIGIN
            and value.get("source_commit") == ticket.get("source_commit")
            and value.get("source_tree") == ticket.get("source_tree")
            and strict_equal(value.get("formal_python"), ticket.get("formal_python"))
            and strict_equal(value.get("guarded_runner"), ticket.get("guarded_runner"))
            and strict_equal(value.get("guard_verifier"), ticket.get("guard_verifier")),
            f"{role} topology binding changed",
        )
        roles[role] = value
        if role == "worker":
            live_host_identity = host_identity_artifact(
                role_host_identity, "worker host identity replay"
            )
            _identity_path, identity_raw, _identity_sha, _identity_size = safe_host_identity_bytes(
                live_host_identity["path"],
                "worker host identity content",
                live_host_identity["sha256"],
                live_host_identity["bytes"],
            )
            require(
                HOST_IDENTITY.fullmatch(identity_raw) is not None,
                "worker host identity content changed",
            )
    require(
        roles["worker"].get("hostname") == hostname
        and strict_equal(roles["worker"].get("machine_id"), machine)
        and strict_equal(roles["worker"].get("host_identity"), host_identity)
        and type(roles["worker"].get("ssh_host")) is str
        and bool(roles["worker"]["ssh_host"]),
        "worker topology identity changed",
    )
    require(
        roles["master"].get("hostname") != hostname
        and roles["master"]["host_identity"]["sha256"]
        != roles["worker"]["host_identity"]["sha256"],
        "dual-host topology collapsed onto one machine",
    )
    _finite_positive(topology.get("created_unix"), "topology creation time")

    wave = values["wave"]
    require(set(wave) == WAVE_KEYS, "wave schema changed")
    _validate_self_hash(wave, "wave_payload_sha256", "wave")
    wave_index = wave.get("wave_index")
    waves = _fresh_waves()
    require(
        wave.get("format") == WAVE_FORMAT
        and wave.get("status") == "active"
        and type(wave_index) is int
        and 1 <= wave_index <= len(waves)
        and wave.get("candidate_epochs") == waves[wave_index - 1]
        and len(wave["candidate_epochs"]) == 2
        and wave.get("master_epoch") == wave["candidate_epochs"][0]
        and wave.get("worker_epoch") == epoch == wave["candidate_epochs"][1]
        and strict_equal(wave.get("campaign"), inputs["campaign"])
        and strict_equal(wave.get("topology"), inputs["topology"]),
        "wave authority changed",
    )
    wave_directory = dual_root / "waves" / ("wave-%04d" % wave_index)
    require(
        inputs["wave"]["path"] == str(wave_directory / "wave.claim.json"),
        "wave path changed",
    )
    _finite_positive(wave.get("created_unix"), "wave creation time")

    dispatcher_artifact, dispatcher_claim = read_canonical_artifact(
        wave.get("dispatcher_claim"), "dispatcher claim"
    )
    require(
        dispatcher_artifact["path"] == str(dual_root / "claim.json"),
        "dispatcher claim path changed",
    )
    require(set(dispatcher_claim) == DISPATCHER_CLAIM_KEYS, "dispatcher claim schema changed")
    _validate_self_hash(dispatcher_claim, "claim_payload_sha256", "dispatcher claim")
    expected_policy = {
        "adopted_candidates": 1,
        "fresh_candidates": 21,
        "paired_fresh_candidates": 20,
        "terminal_singleton_candidates": 1,
        "max_active_waves": 1,
        "retry_authorized": False,
        "rerun_authorized": False,
        "publish_completions_after_full_wave_validation": True,
    }
    require(
        dispatcher_claim.get("format") == DISPATCHER_CLAIM_FORMAT
        and dispatcher_claim.get("status") == "claimed"
        and strict_equal(dispatcher_claim.get("campaign"), inputs["campaign"])
        and strict_equal(dispatcher_claim.get("topology"), inputs["topology"])
        and dispatcher_claim.get("fresh_candidate_epochs")
        == list(supervisor.CANDIDATE_EPOCHS[1:])
        and dispatcher_claim.get("waves") == waves
        and strict_equal(dispatcher_claim.get("policy"), expected_policy)
        and strict_equal(wave.get("dispatcher_claim"), dispatcher_artifact),
        "dispatcher claim changed",
    )
    _finite_positive(dispatcher_claim.get("created_unix"), "dispatcher claim time")

    supervisor_active_artifact, supervisor_active = read_canonical_artifact(
        wave.get("supervisor_active_claim"), "supervisor active claim"
    )
    require(
        supervisor_active_artifact["path"]
        == str(state_root / "active_invocation.claim.json"),
        "supervisor active claim path changed",
    )
    require(set(supervisor_active) == supervisor.ACTIVE_CLAIM_KEYS, "supervisor active claim schema changed")
    _validate_self_hash(supervisor_active, "claim_payload_sha256", "supervisor active claim")
    require(
        supervisor_active.get("format") == supervisor.ACTIVE_CLAIM_FORMAT
        and supervisor_active.get("status") == "active"
        and supervisor_active.get("operation") == f"dual-host-wave-{wave_index:04d}"
        and strict_equal(supervisor_active.get("campaign"), inputs["campaign"])
        and strict_equal(wave.get("supervisor_active_claim"), supervisor_active_artifact),
        "supervisor active claim changed",
    )
    _finite_positive(supervisor_active.get("created_unix"), "supervisor active claim time")

    active = values["active_claim"]
    require(
        inputs["active_claim"]["path"]
        == str(dual_root / "active-wave.lock.json"),
        "active wave claim path changed",
    )
    require(set(active) == ACTIVE_LOCK_KEYS, "active wave claim schema changed")
    _validate_self_hash(active, "lock_payload_sha256", "active wave claim")
    require(
        active.get("format") == ACTIVE_LOCK_FORMAT
        and active.get("status") == "active"
        and strict_equal(active.get("campaign"), inputs["campaign"])
        and strict_equal(active.get("dispatcher_claim"), dispatcher_artifact)
        and strict_equal(active.get("topology"), inputs["topology"])
        and strict_equal(active.get("wave"), inputs["wave"]),
        "active wave claim changed",
    )
    _finite_positive(active.get("created_unix"), "active wave claim time")

    job = values["job_claim"]
    require(
        inputs["job_claim"]["path"]
        == str(state_root / "job_claims" / ("epoch-%04d.json" % epoch)),
        "job claim path changed",
    )
    require(set(job) == supervisor.JOB_CLAIM_KEYS, "job claim schema changed")
    _validate_self_hash(job, "claim_payload_sha256", "job claim")
    candidate_receipt = artifact(job.get("candidate_receipt"), "candidate receipt")
    require(
        job.get("format") == supervisor.JOB_CLAIM_FORMAT
        and job.get("status") == "claimed"
        and job.get("candidate_epoch") == epoch
        and strict_equal(job.get("campaign"), inputs["campaign"])
        and job.get("authority_path") == campaign_job.get("authority_path")
        and job.get("run_root") == campaign_job.get("run_root")
        and job.get("measurement_path") == campaign_job.get("measurement_path"),
        "job claim authority changed",
    )
    _finite_positive(job.get("created_unix"), "job claim time")
    require(
        type(job.get("authorize_argv_sha256")) is str
        and HEX64.fullmatch(job["authorize_argv_sha256"]) is not None,
        "job authorize argv SHA changed",
    )

    authorization = values["authorization"]
    require(
        inputs["authorization"]["path"]
        == str(state_root / "authorizations" / ("epoch-%04d.json" % epoch)),
        "authorization path changed",
    )
    require(set(authorization) == supervisor.AUTHORIZATION_KEYS, "authorization schema changed")
    _validate_self_hash(authorization, "receipt_payload_sha256", "authorization")
    work_authority = artifact(authorization.get("work_authority"), "work authority")
    adapter_argv = authorization.get("adapter_argv")
    require(
        authorization.get("format") == supervisor.AUTHORIZATION_FORMAT
        and authorization.get("status") == "complete"
        and authorization.get("candidate_epoch") == epoch
        and strict_equal(authorization.get("campaign"), inputs["campaign"])
        and strict_equal(authorization.get("candidate_receipt"), candidate_receipt)
        and work_authority["path"] == job.get("authority_path")
        and authorization.get("adapter")
        == values["campaign"]["control_source"]["authority_adapter"]
        and strict_equal(authorization.get("runner_argv"), ticket.get("runner_argv"))
        and type(adapter_argv) is list
        and all(type(token) is str and token and "\0" not in token for token in adapter_argv)
        and _sha256(b"\0".join(os.fsencode(token) for token in adapter_argv) + b"\0")
        == job["authorize_argv_sha256"],
        "authorization authority changed",
    )
    require(
        type(authorization.get("adapter_stdout_sha256")) is str
        and HEX64.fullmatch(authorization["adapter_stdout_sha256"]) is not None,
        "authorization adapter stdout SHA changed",
    )
    _finite_positive(authorization.get("completed_unix"), "authorization completion time")


def write_new_json(path_value: Any, value: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = canonical_output_path(path_value, label)
    require(not os.path.lexists(path), f"{label} already exists")
    raw = canonical_json_bytes(dict(value))
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"{label} short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o400,
        f"{label} did not seal",
    )
    replay_path, replay_raw, digest, size = safe_regular_bytes(path, label)
    require(replay_raw == raw, f"{label} changed after write")
    parent_descriptor = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return {"path": str(replay_path), "sha256": digest, "bytes": size}


def _normalize_link_target(link: Path, target: str) -> Path:
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = link.parent / candidate
    normalized = Path(os.path.normpath(str(candidate)))
    require(normalized.is_absolute(), "formal Python symlink target is not absolute")
    return normalized


def validate_formal_python(value: Any) -> dict[str, Any]:
    require(
        type(value) is dict and set(value) == FORMAL_PYTHON_KEYS,
        "formal Python schema changed",
    )
    binding = dict(value)
    require(
        binding.get("format") == "semtalk.formal_venv_python_binding.v1",
        "formal Python format changed",
    )
    argv0 = _lexical_absolute(binding.get("argv0"), "formal Python argv0")
    require(
        re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", argv0.name) is not None,
        "formal Python leaf changed",
    )
    venv_root = canonical_directory(binding.get("venv_root"), "formal Python venv root")
    require(argv0.parent == venv_root / "bin", "formal Python is outside venv/bin")
    chain = binding.get("symlink_chain")
    require(type(chain) is list and 1 <= len(chain) <= 16, "formal Python symlink chain changed")
    current = argv0
    seen: set[str] = set()
    normalized_chain: list[dict[str, str]] = []
    for row in chain:
        require(type(row) is dict and set(row) == {"path", "target"}, "formal Python symlink row changed")
        require(row.get("path") == str(current) and str(current) not in seen, "formal Python symlink order changed")
        seen.add(str(current))
        before = current.lstat()
        require(stat.S_ISLNK(before.st_mode), "formal Python chain hop is not a symlink")
        target = os.readlink(current)
        after = current.lstat()
        require(_identity(before) == _identity(after) and target == row.get("target"), "formal Python symlink changed")
        normalized_chain.append({"path": str(current), "target": target})
        current = _normalize_link_target(current, target)
    require(
        not current.is_symlink() and argv0.resolve(strict=True) == current,
        "formal Python chain does not end at one file",
    )
    resolved = artifact(binding.get("resolved_target"), "formal Python resolved target", executable=True)
    require(resolved["path"] == str(current), "formal Python resolved target changed")
    cfg = artifact(binding.get("pyvenv_cfg"), "formal Python pyvenv.cfg")
    require(cfg["path"] == str(venv_root / "pyvenv.cfg"), "formal Python pyvenv.cfg path changed")
    binding.update(
        {
            "argv0": str(argv0),
            "venv_root": str(venv_root),
            "symlink_chain": normalized_chain,
            "resolved_target": resolved,
            "pyvenv_cfg": cfg,
        }
    )
    return binding


def _git_capture(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )


def _git_stdout(root: Path, *arguments: str) -> str:
    completed = _git_capture(root, *arguments)
    require(
        completed.returncode == 0 and completed.stderr == "",
        "source Git query failed",
    )
    return completed.stdout.rstrip("\n")


def validate_source(root_value: Any, commit: Any, tree: Any) -> Path:
    root = canonical_directory(root_value, "source root")
    require(type(commit) is str and HEX40.fullmatch(commit) is not None, "source commit is invalid")
    require(type(tree) is str and HEX40.fullmatch(tree) is not None, "source tree is invalid")
    require(_git_stdout(root, "remote") == "origin", "source remotes changed")
    require(
        _git_stdout(root, "remote", "get-url", "origin") == OFFICIAL_ORIGIN
        and _git_stdout(root, "remote", "get-url", "--push", "origin") == OFFICIAL_ORIGIN,
        "source origin changed",
    )
    require(
        _git_stdout(root, "rev-parse", "HEAD") == commit
        and _git_stdout(root, "rev-parse", "HEAD^{tree}") == tree,
        "source commit/tree changed",
    )
    symbolic = _git_capture(root, "symbolic-ref", "-q", "--short", "HEAD")
    require(
        symbolic.returncode == 1 and symbolic.stdout == "" and symbolic.stderr == "",
        "source is not detached",
    )
    require(
        _git_stdout(root, "status", "--porcelain=v1", "--untracked-files=all") == ""
        and _git_stdout(root, "for-each-ref", "--format=%(refname)", "refs/heads") == "",
        "source is not clean zero-branch",
    )
    launcher_relative = "scripts/show_base/run_base_live_val_8shard.sh"
    launcher = root / launcher_relative
    launcher_path, _raw, _sha, _size = safe_regular_bytes(launcher, "tracked launcher")
    require(launcher_path.stat().st_mode & 0o111 != 0, "tracked launcher is not executable")
    require(
        _git_stdout(root, "ls-files", "--error-unmatch", "--", launcher_relative)
        == launcher_relative,
        "live launcher is not tracked",
    )
    index = _git_stdout(root, "ls-files", "-s", "--", launcher_relative)
    require(index.startswith("100755 ") and index.endswith("\t" + launcher_relative), "live launcher Git mode changed")
    return root


def _workload_value(argv: Sequence[str], option: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == option]
    require(len(positions) == 1 and positions[0] + 1 < len(argv), f"workload option {option} changed")
    return argv[positions[0] + 1]


def validate_runner_argv(
    value: Any,
    *,
    formal: Mapping[str, Any],
    source_root: Path,
    source_commit: str,
    source_tree: str,
    runner: Mapping[str, Any],
    status_path: Path,
    log_path: Path,
    authorization_value: Mapping[str, Any],
    job_value: Mapping[str, Any],
    campaign_value: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    require(
        type(value) is list
        and all(type(token) is str and token and "\0" not in token for token in value),
        "runner argv changed",
    )
    argv = list(value)
    require(len(argv) >= 35, "runner argv is incomplete")
    try:
        delimiter = argv.index("--")
    except ValueError as error:
        raise RemoteEndpointError("runner argv lacks the workload delimiter") from error
    require(argv.count("--") == 1, "runner argv delimiter changed")
    expected_outer = [
        formal["argv0"],
        runner["path"],
        "--gpus",
        GPU_LIST,
        "--cwd",
        str(source_root),
        "--status",
        str(status_path),
        "--log",
        str(log_path),
        "--",
    ]
    require(argv[: delimiter + 1] == expected_outer, "outer guarded-runner argv changed")
    workload = argv[delimiter + 1 :]
    launcher = str(source_root / "scripts/show_base/run_base_live_val_8shard.sh")
    require(
        workload[:2] == ["/bin/bash", launcher],
        "runner workload is not the exact tracked Bash launcher",
    )
    option_names = [
        "--repo-root",
        "--python",
        "--work-authority",
        "--expected-work-authority-sha256",
        "--run-root",
        "--source-commit",
        "--source-tree",
        "--paspa-root",
        "--diffsheg-root",
        "--seed",
        "--diffsheg-batch-size",
    ]
    require(
        len(workload) == 2 + 2 * len(option_names)
        and workload[2::2] == option_names,
        "tracked launcher option order changed",
    )
    require(_workload_value(workload, "--repo-root") == str(source_root), "workload source root changed")
    require(_workload_value(workload, "--python") == formal["argv0"], "workload formal Python changed")
    require(_workload_value(workload, "--source-commit") == source_commit, "workload source commit changed")
    require(_workload_value(workload, "--source-tree") == source_tree, "workload source tree changed")
    work_authority = artifact(authorization_value.get("work_authority"), "authorized work authority")
    require(
        _workload_value(workload, "--work-authority") == work_authority["path"]
        and _workload_value(workload, "--expected-work-authority-sha256")
        == work_authority["sha256"],
        "workload work authority changed",
    )
    run_root = canonical_output_path(_workload_value(workload, "--run-root"), "run root")
    require(
        job_value.get("run_root") == str(run_root),
        "workload run root differs from job claim",
    )
    require(
        not os.path.lexists(run_root)
        and not run_root.is_relative_to(source_root),
        "run root must be one new path outside the source checkout",
    )
    for option, campaign_key, label in (
        ("--paspa-root", "paspa_root", "PASPA root"),
        ("--diffsheg-root", "diffsheg_root", "DiffSHEG root"),
    ):
        expected = canonical_directory(campaign_value.get(campaign_key), label)
        require(_workload_value(workload, option) == str(expected), f"workload {label} changed")
    seed = _workload_value(workload, "--seed")
    batch = _workload_value(workload, "--diffsheg-batch-size")
    require(seed.isdigit() and type(campaign_value.get("seed")) is int and int(seed) == campaign_value["seed"] >= 0, "workload seed changed")
    require(batch.isdigit() and type(campaign_value.get("diffsheg_batch_size")) is int and int(batch) == campaign_value["diffsheg_batch_size"] >= 1, "workload batch size changed")
    authorization_argv = authorization_value.get("runner_argv")
    require(strict_equal(authorization_argv, argv), "authorization runner argv changed")
    return argv, workload, work_authority


def validate_runner_status(
    path: Path, workload: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    status_path, raw, digest, size = safe_regular_bytes(path, "runner status")
    value = strict_json(raw, "runner status", canonical=False)
    require(set(value) == RUNNER_STATUS_KEYS, "runner status schema changed")
    require(
        value.get("state") == "finished"
        and type(value.get("return_code")) is int
        and value["return_code"] == 0,
        "guarded runner did not finish successfully",
    )
    for key in ("received_signal", "error", "cleanup_error", "restore_error"):
        require(value.get(key) is None, f"runner status {key} is not explicit null")
    require(strict_equal(value.get("command"), list(workload)), "runner status command changed")
    wrapper = _exact_int(value.get("wrapper_pid"), "runner wrapper PID", 2)
    child = _exact_int(value.get("child_pid"), "runner child PID", 2)
    require(wrapper != child, "runner wrapper/child PIDs collide")
    updated = value.get("updated_at")
    require(
        (type(updated) is str and bool(updated.strip()))
        or (type(updated) in (int, float) and math.isfinite(float(updated))),
        "runner updated_at changed",
    )
    restored = value.get("restored_guards")
    require(type(restored) is dict and set(restored) == GPU_KEYS, "runner guards changed")
    require(
        all(type(restored[str(index)]) is int and restored[str(index)] > 1 for index in range(8))
        and len(set(restored.values())) == 8,
        "runner guard PIDs changed",
    )
    require(wrapper not in restored.values() and child not in restored.values(), "runner PID collides with a guard")
    return _artifact_from_read(status_path, raw, digest, size), value


def validate_runner_log(
    path: Path,
    *,
    job_value: Mapping[str, Any],
    authorization_value: Mapping[str, Any],
    epoch: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    log_path, raw, digest, size = safe_regular_bytes(path, "runner log")
    require(size <= supervisor.MAX_RUNNER_LOG_BYTES, "runner log exceeds bound")
    fields = raw.split(b"\0")
    require(
        len(fields) == 4 and fields[-1] == b"" and all(fields[:3]),
        "runner log is not one exact trailing-NUL triple",
    )
    try:
        measurement_path, measurement_sha, payload_sha = (
            field.decode("ascii") for field in fields[:3]
        )
    except UnicodeDecodeError as error:
        raise RemoteEndpointError("runner log triple is not ASCII") from error
    require(
        measurement_path == job_value.get("measurement_path")
        and HEX64.fullmatch(measurement_sha) is not None
        and HEX64.fullmatch(payload_sha) is not None,
        "runner log triple changed",
    )
    work_authority = artifact(
        authorization_value.get("work_authority"), "runner-log work authority"
    )
    try:
        measurement, measurement_value = supervisor._measurement(
            {
                "measurement_path": measurement_path,
                "epoch": epoch,
                "work_authority": work_authority,
            },
            measurement_sha,
            payload_sha,
        )
    except supervisor.SupervisorError as error:
        raise RemoteEndpointError(f"runner measurement rejected: {error}") from error
    return (
        _artifact_from_read(log_path, raw, digest, size),
        measurement,
        measurement_value,
    )


def _verify_guards(
    formal: Mapping[str, Any],
    verifier: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> tuple[list[str], str]:
    argv = [
        formal["argv0"],
        verifier["path"],
        *[str(restored[str(index)]) for index in range(8)],
    ]
    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        argv,
        shell=False,
        check=False,
        capture_output=True,
        env=environment,
    )
    require(
        type(completed.returncode) is int
        and completed.returncode == 0
        and completed.stderr == b""
        and len(completed.stdout) <= 1024,
        "guard verifier failed",
    )
    try:
        text = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise RemoteEndpointError("guard verifier stdout is not ASCII") from error
    match = GUARD_PASS_RE.fullmatch(text)
    require(match is not None, "guard verifier stdout changed")
    observed = {str(index): int(match.group(index + 1)) for index in range(8)}
    require(strict_equal(observed, dict(restored)), "guard verifier PIDs changed")
    return argv, text


def _artifact_replay(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    return artifact(dict(value), label)


def run_ticket(
    ticket_path_value: Any,
    expected_ticket_sha256: str,
    expected_ticket_bytes: int,
    *,
    hostname_provider: Callable[[], str] = socket.gethostname,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    ticket_path, ticket_raw, ticket_sha, ticket_size = safe_regular_bytes(
        ticket_path_value,
        "remote ticket",
        expected_ticket_sha256,
        expected_ticket_bytes,
    )
    ticket_artifact = _artifact_from_read(ticket_path, ticket_raw, ticket_sha, ticket_size)
    ticket = strict_json(ticket_raw, "remote ticket", canonical=True)
    require(set(ticket) == TICKET_KEYS, "remote ticket schema changed")
    _validate_self_hash(ticket, "ticket_payload_sha256", "remote ticket")
    require(
        ticket.get("format") == FORMAT
        and ticket.get("status") == "authorized"
        and ticket.get("role") == "worker",
        "remote ticket role/state changed",
    )
    hostname = hostname_provider()
    require(type(hostname) is str and hostname and ticket.get("hostname") == hostname, "worker hostname changed")
    epoch = _exact_int(ticket.get("candidate_epoch"), "candidate epoch", 1)

    machine = artifact(ticket.get("machine_id"), "machine-id")
    require(machine["path"] == MACHINE_ID_PATH, "machine-id path changed")
    _machine_path, machine_raw, _machine_sha, _machine_size = safe_regular_bytes(
        machine["path"], "machine-id replay", machine["sha256"], machine["bytes"]
    )
    require(MACHINE_ID.fullmatch(machine_raw) is not None, "machine-id content changed")

    host_identity = host_identity_artifact(
        ticket.get("host_identity"), "worker host identity"
    )

    inputs: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, Any]] = {}
    for key, label in (
        ("campaign", "campaign"),
        ("topology", "topology"),
        ("wave", "wave"),
        ("active_claim", "active claim"),
        ("job_claim", "job claim"),
        ("authorization", "authorization"),
    ):
        inputs[key], values[key] = read_canonical_artifact(ticket.get(key), label)

    campaign_job = _validate_campaign_document(
        inputs["campaign"], values["campaign"], ticket, epoch
    )
    _validate_dispatch_authority(
        ticket=ticket,
        inputs=inputs,
        values=values,
        campaign_job=campaign_job,
        epoch=epoch,
        hostname=hostname,
        machine=machine,
        host_identity=host_identity,
    )

    require(values["job_claim"].get("candidate_epoch") == epoch, "job claim epoch changed")
    require(values["authorization"].get("candidate_epoch") == epoch, "authorization epoch changed")
    for key in ("active_claim", "job_claim", "authorization"):
        if "campaign" in values[key]:
            require(strict_equal(values[key]["campaign"], inputs["campaign"]), f"{key} campaign binding changed")
    if "campaign" in values["wave"]:
        require(strict_equal(values["wave"]["campaign"], inputs["campaign"]), "wave campaign binding changed")

    formal = validate_formal_python(ticket.get("formal_python"))
    require(
        strict_equal(values["campaign"].get("formal_python"), formal),
        "campaign formal Python binding changed",
    )
    source_origin = ticket.get("source_origin")
    require(source_origin == OFFICIAL_ORIGIN, "ticket source origin changed")
    source_commit = ticket.get("source_commit")
    source_tree = ticket.get("source_tree")
    source_root = validate_source(ticket.get("source_root"), source_commit, source_tree)
    control = values["campaign"].get("control_source")
    require(
        type(control) is dict
        and control.get("root") == str(source_root)
        and control.get("origin") == OFFICIAL_ORIGIN
        and control.get("commit") == source_commit
        and control.get("tree") == source_tree,
        "campaign source binding changed",
    )

    runner = artifact(ticket.get("guarded_runner"), "guarded runner", executable=True)
    verifier = artifact(ticket.get("guard_verifier"), "guard verifier")
    require(runner["path"] == GUARDED_RUNNER_PATH, "guarded runner path changed")
    require(verifier["path"] == GUARD_VERIFIER_PATH, "guard verifier path changed")
    require(strict_equal(values["campaign"].get("guarded_runner"), runner), "campaign guarded runner changed")
    require(strict_equal(values["campaign"].get("guard_verifier"), verifier), "campaign guard verifier changed")

    status_path = canonical_output_path(ticket.get("runner_status_path"), "runner status")
    log_path = canonical_output_path(ticket.get("runner_log_path"), "runner log")
    receipt_path = canonical_output_path(ticket.get("receipt_path"), "endpoint receipt")
    require(
        str(status_path) == campaign_job.get("runner_status_path")
        and str(log_path) == campaign_job.get("runner_log_path")
        and str(receipt_path)
        == str(
            Path(inputs["wave"]["path"]).parent
            / "worker-endpoint-receipt.json"
        ),
        "endpoint output paths changed",
    )
    require(len({str(status_path), str(log_path), str(receipt_path)}) == 3, "endpoint output paths collide")
    require(
        all(
            not output.is_relative_to(source_root)
            for output in (status_path, log_path, receipt_path)
        ),
        "endpoint outputs must remain outside the source checkout",
    )
    require(not os.path.lexists(status_path), "runner status already exists")
    require(not os.path.lexists(log_path), "runner log already exists")
    require(not os.path.lexists(receipt_path), "endpoint receipt already exists")

    runner_argv, workload, work_authority = validate_runner_argv(
        ticket.get("runner_argv"),
        formal=formal,
        source_root=source_root,
        source_commit=source_commit,
        source_tree=source_tree,
        runner=runner,
        status_path=status_path,
        log_path=log_path,
        authorization_value=values["authorization"],
        job_value=values["job_claim"],
        campaign_value=values["campaign"],
    )
    del runner  # The exact path and bytes remain bound by the ticket/receipt.

    # Last admission replay immediately before the only mutating subprocess.
    safe_regular_bytes(ticket_path, "remote ticket replay", ticket_sha, ticket_size)
    for key, label in (
        ("campaign", "campaign replay"),
        ("topology", "topology replay"),
        ("wave", "wave replay"),
        ("active_claim", "active claim replay"),
        ("job_claim", "job claim replay"),
        ("authorization", "authorization replay"),
    ):
        _artifact_replay(inputs[key], label)
    _artifact_replay(machine, "machine-id replay")
    host_identity_artifact(host_identity, "worker host identity replay")
    _artifact_replay(work_authority, "work authority replay")
    _artifact_replay(ticket["endpoint"], "remote endpoint replay")
    _artifact_replay(
        values["campaign"]["control_source"]["supervisor"],
        "frozen supervisor replay",
    )
    _artifact_replay(values["topology"]["dispatcher"], "dual dispatcher replay")
    _artifact_replay(ticket["guarded_runner"], "guarded runner replay")
    _artifact_replay(verifier, "guard verifier replay")
    validate_formal_python(formal)
    validate_source(source_root, source_commit, source_tree)

    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        runner_argv,
        shell=False,
        check=False,
        capture_output=True,
        env=environment,
    )
    require(
        type(completed.returncode) is int
        and completed.returncode == 0
        and completed.stdout == b""
        and completed.stderr == b"",
        "guarded runner process failed or emitted endpoint output",
    )

    status_artifact, status = validate_runner_status(status_path, workload)
    log_artifact, measurement_artifact, measurement_value = validate_runner_log(
        log_path,
        job_value=values["job_claim"],
        authorization_value=values["authorization"],
        epoch=epoch,
    )
    verifier_argv, verifier_stdout = _verify_guards(
        formal, verifier, status["restored_guards"]
    )

    # Re-read every mutable or externally supplied input after the verifier.
    replayed_status, replayed_value = validate_runner_status(status_path, workload)
    require(
        strict_equal(replayed_status, status_artifact)
        and strict_equal(replayed_value, status),
        "runner status changed after guard verification",
    )
    replayed_log, replayed_measurement, replayed_measurement_value = validate_runner_log(
        log_path,
        job_value=values["job_claim"],
        authorization_value=values["authorization"],
        epoch=epoch,
    )
    require(
        strict_equal(replayed_log, log_artifact)
        and strict_equal(replayed_measurement, measurement_artifact)
        and strict_equal(replayed_measurement_value, measurement_value),
        "runner log/measurement changed after guard verification",
    )
    require(strict_equal(_artifact_replay(ticket_artifact, "ticket final replay"), ticket_artifact), "ticket changed after execution")
    for key, label in (
        ("campaign", "campaign final replay"),
        ("topology", "topology final replay"),
        ("wave", "wave final replay"),
        ("active_claim", "active claim final replay"),
        ("job_claim", "job claim final replay"),
        ("authorization", "authorization final replay"),
    ):
        require(strict_equal(_artifact_replay(inputs[key], label), inputs[key]), f"{label} changed")
    require(strict_equal(_artifact_replay(machine, "machine-id final replay"), machine), "machine-id changed")
    require(
        strict_equal(
            host_identity_artifact(host_identity, "worker host identity final replay"),
            host_identity,
        ),
        "worker host identity changed",
    )
    require(strict_equal(_artifact_replay(work_authority, "work authority final replay"), work_authority), "work authority changed")
    require(strict_equal(_artifact_replay(ticket["endpoint"], "remote endpoint final replay"), ticket["endpoint"]), "remote endpoint changed")
    require(strict_equal(_artifact_replay(values["campaign"]["control_source"]["supervisor"], "frozen supervisor final replay"), values["campaign"]["control_source"]["supervisor"]), "frozen supervisor changed")
    require(strict_equal(_artifact_replay(values["topology"]["dispatcher"], "dual dispatcher final replay"), values["topology"]["dispatcher"]), "dual dispatcher changed")
    require(strict_equal(_artifact_replay(ticket["guarded_runner"], "guarded runner final replay"), ticket["guarded_runner"]), "guarded runner changed")
    require(strict_equal(_artifact_replay(verifier, "guard verifier final replay"), verifier), "guard verifier changed")
    validate_formal_python(formal)
    validate_source(source_root, source_commit, source_tree)

    completed_unix = clock()
    require(
        type(completed_unix) in (int, float) and math.isfinite(float(completed_unix)) and float(completed_unix) > 0,
        "completion time changed",
    )
    receipt = _add_self_hash(
        {
            "format": RECEIPT_FORMAT,
            "status": "complete",
            "role": "worker",
            "hostname": hostname,
            "machine_id": machine,
            "host_identity": host_identity,
            "candidate_epoch": epoch,
            "ticket": ticket_artifact,
            "campaign": inputs["campaign"],
            "wave": inputs["wave"],
            "active_claim": inputs["active_claim"],
            "job_claim": inputs["job_claim"],
            "authorization": inputs["authorization"],
            "formal_python": formal,
            "source_root": str(source_root),
            "source_origin": OFFICIAL_ORIGIN,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "guarded_runner": dict(ticket["guarded_runner"]),
            "guard_verifier": verifier,
            "runner_argv": runner_argv,
            "runner_argv_sha256": canonical_json_sha256(runner_argv),
            "runner_status": status_artifact,
            "runner_log": log_artifact,
            "guard_verifier_argv": verifier_argv,
            "guard_verifier_stdout": verifier_stdout,
            "restored_guards": dict(status["restored_guards"]),
            "completed_unix": completed_unix,
        },
        "receipt_payload_sha256",
    )
    return write_new_json(receipt_path, receipt, "endpoint receipt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run exactly one frozen worker ticket")
    run.add_argument("--ticket", required=True)
    run.add_argument("--expected-ticket-sha256", required=True)
    run.add_argument("--expected-ticket-bytes", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = run_ticket(
            arguments.ticket,
            arguments.expected_ticket_sha256,
            arguments.expected_ticket_bytes,
        )
    except (RemoteEndpointError, OSError) as error:
        sys.stderr.write(f"remote endpoint rejected ticket: {error}\n")
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
