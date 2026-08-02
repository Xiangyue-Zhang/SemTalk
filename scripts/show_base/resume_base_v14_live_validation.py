#!/usr/bin/env python3
"""Resume one V14 validation job after a pre-authorization supervisor crash.

This is deliberately narrower than the normal supervisor.  It accepts only
the state in which the pinned supervisor wrote both its active-invocation and
job claims, then exited before creating any authority, authorization, runner,
run-root, log, status, measurement, or completion output.  It imports the
campaign-pinned supervisor by exact path and SHA and reuses its validation,
launch, replay, completion, and guard-restoration primitives.

The recovery tool never deletes or rewrites the durable job claim.  It removes
the active claim only through the pinned supervisor's normal success path,
after the completion receipt has been sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
from types import ModuleType
from typing import Any, Dict, Mapping, Sequence


sys.dont_write_bytecode = True

HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")
ACTIVE_KEYS = frozenset({
    "format", "status", "operation", "campaign", "created_unix",
    "claim_payload_sha256",
})
RECOVERY_START_FORMAT = "semtalk_show_base_v14_manual_resume_start_v1"
RECOVERY_COMPLETE_FORMAT = "semtalk_show_base_v14_manual_resume_complete_v1"


class RecoveryError(RuntimeError):
    """Fail-closed recovery contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def _regular_bytes(path_text: str, label: str) -> tuple[Path, bytes, str, int]:
    require(isinstance(path_text, str) and path_text.startswith("/"), f"{label} path must be absolute")
    path = Path(path_text)
    require(not path.is_symlink(), f"{label} must not be a symlink")
    metadata = path.stat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, f"{label} must be a one-link regular file")
    raw = path.read_bytes()
    return path.resolve(strict=True), raw, hashlib.sha256(raw).hexdigest(), len(raw)


def _load_pinned_supervisor(path_text: str, expected_sha256: str) -> tuple[ModuleType, Dict[str, Any]]:
    require(isinstance(expected_sha256, str) and HEX64.fullmatch(expected_sha256) is not None, "supervisor SHA is invalid")
    path, _raw, digest, size = _regular_bytes(path_text, "pinned supervisor")
    require(digest == expected_sha256, "pinned supervisor SHA changed")
    spec = importlib.util.spec_from_file_location("semtalk_pinned_v14_supervisor", path)
    require(spec is not None and spec.loader is not None, "cannot construct pinned supervisor import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"path": str(path), "sha256": digest, "bytes": size}


def _self_artifact(expected_sha256: str) -> Dict[str, Any]:
    require(isinstance(expected_sha256, str) and HEX64.fullmatch(expected_sha256) is not None, "recovery SHA is invalid")
    path, _raw, digest, size = _regular_bytes(__file__, "recovery source")
    require(digest == expected_sha256, "recovery source SHA changed")
    return {"path": str(path), "sha256": digest, "bytes": size}


def _create_audit_root(path_text: str) -> Path:
    require(isinstance(path_text, str) and path_text.startswith("/"), "audit root path must be absolute")
    path = Path(path_text)
    require(not os.path.lexists(path), "audit root already exists")
    require(path.parent.resolve(strict=True) == path.parent, "audit-root parent is not canonical")
    os.mkdir(path, 0o700)
    metadata = path.stat()
    require(stat.S_ISDIR(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o700, "audit root mode changed")
    return path


def _validate_orphaned_claim(module: ModuleType, campaign: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    module._campaign_claim_v2(campaign, time.time)
    completed = []
    head = None
    gap = False
    for job in campaign["_jobs"]:
        claim_exists = os.path.lexists(module._job_claim_path_v2(campaign, job["epoch"]))
        completion_exists = os.path.lexists(module._completion_path_v2(campaign, job["epoch"]))
        if completion_exists:
            require(claim_exists and not gap, "completed queue is not a strict prefix")
            completed.append(module._load_completion_v2(campaign, job))
            continue
        if claim_exists:
            require(head is None and not gap, "more than one incomplete job claim exists")
            head = job
            gap = True
        else:
            gap = True
            for path_text, label in (
                (job["authority_path"], "work authority"),
                (job["authorization_path"], "authorization"),
                (job["run_root"], "run root"),
                (job["runner_status_path"], "runner status"),
                (job["runner_log_path"], "runner log"),
                (job["completion_path"], "completion"),
            ):
                require(not os.path.lexists(path_text), f"unclaimed e{job['epoch']} {label} already exists")
    require(head is not None, "there is no incomplete claimed job to resume")
    expected_head_index = len(completed)
    require(campaign["_jobs"][expected_head_index]["epoch"] == head["epoch"], "incomplete claim is not the queue head")

    candidate_receipt = module._candidate_receipt_artifact(head)
    job_claim_artifact, job_claim = module._load_job_claim_v2(campaign, head, candidate_receipt)
    active_path = campaign["_state_root"] / "active_invocation.claim.json"
    active_artifact, active = module._read_existing_json(active_path, "orphaned active invocation")
    require(set(active) == ACTIVE_KEYS, "active invocation schema changed")
    module._self_hashed(active, "claim_payload_sha256", "orphaned active invocation")
    require(active.get("format") == module.ACTIVE_CLAIM_FORMAT, "active invocation format changed")
    require(active.get("status") == "active" and active.get("operation") == "run-next", "active invocation state changed")
    require(module.strict_json_equal(active.get("campaign"), campaign["_artifact"]), "active invocation campaign changed")
    module._finite_number(active.get("created_unix"), "active invocation time", 0.000001)

    for path_text, label in (
        (head["authority_path"], "work authority"),
        (head["authorization_path"], "authorization"),
        (head["run_root"], "run root"),
        (head["runner_status_path"], "runner status"),
        (head["runner_log_path"], "runner log"),
        (head["completion_path"], "completion"),
    ):
        require(not os.path.lexists(path_text), f"claimed e{head['epoch']} {label} already exists")
    return head, candidate_receipt, active_artifact, job_claim_artifact


def resume(args: argparse.Namespace) -> Dict[str, Any]:
    recovery_source = _self_artifact(args.expected_recovery_sha256)
    module, supervisor_source = _load_pinned_supervisor(args.supervisor, args.expected_supervisor_sha256)
    campaign = module.load_campaign(args.campaign, args.expected_campaign_sha256, args.expected_campaign_bytes)
    require(module.strict_json_equal(campaign["_control_source"]["supervisor"], supervisor_source), "campaign does not pin this supervisor")
    head, candidate_receipt, active_artifact, job_claim_artifact = _validate_orphaned_claim(module, campaign)

    audit_root = _create_audit_root(args.audit_root)
    start_audit = module.write_new_json(
        audit_root / "resume-start.json",
        module._add_self_hash({
            "format": RECOVERY_START_FORMAT,
            "status": "resume_started",
            "campaign": campaign["_artifact"],
            "candidate_epoch": head["epoch"],
            "candidate_receipt": candidate_receipt,
            "active_claim": active_artifact,
            "job_claim": job_claim_artifact,
            "pinned_supervisor": supervisor_source,
            "recovery_source": recovery_source,
            "verified_absent_outputs": [
                head["authority_path"], head["authorization_path"], head["run_root"],
                head["runner_status_path"], head["runner_log_path"], head["completion_path"],
            ],
            "started_unix": time.time(),
        }, "receipt_payload_sha256"),
        "manual-resume start audit",
    )

    module._revalidate_control(campaign)
    module._runtime_contract_check(campaign, module._run_capture)
    authorize_argv = module._authorize_argv(campaign, head)
    module._load_job_claim_v2(campaign, head, candidate_receipt)
    authorize_rc, authorize_stdout, authorize_stderr = module._run_capture(authorize_argv)
    require(type(authorize_rc) is int and authorize_rc == 0 and authorize_stderr == b"", "authority adapter authorize failed")
    authority = module._artifact_from_path(head["authority_path"], "queue-head work authority")
    adapter_stdout_sha = module._adapter_stdout_artifact(authorize_stdout, authority, "authority adapter authorize")
    module._work_authority_runtime_binding(authority, campaign["_runtime_validation_source"], head["epoch"])
    module._work_authority_candidate_binding(authority, candidate_receipt, head["epoch"])
    require(module.strict_json_equal(module._candidate_receipt_artifact(head), candidate_receipt), "candidate receipt changed during authorization")
    dynamic = dict(head)
    dynamic.update({"candidate_receipt": candidate_receipt, "work_authority": authority})
    runner_argv = module._runner_argv_v2(dynamic, campaign)
    module._validate_runner_argv_v2(runner_argv, dynamic, campaign)
    authorization = module.write_new_json(
        dynamic["authorization_path"],
        module._authorization_body(campaign, dynamic, authorize_argv, adapter_stdout_sha, runner_argv, time.time()),
        "authorization receipt",
    )
    dynamic["authorization"] = authorization
    rc = module._run_process(runner_argv)
    require(type(rc) is int and rc == 0, "guarded runner returned nonzero")
    dynamic["_campaign"] = campaign
    status_artifact, status = module._runner_status(dynamic)
    log_artifact, measurement, measurement_value = module._verify_runner_log(dynamic)
    module._revalidate_control(campaign)
    module._runtime_contract_check(campaign, module._run_capture)
    bridge_replay_argv = module._bridge_replay(campaign, dynamic, measurement, module._run_capture)
    verifier_argv, verifier_stdout, guards = module._guard_verify(campaign, status, module._run_capture)
    status_artifact_2, status_2 = module._runner_status(dynamic)
    require(module.strict_json_equal(status_artifact, status_artifact_2) and module.strict_json_equal(status, status_2), "runner status changed during guard verification")
    module._measurement(dynamic, measurement["sha256"], measurement["receipt_payload_sha256"])
    require(module.strict_json_equal(module._candidate_receipt_artifact(dynamic), candidate_receipt), "candidate receipt changed during validation")
    require(module.strict_json_equal(module._artifact_from_path(dynamic["authority_path"], "queue-head work authority"), authority), "work authority changed during validation")
    completion_value = module._completion_body(
        campaign, dynamic, status_artifact, status, log_artifact, measurement,
        measurement_value, bridge_replay_argv, verifier_argv, verifier_stdout,
        guards, time.time(),
    )
    completion = module.write_new_json(dynamic["completion_path"], completion_value, "completion")
    module._release_active_v2(campaign, active_artifact)
    complete_audit = module.write_new_json(
        audit_root / "resume-complete.json",
        module._add_self_hash({
            "format": RECOVERY_COMPLETE_FORMAT,
            "status": "complete",
            "campaign": campaign["_artifact"],
            "candidate_epoch": head["epoch"],
            "start_audit": start_audit,
            "job_claim": job_claim_artifact,
            "authorization": authorization,
            "completion": completion,
            "validation_diffsheg_fgd": completion_value["validation_diffsheg_fgd"],
            "restored_guards": guards,
            "completed_unix": time.time(),
        }, "receipt_payload_sha256"),
        "manual-resume completion audit",
    )
    return {
        "status": "job_complete",
        "candidate_epoch": head["epoch"],
        "validation_diffsheg_fgd": completion_value["validation_diffsheg_fgd"],
        "completion": completion,
        "recovery_audit": complete_audit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--expected-campaign-sha256", required=True)
    parser.add_argument("--expected-campaign-bytes", required=True, type=int)
    parser.add_argument("--supervisor", required=True)
    parser.add_argument("--expected-supervisor-sha256", required=True)
    parser.add_argument("--expected-recovery-sha256", required=True)
    parser.add_argument("--audit-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = resume(_parser().parse_args(argv))
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RecoveryError, OSError, ValueError) as error:
        sys.stderr.write(f"recovery-error: {error}\n")
        raise SystemExit(2)
