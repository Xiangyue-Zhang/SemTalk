#!/usr/bin/env python3
"""Authorize, execute, and seal the one metric-only e1 adoption repair.

The failed formal-v2 state and run roots are immutable inputs.  ``run`` must
itself be launched by ``/tmp/globaldiff_guarded_runner.py``; it writes only
inside the new repair root and executes the exact frozen 4066 evaluator
followed by the unmodified 73ff bridge ``complete`` command.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

SPEC_FORMAT = "semtalk_show_base_v14_metric_repair_spec_v1"
AUTHORITY_FORMAT = "semtalk_show_base_v14_metric_repair_authority_v1"
RESULT_FORMAT = "semtalk_show_base_v14_metric_repair_result_v1"
ADOPTION_FORMAT = "semtalk_show_base_v14_metric_repair_adoption_receipt_v1"
MEASUREMENT_FORMAT = "semtalk_show_base_live_val_measurement_v2"
EXPECTED_FGD_BINARY64_HEX = "3f9a0662d796da74"
OFFICIAL_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
FROZEN_ROOT = "/local-ssd/xiangyuezhang/semtalk_final_4066f20_20260802"
FROZEN_COMMIT = "4066f2096e1675f9c19d725894007ff25f3e9b4b"
FROZEN_TREE = "0b66e3aa1fb23732e76e51492737c4ab1f4db2d0"
FROZEN_EVALUATOR_SHA256 = (
    "18b7dfcc2989a8c2136b8e530e471cfa9a088b29600e084522f0312f1e8e661f"
)
FORMAL_PYTHON = (
    "/local-ssd/xiangyuezhang/semtalk_show_base_env_806b008_20260729/"
    "bin/python"
)
FORMAL_PYTHON_LINK_TARGET = "python3.12"
FORMAL_PYTHON_SECONDARY_TARGET = "/usr/bin/python3.12"
FORMAL_PYTHON_RESOLVED = "/usr/bin/python3.12"
PASPA_ROOT = "/local-ssd/xiangyuezhang/paspa_diffsheg_0df27e6_20260731_exact"
DIFFSHEG_ROOT = "/local-ssd/xiangyuezhang/DiffSHEG_3ebf305_20260802_complete"
DIFFSHEG_BATCH_SIZE = 64
PREDECESSOR_CONTROL_ROOT = (
    "/local-ssd/xiangyuezhang/semtalk_final_control_73ff184_20260802"
)
PREDECESSOR_CONTROL_COMMIT = "73ff184a62fb2d1c3064a4fb03a11b48e6309f7f"
PREDECESSOR_CONTROL_TREE = "2d568cd3f793c0131ee1489b679aa9c7bd76da53"
PREDECESSOR_BRIDGE_SHA256 = (
    "739e241bc18412ca7368549714f4fdce17ed6e4719d48699a6e17037e13565e7"
)
PREDECESSOR_STATE_ROOT = (
    "/efs/xiangyuezhang/semtalk_show_base_v14_formal_400e_20260802_v1/"
    "live_validation_state_73ff184_v2"
)
PREDECESSOR_RUN_ROOT = (
    "/efs/xiangyuezhang/semtalk_show_base_v14_formal_400e_20260802_v1/"
    "live_validation_runs_73ff184_v2/e1"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "9cc1e0f9c7f85556a9598bb8725f76609bde58dd74f1b3886bc5ced0a6482f87"
)
PREDECESSOR_CAMPAIGN_BYTES = 33839
TRAIN_ROOT = (
    "/efs/xiangyuezhang/semtalk_show_base_v14_formal_400e_20260802_v1/"
    "semtalk_show_base_v14_formal_w8g2048_400e_20260802_v1"
)
AUTHORITY_PATH = TRAIN_ROOT + "/live_val_metric_repair_claims/epoch-0001.json"
ADOPTION_PATH = TRAIN_ROOT + "/live_val_consumer_adoption_claims/epoch-0001.json"
SPEC_PATH = TRAIN_ROOT + "/live_val_metric_repair_specs/epoch-0001.v1.json"
REPAIR_ROOT = TRAIN_ROOT + "/live_val_metric_repair_runs/epoch-0001.v1"
RUNNER_CONTROL_ROOT = (
    TRAIN_ROOT + "/live_val_metric_repair_runner_controls/epoch-0001.v1"
)
RESULT_PATH = REPAIR_ROOT + "/repair-result.json"
RUNNER_STATUS_PATH = RUNNER_CONTROL_ROOT + "/.guarded_status.json"
RUNNER_LOG_PATH = RUNNER_CONTROL_ROOT + "/runner.log"
GUARD_PROOF_PATH = RUNNER_CONTROL_ROOT + "/guard-proof.txt"
GUARDED_RUNNER_PATH = "/tmp/globaldiff_guarded_runner.py"
GUARDED_RUNNER_SHA256 = (
    "c7904f32a143cffff16ef7eec4f586a613e8219c4f31a23e47271f5695642b6a"
)
GUARDED_RUNNER_BYTES = 14022
GUARD_VERIFIER_PATH = "/tmp/verify_globaldiff_guards.py"
GUARD_VERIFIER_SHA256 = (
    "dd183d09c9ddf07634b0ca4a2db4da45274c124efc2bbd03489f0c9d82c31888"
)
GUARD_VERIFIER_BYTES = 2988
PREDECESSOR_FIXED = {
    "predecessor_campaign": (
        PREDECESSOR_STATE_ROOT + "/campaign.json",
        PREDECESSOR_CAMPAIGN_SHA256, PREDECESSOR_CAMPAIGN_BYTES,
    ),
    "predecessor_job_claim": (
        PREDECESSOR_STATE_ROOT + "/job_claims/epoch-0001.json",
        "e54f31e167581e60282ca50963a6e6d3b3033373dcd323b3b4be809db80277d2", 1231,
    ),
    "predecessor_active_claim": (
        PREDECESSOR_STATE_ROOT + "/active_invocation.claim.json",
        "71e0c6020dfbce480dbe7ef47db4187b228fc0f74f3b6fc9b849820e5ca53a13", 457,
    ),
    "predecessor_work_authority": (
        PREDECESSOR_STATE_ROOT + "/authorities/epoch-0001.json",
        "995bc4ce5ae9a11dc2707e958eaf4cd60fbb7fbc54ef4fcaeee99843990cb81f", 23578,
    ),
    "predecessor_authorization": (
        PREDECESSOR_STATE_ROOT + "/authorizations/epoch-0001.json",
        "111d0048c90c82fbd3b7f4e961c5c86605682dbbc99e76e2d473ea88e64f0618", 6549,
    ),
    "predecessor_runner_status": (
        PREDECESSOR_STATE_ROOT + "/runner_status/epoch-0001.json",
        "da361716d68b32fe3427a5660eca2af2e85914ee8d093e325fe57e5be7bdc7cf", 2113,
    ),
    "preflight": (
        PREDECESSOR_RUN_ROOT + "/work-preflight.json",
        "497895011915f94d71897a73cbb1aeda13d1f480627fedb3811a4f34c3ed31dc", 6304,
    ),
    "inference_lineage": (
        PREDECESSOR_RUN_ROOT + "/candidates/e1/final/val-inference-lineage.json",
        "bf788d4ec635af39526cc0a1068101ed84ca276954a67830a99399e9f64d64b2", 2344,
    ),
    "final_manifest": (
        PREDECESSOR_RUN_ROOT + "/candidates/e1/final/final_manifest.jsonl",
        "1f916610617d5dbc34832d88385acd05a53d2fb50856eeedd45888cecb95df71", 1561219,
    ),
    "clip_manifest": (
        PREDECESSOR_RUN_ROOT + "/candidates/e1/final/diffsheg_eval_clip_ids.txt",
        "f6ab4334c13f461b99b5d6f8024fcd270ccb812724a8e83f2e5f12c0c57c1fee", 55677,
    ),
    "old_report": (
        PREDECESSOR_RUN_ROOT + "/candidates/e1/diffsheg-val-fgd.json",
        "30afef8b6fa7757723a45422f6c2e6d371af3767fcaf69ce9e40e4e1e52c98e7", 3962,
    ),
    "predecessor_failure_log": (
        PREDECESSOR_RUN_ROOT + "/logs/e1-live-measurement.log",
        "d5e46b5d9fdd7a501c86064e783c107c05799bef60b4112c7c55d2c7328b38d5", 1294,
    ),
    "candidate_receipt": (
        TRAIN_ROOT + "/candidate_receipts/epoch-0001.json",
        "16b2fe66a7f8ec0c38c7ab326b540a6a14043c74cd85fe013e7ca6cf10b24d8a", 2396,
    ),
    "predecessor_recovery_claim": (
        TRAIN_ROOT + "/live_val_consumer_recovery_claims/epoch-0001.json",
        "0fea907f839c38dd4f2e8eb7c0dac4cbb3ca5318cffd8ef63a302d03bb1b0ad8", 3179,
    ),
    "predecessor_recovery_request": (
        PREDECESSOR_STATE_ROOT + "/recovery-request.epoch-0001.json",
        "f5e1b57f3dc1aa284f35b265c3d7b36e25abb604cdabb00879893bcdabc89a5a", 8214,
    ),
    "predecessor_recovery_authority": (
        PREDECESSOR_STATE_ROOT + "/recovery-authority.epoch-0001.json",
        "84a2a0d8c1210ffff235f5b14f4620814e8023b273eab3df5045a7eb2d13661e", 3125,
    ),
}
ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
SOURCE_KEYS = frozenset({
    "origin", "source_root", "commit", "tree", "clean", "detached",
    "local_branches_at_commit",
})
SPEC_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "candidate_epoch", "predecessor_state_root", "predecessor_run_root",
    "predecessor_campaign", "predecessor_job_claim",
    "predecessor_active_claim", "predecessor_work_authority",
    "predecessor_authorization", "predecessor_runner_status",
    "predecessor_recovery_request", "predecessor_recovery_authority",
    "predecessor_recovery_claim", "candidate_receipt", "preflight",
    "inference_lineage", "final_manifest", "clip_manifest", "old_report",
    "predecessor_failure_log", "frozen_evaluator_source",
    "predecessor_control_source",
    "frozen_evaluator", "bridge", "formal_python", "paspa_root",
    "diffsheg_root", "batch_size", "repair_root", "guard_verifier",
    "guarded_runner", "repair_tool", "result_path", "runner_control_root",
    "runner_status_path", "runner_log_path", "guard_proof_path",
    "repair_tool_source",
    "receipt_payload_sha256",
})
AUTHORITY_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "inference_allowed", "metric_replay_count", "candidate_epoch", "spec",
    "terminal_snapshot", "created_unix", "receipt_payload_sha256",
})
RESULT_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "candidate_epoch", "inference_reruns", "metric_replays", "authority",
    "terminal_snapshot_before", "terminal_snapshot_after", "evaluator_argv",
    "bridge_complete_argv", "bridge_complete_stdout_sha256",
    "bridge_measurement_replay_argv", "bridge_measurement_replay_stdout",
    "repaired_report", "report_comparison", "measurement", "completed_unix",
    "receipt_payload_sha256",
})
ADOPTION_KEYS = frozenset({
    "format", "status", "split", "test_visible", "selection_eligible",
    "test_measurements_authorized", "candidate_epoch", "candidate_receipt",
    "predecessor_campaign", "predecessor_job_claim", "predecessor_active_claim",
    "predecessor_work_authority", "predecessor_authorization",
    "predecessor_runner_status", "predecessor_recovery_request",
    "predecessor_recovery_authority", "predecessor_recovery_claim",
    "predecessor_failure_log", "metric_repair_spec", "metric_repair_authority",
    "metric_repair_result", "frozen_evaluator_source",
    "predecessor_control_source", "repair_tool_source", "frozen_evaluator",
    "repaired_report",
    "report_comparison", "measurement", "guarded_runner",
    "repair_runner_status", "repair_runner_log", "guard_verifier", "guard_proof",
    "guard_verifier_argv", "guard_verifier_stdout", "restored_guards",
    "terminal_snapshot_before", "terminal_snapshot_after",
    "validation_diffsheg_fgd", "validation_diffsheg_fgd_binary64_hex",
    "inference_reruns", "metric_replays", "predecessor_return_code",
    "repair_return_code", "failure_phase", "completed_unix",
    "receipt_payload_sha256",
})


class RepairError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def payload_sha(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    require("receipt_payload_sha256" not in result, "self-hash already present")
    result["receipt_payload_sha256"] = payload_sha(result)
    return result


def strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in items:
            require(key not in value, "%s has duplicate key" % label)
            value[key] = child
        return value

    def reject(token: str) -> None:
        raise RepairError("%s has non-finite token %s" % (label, token))

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairError("%s is not strict JSON" % label) from error
    require(isinstance(value, dict), "%s is not an object" % label)
    return value


def regular_bytes(path_text: Any, label: str) -> tuple[Path, bytes]:
    require(isinstance(path_text, str) and path_text.startswith("/"), "%s path invalid" % label)
    lexical = Path(path_text)
    parent = lexical.parent.resolve(strict=True)
    path = parent / lexical.name
    require(str(path) == str(lexical), "%s parent is not canonical" % label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RepairError("%s absent/symlinked" % label) from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "%s is not regular" % label)
        require(before.st_nlink == 1, "%s has multiple hard links" % label)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        require(
            (
                before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
                before.st_size, before.st_mtime_ns, before.st_ctime_ns,
            )
            == (
                after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
                after.st_size, after.st_mtime_ns, after.st_ctime_ns,
            ),
            "%s changed while reading" % label,
        )
        raw = b"".join(chunks)
        require(len(raw) == before.st_size, "%s read length changed" % label)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise RepairError("%s path disappeared while reading" % label) from error
        require(
            stat.S_ISREG(path_after.st_mode)
            and path_after.st_nlink == 1
            and (
                path_after.st_dev, path_after.st_ino, path_after.st_mode,
                path_after.st_nlink, path_after.st_size, path_after.st_mtime_ns,
                path_after.st_ctime_ns,
            )
            == (
                after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
                after.st_size, after.st_mtime_ns, after.st_ctime_ns,
            ),
            "%s path identity changed while reading" % label,
        )
    finally:
        os.close(descriptor)
    return path, raw


def artifact(path_text: Any, label: str, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path, raw = regular_bytes(path_text, label)
    value = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    if expected is not None:
        require(
            isinstance(expected, Mapping)
            and set(expected) == ARTIFACT_KEYS
            and type_sensitive_equal(value, dict(expected)),
            "%s artifact changed" % label,
        )
    return value


def read_document(reference: Mapping[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        isinstance(reference, Mapping) and set(reference) == ARTIFACT_KEYS,
        "%s artifact reference changed" % label,
    )
    path, raw = regular_bytes(reference.get("path"), label)
    observed = {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    require(type_sensitive_equal(observed, dict(reference)), "%s artifact changed" % label)
    return observed, strict_json(raw, label)


def fsync_parent(path: Path, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise RepairError("%s parent fsync failed" % label) from error
    finally:
        os.close(descriptor)


def write_new(path_text: Any, value: Mapping[str, Any], label: str) -> dict[str, Any]:
    require(isinstance(path_text, str) and path_text.startswith("/"), "%s output path invalid" % label)
    lexical = Path(path_text)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    parent = lexical.parent.resolve(strict=True)
    path = parent / lexical.name
    require(str(path) == str(lexical), "%s parent is not canonical" % label)
    require(not os.path.lexists(path), "%s already exists" % label)
    raw = canonical_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "%s write incomplete" % label)
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    fsync_parent(path, label)
    return artifact(str(path), label)


def write_new_bytes(path: Path, raw: bytes, label: str) -> dict[str, Any]:
    require(not os.path.lexists(path), "%s already exists" % label)
    parent = path.parent.resolve(strict=True)
    canonical = parent / path.name
    require(canonical == path, "%s parent is not canonical" % label)
    descriptor = os.open(
        canonical,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "%s write incomplete" % label)
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    fsync_parent(canonical, label)
    return artifact(str(canonical), label)


def git_stdout(root: str, arguments: Sequence[str], label: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", root, *arguments], shell=False, check=False,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    require(
        process.returncode == 0 and process.stderr == b"",
        "%s Git query failed" % label,
    )
    return process.stdout


def inspect_source_checkout(
    root_text: str, expected_commit: str, expected_tree: str, label: str,
) -> dict[str, Any]:
    require(isinstance(root_text, str) and root_text.startswith("/"), "%s root invalid" % label)
    lexical = Path(root_text)
    require(not lexical.is_symlink(), "%s root is symlinked" % label)
    root = lexical.resolve(strict=True)
    require(root.is_dir() and str(root) == root_text, "%s root is not canonical" % label)
    top = git_stdout(root_text, ["rev-parse", "--show-toplevel"], label).decode("utf-8").rstrip("\n")
    origin = git_stdout(root_text, ["remote", "get-url", "origin"], label).decode("utf-8").rstrip("\n")
    commit = git_stdout(root_text, ["rev-parse", "HEAD"], label).decode("ascii").rstrip("\n")
    tree = git_stdout(root_text, ["rev-parse", "HEAD^{tree}"], label).decode("ascii").rstrip("\n")
    dirty = git_stdout(
        root_text, ["status", "--porcelain=v1", "--untracked-files=all"], label,
    )
    symbolic = subprocess.run(
        ["git", "-C", root_text, "symbolic-ref", "-q", "HEAD"],
        shell=False, check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    require(
        symbolic.returncode == 1 and symbolic.stdout == b"" and symbolic.stderr == b"",
        "%s checkout is not detached" % label,
    )
    branches_raw = git_stdout(
        root_text,
        ["for-each-ref", "--format=%(refname)", "refs/heads"],
        label,
    )
    branches = [line for line in branches_raw.decode("utf-8").splitlines() if line]
    observed = {
        "origin": origin, "source_root": root_text, "commit": commit,
        "tree": tree, "clean": dirty == b"", "detached": True,
        "local_branches_at_commit": branches,
    }
    require(
        top == root_text and origin == OFFICIAL_ORIGIN
        and commit == expected_commit and tree == expected_tree
        and dirty == b"" and branches == [],
        "%s source identity changed" % label,
    )
    return observed


def inspect_repair_tool_source(
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tool_path = Path(__file__).resolve(strict=True)
    root = tool_path.parents[2]
    require(
        tool_path == root / "scripts/show_base/adopt_base_v14_metric_repair.py",
        "metric repair tool layout changed",
    )
    commit = git_stdout(str(root), ["rev-parse", "HEAD"], "repair tool").decode(
        "ascii"
    ).rstrip("\n")
    tree = git_stdout(
        str(root), ["rev-parse", "HEAD^{tree}"], "repair tool",
    ).decode("ascii").rstrip("\n")
    observed = inspect_source_checkout(str(root), commit, tree, "repair tool")
    if expected is not None:
        require(
            isinstance(expected, Mapping)
            and set(expected) == SOURCE_KEYS
            and type_sensitive_equal(observed, dict(expected)),
            "metric repair tool source changed",
        )
    return observed


def tree_snapshot(roots: Sequence[str]) -> dict[str, Any]:
    require(list(roots) == [PREDECESSOR_STATE_ROOT, PREDECESSOR_RUN_ROOT], "terminal snapshot roots changed")
    entries: list[dict[str, Any]] = []
    total = 0
    for root_text in roots:
        root = Path(root_text)
        require(not root.is_symlink() and root.is_dir(), "terminal root absent/symlinked")
        for path in sorted([root, *root.rglob("*")], key=lambda item: str(item)):
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), "terminal tree contains a symlink")
            relative = "." if path == root else str(path.relative_to(root))
            if stat.S_ISDIR(mode):
                entries.append({"root": root_text, "path": relative, "type": "directory"})
            else:
                require(stat.S_ISREG(mode), "terminal tree contains a special file")
                _canonical, raw = regular_bytes(str(path), "terminal snapshot file")
                total += len(raw)
                entries.append({
                    "root": root_text, "path": relative, "type": "regular",
                    "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                })
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "roots": list(roots), "entry_count": len(entries), "total_bytes": total,
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def assert_predecessor_incident(spec: Mapping[str, Any]) -> None:
    _campaign_artifact, campaign = read_document(
        spec["predecessor_campaign"], "predecessor campaign"
    )
    jobs = campaign.get("jobs")
    require(
        isinstance(jobs, list)
        and [job.get("epoch") for job in jobs]
        == [1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400],
        "predecessor campaign queue changed",
    )
    absent = [
        PREDECESSOR_STATE_ROOT + "/completions/epoch-0001.json",
        PREDECESSOR_RUN_ROOT + "/candidates/e1/live-measurement.json",
    ]
    for job in jobs[1:]:
        epoch = job["epoch"]
        absent.extend([
            PREDECESSOR_STATE_ROOT + "/job_claims/epoch-%04d.json" % epoch,
            job["authority_path"], job["authorization_path"], job["run_root"],
            job["measurement_path"], job["completion_path"],
            job["runner_status_path"], job["runner_log_path"],
        ])
    require(
        all(not os.path.lexists(path) for path in absent),
        "predecessor contains e1 completion/measurement or e2+ output",
    )
    _status_artifact, status = read_document(
        spec["predecessor_runner_status"], "predecessor runner status"
    )
    require(
        status.get("state") == "failed"
        and type(status.get("return_code")) is int
        and status["return_code"] == 1
        and all(
            status.get(key) is None
            for key in ("error", "cleanup_error", "restore_error", "received_signal")
        ),
        "predecessor failure is not the pinned rc1 phase",
    )
    _failure_path, failure_raw = regular_bytes(
        spec["predecessor_failure_log"]["path"], "predecessor failure log"
    )
    marker = b"DiffSHEG adapter provenance is not the frozen pipeline source"
    require(
        failure_raw.count(marker) == 1
        and failure_raw.rstrip().endswith(marker),
        "predecessor failure is not uniquely provenance-only",
    )


def fixed_artifact(key: str) -> dict[str, Any]:
    path, digest, size = PREDECESSOR_FIXED[key]
    return {"path": path, "sha256": digest, "bytes": size}


def require_canonical_directory(path_text: str, label: str) -> None:
    path = Path(path_text)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RepairError("%s is absent" % label) from error
    require(
        not path.is_symlink() and resolved == path and path.is_dir(),
        "%s is absent, symlinked, or non-canonical" % label,
    )


def validate_formal_python() -> None:
    python_path = Path(FORMAL_PYTHON)
    secondary = python_path.parent / FORMAL_PYTHON_LINK_TARGET
    try:
        first_stat = python_path.lstat()
        second_stat = secondary.lstat()
        resolved = python_path.resolve(strict=True)
        resolved_stat = resolved.lstat()
    except OSError as error:
        raise RepairError("formal Python symlink chain is absent") from error
    require(
        stat.S_ISLNK(first_stat.st_mode)
        and first_stat.st_nlink == 1
        and os.readlink(python_path) == FORMAL_PYTHON_LINK_TARGET
        and stat.S_ISLNK(second_stat.st_mode)
        and second_stat.st_nlink == 1
        and os.readlink(secondary) == FORMAL_PYTHON_SECONDARY_TARGET
        and str(resolved) == FORMAL_PYTHON_RESOLVED
        and stat.S_ISREG(resolved_stat.st_mode)
        and resolved_stat.st_nlink == 1
        and os.access(python_path, os.X_OK),
        "formal Python symlink/executable identity changed",
    )


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    require(args.output == SPEC_PATH, "metric repair spec path changed")
    require(not os.path.lexists(SPEC_PATH), "metric repair spec already exists")
    require(not os.path.lexists(REPAIR_ROOT), "metric repair root already exists")
    require(
        not os.path.lexists(RUNNER_CONTROL_ROOT),
        "metric repair runner control root already exists",
    )
    require_canonical_directory(Path(FORMAL_PYTHON).parent.as_posix(), "formal Python parent")
    validate_formal_python()
    require_canonical_directory(PASPA_ROOT, "PASPA root")
    require_canonical_directory(DIFFSHEG_ROOT, "DiffSHEG root")
    frozen_source = inspect_source_checkout(
        FROZEN_ROOT, FROZEN_COMMIT, FROZEN_TREE, "frozen evaluator",
    )
    control_source = inspect_source_checkout(
        PREDECESSOR_CONTROL_ROOT, PREDECESSOR_CONTROL_COMMIT,
        PREDECESSOR_CONTROL_TREE, "predecessor control",
    )
    repair_tool_source = inspect_repair_tool_source()
    predecessor = {key: fixed_artifact(key) for key in PREDECESSOR_FIXED}
    for key, reference in predecessor.items():
        artifact(reference["path"], key, reference)
    guarded_runner = artifact(
        GUARDED_RUNNER_PATH, "guarded runner",
        {
            "path": GUARDED_RUNNER_PATH, "sha256": GUARDED_RUNNER_SHA256,
            "bytes": GUARDED_RUNNER_BYTES,
        },
    )
    guard_verifier = artifact(
        GUARD_VERIFIER_PATH, "guard verifier",
        {
            "path": GUARD_VERIFIER_PATH, "sha256": GUARD_VERIFIER_SHA256,
            "bytes": GUARD_VERIFIER_BYTES,
        },
    )
    evaluator = artifact(
        FROZEN_ROOT + "/scripts/show_base/evaluate_diffsheg_val_fgd.py",
        "frozen evaluator",
    )
    require(
        evaluator["sha256"] == FROZEN_EVALUATOR_SHA256,
        "frozen evaluator digest changed",
    )
    bridge = artifact(
        PREDECESSOR_CONTROL_ROOT
        + "/scripts/show_base/base_live_val_consumer_bridge.py",
        "predecessor bridge",
    )
    require(
        bridge["sha256"] == PREDECESSOR_BRIDGE_SHA256,
        "predecessor bridge digest changed",
    )
    value = self_hashed({
        "format": SPEC_FORMAT, "status": "frozen", "split": "val",
        "test_visible": False, "selection_eligible": False,
        "candidate_epoch": 1, "predecessor_state_root": PREDECESSOR_STATE_ROOT,
        "predecessor_run_root": PREDECESSOR_RUN_ROOT, **predecessor,
        "frozen_evaluator_source": frozen_source,
        "predecessor_control_source": control_source,
        "repair_tool_source": repair_tool_source,
        "frozen_evaluator": evaluator, "bridge": bridge,
        "formal_python": FORMAL_PYTHON, "paspa_root": PASPA_ROOT,
        "diffsheg_root": DIFFSHEG_ROOT, "batch_size": DIFFSHEG_BATCH_SIZE,
        "repair_root": REPAIR_ROOT, "guarded_runner": guarded_runner,
        "guard_verifier": guard_verifier,
        "repair_tool": artifact(str(Path(__file__).resolve(strict=True)), "repair tool"),
        "result_path": RESULT_PATH, "runner_control_root": RUNNER_CONTROL_ROOT,
        "runner_status_path": RUNNER_STATUS_PATH,
        "runner_log_path": RUNNER_LOG_PATH,
        "guard_proof_path": GUARD_PROOF_PATH,
    })
    require(set(value) == SPEC_KEYS, "constructed metric repair spec schema changed")
    output = write_new(args.output, value, "metric repair spec")
    return {"status": "frozen", "spec": output}


def load_spec(reference: Mapping[str, Any]) -> dict[str, Any]:
    require(reference.get("path") == SPEC_PATH, "metric repair spec path changed")
    _artifact, value = read_document(reference, "metric repair spec")
    require(
        set(value) == SPEC_KEYS
        and value.get("format") == SPEC_FORMAT
        and value.get("status") == "frozen"
        and value.get("receipt_payload_sha256") == payload_sha(value),
        "metric repair spec schema/state changed",
    )
    require(
        value.get("split") == "val"
        and value.get("test_visible") is False
        and value.get("selection_eligible") is False
        and type(value.get("candidate_epoch")) is int
        and value["candidate_epoch"] == 1,
        "metric repair is not e1 val-only",
    )
    require(value.get("predecessor_state_root") == PREDECESSOR_STATE_ROOT and value.get("predecessor_run_root") == PREDECESSOR_RUN_ROOT, "metric repair predecessor roots changed")
    for key, (path, digest, size) in PREDECESSOR_FIXED.items():
        require(
            type_sensitive_equal(
                value.get(key),
                {"path": path, "sha256": digest, "bytes": size},
            ),
            "%s is not the pinned formal-v2 artifact" % key,
        )
    for key in (
        "predecessor_campaign", "predecessor_job_claim", "predecessor_active_claim",
        "predecessor_work_authority", "predecessor_authorization",
        "predecessor_runner_status", "predecessor_recovery_request",
        "predecessor_recovery_authority", "predecessor_recovery_claim",
        "candidate_receipt",
        "preflight", "inference_lineage", "final_manifest", "clip_manifest",
        "old_report", "predecessor_failure_log", "frozen_evaluator",
        "bridge", "guarded_runner", "guard_verifier", "repair_tool",
    ):
        reference_value = value.get(key)
        require(isinstance(reference_value, dict) and set(reference_value) == ARTIFACT_KEYS, "%s reference changed" % key)
        artifact(reference_value["path"], key, reference_value)
    source = value.get("frozen_evaluator_source")
    expected_source = {
        "origin": OFFICIAL_ORIGIN, "source_root": FROZEN_ROOT,
        "commit": FROZEN_COMMIT, "tree": FROZEN_TREE,
        "clean": True, "detached": True, "local_branches_at_commit": [],
    }
    require(
        type_sensitive_equal(source, expected_source)
        and type_sensitive_equal(
            inspect_source_checkout(
                FROZEN_ROOT, FROZEN_COMMIT, FROZEN_TREE, "frozen evaluator",
            ),
            expected_source,
        ),
        "metric repair evaluator source changed",
    )
    control_source = value.get("predecessor_control_source")
    expected_control_source = {
        "origin": OFFICIAL_ORIGIN, "source_root": PREDECESSOR_CONTROL_ROOT,
        "commit": PREDECESSOR_CONTROL_COMMIT, "tree": PREDECESSOR_CONTROL_TREE,
        "clean": True, "detached": True, "local_branches_at_commit": [],
    }
    require(
        type_sensitive_equal(control_source, expected_control_source)
        and type_sensitive_equal(
            inspect_source_checkout(
                PREDECESSOR_CONTROL_ROOT, PREDECESSOR_CONTROL_COMMIT,
                PREDECESSOR_CONTROL_TREE, "predecessor control",
            ),
            expected_control_source,
        ),
        "metric repair predecessor control source changed",
    )
    repair_tool_source = value.get("repair_tool_source")
    require(
        isinstance(repair_tool_source, Mapping)
        and set(repair_tool_source) == SOURCE_KEYS,
        "metric repair tool source schema changed",
    )
    inspect_repair_tool_source(repair_tool_source)
    evaluator = value["frozen_evaluator"]
    require(evaluator["path"] == FROZEN_ROOT + "/scripts/show_base/evaluate_diffsheg_val_fgd.py" and evaluator["sha256"] == FROZEN_EVALUATOR_SHA256, "metric repair evaluator is not frozen 4066")
    bridge = value["bridge"]
    require(bridge["path"] == PREDECESSOR_CONTROL_ROOT + "/scripts/show_base/base_live_val_consumer_bridge.py" and bridge["sha256"] == PREDECESSOR_BRIDGE_SHA256, "metric repair bridge is not unmodified 73ff")
    require(
        value["repair_tool"]["path"] == str(Path(__file__).resolve(strict=True)),
        "metric repair tool is not this exact entrypoint",
    )
    require(
        value.get("formal_python") == FORMAL_PYTHON
        and value.get("paspa_root") == PASPA_ROOT
        and value.get("diffsheg_root") == DIFFSHEG_ROOT
        and type(value.get("batch_size")) is int
        and value["batch_size"] == DIFFSHEG_BATCH_SIZE,
        "metric repair runtime roots/batch changed",
    )
    validate_formal_python()
    require_canonical_directory(PASPA_ROOT, "PASPA root")
    require_canonical_directory(DIFFSHEG_ROOT, "DiffSHEG root")
    require(
        value.get("repair_root") == REPAIR_ROOT
        and value.get("result_path") == RESULT_PATH
        and value.get("runner_control_root") == RUNNER_CONTROL_ROOT
        and value.get("runner_status_path") == RUNNER_STATUS_PATH
        and value.get("runner_log_path") == RUNNER_LOG_PATH
        and value.get("guard_proof_path") == GUARD_PROOF_PATH,
        "metric repair output roots changed",
    )
    require(
        type_sensitive_equal(value["guarded_runner"], {
            "path": GUARDED_RUNNER_PATH, "sha256": GUARDED_RUNNER_SHA256,
            "bytes": GUARDED_RUNNER_BYTES,
        })
        and type_sensitive_equal(value["guard_verifier"], {
            "path": GUARD_VERIFIER_PATH, "sha256": GUARD_VERIFIER_SHA256,
            "bytes": GUARD_VERIFIER_BYTES,
        }),
        "guarded runner/verifier binding changed",
    )
    assert_predecessor_incident(value)
    return value


def authorize(args: argparse.Namespace) -> dict[str, Any]:
    spec_ref = {"path": args.spec, "sha256": args.expected_spec_sha256, "bytes": args.expected_spec_bytes}
    spec = load_spec(spec_ref)
    require(args.output == AUTHORITY_PATH, "metric repair authority path changed")
    require(not os.path.lexists(spec["repair_root"]), "repair root already exists")
    control_root = Path(spec["runner_status_path"]).parent
    require(
        not os.path.lexists(control_root),
        "metric repair runner control root already exists",
    )
    authority = self_hashed({
        "format": AUTHORITY_FORMAT, "status": "authorized", "split": "val",
        "test_visible": False, "selection_eligible": False,
        "inference_allowed": False, "metric_replay_count": 1,
        "candidate_epoch": 1,
        "spec": spec_ref,
        "terminal_snapshot": tree_snapshot([PREDECESSOR_STATE_ROOT, PREDECESSOR_RUN_ROOT]),
        "created_unix": float(time.time()),
    })
    require(set(authority) == AUTHORITY_KEYS, "constructed repair authority schema changed")
    output = write_new(args.output, authority, "metric repair authority")
    return {"status": "authorized", "authority": output}


def load_authority(path_text: str, expected_sha: str, expected_bytes: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(path_text == AUTHORITY_PATH, "metric repair authority path changed")
    reference = {"path": path_text, "sha256": expected_sha, "bytes": expected_bytes}
    observed, value = read_document(reference, "metric repair authority")
    require(
        set(value) == AUTHORITY_KEYS
        and value.get("format") == AUTHORITY_FORMAT
        and value.get("status") == "authorized"
        and type(value.get("candidate_epoch")) is int
        and value["candidate_epoch"] == 1
        and value.get("split") == "val"
        and value.get("test_visible") is False
        and value.get("selection_eligible") is False
        and value.get("inference_allowed") is False
        and type(value.get("metric_replay_count")) is int
        and value["metric_replay_count"] == 1
        and type(value.get("created_unix")) is float
        and math.isfinite(value["created_unix"])
        and value.get("receipt_payload_sha256") == payload_sha(value),
        "metric repair authority changed",
    )
    spec = load_spec(value["spec"])
    require(value.get("terminal_snapshot") == tree_snapshot([PREDECESSOR_STATE_ROOT, PREDECESSOR_RUN_ROOT]), "terminal predecessor changed after repair authorization")
    return observed, value, spec


def prepare_runner_control(args: argparse.Namespace) -> dict[str, Any]:
    authority_artifact, _authority, spec = load_authority(
        args.authority, args.expected_authority_sha256,
        args.expected_authority_bytes,
    )
    del authority_artifact
    require(
        args.output == spec["runner_control_root"] == RUNNER_CONTROL_ROOT,
        "metric repair runner control root changed",
    )
    control = Path(args.output)
    require(not os.path.lexists(control), "metric repair runner control root already exists")
    control.parent.mkdir(parents=True, exist_ok=True)
    parent = control.parent.resolve(strict=True)
    require(parent / control.name == control, "metric repair runner control parent changed")
    os.mkdir(control, 0o700)
    metadata = control.lstat()
    require(
        stat.S_ISDIR(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700
        and control.resolve(strict=True) == control,
        "metric repair runner control root did not seal",
    )
    fsync_parent(control, "metric repair runner control root")
    return {"status": "prepared", "runner_control_root": str(control)}


def capture(argv: Sequence[str], log_path: Path, label: str) -> bytes:
    process = subprocess.run(list(argv), shell=False, check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    write_new_bytes(log_path, process.stdout + process.stderr, label + " log")
    require(process.returncode == 0 and process.stderr == b"", "%s failed or emitted stderr" % label)
    return process.stdout


def type_sensitive_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            type_sensitive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            type_sensitive_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def valid_runner_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%S%z") == value


def compare_reports(old_reference: Mapping[str, Any], new_reference: Mapping[str, Any]) -> dict[str, Any]:
    _old_artifact, old = read_document(old_reference, "old diagnostic report")
    _new_artifact, new = read_document(new_reference, "repaired DiffSHEG report")
    old_adapter = old.get("provenance", {}).get("adapter")
    new_adapter = new.get("provenance", {}).get("adapter")
    require(isinstance(old_adapter, dict) and isinstance(new_adapter, dict), "report adapter provenance absent")
    require(
        old_adapter.get("path") == PREDECESSOR_CONTROL_ROOT + "/scripts/show_base/evaluate_diffsheg_val_fgd.py"
        and old_adapter.get("repository_root") == PREDECESSOR_CONTROL_ROOT
        and old_adapter.get("repository_git_head") == PREDECESSOR_CONTROL_COMMIT
        and new_adapter.get("path") == FROZEN_ROOT + "/scripts/show_base/evaluate_diffsheg_val_fgd.py"
        and new_adapter.get("repository_root") == FROZEN_ROOT
        and new_adapter.get("repository_git_head") == FROZEN_COMMIT,
        "report adapter provenance transition changed",
    )
    normalized = json.loads(json.dumps(new))
    normalized_adapter = normalized["provenance"]["adapter"]
    for key in ("path", "repository_root", "repository_git_head"):
        normalized_adapter[key] = old_adapter[key]
    require(
        type_sensitive_equal(old, normalized),
        "reports differ outside the three authorized adapter provenance fields",
    )
    old_fgd = old.get("metrics", {}).get("fgd")
    new_fgd = new.get("metrics", {}).get("fgd")
    require(
        type(old_fgd) is float
        and type(new_fgd) is float
        and struct.pack(">d", old_fgd) == struct.pack(">d", new_fgd),
        "repaired FGD is not bit-identical binary64",
    )
    require(
        struct.pack(">d", old_fgd).hex() == EXPECTED_FGD_BINARY64_HEX,
        "pinned predecessor FGD binary64 changed",
    )
    return self_hashed({
        "format": "semtalk_show_base_v14_metric_report_comparison_v1",
        "status": "equivalent_except_provenance",
        "split": "val", "test_visible": False,
        "selection_eligible": False, "candidate_epoch": 1,
        "inference_reruns": 0, "metric_replays": 1,
        "old_report": dict(old_reference), "new_report": dict(new_reference),
        "allowed_differences": [
            "provenance.adapter.path",
            "provenance.adapter.repository_git_head",
            "provenance.adapter.repository_root",
        ],
        "fgd_binary64_hex": struct.pack(">d", old_fgd).hex(),
        "raw_feature_bytes_compared": False,
    })


BRIDGE_MEASUREMENT_REPLAY_CODE = """\
import importlib.util
import sys
from pathlib import Path
path, measurement_path, measurement_sha = sys.argv[1:]
spec = importlib.util.spec_from_file_location("metric_repair_bridge", path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load pinned bridge")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
artifact, value = module._validate_live_measurement(Path(measurement_path), measurement_sha)
if artifact.get("sha256") != measurement_sha or value.get("candidate_epoch") != 1:
    raise RuntimeError("measurement replay changed")
sys.stdout.write("PASS\\n")
"""


def evaluator_argv_for(spec: Mapping[str, Any], report_path: str) -> list[str]:
    pred = Path(PREDECESSOR_RUN_ROOT) / "candidates/e1/final/predictions/val"
    gt = Path(PREDECESSOR_RUN_ROOT) / "candidates/e1/final/ground-truth/val"
    return [
        spec["formal_python"], spec["frozen_evaluator"]["path"],
        "--pred-dir", str(pred), "--gt-dir", str(gt),
        "--clip-manifest", spec["clip_manifest"]["path"],
        "--clip-manifest-sha256", spec["clip_manifest"]["sha256"],
        "--paspa-root", spec["paspa_root"],
        "--diffsheg-root", spec["diffsheg_root"],
        "--device", "cuda:0", "--batch-size", str(spec["batch_size"]),
        "--output", report_path,
    ]


def bridge_argv_for(
    spec: Mapping[str, Any], report: Mapping[str, Any], measurement_path: str,
) -> list[str]:
    return [
        spec["formal_python"], spec["bridge"]["path"], "complete",
        "--preflight", spec["preflight"]["path"],
        "--expected-preflight-sha256", spec["preflight"]["sha256"],
        "--inference-lineage", spec["inference_lineage"]["path"],
        "--expected-inference-lineage-sha256", spec["inference_lineage"]["sha256"],
        "--diffsheg-report", report["path"],
        "--expected-diffsheg-report-sha256", report["sha256"],
        "--output", measurement_path,
    ]


def run_repair(args: argparse.Namespace) -> dict[str, Any]:
    authority_artifact, authority, spec = load_authority(args.authority, args.expected_authority_sha256, args.expected_authority_bytes)
    lexical_repair_root = Path(spec["repair_root"])
    lexical_repair_root.parent.mkdir(parents=True, exist_ok=True)
    repair_parent = lexical_repair_root.parent.resolve(strict=True)
    repair_root = repair_parent / lexical_repair_root.name
    require(repair_root == lexical_repair_root, "repair root parent is not canonical")
    require(not os.path.lexists(repair_root), "repair root already exists")
    require(str(repair_root) == REPAIR_ROOT, "metric repair root changed")
    repair_root.mkdir(parents=True, mode=0o700)
    before = tree_snapshot([PREDECESSOR_STATE_ROOT, PREDECESSOR_RUN_ROOT])
    require(before == authority["terminal_snapshot"], "terminal predecessor changed before repair")
    report = repair_root / "diffsheg-val-fgd.frozen-4066.json"
    measurement = repair_root / "live-measurement.json"
    evaluator_argv = evaluator_argv_for(spec, str(report))
    capture(evaluator_argv, repair_root / "evaluator.log", "frozen 4066 evaluator")
    os.chmod(report, 0o400, follow_symlinks=False)
    report_artifact = artifact(str(report), "repaired DiffSHEG report")
    comparison_value = compare_reports(spec["old_report"], report_artifact)
    comparison_artifact = write_new(
        str(repair_root / "report-comparison.json"),
        comparison_value,
        "metric report comparison",
    )
    bridge_argv = bridge_argv_for(spec, report_artifact, str(measurement))
    bridge_stdout = capture(bridge_argv, repair_root / "bridge-complete.log", "unmodified bridge complete")
    os.chmod(measurement, 0o400, follow_symlinks=False)
    measurement_artifact = artifact(str(measurement), "repaired live measurement")
    _measurement_ref, measurement_value = read_document(measurement_artifact, "repaired live measurement")
    require(measurement_value.get("format") == MEASUREMENT_FORMAT and measurement_value.get("status") == "complete" and measurement_value.get("candidate_epoch") == 1 and measurement_value.get("split") == "val" and measurement_value.get("test_visible") is False and measurement_value.get("selection_eligible") is False and measurement_value.get("receipt_payload_sha256") == payload_sha(measurement_value), "repaired measurement changed")
    bridge_result = strict_json(bridge_stdout, "bridge complete stdout")
    expected_bridge_keys = {
        "status", "split", "test_visible", "selection_eligible", "epoch",
        "fgd", "created", "measurement",
    }
    report_fgd = strict_json(
        regular_bytes(report_artifact["path"], "repaired DiffSHEG report")[1],
        "repaired DiffSHEG report",
    ).get("metrics", {}).get("fgd")
    measurement_fgd = measurement_value.get("metrics", {}).get("fgd")
    expected_measurement_stdout = {
        **measurement_artifact,
        "receipt_payload_sha256": measurement_value["receipt_payload_sha256"],
    }
    require(
        set(bridge_result) == expected_bridge_keys
        and bridge_result.get("status") == "complete"
        and bridge_result.get("split") == "val"
        and bridge_result.get("test_visible") is False
        and bridge_result.get("selection_eligible") is False
        and bridge_result.get("epoch") == 1
        and bridge_result.get("created") is True
        and type(bridge_result.get("fgd")) is float
        and type(report_fgd) is float
        and type(measurement_fgd) is float
        and struct.pack(">d", bridge_result["fgd"])
        == struct.pack(">d", report_fgd)
        == struct.pack(">d", measurement_fgd)
        and type_sensitive_equal(
            bridge_result.get("measurement"), expected_measurement_stdout,
        )
        and bridge_stdout
        == (json.dumps(bridge_result, sort_keys=True, allow_nan=False) + "\n").encode(),
        "bridge complete stdout changed",
    )
    replay_argv = [
        spec["formal_python"], "-I", "-B", "-c",
        BRIDGE_MEASUREMENT_REPLAY_CODE, spec["bridge"]["path"],
        measurement_artifact["path"], measurement_artifact["sha256"],
    ]
    replay_stdout = capture(
        replay_argv, repair_root / "bridge-measurement-replay.log",
        "unmodified bridge measurement replay",
    )
    require(replay_stdout == b"PASS\n", "bridge measurement replay changed")
    after = tree_snapshot([PREDECESSOR_STATE_ROOT, PREDECESSOR_RUN_ROOT])
    require(before == after, "terminal predecessor changed during repair")
    result_value = self_hashed({
        "format": RESULT_FORMAT, "status": "complete", "split": "val",
        "test_visible": False, "selection_eligible": False, "candidate_epoch": 1,
        "inference_reruns": 0, "metric_replays": 1,
        "authority": authority_artifact, "terminal_snapshot_before": before,
        "terminal_snapshot_after": after, "evaluator_argv": evaluator_argv,
        "bridge_complete_argv": bridge_argv,
        "bridge_complete_stdout_sha256": hashlib.sha256(bridge_stdout).hexdigest(),
        "bridge_measurement_replay_argv": replay_argv,
        "bridge_measurement_replay_stdout": "PASS\n",
        "repaired_report": report_artifact,
        "report_comparison": comparison_artifact,
        "measurement": {**measurement_artifact, "receipt_payload_sha256": measurement_value["receipt_payload_sha256"]},
        "completed_unix": float(time.time()),
    })
    require(set(result_value) == RESULT_KEYS, "constructed repair result schema changed")
    require(
        str(repair_root / "repair-result.json") == spec["result_path"] == RESULT_PATH,
        "metric repair result path changed",
    )
    result_artifact = write_new(spec["result_path"], result_value, "metric repair result")
    return {"status": "complete", "result": result_artifact}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    authority_artifact, authority, spec = load_authority(args.authority, args.expected_authority_sha256, args.expected_authority_bytes)
    require(args.output == ADOPTION_PATH, "metric repair adoption path changed")
    require(
        args.runner_status == spec["runner_status_path"]
        and args.runner_log == spec["runner_log_path"]
        and args.guard_proof == spec["guard_proof_path"]
        and args.result == spec["result_path"],
        "metric repair runner evidence paths changed",
    )
    result_ref = {"path": args.result, "sha256": args.expected_result_sha256, "bytes": args.expected_result_bytes}
    result_artifact, result = read_document(result_ref, "metric repair result")
    require(
        set(result) == RESULT_KEYS
        and result.get("format") == RESULT_FORMAT
        and result.get("status") == "complete"
        and result.get("split") == "val"
        and result.get("test_visible") is False
        and result.get("selection_eligible") is False
        and type(result.get("candidate_epoch")) is int
        and result["candidate_epoch"] == 1
        and result.get("receipt_payload_sha256") == payload_sha(result)
        and type_sensitive_equal(result.get("authority"), authority_artifact)
        and type_sensitive_equal(
            result.get("terminal_snapshot_before"),
            result.get("terminal_snapshot_after"),
        )
        and type_sensitive_equal(
            result.get("terminal_snapshot_after"), authority["terminal_snapshot"],
        )
        and type(result.get("inference_reruns")) is int
        and result["inference_reruns"] == 0
        and type(result.get("metric_replays")) is int
        and result["metric_replays"] == 1
        and type(result.get("completed_unix")) is float
        and math.isfinite(result["completed_unix"]),
        "metric repair result changed",
    )
    status_ref = {"path": args.runner_status, "sha256": args.expected_runner_status_sha256, "bytes": args.expected_runner_status_bytes}
    status_artifact, status = read_document(status_ref, "metric repair runner status")
    required_status_keys = {"updated_at", "state", "wrapper_pid", "child_pid", "return_code", "received_signal", "error", "cleanup_error", "restored_guards", "restore_error", "command"}
    expected_run_argv = [
        spec["formal_python"], "-I", spec["repair_tool"]["path"], "run",
        "--authority", authority_artifact["path"],
        "--expected-authority-sha256", authority_artifact["sha256"],
        "--expected-authority-bytes", str(authority_artifact["bytes"]),
    ]
    require(
        set(status) == required_status_keys
        and status.get("state") == "finished"
        and valid_runner_timestamp(status.get("updated_at"))
        and type(status.get("wrapper_pid")) is int and status["wrapper_pid"] > 1
        and type(status.get("child_pid")) is int and status["child_pid"] > 1
        and type(status.get("return_code")) is int
        and status["return_code"] == 0
        and type_sensitive_equal(status.get("command"), expected_run_argv)
        and all(
            status.get(key) is None
            for key in ("received_signal", "error", "cleanup_error", "restore_error")
        ),
        "metric repair runner failed or command changed",
    )
    restored = status.get("restored_guards")
    require(isinstance(restored, dict) and set(restored) == {str(i) for i in range(8)} and len(set(restored.values())) == 8 and all(type(restored[str(i)]) is int and restored[str(i)] > 1 for i in range(8)), "metric repair restored guards changed")
    runner_log = artifact(args.runner_log, "metric repair runner log", {"path": args.runner_log, "sha256": args.expected_runner_log_sha256, "bytes": args.expected_runner_log_bytes})
    _runner_log_path, runner_log_raw = regular_bytes(
        runner_log["path"], "metric repair runner log"
    )
    runner_stdout = strict_json(runner_log_raw, "metric repair runner stdout")
    require(
        type_sensitive_equal(
            runner_stdout, {"status": "complete", "result": result_artifact},
        )
        and runner_log_raw
        == (
            json.dumps(runner_stdout, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode(),
        "metric repair runner stdout changed",
    )
    proof_pins = (
        args.expected_guard_proof_sha256,
        args.expected_guard_proof_bytes,
    )
    require(
        all(value is None for value in proof_pins)
        or all(value is not None for value in proof_pins),
        "metric repair guard proof pins must be all-or-none",
    )
    expected_guard_text = "PASS " + " ".join("GPU%d=PID%d" % (i, restored[str(i)]) for i in range(8)) + "\n"
    guard_verifier_argv = [
        spec["formal_python"], spec["guard_verifier"]["path"],
        *[str(restored[str(i)]) for i in range(8)],
    ]
    verifier = subprocess.run(
        guard_verifier_argv, shell=False, check=False,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    require(
        verifier.returncode == 0
        and verifier.stderr == b""
        and verifier.stdout == expected_guard_text.encode("ascii"),
        "independent pinned guard verification failed",
    )
    if proof_pins[0] is None:
        require(
            not os.path.lexists(args.guard_proof),
            "metric repair guard proof already exists without explicit pins",
        )
        guard_proof = write_new_bytes(
            Path(args.guard_proof), verifier.stdout,
            "metric repair guard proof",
        )
        guard_raw = verifier.stdout
    else:
        guard_proof = artifact(args.guard_proof, "metric repair guard proof", {
            "path": args.guard_proof,
            "sha256": args.expected_guard_proof_sha256,
            "bytes": args.expected_guard_proof_bytes,
        })
        _guard_path, guard_raw = regular_bytes(
            guard_proof["path"], "metric repair guard proof"
        )
        require(
            guard_raw == verifier.stdout,
            "metric repair guard proof changed",
        )
    try:
        guard_text = guard_raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise RepairError("metric repair guard proof is not ASCII") from error
    require(guard_text == expected_guard_text, "metric repair guard proof changed")
    require(
        tree_snapshot([PREDECESSOR_STATE_ROOT, PREDECESSOR_RUN_ROOT])
        == result["terminal_snapshot_after"],
        "terminal predecessor changed before adoption finalize",
    )
    measurement_ref = result["measurement"]
    require(
        isinstance(measurement_ref, dict)
        and set(measurement_ref)
        == ARTIFACT_KEYS | {"receipt_payload_sha256"},
        "adopted measurement reference changed",
    )
    _measurement_artifact, measurement = read_document({key: measurement_ref[key] for key in ARTIFACT_KEYS}, "adopted measurement")
    require(
        measurement.get("receipt_payload_sha256") == payload_sha(measurement)
        and measurement_ref["receipt_payload_sha256"]
        == measurement["receipt_payload_sha256"],
        "adopted measurement payload binding changed",
    )
    fgd = measurement.get("metrics", {}).get("fgd")
    require(
        type(fgd) is float and math.isfinite(fgd) and fgd >= 0.0,
        "adopted FGD invalid",
    )
    comparison_artifact, comparison = read_document(
        result["report_comparison"], "metric report comparison"
    )
    expected_comparison = compare_reports(spec["old_report"], result["repaired_report"])
    require(
        type_sensitive_equal(comparison, expected_comparison)
        and comparison.get("receipt_payload_sha256") == payload_sha(comparison),
        "metric report comparison changed",
    )
    expected_report_path = REPAIR_ROOT + "/diffsheg-val-fgd.frozen-4066.json"
    expected_comparison_path = REPAIR_ROOT + "/report-comparison.json"
    expected_measurement_path = REPAIR_ROOT + "/live-measurement.json"
    require(
        isinstance(result.get("repaired_report"), dict)
        and set(result["repaired_report"]) == ARTIFACT_KEYS
        and result["repaired_report"]["path"] == expected_report_path
        and isinstance(result.get("report_comparison"), dict)
        and set(result["report_comparison"]) == ARTIFACT_KEYS
        and result["report_comparison"]["path"] == expected_comparison_path
        and measurement_ref["path"] == expected_measurement_path,
        "metric repair result artifact paths changed",
    )
    expected_bridge_stdout = (
        json.dumps(
            {
                "status": "complete", "split": "val", "test_visible": False,
                "selection_eligible": False, "epoch": 1, "fgd": fgd,
                "created": True, "measurement": measurement_ref,
            },
            sort_keys=True, allow_nan=False,
        )
        + "\n"
    ).encode()
    require(
        type_sensitive_equal(
            result.get("evaluator_argv"),
            evaluator_argv_for(spec, expected_report_path),
        )
        and type_sensitive_equal(
            result.get("bridge_complete_argv"),
            bridge_argv_for(spec, result["repaired_report"], expected_measurement_path),
        )
        and result.get("bridge_complete_stdout_sha256")
        == hashlib.sha256(expected_bridge_stdout).hexdigest()
        and comparison.get("fgd_binary64_hex") == EXPECTED_FGD_BINARY64_HEX
        and struct.pack(">d", fgd).hex() == EXPECTED_FGD_BINARY64_HEX,
        "metric repair command/FGD evidence changed",
    )
    replay_argv = result.get("bridge_measurement_replay_argv")
    require(
        replay_argv
        == [
            spec["formal_python"], "-I", "-B", "-c",
            BRIDGE_MEASUREMENT_REPLAY_CODE, spec["bridge"]["path"],
            measurement_ref["path"], measurement_ref["sha256"],
        ]
        and result.get("bridge_measurement_replay_stdout") == "PASS\n",
        "bridge measurement replay evidence changed",
    )
    replay = subprocess.run(
        replay_argv, shell=False, check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    require(
        replay.returncode == 0 and replay.stdout == b"PASS\n" and replay.stderr == b"",
        "bridge measurement independent replay failed",
    )
    receipt = self_hashed({
        "format": ADOPTION_FORMAT, "status": "complete", "split": "val",
        "test_visible": False, "selection_eligible": False,
        "test_measurements_authorized": 0, "candidate_epoch": 1,
        "candidate_receipt": spec["candidate_receipt"],
        "predecessor_campaign": spec["predecessor_campaign"],
        "predecessor_job_claim": spec["predecessor_job_claim"],
        "predecessor_active_claim": spec["predecessor_active_claim"],
        "predecessor_work_authority": spec["predecessor_work_authority"],
        "predecessor_authorization": spec["predecessor_authorization"],
        "predecessor_runner_status": spec["predecessor_runner_status"],
        "predecessor_recovery_request": spec["predecessor_recovery_request"],
        "predecessor_recovery_authority": spec["predecessor_recovery_authority"],
        "predecessor_recovery_claim": spec["predecessor_recovery_claim"],
        "predecessor_failure_log": spec["predecessor_failure_log"],
        "metric_repair_spec": authority["spec"],
        "metric_repair_authority": authority_artifact,
        "metric_repair_result": result_artifact,
        "frozen_evaluator_source": spec["frozen_evaluator_source"],
        "predecessor_control_source": spec["predecessor_control_source"],
        "repair_tool_source": spec["repair_tool_source"],
        "frozen_evaluator": spec["frozen_evaluator"],
        "repaired_report": result["repaired_report"],
        "report_comparison": comparison_artifact,
        "measurement": measurement_ref,
        "guarded_runner": spec["guarded_runner"],
        "repair_runner_status": status_artifact,
        "repair_runner_log": runner_log,
        "guard_verifier": spec["guard_verifier"],
        "guard_proof": guard_proof,
        "guard_verifier_argv": guard_verifier_argv,
        "guard_verifier_stdout": guard_text,
        "restored_guards": restored,
        "terminal_snapshot_before": result["terminal_snapshot_before"],
        "terminal_snapshot_after": result["terminal_snapshot_after"],
        "validation_diffsheg_fgd": fgd,
        "validation_diffsheg_fgd_binary64_hex": struct.pack(">d", fgd).hex(),
        "inference_reruns": 0,
        "metric_replays": 1,
        "predecessor_return_code": 1,
        "repair_return_code": 0,
        "failure_phase": "provenance_only",
        "completed_unix": float(time.time()),
    })
    require(set(receipt) == ADOPTION_KEYS, "constructed adoption receipt schema changed")
    output = write_new(args.output, receipt, "metric repair adoption receipt")
    return {"status": "complete", "adoption_receipt": output}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build-spec")
    build_parser.add_argument("--output", required=True)
    authorize_parser = commands.add_parser("authorize")
    authorize_parser.add_argument("--spec", required=True)
    authorize_parser.add_argument("--expected-spec-sha256", required=True)
    authorize_parser.add_argument("--expected-spec-bytes", required=True, type=int)
    authorize_parser.add_argument("--output", required=True)
    prepare_parser = commands.add_parser("prepare-runner-control")
    prepare_parser.add_argument("--authority", required=True)
    prepare_parser.add_argument("--expected-authority-sha256", required=True)
    prepare_parser.add_argument("--expected-authority-bytes", required=True, type=int)
    prepare_parser.add_argument("--output", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--authority", required=True)
    run_parser.add_argument("--expected-authority-sha256", required=True)
    run_parser.add_argument("--expected-authority-bytes", required=True, type=int)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--authority", required=True)
    finalize_parser.add_argument("--expected-authority-sha256", required=True)
    finalize_parser.add_argument("--expected-authority-bytes", required=True, type=int)
    for name in ("result", "runner-status", "runner-log"):
        finalize_parser.add_argument("--" + name, required=True)
        finalize_parser.add_argument("--expected-" + name + "-sha256", required=True)
        finalize_parser.add_argument("--expected-" + name + "-bytes", required=True, type=int)
    finalize_parser.add_argument("--guard-proof", required=True)
    finalize_parser.add_argument("--expected-guard-proof-sha256")
    finalize_parser.add_argument("--expected-guard-proof-bytes", type=int)
    finalize_parser.add_argument("--output", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build-spec":
        result = build_spec(args)
    elif args.command == "authorize":
        result = authorize(args)
    elif args.command == "prepare-runner-control":
        result = prepare_runner_control(args)
    elif args.command == "run":
        result = run_repair(args)
    else:
        result = finalize(args)
    sys.stdout.write(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairError as error:
        sys.stderr.write("metric-repair-error: %s\n" % error)
        raise SystemExit(1)
