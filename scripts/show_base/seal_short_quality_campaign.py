#!/usr/bin/env python3
"""Fail-closed transactional sealer for one SemTalk SHOW short-quality run.

The script intentionally has a narrower interface than the retired helper.
One new ``campaign-root`` is the unit of authority.  ``completion.json`` is
created last and is the *only* success marker.  A failed or interrupted
campaign is never allowed to execute the official producer a second time.

This is a CPU-only orchestration program.  It neither opens CUDA nor starts a
guarded GPU runner.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple


EPOCHS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_SOURCE_COMMIT = "70a70f452bdf743e317b583a7770980f0ce744c3"
EXPECTED_SOURCE_TREE = "bdf7680f56f9f53c92e7ab0bf6c6a84ef4f83d69"
EXPECTED_EVIDENCE_SOURCE_COMMIT = "4066f2096e1675f9c19d725894007ff25f3e9b4b"
EXPECTED_EVIDENCE_SOURCE_TREE = "0b66e3aa1fb23732e76e51492737c4ab1f4db2d0"
PRODUCER_RELATIVE = "scripts/show_base/produce_base_topology_short_quality.py"
PRODUCER_SHA256 = "1d7fbd600d6ac0657896688eed0acd47600cacffe1632a655c4f9409eb604dac"
ADAPTER_RELATIVE = "scripts/show_base/base_short_quality_val_adapter.py"
SEALER_RELATIVE = "scripts/show_base/seal_short_quality_campaign.py"

# This is the already frozen SemTalk formal Python runtime.  The executable
# may itself be a venv symlink; its resolved regular file is hash-pinned.
EXPECTED_PYTHON_RUNTIME = Path(
    "/local-ssd/xiangyuezhang/semtalk_show_base_env_806b008_20260729/bin/python"
)
EXPECTED_PYTHON_REAL_SHA256 = (
    "b119173f03ba8558b2ddf74044055c74e8ca83e2bea0a6e9d9f98f0bd9d88e67"
)
MINIMUM_PYTHON = (3, 10)
GIT = Path("/usr/bin/git")

CLAIM_FORMAT = "semtalk_show_short_quality_campaign_claim_v2"
GLOBAL_CLAIM_FORMAT = "semtalk_show_short_quality_global_claim_v2"
COMPLETION_FORMAT = "semtalk_show_short_quality_campaign_completion_v2"
FAILURE_FORMAT = "semtalk_show_short_quality_campaign_failure_v2"
QUALITY_NAME = "quality-report.json"
SHORT_NAME = "short-trajectory.json"
COMPLETION_NAME = "completion.json"
CLAIM_NAME = "claim.json"
SEALED_NAME = "sealed"
FAILED_NAME = "failed"
CLAIM_REGISTRY = Path(
    "/efs/xiangyuezhang/semtalk_show_short_quality_seal_claims_v2"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OID_RE = re.compile(r"[0-9a-f]{40}\Z")


class SealError(RuntimeError):
    """A fail-closed contract violation."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return _sha256_bytes(_canonical_json_bytes(unsigned))


def _require_sha(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise SealError(f"{label} must be one lowercase SHA-256")
    return value


def _require_git_oid(value: Any, label: str) -> str:
    if type(value) is not str or GIT_OID_RE.fullmatch(value) is None:
        raise SealError(f"{label} must be one lowercase 40-hex Git object ID")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise SealError(f"{label} must be one positive integer")
    return value


def _no_duplicate_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SealError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SealError(f"non-finite JSON token {token!r} in {label}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError(f"invalid strict JSON in {label}") from error
    if type(value) is not dict:
        raise SealError(f"{label} must contain one JSON object")
    return value


def _identity(value: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _pathname_stat(path: Path) -> os.stat_result:
    """Separated for deterministic CPU race tests; never follows symlinks."""

    return os.stat(path, follow_symlinks=False)


def _require_absolute_raw(path: Path, label: str) -> Path:
    if not path.is_absolute() or str(path) != os.path.abspath(str(path)):
        raise SealError(f"{label} must be an absolute normalized path")
    return path


def _canonical_existing(path: Path, label: str, *, directory: bool) -> Path:
    path = _require_absolute_raw(path, label)
    try:
        named = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SealError(f"cannot inspect {label}: {path}: {error}") from error
    if stat.S_ISLNK(named.st_mode) or resolved != path:
        raise SealError(f"{label} must be canonical and contain no symlink")
    if directory and not stat.S_ISDIR(named.st_mode):
        raise SealError(f"{label} must be a directory")
    if not directory and not stat.S_ISREG(named.st_mode):
        raise SealError(f"{label} must be a regular file")
    return path


def _safe_read(path: Path, label: str) -> Tuple[bytes, Tuple[int, int, int, int, int]]:
    path = _canonical_existing(path, label, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SealError(f"cannot safely open {label}: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SealError(f"{label} is not a regular file")
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise SealError(f"{label} changed while its descriptor was read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise SealError(f"{label} size changed while it was read")
        current = _pathname_stat(path)
        if _identity(current) != _identity(before):
            raise SealError(f"{label} pathname changed during verification")
        if path.resolve(strict=True) != path:
            raise SealError(f"{label} ceased to be canonical")
        return payload, _identity(before)
    finally:
        os.close(descriptor)


def _strict_receipt(
    value: Any,
    label: str,
    *,
    require_bytes: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full_keys = {"path", "sha256", "bytes", "receipt_payload_sha256"}
    projected_keys = {"path", "sha256", "receipt_payload_sha256"}
    expected_keys = full_keys if require_bytes else projected_keys
    if type(value) is not dict or set(value) != expected_keys:
        raise SealError(f"{label} artifact schema must be exactly {sorted(expected_keys)}")
    raw_path = value.get("path")
    if type(raw_path) is not str:
        raise SealError(f"{label} path must be a string")
    path = _require_absolute_raw(Path(raw_path), f"{label} path")
    expected_sha = _require_sha(value.get("sha256"), f"{label} SHA-256")
    expected_payload = _require_sha(
        value.get("receipt_payload_sha256"), f"{label} payload SHA-256"
    )
    expected_bytes = (
        _positive_int(value.get("bytes"), f"{label} bytes")
        if require_bytes
        else None
    )
    payload, _inode = _safe_read(path, label)
    if _sha256_bytes(payload) != expected_sha:
        raise SealError(f"{label} SHA-256 changed")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise SealError(f"{label} byte count changed")
    parsed = _strict_json_bytes(payload, label)
    if (
        parsed.get("receipt_payload_sha256") != expected_payload
        or _payload_sha(parsed, "receipt_payload_sha256") != expected_payload
    ):
        raise SealError(f"{label} payload closure changed")
    artifact = {
        "path": str(path),
        "sha256": expected_sha,
        "bytes": len(payload),
        "receipt_payload_sha256": expected_payload,
    }
    return artifact, parsed


def _strict_artifact(path: Path, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload, _inode = _safe_read(path, label)
    parsed = _strict_json_bytes(payload, label)
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }, parsed


def _strict_pinned_json(
    path: Path,
    label: str,
    expected_sha: str,
    expected_bytes: int,
    expected_payload: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _strict_receipt(
        {
            "path": str(path),
            "sha256": _require_sha(expected_sha, f"{label} SHA-256"),
            "bytes": _positive_int(expected_bytes, f"{label} bytes"),
            "receipt_payload_sha256": _require_sha(
                expected_payload, f"{label} payload SHA-256"
            ),
        },
        label,
        require_bytes=True,
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SealError(f"subprocess failed before a verified result: {error}") from error


def _git(root: Path, *arguments: str, allow_failure: bool = False) -> str:
    if not GIT.is_file():
        raise SealError(f"fixed git executable is absent: {GIT}")
    process = _run([str(GIT), "-C", str(root), *arguments], timeout=60)
    if process.returncode != 0 and not allow_failure:
        raise SealError(
            "git source audit failed: "
            + process.stderr.decode("utf-8", errors="replace").strip()
        )
    return process.stdout.decode("utf-8", errors="strict").strip()


def verify_sealer_authority(
    expected_commit: str,
    expected_blob_oid: str,
    expected_file_sha256: str,
    *,
    script_path: Path | None = None,
) -> Dict[str, Any]:
    """Bind this exact executing file to a pinned, tracked Git revision.

    The expected values are supplied by the formal launcher, whose campaign
    specification pins them independently of the executing Python process.
    Merely executing a byte-identical untracked copy is therefore forbidden:
    the canonical pathname, HEAD commit/tree, index entry, Git blob, working
    tree bytes, and the launcher-pinned file digest must all agree twice.
    """

    expected_commit = _require_git_oid(expected_commit, "expected sealer commit")
    expected_blob_oid = _require_git_oid(
        expected_blob_oid, "expected sealer Git blob"
    )
    expected_file_sha256 = _require_sha(
        expected_file_sha256, "expected sealer file SHA-256"
    )
    raw_script = Path(__file__) if script_path is None else script_path
    script = _canonical_existing(raw_script, "formal sealer source", directory=False)

    try:
        repository_root = _canonical_existing(
            Path(_git(script.parent, "rev-parse", "--show-toplevel")),
            "formal sealer repository root",
            directory=True,
        )
        relative = script.relative_to(repository_root)
    except (ValueError, OSError) as error:
        raise SealError("formal sealer is outside its Git repository") from error
    if relative.as_posix() != SEALER_RELATIVE:
        raise SealError(
            f"formal sealer must execute from tracked path {SEALER_RELATIVE}"
        )

    origin = _git(repository_root, "remote", "get-url", "origin")
    push_origin = _git(
        repository_root, "remote", "get-url", "--push", "origin"
    )
    commit = _git(repository_root, "rev-parse", "HEAD")
    tree = _git(repository_root, "rev-parse", "HEAD^{tree}")
    if origin != EXPECTED_ORIGIN or push_origin != EXPECTED_ORIGIN:
        raise SealError("formal sealer repository origin changed")
    if commit != expected_commit:
        raise SealError("formal sealer HEAD differs from pinned commit authority")

    # Both the index and HEAD must know this exact pathname.  A matching
    # untracked file beside an otherwise valid repository must never qualify.
    tracked = _run(
        [
            str(GIT),
            "-C",
            str(repository_root),
            "ls-files",
            "--error-unmatch",
            "--stage",
            "--",
            SEALER_RELATIVE,
        ],
        timeout=60,
    )
    if tracked.returncode != 0:
        raise SealError("formal sealer path is not tracked in the pinned commit")
    tracked_lines = tracked.stdout.decode("utf-8", errors="strict").splitlines()
    if len(tracked_lines) != 1:
        raise SealError("formal sealer has an ambiguous Git index entry")
    match = re.fullmatch(
        r"(100644|100755) ([0-9a-f]{40}) 0\t" + re.escape(SEALER_RELATIVE),
        tracked_lines[0],
    )
    if match is None:
        raise SealError("formal sealer must be one stage-0 regular Git blob")
    git_mode = match.group(1)
    index_blob = match.group(2)

    blob_oid = _git(repository_root, "rev-parse", f"HEAD:{SEALER_RELATIVE}")
    if blob_oid != expected_blob_oid or index_blob != expected_blob_oid:
        raise SealError("formal sealer Git blob differs from pinned authority")
    if _git(repository_root, "cat-file", "-t", blob_oid) != "blob":
        raise SealError("formal sealer Git object is not a blob")
    status = _git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        SEALER_RELATIVE,
    )
    if status:
        raise SealError("formal sealer tracked pathname is dirty or untracked")

    payload, first_identity = _safe_read(script, "formal sealer source")
    committed = _run(
        [
            str(GIT),
            "-C",
            str(repository_root),
            "cat-file",
            "blob",
            blob_oid,
        ],
        timeout=60,
    )
    file_sha256 = _sha256_bytes(payload)
    if (
        committed.returncode != 0
        or committed.stdout != payload
        or file_sha256 != expected_file_sha256
    ):
        raise SealError("formal sealer bytes differ from pinned Git/file authority")

    # Close the audit against a concurrent checkout, replacement, or ref move.
    payload_after, second_identity = _safe_read(script, "formal sealer source")
    if payload_after != payload or second_identity != first_identity:
        raise SealError("formal sealer changed during authority verification")
    if (
        _git(repository_root, "rev-parse", "HEAD") != commit
        or _git(repository_root, "rev-parse", "HEAD^{tree}") != tree
        or _git(repository_root, "rev-parse", f"HEAD:{SEALER_RELATIVE}")
        != blob_oid
        or _git(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            SEALER_RELATIVE,
        )
    ):
        raise SealError("formal sealer Git authority changed during verification")

    return {
        "origin": origin,
        "repository_root": str(repository_root),
        "path": str(script),
        "relative_path": SEALER_RELATIVE,
        "commit": commit,
        "tree": tree,
        "git_blob_oid": blob_oid,
        "file_sha256": file_sha256,
        "bytes": len(payload),
        "tracked": True,
        "index_stage": 0,
        "git_mode": git_mode,
        "clean_at_path": True,
        "launcher_pinned": True,
    }


def verify_runtime(runtime: Path = EXPECTED_PYTHON_RUNTIME) -> Dict[str, Any]:
    runtime = _require_absolute_raw(runtime, "formal Python runtime")
    try:
        real = runtime.resolve(strict=True)
    except OSError as error:
        raise SealError(f"formal Python runtime is absent: {runtime}") from error
    payload, _inode = _safe_read(real, "formal Python real executable")
    observed = _sha256_bytes(payload)
    if observed != EXPECTED_PYTHON_REAL_SHA256:
        raise SealError("formal Python real executable SHA-256 changed")
    probe = _run(
        [
            str(runtime),
            "-I",
            "-c",
            (
                "import json,os,sys;"
                "print(json.dumps({'executable':sys.executable,"
                "'realpath':os.path.realpath(sys.executable),"
                "'version':[sys.version_info.major,sys.version_info.minor,"
                "sys.version_info.micro]},sort_keys=True))"
            ),
        ],
        timeout=30,
    )
    if probe.returncode != 0:
        raise SealError("formal Python runtime probe failed")
    value = _strict_json_bytes(probe.stdout.strip(), "formal Python runtime probe")
    version = value.get("version")
    if (
        type(version) is not list
        or len(version) != 3
        or any(type(item) is not int for item in version)
        or tuple(version[:2]) < MINIMUM_PYTHON
        or Path(str(value.get("realpath"))) != real
    ):
        raise SealError("formal Python runtime contract changed")
    return {
        "path": str(runtime),
        "realpath": str(real),
        "execution_path": str(real),
        "real_sha256": observed,
        "version": version,
        "isolated_mode": True,
    }


def verify_source(
    source_root: Path,
    *,
    expected_commit: str = EXPECTED_SOURCE_COMMIT,
    expected_tree: str = EXPECTED_SOURCE_TREE,
    label: str = "runtime validation source",
) -> Dict[str, Any]:
    expected_commit = _require_git_oid(expected_commit, f"{label} commit")
    expected_tree = _require_git_oid(expected_tree, f"{label} tree")
    root = _canonical_existing(source_root, label, directory=True)
    remotes = [line for line in _git(root, "remote").splitlines() if line]
    origin = _git(root, "remote", "get-url", "origin")
    push_origin = _git(root, "remote", "get-url", "--push", "origin")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status_text = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    heads = [
        line
        for line in _git(
            root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
        ).splitlines()
        if line
    ]
    if (
        remotes != ["origin"]
        or origin != EXPECTED_ORIGIN
        or push_origin != EXPECTED_ORIGIN
        or commit != expected_commit
        or tree != expected_tree
        or status_text
        or branch != "HEAD"
        or heads
    ):
        raise SealError(
            f"{label} must be the exact clean detached branchless official "
            f"{expected_commit} tree"
        )
    producer = root / PRODUCER_RELATIVE
    producer_payload, _producer_inode = _safe_read(producer, "official producer")
    producer_sha = _sha256_bytes(producer_payload)
    committed = _run(
        [str(GIT), "-C", str(root), "show", f"{commit}:{PRODUCER_RELATIVE}"],
        timeout=60,
    )
    if (
        producer_sha != PRODUCER_SHA256
        or committed.returncode != 0
        or committed.stdout != producer_payload
    ):
        raise SealError(f"official producer bytes differ from pinned {label}")
    adapter = root / ADAPTER_RELATIVE
    adapter_payload, _adapter_inode = _safe_read(adapter, "official val adapter")
    return {
        "origin": origin,
        "source_root": str(root),
        "commit": commit,
        "tree": tree,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "producer": {
            "path": str(producer),
            "sha256": producer_sha,
            "bytes": len(producer_payload),
        },
        "val_adapter": {
            "path": str(adapter),
            "sha256": _sha256_bytes(adapter_payload),
            "bytes": len(adapter_payload),
        },
    }


def _require_source_binding(preflight: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    for label in ("adapter_source", "pipeline_source"):
        value = preflight.get(label)
        if type(value) is not dict:
            raise SealError(f"preflight {label} is absent")
        for key in (
            "origin",
            "source_root",
            "commit",
            "tree",
            "clean",
            "detached",
            "local_branches_at_commit",
        ):
            if value.get(key) != source.get(key):
                raise SealError(f"preflight {label}.{key} differs from official source")


def replay_preflight(
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
    preflight_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    command = [
        str(runtime["execution_path"]),
        "-I",
        str(source["val_adapter"]["path"]),
        "validate",
        "--split",
        "val",
        "--preflight",
        str(preflight_artifact["path"]),
        "--expected-preflight-sha256",
        str(preflight_artifact["sha256"]),
    ]
    process = _run(command, cwd=Path(str(source["source_root"])), timeout=600)
    if process.returncode != 0:
        raise SealError(
            "official preflight fresh replay failed: "
            + process.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    result = _strict_json_bytes(process.stdout.strip(), "official preflight replay")
    if (
        result.get("path") != preflight_artifact["path"]
        or result.get("sha256") != preflight_artifact["sha256"]
        or result.get("receipt_payload_sha256")
        != preflight_artifact["receipt_payload_sha256"]
    ):
        raise SealError("official preflight replay returned a different receipt")
    return result


def _report_artifact(path: Path, label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact, report = _strict_artifact(path, label)
    protocol = report.get("protocol")
    metrics = report.get("metrics")
    fgd = metrics.get("fgd") if type(metrics) is dict else None
    if (
        report.get("status") != "ok"
        or type(protocol) is not dict
        or protocol.get("selection_split") != "val"
        or protocol.get("test_visible") is not False
        or isinstance(fgd, bool)
        or not isinstance(fgd, (int, float))
        or not math.isfinite(float(fgd))
        or float(fgd) < 0.0
    ):
        raise SealError(f"{label} is not one finite validation-only FGD report")
    return artifact, report


def build_input_closure(
    preflight: Mapping[str, Any],
    validation_root: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    validation_root = _canonical_existing(
        validation_root, "short-quality validation root", directory=True
    )
    if (
        preflight.get("status") != "complete"
        or preflight.get("split") != "val"
        or preflight.get("test_visible") is not False
        or preflight.get("candidate_epochs") != list(EPOCHS)
        or type(preflight.get("coverage")) is not dict
        or preflight["coverage"].get("clip_count") != 1715
    ):
        raise SealError("preflight is not the exact hidden-test SHOW validation authority")
    authority = preflight.get("short_quality_authority")
    if type(authority) is not dict:
        raise SealError("short-quality authority is absent")
    mode = authority.get("topology_mode")
    if (
        type(mode) is not str
        or not mode
        or mode.strip() != mode
        or re.fullmatch(r"[a-z0-9_]+", mode) is None
    ):
        raise SealError("short-quality authority topology mode is invalid")
    ready_values = authority.get("candidate_ready_receipts")
    if type(ready_values) is not list or len(ready_values) != len(EPOCHS):
        raise SealError("candidate-ready receipt set is not exact")
    ready: List[Dict[str, Any]] = []
    for index, epoch in enumerate(EPOCHS):
        artifact, receipt = _strict_receipt(
            ready_values[index], f"e{epoch} candidate-ready receipt", require_bytes=True
        )
        if type(receipt.get("epoch")) is not int or receipt["epoch"] != epoch:
            raise SealError(f"e{epoch} candidate-ready epoch changed")
        ready.append(artifact)
    short_status, short_status_payload = _strict_receipt(
        authority.get("short_quality_status"),
        "short-quality status",
        require_bytes=True,
    )
    authority_throughput = authority.get("throughput_gate")
    status_throughput = short_status_payload.get("throughput_gate")
    if (
        type(authority_throughput) is not dict
        or type(status_throughput) is not dict
        or authority_throughput != status_throughput
        or authority_throughput.get("topology_mode") != mode
    ):
        raise SealError(
            "short-quality topology mode differs from throughput authority"
        )
    val_inputs, _val_payload = _strict_receipt(
        preflight.get("val_inputs_receipt"),
        "validation inputs receipt",
        require_bytes=False,
    )
    pipeline, _pipeline_payload = _strict_receipt(
        preflight.get("pipeline_receipt"),
        "pipeline receipt",
        require_bytes=False,
    )
    lineages: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    command: List[str] = []
    for epoch in EPOCHS:
        candidate = validation_root / "candidates" / f"e{epoch}"
        lineage_path = candidate / "final" / "val-inference-lineage.json"
        lineage_payload, _lineage_inode = _safe_read(
            lineage_path, f"e{epoch} inference lineage"
        )
        lineage_json = _strict_json_bytes(
            lineage_payload, f"e{epoch} inference lineage"
        )
        lineage_receipt = lineage_json.get("receipt_payload_sha256")
        if (
            _require_sha(lineage_receipt, f"e{epoch} lineage payload")
            != _payload_sha(lineage_json, "receipt_payload_sha256")
        ):
            raise SealError(f"e{epoch} lineage payload closure changed")
        lineage = {
            "path": str(lineage_path),
            "sha256": _sha256_bytes(lineage_payload),
            "bytes": len(lineage_payload),
            "receipt_payload_sha256": lineage_receipt,
        }
        report, _report_payload = _report_artifact(
            candidate / "diffsheg-val-fgd.json", f"e{epoch} DiffSHEG report"
        )
        lineages.append(lineage)
        reports.append(report)
        command.extend(
            [
                "--inference-lineage",
                str(epoch),
                lineage["path"],
                lineage["sha256"],
                str(lineage["bytes"]),
                lineage["receipt_payload_sha256"],
                "--diffsheg-report",
                str(epoch),
                report["path"],
                report["sha256"],
                str(report["bytes"]),
            ]
        )
    topology_gate = authority.get("topology_gate_spec")
    quality_gate = authority.get("quality_gate_spec")
    if (
        type(topology_gate) is not dict
        or type(topology_gate.get("path")) is not str
        or type(quality_gate) is not dict
        or type(quality_gate.get("path")) is not str
    ):
        raise SealError("quality/topology gate authority is malformed")
    topology_path = _require_absolute_raw(
        Path(topology_gate["path"]), "topology gate path"
    )
    quality_path = _require_absolute_raw(
        Path(quality_gate["path"]), "quality gate path"
    )
    topology_payload, _ = _safe_read(topology_path, "topology gate")
    quality_payload, _ = _safe_read(quality_path, "quality gate")
    topology_sha = _require_sha(topology_gate.get("sha256"), "topology gate SHA")
    quality_sha = _require_sha(quality_gate.get("sha256"), "quality gate SHA")
    if _sha256_bytes(topology_payload) != topology_sha:
        raise SealError("topology gate SHA changed")
    if _sha256_bytes(quality_payload) != quality_sha:
        raise SealError("quality gate SHA changed")
    topology_gate_payload = _strict_json_bytes(topology_payload, "topology gate")
    candidate_modes = topology_gate_payload.get("candidate_modes")
    topology_matrix = topology_gate_payload.get("topology_matrix")
    topology_specification = (
        topology_matrix.get(mode) if type(topology_matrix) is dict else None
    )
    candidate_bundle = preflight.get("candidate_bundle")
    if (
        type(candidate_modes) is not list
        or any(type(item) is not str for item in candidate_modes)
        or mode not in candidate_modes
        or type(topology_specification) is not dict
        or type(candidate_bundle) is not dict
        or type(candidate_bundle.get("updates_per_epoch")) is not int
        or candidate_bundle.get("updates_per_epoch")
        != topology_specification.get("updates_per_epoch")
    ):
        raise SealError(
            "short-quality topology mode differs from gate/bundle authority"
        )
    closure = {
        "topology_mode": mode,
        "topology_gate_spec": {
            "path": str(topology_path),
            "sha256": topology_sha,
            "bytes": len(topology_payload),
        },
        "quality_gate_spec": {
            "path": str(quality_path),
            "sha256": quality_sha,
            "bytes": len(quality_payload),
        },
        "candidate_ready_receipts": ready,
        "short_quality_status": short_status,
        "val_inputs_receipt": val_inputs,
        "pipeline_receipt": pipeline,
        "validation_root": str(validation_root),
        "inference_lineages": lineages,
        "diffsheg_reports": reports,
        "candidate_epochs": list(EPOCHS),
        "coverage": {"clip_count": 1715},
        "split": "val",
        "test_visible": False,
    }
    common_command: List[str] = [
        "--mode",
        str(closure["topology_mode"]),
        "--topology-gate-spec",
        topology_gate["path"],
        "--expected-topology-gate-spec-sha256",
        topology_sha,
        "--quality-gate-spec",
        quality_gate["path"],
        "--expected-quality-gate-spec-sha256",
        quality_sha,
    ]
    for epoch, artifact in zip(EPOCHS, ready):
        common_command.extend(
            [
                "--candidate-ready-receipt",
                str(epoch),
                artifact["path"],
                artifact["sha256"],
                str(artifact["bytes"]),
                artifact["receipt_payload_sha256"],
            ]
        )
    common_command.extend(
        [
            "--short-quality-status",
            short_status["path"],
            short_status["sha256"],
            str(short_status["bytes"]),
            short_status["receipt_payload_sha256"],
            "--val-inputs-receipt",
            val_inputs["path"],
            val_inputs["sha256"],
            str(val_inputs["bytes"]),
            val_inputs["receipt_payload_sha256"],
            "--pipeline-receipt",
            pipeline["path"],
            pipeline["sha256"],
            str(pipeline["bytes"]),
            pipeline["receipt_payload_sha256"],
        ]
    )
    common_command.extend(command)
    return closure, common_command


def _write_new_at(directory_fd: int, name: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    if "/" in name or name in {"", ".", ".."}:
        raise SealError("unsafe create-new receipt name")
    payload = _pretty_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o400, dir_fd=directory_fd)
    except OSError as error:
        raise SealError(f"cannot create immutable {name}: {error}") from error
    try:
        os.fchmod(descriptor, 0o400)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        or named.st_size != len(payload)
        or stat.S_IMODE(named.st_mode) != 0o400
    ):
        raise SealError(f"immutable {name} inode changed during publication")
    os.fsync(directory_fd)
    return {
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "inode": [int(named.st_dev), int(named.st_ino)],
    }


def _open_canonical_directory(path: Path, label: str) -> int:
    path = _canonical_existing(path, label, directory=True)
    named = os.lstat(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        os.close(descriptor)
        raise SealError(f"{label} changed during open")
    return descriptor


def _open_private_directory(path: Path, label: str) -> int:
    descriptor = _open_canonical_directory(path, label)
    opened = os.fstat(descriptor)
    if stat.S_IMODE(opened.st_mode) != 0o700:
        os.close(descriptor)
        raise SealError(f"{label} must have mode 0700")
    return descriptor


def _create_campaign_root(path: Path) -> int:
    path = _require_absolute_raw(path, "campaign root")
    parent = _canonical_existing(path.parent, "campaign parent", directory=True)
    if os.path.lexists(path):
        raise FileExistsError(f"campaign root already exists: {path}")
    parent_fd = _open_canonical_directory(parent, "campaign parent")
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as error:
        raise SealError(f"could not atomically claim campaign root: {error}") from error
    finally:
        os.close(parent_fd)
    return _open_private_directory(path, "campaign root")


def _ensure_private_directory(path: Path, label: str) -> int:
    path = _require_absolute_raw(path, label)
    if os.path.lexists(path):
        return _open_private_directory(path, label)
    parent = _canonical_existing(path.parent, f"{label} parent", directory=True)
    parent_fd = _open_canonical_directory(parent, f"{label} parent")
    try:
        try:
            os.mkdir(path.name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
    finally:
        os.close(parent_fd)
    return _open_private_directory(path, label)


def _global_claim_value(
    campaign_id: str,
    campaign_root: Path,
    binding: Mapping[str, Any],
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "format": GLOBAL_CLAIM_FORMAT,
        "status": "claimed",
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "binding_sha256": _sha256_bytes(_canonical_json_bytes(binding)),
        "source": binding["source"],
        "evidence_source": binding["evidence_source"],
        "runtime": binding["runtime"],
        "sealer": binding["sealer"],
        "preflight": binding["preflight"],
        "producer": binding["producer"],
        "producer_may_run_once_globally": True,
    }
    value["receipt_payload_sha256"] = _payload_sha(
        value, "receipt_payload_sha256"
    )
    return value


def _acquire_global_claim(
    campaign_id: str,
    campaign_root: Path,
    binding: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    registry = CLAIM_REGISTRY
    registry_fd = _ensure_private_directory(registry, "global claim registry")
    try:
        name = f"{campaign_id}.json"
        path = registry / name
        expected = _global_claim_value(campaign_id, campaign_root, binding)
        if os.path.lexists(path):
            artifact, payload = _strict_artifact(path, "global campaign claim")
            receipt = payload.get("receipt_payload_sha256")
            if (
                payload != expected
                or _require_sha(receipt, "global claim payload")
                != _payload_sha(payload, "receipt_payload_sha256")
            ):
                raise SealError(
                    "this semantic campaign was already claimed for a different root"
                )
            artifact["path"] = str(path)
            artifact["receipt_payload_sha256"] = receipt
            return artifact, payload, False
        published = _write_new_at(registry_fd, name, expected)
        artifact = {
            "path": str(path),
            "sha256": published["sha256"],
            "bytes": published["bytes"],
            "receipt_payload_sha256": expected["receipt_payload_sha256"],
        }
        return artifact, expected, True
    finally:
        os.close(registry_fd)


def _existing_campaign_action(
    campaign_root: Path,
    expected_campaign_id: str,
    runtime: Mapping[str, Any],
    sealer: Mapping[str, Any],
    source: Mapping[str, Any],
    evidence_source: Mapping[str, Any],
    preflight_artifact: Mapping[str, Any],
    input_closure: Mapping[str, Any],
    global_claim: Mapping[str, Any],
) -> Dict[str, Any]:
    root_fd = _open_private_directory(campaign_root, "campaign root")
    try:
        claim_artifact, claim = _strict_artifact(campaign_root / CLAIM_NAME, "campaign claim")
        claim_payload = claim.get("receipt_payload_sha256")
        if (
            claim.get("format") != CLAIM_FORMAT
            or claim.get("status") != "claimed"
            or claim.get("campaign_id") != expected_campaign_id
            or claim.get("global_claim") != global_claim
            or _require_sha(claim_payload, "claim payload")
            != _payload_sha(claim, "receipt_payload_sha256")
        ):
            raise SealError("existing campaign claim differs from requested campaign")
        completion_path = campaign_root / SEALED_NAME / COMPLETION_NAME
        if completion_path.is_file() and not completion_path.is_symlink():
            _completion_artifact, completion = _strict_artifact(
                completion_path, "completion"
            )
            receipt = completion.get("receipt_payload_sha256")
            expected_claim = {
                "path": str(campaign_root / CLAIM_NAME),
                "sha256": claim_artifact["sha256"],
                "bytes": claim_artifact["bytes"],
                "receipt_payload_sha256": claim_payload,
            }
            if (
                completion.get("format") != COMPLETION_FORMAT
                or completion.get("status") != "complete"
                or completion.get("campaign_id") != expected_campaign_id
                or completion.get("campaign_claim") != expected_claim
                or completion.get("global_claim") != global_claim
                or completion.get("source") != source
                or completion.get("evidence_source") != evidence_source
                or completion.get("runtime") != runtime
                or completion.get("sealer") != sealer
                or completion.get("producer") != source["producer"]
                or completion.get("preflight") != preflight_artifact
                or completion.get("input_closure_sha256")
                != _sha256_bytes(_canonical_json_bytes(input_closure))
                or _require_sha(receipt, "completion payload")
                != _payload_sha(completion, "receipt_payload_sha256")
            ):
                raise SealError("existing completion is not the requested campaign")
            quality_path = campaign_root / SEALED_NAME / QUALITY_NAME
            short_path = campaign_root / SEALED_NAME / SHORT_NAME
            quality, _qj, short, _sj = _output_artifacts(
                quality_path, short_path
            )
            if (
                completion.get("quality_report") != quality
                or completion.get("short_trajectory") != short
            ):
                raise SealError("existing completion output closure changed")
            replay_outputs(
                runtime,
                source,
                str(input_closure["topology_mode"]),
                quality_path,
                quality["sha256"],
                input_closure["quality_gate_spec"]["sha256"],
                input_closure["topology_gate_spec"]["sha256"],
            )
            return completion
        sealed = campaign_root / SEALED_NAME
        failed = campaign_root / FAILED_NAME
        if sealed.exists() and not failed.exists():
            os.rename(SEALED_NAME, FAILED_NAME, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.fsync(root_fd)
        raise SealError("campaign was already consumed without a valid completion")
    finally:
        os.close(root_fd)


def _failure_receipt(
    campaign_id: str,
    stage: str,
    error: BaseException,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "format": FAILURE_FORMAT,
        "status": "terminal_failure",
        "campaign_id": campaign_id,
        "stage": stage,
        "error_class": type(error).__name__,
        "error": str(error)[:4000],
        "producer_may_not_be_restarted": True,
    }
    value["receipt_payload_sha256"] = _payload_sha(value, "receipt_payload_sha256")
    return value


def _finalize_failure(
    campaign_root: Path,
    campaign_id: str,
    stage: str,
    error: BaseException,
) -> None:
    root_fd = _open_private_directory(campaign_root, "campaign root")
    try:
        sealed = campaign_root / SEALED_NAME
        failed = campaign_root / FAILED_NAME
        if sealed.is_dir() and not sealed.is_symlink():
            try:
                sealed_fd = _open_private_directory(sealed, "failed staging directory")
                try:
                    if not os.path.lexists(sealed / "failure.json"):
                        _write_new_at(
                            sealed_fd,
                            "failure.json",
                            _failure_receipt(campaign_id, stage, error),
                        )
                finally:
                    os.close(sealed_fd)
            finally:
                if not failed.exists():
                    os.rename(
                        SEALED_NAME,
                        FAILED_NAME,
                        src_dir_fd=root_fd,
                        dst_dir_fd=root_fd,
                    )
                    os.fsync(root_fd)
    finally:
        os.close(root_fd)


def run_producer(
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
    common_command: Sequence[str],
    quality_path: Path,
    short_path: Path,
) -> Dict[str, Any]:
    command = [
        str(runtime["execution_path"]),
        "-I",
        str(source["producer"]["path"]),
        *list(common_command),
        "--short-trajectory-output",
        str(short_path),
        "--output",
        str(quality_path),
    ]
    process = _run(command, cwd=Path(str(source["source_root"])), timeout=3600)
    result = {
        "argv_sha256": _sha256_bytes(_canonical_json_bytes(command)),
        "return_code": int(process.returncode),
        "stdout_sha256": _sha256_bytes(process.stdout),
        "stderr_sha256": _sha256_bytes(process.stderr),
    }
    if process.returncode != 0:
        raise SealError(
            "official producer failed: "
            + process.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    return result


def replay_outputs(
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
    mode: str,
    quality_path: Path,
    quality_sha: str,
    quality_gate_sha: str,
    topology_gate_sha: str,
) -> Dict[str, Any]:
    code = (
        "import json,sys;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);"
        "from scripts.show_base import select_base_training_topology as s;"
        "r=s.validate_quality_report(sys.argv[2],Path(sys.argv[3]),sys.argv[4],"
        "quality_gate_spec_sha256=sys.argv[5],topology_gate_spec_sha256=sys.argv[6]);"
        "print(json.dumps({'mode':r['mode'],'candidate_fgd':r['candidate_fgd'],"
        "'report_sha256':r['report_sha256'],'test_visible':False,'split':'val'},"
        "sort_keys=True,allow_nan=False))"
    )
    command = [
        str(runtime["execution_path"]),
        "-I",
        "-c",
        code,
        str(source["source_root"]),
        mode,
        str(quality_path),
        quality_sha,
        quality_gate_sha,
        topology_gate_sha,
    ]
    process = _run(command, cwd=Path(str(source["source_root"])), timeout=600)
    if process.returncode != 0:
        raise SealError(
            "official output fresh replay failed: "
            + process.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    result = _strict_json_bytes(process.stdout.strip(), "official output replay")
    if (
        result.get("mode") != mode
        or result.get("report_sha256") != quality_sha
        or result.get("test_visible") is not False
        or result.get("split") != "val"
    ):
        raise SealError("official output replay returned a different report")
    return result


def _output_artifacts(
    quality_path: Path,
    short_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    quality, quality_json = _strict_artifact(quality_path, "quality report")
    short, short_json = _strict_artifact(short_path, "short trajectory")
    quality_receipt = _require_sha(
        quality_json.get("receipt_sha256"), "quality report receipt SHA"
    )
    short_payload = _require_sha(
        short_json.get("receipt_payload_sha256"), "short trajectory payload SHA"
    )
    if _payload_sha(quality_json, "receipt_sha256") != quality_receipt:
        raise SealError("quality report receipt closure changed")
    if _payload_sha(short_json, "receipt_payload_sha256") != short_payload:
        raise SealError("short trajectory payload closure changed")
    quality["receipt_sha256"] = quality_receipt
    short["receipt_payload_sha256"] = short_payload
    qstat = os.stat(quality_path, follow_symlinks=False)
    sstat = os.stat(short_path, follow_symlinks=False)
    if (qstat.st_dev, qstat.st_ino) == (sstat.st_dev, sstat.st_ino):
        raise SealError("quality and short outputs must be distinct inodes")
    embedded = quality_json.get("short_trajectory_receipt")
    if embedded != short:
        raise SealError("quality report does not bind the final short path exactly")
    return quality, quality_json, short, short_json


def _fsync_regular(path: Path, label: str) -> None:
    payload, identity = _safe_read(path, label)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if _identity(os.fstat(descriptor)) != identity:
            raise SealError(f"{label} changed before fsync")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not payload:
        raise SealError(f"{label} is unexpectedly empty")


def seal(args: argparse.Namespace) -> Dict[str, Any]:
    sealer = verify_sealer_authority(
        args.expected_sealer_commit,
        args.expected_sealer_blob_oid,
        args.expected_sealer_file_sha256,
    )
    runtime = verify_runtime()
    source = verify_source(args.source_root)
    evidence_source = verify_source(
        args.evidence_source_root,
        expected_commit=EXPECTED_EVIDENCE_SOURCE_COMMIT,
        expected_tree=EXPECTED_EVIDENCE_SOURCE_TREE,
        label="validation evidence source",
    )
    preflight_artifact, preflight = _strict_pinned_json(
        args.preflight,
        "short-quality preflight",
        args.expected_preflight_sha256,
        args.expected_preflight_bytes,
        args.expected_preflight_payload_sha256,
    )
    _require_source_binding(preflight, evidence_source)
    replay_preflight(runtime, evidence_source, preflight_artifact)
    input_closure, common_command = build_input_closure(
        preflight, args.validation_root
    )
    binding = {
        "source": source,
        "evidence_source": evidence_source,
        "runtime": runtime,
        "sealer": sealer,
        "preflight": preflight_artifact,
        "input_closure": input_closure,
        "producer": source["producer"],
        "outputs": {
            "quality_report": f"{SEALED_NAME}/{QUALITY_NAME}",
            "short_trajectory": f"{SEALED_NAME}/{SHORT_NAME}",
            "completion": f"{SEALED_NAME}/{COMPLETION_NAME}",
        },
        "candidate_epochs": list(EPOCHS),
        "split": "val",
        "test_visible": False,
    }
    campaign_id = _sha256_bytes(_canonical_json_bytes(binding))
    campaign_root = _require_absolute_raw(args.campaign_root, "campaign root")
    global_claim, _global_claim_payload, global_claim_created = (
        _acquire_global_claim(campaign_id, campaign_root, binding)
    )
    if os.path.lexists(campaign_root):
        return _existing_campaign_action(
            campaign_root,
            campaign_id,
            runtime,
            sealer,
            source,
            evidence_source,
            preflight_artifact,
            input_closure,
            global_claim,
        )
    if not global_claim_created:
        raise SealError(
            "global campaign claim exists but its campaign root is absent; "
            "the producer may not be started again"
        )
    root_fd = _create_campaign_root(campaign_root)
    stage = "claim"
    try:
        claim: Dict[str, Any] = {
            "format": CLAIM_FORMAT,
            "status": "claimed",
            "campaign_id": campaign_id,
            "producer_may_run_once": True,
            "global_claim": global_claim,
            "binding": binding,
        }
        claim["receipt_payload_sha256"] = _payload_sha(
            claim, "receipt_payload_sha256"
        )
        claim_file = _write_new_at(root_fd, CLAIM_NAME, claim)
        os.mkdir(SEALED_NAME, 0o700, dir_fd=root_fd)
        os.fsync(root_fd)
        sealed_path = campaign_root / SEALED_NAME
        sealed_fd = _open_private_directory(sealed_path, "sealed staging directory")
        try:
            quality_path = sealed_path / QUALITY_NAME
            short_path = sealed_path / SHORT_NAME
            stage = "official_producer"
            producer_result = run_producer(
                runtime,
                source,
                common_command,
                quality_path,
                short_path,
            )
            stage = "output_identity"
            quality, _quality_json, short, _short_json = _output_artifacts(
                quality_path, short_path
            )
            stage = "official_output_fresh_replay"
            replay = replay_outputs(
                runtime,
                source,
                str(input_closure["topology_mode"]),
                quality_path,
                quality["sha256"],
                input_closure["quality_gate_spec"]["sha256"],
                input_closure["topology_gate_spec"]["sha256"],
            )
            # Re-read after the independent replay; any pathname/inode swap is
            # fatal rather than normalized into the completion.
            quality_after, _qj, short_after, _sj = _output_artifacts(
                quality_path, short_path
            )
            if quality_after != quality or short_after != short:
                raise SealError("outputs changed across official fresh replay")
            if verify_source(args.source_root) != source:
                raise SealError("official source changed across producer execution")
            if (
                verify_source(
                    args.evidence_source_root,
                    expected_commit=EXPECTED_EVIDENCE_SOURCE_COMMIT,
                    expected_tree=EXPECTED_EVIDENCE_SOURCE_TREE,
                    label="validation evidence source",
                )
                != evidence_source
            ):
                raise SealError(
                    "validation evidence source changed across producer execution"
                )
            if verify_runtime() != runtime:
                raise SealError("formal Python runtime changed across producer execution")
            if (
                verify_sealer_authority(
                    args.expected_sealer_commit,
                    args.expected_sealer_blob_oid,
                    args.expected_sealer_file_sha256,
                )
                != sealer
            ):
                raise SealError("formal sealer changed across producer execution")
            _fsync_regular(quality_path, "quality report")
            _fsync_regular(short_path, "short trajectory")
            os.fsync(sealed_fd)
            stage = "completion_publish"
            completion: Dict[str, Any] = {
                "format": COMPLETION_FORMAT,
                "status": "complete",
                "campaign_id": campaign_id,
                "split": "val",
                "test_visible": False,
                "candidate_epochs": list(EPOCHS),
                "campaign_claim": {
                    "path": str(campaign_root / CLAIM_NAME),
                    "sha256": claim_file["sha256"],
                    "bytes": claim_file["bytes"],
                    "receipt_payload_sha256": claim[
                        "receipt_payload_sha256"
                    ],
                },
                "global_claim": global_claim,
                "source": source,
                "evidence_source": evidence_source,
                "runtime": runtime,
                "sealer": sealer,
                "producer": source["producer"],
                "preflight": preflight_artifact,
                "input_closure_sha256": _sha256_bytes(
                    _canonical_json_bytes(input_closure)
                ),
                "quality_report": quality,
                "short_trajectory": short,
                "official_producer_execution": producer_result,
                "official_fresh_replay": replay,
                "completion_is_sole_success_marker": True,
            }
            completion["receipt_payload_sha256"] = _payload_sha(
                completion, "receipt_payload_sha256"
            )
            _write_new_at(sealed_fd, COMPLETION_NAME, completion)
            os.fsync(sealed_fd)
            os.fsync(root_fd)
            return completion
        finally:
            os.close(sealed_fd)
    except BaseException as error:
        try:
            _finalize_failure(campaign_root, campaign_id, stage, error)
        except BaseException as cleanup_error:
            raise SealError(
                f"campaign failed at {stage}; failure finalization also failed: "
                f"{cleanup_error}"
            ) from error
        raise
    finally:
        os.close(root_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evidence-source-root", type=Path, required=True)
    parser.add_argument("--expected-sealer-commit", required=True)
    parser.add_argument("--expected-sealer-blob-oid", required=True)
    parser.add_argument("--expected-sealer-file-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--expected-preflight-bytes", type=int, required=True)
    parser.add_argument("--expected-preflight-payload-sha256", required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = seal(parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
