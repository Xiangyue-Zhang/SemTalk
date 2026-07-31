#!/usr/bin/env python3
"""Fail-closed two-node transaction for guarded GPU workloads.

The transaction root must not exist.  Rank zero creates it and publishes an
immutable bootstrap receipt; rank one only joins that exact transaction.  A
node publishes immutable PREPARED and ARMED receipts with ``O_EXCL`` and
``fsync``.  A non-GPU supervisor pins an otherwise empty process group; the
actual workload cannot exec until the single immutable DECISION is GO.  After
both exact workload results, rank zero publishes a single atomic OUTCOME.
Both nodes maintain heartbeats.  A peer failure, abort, stale heartbeat,
identity mismatch, or timeout terminates the exact local group before returning
non-zero to the outer guarded runner.

Filesystem device/inode identities are deliberately node-local evidence.
They never participate in portable-payload equality, which makes the protocol
valid when two hosts see a shared transaction through different mounts.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


SCHEMA = "semtalk.dual_node_guarded_transaction.v1"
PORTABLE_SCHEMA = f"{SCHEMA}.portable"
EXPECTED_RUNNER = "/tmp/globaldiff_guarded_runner.py"
EXPECTED_RUNNER_GPUS = "0,1,2,3,4,5,6,7"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_RECEIPT_BYTES = 1 << 20
EXPECTED_RANKS = (0, 1)
IDENTITY_KEYS = {
    "pid", "ppid", "pgid", "sid", "starttime_ticks", "argv_sha256"
}

# CPU tests import this module and replace the process-evidence provider.  The
# installed CLI never exposes a switch that weakens Linux runner ancestry or
# pinned-fd execution.
_CPU_TEST_MODE = False


class TransactionError(RuntimeError):
    """A fail-closed transaction error."""


class DuplicateInvocation(TransactionError):
    """The transaction or this node's immutable claim already exists."""


class PeerAbort(TransactionError):
    """The peer failed, aborted, became stale, or violated the protocol."""


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _argv_sha256(argv: Sequence[str]) -> str:
    return _sha256_bytes(b"\0".join(os.fsencode(token) for token in argv) + b"\0")


def _exact_int(value: Any, *, minimum: int | None = None) -> bool:
    return type(value) is int and (minimum is None or value >= minimum)


def _proc_identity(pid: int) -> dict[str, Any]:
    if pid <= 0:
        raise TransactionError("invalid process ID")
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not stat_path.exists() or not cmdline_path.exists():
        # The formal hosts are Linux and always take the /proc path below.
        # This narrow fallback keeps the protocol's CPU-only process tests
        # runnable on a developer macOS host; it still snapshots PID, start
        # token, and argv twice and rejects an unstable identity.
        try:
            command = [
                "/bin/ps", "-p", str(pid), "-o", "ppid=", "-o", "pgid=",
                "-o", "sess=", "-o", "lstart=", "-o", "command=",
            ]
            snapshot_before = subprocess.check_output(command)
            snapshot_after = subprocess.check_output(command)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TransactionError(f"process identity unavailable: {pid}") from exc
        if not snapshot_before or snapshot_before != snapshot_after:
            raise TransactionError(
                f"process identity changed during fallback snapshot: {pid}"
            )
        first_line = snapshot_before.splitlines()[0]
        # ``lstart`` is five whitespace-delimited fields; split only through
        # that fixed prefix so the remaining bytes are the command alone.
        fields = first_line.split(maxsplit=8)
        if len(fields) != 9 or not all(item.isdigit() for item in fields[:3]):
            raise TransactionError(f"invalid fallback process identity: {pid}")
        try:
            pgid = os.getpgid(pid)
            sid = os.getsid(pid)
        except OSError as exc:
            raise TransactionError(f"process identity unavailable: {pid}") from exc
        return {
            "pid": pid,
            "ppid": int(fields[0]),
            "pgid": pgid,
            "sid": sid,
            "starttime_ticks": int(
                hashlib.sha256(b" ".join(fields[3:8])).hexdigest()[:15], 16
            ),
            "argv_sha256": _sha256_bytes(fields[8]),
        }
    try:
        stat_line_before = stat_path.read_text(encoding="utf-8")
        argv_bytes = cmdline_path.read_bytes()
        stat_line_after = stat_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TransactionError(f"process identity unavailable: {pid}") from exc
    def parse_starttime(stat_line: str) -> int:
        if ") " not in stat_line:
            raise TransactionError(f"invalid process stat record: {pid}")
        fields = stat_line.rsplit(") ", 1)[1].split()
        if len(fields) < 20 or fields[0] == "Z" or not fields[19].isdigit():
            raise TransactionError(f"invalid or zombie process identity: {pid}")
        return int(fields[19])

    starttime_before = parse_starttime(stat_line_before)
    starttime_after = parse_starttime(stat_line_after)
    if starttime_before != starttime_after:
        raise TransactionError(f"process identity changed during snapshot: {pid}")
    if not argv_bytes:
        raise TransactionError(f"empty process argv: {pid}")
    return {
        "pid": pid,
        "ppid": int(stat_line_after.rsplit(") ", 1)[1].split()[1]),
        "pgid": int(stat_line_after.rsplit(") ", 1)[1].split()[2]),
        "sid": int(stat_line_after.rsplit(") ", 1)[1].split()[3]),
        "starttime_ticks": starttime_before,
        "argv_sha256": _sha256_bytes(argv_bytes),
    }


def _runner_option_values(tokens: Sequence[str], names: set[str]) -> list[str]:
    values: list[str] = []
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        if token in names:
            if index + 1 >= len(tokens) or tokens[index + 1] == "--":
                raise TransactionError(f"runner option has no value: {token}")
            values.append(tokens[index + 1])
            index += 2
            continue
        for name in names:
            prefix = f"{name}="
            if token.startswith(prefix):
                values.append(token[len(prefix) :])
                break
        index += 1
    return values


def _runner_evidence(
    expected_status_path: str,
    expected_log_path: str,
) -> tuple[dict[str, Any], str]:
    pid = os.getppid()
    identity = _proc_identity(pid)
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if not proc_cmdline.is_file():
        raise TransactionError("formal guarded-runner evidence requires Linux /proc")
    raw_cmdline = proc_cmdline.read_bytes()
    if not raw_cmdline.endswith(b"\0"):
        raise TransactionError("incomplete guarded-runner argv")
    tokens = [os.fsdecode(token) for token in raw_cmdline.split(b"\0") if token]
    try:
        delimiter = tokens.index("--", 2)
    except ValueError as exc:
        raise TransactionError("guarded runner has no workload delimiter") from exc
    python_name = Path(tokens[0]).name if tokens else ""
    if (
        len(tokens) < 5
        or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", python_name)
        or tokens[1] != EXPECTED_RUNNER
        or EXPECTED_RUNNER in tokens[2:delimiter]
    ):
        raise TransactionError("coordinator is not the exact guarded-runner child")
    gpu_values = _runner_option_values(tokens, {"--gpus"})
    if gpu_values != [EXPECTED_RUNNER_GPUS]:
        raise TransactionError("guarded runner lacks one exact all-GPU reservation")
    status_values = _runner_option_values(
        tokens, {"--status", "--status-path", "--status-file"}
    )
    log_values = _runner_option_values(
        tokens, {"--log", "--log-path", "--log-file"}
    )
    if status_values != [expected_status_path] or log_values != [expected_log_path]:
        raise TransactionError("runner status/log evidence is not argv-bound")
    runner_path = Path(EXPECTED_RUNNER)
    runner_fd, runner_path = _open_pinned_file(runner_path, executable=False)
    try:
        runner_sha256 = _sha256_fd(runner_fd)
    finally:
        os.close(runner_fd)
    if os.getppid() != pid or _proc_identity(pid) != identity:
        raise TransactionError("guarded-runner identity changed during evidence capture")
    identity.update(
        {
            "path": str(runner_path),
            "sha256": runner_sha256,
            "status_path": expected_status_path,
            "log_path": expected_log_path,
        }
    )
    return identity, runner_sha256


def _open_pinned_file(path: Path, *, executable: bool) -> tuple[int, Path]:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise TransactionError("pinned file path must be safe and absolute")
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise TransactionError(f"pinned file is unavailable: {path}") from exc
    if canonical != path or path.is_symlink():
        raise TransactionError(f"pinned file path is noncanonical or a symlink: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    info = os.fstat(fd)
    public = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or (info.st_dev, info.st_ino) != (public.st_dev, public.st_ino)
        or (executable and info.st_mode & 0o111 == 0)
    ):
        os.close(fd)
        raise TransactionError(f"invalid pinned file: {path}")
    return fd, canonical


def _validate_absolute_evidence_path(raw: str, label: str) -> str:
    path = Path(raw)
    if not path.is_absolute():
        raise TransactionError(f"{label} must be absolute")
    if path.name in {"", ".", ".."}:
        raise TransactionError(f"unsafe {label}")
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise TransactionError(f"{label} parent is unavailable") from exc
    if resolved_parent != path.parent:
        raise TransactionError(f"{label} parent must be canonical and symlink-free")
    try:
        current = path.lstat()
    except FileNotFoundError:
        return str(path)
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise TransactionError(f"{label} must be a regular non-symlink file")
    return str(path)


def _source_evidence(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repository), *arguments],
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TransactionError("source repository evidence is unavailable") from exc

    def git_bytes(*arguments: str) -> bytes:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repository), *arguments],
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TransactionError("source blob evidence is unavailable") from exc

    origin = git("remote", "get-url", "origin")
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if (
        origin != EXPECTED_ORIGIN
        or commit != args.source_commit
        or tree != args.source_tree
        or dirty
    ):
        raise TransactionError("source origin/commit/tree/cleanliness mismatch")
    files: dict[str, str] = {}
    for relative in (
        "scripts/show_base/dual_node_guarded_transaction.py",
        "scripts/show_base/run_dual_node_guarded_transaction.sh",
        "scripts/show_base/guarded_runner_contract.sh",
    ):
        fd, canonical = _open_pinned_file(repository / relative, executable=False)
        try:
            files[relative] = _sha256_fd(fd)
        finally:
            os.close(fd)
        if canonical != repository / relative:
            raise TransactionError(f"noncanonical source entrypoint: {relative}")
        expected_sha256 = _sha256_bytes(git_bytes("show", f"HEAD:{relative}"))
        if files[relative] != expected_sha256:
            raise TransactionError(f"source entrypoint differs from HEAD: {relative}")
    if (
        git("remote", "get-url", "origin") != origin
        or git("rev-parse", "HEAD") != commit
        or git("rev-parse", "HEAD^{tree}") != tree
        or git("status", "--porcelain", "--untracked-files=all")
    ):
        raise TransactionError("source evidence changed during verification")
    return {
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "entrypoint_sha256": files,
    }


def _workload_evidence(
    args: argparse.Namespace,
    workload: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str], int, int, dict[str, int]]:
    executable_fd, executable = _open_pinned_file(
        Path(workload[0]), executable=True
    )
    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        os.close(executable_fd)
        raise TransactionError("workdir must be absolute")
    try:
        canonical_workdir = workdir.resolve(strict=True)
    except OSError as exc:
        os.close(executable_fd)
        raise TransactionError("workdir is unavailable") from exc
    if canonical_workdir != workdir or not workdir.is_dir():
        os.close(executable_fd)
        raise TransactionError("workdir must be a canonical real directory")
    workdir_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        workdir_flags |= os.O_NOFOLLOW
    workdir_fd = os.open(workdir, workdir_flags)
    opened_workdir = os.fstat(workdir_fd)
    public_workdir = workdir.lstat()
    if (opened_workdir.st_dev, opened_workdir.st_ino) != (
        public_workdir.st_dev,
        public_workdir.st_ino,
    ):
        os.close(workdir_fd)
        os.close(executable_fd)
        raise TransactionError("workdir identity changed while pinning")

    allow_names = list(args.allow_env)
    if len(allow_names) != len(set(allow_names)):
        os.close(executable_fd)
        os.close(workdir_fd)
        raise TransactionError("allow-env names must be unique")
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    for name in sorted(allow_names):
        if (
            not ENV_NAME_RE.fullmatch(name)
            or name.startswith("SEMTALK_W16_")
            or name not in os.environ
        ):
            os.close(executable_fd)
            os.close(workdir_fd)
            raise TransactionError(f"invalid or unavailable allow-env name: {name}")
        environment[name] = os.environ[name]

    input_fds: dict[str, int] = {}
    input_hashes: dict[str, str] = {}
    try:
        for raw in args.workload_input:
            if "=" not in raw:
                raise TransactionError("workload-input must be LOGICAL_ID=/absolute/path")
            logical_id, raw_path = raw.split("=", 1)
            if not SAFE_ID_RE.fullmatch(logical_id) or logical_id in input_fds:
                raise TransactionError("workload-input logical IDs must be safe and unique")
            fd, _ = _open_pinned_file(Path(raw_path), executable=False)
            input_fds[logical_id] = fd
            input_hashes[logical_id] = _sha256_fd(fd)
    except BaseException:
        for fd in input_fds.values():
            os.close(fd)
        os.close(workdir_fd)
        os.close(executable_fd)
        raise
    evidence = {
        "argv_sha256": _argv_sha256(workload),
        "executable_path": str(executable),
        "executable_sha256": _sha256_fd(executable_fd),
        "workdir": str(canonical_workdir),
        "environment": environment,
        "input_sha256": input_hashes,
    }
    return evidence, environment, executable_fd, workdir_fd, input_fds


def _validate_tx_path(raw: str, run_id: str) -> tuple[Path, Path, str]:
    path = Path(raw)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise TransactionError("transaction root must be a safe absolute path")
    if not SAFE_ID_RE.fullmatch(path.name):
        raise TransactionError("unsafe transaction-root basename")
    if path.name != run_id:
        raise TransactionError("transaction root must be namespace/run_id")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise TransactionError("transaction parent is unavailable") from exc
    if parent != path.parent:
        raise TransactionError("transaction parent must be canonical and symlink-free")
    return path, parent, path.name


def _participant(node_id: str, rank: int, ip: str) -> dict[str, Any]:
    if not SAFE_ID_RE.fullmatch(node_id):
        raise TransactionError("unsafe node ID")
    if rank not in EXPECTED_RANKS:
        raise TransactionError("node rank must be zero or one")
    try:
        normalized_ip = str(ipaddress.ip_address(ip))
    except ValueError as exc:
        raise TransactionError("invalid node IP address") from exc
    return {"node_id": node_id, "rank": rank, "ip": normalized_ip}


def _portable_payload(
    args: argparse.Namespace,
    runner_sha256: str,
    source: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    if not SAFE_ID_RE.fullmatch(args.run_id):
        raise TransactionError("unsafe run ID")
    if not HEX40_RE.fullmatch(args.source_commit) or not HEX40_RE.fullmatch(
        args.source_tree
    ):
        raise TransactionError("source commit/tree must be lowercase 40-hex IDs")
    if not HEX64_RE.fullmatch(args.common_command_sha256):
        raise TransactionError("common-command SHA-256 must be lowercase 64-hex")
    if args.common_command_sha256 != workload["argv_sha256"]:
        raise TransactionError("common-command SHA-256 does not bind actual argv")
    if args.max_restarts != 0:
        raise TransactionError("max_restarts must be exactly zero")
    integer_timeouts = {
        "heartbeat_ms": args.heartbeat_ms,
        "stale_ms": args.stale_ms,
        "prepare_timeout_ms": args.prepare_timeout_ms,
        "decision_timeout_ms": args.decision_timeout_ms,
        "completion_timeout_ms": args.completion_timeout_ms,
        "arm_timeout_ms": args.arm_timeout_ms,
        "start_timeout_ms": args.start_timeout_ms,
        "shutdown_grace_ms": args.shutdown_grace_ms,
    }
    if any(value <= 0 for value in integer_timeouts.values()):
        raise TransactionError("all transaction intervals must be positive")
    if args.stale_ms <= args.heartbeat_ms:
        raise TransactionError("stale timeout must exceed the heartbeat interval")
    local = _participant(args.node_id, args.node_rank, args.node_ip)
    peer = _participant(args.peer_node_id, args.peer_node_rank, args.peer_node_ip)
    if local["rank"] == peer["rank"] or local["node_id"] == peer["node_id"]:
        raise TransactionError("participants must have distinct IDs and ranks")
    participants = sorted((local, peer), key=lambda item: item["rank"])
    if tuple(item["rank"] for item in participants) != EXPECTED_RANKS:
        raise TransactionError("participants must cover ranks zero and one")
    return {
        "schema": PORTABLE_SCHEMA,
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "source": dict(source),
        "runner_sha256": runner_sha256,
        "common_command_sha256": args.common_command_sha256,
        "workload": dict(workload),
        "participants": participants,
        "max_restarts": 0,
        "timeouts": integer_timeouts,
    }


class TransactionDirectory:
    """Held namespace chain plus public-path identity checks.

    Every component from the process root to the trusted transaction namespace
    remains open.  Replacing any public ancestor therefore fails the next
    operation instead of silently detaching a live transaction.
    """

    def __init__(self, path: Path, parent: Path, basename: str, rank: int):
        self.path = path
        self.parent = parent
        self.basename = basename
        self.rank = rank
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.root_fd = -1
        self.root_identity: tuple[int, int] | None = None
        self.namespace_fds = [os.open("/", flags)]
        self.namespace_names: list[str] = []
        for component in parent.parts[1:]:
            child_fd = os.open(component, flags, dir_fd=self.namespace_fds[-1])
            child = os.fstat(child_fd)
            if not stat.S_ISDIR(child.st_mode):
                os.close(child_fd)
                raise TransactionError("transaction namespace component is not a directory")
            self.namespace_names.append(component)
            self.namespace_fds.append(child_fd)
        self.parent_fd = self.namespace_fds[-1]
        namespace = os.fstat(self.parent_fd)
        if namespace.st_uid != os.geteuid() or namespace.st_mode & 0o022:
            self.close()
            raise TransactionError("transaction namespace has unsafe ownership or mode")

    def _assert_namespace_identity(self) -> None:
        for index, name in enumerate(self.namespace_names):
            parent_fd = self.namespace_fds[index]
            child_fd = self.namespace_fds[index + 1]
            opened = os.fstat(child_fd)
            try:
                public = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise TransactionError("transaction namespace disappeared or changed") from exc
            if (
                not stat.S_ISDIR(public.st_mode)
                or (opened.st_dev, opened.st_ino) != (public.st_dev, public.st_ino)
            ):
                raise TransactionError("public transaction namespace identity changed")

    def create_or_wait(self, deadline: float) -> None:
        self._assert_namespace_identity()
        if self.rank == 0:
            try:
                os.mkdir(self.basename, mode=0o700, dir_fd=self.parent_fd)
                os.fsync(self.parent_fd)
            except FileExistsError as exc:
                raise DuplicateInvocation("transaction root already exists") from exc
        while True:
            try:
                flags = os.O_RDONLY | os.O_DIRECTORY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                self.root_fd = os.open(self.basename, flags, dir_fd=self.parent_fd)
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise TransactionError("timed out waiting for transaction root")
                time.sleep(0.01)
            except OSError as exc:
                raise TransactionError("transaction root is not a real directory") from exc
        opened = os.fstat(self.root_fd)
        if opened.st_uid != os.geteuid() or opened.st_mode & 0o022:
            raise TransactionError("transaction root has unsafe ownership or mode")
        self.root_identity = (opened.st_dev, opened.st_ino)
        self.assert_identity()

    def assert_identity(self) -> None:
        self._assert_namespace_identity()
        if self.root_fd < 0 or self.root_identity is None:
            raise TransactionError("transaction root is not open")
        opened = os.fstat(self.root_fd)
        try:
            public = os.stat(
                self.basename,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise TransactionError("transaction root disappeared or changed") from exc
        opened_identity = (opened.st_dev, opened.st_ino)
        public_identity = (public.st_dev, public.st_ino)
        if not stat.S_ISDIR(public.st_mode) or opened_identity != self.root_identity:
            raise TransactionError("held transaction-root identity changed")
        if public_identity != self.root_identity:
            raise TransactionError("public transaction-root identity changed")

    def local_filesystem_evidence(self) -> dict[str, int]:
        self.assert_identity()
        opened = os.fstat(self.root_fd)
        return {"st_dev": opened.st_dev, "st_ino": opened.st_ino}

    def _open_readonly(self, name: str) -> int:
        self.assert_identity()
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(name, flags, dir_fd=self.root_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise TransactionError(f"unsafe transaction artifact: {name}") from exc
        info = os.fstat(fd)
        expected_mode = 0o600 if name.startswith("HEARTBEAT.rank") else 0o400
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != expected_mode
            or info.st_size > MAX_RECEIPT_BYTES
        ):
            os.close(fd)
            raise TransactionError(f"invalid transaction artifact: {name}")
        return fd

    def read_bytes(self, name: str) -> bytes:
        fd = self._open_readonly(name)
        try:
            before = os.fstat(fd)
            blocks: list[bytes] = []
            total = 0
            while True:
                block = os.read(fd, 65536)
                if not block:
                    break
                total += len(block)
                if total > MAX_RECEIPT_BYTES:
                    raise TransactionError(f"transaction artifact too large: {name}")
                blocks.append(block)
            raw = b"".join(blocks)
            after = os.fstat(fd)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise TransactionError(f"transaction artifact changed while reading: {name}")
            return raw
        finally:
            os.close(fd)

    def read_json(self, name: str) -> tuple[dict[str, Any], str]:
        raw = self.read_bytes(name)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransactionError(f"invalid JSON transaction artifact: {name}") from exc
        if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
            raise TransactionError(f"noncanonical transaction artifact: {name}")
        return payload, _sha256_bytes(raw)

    def exists(self, name: str) -> bool:
        self.assert_identity()
        try:
            info = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise TransactionError(f"unsafe transaction artifact: {name}")
        return True

    def publish_immutable(self, name: str, payload: Mapping[str, Any]) -> str:
        """Publish a complete immutable file atomically without overwriting."""
        self.assert_identity()
        raw = _canonical_json_bytes(payload)
        temporary = f".tmp.{name}.{os.getpid()}.{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temporary, flags, 0o600, dir_fd=self.root_fd)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fchmod(fd, 0o400)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=self.root_fd,
                dst_dir_fd=self.root_fd,
                follow_symlinks=False,
            )
            os.fsync(self.root_fd)
        except FileExistsError as exc:
            raise DuplicateInvocation(f"immutable transaction artifact exists: {name}") from exc
        finally:
            try:
                os.unlink(temporary, dir_fd=self.root_fd)
                os.fsync(self.root_fd)
            except FileNotFoundError:
                pass
        return _sha256_bytes(raw)

    def replace_heartbeat(self, name: str, payload: Mapping[str, Any]) -> None:
        self.assert_identity()
        if self.exists(name):
            # ``exists`` rejects a symlink or non-regular replacement target.
            pass
        raw = _canonical_json_bytes(payload)
        temporary = f".tmp.{name}.{os.getpid()}.{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temporary, flags, 0o600, dir_fd=self.root_fd)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(
                temporary,
                name,
                src_dir_fd=self.root_fd,
                dst_dir_fd=self.root_fd,
            )
            os.fsync(self.root_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.root_fd)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        if self.root_fd >= 0:
            os.close(self.root_fd)
            self.root_fd = -1
        for fd in reversed(self.namespace_fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self.namespace_fds = []
        self.parent_fd = -1


def _set_supervisor_process_controls(parent_pid: int) -> None:
    if sys.platform != "linux":
        if _CPU_TEST_MODE:
            return
        raise TransactionError("formal supervisor requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG and PR_SET_CHILD_SUBREAPER.
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
        raise TransactionError(f"cannot configure exact supervisor: errno={ctypes.get_errno()}")
    if os.getppid() != parent_pid:
        raise TransactionError("coordinator exited while supervisor was arming")


def _group_members(pgid: int) -> set[int]:
    members: set[int] = set()
    if Path("/proc").is_dir():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                line = (entry / "stat").read_text(encoding="utf-8")
                fields = line.rsplit(") ", 1)[1].split()
                if len(fields) >= 3 and int(fields[2]) == pgid:
                    members.add(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
        return members
    if not _CPU_TEST_MODE:
        raise TransactionError("exact process-group inspection requires /proc")
    try:
        output = subprocess.check_output(["/bin/ps", "-axo", "pid=,pgid="], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransactionError("cannot inspect exact test process group") from exc
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            if int(fields[1]) == pgid:
                members.add(int(fields[0]))
    return members


def _write_supervisor_event(fd: int, payload: Mapping[str, Any]) -> None:
    raw = _canonical_json_bytes(payload)
    view = memoryview(raw)
    while view:
        view = view[os.write(fd, view) :]


def _wait_status_returncode(status_value: int) -> int:
    if os.WIFEXITED(status_value):
        return os.WEXITSTATUS(status_value)
    if os.WIFSIGNALED(status_value):
        return -os.WTERMSIG(status_value)
    return 255


def _kill_exact_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _supervisor_cleanup_group(anchor_pid: int, grace_seconds: float) -> None:
    _kill_exact_group(anchor_pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        members = _group_members(anchor_pid)
        if members <= {anchor_pid}:
            break
        time.sleep(0.01)
    # The anchor is deliberately still in this group, so its PGID cannot have
    # been recycled before this exact final kill.  Always target the group:
    # /proc inspection is advisory and a missed member must not escape.
    _kill_exact_group(anchor_pid, signal.SIGKILL)
    reap_deadline = time.monotonic() + max(grace_seconds, 0.2)
    while time.monotonic() < reap_deadline:
        try:
            waited, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if waited == 0:
            time.sleep(0.01)


def _supervisor_main(
    parent_pid: int,
    control_fd: int,
    event_fd: int,
    executable_fd: int,
    workdir_fd: int,
    workload: Sequence[str],
    environment: Mapping[str, str],
    expected_argv_sha256: str,
    executable_sha256: str,
    shutdown_grace_seconds: float,
) -> None:
    anchor_pid = -1
    abort_requested = False

    def request_abort(_signum: int, _frame: Any) -> None:
        nonlocal abort_requested
        abort_requested = True

    try:
        os.setsid()
        _set_supervisor_process_controls(parent_pid)
        signal.signal(signal.SIGTERM, request_abort)
        signal.signal(signal.SIGINT, request_abort)
        os.set_blocking(control_fd, False)

        anchor_read, anchor_write = os.pipe()
        ready_read, ready_write = os.pipe()
        anchor_pid = os.fork()
        if anchor_pid == 0:
            try:
                os.close(anchor_write)
                os.close(ready_read)
                os.close(control_fd)
                os.close(event_fd)
                os.setpgid(0, 0)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                os.write(ready_write, b"1")
                os.close(ready_write)
                while os.read(anchor_read, 1):
                    pass
            finally:
                os._exit(0)
        os.close(anchor_read)
        os.close(ready_write)
        if os.read(ready_read, 1) != b"1":
            raise TransactionError("workgroup anchor failed to arm")
        os.close(ready_read)
        _write_supervisor_event(
            event_fd,
            {"event": "LOCAL_ARMED", "workgroup_pgid": anchor_pid},
        )

        command = b""
        while not command and not abort_requested:
            if os.getppid() != parent_pid:
                abort_requested = True
                break
            readable, _, _ = select.select([control_fd], [], [], 0.05)
            if readable:
                try:
                    command = os.read(control_fd, 1)
                except BlockingIOError:
                    pass
        if abort_requested or command != b"G":
            _supervisor_cleanup_group(anchor_pid, shutdown_grace_seconds)
            os.close(anchor_write)
            _write_supervisor_event(
                event_fd, {"event": "RESULT", "returncode": None, "aborted": True}
            )
            os._exit(0)

        workload_pid = os.fork()
        if workload_pid == 0:
            try:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.setpgid(0, anchor_pid)
                identity = _proc_identity(os.getpid())
                identity["preexec_argv_sha256"] = identity.pop("argv_sha256")
                identity.update(
                    {
                        "expected_argv_sha256": expected_argv_sha256,
                        "executable_sha256": executable_sha256,
                    }
                )
                _write_supervisor_event(
                    event_fd, {"event": "STARTED", "workload": identity}
                )
                os.fchdir(workdir_fd)
                if os.execve in os.supports_fd:
                    os.set_inheritable(executable_fd, True)
                    os.execve(executable_fd, list(workload), dict(environment))
                if _CPU_TEST_MODE:
                    os.execve(workload[0], list(workload), dict(environment))
                raise TransactionError("platform cannot exec the pinned workload fd")
            except BaseException as exc:
                try:
                    _write_supervisor_event(
                        event_fd,
                        {"event": "EXEC_ERROR", "reason": f"{type(exc).__name__}: {exc}"},
                    )
                finally:
                    os._exit(127)

        status_value: int | None = None
        abort_deadline: float | None = None
        while status_value is None:
            waited, observed = os.waitpid(workload_pid, os.WNOHANG)
            if waited == workload_pid:
                status_value = observed
                break
            if abort_requested or os.getppid() != parent_pid:
                if abort_deadline is None:
                    _kill_exact_group(anchor_pid, signal.SIGTERM)
                    abort_deadline = time.monotonic() + shutdown_grace_seconds
                elif time.monotonic() >= abort_deadline:
                    _kill_exact_group(anchor_pid, signal.SIGKILL)
            time.sleep(0.01)
        returncode = _wait_status_returncode(status_value)
        _supervisor_cleanup_group(anchor_pid, shutdown_grace_seconds)
        os.close(anchor_write)
        _write_supervisor_event(
            event_fd,
            {"event": "RESULT", "returncode": returncode, "aborted": abort_requested},
        )
    except BaseException as exc:
        if anchor_pid > 0:
            _kill_exact_group(anchor_pid, signal.SIGKILL)
        try:
            _write_supervisor_event(
                event_fd,
                {"event": "SUPERVISOR_ERROR", "reason": f"{type(exc).__name__}: {exc}"},
            )
        except BaseException:
            pass
    finally:
        os._exit(0)


class Coordinator:
    def __init__(
        self,
        args: argparse.Namespace,
        workload: Sequence[str],
        runner: Mapping[str, Any],
        portable: Mapping[str, Any],
        tx: TransactionDirectory,
        workload_environment: Mapping[str, str],
        executable_fd: int,
        workdir_fd: int,
        workload_input_fds: Mapping[str, int],
    ):
        self.args = args
        self.workload = list(workload)
        self.runner = dict(runner)
        self.portable = dict(portable)
        self.portable_sha256 = _sha256_bytes(_canonical_json_bytes(self.portable))
        self.tx = tx
        self.workload_environment = dict(workload_environment)
        self.executable_fd = executable_fd
        self.workdir_fd = workdir_fd
        self.workload_input_fds = dict(workload_input_fds)
        self.rank = args.node_rank
        self.peer_rank = args.peer_node_rank
        self.prepared_name = f"PREPARED.rank{self.rank}.json"
        self.peer_prepared_name = f"PREPARED.rank{self.peer_rank}.json"
        self.armed_name = f"ARMED.rank{self.rank}.json"
        self.peer_armed_name = f"ARMED.rank{self.peer_rank}.json"
        self.started_name = f"STARTED.rank{self.rank}.json"
        self.peer_started_name = f"STARTED.rank{self.peer_rank}.json"
        self.result_name = f"WORKLOAD_RESULT.rank{self.rank}.json"
        self.peer_result_name = f"WORKLOAD_RESULT.rank{self.peer_rank}.json"
        self.final_name = f"FINAL.rank{self.rank}.json"
        self.heartbeat_name = f"HEARTBEAT.rank{self.rank}.json"
        self.peer_heartbeat_name = f"HEARTBEAT.rank{self.peer_rank}.json"
        self.prepared_sha256 = ""
        self.armed_sha256 = ""
        self.started_sha256 = ""
        self.workload_result_sha256 = ""
        self.final_sha256 = ""
        self.sequence = 0
        self.state = "STARTING"
        self.abort_requested = False
        self.coordinator_identity = _proc_identity(os.getpid())
        self.peer_heartbeat_digest: str | None = None
        self.peer_heartbeat_last_change: float | None = None
        self.peer_heartbeat_missing_since: float | None = None
        self.peer_heartbeat_sequence: int | None = None
        self.peer_heartbeat_status: str | None = None
        self.supervisor_pid: int | None = None
        self.supervisor_identity: dict[str, Any] | None = None
        self.workgroup: dict[str, Any] | None = None
        self.control_fd = -1
        self.event_fd = -1
        self.event_buffer = b""
        self.pending_events: list[dict[str, Any]] = []
        self.decision_go = False
        self.outcome_succeeded = False

    @property
    def heartbeat_seconds(self) -> float:
        return self.args.heartbeat_ms / 1000.0

    @property
    def stale_seconds(self) -> float:
        return self.args.stale_ms / 1000.0

    def request_abort(self, _signum: int, _frame: Any) -> None:
        self.abort_requested = True

    def _assert_runner_identity(self) -> None:
        runner_now = _proc_identity(int(self.runner["pid"]))
        expected = {key: self.runner[key] for key in IDENTITY_KEYS}
        if runner_now != expected or os.getppid() != self.runner["pid"]:
            raise TransactionError("guarded-runner identity or direct ancestry changed")

    def _assert_local_arm_identity(self) -> None:
        if (
            self.supervisor_pid is None
            or self.supervisor_identity is None
            or self.workgroup is None
            or _proc_identity(self.supervisor_pid) != self.supervisor_identity
            or _proc_identity(int(self.workgroup["pid"])) != self.workgroup
        ):
            raise TransactionError("local supervisor/workgroup identity changed")

    @staticmethod
    def _validate_identity(value: Any, label: str) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or set(value) != IDENTITY_KEYS
            or not all(
                _exact_int(value.get(key), minimum=0)
                for key in IDENTITY_KEYS - {"argv_sha256"}
            )
            or value.get("pid", 0) <= 0
            or value.get("pgid", 0) <= 0
            or value.get("sid", 0) <= 0
            or not HEX64_RE.fullmatch(str(value.get("argv_sha256", "")))
        ):
            raise PeerAbort(f"invalid {label} identity")
        return value

    def _local_prepared(self) -> dict[str, Any]:
        self._assert_runner_identity()
        if self.coordinator_identity["ppid"] != self.runner["pid"]:
            raise TransactionError("coordinator is not the recorded runner child")
        return {
            "schema": SCHEMA,
            "status": "PREPARED",
            "portable": self.portable,
            "portable_sha256": self.portable_sha256,
            "node": _participant(self.args.node_id, self.rank, self.args.node_ip),
            "runner": self.runner,
            "coordinator": self.coordinator_identity,
            "transaction_root": str(self.tx.path),
            "node_local_filesystem": self.tx.local_filesystem_evidence(),
            "prepared_unix_ns": time.time_ns(),
        }

    def _bootstrap(self, deadline: float) -> None:
        bootstrap = {
            "schema": SCHEMA,
            "status": "OPEN",
            "portable": self.portable,
            "portable_sha256": self.portable_sha256,
        }
        if self.rank == 0:
            self.tx.publish_immutable("TRANSACTION.json", bootstrap)
        while True:
            if self.abort_requested:
                raise PeerAbort("local abort signal during bootstrap")
            if self.tx.exists("TRANSACTION.json"):
                break
            if time.monotonic() >= deadline:
                raise TransactionError("timed out waiting for transaction bootstrap")
            time.sleep(0.01)
        observed, _ = self.tx.read_json("TRANSACTION.json")
        if observed != bootstrap:
            raise PeerAbort("portable bootstrap mismatch")

    def _publish_prepared(self) -> None:
        self.prepared_sha256 = self.tx.publish_immutable(
            self.prepared_name, self._local_prepared()
        )
        self.state = "PREPARED"
        self._heartbeat()

    def _heartbeat(self) -> None:
        if not self.prepared_sha256:
            return
        self._assert_runner_identity()
        if self.state == "ARMED":
            self._assert_local_arm_identity()
        self.sequence += 1
        self.tx.replace_heartbeat(
            self.heartbeat_name,
            {
                "schema": SCHEMA,
                "status": self.state,
                "rank": self.rank,
                "prepared_sha256": self.prepared_sha256,
                "coordinator": self.coordinator_identity,
                "sequence": self.sequence,
                "heartbeat_unix_ns": time.time_ns(),
            },
        )

    def _validate_prepared(self, payload: Mapping[str, Any], rank: int) -> None:
        expected_node = next(
            item for item in self.portable["participants"] if item["rank"] == rank
        )
        keys = {
            "schema", "status", "portable", "portable_sha256", "node", "runner",
            "coordinator", "transaction_root", "node_local_filesystem", "prepared_unix_ns",
        }
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") != "PREPARED"
            or payload.get("portable") != self.portable
            or payload.get("portable_sha256") != self.portable_sha256
            or payload.get("node") != expected_node
            or not isinstance(payload.get("transaction_root"), str)
            or not Path(str(payload.get("transaction_root"))).is_absolute()
            or Path(str(payload.get("transaction_root"))).name
            != self.portable["run_id"]
            or not _exact_int(payload.get("prepared_unix_ns"), minimum=1)
        ):
            raise PeerAbort(f"PREPARED receipt mismatch for rank {rank}")
        fs_value = payload.get("node_local_filesystem")
        if (
            not isinstance(fs_value, dict)
            or set(fs_value) != {"st_dev", "st_ino"}
            or not all(_exact_int(value, minimum=0) for value in fs_value.values())
        ):
            raise PeerAbort(f"invalid node-local filesystem evidence for rank {rank}")
        runner = payload.get("runner")
        runner_keys = IDENTITY_KEYS | {"path", "sha256", "status_path", "log_path"}
        if not isinstance(runner, dict) or set(runner) != runner_keys:
            raise PeerAbort(f"invalid runner receipt for rank {rank}")
        self._validate_identity({key: runner[key] for key in IDENTITY_KEYS}, "runner")
        if (
            runner.get("path") != EXPECTED_RUNNER
            or runner.get("sha256") != self.portable["runner_sha256"]
            or not HEX64_RE.fullmatch(str(runner.get("sha256", "")))
            or not all(
                isinstance(runner.get(key), str)
                and Path(runner[key]).is_absolute()
                for key in ("status_path", "log_path")
            )
        ):
            raise PeerAbort(f"runner binding mismatch for rank {rank}")
        coordinator = self._validate_identity(payload.get("coordinator"), "coordinator")
        if coordinator["ppid"] != runner["pid"]:
            raise PeerAbort(f"runner/coordinator ancestry mismatch for rank {rank}")

    def _validate_heartbeat(self, peer: Mapping[str, Any], prepared_sha: str) -> int:
        keys = {
            "schema", "status", "rank", "prepared_sha256", "coordinator", "sequence",
            "heartbeat_unix_ns",
        }
        prepared, _ = self.tx.read_json(self.peer_prepared_name)
        self._validate_prepared(prepared, self.peer_rank)
        if (
            set(peer) != keys
            or peer.get("schema") != SCHEMA
            or peer.get("status") not in {"PREPARED", "ARMED", "RUNNING", "RESULT", "FINAL"}
            or peer.get("rank") != self.peer_rank
            or peer.get("prepared_sha256") != prepared_sha
            or peer.get("coordinator") != prepared["coordinator"]
            or not _exact_int(peer.get("sequence"), minimum=1)
            or not _exact_int(peer.get("heartbeat_unix_ns"), minimum=1)
        ):
            raise PeerAbort("peer heartbeat identity mismatch")
        return int(peer["sequence"])

    def _peer_is_stale(self) -> bool:
        if not self.tx.exists(self.peer_heartbeat_name):
            now = time.monotonic()
            if self.peer_heartbeat_missing_since is None:
                self.peer_heartbeat_missing_since = now
                return False
            return now - self.peer_heartbeat_missing_since > self.stale_seconds
        self.peer_heartbeat_missing_since = None
        peer, digest = self.tx.read_json(self.peer_heartbeat_name)
        _, prepared_sha = self.tx.read_json(self.peer_prepared_name)
        sequence = self._validate_heartbeat(peer, prepared_sha)
        self.peer_heartbeat_status = str(peer["status"])
        now = time.monotonic()
        if digest != self.peer_heartbeat_digest:
            if (
                self.peer_heartbeat_sequence is not None
                and sequence <= self.peer_heartbeat_sequence
            ):
                raise PeerAbort("peer heartbeat sequence did not increase")
            self.peer_heartbeat_digest = digest
            self.peer_heartbeat_sequence = sequence
            self.peer_heartbeat_last_change = now
            return False
        if self.peer_heartbeat_sequence != sequence:
            raise PeerAbort("peer heartbeat digest/sequence inconsistency")
        if self.peer_heartbeat_last_change is None:
            self.peer_heartbeat_last_change = now
            return False
        return now - self.peer_heartbeat_last_change > self.stale_seconds

    def _wait_for_peer_heartbeat_status(
        self,
        accepted_statuses: set[str],
        timeout_ms: int,
        phase: str,
    ) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        next_heartbeat = 0.0
        while True:
            self._check_abort()
            if self.tx.exists(self.peer_heartbeat_name):
                if self._peer_is_stale():
                    raise PeerAbort(f"peer heartbeat stale during {phase}")
                if self.peer_heartbeat_status in accepted_statuses:
                    return
            now = time.monotonic()
            if now >= deadline:
                raise PeerAbort(f"timeout waiting for peer heartbeat state during {phase}")
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self.heartbeat_seconds
            time.sleep(min(0.02, self.heartbeat_seconds))

    def _validate_decision(self, payload: Mapping[str, Any]) -> None:
        keys = {
            "schema", "status", "rank", "portable_sha256", "reason", "bindings",
            "decision_unix_ns",
        }
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") not in {"GO", "ABORT"}
            or payload.get("rank") not in EXPECTED_RANKS
            or payload.get("portable_sha256") != self.portable_sha256
            or not isinstance(payload.get("reason"), str)
            or not _exact_int(payload.get("decision_unix_ns"), minimum=1)
        ):
            raise TransactionError("invalid DECISION receipt")
        bindings = payload.get("bindings")
        if payload["status"] == "ABORT":
            if bindings is not None:
                raise TransactionError("ABORT decision must not carry GO bindings")
            return
        if (
            payload.get("rank") != 0
            or not isinstance(bindings, dict)
            or set(bindings) != {"prepared_sha256", "armed_sha256"}
            or any(
                not isinstance(bindings[key], dict)
                or set(bindings[key]) != {"0", "1"}
                or not all(HEX64_RE.fullmatch(str(value)) for value in bindings[key].values())
                for key in ("prepared_sha256", "armed_sha256")
            )
        ):
            raise TransactionError("invalid GO bindings")

    def _validate_outcome(self, payload: Mapping[str, Any]) -> None:
        if (
            set(payload)
            != {
                "schema", "status", "rank", "portable_sha256", "reason", "bindings",
                "outcome_unix_ns",
            }
            or payload.get("schema") != SCHEMA
            or payload.get("status") not in {"SUCCEEDED", "FAILED"}
            or payload.get("rank") not in EXPECTED_RANKS
            or payload.get("portable_sha256") != self.portable_sha256
            or not isinstance(payload.get("reason"), str)
            or not _exact_int(payload.get("outcome_unix_ns"), minimum=1)
        ):
            raise TransactionError("invalid OUTCOME receipt")
        bindings = payload.get("bindings")
        if payload["status"] == "FAILED":
            if bindings is not None:
                raise TransactionError("FAILED outcome must not carry success bindings")
            return
        if (
            payload.get("rank") != 0
            or not isinstance(bindings, dict)
            or set(bindings) != {"workload_result_sha256"}
            or not isinstance(bindings["workload_result_sha256"], dict)
            or set(bindings["workload_result_sha256"]) != {"0", "1"}
            or not all(
                HEX64_RE.fullmatch(str(value))
                for value in bindings["workload_result_sha256"].values()
            )
        ):
            raise TransactionError("invalid successful OUTCOME bindings")

    def _check_abort(self) -> None:
        if self.abort_requested:
            raise PeerAbort("local abort signal")
        if self.tx.exists("OUTCOME.json"):
            outcome, _ = self.tx.read_json("OUTCOME.json")
            self._validate_outcome(outcome)
            if outcome["status"] == "FAILED":
                raise PeerAbort(f"transaction failure: {outcome['reason']}")
        if self.tx.exists("DECISION.json"):
            decision, _ = self.tx.read_json("DECISION.json")
            self._validate_decision(decision)
            if decision["status"] == "ABORT":
                raise PeerAbort(f"transaction aborted: {decision['reason']}")

    def _publish_decision_abort(self, reason: str) -> None:
        payload = {
            "schema": SCHEMA,
            "status": "ABORT",
            "rank": self.rank,
            "portable_sha256": self.portable_sha256,
            "reason": reason,
            "bindings": None,
            "decision_unix_ns": time.time_ns(),
        }
        try:
            self.tx.publish_immutable("DECISION.json", payload)
        except DuplicateInvocation:
            existing, _ = self.tx.read_json("DECISION.json")
            self._validate_decision(existing)
            if existing["status"] == "GO":
                self._publish_failure(reason)

    def _publish_failure(self, reason: str) -> None:
        payload = {
            "schema": SCHEMA,
            "status": "FAILED",
            "rank": self.rank,
            "portable_sha256": self.portable_sha256,
            "reason": reason,
            "bindings": None,
            "outcome_unix_ns": time.time_ns(),
        }
        try:
            self.tx.publish_immutable("OUTCOME.json", payload)
        except DuplicateInvocation:
            existing, _ = self.tx.read_json("OUTCOME.json")
            self._validate_outcome(existing)

    def _publish_success_outcome(self, result_hashes: Mapping[int, str]) -> None:
        bindings = {
            "workload_result_sha256": {
                str(rank): result_hashes[rank] for rank in EXPECTED_RANKS
            }
        }
        if self.rank == 0:
            self._check_abort()
            self.tx.publish_immutable(
                "OUTCOME.json",
                {
                    "schema": SCHEMA,
                    "status": "SUCCEEDED",
                    "rank": 0,
                    "portable_sha256": self.portable_sha256,
                    "reason": "both exact workload results completed",
                    "bindings": bindings,
                    "outcome_unix_ns": time.time_ns(),
                },
            )
        self._wait_for_file("OUTCOME.json", self.args.completion_timeout_ms, "outcome")
        outcome, _ = self.tx.read_json("OUTCOME.json")
        self._validate_outcome(outcome)
        if outcome["status"] != "SUCCEEDED" or outcome["bindings"] != bindings:
            raise PeerAbort("successful OUTCOME binding mismatch")
        self.outcome_succeeded = True

    def _wait_for_file(self, name: str, timeout_ms: int, phase: str) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        next_heartbeat = 0.0
        while True:
            self._check_abort()
            if self.tx.exists(name):
                self._check_abort()
                return
            if self.tx.exists(self.peer_prepared_name) and self._peer_is_stale():
                raise PeerAbort(f"peer heartbeat stale during {phase}")
            now = time.monotonic()
            if now >= deadline:
                raise PeerAbort(f"timeout during {phase}")
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self.heartbeat_seconds
            time.sleep(min(0.02, self.heartbeat_seconds))

    def _prepare_pair(self) -> dict[int, str]:
        self._wait_for_file(self.peer_prepared_name, self.args.prepare_timeout_ms, "prepare")
        hashes: dict[int, str] = {}
        for rank in EXPECTED_RANKS:
            payload, digest = self.tx.read_json(f"PREPARED.rank{rank}.json")
            self._validate_prepared(payload, rank)
            hashes[rank] = digest
        self._wait_for_file(
            self.peer_heartbeat_name,
            self.args.prepare_timeout_ms,
            "fresh heartbeat",
        )
        if self._peer_is_stale():
            raise PeerAbort("peer heartbeat was stale before arming")
        return hashes

    def _read_supervisor_events(self) -> None:
        if self.event_fd < 0:
            return
        while True:
            try:
                block = os.read(self.event_fd, 65536)
            except BlockingIOError:
                break
            if not block:
                break
            self.event_buffer += block
        while b"\n" in self.event_buffer:
            raw, self.event_buffer = self.event_buffer.split(b"\n", 1)
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TransactionError("invalid supervisor event") from exc
            if not isinstance(event, dict):
                raise TransactionError("invalid supervisor event type")
            self.pending_events.append(event)

    def _pop_event(self, event_name: str) -> dict[str, Any] | None:
        self._read_supervisor_events()
        for index, event in enumerate(self.pending_events):
            if event.get("event") in {"SUPERVISOR_ERROR", "EXEC_ERROR"}:
                self.pending_events.pop(index)
                raise TransactionError(str(event.get("reason", "supervisor failure")))
            if event.get("event") == event_name:
                return self.pending_events.pop(index)
        return None

    def _wait_event(self, event_name: str, timeout_ms: int, phase: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_ms / 1000.0
        next_heartbeat = 0.0
        while True:
            event = self._pop_event(event_name)
            if event is not None:
                return event
            self._check_abort()
            if self.tx.exists(self.peer_prepared_name) and self._peer_is_stale():
                raise PeerAbort(f"peer heartbeat stale during {phase}")
            if self.supervisor_pid is not None:
                waited, _ = os.waitpid(self.supervisor_pid, os.WNOHANG)
                if waited == self.supervisor_pid:
                    self.supervisor_pid = None
                    # The process can exit after its final pipe write but
                    # before this nonblocking wait.  Drain the pipe once more
                    # before classifying that ordered terminal event as
                    # missing.
                    event = self._pop_event(event_name)
                    if event is not None:
                        return event
                    raise TransactionError(f"supervisor exited during {phase}")
            now = time.monotonic()
            if now >= deadline:
                raise TransactionError(f"timeout during {phase}")
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self.heartbeat_seconds
            time.sleep(min(0.02, self.heartbeat_seconds))

    def _spawn_supervisor(self) -> None:
        control_read, control_write = os.pipe()
        event_read, event_write = os.pipe()
        parent_pid = os.getpid()
        pid = os.fork()
        if pid == 0:
            os.close(control_write)
            os.close(event_read)
            environment = dict(self.workload_environment)
            environment.update(
                {
                    "SEMTALK_W16_TRANSACTION_ROOT": str(self.tx.path),
                    "SEMTALK_W16_RUN_ID": self.args.run_id,
                    "SEMTALK_W16_NODE_ID": self.args.node_id,
                    "SEMTALK_W16_NODE_RANK": str(self.rank),
                    "SEMTALK_W16_MAX_RESTARTS": "0",
                }
            )
            _supervisor_main(
                parent_pid,
                control_read,
                event_write,
                self.executable_fd,
                self.workdir_fd,
                self.workload,
                environment,
                self.portable["workload"]["argv_sha256"],
                self.portable["workload"]["executable_sha256"],
                self.args.shutdown_grace_ms / 1000.0,
            )
        os.close(control_read)
        os.close(event_write)
        os.set_blocking(event_read, False)
        self.control_fd = control_write
        self.event_fd = event_read
        self.supervisor_pid = pid

    def _validate_armed(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema", "status", "rank", "prepared_sha256", "coordinator",
            "supervisor", "workgroup", "armed_unix_ns",
        }
        prepared, prepared_sha = self.tx.read_json(f"PREPARED.rank{rank}.json")
        self._validate_prepared(prepared, rank)
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") != "ARMED"
            or payload.get("rank") != rank
            or payload.get("prepared_sha256") != prepared_sha
            or payload.get("coordinator") != prepared["coordinator"]
            or not _exact_int(payload.get("armed_unix_ns"), minimum=1)
        ):
            raise PeerAbort(f"ARMED receipt mismatch for rank {rank}")
        supervisor = self._validate_identity(payload.get("supervisor"), "supervisor")
        workgroup = self._validate_identity(payload.get("workgroup"), "workgroup anchor")
        if (
            supervisor["ppid"] != prepared["coordinator"]["pid"]
            or workgroup["ppid"] != supervisor["pid"]
            or workgroup["pid"] != workgroup["pgid"]
            or workgroup["sid"] != supervisor["sid"]
            or supervisor["argv_sha256"] != prepared["coordinator"]["argv_sha256"]
            or workgroup["argv_sha256"] != supervisor["argv_sha256"]
        ):
            raise PeerAbort(f"ARMED ancestry mismatch for rank {rank}")

    def _arm_and_decide(self, prepared_hashes: Mapping[int, str]) -> dict[int, str]:
        self._spawn_supervisor()
        local = self._wait_event("LOCAL_ARMED", self.args.arm_timeout_ms, "local arm")
        pgid = local.get("workgroup_pgid")
        if not _exact_int(pgid, minimum=1) or self.supervisor_pid is None:
            raise TransactionError("invalid local workgroup evidence")
        self.supervisor_identity = _proc_identity(self.supervisor_pid)
        anchor_identity = _proc_identity(int(pgid))
        self.workgroup = anchor_identity
        if (
            self.supervisor_identity["ppid"] != os.getpid()
            or anchor_identity["ppid"] != self.supervisor_pid
            or anchor_identity["pid"] != anchor_identity["pgid"]
            or anchor_identity["sid"] != self.supervisor_identity["sid"]
            or self.supervisor_identity["argv_sha256"]
            != self.coordinator_identity["argv_sha256"]
            or anchor_identity["argv_sha256"]
            != self.supervisor_identity["argv_sha256"]
        ):
            raise TransactionError("local supervisor/workgroup ancestry mismatch")
        self.armed_sha256 = self.tx.publish_immutable(
            self.armed_name,
            {
                "schema": SCHEMA,
                "status": "ARMED",
                "rank": self.rank,
                "prepared_sha256": self.prepared_sha256,
                "coordinator": self.coordinator_identity,
                "supervisor": self.supervisor_identity,
                "workgroup": anchor_identity,
                "armed_unix_ns": time.time_ns(),
            },
        )
        self.state = "ARMED"
        self._heartbeat()
        self._wait_for_file(self.peer_armed_name, self.args.arm_timeout_ms, "peer arm")
        armed_hashes: dict[int, str] = {}
        for rank in EXPECTED_RANKS:
            receipt, digest = self.tx.read_json(f"ARMED.rank{rank}.json")
            self._validate_armed(receipt, rank)
            armed_hashes[rank] = digest
        self._wait_for_peer_heartbeat_status(
            {"ARMED"}, self.args.arm_timeout_ms, "post-ARMED barrier"
        )
        if _source_evidence(self.args) != self.portable["source"]:
            raise TransactionError("source evidence changed before GO")
        if _sha256_fd(self.executable_fd) != self.portable["workload"]["executable_sha256"]:
            raise TransactionError("pinned workload executable changed before GO")
        for logical_id, fd in self.workload_input_fds.items():
            if _sha256_fd(fd) != self.portable["workload"]["input_sha256"][logical_id]:
                raise TransactionError(f"pinned workload input changed: {logical_id}")
        bindings = {
            "prepared_sha256": {str(rank): prepared_hashes[rank] for rank in EXPECTED_RANKS},
            "armed_sha256": {str(rank): armed_hashes[rank] for rank in EXPECTED_RANKS},
        }
        if self.rank == 0:
            self._check_abort()
            self._assert_local_arm_identity()
            self.tx.publish_immutable(
                "DECISION.json",
                {
                    "schema": SCHEMA,
                    "status": "GO",
                    "rank": 0,
                    "portable_sha256": self.portable_sha256,
                    "reason": "both exact guarded nodes armed",
                    "bindings": bindings,
                    "decision_unix_ns": time.time_ns(),
                },
            )
        self._wait_for_file("DECISION.json", self.args.decision_timeout_ms, "GO decision")
        decision, _ = self.tx.read_json("DECISION.json")
        self._validate_decision(decision)
        if decision["status"] != "GO" or decision["bindings"] != bindings:
            raise PeerAbort("GO decision binding mismatch")
        self.decision_go = True
        return armed_hashes

    def _publish_started(self, event: Mapping[str, Any]) -> None:
        workload_identity = event.get("workload")
        expected_keys = {
            "pid", "ppid", "pgid", "sid", "starttime_ticks",
            "preexec_argv_sha256", "expected_argv_sha256", "executable_sha256",
        }
        if (
            not isinstance(workload_identity, dict)
            or set(workload_identity) != expected_keys
            or not all(
                _exact_int(workload_identity.get(key), minimum=1)
                for key in ("pid", "ppid", "pgid", "sid", "starttime_ticks")
            )
            or workload_identity.get("ppid") != self.supervisor_identity["pid"]
            or workload_identity.get("pgid") != self.workgroup["pid"]
            or workload_identity.get("sid") != self.supervisor_identity["sid"]
            or workload_identity.get("preexec_argv_sha256")
            != self.supervisor_identity["argv_sha256"]
            or workload_identity.get("expected_argv_sha256")
            != self.portable["workload"]["argv_sha256"]
            or workload_identity.get("executable_sha256")
            != self.portable["workload"]["executable_sha256"]
        ):
            raise TransactionError("invalid local STARTED process evidence")
        self.started_sha256 = self.tx.publish_immutable(
            self.started_name,
            {
                "schema": SCHEMA,
                "status": "STARTED",
                "rank": self.rank,
                "armed_sha256": self.armed_sha256,
                "coordinator": self.coordinator_identity,
                "supervisor": self.supervisor_identity,
                "workgroup": self.workgroup,
                "workload": workload_identity,
                "started_unix_ns": time.time_ns(),
            },
        )

    def _validate_started(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema", "status", "rank", "armed_sha256", "coordinator", "supervisor",
            "workgroup", "workload", "started_unix_ns",
        }
        armed, armed_sha = self.tx.read_json(f"ARMED.rank{rank}.json")
        self._validate_armed(armed, rank)
        workload = payload.get("workload")
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") != "STARTED"
            or payload.get("rank") != rank
            or payload.get("armed_sha256") != armed_sha
            or payload.get("coordinator") != armed["coordinator"]
            or payload.get("supervisor") != armed["supervisor"]
            or payload.get("workgroup") != armed["workgroup"]
            or not _exact_int(payload.get("started_unix_ns"), minimum=1)
            or not isinstance(workload, dict)
            or set(workload) != {
                "pid", "ppid", "pgid", "sid", "starttime_ticks",
                "preexec_argv_sha256", "expected_argv_sha256",
                "executable_sha256",
            }
            or not all(
                _exact_int(workload.get(key), minimum=1)
                for key in ("pid", "ppid", "pgid", "sid", "starttime_ticks")
            )
            or workload.get("ppid") != armed["supervisor"]["pid"]
            or workload.get("pgid") != armed["workgroup"]["pid"]
            or workload.get("sid") != armed["supervisor"]["sid"]
            or workload.get("preexec_argv_sha256")
            != armed["supervisor"]["argv_sha256"]
            or workload.get("expected_argv_sha256") != self.portable["workload"]["argv_sha256"]
            or workload.get("executable_sha256") != self.portable["workload"]["executable_sha256"]
        ):
            raise PeerAbort(f"STARTED receipt mismatch for rank {rank}")

    def _publish_workload_result(self, status_value: str, rc: int | None, reason: str) -> None:
        if status_value == "COMPLETED":
            valid_rc = rc == 0 and type(rc) is int
        elif status_value == "WORKLOAD_FAILED":
            valid_rc = type(rc) is int and rc != 0
        else:
            valid_rc = status_value == "ABORTED" and rc is None
        if not valid_rc:
            raise TransactionError("invalid local workload-result semantics")
        self.workload_result_sha256 = self.tx.publish_immutable(
            self.result_name,
            {
                "schema": SCHEMA,
                "status": status_value,
                "rank": self.rank,
                "started_sha256": self.started_sha256 or None,
                "coordinator": self.coordinator_identity,
                "reason": reason,
                "workload_returncode": rc,
                "result_unix_ns": time.time_ns(),
            },
        )

    def _validate_workload_result(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema", "status", "rank", "started_sha256", "coordinator", "reason",
            "workload_returncode", "result_unix_ns",
        }
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") not in {"COMPLETED", "WORKLOAD_FAILED", "ABORTED"}
            or payload.get("rank") != rank
            or not isinstance(payload.get("reason"), str)
            or not _exact_int(payload.get("result_unix_ns"), minimum=1)
        ):
            raise PeerAbort(f"WORKLOAD_RESULT schema mismatch for rank {rank}")
        coordinator = self._validate_identity(
            payload.get("coordinator"), "workload-result coordinator"
        )
        prepared_name = f"PREPARED.rank{rank}.json"
        if self.tx.exists(prepared_name):
            prepared, _ = self.tx.read_json(prepared_name)
            self._validate_prepared(prepared, rank)
            if coordinator != prepared["coordinator"]:
                raise PeerAbort(f"WORKLOAD_RESULT coordinator mismatch for rank {rank}")
        if payload["status"] == "ABORTED":
            if payload.get("workload_returncode") is not None:
                raise PeerAbort("ABORTED workload result has a return code")
            if payload.get("started_sha256") is not None:
                started, started_sha = self.tx.read_json(f"STARTED.rank{rank}.json")
                self._validate_started(started, rank)
                if (
                    payload.get("started_sha256") != started_sha
                    or payload.get("coordinator") != started["coordinator"]
                ):
                    raise PeerAbort("ABORTED workload result start binding mismatch")
            return
        started, started_sha = self.tx.read_json(f"STARTED.rank{rank}.json")
        self._validate_started(started, rank)
        rc = payload.get("workload_returncode")
        if (
            payload.get("started_sha256") != started_sha
            or payload.get("coordinator") != started["coordinator"]
            or (payload["status"] == "COMPLETED" and not (type(rc) is int and rc == 0))
            or (payload["status"] == "WORKLOAD_FAILED" and not (type(rc) is int and rc != 0))
        ):
            raise PeerAbort(f"WORKLOAD_RESULT semantics mismatch for rank {rank}")

    def _publish_final(self, status_value: str, coordinator_rc: int, reason: str) -> None:
        self.state = "FINAL"
        self._heartbeat()
        payload = {
            "schema": SCHEMA,
            "status": status_value,
            "rank": self.rank,
            "workload_result_sha256": self.workload_result_sha256 or None,
            "coordinator": self.coordinator_identity,
            "reason": reason,
            "coordinator_returncode": coordinator_rc,
            "final_unix_ns": time.time_ns(),
        }
        self._validate_final(payload, self.rank)
        self.final_sha256 = self.tx.publish_immutable(self.final_name, payload)

    def _validate_final(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema", "status", "rank", "workload_result_sha256", "coordinator",
            "reason", "coordinator_returncode", "final_unix_ns",
        }
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status")
            not in {"SUCCEEDED", "WORKLOAD_FAILED", "ABORTED", "PROTOCOL_FAILED"}
            or payload.get("rank") != rank
            or not isinstance(payload.get("reason"), str)
            or not _exact_int(payload.get("coordinator_returncode"), minimum=0)
            or not _exact_int(payload.get("final_unix_ns"), minimum=1)
            or not isinstance(payload.get("coordinator"), dict)
        ):
            raise PeerAbort(f"FINAL schema mismatch for rank {rank}")
        coordinator = self._validate_identity(payload.get("coordinator"), "final coordinator")
        prepared_name = f"PREPARED.rank{rank}.json"
        if self.tx.exists(prepared_name):
            prepared, _ = self.tx.read_json(prepared_name)
            self._validate_prepared(prepared, rank)
            if coordinator != prepared["coordinator"]:
                raise PeerAbort(f"FINAL coordinator mismatch for rank {rank}")
        result_hash = payload.get("workload_result_sha256")
        result: Mapping[str, Any] | None = None
        if result_hash is not None:
            if not HEX64_RE.fullmatch(str(result_hash)):
                raise PeerAbort(f"FINAL result digest mismatch for rank {rank}")
            result, observed_hash = self.tx.read_json(f"WORKLOAD_RESULT.rank{rank}.json")
            self._validate_workload_result(result, rank)
            if result_hash != observed_hash or payload.get("coordinator") != result["coordinator"]:
                raise PeerAbort(f"FINAL/result binding mismatch for rank {rank}")
        status_value = payload["status"]
        rc = payload["coordinator_returncode"]
        if (
            (
                status_value == "SUCCEEDED"
                and not (rc == 0 and result and result["status"] == "COMPLETED")
            )
            or (
                status_value == "WORKLOAD_FAILED"
                and not (
                    rc == 4 and result and result["status"] == "WORKLOAD_FAILED"
                )
            )
            or (status_value in {"ABORTED", "PROTOCOL_FAILED"} and rc != 3)
        ):
            raise PeerAbort(f"FINAL status/rc semantics mismatch for rank {rank}")

    def _terminate_workload(self) -> None:
        if self.control_fd >= 0:
            try:
                os.write(self.control_fd, b"A")
            except OSError:
                pass
            os.close(self.control_fd)
            self.control_fd = -1
        if self.workgroup is not None:
            pgid = int(self.workgroup["pid"])
            try:
                observed_anchor = _proc_identity(pgid)
                anchor_is_exact = all(
                    observed_anchor[key] == self.workgroup[key]
                    for key in IDENTITY_KEYS - {"ppid"}
                )
            except TransactionError:
                anchor_is_exact = False
            if anchor_is_exact:
                _kill_exact_group(pgid, signal.SIGTERM)
                deadline = time.monotonic() + self.args.shutdown_grace_ms / 1000.0
                while time.monotonic() < deadline:
                    if _group_members(pgid) <= {pgid}:
                        break
                    time.sleep(0.01)
                # The ignored-TERM anchor still pins the PGID here.
                _kill_exact_group(pgid, signal.SIGKILL)
        if self.supervisor_pid is not None:
            deadline = time.monotonic() + self.args.shutdown_grace_ms / 1000.0
            while time.monotonic() < deadline:
                try:
                    waited, _ = os.waitpid(self.supervisor_pid, os.WNOHANG)
                except ChildProcessError:
                    waited = self.supervisor_pid
                if waited == self.supervisor_pid:
                    self.supervisor_pid = None
                    break
                time.sleep(0.01)
            if self.supervisor_pid is not None:
                try:
                    os.kill(self.supervisor_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(self.supervisor_pid, 0)
                except ChildProcessError:
                    pass
                self.supervisor_pid = None

    def _run_workload(self) -> int:
        self._check_abort()
        self._assert_local_arm_identity()
        if self._peer_is_stale():
            raise PeerAbort("peer heartbeat stale immediately before GO")
        if self.control_fd < 0:
            raise TransactionError("local supervisor control is unavailable")
        os.write(self.control_fd, b"G")
        os.close(self.control_fd)
        self.control_fd = -1
        started = self._wait_event("STARTED", self.args.start_timeout_ms, "workload start")
        self._publish_started(started)
        self.state = "RUNNING"
        self._heartbeat()
        deadline = time.monotonic() + self.args.completion_timeout_ms / 1000.0
        next_heartbeat = 0.0
        while True:
            result = self._pop_event("RESULT")
            if result is not None:
                rc = result.get("returncode")
                if type(rc) is not int:
                    raise TransactionError("supervisor returned no exact workload rc")
                if self.supervisor_pid is not None:
                    os.waitpid(self.supervisor_pid, 0)
                    self.supervisor_pid = None
                # The supervisor has proved the anchored group empty.  Drop
                # the live PGID handle so a later peer failure can never act
                # on a numerically recycled process group.
                self.workgroup = None
                if rc == 0:
                    self._publish_workload_result(
                        "COMPLETED",
                        0,
                        "workload and exact process group completed",
                    )
                    self.state = "RESULT"
                    self._heartbeat()
                    break
                self._publish_workload_result("WORKLOAD_FAILED", rc, "workload returned non-zero")
                self.state = "RESULT"
                self._heartbeat()
                self._publish_failure("local workload returned non-zero")
                self._publish_final("WORKLOAD_FAILED", 4, "local workload failed")
                return 4
            if self.supervisor_pid is None or self.supervisor_identity is None:
                raise TransactionError("local supervisor identity is unavailable")
            try:
                supervisor_is_exact = (
                    _proc_identity(self.supervisor_pid) == self.supervisor_identity
                )
            except TransactionError:
                self._read_supervisor_events()
                if any(
                    event.get("event")
                    in {"RESULT", "SUPERVISOR_ERROR", "EXEC_ERROR"}
                    for event in self.pending_events
                ):
                    continue
                raise TransactionError("local supervisor exited without a result")
            if not supervisor_is_exact:
                raise TransactionError("local supervisor identity changed while running")
            self._check_abort()
            if self._peer_is_stale():
                raise PeerAbort("peer heartbeat became stale while running")
            now = time.monotonic()
            if now >= deadline:
                raise PeerAbort("local workload completion timeout")
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self.heartbeat_seconds
            time.sleep(min(0.02, self.heartbeat_seconds))

        self._wait_for_file(self.peer_result_name, self.args.completion_timeout_ms, "peer result")
        peer, peer_result_sha = self.tx.read_json(self.peer_result_name)
        self._validate_workload_result(peer, self.peer_rank)
        if peer["status"] != "COMPLETED":
            raise PeerAbort(f"peer workload result: {peer['status']}")
        self._wait_for_peer_heartbeat_status(
            {"RESULT", "FINAL"},
            self.args.completion_timeout_ms,
            "post-result barrier",
        )
        result_hashes = {
            self.rank: self.workload_result_sha256,
            self.peer_rank: peer_result_sha,
        }
        self._publish_success_outcome(result_hashes)
        self._publish_final("SUCCEEDED", 0, "both exact guarded workloads completed")
        return 0

    def run(self) -> int:
        self._bootstrap(time.monotonic() + self.args.prepare_timeout_ms / 1000.0)
        self._publish_prepared()
        prepared_hashes = self._prepare_pair()
        self._arm_and_decide(prepared_hashes)
        return self._run_workload()

    def fail(self, exc: BaseException) -> None:
        self._terminate_workload()
        if isinstance(exc, DuplicateInvocation):
            return
        reason = f"{type(exc).__name__}: {exc}"
        try:
            if self.decision_go:
                self._publish_failure(reason)
            else:
                self._publish_decision_abort(reason)
            if not self.workload_result_sha256:
                self._publish_workload_result("ABORTED", None, reason)
            if not self.final_sha256:
                status_value = "ABORTED" if isinstance(exc, PeerAbort) else "PROTOCOL_FAILED"
                self._publish_final(status_value, 3, reason)
        except BaseException:
            pass

    def close(self) -> None:
        if self.executable_fd >= 0:
            os.close(self.executable_fd)
            self.executable_fd = -1
        if self.event_fd >= 0:
            os.close(self.event_fd)
            self.event_fd = -1
        if self.workdir_fd >= 0:
            os.close(self.workdir_fd)
            self.workdir_fd = -1
        for fd in self.workload_input_fds.values():
            os.close(fd)
        self.workload_input_fds = {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--common-command-sha256", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--node-rank", required=True, type=int)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--peer-node-id", required=True)
    parser.add_argument("--peer-node-rank", required=True, type=int)
    parser.add_argument("--peer-node-ip", required=True)
    parser.add_argument("--runner-status-path", required=True)
    parser.add_argument("--runner-log-path", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--allow-env", action="append", default=[])
    parser.add_argument("--workload-input", action="append", default=[])
    parser.add_argument("--max-restarts", required=True, type=int)
    parser.add_argument("--heartbeat-ms", type=int, default=1000)
    parser.add_argument("--stale-ms", type=int, default=15000)
    parser.add_argument("--prepare-timeout-ms", type=int, default=120000)
    parser.add_argument("--decision-timeout-ms", type=int, default=120000)
    parser.add_argument("--arm-timeout-ms", type=int, default=120000)
    parser.add_argument("--start-timeout-ms", type=int, default=120000)
    parser.add_argument("--completion-timeout-ms", type=int, default=120000)
    parser.add_argument("--shutdown-grace-ms", type=int, default=10000)
    parser.add_argument("workload", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workload = list(args.workload)
    if workload and workload[0] == "--":
        workload = workload[1:]
    if not workload or not os.path.isabs(workload[0]):
        print("workload executable must be an absolute path", file=sys.stderr)
        return 2
    coordinator: Coordinator | None = None
    tx: TransactionDirectory | None = None
    executable_fd = -1
    workdir_fd = -1
    workload_input_fds: dict[str, int] = {}
    try:
        transaction_root, parent, basename = _validate_tx_path(
            args.transaction_root, args.run_id
        )
        args.runner_status_path = _validate_absolute_evidence_path(
            args.runner_status_path, "runner status path"
        )
        args.runner_log_path = _validate_absolute_evidence_path(
            args.runner_log_path, "runner log path"
        )
        runner, runner_sha256 = _runner_evidence(
            args.runner_status_path, args.runner_log_path
        )
        source = _source_evidence(args)
        (
            workload_spec,
            workload_environment,
            executable_fd,
            workdir_fd,
            workload_input_fds,
        ) = _workload_evidence(args, workload)
        portable = _portable_payload(
            args, runner_sha256, source, workload_spec
        )
        tx = TransactionDirectory(
            transaction_root,
            parent,
            basename,
            args.node_rank,
        )
        tx.create_or_wait(
            time.monotonic() + args.prepare_timeout_ms / 1000.0
        )
        coordinator = Coordinator(
            args,
            workload,
            runner,
            portable,
            tx,
            workload_environment,
            executable_fd,
            workdir_fd,
            workload_input_fds,
        )
        executable_fd = -1
        workdir_fd = -1
        workload_input_fds = {}
        signal.signal(signal.SIGTERM, coordinator.request_abort)
        signal.signal(signal.SIGINT, coordinator.request_abort)
        return coordinator.run()
    except BaseException as exc:
        if coordinator is not None:
            coordinator.fail(exc)
        print(f"dual-node guarded transaction failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if coordinator is not None:
            coordinator.close()
        if executable_fd >= 0:
            os.close(executable_fd)
        if workdir_fd >= 0:
            os.close(workdir_fd)
        for fd in workload_input_fds.values():
            os.close(fd)
        if tx is not None:
            tx.close()


if __name__ == "__main__":
    raise SystemExit(main())
