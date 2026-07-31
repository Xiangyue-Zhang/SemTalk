#!/usr/bin/env python3
"""Fail-closed two-node transaction for guarded GPU workloads.

The transaction root must not exist.  Rank zero creates it and publishes an
immutable bootstrap receipt; rank one only joins that exact transaction.  A
node publishes its immutable PREPARED receipt with ``O_EXCL`` and ``fsync``.
Rank zero publishes COMMIT only after both receipts have the same portable
payload.  Both nodes supervise their workload and maintain heartbeats.  A
peer failure, abort, stale heartbeat, identity mismatch, or timeout terminates
the local workload before returning non-zero to the outer guarded runner.

Filesystem device/inode identities are deliberately node-local evidence.
They never participate in portable-payload equality, which makes the protocol
valid when two hosts see a shared transaction through different mounts.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


SCHEMA = "semtalk.dual_node_guarded_transaction.v1"
PORTABLE_SCHEMA = f"{SCHEMA}.portable"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_RECEIPT_BYTES = 1 << 20
EXPECTED_RANKS = (0, 1)


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
                "/bin/ps",
                "-p",
                str(pid),
                "-o",
                "lstart=",
                "-o",
                "command=",
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
        return {
            "pid": pid,
            "starttime_ticks": int(
                hashlib.sha256(first_line[:24]).hexdigest()[:15], 16
            ),
            "argv_sha256": _sha256_bytes(snapshot_before),
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
        "starttime_ticks": starttime_before,
        "argv_sha256": _sha256_bytes(argv_bytes),
    }


def _runner_evidence() -> tuple[dict[str, Any], str]:
    pid = os.getppid()
    identity = _proc_identity(pid)
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    cmdline = proc_cmdline.read_bytes().split(b"\0") if proc_cmdline.exists() else []
    tokens = [token.decode("utf-8", errors="surrogateescape") for token in cmdline if token]
    guarded = [token for token in tokens if token == "/tmp/globaldiff_guarded_runner.py"]
    if len(guarded) == 1:
        runner_path = Path(guarded[0])
    else:
        try:
            if Path(f"/proc/{pid}/exe").exists():
                runner_path = Path(f"/proc/{pid}/exe").resolve(strict=True)
            else:
                runner_path = Path(sys.executable).resolve(strict=True)
        except OSError as exc:
            raise TransactionError("cannot resolve runner executable") from exc
    if runner_path.is_symlink() or not runner_path.is_file():
        raise TransactionError("runner evidence is not a regular non-symlink file")
    runner_sha256 = _sha256_file(runner_path)
    identity.update(
        {
            "path": str(runner_path),
            "sha256": runner_sha256,
        }
    )
    return identity, runner_sha256


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


def _validate_tx_path(raw: str) -> tuple[Path, Path, str]:
    path = Path(raw)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise TransactionError("transaction root must be a safe absolute path")
    if not SAFE_ID_RE.fullmatch(path.name):
        raise TransactionError("unsafe transaction-root basename")
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


def _portable_payload(args: argparse.Namespace, runner_sha256: str) -> dict[str, Any]:
    if not SAFE_ID_RE.fullmatch(args.run_id):
        raise TransactionError("unsafe run ID")
    if not HEX40_RE.fullmatch(args.source_commit) or not HEX40_RE.fullmatch(
        args.source_tree
    ):
        raise TransactionError("source commit/tree must be lowercase 40-hex IDs")
    if not HEX64_RE.fullmatch(args.common_command_sha256):
        raise TransactionError("common-command SHA-256 must be lowercase 64-hex")
    if args.max_restarts != 0:
        raise TransactionError("max_restarts must be exactly zero")
    integer_timeouts = {
        "heartbeat_ms": args.heartbeat_ms,
        "stale_ms": args.stale_ms,
        "prepare_timeout_ms": args.prepare_timeout_ms,
        "commit_timeout_ms": args.commit_timeout_ms,
        "completion_timeout_ms": args.completion_timeout_ms,
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
        "runner_sha256": runner_sha256,
        "common_command_sha256": args.common_command_sha256,
        "participants": participants,
        "max_restarts": 0,
        "timeouts": integer_timeouts,
    }


class TransactionDirectory:
    """A held dirfd plus public-path identity checks for TOCTOU defense."""

    def __init__(self, path: Path, parent: Path, basename: str, rank: int):
        self.path = path
        self.parent = parent
        self.basename = basename
        self.rank = rank
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.parent_fd = os.open(parent, flags)
        self.root_fd = -1
        self.root_identity: tuple[int, int] | None = None

    def create_or_wait(self, deadline: float) -> None:
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
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECEIPT_BYTES:
            os.close(fd)
            raise TransactionError(f"invalid transaction artifact: {name}")
        return fd

    def read_bytes(self, name: str) -> bytes:
        fd = self._open_readonly(name)
        try:
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
            return b"".join(blocks)
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
        os.replace(
            temporary,
            name,
            src_dir_fd=self.root_fd,
            dst_dir_fd=self.root_fd,
        )
        os.fsync(self.root_fd)

    def close(self) -> None:
        if self.root_fd >= 0:
            os.close(self.root_fd)
            self.root_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


class Coordinator:
    def __init__(
        self,
        args: argparse.Namespace,
        workload: Sequence[str],
        runner: Mapping[str, Any],
        portable: Mapping[str, Any],
        tx: TransactionDirectory,
    ):
        self.args = args
        self.workload = list(workload)
        self.runner = dict(runner)
        self.portable = dict(portable)
        self.portable_sha256 = _sha256_bytes(_canonical_json_bytes(self.portable))
        self.tx = tx
        self.rank = args.node_rank
        self.peer_rank = args.peer_node_rank
        self.prepared_name = f"PREPARED.rank{self.rank}.json"
        self.peer_prepared_name = f"PREPARED.rank{self.peer_rank}.json"
        self.heartbeat_name = f"HEARTBEAT.rank{self.rank}.json"
        self.peer_heartbeat_name = f"HEARTBEAT.rank{self.peer_rank}.json"
        self.terminal_name = f"TERMINAL.rank{self.rank}.json"
        self.peer_terminal_name = f"TERMINAL.rank{self.peer_rank}.json"
        self.prepared_sha256 = ""
        self.sequence = 0
        self.state = "STARTING"
        self.abort_requested = False
        self.workload_process: subprocess.Popen[bytes] | None = None
        self.coordinator_identity = _proc_identity(os.getpid())
        self.peer_heartbeat_digest: str | None = None
        self.peer_heartbeat_last_change: float | None = None
        self.peer_heartbeat_missing_since: float | None = None

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
        runner_identity = {
            key: self.runner[key]
            for key in ("pid", "starttime_ticks", "argv_sha256")
        }
        if runner_now != runner_identity:
            raise TransactionError("guarded-runner identity changed")

    def _local_receipt(self) -> dict[str, Any]:
        self._assert_runner_identity()
        return {
            "schema": SCHEMA,
            "status": "PREPARED",
            "portable": self.portable,
            "portable_sha256": self.portable_sha256,
            "node": _participant(
                self.args.node_id,
                self.args.node_rank,
                self.args.node_ip,
            ),
            "runner": {
                **self.runner,
                "status_path": _validate_absolute_evidence_path(
                    self.args.runner_status_path,
                    "runner status path",
                ),
                "log_path": _validate_absolute_evidence_path(
                    self.args.runner_log_path,
                    "runner log path",
                ),
            },
            "coordinator": self.coordinator_identity,
            "workload_argv_sha256": _sha256_bytes(
                b"\0".join(os.fsencode(token) for token in self.workload) + b"\0"
            ),
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
        while not self.tx.exists("TRANSACTION.json"):
            if self.abort_requested:
                raise PeerAbort("local abort signal during bootstrap")
            if time.monotonic() >= deadline:
                raise TransactionError("timed out waiting for transaction bootstrap")
            time.sleep(0.01)
        observed, _ = self.tx.read_json("TRANSACTION.json")
        if observed != bootstrap:
            self._publish_abort("portable bootstrap mismatch")
            raise PeerAbort("portable bootstrap mismatch")

    def _publish_prepared(self) -> None:
        try:
            self.prepared_sha256 = self.tx.publish_immutable(
                self.prepared_name,
                self._local_receipt(),
            )
        except DuplicateInvocation:
            # A repeated node claim must not mutate the live transaction.
            raise
        self.state = "PREPARED"
        self._heartbeat(force=True)

    def _heartbeat(self, *, force: bool = False) -> None:
        del force
        self._assert_runner_identity()
        self.sequence += 1
        payload = {
            "schema": SCHEMA,
            "status": self.state,
            "rank": self.rank,
            "prepared_sha256": self.prepared_sha256,
            "coordinator": self.coordinator_identity,
            "sequence": self.sequence,
            "heartbeat_unix_ns": time.time_ns(),
        }
        self.tx.replace_heartbeat(self.heartbeat_name, payload)

    def _publish_terminal(self, status_value: str, reason: str, rc: int | None) -> None:
        payload = {
            "schema": SCHEMA,
            "status": status_value,
            "rank": self.rank,
            "prepared_sha256": self.prepared_sha256,
            "coordinator": self.coordinator_identity,
            "reason": reason,
            "workload_returncode": rc,
            "terminal_unix_ns": time.time_ns(),
        }
        try:
            self.tx.publish_immutable(self.terminal_name, payload)
        except DuplicateInvocation:
            existing, _ = self.tx.read_json(self.terminal_name)
            if existing != payload:
                raise TransactionError("conflicting local terminal receipt")

    def _publish_abort(self, reason: str) -> None:
        payload = {
            "schema": SCHEMA,
            "status": "ABORT",
            "rank": self.rank,
            "portable_sha256": self.portable_sha256,
            "reason": reason,
            "abort_unix_ns": time.time_ns(),
        }
        try:
            self.tx.publish_immutable("ABORT.json", payload)
        except DuplicateInvocation:
            # The first immutable abort is authoritative.
            pass

    def _check_abort(self) -> None:
        if self.abort_requested:
            raise PeerAbort("local abort signal")
        if self.tx.exists("ABORT.json"):
            payload, _ = self.tx.read_json("ABORT.json")
            if (
                set(payload)
                != {
                    "schema",
                    "status",
                    "rank",
                    "portable_sha256",
                    "reason",
                    "abort_unix_ns",
                }
                or payload.get("schema") != SCHEMA
                or payload.get("status") != "ABORT"
                or payload.get("rank") not in EXPECTED_RANKS
                or not isinstance(payload.get("reason"), str)
                or not isinstance(payload.get("abort_unix_ns"), int)
            ):
                raise TransactionError("invalid global abort receipt")
            raise PeerAbort(f"transaction aborted: {payload.get('reason', 'unknown')}")
        if self.tx.exists(self.peer_terminal_name):
            peer, _ = self.tx.read_json(self.peer_terminal_name)
            self._validate_peer_terminal(peer)
            status_value = peer.get("status")
            if status_value in {"FAILED", "ABORT"}:
                raise PeerAbort(f"peer terminal state: {status_value}")

    def _validate_prepared(self, payload: Mapping[str, Any], rank: int) -> None:
        expected_node = next(
            item for item in self.portable["participants"] if item["rank"] == rank
        )
        if (
            set(payload)
            != {
                "schema",
                "status",
                "portable",
                "portable_sha256",
                "node",
                "runner",
                "coordinator",
                "workload_argv_sha256",
                "transaction_root",
                "node_local_filesystem",
                "prepared_unix_ns",
            }
            or payload.get("schema") != SCHEMA
            or payload.get("status") != "PREPARED"
            or payload.get("portable") != self.portable
            or payload.get("portable_sha256") != self.portable_sha256
            or payload.get("node") != expected_node
            or payload.get("transaction_root") != str(self.tx.path)
            or not isinstance(payload.get("prepared_unix_ns"), int)
            or not HEX64_RE.fullmatch(str(payload.get("workload_argv_sha256", "")))
        ):
            raise PeerAbort(f"PREPARED receipt mismatch for rank {rank}")
        # Filesystem identity is required local evidence but never compared
        # against the other host's st_dev/st_ino values.
        fs_evidence = payload.get("node_local_filesystem")
        if (
            not isinstance(fs_evidence, dict)
            or set(fs_evidence) != {"st_dev", "st_ino"}
            or not all(isinstance(value, int) for value in fs_evidence.values())
        ):
            raise PeerAbort(f"missing node-local filesystem evidence for rank {rank}")

        identity_keys = {"pid", "starttime_ticks", "argv_sha256"}
        coordinator = payload.get("coordinator")
        if (
            not isinstance(coordinator, dict)
            or set(coordinator) != identity_keys
            or not isinstance(coordinator.get("pid"), int)
            or not isinstance(coordinator.get("starttime_ticks"), int)
            or not HEX64_RE.fullmatch(str(coordinator.get("argv_sha256", "")))
        ):
            raise PeerAbort(f"invalid coordinator identity for rank {rank}")

        runner = payload.get("runner")
        runner_keys = identity_keys | {"path", "sha256", "status_path", "log_path"}
        if (
            not isinstance(runner, dict)
            or set(runner) != runner_keys
            or not isinstance(runner.get("pid"), int)
            or not isinstance(runner.get("starttime_ticks"), int)
            or not HEX64_RE.fullmatch(str(runner.get("argv_sha256", "")))
            or runner.get("sha256") != self.portable["runner_sha256"]
            or not all(
                isinstance(runner.get(key), str)
                and Path(runner[key]).is_absolute()
                for key in ("path", "status_path", "log_path")
            )
        ):
            raise PeerAbort(f"invalid runner identity for rank {rank}")

    def _validate_peer_terminal(self, payload: Mapping[str, Any]) -> None:
        expected_keys = {
            "schema",
            "status",
            "rank",
            "prepared_sha256",
            "coordinator",
            "reason",
            "workload_returncode",
            "terminal_unix_ns",
        }
        peer_prepared, peer_prepared_sha256 = self.tx.read_json(
            self.peer_prepared_name
        )
        self._validate_prepared(peer_prepared, self.peer_rank)
        if (
            set(payload) != expected_keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") not in {"COMPLETED", "FAILED", "ABORT"}
            or payload.get("rank") != self.peer_rank
            or payload.get("prepared_sha256") != peer_prepared_sha256
            or payload.get("coordinator") != peer_prepared.get("coordinator")
            or not isinstance(payload.get("reason"), str)
            or not isinstance(payload.get("terminal_unix_ns"), int)
        ):
            raise PeerAbort("peer terminal identity mismatch")

    def _peer_is_stale(self) -> bool:
        if not self.tx.exists(self.peer_heartbeat_name):
            now = time.monotonic()
            if self.peer_heartbeat_missing_since is None:
                self.peer_heartbeat_missing_since = now
                return False
            return now - self.peer_heartbeat_missing_since > self.stale_seconds
        self.peer_heartbeat_missing_since = None
        peer, heartbeat_digest = self.tx.read_json(self.peer_heartbeat_name)
        peer_prepared, peer_prepared_sha256 = self.tx.read_json(
            self.peer_prepared_name
        )
        self._validate_prepared(peer_prepared, self.peer_rank)
        if (
            set(peer)
            != {
                "schema",
                "status",
                "rank",
                "prepared_sha256",
                "coordinator",
                "sequence",
                "heartbeat_unix_ns",
            }
            or peer.get("schema") != SCHEMA
            or peer.get("rank") != self.peer_rank
            or peer.get("prepared_sha256") != peer_prepared_sha256
            or peer.get("coordinator") != peer_prepared.get("coordinator")
            or peer.get("status")
            not in {"PREPARED", "RUNNING", "COMPLETED", "FAILED", "ABORT"}
            or not isinstance(peer.get("sequence"), int)
            or not isinstance(peer.get("heartbeat_unix_ns"), int)
        ):
            raise PeerAbort("peer heartbeat identity mismatch")
        now = time.monotonic()
        if heartbeat_digest != self.peer_heartbeat_digest:
            self.peer_heartbeat_digest = heartbeat_digest
            self.peer_heartbeat_last_change = now
            return False
        if self.peer_heartbeat_last_change is None:
            self.peer_heartbeat_last_change = now
            return False
        return now - self.peer_heartbeat_last_change > self.stale_seconds

    def _wait_for_file(self, name: str, timeout_ms: int, phase: str) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        next_heartbeat = 0.0
        while not self.tx.exists(name):
            self._check_abort()
            if self.tx.exists(self.peer_prepared_name) and self._peer_is_stale():
                raise PeerAbort(f"peer heartbeat stale during {phase}")
            now = time.monotonic()
            if now >= deadline:
                raise PeerAbort(f"timeout during {phase}")
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self.heartbeat_seconds
            time.sleep(min(0.02, self.heartbeat_seconds))

    def _prepare_and_commit(self) -> dict[int, str]:
        self._wait_for_file(
            self.peer_prepared_name,
            self.args.prepare_timeout_ms,
            "prepare",
        )
        prepared_hashes: dict[int, str] = {}
        for rank in EXPECTED_RANKS:
            payload, digest = self.tx.read_json(f"PREPARED.rank{rank}.json")
            try:
                self._validate_prepared(payload, rank)
            except PeerAbort as exc:
                self._publish_abort(str(exc))
                raise
            prepared_hashes[rank] = digest
        if self.rank == 0:
            commit = {
                "schema": SCHEMA,
                "status": "COMMITTED",
                "portable_sha256": self.portable_sha256,
                "prepared_sha256": {
                    str(rank): prepared_hashes[rank] for rank in EXPECTED_RANKS
                },
                "committed_by_rank": 0,
                "commit_unix_ns": time.time_ns(),
            }
            self.tx.publish_immutable("COMMIT.json", commit)
        self._wait_for_file("COMMIT.json", self.args.commit_timeout_ms, "commit")
        commit, _ = self.tx.read_json("COMMIT.json")
        if (
            commit.get("schema") != SCHEMA
            or commit.get("status") != "COMMITTED"
            or commit.get("portable_sha256") != self.portable_sha256
            or commit.get("committed_by_rank") != 0
            or commit.get("prepared_sha256")
            != {str(rank): prepared_hashes[rank] for rank in EXPECTED_RANKS}
        ):
            self._publish_abort("COMMIT receipt mismatch")
            raise PeerAbort("COMMIT receipt mismatch")
        return prepared_hashes

    def _terminate_workload(self) -> None:
        process = self.workload_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.args.shutdown_grace_ms / 1000.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.args.shutdown_grace_ms / 1000.0)
            except subprocess.TimeoutExpired:
                pass

    def _run_workload(self) -> int:
        environment = os.environ.copy()
        environment.update(
            {
                "SEMTALK_W16_TRANSACTION_ROOT": str(self.tx.path),
                "SEMTALK_W16_RUN_ID": self.args.run_id,
                "SEMTALK_W16_NODE_ID": self.args.node_id,
                "SEMTALK_W16_NODE_RANK": str(self.rank),
                "SEMTALK_W16_MAX_RESTARTS": "0",
            }
        )
        self.state = "RUNNING"
        self._heartbeat(force=True)
        self.workload_process = subprocess.Popen(
            self.workload,
            env=environment,
            start_new_session=True,
        )
        next_heartbeat = time.monotonic() + self.heartbeat_seconds
        try:
            while True:
                rc = self.workload_process.poll()
                if rc is not None:
                    if rc != 0:
                        self.state = "FAILED"
                        self._publish_terminal("FAILED", "workload returned non-zero", rc)
                        self._publish_abort("peer workload returned non-zero")
                        return rc if rc > 0 else 1
                    self.state = "COMPLETED"
                    self._publish_terminal("COMPLETED", "workload completed", 0)
                    break
                self._check_abort()
                if self._peer_is_stale():
                    raise PeerAbort("peer heartbeat became stale while running")
                now = time.monotonic()
                if now >= next_heartbeat:
                    self._heartbeat()
                    next_heartbeat = now + self.heartbeat_seconds
                time.sleep(min(0.02, self.heartbeat_seconds))
        except BaseException:
            self._terminate_workload()
            raise

        deadline = time.monotonic() + self.args.completion_timeout_ms / 1000.0
        next_heartbeat = 0.0
        while True:
            self._check_abort()
            if self.tx.exists(self.peer_terminal_name):
                peer, _ = self.tx.read_json(self.peer_terminal_name)
                self._validate_peer_terminal(peer)
                if peer.get("status") == "COMPLETED":
                    return 0
                raise PeerAbort(f"peer terminal state: {peer.get('status')}")
            if self._peer_is_stale():
                raise PeerAbort("peer heartbeat stale during completion")
            now = time.monotonic()
            if now >= deadline:
                raise PeerAbort("timeout waiting for peer completion")
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self.heartbeat_seconds
            time.sleep(min(0.02, self.heartbeat_seconds))

    def run(self) -> int:
        self._bootstrap(
            time.monotonic() + self.args.prepare_timeout_ms / 1000.0
        )
        self._publish_prepared()
        self._prepare_and_commit()
        return self._run_workload()

    def fail(self, exc: BaseException) -> None:
        self._terminate_workload()
        if isinstance(exc, DuplicateInvocation):
            return
        reason = f"{type(exc).__name__}: {exc}"
        try:
            self.state = "ABORT" if isinstance(exc, PeerAbort) else "FAILED"
            # Publish the shared abort first.  The local immutable terminal may
            # already say COMPLETED while this node is waiting for its peer;
            # that receipt must never be overwritten, but the later global
            # failure must still become visible to the other runner.
            self._publish_abort(reason)
            self._publish_terminal(self.state, reason, None)
        except BaseException:
            # Preserve the original fail-closed outcome even if evidence
            # publication itself is impossible after a TOCTOU replacement.
            pass


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
    parser.add_argument("--max-restarts", required=True, type=int)
    parser.add_argument("--heartbeat-ms", type=int, default=1000)
    parser.add_argument("--stale-ms", type=int, default=15000)
    parser.add_argument("--prepare-timeout-ms", type=int, default=120000)
    parser.add_argument("--commit-timeout-ms", type=int, default=120000)
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
    try:
        transaction_root, parent, basename = _validate_tx_path(args.transaction_root)
        runner, runner_sha256 = _runner_evidence()
        portable = _portable_payload(args, runner_sha256)
        tx = TransactionDirectory(
            transaction_root,
            parent,
            basename,
            args.node_rank,
        )
        tx.create_or_wait(
            time.monotonic() + args.prepare_timeout_ms / 1000.0
        )
        coordinator = Coordinator(args, workload, runner, portable, tx)
        signal.signal(signal.SIGTERM, coordinator.request_abort)
        signal.signal(signal.SIGINT, coordinator.request_abort)
        return coordinator.run()
    except BaseException as exc:
        if coordinator is not None:
            coordinator.fail(exc)
        print(f"dual-node guarded transaction failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if tx is not None:
            tx.close()


if __name__ == "__main__":
    raise SystemExit(main())
