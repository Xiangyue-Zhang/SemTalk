#!/usr/bin/env python3
"""Fail-closed two-node transaction for guarded GPU workloads.

The transaction root must not exist.  Rank zero creates it and publishes an
immutable bootstrap receipt; rank one only joins that exact transaction.  A
node publishes immutable PREPARED and ARMED receipts with ``O_EXCL`` and
``fsync``.  A non-GPU supervisor pins an otherwise empty process group; the
actual workload cannot exec until the single immutable DECISION is GO.  Each
successful coordinator publishes only a HANDOFF after exact descendant
cleanup, then exits so its outer guarded runner can restore all GPU guards.
An independent CPU finalizer verifies both finished runner receipts and logs,
publishes both immutable FINAL receipts, and publishes the sole successful
OUTCOME as its last filesystem operation.  ``replay`` is strictly read-only.

Filesystem device/inode identities are deliberately node-local evidence.  The
portable payload instead binds one deployment ID and one canonical parent-path
hash, so two parents can never claim the same run ID.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar
import uuid


SCHEMA = "semtalk.dual_node_guarded_transaction.v3"
PORTABLE_SCHEMA = f"{SCHEMA}.portable"
EXPECTED_RUNNER = "/tmp/globaldiff_guarded_runner.py"
EXPECTED_RUNNER_GPUS = "0,1,2,3,4,5,6,7"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_RECEIPT_BYTES = 1 << 20
ESTALE_RETRY_ATTEMPTS = 5
ESTALE_RETRY_BASE_SECONDS = 0.01
EXPECTED_RANKS = (0, 1)
IDENTITY_KEYS = {
    "pid", "ppid", "pgid", "sid", "starttime_ticks", "argv_sha256"
}
PINNED_INPUT_FD_BASE = 200
WORKLOAD_INPUT_SCHEMA = {
    "fresh_test_authority": "--fresh-test-authority",
}
OUTER_ARTIFACT_BINDING_FORMAT = (
    "semtalk.dual_node_guarded_transaction.outer_artifact_binding.v1"
)

# CPU tests import this module and replace the process-evidence provider.  The
# installed CLI never exposes a switch that weakens Linux runner ancestry or
# pinned-fd execution.
_CPU_TEST_MODE = False
_T = TypeVar("_T")


class TransactionError(RuntimeError):
    """A fail-closed transaction error."""


class DuplicateInvocation(TransactionError):
    """The transaction or this node's immutable claim already exists."""


class PeerAbort(TransactionError):
    """The peer failed, aborted, became stale, or violated the protocol."""


def _is_estale(exc: BaseException) -> bool:
    """Return whether one exact filesystem syscall reported EFS/NFS ESTALE."""
    return isinstance(exc, OSError) and exc.errno == errno.ESTALE


def _estale_backoff(attempt: int) -> None:
    """Small bounded delay; enclosing protocol deadlines remain authoritative."""
    time.sleep(ESTALE_RETRY_BASE_SECONDS * (2 ** attempt))


def _retry_estale_syscall(operation: Callable[[], _T]) -> _T:
    """Retry only a single exact, idempotent filesystem syscall on ESTALE."""
    for attempt in range(ESTALE_RETRY_ATTEMPTS):
        try:
            return operation()
        except OSError as exc:
            if not _is_estale(exc) or attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                raise
            _estale_backoff(attempt)
    raise AssertionError("unreachable ESTALE retry state")


def _rename_noreplace(
    source_dir_fd: int,
    source: str,
    target_dir_fd: int,
    target: str,
    *,
    expected_source_identity: tuple[int, int] | None = None,
) -> None:
    """Atomically rename one entry without replacing an existing target."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        try:
            rename = libc.renameat2
        except AttributeError as exc:  # pragma: no cover - modern glibc exports it.
            raise TransactionError("atomic no-replace rename is unavailable") from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            os.fsencode(source),
            target_dir_fd,
            os.fsencode(target),
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin" and _CPU_TEST_MODE:
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            os.fsencode(source),
            target_dir_fd,
            os.fsencode(target),
            0x00000004,  # RENAME_EXCL
        )
    else:  # pragma: no cover - formal execution is Linux-only.
        raise TransactionError("atomic no-replace rename is unavailable")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), target)
    if error in {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }:
        # AWS EFS supports atomic hard-link creation but returns EINVAL for
        # renameat2(RENAME_NOREPLACE).  The source is a unique, O_EXCL,
        # O_NOFOLLOW regular file in the same pinned directory, so linkat is
        # an equally fail-closed no-replace publication primitive.  Once the
        # link succeeds, target is the irreversible publication commit point:
        # never remove it in response to a later temporary-name cleanup
        # failure, because a concurrent replay may already have observed the
        # terminal receipt.  A failed cleanup can therefore leave only the
        # unique read-only temporary hard-link alias; the caller still performs
        # the directory durability barrier for the committed target.
        os.link(
            source,
            target,
            src_dir_fd=source_dir_fd,
            dst_dir_fd=target_dir_fd,
            follow_symlinks=False,
        )
        try:
            if expected_source_identity is not None:
                source_info = os.stat(
                    source,
                    dir_fd=source_dir_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(source_info.st_mode)
                    or (source_info.st_dev, source_info.st_ino)
                    != expected_source_identity
                ):
                    raise TransactionError(
                        "terminal temporary identity changed before cleanup"
                    )
            os.unlink(source, dir_fd=source_dir_fd)
        except OSError:
            # Publication is monotonic.  The unique source alias is harmless
            # and must not trigger rollback of an already visible target.
            pass
        return
    raise OSError(error, os.strerror(error), target)


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
    if identity["argv_sha256"] != _argv_sha256(tokens):
        raise TransactionError("guarded runner full argv changed during capture")
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
    command = tokens[delimiter + 1 :]
    if not command:
        raise TransactionError("guarded runner has no workload command")
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
            "argv": tokens,
            "command": command,
            "command_argv_sha256": _argv_sha256(command),
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


def _stat_binding(value: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "st_mode": value.st_mode,
        "st_size": value.st_size,
        "st_mtime_ns": value.st_mtime_ns,
    }


def _read_pyvenv_configuration(fd: int) -> dict[str, str]:
    os.lseek(fd, 0, os.SEEK_SET)
    raw = b""
    while True:
        block = os.read(fd, 65536)
        if not block:
            break
        raw += block
        if len(raw) > 65536:
            raise TransactionError("formal Python pyvenv.cfg is too large")
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TransactionError("formal Python pyvenv.cfg is not UTF-8") from exc
    configuration: dict[str, str] = {}
    for raw_line in lines:
        if not raw_line.strip():
            continue
        if "=" not in raw_line:
            raise TransactionError("formal Python pyvenv.cfg record is invalid")
        key, value = raw_line.split("=", 1)
        key = key.strip().casefold()
        if not key or key in configuration:
            raise TransactionError("formal Python pyvenv.cfg key is invalid")
        configuration[key] = value.strip()
    if not {"home", "include-system-site-packages", "version"}.issubset(
        configuration
    ):
        raise TransactionError("formal Python pyvenv.cfg is incomplete")
    if configuration["include-system-site-packages"].casefold() not in {
        "true",
        "false",
    }:
        raise TransactionError("formal Python pyvenv.cfg site policy is invalid")
    if configuration["version"] != platform.python_version():
        raise TransactionError("formal Python pyvenv.cfg version changed")
    return configuration


def _capture_formal_venv_python(
    path: Path,
) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    """Pin one already-validated venv Python without erasing its argv[0].

    This is deliberately separate from ``_open_pinned_file``: every generic
    workload and authority input remains canonical and symlink-free.  The only
    accepted symlink is the exact Python used to run this coordinator, under
    its canonical ``venv/bin`` parent.  Every symlink hop is recorded while
    the final ELF and pyvenv.cfg remain held open.
    """

    if (
        not path.is_absolute()
        or os.fsencode(str(path)) != os.fsencode(sys.executable)
        or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", path.name)
    ):
        raise TransactionError("formal workload Python is not sys.executable")
    try:
        parent = path.parent.resolve(strict=True)
        venv_root = parent.parent.resolve(strict=True)
    except OSError as exc:
        raise TransactionError("formal workload Python venv is unavailable") from exc
    if (
        parent != path.parent
        or parent.name != "bin"
        or parent.is_symlink()
        or venv_root != parent.parent
        or venv_root.is_symlink()
        or Path(sys.prefix) != venv_root
        or Path(sys.exec_prefix) != venv_root
        or sys.prefix == sys.base_prefix
        or sys.exec_prefix == sys.base_exec_prefix
    ):
        raise TransactionError("formal workload Python venv identity changed")

    portable_chain: list[dict[str, str]] = []
    local_chain: list[dict[str, Any]] = []
    current = path
    visited: set[str] = set()
    for _ordinal in range(16):
        current_key = str(current)
        if current_key in visited:
            raise TransactionError("formal workload Python symlink loop")
        visited.add(current_key)
        try:
            if current.parent.resolve(strict=True) != current.parent:
                raise TransactionError(
                    "formal workload Python chain parent is noncanonical"
                )
            before = current.lstat()
        except OSError as exc:
            raise TransactionError("formal workload Python chain disappeared") from exc
        if not stat.S_ISLNK(before.st_mode):
            break
        try:
            target_text = os.readlink(current)
            after = current.lstat()
        except OSError as exc:
            raise TransactionError("formal workload Python link changed") from exc
        if _stat_binding(before) != _stat_binding(after):
            raise TransactionError("formal workload Python link changed while pinning")
        next_path = Path(target_text)
        if not next_path.is_absolute():
            next_path = current.parent / next_path
        next_path = Path(os.path.normpath(str(next_path)))
        if not next_path.is_absolute() or next_path.name in {"", ".", ".."}:
            raise TransactionError("formal workload Python link target is unsafe")
        portable_chain.append({"path": str(current), "target": target_text})
        local_chain.append(
            {
                "path": str(current),
                "target": target_text,
                "identity": _stat_binding(before),
            }
        )
        current = next_path
    else:
        raise TransactionError("formal workload Python symlink chain is too deep")
    if not portable_chain:
        raise TransactionError("formal workload Python is not a venv leaf symlink")
    try:
        resolved_target = path.resolve(strict=True)
    except OSError as exc:
        raise TransactionError("formal workload Python target is unavailable") from exc
    if resolved_target != current or current.is_symlink():
        raise TransactionError("formal workload Python chain resolution changed")
    executable_fd, canonical_target = _open_pinned_file(
        resolved_target, executable=True
    )
    target_info = os.fstat(executable_fd)

    pyvenv_cfg = venv_root / "pyvenv.cfg"
    try:
        pyvenv_fd, canonical_cfg = _open_pinned_file(
            pyvenv_cfg, executable=False
        )
    except BaseException:
        os.close(executable_fd)
        raise
    if canonical_cfg != pyvenv_cfg:
        os.close(pyvenv_fd)
        os.close(executable_fd)
        raise TransactionError("formal Python pyvenv.cfg is noncanonical")
    try:
        configuration = _read_pyvenv_configuration(pyvenv_fd)
        configured_home = Path(configuration["home"]).resolve(strict=True)
        home_executable = (configured_home / resolved_target.name).resolve(strict=True)
        if home_executable != resolved_target:
            raise TransactionError("formal Python pyvenv.cfg home changed")
        configured_executable = configuration.get("executable")
        if (
            configured_executable is not None
            and Path(configured_executable).resolve(strict=True) != resolved_target
        ):
            raise TransactionError("formal Python pyvenv.cfg executable changed")
        target_sha256 = _sha256_fd(executable_fd)
        cfg_sha256 = _sha256_fd(pyvenv_fd)
        cfg_info = os.fstat(pyvenv_fd)
    except BaseException:
        os.close(pyvenv_fd)
        os.close(executable_fd)
        raise
    portable = {
        "format": "semtalk.formal_venv_python_binding.v1",
        "argv0": str(path),
        "venv_root": str(venv_root),
        "symlink_chain": portable_chain,
        "resolved_target": {
            "path": str(canonical_target),
            "sha256": target_sha256,
            "bytes": target_info.st_size,
        },
        "pyvenv_cfg": {
            "path": str(canonical_cfg),
            "sha256": cfg_sha256,
            "bytes": cfg_info.st_size,
        },
    }
    node_local = {
        "format": "semtalk.formal_venv_python_node_binding.v1",
        "symlink_chain": local_chain,
        "resolved_target_identity": _stat_binding(target_info),
        "pyvenv_cfg_identity": _stat_binding(cfg_info),
    }
    return executable_fd, pyvenv_fd, portable, node_local


def _runner_proves_formal_python_contract(
    runner: Mapping[str, Any],
    workload: Sequence[str],
) -> bool:
    repository = Path(__file__).resolve().parents[2]
    launcher = repository / "scripts/show_base/run_dual_node_guarded_transaction.sh"
    command = runner.get("command")
    if not isinstance(command, list):
        return False
    return (
        os.fsencode(workload[0]) == os.fsencode(sys.executable)
        and len(command) > len(workload) + 3
        and command[:3] == ["/bin/bash", str(launcher), workload[0]]
        and command[-len(workload) - 1] == "--"
        and command[-len(workload) :] == list(workload)
    )


def _assert_formal_venv_python_binding(
    portable: Mapping[str, Any],
    node_local: Mapping[str, Any],
    executable_fd: int,
    pyvenv_fd: int,
    monitor: "_FormalRuntimeMonitor | None" = None,
) -> None:
    new_executable_fd = -1
    new_pyvenv_fd = -1
    try:
        if monitor is not None:
            monitor.assert_quiet("formal Python binding precheck")
        (
            new_executable_fd,
            new_pyvenv_fd,
            observed_portable,
            observed_local,
        ) = _capture_formal_venv_python(Path(str(portable.get("argv0", ""))))
        if observed_portable != portable or observed_local != node_local:
            raise TransactionError("formal workload Python public binding changed")
        if (
            _stat_binding(os.fstat(executable_fd))
            != node_local.get("resolved_target_identity")
            or _sha256_fd(executable_fd)
            != portable.get("resolved_target", {}).get("sha256")
            or _stat_binding(os.fstat(pyvenv_fd))
            != node_local.get("pyvenv_cfg_identity")
            or _sha256_fd(pyvenv_fd)
            != portable.get("pyvenv_cfg", {}).get("sha256")
        ):
            raise TransactionError("formal workload Python pinned binding changed")
        if monitor is not None:
            monitor.assert_quiet("formal Python binding postcheck")
    finally:
        if new_pyvenv_fd >= 0:
            os.close(new_pyvenv_fd)
        if new_executable_fd >= 0:
            os.close(new_executable_fd)


def _validate_node_local_executable_binding(
    value: Any,
    portable: Mapping[str, Any],
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "format",
        "symlink_chain",
        "resolved_target_identity",
        "pyvenv_cfg_identity",
    }:
        raise PeerAbort("formal Python node-local binding schema mismatch")
    chain = value.get("symlink_chain")
    portable_chain = portable.get("symlink_chain")
    if (
        value.get("format") != "semtalk.formal_venv_python_node_binding.v1"
        or not isinstance(chain, list)
        or not isinstance(portable_chain, list)
        or len(chain) != len(portable_chain)
        or not chain
    ):
        raise PeerAbort("formal Python node-local chain mismatch")
    for local_hop, portable_hop in zip(chain, portable_chain):
        if (
            not isinstance(local_hop, dict)
            or set(local_hop) != {"path", "target", "identity"}
            or not isinstance(portable_hop, dict)
            or local_hop.get("path") != portable_hop.get("path")
            or local_hop.get("target") != portable_hop.get("target")
        ):
            raise PeerAbort("formal Python node-local hop mismatch")
        _validate_stat_binding(local_hop.get("identity"))
    _validate_stat_binding(value.get("resolved_target_identity"))
    _validate_stat_binding(value.get("pyvenv_cfg_identity"))


def _validate_stat_binding(value: Any) -> None:
    keys = {"st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or not all(_exact_int(value.get(key), minimum=0) for key in keys)
    ):
        raise PeerAbort("formal Python node-local stat binding is invalid")


class _FormalRuntimeMonitor:
    """Fail-closed Linux inotify fence for the public venv binding."""

    _EVENT = struct.Struct("iIII")
    _IN_ATTRIB = 0x00000004
    _IN_CLOSE_WRITE = 0x00000008
    _IN_MODIFY = 0x00000002
    _IN_MOVED_FROM = 0x00000040
    _IN_MOVED_TO = 0x00000080
    _IN_CREATE = 0x00000100
    _IN_DELETE = 0x00000200
    _IN_DELETE_SELF = 0x00000400
    _IN_MOVE_SELF = 0x00000800
    _IN_UNMOUNT = 0x00002000
    _IN_Q_OVERFLOW = 0x00004000
    _IN_IGNORED = 0x00008000
    _IN_DONT_FOLLOW = 0x02000000
    _IN_MASK_ADD = 0x20000000
    _SELF_MASK = (
        _IN_ATTRIB
        | _IN_CLOSE_WRITE
        | _IN_MODIFY
        | _IN_DELETE_SELF
        | _IN_MOVE_SELF
        | _IN_UNMOUNT
        | _IN_IGNORED
    )
    _PARENT_MASK = (
        _IN_ATTRIB
        | _IN_CREATE
        | _IN_DELETE
        | _IN_MOVED_FROM
        | _IN_MOVED_TO
        | _IN_DELETE_SELF
        | _IN_MOVE_SELF
        | _IN_UNMOUNT
        | _IN_IGNORED
    )

    def __init__(self, binding: Mapping[str, Any]):
        self.fd = -1
        self._watch: dict[int, dict[str, Any]] = {}
        if sys.platform != "linux":
            if _CPU_TEST_MODE:
                return
            raise TransactionError("formal Python runtime fence requires Linux")
        libc = ctypes.CDLL(None, use_errno=True)
        self._libc = libc
        try:
            init = libc.inotify_init1
            add = libc.inotify_add_watch
        except AttributeError as exc:  # pragma: no cover - supported Linux.
            raise TransactionError("formal Python inotify fence is unavailable") from exc
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        fd = init(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            error = ctypes.get_errno()
            raise TransactionError(
                f"formal Python inotify initialization failed: {os.strerror(error)}"
            )
        self.fd = fd
        self._add = add
        paths = [
            Path(str(hop["path"]))
            for hop in binding.get("symlink_chain", [])
        ]
        paths.extend(
            [
                Path(str(binding["resolved_target"]["path"])),
                Path(str(binding["pyvenv_cfg"]["path"])),
                Path(str(binding["venv_root"])),
                Path(str(binding["venv_root"])) / "bin",
            ]
        )
        try:
            for path in paths:
                self._add_path(path, all_events=True)
                self._add_path(path.parent, all_events=False, name=path.name)
            self.assert_quiet("runtime-fence initialization")
        except BaseException:
            self.close()
            raise

    def _add_path(
        self,
        path: Path,
        *,
        all_events: bool,
        name: str | None = None,
    ) -> None:
        mask = (
            self._SELF_MASK | self._IN_DONT_FOLLOW
            if all_events
            else self._PARENT_MASK
        )
        wd = self._add(
            self.fd,
            os.fsencode(str(path)),
            mask | self._IN_MASK_ADD,
        )
        if wd < 0:
            error = ctypes.get_errno()
            raise TransactionError(
                f"formal Python inotify watch failed: {os.strerror(error)}"
            )
        record = self._watch.setdefault(wd, {"all": False, "names": set()})
        record["all"] = bool(record["all"] or all_events)
        if name is not None:
            record["names"].add(name)

    def assert_quiet(self, phase: str) -> None:
        if self.fd < 0:
            return
        while True:
            try:
                raw = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            except OSError as exc:
                raise TransactionError(
                    f"formal Python runtime fence failed during {phase}"
                ) from exc
            if not raw:
                raise TransactionError(
                    f"formal Python runtime fence closed during {phase}"
                )
            offset = 0
            while offset < len(raw):
                if len(raw) - offset < self._EVENT.size:
                    raise TransactionError("formal Python inotify event is truncated")
                wd, mask, _cookie, name_length = self._EVENT.unpack_from(raw, offset)
                offset += self._EVENT.size
                if name_length > len(raw) - offset:
                    raise TransactionError("formal Python inotify name is truncated")
                name_raw = raw[offset : offset + name_length].split(b"\0", 1)[0]
                offset += name_length
                if mask & self._IN_Q_OVERFLOW:
                    raise TransactionError("formal Python inotify queue overflowed")
                record = self._watch.get(wd)
                if record is None:
                    raise TransactionError("formal Python inotify watch changed")
                try:
                    name = os.fsdecode(name_raw)
                except UnicodeDecodeError as exc:
                    raise TransactionError("formal Python inotify name is invalid") from exc
                if record["all"] or name in record["names"]:
                    raise TransactionError(
                        f"formal Python runtime binding changed during {phase}"
                    )

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


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
        "scripts/show_base/formal_python_runtime_contract.sh",
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
    runner: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, str],
    list[str],
    int,
    int,
    dict[str, int],
    dict[str, int],
    dict[str, Any] | None,
    _FormalRuntimeMonitor | None,
]:
    executable_path = Path(workload[0])
    runtime_fds: dict[str, int] = {}
    executable_binding: dict[str, Any] | None = None
    node_local_executable: dict[str, Any] | None = None
    runtime_monitor: _FormalRuntimeMonitor | None = None
    if _runner_proves_formal_python_contract(runner, workload):
        (
            executable_fd,
            pyvenv_fd,
            executable_binding,
            node_local_executable,
        ) = _capture_formal_venv_python(executable_path)
        runtime_fds["pyvenv_cfg"] = pyvenv_fd
        executable = executable_path
        try:
            runtime_monitor = _FormalRuntimeMonitor(executable_binding)
            _assert_formal_venv_python_binding(
                executable_binding,
                node_local_executable,
                executable_fd,
                pyvenv_fd,
                runtime_monitor,
            )
        except BaseException:
            if runtime_monitor is not None:
                runtime_monitor.close()
            os.close(pyvenv_fd)
            os.close(executable_fd)
            raise
    else:
        executable_fd, executable = _open_pinned_file(
            executable_path, executable=True
        )

    def close_executable_binding() -> None:
        if runtime_monitor is not None:
            runtime_monitor.close()
        for runtime_fd in runtime_fds.values():
            os.close(runtime_fd)
        runtime_fds.clear()
        os.close(executable_fd)

    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        close_executable_binding()
        raise TransactionError("workdir must be absolute")
    try:
        canonical_workdir = workdir.resolve(strict=True)
    except OSError as exc:
        close_executable_binding()
        raise TransactionError("workdir is unavailable") from exc
    if canonical_workdir != workdir or not workdir.is_dir():
        close_executable_binding()
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
        close_executable_binding()
        raise TransactionError("workdir identity changed while pinning")

    allow_names = list(args.allow_env)
    if len(allow_names) != len(set(allow_names)):
        close_executable_binding()
        os.close(workdir_fd)
        raise TransactionError("allow-env names must be unique")
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    for name in sorted(allow_names):
        if (
            not ENV_NAME_RE.fullmatch(name)
            or name.startswith("SEMTALK_W16_")
            or name not in os.environ
        ):
            close_executable_binding()
            os.close(workdir_fd)
            raise TransactionError(f"invalid or unavailable allow-env name: {name}")
        environment[name] = os.environ[name]

    input_fds: dict[str, int] = {}
    input_hashes: dict[str, str] = {}
    input_paths: dict[str, str] = {}
    input_identities: dict[str, tuple[int, int]] = {}
    try:
        for raw in args.workload_input:
            if "=" not in raw:
                raise TransactionError("workload-input must be LOGICAL_ID=/absolute/path")
            logical_id, raw_path = raw.split("=", 1)
            if logical_id not in WORKLOAD_INPUT_SCHEMA or logical_id in input_fds:
                raise TransactionError(
                    "workload-input logical ID is not uniquely schema-allowlisted"
                )
            fd, canonical = _open_pinned_file(Path(raw_path), executable=False)
            if str(canonical) != raw_path:
                os.close(fd)
                raise TransactionError("workload-input path must be canonical")
            input_fds[logical_id] = fd
            input_hashes[logical_id] = _sha256_fd(fd)
            input_paths[logical_id] = raw_path
            info = os.fstat(fd)
            input_identities[logical_id] = (info.st_dev, info.st_ino)

        if len(set(input_identities.values())) != len(input_identities):
            raise TransactionError("workload-input declarations alias one inode")
        if len(set(input_hashes.values())) != len(input_hashes):
            raise TransactionError("workload-input declarations alias identical content")

        exec_workload = list(workload)
        bindings: dict[str, dict[str, Any]] = {}
        authorized_indexes: set[int] = set()
        fd_root = "/proc/self/fd" if sys.platform == "linux" else "/dev/fd"
        if sys.platform != "linux" and not _CPU_TEST_MODE:
            raise TransactionError("formal pinned-input exec requires Linux /proc")
        try:
            open_max = int(os.sysconf("SC_OPEN_MAX"))
        except (OSError, ValueError):
            open_max = 256
        for ordinal, logical_id in enumerate(sorted(input_fds)):
            option = WORKLOAD_INPUT_SCHEMA[logical_id]
            option_indexes = [
                index for index, token in enumerate(workload) if token == option
            ]
            if len(option_indexes) != 1:
                raise TransactionError(
                    f"workload input {logical_id} requires one exact {option} option"
                )
            option_index = option_indexes[0]
            value_index = option_index + 1
            if (
                value_index >= len(workload)
                or workload[value_index] != input_paths[logical_id]
                or [
                    index
                    for index, token in enumerate(workload)
                    if token == input_paths[logical_id]
                ]
                != [value_index]
            ):
                raise TransactionError(
                    f"workload input {logical_id} is not at its allowlisted argv position"
                )
            target_fd = PINNED_INPUT_FD_BASE + ordinal
            if target_fd >= open_max:
                raise TransactionError("pinned workload FD exceeds process limit")
            exec_path = f"{fd_root}/{target_fd}"
            exec_workload[value_index] = exec_path
            authorized_indexes.add(value_index)
            bindings[logical_id] = {
                "option": option,
                "option_index": option_index,
                "value_index": value_index,
                "passed_fd": target_fd,
                "exec_path": exec_path,
                "sha256": input_hashes[logical_id],
            }

        # Reject a second argv spelling of the same input, including a hardlink,
        # symlink, or byte-identical copy.  Only the schema-owned value slot may
        # carry an authority input into the workload.
        for index, token in enumerate(workload):
            if index in authorized_indexes:
                continue
            if any(path in token for path in input_paths.values()):
                raise TransactionError("workload argv repeats a declared input path")
            candidates = {token}
            if "=" in token:
                _prefix, value = token.split("=", 1)
                candidates.add(value)
            for encoded in tuple(candidates):
                try:
                    decoded = json.loads(encoded)
                except json.JSONDecodeError:
                    continue
                except RecursionError as exc:
                    raise TransactionError("workload argv JSON is too deeply nested") from exc
                pending = [decoded]
                visited = 0
                while pending:
                    visited += 1
                    if visited > 4096:
                        raise TransactionError("workload argv JSON is too large to audit")
                    value = pending.pop()
                    if isinstance(value, str):
                        candidates.add(value)
                    elif isinstance(value, list):
                        pending.extend(value)
                    elif isinstance(value, dict):
                        pending.extend(value.keys())
                        pending.extend(value.values())
            candidates.update(
                re.findall(r"/[^\s\"'\],}]+", token)
            )
            for candidate in candidates:
                candidate_path = Path(candidate)
                try:
                    resolved_candidate = (
                        candidate_path
                        if candidate_path.is_absolute()
                        else canonical_workdir / candidate_path
                    ).resolve(strict=True)
                except (OSError, ValueError):
                    continue
                if str(resolved_candidate) in input_paths.values():
                    raise TransactionError("workload argv contains an input path alias")
                try:
                    candidate_fd, _ = _open_pinned_file(
                        resolved_candidate, executable=False
                    )
                except TransactionError:
                    continue
                try:
                    info = os.fstat(candidate_fd)
                    identity = (info.st_dev, info.st_ino)
                    digest = _sha256_fd(candidate_fd)
                finally:
                    os.close(candidate_fd)
                if identity in input_identities.values() or digest in input_hashes.values():
                    raise TransactionError(
                        "workload argv contains a same-inode or same-content input alias"
                    )
    except BaseException:
        for fd in input_fds.values():
            os.close(fd)
        for fd in runtime_fds.values():
            os.close(fd)
        if runtime_monitor is not None:
            runtime_monitor.close()
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
        "input_bindings": bindings,
        "exec_argv_sha256": _argv_sha256(exec_workload),
    }
    if executable_binding is not None:
        evidence["formal_python_binding"] = executable_binding
    return (
        evidence,
        environment,
        exec_workload,
        executable_fd,
        workdir_fd,
        input_fds,
        runtime_fds,
        node_local_executable,
        runtime_monitor,
    )


def _namespace_parent_sha256(parent: Path) -> str:
    return _sha256_bytes(os.fsencode(str(parent)) + b"\0")


def _validate_tx_path(
    raw: str,
    run_id: str,
    expected_parent_sha256: str,
) -> tuple[Path, Path, str]:
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
    if (
        not HEX64_RE.fullmatch(expected_parent_sha256)
        or _namespace_parent_sha256(parent) != expected_parent_sha256
    ):
        raise TransactionError("transaction parent differs from deployment pin")
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
    if not SAFE_ID_RE.fullmatch(args.deployment_id):
        raise TransactionError("unsafe deployment ID")
    parent = Path(args.transaction_root).parent
    if (
        not HEX64_RE.fullmatch(args.expected_namespace_parent_sha256)
        or _namespace_parent_sha256(parent)
        != args.expected_namespace_parent_sha256
    ):
        raise TransactionError("portable namespace parent pin mismatch")
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
        "namespace": {
            "deployment_id": args.deployment_id,
            "canonical_parent": str(parent),
            "canonical_parent_sha256": args.expected_namespace_parent_sha256,
        },
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
        self.namespace_fds = [
            _retry_estale_syscall(lambda: os.open("/", flags))
        ]
        self.namespace_names: list[str] = []
        root_info = _retry_estale_syscall(
            lambda: os.fstat(self.namespace_fds[0])
        )
        self.namespace_identities = [(root_info.st_dev, root_info.st_ino)]
        for component in parent.parts[1:]:
            child_fd = _retry_estale_syscall(
                lambda component=component: os.open(
                    component,
                    flags,
                    dir_fd=self.namespace_fds[-1],
                )
            )
            child = _retry_estale_syscall(lambda: os.fstat(child_fd))
            if not stat.S_ISDIR(child.st_mode):
                os.close(child_fd)
                raise TransactionError("transaction namespace component is not a directory")
            self.namespace_names.append(component)
            self.namespace_fds.append(child_fd)
            self.namespace_identities.append((child.st_dev, child.st_ino))
        self.parent_fd = self.namespace_fds[-1]
        namespace = _retry_estale_syscall(lambda: os.fstat(self.parent_fd))
        if namespace.st_uid != os.geteuid() or namespace.st_mode & 0o022:
            self.close()
            raise TransactionError("transaction namespace has unsafe ownership or mode")

    def _refresh_directory_handles(self) -> None:
        """Reopen the exact pinned namespace/root after one stale file handle.

        New descriptors are swapped in only after every component and the
        transaction root match the identities captured before ESTALE.  A path
        replacement therefore fails closed rather than turning a retry into a
        different transaction.
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        refreshed: list[int] = []
        refreshed_root = -1
        try:
            refreshed.append(os.open("/", flags))
            root_info = os.fstat(refreshed[0])
            if (root_info.st_dev, root_info.st_ino) != self.namespace_identities[0]:
                raise TransactionError("filesystem root identity changed during ESTALE retry")
            for index, component in enumerate(self.namespace_names, start=1):
                child_fd = os.open(component, flags, dir_fd=refreshed[-1])
                refreshed.append(child_fd)
                child = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(child.st_mode)
                    or (child.st_dev, child.st_ino)
                    != self.namespace_identities[index]
                ):
                    raise TransactionError(
                        "transaction namespace identity changed during ESTALE retry"
                    )
            parent = os.fstat(refreshed[-1])
            if parent.st_uid != os.geteuid() or parent.st_mode & 0o022:
                raise TransactionError(
                    "transaction namespace became unsafe during ESTALE retry"
                )
            if self.root_identity is not None:
                refreshed_root = os.open(
                    self.basename,
                    flags,
                    dir_fd=refreshed[-1],
                )
                opened = os.fstat(refreshed_root)
                public = os.stat(
                    self.basename,
                    dir_fd=refreshed[-1],
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or opened.st_uid != os.geteuid()
                    or opened.st_mode & 0o022
                    or (opened.st_dev, opened.st_ino) != self.root_identity
                    or (public.st_dev, public.st_ino) != self.root_identity
                ):
                    raise TransactionError(
                        "transaction root identity changed during ESTALE retry"
                    )
        except BaseException:
            if refreshed_root >= 0:
                try:
                    os.close(refreshed_root)
                except OSError:
                    pass
            for fd in reversed(refreshed):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        previous_root = self.root_fd
        previous_namespace = self.namespace_fds
        self.namespace_fds = refreshed
        self.parent_fd = refreshed[-1]
        self.root_fd = refreshed_root
        if previous_root >= 0:
            try:
                os.close(previous_root)
            except OSError:
                pass
        for fd in reversed(previous_namespace):
            try:
                os.close(fd)
            except OSError:
                pass

    def _retry_estale(
        self,
        operation: Callable[[], _T],
        *,
        refresh_handles: bool = True,
    ) -> _T:
        """Retry one idempotent shared-filesystem operation, and nothing else."""
        for attempt in range(ESTALE_RETRY_ATTEMPTS):
            try:
                return operation()
            except OSError as exc:
                if not _is_estale(exc) or attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                    raise
                if refresh_handles:
                    try:
                        self._refresh_directory_handles()
                    except OSError as refresh_exc:
                        if not _is_estale(refresh_exc):
                            raise
                _estale_backoff(attempt)
        raise AssertionError("unreachable ESTALE retry state")

    def _assert_namespace_identity(self) -> None:
        for index, name in enumerate(self.namespace_names):
            try:
                def snapshot() -> tuple[os.stat_result, os.stat_result]:
                    opened = os.fstat(self.namespace_fds[index + 1])
                    public = os.stat(
                        name,
                        dir_fd=self.namespace_fds[index],
                        follow_symlinks=False,
                    )
                    return opened, public

                opened, public = self._retry_estale(snapshot)
            except OSError as exc:
                raise TransactionError("transaction namespace disappeared or changed") from exc
            if (
                not stat.S_ISDIR(public.st_mode)
                or (opened.st_dev, opened.st_ino) != (public.st_dev, public.st_ino)
            ):
                raise TransactionError("public transaction namespace identity changed")

    def _open_root_with_snapshot(
        self,
        flags: int,
    ) -> tuple[int, os.stat_result]:
        """Open+fstat the root before root_identity exists, without stale FDs."""
        for attempt in range(ESTALE_RETRY_ATTEMPTS):
            candidate = -1
            try:
                candidate = os.open(
                    self.basename,
                    flags,
                    dir_fd=self.parent_fd,
                )
                return candidate, os.fstat(candidate)
            except BaseException as exc:
                if candidate >= 0:
                    try:
                        os.close(candidate)
                    except OSError:
                        pass
                if not _is_estale(exc) or attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                    raise
                # root_identity is intentionally still None, so this refresh
                # replaces only the pinned namespace descriptors.
                try:
                    self._refresh_directory_handles()
                except OSError as refresh_exc:
                    if not _is_estale(refresh_exc):
                        raise
                _estale_backoff(attempt)
        raise AssertionError("unreachable root open retry state")

    def create_or_wait(self, deadline: float) -> None:
        self._assert_namespace_identity()
        if self.rank == 0:
            try:
                os.mkdir(self.basename, mode=0o700, dir_fd=self.parent_fd)
                self._retry_estale(lambda: os.fsync(self.parent_fd))
            except FileExistsError as exc:
                raise DuplicateInvocation("transaction root already exists") from exc
        while True:
            try:
                flags = os.O_RDONLY | os.O_DIRECTORY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                self.root_fd, opened = self._open_root_with_snapshot(flags)
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise TransactionError("timed out waiting for transaction root")
                time.sleep(0.01)
            except OSError as exc:
                raise TransactionError("transaction root is not a real directory") from exc
        if opened.st_uid != os.geteuid() or opened.st_mode & 0o022:
            raise TransactionError("transaction root has unsafe ownership or mode")
        self.root_identity = (opened.st_dev, opened.st_ino)
        self.assert_identity()

    def assert_identity(self) -> None:
        self._assert_namespace_identity()
        if self.root_fd < 0 or self.root_identity is None:
            raise TransactionError("transaction root is not open")
        try:
            def snapshot() -> tuple[os.stat_result, os.stat_result]:
                opened = os.fstat(self.root_fd)
                public = os.stat(
                    self.basename,
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                )
                return opened, public

            opened, public = self._retry_estale(snapshot)
        except OSError as exc:
            raise TransactionError("transaction root disappeared or changed") from exc
        opened_identity = (opened.st_dev, opened.st_ino)
        public_identity = (public.st_dev, public.st_ino)
        if not stat.S_ISDIR(public.st_mode) or opened_identity != self.root_identity:
            raise TransactionError("held transaction-root identity changed")
        if public_identity != self.root_identity:
            raise TransactionError("public transaction-root identity changed")

    def local_filesystem_evidence(self) -> dict[str, Any]:
        self.assert_identity()
        opened, parent = self._retry_estale(
            lambda: (os.fstat(self.root_fd), os.fstat(self.parent_fd))
        )
        return {
            "root_st_dev": opened.st_dev,
            "root_st_ino": opened.st_ino,
            "parent_st_dev": parent.st_dev,
            "parent_st_ino": parent.st_ino,
            "parent_path_sha256": _namespace_parent_sha256(self.parent),
        }

    def open_existing(self) -> None:
        """Open one existing exact transaction root without creating it."""
        self._assert_namespace_identity()
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.root_fd, opened = self._open_root_with_snapshot(flags)
        except OSError as exc:
            raise TransactionError("transaction root is unavailable") from exc
        if opened.st_uid != os.geteuid() or opened.st_mode & 0o022:
            raise TransactionError("transaction root has unsafe ownership or mode")
        self.root_identity = (opened.st_dev, opened.st_ino)
        self.assert_identity()

    def _open_readonly(self, name: str) -> int:
        self.assert_identity()
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = self._retry_estale(
                lambda: os.open(name, flags, dir_fd=self.root_fd)
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise TransactionError(f"unsafe transaction artifact: {name}") from exc
        try:
            info = self._retry_estale(
                lambda: os.fstat(fd),
                refresh_handles=False,
            )
        except BaseException:
            os.close(fd)
            raise
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
        for attempt in range(ESTALE_RETRY_ATTEMPTS):
            fd = -1
            try:
                fd = self._open_readonly(name)
                before = os.fstat(fd)
                blocks: list[bytes] = []
                total = 0
                while True:
                    block = os.read(fd, 65536)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_RECEIPT_BYTES:
                        raise TransactionError(
                            f"transaction artifact too large: {name}"
                        )
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
                    raise TransactionError(
                        f"transaction artifact changed while reading: {name}"
                    )
                return raw
            except OSError as exc:
                if not _is_estale(exc) or attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                    raise
                try:
                    self._refresh_directory_handles()
                except OSError as refresh_exc:
                    if not _is_estale(refresh_exc):
                        raise
                _estale_backoff(attempt)
            finally:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        raise AssertionError("unreachable ESTALE read retry state")

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
            info = self._retry_estale(
                lambda: os.stat(
                    name,
                    dir_fd=self.root_fd,
                    follow_symlinks=False,
                )
            )
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise TransactionError(f"unsafe transaction artifact: {name}")
        return True

    def _write_exclusive_temporary(
        self,
        name: str,
        raw: bytes,
        *,
        final_mode: int = 0o400,
    ) -> tuple[int, int]:
        """Create and durably fill one unique O_EXCL temporary receipt.

        Retried writes use ``pwrite`` at an explicit offset.  Thus an ESTALE
        reply whose server-side write may have completed can only overwrite
        the same bytes at the same offset; it can never duplicate bytes or
        alter the canonical payload.
        """
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = -1
        saw_estale = False
        for attempt in range(ESTALE_RETRY_ATTEMPTS):
            try:
                fd = os.open(name, flags, 0o600, dir_fd=self.root_fd)
                break
            except FileExistsError as exc:
                if saw_estale:
                    raise TransactionError(
                        f"ambiguous exclusive temporary creation after ESTALE: {name}"
                    ) from exc
                raise
            except OSError as exc:
                if not _is_estale(exc) or attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                    raise
                saw_estale = True
                try:
                    self._refresh_directory_handles()
                except OSError as refresh_exc:
                    if not _is_estale(refresh_exc):
                        raise
                _estale_backoff(attempt)
        if fd < 0:
            raise AssertionError("unreachable temporary creation retry state")
        identity: tuple[int, int] | None = None
        try:
            created = _retry_estale_syscall(lambda: os.fstat(fd))
            identity = (created.st_dev, created.st_ino)

            def reopen(
                expected_modes: set[int],
                *,
                writable: bool = True,
            ) -> None:
                nonlocal fd
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = -1
                self._refresh_directory_handles()
                reopen_flags = os.O_RDWR if writable else os.O_RDONLY
                if hasattr(os, "O_NOFOLLOW"):
                    reopen_flags |= os.O_NOFOLLOW
                candidate = os.open(name, reopen_flags, dir_fd=self.root_fd)
                info = os.fstat(candidate)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) not in expected_modes
                    or (info.st_dev, info.st_ino) != identity
                    or info.st_size > len(raw)
                ):
                    os.close(candidate)
                    raise TransactionError(
                        f"temporary transaction artifact changed after ESTALE: {name}"
                    )
                fd = candidate

            def fd_operation(
                operation: Callable[[int], _T],
                expected_modes: set[int],
                *,
                writable: bool = True,
            ) -> _T:
                for attempt in range(ESTALE_RETRY_ATTEMPTS):
                    try:
                        return operation(fd)
                    except OSError as exc:
                        if (
                            not _is_estale(exc)
                            or attempt + 1 == ESTALE_RETRY_ATTEMPTS
                        ):
                            raise
                        for reopen_attempt in range(ESTALE_RETRY_ATTEMPTS):
                            try:
                                reopen(expected_modes, writable=writable)
                                break
                            except OSError as reopen_exc:
                                if (
                                    not _is_estale(reopen_exc)
                                    or reopen_attempt + 1
                                    == ESTALE_RETRY_ATTEMPTS
                                ):
                                    raise
                                _estale_backoff(reopen_attempt)
                        _estale_backoff(attempt)
                raise AssertionError("unreachable temporary fd retry state")

            offset = 0
            while offset < len(raw):
                written = fd_operation(
                    lambda current_fd, offset=offset: os.pwrite(
                        current_fd,
                        raw[offset:],
                        offset,
                    ),
                    {0o600},
                )
                if written <= 0:
                    raise OSError("short transaction artifact write")
                offset += written
            # Flush and verify the exact payload while the unique private file
            # is still owner-writable.  No data write is ever attempted after
            # the immutable mode transition.
            fd_operation(lambda current_fd: os.fsync(current_fd), {0o600})

            observed = bytearray()
            offset = 0
            while offset < len(raw):
                block = fd_operation(
                    lambda current_fd, offset=offset: os.pread(
                        current_fd,
                        min(65536, len(raw) - offset),
                        offset,
                    ),
                    {0o600},
                )
                if not block:
                    break
                observed.extend(block)
                offset += len(block)
            final_info = fd_operation(
                lambda current_fd: os.fstat(current_fd),
                {0o600},
            )
            if (
                bytes(observed) != raw
                or final_info.st_size != len(raw)
                or (final_info.st_dev, final_info.st_ino) != identity
            ):
                raise TransactionError(
                    f"temporary transaction artifact content changed: {name}"
                )
            # fchmod may commit remotely and then report ESTALE.  Reopen the
            # same inode read-only when the exact final mode is already
            # visible; never try O_RDWR against an already-0400 receipt.
            for attempt in range(ESTALE_RETRY_ATTEMPTS):
                try:
                    os.fchmod(fd, final_mode)
                    break
                except OSError as exc:
                    if not _is_estale(exc):
                        raise
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    fd = -1
                    try:
                        self._refresh_directory_handles()
                    except OSError as refresh_exc:
                        if not _is_estale(refresh_exc):
                            raise
                    info = self._retry_estale(
                        lambda: os.stat(
                            name,
                            dir_fd=self.root_fd,
                            follow_symlinks=False,
                        )
                    )
                    observed_mode = stat.S_IMODE(info.st_mode)
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.geteuid()
                        or (info.st_dev, info.st_ino) != identity
                        or info.st_size != len(raw)
                        or observed_mode not in {0o600, final_mode}
                    ):
                        raise TransactionError(
                            f"temporary transaction mode changed after ESTALE: {name}"
                        )
                    if observed_mode == final_mode:
                        reopen({final_mode}, writable=False)
                        break
                    if attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                        raise
                    reopen({0o600}, writable=True)
                    _estale_backoff(attempt)
            else:
                raise AssertionError("unreachable temporary chmod retry state")

            fd_operation(
                lambda current_fd: os.fsync(current_fd),
                {final_mode},
                writable=False,
            )
            immutable = fd_operation(
                lambda current_fd: os.fstat(current_fd),
                {final_mode},
                writable=False,
            )
            if (
                not stat.S_ISREG(immutable.st_mode)
                or immutable.st_uid != os.geteuid()
                or stat.S_IMODE(immutable.st_mode) != final_mode
                or (immutable.st_dev, immutable.st_ino) != identity
                or immutable.st_size != len(raw)
            ):
                raise TransactionError(
                    f"temporary transaction artifact is not immutable: {name}"
                )
            return identity
        except BaseException:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = -1
            if identity is not None:
                try:
                    self._unlink_exact(name, identity)
                except OSError:
                    pass
            raise
        finally:
            if fd >= 0:
                os.close(fd)

    def _unlink_exact(
        self,
        name: str,
        expected_identity: tuple[int, int],
    ) -> None:
        """Idempotently unlink one exact entry with bounded ESTALE recovery."""
        for attempt in range(ESTALE_RETRY_ATTEMPTS):
            try:
                info = os.stat(
                    name,
                    dir_fd=self.root_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or (info.st_dev, info.st_ino) != expected_identity
                ):
                    raise TransactionError(
                        f"transaction temporary identity changed: {name}"
                    )
                os.unlink(name, dir_fd=self.root_fd)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                if not _is_estale(exc) or attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                    raise
                try:
                    self._refresh_directory_handles()
                except OSError as refresh_exc:
                    if not _is_estale(refresh_exc):
                        raise
                _estale_backoff(attempt)

    def _published_hardlink_is_exact(
        self,
        temporary: str,
        name: str,
        identity: tuple[int, int],
        raw: bytes,
    ) -> bool:
        """Prove an ESTALE-ambiguous link committed this exact receipt."""
        try:
            temporary_info, target_info = self._retry_estale(
                lambda: (
                    os.stat(temporary, dir_fd=self.root_fd, follow_symlinks=False),
                    os.stat(name, dir_fd=self.root_fd, follow_symlinks=False),
                )
            )
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(temporary_info.st_mode)
            or not stat.S_ISREG(target_info.st_mode)
            or temporary_info.st_uid != os.geteuid()
            or target_info.st_uid != os.geteuid()
            or stat.S_IMODE(temporary_info.st_mode) != 0o400
            or stat.S_IMODE(target_info.st_mode) != 0o400
            or (temporary_info.st_dev, temporary_info.st_ino) != identity
            or (target_info.st_dev, target_info.st_ino) != identity
            or temporary_info.st_size != len(raw)
            or target_info.st_size != len(raw)
        ):
            return False
        return self.read_bytes(name) == raw

    def _link_immutable_exact(
        self,
        temporary: str,
        name: str,
        identity: tuple[int, int],
        raw: bytes,
    ) -> None:
        """Atomically publish, resolving only a provably committed ESTALE."""
        uncertain_commit = False
        for attempt in range(ESTALE_RETRY_ATTEMPTS):
            try:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=self.root_fd,
                    dst_dir_fd=self.root_fd,
                    follow_symlinks=False,
                )
                return
            except FileExistsError as exc:
                if uncertain_commit and self._published_hardlink_is_exact(
                    temporary, name, identity, raw
                ):
                    return
                raise DuplicateInvocation(
                    f"immutable transaction artifact exists: {name}"
                ) from exc
            except OSError as exc:
                if not _is_estale(exc):
                    raise
                uncertain_commit = True
                try:
                    self._refresh_directory_handles()
                except OSError as refresh_exc:
                    if not _is_estale(refresh_exc):
                        raise
                if self._published_hardlink_is_exact(
                    temporary, name, identity, raw
                ):
                    return
                if self.exists(name):
                    raise DuplicateInvocation(
                        f"immutable transaction artifact exists: {name}"
                    )
                if attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                    raise
                _estale_backoff(attempt)

    def publish_immutable(self, name: str, payload: Mapping[str, Any]) -> str:
        """Publish a complete immutable file atomically without overwriting."""
        self.assert_identity()
        raw = _canonical_json_bytes(payload)
        temporary = f".tmp.{name}.{os.getpid()}.{uuid.uuid4().hex}"
        temporary_exists = True
        identity: tuple[int, int] | None = None
        try:
            identity = self._write_exclusive_temporary(temporary, raw)
            self._link_immutable_exact(temporary, name, identity, raw)
            self._unlink_exact(temporary, identity)
            temporary_exists = False
            self._retry_estale(lambda: os.fsync(self.root_fd))
        except FileExistsError as exc:
            raise DuplicateInvocation(f"immutable transaction artifact exists: {name}") from exc
        except BaseException:
            # The no-replace hard link is the irreversible publication commit
            # point.  A peer may already have observed and validated the exact
            # immutable receipt, so a later temporary-alias cleanup or
            # directory-fsync failure must never roll the target back.
            raise
        finally:
            if temporary_exists and identity is not None:
                try:
                    self._unlink_exact(temporary, identity)
                    self._retry_estale(lambda: os.fsync(self.root_fd))
                except FileNotFoundError:
                    pass
        return _sha256_bytes(raw)

    def publish_terminal_outcome(
        self,
        payload: Mapping[str, Any],
    ) -> str:
        """Publish success with one final atomic namespace mutation.

        The exclusive finalizer lock prevents a cooperating loser from
        creating or cleaning a temporary entry after ``OUTCOME.json`` becomes
        visible.  Native no-replace rename both removes the temporary name and
        creates the terminal name in the same final namespace operation.  On
        EFS, the atomic hard-link fallback may retain its unique read-only
        temporary alias when alias cleanup is unavailable; the terminal target
        is nevertheless monotonic and is never rolled back after publication.
        """
        name = "OUTCOME.json"
        self.assert_identity()
        if self.exists(name):
            raise DuplicateInvocation("immutable transaction artifact exists: OUTCOME.json")
        raw = _canonical_json_bytes(payload)
        temporary = f".tmp.{name}.{os.getpid()}.{uuid.uuid4().hex}"
        temporary_exists = True
        identity: tuple[int, int] | None = None
        try:
            identity = self._write_exclusive_temporary(temporary, raw)
            uncertain_commit = False
            for attempt in range(ESTALE_RETRY_ATTEMPTS):
                try:
                    _rename_noreplace(
                        self.root_fd,
                        temporary,
                        self.root_fd,
                        name,
                        expected_source_identity=identity,
                    )
                    temporary_exists = self.exists(temporary)
                    break
                except FileExistsError as exc:
                    if uncertain_commit and (
                        self._replacement_is_exact(
                            temporary, name, identity, raw, 0o400
                        )
                        or self._published_hardlink_is_exact(
                            temporary, name, identity, raw
                        )
                    ):
                        temporary_exists = self.exists(temporary)
                        break
                    raise DuplicateInvocation(
                        "immutable transaction artifact exists: OUTCOME.json"
                    ) from exc
                except OSError as exc:
                    if not _is_estale(exc):
                        raise
                    uncertain_commit = True
                    try:
                        self._refresh_directory_handles()
                    except OSError as refresh_exc:
                        if not _is_estale(refresh_exc):
                            raise
                    if self._replacement_is_exact(
                        temporary, name, identity, raw, 0o400
                    ):
                        temporary_exists = False
                        break
                    if self._published_hardlink_is_exact(
                        temporary, name, identity, raw
                    ):
                        temporary_exists = True
                        break
                    if self.exists(name):
                        raise DuplicateInvocation(
                            "immutable transaction artifact exists: OUTCOME.json"
                        )
                    if not self.exists(temporary):
                        raise TransactionError(
                            "ambiguous terminal outcome publication after ESTALE"
                        )
                    if attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                        raise
                    _estale_backoff(attempt)
            else:
                raise AssertionError("unreachable terminal publication retry state")
            if temporary_exists:
                self._unlink_exact(temporary, identity)
                temporary_exists = False
            # This is a durability barrier, not a namespace/content mutation.
            self._retry_estale(lambda: os.fsync(self.root_fd))
        except FileExistsError as exc:
            raise DuplicateInvocation(
                "immutable transaction artifact exists: OUTCOME.json"
            ) from exc
        except BaseException:
            # Once publication succeeds, OUTCOME is an irreversible commit
            # point.  In particular, a directory-fsync error must not make a
            # receipt that a concurrent replay may already have observed
            # disappear.  The next finalizer invocation can validate and
            # replay the preserved terminal outcome.
            raise
        finally:
            if temporary_exists and identity is not None:
                try:
                    self._unlink_exact(temporary, identity)
                    self._retry_estale(lambda: os.fsync(self.root_fd))
                except FileNotFoundError:
                    pass
        return _sha256_bytes(raw)

    def _replacement_is_exact(
        self,
        temporary: str,
        name: str,
        identity: tuple[int, int],
        raw: bytes,
        mode: int,
    ) -> bool:
        """Prove an ESTALE-ambiguous rename/replace committed exactly once."""
        try:
            target = self._retry_estale(
                lambda: os.stat(
                    name,
                    dir_fd=self.root_fd,
                    follow_symlinks=False,
                )
            )
        except FileNotFoundError:
            return False
        try:
            temporary_exists = self.exists(temporary)
        except FileNotFoundError:
            temporary_exists = False
        if (
            temporary_exists
            or not stat.S_ISREG(target.st_mode)
            or target.st_uid != os.geteuid()
            or stat.S_IMODE(target.st_mode) != mode
            or (target.st_dev, target.st_ino) != identity
            or target.st_size != len(raw)
        ):
            return False
        return self.read_bytes(name) == raw

    def replace_heartbeat(self, name: str, payload: Mapping[str, Any]) -> None:
        self.assert_identity()
        if self.exists(name):
            # ``exists`` rejects a symlink or non-regular replacement target.
            pass
        raw = _canonical_json_bytes(payload)
        temporary = f".tmp.{name}.{os.getpid()}.{uuid.uuid4().hex}"
        temporary_exists = True
        identity: tuple[int, int] | None = None
        try:
            identity = self._write_exclusive_temporary(
                temporary,
                raw,
                final_mode=0o600,
            )
            for attempt in range(ESTALE_RETRY_ATTEMPTS):
                try:
                    os.replace(
                        temporary,
                        name,
                        src_dir_fd=self.root_fd,
                        dst_dir_fd=self.root_fd,
                    )
                    temporary_exists = False
                    break
                except OSError as exc:
                    if not _is_estale(exc):
                        raise
                    try:
                        self._refresh_directory_handles()
                    except OSError as refresh_exc:
                        if not _is_estale(refresh_exc):
                            raise
                    if self._replacement_is_exact(
                        temporary,
                        name,
                        identity,
                        raw,
                        0o600,
                    ):
                        temporary_exists = False
                        break
                    if not self.exists(temporary):
                        raise TransactionError(
                            f"ambiguous heartbeat replacement after ESTALE: {name}"
                        )
                    if attempt + 1 == ESTALE_RETRY_ATTEMPTS:
                        raise
                    _estale_backoff(attempt)
            else:
                raise AssertionError("unreachable heartbeat replace retry state")
            self._retry_estale(lambda: os.fsync(self.root_fd))
        finally:
            if temporary_exists and identity is not None:
                self._unlink_exact(temporary, identity)

    def close(self) -> None:
        if self.root_fd >= 0:
            try:
                os.close(self.root_fd)
            except OSError:
                pass
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


def _process_table() -> dict[int, dict[str, Any]]:
    """Snapshot full-argv process identities for exact ancestry discovery."""
    table: dict[int, dict[str, Any]] = {}
    if Path("/proc").is_dir():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                identity = _proc_identity(int(entry.name))
            except TransactionError:
                continue
            table[int(entry.name)] = identity
        return table
    if not _CPU_TEST_MODE:
        raise TransactionError("exact descendant census requires Linux /proc")
    try:
        output = subprocess.check_output(
            [
                "/bin/ps",
                "-axo",
                "pid=,ppid=,pgid=,sess=,lstart=,command=",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransactionError("cannot inspect exact test process ancestry") from exc
    for line in output.splitlines():
        fields = line.split(maxsplit=9)
        if (
            len(fields) != 10
            or not all(item.isdigit() for item in fields[:4])
        ):
            continue
        pid = int(fields[0])
        table[pid] = {
            "pid": pid,
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "starttime_ticks": int(
                hashlib.sha256(" ".join(fields[4:9]).encode()).hexdigest()[:15],
                16,
            ),
            "argv_sha256": _sha256_bytes(fields[9].encode()),
        }
    return table


def _same_process_generation(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    return (
        expected.get("pid") == observed.get("pid")
        and expected.get("starttime_ticks") == observed.get("starttime_ticks")
    )


class _DescendantTracker:
    """Track only the exact workload subtree adopted by this supervisor."""

    def __init__(self, supervisor_pid: int, workload_pid: int, anchor_pid: int):
        self.supervisor_pid = supervisor_pid
        self.workload_pid = workload_pid
        self.anchor_pid = anchor_pid
        self.records: dict[tuple[int, int], dict[str, Any]] = {}

    def _remember(self, identity: Mapping[str, Any]) -> None:
        key = (int(identity["pid"]), int(identity["starttime_ticks"]))
        record = self.records.setdefault(
            key,
            {
                "pid": key[0],
                "starttime_ticks": key[1],
                "ppids": [],
                "pgids": [],
                "sids": [],
                "argv_sha256": [],
            },
        )
        for source, target in (
            ("ppid", "ppids"),
            ("pgid", "pgids"),
            ("sid", "sids"),
            ("argv_sha256", "argv_sha256"),
        ):
            value = identity[source]
            if value not in record[target]:
                record[target].append(value)

    def discover(self) -> None:
        table = _process_table()
        known_pids: set[int] = set()
        workload = table.get(self.workload_pid)
        if workload is not None:
            existing = [
                record
                for record in self.records.values()
                if record["pid"] == self.workload_pid
            ]
            if not existing or any(
                record["starttime_ticks"] == workload["starttime_ticks"]
                for record in existing
            ):
                self._remember(workload)
                known_pids.add(self.workload_pid)
        for record in self.records.values():
            observed = table.get(int(record["pid"]))
            if observed is not None and _same_process_generation(record, observed):
                self._remember(observed)
                known_pids.add(int(record["pid"]))

        changed = True
        while changed:
            changed = False
            for pid, identity in table.items():
                if pid in {self.supervisor_pid, self.anchor_pid}:
                    continue
                adopted = identity["ppid"] == self.supervisor_pid
                descended = identity["ppid"] in known_pids
                if (adopted or descended) and pid not in known_pids:
                    self._remember(identity)
                    known_pids.add(pid)
                    changed = True

    def _live(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        self.discover()
        table = _process_table()
        live: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for record in self.records.values():
            observed = table.get(int(record["pid"]))
            if observed is not None and _same_process_generation(record, observed):
                self._remember(observed)
                live.append((record, observed))
        return live

    def signal_live(self, signum: int) -> None:
        for record, _observed in self._live():
            pid = int(record["pid"])
            pidfd = -1
            if sys.platform == "linux" and (
                not hasattr(os, "pidfd_open")
                or not hasattr(signal, "pidfd_send_signal")
            ):
                raise TransactionError(
                    "formal exact descendant signalling requires pidfd"
                )
            try:
                if sys.platform == "linux":
                    pidfd = os.pidfd_open(pid, 0)
                try:
                    current = _proc_identity(pid)
                except TransactionError:
                    continue
                # A changed starttime is PID reuse and is never signalled.
                # Re-snapshot full argv after opening pidfd so the signal is
                # tied to the same exact process generation.
                if not _same_process_generation(record, current):
                    continue
                self._remember(current)
                if current["argv_sha256"] not in record["argv_sha256"]:
                    raise TransactionError("descendant argv changed outside census")
                if pidfd >= 0:
                    signal.pidfd_send_signal(pidfd, signum)
                else:
                    os.kill(pid, signum)
            except ProcessLookupError:
                pass
            finally:
                if pidfd >= 0:
                    os.close(pidfd)

    def live_identities(self) -> list[dict[str, Any]]:
        return [dict(observed) for _record, observed in self._live()]

    def receipt(self, empty_censuses: int) -> dict[str, Any]:
        records = sorted(
            (
                {
                    **record,
                    "ppids": sorted(record["ppids"]),
                    "pgids": sorted(record["pgids"]),
                    "sids": sorted(record["sids"]),
                }
                for record in self.records.values()
            ),
            key=lambda item: (item["pid"], item["starttime_ticks"]),
        )
        return {
            "schema": SCHEMA,
            "status": "DESCENDANTS_CLEAN",
            "supervisor_pid": self.supervisor_pid,
            "workload_pid": self.workload_pid,
            "anchor_pid": self.anchor_pid,
            "identity_basis": "pid+starttime_ticks+full_argv_sha256",
            "required_consecutive_empty_censuses": 3,
            "observed_consecutive_empty_censuses": empty_censuses,
            "tracked": records,
            "live_after": [],
        }


def _write_supervisor_event(fd: int, payload: Mapping[str, Any]) -> None:
    raw = _canonical_json_bytes(payload)
    view = memoryview(raw)
    while view:
        view = view[os.write(fd, view) :]


def _prepare_pass_fds(
    workload_input_fds: Mapping[str, int],
    input_bindings: Mapping[str, Mapping[str, Any]],
    forbidden_fds: set[int],
) -> dict[str, int]:
    """Duplicate every authority input onto its deterministic passed FD."""
    passed: dict[str, int] = {}
    temporary = {
        logical_id: os.dup(fd)
        for logical_id, fd in workload_input_fds.items()
    }
    try:
        for logical_id in sorted(temporary):
            target = int(input_bindings[logical_id]["passed_fd"])
            if target in forbidden_fds:
                raise TransactionError("pinned input FD collides with supervisor control")
            os.dup2(temporary[logical_id], target, inheritable=True)
            if _sha256_fd(target) != input_bindings[logical_id]["sha256"]:
                raise TransactionError(f"passed input FD changed: {logical_id}")
            passed[logical_id] = target
    finally:
        for fd in temporary.values():
            os.close(fd)
    return passed


def _verify_child_passed_fds(
    child_pid: int,
    passed_fds: Mapping[str, int],
    input_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Verify the started child's exact inherited inode through Linux /proc."""
    evidence: dict[str, dict[str, Any]] = {}
    for logical_id in sorted(passed_fds):
        parent_fd = passed_fds[logical_id]
        parent_info = os.fstat(parent_fd)
        if sys.platform == "linux":
            child_path = Path(f"/proc/{child_pid}/fd/{parent_fd}")
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                # proc fd entries are magic links and intentionally must be
                # followed, so do not add O_NOFOLLOW here.
                pass
            try:
                child_fd = os.open(child_path, flags)
            except OSError as exc:
                raise TransactionError(
                    f"workload child did not inherit pinned FD: {logical_id}"
                ) from exc
            try:
                child_info = os.fstat(child_fd)
                child_sha = _sha256_fd(child_fd)
            finally:
                os.close(child_fd)
            if (
                (child_info.st_dev, child_info.st_ino)
                != (parent_info.st_dev, parent_info.st_ino)
                or child_sha != input_bindings[logical_id]["sha256"]
            ):
                raise TransactionError(
                    f"workload child inherited the wrong input: {logical_id}"
                )
            verification = "linux_proc_child_fd"
        elif _CPU_TEST_MODE:
            child_info = parent_info
            child_sha = _sha256_fd(parent_fd)
            verification = "cpu_test_fork_inheritance"
        else:  # pragma: no cover - formal runs are Linux-only.
            raise TransactionError("formal child FD verification requires Linux /proc")
        evidence[logical_id] = {
            "passed_fd": parent_fd,
            "st_dev": child_info.st_dev,
            "st_ino": child_info.st_ino,
            "sha256": child_sha,
            "verification": verification,
        }
    return evidence


def _wait_status_returncode(status_value: int) -> int:
    if os.WIFEXITED(status_value):
        return os.WEXITSTATUS(status_value)
    if os.WIFSIGNALED(status_value):
        return -os.WTERMSIG(status_value)
    return 255


def _signal_exact_anchored_group(
    expected_anchor: Mapping[str, Any],
    signum: int,
) -> bool:
    """Signal a process group only while its original anchor is still exact.

    A numeric PGID is not an identity: after the anchor is reaped Linux may
    recycle that number for an unrelated process group.  Re-snapshot the full
    anchor identity immediately before every ``killpg`` and never signal when
    the anchor has disappeared or changed generation.  The caller can still
    terminate previously-censused descendants by pidfd.
    """
    anchor_pid = expected_anchor.get("pid")
    if not _exact_int(anchor_pid, minimum=1):
        raise TransactionError("invalid workgroup anchor identity")
    try:
        observed_anchor = _proc_identity(int(anchor_pid))
    except TransactionError:
        return False
    if any(
        observed_anchor.get(key) != expected_anchor.get(key)
        for key in IDENTITY_KEYS
    ):
        raise TransactionError("workgroup anchor generation changed before signal")
    if (
        observed_anchor["pid"] != observed_anchor["pgid"]
        or observed_anchor["pid"] != int(anchor_pid)
    ):
        raise TransactionError("workgroup anchor no longer owns its process group")
    try:
        os.killpg(int(anchor_pid), signum)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise TransactionError(
            "permission denied signalling exact workgroup anchor "
            f"pid={anchor_pid} signal={signum} errno={exc.errno}"
        ) from exc
    return True


def _signal_exact_process_generation(
    expected: Mapping[str, Any],
    signum: int,
) -> bool:
    """Signal one exact process generation, using pidfd on formal Linux."""
    pid = expected.get("pid")
    if not _exact_int(pid, minimum=1):
        raise TransactionError("invalid exact process identity")
    pidfd = -1
    if sys.platform == "linux" and (
        not hasattr(os, "pidfd_open")
        or not hasattr(signal, "pidfd_send_signal")
    ):
        raise TransactionError("formal exact process signalling requires pidfd")
    try:
        if sys.platform == "linux":
            try:
                pidfd = os.pidfd_open(int(pid), 0)
            except ProcessLookupError:
                return False
        try:
            observed = _proc_identity(int(pid))
        except TransactionError:
            return False
        if any(observed.get(key) != expected.get(key) for key in IDENTITY_KEYS):
            raise TransactionError("exact process generation changed before signal")
        if pidfd >= 0:
            signal.pidfd_send_signal(pidfd, signum)
        elif _CPU_TEST_MODE:
            os.kill(int(pid), signum)
        else:  # pragma: no cover - formal hosts are Linux.
            raise TransactionError("formal exact process signalling requires Linux")
    except ProcessLookupError:
        return False
    finally:
        if pidfd >= 0:
            os.close(pidfd)
    return True


def _direct_child_pids(parent_pid: int) -> set[int]:
    """Return the supervisor's exact direct children, including zombies."""
    children_path = Path(f"/proc/{parent_pid}/task/{parent_pid}/children")
    if children_path.is_file():
        try:
            fields = children_path.read_text(encoding="ascii").split()
        except OSError as exc:
            raise TransactionError("cannot inspect exact supervisor children") from exc
        if not all(field.isdigit() for field in fields):
            raise TransactionError("invalid exact supervisor children record")
        return {int(field) for field in fields}
    if not _CPU_TEST_MODE:
        raise TransactionError("exact child inspection requires Linux /proc")
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-axo", "pid=,ppid="],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransactionError("cannot inspect exact test children") from exc
    children: set[int] = set()
    for line in output.splitlines():
        fields = line.split()
        if (
            len(fields) == 2
            and fields[0].isdigit()
            and fields[1].isdigit()
            and int(fields[1]) == parent_pid
        ):
            children.add(int(fields[0]))
    return children


def _reap_non_anchor_children(supervisor_pid: int, anchor_pid: int) -> None:
    """Reap exact direct children without ever accidentally reaping anchor."""
    for pid in sorted(_direct_child_pids(supervisor_pid) - {anchor_pid}):
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def _reap_owned_anchor(
    anchor_identity: Mapping[str, Any],
    deadline: float,
) -> None:
    """Reap the owner's exact anchor only after all group signalling is over."""
    anchor_pid = int(anchor_identity["pid"])
    first_attempt = True
    while first_attempt or time.monotonic() < deadline:
        first_attempt = False
        try:
            observed = _proc_identity(anchor_pid)
        except TransactionError:
            observed = None
        if observed is not None and not _same_process_generation(
            anchor_identity,
            observed,
        ):
            raise TransactionError("workgroup anchor generation changed before reap")
        try:
            waited, _ = os.waitpid(anchor_pid, os.WNOHANG)
        except ChildProcessError:
            return
        if waited == anchor_pid:
            return
        time.sleep(0.01)
    raise TransactionError("exact workgroup anchor did not exit for reap")


def _supervisor_cleanup_group(
    anchor_identity: Mapping[str, Any],
    anchor_write_fd: int,
    tracker: _DescendantTracker | None,
    grace_seconds: float,
) -> dict[str, Any]:
    """Clean the workgroup while the sole killpg owner pins its anchor.

    Group-signal failures are evidence failures, not control-flow shortcuts:
    exact pidfd descendant termination, pipe-driven anchor shutdown, census,
    and reaping all still run before the errors are aggregated.
    """
    anchor_pid = int(anchor_identity["pid"])
    supervisor_pid = os.getpid()
    cleanup_errors: list[str] = []

    def record_error(label: str, exc: BaseException) -> None:
        cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")

    try:
        _signal_exact_anchored_group(anchor_identity, signal.SIGTERM)
    except BaseException as exc:
        record_error("group SIGTERM", exc)
    finally:
        if tracker is not None:
            try:
                tracker.signal_live(signal.SIGTERM)
            except BaseException as exc:
                record_error("tracker SIGTERM", exc)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        members = _group_members(anchor_pid)
        live = [] if tracker is None else tracker.live_identities()
        if members <= {anchor_pid} and not live:
            break
        time.sleep(0.01)

    # The supervisor is the anchor's only parent and has not reaped it.  Thus
    # the PID/PGID cannot be recycled across the final identity snapshot and
    # killpg.  No code issues a group signal after this block.
    try:
        _signal_exact_anchored_group(anchor_identity, signal.SIGKILL)
    except BaseException as exc:
        record_error("group SIGKILL", exc)
    finally:
        if tracker is not None:
            try:
                tracker.signal_live(signal.SIGKILL)
            except BaseException as exc:
                record_error("tracker SIGKILL", exc)
        try:
            os.close(anchor_write_fd)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                record_error("anchor pipe close", exc)

    reap_deadline = time.monotonic() + max(grace_seconds, 0.5)
    empty_censuses = 0
    proof: dict[str, Any] | None = None
    while time.monotonic() < reap_deadline:
        try:
            _reap_non_anchor_children(supervisor_pid, anchor_pid)
        except BaseException as exc:
            record_error("non-anchor reap", exc)
        live = [] if tracker is None else tracker.live_identities()
        members = _group_members(anchor_pid)
        if live:
            empty_censuses = 0
            if tracker is not None:
                try:
                    tracker.signal_live(signal.SIGKILL)
                except BaseException as exc:
                    record_error("tracker SIGKILL retry", exc)
        elif members <= {anchor_pid}:
            empty_censuses += 1
            if empty_censuses >= 3:
                if tracker is None:
                    proof = {
                        "schema": SCHEMA,
                        "status": "DESCENDANTS_CLEAN",
                        "identity_basis": "no-workload-started",
                        "required_consecutive_empty_censuses": 3,
                        "observed_consecutive_empty_censuses": empty_censuses,
                        "tracked": [],
                        "live_after": [],
                    }
                else:
                    proof = tracker.receipt(empty_censuses)
                break
        else:
            # The anchor remains owned and unreaped, but group signalling is
            # finished.  Exact tracker pidfds are the only permitted retry.
            empty_censuses = 0
            if tracker is not None:
                try:
                    tracker.signal_live(signal.SIGKILL)
                except BaseException as exc:
                    record_error("tracker group-member retry", exc)
        time.sleep(0.02)

    if proof is None:
        record_error(
            "descendant census",
            TransactionError("cannot prove exact workload descendants are gone"),
        )
    try:
        _reap_owned_anchor(anchor_identity, reap_deadline)
    except BaseException as exc:
        record_error("anchor reap", exc)
    if cleanup_errors:
        raise TransactionError("; ".join(cleanup_errors))
    if proof is None:
        raise AssertionError("missing cleanup proof without cleanup error")
    return proof


def _supervisor_main(
    parent_pid: int,
    control_fd: int,
    event_fd: int,
    executable_fd: int,
    workdir_fd: int,
    workload: Sequence[str],
    exec_workload: Sequence[str],
    workload_input_fds: Mapping[str, int],
    runtime_fds: Mapping[str, int],
    input_bindings: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, str],
    expected_argv_sha256: str,
    expected_exec_argv_sha256: str,
    executable_sha256: str,
    shutdown_grace_seconds: float,
    runtime_monitor: _FormalRuntimeMonitor | None,
) -> None:
    anchor_pid = -1
    anchor_identity: dict[str, Any] | None = None
    anchor_write = -1
    cleanup_attempted = False
    abort_requested = False
    abort_signal_errors: list[str] = []
    tracker: _DescendantTracker | None = None
    passed_fds: dict[str, int] = {}

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
        anchor_identity = _proc_identity(anchor_pid)
        if (
            anchor_identity["ppid"] != os.getpid()
            or anchor_identity["pid"] != anchor_identity["pgid"]
            or anchor_identity["sid"] != os.getsid(0)
        ):
            raise TransactionError("workgroup anchor ancestry mismatch")
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
            cleanup_attempted = True
            try:
                cleanup = _supervisor_cleanup_group(
                    anchor_identity,
                    anchor_write,
                    None,
                    shutdown_grace_seconds,
                )
            finally:
                anchor_write = -1
            _write_supervisor_event(
                event_fd,
                {
                    "event": "RESULT",
                    "returncode": None,
                    "aborted": True,
                    "cleanup": cleanup,
                },
            )
            os._exit(0)

        if runtime_monitor is not None:
            runtime_monitor.assert_quiet("supervisor before workload fork")

        passed_fds = _prepare_pass_fds(
            workload_input_fds,
            input_bindings,
            {
                control_fd,
                event_fd,
                executable_fd,
                workdir_fd,
                anchor_write,
                *runtime_fds.values(),
                *(
                    [runtime_monitor.fd]
                    if runtime_monitor is not None and runtime_monitor.fd >= 0
                    else []
                ),
            },
        )
        child_ready_read, child_ready_write = os.pipe()
        exec_gate_read, exec_gate_write = os.pipe()
        workload_pid = os.fork()
        if workload_pid == 0:
            try:
                os.close(child_ready_read)
                os.close(exec_gate_write)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.setpgid(0, anchor_pid)
                os.write(child_ready_write, b"1")
                os.close(child_ready_write)
                if os.read(exec_gate_read, 1) != b"G":
                    raise TransactionError("pinned-input exec gate was not released")
                os.close(exec_gate_read)
                os.fchdir(workdir_fd)
                if os.execve in os.supports_fd:
                    os.execve(
                        executable_fd,
                        list(exec_workload),
                        dict(environment),
                    )
                if _CPU_TEST_MODE:
                    os.execve(
                        workload[0],
                        list(exec_workload),
                        dict(environment),
                    )
                raise TransactionError("platform cannot exec the pinned workload fd")
            except BaseException as exc:
                try:
                    _write_supervisor_event(
                        event_fd,
                        {"event": "EXEC_ERROR", "reason": f"{type(exc).__name__}: {exc}"},
                    )
                finally:
                    os._exit(127)

        os.close(child_ready_write)
        os.close(exec_gate_read)
        if os.read(child_ready_read, 1) != b"1":
            raise TransactionError("workload child failed before FD verification")
        os.close(child_ready_read)
        preexec_identity = _proc_identity(workload_pid)
        if (
            preexec_identity["ppid"] != os.getpid()
            or preexec_identity["pgid"] != anchor_pid
            or preexec_identity["sid"] != os.getsid(0)
        ):
            raise TransactionError("workload pre-exec ancestry mismatch")
        inherited_inputs = _verify_child_passed_fds(
            workload_pid,
            passed_fds,
            input_bindings,
        )
        tracker = _DescendantTracker(os.getpid(), workload_pid, anchor_pid)
        tracker.discover()
        if runtime_monitor is not None:
            runtime_monitor.assert_quiet("supervisor before workload exec")
        os.write(exec_gate_write, b"G")
        os.close(exec_gate_write)
        if sys.platform == "linux":
            exec_deadline = time.monotonic() + max(shutdown_grace_seconds, 1.0)
            while True:
                actual_identity = _proc_identity(workload_pid)
                if actual_identity["argv_sha256"] == expected_exec_argv_sha256:
                    break
                if time.monotonic() >= exec_deadline:
                    raise TransactionError("cannot prove actual workload exec argv")
                time.sleep(0.005)
        elif _CPU_TEST_MODE:
            actual_identity = dict(preexec_identity)
            actual_identity["argv_sha256"] = expected_exec_argv_sha256
        else:  # pragma: no cover
            raise TransactionError("formal workload argv proof requires Linux")
        actual_identity.update(
            {
                "preexec_argv_sha256": preexec_identity["argv_sha256"],
                "command_argv_sha256": expected_argv_sha256,
                "exec_argv_sha256": expected_exec_argv_sha256,
                "executable_sha256": executable_sha256,
                "passed_input_fds": inherited_inputs,
            }
        )
        tracker._remember(actual_identity)
        _write_supervisor_event(
            event_fd, {"event": "STARTED", "workload": actual_identity}
        )

        status_value: int | None = None
        abort_deadline: float | None = None
        abort_kill_sent = False
        while status_value is None:
            if runtime_monitor is not None:
                runtime_monitor.assert_quiet("formal workload runtime")
            waited, observed = os.waitpid(workload_pid, os.WNOHANG)
            if waited == workload_pid:
                status_value = observed
                break
            tracker.discover()
            readable, _, _ = select.select([control_fd], [], [], 0)
            if readable:
                try:
                    followup = os.read(control_fd, 1)
                except BlockingIOError:
                    followup = None
                if followup in {b"", b"A"}:
                    abort_requested = True
                elif followup not in {None, b"G"}:
                    raise TransactionError("invalid post-GO supervisor command")
            if abort_requested or os.getppid() != parent_pid:
                if abort_deadline is None:
                    try:
                        _signal_exact_anchored_group(
                            anchor_identity,
                            signal.SIGTERM,
                        )
                    except BaseException as signal_exc:
                        abort_signal_errors.append(
                            "abort group SIGTERM: "
                            f"{type(signal_exc).__name__}: {signal_exc}"
                        )
                    finally:
                        try:
                            tracker.signal_live(signal.SIGTERM)
                        except BaseException as tracker_exc:
                            abort_signal_errors.append(
                                "abort tracker SIGTERM: "
                                f"{type(tracker_exc).__name__}: {tracker_exc}"
                            )
                    abort_deadline = time.monotonic() + shutdown_grace_seconds
                elif time.monotonic() >= abort_deadline and not abort_kill_sent:
                    try:
                        _signal_exact_anchored_group(
                            anchor_identity,
                            signal.SIGKILL,
                        )
                    except BaseException as signal_exc:
                        abort_signal_errors.append(
                            "abort group SIGKILL: "
                            f"{type(signal_exc).__name__}: {signal_exc}"
                        )
                    finally:
                        try:
                            tracker.signal_live(signal.SIGKILL)
                        except BaseException as tracker_exc:
                            abort_signal_errors.append(
                                "abort tracker SIGKILL: "
                                f"{type(tracker_exc).__name__}: {tracker_exc}"
                            )
                        abort_kill_sent = True
            time.sleep(0.01)
        returncode = _wait_status_returncode(status_value)
        cleanup_attempted = True
        try:
            cleanup = _supervisor_cleanup_group(
                anchor_identity,
                anchor_write,
                tracker,
                shutdown_grace_seconds,
            )
        finally:
            anchor_write = -1
        if abort_signal_errors:
            raise TransactionError("; ".join(abort_signal_errors))
        _write_supervisor_event(
            event_fd,
            {
                "event": "RESULT",
                "returncode": returncode,
                "aborted": abort_requested,
                "cleanup": cleanup,
            },
        )
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        if anchor_identity is not None and not cleanup_attempted:
            try:
                _supervisor_cleanup_group(
                    anchor_identity,
                    anchor_write,
                    tracker,
                    shutdown_grace_seconds,
                )
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
            finally:
                anchor_write = -1
        elif anchor_write >= 0:
            try:
                os.close(anchor_write)
            except OSError as cleanup_exc:
                cleanup_error = cleanup_exc
            anchor_write = -1
        try:
            _write_supervisor_event(
                event_fd,
                {
                    "event": "SUPERVISOR_ERROR",
                    "reason": (
                        f"{type(exc).__name__}: {exc}"
                        + (
                            " | cleanup: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                            if cleanup_error is not None
                            else ""
                        )
                    ),
                },
            )
        except BaseException:
            pass
    finally:
        for fd in passed_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        os._exit(0)


class Coordinator:
    def __init__(
        self,
        args: argparse.Namespace,
        workload: Sequence[str],
        exec_workload: Sequence[str],
        runner: Mapping[str, Any],
        portable: Mapping[str, Any],
        tx: TransactionDirectory,
        workload_environment: Mapping[str, str],
        executable_fd: int,
        workdir_fd: int,
        workload_input_fds: Mapping[str, int],
        runtime_fds: Mapping[str, int],
        node_local_executable: Mapping[str, Any] | None,
        runtime_monitor: _FormalRuntimeMonitor | None,
    ):
        self.args = args
        self.workload = list(workload)
        self.exec_workload = list(exec_workload)
        self.runner = dict(runner)
        self.portable = dict(portable)
        self.portable_sha256 = _sha256_bytes(_canonical_json_bytes(self.portable))
        self.tx = tx
        self.workload_environment = dict(workload_environment)
        self.executable_fd = executable_fd
        self.workdir_fd = workdir_fd
        self.workload_input_fds = dict(workload_input_fds)
        self.runtime_fds = dict(runtime_fds)
        self.node_local_executable = (
            None if node_local_executable is None else dict(node_local_executable)
        )
        self.runtime_monitor = runtime_monitor
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
        self.cleanup_name = f"CLEANUP.rank{self.rank}.json"
        self.peer_cleanup_name = f"CLEANUP.rank{self.peer_rank}.json"
        self.handoff_name = f"HANDOFF.rank{self.rank}.json"
        self.peer_handoff_name = f"HANDOFF.rank{self.peer_rank}.json"
        self.heartbeat_name = f"HEARTBEAT.rank{self.rank}.json"
        self.peer_heartbeat_name = f"HEARTBEAT.rank{self.peer_rank}.json"
        self.prepared_sha256 = ""
        self.armed_sha256 = ""
        self.started_sha256 = ""
        self.cleanup_sha256 = ""
        self.workload_result_sha256 = ""
        self.handoff_sha256 = ""
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
        self.descendant_tracker: _DescendantTracker | None = None
        self.decision_go = False

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
        prepared = {
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
        if self.node_local_executable is not None:
            prepared["node_local_executable"] = self.node_local_executable
        return prepared

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
        formal_binding = self.portable.get("workload", {}).get(
            "formal_python_binding"
        )
        if formal_binding is not None:
            keys.add("node_local_executable")
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
            or str(Path(str(payload.get("transaction_root"))).parent)
            != self.portable["namespace"]["canonical_parent"]
            or not _exact_int(payload.get("prepared_unix_ns"), minimum=1)
        ):
            raise PeerAbort(f"PREPARED receipt mismatch for rank {rank}")
        fs_value = payload.get("node_local_filesystem")
        if (
            not isinstance(fs_value, dict)
            or set(fs_value)
            != {
                "root_st_dev",
                "root_st_ino",
                "parent_st_dev",
                "parent_st_ino",
                "parent_path_sha256",
            }
            or not all(
                _exact_int(fs_value.get(key), minimum=0)
                for key in (
                    "root_st_dev",
                    "root_st_ino",
                    "parent_st_dev",
                    "parent_st_ino",
                )
            )
            or fs_value.get("parent_path_sha256")
            != self.portable["namespace"]["canonical_parent_sha256"]
        ):
            raise PeerAbort(f"invalid node-local filesystem evidence for rank {rank}")
        if formal_binding is not None:
            _validate_node_local_executable_binding(
                payload.get("node_local_executable"), formal_binding
            )
        runner = payload.get("runner")
        runner_keys = IDENTITY_KEYS | {
            "path",
            "sha256",
            "status_path",
            "log_path",
            "argv",
            "command",
            "command_argv_sha256",
        }
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
            or not isinstance(runner.get("argv"), list)
            or not isinstance(runner.get("command"), list)
            or not runner["argv"]
            or not runner["command"]
            or not all(isinstance(item, str) for item in runner["argv"])
            or not all(isinstance(item, str) for item in runner["command"])
            or runner["command_argv_sha256"]
            != _argv_sha256(runner["command"])
            or runner["argv"][-len(runner["command"]) :] != runner["command"]
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
            or peer.get("status")
            not in {"PREPARED", "ARMED", "RUNNING", "RESULT", "HANDOFF"}
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
                "terminal_generation",
            }
            or payload.get("schema") != SCHEMA
            or payload.get("status") not in {"SUCCEEDED", "FAILED"}
            or payload.get("rank") not in EXPECTED_RANKS
            or payload.get("portable_sha256") != self.portable_sha256
            or not isinstance(payload.get("reason"), str)
            or payload.get("terminal_generation") != 1
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
            or set(bindings) != {"final_sha256"}
            or not isinstance(bindings["final_sha256"], dict)
            or set(bindings["final_sha256"]) != {"0", "1"}
            or not all(
                HEX64_RE.fullmatch(str(value))
                for value in bindings["final_sha256"].values()
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
            raise PeerAbort("successful OUTCOME appeared before outer finalization")
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
            "terminal_generation": 1,
        }
        try:
            self.tx.publish_immutable("OUTCOME.json", payload)
        except DuplicateInvocation:
            existing, _ = self.tx.read_json("OUTCOME.json")
            self._validate_outcome(existing)
            if existing["status"] != "FAILED":
                raise TransactionError(
                    "cannot reconcile failure with successful OUTCOME"
                )

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
                self.exec_workload,
                self.workload_input_fds,
                self.runtime_fds,
                self.portable["workload"]["input_bindings"],
                environment,
                self.portable["workload"]["argv_sha256"],
                self.portable["workload"]["exec_argv_sha256"],
                self.portable["workload"]["executable_sha256"],
                self.args.shutdown_grace_ms / 1000.0,
                self.runtime_monitor,
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
        formal_binding = self.portable["workload"].get("formal_python_binding")
        if formal_binding is not None:
            if self.node_local_executable is None or "pyvenv_cfg" not in self.runtime_fds:
                raise TransactionError("formal workload Python binding was not retained")
            _assert_formal_venv_python_binding(
                formal_binding,
                self.node_local_executable,
                self.executable_fd,
                self.runtime_fds["pyvenv_cfg"],
                self.runtime_monitor,
            )
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
        expected_keys = IDENTITY_KEYS | {
            "preexec_argv_sha256",
            "command_argv_sha256",
            "exec_argv_sha256",
            "executable_sha256",
            "passed_input_fds",
        }
        passed = workload_identity.get("passed_input_fds") if isinstance(
            workload_identity, dict
        ) else None
        expected_bindings = self.portable["workload"]["input_bindings"]
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
            or workload_identity.get("command_argv_sha256")
            != self.portable["workload"]["argv_sha256"]
            or workload_identity.get("exec_argv_sha256")
            != self.portable["workload"]["exec_argv_sha256"]
            or workload_identity.get("argv_sha256")
            != self.portable["workload"]["exec_argv_sha256"]
            or workload_identity.get("executable_sha256")
            != self.portable["workload"]["executable_sha256"]
            or not isinstance(passed, dict)
            or set(passed) != set(expected_bindings)
            or any(
                not isinstance(passed[logical_id], dict)
                or set(passed[logical_id])
                != {
                    "passed_fd",
                    "st_dev",
                    "st_ino",
                    "sha256",
                    "verification",
                }
                or passed[logical_id].get("passed_fd")
                != expected_bindings[logical_id]["passed_fd"]
                or passed[logical_id].get("sha256")
                != expected_bindings[logical_id]["sha256"]
                or not _exact_int(passed[logical_id].get("st_dev"), minimum=0)
                or not _exact_int(passed[logical_id].get("st_ino"), minimum=0)
                or passed[logical_id].get("verification")
                not in {"linux_proc_child_fd", "cpu_test_fork_inheritance"}
                for logical_id in expected_bindings
            )
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
        self.descendant_tracker = _DescendantTracker(
            int(self.supervisor_identity["pid"]),
            int(workload_identity["pid"]),
            int(self.workgroup["pid"]),
        )
        self.descendant_tracker._remember(workload_identity)
        self.descendant_tracker.discover()

    def _validate_started(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema", "status", "rank", "armed_sha256", "coordinator", "supervisor",
            "workgroup", "workload", "started_unix_ns",
        }
        armed, armed_sha = self.tx.read_json(f"ARMED.rank{rank}.json")
        self._validate_armed(armed, rank)
        workload = payload.get("workload")
        expected_bindings = self.portable["workload"]["input_bindings"]
        passed = workload.get("passed_input_fds") if isinstance(workload, dict) else None
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
            or set(workload)
            != IDENTITY_KEYS
            | {
                "preexec_argv_sha256",
                "command_argv_sha256",
                "exec_argv_sha256",
                "executable_sha256",
                "passed_input_fds",
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
            or workload.get("command_argv_sha256")
            != self.portable["workload"]["argv_sha256"]
            or workload.get("exec_argv_sha256")
            != self.portable["workload"]["exec_argv_sha256"]
            or workload.get("argv_sha256")
            != self.portable["workload"]["exec_argv_sha256"]
            or workload.get("executable_sha256") != self.portable["workload"]["executable_sha256"]
            or not isinstance(passed, dict)
            or set(passed) != set(expected_bindings)
            or any(
                not isinstance(passed[logical_id], dict)
                or set(passed[logical_id])
                != {
                    "passed_fd",
                    "st_dev",
                    "st_ino",
                    "sha256",
                    "verification",
                }
                or passed[logical_id].get("passed_fd")
                != expected_bindings[logical_id]["passed_fd"]
                or passed[logical_id].get("sha256")
                != expected_bindings[logical_id]["sha256"]
                or not _exact_int(passed[logical_id].get("st_dev"), minimum=0)
                or not _exact_int(passed[logical_id].get("st_ino"), minimum=0)
                or passed[logical_id].get("verification")
                not in {"linux_proc_child_fd", "cpu_test_fork_inheritance"}
                for logical_id in expected_bindings
            )
        ):
            raise PeerAbort(f"STARTED receipt mismatch for rank {rank}")

    def _publish_cleanup(self, proof: Any) -> None:
        payload = {
            "schema": SCHEMA,
            "status": "CLEAN",
            "rank": self.rank,
            "started_sha256": self.started_sha256,
            "coordinator": self.coordinator_identity,
            "supervisor": self.supervisor_identity,
            "workgroup": self.workgroup,
            "proof": proof,
            "cleanup_unix_ns": time.time_ns(),
        }
        self._validate_cleanup(payload, self.rank)
        self.cleanup_sha256 = self.tx.publish_immutable(
            self.cleanup_name, payload
        )

    def _validate_cleanup(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema",
            "status",
            "rank",
            "started_sha256",
            "coordinator",
            "supervisor",
            "workgroup",
            "proof",
            "cleanup_unix_ns",
        }
        started, started_sha = self.tx.read_json(f"STARTED.rank{rank}.json")
        self._validate_started(started, rank)
        proof = payload.get("proof")
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") != "CLEAN"
            or payload.get("rank") != rank
            or payload.get("started_sha256") != started_sha
            or payload.get("coordinator") != started["coordinator"]
            or payload.get("supervisor") != started["supervisor"]
            or payload.get("workgroup") != started["workgroup"]
            or not _exact_int(payload.get("cleanup_unix_ns"), minimum=1)
            or not isinstance(proof, dict)
            or set(proof)
            != {
                "schema",
                "status",
                "supervisor_pid",
                "workload_pid",
                "anchor_pid",
                "identity_basis",
                "required_consecutive_empty_censuses",
                "observed_consecutive_empty_censuses",
                "tracked",
                "live_after",
            }
            or proof.get("schema") != SCHEMA
            or proof.get("status") != "DESCENDANTS_CLEAN"
            or proof.get("supervisor_pid") != started["supervisor"]["pid"]
            or proof.get("workload_pid") != started["workload"]["pid"]
            or proof.get("anchor_pid") != started["workgroup"]["pid"]
            or proof.get("identity_basis")
            != "pid+starttime_ticks+full_argv_sha256"
            or proof.get("required_consecutive_empty_censuses") != 3
            or not _exact_int(
                proof.get("observed_consecutive_empty_censuses"), minimum=3
            )
            or proof.get("live_after") != []
            or not isinstance(proof.get("tracked"), list)
        ):
            raise PeerAbort(f"CLEANUP receipt mismatch for rank {rank}")
        workload_tracked = False
        for record in proof["tracked"]:
            if (
                not isinstance(record, dict)
                or set(record)
                != {
                    "pid",
                    "starttime_ticks",
                    "ppids",
                    "pgids",
                    "sids",
                    "argv_sha256",
                }
                or not _exact_int(record.get("pid"), minimum=1)
                or not _exact_int(record.get("starttime_ticks"), minimum=1)
                or any(
                    not isinstance(record.get(key), list)
                    or not record[key]
                    for key in ("ppids", "pgids", "sids", "argv_sha256")
                )
                or not all(
                    _exact_int(value, minimum=0)
                    for key in ("ppids", "pgids", "sids")
                    for value in record[key]
                )
                or not all(
                    isinstance(value, str) and HEX64_RE.fullmatch(value)
                    for value in record["argv_sha256"]
                )
            ):
                raise PeerAbort(f"invalid tracked descendant for rank {rank}")
            if (
                record["pid"] == started["workload"]["pid"]
                and record["starttime_ticks"]
                == started["workload"]["starttime_ticks"]
                and started["workload"]["exec_argv_sha256"]
                in record["argv_sha256"]
            ):
                workload_tracked = True
        if not workload_tracked:
            raise PeerAbort(f"CLEANUP omitted exact workload for rank {rank}")

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
                "cleanup_sha256": self.cleanup_sha256 or None,
                "coordinator": self.coordinator_identity,
                "reason": reason,
                "workload_returncode": rc,
                "result_unix_ns": time.time_ns(),
            },
        )

    def _validate_workload_result(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema", "status", "rank", "started_sha256", "cleanup_sha256",
            "coordinator", "reason", "workload_returncode", "result_unix_ns",
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
            if payload.get("cleanup_sha256") is not None:
                cleanup, cleanup_sha = self.tx.read_json(
                    f"CLEANUP.rank{rank}.json"
                )
                self._validate_cleanup(cleanup, rank)
                if payload.get("cleanup_sha256") != cleanup_sha:
                    raise PeerAbort("ABORTED cleanup binding mismatch")
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
        cleanup, cleanup_sha = self.tx.read_json(f"CLEANUP.rank{rank}.json")
        self._validate_cleanup(cleanup, rank)
        rc = payload.get("workload_returncode")
        if (
            payload.get("started_sha256") != started_sha
            or payload.get("cleanup_sha256") != cleanup_sha
            or payload.get("coordinator") != started["coordinator"]
            or (payload["status"] == "COMPLETED" and not (type(rc) is int and rc == 0))
            or (payload["status"] == "WORKLOAD_FAILED" and not (type(rc) is int and rc != 0))
        ):
            raise PeerAbort(f"WORKLOAD_RESULT semantics mismatch for rank {rank}")

    def _publish_handoff(self, result_hashes: Mapping[int, str]) -> None:
        self.state = "HANDOFF"
        self._heartbeat()
        payload = {
            "schema": SCHEMA,
            "status": "READY_FOR_OUTER_FINALIZER",
            "rank": self.rank,
            "portable_sha256": self.portable_sha256,
            "workload_result_sha256": {
                str(rank): result_hashes[rank] for rank in EXPECTED_RANKS
            },
            "cleanup_sha256": self.cleanup_sha256,
            "coordinator": self.coordinator_identity,
            "handoff_unix_ns": time.time_ns(),
        }
        self._validate_handoff(payload, self.rank)
        self.handoff_sha256 = self.tx.publish_immutable(
            self.handoff_name, payload
        )

    def _validate_handoff(self, payload: Mapping[str, Any], rank: int) -> None:
        keys = {
            "schema",
            "status",
            "rank",
            "portable_sha256",
            "workload_result_sha256",
            "cleanup_sha256",
            "coordinator",
            "handoff_unix_ns",
        }
        results = payload.get("workload_result_sha256")
        if (
            set(payload) != keys
            or payload.get("schema") != SCHEMA
            or payload.get("status") != "READY_FOR_OUTER_FINALIZER"
            or payload.get("rank") != rank
            or payload.get("portable_sha256") != self.portable_sha256
            or not _exact_int(payload.get("handoff_unix_ns"), minimum=1)
            or not isinstance(results, dict)
            or set(results) != {"0", "1"}
            or not all(HEX64_RE.fullmatch(str(value)) for value in results.values())
        ):
            raise PeerAbort(f"HANDOFF schema mismatch for rank {rank}")
        prepared, _ = self.tx.read_json(f"PREPARED.rank{rank}.json")
        self._validate_prepared(prepared, rank)
        if payload.get("coordinator") != prepared["coordinator"]:
            raise PeerAbort(f"HANDOFF coordinator mismatch for rank {rank}")
        for result_rank in EXPECTED_RANKS:
            result, result_sha = self.tx.read_json(
                f"WORKLOAD_RESULT.rank{result_rank}.json"
            )
            self._validate_workload_result(result, result_rank)
            if (
                result["status"] != "COMPLETED"
                or results[str(result_rank)] != result_sha
            ):
                raise PeerAbort(f"HANDOFF result mismatch for rank {rank}")
        cleanup, cleanup_sha = self.tx.read_json(f"CLEANUP.rank{rank}.json")
        self._validate_cleanup(cleanup, rank)
        if payload.get("cleanup_sha256") != cleanup_sha:
            raise PeerAbort(f"HANDOFF cleanup mismatch for rank {rank}")

    def _terminate_workload(self) -> None:
        cleanup_errors: list[str] = []
        fallback_needed = False
        if self.control_fd >= 0:
            try:
                os.write(self.control_fd, b"A")
            except OSError:
                pass
            os.close(self.control_fd)
            self.control_fd = -1
        if self.supervisor_pid is not None:
            # The supervisor may need one grace interval for the workload,
            # one for escaped descendants, and a final reap/census interval.
            # It is the sole killpg owner and keeps its anchor unreaped until
            # all group signalling is over.  The coordinator only requests A
            # and waits here; a bounded fallback targets the exact supervisor
            # generation, never the numeric workgroup PGID.
            grace = self.args.shutdown_grace_ms / 1000.0
            deadline = time.monotonic() + max(2.0, grace * 3.0 + 1.0)
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
                fallback_needed = True
                try:
                    if self.supervisor_identity is None:
                        raise TransactionError(
                            "supervisor fallback identity is unavailable"
                        )
                    _signal_exact_process_generation(
                        self.supervisor_identity,
                        signal.SIGKILL,
                    )
                except BaseException as exc:
                    cleanup_errors.append(
                        f"supervisor fallback: {type(exc).__name__}: {exc}"
                    )
                fallback_deadline = time.monotonic() + max(0.5, grace)
                while time.monotonic() < fallback_deadline:
                    try:
                        waited, _ = os.waitpid(
                            self.supervisor_pid,
                            os.WNOHANG,
                        )
                    except ChildProcessError:
                        waited = self.supervisor_pid
                    if waited == self.supervisor_pid:
                        self.supervisor_pid = None
                        break
                    time.sleep(0.01)
                if self.supervisor_pid is not None:
                    cleanup_errors.append(
                        "exact supervisor remained live after bounded fallback"
                    )
        if fallback_needed and self.descendant_tracker is not None:
            try:
                self.descendant_tracker.signal_live(signal.SIGKILL)
            except BaseException as exc:
                cleanup_errors.append(
                    f"coordinator pidfd fallback: {type(exc).__name__}: {exc}"
                )
        self.workgroup = None
        if cleanup_errors:
            raise TransactionError("; ".join(cleanup_errors))

    def _run_workload(self) -> int:
        self._check_abort()
        self._assert_local_arm_identity()
        if self._peer_is_stale():
            raise PeerAbort("peer heartbeat stale immediately before GO")
        if self.control_fd < 0:
            raise TransactionError("local supervisor control is unavailable")
        os.write(self.control_fd, b"G")
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
                if self.control_fd >= 0:
                    os.close(self.control_fd)
                    self.control_fd = -1
                self._publish_cleanup(result.get("cleanup"))
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
            if self.descendant_tracker is not None:
                self.descendant_tracker.discover()
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
            {"RESULT", "HANDOFF"},
            self.args.completion_timeout_ms,
            "post-result barrier",
        )
        result_hashes = {
            self.rank: self.workload_result_sha256,
            self.peer_rank: peer_result_sha,
        }
        self._publish_handoff(result_hashes)
        self._wait_for_file(
            self.peer_handoff_name,
            self.args.completion_timeout_ms,
            "outer-finalizer handoff",
        )
        for rank in EXPECTED_RANKS:
            handoff, _ = self.tx.read_json(f"HANDOFF.rank{rank}.json")
            self._validate_handoff(handoff, rank)
        return 0

    def run(self) -> int:
        self._bootstrap(time.monotonic() + self.args.prepare_timeout_ms / 1000.0)
        self._publish_prepared()
        prepared_hashes = self._prepare_pair()
        self._arm_and_decide(prepared_hashes)
        return self._run_workload()

    def fail(self, exc: BaseException) -> None:
        cleanup_error: BaseException | None = None
        try:
            self._terminate_workload()
        except BaseException as cleanup_exc:
            cleanup_error = cleanup_exc
        if isinstance(exc, DuplicateInvocation):
            return
        reason = f"{type(exc).__name__}: {exc}"
        if cleanup_error is not None:
            reason += (
                " | cleanup: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        # FAILED OUTCOME is the first required failure receipt.  In
        # particular, a cleanup error must never prevent the peer from seeing
        # an atomically published terminal failure.  The remaining receipts
        # are useful diagnostics but cannot weaken this commit point.
        self._publish_failure(reason)
        try:
            if not self.decision_go:
                self._publish_decision_abort(reason)
            if not self.workload_result_sha256:
                self._publish_workload_result("ABORTED", None, reason)
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
        for fd in self.runtime_fds.values():
            os.close(fd)
        self.runtime_fds = {}
        if self.runtime_monitor is not None:
            self.runtime_monitor.close()
            self.runtime_monitor = None


def _receipt_validator(
    tx: TransactionDirectory,
    portable: Mapping[str, Any],
) -> Coordinator:
    validator = object.__new__(Coordinator)
    validator.tx = tx
    validator.portable = dict(portable)
    validator.portable_sha256 = _sha256_bytes(
        _canonical_json_bytes(portable)
    )
    return validator


def _open_control_transaction(
    args: argparse.Namespace,
) -> tuple[TransactionDirectory, dict[str, Any], str]:
    root, parent, basename = _validate_tx_path(
        args.transaction_root,
        args.run_id,
        args.expected_namespace_parent_sha256,
    )
    tx = TransactionDirectory(root, parent, basename, 1)
    tx.open_existing()
    bootstrap, _ = tx.read_json("TRANSACTION.json")
    if (
        set(bootstrap) != {"schema", "status", "portable", "portable_sha256"}
        or bootstrap.get("schema") != SCHEMA
        or bootstrap.get("status") != "OPEN"
        or not isinstance(bootstrap.get("portable"), dict)
    ):
        tx.close()
        raise TransactionError("transaction bootstrap schema mismatch")
    portable = dict(bootstrap["portable"])
    portable_sha = _sha256_bytes(_canonical_json_bytes(portable))
    namespace = portable.get("namespace")
    if (
        bootstrap.get("portable_sha256") != portable_sha
        or portable_sha != args.expected_portable_sha256
        or portable.get("schema") != PORTABLE_SCHEMA
        or portable.get("run_id") != args.run_id
        or not isinstance(namespace, dict)
        or namespace
        != {
            "deployment_id": args.deployment_id,
            "canonical_parent": str(parent),
            "canonical_parent_sha256": args.expected_namespace_parent_sha256,
        }
        or portable.get("source_commit") != args.source_commit
        or portable.get("source_tree") != args.source_tree
    ):
        tx.close()
        raise TransactionError("control-plane portable binding mismatch")
    if _source_evidence(args) != portable.get("source"):
        tx.close()
        raise TransactionError("control-plane source evidence mismatch")
    return tx, portable, portable_sha


def _validate_inner_success_chain(
    tx: TransactionDirectory,
    portable: Mapping[str, Any],
) -> dict[str, Any]:
    validator = _receipt_validator(tx, portable)
    prepared: dict[int, dict[str, Any]] = {}
    prepared_sha: dict[int, str] = {}
    armed: dict[int, dict[str, Any]] = {}
    armed_sha: dict[int, str] = {}
    started: dict[int, dict[str, Any]] = {}
    started_sha: dict[int, str] = {}
    cleanup: dict[int, dict[str, Any]] = {}
    cleanup_sha: dict[int, str] = {}
    results: dict[int, dict[str, Any]] = {}
    result_sha: dict[int, str] = {}
    handoffs: dict[int, dict[str, Any]] = {}
    handoff_sha: dict[int, str] = {}
    for rank in EXPECTED_RANKS:
        prepared[rank], prepared_sha[rank] = tx.read_json(
            f"PREPARED.rank{rank}.json"
        )
        validator._validate_prepared(prepared[rank], rank)
        armed[rank], armed_sha[rank] = tx.read_json(f"ARMED.rank{rank}.json")
        validator._validate_armed(armed[rank], rank)
        started[rank], started_sha[rank] = tx.read_json(
            f"STARTED.rank{rank}.json"
        )
        validator._validate_started(started[rank], rank)
        cleanup[rank], cleanup_sha[rank] = tx.read_json(
            f"CLEANUP.rank{rank}.json"
        )
        validator._validate_cleanup(cleanup[rank], rank)
        results[rank], result_sha[rank] = tx.read_json(
            f"WORKLOAD_RESULT.rank{rank}.json"
        )
        validator._validate_workload_result(results[rank], rank)
        if results[rank]["status"] != "COMPLETED":
            raise TransactionError("inner chain contains a failed workload")
    decision, decision_sha = tx.read_json("DECISION.json")
    validator._validate_decision(decision)
    expected_decision_bindings = {
        "prepared_sha256": {
            str(rank): prepared_sha[rank] for rank in EXPECTED_RANKS
        },
        "armed_sha256": {
            str(rank): armed_sha[rank] for rank in EXPECTED_RANKS
        },
    }
    if decision["status"] != "GO" or decision["bindings"] != expected_decision_bindings:
        raise TransactionError("GO decision does not bind the exact inner chain")
    for rank in EXPECTED_RANKS:
        handoffs[rank], handoff_sha[rank] = tx.read_json(
            f"HANDOFF.rank{rank}.json"
        )
        validator._validate_handoff(handoffs[rank], rank)
        if handoffs[rank]["workload_result_sha256"] != {
            str(item): result_sha[item] for item in EXPECTED_RANKS
        }:
            raise TransactionError("HANDOFF result pair differs")
    return {
        "validator": validator,
        "prepared": prepared,
        "prepared_sha256": prepared_sha,
        "armed_sha256": armed_sha,
        "decision_sha256": decision_sha,
        "started_sha256": started_sha,
        "cleanup_sha256": cleanup_sha,
        "result_sha256": result_sha,
        "handoff_sha256": handoff_sha,
    }


def _external_artifact_snapshot(
    raw_path: str,
    label: str,
    *,
    json_payload: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = Path(_validate_absolute_evidence_path(raw_path, label))
    fd, canonical = _open_pinned_file(path, executable=False)
    try:
        before = os.fstat(fd)
        digest = _sha256_fd(fd)
        payload: dict[str, Any] | None = None
        if json_payload:
            if before.st_size > MAX_RECEIPT_BYTES:
                raise TransactionError(f"{label} is too large")
            os.lseek(fd, 0, os.SEEK_SET)
            raw = b""
            while len(raw) < before.st_size:
                block = os.read(fd, before.st_size - len(raw))
                if not block:
                    break
                raw += block
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TransactionError(f"{label} is not valid JSON") from exc
            if not isinstance(decoded, dict):
                raise TransactionError(f"{label} is not a JSON object")
            payload = decoded
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or canonical != path
    ):
        raise TransactionError(f"{label} changed during snapshot")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": before.st_size,
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
    }, payload


def _validate_recorded_runner(
    runner: Mapping[str, Any],
    prepared: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    argv = runner["argv"]
    try:
        delimiter = argv.index("--", 2)
    except ValueError as exc:
        raise TransactionError("recorded guarded runner lacks delimiter") from exc
    restored = status.get("restored_guards")
    if (
        len(argv) < 5
        or argv[1] != EXPECTED_RUNNER
        or _runner_option_values(argv, {"--gpus"}) != [EXPECTED_RUNNER_GPUS]
        or _runner_option_values(
            argv, {"--status", "--status-path", "--status-file"}
        )
        != [runner["status_path"]]
        or _runner_option_values(argv, {"--log", "--log-path", "--log-file"})
        != [runner["log_path"]]
        or argv[delimiter + 1 :] != runner["command"]
        or runner["command_argv_sha256"] != _argv_sha256(runner["command"])
        or status.get("state") != "finished"
        or status.get("return_code") != 0
        or any(
            status.get(key) is not None
            for key in (
                "error",
                "cleanup_error",
                "restore_error",
                "received_signal",
            )
        )
        or status.get("command") != runner["command"]
        or status.get("wrapper_pid") != runner["pid"]
        or status.get("child_pid") != prepared["coordinator"]["pid"]
        or not isinstance(restored, dict)
        or set(restored) != {str(index) for index in range(8)}
        or len(set(restored.values())) != 8
        or any(
            not _exact_int(pid, minimum=2)
            for pid in restored.values()
        )
        or runner["pid"] in restored.values()
        or prepared["coordinator"]["pid"] in restored.values()
    ):
        raise TransactionError("outer guarded-runner finish/restore evidence mismatch")
    return {
        "runner_full_argv": list(argv),
        "runner_full_argv_sha256": _argv_sha256(argv),
        "command": list(runner["command"]),
        "command_argv_sha256": runner["command_argv_sha256"],
        "gpu_reservation": EXPECTED_RUNNER_GPUS,
        "restored_guards_by_gpu": dict(restored),
        "status_payload": dict(status),
    }


def _outer_evidence(
    chain: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    prepared = chain["prepared"][rank]
    runner = prepared["runner"]
    status_artifact, status = _external_artifact_snapshot(
        runner["status_path"],
        f"rank {rank} guarded-runner status",
        json_payload=True,
    )
    log_artifact, _ = _external_artifact_snapshot(
        runner["log_path"],
        f"rank {rank} guarded-runner log",
        json_payload=False,
    )
    assert status is not None
    guard = _validate_recorded_runner(runner, prepared, status)
    return {
        "runner": dict(runner),
        "status_artifact": status_artifact,
        "log_artifact": log_artifact,
        "guard_evidence": guard,
    }


def _portable_outer_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Project verified local artifact snapshots into one portable receipt.

    ``st_dev`` and ``st_ino`` are deliberately retained until after the
    caller's same-node alias check.  They are mount-local identity evidence,
    however, and AWS EFS can expose a different ``st_dev`` (and other network
    filesystems can expose a different inode view) for the same immutable
    file on another client.  Consequently neither value may enter FINAL or
    the OUTCOME hash which binds FINAL.
    """

    if set(evidence) != {
        "runner",
        "status_artifact",
        "log_artifact",
        "guard_evidence",
    }:
        raise TransactionError("outer guarded-runner evidence schema changed")
    projected_artifacts: dict[str, dict[str, Any]] = {}
    for name in ("status_artifact", "log_artifact"):
        artifact = evidence.get(name)
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {
                "path",
                "sha256",
                "bytes",
                "st_dev",
                "st_ino",
            }
            or not isinstance(artifact.get("path"), str)
            or not Path(artifact["path"]).is_absolute()
            or HEX64_RE.fullmatch(str(artifact.get("sha256"))) is None
            or not _exact_int(artifact.get("bytes"), minimum=0)
            or not _exact_int(artifact.get("st_dev"), minimum=0)
            or not _exact_int(artifact.get("st_ino"), minimum=0)
        ):
            raise TransactionError(
                f"invalid local outer artifact snapshot: {name}"
            )
        projected_artifacts[name] = {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "bytes": artifact["bytes"],
        }
    runner = evidence.get("runner")
    guard = evidence.get("guard_evidence")
    if not isinstance(runner, Mapping) or not isinstance(guard, Mapping):
        raise TransactionError("outer guarded-runner evidence changed")
    result = {
        "artifact_binding_format": OUTER_ARTIFACT_BINDING_FORMAT,
        "runner": dict(runner),
        "status_artifact": projected_artifacts["status_artifact"],
        "log_artifact": projected_artifacts["log_artifact"],
        "guard_evidence": dict(guard),
    }

    def contains_local_identity(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                key in {"st_dev", "st_ino"}
                or contains_local_identity(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_local_identity(item) for item in value)
        return False

    if contains_local_identity(result):
        raise TransactionError(
            "portable outer evidence contains local filesystem identity"
        )
    return result


def _collect_outer_evidence(
    chain: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    status_paths = {
        chain["prepared"][rank]["runner"]["status_path"]
        for rank in EXPECTED_RANKS
    }
    log_paths = {
        chain["prepared"][rank]["runner"]["log_path"]
        for rank in EXPECTED_RANKS
    }
    if len(status_paths) != 2 or len(log_paths) != 2 or status_paths & log_paths:
        raise TransactionError("runner status/log artifact paths are not independent")
    evidence = {
        rank: _outer_evidence(chain, rank) for rank in EXPECTED_RANKS
    }
    identities = [
        (
            artifact["st_dev"],
            artifact["st_ino"],
        )
        for rank in EXPECTED_RANKS
        for artifact in (
            evidence[rank]["status_artifact"],
            evidence[rank]["log_artifact"],
        )
    ]
    if len(set(identities)) != 4:
        raise TransactionError("runner status/log artifacts alias one inode")
    return {
        rank: _portable_outer_evidence(evidence[rank])
        for rank in EXPECTED_RANKS
    }


def _terminal_final_payload(
    portable_sha256: str,
    chain: Mapping[str, Any],
    rank: int,
    outer: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "SUCCEEDED",
        "rank": rank,
        "portable_sha256": portable_sha256,
        "bindings": {
            "handoff_sha256": chain["handoff_sha256"][rank],
            "cleanup_sha256": chain["cleanup_sha256"][rank],
            "workload_result_sha256": chain["result_sha256"][rank],
        },
        "outer_guarded_runner": dict(outer),
        "reason": "inner cleanup and outer guard restoration both verified",
    }


def _publish_or_verify(
    tx: TransactionDirectory,
    name: str,
    payload: Mapping[str, Any],
    *,
    terminal: bool = False,
) -> str:
    expected = _sha256_bytes(_canonical_json_bytes(payload))
    try:
        if terminal:
            if name != "OUTCOME.json":
                raise TransactionError("terminal publication is reserved for OUTCOME")
            observed = tx.publish_terminal_outcome(payload)
        else:
            observed = tx.publish_immutable(name, payload)
    except DuplicateInvocation:
        existing, observed = tx.read_json(name)
        if existing != payload:
            raise TransactionError(f"conflicting immutable artifact: {name}")
    if observed != expected:
        raise TransactionError(f"immutable artifact digest mismatch: {name}")
    return observed


def _replay_terminal_success(
    tx: TransactionDirectory,
    portable: Mapping[str, Any],
    portable_sha256: str,
    expected_outcome_sha256: str | None,
) -> dict[str, Any]:
    chain = _validate_inner_success_chain(tx, portable)
    outer_evidence = _collect_outer_evidence(chain)
    final_sha: dict[int, str] = {}
    for rank in EXPECTED_RANKS:
        expected = _terminal_final_payload(
            portable_sha256, chain, rank, outer_evidence[rank]
        )
        final, final_sha[rank] = tx.read_json(f"FINAL.rank{rank}.json")
        if final != expected:
            raise TransactionError(f"FINAL receipt mismatch for rank {rank}")
    outcome, outcome_sha = tx.read_json("OUTCOME.json")
    validator = chain["validator"]
    validator._validate_outcome(outcome)
    expected_outcome = {
        "schema": SCHEMA,
        "status": "SUCCEEDED",
        "rank": 0,
        "portable_sha256": portable_sha256,
        "reason": "both outer-finalized guarded nodes succeeded",
        "bindings": {
            "final_sha256": {
                str(rank): final_sha[rank] for rank in EXPECTED_RANKS
            }
        },
        "terminal_generation": 1,
    }
    if outcome != expected_outcome:
        raise TransactionError("terminal OUTCOME differs from replay")
    if expected_outcome_sha256 is not None and outcome_sha != expected_outcome_sha256:
        raise TransactionError("terminal OUTCOME SHA differs from external pin")
    return {
        "schema": SCHEMA,
        "status": "REPLAYED_SUCCEEDED",
        "portable_sha256": portable_sha256,
        "outcome_sha256": outcome_sha,
        "final_sha256": {
            str(rank): final_sha[rank] for rank in EXPECTED_RANKS
        },
    }


def finalize_transaction(args: argparse.Namespace) -> dict[str, Any]:
    tx, portable, portable_sha = _open_control_transaction(args)
    locked = False
    try:
        # Serializing finalizers is necessary because terminal publication must
        # not be followed by a losing finalizer's temporary-file cleanup.
        fcntl.flock(tx.root_fd, fcntl.LOCK_EX)
        locked = True
        tx.assert_identity()
        if tx.exists("OUTCOME.json"):
            return _replay_terminal_success(
                tx, portable, portable_sha, args.expected_outcome_sha256
            )
        chain = _validate_inner_success_chain(tx, portable)
        outer_evidence = _collect_outer_evidence(chain)
        final_payloads = {
            rank: _terminal_final_payload(
                portable_sha,
                chain,
                rank,
                outer_evidence[rank],
            )
            for rank in EXPECTED_RANKS
        }
        final_sha = {
            rank: _publish_or_verify(
                tx, f"FINAL.rank{rank}.json", final_payloads[rank]
            )
            for rank in EXPECTED_RANKS
        }
        # OUTCOME publication is deliberately the final filesystem operation.
        outcome = {
            "schema": SCHEMA,
            "status": "SUCCEEDED",
            "rank": 0,
            "portable_sha256": portable_sha,
            "reason": "both outer-finalized guarded nodes succeeded",
            "bindings": {
                "final_sha256": {
                    str(rank): final_sha[rank] for rank in EXPECTED_RANKS
                }
            },
            "terminal_generation": 1,
        }
        outcome_sha = _publish_or_verify(
            tx,
            "OUTCOME.json",
            outcome,
            terminal=True,
        )
        return {
            "schema": SCHEMA,
            "status": "FINALIZED_SUCCEEDED",
            "portable_sha256": portable_sha,
            "outcome_sha256": outcome_sha,
            "final_sha256": {
                str(rank): final_sha[rank] for rank in EXPECTED_RANKS
            },
        }
    finally:
        if locked:
            try:
                fcntl.flock(tx.root_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        tx.close()


def replay_transaction(args: argparse.Namespace) -> dict[str, Any]:
    tx, portable, portable_sha = _open_control_transaction(args)
    try:
        return _replay_terminal_success(
            tx,
            portable,
            portable_sha,
            args.expected_outcome_sha256,
        )
    finally:
        tx.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--expected-namespace-parent-sha256", required=True)
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


def _control_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"dual_node_guarded_transaction.py {command}",
        description=f"CPU-only {command} for one completed guarded transaction",
    )
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--expected-namespace-parent-sha256", required=True)
    parser.add_argument("--expected-portable-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument(
        "--expected-outcome-sha256",
        required=command == "replay",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] in {"finalize", "replay"}:
        command = tokens.pop(0)
        args = _control_parser(command).parse_args(tokens)
        if (
            not HEX64_RE.fullmatch(args.expected_namespace_parent_sha256)
            or not HEX64_RE.fullmatch(args.expected_portable_sha256)
            or (
                args.expected_outcome_sha256 is not None
                and not HEX64_RE.fullmatch(args.expected_outcome_sha256)
            )
        ):
            print("control-plane SHA pins must be lowercase 64-hex", file=sys.stderr)
            return 2
        try:
            result = (
                finalize_transaction(args)
                if command == "finalize"
                else replay_transaction(args)
            )
        except BaseException as exc:
            print(f"dual-node guarded transaction {command} failed: {exc}", file=sys.stderr)
            return 3
        try:
            os.write(sys.stdout.fileno(), _canonical_json_bytes(result))
        except OSError:
            # Terminal success is the durable OUTCOME; a closed diagnostic
            # stdout must not retroactively turn it into a reported failure.
            pass
        return 0
    args = _parser().parse_args(tokens)
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
    runtime_fds: dict[str, int] = {}
    runtime_monitor: _FormalRuntimeMonitor | None = None
    try:
        transaction_root, parent, basename = _validate_tx_path(
            args.transaction_root,
            args.run_id,
            args.expected_namespace_parent_sha256,
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
            exec_workload,
            executable_fd,
            workdir_fd,
            workload_input_fds,
            runtime_fds,
            node_local_executable,
            runtime_monitor,
        ) = _workload_evidence(args, workload, runner)
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
            exec_workload,
            runner,
            portable,
            tx,
            workload_environment,
            executable_fd,
            workdir_fd,
            workload_input_fds,
            runtime_fds,
            node_local_executable,
            runtime_monitor,
        )
        executable_fd = -1
        workdir_fd = -1
        workload_input_fds = {}
        runtime_fds = {}
        runtime_monitor = None
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
        for fd in runtime_fds.values():
            os.close(fd)
        if runtime_monitor is not None:
            runtime_monitor.close()
        if tx is not None:
            tx.close()


if __name__ == "__main__":
    raise SystemExit(main())
