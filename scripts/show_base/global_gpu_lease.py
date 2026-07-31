#!/usr/bin/env python3
"""Per-host, cross-mode lease for formal SemTalk GPU runner invocations.

The guarded runner owns all eight physical GPUs while it temporarily removes
the idle guards.  That ownership is host-global, not specific to one SemTalk
mode.  This helper gives every formal caller one common fail-closed mutex:

* ``acquire`` creates the active directory atomically and publishes one
  immutable, token-bound lease receipt;
* an existing directory is never inspected for staleness and never stolen;
* ``release`` accepts only the runner-status path bound at acquisition and
  archives the whole directory with an atomic no-replace rename, but only
  after successful runner exit, cleanup, and exact guard restoration are
  proved and the caller explicitly confirms descendant/real-guard checks.

The helper is deliberately CPU-only and never starts, signals, or inspects a
GPU workload.  Formal two-host W16 orchestration acquires host slot 0 first,
then passes its lease-receipt SHA-256 as slot 1's predecessor binding.  Each
host still owns an independent local lease.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


SCHEMA = "semtalk.formal_global_gpu_lease.v1"
GLOBAL_LEASE_DIR = Path("/tmp/semtalk_formal_global_gpu_lease.active")
LEASE_FILE = "LEASE.json"
RUNNER_STATUS_FILE = "RUNNER_STATUS.json"
RELEASE_FILE = "RELEASE.json"
EXPECTED_GPUS = tuple(range(8))
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_EVIDENCE_BYTES = 1 << 20


class LeaseError(RuntimeError):
    """The lease cannot be safely acquired, validated, or archived."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
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


def _require_sha256(value: str, label: str) -> str:
    if not HEX64_RE.fullmatch(value):
        raise LeaseError(f"{label} must be one lowercase SHA-256")
    return value


def _require_safe_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise LeaseError(f"{label} is not a safe identifier")
    return value


def _exact_int(value: Any, *, minimum: int | None = None) -> bool:
    return type(value) is int and (minimum is None or value >= minimum)


def _stable_existing_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        before = resolved.stat()
        after = resolved.stat()
    except OSError as exc:
        raise LeaseError(f"{label} is unavailable: {path}") from exc
    mode = stat.S_IMODE(before.st_mode)
    safe_owner = before.st_uid == os.getuid()
    safe_shared_tmp = before.st_uid == 0 and bool(mode & stat.S_ISVTX)
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or not (safe_owner or safe_shared_tmp)
    ):
        raise LeaseError(
            f"{label} is not one stable caller-owned or root-sticky directory"
        )
    return resolved


def _normalize_new_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise LeaseError(f"{label} must be an absolute child path")
    parent = _stable_existing_directory(path.parent, f"{label} parent")
    return parent / path.name


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_stable_regular_file(
    path: Path,
    label: str,
    *,
    require_owner: bool = True,
) -> tuple[bytes, dict[str, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise LeaseError(f"{label} is unavailable or unsafe: {path}") from exc
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (require_owner and before.st_uid != os.getuid())
            or before.st_size < 2
            or before.st_size > MAX_EVIDENCE_BYTES
        ):
            raise LeaseError(f"{label} is not one safe bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(remaining, 1024 * 1024))
            if not block:
                raise LeaseError(f"{label} was truncated during snapshot")
            chunks.append(block)
            remaining -= len(block)
        if os.read(fd, 1):
            raise LeaseError(f"{label} grew during snapshot")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise LeaseError(f"{label} changed during snapshot")
    return b"".join(chunks), {
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_size": before.st_size,
        "st_mtime_ns": before.st_mtime_ns,
        "st_ctime_ns": before.st_ctime_ns,
        "st_uid": before.st_uid,
        "st_mode": stat.S_IMODE(before.st_mode),
    }


def _decode_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LeaseError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeaseError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LeaseError(f"{label} is not one JSON object")
    return value


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        stat.S_IMODE(value.st_mode),
    )


def _read_stable_regular_file_at(
    directory_fd: int,
    name: str,
    label: str,
) -> tuple[bytes, dict[str, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise LeaseError(f"{label} is unavailable or unsafe") from exc
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_size < 2
            or before.st_size > MAX_EVIDENCE_BYTES
        ):
            raise LeaseError(f"{label} is not one safe bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(remaining, 1024 * 1024))
            if not block:
                raise LeaseError(f"{label} was truncated during snapshot")
            chunks.append(block)
            remaining -= len(block)
        if os.read(fd, 1):
            raise LeaseError(f"{label} grew during snapshot")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise LeaseError(f"{label} changed during snapshot")
    return b"".join(chunks), {
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "st_size": before.st_size,
        "st_mtime_ns": before.st_mtime_ns,
        "st_ctime_ns": before.st_ctime_ns,
        "st_uid": before.st_uid,
        "st_mode": stat.S_IMODE(before.st_mode),
    }


def _write_exclusive_at(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(
    source: Path,
    target: Path,
    expected_source_identity: tuple[int, int, int, int],
) -> None:
    if source.parent != target.parent:
        raise LeaseError("lease archive rename must remain in one directory")
    directory_fd = os.open(
        source.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        try:
            source_stat = os.stat(
                source.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise LeaseError("active lease path disappeared before archive") from exc
        if (
            not stat.S_ISDIR(source_stat.st_mode)
            or _directory_identity(source_stat) != expected_source_identity
        ):
            raise LeaseError("active lease path identity changed before archive")
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "linux":
            try:
                rename = libc.renameat2
            except AttributeError as exc:  # pragma: no cover - formal glibc.
                raise LeaseError("atomic no-replace rename is unavailable") from exc
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(
                directory_fd,
                os.fsencode(source.name),
                directory_fd,
                os.fsencode(target.name),
                1,  # RENAME_NOREPLACE
            )
        elif sys.platform == "darwin":
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
                directory_fd,
                os.fsencode(source.name),
                directory_fd,
                os.fsencode(target.name),
                0x00000004,  # RENAME_EXCL
            )
        else:  # pragma: no cover - formal execution is Linux-only.
            raise LeaseError("atomic no-replace rename is unavailable")
        if result == 0:
            archived_stat = os.stat(
                target.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(archived_stat.st_mode)
                or _directory_identity(archived_stat)
                != expected_source_identity
            ):
                raise LeaseError("archived lease path identity mismatch")
            os.fsync(directory_fd)
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise LeaseError("lease archive target already exists")
        raise LeaseError(
            f"atomic lease archive rename failed: {os.strerror(error)}"
        )
    finally:
        os.close(directory_fd)


def _current_host(expected: str) -> str:
    actual = socket.gethostname()
    if actual != expected:
        raise LeaseError(
            f"formal host mismatch: actual={actual!r}, expected={expected!r}"
        )
    return actual


def _validate_host_order(
    host_count: int,
    host_slot: int,
    predecessor_sha256: str | None,
) -> str | None:
    if host_count not in {1, 2}:
        raise LeaseError("formal host count must be exactly 1 or 2")
    if host_slot not in {0, 1}:
        raise LeaseError("formal host slot must be exactly 0 or 1")
    if host_count == 1:
        if predecessor_sha256 is not None:
            raise LeaseError("single-host acquisition forbids a predecessor")
        return None
    if host_slot == 0:
        if predecessor_sha256 is not None:
            raise LeaseError("two-host slot 0 forbids a predecessor")
        return None
    if predecessor_sha256 is None:
        raise LeaseError(
            "two-host slot 1 requires slot 0 lease-receipt SHA-256"
        )
    return _require_sha256(
        predecessor_sha256,
        "slot 0 predecessor lease receipt",
    )


def _acquire(args: argparse.Namespace) -> dict[str, Any]:
    # This path is code-owned so every SemTalk mode contends for the same
    # host-global authority.  It is intentionally not a CLI option.
    lease_dir = _normalize_new_path(
        str(GLOBAL_LEASE_DIR),
        "active lease directory",
    )
    formal_host = _current_host(args.formal_host)
    mode = _require_safe_id(args.mode, "formal GPU mode")
    run_id = _require_safe_id(args.run_id, "formal run ID")
    predecessor = _validate_host_order(
        args.formal_host_count,
        args.formal_host_slot,
        args.predecessor_lease_receipt_sha256,
    )
    status_path = _normalize_new_path(
        args.runner_status,
        "guarded-runner status",
    )
    if status_path == lease_dir:
        raise LeaseError(
            "guarded-runner status path cannot be the active lease directory"
        )
    if status_path.exists() or status_path.is_symlink():
        raise LeaseError("fresh guarded-runner status path already exists")

    lease_id = uuid.uuid4().hex
    token = uuid.uuid4().hex + uuid.uuid4().hex
    archive_dir = lease_dir.with_name(f"{lease_dir.name}.archive.{lease_id}")
    if archive_dir.exists() or archive_dir.is_symlink():
        raise LeaseError("lease archive target already exists")
    try:
        os.mkdir(lease_dir, 0o700)
    except FileExistsError as exc:
        raise LeaseError(
            "global GPU lease is already held; leases are never stolen"
        ) from exc
    except OSError as exc:
        raise LeaseError(f"cannot atomically acquire global GPU lease: {exc}") from exc

    # A crash after mkdir intentionally leaves an unstealable, fail-closed
    # directory.  There is no automatic stale-owner recovery path.
    receipt = {
        "schema": SCHEMA,
        "state": "ACTIVE",
        "lease_id": lease_id,
        "lease_dir": str(lease_dir),
        "archive_dir": str(archive_dir),
        "token_sha256": _sha256_bytes(token.encode("ascii")),
        "formal_host": formal_host,
        "formal_host_count": args.formal_host_count,
        "formal_host_slot": args.formal_host_slot,
        "two_host_acquisition_order": "ascending_formal_host_slot",
        "predecessor_lease_receipt_sha256": predecessor,
        "mode": mode,
        "run_id": run_id,
        "runner_status_path": str(status_path),
        "acquired_time_ns": time.time_ns(),
        "acquirer": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "uid": os.getuid(),
            "argv_sha256": _sha256_bytes(
                b"\0".join(os.fsencode(token_) for token_ in sys.argv) + b"\0"
            ),
        },
        "utility": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "policy": {
            "gpu_reservation": "0,1,2,3,4,5,6,7",
            "cross_mode": True,
            "stale_lease_stealing": False,
            "archive_requires_successful_runner_and_caller_confirmation": True,
        },
    }
    raw = _canonical_json_bytes(receipt)
    try:
        _write_exclusive(lease_dir / LEASE_FILE, raw)
        _fsync_directory(lease_dir)
        _fsync_directory(lease_dir.parent)
    except Exception:
        # Do not remove the directory: uncertainty after acquisition must
        # remain fail closed and visible to the operator.
        raise
    return {
        "status": "ACQUIRED",
        "lease_id": lease_id,
        "lease_token": token,
        "lease_dir": str(lease_dir),
        "archive_dir": str(archive_dir),
        "lease_receipt_sha256": _sha256_bytes(raw),
        "formal_host_slot": args.formal_host_slot,
    }


def _parse_confirmed_guards(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise LeaseError("each confirmed guard must be GPU=PID")
        gpu, pid_text = value.split("=", 1)
        if gpu in result or not gpu.isdigit() or not pid_text.isdigit():
            raise LeaseError("confirmed guard GPU/PID syntax is invalid")
        pid = int(pid_text)
        if not _exact_int(pid, minimum=2):
            raise LeaseError("confirmed guard PID is invalid")
        result[gpu] = pid
    if set(result) != {str(gpu) for gpu in EXPECTED_GPUS}:
        raise LeaseError("caller must confirm exactly one guard for GPU 0..7")
    if len(set(result.values())) != len(EXPECTED_GPUS):
        raise LeaseError("caller-confirmed guard PIDs must be unique")
    return result


def _validate_runner_status(
    status: Mapping[str, Any],
    confirmed_guards: Mapping[str, int],
) -> dict[str, int]:
    restored = status.get("restored_guards")
    null_fields = (
        "error",
        "cleanup_error",
        "restore_error",
        "received_signal",
    )
    if (
        status.get("state") != "finished"
        or not _exact_int(status.get("return_code"))
        or status.get("return_code") != 0
        or any(key not in status or status[key] is not None for key in null_fields)
        or not isinstance(restored, dict)
        or set(restored) != {str(gpu) for gpu in EXPECTED_GPUS}
        or any(not _exact_int(pid, minimum=2) for pid in restored.values())
        or len(set(restored.values())) != len(EXPECTED_GPUS)
    ):
        raise LeaseError(
            "guarded runner did not prove finished rc0, clean cleanup/restore, "
            "and exact unique guards on GPU 0..7"
        )
    normalized = {key: int(value) for key, value in restored.items()}
    if normalized != dict(confirmed_guards):
        raise LeaseError(
            "caller-confirmed real guards differ from runner restoration evidence"
        )
    return normalized


def _release_pinned(
    args: argparse.Namespace,
    *,
    lease_dir: Path,
    lease_fd: int,
    lease_identity: tuple[int, int, int, int],
    formal_host: str,
    confirmed_guards: Mapping[str, int],
) -> dict[str, Any]:
    if set(os.listdir(lease_fd)) != {LEASE_FILE}:
        raise LeaseError("active lease has unexpected or partial evidence")
    lease_raw, lease_artifact = _read_stable_regular_file_at(
        lease_fd,
        LEASE_FILE,
        "active lease receipt",
    )
    lease_sha256 = _sha256_bytes(lease_raw)
    if lease_sha256 != _require_sha256(
        args.expected_lease_receipt_sha256,
        "expected lease receipt",
    ):
        raise LeaseError("active lease receipt SHA-256 mismatch")
    lease = _decode_json_object(lease_raw, "active lease receipt")
    archive_dir = lease_dir.with_name(
        f"{lease_dir.name}.archive.{args.lease_id}"
    )
    if (
        lease.get("schema") != SCHEMA
        or lease.get("state") != "ACTIVE"
        or lease.get("lease_id") != args.lease_id
        or lease.get("lease_dir") != str(lease_dir)
        or lease.get("archive_dir") != str(archive_dir)
        or lease.get("formal_host") != formal_host
        or lease.get("token_sha256")
        != _sha256_bytes(args.lease_token.encode("ascii"))
    ):
        raise LeaseError("active lease identity/token/host binding mismatch")
    if archive_dir.exists() or archive_dir.is_symlink():
        raise LeaseError("lease archive target already exists")

    status_path = _normalize_new_path(
        args.runner_status,
        "guarded-runner status",
    )
    if lease.get("runner_status_path") != str(status_path):
        raise LeaseError("runner status path differs from the acquired lease")
    status_raw, status_artifact = _read_stable_regular_file(
        status_path,
        "guarded-runner status",
    )
    status_sha256 = _sha256_bytes(status_raw)
    if status_sha256 != _require_sha256(
        args.expected_runner_status_sha256,
        "expected guarded-runner status",
    ):
        raise LeaseError("guarded-runner status SHA-256 mismatch")
    status = _decode_json_object(status_raw, "guarded-runner status")
    restored = _validate_runner_status(status, confirmed_guards)

    release = {
        "schema": SCHEMA,
        "state": "ARCHIVED",
        "lease_id": args.lease_id,
        "lease_receipt_sha256": lease_sha256,
        "formal_host": formal_host,
        "mode": lease.get("mode"),
        "run_id": lease.get("run_id"),
        "runner_status": {
            "path": str(status_path),
            "sha256": status_sha256,
            "artifact": status_artifact,
            "state": "finished",
            "return_code": 0,
            "error": None,
            "cleanup_error": None,
            "restore_error": None,
            "received_signal": None,
            "restored_guards": restored,
        },
        "caller_confirmations": {
            "no_live_workload_descendants": True,
            "exact_one_guard_per_gpu_0_through_7": True,
            "guards_are_real_torchvision_resnet18_globaldiff_gpu_guard_cnn": True,
            "confirmed_guards": dict(confirmed_guards),
        },
        "lease_artifact": lease_artifact,
        "released_time_ns": time.time_ns(),
        "release_policy": "atomic_no_replace_archive_after_complete_evidence",
    }
    release_raw = _canonical_json_bytes(release)
    _write_exclusive_at(lease_fd, RUNNER_STATUS_FILE, status_raw)
    _write_exclusive_at(lease_fd, RELEASE_FILE, release_raw)
    os.fsync(lease_fd)

    expected_entries = {LEASE_FILE, RUNNER_STATUS_FILE, RELEASE_FILE}
    if set(os.listdir(lease_fd)) != expected_entries:
        raise LeaseError("lease evidence changed before archival")
    # Reopen every artifact through the pinned directory immediately before
    # the rename.  A path-based snapshot from earlier in release is not
    # sufficient: same-UID mutation or directory replacement must fail closed.
    final_lease_raw, final_lease_artifact = _read_stable_regular_file_at(
        lease_fd,
        LEASE_FILE,
        "final active lease receipt",
    )
    final_status_raw, _ = _read_stable_regular_file_at(
        lease_fd,
        RUNNER_STATUS_FILE,
        "final runner status copy",
    )
    final_release_raw, _ = _read_stable_regular_file_at(
        lease_fd,
        RELEASE_FILE,
        "final release receipt",
    )
    if (
        _sha256_bytes(final_lease_raw) != lease_sha256
        or final_lease_artifact != lease_artifact
        or final_status_raw != status_raw
        or final_release_raw != release_raw
        or _directory_identity(os.fstat(lease_fd)) != lease_identity
    ):
        raise LeaseError("pinned lease evidence changed before archival")

    _rename_noreplace(lease_dir, archive_dir, lease_identity)
    # The descriptor remains pinned across rename.  Verify that the exact
    # archived inode still contains the exact three byte-bound artifacts before
    # reporting success; a raced replacement can never yield ARCHIVED success.
    if (
        _directory_identity(os.fstat(lease_fd)) != lease_identity
        or set(os.listdir(lease_fd)) != expected_entries
        or _read_stable_regular_file_at(
            lease_fd, LEASE_FILE, "archived lease receipt"
        )[0]
        != lease_raw
        or _read_stable_regular_file_at(
            lease_fd, RUNNER_STATUS_FILE, "archived runner status copy"
        )[0]
        != status_raw
        or _read_stable_regular_file_at(
            lease_fd, RELEASE_FILE, "archived release receipt"
        )[0]
        != release_raw
    ):
        raise LeaseError("archived lease evidence changed before success")
    return {
        "status": "ARCHIVED",
        "lease_id": args.lease_id,
        "archive_dir": str(archive_dir),
        "lease_receipt_sha256": lease_sha256,
        "runner_status_sha256": status_sha256,
        "release_receipt_sha256": _sha256_bytes(release_raw),
    }


def _release(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_no_live_descendants:
        raise LeaseError("caller must confirm that no workload descendant is live")
    if not args.confirm_restored_guards_are_real_resnet18:
        raise LeaseError(
            "caller must confirm each restored guard is the real torchvision "
            "ResNet18 globaldiff_gpu_guard_cnn"
        )
    confirmed_guards = _parse_confirmed_guards(args.confirmed_guard)
    formal_host = _current_host(args.formal_host)
    lease_dir = _normalize_new_path(
        str(GLOBAL_LEASE_DIR),
        "active lease directory",
    )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        lease_fd = os.open(lease_dir, flags)
    except OSError as exc:
        raise LeaseError("active global GPU lease is unavailable or unsafe") from exc
    try:
        lease_stat = os.fstat(lease_fd)
        if (
            not stat.S_ISDIR(lease_stat.st_mode)
            or lease_stat.st_uid != os.getuid()
            or stat.S_IMODE(lease_stat.st_mode) & 0o077
        ):
            raise LeaseError("active lease directory is unsafe")
        return _release_pinned(
            args,
            lease_dir=lease_dir,
            lease_fd=lease_fd,
            lease_identity=_directory_identity(lease_stat),
            formal_host=formal_host,
            confirmed_guards=confirmed_guards,
        )
    finally:
        os.close(lease_fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed per-host global GPU lease for formal SemTalk",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--formal-host", required=True)
    acquire.add_argument("--formal-host-count", required=True, type=int)
    acquire.add_argument("--formal-host-slot", required=True, type=int)
    acquire.add_argument("--predecessor-lease-receipt-sha256")
    acquire.add_argument("--mode", required=True)
    acquire.add_argument("--run-id", required=True)
    acquire.add_argument("--runner-status", required=True)
    acquire.set_defaults(handler=_acquire)

    release = subparsers.add_parser("release")
    release.add_argument("--formal-host", required=True)
    release.add_argument("--lease-id", required=True)
    release.add_argument("--lease-token", required=True)
    release.add_argument("--expected-lease-receipt-sha256", required=True)
    release.add_argument("--runner-status", required=True)
    release.add_argument("--expected-runner-status-sha256", required=True)
    release.add_argument(
        "--confirm-no-live-descendants",
        action="store_true",
    )
    release.add_argument(
        "--confirm-restored-guards-are-real-resnet18",
        action="store_true",
    )
    release.add_argument(
        "--confirmed-guard",
        action="append",
        default=[],
        metavar="GPU=PID",
    )
    release.set_defaults(handler=_release)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (LeaseError, OSError, ValueError) as exc:
        parser.exit(1, f"global GPU lease error: {exc}\n")
    sys.stdout.buffer.write(_canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
