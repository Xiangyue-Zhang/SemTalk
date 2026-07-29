#!/usr/bin/env python3
"""Run the formal lower-target cache builder and time its full child lifetime.

This wrapper is intentionally standard-library-only.  It must be the direct
child of ``/tmp/globaldiff_guarded_runner.py``.  The measured interval starts
immediately before spawning ``build_lower_target_joints_cache.py`` and ends
only after that exact child has fully exited.  Receipt validation and writing
happen after the interval, so checker work and guarded-runner restoration are
never charged to the builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any


FORMAT = "semtalk_show_lower_target_cache_builder_process_v1"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_RUNNER = "/tmp/globaldiff_guarded_runner.py"
EXPECTED_RUNNER_GPUS = "0,1,2,3,4,5,6,7"
EXPECTED_ENTRIES = 127_286
EXPECTED_COMPUTE_BATCH_WINDOWS = 64
EXPECTED_MAP_SIZE_GIB = 16
EXPECTED_DEVICE = "cuda:0"
ROOT = Path(__file__).resolve().parents[2]
BUILDER = (
    ROOT / "scripts" / "show_base" / "build_lower_target_joints_cache.py"
)
BOOTSTRAP = (
    ROOT
    / "scripts"
    / "show_base"
    / "run_lower_target_cache_builder_guarded.sh"
)


class BuilderProcessError(RuntimeError):
    """The external builder-process contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def argv_bytes(argv: list[str]) -> bytes:
    if not argv or any(type(item) is not str or "\0" in item for item in argv):
        raise BuilderProcessError("argv must be non-empty NUL-free strings")
    return b"\0".join(os.fsencode(item) for item in argv) + b"\0"


def argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(argv_bytes(argv)).hexdigest()


def require_hex(value: Any, length: int, label: str) -> str:
    if type(value) is not str:
        raise BuilderProcessError(f"{label} must be lowercase hexadecimal")
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BuilderProcessError(f"{label} must contain exactly {length} hex")
    return value


def require_exact_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise BuilderProcessError(
            f"{label}={value!r}, expected exact integer {expected}"
        )
    return value


def atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_regular_json(path_value: Path, label: str) -> tuple[Path, bytes, Any]:
    if path_value.is_symlink():
        raise BuilderProcessError(f"{label} must not be a symlink")
    path = path_value.resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BuilderProcessError(f"cannot open {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BuilderProcessError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
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
            raise BuilderProcessError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuilderProcessError(f"{label} is not valid JSON") from error
    return path, raw, value


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_receipt(
    expected_commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    commit = require_hex(expected_commit, 40, "source commit")
    tree = require_hex(expected_tree, 40, "source tree")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise BuilderProcessError(
            "builder wrapper requires a clean checkout; first change: "
            f"{status.splitlines()[0]}"
        )
    origin = _git("remote", "get-url", "origin")
    observed_commit = _git("rev-parse", "HEAD")
    observed_tree = _git("rev-parse", "HEAD^{tree}")
    if (
        origin != EXPECTED_ORIGIN
        or observed_commit != commit
        or observed_tree != tree
    ):
        raise BuilderProcessError("builder wrapper source binding mismatch")
    wrapper = Path(__file__).resolve()
    builder = BUILDER.resolve()
    bootstrap = BOOTSTRAP.resolve()
    if (
        wrapper.is_symlink()
        or builder.is_symlink()
        or bootstrap.is_symlink()
        or not wrapper.is_file()
        or not builder.is_file()
        or not bootstrap.is_file()
    ):
        raise BuilderProcessError("wrapper/builder entrypoint is invalid")
    return {
        "origin": origin,
        "commit": observed_commit,
        "tree": observed_tree,
        "entrypoint": str(wrapper),
        "entrypoint_sha256": sha256_file(wrapper),
        "builder_entrypoint": str(builder),
        "builder_entrypoint_sha256": sha256_file(builder),
        "bootstrap_entrypoint": str(bootstrap),
        "bootstrap_entrypoint_sha256": sha256_file(bootstrap),
    }


def proc_identity(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 1:
        raise BuilderProcessError(f"invalid PID: {pid!r}")
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text()
        raw_argv = (proc / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError) as error:
        raise BuilderProcessError(f"process disappeared: {pid}") from error
    close_parenthesis = stat_text.rfind(")")
    if close_parenthesis < 0:
        raise BuilderProcessError(f"invalid /proc stat for PID {pid}")
    fields = stat_text[close_parenthesis + 2 :].split()
    if len(fields) <= 19 or not raw_argv.endswith(b"\0"):
        raise BuilderProcessError(f"incomplete /proc identity for PID {pid}")
    argv = [
        os.fsdecode(item)
        for item in raw_argv.split(b"\0")
        if item
    ]
    if not argv:
        raise BuilderProcessError(f"empty process argv for PID {pid}")
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "state": fields[0],
        "starttime": fields[19],
        "argv": argv,
        "argv_sha256": hashlib.sha256(raw_argv).hexdigest(),
    }


def guarded_runner_receipt(parsed: argparse.Namespace) -> dict[str, Any]:
    expected_pid = parsed.expected_runner_pid
    expected_starttime = str(parsed.expected_runner_starttime)
    expected_argv_sha = require_hex(
        parsed.expected_runner_argv_sha256,
        64,
        "runner argv SHA-256",
    )
    if (
        type(expected_pid) is not int
        or expected_pid <= 1
        or not expected_starttime.isdigit()
        or os.getppid() != expected_pid
    ):
        raise BuilderProcessError("wrapper is not the exact runner child")
    runner = proc_identity(expected_pid)
    gpu_reservation = any(
        (
            item == "--gpus"
            and index + 1 < len(runner["argv"])
            and runner["argv"][index + 1] == EXPECTED_RUNNER_GPUS
        )
        or item == f"--gpus={EXPECTED_RUNNER_GPUS}"
        for index, item in enumerate(runner["argv"])
    )
    if (
        runner["starttime"] != expected_starttime
        or runner["argv_sha256"] != expected_argv_sha
        or EXPECTED_RUNNER not in runner["argv"]
        or not gpu_reservation
        or runner["state"] == "Z"
    ):
        raise BuilderProcessError("guarded-runner identity binding mismatch")
    return runner


def stable_proc_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: identity[key]
        for key in (
            "pid",
            "ppid",
            "starttime",
            "argv",
            "argv_sha256",
        )
    }


def builder_command(parsed: argparse.Namespace) -> list[str]:
    require_exact_int(
        parsed.expected_entries,
        EXPECTED_ENTRIES,
        "expected entries",
    )
    require_exact_int(
        parsed.compute_batch_windows,
        EXPECTED_COMPUTE_BATCH_WINDOWS,
        "compute batch windows",
    )
    require_exact_int(
        parsed.map_size_gib,
        EXPECTED_MAP_SIZE_GIB,
        "map size GiB",
    )
    if parsed.device != EXPECTED_DEVICE:
        raise BuilderProcessError("formal builder device must remain cuda:0")
    return [
        # Preserve the venv interpreter path.  Resolving a ``bin/python``
        # symlink can silently drop the venv's site-packages.
        sys.executable,
        str(BUILDER.resolve()),
        "--representation-lmdb",
        str(parsed.representation_lmdb),
        "--representation-summary",
        str(parsed.representation_summary),
        "--representation-lineage",
        str(parsed.representation_lineage),
        "--expected-representation-data-sha256",
        str(parsed.expected_representation_data_sha256),
        "--expected-representation-summary-sha256",
        str(parsed.expected_representation_summary_sha256),
        "--expected-representation-lineage-sha256",
        str(parsed.expected_representation_lineage_sha256),
        "--expected-representation-entry-aggregate-sha256",
        str(parsed.expected_representation_entry_aggregate_sha256),
        "--smplx-model-dir",
        str(parsed.smplx_model_dir),
        "--smplx-asset",
        str(parsed.smplx_asset),
        "--expected-smplx-asset-sha256",
        str(parsed.expected_smplx_asset_sha256),
        "--output-lmdb",
        str(parsed.output_lmdb),
        "--output-manifest",
        str(parsed.output_manifest),
        "--expected-source-commit",
        str(parsed.expected_source_commit),
        "--expected-source-tree",
        str(parsed.expected_source_tree),
        "--device",
        parsed.device,
        "--expected-device-name",
        str(parsed.expected_device_name),
        "--expected-entries",
        str(parsed.expected_entries),
        "--compute-batch-windows",
        str(parsed.compute_batch_windows),
        "--map-size-gib",
        str(parsed.map_size_gib),
    ]


def capture_exact_child(
    process: subprocess.Popen[Any],
    command: list[str],
) -> dict[str, Any]:
    expected_sha = argv_sha256(command)
    deadline = time.monotonic() + 10.0
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            identity = proc_identity(process.pid)
        except BuilderProcessError:
            if process.poll() is not None:
                break
            time.sleep(0.01)
            continue
        last = identity
        if (
            identity["ppid"] == os.getpid()
            and identity["state"] != "Z"
            and identity["argv"] == command
            and identity["argv_sha256"] == expected_sha
        ):
            return identity
        if process.poll() is not None:
            break
        time.sleep(0.01)
    # Do not orphan a known Popen child if exec identity capture failed.
    return_code = process.wait()
    raise BuilderProcessError(
        "builder child never reached the exact expected argv; "
        f"rc={return_code}, last={last!r}"
    )


def validate_completed_builder(
    *,
    parsed: argparse.Namespace,
    command: list[str],
    source: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    manifest_path, raw, manifest = read_regular_json(
        Path(parsed.output_manifest),
        "builder manifest",
    )
    if type(manifest) is not dict:
        raise BuilderProcessError("builder manifest must be one JSON object")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest_source = manifest.get("source_receipt")
    manifest_lmdb = manifest.get("lmdb")
    if (
        manifest.get("status") != "complete"
        or manifest.get("format")
        != "semtalk_show_lower_target_joints_raw_lmdb_v1"
        or manifest.get("argv") != command[1:]
        or type(manifest_source) is not dict
        or {
            key: manifest_source.get(key)
            for key in ("origin", "commit", "tree")
        }
        != {
            key: source.get(key)
            for key in ("origin", "commit", "tree")
        }
        or manifest_source.get("entrypoint")
        != source.get("builder_entrypoint")
        or manifest_source.get("entrypoint_sha256")
        != source.get("builder_entrypoint_sha256")
        or type(manifest_lmdb) is not dict
        or Path(str(manifest_lmdb.get("path", ""))).resolve()
        != Path(parsed.output_lmdb).resolve()
        or not Path(parsed.output_lmdb).resolve().is_dir()
        or manifest.get("entries") != EXPECTED_ENTRIES
        or manifest.get("finite") is not True
        or manifest.get("exact_once") is not True
    ):
        raise BuilderProcessError("completed builder manifest binding is invalid")
    return manifest_path, manifest_sha, manifest


def run_builder(parsed: argparse.Namespace) -> dict[str, Any]:
    output_lmdb = Path(parsed.output_lmdb)
    output_manifest = Path(parsed.output_manifest)
    output_receipt = Path(parsed.output_process_receipt)
    for path, label in (
        (output_lmdb, "output LMDB"),
        (output_manifest, "output manifest"),
        (output_receipt, "output process receipt"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{label} already exists: {path}")
    source = source_receipt(
        parsed.expected_source_commit,
        parsed.expected_source_tree,
    )
    runner = guarded_runner_receipt(parsed)
    wrapper = proc_identity(os.getpid())
    if (
        wrapper["ppid"] != runner["pid"]
        or wrapper["state"] == "Z"
        or len(wrapper["argv"]) < 2
        or Path(wrapper["argv"][1]).resolve() != Path(__file__).resolve()
    ):
        raise BuilderProcessError("wrapper process identity is invalid")
    command = builder_command(parsed)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "0"

    # These are the authoritative full-child timing boundaries.  Do not move
    # either timestamp inside the builder or around receipt/checker work.
    started_unix_ns = time.time_ns()
    started_monotonic_ns = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
    )
    child = capture_exact_child(process, command)
    return_code = process.wait()
    completed_monotonic_ns = time.perf_counter_ns()
    completed_unix_ns = time.time_ns()

    elapsed_ns = completed_monotonic_ns - started_monotonic_ns
    if return_code != 0 or elapsed_ns <= 0:
        raise BuilderProcessError(
            f"builder child failed rc={return_code}, elapsed_ns={elapsed_ns}"
        )
    final_runner = guarded_runner_receipt(parsed)
    if stable_proc_identity(final_runner) != stable_proc_identity(runner):
        raise BuilderProcessError("guarded-runner identity changed")
    if stable_proc_identity(proc_identity(os.getpid())) != stable_proc_identity(
        wrapper
    ):
        raise BuilderProcessError("builder wrapper identity changed")
    final_source = source_receipt(
        parsed.expected_source_commit,
        parsed.expected_source_tree,
    )
    if final_source != source:
        raise BuilderProcessError("wrapper/builder source changed")
    manifest_path, manifest_sha, manifest = validate_completed_builder(
        parsed=parsed,
        command=command,
        source=source,
    )
    receipt = {
        "format": FORMAT,
        "status": "complete",
        "scope": {
            "dataset": "show_base",
            "formal_stage": "lower",
            "speaker_scope": "All",
            "speaker_ids": [0, 1, 2, 3],
        },
        "source_receipt": source,
        "guarded_runner": runner,
        "wrapper_process": wrapper,
        "builder_process": {
            "pid": child["pid"],
            "ppid": child["ppid"],
            "starttime": child["starttime"],
            "argv": command,
            "argv_sha256": argv_sha256(command),
            "observed_argv_sha256": child["argv_sha256"],
            "return_code": return_code,
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
            "started_unix_ns": started_unix_ns,
            "completed_unix_ns": completed_unix_ns,
            "started_monotonic_ns": started_monotonic_ns,
            "completed_monotonic_ns": completed_monotonic_ns,
            "elapsed_monotonic_ns": elapsed_ns,
            "elapsed_seconds": elapsed_ns / 1_000_000_000.0,
            "timing_scope": (
                "immediately_before_popen_through_complete_child_exit"
            ),
            "excludes": [
                "independent_checker",
                "formal_gate",
                "guard_restore",
                "receipt_validation_and_write",
            ],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "status": manifest["status"],
            "entries": manifest["entries"],
            "entry_aggregate_sha256": manifest[
                "entry_aggregate_sha256"
            ],
        },
        "completed_unix_ns": time.time_ns(),
    }
    atomic_json_new(output_receipt.resolve(), receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--representation-lmdb", required=True)
    parser.add_argument("--representation-summary", required=True)
    parser.add_argument("--representation-lineage", required=True)
    parser.add_argument("--expected-representation-data-sha256", required=True)
    parser.add_argument("--expected-representation-summary-sha256", required=True)
    parser.add_argument("--expected-representation-lineage-sha256", required=True)
    parser.add_argument(
        "--expected-representation-entry-aggregate-sha256",
        required=True,
    )
    parser.add_argument("--smplx-model-dir", required=True)
    parser.add_argument("--smplx-asset", required=True)
    parser.add_argument("--expected-smplx-asset-sha256", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-process-receipt", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--device", default=EXPECTED_DEVICE)
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument(
        "--expected-entries",
        type=int,
        default=EXPECTED_ENTRIES,
    )
    parser.add_argument(
        "--compute-batch-windows",
        type=int,
        default=EXPECTED_COMPUTE_BATCH_WINDOWS,
    )
    parser.add_argument(
        "--map-size-gib",
        type=int,
        default=EXPECTED_MAP_SIZE_GIB,
    )
    parser.add_argument("--expected-runner-pid", type=int, required=True)
    parser.add_argument("--expected-runner-starttime", required=True)
    parser.add_argument("--expected-runner-argv-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    receipt_path = Path(parsed.output_process_receipt)
    try:
        receipt = run_builder(parsed)
    except BaseException as error:
        failure_path = receipt_path.with_name(
            f"{receipt_path.name}.failure.json"
        )
        if not failure_path.exists() and not failure_path.is_symlink():
            try:
                atomic_json_new(
                    failure_path.resolve(),
                    {
                        "format": FORMAT,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "failed_unix_ns": time.time_ns(),
                        "argv": sys.argv,
                    },
                )
            except BaseException:
                pass
        raise
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path.resolve()),
                "receipt_sha256": sha256_file(receipt_path.resolve()),
                "manifest_sha256": receipt["manifest"]["sha256"],
                "builder_elapsed_seconds": receipt["builder_process"][
                    "elapsed_seconds"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
