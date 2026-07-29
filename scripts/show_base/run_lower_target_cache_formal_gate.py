#!/usr/bin/env python3
"""Fail-closed formal gate for the SHOW lower-target joint cache.

The public mode must itself be launched through
``/tmp/globaldiff_guarded_runner.py`` with exactly one H200 visible.  It:

* binds the cache builder/checker, representation, source and SMPL-X receipts;
* runs four isolated deterministic children on real SHOW-All lower batches:
  two legacy repeats and two cache repeats.  It first proves the legacy A/A
  and cache B/B repeat controls exact, then proves both A/B pairs exact after
  the initial state, both complete optimizer updates, and final scheduler
  state;
* requires raw target-joint bytes to be identical for both updates;
* benchmarks the production path with four fresh child processes in A-B-B-A
  order.  Every child has five warm-up and twenty-five measured real updates;
* requires pooled and paired-block wall-clock and CUDA-event speedups >= 1.05;
* amortizes the *full* recorded builder process time over 600 * 1989 updates
  and requires that end-to-end speedup >= 1.05.

No output root is reusable.  Any exception leaves only a ``failure.json`` and
partial evidence; only an atomically created ``formal_gate_report.json`` with
``status=pass`` authorizes the cache for training.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import contextlib
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import stat
import subprocess
import sys
import time
import traceback
import types
from typing import Any
import warnings

sys.dont_write_bytecode = True


FORMAT = "semtalk_show_lower_target_cache_formal_gate_v1"
BUILDER_PROCESS_FORMAT = (
    "semtalk_show_lower_target_cache_builder_process_v1"
)
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_SPEAKERS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
EXPECTED_SPEAKER_IDS = (0, 1, 2, 3)
EXPECTED_ENTRIES = 127_309
EXPECTED_UPDATES_PER_EPOCH = 1_989
EXPECTED_EPOCHS = 600
EXPECTED_BATCH_SIZE = 64
EXPECTED_FRAMES = 64
EQUIVALENCE_UPDATES = 2
ABBA_ORDER = ("legacy", "cache", "cache", "legacy")
EQUIVALENCE_CHILDREN = (
    ("equivalence-legacy-0", "legacy", 0),
    ("equivalence-legacy-1", "legacy", 1),
    ("equivalence-cache-0", "cache", 0),
    ("equivalence-cache-1", "cache", 1),
)
BENCHMARK_CHILDREN = (
    ("benchmark-0-legacy", "legacy", 0),
    ("benchmark-1-cache", "cache", 1),
    ("benchmark-2-cache", "cache", 2),
    ("benchmark-3-legacy", "legacy", 3),
)
INTERNAL_MODES = tuple(
    item[0] for item in (*EQUIVALENCE_CHILDREN, *BENCHMARK_CHILDREN)
)
WARMUP_UPDATES = 5
MEASURED_UPDATES = 25
MIN_SPEEDUP = 1.05
FORMAL_SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
CRITICAL_SOURCE_FILES = (
    "train.py",
    "show_base_train.py",
    "dataloaders/show_base.py",
    "aelower_trainer.py",
    "utils/config.py",
    "utils/lower_target_cache.py",
    "utils/rotation_conversions.py",
    "utils/show_base_tensor_ops.py",
    "utils/smplx_training.py",
    "scripts/show_base/build_lower_target_joints_cache.py",
    "scripts/show_base/run_lower_target_cache_builder.py",
    "scripts/show_base/run_lower_target_cache_builder_guarded.sh",
    "scripts/show_base/check_lower_target_joints_cache.py",
    "scripts/show_base/run_lower_target_cache_formal_gate.py",
    "configs/cnn_vqvae_lower_30.yaml",
)
EXPECTED_RUNNER = "/tmp/globaldiff_guarded_runner.py"
EXPECTED_RUNNER_GPUS = "0,1,2,3,4,5,6,7"


class GateError(RuntimeError):
    """A formal gate condition failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def argv_bytes(argv: Sequence[str]) -> bytes:
    if not argv or any(type(item) is not str or "\0" in item for item in argv):
        raise GateError("argv must be non-empty NUL-free strings")
    return b"\0".join(os.fsencode(item) for item in argv) + b"\0"


def argv_sha256(argv: Sequence[str]) -> str:
    return hashlib.sha256(argv_bytes(argv)).hexdigest()


def require_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str):
        raise GateError(f"{label} must be a lowercase hexadecimal string")
    normalized = value.strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise GateError(f"{label} must contain exactly {length} lowercase hex")
    return normalized


def require_exact_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise GateError(f"{label}={value!r}, expected exact integer {expected}")
    return value


def require_finite_positive(value: Any, label: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise GateError(f"{label} must be an exact finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GateError(f"{label} must be finite and positive")
    return result


def read_regular_bytes(path_value: Path, label: str) -> tuple[Path, bytes]:
    if path_value.is_symlink():
        raise GateError(f"{label} must not be a symlink: {path_value}")
    path = path_value.resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GateError(f"cannot open regular {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GateError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
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
            raise GateError(f"{label} changed while being read: {path}")
        return path, payload
    finally:
        os.close(descriptor)


def verified_json(
    path_value: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    expected = require_hex(expected_sha256, 64, f"{label} SHA-256")
    path, payload = read_regular_bytes(path_value, label)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise GateError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise GateError(f"{label} must contain one JSON object")
    return path, value, observed


def atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
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


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _proc_identity(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 1:
        raise GateError(f"invalid process identity PID: {pid!r}")
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text()
        cmdline = (proc / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError) as error:
        raise GateError(f"process disappeared during identity read: {pid}") from error
    argv = [
        item.decode("utf-8", "surrogateescape")
        for item in cmdline.split(b"\0")
        if item
    ]
    close_parenthesis = stat_text.rfind(")")
    if close_parenthesis < 0:
        raise GateError(f"invalid /proc stat record for PID {pid}")
    stat_tail = stat_text[close_parenthesis + 2 :].split()
    # stat_tail starts at field 3 (state): PPID is offset 1 and process
    # starttime (field 22) is offset 19.  Parsing after the final ')' remains
    # correct even when the kernel comm field contains spaces or parentheses.
    if len(stat_tail) <= 19 or not argv:
        raise GateError(f"incomplete process identity for PID {pid}")
    return {
        "pid": pid,
        "ppid": int(stat_tail[1]),
        "starttime": stat_tail[19],
        "argv": argv,
        "argv_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def guarded_ancestry(
    parsed: argparse.Namespace,
    *,
    internal: bool,
) -> list[dict[str, Any]]:
    expected_pid = parsed.expected_runner_pid
    if type(expected_pid) is not int or expected_pid <= 1:
        raise GateError("expected runner PID must be greater than one")
    expected_starttime = str(parsed.expected_runner_starttime)
    if not expected_starttime.isdigit():
        raise GateError("expected runner starttime must be /proc clock ticks")
    expected_argv_sha = require_hex(
        parsed.expected_runner_argv_sha256,
        64,
        "expected runner argv SHA-256",
    )
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        identity = _proc_identity(pid)
        chain.append(identity)
        pid = identity["ppid"]
    runner_index = 2 if internal else 1
    if len(chain) <= runner_index:
        raise GateError("guarded-runner ancestry is shorter than expected")
    runner = chain[runner_index]
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
        runner["pid"] != expected_pid
        or runner["starttime"] != expected_starttime
        or runner["argv_sha256"] != expected_argv_sha
        or EXPECTED_RUNNER not in runner["argv"]
        or not gpu_reservation
    ):
        raise GateError("exact guarded-runner PID/starttime/argv binding mismatch")
    if internal:
        if (
            chain[1]["pid"] != int(parsed.internal_parent_pid)
            or chain[1]["ppid"] != expected_pid
        ):
            raise GateError("child -> gate -> runner exact ancestor chain mismatch")
    elif chain[0]["ppid"] != expected_pid:
        raise GateError("public gate is not the guarded runner's immediate child")
    return chain


def source_receipt(
    repository: Path,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    commit = require_hex(expected_commit, 40, "expected source commit")
    tree = require_hex(expected_tree, 40, "expected source tree")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise GateError(
            "formal gate requires a clean checkout; first change: "
            f"{status.splitlines()[0]}"
        )
    receipt = {
        "origin": _git(repository, "remote", "get-url", "origin"),
        "commit": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "critical_files": {
            relative: sha256_file(repository / relative)
            for relative in CRITICAL_SOURCE_FILES
        },
    }
    if receipt["origin"] != EXPECTED_ORIGIN:
        raise GateError(f"unexpected origin: {receipt['origin']!r}")
    if receipt["commit"] != commit or receipt["tree"] != tree:
        raise GateError("source commit/tree does not match the explicit receipt")
    return receipt


def _exact_speaker_contract(value: Any, label: str) -> None:
    if type(value) is not dict or set(value) != set(EXPECTED_SPEAKERS):
        raise GateError(f"{label} must contain exactly all four SHOW speakers")
    for speaker, expected_id in EXPECTED_SPEAKERS.items():
        require_exact_int(value[speaker], expected_id, f"{label}.{speaker}")


def validate_static_gate_contract() -> dict[str, Any]:
    """Pure-Python protocol audit used by unit tests and the formal run."""

    if (
        ABBA_ORDER != ("legacy", "cache", "cache", "legacy")
        or len(ABBA_ORDER) != 4
        or ABBA_ORDER.count("legacy") != 2
        or ABBA_ORDER.count("cache") != 2
    ):
        raise GateError("ABBA order contract changed")
    if (
        tuple(item[1] for item in EQUIVALENCE_CHILDREN)
        != ("legacy", "legacy", "cache", "cache")
        or tuple(item[2] for item in EQUIVALENCE_CHILDREN) != (0, 1, 0, 1)
        or tuple(item[1] for item in BENCHMARK_CHILDREN) != ABBA_ORDER
        or tuple(item[2] for item in BENCHMARK_CHILDREN) != (0, 1, 2, 3)
        or len(set(INTERNAL_MODES)) != 8
    ):
        raise GateError("fresh-process child protocol changed")
    require_exact_int(WARMUP_UPDATES, 5, "warmup updates")
    require_exact_int(MEASURED_UPDATES, 25, "measured updates")
    require_exact_int(EQUIVALENCE_UPDATES, 2, "equivalence updates")
    require_exact_int(EXPECTED_EPOCHS, 600, "lower epochs")
    require_exact_int(
        EXPECTED_UPDATES_PER_EPOCH,
        1_989,
        "updates per epoch",
    )
    require_finite_positive(MIN_SPEEDUP, "minimum speedup")
    if MIN_SPEEDUP != 1.05:
        raise GateError("minimum speedup must remain exactly 1.05")
    return {
        "speaker_scope": "All",
        "speakers": dict(EXPECTED_SPEAKERS),
        "equivalence_updates": EQUIVALENCE_UPDATES,
        "equivalence_fresh_processes": 4,
        "equivalence_repeats_per_mode": 2,
        "abba_order": list(ABBA_ORDER),
        "benchmark_fresh_processes": 4,
        "warmup_updates_per_block": WARMUP_UPDATES,
        "measured_updates_per_block": MEASURED_UPDATES,
        "minimum_speedup": MIN_SPEEDUP,
        "training_updates": EXPECTED_EPOCHS * EXPECTED_UPDATES_PER_EPOCH,
    }


def calculate_speedup_report(
    blocks: Sequence[Mapping[str, Any]],
    *,
    full_builder_seconds: float,
    minimum_speedup: float = MIN_SPEEDUP,
    epochs: int = EXPECTED_EPOCHS,
    updates_per_epoch: int = EXPECTED_UPDATES_PER_EPOCH,
) -> dict[str, Any]:
    """Validate exact ABBA samples and calculate every fail-closed threshold."""

    if len(blocks) != 4:
        raise GateError("benchmark must contain exactly four ABBA blocks")
    minimum = require_finite_positive(minimum_speedup, "minimum speedup")
    builder = require_finite_positive(
        full_builder_seconds,
        "full builder seconds",
    )
    require_exact_int(epochs, EXPECTED_EPOCHS, "amortization epochs")
    require_exact_int(
        updates_per_epoch,
        EXPECTED_UPDATES_PER_EPOCH,
        "amortization updates per epoch",
    )
    normalized: list[dict[str, Any]] = []
    for index, (block, expected_mode) in enumerate(zip(blocks, ABBA_ORDER)):
        if type(block) is not dict:
            raise GateError(f"benchmark block {index} must be an object")
        if block.get("mode") != expected_mode:
            raise GateError(
                f"benchmark block {index} mode {block.get('mode')!r} "
                f"!= {expected_mode!r}"
            )
        require_exact_int(
            block.get("warmup_updates"),
            WARMUP_UPDATES,
            f"benchmark block {index} warmup_updates",
        )
        require_exact_int(
            block.get("measured_updates"),
            MEASURED_UPDATES,
            f"benchmark block {index} measured_updates",
        )
        wall = block.get("wall_seconds")
        cuda = block.get("cuda_event_seconds")
        if (
            type(wall) is not list
            or type(cuda) is not list
            or len(wall) != MEASURED_UPDATES
            or len(cuda) != MEASURED_UPDATES
        ):
            raise GateError(
                f"benchmark block {index} lacks 25 wall/CUDA measurements"
            )
        wall_values = [
            require_finite_positive(value, f"block {index} wall[{offset}]")
            for offset, value in enumerate(wall)
        ]
        cuda_values = [
            require_finite_positive(value, f"block {index} cuda[{offset}]")
            for offset, value in enumerate(cuda)
        ]
        normalized.append(
            {
                "block_index": index,
                "mode": expected_mode,
                "warmup_updates": WARMUP_UPDATES,
                "measured_updates": MEASURED_UPDATES,
                "wall_seconds": wall_values,
                "cuda_event_seconds": cuda_values,
                "wall_total_seconds": sum(wall_values),
                "cuda_event_total_seconds": sum(cuda_values),
                "wall_mean_seconds": sum(wall_values) / len(wall_values),
                "cuda_event_mean_seconds": sum(cuda_values) / len(cuda_values),
            }
        )

    legacy = [normalized[0], normalized[3]]
    cache = [normalized[1], normalized[2]]
    pooled: dict[str, Any] = {}
    for label in ("wall", "cuda_event"):
        key = f"{label}_seconds"
        legacy_values = [
            value for block in legacy for value in block[key]
        ]
        cache_values = [value for block in cache for value in block[key]]
        legacy_total = sum(legacy_values)
        cache_total = sum(cache_values)
        speedup = legacy_total / cache_total
        if not math.isfinite(speedup) or speedup < minimum:
            raise GateError(
                f"pooled {label} speedup {speedup:.9f} < {minimum:.9f}"
            )
        pooled[label] = {
            "legacy_samples": len(legacy_values),
            "cache_samples": len(cache_values),
            "legacy_total_seconds": legacy_total,
            "cache_total_seconds": cache_total,
            "legacy_mean_seconds": legacy_total / len(legacy_values),
            "cache_mean_seconds": cache_total / len(cache_values),
            "speedup": speedup,
            "threshold": minimum,
            "pass": True,
        }

    # Chronological pairs remove first/last-position bias: A0/B1 and A3/B2.
    pair_indices = ((0, 1), (3, 2))
    block_pairs: list[dict[str, Any]] = []
    for pair_index, (legacy_index, cache_index) in enumerate(pair_indices):
        item: dict[str, Any] = {
            "pair_index": pair_index,
            "legacy_block": legacy_index,
            "cache_block": cache_index,
        }
        for label in ("wall", "cuda_event"):
            legacy_total = normalized[legacy_index][f"{label}_total_seconds"]
            cache_total = normalized[cache_index][f"{label}_total_seconds"]
            speedup = legacy_total / cache_total
            if not math.isfinite(speedup) or speedup < minimum:
                raise GateError(
                    f"paired block {pair_index} {label} speedup "
                    f"{speedup:.9f} < {minimum:.9f}"
                )
            item[label] = {
                "legacy_total_seconds": legacy_total,
                "cache_total_seconds": cache_total,
                "speedup": speedup,
                "threshold": minimum,
                "pass": True,
            }
        block_pairs.append(item)

    training_updates = epochs * updates_per_epoch
    legacy_wall_per_update = pooled["wall"]["legacy_mean_seconds"]
    cache_wall_per_update = pooled["wall"]["cache_mean_seconds"]
    projected_legacy = legacy_wall_per_update * training_updates
    projected_cache = builder + cache_wall_per_update * training_updates
    amortized_speedup = projected_legacy / projected_cache
    if (
        not math.isfinite(amortized_speedup)
        or amortized_speedup < minimum
    ):
        raise GateError(
            "builder-amortized wall speedup "
            f"{amortized_speedup:.9f} < {minimum:.9f}"
        )
    return {
        "status": "pass",
        "order": list(ABBA_ORDER),
        "blocks": normalized,
        "pooled": pooled,
        "block_pairs": block_pairs,
        "amortization": {
            "full_builder_seconds": builder,
            "epochs": epochs,
            "updates_per_epoch": updates_per_epoch,
            "training_updates": training_updates,
            "legacy_wall_seconds_per_update": legacy_wall_per_update,
            "cache_wall_seconds_per_update": cache_wall_per_update,
            "projected_legacy_seconds": projected_legacy,
            "projected_cache_seconds_including_builder": projected_cache,
            "speedup": amortized_speedup,
            "threshold": minimum,
            "pass": True,
        },
    }


def _tensor_bytes(tensor: Any, torch: Any) -> bytes:
    contiguous = tensor.detach().cpu().contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def _canonical_tree_update(
    digest: Any,
    value: Any,
    *,
    torch: Any,
    np: Any,
) -> None:
    """Hash state by explicit type/schema/raw bytes, never pickle identity."""

    if torch.is_tensor(value):
        raw = _tensor_bytes(value, torch)
        digest.update(b"T")
        digest.update(str(value.dtype).encode())
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        raw = array.tobytes(order="C")
        digest.update(b"N")
        digest.update(array.dtype.str.encode())
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if isinstance(value, np.generic):
        _canonical_tree_update(
            digest,
            np.asarray(value),
            torch=torch,
            np=np,
        )
        return
    if value is None:
        digest.update(b"0")
        return
    if type(value) is bool:
        digest.update(b"B1" if value else b"B0")
        return
    if type(value) is int:
        digest.update(b"I" + str(value).encode() + b";")
        return
    if type(value) is float:
        digest.update(b"F" + value.hex().encode() + b";")
        return
    if type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
        return
    if type(value) is bytes:
        digest.update(b"Y" + len(value).to_bytes(8, "big") + value)
        return
    if isinstance(value, Mapping):
        digest.update(b"D" + len(value).to_bytes(8, "big"))
        ordered = []
        for key in value:
            key_digest = hashlib.sha256()
            _canonical_tree_update(
                key_digest,
                key,
                torch=torch,
                np=np,
            )
            ordered.append((key_digest.digest(), key, value[key]))
        for key_digest, key, child in sorted(ordered, key=lambda item: item[0]):
            digest.update(key_digest)
            _canonical_tree_update(
                digest,
                key,
                torch=torch,
                np=np,
            )
            _canonical_tree_update(
                digest,
                child,
                torch=torch,
                np=np,
            )
        return
    if isinstance(value, tuple):
        digest.update(b"U" + len(value).to_bytes(8, "big"))
        for child in value:
            _canonical_tree_update(
                digest,
                child,
                torch=torch,
                np=np,
            )
        return
    if isinstance(value, list):
        digest.update(b"L" + len(value).to_bytes(8, "big"))
        for child in value:
            _canonical_tree_update(
                digest,
                child,
                torch=torch,
                np=np,
            )
        return
    raise GateError(f"unsupported state leaf: {type(value).__name__}")


def canonical_tree_sha256(value: Any, *, torch: Any, np: Any) -> str:
    digest = hashlib.sha256()
    _canonical_tree_update(digest, value, torch=torch, np=np)
    return digest.hexdigest()


def assert_tree_byte_exact(
    left: Any,
    right: Any,
    *,
    torch: Any,
    np: Any,
    path: str = "state",
) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise GateError(f"{path}: tensor/non-tensor type mismatch")
        if (
            left.dtype != right.dtype
            or tuple(left.shape) != tuple(right.shape)
            or _tensor_bytes(left, torch) != _tensor_bytes(right, torch)
        ):
            raise GateError(f"{path}: tensor bytes differ")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C")
        ):
            raise GateError(f"{path}: ndarray bytes differ")
        return
    if type(left) is not type(right):
        raise GateError(
            f"{path}: type mismatch {type(left).__name__}/"
            f"{type(right).__name__}"
        )
    if isinstance(left, Mapping):
        if set(left) != set(right):
            raise GateError(f"{path}: mapping keys differ")
        for key in left:
            assert_tree_byte_exact(
                left[key],
                right[key],
                torch=torch,
                np=np,
                path=f"{path}[{key!r}]",
            )
        return
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise GateError(f"{path}: sequence lengths differ")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_tree_byte_exact(
                left_item,
                right_item,
                torch=torch,
                np=np,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(left, float):
        if left.hex() != right.hex():
            raise GateError(f"{path}: float bytes differ")
        return
    if left != right:
        raise GateError(f"{path}: scalar values differ")


def _cpu_clone(value: Any, *, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(child, torch=torch) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(child, torch=torch) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(child, torch=torch) for child in value)
    return pickle.loads(pickle.dumps(value, protocol=5))


def _gradient_state(model: Any) -> dict[str, Any]:
    return {
        name: None if parameter.grad is None else parameter.grad
        for name, parameter in model.named_parameters()
    }


def _rvq_ema_state(model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "QuantizeEMAReset":
            continue
        item: dict[str, Any] = {"init": bool(module.init)}
        if item["init"]:
            if module.code_sum is None or module.code_count is None:
                raise GateError(f"incomplete RVQ EMA state: {name}")
            item["code_sum"] = module.code_sum
            item["code_count"] = module.code_count
        state[name] = item
    if len(state) != 6:
        raise GateError(f"lower RVQ must expose six EMA modules, got {len(state)}")
    return state


def _tracker_state(trainer: Any) -> dict[str, Any]:
    tracker = trainer.tracker
    return {
        "metric_names": list(tracker.metric_names),
        "states": list(tracker.states),
        "types": list(tracker.types),
        "values": tracker.values,
        "meters": {
            metric_name: {
                state: {
                    "name": tracker.loss_meters[metric_name][state].name,
                    "fmt": tracker.loss_meters[metric_name][state].fmt,
                    "val": tracker.loss_meters[metric_name][state].val,
                    "avg": tracker.loss_meters[metric_name][state].avg,
                    "sum": tracker.loss_meters[metric_name][state].sum,
                    "count": tracker.loss_meters[metric_name][state].count,
                }
                for state in tracker.states
            }
            for metric_name in tracker.metric_names
        },
        "train_history": tracker.train_history,
        "val_history": tracker.val_history,
        "formal_metric_sums": trainer._formal_train_metric_sums,
        "formal_metric_counts": trainer._formal_train_metric_counts,
    }


def _scheduler_state(scheduler: Any) -> dict[str, Any]:
    state = scheduler.state_dict()
    if not isinstance(state, dict):
        raise GateError("scheduler state_dict is not an object")
    # A temporary exact-step wrapper is installed as an instance attribute and
    # therefore appears in PyTorch scheduler state_dict.  Exclude only that
    # gate-owned callable after verifying exact identity.
    override = vars(scheduler).get("step")
    if override is not None:
        if not callable(override) or state.get("step") is not override:
            raise GateError("scheduler step audit wrapper identity changed")
        state = dict(state)
        del state["step"]
    elif "step" in state:
        raise GateError("production scheduler unexpectedly serializes step")
    return state


def _rng_state(torch: Any, np: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _smplx_state(trainer: Any) -> dict[str, Any]:
    model = trainer.smplx
    metadata = {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "grad_is_none": parameter.grad is None,
        }
        for name, parameter in model.named_parameters()
    }
    if (
        model.training
        or not metadata
        or any(item["requires_grad"] for item in metadata.values())
        or any(not item["grad_is_none"] for item in metadata.values())
    ):
        raise GateError("formal lower SMPL-X must remain frozen/eval/no-grad")
    return {
        "state_dict": model.state_dict(),
        "grads": _gradient_state(model),
        "training": bool(model.training),
        "metadata": metadata,
    }


def _snapshot_state(
    trainer: Any,
    *,
    torch: Any,
    np: Any,
    batch_indices: list[int],
) -> dict[str, Any]:
    value = {
        "model": trainer.model.state_dict(),
        "grads": _gradient_state(trainer.model),
        "optimizer": trainer.opt.state_dict(),
        "scheduler": _scheduler_state(trainer.opt_s),
        "rvq_ema": _rvq_ema_state(trainer.model),
        "rng": _rng_state(torch, np),
        "tracker": _tracker_state(trainer),
        "formal_optimizer_updates": int(trainer.formal_optimizer_updates),
        "smplx": _smplx_state(trainer),
        "batch_indices": list(batch_indices),
    }
    return _cpu_clone(value, torch=torch)


def _assert_finite_training_state(
    value: Any,
    *,
    torch: Any,
    np: Any,
    path: str = "state",
) -> None:
    """Reject non-finite state except EpochTracker's untouched +/-inf values."""

    if path.endswith("['tracker']['values']"):
        return
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise GateError(f"{path}: non-finite tensor")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise GateError(f"{path}: non-finite ndarray")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_training_state(
                child,
                torch=torch,
                np=np,
                path=f"{path}[{key!r}]",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_training_state(
                child,
                torch=torch,
                np=np,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise GateError(f"{path}: non-finite scalar")


@contextlib.contextmanager
def _temporary_argv(arguments: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = arguments
    try:
        yield
    finally:
        sys.argv = previous


class _NullWriter:
    def add_scalar(self, *_: Any, **__: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _internal_mode_contract(mode: str) -> tuple[str, str, int]:
    for child_mode, target_mode, ordinal in EQUIVALENCE_CHILDREN:
        if mode == child_mode:
            return "equivalence", target_mode, ordinal
    for child_mode, target_mode, ordinal in BENCHMARK_CHILDREN:
        if mode == child_mode:
            return "benchmark", target_mode, ordinal
    raise GateError(f"invalid internal mode: {mode!r}")


def _build_lower_args(
    parsed: argparse.Namespace,
    *,
    output: Path,
    cache_enabled: bool,
) -> Any:
    from utils import config

    repository = Path(__file__).resolve().parents[2]
    argv = [
        str(Path(__file__).resolve()),
        "--config",
        str(repository / "configs/cnn_vqvae_lower_30.yaml"),
        "--formal_stage",
        "lower",
        "--train_only",
        "true",
        "--train_rvq",
        "--dataset",
        "show_base",
        "--training_speakers",
        "0",
        "1",
        "2",
        "3",
        "--train_path",
        str(Path(parsed.representation_lmdb).resolve()),
        "--dataset_summary",
        str(Path(parsed.representation_summary).resolve()),
        "--lineage_manifest",
        str(Path(parsed.representation_lineage).resolve()),
        "--data_path_1",
        str(Path(parsed.asset_root).resolve()) + "/",
        "--expected_smplx_asset_sha256",
        str(parsed.expected_smplx_asset_sha256),
        "--out_path",
        str(output) + "/",
        "--run_name",
        f"lower_cache_gate_{parsed.internal_mode}",
        "--epochs",
        str(EXPECTED_EPOCHS),
        "--batch_size",
        str(EXPECTED_BATCH_SIZE),
        "--log_period",
        str(EXPECTED_UPDATES_PER_EPOCH),
        "--loader_workers",
        "0",
        "--random_seed",
        "2021",
        "--debug",
        "false",
        "--pretrain",
        "false",
        "--sparse",
        "0",
        "--word_cache",
        "false",
        "--word_rep",
        "disabled_zero_placeholder",
        "--t_pre_encoder",
        "disabled",
        "--word_index_num",
        "0",
        "--word_dims",
        "0",
        "--word_f",
        "0",
        "--freeze_wordembed",
        "true",
        "--hubert_mean_path",
        "",
        "--hubert_std_path",
        "",
        "--audio_infer_path",
        "",
        "--base_ckpt",
        "",
        "--test_ckpt",
        "",
        "--load_ckpt",
        "",
    ]
    if cache_enabled:
        argv.extend(
            [
                "--use_lower_target_joints_cache",
                "true",
                "--lower_target_joints_cache",
                str(Path(parsed.cache_lmdb).resolve()),
                "--lower_target_joints_cache_manifest",
                str(Path(parsed.cache_manifest).resolve()),
                "--expected_lower_target_joints_cache_manifest_sha256",
                str(parsed.expected_cache_manifest_sha256),
                "--lower_target_joints_cache_checker_receipt",
                str(Path(parsed.checker_receipt).resolve()),
                "--expected_lower_target_joints_cache_checker_sha256",
                str(parsed.expected_checker_receipt_sha256),
            ]
        )
    with _temporary_argv(argv):
        args = config.parse_args()
    args.local_rank = 0
    args.ddp = False
    args.gpus = [0]
    args.skip_test_init = True
    if (
        args.formal_stage != "lower"
        or args.dataset != "show_base"
        or args.trainer != "aelower"
        or args.tar_joints != "beat_smplx_lower"
        or args.g_name != "RVQVAE"
        or not bool(args.train_rvq)
        or int(args.batch_size) != EXPECTED_BATCH_SIZE
        or int(args.pose_length) != EXPECTED_FRAMES
        or list(args.training_speakers) != list(EXPECTED_SPEAKER_IDS)
        or int(args.epochs) != EXPECTED_EPOCHS
        or int(args.sparse) != 0
        or bool(args.use_lower_target_joints_cache) is not cache_enabled
        or not bool(args.train_only)
    ):
        raise GateError("parsed lower formal configuration changed")
    return args


def _semantic_receipts(
    parsed: argparse.Namespace,
    *,
    args: Any,
    source: dict[str, Any],
    torch: Any,
    target_mode: str,
    kind: str,
) -> dict[str, Any]:
    if target_mode not in {"legacy", "cache"}:
        raise GateError(f"invalid semantic target mode: {target_mode!r}")
    return {
        "common": {
            "format": FORMAT,
            "kind": kind,
            "source": source,
            "representation": {
                "lmdb": str(Path(parsed.representation_lmdb).resolve()),
                "summary": str(
                    Path(parsed.representation_summary).resolve()
                ),
                "lineage": str(
                    Path(parsed.representation_lineage).resolve()
                ),
                "data_mdb_sha256": (
                    parsed.expected_representation_data_sha256
                ),
                "summary_sha256": (
                    parsed.expected_representation_summary_sha256
                ),
                "lineage_sha256": (
                    parsed.expected_representation_lineage_sha256
                ),
                "entry_aggregate_sha256": (
                    parsed.expected_representation_entry_aggregate_sha256
                ),
            },
            "lower_target_cache": {
                "lmdb": str(Path(parsed.cache_lmdb).resolve()),
                "manifest": str(Path(parsed.cache_manifest).resolve()),
                "manifest_sha256": parsed.expected_cache_manifest_sha256,
                "checker_receipt": str(
                    Path(parsed.checker_receipt).resolve()
                ),
                "checker_receipt_sha256": (
                    parsed.expected_checker_receipt_sha256
                ),
                "builder_process_receipt": str(
                    Path(parsed.builder_process_receipt).resolve()
                ),
                "builder_process_receipt_sha256": (
                    parsed.expected_builder_process_receipt_sha256
                ),
            },
            "smplx": {
                "asset_root": str(Path(parsed.asset_root).resolve()),
                "asset_sha256": parsed.expected_smplx_asset_sha256,
            },
            "configuration": {
                "dataset": args.dataset,
                "formal_stage": args.formal_stage,
                "trainer": args.trainer,
                "model": args.model,
                "g_name": args.g_name,
                "tar_joints": args.tar_joints,
                "training_speakers": list(args.training_speakers),
                "batch_size": int(args.batch_size),
                "pose_length": int(args.pose_length),
                "epochs": int(args.epochs),
                "train_only": bool(args.train_only),
                "train_rvq": bool(args.train_rvq),
                "sparse": int(args.sparse),
                "random_seed": int(args.random_seed),
            },
            "runtime": {
                "device_name": torch.cuda.get_device_name(0),
                "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda),
                "cudnn_version": int(torch.backends.cudnn.version()),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": bool(
                    torch.backends.cudnn.deterministic
                ),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
            },
        },
        "mode": {
            "target_mode": target_mode,
            "cache_enabled": target_mode == "cache",
        },
    }


def _speaker_id(dataset: Any, index: int, *, np: Any) -> int:
    raw = dataset._read(index, ("speaker_id",))["speaker_id"]
    array = np.asarray(raw)
    if (
        array.shape != (EXPECTED_FRAMES, 1)
        or array.dtype.kind not in "iu"
        or not bool((array == array[0, 0]).all())
    ):
        raise GateError(f"sample {index}: invalid speaker_id field")
    speaker = int(array[0, 0])
    if speaker not in EXPECTED_SPEAKER_IDS:
        raise GateError(f"sample {index}: forbidden speaker ID {speaker}")
    return speaker


def _select_real_indices(
    dataset: Any,
    *,
    per_speaker: int,
    np: Any,
) -> list[int]:
    if per_speaker <= 0:
        raise ValueError("per_speaker must be positive")
    length = len(dataset)
    require_exact_int(length, EXPECTED_ENTRIES, "representation entries")
    selected: dict[int, list[int]] = {
        speaker: [] for speaker in EXPECTED_SPEAKER_IDS
    }
    # Probe the complete key range at fixed resolution.  SHOW windows are
    # emitted in sorted clip order, so this locates every large speaker region
    # without decoding all 127309 NPZ records.  A complete fallback remains
    # fail-safe if ordering ever changes.
    probe_count = min(length, 4_096)
    probes = np.linspace(0, length - 1, probe_count, dtype=np.int64)
    seeds: dict[int, int] = {}
    for probe in probes.tolist():
        speaker = _speaker_id(dataset, int(probe), np=np)
        seeds.setdefault(speaker, int(probe))
        if len(seeds) == len(EXPECTED_SPEAKER_IDS):
            break
    for speaker, seed in seeds.items():
        for direction in (1, -1):
            index = seed if direction == 1 else seed - 1
            while (
                0 <= index < length
                and len(selected[speaker]) < per_speaker
            ):
                observed = _speaker_id(dataset, index, np=np)
                if observed != speaker:
                    break
                selected[speaker].append(index)
                index += direction
    if any(len(values) < per_speaker for values in selected.values()):
        selected = {speaker: [] for speaker in EXPECTED_SPEAKER_IDS}
        for index in range(length):
            speaker = _speaker_id(dataset, index, np=np)
            if len(selected[speaker]) < per_speaker:
                selected[speaker].append(index)
            if all(
                len(values) == per_speaker for values in selected.values()
            ):
                break
    if any(len(values) != per_speaker for values in selected.values()):
        raise GateError("cannot select enough real windows for every SHOW speaker")
    batches = per_speaker // (EXPECTED_BATCH_SIZE // 4)
    if (
        per_speaker % (EXPECTED_BATCH_SIZE // 4) != 0
        or batches <= 0
    ):
        raise GateError("per-speaker selection cannot form balanced batches")
    result: list[int] = []
    per_batch = EXPECTED_BATCH_SIZE // 4
    for batch_index in range(batches):
        for speaker in EXPECTED_SPEAKER_IDS:
            start = batch_index * per_batch
            result.extend(selected[speaker][start:start + per_batch])
    if len(result) != batches * EXPECTED_BATCH_SIZE or len(set(result)) != len(
        result
    ):
        raise GateError("real batch index selection is not unique/exact")
    return result


class _FixedRealDataset:
    def __init__(self, base: Any, indices: Sequence[int], *, np: Any):
        self.base = base
        self.indices = [int(index) for index in indices]
        self.np = np

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, ordinal: int) -> dict[str, Any]:
        index = self.indices[ordinal]
        item = dict(self.base[index])
        item["speaker_id"] = self.base._read(
            index,
            ("speaker_id",),
        )["speaker_id"]
        return item


def _make_loader(
    trainer: Any,
    indices: list[int],
    *,
    torch: Any,
    np: Any,
) -> Any:
    dataset = _FixedRealDataset(trainer.train_data, indices, np=np)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=EXPECTED_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )


def _batch_speaker_audit(batch: Mapping[str, Any], *, torch: Any) -> list[int]:
    speaker = batch.get("speaker_id")
    sample_index = batch.get("sample_index")
    if not torch.is_tensor(speaker) or not torch.is_tensor(sample_index):
        raise GateError("real lower batch lacks speaker_id/sample_index")
    speakers = sorted(
        int(value)
        for value in torch.unique(speaker.detach().cpu()).tolist()
    )
    if speakers != list(EXPECTED_SPEAKER_IDS):
        raise GateError(f"real lower batch speaker coverage changed: {speakers}")
    if int(sample_index.numel()) != EXPECTED_BATCH_SIZE:
        raise GateError("real lower batch does not contain exactly 64 windows")
    return speakers


class _ObservedLoader:
    def __init__(self, loader: Any, *, torch: Any):
        self.loader = loader
        self.torch = torch
        self.batches: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        for batch in self.loader:
            sample_indices = [
                int(value)
                for value in batch["sample_index"].detach().cpu().tolist()
            ]
            speakers = _batch_speaker_audit(batch, torch=self.torch)
            self.batches.append(
                {
                    "batch_index": len(self.batches),
                    "sample_indices": sample_indices,
                    "speaker_ids": speakers,
                }
            )
            yield batch


@contextlib.contextmanager
def _count_steps(
    target: Any,
    label: str,
    *,
    after_step: Any | None = None,
) -> Iterator[dict[str, int]]:
    if "step" in vars(target):
        raise GateError(f"{label}: pre-existing instance step override")
    original = target.step
    counter = {"count": 0}

    def counted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        counter["count"] += 1
        if after_step is not None:
            after_step(counter["count"])
        return result

    target.step = counted
    try:
        yield counter
    finally:
        if vars(target).get("step") is not counted:
            raise GateError(f"{label}: step wrapper identity changed")
        del vars(target)["step"]


@contextlib.contextmanager
def _observe_completed_formal_updates(
    trainer: Any,
    *,
    after_update: Any,
) -> Iterator[dict[str, int]]:
    method_name = "_formal_optimizer_step"
    if method_name in vars(trainer):
        raise GateError("pre-existing formal optimizer-step instance override")
    original = trainer._formal_optimizer_step
    initial_updates = int(trainer.formal_optimizer_updates)
    counter = {"count": 0}

    def observed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        counter["count"] += 1
        expected_updates = initial_updates + counter["count"]
        if int(trainer.formal_optimizer_updates) != expected_updates:
            raise GateError(
                "formal optimizer counter was not updated before snapshot"
            )
        after_update(counter["count"])
        return result

    trainer._formal_optimizer_step = observed
    try:
        yield counter
    finally:
        if vars(trainer).get(method_name) is not observed:
            raise GateError("formal optimizer-step wrapper identity changed")
        del vars(trainer)[method_name]


@contextlib.contextmanager
def _observe_target_joints(
    *,
    mode: str,
    trainer: Any,
    trainer_module: Any,
    output: Path,
    torch: Any,
) -> Iterator[list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []

    def write_target(value: Any, call_index: int) -> None:
        normalized = value.detach().reshape(-1, 127, 3).cpu().contiguous()
        if (
            normalized.dtype != torch.float32
            or normalized.requires_grad
            or normalized.grad_fn is not None
            or not bool(torch.isfinite(normalized).all())
        ):
            raise GateError("observed target joints violate finite no-grad float32")
        raw = _tensor_bytes(normalized, torch)
        path = output / f"target_{call_index:02d}.bin"
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        observations.append(
            {
                "call_index": call_index,
                "normalized_shape": list(normalized.shape),
                "dtype": str(normalized.dtype),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "path": str(path),
            }
        )

    if mode == "legacy":
        original = trainer_module.smplx_target_forward

        def observed(*args: Any, **kwargs: Any) -> Any:
            value = original(*args, **kwargs)
            write_target(value["joints"], len(observations))
            return value

        trainer_module.smplx_target_forward = observed
        try:
            yield observations
        finally:
            if trainer_module.smplx_target_forward is not observed:
                raise GateError("legacy target helper wrapper changed")
            trainer_module.smplx_target_forward = original
        return

    if mode != "cache" or trainer.lower_target_joints_cache is None:
        raise GateError(f"invalid target observation mode: {mode}")
    cache = trainer.lower_target_joints_cache
    original_instance = vars(cache).get("index_select")
    if original_instance is not None:
        raise GateError("cache has an unexpected index_select override")
    original = cache.index_select

    def observed(this: Any, indices: Any) -> Any:
        if this is not cache:
            raise GateError("cache index_select wrapper rebound")
        value = original(indices)
        write_target(value, len(observations))
        return value

    cache.index_select = types.MethodType(observed, cache)
    try:
        yield observations
    finally:
        wrapped = vars(cache).get("index_select")
        if wrapped is None or getattr(wrapped, "__func__", None) is not observed:
            raise GateError("cache target wrapper changed")
        del vars(cache)["index_select"]


def _save_state(
    path: Path,
    state: dict[str, Any],
    *,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    _assert_finite_training_state(state, torch=torch, np=np)
    torch.save(state, path)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "canonical_state_sha256": canonical_tree_sha256(
            state,
            torch=torch,
            np=np,
        ),
    }


def _run_equivalence_child(
    parsed: argparse.Namespace,
    *,
    mode: str,
    repeat_index: int,
    semantic_receipts: dict[str, Any],
    torch: Any,
    np: Any,
    trainer: Any,
    trainer_module: Any,
    snapshot: Path,
) -> dict[str, Any]:
    per_speaker = EQUIVALENCE_UPDATES * (EXPECTED_BATCH_SIZE // 4)
    indices = _select_real_indices(
        trainer.train_data,
        per_speaker=per_speaker,
        np=np,
    )
    loader = _ObservedLoader(
        _make_loader(trainer, indices, torch=torch, np=np),
        torch=torch,
    )
    trainer.train_loader = loader
    trainer.train_length = EQUIVALENCE_UPDATES
    states: dict[str, Any] = {}
    initial = _snapshot_state(
        trainer,
        torch=torch,
        np=np,
        batch_indices=indices,
    )
    states["initial"] = _save_state(
        snapshot / "state_initial.pt",
        initial,
        torch=torch,
        np=np,
    )

    def after_step(step: int) -> None:
        state = _snapshot_state(
            trainer,
            torch=torch,
            np=np,
            batch_indices=indices,
        )
        states[f"step_{step}"] = _save_state(
            snapshot / f"state_step_{step}.pt",
            state,
            torch=torch,
            np=np,
        )

    with _count_steps(
        trainer.opt,
        f"{mode} optimizer",
    ) as optimizer:
        with _observe_completed_formal_updates(
            trainer,
            after_update=after_step,
        ) as completed_updates:
            with _count_steps(
                trainer.opt_s,
                f"{mode} scheduler",
            ) as scheduler:
                with _observe_target_joints(
                    mode=mode,
                    trainer=trainer,
                    trainer_module=trainer_module,
                    output=snapshot,
                    torch=torch,
                ) as targets:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        trainer.train(0)
    final = _snapshot_state(
        trainer,
        torch=torch,
        np=np,
        batch_indices=indices,
    )
    states["final"] = _save_state(
        snapshot / "state_final.pt",
        final,
        torch=torch,
        np=np,
    )
    deterministic_warnings = [
        str(warning.message)
        for warning in caught
        if "deterministic" in str(warning.message).lower()
    ]
    if deterministic_warnings:
        raise GateError(
            f"{mode}: deterministic warning observed: {deterministic_warnings}"
        )
    if (
        optimizer["count"] != EQUIVALENCE_UPDATES
        or completed_updates["count"] != EQUIVALENCE_UPDATES
        or scheduler["count"] != 1
        or trainer.formal_optimizer_updates != EQUIVALENCE_UPDATES
        or len(loader.batches) != EQUIVALENCE_UPDATES
        or len(targets) != EQUIVALENCE_UPDATES
    ):
        raise GateError(f"{mode}: incomplete two-update equivalence execution")
    return {
        "format": FORMAT,
        "status": "pass",
        "kind": "equivalence",
        "mode": mode,
        "repeat_index": repeat_index,
        "semantic_receipts": semantic_receipts,
        "real_batches": loader.batches,
        "states": states,
        "target_joints": targets,
        "optimizer_steps": optimizer["count"],
        "completed_formal_optimizer_updates": completed_updates["count"],
        "scheduler_steps": scheduler["count"],
        "formal_optimizer_updates": trainer.formal_optimizer_updates,
        "runtime": {
            "device_name": torch.cuda.get_device_name(0),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
    }


class _TimedLoader:
    def __init__(
        self,
        loader: Any,
        *,
        torch: Any,
        warmup: int,
        measured: int,
    ):
        self.loader = loader
        self.torch = torch
        self.warmup = warmup
        self.measured = measured
        self.wall_seconds: list[float] = []
        self.cuda_seconds: list[float] = []
        self.observed_batches: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.loader)
        active_index: int | None = None
        started_wall = 0.0
        started_event = None

        def close_active() -> None:
            nonlocal active_index, started_event
            if active_index is None or started_event is None:
                return
            ended_event = self.torch.cuda.Event(enable_timing=True)
            ended_event.record()
            ended_event.synchronize()
            wall = time.perf_counter() - started_wall
            cuda = float(started_event.elapsed_time(ended_event)) / 1000.0
            if active_index >= self.warmup:
                self.wall_seconds.append(wall)
                self.cuda_seconds.append(cuda)
            active_index = None
            started_event = None

        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                close_active()
                break
            close_active()
            batch_index = len(self.observed_batches)
            speakers = _batch_speaker_audit(batch, torch=self.torch)
            self.observed_batches.append(
                {
                    "batch_index": batch_index,
                    "speaker_ids": speakers,
                    "sample_indices_sha256": hashlib.sha256(
                        canonical_json_bytes(
                            [
                                int(value)
                                for value in batch["sample_index"].tolist()
                            ]
                        )
                    ).hexdigest(),
                }
            )
            active_index = batch_index
            started_event = self.torch.cuda.Event(enable_timing=True)
            started_event.record()
            started_wall = time.perf_counter()
            yield batch
        if (
            len(self.observed_batches) != self.warmup + self.measured
            or len(self.wall_seconds) != self.measured
            or len(self.cuda_seconds) != self.measured
        ):
            raise GateError("timed loader did not record exact warmup/measured counts")


def _run_benchmark_child(
    parsed: argparse.Namespace,
    *,
    mode: str,
    block_index: int,
    semantic_receipts: dict[str, Any],
    torch: Any,
    np: Any,
    trainer: Any,
) -> dict[str, Any]:
    updates_per_block = WARMUP_UPDATES + MEASURED_UPDATES
    per_speaker = updates_per_block * (EXPECTED_BATCH_SIZE // 4)
    indices = _select_real_indices(
        trainer.train_data,
        per_speaker=per_speaker,
        np=np,
    )
    if (
        block_index < 0
        or block_index >= len(ABBA_ORDER)
        or ABBA_ORDER[block_index] != mode
        or (trainer.lower_target_joints_cache is not None)
        != (mode == "cache")
    ):
        raise GateError("fresh benchmark child mode/cache binding changed")
    initial_state = _snapshot_state(
        trainer,
        torch=torch,
        np=np,
        batch_indices=indices,
    )
    _assert_finite_training_state(initial_state, torch=torch, np=np)
    initial_state_sha256 = canonical_tree_sha256(
        initial_state,
        torch=torch,
        np=np,
    )
    timed = _TimedLoader(
        _make_loader(trainer, indices, torch=torch, np=np),
        torch=torch,
        warmup=WARMUP_UPDATES,
        measured=MEASURED_UPDATES,
    )
    trainer.train_loader = timed
    trainer.train_length = updates_per_block
    with _count_steps(
        trainer.opt,
        f"benchmark {block_index} optimizer",
    ) as optimizer:
        with _count_steps(
            trainer.opt_s,
            f"benchmark {block_index} scheduler",
        ) as scheduler:
            initial_updates = int(trainer.formal_optimizer_updates)
            if initial_updates != 0:
                raise GateError("fresh benchmark trainer did not start at zero")
            trainer.train(0)
    observed_updates = int(trainer.formal_optimizer_updates)
    if (
        observed_updates != updates_per_block
        or optimizer["count"] != updates_per_block
        or scheduler["count"] != 1
    ):
        raise GateError(f"benchmark block {block_index} update count changed")
    block = {
        "block_index": block_index,
        "mode": mode,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "wall_seconds": timed.wall_seconds,
        "cuda_event_seconds": timed.cuda_seconds,
        "real_batch_count": len(timed.observed_batches),
        "real_batch_receipts": timed.observed_batches,
    }
    return {
        "format": FORMAT,
        "status": "pass",
        "kind": "benchmark",
        "mode": mode,
        "block_index": block_index,
        "semantic_receipts": semantic_receipts,
        "initial_state_sha256": initial_state_sha256,
        "initial_state_exact_scope": (
            "model_grads_optimizer_scheduler_rvq_rng_tracker_counter_smplx_batches"
        ),
        "block": block,
        "optimizer_steps": optimizer["count"],
        "scheduler_steps": scheduler["count"],
        "formal_optimizer_updates": observed_updates,
        "runtime": {
            "device_name": torch.cuda.get_device_name(0),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
    }


def _internal_main(parsed: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

    internal_mode = str(parsed.internal_mode)
    kind, target_mode, ordinal = _internal_mode_contract(internal_mode)
    if int(parsed.internal_parent_pid) != os.getppid():
        raise GateError("internal child PPID does not match its exact orchestrator")
    ancestry = guarded_ancestry(parsed, internal=True)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise GateError("internal gate child requires exactly one visible CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if (
        "H200" not in device_name.upper()
        or device_name != parsed.expected_device_name
    ):
        raise GateError(f"internal gate child requires bound H200, got {device_name!r}")
    torch.cuda.set_device(0)
    repository = Path(__file__).resolve().parents[2]
    source = source_receipt(
        repository,
        parsed.expected_source_commit,
        parsed.expected_source_tree,
    )
    snapshot = Path(parsed.snapshot_dir)
    if snapshot.exists() or snapshot.is_symlink():
        raise FileExistsError(snapshot)
    snapshot.mkdir(parents=True, exist_ok=False)
    output = snapshot / "writer"
    output.mkdir()
    cache_enabled = target_mode == "cache"
    args = _build_lower_args(
        parsed,
        output=output,
        cache_enabled=cache_enabled,
    )
    from utils import other_tools

    other_tools.set_random_seed(args)
    if kind == "equivalence":
        # Production remains benchmark=True.  Only the cross-process exactness
        # proof pins a single deterministic cuDNN algorithm.
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    if (
        not bool(torch.backends.cudnn.deterministic)
        or bool(torch.backends.cudnn.benchmark)
        is (kind == "equivalence")
    ):
        raise GateError("cuDNN runtime does not match gate mode")
    init_file = snapshot / "dist_init"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    trainer = None
    try:
        trainer_module = importlib.import_module("aelower_trainer")
        trainer = trainer_module.CustomTrainer(args)
        if trainer.writer is not None:
            trainer.writer.close()
        trainer.writer = _NullWriter()
        if cache_enabled != (trainer.lower_target_joints_cache is not None):
            raise GateError("trainer cache activation mismatch")
        semantic_receipts = _semantic_receipts(
            parsed,
            args=args,
            source=source,
            torch=torch,
            target_mode=target_mode,
            kind=kind,
        )
        if kind == "benchmark":
            result = _run_benchmark_child(
                parsed,
                mode=target_mode,
                block_index=ordinal,
                semantic_receipts=semantic_receipts,
                torch=torch,
                np=np,
                trainer=trainer,
            )
        else:
            result = _run_equivalence_child(
                parsed,
                mode=target_mode,
                repeat_index=ordinal,
                semantic_receipts=semantic_receipts,
                torch=torch,
                np=np,
                trainer=trainer,
                trainer_module=trainer_module,
                snapshot=snapshot,
            )
        if guarded_ancestry(parsed, internal=True) != ancestry:
            raise GateError("internal guarded ancestry changed during execution")
        result["guarded_ancestry"] = ancestry
        atomic_json_new(snapshot / "result.json", result)
    finally:
        if trainer is not None:
            if trainer.writer is not None:
                trainer.writer.close()
            del trainer
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


def validate_builder_process_payload(
    payload: Any,
    *,
    receipt_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    manifest: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise GateError("builder process receipt must be one JSON object")
    scope = payload.get("scope")
    receipt_source = payload.get("source_receipt")
    process = payload.get("builder_process")
    manifest_receipt = payload.get("manifest")
    runner = payload.get("guarded_runner")
    wrapper = payload.get("wrapper_process")
    if (
        payload.get("format") != BUILDER_PROCESS_FORMAT
        or payload.get("status") != "complete"
        or type(scope) is not dict
        or scope.get("dataset") != "show_base"
        or scope.get("formal_stage") != "lower"
        or scope.get("speaker_scope") != "All"
        or scope.get("speaker_ids") != list(EXPECTED_SPEAKER_IDS)
        or type(receipt_source) is not dict
        or type(process) is not dict
        or type(manifest_receipt) is not dict
        or type(runner) is not dict
        or type(wrapper) is not dict
    ):
        raise GateError("builder process receipt protocol is incomplete")
    if {
        key: receipt_source.get(key)
        for key in ("origin", "commit", "tree")
    } != {
        key: source.get(key)
        for key in ("origin", "commit", "tree")
    }:
        raise GateError("builder process/source binding mismatch")
    expected_wrapper = (
        Path(__file__).resolve().parent
        / "run_lower_target_cache_builder.py"
    )
    expected_builder = (
        Path(__file__).resolve().parent
        / "build_lower_target_joints_cache.py"
    )
    expected_bootstrap = (
        Path(__file__).resolve().parent
        / "run_lower_target_cache_builder_guarded.sh"
    )
    if (
        Path(str(receipt_source.get("entrypoint", ""))).resolve()
        != expected_wrapper.resolve()
        or receipt_source.get("entrypoint_sha256")
        != sha256_file(expected_wrapper)
        or Path(
            str(receipt_source.get("builder_entrypoint", ""))
        ).resolve()
        != expected_builder.resolve()
        or receipt_source.get("builder_entrypoint_sha256")
        != sha256_file(expected_builder)
        or Path(
            str(receipt_source.get("bootstrap_entrypoint", ""))
        ).resolve()
        != expected_bootstrap.resolve()
        or receipt_source.get("bootstrap_entrypoint_sha256")
        != sha256_file(expected_bootstrap)
    ):
        raise GateError("builder process entrypoint binding mismatch")
    command = process.get("argv")
    wrapper_argv = wrapper.get("argv")
    runner_argv = runner.get("argv")
    runner_gpu_reservation = (
        type(runner_argv) is list
        and all(type(item) is str and "\0" not in item for item in runner_argv)
        and any(
            (
                item == "--gpus"
                and index + 1 < len(runner_argv)
                and runner_argv[index + 1] == EXPECTED_RUNNER_GPUS
            )
            or item == f"--gpus={EXPECTED_RUNNER_GPUS}"
            for index, item in enumerate(runner_argv)
        )
    )
    if (
        type(command) is not list
        or len(command) < 3
        or any(type(item) is not str or "\0" in item for item in command)
        or type(wrapper_argv) is not list
        or len(wrapper_argv) < 2
        or any(
            type(item) is not str or "\0" in item
            for item in wrapper_argv
        )
        or command[0] != wrapper_argv[0]
        or Path(command[1]).resolve() != expected_builder.resolve()
        or command[1:] != manifest.get("argv")
        or process.get("argv_sha256") != argv_sha256(command)
        or process.get("observed_argv_sha256") != argv_sha256(command)
        or require_exact_int(
            process.get("return_code"),
            0,
            "builder return code",
        )
        != 0
        or process.get("cuda_visible_devices") != "0"
        or process.get("timing_scope")
        != "immediately_before_popen_through_complete_child_exit"
        or process.get("excludes")
        != [
            "independent_checker",
            "formal_gate",
            "guard_restore",
            "receipt_validation_and_write",
        ]
    ):
        raise GateError("builder process argv/runtime binding mismatch")
    integer_timings: dict[str, int] = {}
    for key in (
        "started_unix_ns",
        "completed_unix_ns",
        "started_monotonic_ns",
        "completed_monotonic_ns",
        "elapsed_monotonic_ns",
    ):
        value = process.get(key)
        if type(value) is not int:
            raise GateError(f"builder process {key} must be an exact integer")
        integer_timings[key] = value
    started_unix_ns = integer_timings["started_unix_ns"]
    completed_unix_ns = integer_timings["completed_unix_ns"]
    started_monotonic_ns = integer_timings["started_monotonic_ns"]
    completed_monotonic_ns = integer_timings["completed_monotonic_ns"]
    elapsed_ns = integer_timings["elapsed_monotonic_ns"]
    elapsed_seconds = require_finite_positive(
        process.get("elapsed_seconds"),
        "builder elapsed seconds",
    )
    if (
        started_unix_ns <= 0
        or completed_unix_ns <= started_unix_ns
        or started_monotonic_ns <= 0
        or completed_monotonic_ns <= started_monotonic_ns
        or elapsed_ns
        != completed_monotonic_ns - started_monotonic_ns
        or elapsed_ns <= 0
        or elapsed_seconds != elapsed_ns / 1_000_000_000.0
        or type(process.get("pid")) is not int
        or process["pid"] <= 1
        or type(process.get("ppid")) is not int
        or process["ppid"] <= 1
        or type(process.get("starttime")) is not str
        or not process["starttime"].isdigit()
        or type(payload.get("completed_unix_ns")) is not int
        or payload["completed_unix_ns"] < completed_unix_ns
    ):
        raise GateError("builder process full-lifetime timing is invalid")
    if (
        manifest_receipt.get("path") != str(manifest_path)
        or manifest_receipt.get("sha256") != manifest_sha256
        or manifest_receipt.get("status") != "complete"
        or manifest_receipt.get("entries") != EXPECTED_ENTRIES
        or manifest_receipt.get("entry_aggregate_sha256")
        != manifest.get("entry_aggregate_sha256")
        or type(wrapper.get("pid")) is not int
        or wrapper["pid"] <= 1
        or wrapper.get("pid") != process.get("ppid")
        or wrapper.get("ppid") != runner.get("pid")
        or type(wrapper.get("starttime")) is not str
        or not wrapper["starttime"].isdigit()
        or wrapper.get("state") == "Z"
        or Path(wrapper_argv[1]).resolve() != expected_wrapper.resolve()
        or wrapper.get("argv_sha256") != argv_sha256(wrapper_argv)
        or type(runner.get("pid")) is not int
        or runner["pid"] <= 1
        or type(runner.get("ppid")) is not int
        or runner["ppid"] <= 0
        or type(runner.get("starttime")) is not str
        or not runner["starttime"].isdigit()
        or runner.get("state") == "Z"
        or not runner_gpu_reservation
        or EXPECTED_RUNNER not in runner_argv
        or runner.get("argv_sha256") != argv_sha256(runner_argv)
        or require_hex(
            receipt_sha256,
            64,
            "builder process receipt SHA-256",
        )
        != receipt_sha256
    ):
        raise GateError("builder process manifest/runner binding is invalid")
    return {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_monotonic_ns": elapsed_ns,
        "argv_sha256": process["argv_sha256"],
        "return_code": 0,
    }


def _preflight_inputs(
    parsed: argparse.Namespace,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    source = source_receipt(
        repository,
        parsed.expected_source_commit,
        parsed.expected_source_tree,
    )
    from utils.lower_target_cache import (
        validate_checker_payload,
        validate_manifest_payload,
    )

    manifest_path, manifest, manifest_sha = verified_json(
        parsed.cache_manifest,
        parsed.expected_cache_manifest_sha256,
        "lower target cache manifest",
    )
    checker_path, checker, checker_sha = verified_json(
        parsed.checker_receipt,
        parsed.expected_checker_receipt_sha256,
        "lower target cache checker receipt",
    )
    (
        builder_process_path,
        builder_process,
        builder_process_sha,
    ) = verified_json(
        parsed.builder_process_receipt,
        parsed.expected_builder_process_receipt_sha256,
        "lower target cache builder process receipt",
    )
    validate_manifest_payload(manifest)
    validate_checker_payload(
        checker,
        manifest_sha256=manifest_sha,
        manifest=manifest,
    )
    builder_process_validation = validate_builder_process_payload(
        builder_process,
        receipt_sha256=builder_process_sha,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        source=source,
    )
    for receipt_label, receipt in (
        ("builder", manifest["source_receipt"]),
        ("checker", checker["source_receipt"]),
    ):
        if any(
            receipt.get(key) != source[key]
            for key in ("origin", "commit", "tree")
        ):
            raise GateError(f"{receipt_label}/gate source receipt mismatch")
    protocol = manifest["protocol"]
    if (
        protocol.get("speaker_scope") != "All"
        or protocol.get("speaker_ids") != list(EXPECTED_SPEAKER_IDS)
        or protocol.get("split") != "train"
        or protocol.get("formal_stage") != "lower"
        or protocol.get("dataset") != "show_base"
    ):
        raise GateError("cache manifest is not strict SHOW-All lower train")
    _exact_speaker_contract(protocol.get("speaker_map"), "cache speaker map")
    representation = manifest["representation_receipt"]
    representation_lmdb = Path(parsed.representation_lmdb)
    if (
        representation_lmdb.is_symlink()
        or representation_lmdb.resolve()
        != Path(representation["lmdb_path"]).resolve()
        or not representation_lmdb.resolve().is_dir()
    ):
        raise GateError("representation LMDB path/receipt mismatch")
    representation_data = representation_lmdb.resolve() / "data.mdb"
    if (
        sha256_file(representation_data)
        != require_hex(
            parsed.expected_representation_data_sha256,
            64,
            "representation data SHA-256",
        )
        or representation["data_mdb_sha256"]
        != parsed.expected_representation_data_sha256
    ):
        raise GateError("representation data.mdb SHA mismatch")
    summary_path, summary, summary_sha = verified_json(
        parsed.representation_summary,
        parsed.expected_representation_summary_sha256,
        "representation summary",
    )
    lineage_path, lineage, lineage_sha = verified_json(
        parsed.representation_lineage,
        parsed.expected_representation_lineage_sha256,
        "representation lineage",
    )
    if (
        summary_path != lineage_path
        or summary_sha != lineage_sha
        or summary != lineage
        or summary.get("status") != "complete"
        or require_exact_int(
            summary.get("entries"),
            EXPECTED_ENTRIES,
            "representation entries",
        )
        != EXPECTED_ENTRIES
        or summary.get("entry_aggregate_sha256")
        != require_hex(
            parsed.expected_representation_entry_aggregate_sha256,
            64,
            "representation aggregate SHA-256",
        )
        or representation["entry_aggregate_sha256"]
        != parsed.expected_representation_entry_aggregate_sha256
    ):
        raise GateError("representation summary/lineage binding mismatch")
    _exact_speaker_contract(
        summary.get("protocol", {}).get("speaker_map"),
        "representation speaker map",
    )
    cache_input = Path(parsed.cache_lmdb)
    cache = cache_input.resolve()
    if (
        cache_input.is_symlink()
        or not cache.is_dir()
        or cache != Path(manifest["lmdb"]["path"]).resolve()
    ):
        raise GateError("cache LMDB path/manifest mismatch")
    for name, key in (
        ("data.mdb", "data_mdb_sha256"),
        ("lock.mdb", "lock_mdb_sha256"),
    ):
        artifact = cache / name
        if artifact.is_symlink() or not artifact.is_file():
            raise GateError(f"cache artifact is missing: {artifact}")
        if sha256_file(artifact) != manifest["lmdb"][key]:
            raise GateError(f"cache artifact SHA mismatch: {artifact}")
    asset_root = Path(parsed.asset_root)
    smplx_asset = (
        asset_root.resolve()
        / "smplx_models"
        / "smplx"
        / "SMPLX_NEUTRAL_2020.npz"
    )
    expected_smplx = require_hex(
        parsed.expected_smplx_asset_sha256,
        64,
        "SMPL-X SHA-256",
    )
    if (
        asset_root.is_symlink()
        or smplx_asset.is_symlink()
        or not smplx_asset.is_file()
        or sha256_file(smplx_asset) != expected_smplx
        or expected_smplx != FORMAL_SMPLX_SHA256
        or manifest["smplx_receipt"]["asset_sha256"] != expected_smplx
    ):
        raise GateError("SMPL-X asset/manifest binding mismatch")
    return {
        "source": source,
        "cache_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "payload": manifest,
        },
        "checker_receipt": {
            "path": str(checker_path),
            "sha256": checker_sha,
            "payload": checker,
        },
        "builder_process_receipt": {
            "path": str(builder_process_path),
            "sha256": builder_process_sha,
            "payload": builder_process,
            "validation": builder_process_validation,
        },
        "representation": {
            "lmdb": str(representation_lmdb.resolve()),
            "data_mdb_sha256": sha256_file(representation_data),
            "summary": str(summary_path),
            "summary_sha256": summary_sha,
            "lineage": str(lineage_path),
            "lineage_sha256": lineage_sha,
            "entry_aggregate_sha256": (
                summary["entry_aggregate_sha256"]
            ),
        },
        "cache_lmdb": {
            "path": str(cache),
            "data_mdb_sha256": manifest["lmdb"]["data_mdb_sha256"],
            "lock_mdb_sha256": manifest["lmdb"]["lock_mdb_sha256"],
            "entry_aggregate_sha256": manifest["entry_aggregate_sha256"],
        },
        "smplx": {
            "asset_root": str(asset_root.resolve()),
            "asset": str(smplx_asset),
            "asset_sha256": expected_smplx,
        },
        "builder_timing": {
            "source": "external_full_child_process_receipt",
            "receipt_sha256": builder_process_sha,
            "full_builder_seconds": builder_process_validation[
                "elapsed_seconds"
            ],
            "elapsed_monotonic_ns": builder_process_validation[
                "elapsed_monotonic_ns"
            ],
        },
    }


def _child_command(
    parsed: argparse.Namespace,
    *,
    mode: str,
    snapshot: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
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
        "--asset-root",
        str(parsed.asset_root),
        "--expected-smplx-asset-sha256",
        str(parsed.expected_smplx_asset_sha256),
        "--cache-lmdb",
        str(parsed.cache_lmdb),
        "--cache-manifest",
        str(parsed.cache_manifest),
        "--expected-cache-manifest-sha256",
        str(parsed.expected_cache_manifest_sha256),
        "--checker-receipt",
        str(parsed.checker_receipt),
        "--expected-checker-receipt-sha256",
        str(parsed.expected_checker_receipt_sha256),
        "--builder-process-receipt",
        str(parsed.builder_process_receipt),
        "--expected-builder-process-receipt-sha256",
        str(parsed.expected_builder_process_receipt_sha256),
        "--expected-source-commit",
        str(parsed.expected_source_commit),
        "--expected-source-tree",
        str(parsed.expected_source_tree),
        "--expected-device-name",
        str(parsed.expected_device_name),
        "--expected-runner-pid",
        str(parsed.expected_runner_pid),
        "--expected-runner-starttime",
        str(parsed.expected_runner_starttime),
        "--expected-runner-argv-sha256",
        str(parsed.expected_runner_argv_sha256),
        "--output-root",
        str(parsed.output_root),
        "--internal-mode",
        mode,
        "--snapshot-dir",
        str(snapshot),
        "--internal-parent-pid",
        str(os.getpid()),
    ]


def _load_child_result(
    snapshot: Path,
    expected_kind: str,
    *,
    command: list[str],
    expected_pid: int,
) -> dict[str, Any]:
    path = snapshot / "result.json"
    if path.is_symlink() or not path.is_file():
        raise GateError(f"child result is missing: {path}")
    payload = json.loads(path.read_text())
    ancestry = payload.get("guarded_ancestry")
    if (
        type(payload) is not dict
        or payload.get("format") != FORMAT
        or payload.get("status") != "pass"
        or payload.get("kind") != expected_kind
        or type(ancestry) is not list
        or len(ancestry) < 3
        or type(ancestry[0]) is not dict
        or ancestry[0].get("pid") != expected_pid
        or ancestry[0].get("ppid") != os.getpid()
        or ancestry[0].get("argv") != command
        or ancestry[0].get("argv_sha256") != argv_sha256(command)
    ):
        raise GateError(f"invalid child result: {path}")
    return payload


def _compare_equivalence_pair(
    left_root: Path,
    right_root: Path,
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    *,
    relationship: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if relationship not in {"repeat_control", "cross_mode"}:
        raise GateError(f"invalid equivalence relationship: {relationship}")
    left_mode = left_payload.get("mode")
    right_mode = right_payload.get("mode")
    if (
        left_payload.get("real_batches") != right_payload.get("real_batches")
        or left_mode not in {"legacy", "cache"}
        or right_mode not in {"legacy", "cache"}
        or (
            relationship == "repeat_control"
            and left_mode != right_mode
        )
        or (
            relationship == "cross_mode"
            and (left_mode, right_mode) != ("legacy", "cache")
        )
    ):
        raise GateError("equivalence child batch/mode identities differ")
    left_semantic = left_payload.get("semantic_receipts")
    right_semantic = right_payload.get("semantic_receipts")
    if (
        type(left_semantic) is not dict
        or type(right_semantic) is not dict
        or left_semantic.get("common") != right_semantic.get("common")
        or (
            relationship == "repeat_control"
            and left_semantic != right_semantic
        )
        or (
            relationship == "cross_mode"
            and (
                left_semantic.get("mode")
                != {"target_mode": "legacy", "cache_enabled": False}
                or right_semantic.get("mode")
                != {"target_mode": "cache", "cache_enabled": True}
            )
        )
        or left_payload.get("runtime") != right_payload.get("runtime")
    ):
        raise GateError("equivalence semantic/runtime receipts differ")
    for payload, label in (
        (left_payload, "left"),
        (right_payload, "right"),
    ):
        if (
            payload.get("optimizer_steps") != EQUIVALENCE_UPDATES
            or payload.get("completed_formal_optimizer_updates")
            != EQUIVALENCE_UPDATES
            or payload.get("formal_optimizer_updates")
            != EQUIVALENCE_UPDATES
            or payload.get("scheduler_steps") != 1
        ):
            raise GateError(f"{label} equivalence update receipt is incomplete")

    state_names = ("initial", "step_1", "step_2", "final")
    state_receipts: dict[str, Any] = {}
    for name in state_names:
        left_record = left_payload.get("states", {}).get(name)
        right_record = right_payload.get("states", {}).get(name)
        if type(left_record) is not dict or type(right_record) is not dict:
            raise GateError(f"equivalence state {name} is missing")
        left_path = Path(left_record["path"])
        right_path = Path(right_record["path"])
        if (
            left_path.parent != left_root
            or right_path.parent != right_root
            or sha256_file(left_path) != left_record["file_sha256"]
            or sha256_file(right_path) != right_record["file_sha256"]
        ):
            raise GateError(f"equivalence state {name} path/SHA changed")
        left = torch.load(left_path, map_location="cpu", weights_only=False)
        right = torch.load(right_path, map_location="cpu", weights_only=False)
        assert_tree_byte_exact(
            left,
            right,
            torch=torch,
            np=np,
            path=name,
        )
        left_digest = canonical_tree_sha256(left, torch=torch, np=np)
        right_digest = canonical_tree_sha256(right, torch=torch, np=np)
        if (
            left_digest != left_record["canonical_state_sha256"]
            or right_digest != right_record["canonical_state_sha256"]
            or left_digest != right_digest
        ):
            raise GateError(f"equivalence state {name} digest differs")
        state_receipts[name] = {
            "canonical_state_sha256": left_digest,
            "left_file_sha256": left_record["file_sha256"],
            "right_file_sha256": right_record["file_sha256"],
            "byte_exact": True,
        }

    left_targets = left_payload.get("target_joints")
    right_targets = right_payload.get("target_joints")
    if (
        type(left_targets) is not list
        or type(right_targets) is not list
        or len(left_targets) != EQUIVALENCE_UPDATES
        or len(right_targets) != EQUIVALENCE_UPDATES
    ):
        raise GateError("target-joint equivalence evidence is incomplete")
    targets: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(left_targets, right_targets)):
        left_path = Path(left["path"])
        right_path = Path(right["path"])
        if left_path.parent != left_root or right_path.parent != right_root:
            raise GateError("target-joint evidence escaped its child root")
        left_bytes = left_path.read_bytes()
        right_bytes = right_path.read_bytes()
        if (
            left["call_index"] != index
            or right["call_index"] != index
            or left["dtype"] != "torch.float32"
            or right["dtype"] != "torch.float32"
            or left["normalized_shape"]
            != [EXPECTED_BATCH_SIZE * EXPECTED_FRAMES, 127, 3]
            or right["normalized_shape"]
            != [EXPECTED_BATCH_SIZE * EXPECTED_FRAMES, 127, 3]
            or left["sha256"] != hashlib.sha256(left_bytes).hexdigest()
            or right["sha256"] != hashlib.sha256(right_bytes).hexdigest()
            or left_bytes != right_bytes
        ):
            raise GateError(f"target-joint bytes differ at update {index + 1}")
        targets.append(
            {
                "update": index + 1,
                "bytes": len(left_bytes),
                "sha256": left["sha256"],
                "byte_exact": True,
            }
        )
    if (
        left_payload.get("runtime", {}).get("cudnn_benchmark") is not False
        or right_payload.get("runtime", {}).get("cudnn_benchmark") is not False
        or left_payload.get("runtime", {}).get("deterministic_algorithms")
        is not True
        or right_payload.get("runtime", {}).get("deterministic_algorithms")
        is not True
    ):
        raise GateError("equivalence child runtime is not deterministic")
    return {
        "status": "pass",
        "relationship": relationship,
        "left_mode": left_mode,
        "right_mode": right_mode,
        "real_batches": left_payload["real_batches"],
        "states": state_receipts,
        "target_joints": targets,
        "semantic_receipts_exact": True,
        "initial_and_two_complete_updates_and_final_byte_exact": True,
    }


def _build_equivalence_report(
    child_roots: dict[str, Path],
    child_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    legacy_0 = "equivalence-legacy-0"
    legacy_1 = "equivalence-legacy-1"
    cache_0 = "equivalence-cache-0"
    cache_1 = "equivalence-cache-1"
    repeat_legacy = _compare_equivalence_pair(
        child_roots[legacy_0],
        child_roots[legacy_1],
        child_results[legacy_0],
        child_results[legacy_1],
        relationship="repeat_control",
    )
    repeat_cache = _compare_equivalence_pair(
        child_roots[cache_0],
        child_roots[cache_1],
        child_results[cache_0],
        child_results[cache_1],
        relationship="repeat_control",
    )
    cross_0 = _compare_equivalence_pair(
        child_roots[legacy_0],
        child_roots[cache_0],
        child_results[legacy_0],
        child_results[cache_0],
        relationship="cross_mode",
    )
    cross_1 = _compare_equivalence_pair(
        child_roots[legacy_1],
        child_roots[cache_1],
        child_results[legacy_1],
        child_results[cache_1],
        relationship="cross_mode",
    )
    return {
        "status": "pass",
        "fresh_processes": 4,
        "repeat_controls": {
            "legacy_a_a": repeat_legacy,
            "cache_b_b": repeat_cache,
        },
        "cross_mode_pairs": [cross_0, cross_1],
        "states": cross_0["states"],
        "target_joints": cross_0["target_joints"],
        "repeat_controls_exact": True,
        "semantic_receipts_exact": True,
        "initial_and_two_complete_updates_and_final_byte_exact": True,
    }


def _public_main(parsed: argparse.Namespace) -> None:
    validate_static_gate_contract()
    ancestry = guarded_ancestry(parsed, internal=False)
    output_input = Path(parsed.output_root)
    if output_input.is_symlink():
        raise GateError("output root must not be a symlink")
    output = output_input.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    child_commands: list[dict[str, Any]] = []
    try:
        preflight = _preflight_inputs(parsed)
        atomic_json_new(output / "preflight.json", preflight)
        child_results: dict[str, dict[str, Any]] = {}
        child_roots: dict[str, Path] = {}
        child_specs = [
            (mode, "equivalence")
            for mode, _, _ in EQUIVALENCE_CHILDREN
        ] + [
            (mode, "benchmark")
            for mode, _, _ in BENCHMARK_CHILDREN
        ]
        for sequence_index, (mode, kind) in enumerate(child_specs):
            snapshot = output / mode
            command = _child_command(parsed, mode=mode, snapshot=snapshot)
            command_sha = argv_sha256(command)
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONHASHSEED"] = "2021"
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
            )
            return_code = int(process.wait())
            command_receipt = {
                "sequence_index": sequence_index,
                "mode": mode,
                "kind": kind,
                "pid": int(process.pid),
                "argv": command,
                "argv_sha256": command_sha,
                "return_code": return_code,
                "cuda_visible_devices": "0",
            }
            child_commands.append(command_receipt)
            if return_code != 0:
                raise GateError(
                    f"formal gate child {mode} failed rc={return_code}"
                )
            child_result = _load_child_result(
                snapshot,
                kind,
                command=command,
                expected_pid=int(process.pid),
            )
            reported = child_result["guarded_ancestry"][0]
            command_receipt["reported_starttime"] = reported["starttime"]
            command_receipt["reported_argv_sha256"] = reported[
                "argv_sha256"
            ]
            command_receipt["exact_argv_and_rc"] = True
            child_results[mode] = child_result
            child_roots[mode] = snapshot

        process_identities = {
            (item["pid"], item["reported_starttime"])
            for item in child_commands
        }
        if len(process_identities) != len(child_commands):
            raise GateError("formal gate children were not eight fresh processes")
        equivalence = _build_equivalence_report(
            child_roots,
            child_results,
        )
        benchmark_blocks: list[dict[str, Any]] = []
        benchmark_common: Any = None
        benchmark_real_batches: Any = None
        benchmark_initial_state_sha256: Any = None
        for mode, expected_target_mode, block_index in BENCHMARK_CHILDREN:
            payload = child_results[mode]
            semantic = payload.get("semantic_receipts")
            block = payload.get("block")
            if (
                payload.get("mode") != expected_target_mode
                or payload.get("block_index") != block_index
                or payload.get("formal_optimizer_updates")
                != WARMUP_UPDATES + MEASURED_UPDATES
                or payload.get("optimizer_steps")
                != WARMUP_UPDATES + MEASURED_UPDATES
                or payload.get("scheduler_steps") != 1
                or payload.get("runtime", {}).get("cudnn_benchmark")
                is not True
                or payload.get("initial_state_exact_scope")
                != (
                    "model_grads_optimizer_scheduler_rvq_rng_tracker_"
                    "counter_smplx_batches"
                )
                or type(payload.get("initial_state_sha256")) is not str
                or len(payload["initial_state_sha256"]) != 64
                or type(semantic) is not dict
                or semantic.get("mode")
                != {
                    "target_mode": expected_target_mode,
                    "cache_enabled": expected_target_mode == "cache",
                }
                or type(block) is not dict
            ):
                raise GateError(
                    f"fresh benchmark child {block_index} is incomplete"
                )
            if benchmark_common is None:
                benchmark_common = semantic.get("common")
                benchmark_real_batches = block.get("real_batch_receipts")
                benchmark_initial_state_sha256 = payload[
                    "initial_state_sha256"
                ]
            elif (
                semantic.get("common") != benchmark_common
                or block.get("real_batch_receipts")
                != benchmark_real_batches
                or payload.get("initial_state_sha256")
                != benchmark_initial_state_sha256
            ):
                raise GateError(
                    "fresh benchmark children used different receipts/batches"
                )
            benchmark_blocks.append(block)
        performance = calculate_speedup_report(
            benchmark_blocks,
            full_builder_seconds=preflight["builder_timing"][
                "full_builder_seconds"
            ],
        )
        performance["fresh_processes"] = 4
        performance["semantic_receipts_exact"] = True
        performance["real_batch_receipts_exact"] = True
        performance["initial_whole_state_exact"] = True
        performance["initial_state_sha256"] = (
            benchmark_initial_state_sha256
        )
        performance["builder_process_receipt_sha256"] = preflight[
            "builder_timing"
        ]["receipt_sha256"]
        final_preflight = _preflight_inputs(parsed)
        if canonical_json_sha256(final_preflight) != canonical_json_sha256(
            preflight
        ):
            raise GateError("source/artifact receipts changed during formal gate")
        report = {
            "format": FORMAT,
            "status": "pass",
            "scope": {
                "dataset": "show_base",
                "formal_stage": "lower",
                "speaker_scope": "All",
                "speaker_ids": list(EXPECTED_SPEAKER_IDS),
                "speaker_map": dict(EXPECTED_SPEAKERS),
                "forbidden": [
                    "SemGate",
                    "Sparse Motion Generation",
                    "speaker2",
                    "BEAT2 motion weights",
                ],
            },
            "protocol": validate_static_gate_contract(),
            "guarded_ancestry": ancestry,
            "preflight": preflight,
            "equivalence": equivalence,
            "performance": performance,
            "child_commands": child_commands,
            "started_unix": started,
            "completed_unix": time.time(),
            "argv": sys.argv,
        }
        atomic_json_new(output / "formal_gate_report.json", report)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "report": str(output / "formal_gate_report.json"),
                    "report_sha256": sha256_file(
                        output / "formal_gate_report.json"
                    ),
                    "pooled_wall_speedup": performance["pooled"]["wall"][
                        "speedup"
                    ],
                    "pooled_cuda_speedup": performance["pooled"][
                        "cuda_event"
                    ]["speedup"],
                    "amortized_speedup": performance["amortization"][
                        "speedup"
                    ],
                },
                sort_keys=True,
            )
        )
    except BaseException as error:
        failure = {
            "format": FORMAT,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "started_unix": started,
            "failed_unix": time.time(),
            "child_commands": child_commands,
            "argv": sys.argv,
        }
        try:
            atomic_json_new(output / "failure.json", failure)
        except BaseException:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--representation-lmdb", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--representation-lineage", type=Path, required=True)
    parser.add_argument(
        "--expected-representation-data-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-representation-summary-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-representation-lineage-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-representation-entry-aggregate-sha256",
        required=True,
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--expected-smplx-asset-sha256", required=True)
    parser.add_argument("--cache-lmdb", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--checker-receipt", type=Path, required=True)
    parser.add_argument("--expected-checker-receipt-sha256", required=True)
    parser.add_argument("--builder-process-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-builder-process-receipt-sha256",
        required=True,
    )
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument("--expected-runner-pid", type=int, required=True)
    parser.add_argument("--expected-runner-starttime", required=True)
    parser.add_argument("--expected-runner-argv-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--internal-mode",
        choices=INTERNAL_MODES,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--snapshot-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-parent-pid",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    if parsed.internal_mode is None:
        if parsed.snapshot_dir is not None or parsed.internal_parent_pid is not None:
            raise GateError("internal child arguments require --internal-mode")
        _public_main(parsed)
        return
    if parsed.snapshot_dir is None or parsed.internal_parent_pid is None:
        raise GateError("internal mode requires snapshot-dir and exact parent PID")
    _internal_main(parsed)


if __name__ == "__main__":
    main()
