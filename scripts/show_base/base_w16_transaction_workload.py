#!/usr/bin/env python3
"""Audited W16 Base workload nested inside the two-node transaction.

``run_base_official_adapt_long.sh`` deliberately requires the guarded runner
as its direct parent.  A two-node transaction necessarily inserts its
coordinator and supervisor, so that launcher must never be nested.  This shim
is the transaction-owned alternative: it replays the transaction, source, and
SHOW input bindings and then replaces itself with the one fixed W16 torchrun
command.  It never invokes a shell and never accepts a free-form executable.
The outer transaction must pass ``--completion-timeout-ms 86400000`` (or a
larger explicit value) before its workload delimiter; the generic 120-second
transaction default is intentionally rejected for every W16 workload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


TRANSACTION_SCHEMA = "semtalk.dual_node_guarded_transaction.v2"
PORTABLE_SCHEMA = f"{TRANSACTION_SCHEMA}.portable"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_RUNNER = "/tmp/globaldiff_guarded_runner.py"
EXPECTED_GPUS = "0,1,2,3,4,5,6,7"
EXPECTED_HOST_BY_RANK = {
    0: "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0",
    1: "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0",
}
EXPECTED_SHOW_SPEAKERS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
W16_TOPOLOGIES = {
    "official_objective_w16_l4_g64_ddp_adaptation": {
        "local_batch_size": 4,
        "learning_rate": 0.00005,
        "precision": "bf16",
    },
    "validation_gated_w16_l32_g512_empirical_acceleration": {
        "local_batch_size": 32,
        "learning_rate": 0.00003,
        "precision": "bf16",
    },
}
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
SAFE_MASTER = re.compile(r"^[A-Za-z0-9.-]+$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
E30 = re.compile(r"(?<![a-z0-9])e[-_ ]?30(?![0-9])", re.IGNORECASE)
SPEAKER2 = re.compile(
    r"(?<![a-z0-9])speaker[-_ ]?2(?![0-9])", re.IGNORECASE
)
MAX_JSON_BYTES = 8 << 20
W16_MIN_COMPLETION_TIMEOUT_MS = 24 * 60 * 60 * 1000
IDENTITY_KEYS = {
    "pid",
    "ppid",
    "pgid",
    "sid",
    "starttime_ticks",
    "argv_sha256",
}
RESERVED_TRAINER_OPTIONS = {
    "--formal-node-rank",
    "--formal-master-addr",
    "--formal-master-port",
    "--formal-run-id",
    "--topology-mode",
    "--local-batch-size",
    "--learning-rate",
    "--precision",
    "--seed",
}


class W16ShimError(RuntimeError):
    """A fail-closed workload-shim error."""


def _canonical_json_bytes(value: Any) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _argv_sha256(argv: Sequence[str]) -> str:
    return _sha256_bytes(
        b"\0".join(os.fsencode(token) for token in argv) + b"\0"
    )


def _exact_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _open_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    if not path.is_absolute():
        raise W16ShimError(f"{label} must be absolute")
    try:
        canonical = path.resolve(strict=True)
        public = path.lstat()
    except OSError as exc:
        raise W16ShimError(f"{label} is unavailable") from exc
    if canonical != path or stat.S_ISLNK(public.st_mode):
        raise W16ShimError(f"{label} must be canonical and symlink-free")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    opened = os.fstat(fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (public.st_dev, public.st_ino)
    ):
        os.close(fd)
        raise W16ShimError(f"{label} is not a stable regular file")
    return fd, opened


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 8 << 20)
        if not block:
            break
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _snapshot_file(path: Path, label: str) -> dict[str, Any]:
    fd, before = _open_regular(path, label)
    try:
        digest = _sha256_fd(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
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
        raise W16ShimError(f"{label} changed while hashing")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": before.st_size,
    }


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    fd, before = _open_regular(path, label)
    try:
        if before.st_size <= 0 or before.st_size > MAX_JSON_BYTES:
            raise W16ShimError(f"{label} has an unsafe size")
        raw = b""
        while len(raw) < before.st_size:
            block = os.read(fd, before.st_size - len(raw))
            if not block:
                break
            raw += block
        after = os.fstat(fd)
    finally:
        os.close(fd)
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
        raise W16ShimError(f"{label} changed while reading")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W16ShimError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise W16ShimError(f"{label} is not a canonical JSON object")
    return value, _sha256_bytes(raw)


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W16ShimError("Git source evidence is unavailable") from exc


def audit_source(
    repository: Path,
    *,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    """Prove one clean detached source and its two executed blobs."""

    if not HEX40.fullmatch(expected_commit) or not HEX40.fullmatch(expected_tree):
        raise W16ShimError("source commit/tree pins must be lowercase 40-hex")
    if repository.resolve(strict=True) != repository or repository.is_symlink():
        raise W16ShimError("source repository must be canonical and symlink-free")
    origin = _git(repository, "remote", "get-url", "origin").stdout.decode().strip()
    commit = _git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    tree = _git(repository, "rev-parse", "HEAD^{tree}").stdout.decode().strip()
    dirty = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    ).stdout
    symbolic = _git(repository, "symbolic-ref", "-q", "HEAD", check=False)
    if (
        origin != EXPECTED_ORIGIN
        or commit != expected_commit
        or tree != expected_tree
        or dirty
        or symbolic.returncode != 1
    ):
        raise W16ShimError(
            "source must be the exact clean detached SemTalk commit/tree"
        )
    blobs: dict[str, str] = {}
    for relative in (
        "scripts/show_base/dual_node_guarded_transaction.py",
        "scripts/show_base/run_dual_node_guarded_transaction.sh",
        "scripts/show_base/guarded_runner_contract.sh",
        "scripts/show_base/base_w16_transaction_workload.py",
        "scripts/show_base/train_base_official_adapt_long.py",
    ):
        path = repository / relative
        observed = _snapshot_file(path, f"source blob {relative}")["sha256"]
        committed = _git(repository, "show", f"HEAD:{relative}").stdout
        if observed != _sha256_bytes(committed):
            raise W16ShimError(f"source blob differs from HEAD: {relative}")
        blobs[relative] = observed
    return {
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "detached": True,
        "entrypoint_sha256": blobs,
    }


def _proc_identity(pid: int) -> dict[str, Any]:
    if pid <= 0:
        raise W16ShimError("invalid parent PID")
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        raw_argv = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as exc:
        raise W16ShimError("transaction supervisor is unavailable") from exc
    if ") " not in stat_line or not raw_argv.endswith(b"\0"):
        raise W16ShimError("invalid transaction supervisor process evidence")
    fields = stat_line.rsplit(") ", 1)[1].split()
    if len(fields) < 20 or fields[0] == "Z":
        raise W16ShimError("transaction supervisor is not live")
    argv = [os.fsdecode(token) for token in raw_argv[:-1].split(b"\0")]
    if not argv:
        raise W16ShimError("transaction supervisor argv is empty")
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime_ticks": int(fields[19]),
        "argv_sha256": _argv_sha256(argv),
    }


def _validate_identity(value: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != IDENTITY_KEYS
        or not all(
            _exact_int(value.get(key), minimum=1)
            for key in IDENTITY_KEYS - {"argv_sha256"}
        )
        or not HEX64.fullmatch(str(value.get("argv_sha256", "")))
    ):
        raise W16ShimError(f"invalid {label} identity")
    return dict(value)


def validate_transaction_context(
    *,
    transaction_root: Path,
    run_id: str,
    node_id: str,
    node_rank: int,
    source: Mapping[str, Any],
    master_addr: str,
    master_port: int,
    original_argv: Sequence[str],
    parent_identity: Mapping[str, Any],
    hostname: str,
) -> dict[str, Any]:
    """Replay the immutable two-node GO chain before torchrun replaces us."""

    if (
        node_rank not in (0, 1)
        or node_id != EXPECTED_HOST_BY_RANK[node_rank]
        or hostname != EXPECTED_HOST_BY_RANK[node_rank]
        or not SAFE_ID.fullmatch(run_id)
        or transaction_root.name != run_id
        or transaction_root.resolve(strict=True) != transaction_root
    ):
        raise W16ShimError("transaction-provided rank/host/run identity changed")
    transaction, _ = _read_canonical_json(
        transaction_root / "TRANSACTION.json", "transaction bootstrap"
    )
    if (
        set(transaction) != {"schema", "status", "portable", "portable_sha256"}
        or transaction.get("schema") != TRANSACTION_SCHEMA
        or transaction.get("status") != "OPEN"
        or not isinstance(transaction.get("portable"), dict)
    ):
        raise W16ShimError("invalid transaction bootstrap")
    portable = transaction["portable"]
    portable_sha = _sha256_bytes(_canonical_json_bytes(portable))
    transaction_source = portable.get("source")
    transaction_entrypoints = (
        transaction_source.get("entrypoint_sha256")
        if isinstance(transaction_source, dict)
        else None
    )
    expected_transaction_entrypoints = {
        relative: source["entrypoint_sha256"][relative]
        for relative in (
            "scripts/show_base/dual_node_guarded_transaction.py",
            "scripts/show_base/run_dual_node_guarded_transaction.sh",
            "scripts/show_base/guarded_runner_contract.sh",
        )
    }
    if (
        transaction.get("portable_sha256") != portable_sha
        or portable.get("schema") != PORTABLE_SCHEMA
        or portable.get("run_id") != run_id
        or portable.get("source_commit") != source.get("commit")
        or portable.get("source_tree") != source.get("tree")
        or not isinstance(transaction_source, dict)
        or set(transaction_source) != {"origin", "commit", "tree", "entrypoint_sha256"}
        or transaction_source.get("origin") != source.get("origin")
        or transaction_source.get("commit") != source.get("commit")
        or transaction_source.get("tree") != source.get("tree")
        or transaction_entrypoints != expected_transaction_entrypoints
        or portable.get("max_restarts") != 0
    ):
        raise W16ShimError("portable transaction/source binding changed")
    timeouts = portable.get("timeouts")
    completion_timeout_ms = (
        timeouts.get("completion_timeout_ms")
        if isinstance(timeouts, dict)
        else None
    )
    if (
        not isinstance(completion_timeout_ms, int)
        or isinstance(completion_timeout_ms, bool)
        or completion_timeout_ms < W16_MIN_COMPLETION_TIMEOUT_MS
    ):
        raise W16ShimError(
            "W16 transaction requires an explicit completion timeout of at least 24 hours"
        )
    participants = portable.get("participants")
    if (
        not isinstance(participants, list)
        or len(participants) != 2
        or not all(isinstance(item, dict) for item in participants)
        or [item.get("rank") for item in participants] != [0, 1]
        or any(
            item.get("node_id") != EXPECTED_HOST_BY_RANK[item["rank"]]
            for item in participants
        )
    ):
        raise W16ShimError("transaction participants are not the exact two hosts")
    master = participants[0]
    if (
        not SAFE_MASTER.fullmatch(master_addr)
        or master_addr not in {master.get("node_id"), master.get("ip")}
        or not (1024 <= master_port <= 65535)
    ):
        raise W16ShimError("master address/port is not bound to rank zero")
    workload = portable.get("workload")
    shim_path = Path(original_argv[1])
    repository = shim_path.parents[2]
    expected_launcher = repository / (
        "scripts/show_base/run_dual_node_guarded_transaction.sh"
    )
    if (
        not isinstance(workload, dict)
        or portable.get("common_command_sha256") != _argv_sha256(original_argv)
        or workload.get("argv_sha256") != _argv_sha256(original_argv)
        or workload.get("exec_argv_sha256") != workload.get("argv_sha256")
        or workload.get("input_sha256") != {}
        or workload.get("input_bindings") != {}
        or workload.get("executable_path") != original_argv[0]
        or workload.get("workdir") != str(repository)
    ):
        raise W16ShimError("two-node immutable workload argv/input binding changed")
    executable = _snapshot_file(
        Path(str(workload.get("executable_path", ""))),
        "transaction workload Python",
    )
    if executable["sha256"] != workload.get("executable_sha256"):
        raise W16ShimError("transaction workload Python changed")

    prepared: dict[int, dict[str, Any]] = {}
    prepared_sha: dict[int, str] = {}
    armed: dict[int, dict[str, Any]] = {}
    armed_sha: dict[int, str] = {}
    for rank in (0, 1):
        prepared[rank], prepared_sha[rank] = _read_canonical_json(
            transaction_root / f"PREPARED.rank{rank}.json",
            f"rank {rank} PREPARED",
        )
        armed[rank], armed_sha[rank] = _read_canonical_json(
            transaction_root / f"ARMED.rank{rank}.json",
            f"rank {rank} ARMED",
        )
        runner = prepared[rank].get("runner")
        coordinator = prepared[rank].get("coordinator")
        if not isinstance(runner, dict) or not isinstance(coordinator, dict):
            raise W16ShimError(f"rank {rank} omitted runner/coordinator evidence")
        runner_identity = _validate_identity(
            {key: runner.get(key) for key in IDENTITY_KEYS},
            f"rank {rank} guarded runner",
        )
        coordinator_identity = _validate_identity(
            coordinator, f"rank {rank} transaction coordinator"
        )
        command = runner.get("command")
        try:
            workload_delimiter = command.index("--") if isinstance(command, list) else -1
        except ValueError:
            workload_delimiter = -1
        coordinator_argv = (
            command[:workload_delimiter] if workload_delimiter >= 0 else []
        )
        timeout_positions = [
            index
            for index, token in enumerate(coordinator_argv)
            if token == "--completion-timeout-ms"
        ]
        explicit_timeout = None
        if (
            len(timeout_positions) == 1
            and timeout_positions[0] + 1 < len(coordinator_argv)
        ):
            try:
                explicit_timeout = int(coordinator_argv[timeout_positions[0] + 1])
            except ValueError:
                explicit_timeout = None
        if (
            runner.get("path") != EXPECTED_RUNNER
            or not isinstance(command, list)
            or not all(isinstance(token, str) for token in command)
            or len(command) <= len(original_argv) + 3
            or command[:3]
            != ["/bin/bash", str(expected_launcher), original_argv[0]]
            or command[-len(original_argv) :] != list(original_argv)
            or explicit_timeout != completion_timeout_ms
            or runner.get("command_argv_sha256") != _argv_sha256(command)
            or coordinator_identity["ppid"] != runner_identity["pid"]
        ):
            raise W16ShimError(f"rank {rank} guarded transaction launcher chain changed")
        if (
            prepared[rank].get("schema") != TRANSACTION_SCHEMA
            or prepared[rank].get("status") != "PREPARED"
            or prepared[rank].get("portable") != portable
            or prepared[rank].get("portable_sha256") != portable_sha
            or prepared[rank].get("node") != participants[rank]
            or armed[rank].get("schema") != TRANSACTION_SCHEMA
            or armed[rank].get("status") != "ARMED"
            or armed[rank].get("rank") != rank
            or armed[rank].get("prepared_sha256") != prepared_sha[rank]
            or _validate_identity(
                armed[rank].get("supervisor"),
                f"rank {rank} transaction supervisor",
            )["ppid"]
            != coordinator_identity["pid"]
        ):
            raise W16ShimError(f"rank {rank} transaction evidence mismatch")
    local_supervisor = _validate_identity(
        armed[node_rank].get("supervisor"), "local transaction supervisor"
    )
    if dict(parent_identity) != local_supervisor:
        raise W16ShimError("shim is not the exact transaction supervisor child")
    decision, _ = _read_canonical_json(
        transaction_root / "DECISION.json", "transaction GO decision"
    )
    if (
        decision.get("schema") != TRANSACTION_SCHEMA
        or decision.get("status") != "GO"
        or decision.get("rank") != 0
        or decision.get("portable_sha256") != portable_sha
        or decision.get("bindings")
        != {
            "prepared_sha256": {str(rank): prepared_sha[rank] for rank in (0, 1)},
            "armed_sha256": {str(rank): armed_sha[rank] for rank in (0, 1)},
        }
    ):
        raise W16ShimError("transaction did not publish the exact two-node GO")
    return {
        "portable_sha256": portable_sha,
        "python": executable,
        "participant": participants[node_rank],
    }


def _load_trainer(repository: Path) -> ModuleType:
    path = repository / "scripts" / "show_base" / "train_base_official_adapt_long.py"
    spec = importlib.util.spec_from_file_location(
        "semtalk_base_w16_transaction_trainer_contract", path
    )
    if spec is None or spec.loader is None:
        raise W16ShimError("cannot load the exact Base trainer contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option_name(token: str) -> str | None:
    if not token.startswith("--") or token == "--":
        return None
    return token.split("=", 1)[0]


def _validate_trainer_tokens(tokens: Sequence[str]) -> None:
    seen: set[str] = set()
    for token in tokens:
        option = _option_name(token)
        if option is not None:
            if option in seen:
                raise W16ShimError(f"duplicate trainer option: {option}")
            seen.add(option)
            if option in RESERVED_TRAINER_OPTIONS:
                raise W16ShimError(f"shim owns trainer option: {option}")
        if E30.search(token) or SPEAKER2.search(token):
            raise W16ShimError("e30/Speaker2 sources are forbidden")
        if "semgate" in token.casefold() or "sparse" in token.casefold():
            raise W16ShimError("SemGate/Sparse paths or modes are forbidden")


def prepare_trainer_launch(
    *,
    trainer: ModuleType,
    trainer_tokens: Sequence[str],
    topology_mode: str,
    node_rank: int,
    master_addr: str,
    master_port: int,
    formal_run_id: str,
) -> tuple[argparse.Namespace, list[str]]:
    if topology_mode not in W16_TOPOLOGIES:
        raise W16ShimError("transaction shim accepts only exact W16 topologies")
    _validate_trainer_tokens(trainer_tokens)
    topology = W16_TOPOLOGIES[topology_mode]
    derived = [
        "--formal-node-rank",
        str(node_rank),
        "--formal-master-addr",
        master_addr,
        "--formal-master-port",
        str(master_port),
        "--formal-run-id",
        formal_run_id,
        "--topology-mode",
        topology_mode,
        "--local-batch-size",
        str(topology["local_batch_size"]),
        "--learning-rate",
        str(topology["learning_rate"]),
        "--precision",
        str(topology["precision"]),
        "--seed",
        "43",
    ]
    try:
        args = trainer.build_parser().parse_args([*trainer_tokens, *derived])
        trainer.validate_args(args)
    except SystemExit as exc:
        raise W16ShimError("invalid Base trainer argv") from exc
    except BaseException as exc:
        raise W16ShimError(f"Base trainer contract rejected argv: {exc}") from exc
    return args, [*trainer_tokens, *derived]


def build_input_set_receipt(
    *,
    trainer: ModuleType,
    trainer_args: argparse.Namespace,
) -> dict[str, Any]:
    """Hash and semantically replay every immutable Base training input."""
    try:
        dataset = trainer.validate_dataset_receipts(trainer_args)
        contract = trainer.validate_long_contract_receipts(
            trainer_args, dataset_receipt=dataset
        )
        topology = trainer.validate_topology_gate_spec(trainer_args)
    except BaseException as exc:
        raise W16ShimError(f"SHOW All input contract failed: {exc}") from exc
    if (
        dataset.get("format")
        != "semtalk_show_base_selected_feature_dataset_receipt_v1"
        or dataset.get("prerequisite_source")
        != trainer.SHOW_VAL_SELECTED_SOURCE
        or dataset.get("split") != "train"
        or dataset.get("test_visible") is not False
        or dataset.get("global_verified_not_consumed") is not True
        or trainer.SHOW_SPEAKERS != EXPECTED_SHOW_SPEAKERS
    ):
        raise W16ShimError("inputs are not the exact SHOW All five-stage Base set")
    file_arguments = {
        "official_base_checkpoint": trainer_args.official_base_checkpoint,
        "dataset_summary": trainer_args.dataset_summary,
        "lineage_manifest": trainer_args.lineage_manifest,
        "prerequisite_selection": trainer_args.prerequisite_selection_json,
        "schedule": trainer_args.schedule_json,
        "topology_gate_spec": trainer_args.topology_gate_spec,
    }
    if trainer_args.throughput_gate_report is not None:
        file_arguments["throughput_gate_report"] = trainer_args.throughput_gate_report
    if trainer_args.topology_selection_report is not None:
        file_arguments["topology_selection_report"] = (
            trainer_args.topology_selection_report
        )
    files = {
        logical_id: _snapshot_file(Path(str(path)), logical_id)
        for logical_id, path in sorted(file_arguments.items())
    }
    if files["official_base_checkpoint"]["sha256"] != trainer.OFFICIAL_BASE_SPEC["sha256"]:
        raise W16ShimError("official All-Speakers Base checkpoint changed")
    lmdb = Path(trainer_args.train_lmdb).resolve(strict=True)
    if lmdb != Path(trainer_args.train_lmdb) or not lmdb.is_dir() or lmdb.is_symlink():
        raise W16ShimError("train LMDB must be a canonical non-symlink directory")
    lmdb_binding = dataset.get("lmdb_inode_binding")
    if (
        not isinstance(lmdb_binding, dict)
        or lmdb_binding.get("format")
        != "semtalk_show_base_lmdb_inode_binding_v1"
        or not isinstance(lmdb_binding.get("files"), dict)
    ):
        raise W16ShimError("dataset replay omitted immutable LMDB evidence")
    for filename, receipt_key in (("data.mdb", "data_mdb_sha256"), ("lock.mdb", "lock_mdb_sha256")):
        path = lmdb / filename
        recorded = lmdb_binding["files"].get(filename)
        if not isinstance(recorded, dict) or not isinstance(recorded.get("identity"), dict):
            raise W16ShimError(f"dataset replay omitted {filename} identity")
        try:
            current_stat = os.stat(path, follow_symlinks=False)
            current_identity = trainer._stat_identity(current_stat)
        except OSError as exc:
            raise W16ShimError(f"train LMDB {filename} disappeared") from exc
        if (
            stat.S_ISLNK(current_stat.st_mode)
            or not stat.S_ISREG(current_stat.st_mode)
            or recorded.get("sha256") != dataset[receipt_key]
            or current_identity != recorded["identity"]
        ):
            raise W16ShimError(f"train LMDB {filename} changed after replay")
        info = {
            "path": str(path),
            "sha256": dataset[receipt_key],
            "bytes": current_identity["size"],
        }
        files[f"train_lmdb_{filename.replace('.', '_')}"] = info
    receipt = {
        "format": "semtalk_show_base_w16_transaction_input_set_v1",
        "dataset": "SHOW",
        "speaker_scope": "All",
        "speakers": EXPECTED_SHOW_SPEAKERS,
        "base_scope": "SemTalk Base only",
        "forbidden_training_components": ["SemGate", "Sparse"],
        "forbidden_sources": ["e30", "Speaker2"],
        "files": files,
        "dataset_receipt_payload_sha256": trainer.canonical_json_sha256(
            trainer._portable_dataset_receipt(dataset)
        ),
        "long_contract_schedule_sha256": contract["schedule"]["sha256"],
        "topology_gate_spec_sha256": topology["sha256"],
    }
    observed = _sha256_bytes(_canonical_json_bytes(receipt))
    return {"sha256": observed, "receipt": receipt}


def audit_inputs(
    *,
    trainer: ModuleType,
    trainer_args: argparse.Namespace,
    expected_input_set_sha256: str,
) -> dict[str, Any]:
    """Compare a replayed input set to the pin in both nodes' common argv."""

    if not HEX64.fullmatch(expected_input_set_sha256):
        raise W16ShimError("expected input-set SHA-256 must be lowercase 64-hex")
    result = build_input_set_receipt(
        trainer=trainer,
        trainer_args=trainer_args,
    )
    observed = result["sha256"]
    if observed != expected_input_set_sha256:
        raise W16ShimError(
            f"two-node input-set SHA-256 {observed} != external pin "
            f"{expected_input_set_sha256}"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audited W16 transaction workload for SemTalk Base on SHOW All"
    )
    parser.allow_abbrev = False
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument("--topology-mode", choices=tuple(W16_TOPOLOGIES), required=True)
    parser.add_argument("--expected-input-set-sha256", required=True)
    parser.add_argument("trainer_argv", nargs=argparse.REMAINDER)
    return parser


def _derive_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only derivation of the exact W16 input-set SHA-256"
    )
    parser.allow_abbrev = False
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--node-rank", required=True, type=int, choices=(0, 1))
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument("--topology-mode", choices=tuple(W16_TOPOLOGIES), required=True)
    parser.add_argument("trainer_argv", nargs=argparse.REMAINDER)
    return parser


def derive_input_set_sha256(argv: Sequence[str]) -> int:
    """Replay the same CPU input preflight before constructing common argv."""

    args = _derive_parser().parse_args(argv)
    trainer_tokens = list(args.trainer_argv)
    if not trainer_tokens or trainer_tokens.pop(0) != "--":
        raise W16ShimError("trainer arguments require one explicit -- delimiter")
    repository = Path(__file__).resolve().parents[2]
    source = audit_source(
        repository,
        expected_commit=args.source_commit,
        expected_tree=args.source_tree,
    )
    if source["detached"] is not True:
        raise W16ShimError("input derivation source is not detached")
    trainer = _load_trainer(repository)
    trainer_args, _ = prepare_trainer_launch(
        trainer=trainer,
        trainer_tokens=trainer_tokens,
        topology_mode=args.topology_mode,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        formal_run_id=args.formal_run_id,
    )
    result = build_input_set_receipt(
        trainer=trainer,
        trainer_args=trainer_args,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "derive-input-set-sha256":
        return derive_input_set_sha256(tokens[1:])
    args = _parser().parse_args(tokens)
    trainer_tokens = list(args.trainer_argv)
    if not trainer_tokens or trainer_tokens.pop(0) != "--":
        raise W16ShimError("trainer arguments require one explicit -- delimiter")
    repository = Path(__file__).resolve().parents[2]
    source = audit_source(
        repository,
        expected_commit=args.source_commit,
        expected_tree=args.source_tree,
    )
    required_environment = {
        "SEMTALK_W16_TRANSACTION_ROOT",
        "SEMTALK_W16_RUN_ID",
        "SEMTALK_W16_NODE_ID",
        "SEMTALK_W16_NODE_RANK",
        "SEMTALK_W16_MAX_RESTARTS",
    }
    if not required_environment.issubset(os.environ):
        raise W16ShimError("transaction workload environment is incomplete")
    try:
        node_rank = int(os.environ["SEMTALK_W16_NODE_RANK"])
    except ValueError as exc:
        raise W16ShimError("transaction node rank is not an integer") from exc
    if (
        os.environ["SEMTALK_W16_RUN_ID"] != args.formal_run_id
        or os.environ["SEMTALK_W16_MAX_RESTARTS"] != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_GPUS
    ):
        raise W16ShimError("transaction run/restart/GPU environment changed")
    executable = Path(sys.executable).resolve(strict=True)
    shim = Path(__file__).resolve(strict=True)
    original_argv = [str(executable), str(shim), *sys.argv[1:]]
    context = validate_transaction_context(
        transaction_root=Path(os.environ["SEMTALK_W16_TRANSACTION_ROOT"]),
        run_id=args.formal_run_id,
        node_id=os.environ["SEMTALK_W16_NODE_ID"],
        node_rank=node_rank,
        source=source,
        master_addr=args.master_addr,
        master_port=args.master_port,
        original_argv=original_argv,
        parent_identity=_proc_identity(os.getppid()),
        hostname=socket.gethostname(),
    )
    if context["python"]["path"] != str(executable):
        raise W16ShimError("current Python differs from transaction executable")
    trainer = _load_trainer(repository)
    trainer_args, normalized_trainer_argv = prepare_trainer_launch(
        trainer=trainer,
        trainer_tokens=trainer_tokens,
        topology_mode=args.topology_mode,
        node_rank=node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        formal_run_id=args.formal_run_id,
    )
    audit_inputs(
        trainer=trainer,
        trainer_args=trainer_args,
        expected_input_set_sha256=args.expected_input_set_sha256,
    )
    trainer_path = repository / "scripts" / "show_base" / "train_base_official_adapt_long.py"
    command = [
        str(executable),
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc_per_node=8",
        f"--node_rank={node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        str(trainer_path),
        *normalized_trainer_argv,
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "43"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.execve(str(executable), command, environment)
    raise AssertionError("os.execve returned")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except W16ShimError as exc:
        print(f"W16 transaction workload rejected: {exc}", file=sys.stderr)
        raise SystemExit(3)
