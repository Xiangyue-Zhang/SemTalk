from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "base_w16_transaction_workload.py"
)
SPEC = importlib.util.spec_from_file_location("base_w16_transaction_workload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SHIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHIM)


def _write_json(path: Path, payload: object) -> str:
    raw = SHIM._canonical_json_bytes(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _identity(pid: int) -> dict[str, object]:
    return {
        "pid": pid,
        "ppid": 1,
        "pgid": pid,
        "sid": pid,
        "starttime_ticks": pid + 100,
        "argv_sha256": "a" * 64,
    }


class BaseW16TransactionWorkloadTests(unittest.TestCase):
    def _transaction_fixture(
        self,
        root: Path,
        *,
        argv: list[str],
        source: dict[str, object],
        run_id: str = "formal-w16-run-001",
        completion_timeout_ms: int = SHIM.W16_MIN_COMPLETION_TIMEOUT_MS,
        command_timeout_ms: int | None = None,
    ) -> tuple[Path, dict[str, object]]:
        transaction_root = root / run_id
        transaction_root.mkdir()
        python = Path(argv[0])
        executable = SHIM._snapshot_file(python, "test Python")
        participants = [
            {"node_id": SHIM.EXPECTED_HOST_BY_RANK[rank], "rank": rank, "ip": f"127.0.0.{rank + 1}"}
            for rank in (0, 1)
        ]
        portable = {
            "schema": SHIM.PORTABLE_SCHEMA,
            "run_id": run_id,
            "namespace": {},
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "source": {
                "origin": source["origin"],
                "commit": source["commit"],
                "tree": source["tree"],
                "entrypoint_sha256": {
                    relative: source["entrypoint_sha256"][relative]
                    for relative in (
                        "scripts/show_base/dual_node_guarded_transaction.py",
                        "scripts/show_base/run_dual_node_guarded_transaction.sh",
                        "scripts/show_base/guarded_runner_contract.sh",
                    )
                },
            },
            "runner_sha256": "b" * 64,
            "common_command_sha256": SHIM._argv_sha256(argv),
            "workload": {
                "argv_sha256": SHIM._argv_sha256(argv),
                "executable_path": str(python),
                "executable_sha256": executable["sha256"],
                "workdir": str(SCRIPT.parents[2]),
                "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                "input_sha256": {},
                "input_bindings": {},
                "exec_argv_sha256": SHIM._argv_sha256(argv),
            },
            "participants": participants,
            "max_restarts": 0,
            "timeouts": {"completion_timeout_ms": completion_timeout_ms},
        }
        portable_sha = hashlib.sha256(SHIM._canonical_json_bytes(portable)).hexdigest()
        _write_json(
            transaction_root / "TRANSACTION.json",
            {
                "schema": SHIM.TRANSACTION_SCHEMA,
                "status": "OPEN",
                "portable": portable,
                "portable_sha256": portable_sha,
            },
        )
        prepared_sha: dict[int, str] = {}
        armed_sha: dict[int, str] = {}
        for rank in (0, 1):
            runner_identity = _identity(800 + rank)
            command = [
                "/bin/bash",
                str(
                    SCRIPT.parents[2]
                    / "scripts"
                    / "show_base"
                    / "run_dual_node_guarded_transaction.sh"
                ),
                argv[0],
                "--transaction-root",
                str(transaction_root),
                "--completion-timeout-ms",
                str(
                    completion_timeout_ms
                    if command_timeout_ms is None
                    else command_timeout_ms
                ),
                "--",
                *argv,
            ]
            runner = {
                **runner_identity,
                "path": SHIM.EXPECTED_RUNNER,
                "command": command,
                "command_argv_sha256": SHIM._argv_sha256(command),
            }
            coordinator = _identity(900 + rank)
            coordinator["ppid"] = runner_identity["pid"]
            prepared = {
                "schema": SHIM.TRANSACTION_SCHEMA,
                "status": "PREPARED",
                "portable": portable,
                "portable_sha256": portable_sha,
                "node": participants[rank],
                "runner": runner,
                "coordinator": coordinator,
            }
            prepared_sha[rank] = _write_json(
                transaction_root / f"PREPARED.rank{rank}.json", prepared
            )
            armed = {
                "schema": SHIM.TRANSACTION_SCHEMA,
                "status": "ARMED",
                "rank": rank,
                "prepared_sha256": prepared_sha[rank],
                "supervisor": {
                    **_identity(1000 + rank),
                    "ppid": coordinator["pid"],
                },
            }
            armed_sha[rank] = _write_json(
                transaction_root / f"ARMED.rank{rank}.json", armed
            )
        _write_json(
            transaction_root / "DECISION.json",
            {
                "schema": SHIM.TRANSACTION_SCHEMA,
                "status": "GO",
                "rank": 0,
                "portable_sha256": portable_sha,
                "bindings": {
                    "prepared_sha256": {str(rank): prepared_sha[rank] for rank in (0, 1)},
                    "armed_sha256": {str(rank): armed_sha[rank] for rank in (0, 1)},
                },
            },
        )
        return transaction_root, portable

    def test_transaction_replay_accepts_exact_rank_host_argv_and_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = Path(os.path.realpath(os.sys.executable))
            argv = [str(python), str(SCRIPT), "--source-commit", "c" * 40]
            source = {
                "origin": SHIM.EXPECTED_ORIGIN,
                "commit": "c" * 40,
                "tree": "d" * 40,
                "detached": True,
                "entrypoint_sha256": {
                    "scripts/show_base/dual_node_guarded_transaction.py": "1" * 64,
                    "scripts/show_base/run_dual_node_guarded_transaction.sh": "2" * 64,
                    "scripts/show_base/guarded_runner_contract.sh": "3" * 64,
                    "scripts/show_base/base_w16_transaction_workload.py": "4" * 64,
                    "scripts/show_base/train_base_official_adapt_long.py": "5" * 64,
                },
            }
            transaction_root, _ = self._transaction_fixture(
                root, argv=argv, source=source
            )
            result = SHIM.validate_transaction_context(
                transaction_root=transaction_root,
                run_id=transaction_root.name,
                node_id=SHIM.EXPECTED_HOST_BY_RANK[0],
                node_rank=0,
                source=source,
                master_addr=SHIM.EXPECTED_HOST_BY_RANK[0],
                master_port=29601,
                original_argv=argv,
                parent_identity={**_identity(1000), "ppid": 900},
                hostname=SHIM.EXPECTED_HOST_BY_RANK[0],
            )
            self.assertRegex(result["portable_sha256"], r"^[0-9a-f]{64}$")

    def test_transaction_replay_rejects_inherited_short_completion_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = Path(os.path.realpath(os.sys.executable))
            argv = [str(python), str(SCRIPT), "--source-commit", "c" * 40]
            source = {
                "origin": SHIM.EXPECTED_ORIGIN,
                "commit": "c" * 40,
                "tree": "d" * 40,
                "detached": True,
                "entrypoint_sha256": {
                    "scripts/show_base/dual_node_guarded_transaction.py": "1" * 64,
                    "scripts/show_base/run_dual_node_guarded_transaction.sh": "2" * 64,
                    "scripts/show_base/guarded_runner_contract.sh": "3" * 64,
                    "scripts/show_base/base_w16_transaction_workload.py": "4" * 64,
                    "scripts/show_base/train_base_official_adapt_long.py": "5" * 64,
                },
            }
            transaction_root, _ = self._transaction_fixture(
                root,
                argv=argv,
                source=source,
                completion_timeout_ms=120_000,
            )
            with self.assertRaisesRegex(SHIM.W16ShimError, "at least 24 hours"):
                SHIM.validate_transaction_context(
                    transaction_root=transaction_root,
                    run_id=transaction_root.name,
                    node_id=SHIM.EXPECTED_HOST_BY_RANK[0],
                    node_rank=0,
                    source=source,
                    master_addr=SHIM.EXPECTED_HOST_BY_RANK[0],
                    master_port=29601,
                    original_argv=argv,
                    parent_identity={**_identity(1000), "ppid": 900},
                    hostname=SHIM.EXPECTED_HOST_BY_RANK[0],
                )

    def test_transaction_replay_rejects_timeout_argv_portable_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = Path(os.path.realpath(os.sys.executable))
            argv = [str(python), str(SCRIPT), "--source-commit", "c" * 40]
            source = {
                "origin": SHIM.EXPECTED_ORIGIN,
                "commit": "c" * 40,
                "tree": "d" * 40,
                "detached": True,
                "entrypoint_sha256": {
                    "scripts/show_base/dual_node_guarded_transaction.py": "1" * 64,
                    "scripts/show_base/run_dual_node_guarded_transaction.sh": "2" * 64,
                    "scripts/show_base/guarded_runner_contract.sh": "3" * 64,
                    "scripts/show_base/base_w16_transaction_workload.py": "4" * 64,
                    "scripts/show_base/train_base_official_adapt_long.py": "5" * 64,
                },
            }
            transaction_root, _ = self._transaction_fixture(
                root,
                argv=argv,
                source=source,
                command_timeout_ms=SHIM.W16_MIN_COMPLETION_TIMEOUT_MS + 1,
            )
            with self.assertRaisesRegex(
                SHIM.W16ShimError, "launcher chain changed"
            ):
                SHIM.validate_transaction_context(
                    transaction_root=transaction_root,
                    run_id=transaction_root.name,
                    node_id=SHIM.EXPECTED_HOST_BY_RANK[0],
                    node_rank=0,
                    source=source,
                    master_addr=SHIM.EXPECTED_HOST_BY_RANK[0],
                    master_port=29601,
                    original_argv=argv,
                    parent_identity={**_identity(1000), "ppid": 900},
                    hostname=SHIM.EXPECTED_HOST_BY_RANK[0],
                )

    def test_transaction_replay_rejects_cross_node_argv_or_parent_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = Path(os.path.realpath(os.sys.executable))
            argv = [str(python), str(SCRIPT), "--formal-run-id", "formal-w16-run-001"]
            source = {
                "origin": SHIM.EXPECTED_ORIGIN,
                "commit": "c" * 40,
                "tree": "d" * 40,
                "detached": True,
                "entrypoint_sha256": {
                    "scripts/show_base/dual_node_guarded_transaction.py": "1" * 64,
                    "scripts/show_base/run_dual_node_guarded_transaction.sh": "2" * 64,
                    "scripts/show_base/guarded_runner_contract.sh": "3" * 64,
                    "scripts/show_base/base_w16_transaction_workload.py": "4" * 64,
                    "scripts/show_base/train_base_official_adapt_long.py": "5" * 64,
                },
            }
            transaction_root, _ = self._transaction_fixture(
                root, argv=argv, source=source
            )
            common = dict(
                transaction_root=transaction_root,
                run_id=transaction_root.name,
                node_id=SHIM.EXPECTED_HOST_BY_RANK[1],
                node_rank=1,
                source=source,
                master_addr=SHIM.EXPECTED_HOST_BY_RANK[0],
                master_port=29601,
                hostname=SHIM.EXPECTED_HOST_BY_RANK[1],
            )
            with self.assertRaisesRegex(SHIM.W16ShimError, "argv/input"):
                SHIM.validate_transaction_context(
                    **common,
                    original_argv=[*argv, "--tampered"],
                    parent_identity={**_identity(1001), "ppid": 901},
                )
            with self.assertRaisesRegex(SHIM.W16ShimError, "supervisor child"):
                SHIM.validate_transaction_context(
                    **common,
                    original_argv=argv,
                    parent_identity=_identity(9999),
                )

    def test_source_audit_requires_detached_clean_exact_commit_and_blobs(self) -> None:
        commit = subprocess_output = __import__("subprocess").check_output(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
        ).strip()
        tree = __import__("subprocess").check_output(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        # The worktree is intentionally dirty before its final commit, so use
        # mocks to exercise the detached/dirty decision independently of Git.
        completed = types.SimpleNamespace
        outputs = {
            ("remote", "get-url", "origin"): (0, SHIM.EXPECTED_ORIGIN + "\n"),
            ("rev-parse", "HEAD"): (0, commit + "\n"),
            ("rev-parse", "HEAD^{tree}"): (0, tree + "\n"),
            ("status", "--porcelain", "--untracked-files=all"): (0, ""),
            ("symbolic-ref", "-q", "HEAD"): (1, ""),
        }

        def fake_git(_repository: Path, *arguments: str, check: bool = True):
            if arguments[0] == "show":
                relative = arguments[1].split(":", 1)[1]
                return completed(returncode=0, stdout=(REPOSITORY / relative).read_bytes(), stderr=b"")
            returncode, stdout = outputs[arguments]
            return completed(returncode=returncode, stdout=stdout.encode(), stderr=b"")

        with mock.patch.object(SHIM, "_git", side_effect=fake_git):
            receipt = SHIM.audit_source(
                REPOSITORY, expected_commit=commit, expected_tree=tree
            )
            self.assertTrue(receipt["detached"])
        outputs[("symbolic-ref", "-q", "HEAD")] = (0, "refs/heads/main\n")
        with mock.patch.object(SHIM, "_git", side_effect=fake_git):
            with self.assertRaisesRegex(SHIM.W16ShimError, "clean detached"):
                SHIM.audit_source(
                    REPOSITORY, expected_commit=commit, expected_tree=tree
                )

    def test_trainer_launch_is_w16_fixed_and_rejects_forbidden_sources(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", required=True)
        parser.add_argument("--epochs", type=int, required=True)
        for option in SHIM.RESERVED_TRAINER_OPTIONS:
            parser.add_argument(option)
        trainer = types.SimpleNamespace(
            build_parser=lambda: parser,
            validate_args=lambda args: None,
        )
        args, tokens = SHIM.prepare_trainer_launch(
            trainer=trainer,
            trainer_tokens=["--mode", "train", "--epochs", "400"],
            topology_mode="validation_gated_w16_l32_g512_empirical_acceleration",
            node_rank=1,
            master_addr=SHIM.EXPECTED_HOST_BY_RANK[0],
            master_port=29601,
            formal_run_id="formal-w16-run-001",
        )
        self.assertEqual(args.formal_node_rank, "1")
        self.assertEqual(args.formal_host_slot, "1")
        self.assertEqual(
            tokens[tokens.index("--formal-host-slot") + 1],
            "1",
        )
        self.assertIn("--local-batch-size", tokens)
        self.assertIn("32", tokens)
        with self.assertRaisesRegex(
            SHIM.W16ShimError,
            "shim owns trainer option",
        ):
            SHIM.prepare_trainer_launch(
                trainer=trainer,
                trainer_tokens=[
                    "--mode", "train", "--epochs", "400",
                    "--formal-host-slot", "0",
                ],
                topology_mode=(
                    "validation_gated_w16_l32_g512_empirical_acceleration"
                ),
                node_rank=0,
                master_addr=SHIM.EXPECTED_HOST_BY_RANK[0],
                master_port=29601,
                formal_run_id="formal-w16-run-001",
            )
        with self.assertRaisesRegex(SHIM.W16ShimError, "Speaker2"):
            SHIM.prepare_trainer_launch(
                trainer=trainer,
                trainer_tokens=["--mode", "train", "--epochs", "400", "--x=speaker2"],
                topology_mode="validation_gated_w16_l32_g512_empirical_acceleration",
                node_rank=0,
                master_addr=SHIM.EXPECTED_HOST_BY_RANK[0],
                master_port=29601,
                formal_run_id="formal-w16-run-001",
            )

    def test_input_receipt_requires_show_all_semantics_and_external_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = {}
            for name in (
                "best_semtalk_base.bin",
                "summary.json",
                "lineage.json",
                "selection.json",
                "schedule.json",
                "topology.json",
            ):
                path = root / name
                path.write_bytes(name.encode())
                paths[name] = path
            lmdb = root / "lmdb"
            lmdb.mkdir()
            (lmdb / "data.mdb").write_bytes(b"data")
            (lmdb / "lock.mdb").write_bytes(b"lock")
            args = argparse.Namespace(
                official_base_checkpoint=str(paths["best_semtalk_base.bin"]),
                dataset_summary=str(paths["summary.json"]),
                lineage_manifest=str(paths["lineage.json"]),
                prerequisite_selection_json=str(paths["selection.json"]),
                schedule_json=str(paths["schedule.json"]),
                topology_gate_spec=str(paths["topology.json"]),
                throughput_gate_report=None,
                topology_selection_report=None,
                train_lmdb=str(lmdb),
            )
            data_sha = hashlib.sha256(b"data").hexdigest()
            lock_sha = hashlib.sha256(b"lock").hexdigest()
            official_sha = hashlib.sha256(b"best_semtalk_base.bin").hexdigest()
            trainer = types.SimpleNamespace(
                SHOW_VAL_SELECTED_SOURCE="show_val_selected_five_v1",
                SHOW_SPEAKERS=SHIM.EXPECTED_SHOW_SPEAKERS,
                OFFICIAL_BASE_SPEC={"sha256": official_sha},
                validate_dataset_receipts=lambda _args: {
                    "format": "semtalk_show_base_selected_feature_dataset_receipt_v1",
                    "prerequisite_source": "show_val_selected_five_v1",
                    "split": "train",
                    "test_visible": False,
                    "global_verified_not_consumed": True,
                    "data_mdb_sha256": data_sha,
                    "lock_mdb_sha256": lock_sha,
                    "lmdb_inode_binding": {
                        "format": "semtalk_show_base_lmdb_inode_binding_v1",
                        "files": {
                            "data.mdb": {
                                "sha256": data_sha,
                                "identity": {
                                    "device": (lmdb / "data.mdb").stat().st_dev,
                                    "inode": (lmdb / "data.mdb").stat().st_ino,
                                    "size": 4,
                                    "mtime_ns": (lmdb / "data.mdb").stat().st_mtime_ns,
                                    "ctime_ns": (lmdb / "data.mdb").stat().st_ctime_ns,
                                },
                            },
                            "lock.mdb": {
                                "sha256": lock_sha,
                                "identity": {
                                    "device": (lmdb / "lock.mdb").stat().st_dev,
                                    "inode": (lmdb / "lock.mdb").stat().st_ino,
                                    "size": 4,
                                    "mtime_ns": (lmdb / "lock.mdb").stat().st_mtime_ns,
                                    "ctime_ns": (lmdb / "lock.mdb").stat().st_ctime_ns,
                                },
                            },
                        },
                    },
                },
                validate_long_contract_receipts=lambda _args, dataset_receipt: {
                    "schedule": {"sha256": hashlib.sha256(b"schedule.json").hexdigest()}
                },
                validate_topology_gate_spec=lambda _args: {
                    "sha256": hashlib.sha256(b"topology.json").hexdigest()
                },
                _portable_dataset_receipt=lambda receipt: receipt,
                canonical_json_sha256=lambda payload: hashlib.sha256(
                    SHIM._canonical_json_bytes(payload)
                ).hexdigest(),
                _stat_identity=lambda value: {
                    "device": value.st_dev,
                    "inode": value.st_ino,
                    "size": value.st_size,
                    "mtime_ns": value.st_mtime_ns,
                    "ctime_ns": value.st_ctime_ns,
                },
            )
            derived = SHIM.build_input_set_receipt(
                trainer=trainer,
                trainer_args=args,
            )
            self.assertRegex(derived["sha256"], r"^[0-9a-f]{64}$")
            accepted = SHIM.audit_inputs(
                trainer=trainer,
                trainer_args=args,
                expected_input_set_sha256=derived["sha256"],
            )
            self.assertEqual(accepted, derived)
            with self.assertRaisesRegex(SHIM.W16ShimError, "external pin"):
                SHIM.audit_inputs(
                    trainer=trainer,
                    trainer_args=args,
                    expected_input_set_sha256="0" * 64,
                )

    def test_source_contains_no_shell_eval_and_execs_fixed_torchrun(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)
        self.assertNotRegex(source, r"\beval\(")
        self.assertIn('"torch.distributed.run"', source)
        self.assertIn('"--nnodes=2"', source)
        self.assertIn('"--nproc_per_node=8"', source)
        self.assertIn("os.execve", source)
        self.assertIn('"scripts/show_base/train_base_official_adapt_long.py"', source)
        self.assertIn("derive-input-set-sha256", source)


if __name__ == "__main__":
    unittest.main()
