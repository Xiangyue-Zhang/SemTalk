from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts.show_base import dual_node_guarded_transaction as TRANSACTION


REPOSITORY = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY / "scripts" / "show_base" / "dual_node_guarded_transaction.py"
LAUNCHER = REPOSITORY / "scripts" / "show_base" / "run_dual_node_guarded_transaction.sh"
PYTHON = str(Path(sys.executable).resolve())

CPU_HEARTBEAT_MS = 250
CPU_STALE_MS = 1500
CPU_PREPARE_TIMEOUT_MS = 5000
CPU_DECISION_TIMEOUT_MS = 5000
CPU_ARM_TIMEOUT_MS = 5000
CPU_START_TIMEOUT_MS = 5000
CPU_COMPLETION_TIMEOUT_MS = 8000
CPU_SHUTDOWN_GRACE_MS = 2500
CPU_COORDINATOR_EXIT_TIMEOUT_SECONDS = (
    max(
        CPU_PREPARE_TIMEOUT_MS,
        CPU_DECISION_TIMEOUT_MS,
        CPU_ARM_TIMEOUT_MS,
        CPU_START_TIMEOUT_MS,
        CPU_COMPLETION_TIMEOUT_MS,
    )
    / 1000.0
    + max(2.0, 3.0 * CPU_SHUTDOWN_GRACE_MS / 1000.0 + 1.0)
    + 1.0
)


class RenameNoReplaceCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _unsupported_rename_libc() -> object:
        class UnsupportedRename:
            argtypes = None
            restype = None

            def __call__(self, *_args: object) -> int:
                return -1

        class UnsupportedLibc:
            renameat2 = UnsupportedRename()
            renameatx_np = UnsupportedRename()

        return UnsupportedLibc()

    def test_efs_einval_falls_back_to_atomic_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / ".tmp.OUTCOME.json"
            target = root / "OUTCOME.json"
            source.write_bytes(b"terminal\n")
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            previous_test_mode = TRANSACTION._CPU_TEST_MODE
            TRANSACTION._CPU_TEST_MODE = True
            try:
                with (
                    mock.patch.object(
                        TRANSACTION.ctypes,
                        "CDLL",
                        return_value=self._unsupported_rename_libc(),
                    ),
                    mock.patch.object(
                        TRANSACTION.ctypes,
                        "get_errno",
                        return_value=errno.EINVAL,
                    ),
                ):
                    TRANSACTION._rename_noreplace(
                        directory_fd,
                        source.name,
                        directory_fd,
                        target.name,
                    )
            finally:
                TRANSACTION._CPU_TEST_MODE = previous_test_mode
                os.close(directory_fd)
            self.assertFalse(source.exists())
            self.assertEqual(target.read_bytes(), b"terminal\n")

    def test_efs_fallback_never_rolls_back_published_target_on_cleanup_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / ".tmp.OUTCOME.json"
            target = root / "OUTCOME.json"
            source.write_bytes(b"terminal\n")
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            previous_test_mode = TRANSACTION._CPU_TEST_MODE
            TRANSACTION._CPU_TEST_MODE = True
            real_unlink = TRANSACTION.os.unlink

            def reject_source_cleanup(
                path: os.PathLike[str] | str,
                *,
                dir_fd: int | None = None,
            ) -> None:
                if os.fspath(path) == source.name and dir_fd == directory_fd:
                    raise OSError(errno.EIO, "injected source cleanup failure")
                real_unlink(path, dir_fd=dir_fd)

            try:
                with (
                    mock.patch.object(
                        TRANSACTION.ctypes,
                        "CDLL",
                        return_value=self._unsupported_rename_libc(),
                    ),
                    mock.patch.object(
                        TRANSACTION.ctypes,
                        "get_errno",
                        return_value=errno.EINVAL,
                    ),
                    mock.patch.object(
                        TRANSACTION.os,
                        "unlink",
                        side_effect=reject_source_cleanup,
                    ),
                ):
                    TRANSACTION._rename_noreplace(
                        directory_fd,
                        source.name,
                        directory_fd,
                        target.name,
                    )
            finally:
                TRANSACTION._CPU_TEST_MODE = previous_test_mode
                os.close(directory_fd)
            self.assertEqual(target.read_bytes(), b"terminal\n")
            self.assertEqual(source.read_bytes(), b"terminal\n")
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)

    def test_terminal_target_survives_post_publication_directory_fsync_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = parent / "transaction"
            transaction = TRANSACTION.TransactionDirectory(
                root,
                parent,
                root.name,
                0,
            )
            transaction.create_or_wait(time.monotonic() + 1.0)
            real_fsync = TRANSACTION.os.fsync

            def fail_root_directory_fsync(fd: int) -> None:
                if fd == transaction.root_fd:
                    raise OSError(errno.EIO, "injected directory fsync failure")
                real_fsync(fd)

            payload = {"schema": "test", "status": "SUCCEEDED"}
            previous_test_mode = TRANSACTION._CPU_TEST_MODE
            TRANSACTION._CPU_TEST_MODE = True
            try:
                with mock.patch.object(
                    TRANSACTION.os,
                    "fsync",
                    side_effect=fail_root_directory_fsync,
                ):
                    with self.assertRaisesRegex(OSError, "injected"):
                        transaction.publish_terminal_outcome(payload)
                self.assertTrue((root / "OUTCOME.json").is_file())
                observed, _ = transaction.read_json("OUTCOME.json")
                self.assertEqual(observed, payload)
            finally:
                TRANSACTION._CPU_TEST_MODE = previous_test_mode
                transaction.close()

HARNESS = r"""
import json
import os
import sys
from scripts.show_base import dual_node_guarded_transaction as module

module._CPU_TEST_MODE = True

def fake_runner(status_path, log_path):
    identity = module._proc_identity(os.getppid())
    formal_command = os.environ.get("SEMTALK_TEST_FORMAL_RUNNER_COMMAND")
    command = (
        json.loads(formal_command)
        if formal_command
        else ["/test/fake-transaction-coordinator"]
    )
    argv = [
        sys.executable,
        module.EXPECTED_RUNNER,
        "--gpus",
        module.EXPECTED_RUNNER_GPUS,
        "--status",
        status_path,
        "--log",
        log_path,
        "--",
        *command,
    ]
    identity.update({
        "path": module.EXPECTED_RUNNER,
        "sha256": "e" * 64,
        "status_path": status_path,
        "log_path": log_path,
        "argv": argv,
        "command": command,
        "command_argv_sha256": module._argv_sha256(command),
    })
    return identity, "e" * 64

def fake_source(args):
    return {
        "origin": module.EXPECTED_ORIGIN,
        "commit": args.source_commit,
        "tree": args.source_tree,
        "entrypoint_sha256": {
            "scripts/show_base/dual_node_guarded_transaction.py": "1" * 64,
            "scripts/show_base/run_dual_node_guarded_transaction.sh": "2" * 64,
            "scripts/show_base/guarded_runner_contract.sh": "3" * 64,
        },
    }

module._runner_evidence = fake_runner
module._source_evidence = fake_source

if os.environ.get("SEMTALK_TEST_FINAL_PUBLISH_FAIL"):
    original_publish = module.TransactionDirectory.publish_immutable
    def failing_publish(self, name, payload):
        if name == os.environ["SEMTALK_TEST_FINAL_PUBLISH_FAIL"]:
            raise OSError("injected finalizer publication failure")
        return original_publish(self, name, payload)
    module.TransactionDirectory.publish_immutable = failing_publish

if os.environ.get("SEMTALK_TEST_ARM_DELAY"):
    original_spawn = module.Coordinator._spawn_supervisor
    def delayed_spawn(self):
        import time
        time.sleep(float(os.environ["SEMTALK_TEST_ARM_DELAY"]))
        return original_spawn(self)
    module.Coordinator._spawn_supervisor = delayed_spawn

if os.environ.get("SEMTALK_TEST_REPLAY_RANK"):
    original_heartbeat = module.Coordinator._heartbeat
    def replaying_heartbeat(self):
        original_heartbeat(self)
        if (
            self.rank == int(os.environ["SEMTALK_TEST_REPLAY_RANK"])
            and self.sequence == 10
        ):
            self.sequence = 1
    module.Coordinator._heartbeat = replaying_heartbeat

raise SystemExit(module.main(sys.argv[1:]))
"""

WORKLOAD = r"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

configuration = json.loads(sys.argv[1])
rank = os.environ["SEMTALK_W16_NODE_RANK"]
entry = configuration[rank]
marker_root = Path(configuration["marker_root"])
marker_root.mkdir(parents=True, exist_ok=True)
(marker_root / f"started.rank{rank}").write_text(str(os.getpid()))
if entry.get("record_runtime"):
    python_target_fd_leaks = []
    proc_fd = Path("/proc/self/fd")
    if proc_fd.is_dir():
        target_stat = os.stat(Path(sys.executable).resolve())
        for fd_path in proc_fd.iterdir():
            try:
                fd = int(fd_path.name)
                if fd <= 2:
                    continue
                observed = os.fstat(fd)
            except (OSError, ValueError):
                continue
            if (
                observed.st_dev == target_stat.st_dev
                and observed.st_ino == target_stat.st_ino
            ):
                python_target_fd_leaks.append(fd)
    (marker_root / f"runtime.rank{rank}.json").write_text(json.dumps({
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "python_target_fd_leaks": python_target_fd_leaks,
    }, sort_keys=True))
if entry.get("spawn_descendant"):
    descendant = r'''import os,signal,sys,time
from pathlib import Path
if sys.argv[2] == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(20)
'''
    descendant_path = marker_root / f"descendant.rank{rank}"
    subprocess.Popen([
        sys.executable,
        "-c",
        descendant,
        str(descendant_path),
        "ignore" if entry.get("ignore_term") else "default",
    ], start_new_session=bool(entry.get("setsid")))
    deadline = time.monotonic() + 1.0
    while not descendant_path.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
if entry.get("read_authority"):
    option_index = sys.argv.index("--fresh-test-authority")
    observed = Path(sys.argv[option_index + 1]).read_text(encoding="utf-8")
    (marker_root / f"authority.rank{rank}").write_text(observed, encoding="utf-8")
    if observed != entry["expected_authority"]:
        sys.exit(91)
time.sleep(float(entry.get("seconds", 0.05)))
sys.exit(int(entry.get("rc", 0)))
"""


def argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(
        b"\0".join(os.fsencode(token) for token in argv) + b"\0"
    ).hexdigest()


def namespace_parent_sha256(path: Path) -> str:
    return hashlib.sha256(os.fsencode(str(path.resolve())) + b"\0").hexdigest()


class DualNodeGuardedTransactionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="semtalk-w16-2pc-")
        self.root = Path(self.temporary.name).resolve()
        self.status_paths: list[Path] = []
        self.log_paths: list[Path] = []
        for rank in (0, 1):
            status = self.root / f"runner.rank{rank}.status.json"
            log = self.root / f"runner.rank{rank}.log"
            status.write_text("{}\n", encoding="utf-8")
            log.write_text("", encoding="utf-8")
            self.status_paths.append(status)
            self.log_paths.append(log)
        self.processes: list[subprocess.Popen[str]] = []

    def tearDown(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGCONT)
                    process.terminate()
                    process.wait(timeout=3)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                        process.wait(timeout=2)
                    except ProcessLookupError:
                        pass
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        self.temporary.cleanup()

    def _configuration(
        self,
        *,
        seconds0: float = 0.05,
        seconds1: float = 0.05,
        rc0: int = 0,
        rc1: int = 0,
        descendants: bool = False,
        ignore_term: bool = False,
        setsid: bool = False,
        read_authority: bool = False,
        expected_authority: str = "",
        record_runtime: bool = False,
    ) -> dict[str, object]:
        return {
            "marker_root": str(self.root / "markers"),
            "0": {
                "seconds": seconds0,
                "rc": rc0,
                "spawn_descendant": descendants,
                "ignore_term": ignore_term,
                "setsid": setsid,
                "read_authority": read_authority,
                "expected_authority": expected_authority,
                "record_runtime": record_runtime,
            },
            "1": {
                "seconds": seconds1,
                "rc": rc1,
                "spawn_descendant": descendants,
                "ignore_term": ignore_term,
                "setsid": setsid,
                "read_authority": read_authority,
                "expected_authority": expected_authority,
                "record_runtime": record_runtime,
            },
        }

    def _protocol_args(
        self,
        rank: int,
        transaction_root: Path,
        configuration: dict[str, object],
        *,
        common_sha256: str | None = None,
        run_id: str | None = None,
        workload_suffix: list[str] | None = None,
        workload_inputs: dict[str, Path] | None = None,
        allow_env: list[str] | None = None,
        expected_parent_sha256: str | None = None,
        workload_python: str = PYTHON,
    ) -> list[str]:
        peer = 1 - rank
        config_text = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        workload = [workload_python, "-c", WORKLOAD, config_text]
        for logical_id, path in sorted((workload_inputs or {}).items()):
            workload.extend(
                [
                    "--fresh-test-authority",
                    str(path),
                ]
            )
        if workload_suffix:
            workload.extend(workload_suffix)
        common_sha256 = common_sha256 or argv_sha256(workload)
        # Linux production nodes need multiple /proc censuses to prove that
        # the exact descendant set is gone.  Keep the CPU fixture short, but
        # leave enough time for those real scans instead of assuming a
        # sub-200 ms cleanup path.
        arguments = [
            "--transaction-root",
            str(transaction_root),
            "--run-id",
            run_id or transaction_root.name,
            "--deployment-id",
            "cpu-two-node",
            "--expected-namespace-parent-sha256",
            expected_parent_sha256
            or namespace_parent_sha256(transaction_root.parent),
            "--source-commit",
            "a" * 40,
            "--source-tree",
            "b" * 40,
            "--common-command-sha256",
            common_sha256,
            "--node-id",
            f"node{rank}",
            "--node-rank",
            str(rank),
            "--node-ip",
            f"127.0.0.{rank + 1}",
            "--peer-node-id",
            f"node{peer}",
            "--peer-node-rank",
            str(peer),
            "--peer-node-ip",
            f"127.0.0.{peer + 1}",
            "--runner-status-path",
            str(self.status_paths[rank]),
            "--runner-log-path",
            str(self.log_paths[rank]),
            "--workdir",
            str(self.root),
            "--max-restarts",
            "0",
            "--heartbeat-ms",
            str(CPU_HEARTBEAT_MS),
            "--stale-ms",
            str(CPU_STALE_MS),
            "--prepare-timeout-ms",
            str(CPU_PREPARE_TIMEOUT_MS),
            "--decision-timeout-ms",
            str(CPU_DECISION_TIMEOUT_MS),
            "--arm-timeout-ms",
            str(CPU_ARM_TIMEOUT_MS),
            "--start-timeout-ms",
            str(CPU_START_TIMEOUT_MS),
            "--completion-timeout-ms",
            str(CPU_COMPLETION_TIMEOUT_MS),
            "--shutdown-grace-ms",
            str(CPU_SHUTDOWN_GRACE_MS),
        ]
        for name in allow_env or []:
            arguments.extend(["--allow-env", name])
        for logical_id, path in sorted((workload_inputs or {}).items()):
            arguments.extend(["--workload-input", f"{logical_id}={path}"])
        arguments.extend(["--", *workload])
        return arguments

    def _command(self, *args: object, **kwargs: object) -> list[str]:
        return [PYTHON, "-c", HARNESS, *self._protocol_args(*args, **kwargs)]

    def _start(
        self,
        command: list[str],
        *,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.Popen[str]:
        environment = os.environ.copy()
        environment.update(extra_environment or {})
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(process)
        return process

    def _run_pair(
        self,
        transaction_root: Path,
        configuration: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[subprocess.Popen[str], subprocess.Popen[str]]:
        configuration = configuration or self._configuration()
        node1 = self._start(self._command(1, transaction_root, configuration, **kwargs))
        node0 = self._start(self._command(0, transaction_root, configuration, **kwargs))
        return node0, node1

    def _formal_venv(self) -> tuple[Path, Path]:
        venv_root = self.root / "formal-venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        leaf = venv_root / "bin" / "python"
        self.assertTrue(leaf.is_symlink())
        return venv_root, leaf

    def _run_formal_pair(
        self,
        transaction_root: Path,
        venv_python: Path,
        configuration: dict[str, object],
        *,
        extra_environment: dict[str, str] | None = None,
    ) -> tuple[subprocess.Popen[str], subprocess.Popen[str]]:
        commands: list[list[str]] = []
        environments: list[dict[str, str]] = []
        for rank in (1, 0):
            arguments = self._protocol_args(
                rank,
                transaction_root,
                configuration,
                workload_python=str(venv_python),
            )
            workload = arguments[arguments.index("--") + 1 :]
            runner_command = [
                "/bin/bash",
                str(LAUNCHER),
                str(venv_python),
                "--transaction-root",
                str(transaction_root),
                "--",
                *workload,
            ]
            commands.append([str(venv_python), "-c", HARNESS, *arguments])
            environment = {
                "SEMTALK_TEST_FORMAL_RUNNER_COMMAND": json.dumps(runner_command),
                **(extra_environment or {}),
            }
            environments.append(environment)
        node1 = self._start(commands[0], extra_environment=environments[0])
        node0 = self._start(commands[1], extra_environment=environments[1])
        return node0, node1

    @staticmethod
    def _wait_process(process: subprocess.Popen[str]) -> int:
        return process.wait(timeout=CPU_COORDINATOR_EXIT_TIMEOUT_SECONDS)

    def _wait_for(
        self,
        path: Path,
        timeout: float = CPU_COORDINATOR_EXIT_TIMEOUT_SECONDS,
    ) -> None:
        deadline = time.monotonic() + timeout
        while not path.exists():
            if time.monotonic() >= deadline:
                errors = []
                for process in self.processes:
                    if process.poll() is not None and process.stderr is not None:
                        errors.append(process.stderr.read())
                self.fail(f"timed out waiting for {path}; errors={errors}")
            time.sleep(0.01)

    @staticmethod
    def _pid_is_live(
        pid: int,
        expected_identity: dict[str, object] | None = None,
    ) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        status = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not status or status.startswith("Z"):
            return False
        if expected_identity is not None:
            from scripts.show_base import dual_node_guarded_transaction as module

            try:
                observed = module._proc_identity(pid)
            except module.TransactionError:
                return False
            return (
                observed["starttime_ticks"]
                == expected_identity["starttime_ticks"]
                and observed["argv_sha256"]
                == expected_identity["argv_sha256"]
            )
        return True

    @staticmethod
    def _publish_artifact(path: Path, payload: dict[str, object]) -> None:
        raw = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode()
        temporary = path.with_name(f".test.{path.name}.{os.getpid()}")
        temporary.write_bytes(raw)
        temporary.chmod(0o400)
        os.link(temporary, path)
        temporary.unlink()

    def _finish_runner_receipts(self, transaction_root: Path) -> None:
        for rank in (0, 1):
            prepared = json.loads(
                (transaction_root / f"PREPARED.rank{rank}.json").read_text()
            )
            runner = prepared["runner"]
            status = {
                "state": "finished",
                "return_code": 0,
                "error": None,
                "cleanup_error": None,
                "restore_error": None,
                "received_signal": None,
                "command": runner["command"],
                "wrapper_pid": runner["pid"],
                "child_pid": prepared["coordinator"]["pid"],
                "restored_guards": {
                    str(index): 90000 + rank * 100 + index
                    for index in range(8)
                },
            }
            Path(runner["status_path"]).write_text(
                json.dumps(status, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            Path(runner["log_path"]).write_text(
                f"rank {rank} guarded runner restored\n",
                encoding="utf-8",
            )

    def _control_command(
        self,
        command: str,
        transaction_root: Path,
        *,
        outcome_sha256: str | None = None,
    ) -> list[str]:
        bootstrap = json.loads(
            (transaction_root / "TRANSACTION.json").read_text()
        )
        portable = bootstrap["portable"]
        result = [
            PYTHON,
            "-c",
            HARNESS,
            command,
            "--transaction-root",
            str(transaction_root),
            "--run-id",
            transaction_root.name,
            "--deployment-id",
            "cpu-two-node",
            "--expected-namespace-parent-sha256",
            namespace_parent_sha256(transaction_root.parent),
            "--expected-portable-sha256",
            bootstrap["portable_sha256"],
            "--source-commit",
            portable["source_commit"],
            "--source-tree",
            portable["source_tree"],
        ]
        if outcome_sha256 is not None:
            result.extend(["--expected-outcome-sha256", outcome_sha256])
        return result

    def _finalize(self, transaction_root: Path) -> dict[str, object]:
        self._finish_runner_receipts(transaction_root)
        result = subprocess.run(
            self._control_command("finalize", transaction_root),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def _replay(self, transaction_root: Path) -> dict[str, object]:
        outcome_sha = hashlib.sha256(
            (transaction_root / "OUTCOME.json").read_bytes()
        ).hexdigest()
        result = subprocess.run(
            self._control_command(
                "replay",
                transaction_root,
                outcome_sha256=outcome_sha,
            ),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_success_uses_armed_go_result_and_final_receipts(self) -> None:
        transaction_root = self.root / "success_tx"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        self.assertFalse((transaction_root / "OUTCOME.json").exists())
        self.assertFalse((transaction_root / "FINAL.rank0.json").exists())
        self.assertFalse((transaction_root / "FINAL.rank1.json").exists())
        finalized = self._finalize(transaction_root)
        self.assertEqual(finalized["status"], "FINALIZED_SUCCEEDED")
        decision = json.loads((transaction_root / "DECISION.json").read_text())
        self.assertEqual(set(decision), {
            "schema", "status", "rank", "portable_sha256", "reason", "bindings",
            "decision_unix_ns",
        })
        self.assertEqual(decision["status"], "GO")
        self.assertFalse((transaction_root / "COMMIT.json").exists())
        self.assertEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "SUCCEEDED",
        )
        for rank in (0, 1):
            self.assertTrue((transaction_root / f"ARMED.rank{rank}.json").is_file())
            started = json.loads((transaction_root / f"STARTED.rank{rank}.json").read_text())
            self.assertEqual(started["workload"]["pgid"], started["workgroup"]["pid"])
            result = json.loads(
                (transaction_root / f"WORKLOAD_RESULT.rank{rank}.json").read_text()
            )
            final = json.loads((transaction_root / f"FINAL.rank{rank}.json").read_text())
            self.assertEqual((result["status"], result["workload_returncode"]), ("COMPLETED", 0))
            self.assertEqual(final["status"], "SUCCEEDED")
            self.assertIn("restored_guards_by_gpu", final["outer_guarded_runner"]["guard_evidence"])
        replay = self._replay(transaction_root)
        self.assertEqual(replay["status"], "REPLAYED_SUCCEEDED")

    def test_actual_argv_must_equal_common_digest_before_spawn(self) -> None:
        transaction_root = self.root / "argv_mismatch"
        configuration = self._configuration()
        canonical = self._protocol_args(0, transaction_root, configuration)
        delimiter = canonical.index("--")
        workload = canonical[delimiter + 1 :]
        common = argv_sha256(workload)
        node1 = self._start(
            self._command(
                1,
                transaction_root,
                configuration,
                common_sha256=common,
                workload_suffix=["different"],
            )
        )
        node0 = self._start(
            self._command(0, transaction_root, configuration, common_sha256=common)
        )
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_pre_go_abort_wins_single_atomic_decision_and_never_spawns(self) -> None:
        transaction_root = self.root / "pre_go_abort"
        configuration = self._configuration(seconds0=8.0, seconds1=8.0)
        environment = {"SEMTALK_TEST_ARM_DELAY": "0.4"}
        node1 = self._start(
            self._command(1, transaction_root, configuration),
            extra_environment=environment,
        )
        node0 = self._start(
            self._command(0, transaction_root, configuration),
            extra_environment=environment,
        )
        self._wait_for(transaction_root / "PREPARED.rank0.json")
        self._wait_for(transaction_root / "PREPARED.rank1.json")
        prepared = json.loads((transaction_root / "PREPARED.rank0.json").read_text())
        self._publish_artifact(
            transaction_root / "DECISION.json",
            {
                "schema": prepared["schema"],
                "status": "ABORT",
                "rank": 1,
                "portable_sha256": prepared["portable_sha256"],
                "reason": "test abort before either gated workload can exec",
                "bindings": None,
                "decision_unix_ns": time.time_ns(),
            },
        )
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertEqual(
            json.loads((transaction_root / "DECISION.json").read_text())["status"],
            "ABORT",
        )
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_real_cli_rejects_unguarded_parent(self) -> None:
        transaction_root = self.root / "unguarded"
        arguments = self._protocol_args(0, transaction_root, self._configuration())
        result = subprocess.run(
            [PYTHON, str(HELPER), *arguments],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, "guarded|Linux /proc|exact")
        self.assertFalse((self.root / "markers" / "started.rank0").exists())

    def test_nonzero_result_has_exact_status_and_no_false_final(self) -> None:
        transaction_root = self.root / "failed_workload"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(rc0=7, seconds1=0.8),
        )
        self.assertEqual(self._wait_process(node0), 4, node0.stderr.read())
        self.assertNotEqual(self._wait_process(node1), 0)
        result = json.loads((transaction_root / "WORKLOAD_RESULT.rank0.json").read_text())
        self.assertEqual((result["status"], result["workload_returncode"]), ("WORKLOAD_FAILED", 7))
        self.assertFalse((transaction_root / "FINAL.rank0.json").exists())
        self.assertEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "FAILED",
        )

    def test_peer_signal_aborts_and_cleans_other_node(self) -> None:
        transaction_root = self.root / "peer_abort"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(seconds0=8.0, seconds1=8.0),
        )
        self._wait_for(transaction_root / "STARTED.rank0.json")
        self._wait_for(transaction_root / "STARTED.rank1.json")
        node1.terminate()
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "FAILED",
        )

    def test_stale_peer_after_local_result_changes_final_outcome(self) -> None:
        transaction_root = self.root / "stale_peer"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(seconds0=0.03, seconds1=8.0),
        )
        self._wait_for(transaction_root / "WORKLOAD_RESULT.rank0.json")
        os.kill(node1.pid, signal.SIGSTOP)
        self.assertNotEqual(self._wait_process(node0), 0)
        result = json.loads((transaction_root / "WORKLOAD_RESULT.rank0.json").read_text())
        self.assertEqual(result["status"], "COMPLETED")
        self.assertFalse((transaction_root / "FINAL.rank0.json").exists())
        self.assertNotEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "SUCCEEDED",
        )
        os.kill(node1.pid, signal.SIGCONT)
        node1.terminate()
        self._wait_process(node1)

    def test_descendant_cannot_outlive_successful_group_leader(self) -> None:
        transaction_root = self.root / "descendant_cleanup"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(descendants=True, ignore_term=True),
        )
        marker0 = self.root / "markers" / "descendant.rank0"
        marker1 = self.root / "markers" / "descendant.rank1"
        self._wait_for(marker0)
        self._wait_for(marker1)
        pids = [int(marker0.read_text()), int(marker1.read_text())]
        from scripts.show_base import dual_node_guarded_transaction as module

        identities = [module._proc_identity(pid) for pid in pids]
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            living = []
            for pid, identity in zip(pids, identities):
                if self._pid_is_live(pid, identity):
                    living.append(pid)
            if not living:
                break
            time.sleep(0.02)
        self.assertFalse(living, f"escaped descendants: {living}")

    def test_setsid_descendant_is_tracked_killed_and_proved_absent(self) -> None:
        transaction_root = self.root / "setsid_descendant_cleanup"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(
                descendants=True,
                ignore_term=True,
                setsid=True,
            ),
        )
        markers = [
            self.root / "markers" / f"descendant.rank{rank}"
            for rank in (0, 1)
        ]
        for marker in markers:
            self._wait_for(marker)
        pids = [int(marker.read_text()) for marker in markers]
        from scripts.show_base import dual_node_guarded_transaction as module

        identities = [module._proc_identity(pid) for pid in pids]
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        for rank, (pid, identity) in enumerate(zip(pids, identities)):
            cleanup = json.loads(
                (transaction_root / f"CLEANUP.rank{rank}.json").read_text()
            )
            self.assertEqual(cleanup["proof"]["live_after"], [])
            self.assertEqual(
                cleanup["proof"]["observed_consecutive_empty_censuses"],
                3,
            )
            self.assertTrue(
                any(record["pid"] == pid for record in cleanup["proof"]["tracked"])
            )
            self.assertFalse(self._pid_is_live(pid, identity))

    def test_peer_abort_also_cleans_setsid_descendants(self) -> None:
        transaction_root = self.root / "setsid_peer_abort"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(
                seconds0=8.0,
                seconds1=8.0,
                descendants=True,
                ignore_term=True,
                setsid=True,
            ),
        )
        markers = [
            self.root / "markers" / f"descendant.rank{rank}"
            for rank in (0, 1)
        ]
        for marker in markers:
            self._wait_for(marker)
        pids = [int(marker.read_text()) for marker in markers]
        from scripts.show_base import dual_node_guarded_transaction as module

        identities = [module._proc_identity(pid) for pid in pids]
        # macOS has no PR_SET_CHILD_SUBREAPER; allow the CPU-test census to
        # observe the already-escaped sessions before triggering peer abort.
        # Formal Linux does not need this delay because orphaned descendants
        # are adopted by the transaction supervisor.
        time.sleep(0.2)
        node1.terminate()
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertNotEqual(self._wait_process(node0), 0)
        deadline = time.monotonic() + 2
        living = pids
        while living and time.monotonic() < deadline:
            living = []
            for pid, identity in zip(pids, identities):
                if self._pid_is_live(pid, identity):
                    living.append(pid)
            time.sleep(0.02)
        details = subprocess.run(
            [
                "/bin/ps",
                "-p",
                ",".join(map(str, pids)),
                "-o",
                "pid=,ppid=,pgid=,sess=,stat=,lstart=,command=",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        self.assertFalse(
            living,
            f"setsid descendants survived abort: {living}; details={details}",
        )
        self.assertNotEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "SUCCEEDED",
        )

    def test_pid_reuse_starttime_change_is_never_signalled(self) -> None:
        from scripts.show_base import dual_node_guarded_transaction as module

        tracker = module._DescendantTracker(100, 200, 300)
        old = {
            "pid": 444,
            "ppid": 200,
            "pgid": 200,
            "sid": 100,
            "starttime_ticks": 10,
            "argv_sha256": "a" * 64,
        }
        reused = {
            **old,
            "ppid": 1,
            "starttime_ticks": 11,
            "argv_sha256": "b" * 64,
        }
        tracker._remember(old)
        with mock.patch.object(module, "_process_table", return_value={444: reused}), mock.patch.object(
            module.os, "kill"
        ) as kill:
            tracker.signal_live(signal.SIGKILL)
        kill.assert_not_called()

    def test_replayed_lower_heartbeat_sequence_aborts_both_nodes(self) -> None:
        transaction_root = self.root / "heartbeat_replay"
        configuration = self._configuration(seconds0=8.0, seconds1=8.0)
        environment = {"SEMTALK_TEST_REPLAY_RANK": "1"}
        node1 = self._start(
            self._command(1, transaction_root, configuration),
            extra_environment=environment,
        )
        node0 = self._start(
            self._command(0, transaction_root, configuration),
            extra_environment=environment,
        )
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        failure = json.loads((transaction_root / "OUTCOME.json").read_text())
        self.assertEqual(failure["status"], "FAILED")
        self.assertIn("sequence", failure["reason"])

    def test_forged_completed_result_with_nonzero_rc_is_rejected(self) -> None:
        transaction_root = self.root / "forged_result"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(seconds0=0.03, seconds1=8.0),
        )
        self._wait_for(transaction_root / "STARTED.rank1.json")
        self._wait_for(transaction_root / "WORKLOAD_RESULT.rank0.json")
        started_raw = (transaction_root / "STARTED.rank1.json").read_bytes()
        started = json.loads(started_raw)
        self._publish_artifact(
            transaction_root / "WORKLOAD_RESULT.rank1.json",
            {
                "schema": started["schema"],
                "status": "COMPLETED",
                "rank": 1,
                "started_sha256": hashlib.sha256(started_raw).hexdigest(),
                "coordinator": started["coordinator"],
                "reason": "forged invalid result",
                "workload_returncode": 9,
                "result_unix_ns": time.time_ns(),
            },
        )
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "FAILED",
        )

    def test_parent_namespace_replacement_is_detected(self) -> None:
        parent = self.root / "trusted_namespace"
        parent.mkdir(mode=0o700)
        transaction_root = parent / "parent_swap"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(seconds0=8.0, seconds1=8.0),
        )
        self._wait_for(transaction_root / "DECISION.json")
        moved = self.root / "moved_namespace"
        parent.rename(moved)
        parent.mkdir(mode=0o700)
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse(transaction_root.exists())

    def test_same_run_id_under_two_namespace_parents_cannot_join(self) -> None:
        parent0 = self.root / "deployment_parent0"
        parent1 = self.root / "deployment_parent1"
        parent0.mkdir(mode=0o700)
        parent1.mkdir(mode=0o700)
        configuration = self._configuration(seconds0=8.0, seconds1=8.0)
        deployment_parent_pin = namespace_parent_sha256(parent0)
        node1 = self._start(
            self._command(
                1,
                parent1 / "same_run",
                configuration,
                expected_parent_sha256=deployment_parent_pin,
            )
        )
        node0 = self._start(
            self._command(
                0,
                parent0 / "same_run",
                configuration,
                expected_parent_sha256=deployment_parent_pin,
            )
        )
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse((parent0 / "same_run" / "OUTCOME.json").exists())
        self.assertFalse((parent1 / "same_run" / "OUTCOME.json").exists())

    def test_run_id_is_bound_to_root_basename(self) -> None:
        transaction_root = self.root / "actual_name"
        result = subprocess.run(
            self._command(
                0,
                transaction_root,
                self._configuration(),
                run_id="different_name",
            ),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("namespace/run_id", result.stderr)

    def test_duplicate_root_never_overwrites_decision(self) -> None:
        transaction_root = self.root / "duplicate"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0)
        self.assertEqual(self._wait_process(node1), 0)
        before = (transaction_root / "DECISION.json").read_bytes()
        duplicate = subprocess.run(
            self._command(0, transaction_root, self._configuration()),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertEqual((transaction_root / "DECISION.json").read_bytes(), before)

    def test_peer_cannot_claim_another_canonical_namespace_parent(self) -> None:
        transaction_root = self.root / "mount_evidence"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0)
        self.assertEqual(self._wait_process(node1), 0)
        prepared = json.loads((transaction_root / "PREPARED.rank1.json").read_text())
        prepared["transaction_root"] = "/different/local/efs/mount/mount_evidence"
        from scripts.show_base import dual_node_guarded_transaction as module

        validator = object.__new__(module.Coordinator)
        validator.portable = prepared["portable"]
        validator.portable_sha256 = prepared["portable_sha256"]
        with self.assertRaises(module.PeerAbort):
            validator._validate_prepared(prepared, 1)

    def test_workload_input_content_and_allowlisted_environment_are_portable(self) -> None:
        authority = self.root / "fresh_test_authority.json"
        authority.write_text(
            json.dumps(
                {
                    "candidate_count": 22,
                    "epochs": 400,
                    "fresh_forward_count": 3,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        transaction_root = self.root / "bound_authority"
        configuration = self._configuration()
        previous = os.environ.get("SEMTALK_TEST_BOUND_ENV")
        os.environ["SEMTALK_TEST_BOUND_ENV"] = "fresh-authority-v1"
        try:
            node0, node1 = self._run_pair(
                transaction_root,
                configuration,
                workload_inputs={"fresh_test_authority": authority},
                allow_env=["SEMTALK_TEST_BOUND_ENV"],
            )
            self.assertEqual(self._wait_process(node0), 0)
            self.assertEqual(self._wait_process(node1), 0)
        finally:
            if previous is None:
                os.environ.pop("SEMTALK_TEST_BOUND_ENV", None)
            else:
                os.environ["SEMTALK_TEST_BOUND_ENV"] = previous
        portable = json.loads((transaction_root / "PREPARED.rank0.json").read_text())["portable"]
        self.assertEqual(
            portable["workload"]["input_sha256"]["fresh_test_authority"],
            hashlib.sha256(authority.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            portable["workload"]["environment"],
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SEMTALK_TEST_BOUND_ENV": "fresh-authority-v1",
            },
        )
        binding = portable["workload"]["input_bindings"][
            "fresh_test_authority"
        ]
        self.assertEqual(binding["option"], "--fresh-test-authority")
        self.assertRegex(binding["exec_path"], r"^/(?:proc/self|dev)/fd/\d+$")
        for rank in (0, 1):
            started = json.loads(
                (transaction_root / f"STARTED.rank{rank}.json").read_text()
            )
            inherited = started["workload"]["passed_input_fds"][
                "fresh_test_authority"
            ]
            self.assertEqual(inherited["passed_fd"], binding["passed_fd"])
            self.assertEqual(inherited["sha256"], binding["sha256"])

    def test_unknown_input_id_is_rejected_before_spawn(self) -> None:
        authority = self.root / "unknown-input.json"
        authority.write_text('{"version":1}\n', encoding="utf-8")
        transaction_root = self.root / "unknown_input_id"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(),
            workload_inputs={"not_allowlisted": authority},
        )
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_input_path_must_occupy_its_schema_owned_argv_slot(self) -> None:
        authority = self.root / "mispositioned-input.json"
        authority.write_text('{"version":1}\n', encoding="utf-8")
        transaction_root = self.root / "mispositioned_input"
        configuration = self._configuration()
        processes = []
        for rank in (1, 0):
            arguments = self._protocol_args(
                rank,
                transaction_root,
                configuration,
                workload_inputs={"fresh_test_authority": authority},
            )
            delimiter = arguments.index("--")
            workload = arguments[delimiter + 1 :]
            option = workload.index("--fresh-test-authority")
            workload.insert(option + 1, "decoy-not-the-authority")
            arguments[delimiter + 1 :] = workload
            digest_index = arguments.index("--common-command-sha256") + 1
            arguments[digest_index] = argv_sha256(workload)
            processes.append(
                self._start([PYTHON, "-c", HARNESS, *arguments])
            )
        for process in processes:
            self.assertNotEqual(self._wait_process(process), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_pinned_workload_input_mutation_before_go_aborts(self) -> None:
        authority = self.root / "mutable_authority.json"
        authority.write_text('{"version":1}\n', encoding="utf-8")
        transaction_root = self.root / "mutated_authority"
        configuration = self._configuration(seconds0=8.0, seconds1=8.0)
        arguments = {
            "workload_inputs": {"fresh_test_authority": authority},
        }
        delay = {"SEMTALK_TEST_ARM_DELAY": "0.12"}
        node1 = self._start(
            self._command(1, transaction_root, configuration, **arguments),
            extra_environment=delay,
        )
        node0 = self._start(
            self._command(0, transaction_root, configuration, **arguments),
            extra_environment=delay,
        )
        self._wait_for(transaction_root / "PREPARED.rank0.json")
        self._wait_for(transaction_root / "PREPARED.rank1.json")
        authority.write_text('{"version":2}\n', encoding="utf-8")
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_atomic_replaced_input_is_consumed_from_pinned_fd_v1(self) -> None:
        authority = self.root / "atomic_authority.json"
        v1 = '{"version":1}\n'
        v2 = '{"version":2}\n'
        authority.write_text(v1, encoding="utf-8")
        transaction_root = self.root / "atomic_authority_tx"
        configuration = self._configuration(
            seconds0=0.1,
            seconds1=0.1,
            read_authority=True,
            expected_authority=v1,
        )
        arguments = {
            "workload_inputs": {"fresh_test_authority": authority},
        }
        delay = {"SEMTALK_TEST_ARM_DELAY": "0.12"}
        node1 = self._start(
            self._command(1, transaction_root, configuration, **arguments),
            extra_environment=delay,
        )
        node0 = self._start(
            self._command(0, transaction_root, configuration, **arguments),
            extra_environment=delay,
        )
        self._wait_for(transaction_root / "PREPARED.rank0.json")
        self._wait_for(transaction_root / "PREPARED.rank1.json")
        replacement = authority.with_name("replacement.json")
        replacement.write_text(v2, encoding="utf-8")
        replacement.replace(authority)
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        self.assertEqual(authority.read_text(encoding="utf-8"), v2)
        for rank in (0, 1):
            observed = self.root / "markers" / f"authority.rank{rank}"
            self.assertEqual(observed.read_text(encoding="utf-8"), v1)
            started = json.loads(
                (transaction_root / f"STARTED.rank{rank}.json").read_text()
            )
            passed = started["workload"]["passed_input_fds"][
                "fresh_test_authority"
            ]
            self.assertEqual(passed["sha256"], hashlib.sha256(v1.encode()).hexdigest())
        self._finalize(transaction_root)
        self.assertEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "SUCCEEDED",
        )

    def test_same_content_input_alias_is_rejected_before_spawn(self) -> None:
        authority = self.root / "authority.json"
        alias = self.root / "authority-copy.json"
        alias_link = self.root / "authority-copy-link.json"
        authority.write_text('{"version":1}\n', encoding="utf-8")
        alias.write_bytes(authority.read_bytes())
        alias_link.symlink_to(alias)
        variants = {
            "plain_json": json.dumps({"authority": str(alias)}),
            "symlink_to_copy": str(alias_link),
            "escaped_json": json.dumps(
                {"authority": str(alias)}
            ).replace("/", "\\/"),
        }
        for name, encoded_alias in variants.items():
            with self.subTest(name=name):
                transaction_root = self.root / f"aliased_authority_{name}"
                node0, node1 = self._run_pair(
                    transaction_root,
                    self._configuration(),
                    workload_inputs={"fresh_test_authority": authority},
                    workload_suffix=[
                        "--authority-alias-json",
                        encoded_alias,
                    ],
                )
                self.assertNotEqual(self._wait_process(node0), 0)
                self.assertNotEqual(self._wait_process(node1), 0)
                self.assertFalse(
                    (self.root / "markers" / "started.rank0").exists()
                )
                self.assertFalse(
                    (self.root / "markers" / "started.rank1").exists()
                )

    def test_symlinked_workload_provenance_is_rejected(self) -> None:
        from scripts.show_base import dual_node_guarded_transaction as module

        link = self.root / "python-link"
        link.symlink_to(PYTHON)
        with self.assertRaises(module.TransactionError):
            module._open_pinned_file(link, executable=True)

    def test_two_hop_formal_venv_python_preserves_exact_runtime(self) -> None:
        venv_root, venv_python = self._formal_venv()
        transaction_root = self.root / "formal_venv_success"
        node0, node1 = self._run_formal_pair(
            transaction_root,
            venv_python,
            self._configuration(record_runtime=True),
        )
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        portable = json.loads(
            (transaction_root / "PREPARED.rank0.json").read_text()
        )["portable"]
        binding = portable["workload"]["formal_python_binding"]
        self.assertEqual(portable["workload"]["executable_path"], str(venv_python))
        self.assertEqual(binding["argv0"], str(venv_python))
        self.assertGreaterEqual(len(binding["symlink_chain"]), 1)
        self.assertEqual(
            binding["resolved_target"]["sha256"],
            portable["workload"]["executable_sha256"],
        )
        for rank in (0, 1):
            runtime = json.loads(
                (self.root / "markers" / f"runtime.rank{rank}.json").read_text()
            )
            self.assertEqual(runtime["sys_executable"], str(venv_python))
            self.assertEqual(runtime["sys_prefix"], str(venv_root))
            self.assertNotEqual(runtime["sys_base_prefix"], str(venv_root))
            self.assertEqual(runtime["python_target_fd_leaks"], [])

    def test_formal_pyvenv_replacement_aborts_before_go(self) -> None:
        venv_root, venv_python = self._formal_venv()
        transaction_root = self.root / "formal_cfg_replaced"
        node0, node1 = self._run_formal_pair(
            transaction_root,
            venv_python,
            self._configuration(record_runtime=True),
            extra_environment={"SEMTALK_TEST_ARM_DELAY": "0.5"},
        )
        self._wait_for(transaction_root / "PREPARED.rank0.json")
        self._wait_for(transaction_root / "PREPARED.rank1.json")
        cfg = venv_root / "pyvenv.cfg"
        replacement = venv_root / "pyvenv.cfg.replacement"
        replacement.write_bytes(cfg.read_bytes())
        replacement.replace(cfg)
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())
        decision = transaction_root / "DECISION.json"
        if decision.exists():
            self.assertNotEqual(json.loads(decision.read_text()).get("status"), "GO")

    def test_formal_leaf_redirect_aborts_before_go(self) -> None:
        _venv_root, venv_python = self._formal_venv()
        transaction_root = self.root / "formal_leaf_redirected"
        node0, node1 = self._run_formal_pair(
            transaction_root,
            venv_python,
            self._configuration(record_runtime=True),
            extra_environment={"SEMTALK_TEST_ARM_DELAY": "0.5"},
        )
        self._wait_for(transaction_root / "PREPARED.rank0.json")
        self._wait_for(transaction_root / "PREPARED.rank1.json")
        replacement = venv_python.with_name("python.replacement")
        replacement.symlink_to("/bin/sh")
        replacement.replace(venv_python)
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_declared_workload_input_symlink_is_rejected_before_spawn(self) -> None:
        authority = self.root / "authority.json"
        authority.write_text('{"version":1}\n', encoding="utf-8")
        alias = self.root / "authority-link.json"
        alias.symlink_to(authority)
        transaction_root = self.root / "symlinked_authority"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(),
            workload_inputs={"fresh_test_authority": alias},
        )
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_formal_leaf_replaced_by_regular_file_is_rejected_before_prepared(self) -> None:
        _venv_root, venv_python = self._formal_venv()
        target = venv_python.resolve(strict=True)
        venv_python.unlink()
        shutil.copy2(target, venv_python)
        transaction_root = self.root / "formal_leaf_regular"
        node0, node1 = self._run_formal_pair(
            transaction_root,
            venv_python,
            self._configuration(record_runtime=True),
        )
        self.assertNotEqual(self._wait_process(node0), 0)
        self.assertNotEqual(self._wait_process(node1), 0)
        self.assertFalse((transaction_root / "PREPARED.rank0.json").exists())
        self.assertFalse((transaction_root / "PREPARED.rank1.json").exists())

    def test_finalizer_file_failure_never_publishes_success_outcome(self) -> None:
        transaction_root = self.root / "finalizer_file_failure"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        self._finish_runner_receipts(transaction_root)
        environment = os.environ.copy()
        environment["SEMTALK_TEST_FINAL_PUBLISH_FAIL"] = "FINAL.rank1.json"
        failed = subprocess.run(
            self._control_command("finalize", transaction_root),
            cwd=REPOSITORY,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertTrue((transaction_root / "FINAL.rank0.json").exists())
        self.assertFalse((transaction_root / "FINAL.rank1.json").exists())
        self.assertFalse((transaction_root / "OUTCOME.json").exists())
        recovered = subprocess.run(
            self._control_command("finalize", transaction_root),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        self.assertEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "SUCCEEDED",
        )

    def test_concurrent_finalizers_publish_one_terminal_outcome(self) -> None:
        transaction_root = self.root / "concurrent_finalizers"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        self._finish_runner_receipts(transaction_root)
        finalizers = [
            subprocess.Popen(
                self._control_command("finalize", transaction_root),
                cwd=REPOSITORY,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        results = []
        for process in finalizers:
            stdout, stderr = process.communicate(timeout=6)
            self.assertEqual(process.returncode, 0, stderr)
            results.append(json.loads(stdout)["status"])
        self.assertEqual(
            sorted(results),
            ["FINALIZED_SUCCEEDED", "REPLAYED_SUCCEEDED"],
        )
        self.assertEqual(
            json.loads((transaction_root / "OUTCOME.json").read_text())["status"],
            "SUCCEEDED",
        )
        self.assertFalse(
            any(
                path.name.startswith(".tmp.OUTCOME.json.")
                for path in transaction_root.iterdir()
            )
        )

    def test_finalizer_rejects_runner_restore_error_before_final_files(self) -> None:
        transaction_root = self.root / "restore_failure"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        self._finish_runner_receipts(transaction_root)
        status = self.status_paths[1]
        payload = json.loads(status.read_text())
        payload["restore_error"] = "injected guard restore failure"
        status.write_text(json.dumps(payload, sort_keys=True) + "\n")
        failed = subprocess.run(
            self._control_command("finalize", transaction_root),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((transaction_root / "FINAL.rank0.json").exists())
        self.assertFalse((transaction_root / "FINAL.rank1.json").exists())
        self.assertFalse((transaction_root / "OUTCOME.json").exists())

    def test_finalizer_rejects_hardlinked_outer_evidence(self) -> None:
        transaction_root = self.root / "hardlinked_outer_evidence"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        self._finish_runner_receipts(transaction_root)
        self.log_paths[1].unlink()
        os.link(self.log_paths[0], self.log_paths[1])
        failed = subprocess.run(
            self._control_command("finalize", transaction_root),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("alias one inode", failed.stderr)
        self.assertFalse((transaction_root / "FINAL.rank0.json").exists())
        self.assertFalse((transaction_root / "FINAL.rank1.json").exists())
        self.assertFalse((transaction_root / "OUTCOME.json").exists())

    def test_replay_is_read_only_and_detects_runner_status_replacement(self) -> None:
        transaction_root = self.root / "read_only_replay"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0, node0.stderr.read())
        self.assertEqual(self._wait_process(node1), 0, node1.stderr.read())
        self._finalize(transaction_root)
        before = {
            path.name: path.read_bytes()
            for path in transaction_root.iterdir()
            if path.is_file()
        }
        replay = self._replay(transaction_root)
        self.assertEqual(replay["status"], "REPLAYED_SUCCEEDED")
        after = {
            path.name: path.read_bytes()
            for path in transaction_root.iterdir()
            if path.is_file()
        }
        self.assertEqual(after, before)

        status = self.status_paths[0]
        replacement = status.with_name("late-status-replacement.json")
        payload = json.loads(status.read_text())
        payload["restore_error"] = "late replacement"
        replacement.write_text(json.dumps(payload, sort_keys=True) + "\n")
        replacement.replace(status)
        outcome_sha = hashlib.sha256(
            (transaction_root / "OUTCOME.json").read_bytes()
        ).hexdigest()
        rejected = subprocess.run(
            self._control_command(
                "replay",
                transaction_root,
                outcome_sha256=outcome_sha,
            ),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(rejected.returncode, 0)

    def test_decision_and_outcome_validators_reject_extra_or_wrong_semantics(self) -> None:
        transaction_root = self.root / "validator_exactness"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(self._wait_process(node0), 0)
        self.assertEqual(self._wait_process(node1), 0)
        self._finalize(transaction_root)
        from scripts.show_base import dual_node_guarded_transaction as module

        prepared = json.loads((transaction_root / "PREPARED.rank0.json").read_text())
        validator = object.__new__(module.Coordinator)
        validator.portable = prepared["portable"]
        validator.portable_sha256 = prepared["portable_sha256"]
        decision = json.loads((transaction_root / "DECISION.json").read_text())
        decision["extra"] = True
        with self.assertRaises(module.TransactionError):
            validator._validate_decision(decision)
        outcome = json.loads((transaction_root / "OUTCOME.json").read_text())
        outcome["bindings"]["final_sha256"]["0"] = "bad"
        with self.assertRaises(module.TransactionError):
            validator._validate_outcome(outcome)

    def test_launcher_syntax_and_nested_delimiter_contract(self) -> None:
        subprocess.run(["/usr/bin/env", "bash", "-n", str(LAUNCHER)], check=True)
        helper = REPOSITORY / "scripts" / "show_base" / "guarded_runner_contract.sh"
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; shift; _semtalk_guarded_runner_argv_is_exact "$@"',
                "nested-contract",
                str(helper),
                PYTHON,
                "/tmp/globaldiff_guarded_runner.py",
                "--gpus",
                "0,1,2,3,4,5,6,7",
                "--",
                str(LAUNCHER),
                PYTHON,
                "--transaction-root",
                "/efs/run",
                "--",
                PYTHON,
                "-c",
                "pass",
                "--",
                "inner",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
