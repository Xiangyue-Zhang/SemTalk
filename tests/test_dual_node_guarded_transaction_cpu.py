from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY / "scripts" / "show_base" / "dual_node_guarded_transaction.py"
LAUNCHER = REPOSITORY / "scripts" / "show_base" / "run_dual_node_guarded_transaction.sh"


class DualNodeGuardedTransactionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="semtalk-w16-2pc-")
        self.root = Path(self.temporary.name).resolve()
        self.status_paths = []
        self.log_paths = []
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
                process.send_signal(signal.SIGCONT)
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        self.temporary.cleanup()

    def _command(
        self,
        rank: int,
        transaction_root: Path,
        *,
        common_sha256: str = "c" * 64,
        workload_seconds: float = 0.2,
        workload_rc: int = 0,
    ) -> list[str]:
        peer = 1 - rank
        workload = (
            "import sys,time;"
            f"time.sleep({workload_seconds!r});"
            f"sys.exit({workload_rc})"
        )
        return [
            sys.executable,
            str(HELPER),
            "--transaction-root",
            str(transaction_root),
            "--run-id",
            "cpu_test_transaction",
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
            "--max-restarts",
            "0",
            "--heartbeat-ms",
            "40",
            "--stale-ms",
            "240",
            "--prepare-timeout-ms",
            "1800",
            "--commit-timeout-ms",
            "1800",
            "--completion-timeout-ms",
            "1800",
            "--shutdown-grace-ms",
            "250",
            "--",
            sys.executable,
            "-c",
            workload,
        ]

    def _start(self, command: list[str]) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(process)
        return process

    def _wait_for(self, path: Path, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not path.exists():
            if time.monotonic() >= deadline:
                self.fail(f"timed out waiting for {path}")
            time.sleep(0.01)

    def _run_pair(
        self,
        transaction_root: Path,
        *,
        sha0: str = "c" * 64,
        sha1: str = "c" * 64,
        seconds0: float = 0.2,
        seconds1: float = 0.2,
    ) -> tuple[subprocess.Popen[str], subprocess.Popen[str]]:
        # Starting rank one first exercises its bounded bootstrap wait.
        node1 = self._start(
            self._command(1, transaction_root, common_sha256=sha1, workload_seconds=seconds1)
        )
        node0 = self._start(
            self._command(0, transaction_root, common_sha256=sha0, workload_seconds=seconds0)
        )
        return node0, node1

    def test_success_publishes_exact_receipts_and_commit(self) -> None:
        transaction_root = self.root / "success_tx"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(node0.wait(timeout=5), 0, node0.stderr.read())
        self.assertEqual(node1.wait(timeout=5), 0, node1.stderr.read())
        prepared = [
            json.loads((transaction_root / f"PREPARED.rank{rank}.json").read_text())
            for rank in (0, 1)
        ]
        self.assertEqual(prepared[0]["portable"], prepared[1]["portable"])
        self.assertEqual(prepared[0]["portable"]["max_restarts"], 0)
        self.assertEqual(
            set(prepared[0]["node_local_filesystem"]),
            {"st_dev", "st_ino"},
        )
        commit = json.loads((transaction_root / "COMMIT.json").read_text())
        self.assertEqual(commit["status"], "COMMITTED")
        for rank in (0, 1):
            terminal = json.loads(
                (transaction_root / f"TERMINAL.rank{rank}.json").read_text()
            )
            self.assertEqual(terminal["status"], "COMPLETED")

    def test_portable_mismatch_aborts_both_nodes(self) -> None:
        transaction_root = self.root / "mismatch_tx"
        node0, node1 = self._run_pair(
            transaction_root,
            sha0="c" * 64,
            sha1="d" * 64,
        )
        self.assertNotEqual(node0.wait(timeout=5), 0)
        self.assertNotEqual(node1.wait(timeout=5), 0)
        abort = json.loads((transaction_root / "ABORT.json").read_text())
        self.assertEqual(abort["status"], "ABORT")
        self.assertFalse((transaction_root / "COMMIT.json").exists())

    def test_stale_peer_terminates_local_workload_boundedly(self) -> None:
        transaction_root = self.root / "stale_tx"
        node0, node1 = self._run_pair(
            transaction_root,
            seconds0=0.05,
            seconds1=8.0,
        )
        self._wait_for(transaction_root / "COMMIT.json")
        # Exercise staleness after rank zero's local workload has already
        # succeeded.  Its immutable COMPLETED receipt cannot be overwritten,
        # but the shared ABORT must still be published.
        self._wait_for(transaction_root / "TERMINAL.rank0.json")
        os.kill(node1.pid, signal.SIGSTOP)
        started = time.monotonic()
        self.assertNotEqual(node0.wait(timeout=3), 0)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(
            json.loads((transaction_root / "ABORT.json").read_text())["status"],
            "ABORT",
        )
        os.kill(node1.pid, signal.SIGCONT)
        node1.terminate()
        self.assertNotEqual(node1.wait(timeout=3), 0)

    def test_peer_abort_terminates_other_node(self) -> None:
        transaction_root = self.root / "abort_tx"
        node0, node1 = self._run_pair(
            transaction_root,
            seconds0=8.0,
            seconds1=8.0,
        )
        self._wait_for(transaction_root / "COMMIT.json")
        node1.terminate()
        self.assertNotEqual(node1.wait(timeout=3), 0)
        self.assertNotEqual(node0.wait(timeout=3), 0)
        abort = json.loads((transaction_root / "ABORT.json").read_text())
        self.assertEqual(abort["status"], "ABORT")

    def test_duplicate_invocation_fails_without_overwrite(self) -> None:
        transaction_root = self.root / "duplicate_tx"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(node0.wait(timeout=5), 0)
        self.assertEqual(node1.wait(timeout=5), 0)
        commit_before = (transaction_root / "COMMIT.json").read_bytes()
        duplicate = subprocess.run(
            self._command(0, transaction_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("already exists", duplicate.stderr)
        self.assertEqual((transaction_root / "COMMIT.json").read_bytes(), commit_before)

    def test_symlink_and_root_replacement_toctou_fail_closed(self) -> None:
        target = self.root / "symlink_target"
        target.mkdir()
        symlink_root = self.root / "symlink_tx"
        symlink_root.symlink_to(target, target_is_directory=True)
        rejected = subprocess.run(
            self._command(0, symlink_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        self.assertNotEqual(rejected.returncode, 0)

        transaction_root = self.root / "toctou_tx"
        node0, node1 = self._run_pair(
            transaction_root,
            seconds0=8.0,
            seconds1=8.0,
        )
        self._wait_for(transaction_root / "COMMIT.json")
        moved = self.root / "toctou_tx_moved"
        transaction_root.rename(moved)
        transaction_root.mkdir(mode=0o700)
        self.assertNotEqual(node0.wait(timeout=3), 0)
        self.assertNotEqual(node1.wait(timeout=3), 0)
        self.assertTrue((moved / "COMMIT.json").is_file())

    def test_launcher_shell_syntax_and_guard_contract(self) -> None:
        subprocess.run(["/usr/bin/env", "bash", "-n", str(LAUNCHER)], check=True)
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("semtalk_require_exact_guarded_runner_all_gpus", source)
        self.assertIn('exec "$python_bin" "$helper" "$@"', source)


if __name__ == "__main__":
    unittest.main()
