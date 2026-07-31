from __future__ import annotations

import hashlib
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
PYTHON = str(Path(sys.executable).resolve())

HARNESS = r"""
import os
import sys
from scripts.show_base import dual_node_guarded_transaction as module

module._CPU_TEST_MODE = True

def fake_runner(status_path, log_path):
    identity = module._proc_identity(os.getppid())
    identity.update({
        "path": module.EXPECTED_RUNNER,
        "sha256": "e" * 64,
        "status_path": status_path,
        "log_path": log_path,
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
    ])
    deadline = time.monotonic() + 1.0
    while not descendant_path.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
time.sleep(float(entry.get("seconds", 0.05)))
sys.exit(int(entry.get("rc", 0)))
"""


def argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(
        b"\0".join(os.fsencode(token) for token in argv) + b"\0"
    ).hexdigest()


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
    ) -> dict[str, object]:
        return {
            "marker_root": str(self.root / "markers"),
            "0": {
                "seconds": seconds0,
                "rc": rc0,
                "spawn_descendant": descendants,
                "ignore_term": ignore_term,
            },
            "1": {
                "seconds": seconds1,
                "rc": rc1,
                "spawn_descendant": descendants,
                "ignore_term": ignore_term,
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
    ) -> list[str]:
        peer = 1 - rank
        config_text = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        workload = [PYTHON, "-c", WORKLOAD, config_text]
        if workload_suffix:
            workload.extend(workload_suffix)
        common_sha256 = common_sha256 or argv_sha256(workload)
        arguments = [
            "--transaction-root",
            str(transaction_root),
            "--run-id",
            run_id or transaction_root.name,
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
            "35",
            "--stale-ms",
            "220",
            "--prepare-timeout-ms",
            "1800",
            "--decision-timeout-ms",
            "1800",
            "--arm-timeout-ms",
            "1800",
            "--start-timeout-ms",
            "1800",
            "--completion-timeout-ms",
            "2600",
            "--shutdown-grace-ms",
            "180",
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

    def _wait_for(self, path: Path, timeout: float = 4.0) -> None:
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

    def test_success_uses_armed_go_result_and_final_receipts(self) -> None:
        transaction_root = self.root / "success_tx"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(node0.wait(timeout=6), 0, node0.stderr.read())
        self.assertEqual(node1.wait(timeout=6), 0, node1.stderr.read())
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
            self.assertEqual((final["status"], final["coordinator_returncode"]), ("SUCCEEDED", 0))

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
        self.assertNotEqual(node1.wait(timeout=4), 0)
        self.assertNotEqual(node0.wait(timeout=4), 0)
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
        self.assertNotEqual(node0.wait(timeout=5), 0)
        self.assertNotEqual(node1.wait(timeout=5), 0)
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

    def test_nonzero_result_has_exact_status_and_final_rc(self) -> None:
        transaction_root = self.root / "failed_workload"
        node0, node1 = self._run_pair(
            transaction_root,
            self._configuration(rc0=7, seconds1=0.8),
        )
        self.assertEqual(node0.wait(timeout=6), 4, node0.stderr.read())
        self.assertNotEqual(node1.wait(timeout=6), 0)
        result = json.loads((transaction_root / "WORKLOAD_RESULT.rank0.json").read_text())
        final = json.loads((transaction_root / "FINAL.rank0.json").read_text())
        self.assertEqual((result["status"], result["workload_returncode"]), ("WORKLOAD_FAILED", 7))
        self.assertEqual(
            (final["status"], final["coordinator_returncode"]),
            ("WORKLOAD_FAILED", 4),
        )
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
        self.assertNotEqual(node1.wait(timeout=5), 0)
        self.assertNotEqual(node0.wait(timeout=5), 0)
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
        self.assertNotEqual(node0.wait(timeout=5), 0)
        result = json.loads((transaction_root / "WORKLOAD_RESULT.rank0.json").read_text())
        final = json.loads((transaction_root / "FINAL.rank0.json").read_text())
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(final["status"], "ABORTED")
        self.assertEqual(final["coordinator_returncode"], 3)
        os.kill(node1.pid, signal.SIGCONT)
        node1.terminate()
        node1.wait(timeout=5)

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
        self.assertEqual(node0.wait(timeout=7), 0, node0.stderr.read())
        self.assertEqual(node1.wait(timeout=7), 0, node1.stderr.read())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            living = []
            for pid in pids:
                try:
                    os.kill(pid, 0)
                    living.append(pid)
                except ProcessLookupError:
                    pass
            if not living:
                break
            time.sleep(0.02)
        self.assertFalse(living, f"escaped descendants: {living}")

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
        self.assertNotEqual(node0.wait(timeout=6), 0)
        self.assertNotEqual(node1.wait(timeout=6), 0)
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
        self.assertNotEqual(node0.wait(timeout=5), 0)
        self.assertNotEqual(node1.wait(timeout=5), 0)
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
        self.assertNotEqual(node0.wait(timeout=5), 0)
        self.assertNotEqual(node1.wait(timeout=5), 0)
        self.assertFalse(transaction_root.exists())

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
        self.assertEqual(node0.wait(timeout=6), 0)
        self.assertEqual(node1.wait(timeout=6), 0)
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

    def test_peer_mount_path_is_node_local_evidence(self) -> None:
        transaction_root = self.root / "mount_evidence"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(node0.wait(timeout=6), 0)
        self.assertEqual(node1.wait(timeout=6), 0)
        prepared = json.loads((transaction_root / "PREPARED.rank1.json").read_text())
        prepared["transaction_root"] = "/different/local/efs/mount/mount_evidence"
        prepared["node_local_filesystem"] = {"st_dev": 987654321, "st_ino": 123456789}
        from scripts.show_base import dual_node_guarded_transaction as module

        validator = object.__new__(module.Coordinator)
        validator.portable = prepared["portable"]
        validator.portable_sha256 = prepared["portable_sha256"]
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
            self.assertEqual(node0.wait(timeout=6), 0)
            self.assertEqual(node1.wait(timeout=6), 0)
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

    def test_pinned_workload_input_mutation_before_go_aborts(self) -> None:
        authority = self.root / "mutable_authority.json"
        authority.write_text('{"version":1}\n', encoding="utf-8")
        transaction_root = self.root / "mutated_authority"
        configuration = self._configuration(seconds0=8.0, seconds1=8.0)
        arguments = {
            "workload_inputs": {"fresh_test_authority": authority},
        }
        delay = {"SEMTALK_TEST_ARM_DELAY": "0.35"}
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
        self.assertNotEqual(node0.wait(timeout=5), 0)
        self.assertNotEqual(node1.wait(timeout=5), 0)
        self.assertFalse((self.root / "markers" / "started.rank0").exists())
        self.assertFalse((self.root / "markers" / "started.rank1").exists())

    def test_symlinked_workload_provenance_is_rejected(self) -> None:
        from scripts.show_base import dual_node_guarded_transaction as module

        link = self.root / "python-link"
        link.symlink_to(PYTHON)
        with self.assertRaises(module.TransactionError):
            module._open_pinned_file(link, executable=True)

    def test_decision_and_final_validators_reject_extra_or_wrong_semantics(self) -> None:
        transaction_root = self.root / "validator_exactness"
        node0, node1 = self._run_pair(transaction_root)
        self.assertEqual(node0.wait(timeout=6), 0)
        self.assertEqual(node1.wait(timeout=6), 0)
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
        outcome["bindings"]["workload_result_sha256"]["0"] = "bad"
        with self.assertRaises(module.TransactionError):
            validator._validate_outcome(outcome)

        root, parent, basename = module._validate_tx_path(
            str(transaction_root), transaction_root.name
        )
        tx = module.TransactionDirectory(root, parent, basename, 1)
        try:
            tx.create_or_wait(time.monotonic() + 1)
            validator.tx = tx
            final = json.loads((transaction_root / "FINAL.rank0.json").read_text())
            final["coordinator_returncode"] = 9
            with self.assertRaises(module.PeerAbort):
                validator._validate_final(final, 0)
        finally:
            tx.close()

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
