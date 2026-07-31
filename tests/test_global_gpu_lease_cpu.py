from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
UTILITY = REPOSITORY / "scripts" / "show_base" / "global_gpu_lease.py"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class GlobalGpuLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.lease = self.root / "all-gpu.active"
        self.status = self.root / "runner-status.json"
        self.host = socket.gethostname()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _command(self, *arguments: str) -> list[str]:
        # Production has one code-owned /tmp lease path and exposes no path
        # override.  This import-only CPU shim replaces the module constant
        # before calling main, keeping tests isolated without weakening CLI.
        shim = (
            "import sys; from pathlib import Path; "
            "from scripts.show_base import global_gpu_lease as module; "
            "module.GLOBAL_LEASE_DIR = Path(sys.argv[1]); "
            "raise SystemExit(module.main(sys.argv[2:]))"
        )
        return [sys.executable, "-c", shim, str(self.lease), *arguments]

    def _acquire(
        self,
        *,
        host_count: int = 1,
        host_slot: int = 0,
        predecessor: str | None = None,
        mode: str = "base.probe.W16",
        status_path: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object] | None]:
        command = self._command(
            "acquire",
            "--formal-host",
            self.host,
            "--formal-host-count",
            str(host_count),
            "--formal-host-slot",
            str(host_slot),
            "--mode",
            mode,
            "--run-id",
            "formal-run-0001",
            "--runner-status",
            str(self.status if status_path is None else status_path),
        )
        if predecessor is not None:
            command.extend(
                ["--predecessor-lease-receipt-sha256", predecessor]
            )
        result = subprocess.run(
            command,
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else None
        return result, payload

    def _status_payload(self) -> dict[str, object]:
        return {
            "state": "finished",
            "return_code": 0,
            "error": None,
            "cleanup_error": None,
            "restore_error": None,
            "received_signal": None,
            "restored_guards": {
                str(gpu): 9000 + gpu for gpu in range(8)
            },
            "command": ["/bin/bash", "/formal/workload.sh"],
        }

    def _write_status(self, payload: dict[str, object] | None = None) -> str:
        raw = (
            json.dumps(
                self._status_payload() if payload is None else payload,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        self.status.write_bytes(raw)
        return sha256_bytes(raw)

    def _release_command(
        self,
        acquired: dict[str, object],
        status_sha256: str,
        *,
        confirmed_pid_offset: int = 9000,
    ) -> list[str]:
        command = self._command(
            "release",
            "--formal-host",
            self.host,
            "--lease-id",
            str(acquired["lease_id"]),
            "--lease-token",
            str(acquired["lease_token"]),
            "--expected-lease-receipt-sha256",
            str(acquired["lease_receipt_sha256"]),
            "--runner-status",
            str(self.status),
            "--expected-runner-status-sha256",
            status_sha256,
            "--confirm-no-live-descendants",
            "--confirm-restored-guards-are-real-resnet18",
        )
        for gpu in range(8):
            command.extend(
                ["--confirmed-guard", f"{gpu}={confirmed_pid_offset + gpu}"]
            )
        return command

    def _release_namespace(
        self,
        acquired: dict[str, object],
        status_sha256: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            confirm_no_live_descendants=True,
            confirm_restored_guards_are_real_resnet18=True,
            confirmed_guard=[f"{gpu}={9000 + gpu}" for gpu in range(8)],
            formal_host=self.host,
            lease_id=str(acquired["lease_id"]),
            lease_token=str(acquired["lease_token"]),
            expected_lease_receipt_sha256=str(
                acquired["lease_receipt_sha256"]
            ),
            runner_status=str(self.status),
            expected_runner_status_sha256=status_sha256,
        )

    def test_acquire_is_atomic_token_bound_and_cross_mode_duplicate_fails(self) -> None:
        first, acquired = self._acquire()
        self.assertEqual(first.returncode, 0, first.stderr)
        assert acquired is not None
        receipt_raw = (self.lease / "LEASE.json").read_bytes()
        receipt = json.loads(receipt_raw)
        self.assertEqual(
            acquired["lease_receipt_sha256"], sha256_bytes(receipt_raw)
        )
        self.assertNotIn(str(acquired["lease_token"]), receipt_raw.decode())
        self.assertEqual(
            receipt["token_sha256"],
            sha256_bytes(str(acquired["lease_token"]).encode()),
        )
        self.assertTrue(receipt["policy"]["cross_mode"])
        second, _ = self._acquire(mode="canonical-cache.other-mode")
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("never stolen", second.stderr)

    def test_production_cli_exposes_no_lease_path_override(self) -> None:
        source = UTILITY.read_text(encoding="utf-8")
        self.assertIn(
            'GLOBAL_LEASE_DIR = Path("/tmp/semtalk_formal_global_gpu_lease.active")',
            source,
        )
        self.assertNotIn('add_argument("--lease-dir"', source)

    def test_abandoned_or_incomplete_directory_is_never_stolen(self) -> None:
        self.lease.mkdir(mode=0o700)
        result, _ = self._acquire()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.lease.is_dir())
        self.assertEqual(list(self.lease.iterdir()), [])

    def test_acquire_rejects_status_path_equal_to_active_lease(self) -> None:
        result, acquired = self._acquire(status_path=self.lease)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(acquired)
        self.assertIn("cannot be the active lease", result.stderr)
        self.assertFalse(self.lease.exists())

    def test_concurrent_acquirers_have_exactly_one_winner(self) -> None:
        command = self._command(
            "acquire",
            "--formal-host",
            self.host,
            "--formal-host-count",
            "1",
            "--formal-host-slot",
            "0",
            "--mode",
            "concurrent-mode",
            "--run-id",
            "formal-run-concurrent",
            "--runner-status",
            str(self.status),
        )
        processes = [
            subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        results = [process.communicate(timeout=5) for process in processes]
        returncodes = [process.returncode for process in processes]
        self.assertEqual(sorted(returncodes), [0, 1], results)
        self.assertEqual(
            {path.name for path in self.lease.iterdir()}, {"LEASE.json"}
        )

    def test_two_host_slot_one_requires_slot_zero_receipt_binding(self) -> None:
        rejected, _ = self._acquire(host_count=2, host_slot=1)
        self.assertNotEqual(rejected.returncode, 0)
        predecessor = "a" * 64
        accepted, acquired = self._acquire(
            host_count=2,
            host_slot=1,
            predecessor=predecessor,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        assert acquired is not None
        receipt = json.loads((self.lease / "LEASE.json").read_text())
        self.assertEqual(
            receipt["two_host_acquisition_order"],
            "ascending_formal_host_slot",
        )
        self.assertEqual(
            receipt["predecessor_lease_receipt_sha256"], predecessor
        )

    def test_single_host_worker_slot_needs_no_remote_predecessor(self) -> None:
        accepted, acquired = self._acquire(host_count=1, host_slot=1)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        assert acquired is not None
        receipt = json.loads((self.lease / "LEASE.json").read_text())
        self.assertEqual(receipt["formal_host_slot"], 1)
        self.assertIsNone(receipt["predecessor_lease_receipt_sha256"])

    def test_successful_release_archives_complete_immutable_evidence(self) -> None:
        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        status_sha256 = self._write_status()
        release = subprocess.run(
            self._release_command(acquired, status_sha256),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(release.returncode, 0, release.stderr)
        result = json.loads(release.stdout)
        archive = Path(result["archive_dir"])
        self.assertFalse(self.lease.exists())
        self.assertTrue(archive.is_dir())
        self.assertEqual(
            {path.name for path in archive.iterdir()},
            {"LEASE.json", "RUNNER_STATUS.json", "RELEASE.json"},
        )
        self.assertEqual(
            (archive / "RUNNER_STATUS.json").read_bytes(),
            self.status.read_bytes(),
        )
        evidence = json.loads((archive / "RELEASE.json").read_text())
        self.assertTrue(
            evidence["caller_confirmations"]["no_live_workload_descendants"]
        )
        self.assertTrue(
            evidence["caller_confirmations"]
            ["guards_are_real_torchvision_resnet18_globaldiff_gpu_guard_cnn"]
        )
        self.assertEqual(
            set(evidence["runner_status"]["restored_guards"]),
            {str(gpu) for gpu in range(8)},
        )

    def test_release_rejects_failure_or_inexact_guard_restore(self) -> None:
        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        payload = self._status_payload()
        payload["cleanup_error"] = "injected"
        status_sha256 = self._write_status(payload)
        release = subprocess.run(
            self._release_command(acquired, status_sha256),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(release.returncode, 0)
        self.assertTrue(self.lease.is_dir())
        self.assertFalse(Path(acquired["archive_dir"]).exists())

    def test_runner_terminal_status_validator_is_exact(self) -> None:
        from scripts.show_base import global_gpu_lease as module

        confirmed = {str(gpu): 9000 + gpu for gpu in range(8)}
        mutations = []
        for value in (True, 0.0, 1, -1):
            mutations.append((f"return_code={value!r}", {"return_code": value}))
        for key in (
            "error",
            "cleanup_error",
            "restore_error",
            "received_signal",
        ):
            mutations.append((f"missing {key}", {key: "__DELETE__"}))
            mutations.append((f"non-null {key}", {key: "failure"}))
        mutations.extend(
            [
                ("missing GPU", {"guards": {str(gpu): 9000 + gpu for gpu in range(7)}}),
                ("extra GPU", {"guards": {str(gpu): 9000 + gpu for gpu in range(9)}}),
                ("duplicate PID", {"guards": {str(gpu): 9000 for gpu in range(8)}}),
                ("boolean PID", {"guards": {**confirmed, "0": True}}),
            ]
        )
        for label, mutation in mutations:
            with self.subTest(label=label):
                status = self._status_payload()
                if "guards" in mutation:
                    status["restored_guards"] = mutation["guards"]
                else:
                    for key, value in mutation.items():
                        if value == "__DELETE__":
                            del status[key]
                        else:
                            status[key] = value
                with self.assertRaises(module.LeaseError):
                    module._validate_runner_status(status, confirmed)

    def test_release_requires_both_explicit_caller_confirmations(self) -> None:
        from scripts.show_base import global_gpu_lease as module

        with self.assertRaisesRegex(module.LeaseError, "descendant"):
            module._release(
                SimpleNamespace(confirm_no_live_descendants=False)
            )
        with self.assertRaisesRegex(module.LeaseError, "real torchvision"):
            module._release(
                SimpleNamespace(
                    confirm_no_live_descendants=True,
                    confirm_restored_guards_are_real_resnet18=False,
                )
            )

    def test_release_rejects_duplicate_json_keys_in_runner_evidence(self) -> None:
        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        guards = ",".join(
            ["\"0\":9000", "\"0\":9000"]
            + [f'"{gpu}":{9000 + gpu}' for gpu in range(1, 8)]
        )
        raw = (
            "{\"state\":\"finished\",\"return_code\":0,"
            "\"error\":null,\"cleanup_error\":null,"
            "\"restore_error\":null,\"received_signal\":null,"
            f'"restored_guards":{{{guards}}}}}\n'
        ).encode()
        self.status.write_bytes(raw)
        release = subprocess.run(
            self._release_command(acquired, sha256_bytes(raw)),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(release.returncode, 0)
        self.assertIn("duplicate JSON key", release.stderr)
        self.assertEqual(
            {path.name for path in self.lease.iterdir()}, {"LEASE.json"}
        )

    def test_final_revalidation_rejects_lease_mutation_after_snapshot(self) -> None:
        from scripts.show_base import global_gpu_lease as module

        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        status_sha256 = self._write_status()
        original = module._validate_runner_status

        def mutate_after_status(*arguments, **keywords):
            result = original(*arguments, **keywords)
            (self.lease / "LEASE.json").write_text("{}\n")
            return result

        module.GLOBAL_LEASE_DIR = self.lease
        with mock.patch.object(
            module,
            "_validate_runner_status",
            side_effect=mutate_after_status,
        ):
            with self.assertRaisesRegex(module.LeaseError, "evidence changed"):
                module._release(
                    self._release_namespace(acquired, status_sha256)
                )
        self.assertFalse(Path(acquired["archive_dir"]).exists())

    def test_pinned_directory_rejects_path_replacement_before_archive(self) -> None:
        from scripts.show_base import global_gpu_lease as module

        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        status_sha256 = self._write_status()
        displaced = self.root / "displaced-active-lease"
        original = module._validate_runner_status

        def replace_path_after_status(*arguments, **keywords):
            result = original(*arguments, **keywords)
            self.lease.rename(displaced)
            self.lease.symlink_to(displaced, target_is_directory=True)
            return result

        module.GLOBAL_LEASE_DIR = self.lease
        with mock.patch.object(
            module,
            "_validate_runner_status",
            side_effect=replace_path_after_status,
        ):
            with self.assertRaisesRegex(module.LeaseError, "identity changed"):
                module._release(
                    self._release_namespace(acquired, status_sha256)
                )
        self.assertTrue(self.lease.is_symlink())
        self.assertTrue(displaced.is_dir())
        self.assertFalse(Path(acquired["archive_dir"]).exists())

    def test_release_rejects_caller_guard_disagreement(self) -> None:
        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        status_sha256 = self._write_status()
        release = subprocess.run(
            self._release_command(
                acquired,
                status_sha256,
                confirmed_pid_offset=9100,
            ),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(release.returncode, 0)
        self.assertIn("caller-confirmed", release.stderr)
        self.assertTrue(self.lease.is_dir())

    def test_release_refuses_existing_archive_target_without_overwrite(self) -> None:
        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        archive = Path(acquired["archive_dir"])
        archive.mkdir()
        sentinel = archive / "sentinel"
        sentinel.write_text("preserve\n")
        status_sha256 = self._write_status()
        release = subprocess.run(
            self._release_command(acquired, status_sha256),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(release.returncode, 0)
        self.assertEqual(sentinel.read_text(), "preserve\n")
        self.assertTrue(self.lease.is_dir())

    def test_status_symlink_is_rejected(self) -> None:
        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        real = self.root / "real-status.json"
        raw = (json.dumps(self._status_payload()) + "\n").encode()
        real.write_bytes(raw)
        self.status.symlink_to(real)
        release = subprocess.run(
            self._release_command(acquired, sha256_bytes(raw)),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(release.returncode, 0)
        self.assertTrue(self.lease.is_dir())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO is unavailable")
    def test_status_fifo_is_rejected_without_blocking(self) -> None:
        acquired_result, acquired = self._acquire()
        self.assertEqual(acquired_result.returncode, 0, acquired_result.stderr)
        assert acquired is not None
        os.mkfifo(self.status)
        release = subprocess.run(
            self._release_command(acquired, "a" * 64),
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        self.assertNotEqual(release.returncode, 0)
        self.assertIn("safe bounded regular file", release.stderr)
        self.assertTrue(self.lease.is_dir())


if __name__ == "__main__":
    unittest.main()
