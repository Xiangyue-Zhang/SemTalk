from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_live_val_remote_endpoint as endpoint
from scripts.show_base import supervise_base_v14_live_validation as supervisor


REAL_SUBPROCESS_RUN = subprocess.run


def artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def write_bytes(path: Path, raw: bytes, *, executable: bool = False) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if executable:
        path.chmod(0o755)
    return artifact(path)


def write_json(path: Path, value: dict[str, object]) -> dict[str, object]:
    return write_bytes(path, endpoint.canonical_json_bytes(value))


class RemoteEndpointFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.hostname = "worker-0.example"
        self.epoch = 4
        self.guards = {str(index): 9000 + index for index in range(8)}
        self.status_mode = "ok"
        self.guard_mode = "ok"
        self.log_tamper_during_verify = False
        self.inner_calls = 0

        self.machine_id = self.root / "machine-id"
        self.machine_artifact = write_bytes(
            self.machine_id, b"0123456789abcdef0123456789abcdef\n"
        )
        self.worker_host_identity = self.root / "worker-product-uuid"
        self.worker_host_identity_artifact = write_bytes(
            self.worker_host_identity,
            b"11111111-2222-3333-4444-555555555555\n",
        )
        self.master_host_identity = self.root / "master-product-uuid"
        self.master_host_identity_artifact = write_bytes(
            self.master_host_identity,
            b"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n",
        )
        # Both machines expose the UUID through the same canonical sysfs path.
        # In this single-host fixture that path is the worker file patched into
        # HOST_IDENTITY_PATH, while the master descriptor retains its distinct
        # bytes and digest as if read on the master host.
        self.master_host_identity_artifact["path"] = str(
            self.worker_host_identity
        )
        self.runner = self.root / "globaldiff_guarded_runner.py"
        self.runner_artifact = write_bytes(
            self.runner, b"#!/usr/bin/env python3\n", executable=True
        )
        self.verifier = self.root / "verify_globaldiff_guards.py"
        self.verifier_artifact = write_bytes(
            self.verifier, b"#!/usr/bin/env python3\n"
        )

        self.venv = self.root / "formal-env"
        (self.venv / "bin").mkdir(parents=True)
        self.python_target = self.venv / "python-final"
        write_bytes(
            self.python_target,
            b"#!/bin/sh\nexit 0\n",
            executable=True,
        )
        self.python = self.venv / "bin" / "python"
        self.python.symlink_to("../python-final")
        self.pyvenv_cfg = self.venv / "pyvenv.cfg"
        write_bytes(
            self.pyvenv_cfg,
            (
                f"home = {self.venv}\n"
                "include-system-site-packages = false\n"
                "version = 3.11.0\n"
                f"executable = {self.python_target}\n"
            ).encode("utf-8"),
        )
        self.formal = {
            "format": "semtalk.formal_venv_python_binding.v1",
            "argv0": str(self.python),
            "venv_root": str(self.venv),
            "symlink_chain": [
                {"path": str(self.python), "target": "../python-final"}
            ],
            "resolved_target": artifact(self.python_target),
            "pyvenv_cfg": artifact(self.pyvenv_cfg),
        }

        self.source = self.root / "source"
        launcher = self.source / "scripts/show_base/run_base_live_val_8shard.sh"
        write_bytes(launcher, b"#!/usr/bin/env bash\nexit 0\n", executable=True)
        self.endpoint_copy = (
            self.source / "scripts/show_base/base_live_val_remote_endpoint.py"
        )
        write_bytes(
            self.endpoint_copy,
            Path(endpoint.__file__).resolve(strict=True).read_bytes(),
        )
        self.supervisor_copy = (
            self.source
            / "scripts/show_base/supervise_base_v14_live_validation.py"
        )
        write_bytes(
            self.supervisor_copy,
            Path(supervisor.__file__).resolve(strict=True).read_bytes(),
        )
        self.dispatcher_copy = (
            self.source
            / "scripts/show_base/dispatch_base_v14_live_validation_dual_host.py"
        )
        write_bytes(self.dispatcher_copy, b"#!/usr/bin/env python3\n")
        self._git("init", "-q")
        self._git("config", "user.name", "Xiangyue-Zhang")
        self._git(
            "config",
            "user.email",
            "85532891+Xiangyue-Zhang@users.noreply.github.com",
        )
        self._git("remote", "add", "origin", endpoint.OFFICIAL_ORIGIN)
        self._git("add", "scripts/show_base")
        self._git("commit", "-q", "-m", "fixture")
        branch = self._git_stdout("symbolic-ref", "--short", "HEAD")
        self.commit = self._git_stdout("rev-parse", "HEAD")
        self.tree = self._git_stdout("rev-parse", "HEAD^{tree}")
        self._git("checkout", "-q", "--detach", self.commit)
        self._git("branch", "-D", branch)

        self.paspa = self.root / "PASPA"
        self.diffsheg = self.root / "DiffSHEG"
        self.paspa.mkdir()
        self.diffsheg.mkdir()
        self.state = self.root / "state"
        self.state.mkdir()
        self.dual_root = self.state / "dual_dispatch"
        self.wave_dir = self.dual_root / "waves" / "wave-0001"
        self.wave_dir.mkdir(parents=True)
        (self.state / "job_claims").mkdir()
        (self.state / "authorizations").mkdir()
        self.status_path = self.state / "runner-status.json"
        self.log_path = self.state / "runner.log"
        self.receipt_path = self.wave_dir / "worker-endpoint-receipt.json"
        self.run_root = self.root / "runs" / "e4"
        self.run_root.parent.mkdir()

        self.work_authority = write_json(
            self.state / "work-authority.json",
            {"candidate_epoch": self.epoch, "status": "authorized"},
        )
        self.candidate_receipt = write_json(
            self.state / "candidate-receipt.json",
            {"candidate_epoch": self.epoch, "status": "ready"},
        )
        self.adopted_e1 = write_json(
            self.state / "adopted-e1.json",
            {"candidate_epoch": 1, "status": "complete"},
        )
        self.adapter_argv = ["/bin/true", "authorize", "4"]
        adapter_argv_sha = hashlib.sha256(
            b"\0".join(os.fsencode(token) for token in self.adapter_argv) + b"\0"
        ).hexdigest()
        launcher_artifact = artifact(launcher)
        control_source = {
            "root": str(self.source),
            "origin": endpoint.OFFICIAL_ORIGIN,
            "commit": self.commit,
            "tree": self.tree,
            "supervisor": artifact(self.supervisor_copy),
            "launcher": copy.deepcopy(launcher_artifact),
            "bridge": copy.deepcopy(launcher_artifact),
            "authority_adapter": copy.deepcopy(launcher_artifact),
        }
        jobs = []
        for candidate_epoch in supervisor.CANDIDATE_EPOCHS:
            selected = candidate_epoch == self.epoch
            candidate_run = (
                self.run_root
                if selected
                else self.root / "runs" / f"e{candidate_epoch}"
            )
            jobs.append(
                {
                    "epoch": candidate_epoch,
                    "candidate_receipt_path": (
                        str(self.candidate_receipt["path"])
                        if selected
                        else str(self.state / f"candidate-{candidate_epoch}.json")
                    ),
                    "authority_path": (
                        str(self.work_authority["path"])
                        if selected
                        else str(self.state / f"authority-{candidate_epoch}.json")
                    ),
                    "authorization_path": (
                        str(self.state / "authorization.json")
                        if selected
                        else str(self.state / f"authorization-{candidate_epoch}.json")
                    ),
                    "run_root": str(candidate_run),
                    "measurement_path": str(
                        candidate_run
                        / "candidates"
                        / f"e{candidate_epoch}"
                        / "live-measurement.json"
                    ),
                    "completion_path": str(
                        self.state / f"completion-{candidate_epoch}.json"
                    ),
                    "runner_status_path": (
                        str(self.status_path)
                        if selected
                        else str(self.state / f"status-{candidate_epoch}.json")
                    ),
                    "runner_log_path": (
                        str(self.log_path)
                        if selected
                        else str(self.state / f"log-{candidate_epoch}.log")
                    ),
                }
            )
        campaign_value = endpoint._add_self_hash({
            "format": supervisor.ADOPTION_CAMPAIGN_FORMAT,
            "status": "frozen_before_execution",
            "split": "val",
            "test_visible": False,
            "test_measurements_authorized": 0,
            "candidate_epochs": list(supervisor.CANDIDATE_EPOCHS),
            "state_root": str(self.state),
            "campaign_claim_path": str(self.state / "campaign.claim.json"),
            "summary_path": str(self.state / "summary.json"),
            "control_source": control_source,
            "runtime_validation_source": {"format": "fixture"},
            "authority_adapter_config": {"format": "fixture"},
            "final_manifest_path": str(self.state / "final-manifest.jsonl"),
            "final_status_path": str(self.state / "final-status.json"),
            "paspa_root": str(self.paspa),
            "diffsheg_root": str(self.diffsheg),
            "seed": 2026,
            "diffsheg_batch_size": 64,
            "formal_python": copy.deepcopy(self.formal),
            "formal_python_runtime_contract": copy.deepcopy(launcher_artifact),
            "guarded_runner": copy.deepcopy(self.runner_artifact),
            "guard_verifier": copy.deepcopy(self.verifier_artifact),
            "reconciliation_path": str(self.state / "reconciliation.json"),
            "reconcile_selection_root": str(self.state / "selection"),
            "jobs": jobs,
            "adopted_e1": copy.deepcopy(self.adopted_e1),
        }, "campaign_payload_sha256")
        self.campaign = write_json(self.state / "campaign.json", campaign_value)
        endpoint_artifact = artifact(self.endpoint_copy)
        master_machine = {
            "path": str(self.machine_id),
            "sha256": "f" * 64,
            "bytes": int(self.machine_artifact["bytes"]),
        }
        role_common = {
            "source_root": str(self.source),
            "source_origin": endpoint.OFFICIAL_ORIGIN,
            "source_commit": self.commit,
            "source_tree": self.tree,
            "formal_python": copy.deepcopy(self.formal),
            "guarded_runner": copy.deepcopy(self.runner_artifact),
            "guard_verifier": copy.deepcopy(self.verifier_artifact),
        }
        topology_value = endpoint._add_self_hash({
            "format": endpoint.TOPOLOGY_FORMAT,
            "status": "frozen_before_execution",
            "campaign": copy.deepcopy(self.campaign),
            "master": {
                "role": "master",
                "hostname": "master-0.example",
                "machine_id": master_machine,
                "host_identity": copy.deepcopy(
                    self.master_host_identity_artifact
                ),
                **copy.deepcopy(role_common),
            },
            "worker": {
                "role": "worker",
                "hostname": self.hostname,
                "machine_id": copy.deepcopy(self.machine_artifact),
                "host_identity": copy.deepcopy(
                    self.worker_host_identity_artifact
                ),
                "ssh_host": self.hostname,
                **copy.deepcopy(role_common),
            },
            "ssh_binary": artifact(Path("/usr/bin/ssh")),
            "ssh_options": [
                "-T", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
                "-o", "ExitOnForwardFailure=yes", "-o", "LogLevel=ERROR",
                "-o", "RequestTTY=no", "--",
            ],
            "endpoint": endpoint_artifact,
            "dispatcher": artifact(self.dispatcher_copy),
            "created_unix": 1785729000.0,
        }, "topology_payload_sha256")
        self.topology = write_json(
            self.dual_root / "topology.json", topology_value
        )
        waves = endpoint._fresh_waves()
        dispatcher_claim_value = endpoint._add_self_hash({
            "format": endpoint.DISPATCHER_CLAIM_FORMAT,
            "status": "claimed",
            "campaign": copy.deepcopy(self.campaign),
            "topology": copy.deepcopy(self.topology),
            "fresh_candidate_epochs": list(supervisor.CANDIDATE_EPOCHS[1:]),
            "waves": waves,
            "policy": {
                "adopted_candidates": 1,
                "fresh_candidates": 21,
                "paired_fresh_candidates": 20,
                "terminal_singleton_candidates": 1,
                "max_active_waves": 1,
                "retry_authorized": False,
                "rerun_authorized": False,
                "publish_completions_after_full_wave_validation": True,
            },
            "created_unix": 1785729010.0,
        }, "claim_payload_sha256")
        self.dispatcher_claim = write_json(
            self.dual_root / "claim.json", dispatcher_claim_value
        )
        supervisor_active_value = endpoint._add_self_hash({
            "format": supervisor.ACTIVE_CLAIM_FORMAT,
            "status": "active",
            "operation": "dual-host-wave-0001",
            "campaign": copy.deepcopy(self.campaign),
            "created_unix": 1785729020.0,
        }, "claim_payload_sha256")
        self.supervisor_active = write_json(
            self.state / "active_invocation.claim.json", supervisor_active_value
        )
        wave_value = endpoint._add_self_hash({
            "format": endpoint.WAVE_FORMAT,
            "status": "active",
            "campaign": copy.deepcopy(self.campaign),
            "dispatcher_claim": copy.deepcopy(self.dispatcher_claim),
            "topology": copy.deepcopy(self.topology),
            "wave_index": 1,
            "candidate_epochs": waves[0],
            "master_epoch": waves[0][0],
            "worker_epoch": self.epoch,
            "supervisor_active_claim": copy.deepcopy(self.supervisor_active),
            "created_unix": 1785729030.0,
        }, "wave_payload_sha256")
        self.wave = write_json(self.wave_dir / "wave.claim.json", wave_value)
        active_value = endpoint._add_self_hash({
            "format": endpoint.ACTIVE_LOCK_FORMAT,
            "status": "active",
            "campaign": copy.deepcopy(self.campaign),
            "dispatcher_claim": copy.deepcopy(self.dispatcher_claim),
            "topology": copy.deepcopy(self.topology),
            "wave": copy.deepcopy(self.wave),
            "created_unix": 1785729040.0,
        }, "lock_payload_sha256")
        self.active = write_json(
            self.dual_root / "active-wave.lock.json", active_value
        )
        job_value = endpoint._add_self_hash({
            "format": supervisor.JOB_CLAIM_FORMAT,
            "status": "claimed",
            "candidate_epoch": self.epoch,
            "campaign": copy.deepcopy(self.campaign),
            "candidate_receipt": copy.deepcopy(self.candidate_receipt),
            "authority_path": str(self.work_authority["path"]),
            "run_root": str(self.run_root),
            "measurement_path": jobs[2]["measurement_path"],
            "authorize_argv_sha256": adapter_argv_sha,
            "created_unix": 1785729050.0,
        }, "claim_payload_sha256")
        self.job = write_json(
            self.state / "job_claims" / "epoch-0004.json", job_value
        )
        self.runner_argv = [
            str(self.python),
            str(self.runner),
            "--gpus",
            endpoint.GPU_LIST,
            "--cwd",
            str(self.source),
            "--status",
            str(self.status_path),
            "--log",
            str(self.log_path),
            "--",
            "/bin/bash",
            str(self.source / "scripts/show_base/run_base_live_val_8shard.sh"),
            "--repo-root",
            str(self.source),
            "--python",
            str(self.python),
            "--work-authority",
            str(self.work_authority["path"]),
            "--expected-work-authority-sha256",
            str(self.work_authority["sha256"]),
            "--run-root",
            str(self.run_root),
            "--source-commit",
            self.commit,
            "--source-tree",
            self.tree,
            "--paspa-root",
            str(self.paspa),
            "--diffsheg-root",
            str(self.diffsheg),
            "--seed",
            "2026",
            "--diffsheg-batch-size",
            "64",
        ]
        self.authorization_value = endpoint._add_self_hash({
            "format": supervisor.AUTHORIZATION_FORMAT,
            "status": "complete",
            "campaign": copy.deepcopy(self.campaign),
            "candidate_epoch": self.epoch,
            "candidate_receipt": copy.deepcopy(self.candidate_receipt),
            "work_authority": copy.deepcopy(self.work_authority),
            "adapter": copy.deepcopy(control_source["authority_adapter"]),
            "adapter_argv": list(self.adapter_argv),
            "adapter_stdout_sha256": "a" * 64,
            "runner_argv": list(self.runner_argv),
            "completed_unix": 1785729060.0,
        }, "receipt_payload_sha256")
        self.authorization = write_json(
            self.state / "authorizations" / "epoch-0004.json",
            self.authorization_value,
        )
        self.ticket_value = {
            "format": endpoint.FORMAT,
            "status": "authorized",
            "role": "worker",
            "hostname": self.hostname,
            "machine_id": copy.deepcopy(self.machine_artifact),
            "host_identity": copy.deepcopy(
                self.worker_host_identity_artifact
            ),
            "candidate_epoch": self.epoch,
            "campaign": copy.deepcopy(self.campaign),
            "topology": copy.deepcopy(self.topology),
            "endpoint": endpoint_artifact,
            "wave": copy.deepcopy(self.wave),
            "active_claim": copy.deepcopy(self.active),
            "job_claim": copy.deepcopy(self.job),
            "authorization": copy.deepcopy(self.authorization),
            "formal_python": copy.deepcopy(self.formal),
            "source_root": str(self.source),
            "source_origin": endpoint.OFFICIAL_ORIGIN,
            "source_commit": self.commit,
            "source_tree": self.tree,
            "guarded_runner": copy.deepcopy(self.runner_artifact),
            "guard_verifier": copy.deepcopy(self.verifier_artifact),
            "runner_argv": list(self.runner_argv),
            "runner_status_path": str(self.status_path),
            "runner_log_path": str(self.log_path),
            "receipt_path": str(self.receipt_path),
        }
        self.ticket = self.state / "ticket.json"
        self.ticket_artifact = self._publish_ticket(self.ticket_value)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> None:
        completed = REAL_SUBPROCESS_RUN(
            ["git", "-C", str(self.source), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            self.fail(completed.stderr)

    def _git_stdout(self, *arguments: str) -> str:
        completed = REAL_SUBPROCESS_RUN(
            ["git", "-C", str(self.source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip("\n")

    def _publish_ticket(
        self, value: dict[str, object], *, rehash: bool = True, canonical: bool = True
    ) -> dict[str, object]:
        published = copy.deepcopy(value)
        published.pop("ticket_payload_sha256", None)
        if rehash:
            published["ticket_payload_sha256"] = endpoint.canonical_json_sha256(
                published
            )
        else:
            published["ticket_payload_sha256"] = "f" * 64
        raw = (
            endpoint.canonical_json_bytes(published)
            if canonical
            else (json.dumps(published, indent=2, sort_keys=True) + "\n").encode()
        )
        self.ticket.write_bytes(raw)
        self.ticket_value = published
        self.ticket_artifact = artifact(self.ticket)
        return self.ticket_artifact

    def _write_runner_outputs(self) -> None:
        delimiter = self.runner_argv.index("--")
        command = list(self.runner_argv[delimiter + 1 :])
        if self.status_mode == "command-tamper":
            command[-1] = "65"
        status = {
            "updated_at": "2026-08-03T12:00:00Z",
            "state": "finished" if self.status_mode != "failed" else "failed",
            "wrapper_pid": 7001,
            "child_pid": 7002,
            "return_code": 0 if self.status_mode != "failed" else 1,
            "received_signal": None,
            "error": None,
            "cleanup_error": None,
            "restored_guards": dict(self.guards),
            "restore_error": None,
            "command": command,
        }
        self.status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        measurement_path = (
            self.run_root / "candidates" / f"e{self.epoch}" / "live-measurement.json"
        )
        measurement_path.parent.mkdir(parents=True, exist_ok=True)
        measurement_value = {
            "format": supervisor.MEASUREMENT_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "candidate_epoch": self.epoch,
            "candidate_checkpoint": copy.deepcopy(self.candidate_receipt),
            "work_authority": copy.deepcopy(self.work_authority),
            "consumer_claim": {"status": "complete"},
            "execution_preflight": {"status": "complete"},
            "val_inputs_receipt": {"status": "complete"},
            "pipeline_receipt": {"status": "complete"},
            "inference_lineage": {"status": "complete"},
            "diffsheg_report": {"status": "complete"},
            "metrics": {"fgd": 0.125},
            "execution_contract": {
                "expected_shards": 8,
                "exact_once": True,
                "finite": True,
                "may_influence_training": False,
                "requires_e400_reconciliation_for_selection": True,
            },
        }
        measurement_value["receipt_payload_sha256"] = supervisor._bridge_payload_sha(
            measurement_value
        )
        measurement_path.write_text(
            json.dumps(measurement_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        measurement_artifact = artifact(measurement_path)
        self.log_path.write_bytes(
            os.fsencode(str(measurement_path))
            + b"\0"
            + os.fsencode(str(measurement_artifact["sha256"]))
            + b"\0"
            + os.fsencode(str(measurement_value["receipt_payload_sha256"]))
            + b"\0"
        )

    def _subprocess_side_effect(self, argv, **kwargs):
        arguments = list(argv)
        if arguments and arguments[0] == "git":
            return REAL_SUBPROCESS_RUN(arguments, **kwargs)
        if arguments == self.runner_argv:
            self.inner_calls += 1
            self._write_runner_outputs()
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        verifier_argv = [
            str(self.python),
            str(self.verifier),
            *[str(self.guards[str(index)]) for index in range(8)],
        ]
        if arguments == verifier_argv:
            if self.log_tamper_during_verify:
                self.log_path.write_bytes(b"tampered after first read\n")
            stdout = (
                "PASS "
                + " ".join(
                    "GPU%d=PID%d" % (index, self.guards[str(index)])
                    for index in range(8)
                )
                + "\n"
            ).encode("ascii")
            if self.guard_mode == "pid-tamper":
                stdout = stdout.replace(b"GPU7=PID9007", b"GPU7=PID9999")
            if self.guard_mode == "failed":
                return subprocess.CompletedProcess(arguments, 1, b"", b"bad guard")
            return subprocess.CompletedProcess(arguments, 0, stdout, b"")
        self.fail("unexpected subprocess argv: %r" % arguments)

    def run_endpoint(self) -> dict[str, object]:
        with (
            mock.patch.object(endpoint, "MACHINE_ID_PATH", str(self.machine_id)),
            mock.patch.object(
                endpoint,
                "HOST_IDENTITY_PATH",
                str(self.worker_host_identity),
            ),
            mock.patch.object(endpoint, "GUARDED_RUNNER_PATH", str(self.runner)),
            mock.patch.object(endpoint, "GUARD_VERIFIER_PATH", str(self.verifier)),
            mock.patch.object(endpoint, "__file__", str(self.endpoint_copy)),
            mock.patch.object(endpoint, "_SUPERVISOR_PATH", self.supervisor_copy),
            mock.patch.object(
                endpoint.subprocess, "run", side_effect=self._subprocess_side_effect
            ),
        ):
            return endpoint.run_ticket(
                self.ticket,
                str(self.ticket_artifact["sha256"]),
                int(self.ticket_artifact["bytes"]),
                hostname_provider=lambda: self.hostname,
                clock=lambda: 1785729600.0,
            )


class RemoteEndpointPositiveTests(RemoteEndpointFixture):
    def test_success_binds_exact_runner_status_log_and_guards(self) -> None:
        receipt_artifact = self.run_endpoint()
        self.assertEqual(self.inner_calls, 1)
        self.assertEqual(receipt_artifact, artifact(self.receipt_path))
        raw = self.receipt_path.read_bytes()
        receipt = endpoint.strict_json(raw, "receipt", canonical=True)
        claimed = receipt.pop("receipt_payload_sha256")
        self.assertEqual(claimed, endpoint.canonical_json_sha256(receipt))
        self.assertEqual(receipt["candidate_epoch"], self.epoch)
        self.assertEqual(receipt["machine_id"], self.machine_artifact)
        self.assertEqual(
            receipt["host_identity"], self.worker_host_identity_artifact
        )
        self.assertEqual(receipt["restored_guards"], self.guards)
        self.assertEqual(receipt["runner_argv"], self.runner_argv)
        self.assertEqual(receipt["runner_status"], artifact(self.status_path))
        self.assertEqual(receipt["runner_log"], artifact(self.log_path))
        self.assertEqual(receipt["source_commit"], self.commit)
        self.assertEqual(receipt["source_tree"], self.tree)

    def test_main_stdout_is_one_canonical_artifact_and_no_other_output(self) -> None:
        expected = {"path": "/receipt", "sha256": "a" * 64, "bytes": 10}
        stream = io.BytesIO()
        text_stream = io.TextIOWrapper(stream, encoding="utf-8")
        with (
            mock.patch.object(endpoint, "run_ticket", return_value=expected),
            mock.patch.object(sys, "stdout", text_stream),
        ):
            return_code = endpoint.main(
                [
                    "run",
                    "--ticket",
                    "/ticket",
                    "--expected-ticket-sha256",
                    "b" * 64,
                    "--expected-ticket-bytes",
                    "1",
                ]
            )
            text_stream.flush()
        self.assertEqual(return_code, 0)
        self.assertEqual(stream.getvalue(), endpoint.canonical_json_bytes(expected))


class RemoteEndpointNegativeTests(RemoteEndpointFixture):
    def assertRejected(self, pattern: str) -> None:  # noqa: N802 - unittest idiom
        with self.assertRaisesRegex(endpoint.RemoteEndpointError, pattern):
            self.run_endpoint()
        self.assertFalse(self.receipt_path.exists())

    def test_runner_argv_tamper_is_rejected_before_execution(self) -> None:
        tampered = copy.deepcopy(self.ticket_value)
        tampered["runner_argv"][3] = "0,1,2,3,4,5,6"
        self._publish_ticket(tampered)
        self.assertRejected("guarded-runner argv|authorization runner argv|authorization authority")
        self.assertEqual(self.inner_calls, 0)

    def test_hostname_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.ticket_value)
        tampered["hostname"] = "another-worker"
        self._publish_ticket(tampered)
        self.assertRejected("hostname")
        self.assertEqual(self.inner_calls, 0)

    def test_machine_id_tamper_is_rejected(self) -> None:
        self.machine_id.write_bytes(b"fedcba9876543210fedcba9876543210\n")
        self.assertRejected("machine-id SHA")
        self.assertEqual(self.inner_calls, 0)

    def test_live_host_identity_tamper_is_rejected(self) -> None:
        self.worker_host_identity.write_bytes(
            b"99999999-8888-7777-6666-555555555555\n"
        )
        self.assertRejected("worker host identity SHA")
        self.assertEqual(self.inner_calls, 0)

    def test_dirty_source_is_rejected(self) -> None:
        (self.source / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        self.assertRejected("clean zero-branch")
        self.assertEqual(self.inner_calls, 0)

    def test_source_commit_ticket_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.ticket_value)
        tampered["source_commit"] = "0" * 40
        self._publish_ticket(tampered)
        self.assertRejected("commit/tree|campaign source binding")
        self.assertEqual(self.inner_calls, 0)

    def test_runner_status_command_tamper_never_writes_receipt(self) -> None:
        self.status_mode = "command-tamper"
        self.assertRejected("status command")
        self.assertEqual(self.inner_calls, 1)

    def test_runner_failure_never_writes_receipt(self) -> None:
        self.status_mode = "failed"
        self.assertRejected("did not finish successfully")
        self.assertEqual(self.inner_calls, 1)

    def test_log_changed_during_guard_verification_is_rejected(self) -> None:
        self.log_tamper_during_verify = True
        self.assertRejected("runner log|measurement")
        self.assertEqual(self.inner_calls, 1)

    def test_guard_pid_tamper_is_rejected(self) -> None:
        self.guard_mode = "pid-tamper"
        self.assertRejected("verifier PIDs")
        self.assertEqual(self.inner_calls, 1)

    def test_guard_failure_is_rejected(self) -> None:
        self.guard_mode = "failed"
        self.assertRejected("guard verifier failed")
        self.assertEqual(self.inner_calls, 1)

    def test_ticket_self_hash_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.ticket_value)
        tampered["hostname"] = "self-hash-attack"
        self._publish_ticket(tampered, rehash=False)
        self.assertRejected("self-hash")
        self.assertEqual(self.inner_calls, 0)

    def test_noncanonical_ticket_is_rejected(self) -> None:
        self._publish_ticket(self.ticket_value, canonical=False)
        self.assertRejected("not canonical JSON")
        self.assertEqual(self.inner_calls, 0)

    def test_live_endpoint_bytes_tamper_is_rejected_before_execution(self) -> None:
        self.endpoint_copy.write_bytes(b"tampered endpoint\n")
        self.assertRejected("ticket endpoint SHA")
        self.assertEqual(self.inner_calls, 0)

    def test_endpoint_artifact_must_name_the_running_endpoint(self) -> None:
        alternate = write_bytes(self.root / "alternate-endpoint.py", b"alternate\n")
        tampered = copy.deepcopy(self.ticket_value)
        tampered["endpoint"] = alternate
        self._publish_ticket(tampered)
        self.assertRejected("endpoint path")
        self.assertEqual(self.inner_calls, 0)

    def test_topology_self_hash_tamper_is_rejected_before_execution(self) -> None:
        topology_path = Path(str(self.topology["path"]))
        topology = endpoint.strict_json(
            topology_path.read_bytes(), "fixture topology", canonical=True
        )
        topology["worker"]["hostname"] = "tampered-worker"
        topology_path.write_bytes(endpoint.canonical_json_bytes(topology))
        tampered = copy.deepcopy(self.ticket_value)
        tampered["topology"] = artifact(topology_path)
        self._publish_ticket(tampered)
        self.assertRejected("topology self-hash")
        self.assertEqual(self.inner_calls, 0)

    def test_topology_cannot_collapse_both_roles_onto_worker(self) -> None:
        topology_path = Path(str(self.topology["path"]))
        topology = endpoint.strict_json(
            topology_path.read_bytes(), "fixture topology", canonical=True
        )
        topology["master"]["hostname"] = self.hostname
        topology["master"]["machine_id"] = copy.deepcopy(self.machine_artifact)
        topology["master"]["host_identity"] = copy.deepcopy(
            self.worker_host_identity_artifact
        )
        topology.pop("topology_payload_sha256")
        topology = endpoint._add_self_hash(topology, "topology_payload_sha256")
        topology_path.write_bytes(endpoint.canonical_json_bytes(topology))
        tampered = copy.deepcopy(self.ticket_value)
        tampered["topology"] = artifact(topology_path)
        self._publish_ticket(tampered)
        self.assertRejected("collapsed onto one machine")
        self.assertEqual(self.inner_calls, 0)

    def test_wave_self_hash_tamper_is_rejected_before_execution(self) -> None:
        wave_path = Path(str(self.wave["path"]))
        wave = endpoint.strict_json(wave_path.read_bytes(), "fixture wave", canonical=True)
        wave["worker_epoch"] = 8
        wave_path.write_bytes(endpoint.canonical_json_bytes(wave))
        tampered = copy.deepcopy(self.ticket_value)
        tampered["wave"] = artifact(wave_path)
        self._publish_ticket(tampered)
        self.assertRejected("wave self-hash")
        self.assertEqual(self.inner_calls, 0)

    def test_job_self_hash_tamper_is_rejected_before_execution(self) -> None:
        job_path = Path(str(self.job["path"]))
        job = endpoint.strict_json(job_path.read_bytes(), "fixture job", canonical=True)
        job["run_root"] = str(self.root / "wrong-run")
        job_path.write_bytes(endpoint.canonical_json_bytes(job))
        tampered = copy.deepcopy(self.ticket_value)
        tampered["job_claim"] = artifact(job_path)
        self._publish_ticket(tampered)
        self.assertRejected("job claim self-hash")
        self.assertEqual(self.inner_calls, 0)

    def test_authorization_self_hash_tamper_is_rejected_before_execution(self) -> None:
        authorization_path = Path(str(self.authorization["path"]))
        authorization = endpoint.strict_json(
            authorization_path.read_bytes(), "fixture authorization", canonical=True
        )
        authorization["adapter_stdout_sha256"] = "b" * 64
        authorization_path.write_bytes(endpoint.canonical_json_bytes(authorization))
        tampered = copy.deepcopy(self.ticket_value)
        tampered["authorization"] = artifact(authorization_path)
        self._publish_ticket(tampered)
        self.assertRejected("authorization self-hash")
        self.assertEqual(self.inner_calls, 0)

    def test_existing_receipt_is_an_o_excl_terminal_rejection(self) -> None:
        original = b"existing receipt must survive\n"
        self.receipt_path.write_bytes(original)
        with self.assertRaisesRegex(endpoint.RemoteEndpointError, "already exists"):
            self.run_endpoint()
        self.assertEqual(self.receipt_path.read_bytes(), original)
        self.assertEqual(self.inner_calls, 0)


if __name__ == "__main__":
    unittest.main()
