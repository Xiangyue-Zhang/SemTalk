#!/usr/bin/env python3
"""CPU-only adversarial checks for the V14 live-validation supervisor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/show_base/supervise_base_v14_live_validation.py"
SPEC = importlib.util.spec_from_file_location("v14_live_supervisor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sup)


def write_bytes(path: Path, raw: bytes, executable: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if executable:
        path.chmod(0o755)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def write_json(path: Path, value: dict) -> dict:
    return write_bytes(path, sup.canonical_json_bytes(value))


def write_bridge_json(path: Path, value: dict) -> dict:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    return write_bytes(path, raw)


def bridge_hash(value: dict) -> str:
    return sup._bridge_payload_sha(value)


class Fixture:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="v14-live-v3-")
        self.root = Path(self.temp.name).resolve()
        self.state = self.root / "state"
        self.train = self.root / "train"
        (self.train / "candidate_receipts").mkdir(parents=True)
        for name in ("job_claims", "completions", "runner_status", "runner_logs", "authorities", "authorizations"):
            (self.state / name).mkdir(parents=True)
        self.launcher = write_bytes(self.root / "control/run_base_live_val_8shard.sh", b"#!/bin/bash\n", executable=True)
        self.adapter = write_bytes(self.root / "control/base_v14_live_validation_authority.py", b"# adapter\n")
        self.bridge = write_bytes(self.root / "control/base_live_val_consumer_bridge.py", b"# bridge\n")
        self.runner = {"path": "/tmp/globaldiff_guarded_runner.py", "sha256": "1" * 64, "bytes": 1}
        self.verifier = {"path": "/tmp/verify_globaldiff_guards.py", "sha256": "2" * 64, "bytes": 1}
        self.formal = {"argv0": str(self.root / "venv/bin/python")}
        self.runtime = {
            "source_root": sup.RUNTIME_VALIDATION_ROOT,
            "contract": {"path": sup.RUNTIME_VALIDATION_ROOT + "/scripts/show_base/base_long_val_contract.py", "sha256": sup.RUNTIME_VALIDATION_CONTRACT_SHA256, "bytes": 1},
            "selector": {"path": sup.RUNTIME_VALIDATION_ROOT + "/scripts/show_base/select_base_official_adapt.py", "sha256": sup.RUNTIME_VALIDATION_SELECTOR_SHA256, "bytes": 1},
            "validation_semantics_source": {
                "source_root": sup.VALIDATION_SEMANTICS_ROOT,
                "contract": {"path": sup.VALIDATION_SEMANTICS_ROOT + "/scripts/show_base/base_long_val_contract.py", "sha256": sup.VALIDATION_SEMANTICS_CONTRACT_SHA256, "bytes": 1},
                "selector": {"path": sup.VALIDATION_SEMANTICS_ROOT + "/scripts/show_base/select_base_official_adapt.py", "sha256": sup.VALIDATION_SEMANTICS_SELECTOR_SHA256, "bytes": 1},
            },
        }
        self.adapter_config = {
            "train_root": str(self.train), "topology_mode": "validation_gated_w8_l128_g1024_empirical_acceleration",
            "producer_source_root": str(self.root / "producer"),
            "expected_producer_trainer_sha256": sup.PRODUCER_TRAINER_SHA256,
            "expected_producer_contract_sha256": sup.PRODUCER_CONTRACT_SHA256,
            "schedule": {"path": str(self.root / "schedule.json"), "sha256": "5" * 64, "bytes": 1},
            "expected_frozen_inputs_sha256": "6" * 64,
            "val_inputs": {"path": str(self.root / "val.json"), "sha256": "7" * 64, "bytes": 1},
            "pipeline": {"path": str(self.root / "pipeline.json"), "sha256": "8" * 64, "bytes": 1},
        }
        (self.root / "producer").mkdir()
        self.jobs = []
        for epoch in sup.CANDIDATE_EPOCHS:
            run_root = self.root / "runs" / ("e%d" % epoch)
            self.jobs.append({
                "epoch": epoch,
                "candidate_receipt_path": str(self.train / "candidate_receipts" / ("epoch-%04d.json" % epoch)),
                "authority_path": str(self.state / "authorities" / ("epoch-%04d.json" % epoch)),
                "authorization_path": str(self.state / "authorizations" / ("epoch-%04d.json" % epoch)),
                "run_root": str(run_root),
                "measurement_path": str(run_root / "candidates" / ("e%d" % epoch) / "live-measurement.json"),
                "completion_path": str(self.state / "completions" / ("epoch-%04d.json" % epoch)),
                "runner_status_path": str(self.state / "runner_status" / ("epoch-%04d.json" % epoch)),
                "runner_log_path": str(self.state / "runner_logs" / ("epoch-%04d.log" % epoch)),
            })

    def close(self) -> None:
        self.temp.cleanup()

    def campaign(self) -> dict:
        return {
            "_artifact": {"path": str(self.root / "campaign.json"), "sha256": "9" * 64, "bytes": 1},
            "_state_root": self.state,
            "_claims_dir": self.state / "job_claims", "_completions_dir": self.state / "completions",
            "_runner_status_dir": self.state / "runner_status", "_runner_logs_dir": self.state / "runner_logs",
            "_authorities_dir": self.state / "authorities", "_authorizations_dir": self.state / "authorizations",
            "_selection_root": self.root / "selection", "_reconciliation_path": self.state / "reconciliation.json",
            "_final_manifest_path": self.train / "candidate_manifest.json", "_final_status_path": self.train / "status.json",
            "campaign_claim_path": str(self.state / "campaign.claim.json"),
            "summary_path": str(self.state / "final_summary.json"), "_jobs": self.jobs,
            "_formal_python": self.formal, "_guarded_runner": self.runner, "_guard_verifier": self.verifier,
            "_control_source": {"root": str(self.root / "control"), "commit": "a" * 40, "tree": "b" * 40,
                "launcher": self.launcher, "bridge": self.bridge, "authority_adapter": self.adapter},
            "_runtime_validation_source": self.runtime, "_adapter_config": self.adapter_config,
            "_paspa_root": str(self.root / "paspa"), "_diffsheg_root": str(self.root / "diffsheg"),
            "_seed": 20260731, "_diffsheg_batch_size": 64,
            "_runtime_contract": {"path": str(self.root / "runtime.sh"), "sha256": "0" * 64, "bytes": 1},
        }


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()

    def tearDown(self) -> None:
        self.fx.close()

    def test_exact_candidate_queue(self) -> None:
        self.assertEqual(sup.CANDIDATE_EPOCHS, (1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400))

    def test_formal_producer_is_frozen_at_8f_not_the_control_successor(self) -> None:
        self.assertEqual(
            sup.PRODUCER_SOURCE_COMMIT,
            "8f1fa7b85ed8253600a4c571e98eb9927edeb073",
        )
        self.assertEqual(
            sup.PRODUCER_SOURCE_TREE,
            "d5e5eb74acdc6d9ca330d5731e718e33b67cf626",
        )
        self.assertEqual(
            sup.PRODUCER_TRAINER_SHA256,
            "29fdd5d3e9bdfc61904f649b71d4dae1766b42a4a6a5a40b2bbd37a4c6b33173",
        )
        self.assertEqual(
            sup.PRODUCER_CONTRACT_SHA256,
            "3526ca896f23e7849545e3f81553dd242dc1ca3049eabdeeb89fc38357616323",
        )

    def test_real_double_symlink_venv_chain_is_bound(self) -> None:
        base = self.fx.root / "base/bin"
        base.mkdir(parents=True)
        final = base / "python3"
        final.write_bytes(b"ELF-python\n")
        final.chmod(0o755)
        venv_bin = self.fx.root / "realvenv/bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python3").symlink_to(final)
        (venv_bin / "python").symlink_to("python3")
        cfg = write_bytes(self.fx.root / "realvenv/pyvenv.cfg", ("home = %s\ninclude-system-site-packages = false\nversion = 3.10.0\nexecutable = %s\n" % (base, final)).encode())
        binding = {"format": "semtalk.formal_venv_python_binding.v1", "argv0": str(venv_bin / "python"), "venv_root": str(self.fx.root / "realvenv"),
            "symlink_chain": [{"path": str(venv_bin / "python"), "target": "python3"}, {"path": str(venv_bin / "python3"), "target": str(final)}],
            "resolved_target": write_bytes(final, final.read_bytes(), executable=True), "pyvenv_cfg": cfg}
        observed = sup._formal_python(binding, running_executable=str(venv_bin / "python"))
        self.assertEqual(len(observed["symlink_chain"]), 2)
        (venv_bin / "python3").unlink()
        (venv_bin / "python3").symlink_to(base / "other")
        with self.assertRaises(sup.SupervisorError):
            sup._formal_python(binding, running_executable=str(venv_bin / "python"))

    def test_exact_runner_workload_is_tracked_bash_launcher(self) -> None:
        campaign = self.fx.campaign()
        job = {**self.fx.jobs[0], "work_authority": {"path": str(self.fx.root / "authority.json"), "sha256": "c" * 64, "bytes": 10}}
        workload = sup._runner_workload(job, campaign)
        control = campaign["_control_source"]
        runtime_root = campaign["_runtime_validation_source"]["source_root"]
        self.assertEqual(workload[:2], ["/bin/bash", control["launcher"]["path"]])
        self.assertEqual(workload[workload.index("--repo-root") + 1], control["root"])
        self.assertEqual(workload[workload.index("--source-commit") + 1], control["commit"])
        self.assertEqual(workload[workload.index("--source-tree") + 1], control["tree"])
        self.assertNotEqual(runtime_root, control["root"])
        self.assertNotIn(runtime_root, workload)
        authorize = sup._authorize_argv(campaign, job)
        authority_reconcile = sup._authority_reconcile_argv(
            campaign,
            {"path": "/final-manifest", "sha256": "d" * 64, "bytes": 1},
            {"path": "/final-status", "sha256": "e" * 64, "bytes": 1},
        )
        bridge_reconcile = sup._reconcile_argv(campaign, [], "f" * 64)
        bridge_replay = sup._bridge_replay(
            campaign, job,
            {"path": "/measurement", "sha256": "1" * 64, "receipt_payload_sha256": "2" * 64},
            lambda _argv: (0, b"PASS\n", b""),
        )
        self.assertEqual(authorize[2], control["authority_adapter"]["path"])
        self.assertEqual(authority_reconcile[2], control["authority_adapter"]["path"])
        self.assertEqual(bridge_reconcile[2], control["bridge"]["path"])
        self.assertEqual(bridge_replay[4], control["bridge"]["path"])
        argv = sup._runner_argv_v2(job, campaign)
        self.assertEqual(argv.count("--cwd"), 1)
        self.assertEqual(argv[argv.index("--cwd") + 1], control["root"])
        self.assertLess(argv.index("--cwd"), argv.index("--"))
        self.assertEqual(argv[argv.index("--") + 1:argv.index("--") + 3], ["/bin/bash", self.fx.launcher["path"]])
        self.assertEqual(sup._validate_runner_argv_v2(argv, job, campaign), argv)
        missing_cwd = list(argv)
        cwd_index = missing_cwd.index("--cwd")
        del missing_cwd[cwd_index:cwd_index + 2]
        with self.assertRaisesRegex(sup.SupervisorError, "exact tracked"):
            sup._validate_runner_argv_v2(missing_cwd, job, campaign)
        changed = list(argv)
        changed[changed.index("--") + 1] = self.fx.formal["argv0"]
        with self.assertRaisesRegex(sup.SupervisorError, "exact tracked"):
            sup._validate_runner_argv_v2(changed, job, campaign)

    def test_build_campaign_succeeds_with_zero_candidate_receipts(self) -> None:
        build_root = self.fx.root / "build"
        for name in ("control/scripts/show_base", "runtime/scripts/show_base", "evidence/scripts/show_base", "runs", "paspa", "diffsheg", "producer", "train"):
            (build_root / name).mkdir(parents=True, exist_ok=True)
        runtime_contract = write_bytes(build_root / "control/scripts/show_base/formal_python_runtime_contract.sh", b"#!/bin/bash\n", executable=True)
        schedule = write_bytes(build_root / "schedule.json", b"{}\n")
        val_inputs = write_bytes(build_root / "val.json", b"{}\n")
        pipeline = write_bytes(build_root / "pipeline.json", b"{}\n")
        control = {"root": str(build_root / "control"), "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git", "commit": "a" * 40, "tree": "b" * 40,
            "supervisor": {"path": str(SCRIPT), "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), "bytes": SCRIPT.stat().st_size},
            "launcher": self.fx.launcher, "bridge": self.fx.bridge, "authority_adapter": self.fx.adapter}
        runtime = {"source_root": str(build_root / "runtime"), "contract": {"path": str(build_root / "runtime/scripts/show_base/base_long_val_contract.py"), "sha256": sup.RUNTIME_VALIDATION_CONTRACT_SHA256, "bytes": 1},
            "selector": {"path": str(build_root / "runtime/scripts/show_base/select_base_official_adapt.py"), "sha256": sup.RUNTIME_VALIDATION_SELECTOR_SHA256, "bytes": 1},
            "validation_semantics_source": {"source_root": str(build_root / "evidence"), "contract": {"path": str(build_root / "evidence/scripts/show_base/base_long_val_contract.py"), "sha256": sup.VALIDATION_SEMANTICS_CONTRACT_SHA256, "bytes": 1}, "selector": {"path": str(build_root / "evidence/scripts/show_base/select_base_official_adapt.py"), "sha256": sup.VALIDATION_SEMANTICS_SELECTOR_SHA256, "bytes": 1}}}
        args = argparse.Namespace(output=str(build_root / "state/campaign.json"), state_root=str(build_root / "state"), control_root=str(build_root / "control"), control_commit="a" * 40, control_tree="b" * 40,
            runtime_validation_root=sup.RUNTIME_VALIDATION_ROOT, runtime_validation_commit=sup.RUNTIME_VALIDATION_COMMIT, runtime_validation_tree=sup.RUNTIME_VALIDATION_TREE,
            runtime_contract_sha256=sup.RUNTIME_VALIDATION_CONTRACT_SHA256, runtime_evaluator_sha256=sup.RUNTIME_VALIDATION_EVALUATOR_SHA256, runtime_selector_sha256=sup.RUNTIME_VALIDATION_SELECTOR_SHA256,
            validation_evidence_root=sup.VALIDATION_SEMANTICS_ROOT, validation_evidence_contract_sha256=sup.VALIDATION_SEMANTICS_CONTRACT_SHA256, validation_evidence_selector_sha256=sup.VALIDATION_SEMANTICS_SELECTOR_SHA256,
            train_root=str(build_root / "train"), topology_mode="validation_gated_w8_l128_g1024_empirical_acceleration", producer_source_root=str(build_root / "producer"),
            expected_producer_trainer_sha256=sup.PRODUCER_TRAINER_SHA256, expected_producer_contract_sha256=sup.PRODUCER_CONTRACT_SHA256, schedule=schedule["path"], expected_schedule_sha256=schedule["sha256"], expected_frozen_inputs_sha256="3" * 64,
            val_inputs=val_inputs["path"], expected_val_inputs_sha256=val_inputs["sha256"], pipeline=pipeline["path"], expected_pipeline_sha256=pipeline["sha256"],
            formal_python=str(build_root / "venv/bin/python"), guarded_runner="/tmp/globaldiff_guarded_runner.py",
            expected_guarded_runner_sha256="4" * 64, expected_guarded_runner_bytes=1,
            guard_verifier="/tmp/verify_globaldiff_guards.py",
            expected_guard_verifier_sha256="5" * 64, expected_guard_verifier_bytes=1,
            run_root_base=str(build_root / "runs"),
            reconcile_selection_root=str(build_root / "selection"), paspa_root=str(build_root / "paspa"), diffsheg_root=str(build_root / "diffsheg"), seed=20260731, diffsheg_batch_size=64)
        formal = {"argv0": args.formal_python}
        real_artifact = sup._artifact_from_path
        def fake_artifact(path, label, executable=False):
            if path == "/tmp/globaldiff_guarded_runner.py":
                return {"path": path, "sha256": "4" * 64, "bytes": 1}
            if path == "/tmp/verify_globaldiff_guards.py":
                return {"path": path, "sha256": "5" * 64, "bytes": 1}
            if path == str(build_root / "producer/scripts/show_base/train_base_official_adapt_long.py"):
                return {"path": path, "sha256": sup.PRODUCER_TRAINER_SHA256, "bytes": 1}
            if path == str(build_root / "producer/scripts/show_base/base_v14_formal_contract.py"):
                return {"path": path, "sha256": sup.PRODUCER_CONTRACT_SHA256, "bytes": 1}
            return real_artifact(path, label, executable=executable)
        with mock.patch.object(sup, "_freeze_control_source", return_value=control), mock.patch.object(sup, "_freeze_runtime_validation_source", return_value=runtime), mock.patch.object(sup, "_capture_formal_python_binding", return_value=formal), mock.patch.object(sup, "_artifact_from_path", side_effect=fake_artifact), mock.patch.object(sup, "_git_authority") as git_authority, mock.patch.object(sup, "_git_stdout", side_effect=lambda _root, *items: items[-1]), mock.patch.object(sup, "load_campaign"):
            result = sup.build_campaign(args)
        git_authority.assert_called_once_with(
            build_root / "producer", sup.PRODUCER_SOURCE_COMMIT,
            sup.PRODUCER_SOURCE_TREE, "producer source",
        )
        self.assertEqual(result["candidate_count"], 22)
        campaign = json.loads((build_root / "state/campaign.json").read_text())
        self.assertEqual(len(campaign["jobs"]), 22)
        self.assertTrue(all(set(job) == sup.JOB_KEYS for job in campaign["jobs"]))
        self.assertFalse((build_root / "train/candidate_receipts").exists())
        self.assertEqual(list((build_root / "state/authorities").iterdir()), [])
        self.assertNotIn("work_authority", campaign["jobs"][0])

    def test_no_receipt_returns_waiting_without_job_or_active_claim(self) -> None:
        campaign = self.fx.campaign()
        result = sup.run_next(campaign, clock=lambda: 1.0)
        self.assertEqual(result["status"], "waiting_for_candidate_receipt")
        self.assertFalse(result["job_claim_created"])
        self.assertFalse((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertFalse((self.fx.state / "active_invocation.claim.json").exists())

    def test_receipts_appearing_in_order_authorize_each_head(self) -> None:
        campaign = self.fx.campaign()
        jobs = self.fx.jobs[:2]
        for job in jobs:
            write_json(Path(job["candidate_receipt_path"]), {"epoch": job["epoch"]})
        scans = [([], jobs[0]), ([({}, {})], jobs[1])]
        authorize_epochs = []
        def capture(argv):
            if "authorize" in argv:
                epoch = int(argv[argv.index("--epoch") + 1])
                authorize_epochs.append(epoch)
                job = jobs[len(authorize_epochs) - 1]
                authority = write_json(Path(job["authority_path"]), {"candidate_epoch": epoch})
                stdout = (json.dumps(authority, ensure_ascii=False, sort_keys=True) + "\n").encode()
                return 0, stdout, b""
            return 0, b"", b""
        fake_status_artifact = {"path": str(self.fx.root / "status"), "sha256": "a" * 64, "bytes": 1}
        fake_status = {"restored_guards": {str(i): 100 + i for i in range(8)}}
        fake_measurement = {"path": str(self.fx.root / "measurement"), "sha256": "b" * 64, "bytes": 1, "receipt_payload_sha256": "c" * 64}
        fake_value = {"metrics": {"fgd": 0.25}}
        with mock.patch.object(sup, "_scan_v2", side_effect=scans), mock.patch.object(sup, "_revalidate_control"), mock.patch.object(sup, "_runtime_contract_check"), mock.patch.object(sup, "_work_authority_runtime_binding"), mock.patch.object(sup, "_work_authority_candidate_binding"), mock.patch.object(sup, "_runner_status", return_value=(fake_status_artifact, fake_status)), mock.patch.object(sup, "_verify_runner_log", return_value=({"path": str(self.fx.root / "log"), "sha256": "d" * 64, "bytes": 1}, fake_measurement, fake_value)), mock.patch.object(sup, "_bridge_replay", return_value=["bridge"]), mock.patch.object(sup, "_guard_verify", return_value=(["verify"], "PASS " + " ".join("GPU%d=PID%d" % (i, 100 + i) for i in range(8)) + "\n", fake_status["restored_guards"])), mock.patch.object(sup, "_measurement", return_value=(fake_measurement, fake_value)):
            first = sup.run_next(campaign, runner=lambda _argv: 0, capture=capture, clock=lambda: 1.0)
            second = sup.run_next(campaign, runner=lambda _argv: 0, capture=capture, clock=lambda: 2.0)
        self.assertEqual(authorize_epochs, [1, 2])
        self.assertEqual([first["candidate_epoch"], second["candidate_epoch"]], [1, 2])
        self.assertTrue(Path(jobs[0]["authorization_path"]).exists() and Path(jobs[1]["authorization_path"]).exists())

    def test_authorize_failure_is_terminal_and_never_retried(self) -> None:
        campaign = self.fx.campaign()
        write_json(Path(self.fx.jobs[0]["candidate_receipt_path"]), {"epoch": 1})
        calls = []
        def fail(argv):
            calls.append(list(argv))
            return 9, b"", b"adapter failed"
        with mock.patch.object(sup, "_revalidate_control"), mock.patch.object(sup, "_runtime_contract_check"):
            with self.assertRaisesRegex(sup.SupervisorError, "adapter authorize failed"):
                sup.run_next(campaign, capture=fail, clock=lambda: 1.0)
        self.assertTrue((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertTrue((self.fx.state / "active_invocation.claim.json").exists())
        with self.assertRaisesRegex(sup.SupervisorError, "claim exists without completion"):
            sup.run_next(campaign, capture=lambda _argv: self.fail("adapter was retried"), clock=lambda: 2.0)
        self.assertEqual(len(calls), 1)

    def test_e400_final_absent_waits_without_reconcile_claim(self) -> None:
        campaign = self.fx.campaign()
        result = sup._finalize_v2(campaign, [({}, {}) for _ in sup.CANDIDATE_EPOCHS], lambda _argv: 0, lambda _argv: (0, b"", b""), lambda: 1.0)
        self.assertEqual(result["status"], "waiting_for_e400_final")
        self.assertFalse((self.fx.state / "producer_reconcile.claim.json").exists())
        self.assertFalse((self.fx.state / "bridge_reconcile.claim.json").exists())
        self.assertFalse((self.fx.state / "active_invocation.claim.json").exists())

    def test_state_directories_reject_unknown_entries(self) -> None:
        campaign = self.fx.campaign()
        (self.fx.state / "authorizations/surprise.json").write_bytes(b"x")
        with self.assertRaisesRegex(sup.SupervisorError, "unknown entry"):
            sup._scan_v2(campaign)

    def test_runner_stdout_is_exact_bounded_trailing_nul_triple(self) -> None:
        job = {**self.fx.jobs[0], "work_authority": {"path": "/authority", "sha256": "a" * 64, "bytes": 1}}
        value = {"format": sup.MEASUREMENT_FORMAT, "status": "complete", "split": "val", "test_visible": False, "selection_eligible": False, "candidate_epoch": 1,
            "candidate_checkpoint": {}, "work_authority": job["work_authority"], "consumer_claim": {}, "execution_preflight": {}, "val_inputs_receipt": {}, "pipeline_receipt": {}, "inference_lineage": {}, "diffsheg_report": {}, "metrics": {"fgd": 0.25},
            "execution_contract": {"expected_shards": 8, "exact_once": True, "finite": True, "may_influence_training": False, "requires_e400_reconciliation_for_selection": True}}
        value["receipt_payload_sha256"] = bridge_hash(value)
        artifact = write_json(Path(job["measurement_path"]), value)
        raw = (job["measurement_path"] + "\0" + artifact["sha256"] + "\0" + value["receipt_payload_sha256"] + "\0").encode()
        write_bytes(Path(job["runner_log_path"]), raw)
        sup._verify_runner_log(job)
        Path(job["runner_log_path"]).write_bytes(raw + b"x")
        with self.assertRaisesRegex(sup.SupervisorError, "trailing-NUL"):
            sup._verify_runner_log(job)

    def test_guard_verifier_exact_line_and_pid_order(self) -> None:
        campaign = {"_formal_python": self.fx.formal, "_guard_verifier": self.fx.verifier}
        status = {"restored_guards": {str(i): 300 + i for i in range(8)}}
        def good(_argv):
            return 0, ("PASS " + " ".join("GPU%d=PID%d" % (i, 300 + i) for i in range(8)) + "\n").encode(), b""
        _argv, _text, observed = sup._guard_verify(campaign, status, good)
        self.assertEqual(observed, status["restored_guards"])
        with self.assertRaisesRegex(sup.SupervisorError, "differ"):
            sup._guard_verify(campaign, status, lambda _argv: (0, ("PASS " + " ".join("GPU%d=PID%d" % (i, 301 if i == 0 else 300 + i) for i in range(8)) + "\n").encode(), b""))

    def test_official_selection_requires_selected_state_and_v2_format(self) -> None:
        selection_root = self.fx.root / "selection"
        selection_root.mkdir()
        selected = {"epoch": 80, "candidate_checkpoint": {"path": "/checkpoint", "sha256": "3" * 64}, "fgd": 0.125, "inference_lineage": {"path": "/lineage", "sha256": "4" * 64}, "diffsheg_report": {"path": "/report", "sha256": "5" * 64}}
        completed = [({}, {"measurement": {"path": "/measurement/e%d.json" % epoch, "sha256": "%064x" % epoch, "bytes": 100 + epoch, "receipt_payload_sha256": "%064x" % (1000 + epoch)}}) for epoch in sup.CANDIDATE_EPOCHS]
        reconciliation = {"path": "/reconciliation.json", "sha256": "6" * 64, "bytes": 123, "receipt_payload_sha256": "7" * 64}
        value = {"format": sup.SELECTION_FORMAT, "status": "selected", "split": "val", "test_visible": False, "selection_eligible": True, "candidate_epochs": list(sup.CANDIDATE_EPOCHS), "reconciliation_receipt": reconciliation,
            "live_measurements": [row[1]["measurement"] for row in completed], "reconciled_measurements": [{"path": "/standard/e%d" % epoch} for epoch in sup.CANDIDATE_EPOCHS], "formal_selection": {"selected": selected}, "selected": selected, "test_evaluations_observed": 0}
        value["receipt_payload_sha256"] = bridge_hash(value)
        write_bridge_json(selection_root / "selection.json", value)
        artifact, observed = sup._selection_v2({"_selection_root": selection_root}, completed, reconciliation)
        self.assertEqual(artifact["receipt_payload_sha256"], value["receipt_payload_sha256"])
        self.assertEqual(observed["format"], "semtalk_show_base_live_val_22way_selection_v2")

    def test_bridge_reconcile_stdout_is_one_canonical_bound_selection(self) -> None:
        artifact = {
            "path": "/selection.json", "sha256": "a" * 64, "bytes": 123,
            "receipt_payload_sha256": "b" * 64,
        }
        selection = {"selected": {"epoch": 80, "fgd": 0.125}}
        value = {
            "status": "selected", "split": "val", "test_visible": False,
            "selection_eligible": True,
            "candidate_count": len(sup.CANDIDATE_EPOCHS),
            "selected_epoch": 80, "selected_fgd": 0.125,
            "selection": artifact,
        }
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        self.assertEqual(
            sup._validate_bridge_reconcile_stdout(raw, artifact, selection), value,
        )
        with self.assertRaisesRegex(sup.SupervisorError, "canonical"):
            sup._validate_bridge_reconcile_stdout(raw + b"\n", artifact, selection)
        attacked = dict(value)
        attacked["selected_epoch"] = 100
        attacked_raw = (
            json.dumps(attacked, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        with self.assertRaisesRegex(sup.SupervisorError, "authoritative selection"):
            sup._validate_bridge_reconcile_stdout(attacked_raw, artifact, selection)


if __name__ == "__main__":
    unittest.main(verbosity=2)
