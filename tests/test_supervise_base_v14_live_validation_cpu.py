#!/usr/bin/env python3
"""CPU-only adversarial checks for the V14 live-validation supervisor."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_live_val_consumer_bridge as bridge


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


def write_failed_run_fixture(root: Path) -> dict:
    for relative in ("candidates/e1/shards", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    fixture_pins = {}
    for relative in (
        "diffsheg-evaluator-bundle.json", "work-preflight.json",
        "logs/evaluator-preflight.log", "logs/work-inspect.json",
        "logs/work-preflight.log",
    ):
        artifact = write_bytes(root / relative, (relative + "\n").encode())
        fixture_pins[relative] = (artifact["sha256"], artifact["bytes"])
    shard_raw = ("prefix\n" + sup.FAILED_SHARD_MARKER + "\nsuffix\n").encode()
    for shard in range(8):
        relative = "logs/e1-shard%d.log" % shard
        artifact = write_bytes(root / relative, shard_raw)
        fixture_pins[relative] = (artifact["sha256"], artifact["bytes"])
    return fixture_pins


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

    def test_recovery_request_uses_bridge_payload_hash_abi(self) -> None:
        artifact = {
            "path": "/tmp/恢复验证/artifact.json",
            "sha256": "a" * 64,
            "bytes": 1,
        }
        payload = {
            "format": sup.RECOVERY_REQUEST_FORMAT,
            "status": "ready_for_single_recovery",
            "candidate_epoch": 1,
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "failed_campaign": artifact,
            "failed_job_claim": artifact,
            "failed_active_claim": artifact,
            "failed_authorization": artifact,
            "failed_work_authority": artifact,
            "failed_consumer_claim": artifact,
            "failed_runner_status": artifact,
            "failed_runner_log": artifact,
            "failed_run_root": "/tmp/恢复验证/failed-run",
            "failed_run_inventory": {
                "semantic_outputs": [],
                "marker": "导入来源错误",
            },
            "failed_process_proof": {
                "wrapper_proc_state": "absent",
                "child_proc_state": "absent",
            },
            "guard_proof": {"stdout": "PASS 恢复验证\n"},
            "new_campaign": artifact,
            "new_control_source": {"root": "/tmp/恢复验证/control"},
            "new_work_authority": artifact,
            "new_run_root": "/tmp/恢复验证/new-run",
            "created_unix": 1.0,
        }
        self.assertEqual(
            set(payload) | {"receipt_payload_sha256"},
            sup.RECOVERY_REQUEST_KEYS,
        )
        value = sup._add_bridge_self_hash(payload, "receipt_payload_sha256")
        self.assertEqual(
            value["receipt_payload_sha256"], bridge._payload_sha(value)
        )
        self.assertNotEqual(
            value["receipt_payload_sha256"],
            bridge._supervisor_payload_sha(
                value, "receipt_payload_sha256"
            ),
        )
        with self.assertRaisesRegex(
            sup.SupervisorError, "bridge self-hash field already present"
        ):
            sup._add_bridge_self_hash(value, "receipt_payload_sha256")

        tree = ast.parse(Path(sup.__file__).read_text(encoding="utf-8"))
        request_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_recovery_request_value"
        )
        bridge_hash_calls = [
            node
            for node in ast.walk(request_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_add_bridge_self_hash"
        ]
        self.assertEqual(len(bridge_hash_calls), 1)
        call = bridge_hash_calls[0]
        self.assertGreaterEqual(len(call.args), 2)
        self.assertIsInstance(call.args[1], ast.Constant)
        self.assertEqual(call.args[1].value, "receipt_payload_sha256")
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_add_self_hash"
                for node in ast.walk(request_function)
            )
        )

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
        recovery = {
            label: write_json(self.fx.root / ("recovery-%s.json" % label), {label: True})
            for label in ("request", "authority", "claim")
        }
        recovered_job = {**job, "consumer_recovery": recovery}
        recovered_workload = sup._runner_workload(recovered_job, campaign)
        self.assertEqual(
            recovered_workload[-8:],
            [
                "--recovery-authority", recovery["authority"]["path"],
                "--expected-recovery-authority-sha256",
                recovery["authority"]["sha256"],
                "--recovery-claim", recovery["claim"]["path"],
                "--expected-recovery-claim-sha256",
                recovery["claim"]["sha256"],
            ],
        )

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

        adoption_value = {
            key: None
            for key in sup.ADOPTED_E1_KEYS
            if key != "receipt_payload_sha256"
        }
        adoption_value.update({
            "format": sup.ADOPTED_E1_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "candidate_epoch": 1,
        })
        adoption_value["receipt_payload_sha256"] = sup._bridge_payload_sha(
            adoption_value
        )
        adoption = write_json(
            build_root
            / "train/live_val_consumer_adoption_claims/epoch-0001.json",
            adoption_value,
        )
        (build_root / "runs-adopted").mkdir()
        adopted_args = argparse.Namespace(**vars(args))
        adopted_args.state_root = str(build_root / "state-adopted")
        adopted_args.output = str(
            build_root / "state-adopted/campaign.json"
        )
        adopted_args.run_root_base = str(build_root / "runs-adopted")
        adopted_args.reconcile_selection_root = str(
            build_root / "selection-adopted"
        )
        adopted_args.adopted_e1 = adoption["path"]
        adopted_args.expected_adopted_e1_sha256 = adoption["sha256"]
        adopted_args.expected_adopted_e1_bytes = adoption["bytes"]
        with mock.patch.object(
            sup, "_freeze_control_source", return_value=control
        ), mock.patch.object(
            sup, "_freeze_runtime_validation_source", return_value=runtime
        ), mock.patch.object(
            sup, "_capture_formal_python_binding", return_value=formal
        ), mock.patch.object(
            sup, "_artifact_from_path", side_effect=fake_artifact
        ), mock.patch.object(
            sup, "_git_authority"
        ), mock.patch.object(
            sup, "_git_stdout", side_effect=lambda _root, *items: items[-1]
        ), mock.patch.object(
            sup,
            "_load_adopted_e1",
            return_value=(adoption, {"candidate_epoch": 1}),
        ) as adoption_loader, mock.patch.object(sup, "load_campaign"):
            adopted_result = sup.build_campaign(adopted_args)
        adoption_loader.assert_called_once()
        self.assertTrue(
            adoption_loader.call_args.kwargs["revalidate_terminal_tree"]
        )
        self.assertTrue(
            adoption_loader.call_args.kwargs["verify_live_guards"]
        )
        self.assertEqual(adopted_result["candidate_count"], 22)
        adopted_campaign = json.loads(
            (build_root / "state-adopted/campaign.json").read_text()
        )
        self.assertEqual(
            adopted_campaign["format"], sup.ADOPTION_CAMPAIGN_FORMAT
        )
        self.assertEqual(adopted_campaign["adopted_e1"], adoption)
        self.assertEqual(
            list((build_root / "state-adopted/authorities").iterdir()), []
        )

        (build_root / "runs-rejected").mkdir()
        rejected_args = argparse.Namespace(**vars(adopted_args))
        rejected_args.state_root = str(build_root / "state-rejected")
        rejected_args.output = str(
            build_root / "state-rejected/campaign.json"
        )
        rejected_args.run_root_base = str(build_root / "runs-rejected")
        rejected_args.reconcile_selection_root = str(
            build_root / "selection-rejected"
        )
        with mock.patch.object(
            sup, "_freeze_control_source", return_value=control
        ), mock.patch.object(
            sup, "_freeze_runtime_validation_source", return_value=runtime
        ), mock.patch.object(
            sup, "_capture_formal_python_binding", return_value=formal
        ), mock.patch.object(
            sup, "_artifact_from_path", side_effect=fake_artifact
        ), mock.patch.object(sup, "_git_authority"), mock.patch.object(
            sup, "_git_stdout", side_effect=lambda _root, *items: items[-1]
        ), mock.patch.object(
            sup,
            "_load_adopted_e1",
            side_effect=sup.SupervisorError("adoption replay failed"),
        ):
            with self.assertRaisesRegex(
                sup.SupervisorError, "adoption replay failed"
            ):
                sup.build_campaign(rejected_args)
        self.assertFalse((build_root / "state-rejected").exists())

        partial_args = argparse.Namespace(**vars(args))
        partial_args.state_root = str(build_root / "state-partial")
        partial_args.output = str(build_root / "state-partial/campaign.json")
        partial_args.adopted_e1 = adoption["path"]
        partial_args.expected_adopted_e1_sha256 = None
        partial_args.expected_adopted_e1_bytes = adoption["bytes"]
        with mock.patch.object(
            sup, "_freeze_control_source", return_value=control
        ), mock.patch.object(
            sup, "_freeze_runtime_validation_source", return_value=runtime
        ), mock.patch.object(
            sup, "_capture_formal_python_binding", return_value=formal
        ), mock.patch.object(
            sup, "_artifact_from_path", side_effect=fake_artifact
        ), mock.patch.object(sup, "_git_authority"), mock.patch.object(
            sup, "_git_stdout", side_effect=lambda _root, *items: items[-1]
        ):
            with self.assertRaisesRegex(
                sup.SupervisorError, "artifact triple must be all-or-none"
            ):
                sup.build_campaign(partial_args)

    def test_repair_hash_abi_is_distinct_and_fail_closed(self) -> None:
        payload = {
            "format": sup.ADOPTED_E1_FORMAT,
            "status": "complete",
            "candidate_epoch": 1,
        }
        repair_value = dict(payload)
        repair_value["receipt_payload_sha256"] = sup._bridge_payload_sha(
            repair_value
        )
        self.assertNotEqual(
            repair_value["receipt_payload_sha256"],
            sup.payload_sha256(payload),
        )
        self.assertEqual(
            sup._repair_self_hashed(
                repair_value, "receipt_payload_sha256", "repair fixture"
            ),
            repair_value,
        )
        with self.assertRaisesRegex(sup.SupervisorError, "self-hash changed"):
            sup._self_hashed(
                repair_value, "receipt_payload_sha256", "wrong ABI fixture"
            )

    def test_recovered_e1_schema_uses_unambiguous_total_and_recovery_counts(self) -> None:
        value = {
            key: None
            for key in sup.RECOVERED_E1_KEYS
        }
        value.update({
            "format": sup.RECOVERED_E1_FORMAT,
            "original_metric_replays": 1,
            "recovery_evaluator_invocations": 0,
            "recovery_inference_runs": 0,
            "recovery_metric_replays": 0,
            "metric_replays_total": 1,
        })
        self.assertTrue(sup._adopted_e1_schema(value))
        self.assertNotIn("evaluator_invocations", sup.RECOVERED_E1_KEYS)
        self.assertNotIn("inference_reruns", sup.RECOVERED_E1_KEYS)
        self.assertNotIn("metric_replays", sup.RECOVERED_E1_KEYS)
        missing = dict(value)
        missing.pop("recovery_metric_replays")
        self.assertFalse(sup._adopted_e1_schema(missing))

    def test_load_adopted_e1_dispatches_recovery_format_before_legacy_chain(self) -> None:
        campaign = self.fx.campaign()
        artifact = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        campaign["_adopted_e1"] = artifact
        value = {key: None for key in sup.RECOVERED_E1_KEYS}
        value["format"] = sup.RECOVERED_E1_FORMAT
        expected = (artifact, {"status": "recovered"})
        with mock.patch.object(
            sup, "_read_existing_json", return_value=(artifact, value)
        ), mock.patch.object(
            sup, "_load_recovered_e1", return_value=expected
        ) as loader, mock.patch.object(
            sup, "_load_metric_repair_chain"
        ) as legacy:
            self.assertEqual(sup._load_adopted_e1(campaign), expected)
        loader.assert_called_once()
        legacy.assert_not_called()

    def test_incident_snapshots_require_exact_two_file_inventories(self) -> None:
        repair_root = self.fx.root / "incident-repair"
        repair_root.mkdir(mode=0o700)
        report_raw = b"report\n"
        evaluator_raw = b"evaluator\n"
        report = repair_root / "diffsheg-val-fgd.frozen-4066.json"
        evaluator = repair_root / "evaluator.log"
        report.write_bytes(report_raw)
        evaluator.write_bytes(evaluator_raw)
        report.chmod(0o600)
        evaluator.chmod(0o400)
        entries = [
            {
                "name": "diffsheg-val-fgd.frozen-4066.json",
                "bytes": len(report_raw),
                "sha256": hashlib.sha256(report_raw).hexdigest(),
                "mode": 0o600,
                "type": "regular",
            },
            {
                "name": "evaluator.log",
                "bytes": len(evaluator_raw),
                "sha256": hashlib.sha256(evaluator_raw).hexdigest(),
                "mode": 0o400,
                "type": "regular",
            },
        ]
        snapshot = {
            "root": str(repair_root), "root_mode": 0o700,
            "entry_count": 2,
            "total_bytes": len(report_raw) + len(evaluator_raw),
            "entries": entries,
            "inventory_sha256": hashlib.sha256(json.dumps(
                entries, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
        }
        with mock.patch.multiple(
            sup,
            METRIC_REPAIR_INCIDENT_REPORT_SHA256=entries[0]["sha256"],
            METRIC_REPAIR_INCIDENT_REPORT_BYTES=len(report_raw),
            METRIC_REPAIR_INCIDENT_EVALUATOR_LOG_SHA256=entries[1]["sha256"],
            METRIC_REPAIR_INCIDENT_EVALUATOR_LOG_BYTES=len(evaluator_raw),
        ):
            self.assertEqual(
                sup._metric_repair_incident_snapshot(
                    repair_root, snapshot, "incident fixture"
                ),
                snapshot,
            )
            (repair_root / "forbidden-result.json").write_bytes(b"{}\n")
            with self.assertRaisesRegex(
                sup.SupervisorError, "inventory changed"
            ):
                sup._metric_repair_incident_snapshot(
                    repair_root, snapshot, "incident fixture"
                )

    def test_recovery_chain_has_no_evaluator_invocation_surface(self) -> None:
        tree = ast.parse(Path(sup.__file__).read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_load_metric_repair_recovery_chain"
        )
        literals = {
            node.value for node in ast.walk(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("run-recovery", literals)
        self.assertNotIn("--pred-dir", literals)
        self.assertNotIn("evaluator_argv", literals)
        self.assertIn(
            "recovery_evaluator_invocations_authorized",
            sup.METRIC_REPAIR_RECOVERY_SPEC_KEYS,
        )
        self.assertIn(
            "incident_runner_control_snapshot_after",
            sup.METRIC_REPAIR_RECOVERY_RESULT_KEYS,
        )

    def test_finalize_revalidates_adoption_before_consuming_state(self) -> None:
        campaign = self.fx.campaign()
        campaign["_adopted_e1"] = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        with mock.patch.object(
            sup, "_load_adopted_e1",
            side_effect=sup.SupervisorError("terminal predecessor changed"),
        ), mock.patch.object(sup, "_campaign_claim_v2") as campaign_claim:
            with self.assertRaisesRegex(
                sup.SupervisorError, "terminal predecessor changed"
            ):
                sup.finalize_campaign(campaign)
        campaign_claim.assert_not_called()

    def test_dual_dispatch_state_is_allowlisted_only_as_a_canonical_directory(self) -> None:
        campaign = self.fx.campaign()
        dual = self.fx.state / sup.DUAL_DISPATCH_DIR_NAME
        dual.mkdir()
        completed, head = sup._scan_v2(campaign)
        self.assertEqual(completed, [])
        self.assertEqual(head["epoch"], 1)
        dual.rmdir()
        target = self.fx.root / "redirected-dual"
        target.mkdir()
        dual.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(
            sup.SupervisorError, "not a canonical directory"
        ):
            sup._scan_v2(campaign)

    def test_legacy_run_next_refuses_dispatcher_state_before_claiming_campaign(self) -> None:
        campaign = self.fx.campaign()
        dual = self.fx.state / sup.DUAL_DISPATCH_DIR_NAME
        dual.mkdir()
        (dual / sup.DUAL_DISPATCH_WAVES_NAME).mkdir()
        with mock.patch.object(sup, "_campaign_claim_v2") as campaign_claim:
            with self.assertRaisesRegex(
                sup.SupervisorError, "legacy run-next is disabled"
            ):
                sup.run_next(campaign, clock=lambda: 1.0)
        campaign_claim.assert_not_called()

    def test_finalize_refuses_active_dual_wave_before_campaign_claim(self) -> None:
        campaign = self.fx.campaign()
        campaign["_adopted_e1"] = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        dual = self.fx.state / sup.DUAL_DISPATCH_DIR_NAME
        dual.mkdir()
        for name in (
            sup.DUAL_DISPATCH_TOPOLOGY_NAME,
            sup.DUAL_DISPATCH_CLAIM_NAME,
            sup.DUAL_DISPATCH_ACTIVE_NAME,
        ):
            write_bytes(dual / name, b"evidence\n")
        (dual / sup.DUAL_DISPATCH_WAVES_NAME).mkdir()
        adopted = (campaign["_adopted_e1"], {"candidate_epoch": 1})
        with mock.patch.object(
            sup, "_load_adopted_e1", return_value=adopted
        ), mock.patch.object(sup, "_campaign_claim_v2") as campaign_claim:
            with self.assertRaisesRegex(
                sup.SupervisorError, "dual-dispatch wave is active"
            ):
                sup.finalize_campaign(campaign)
        campaign_claim.assert_not_called()

    def test_adopted_finalize_refuses_missing_dual_ledger_without_writing(self) -> None:
        campaign = self.fx.campaign()
        campaign["_adopted_e1"] = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        adopted = (campaign["_adopted_e1"], {"candidate_epoch": 1})
        with mock.patch.object(
            sup, "_load_adopted_e1", return_value=adopted
        ), mock.patch.object(sup, "_campaign_claim_v2") as campaign_claim:
            with self.assertRaisesRegex(
                sup.SupervisorError, "complete dual-dispatch ledger"
            ):
                sup.finalize_campaign(campaign)
        campaign_claim.assert_not_called()

    def test_finalize_refuses_incomplete_dual_wave_coverage_without_writing(self) -> None:
        campaign = self.fx.campaign()
        campaign["_adopted_e1"] = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        dual = self.fx.state / sup.DUAL_DISPATCH_DIR_NAME
        dual.mkdir()
        write_bytes(dual / sup.DUAL_DISPATCH_TOPOLOGY_NAME, b"topology\n")
        write_bytes(dual / sup.DUAL_DISPATCH_CLAIM_NAME, b"claim\n")
        waves = dual / sup.DUAL_DISPATCH_WAVES_NAME
        waves.mkdir()
        for index in range(1, 11):
            (waves / ("wave-%04d" % index)).mkdir()
        adopted = (campaign["_adopted_e1"], {"candidate_epoch": 1})
        with mock.patch.object(
            sup, "_load_adopted_e1", return_value=adopted
        ), mock.patch.object(sup, "_campaign_claim_v2") as campaign_claim:
            with self.assertRaisesRegex(
                sup.SupervisorError, "exactly 11 committed waves"
            ):
                sup.finalize_campaign(campaign)
        campaign_claim.assert_not_called()

    def _complete_dual_fixture(self, campaign: dict):
        campaign["_adopted_e1"] = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        dual = self.fx.state / sup.DUAL_DISPATCH_DIR_NAME
        dual.mkdir()
        topology = write_bytes(
            dual / sup.DUAL_DISPATCH_TOPOLOGY_NAME, b"topology\n"
        )
        claim = write_bytes(
            dual / sup.DUAL_DISPATCH_CLAIM_NAME, b"claim\n"
        )
        waves = dual / sup.DUAL_DISPATCH_WAVES_NAME
        waves.mkdir()
        for index in range(1, 12):
            (waves / ("wave-%04d" % index)).mkdir()
        dispatcher = mock.Mock()
        dispatcher.DUAL_DIR_NAME = sup.DUAL_DISPATCH_DIR_NAME
        dispatcher.TOPOLOGY_NAME = sup.DUAL_DISPATCH_TOPOLOGY_NAME
        dispatcher.CLAIM_NAME = sup.DUAL_DISPATCH_CLAIM_NAME
        dispatcher.ACTIVE_LOCK_NAME = sup.DUAL_DISPATCH_ACTIVE_NAME
        dispatcher.WAVES_DIR_NAME = sup.DUAL_DISPATCH_WAVES_NAME
        fresh = list(sup.CANDIDATE_EPOCHS[1:])
        dispatcher.fresh_waves.return_value = [
            fresh[index:index + 2] for index in range(0, 21, 2)
        ]
        dispatcher.load_topology.side_effect = (
            lambda _campaign, artifact: (artifact, {"complete": True})
        )
        dispatcher.load_or_create_dispatcher_claim.return_value = (
            claim, {"complete": True}
        )
        return dispatcher, topology, claim

    def test_finalize_accepts_exact_11_wave_21_fresh_dispatcher_ledger(self) -> None:
        campaign = self.fx.campaign()
        dispatcher, topology, claim = self._complete_dual_fixture(campaign)
        adopted = (campaign["_adopted_e1"], {"candidate_epoch": 1})
        expected = {"status": "finalized-after-dual-replay"}
        completed = [({}, {}) for _ in sup.CANDIDATE_EPOCHS]
        with mock.patch.object(
            sup, "_load_adopted_e1", return_value=adopted
        ), mock.patch.object(
            sup, "_load_dual_dispatcher_module", return_value=dispatcher
        ), mock.patch.object(
            sup, "_campaign_claim_v2"
        ), mock.patch.object(
            sup, "_scan_v2", return_value=(completed, None)
        ), mock.patch.object(
            sup, "_finalize_v2", return_value=expected
        ):
            self.assertEqual(sup.finalize_campaign(campaign), expected)
        dispatcher.load_topology.assert_called_once_with(campaign, topology)
        dispatcher.load_or_create_dispatcher_claim.assert_called_once_with(
            campaign, topology
        )
        dispatcher._validate_prior_waves.assert_called_once_with(
            campaign, claim, topology, {"complete": True}, 11
        )

    def test_finalize_maps_dispatcher_wave_tamper_to_fail_closed_error(self) -> None:
        campaign = self.fx.campaign()
        dispatcher, _topology, _claim = self._complete_dual_fixture(campaign)
        dispatcher._validate_prior_waves.side_effect = RuntimeError(
            "commit payload changed"
        )
        adopted = (campaign["_adopted_e1"], {"candidate_epoch": 1})
        with mock.patch.object(
            sup, "_load_adopted_e1", return_value=adopted
        ), mock.patch.object(
            sup, "_load_dual_dispatcher_module", return_value=dispatcher
        ), mock.patch.object(sup, "_campaign_claim_v2") as campaign_claim:
            with self.assertRaisesRegex(
                sup.SupervisorError,
                "finalization evidence is invalid: commit payload changed",
            ):
                sup.finalize_campaign(campaign)
        campaign_claim.assert_not_called()

    def test_no_receipt_returns_waiting_without_job_or_active_claim(self) -> None:
        campaign = self.fx.campaign()
        result = sup.run_next(campaign, clock=lambda: 1.0)
        self.assertEqual(result["status"], "waiting_for_candidate_receipt")
        self.assertFalse(result["job_claim_created"])
        self.assertFalse((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertFalse((self.fx.state / "active_invocation.claim.json").exists())

    def test_adopted_e1_advances_to_e2_without_successor_authority(self) -> None:
        campaign = self.fx.campaign()
        campaign["_adopted_e1"] = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        adopted = (
            campaign["_adopted_e1"],
            {
                "format": sup.ADOPTED_E1_FORMAT,
                "status": "complete",
                "candidate_epoch": 1,
                "measurement": {
                    "path": str(self.fx.root / "measurement.json"),
                    "sha256": "b" * 64,
                    "bytes": 1,
                    "receipt_payload_sha256": "c" * 64,
                },
                "validation_diffsheg_fgd": 0.25,
            },
        )
        with mock.patch.object(sup, "_load_adopted_e1", return_value=adopted):
            completed, head = sup._scan_v2(campaign)
        self.assertEqual(len(completed), 1)
        self.assertEqual(head["epoch"], 2)
        Path(self.fx.jobs[0]["authority_path"]).write_text("forbidden\n")
        with mock.patch.object(sup, "_load_adopted_e1", return_value=adopted):
            with self.assertRaisesRegex(
                sup.SupervisorError, "must not contain successor execution output"
            ):
                sup._scan_v2(campaign)

    def test_adopted_reconcile_uses_pinned_old_e1_authority(self) -> None:
        campaign = self.fx.campaign()
        receipt_artifact = {
            "path": str(self.fx.root / "adoption.json"),
            "sha256": "a" * 64,
            "bytes": 1,
        }
        campaign["_adopted_e1"] = receipt_artifact
        receipt = {
            key: None
            for key in sup.ADOPTED_E1_KEYS
        }
        receipt.update({
            "format": sup.ADOPTED_E1_FORMAT,
            "predecessor_work_authority": (
                sup.ADOPTION_PREDECESSOR_FIXED["predecessor_work_authority"]
            ),
        })
        with mock.patch.object(
            sup, "_read_existing_json", return_value=(receipt_artifact, receipt)
        ), mock.patch.object(
            sup,
            "_artifact",
            return_value=sup.ADOPTION_PREDECESSOR_FIXED[
                "predecessor_work_authority"
            ],
        ):
            argv = sup._authority_reconcile_argv(
                campaign,
                {"path": "/manifest", "sha256": "b" * 64, "bytes": 1},
                {"path": "/status", "sha256": "c" * 64, "bytes": 1},
            )
        authority = sup.ADOPTION_PREDECESSOR_FIXED[
            "predecessor_work_authority"
        ]
        self.assertEqual(
            argv[argv.index("--adopted-e1-authority") + 1],
            authority["path"],
        )
        self.assertEqual(
            argv[
                argv.index("--expected-adopted-e1-authority-sha256") + 1
            ],
            authority["sha256"],
        )
        self.assertEqual(
            argv[argv.index("--expected-adopted-e1-authority-bytes") + 1],
            str(authority["bytes"]),
        )

    def test_adopted_authority_must_match_successor_campaign_inputs(self) -> None:
        config = self.fx.adapter_config
        value = {
            "selected_topology": {
                "mode": config["topology_mode"], "world_size": 8,
            },
            "schedule": dict(config["schedule"]),
            "frozen_inputs": {
                "path": str(self.fx.train / "frozen_inputs.json"),
                "sha256": config["expected_frozen_inputs_sha256"],
            },
            "val_inputs_receipt": {
                "path": config["val_inputs"]["path"],
                "sha256": config["val_inputs"]["sha256"],
            },
            "pipeline_receipt": {
                "path": config["pipeline"]["path"],
                "sha256": config["pipeline"]["sha256"],
            },
            "producer_source": {
                "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                "commit": sup.PRODUCER_SOURCE_COMMIT,
                "tree": sup.PRODUCER_SOURCE_TREE,
                "entrypoint_sha256": config[
                    "expected_producer_trainer_sha256"
                ],
                "node_local_clones": [{
                    "entrypoint": str(
                        Path(config["producer_source_root"])
                        / "scripts/show_base/train_base_official_adapt_long.py"
                    ),
                }],
            },
            "candidate_checkpoint": {
                "path": str(
                    self.fx.train
                    / "candidates/base_official_adapt_epoch_01.bin"
                ),
            },
        }
        artifact = write_json(self.fx.root / "old-authority.json", value)
        sup._work_authority_campaign_binding(artifact, config, 1)
        value["selected_topology"]["mode"] = (
            "validation_gated_w8_l256_g2048_empirical_acceleration"
        )
        tampered = write_json(self.fx.root / "old-authority-bad.json", value)
        with self.assertRaisesRegex(
            sup.SupervisorError, "topology differs"
        ):
            sup._work_authority_campaign_binding(tampered, config, 1)

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

    def test_plain_run_next_refuses_occupied_global_e1_claim_before_private_claim(self) -> None:
        campaign = self.fx.campaign()
        write_json(Path(self.fx.jobs[0]["candidate_receipt_path"]), {"epoch": 1})
        original = self.fx.train / "live_val_consumer_claims/epoch-0001.json"
        write_json(original, {"occupied": True})
        with self.assertRaisesRegex(sup.SupervisorError, "recover-e1"):
            sup.run_next(
                campaign,
                capture=lambda _argv: self.fail("plain run-next executed work"),
                clock=lambda: 1.0,
            )
        self.assertFalse((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertFalse((self.fx.state / "active_invocation.claim.json").exists())
        self.assertFalse(Path(self.fx.jobs[0]["authority_path"]).exists())

    def test_recovery_read_only_admission_failure_leaves_zero_new_state(self) -> None:
        campaign = self.fx.campaign()
        write_json(Path(self.fx.jobs[0]["candidate_receipt_path"]), {"epoch": 1})
        write_json(
            self.fx.train / "live_val_consumer_claims/epoch-0001.json",
            {"occupied": True},
        )
        before_state = sorted(
            path.relative_to(self.fx.state).as_posix()
            for path in self.fx.state.rglob("*")
        )
        failed_root = self.fx.root / "failed-run-admission"
        fixture_pins = write_failed_run_fixture(failed_root)
        write_bytes(failed_root / "work-preflight.json", b"tampered\n")

        def reject_tampered_inventory(*_args, **_kwargs):
            sup._failed_run_inventory(failed_root)
            self.fail("tampered non-shard preflight unexpectedly passed")

        with mock.patch.object(sup, "FAILED_RUN_FILE_PINS", fixture_pins), \
             mock.patch.object(sup, "_load_failed_campaign", return_value={}), \
             mock.patch.object(sup, "_revalidate_control"), \
             mock.patch.object(sup, "_runtime_contract_check"), \
             mock.patch.object(
                 sup, "_recovery_request_value",
                 side_effect=reject_tampered_inventory,
             ):
            with self.assertRaisesRegex(sup.SupervisorError, "SHA/byte pins"):
                sup.recover_e1(
                    campaign, "/failed-campaign.json",
                    sup.FAILED_CAMPAIGN_SHA256, sup.FAILED_CAMPAIGN_BYTES,
                    runner=lambda _argv: self.fail("runner was invoked"),
                    capture=lambda _argv: self.fail("unexpected capture"),
                    clock=lambda: 1.0,
                )
        after_state = sorted(
            path.relative_to(self.fx.state).as_posix()
            for path in self.fx.state.rglob("*")
        )
        self.assertEqual(after_state, before_state)
        self.assertFalse(Path(campaign["campaign_claim_path"]).exists())
        self.assertFalse((self.fx.state / "active_invocation.claim.json").exists())
        self.assertFalse((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertFalse(Path(self.fx.jobs[0]["authority_path"]).exists())
        self.assertFalse(
            (self.fx.train / "live_val_consumer_recovery_claims").exists()
        )

    def test_recovery_reserves_once_before_runner_and_failure_is_terminal(self) -> None:
        campaign = self.fx.campaign()
        job = self.fx.jobs[0]
        write_json(Path(job["candidate_receipt_path"]), {"epoch": 1})
        write_json(
            self.fx.train / "live_val_consumer_claims/epoch-0001.json",
            {"occupied": True},
        )
        preview_raw = sup.canonical_json_bytes({
            "candidate_epoch": 1,
            "published_unix": 1.0,
            "receipt_payload_sha256": "a" * 64,
        })
        authority_preview = {
            "path": job["authority_path"],
            "sha256": hashlib.sha256(preview_raw).hexdigest(),
            "bytes": len(preview_raw),
        }
        authority_raw = sup.canonical_json_bytes({
            "candidate_epoch": 1,
            "published_unix": 2.0,
            "receipt_payload_sha256": "b" * 64,
        })
        actual_authority = {
            "path": job["authority_path"],
            "sha256": hashlib.sha256(authority_raw).hexdigest(),
            "bytes": len(authority_raw),
        }
        request_value = {
            "format": sup.RECOVERY_REQUEST_FORMAT,
            "status": "ready_for_single_recovery",
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "candidate_epoch": 1,
            "failed_campaign": {"path": "/failed/campaign", "sha256": "1" * 64, "bytes": 1},
            "failed_job_claim": {"path": "/failed/job", "sha256": "2" * 64, "bytes": 2},
            "failed_active_claim": {"path": "/failed/active", "sha256": "3" * 64, "bytes": 3},
            "failed_authorization": {"path": "/failed/auth", "sha256": "4" * 64, "bytes": 4},
            "failed_work_authority": {"path": "/failed/work", "sha256": "5" * 64, "bytes": 5},
            "failed_consumer_claim": {"path": "/failed/consumer", "sha256": "6" * 64, "bytes": 6},
            "failed_runner_status": {"path": "/failed/status", "sha256": "7" * 64, "bytes": 7},
            "failed_runner_log": {"path": "/failed/log", "sha256": "8" * 64, "bytes": 8},
            "failed_run_root": "/failed/run",
            "failed_run_inventory": {},
            "failed_process_proof": {},
            "guard_proof": {},
            "new_campaign": {"path": "/new/campaign", "sha256": "9" * 64, "bytes": 9},
            "new_control_source": {},
            "new_work_authority": authority_preview,
            "new_run_root": job["run_root"],
            "created_unix": 1.0,
        }
        request_value = sup._add_bridge_self_hash(
            request_value, "receipt_payload_sha256"
        )
        recovery_authority_value = {"recovery": "authority"}
        recovery_authority_raw = sup.canonical_json_bytes(
            recovery_authority_value
        )
        recovery_authority_preview = {
            "path": str(self.fx.state / "recovery-authority.epoch-0001.json"),
            "sha256": hashlib.sha256(recovery_authority_raw).hexdigest(),
            "bytes": len(recovery_authority_raw),
        }
        recovery_claim_path = (
            self.fx.train
            / "live_val_consumer_recovery_claims/epoch-0001.json"
        )
        events = []

        def capture(argv):
            if "authorize" in argv:
                events.append("authorize")
                Path(job["authority_path"]).write_bytes(authority_raw)
                stdout = (
                    json.dumps(actual_authority, sort_keys=True) + "\n"
                ).encode()
                return 0, stdout, b""
            if "inspect-recovery" in argv:
                events.append("inspect")
                request_path = Path(argv[argv.index("--request") + 1])
                rebound_request = json.loads(request_path.read_text())
                self.assertEqual(
                    rebound_request["new_work_authority"], actual_authority
                )
                self.assertEqual(
                    rebound_request["receipt_payload_sha256"],
                    sup._bridge_payload_sha(rebound_request),
                )
                value = {
                    "status": "ready",
                    "recovery_authority": recovery_authority_preview,
                    "recovery_claim_path": str(recovery_claim_path),
                }
                return 0, (json.dumps(value, sort_keys=True) + "\n").encode(), b""
            if "reserve-recovery" in argv:
                events.append("reserve")
                recovery_authority = write_json(
                    Path(recovery_authority_preview["path"]),
                    recovery_authority_value,
                )
                recovery_claim = write_json(
                    recovery_claim_path, {"recovery": "claim"}
                )
                value = {
                    "status": "reserved",
                    "recovery_authority": recovery_authority,
                    "recovery_claim": recovery_claim,
                }
                return 0, (json.dumps(value, sort_keys=True) + "\n").encode(), b""
            self.fail("unexpected captured command: %r" % (argv,))

        def runner(_argv):
            events.append("runner")
            return 1

        patches = (
            mock.patch.object(sup, "_load_failed_campaign", return_value={}),
            mock.patch.object(sup, "_revalidate_control"),
            mock.patch.object(sup, "_runtime_contract_check"),
            mock.patch.object(
                sup, "_recovery_request_value",
                return_value=(request_value, authority_preview),
            ),
            mock.patch.object(sup, "_work_authority_runtime_binding"),
            mock.patch.object(sup, "_work_authority_candidate_binding"),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with self.assertRaisesRegex(sup.SupervisorError, "runner returned nonzero"):
                sup.recover_e1(
                    campaign, "/failed-campaign.json",
                    sup.FAILED_CAMPAIGN_SHA256, sup.FAILED_CAMPAIGN_BYTES,
                    runner=runner, capture=capture, clock=lambda: 1.0,
                )
        self.assertEqual(events, ["authorize", "inspect", "reserve", "runner"])
        self.assertTrue(recovery_claim_path.exists())
        self.assertTrue(Path(recovery_authority_preview["path"]).exists())
        self.assertTrue((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertTrue((self.fx.state / "active_invocation.claim.json").exists())
        events.clear()
        with mock.patch.object(sup, "_load_failed_campaign", return_value={}):
            with self.assertRaisesRegex(sup.SupervisorError, "claim exists without completion"):
                sup.recover_e1(
                    campaign, "/failed-campaign.json",
                    sup.FAILED_CAMPAIGN_SHA256, sup.FAILED_CAMPAIGN_BYTES,
                    runner=lambda _argv: self.fail("recovery runner was retried"),
                    capture=lambda _argv: self.fail("recovery command was retried"),
                    clock=lambda: 2.0,
                )
        self.assertEqual(events, [])

    def test_recovery_authorize_failure_is_private_terminal_without_global_slot(self) -> None:
        campaign = self.fx.campaign()
        job = self.fx.jobs[0]
        write_json(Path(job["candidate_receipt_path"]), {"epoch": 1})
        original_claim = write_json(
            self.fx.train / "live_val_consumer_claims/epoch-0001.json",
            {"occupied": True},
        )
        original_raw = Path(original_claim["path"]).read_bytes()
        authority_raw = sup.canonical_json_bytes({"candidate_epoch": 1})
        authority_preview = {
            "path": job["authority_path"],
            "sha256": hashlib.sha256(authority_raw).hexdigest(),
            "bytes": len(authority_raw),
        }
        calls = []

        def capture(argv):
            calls.append(list(argv))
            self.assertIn("authorize", argv)
            return 9, b"", b"authorize failed"

        with mock.patch.object(sup, "_load_failed_campaign", return_value={}), \
             mock.patch.object(sup, "_revalidate_control"), \
             mock.patch.object(sup, "_runtime_contract_check"), \
             mock.patch.object(
                 sup, "_recovery_request_value",
                 return_value=(
                     {"new_work_authority": authority_preview},
                     authority_preview,
                 ),
             ):
            with self.assertRaisesRegex(
                sup.SupervisorError, "authorize failed"
            ):
                sup.recover_e1(
                    campaign, "/failed-campaign.json",
                    sup.FAILED_CAMPAIGN_SHA256, sup.FAILED_CAMPAIGN_BYTES,
                    runner=lambda _argv: self.fail("runner was invoked"),
                    capture=capture, clock=lambda: 1.0,
                )
        self.assertEqual(len(calls), 1)
        self.assertTrue(Path(campaign["campaign_claim_path"]).exists())
        self.assertTrue((self.fx.state / "active_invocation.claim.json").exists())
        self.assertTrue((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertFalse(
            (self.fx.train / "live_val_consumer_recovery_claims").exists()
        )
        self.assertEqual(Path(original_claim["path"]).read_bytes(), original_raw)
        with mock.patch.object(sup, "_load_failed_campaign", return_value={}):
            with self.assertRaisesRegex(
                sup.SupervisorError, "claim exists without completion"
            ):
                sup.recover_e1(
                    campaign, "/failed-campaign.json",
                    sup.FAILED_CAMPAIGN_SHA256, sup.FAILED_CAMPAIGN_BYTES,
                    runner=lambda _argv: self.fail("runner was retried"),
                    capture=lambda _argv: self.fail("capture was retried"),
                    clock=lambda: 2.0,
                )

    def test_recovery_inspect_failure_is_private_terminal_without_global_slot(self) -> None:
        campaign = self.fx.campaign()
        job = self.fx.jobs[0]
        write_json(Path(job["candidate_receipt_path"]), {"epoch": 1})
        original_claim = write_json(
            self.fx.train / "live_val_consumer_claims/epoch-0001.json",
            {"occupied": True},
        )
        original_raw = Path(original_claim["path"]).read_bytes()
        authority_raw = sup.canonical_json_bytes({"candidate_epoch": 1})
        authority_preview = {
            "path": job["authority_path"],
            "sha256": hashlib.sha256(authority_raw).hexdigest(),
            "bytes": len(authority_raw),
        }
        request_value = {
            key: None
            for key in sup.RECOVERY_REQUEST_KEYS
            if key != "receipt_payload_sha256"
        }
        request_value["new_work_authority"] = authority_preview
        request_value = sup._add_bridge_self_hash(
            request_value, "receipt_payload_sha256"
        )
        events = []

        def capture(argv):
            if "authorize" in argv:
                events.append("authorize")
                Path(job["authority_path"]).write_bytes(authority_raw)
                stdout = (
                    json.dumps(authority_preview, sort_keys=True) + "\n"
                ).encode()
                return 0, stdout, b""
            if "inspect-recovery" in argv:
                events.append("inspect")
                return 7, b"", b"inspect failed"
            self.fail("unexpected captured command: %r" % (argv,))

        patches = (
            mock.patch.object(sup, "_load_failed_campaign", return_value={}),
            mock.patch.object(sup, "_revalidate_control"),
            mock.patch.object(sup, "_runtime_contract_check"),
            mock.patch.object(
                sup, "_recovery_request_value",
                return_value=(request_value, authority_preview),
            ),
            mock.patch.object(sup, "_work_authority_runtime_binding"),
            mock.patch.object(sup, "_work_authority_candidate_binding"),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with self.assertRaisesRegex(sup.SupervisorError, "inspect-recovery"):
                sup.recover_e1(
                    campaign, "/failed-campaign.json",
                    sup.FAILED_CAMPAIGN_SHA256, sup.FAILED_CAMPAIGN_BYTES,
                    runner=lambda _argv: self.fail("runner was invoked"),
                    capture=capture, clock=lambda: 1.0,
                )
        self.assertEqual(events, ["authorize", "inspect"])
        self.assertTrue(Path(campaign["campaign_claim_path"]).exists())
        self.assertTrue((self.fx.state / "active_invocation.claim.json").exists())
        self.assertTrue((self.fx.state / "job_claims/epoch-0001.json").exists())
        self.assertTrue((self.fx.state / "recovery-request.epoch-0001.json").exists())
        self.assertFalse(
            (self.fx.train / "live_val_consumer_recovery_claims").exists()
        )
        self.assertFalse(
            (self.fx.state / "recovery-authority.epoch-0001.json").exists()
        )
        self.assertEqual(Path(original_claim["path"]).read_bytes(), original_raw)
        events.clear()
        with mock.patch.object(sup, "_load_failed_campaign", return_value={}):
            with self.assertRaisesRegex(
                sup.SupervisorError, "claim exists without completion"
            ):
                sup.recover_e1(
                    campaign, "/failed-campaign.json",
                    sup.FAILED_CAMPAIGN_SHA256, sup.FAILED_CAMPAIGN_BYTES,
                    runner=lambda _argv: self.fail("runner was retried"),
                    capture=lambda _argv: self.fail("capture was retried"),
                    clock=lambda: 2.0,
                )
        self.assertEqual(events, [])

    def test_unrelated_7d_incident_is_not_an_admissible_failed_campaign(self) -> None:
        raw = sup.canonical_json_bytes({
            "control_source": {"supervisor": {"path": "/old/supervisor.py"}},
            "formal_python": {"argv0": "/old/venv/bin/python"},
        })
        failed_7d = {"_control_source": {
            "commit": "7d9967c9c5124d6f1ba041c2cb315dee88554a76",
            "tree": "a965cf1ccff214bd8922692603b33832b7357543",
        }}
        with mock.patch.object(
            sup, "safe_regular_bytes",
            return_value=(Path("/old/campaign.json"), raw,
                          sup.FAILED_CAMPAIGN_SHA256,
                          sup.FAILED_CAMPAIGN_BYTES),
        ), mock.patch.object(sup, "load_campaign", return_value=failed_7d):
            with self.assertRaisesRegex(sup.SupervisorError, "pinned 58407ed"):
                sup._load_failed_campaign(
                    "/old/campaign.json", sup.FAILED_CAMPAIGN_SHA256,
                    sup.FAILED_CAMPAIGN_BYTES,
                )

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

    def test_completion_binds_recovery_claim_and_normal_completion_is_explicit_null(self) -> None:
        campaign = self.fx.campaign()
        base_job = {
            **self.fx.jobs[0],
            "candidate_receipt": {"path": "/candidate", "sha256": "1" * 64, "bytes": 1},
            "work_authority": {"path": "/authority", "sha256": "2" * 64, "bytes": 1},
            "authorization": {"path": "/authorization", "sha256": "3" * 64, "bytes": 1},
        }
        status_artifact = {"path": "/status", "sha256": "4" * 64, "bytes": 1}
        log_artifact = {"path": "/log", "sha256": "5" * 64, "bytes": 1}
        measurement = {"path": "/measurement", "sha256": "6" * 64,
                       "bytes": 1, "receipt_payload_sha256": "7" * 64}
        guards = {str(i): 100 + i for i in range(8)}
        normal = sup._completion_body(
            campaign, base_job, status_artifact, {}, log_artifact,
            measurement, {"metrics": {"fgd": 0.25}}, ["bridge"],
            ["verify"], "PASS\n", guards, 1.0,
        )
        self.assertIsNone(normal["consumer_recovery"])
        recovery = {
            label: write_json(self.fx.root / ("bound-%s.json" % label), {label: 1})
            for label in ("request", "authority", "claim")
        }
        recovered_job = {**base_job, "consumer_recovery": recovery}
        recovered_value = {
            "metrics": {"fgd": 0.25},
            "consumer_claim": {**recovery["claim"], "receipt_payload_sha256": "8" * 64},
        }
        completion = sup._completion_body(
            campaign, recovered_job, status_artifact, {}, log_artifact,
            measurement, recovered_value, ["bridge"], ["verify"],
            "PASS\n", guards, 1.0,
        )
        self.assertEqual(completion["consumer_recovery"], recovery)
        recovered_value["consumer_claim"] = {
            **recovered_value["consumer_claim"], "sha256": "9" * 64,
        }
        with self.assertRaisesRegex(sup.SupervisorError, "reserved recovery claim"):
            sup._completion_body(
                campaign, recovered_job, status_artifact, {}, log_artifact,
                measurement, recovered_value, ["bridge"], ["verify"],
                "PASS\n", guards, 1.0,
            )

    def test_guard_verifier_exact_line_and_pid_order(self) -> None:
        campaign = {"_formal_python": self.fx.formal, "_guard_verifier": self.fx.verifier}
        status = {"restored_guards": {str(i): 300 + i for i in range(8)}}
        def good(_argv):
            return 0, ("PASS " + " ".join("GPU%d=PID%d" % (i, 300 + i) for i in range(8)) + "\n").encode(), b""
        _argv, _text, observed = sup._guard_verify(campaign, status, good)
        self.assertEqual(observed, status["restored_guards"])
        with self.assertRaisesRegex(sup.SupervisorError, "differ"):
            sup._guard_verify(campaign, status, lambda _argv: (0, ("PASS " + " ".join("GPU%d=PID%d" % (i, 301 if i == 0 else 300 + i) for i in range(8)) + "\n").encode(), b""))

    def test_failed_process_proof_checks_only_the_two_pinned_proc_paths(self) -> None:
        status = {
            "wrapper_pid": sup.FAILED_WRAPPER_PID,
            "child_pid": sup.FAILED_CHILD_PID,
            "command": ["/bin/bash", "/pinned/launcher.sh"],
        }
        observed = []
        def absent(path):
            observed.append(path)
            return False
        with mock.patch.object(sup.os.path, "lexists", side_effect=absent):
            proof = sup._failed_process_proof(status, 1.0)
        self.assertEqual(
            observed,
            ["/proc/%d" % sup.FAILED_WRAPPER_PID,
             "/proc/%d" % sup.FAILED_CHILD_PID],
        )
        self.assertEqual(proof["wrapper_proc_state"], "absent")
        self.assertEqual(proof["child_proc_state"], "absent")
        self.assertEqual(proof["runner_command"], status["command"])
        with mock.patch.object(
            sup.os.path, "lexists",
            side_effect=lambda path: path.endswith(str(sup.FAILED_CHILD_PID)),
        ):
            with self.assertRaisesRegex(sup.SupervisorError, "still live"):
                sup._failed_process_proof(status, 1.0)

    def test_failed_run_inventory_requires_exact_zero_output_tree(self) -> None:
        root = self.fx.root / "failed-run"
        fixture_pins = write_failed_run_fixture(root)
        with mock.patch.object(sup, "FAILED_RUN_FILE_PINS", fixture_pins):
            inventory = sup._failed_run_inventory(root)
            self.assertEqual(inventory["semantic_outputs"], [])
            self.assertEqual(len(inventory["files"]), 13)
            write_bytes(root / "work-preflight.json", b"tampered\n")
            with self.assertRaisesRegex(sup.SupervisorError, "SHA/byte pins"):
                sup._failed_run_inventory(root)
            write_bytes(
                root / "work-preflight.json",
                b"work-preflight.json\n",
            )
            write_bytes(root / "candidates/e1/shards/receipt.json", b"forbidden\n")
            with self.assertRaisesRegex(sup.SupervisorError, "zero-semantic-output"):
                sup._failed_run_inventory(root)

    def test_failed_runner_status_is_exact_rc1_with_pinned_processes(self) -> None:
        campaign = self.fx.campaign()
        authority = write_json(self.fx.root / "failed-authority.json", {"x": 1})
        job = {**self.fx.jobs[0], "work_authority": authority,
               "_campaign": campaign}
        value = {
            "updated_at": "2026-08-02T00:00:00Z", "state": "failed",
            "wrapper_pid": sup.FAILED_WRAPPER_PID,
            "child_pid": sup.FAILED_CHILD_PID, "return_code": 1,
            "received_signal": None, "error": None, "cleanup_error": None,
            "restored_guards": {str(i): 300000 + i for i in range(8)},
            "restore_error": None,
            "command": sup._runner_workload(job, campaign),
        }
        artifact = write_json(Path(job["runner_status_path"]), value)
        sup._failed_runner_status(job, artifact["sha256"])
        value["return_code"] = 2
        write_json(Path(job["runner_status_path"]), value)
        with self.assertRaisesRegex(sup.SupervisorError, "exact rc1"):
            sup._failed_runner_status(job)

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
