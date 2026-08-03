#!/usr/bin/env python3
"""CPU-only focused tests for successor e1 metric-repair adoption."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from scripts.show_base import supervise_base_v14_live_validation as supervisor


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/show_base/adopt_base_v14_metric_repair.py"
)
SPEC = importlib.util.spec_from_file_location("metric_repair_adoption", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def write_bytes(path: Path, raw: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def write_json(path: Path, value: dict) -> dict:
    return write_bytes(path, repair.canonical_bytes(value))


class RepairFixture:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="metric-adoption-")
        self.root = Path(self.temp.name).resolve()
        self.state = self.root / "terminal-state"
        self.run = self.root / "terminal-run"
        self.state.mkdir()
        self.run.mkdir()
        self.frozen = self.root / "frozen-4066"
        self.control = self.root / "control-73ff"
        self.repair_root = self.root / "repair"
        self.control_root = self.root / "new-runner-control"
        self.train_root = self.root / "train"
        self.train_root.mkdir()
        self.formal_python = self.root / "formal/bin/python"
        self.formal_target = self.root / "formal-target/python3.12"
        write_bytes(self.formal_target, b"#!/bin/sh\n")
        self.formal_target.chmod(0o700)
        self.formal_python.parent.mkdir(parents=True)
        os.symlink(str(self.formal_target), self.formal_python.parent / "python3.12")
        os.symlink("python3.12", self.formal_python)
        self.paspa = self.root / "paspa"
        self.diffsheg = self.root / "diffsheg"
        self.paspa.mkdir()
        self.diffsheg.mkdir()
        self.frozen_evaluator = write_bytes(
            self.frozen / "scripts/show_base/evaluate_diffsheg_val_fgd.py",
            b"# exact evaluator\n",
        )
        self.bridge = write_bytes(
            self.control / "scripts/show_base/base_live_val_consumer_bridge.py",
            b"# exact bridge\n",
        )
        self.runner = write_bytes(self.root / "runner.py", b"# runner\n")
        self.guard = write_bytes(self.root / "verify.py", b"# guard\n")
        self.tool = repair.artifact(str(SCRIPT), "repair tool")
        self.repair_source = {
            "origin": repair.OFFICIAL_ORIGIN,
            "source_root": str(SCRIPT.parents[2]),
            "commit": "9" * 40,
            "tree": "a" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        self.old_report_value = {
            "format": "test-report-v1",
            "metrics": {"fgd": 1.25, "count": 17},
            "provenance": {"adapter": {
                "path": str(self.control / "scripts/show_base/evaluate_diffsheg_val_fgd.py"),
                "repository_root": str(self.control),
                "repository_git_head": "7" * 40,
                "unchanged": True,
            }},
            "nested": [1, 2.0, {"ok": False}],
        }
        refs = {
            "job-claim": write_bytes(self.state / "job_claims/epoch-0001.json", b"job-claim\n"),
            "active": write_bytes(self.state / "active_invocation.claim.json", b"active\n"),
            "work-authority": write_bytes(self.state / "authorities/epoch-0001.json", b"work-authority\n"),
            "authorization": write_bytes(self.state / "authorizations/epoch-0001.json", b"authorization\n"),
            "runner-status": write_json(self.state / "runner_status/epoch-0001.json", {
                "state": "failed", "return_code": 1, "error": None,
                "cleanup_error": None, "restore_error": None,
                "received_signal": None,
            }),
            "candidate-receipt": write_bytes(self.train_root / "candidate_receipts/epoch-0001.json", b"candidate-receipt\n"),
            "recovery-claim": write_bytes(self.train_root / "live_val_consumer_recovery_claims/epoch-0001.json", b"recovery\n"),
            "recovery-request": write_bytes(self.state / "recovery-request.epoch-0001.json", b"recovery-request\n"),
            "recovery-authority": write_bytes(self.state / "recovery-authority.epoch-0001.json", b"recovery-authority\n"),
            "preflight": write_bytes(self.run / "work-preflight.json", b"preflight\n"),
            "lineage": write_bytes(self.run / "candidates/e1/final/val-inference-lineage.json", b"lineage\n"),
            "final-manifest": write_bytes(self.run / "candidates/e1/final/final_manifest.jsonl", b"manifest\n"),
            "clip-manifest": write_bytes(self.run / "candidates/e1/final/diffsheg_eval_clip_ids.txt", b"clip\n"),
            "old-report": write_json(self.run / "candidates/e1/diffsheg-val-fgd.json", self.old_report_value),
            "failure-log": write_bytes(self.run / "logs/e1-live-measurement.log", b"prefix\nDiffSHEG adapter provenance is not the frozen pipeline source\n"),
        }
        jobs = []
        for epoch in repair.CANDIDATE_EPOCHS if hasattr(repair, "CANDIDATE_EPOCHS") else supervisor.CANDIDATE_EPOCHS:
            run_root = self.root / "old-runs" / ("e%d" % epoch)
            jobs.append({
                "epoch": epoch,
                "authority_path": str(self.state / "authorities" / ("epoch-%04d.json" % epoch)),
                "authorization_path": str(self.state / "authorizations" / ("epoch-%04d.json" % epoch)),
                "run_root": str(run_root),
                "measurement_path": str(run_root / "live-measurement.json"),
                "completion_path": str(self.state / "completions" / ("epoch-%04d.json" % epoch)),
                "runner_status_path": str(self.state / "runner_status" / ("epoch-%04d.json" % epoch)),
                "runner_log_path": str(self.state / "runner_logs" / ("epoch-%04d.log" % epoch)),
            })
        refs["campaign"] = write_json(self.state / "campaign.json", {"jobs": jobs})
        self.refs = refs
        self.fixed = {
            "predecessor_campaign": (refs["campaign"]["path"], refs["campaign"]["sha256"], refs["campaign"]["bytes"]),
            "predecessor_job_claim": (refs["job-claim"]["path"], refs["job-claim"]["sha256"], refs["job-claim"]["bytes"]),
            "predecessor_active_claim": (refs["active"]["path"], refs["active"]["sha256"], refs["active"]["bytes"]),
            "predecessor_work_authority": (refs["work-authority"]["path"], refs["work-authority"]["sha256"], refs["work-authority"]["bytes"]),
            "predecessor_authorization": (refs["authorization"]["path"], refs["authorization"]["sha256"], refs["authorization"]["bytes"]),
            "predecessor_runner_status": (refs["runner-status"]["path"], refs["runner-status"]["sha256"], refs["runner-status"]["bytes"]),
            "preflight": (refs["preflight"]["path"], refs["preflight"]["sha256"], refs["preflight"]["bytes"]),
            "inference_lineage": (refs["lineage"]["path"], refs["lineage"]["sha256"], refs["lineage"]["bytes"]),
            "final_manifest": (refs["final-manifest"]["path"], refs["final-manifest"]["sha256"], refs["final-manifest"]["bytes"]),
            "clip_manifest": (refs["clip-manifest"]["path"], refs["clip-manifest"]["sha256"], refs["clip-manifest"]["bytes"]),
            "old_report": (refs["old-report"]["path"], refs["old-report"]["sha256"], refs["old-report"]["bytes"]),
            "predecessor_failure_log": (refs["failure-log"]["path"], refs["failure-log"]["sha256"], refs["failure-log"]["bytes"]),
            "candidate_receipt": (refs["candidate-receipt"]["path"], refs["candidate-receipt"]["sha256"], refs["candidate-receipt"]["bytes"]),
            "predecessor_recovery_claim": (refs["recovery-claim"]["path"], refs["recovery-claim"]["sha256"], refs["recovery-claim"]["bytes"]),
            "predecessor_recovery_request": (refs["recovery-request"]["path"], refs["recovery-request"]["sha256"], refs["recovery-request"]["bytes"]),
            "predecessor_recovery_authority": (refs["recovery-authority"]["path"], refs["recovery-authority"]["sha256"], refs["recovery-authority"]["bytes"]),
        }

    def close(self) -> None:
        self.temp.cleanup()

    def patches(self):
        return mock.patch.multiple(
            repair,
            PREDECESSOR_STATE_ROOT=str(self.state),
            PREDECESSOR_RUN_ROOT=str(self.run),
            PREDECESSOR_CAMPAIGN_SHA256=self.refs["campaign"]["sha256"],
            PREDECESSOR_CAMPAIGN_BYTES=self.refs["campaign"]["bytes"],
            FROZEN_ROOT=str(self.frozen),
            FROZEN_COMMIT="4" * 40,
            FROZEN_TREE="5" * 40,
            FROZEN_EVALUATOR_SHA256=self.frozen_evaluator["sha256"],
            PREDECESSOR_CONTROL_ROOT=str(self.control),
            PREDECESSOR_CONTROL_COMMIT="7" * 40,
            PREDECESSOR_CONTROL_TREE="8" * 40,
            PREDECESSOR_BRIDGE_SHA256=self.bridge["sha256"],
            PREDECESSOR_FIXED=self.fixed,
            TRAIN_ROOT=str(self.train_root),
            AUTHORITY_PATH=str(self.train_root / "live_val_metric_repair_claims/epoch-0001.json"),
            ADOPTION_PATH=str(self.train_root / "live_val_consumer_adoption_claims/epoch-0001.json"),
            SPEC_PATH=str(self.train_root / "live_val_metric_repair_specs/epoch-0001.v1.json"),
            REPAIR_ROOT=str(self.repair_root),
            RUNNER_CONTROL_ROOT=str(self.control_root),
            RESULT_PATH=str(self.repair_root / "repair-result.json"),
            RUNNER_STATUS_PATH=str(self.control_root / ".guarded_status.json"),
            RUNNER_LOG_PATH=str(self.control_root / "runner.log"),
            GUARD_PROOF_PATH=str(self.control_root / "guard-proof.txt"),
            FORMAL_PYTHON=str(self.formal_python),
            FORMAL_PYTHON_LINK_TARGET="python3.12",
            FORMAL_PYTHON_SECONDARY_TARGET=str(self.formal_target),
            FORMAL_PYTHON_RESOLVED=str(self.formal_target),
            PASPA_ROOT=str(self.paspa),
            DIFFSHEG_ROOT=str(self.diffsheg),
            GUARDED_RUNNER_PATH=self.runner["path"],
            GUARDED_RUNNER_SHA256=self.runner["sha256"],
            GUARDED_RUNNER_BYTES=self.runner["bytes"],
            GUARD_VERIFIER_PATH=self.guard["path"],
            GUARD_VERIFIER_SHA256=self.guard["sha256"],
            GUARD_VERIFIER_BYTES=self.guard["bytes"],
            EXPECTED_FGD_BINARY64_HEX="3ff4000000000000",
        )

    def source_identity(self, root: str, commit: str, tree: str, _label: str) -> dict:
        return {
            "origin": repair.OFFICIAL_ORIGIN, "source_root": root,
            "commit": commit, "tree": tree, "clean": True,
            "detached": True, "local_branches_at_commit": [],
        }

    def repair_tool_identity(self, expected=None) -> dict:
        if expected is not None and expected != self.repair_source:
            raise repair.RepairError("metric repair tool source changed")
        return dict(self.repair_source)

    def spec_value(self) -> dict:
        value = {
            "format": repair.SPEC_FORMAT,
            "status": "frozen",
            "split": "val",
            "test_visible": False,
            "selection_eligible": False,
            "candidate_epoch": 1,
            "predecessor_state_root": str(self.state),
            "predecessor_run_root": str(self.run),
            "predecessor_campaign": self.refs["campaign"],
            "predecessor_job_claim": self.refs["job-claim"],
            "predecessor_active_claim": self.refs["active"],
            "predecessor_work_authority": self.refs["work-authority"],
            "predecessor_authorization": self.refs["authorization"],
            "predecessor_runner_status": self.refs["runner-status"],
            "predecessor_recovery_request": self.refs["recovery-request"],
            "predecessor_recovery_authority": self.refs["recovery-authority"],
            "predecessor_recovery_claim": self.refs["recovery-claim"],
            "candidate_receipt": self.refs["candidate-receipt"],
            "preflight": self.refs["preflight"],
            "inference_lineage": self.refs["lineage"],
            "final_manifest": self.refs["final-manifest"],
            "clip_manifest": self.refs["clip-manifest"],
            "old_report": self.refs["old-report"],
            "predecessor_failure_log": self.refs["failure-log"],
            "frozen_evaluator_source": {
                "origin": repair.OFFICIAL_ORIGIN,
                "source_root": str(self.frozen),
                "commit": "4" * 40,
                "tree": "5" * 40,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            },
            "predecessor_control_source": {
                "origin": repair.OFFICIAL_ORIGIN,
                "source_root": str(self.control),
                "commit": "7" * 40,
                "tree": "8" * 40,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            },
            "repair_tool_source": self.repair_source,
            "frozen_evaluator": self.frozen_evaluator,
            "bridge": self.bridge,
            "formal_python": str(self.formal_python),
            "paspa_root": str(self.paspa),
            "diffsheg_root": str(self.diffsheg),
            "batch_size": 64,
            "repair_root": str(self.repair_root),
            "guarded_runner": self.runner,
            "guard_verifier": self.guard,
            "repair_tool": self.tool,
            "result_path": str(self.repair_root / "repair-result.json"),
            "runner_control_root": str(self.control_root),
            "runner_status_path": str(self.control_root / ".guarded_status.json"),
            "runner_log_path": str(self.control_root / "runner.log"),
            "guard_proof_path": str(self.control_root / "guard-proof.txt"),
        }
        return repair.self_hashed(value)

    def rewrite_spec(self, value: dict, path: str = "spec.json") -> dict:
        value = dict(value)
        value.pop("receipt_payload_sha256", None)
        output = Path(repair.SPEC_PATH)
        if output.exists():
            output.unlink()
        return write_json(output, repair.self_hashed(value))

    def repaired_report(self, **adapter_changes: str) -> dict:
        value = json.loads(json.dumps(self.old_report_value))
        value["provenance"]["adapter"].update({
            "path": str(self.frozen / "scripts/show_base/evaluate_diffsheg_val_fgd.py"),
            "repository_root": str(self.frozen),
            "repository_git_head": "4" * 40,
            **adapter_changes,
        })
        return value


class MetricRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = RepairFixture()
        self.patch = self.fx.patches()
        self.patch.start()
        self.source_patch = mock.patch.object(
            repair, "inspect_source_checkout", side_effect=self.fx.source_identity,
        )
        self.source_patch.start()
        self.tool_source_patch = mock.patch.object(
            repair, "inspect_repair_tool_source",
            side_effect=self.fx.repair_tool_identity,
        )
        self.tool_source_patch.start()

    def tearDown(self) -> None:
        self.tool_source_patch.stop()
        self.source_patch.stop()
        self.patch.stop()
        self.fx.close()

    def test_authorize_binds_terminal_snapshot_and_is_create_new(self) -> None:
        spec_artifact = self.fx.rewrite_spec(self.fx.spec_value())
        output = Path(repair.AUTHORITY_PATH)
        args = argparse.Namespace(
            spec=spec_artifact["path"],
            expected_spec_sha256=spec_artifact["sha256"],
            expected_spec_bytes=spec_artifact["bytes"],
            output=str(output),
        )
        with mock.patch.object(repair.time, "time", return_value=7.0):
            result = repair.authorize(args)
        self.assertEqual(result["status"], "authorized")
        authority = json.loads(output.read_text())
        self.assertEqual(
            authority["terminal_snapshot"],
            repair.tree_snapshot([str(self.fx.state), str(self.fx.run)]),
        )
        self.assertEqual(authority["receipt_payload_sha256"], repair.payload_sha(authority))
        with self.assertRaisesRegex(repair.RepairError, "already exists"):
            repair.authorize(args)

    def test_prepare_runner_control_is_exact_create_new_0700(self) -> None:
        spec_artifact = self.fx.rewrite_spec(self.fx.spec_value())
        authorize_args = argparse.Namespace(
            spec=spec_artifact["path"],
            expected_spec_sha256=spec_artifact["sha256"],
            expected_spec_bytes=spec_artifact["bytes"],
            output=repair.AUTHORITY_PATH,
        )
        with mock.patch.object(repair.time, "time", return_value=7.0):
            authorized = repair.authorize(authorize_args)
        authority = authorized["authority"]
        prepare_args = argparse.Namespace(
            authority=authority["path"],
            expected_authority_sha256=authority["sha256"],
            expected_authority_bytes=authority["bytes"],
            output=repair.RUNNER_CONTROL_ROOT,
        )
        prepared = repair.prepare_runner_control(prepare_args)
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(
            stat.S_IMODE(Path(repair.RUNNER_CONTROL_ROOT).stat().st_mode),
            0o700,
        )
        with self.assertRaisesRegex(repair.RepairError, "already exists"):
            repair.prepare_runner_control(prepare_args)

    def test_build_spec_is_fixed_schema_and_create_new(self) -> None:
        args = argparse.Namespace(output=repair.SPEC_PATH)
        result = repair.build_spec(args)
        self.assertEqual(result["status"], "frozen")
        value = json.loads(Path(repair.SPEC_PATH).read_text())
        self.assertEqual(set(value), set(repair.SPEC_KEYS))
        self.assertNotIn("successor_work_authority", value)
        self.assertEqual(value["formal_python"], str(self.fx.formal_python))
        self.assertEqual(value["batch_size"], 64)
        self.assertEqual(value["guarded_runner"], self.fx.runner)
        self.assertEqual(value["guard_verifier"], self.fx.guard)
        self.assertEqual(value["repair_root"], str(self.fx.repair_root))
        self.assertEqual(value["runner_control_root"], str(self.fx.control_root))
        with self.assertRaisesRegex(repair.RepairError, "already exists"):
            repair.build_spec(args)

    def test_spec_rejects_nonfrozen_evaluator_source(self) -> None:
        value = self.fx.spec_value()
        value["frozen_evaluator_source"]["commit"] = "6" * 40
        spec_artifact = self.fx.rewrite_spec(value, "bad-spec.json")
        with self.assertRaisesRegex(repair.RepairError, "evaluator source"):
            repair.load_spec(spec_artifact)

    def test_terminal_snapshot_detects_append_only_mutation(self) -> None:
        before = repair.tree_snapshot([str(self.fx.state), str(self.fx.run)])
        (self.fx.run / "new-output").write_text("forbidden\n")
        after = repair.tree_snapshot([str(self.fx.state), str(self.fx.run)])
        self.assertNotEqual(before, after)

    def test_repair_root_may_not_overlap_terminal_roots(self) -> None:
        value = self.fx.spec_value()
        value["repair_root"] = str(self.fx.run / "repair")
        spec_artifact = self.fx.rewrite_spec(value, "overlap-spec.json")
        with self.assertRaisesRegex(repair.RepairError, "output roots changed"):
            repair.load_spec(spec_artifact)

    def test_spec_rejects_split_runner_control_roots(self) -> None:
        value = self.fx.spec_value()
        value["runner_log_path"] = str(self.fx.root / "other-control/stdout.log")
        spec_artifact = self.fx.rewrite_spec(value, "split-control-spec.json")
        with self.assertRaisesRegex(repair.RepairError, "output roots changed"):
            repair.load_spec(spec_artifact)

    def test_spec_rejects_runtime_or_batch_drift(self) -> None:
        for key, replacement in (
            ("formal_python", "/wrong/python"),
            ("paspa_root", "/wrong/paspa"),
            ("diffsheg_root", "/wrong/diffsheg"),
            ("batch_size", 32),
        ):
            with self.subTest(key=key):
                value = self.fx.spec_value()
                value[key] = replacement
                spec_artifact = self.fx.rewrite_spec(value)
                with self.assertRaisesRegex(
                    repair.RepairError, "runtime roots/batch changed",
                ):
                    repair.load_spec(spec_artifact)

    def test_spec_rejects_formal_python_symlink_drift(self) -> None:
        spec_artifact = self.fx.rewrite_spec(self.fx.spec_value())
        self.fx.formal_python.unlink()
        os.symlink("wrong-python", self.fx.formal_python)
        with self.assertRaisesRegex(repair.RepairError, "formal Python"):
            repair.load_spec(spec_artifact)

    def test_spec_rejects_guarded_runner_byte_drift(self) -> None:
        spec_artifact = self.fx.rewrite_spec(self.fx.spec_value())
        Path(self.fx.runner["path"]).write_bytes(b"changed runner\n")
        with self.assertRaisesRegex(repair.RepairError, "guarded_runner artifact changed"):
            repair.load_spec(spec_artifact)

    def test_spec_rejects_type_only_artifact_size_change(self) -> None:
        value = self.fx.spec_value()
        value["guarded_runner"] = dict(value["guarded_runner"])
        value["guarded_runner"]["bytes"] = float(
            value["guarded_runner"]["bytes"]
        )
        spec_artifact = self.fx.rewrite_spec(value)
        with self.assertRaisesRegex(repair.RepairError, "guarded_runner artifact changed"):
            repair.load_spec(spec_artifact)

    def test_spec_rejects_repair_tool_source_drift(self) -> None:
        value = self.fx.spec_value()
        value["repair_tool_source"] = dict(value["repair_tool_source"])
        value["repair_tool_source"]["commit"] = "b" * 40
        spec_artifact = self.fx.rewrite_spec(value)
        with self.assertRaisesRegex(repair.RepairError, "repair tool source changed"):
            repair.load_spec(spec_artifact)

    def test_regular_bytes_rejects_hard_link(self) -> None:
        first = self.fx.root / "one.bin"
        second = self.fx.root / "two.bin"
        first.write_bytes(b"same inode")
        os.link(first, second)
        with self.assertRaisesRegex(repair.RepairError, "multiple hard links"):
            repair.regular_bytes(str(first), "hard-linked evidence")

    def test_write_new_fsyncs_file_and_parent(self) -> None:
        target = self.fx.root / "durable/output.json"
        real_fsync = repair.os.fsync
        calls: list[int] = []

        def tracked_fsync(descriptor: int) -> None:
            calls.append(descriptor)
            real_fsync(descriptor)

        with mock.patch.object(repair.os, "fsync", side_effect=tracked_fsync):
            repair.write_new(str(target), {"ok": True}, "durable output")
        self.assertGreaterEqual(len(calls), 2)

    def test_bridge_replay_uses_path_object(self) -> None:
        self.assertIn("from pathlib import Path", repair.BRIDGE_MEASUREMENT_REPLAY_CODE)
        self.assertIn(
            "_validate_live_measurement(Path(measurement_path), measurement_sha)",
            repair.BRIDGE_MEASUREMENT_REPLAY_CODE,
        )

    def test_finalize_reexecutes_verifier_and_emits_full_adoption_schema(self) -> None:
        spec_ref = {"path": repair.SPEC_PATH, "sha256": "1" * 64, "bytes": 10}
        authority_artifact = {
            "path": repair.AUTHORITY_PATH, "sha256": "2" * 64, "bytes": 20,
        }
        snapshot = {"roots": [str(self.fx.state), str(self.fx.run)]}
        authority = {"spec": spec_ref, "terminal_snapshot": snapshot}
        measurement = repair.self_hashed({"metrics": {"fgd": 1.25}})
        measurement_ref = {
            "path": str(self.fx.repair_root / "live-measurement.json"),
            "sha256": "3" * 64, "bytes": 30,
            "receipt_payload_sha256": measurement["receipt_payload_sha256"],
        }
        report_ref = {
            "path": str(self.fx.repair_root / "diffsheg-val-fgd.frozen-4066.json"),
            "sha256": "5" * 64, "bytes": 50,
        }
        comparison_ref = {
            "path": str(self.fx.repair_root / "report-comparison.json"),
            "sha256": "6" * 64, "bytes": 60,
        }
        spec = self.fx.spec_value()
        expected_bridge_stdout = (
            json.dumps(
                {
                    "status": "complete", "split": "val",
                    "test_visible": False, "selection_eligible": False,
                    "epoch": 1, "fgd": 1.25, "created": True,
                    "measurement": measurement_ref,
                },
                sort_keys=True, allow_nan=False,
            )
            + "\n"
        ).encode()
        result = repair.self_hashed({
            "format": repair.RESULT_FORMAT, "status": "complete",
            "split": "val", "test_visible": False,
            "selection_eligible": False, "candidate_epoch": 1,
            "inference_reruns": 0, "metric_replays": 1,
            "authority": authority_artifact,
            "terminal_snapshot_before": snapshot,
            "terminal_snapshot_after": snapshot,
            "evaluator_argv": repair.evaluator_argv_for(spec, report_ref["path"]),
            "bridge_complete_argv": repair.bridge_argv_for(
                spec, report_ref, measurement_ref["path"],
            ),
            "bridge_complete_stdout_sha256": hashlib.sha256(
                expected_bridge_stdout
            ).hexdigest(),
            "bridge_measurement_replay_argv": [
                str(self.fx.formal_python), "-I", "-B", "-c",
                repair.BRIDGE_MEASUREMENT_REPLAY_CODE, self.fx.bridge["path"],
                measurement_ref["path"], measurement_ref["sha256"],
            ],
            "bridge_measurement_replay_stdout": "PASS\n",
            "repaired_report": report_ref,
            "report_comparison": comparison_ref,
            "measurement": measurement_ref, "completed_unix": 8.0,
        })
        result_artifact = {
            "path": repair.RESULT_PATH,
            "sha256": hashlib.sha256(repair.canonical_bytes(result)).hexdigest(),
            "bytes": len(repair.canonical_bytes(result)),
        }
        restored = {str(index): 1000 + index for index in range(8)}
        expected_run_argv = [
            str(self.fx.formal_python), "-I", self.fx.tool["path"], "run",
            "--authority", authority_artifact["path"],
            "--expected-authority-sha256", authority_artifact["sha256"],
            "--expected-authority-bytes", str(authority_artifact["bytes"]),
        ]
        status = {
            "updated_at": "2026-08-03T12:00:00+0800", "state": "finished", "wrapper_pid": 10,
            "child_pid": 11, "return_code": 0, "received_signal": None,
            "error": None, "cleanup_error": None, "restored_guards": restored,
            "restore_error": None, "command": expected_run_argv,
        }
        status_artifact = {
            "path": repair.RUNNER_STATUS_PATH, "sha256": "8" * 64, "bytes": 80,
        }
        log_artifact = {
            "path": repair.RUNNER_LOG_PATH, "sha256": "9" * 64, "bytes": 90,
        }
        runner_stdout = {"status": "complete", "result": result_artifact}
        runner_raw = (
            json.dumps(runner_stdout, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        guard_artifact = {
            "path": repair.GUARD_PROOF_PATH, "sha256": "a" * 64, "bytes": 100,
        }
        guard_text = "PASS " + " ".join(
            "GPU%d=PID%d" % (index, restored[str(index)]) for index in range(8)
        ) + "\n"
        guard_raw = guard_text.encode("ascii")
        comparison = repair.self_hashed({
            "comparison": True, "fgd_binary64_hex": "3ff4000000000000",
        })
        written: dict = {}

        def fake_read(reference, label):
            if label == "metric repair result":
                return result_artifact, result
            if label == "metric repair runner status":
                return status_artifact, status
            if label == "adopted measurement":
                return dict(reference), measurement
            if label == "metric report comparison":
                return comparison_ref, comparison
            raise AssertionError(label)

        def fake_artifact(path, label, expected=None):
            if label == "metric repair runner log":
                return log_artifact
            raise AssertionError(label)

        def fake_regular(path, label):
            if label == "metric repair runner log":
                return Path(path), runner_raw
            if label == "metric repair guard proof":
                return Path(path), guard_raw
            raise AssertionError(label)

        def fake_write(path, value, label):
            written.update(value)
            return {"path": path, "sha256": "b" * 64, "bytes": 1}

        verifier = mock.Mock(returncode=0, stdout=guard_raw, stderr=b"")
        replay = mock.Mock(returncode=0, stdout=b"PASS\n", stderr=b"")
        args = argparse.Namespace(
            authority=repair.AUTHORITY_PATH,
            expected_authority_sha256=authority_artifact["sha256"],
            expected_authority_bytes=authority_artifact["bytes"],
            result=repair.RESULT_PATH, expected_result_sha256=result_artifact["sha256"],
            expected_result_bytes=result_artifact["bytes"],
            runner_status=repair.RUNNER_STATUS_PATH,
            expected_runner_status_sha256=status_artifact["sha256"],
            expected_runner_status_bytes=status_artifact["bytes"],
            runner_log=repair.RUNNER_LOG_PATH,
            expected_runner_log_sha256=log_artifact["sha256"],
            expected_runner_log_bytes=log_artifact["bytes"],
            guard_proof=repair.GUARD_PROOF_PATH,
            expected_guard_proof_sha256=None,
            expected_guard_proof_bytes=None,
            output=repair.ADOPTION_PATH,
        )
        with mock.patch.object(
            repair, "load_authority",
            return_value=(authority_artifact, authority, spec),
        ), mock.patch.object(repair, "read_document", side_effect=fake_read), \
                mock.patch.object(repair, "artifact", side_effect=fake_artifact), \
                mock.patch.object(repair, "regular_bytes", side_effect=fake_regular), \
                mock.patch.object(repair, "tree_snapshot", return_value=snapshot), \
                mock.patch.object(repair, "compare_reports", return_value=comparison), \
                mock.patch.object(repair, "write_new", side_effect=fake_write), \
                mock.patch.object(
                    repair, "write_new_bytes", return_value=guard_artifact,
                ) as proof_writer, \
                mock.patch.object(
                    repair.subprocess, "run", side_effect=[verifier, replay],
                ) as run:
            repair.finalize(args)
        self.assertEqual(set(written), set(repair.ADOPTION_KEYS))
        self.assertEqual(written["metric_repair_spec"], spec_ref)
        self.assertEqual(written["metric_repair_result"], result_artifact)
        self.assertEqual(written["guard_proof"], guard_artifact)
        self.assertNotIn("successor_work_authority", written)
        self.assertEqual(run.call_count, 2)
        proof_writer.assert_called_once_with(
            Path(repair.GUARD_PROOF_PATH), guard_raw,
            "metric repair guard proof",
        )

    def test_report_compare_allows_exact_three_field_transition(self) -> None:
        new_ref = write_json(
            self.fx.root / "new-report.json", self.fx.repaired_report()
        )
        comparison = repair.compare_reports(self.fx.refs["old-report"], new_ref)
        self.assertFalse(comparison["raw_feature_bytes_compared"])
        self.assertEqual(
            comparison["fgd_binary64_hex"],
            "3ff4000000000000",
        )

    def test_report_compare_rejects_type_only_change(self) -> None:
        value = self.fx.repaired_report()
        value["nested"][0] = 1.0
        new_ref = write_json(self.fx.root / "type-change.json", value)
        with self.assertRaisesRegex(repair.RepairError, "outside the three"):
            repair.compare_reports(self.fx.refs["old-report"], new_ref)

    def test_report_compare_rejects_one_ulp_fgd_change(self) -> None:
        value = self.fx.repaired_report()
        value["metrics"]["fgd"] = math.nextafter(1.25, math.inf)
        new_ref = write_json(self.fx.root / "fgd-change.json", value)
        with self.assertRaises(repair.RepairError):
            repair.compare_reports(self.fx.refs["old-report"], new_ref)


class SupervisorAdoptionScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="adoption-scan-")
        self.root = Path(self.temp.name).resolve()
        self.state = self.root / "state"
        for name in (
            "job_claims", "completions", "runner_status", "runner_logs",
            "authorities", "authorizations",
        ):
            (self.state / name).mkdir(parents=True, exist_ok=True)
        (self.state / "campaign.json").write_text("{}\n")
        self.jobs = []
        for epoch in supervisor.CANDIDATE_EPOCHS:
            run_root = self.root / "runs" / ("e%d" % epoch)
            self.jobs.append({
                "epoch": epoch,
                "candidate_receipt_path": str(self.root / "receipts" / ("epoch-%04d.json" % epoch)),
                "authority_path": str(self.state / "authorities" / ("epoch-%04d.json" % epoch)),
                "authorization_path": str(self.state / "authorizations" / ("epoch-%04d.json" % epoch)),
                "run_root": str(run_root),
                "measurement_path": str(run_root / "candidates" / ("e%d" % epoch) / "live-measurement.json"),
                "completion_path": str(self.state / "completions" / ("epoch-%04d.json" % epoch)),
                "runner_status_path": str(self.state / "runner_status" / ("epoch-%04d.json" % epoch)),
                "runner_log_path": str(self.state / "runner_logs" / ("epoch-%04d.log" % epoch)),
            })
        self.campaign = {
            "_state_root": self.state,
            "_claims_dir": self.state / "job_claims",
            "_completions_dir": self.state / "completions",
            "_runner_status_dir": self.state / "runner_status",
            "_runner_logs_dir": self.state / "runner_logs",
            "_authorities_dir": self.state / "authorities",
            "_authorizations_dir": self.state / "authorizations",
            "_selection_root": self.root / "selection",
            "_reconciliation_path": self.state / "reconciliation.json",
            "summary_path": str(self.state / "final_summary.json"),
            "_jobs": self.jobs,
            "_adopted_e1": {"path": str(self.root / "adoption.json"), "sha256": "a" * 64, "bytes": 1},
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_adoption_is_completed_zero_and_e2_is_head(self) -> None:
        adopted = (
            {"path": "/adoption", "sha256": "b" * 64, "bytes": 1},
            {"candidate_epoch": 1, "measurement": {"path": "/m", "sha256": "c" * 64}},
        )
        with mock.patch.object(supervisor, "_load_adopted_e1", return_value=adopted):
            completed, head = supervisor._scan_v2(self.campaign)
        self.assertEqual(completed, [adopted])
        self.assertEqual(head["epoch"], 2)

    def test_adoption_rejects_any_successor_e1_claim(self) -> None:
        (self.state / "job_claims/epoch-0001.json").write_text("{}\n")
        with mock.patch.object(supervisor, "_load_adopted_e1"):
            with self.assertRaisesRegex(supervisor.SupervisorError, "successor execution output"):
                supervisor._scan_v2(self.campaign)


if __name__ == "__main__":
    unittest.main()
