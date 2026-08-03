#!/usr/bin/env python3
"""Focused CPU-only tests for the no-rerun metric incident recovery path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.show_base import adopt_base_v14_metric_repair as recovery


SCRIPT = Path(recovery.__file__).resolve()


def write_bytes(path: Path, raw: bytes, mode: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


class IncidentRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="metric-incident-recovery-")
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_directory_snapshot_binds_inventory_modes_and_bytes(self) -> None:
        incident = self.root / "incident"
        incident.mkdir(mode=0o700)
        first = write_bytes(incident / "report.json", b"report\n", 0o600)
        second = write_bytes(incident / "evaluator.log", b"progress\n", 0o400)
        snapshot = recovery.exact_directory_snapshot(
            str(incident),
            {"report.json": (first, 0o600), "evaluator.log": (second, 0o400)},
            "test incident",
        )
        self.assertEqual(snapshot["root_mode"], 0o700)
        self.assertEqual(snapshot["entry_count"], 2)
        self.assertEqual(snapshot["total_bytes"], 16)
        self.assertEqual(
            [(entry["name"], entry["mode"]) for entry in snapshot["entries"]],
            [("evaluator.log", 0o400), ("report.json", 0o600)],
        )

    def test_exact_directory_snapshot_rejects_extra_mode_and_content_drift(self) -> None:
        for mutation in ("extra", "mode", "content"):
            with self.subTest(mutation=mutation):
                incident = self.root / mutation
                incident.mkdir(mode=0o700)
                report = write_bytes(incident / "report.json", b"report\n", 0o600)
                expected = {"report.json": (report, 0o600)}
                if mutation == "extra":
                    write_bytes(incident / "extra", b"x", 0o400)
                elif mutation == "mode":
                    (incident / "report.json").chmod(0o400)
                else:
                    (incident / "report.json").write_bytes(b"changed\n")
                with self.assertRaises(recovery.RepairError):
                    recovery.exact_directory_snapshot(
                        str(incident), expected, "mutated incident",
                    )

    def test_recovery_cli_isolated_help_exposes_only_explicit_recovery_commands(self) -> None:
        process = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--help"],
            cwd="/", stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        help_text = process.stdout.decode()
        for command in (
            "build-recovery-spec", "authorize-recovery",
            "prepare-recovery-runner-control", "run-recovery",
            "finalize-recovery",
        ):
            self.assertIn(command, help_text)

    def test_recovery_schemas_separate_original_and_recovery_counts(self) -> None:
        self.assertIn("recovery_evaluator_invocations", recovery.RECOVERY_RESULT_KEYS)
        self.assertIn("recovery_metric_replays", recovery.RECOVERY_RESULT_KEYS)
        self.assertIn("metric_replays_total", recovery.RECOVERY_RESULT_KEYS)
        self.assertNotIn("evaluator_argv", recovery.RECOVERY_RESULT_KEYS)
        self.assertNotIn("inference_argv", recovery.RECOVERY_RESULT_KEYS)
        self.assertIn("original_metric_replays", recovery.RECOVERY_ADOPTION_KEYS)
        self.assertIn(
            "incident_runner_control_snapshot_before",
            recovery.RECOVERY_ADOPTION_KEYS,
        )

    def test_run_recovery_never_invokes_evaluator_or_inference(self) -> None:
        recovery_root = self.root / "recovery"
        report_value = {"metrics": {"fgd": 1.25}}
        report_raw = recovery.canonical_bytes(report_value)
        report = write_bytes(self.root / "partial-report.json", report_raw, 0o600)
        evaluator_log = write_bytes(
            self.root / "evaluator.log", b"complete evidence\n", 0o400,
        )
        preflight = write_bytes(self.root / "preflight.json", b"preflight\n", 0o400)
        lineage = write_bytes(self.root / "lineage.json", b"lineage\n", 0o400)
        bridge = write_bytes(self.root / "bridge.py", b"# bridge\n", 0o400)
        authority_artifact = {
            "path": str(self.root / "authority.json"),
            "sha256": "a" * 64, "bytes": 1,
        }
        terminal = {"inventory_sha256": "1" * 64}
        incident = {"inventory_sha256": "2" * 64}
        control = {"inventory_sha256": "3" * 64}
        authority = {
            "terminal_snapshot": terminal, "incident_snapshot": incident,
            "incident_runner_control_snapshot": control,
        }
        spec = {
            "formal_python": sys.executable, "bridge": bridge,
        }
        measurement_value = recovery.self_hashed({
            "format": recovery.MEASUREMENT_FORMAT, "status": "complete",
            "candidate_epoch": 1, "split": "val", "test_visible": False,
            "selection_eligible": False, "metrics": {"fgd": 1.25},
            "diffsheg_report": {
                "path": report["path"], "sha256": report["sha256"],
            },
        })
        comparison_value = recovery.self_hashed({
            "status": "equivalent", "fgd_binary64_hex": "3ff4000000000000",
        })
        calls: list[tuple[list[str], str]] = []

        def fake_capture(argv, log_path, label):
            calls.append((list(argv), label))
            if label == "recovery bridge complete":
                measurement_path = recovery_root / "live-measurement.json"
                measurement_path.write_bytes(recovery.canonical_bytes(measurement_value))
                measurement_path.chmod(0o400)
                observed = recovery.artifact(str(measurement_path), "measurement")
                reference = {
                    **observed,
                    "receipt_payload_sha256": measurement_value["receipt_payload_sha256"],
                }
                stdout = {
                    "status": "complete", "split": "val",
                    "test_visible": False, "selection_eligible": False,
                    "epoch": 1, "fgd": 1.25, "created": True,
                    "measurement": reference,
                }
                recovery.write_new_bytes(
                    log_path,
                    (json.dumps(stdout, sort_keys=True) + "\n").encode(),
                    "bridge log",
                )
                return (json.dumps(stdout, sort_keys=True) + "\n").encode()
            recovery.write_new_bytes(log_path, b"PASS\n", "replay log")
            return b"PASS\n"

        args = argparse.Namespace(
            authority=authority_artifact["path"],
            expected_authority_sha256=authority_artifact["sha256"],
            expected_authority_bytes=authority_artifact["bytes"],
        )
        fixed = {
            "old_report": (str(self.root / "old.json"), "b" * 64, 1),
            "preflight": (preflight["path"], preflight["sha256"], preflight["bytes"]),
            "inference_lineage": (lineage["path"], lineage["sha256"], lineage["bytes"]),
        }
        with mock.patch.multiple(
            recovery,
            RECOVERY_ROOT=str(recovery_root),
            RECOVERY_RESULT_PATH=str(recovery_root / "recovery-result.json"),
            INCIDENT_REPAIRED_REPORT=report,
            INCIDENT_EVALUATOR_LOG=evaluator_log,
            PREDECESSOR_FIXED=fixed,
            EXPECTED_FGD_BINARY64_HEX=struct.pack(">d", 1.25).hex(),
        ), mock.patch.object(
            recovery, "load_recovery_authority",
            return_value=(authority_artifact, authority, spec),
        ), mock.patch.object(
            recovery, "tree_snapshot", return_value=terminal,
        ), mock.patch.object(
            recovery, "incident_partial_snapshot", return_value=incident,
        ), mock.patch.object(
            recovery, "incident_runner_control_snapshot", return_value=control,
        ), mock.patch.object(
            recovery, "compare_reports", return_value=comparison_value,
        ), mock.patch.object(
            recovery, "capture", side_effect=fake_capture,
        ):
            output = recovery.run_recovery(args)
        result = json.loads(Path(output["result"]["path"]).read_text())
        self.assertEqual(result["recovery_evaluator_invocations"], 0)
        self.assertEqual(result["recovery_inference_runs"], 0)
        self.assertEqual(result["recovery_metric_replays"], 0)
        self.assertEqual(result["metric_replays_total"], 1)
        self.assertEqual([label for _argv, label in calls], [
            "recovery bridge complete", "recovery bridge measurement replay",
        ])
        flattened = "\n".join(" ".join(argv) for argv, _label in calls)
        self.assertNotIn("evaluate_diffsheg_val_fgd.py", flattened)
        self.assertNotIn("infer_val.py", flattened)
        self.assertNotIn("test_camn_audio.py", flattened)


if __name__ == "__main__":
    unittest.main()
