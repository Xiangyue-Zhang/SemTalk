from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_diffsheg_val_partition_contract as contract
from scripts.show_base import base_long_val_contract as long_contract
from scripts.show_base import select_base_official_adapt as decision_core
from scripts.show_base import select_base_official_adapt_long as long_selector


REPOSITORY = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY / "scripts/show_base/run_base_diffsheg_val_8shard.sh"
FINALIZER = (
    REPOSITORY
    / "scripts/show_base/finalize_base_diffsheg_val_partitions.sh"
)
PRODUCER = REPOSITORY / "scripts/show_base/produce_base_val_measurement.py"
SELECTOR = REPOSITORY / "scripts/show_base/select_base_official_adapt_long.py"


class StaticPartitionContractTests(unittest.TestCase):
    def test_exact_static_modulo_partitions_cover_all_22_once(self) -> None:
        self.assertEqual(
            contract.PARTITION_EPOCHS[0],
            (1, 4, 16, 40, 60, 80, 120, 160, 200, 280, 360),
        )
        self.assertEqual(
            contract.PARTITION_EPOCHS[1],
            (2, 8, 32, 50, 70, 100, 140, 180, 240, 320, 400),
        )
        self.assertEqual(
            sorted(contract.PARTITION_EPOCHS[0] + contract.PARTITION_EPOCHS[1]),
            sorted(long_contract.EXPECTED_CANDIDATE_EPOCHS),
        )
        self.assertEqual(
            set(contract.PARTITION_EPOCHS[0]).intersection(
                contract.PARTITION_EPOCHS[1]
            ),
            set(),
        )

    def _partition_receipt(
        self,
        root: Path,
        partition_id: int,
        *,
        inputs: dict[str, object],
        source: dict[str, object],
    ) -> tuple[Path, str, str]:
        run_root = root / f"partition-{partition_id}"
        run_root.mkdir()
        rows = []
        for epoch in contract.PARTITION_EPOCHS[partition_id]:
            candidate = run_root / "candidates" / f"e{epoch}"
            rows.append(
                {
                    "epoch": epoch,
                    "fgd": float(epoch),
                    "measurement": {
                        "path": str(candidate / "diffsheg-val-measurement.json"),
                        "sha256": f"{epoch:064x}",
                        "receipt_payload_sha256": f"{epoch + 1:064x}",
                    },
                    "inference_lineage": {
                        "path": str(candidate / "final/val-inference-lineage.json"),
                        "sha256": f"{epoch + 2:064x}",
                        "receipt_payload_sha256": f"{epoch + 3:064x}",
                    },
                    "diffsheg_report": {
                        "path": str(candidate / "diffsheg-val-fgd.json"),
                        "sha256": f"{epoch + 4:064x}",
                    },
                    "prediction_dir": str(candidate / "final/predictions/val"),
                    "ground_truth_dir": str(
                        candidate / "final/ground-truth/val"
                    ),
                    "exact_once": True,
                    "finite": True,
                }
            )
        receipt = contract._with_payload_sha(
            {
                "format": contract.PARTITION_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "selection_metric": "fgd",
                "partition_id": partition_id,
                "partition_count": 2,
                "formal_host": contract.EXPECTED_HOSTS[partition_id],
                "source": source,
                "run_root": str(run_root),
                "candidate_epochs": list(
                    contract.PARTITION_EPOCHS[partition_id]
                ),
                "candidate_count": 11,
                "shards_per_candidate": 8,
                "clip_evaluations": 11 * 1_715,
                "inputs": inputs,
                "preflight": {
                    "path": str(run_root / "common-preflight.json"),
                    "sha256": "a" * 64,
                    "receipt_payload_sha256": "b" * 64,
                },
                "evaluator_bundle": {
                    "path": str(run_root / "diffsheg-evaluator-bundle.json"),
                    "sha256": "c" * 64,
                    "receipt_payload_sha256": "d" * 64,
                },
                "evaluator_bundle_identity": {
                    "paspa_commit": "1" * 40,
                    "paspa_tree": "2" * 40,
                    "paspa_evaluator_sha256": "e" * 64,
                    "diffsheg_commit": "3" * 40,
                    "stats_sha256": "f" * 64,
                    "gesture_autoencoder_sha256": "1" * 64,
                },
                "measurements": rows,
                "exact_once": True,
                "finite": True,
            }
        )
        path = run_root / "partition-receipt.json"
        contract._write_new_json(path, receipt, "partition receipt")
        digest, _size, payload = contract._artifact_fields(path, True)
        return path, digest, payload

    def test_union_recovers_registered_epoch_order_and_rejects_swap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-val-union-") as raw:
            root = Path(raw).resolve()
            source = {
                "origin": contract.SOURCE_ORIGIN,
                "commit": "1" * 40,
                "tree": "2" * 40,
                "clean": True,
                "detached": True,
                "local_branches": [],
            }
            candidate_bundle = {
                "manifest": {"path": "/validation/manifest.json", "sha256": "1" * 64},
                "status": {"path": "/validation/status.json", "sha256": "2" * 64},
                "frozen_inputs": {
                    "path": "/validation/frozen.json",
                    "sha256": "3" * 64,
                    "receipt_sha256": "4" * 64,
                },
            }
            val = {
                "path": "/validation/val-inputs.json",
                "sha256": "5" * 64,
                "receipt_payload_sha256": "6" * 64,
            }
            pipeline = {
                "path": "/validation/pipeline.json",
                "sha256": "7" * 64,
                "receipt_payload_sha256": "8" * 64,
            }
            inputs = contract._input_artifacts(candidate_bundle, val, pipeline)
            triples = [
                self._partition_receipt(
                    root, partition_id, inputs=inputs, source=source
                )
                for partition_id in range(2)
            ]
            args = argparse.Namespace(
                partition_count=2,
                source_commit=source["commit"],
                source_tree=source["tree"],
                partition_receipt=[triples[0][0], triples[1][0]],
                expected_partition_receipt_sha256=[
                    triples[0][1],
                    triples[1][1],
                ],
                expected_partition_receipt_payload_sha256=[
                    triples[0][2],
                    triples[1][2],
                ],
            )
            with mock.patch.object(
                contract,
                "_candidate_context",
                return_value=(candidate_bundle, val, {}, pipeline),
            ):
                union = contract.build_union(args)
                self.assertEqual(
                    [row["epoch"] for row in union["measurements"]],
                    list(contract.EXPECTED_EPOCHS),
                )
                self.assertEqual(union["candidate_count"], 22)
                self.assertEqual(union["total_shards"], 176)
                self.assertEqual(union["clip_evaluations"], 22 * 1_715)
                args.partition_receipt.reverse()
                args.expected_partition_receipt_sha256.reverse()
                args.expected_partition_receipt_payload_sha256.reverse()
                with self.assertRaises(contract.PartitionContractError):
                    contract.build_union(args)

    def test_receipt_publication_is_create_new(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-val-new-") as raw:
            path = Path(raw).resolve() / "receipt.json"
            contract._write_new_json(path, {"status": "one"}, "receipt")
            with self.assertRaises(contract.PartitionContractError):
                contract._write_new_json(path, {"status": "two"}, "receipt")


class FormalLauncherStaticTests(unittest.TestCase):
    def test_worker_is_guarded_exact_eight_shard_diffsheg_only(self) -> None:
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn("semtalk_require_exact_guarded_runner_all_gpus", source)
        self.assertIn("/tmp/globaldiff_guarded_runner.py", (
            REPOSITORY / "scripts/show_base/guarded_runner_contract.sh"
        ).read_text(encoding="utf-8"))
        self.assertIn("for shard_id in 0 1 2 3 4 5 6 7", source)
        self.assertIn("--num-shards 8", source)
        self.assertIn("evaluate_diffsheg_val_fgd.py", source)
        self.assertIn("produce_base_val_measurement.py", source)
        self.assertIn("pending_pid", source)
        self.assertIn("PROC_PPID", source)
        self.assertIn("PROC_STARTTIME", source)
        self.assertIn("PROC_CMDLINE_SHA256", source)
        self.assertIn("sha256_argv", source)
        self.assertIn("launcher_path != \"$launcher_dir/$launcher_name\"", source)
        self.assertNotIn("pgrep", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("evaluate_talkshow_show_metrics.py", source)
        self.assertNotIn("replay_released2_primary.py", source)
        self.assertNotIn("released2", source.casefold())
        self.assertNotIn("speaker2", source.casefold())

    def test_cpu_finalizer_calls_official_22_way_selector(self) -> None:
        source = FINALIZER.read_text(encoding="utf-8")
        self.assertIn("partition_count != 2", source)
        self.assertLess(
            source.index("partition_count != 2"),
            source.index("create-run-root"),
        )
        self.assertIn("select_base_official_adapt_long.py", source)
        self.assertIn("measurement_fields[@]} -ne 44", source)
        self.assertIn("for ((index = 0; index < 44; index += 2))", source)
        self.assertNotIn("--device", source)
        self.assertNotIn("released2", source.casefold())
        self.assertNotIn("speaker2", source.casefold())

    def test_measurement_is_validated_before_formal_hardlink(self) -> None:
        source = PRODUCER.read_text(encoding="utf-8")
        self.assertLess(
            source.index("long_selector.validate_measurement"),
            source.index("os.link(staging, output"),
        )
        self.assertNotIn("output.unlink", source)


class LongSelectorClosureTests(unittest.TestCase):
    def test_long_profile_patches_inference_source_and_restores_it(self) -> None:
        original = decision_core.VAL_INFERENCE_SOURCE
        with long_selector._long_profile():
            self.assertEqual(
                decision_core.VAL_INFERENCE_SOURCE,
                long_contract.VAL_INFERENCE_SOURCE,
            )
        self.assertIs(decision_core.VAL_INFERENCE_SOURCE, original)

    def test_minimum_is_unique_by_fgd_then_epoch_under_long_profile(self) -> None:
        rows = [
            {"epoch": epoch, "metrics": {"fgd": 2.0}}
            for epoch in contract.EXPECTED_EPOCHS
        ]
        rows[10]["metrics"]["fgd"] = 1.0
        rows[11]["metrics"]["fgd"] = 1.0
        with long_selector._long_profile():
            winner = decision_core.select_minimum_fgd(rows)
        self.assertEqual(winner["epoch"], contract.EXPECTED_EPOCHS[10])

    def test_producer_and_selector_help_work_from_foreign_cwd(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-val-cwd-") as raw:
            for path in (PRODUCER, SELECTOR):
                result = subprocess.run(
                    [sys.executable, str(path), "--help"],
                    cwd=raw,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
