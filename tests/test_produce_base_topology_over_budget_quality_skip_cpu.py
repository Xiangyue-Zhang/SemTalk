from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import (
    produce_base_topology_over_budget_quality_skip as producer,
)
from scripts.show_base import select_base_training_topology as selector
from scripts.show_base import train_base_official_adapt_long as contract


REPOSITORY = Path(__file__).resolve().parents[1]
TOPOLOGY_SPEC = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_topology_gate_spec_20260731.json"
)
QUALITY_SPEC = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_topology_quality_gate_spec_20260731.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_probe(path: Path, mode: str, eta: float) -> Path:
    specification = contract.TOPOLOGY_SPECS[mode]
    world_size = int(specification["world_size"])
    global_batch_size = int(specification["global_batch_size"])
    probe_samples = contract.TRAJECTORY_PROBE_UPDATES * global_batch_size
    probe_epoch_updates = list(
        contract._probe_epoch_update_counts(
            int(specification["updates_per_epoch"])
        )
    )
    probe_epoch_samples = [
        updates * global_batch_size for updates in probe_epoch_updates
    ]
    cross_epoch_duplicates = 0 if len(probe_epoch_updates) == 1 else 1
    median_seconds = (
        float(eta)
        / float(specification["updates_per_epoch"])
        / float(contract.TOTAL_EPOCHS)
    )
    derived_eta = (
        median_seconds
        * float(specification["updates_per_epoch"])
        * float(contract.TOTAL_EPOCHS)
    )
    body: dict[str, object] = {
        "format": contract.GATE_FORMAT,
        "status": "pass",
        "topology_mode": mode,
        "topology_classification": specification["classification"],
        "topology_gate_spec_sha256": _sha(TOPOLOGY_SPEC),
        "topology_independent_input_sha256": "a" * 64,
        "frozen_receipt_sha256": "b" * 64,
        "frozen_gate_compatibility_sha256": "c" * 64,
        "topology_receipt_sha256": "d" * 64,
        "node_count": specification["node_count"],
        "local_world_size": specification["local_world_size"],
        "world_size": world_size,
        "local_batch_size": specification["local_batch_size"],
        "global_batch_size": global_batch_size,
        "updates_per_epoch": specification["updates_per_epoch"],
        "unique_samples_per_epoch": specification[
            "unique_samples_per_epoch"
        ],
        "learning_rate": specification["learning_rate"],
        "warmup_updates": contract.THROUGHPUT_WARMUP_UPDATES,
        "timed_updates": contract.THROUGHPUT_TIMED_UPDATES,
        "optimizer_updates": contract.TRAJECTORY_PROBE_UPDATES,
        "precision": specification["precision"],
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "oom": False,
        "seconds_per_update": median_seconds,
        "median_seconds": median_seconds,
        "p90_seconds": median_seconds * 1.1,
        "p99_seconds": median_seconds * 1.2,
        "estimated_training_seconds": derived_eta,
        "estimated_epochs": contract.TOTAL_EPOCHS,
        "samples_per_second": global_batch_size / median_seconds,
        "peak_cuda_memory_bytes_all_ranks": [1024] * world_size,
        "data_wait_seconds": {"median": 0.01, "p99": 0.02},
        "collective_seconds": {
            "probe": "ten_scalar_nccl_all_reduce_calls",
            "median": 0.001,
            "p99": 0.002,
        },
        "batchnorm_inventory": [
            {
                "rank": rank,
                "before": [{"name": "hubert", "sha256": "e" * 64}],
                "after": [{"name": "hubert", "sha256": "e" * 64}],
            }
            for rank in range(world_size)
        ],
        "rng_inventory": [
            {
                "rank": rank,
                "torch_cuda_rng_state_sha256": f"{rank + 1:064x}",
            }
            for rank in range(world_size)
        ],
        "sample_inventory": {
            "sampler_drop_last": True,
            "padding_duplicates": 0,
            "probe_samples": probe_samples,
            "probe_unique_samples": probe_samples - cross_epoch_duplicates,
            "probe_sampler_epochs": list(range(len(probe_epoch_updates))),
            "probe_epoch_updates": probe_epoch_updates,
            "probe_epoch_samples": probe_epoch_samples,
            "probe_epoch_unique_samples": probe_epoch_samples,
            "probe_cross_epoch_duplicates": cross_epoch_duplicates,
            "full_epoch_samples": specification["unique_samples_per_epoch"],
            "full_epoch_unique_samples": specification[
                "unique_samples_per_epoch"
            ],
            "dataset_samples": contract.EXPECTED_TRAIN_SAMPLES,
            "dropped_tail_samples": (
                contract.EXPECTED_TRAIN_SAMPLES
                - int(specification["unique_samples_per_epoch"])
            ),
        },
    }
    body["receipt_sha256"] = contract.canonical_json_sha256(body)
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _argv(mode: str, probe: Path, output: Path) -> list[str]:
    return [
        "--mode",
        mode,
        "--topology-gate-spec",
        str(TOPOLOGY_SPEC),
        "--expected-topology-gate-spec-sha256",
        _sha(TOPOLOGY_SPEC),
        "--quality-gate-spec",
        str(QUALITY_SPEC),
        "--expected-quality-gate-spec-sha256",
        _sha(QUALITY_SPEC),
        "--probe-report",
        str(probe),
        "--expected-probe-report-sha256",
        _sha(probe),
        "--output",
        str(output),
    ]


class ProduceOverBudgetQualitySkipTests(unittest.TestCase):
    def test_create_new_receipt_binds_source_specs_probe_and_input(self) -> None:
        mode = contract.W16_GLOBAL64_MODE
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            probe = _write_probe(root / "probe.json", mode, 90_000.0)
            output = root / "skip.json"
            self.assertEqual(producer.main(_argv(mode, probe, output)), 0)
            validated = selector.validate_quality_skip(
                mode,
                output,
                _sha(output),
                topology_gate_spec_sha256=_sha(TOPOLOGY_SPEC),
                quality_gate_spec_sha256=_sha(QUALITY_SPEC),
            )
            self.assertEqual(validated["status"], "skipped_over_eta_budget")
            self.assertEqual(
                validated["probe_report"]["sha256"], _sha(probe)
            )
            self.assertEqual(
                validated["source_binding"]["frozen_receipt_sha256"],
                "b" * 64,
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "refusing to overwrite immutable receipt",
            ):
                producer.main(_argv(mode, probe, output))

    def test_within_budget_probe_requires_full_quality(self) -> None:
        mode = contract.W8_GLOBAL64_MODE
        for eta in (86_400.0, 10_000.0):
            with self.subTest(eta=eta), tempfile.TemporaryDirectory() as raw:
                root = Path(raw).resolve()
                probe = _write_probe(root / "probe.json", mode, eta)
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "within 24 hours and requires full "
                    "e1/e2/e4/e8/e16/e32",
                ):
                    producer.main(_argv(mode, probe, root / "skip.json"))

    def test_probe_and_input_tampering_are_rejected(self) -> None:
        mode = contract.W16_GLOBAL512_MODE
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            probe = _write_probe(root / "probe.json", mode, 90_000.0)
            output = root / "skip.json"
            self.assertEqual(producer.main(_argv(mode, probe, output)), 0)

            original = json.loads(output.read_text(encoding="utf-8"))
            original.pop("receipt_sha256")
            original["topology_independent_input_sha256"] = "f" * 64
            original["receipt_sha256"] = contract.canonical_json_sha256(
                original
            )
            forged = root / "forged-skip.json"
            forged.write_text(
                json.dumps(original, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                selector.TopologySelectionError,
                "probe binding changed",
            ):
                selector.validate_quality_skip(
                    mode,
                    forged,
                    _sha(forged),
                    topology_gate_spec_sha256=_sha(TOPOLOGY_SPEC),
                    quality_gate_spec_sha256=_sha(QUALITY_SPEC),
                )

            probe.write_text(probe.read_text(encoding="utf-8") + " ")
            with self.assertRaisesRegex(
                selector.TopologySelectionError,
                "artifact changed|probe binding changed",
            ):
                selector.validate_quality_skip(
                    mode,
                    output,
                    _sha(output),
                    topology_gate_spec_sha256=_sha(TOPOLOGY_SPEC),
                    quality_gate_spec_sha256=_sha(QUALITY_SPEC),
                )

    def test_w1_builder_always_rejects_quality_skip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            mode = contract.OFFICIAL_W1_REFERENCE_MODE
            probe = _write_probe(root / "probe.json", mode, 90_000.0)
            with self.assertRaisesRegex(
                selector.TopologySelectionError,
                "mandatory quality reference",
            ):
                selector.build_quality_skip_receipt(
                    mode,
                    probe,
                    _sha(probe),
                    topology_gate_spec_sha256=_sha(TOPOLOGY_SPEC),
                    quality_gate_spec_sha256=_sha(QUALITY_SPEC),
                )

    def test_selector_cli_accepts_mixed_reports_and_skip_receipts(self) -> None:
        modes = list(contract.TOPOLOGY_SPECS)
        skipped_modes = {modes[2], modes[4]}
        probes = []
        reports = []
        skips = []
        for index, (mode, specification) in enumerate(
            contract.TOPOLOGY_SPECS.items()
        ):
            eta = 90_000.0 if mode in skipped_modes else 10_000.0 + index
            probe = {
                "mode": mode,
                "status": "pass",
                "report_path": f"/probes/{mode}.json",
                "report_sha256": f"{index + 1:x}" * 64,
                "classification": specification["classification"],
                "precision": specification["precision"],
                "formal_training_eligible": True,
                "topology_independent_input_sha256": "a" * 64,
                "median_seconds": 1.0,
                "p90_seconds": 1.1,
                "p99_seconds": 1.2,
                "estimated_training_seconds": eta,
                "samples_per_second": 1000.0,
            }
            probes.append(probe)
            if mode in skipped_modes:
                skips.append(
                    {
                        "mode": mode,
                        "status": "skipped_over_eta_budget",
                        "receipt_path": f"/skips/{mode}.json",
                        "receipt_sha256": "b" * 64,
                        "receipt_payload_sha256": "c" * 64,
                        "topology_gate_spec_sha256": "d" * 64,
                        "quality_gate_spec_sha256": "e" * 64,
                        "topology_independent_input_sha256": "a" * 64,
                        "source_binding": {
                            "frozen_receipt_sha256": "1" * 64,
                            "frozen_gate_compatibility_sha256": "2" * 64,
                            "topology_receipt_sha256": "3" * 64,
                        },
                        "probe_report": {
                            "path": probe["report_path"],
                            "sha256": probe["report_sha256"],
                            "bytes": 100,
                            "receipt_sha256": "4" * 64,
                        },
                        "estimated_training_seconds": eta,
                        "maximum_estimated_training_seconds": 86_400,
                    }
                )
            else:
                reports.append(
                    {
                        "mode": mode,
                        "report_path": f"/quality/{mode}.json",
                        "report_sha256": "f" * 64,
                        "topology_independent_input_sha256": "a" * 64,
                        "short_trajectory_receipt": {},
                        "val_inputs_receipt": {
                            "path": "/val/inputs.json",
                            "sha256": "5" * 64,
                            "bytes": 10,
                            "receipt_payload_sha256": "6" * 64,
                        },
                        "pipeline_receipt": {
                            "path": "/val/pipeline.json",
                            "sha256": "7" * 64,
                            "bytes": 10,
                            "receipt_payload_sha256": "8" * 64,
                        },
                        "candidate_fgd": {
                            str(epoch): 0.5
                            for epoch in selector.QUALITY_EPOCHS
                        },
                        "candidates": [],
                    }
                )
        argv = [
            "--gate-spec",
            "/spec/topology.json",
            "--expected-gate-spec-sha256",
            "d" * 64,
            "--quality-gate-spec",
            "/spec/quality.json",
            "--expected-quality-gate-spec-sha256",
            "e" * 64,
        ]
        for mode in modes:
            argv.extend(["--probe", mode, f"/probes/{mode}.json", "8" * 64])
        for mode in modes:
            option = "--quality-skip" if mode in skipped_modes else "--quality-report"
            root = "/skips" if mode in skipped_modes else "/quality"
            argv.extend([option, mode, f"{root}/{mode}.json", "9" * 64])
        argv.extend(["--output", "/selection/output.json"])
        published: list[dict[str, object]] = []
        with (
            mock.patch.object(
                selector.contract,
                "validate_topology_gate_spec",
                return_value={"sha256": "d" * 64},
            ),
            mock.patch.object(
                selector,
                "validate_quality_gate_spec",
                return_value={"sha256": "e" * 64},
            ),
            mock.patch.object(
                selector,
                "validate_probe",
                side_effect=probes,
            ),
            mock.patch.object(
                selector,
                "validate_quality_report",
                side_effect=reports,
            ),
            mock.patch.object(
                selector,
                "validate_quality_skip",
                side_effect=skips,
            ),
            mock.patch.object(
                selector.contract,
                "_write_new_json",
                side_effect=lambda _path, payload: published.append(payload),
            ),
        ):
            self.assertEqual(selector.main(argv), 0)
        self.assertEqual(len(published), 1)
        self.assertEqual(
            [item["mode"] for item in published[0]["quality_skips"]],
            [modes[2], modes[4]],
        )


if __name__ == "__main__":
    unittest.main()
