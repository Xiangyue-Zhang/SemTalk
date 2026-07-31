from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import show_base_train as formal
from test_smplx_training_pool_gate import (
    FormalPoolGateFixture,
    argv_sha256,
    base_inference,
    build_features,
    sha256,
)


class TargetOffloadFormalPoolGateFixture(FormalPoolGateFixture):
    """Convert the full sharded fixture into a target-offload report."""

    mode = "target_offload"
    helpers = [1]
    helper_csv = "1"
    visible = "0,1"
    visible_count = 2
    minimum_speedup = 1.03

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        spec = formal.SMPLX_TRAINING_POOL_MODE_SPECS[self.mode]
        self.gate_source["harness_sha256"] = spec["harness_sha256"]
        self.gate_source["adapter_sha256"] = spec["adapter_sha256"]

        self.runner["argv"] = [
            "python",
            "/tmp/globaldiff_guarded_runner.py",
            "--gpus",
            "0,1,2,3,4,5,6,7",
        ]
        self.runner["argv_sha256"] = argv_sha256(self.runner["argv"])

        asset_root = self.root / "asset_root"
        asset = (
            asset_root
            / "smplx_models"
            / "smplx"
            / formal.FORMAL_SMPLX_FILENAME
        )
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"fixture-smplx-asset\n")
        self.smplx.update(
            {
                "format": "semtalk_show_smplx_asset_v1",
                "filename": formal.FORMAL_SMPLX_FILENAME,
                "path": str(asset.resolve()),
                "bytes": asset.stat().st_size,
                "regular_file": True,
                "symlink": False,
            }
        )
        self.asset = asset
        numerical_script = self.root / "quick_numerical_gate.py"
        numerical_script.write_bytes(b"numerical fixture\n")
        numerical_report = self.root / "quick_numerical_report.json"
        exact_numeric = {
            "dtype": "torch.float32",
            "equal": True,
            "max_abs": 0.0,
            "mismatch_count": 0,
        }
        self.write_json(
            numerical_report,
            {
                "format": "semtalk_smplx_full_batch_grad_quick_gate_v1",
                "status": "pass",
                "asset_sha256": formal.FORMAL_SMPLX_SHA256,
                "batch": 4096,
                "clips": 64,
                "frames": 64,
                "device_names": ["NVIDIA H200"] * 8,
                "comparisons": {
                    "vertices": copy.deepcopy(exact_numeric),
                    "loss": copy.deepcopy(exact_numeric),
                    "input_gradients": {
                        name: copy.deepcopy(exact_numeric)
                        for name in (
                            "body_pose",
                            "expression",
                            "global_orient",
                            "jaw_pose",
                            "left_hand_pose",
                            "right_hand_pose",
                        )
                    },
                },
            },
        )
        numerical_status = self.root / "quick_numerical_status.json"
        self.write_json(
            numerical_status,
            {
                "state": "finished",
                "return_code": 0,
                "error": None,
                "cleanup_error": None,
                "restore_error": None,
                "received_signal": None,
                "command": [
                    "/usr/bin/python3.12",
                    str(numerical_script.resolve()),
                    "--asset-root",
                    str(asset_root.resolve()),
                    "--report",
                    str(numerical_report.resolve()),
                    "--clips",
                    "64",
                    "--frames",
                    "64",
                ],
                "wrapper_pid": 12000,
                "child_pid": 12001,
                "restored_guards": {
                    str(index): 12100 + index
                    for index in range(8)
                },
            },
        )
        self.numerical_files = {
            "numerical report": numerical_report,
            "numerical script": numerical_script,
            "numerical runner status": numerical_status,
        }
        preliminary = {
            "classification": "preliminary_not_source_bound",
            "authorization": False,
            "files": {
                label: {
                    "path": str(file.resolve()),
                    "sha256": sha256(file),
                }
                for label, file in self.numerical_files.items()
            },
            "format": "semtalk_smplx_full_batch_grad_quick_gate_v1",
            "batch": 4096,
            "clips": 64,
            "frames": 64,
            "vertices_loss_six_input_gradients_byte_exact": True,
            "runner_status": {
                "wrapper_pid": 12000,
                "child_pid": 12001,
                "return_code": 0,
                "restored_guards": {
                    str(index): 12100 + index
                    for index in range(8)
                },
            },
        }

        self.semantic_common["format"] = spec["gate_format"]
        self.semantic_common["source"] = self.gate_source
        self.semantic_common["runtime"]["visible"] = self.visible
        self.semantic_common["runtime"]["device_names"] = [
            "NVIDIA H200"
        ] * self.visible_count
        self.semantic_common[
            "preliminary_full_batch_numerical_gate"
        ] = preliminary

        report = self.report
        report["format"] = spec["gate_format"]
        report["topology"] = {
            "primary_device": 0,
            "helper_devices": self.helpers,
            "visible_device_count": self.visible_count,
            "device_name": "NVIDIA H200",
        }
        report["protocol"].update(
            {
                "candidate_mode": self.mode,
                "helper_devices": self.helpers,
                "visible_device_count": self.visible_count,
                "minimum_speedup": self.minimum_speedup,
                "bounded_tolerances": None,
            }
        )
        report["preflight"].update(
            {
                "format": spec["gate_format"],
                "source": self.gate_source,
                "candidate_mode": self.mode,
                "helper_devices": self.helpers,
                "preliminary_numerical_gate": preliminary,
            }
        )
        report["semantic_common"] = self.semantic_common

        exact_state = {
            "comparison": "byte_exact",
            "pass": True,
            "canonical_sha256": "5" * 64,
        }
        cross_proof = {
            "status": "pass",
            "real_batches_exact": True,
            "comparison": "cross_mode_byte_exact",
            "states": {
                state: copy.deepcopy(exact_state)
                for state in ("initial", "step_1", "step_2", "final")
            },
            "numeric_scope": (
                "loss_tracker_gradients_model_optimizer_scheduler_rvq_ema_rng"
            ),
            "initial_byte_exact": True,
            "two_complete_updates_and_final": "byte_exact",
        }
        report["equivalence"]["stock_candidate_0"] = copy.deepcopy(
            cross_proof
        )
        report["equivalence"]["stock_candidate_1"] = copy.deepcopy(
            cross_proof
        )
        for metric in report["performance"].values():
            if isinstance(metric, dict):
                metric["minimum_required"] = self.minimum_speedup

        for child in report["children"]:
            argv = self._target_argv(child["argv"])
            child["argv"] = argv
            child["argv_sha256"] = argv_sha256(argv)
            result_path = Path(child["result_path"])
            with result_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            result["format"] = spec["gate_format"]
            result["semantic"]["common"] = copy.deepcopy(
                self.semantic_common
            )
            if result["mode"] == "candidate":
                result["semantic"]["mode"]["smplx_parallel_mode"] = self.mode
                result["semantic"]["mode"]["helper_devices"] = self.helpers
            result["ancestry"][0]["argv"] = argv
            result["ancestry"][0]["argv_sha256"] = argv_sha256(argv)
            result["ancestry"][1:] = copy.deepcopy(self.guarded)
            self.write_json(result_path, result)
            child["result_sha256"] = sha256(result_path)
        self.write()

    @staticmethod
    def _replace_cli_value(
        argv: list[str],
        flag: str,
        value: str,
    ) -> None:
        indices = [
            index
            for index, item in enumerate(argv)
            if item == flag
        ]
        if len(indices) != 1 or indices[0] + 1 >= len(argv):
            raise AssertionError(f"fixture requires one {flag}")
        argv[indices[0] + 1] = value

    def _target_argv(self, source: list[str]) -> list[str]:
        argv = list(source)
        for flag, value in (
            ("--candidate-mode", self.mode),
            ("--helper-devices", self.helper_csv),
            ("--expected-visible-device-count", str(self.visible_count)),
            ("--runner-gpus", "0,1,2,3,4,5,6,7"),
            (
                "--expected-runner-argv-sha256",
                self.runner["argv_sha256"],
            ),
            ("--minimum-speedup", str(self.minimum_speedup)),
        ):
            self._replace_cli_value(argv, flag, value)
        preliminary = self.report["preflight"]["preliminary_numerical_gate"]
        argv.extend(
            [
                "--asset-root",
                str(self.asset.parents[2]),
                "--numerical-gate-report",
                preliminary["files"]["numerical report"]["path"],
                "--expected-numerical-gate-report-sha256",
                preliminary["files"]["numerical report"]["sha256"],
                "--numerical-gate-script",
                preliminary["files"]["numerical script"]["path"],
                "--expected-numerical-gate-script-sha256",
                preliminary["files"]["numerical script"]["sha256"],
                "--numerical-gate-runner-status",
                preliminary["files"]["numerical runner status"]["path"],
                "--expected-numerical-gate-runner-status-sha256",
                preliminary["files"]["numerical runner status"]["sha256"],
            ]
        )
        return argv

    def args(self) -> SimpleNamespace:
        return SimpleNamespace(
            formal_stage="face",
            smplx_training_pool_mode=self.mode,
            smplx_training_helper_devices=self.helper_csv,
            smplx_training_pool_gate_report=str(self.report_path),
            expected_smplx_training_pool_gate_sha256=sha256(
                self.report_path
            ),
        )

    def validate(self) -> dict[str, object]:
        real_sha256 = formal._sha256
        target_spec = formal.SMPLX_TRAINING_POOL_MODE_SPECS[self.mode]
        bindings = {
            self.harness.resolve(): target_spec["harness_sha256"],
            self.adapter.resolve(): target_spec["adapter_sha256"],
            self.asset.resolve(): formal.FORMAL_SMPLX_SHA256,
        }

        def bound_sha256(path: Path) -> str:
            resolved = Path(path).resolve()
            if resolved in bindings:
                return bindings[resolved]
            return real_sha256(path)

        with mock.patch.object(
            formal,
            "_sha256",
            side_effect=bound_sha256,
        ):
            receipt = formal._formal_smplx_training_pool_gate_receipt(
                self.args(),
                current_source=self.source,
                representation=self.representation,
                smplx_asset_receipt=self.smplx,
            )
        assert receipt is not None
        return receipt

    @staticmethod
    def runtime_evidence(
        gate_receipt: dict[str, object],
        *,
        completed_forward_pairs: int = 2,
    ) -> dict[str, object]:
        body = {
            "format": (
                formal.SMPLX_TRAINING_POOL_TARGET_OFFLOAD_RUNTIME_EVIDENCE_FORMAT
            ),
            "formal_stage": "face",
            "mode": "target_offload",
            "gate_report_sha256": gate_receipt["sha256"],
            "gate_topology_sha256": gate_receipt["topology_sha256"],
            "source_binding": copy.deepcopy(
                gate_receipt["source_binding"]
            ),
            "pool_runtime": {
                "format": "semtalk_smplx_training_pool_v2",
                "mode": "target_offload",
                "primary_device": 0,
                "helper_devices": [1],
                "replica_devices": [0, 1],
                "partition": (
                    "primary_stock_full_batch_reconstruction_and_"
                    "helper_full_batch_target"
                ),
                "transfer_to_primary": (
                    "detached_target_full_outputs_only"
                ),
                "completed_forward_pairs": completed_forward_pairs,
                "last_forward": {
                    "stage": "face",
                    "total_rows": 4096,
                    "clip_length": 64,
                    "total_clips": 64,
                    "helpers": [1],
                    "reconstruction_device": 0,
                    "reconstruction_execution": (
                        "stock_full_batch_primary_current_stream"
                    ),
                    "target_device": 1,
                    "target_execution": "detached_full_batch_helper",
                    "transfer_to_primary": (
                        "detached_target_full_outputs_only"
                    ),
                    "output_keys": ["vertices"],
                },
            },
        }
        return {
            **body,
            "receipt_sha256": formal._payload_sha256(body),
        }


class TargetOffloadFormalGateContractTest(unittest.TestCase):
    def test_target_report_passes_production_validator_end_to_end(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = TargetOffloadFormalPoolGateFixture(Path(directory))
            receipt = fixture.validate()

        spec = formal.SMPLX_TRAINING_POOL_MODE_SPECS["target_offload"]
        self.assertEqual(receipt["format"], spec["gate_format"])
        self.assertEqual(receipt["mode"], "target_offload")
        self.assertEqual(receipt["helper_devices"], [1])
        self.assertEqual(receipt["runtime"]["visible_device_count"], 2)
        self.assertEqual(receipt["runtime"]["device_names"], ["NVIDIA H200"] * 2)
        self.assertEqual(receipt["performance"]["minimum_speedup"], 1.03)
        self.assertEqual(
            receipt["equivalence"]["cross_mode"],
            "byte_exact",
        )
        self.assertNotIn(
            "bounded_tolerances",
            receipt["equivalence"],
        )
        self.assertEqual(
            receipt["gate_artifacts"]["harness_sha256"],
            spec["harness_sha256"],
        )
        self.assertEqual(
            receipt["gate_artifacts"]["adapter_sha256"],
            spec["adapter_sha256"],
        )
        self.assertEqual(receipt["children"]["count"], 8)
        self.assertEqual(receipt["children"]["fresh_processes"], 8)
        self.assertEqual(len(receipt["children"]["result_sha256"]), 8)

    def test_target_report_rejects_non_none_bounded_tolerances(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = TargetOffloadFormalPoolGateFixture(Path(directory))
            fixture.report["protocol"]["bounded_tolerances"] = copy.deepcopy(
                formal.SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES
            )
            fixture.write()
            with self.assertRaises(RuntimeError):
                fixture.validate()

    def test_target_child_runner_gpus_match_outer_guarded_runner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = TargetOffloadFormalPoolGateFixture(Path(directory))
            child = fixture.report["children"][0]
            argv = list(child["argv"])
            fixture._replace_cli_value(argv, "--runner-gpus", fixture.visible)
            child["argv"] = argv
            child["argv_sha256"] = argv_sha256(argv)
            result_path = Path(child["result_path"])
            with result_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            result["ancestry"][0]["argv"] = argv
            result["ancestry"][0]["argv_sha256"] = argv_sha256(argv)
            fixture.write_json(result_path, result)
            child["result_sha256"] = sha256(result_path)
            fixture.write()
            with self.assertRaises(RuntimeError):
                fixture.validate()

    def test_target_outer_guarded_runner_requires_all_eight_gpus(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = TargetOffloadFormalPoolGateFixture(Path(directory))
            fixture.runner["argv"] = [
                "python",
                "/tmp/globaldiff_guarded_runner.py",
                "--gpus",
                "0,1",
            ]
            fixture.runner["argv_sha256"] = argv_sha256(
                fixture.runner["argv"]
            )
            for child in fixture.report["children"]:
                argv = list(child["argv"])
                fixture._replace_cli_value(argv, "--runner-gpus", "0,1")
                fixture._replace_cli_value(
                    argv,
                    "--expected-runner-argv-sha256",
                    fixture.runner["argv_sha256"],
                )
                child["argv"] = argv
                child["argv_sha256"] = argv_sha256(argv)
                result_path = Path(child["result_path"])
                with result_path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
                result["ancestry"][0]["argv"] = argv
                result["ancestry"][0]["argv_sha256"] = argv_sha256(argv)
                result["ancestry"][1:] = copy.deepcopy(fixture.guarded)
                fixture.write_json(result_path, result)
                child["result_sha256"] = sha256(result_path)
            fixture.write()
            with self.assertRaises(RuntimeError):
                fixture.validate()

    def test_target_preliminary_artifacts_are_rehashed(self) -> None:
        for label in (
            "numerical report",
            "numerical script",
            "numerical runner status",
        ):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = TargetOffloadFormalPoolGateFixture(
                        Path(directory)
                    )
                    fixture.numerical_files[label].write_bytes(b"tampered\n")
                    with self.assertRaises(RuntimeError):
                        fixture.validate()

    def test_target_gate_and_runtime_pass_both_final_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = TargetOffloadFormalPoolGateFixture(Path(directory))
            receipt = fixture.validate()
            runtime = fixture.runtime_evidence(receipt)
            key = formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
            runtime_key = (
                formal.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY
            )
            dataset_receipt = {
                "data_mdb_sha256": receipt["representation"]["data_sha256"],
                "summary_sha256": receipt["representation"][
                    "summary_sha256"
                ],
                "lineage_sha256": receipt["representation"][
                    "lineage_sha256"
                ],
                "smplx_asset": copy.deepcopy(fixture.smplx),
                key: receipt,
            }
            audit = {
                "source_receipt": copy.deepcopy(fixture.source),
                "smplx_training_pool_mode": "target_offload",
                "optimizer_updates": 2,
                runtime_key: runtime,
                key: receipt,
            }
            status = {
                "smplx_training_pool_mode": "target_offload",
                "optimizer_updates": 2,
                runtime_key: runtime,
                key: receipt,
            }
            real_feature_sha = build_features.sha256
            real_inference_sha = base_inference.sha256_file

            def feature_sha(path: Path) -> str:
                if Path(path).resolve() == fixture.asset.resolve():
                    return formal.FORMAL_SMPLX_SHA256
                return real_feature_sha(path)

            def inference_sha(path: str | Path) -> str:
                if Path(path).resolve() == fixture.asset.resolve():
                    return formal.FORMAL_SMPLX_SHA256
                return real_inference_sha(path)

            with mock.patch.object(
                build_features,
                "sha256",
                side_effect=feature_sha,
            ):
                build_features.validate_smplx_training_pool_gate_binding(
                    formal_stage="face",
                    audit=audit,
                    dataset_receipt=dataset_receipt,
                    status=status,
                    path=Path("/formal/checkpoint.bin"),
                )
            with mock.patch.object(
                base_inference,
                "sha256_file",
                side_effect=inference_sha,
            ):
                base_inference._validate_smplx_training_pool_gate_binding(
                    formal_stage="face",
                    audit=audit,
                    dataset_receipt=dataset_receipt,
                    status=status,
                    path=Path("/formal/checkpoint.bin"),
                )


if __name__ == "__main__":
    unittest.main()
