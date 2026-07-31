from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import show_base_train as formal
from utils import config


REPOSITORY = Path(__file__).resolve().parents[1]


def load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name,
        REPOSITORY / relative_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_features = load_script_module(
    "test_smplx_pool_build_base_features",
    "scripts/show_base/build_base_features.py",
)
base_inference = load_script_module(
    "test_smplx_pool_run_base_inference",
    "scripts/show_base/run_base_inference.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(
        b"\0".join(item.encode() for item in argv) + b"\0"
    ).hexdigest()


class FormalPoolGateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.report_path = root / "formal_gate_report.json"
        self.harness = root / "gate.py"
        self.adapter = root / "adapter.py"
        self.reference = root / "reference.py"
        self.coordinator = root / "run_parallel_formal_gate.sh"
        for path, content in (
            (self.harness, b"harness\n"),
            (self.adapter, b"adapter\n"),
            (self.reference, b"reference\n"),
            (self.coordinator, b"#!/bin/bash\n"),
        ):
            path.write_bytes(content)
        self.source = {
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "commit": "a" * 40,
            "tree": "b" * 40,
        }
        implementation_files = {
            relative: sha256(REPOSITORY / relative)
            for relative in formal.SMPLX_TRAINING_POOL_IMPLEMENTATION_FILES
        }
        self.gate_source = {
            **self.source,
            "implementation_files": implementation_files,
            "reference_gate": str(self.reference),
            "reference_gate_sha256": sha256(self.reference),
            "adapter": str(self.adapter),
            "adapter_sha256": sha256(self.adapter),
            "harness_sha256": sha256(self.harness),
            "coordinator": {
                "path": str(self.coordinator.resolve()),
                "sha256": sha256(self.coordinator),
            },
        }
        self.representation = {
            "lmdb": str((root / "representation_lmdb").resolve()),
            "data_sha256": "c" * 64,
            "summary": str(
                (root / "representation_summary.json").resolve()
            ),
            "summary_sha256": "d" * 64,
            "lineage": str(
                (root / "representation_summary.json").resolve()
            ),
            "lineage_sha256": "d" * 64,
            "entry_aggregate_sha256": "e" * 64,
        }
        self.smplx = {"sha256": formal.FORMAL_SMPLX_SHA256}
        self.public = {
            "pid": 9000,
            "ppid": 8500,
            "starttime": "900000",
            "argv": ["python", str(self.harness)],
            "argv_sha256": "1" * 64,
        }
        coordinator_argv = ["bash", str(self.coordinator.resolve())]
        self.coordinator_identity = {
            "pid": 8500,
            "ppid": 8000,
            "starttime": "850000",
            "argv": coordinator_argv,
            "argv_sha256": argv_sha256(coordinator_argv),
        }
        self.gate_source["coordinator"]["identity"] = copy.deepcopy(
            self.coordinator_identity
        )
        runner_argv = [
            "python",
            "/tmp/globaldiff_guarded_runner.py",
            "--gpus",
            "0,1,2",
        ]
        self.runner = {
            "pid": 8000,
            "ppid": 1,
            "starttime": "800000",
            "argv": runner_argv,
            "argv_sha256": argv_sha256(runner_argv),
        }
        self.guarded = [
            self.public,
            self.coordinator_identity,
            self.runner,
        ]
        preliminary = {"status": "pass", "authorization": False}
        self.semantic_common = {
            "format": formal.SMPLX_TRAINING_POOL_GATE_FORMAT,
            "source": self.gate_source,
            "stage": "face",
            "dataset": "show_base",
            "speaker_scope": "All",
            "speaker_ids": [0, 1, 2, 3],
            "batch_size": 64,
            "frames": 64,
            "representation": self.representation,
            "smplx_asset_sha256": formal.FORMAL_SMPLX_SHA256,
            "preliminary_full_batch_numerical_gate": preliminary,
            "runtime": {
                "visible": "0,1,2",
                "device_names": ["NVIDIA H200"] * 3,
                "torch": "2.7.1+cu128",
                "cuda": "12.8",
                "cudnn": 91002,
            },
        }
        specs = [
            ("equivalence-stock-0", "equivalence", "stock"),
            ("equivalence-candidate-0", "equivalence", "candidate"),
            ("equivalence-stock-1", "equivalence", "stock"),
            ("equivalence-candidate-1", "equivalence", "candidate"),
            ("benchmark-0-stock", "benchmark", "stock"),
            ("benchmark-1-candidate", "benchmark", "candidate"),
            ("benchmark-2-candidate", "benchmark", "candidate"),
            ("benchmark-3-stock", "benchmark", "stock"),
        ]
        children = []
        for sequence, (name, kind, mode) in enumerate(specs):
            snapshot = root / name
            snapshot.mkdir()
            argv = self.child_argv(name, snapshot)
            pid = 10000 + sequence
            result = {
                "format": formal.SMPLX_TRAINING_POOL_GATE_FORMAT,
                "status": "pass",
                "kind": kind,
                "mode": mode,
                "semantic": {
                    "common": self.semantic_common,
                    "mode": {
                        "label": mode,
                        "smplx_parallel_mode": (
                            "disabled"
                            if mode == "stock"
                            else "sharded_local_loss"
                        ),
                        "helper_devices": (
                            [] if mode == "stock" else [1, 2]
                        ),
                        "adapter_receipt": {"status": "pass"},
                    },
                },
                "ancestry": [
                    {
                        "pid": pid,
                        "ppid": self.public["pid"],
                        "starttime": str(700000 + sequence),
                        "argv": argv,
                        "argv_sha256": argv_sha256(argv),
                    },
                    *self.guarded,
                ],
            }
            if kind == "equivalence":
                result.update(
                    {
                        "batches": [["batch", 0], ["batch", 1]],
                        "states": {
                            state: {
                                "path": str(snapshot / f"{state}.pt"),
                                "file_sha256": "2" * 64,
                                "canonical_sha256": "3" * 64,
                            }
                            for state in (
                                "initial",
                                "step_1",
                                "step_2",
                                "final",
                            )
                        },
                        "optimizer_steps": 2,
                        "formal_updates": 2,
                        "scheduler_steps": 1,
                    }
                )
            else:
                stock = mode == "stock"
                result.update(
                    {
                        "initial_state_sha256": "4" * 64,
                        "timing": {
                            "mode": mode,
                            "block": sequence - 4,
                            "warmup": 5,
                            "measured": 25,
                            "wall_seconds": [2.0 if stock else 1.0] * 25,
                            "cuda_seconds": [4.0 if stock else 2.0] * 25,
                            "batches": [["batch", index] for index in range(30)],
                        },
                        "optimizer_steps": 30,
                        "formal_updates": 30,
                        "scheduler_steps": 1,
                    }
                )
            result_path = snapshot / "result.json"
            self.write_json(result_path, result)
            children.append(
                {
                    "sequence": sequence,
                    "name": name,
                    "kind": kind,
                    "mode": mode,
                    "pid": pid,
                    "argv": argv,
                    "argv_sha256": argv_sha256(argv),
                    "return_code": 0,
                    "result_path": str(result_path),
                    "result_sha256": sha256(result_path),
                }
            )
        exact_state = {
            "comparison": "byte_exact",
            "pass": True,
            "canonical_sha256": "5" * 64,
        }
        bounded_state = {
            "comparison": (
                "bounded_float_exact_discrete_rng_and_smplx"
            ),
            "pass": True,
            "tolerances": {
                category: {
                    "atol": tolerance["atol"],
                    "rtol": tolerance["rtol"],
                    "floating_nodes": 1,
                    "elements": 2,
                    "bitwise_mismatch_elements": 1,
                    "max_abs": tolerance["atol"] / 2,
                    "max_normalized_error": 0.5,
                }
                for category, tolerance in (
                    formal.SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES.items()
                )
            },
            "exact_nodes": 1,
            "all_floating_values_within_tolerance": True,
            "rng_indices_scheduler_smplx_and_nonfloating_exact": True,
        }
        repeat_proof = {
            "status": "pass",
            "real_batches_exact": True,
            "comparison": "repeat_byte_exact",
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
        cross_proof = {
            "status": "pass",
            "real_batches_exact": True,
            "comparison": "cross_mode_bounded",
            "states": {
                "initial": copy.deepcopy(exact_state),
                "step_1": copy.deepcopy(bounded_state),
                "step_2": copy.deepcopy(bounded_state),
                "final": copy.deepcopy(bounded_state),
            },
            "numeric_scope": (
                "loss_tracker_gradients_model_optimizer_scheduler_rvq_ema_rng"
            ),
            "initial_byte_exact": True,
            "two_complete_updates_and_final": (
                "bounded_float_exact_discrete_rng_and_smplx"
            ),
        }
        metric = {
            "stock_median_seconds": 2.0,
            "candidate_median_seconds": 1.0,
            "pooled_speedup": 2.0,
            "paired_speedups": [2.0, 2.0],
            "minimum_required": 1.50,
            "pass": True,
        }
        self.report = {
            "format": formal.SMPLX_TRAINING_POOL_GATE_FORMAT,
            "status": "pass",
            "authorization": True,
            "scope": {
                "stage": "face",
                "dataset": "show_base",
                "speaker_scope": "All",
                "speaker_ids": [0, 1, 2, 3],
                "forbidden": ["speaker2-only", "SemGate", "sparse motion"],
            },
            "topology": {
                "primary_device": 0,
                "helper_devices": [1, 2],
                "visible_device_count": 3,
                "device_name": "NVIDIA H200",
            },
            "protocol": {
                "candidate_mode": "sharded_local_loss",
                "helper_devices": [1, 2],
                "visible_device_count": 3,
                "equivalence_updates": 2,
                "abba": ["stock", "candidate", "candidate", "stock"],
                "warmup": 5,
                "measured": 25,
                "minimum_speedup": 1.50,
                "bounded_tolerances": copy.deepcopy(
                    formal.SMPLX_TRAINING_POOL_GATE_BOUNDED_TOLERANCES
                ),
            },
            "preflight": {
                "format": formal.SMPLX_TRAINING_POOL_GATE_FORMAT,
                "source": self.gate_source,
                "stage": "face",
                "candidate_mode": "sharded_local_loss",
                "helper_devices": [1, 2],
                "real_show_all": True,
                "preliminary_numerical_gate": preliminary,
            },
            "semantic_common": self.semantic_common,
            "equivalence": {
                "stock_repeat": copy.deepcopy(repeat_proof),
                "candidate_repeat": copy.deepcopy(repeat_proof),
                "stock_candidate_0": copy.deepcopy(cross_proof),
                "stock_candidate_1": copy.deepcopy(cross_proof),
            },
            "performance": {
                "order": ["stock", "candidate", "candidate", "stock"],
                "fresh_processes": 4,
                "wall": copy.deepcopy(metric),
                "cuda": {
                    **metric,
                    "stock_median_seconds": 4.0,
                    "candidate_median_seconds": 2.0,
                },
            },
            "guarded_ancestry": self.guarded,
            "children": children,
            "started_unix": 1.0,
            "completed_unix": 2.0,
        }
        self.write()

    def child_argv(self, name: str, snapshot: Path) -> list[str]:
        argv = [
            "python",
            str(self.harness),
            "--repository",
            str(REPOSITORY),
            "--adapter",
            str(self.adapter),
            "--stage",
            "face",
            "--candidate-mode",
            "sharded_local_loss",
            "--helper-devices",
            "1,2",
            "--representation-lmdb",
            self.representation["lmdb"],
            "--representation-summary",
            self.representation["summary"],
            "--representation-lineage",
            self.representation["lineage"],
            "--expected-representation-data-sha256",
            self.representation["data_sha256"],
            "--expected-representation-summary-sha256",
            self.representation["summary_sha256"],
            "--expected-representation-lineage-sha256",
            self.representation["lineage_sha256"],
            "--expected-representation-entry-aggregate-sha256",
            self.representation["entry_aggregate_sha256"],
            "--expected-smplx-asset-sha256",
            formal.FORMAL_SMPLX_SHA256,
            "--expected-source-commit",
            self.source["commit"],
            "--expected-source-tree",
            self.source["tree"],
        ]
        for relative in sorted(
            formal.SMPLX_TRAINING_POOL_IMPLEMENTATION_FILES
        ):
            argv.extend(["--implementation-file", relative])
        argv.extend(
            [
            "--expected-device-name",
            "NVIDIA H200",
            "--expected-visible-device-count",
            "3",
            "--runner-gpus",
            "0,1,2",
            "--expected-runner-pid",
            str(self.runner["pid"]),
            "--expected-runner-starttime",
            self.runner["starttime"],
            "--expected-runner-argv-sha256",
            self.runner["argv_sha256"],
            "--coordinator-pid",
            str(self.coordinator_identity["pid"]),
            "--coordinator-script",
            str(self.coordinator.resolve()),
            "--expected-coordinator-script-sha256",
            sha256(self.coordinator),
            "--expected-coordinator-starttime",
            self.coordinator_identity["starttime"],
            "--expected-coordinator-argv-sha256",
            self.coordinator_identity["argv_sha256"],
            "--minimum-speedup",
            "1.50",
            "--output-root",
            str(self.root),
            "--internal-mode",
            name,
            "--snapshot-dir",
            str(snapshot),
            "--internal-parent-pid",
            str(self.public["pid"]),
            ]
        )
        return argv

    @staticmethod
    def write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write(self) -> str:
        self.write_json(self.report_path, self.report)
        return sha256(self.report_path)

    def args(self) -> SimpleNamespace:
        return SimpleNamespace(
            formal_stage="face",
            smplx_training_pool_mode="sharded_local_loss",
            smplx_training_helper_devices="1,2",
            smplx_training_pool_gate_report=str(self.report_path),
            expected_smplx_training_pool_gate_sha256=sha256(
                self.report_path
            ),
        )

    def validate(self) -> dict[str, object]:
        with mock.patch.dict(
            formal.SMPLX_TRAINING_POOL_MODE_SPECS[
                "sharded_local_loss"
            ],
            {
                "harness_sha256": sha256(self.harness),
                "adapter_sha256": sha256(self.adapter),
            },
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
    ) -> dict[str, object]:
        body = {
            "format": (
                formal.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_FORMAT
            ),
            "formal_stage": "face",
            "mode": "sharded_local_loss",
            "gate_report_sha256": gate_receipt["sha256"],
            "gate_topology_sha256": gate_receipt["topology_sha256"],
            "source_binding": copy.deepcopy(
                gate_receipt["source_binding"]
            ),
            "pool_runtime": {
                "format": "semtalk_smplx_training_pool_v2",
                "mode": "sharded_local_loss",
                "primary_device": 0,
                "helper_devices": [1, 2],
                "replica_devices": [0, 1, 2],
                "partition": "balanced_contiguous_whole_clips",
                "transfer_to_primary": (
                    "differentiable_scalar_numerators_only"
                ),
                "last_forward": {
                    "stage": "face",
                    "total_rows": 256,
                    "clip_length": 64,
                    "total_clips": 4,
                    "helpers": [1, 2],
                    "spans": [
                        {
                            "device": 1,
                            "row_start": 0,
                            "row_end": 128,
                            "clip_start": 0,
                            "clip_end": 2,
                        },
                        {
                            "device": 2,
                            "row_start": 128,
                            "row_end": 256,
                            "clip_start": 2,
                            "clip_end": 4,
                        },
                    ],
                    "component_counts": {
                        "ver": [3840, 3840],
                        "ver_vel": [3456, 3456],
                        "ver_acc": [3072, 3072],
                    },
                    "aggregation": (
                        "ordered_numerator_sum_over_exact_global_count"
                    ),
                },
            },
        }
        return {
            **body,
            "receipt_sha256": formal._payload_sha256(body),
        }


class SmplxTrainingPoolProductionTest(unittest.TestCase):
    def test_gate_artifact_shas_are_bound_per_mode_in_all_consumers(
        self,
    ) -> None:
        expected_target = {
            "harness_sha256": (
                "5423ed9a5a8076e23e72ffaad3104f94c81ea5b2cc7d48aec32d50c10e6772de"
            ),
            "adapter_sha256": (
                "2fd00b05eec9f68267ab74d36d9a074a2692465f55e422ed2c6477c25821a269"
            ),
        }
        expected_sharded = {
            "harness_sha256": (
                "01c77aa005039f9936f667b8b97052084be3e3364f64be402b53df383ed7ae2d"
            ),
            "adapter_sha256": (
                "5b3ac7fdeb4ca680036998159f64db7369430d4cc0e5852a49436d97f04aeeba"
            ),
        }
        for module in (formal, build_features, base_inference):
            with self.subTest(module=module.__name__):
                target = module.SMPLX_TRAINING_POOL_MODE_SPECS[
                    "target_offload"
                ]
                sharded = module.SMPLX_TRAINING_POOL_MODE_SPECS[
                    "sharded_local_loss"
                ]
                self.assertEqual(
                    {
                        key: target[key]
                        for key in expected_target
                    },
                    expected_target,
                )
                self.assertEqual(
                    {
                        key: sharded[key]
                        for key in expected_sharded
                    },
                    expected_sharded,
                )
                self.assertNotEqual(
                    target["harness_sha256"],
                    sharded["harness_sha256"],
                )
                self.assertNotEqual(
                    target["adapter_sha256"],
                    sharded["adapter_sha256"],
                )

    def test_parser_exposes_explicit_gate_roots(self) -> None:
        with mock.patch.object(sys, "argv", ["config"]):
            args = config.parse_args()
        self.assertIsNone(args.smplx_training_pool_gate_report)
        self.assertIsNone(
            args.expected_smplx_training_pool_gate_sha256
        )

    def test_device_matrix_is_exact_and_fail_closed(self) -> None:
        args = SimpleNamespace(
            formal_stage="face",
            smplx_training_pool_mode="sharded_local_loss",
            smplx_training_helper_devices="1,2",
            smplx_training_pool_gate_report="/formal/report.json",
            expected_smplx_training_pool_gate_sha256="a" * 64,
        )
        formal._validate_smplx_training_pool_device_matrix(
            args,
            world_size=1,
            cuda_available=True,
            visible_device_count=3,
        )
        four_device_args = copy.copy(args)
        four_device_args.formal_stage = "upper"
        four_device_args.smplx_training_helper_devices = "1,2,3"
        formal._validate_smplx_training_pool_device_matrix(
            four_device_args,
            world_size=1,
            cuda_available=True,
            visible_device_count=4,
        )
        with self.assertRaises(RuntimeError):
            formal._validate_smplx_training_pool_device_matrix(
                four_device_args,
                world_size=1,
                cuda_available=True,
                visible_device_count=3,
            )
        for field, value in (
            ("smplx_training_helper_devices", "1"),
            ("formal_stage", "global"),
            ("smplx_training_pool_gate_report", ""),
        ):
            with self.subTest(field=field):
                changed = copy.copy(args)
                setattr(changed, field, value)
                with self.assertRaises(RuntimeError):
                    formal._validate_smplx_training_pool_device_matrix(
                        changed,
                        world_size=1,
                        cuda_available=True,
                        visible_device_count=3,
                    )
        disabled = SimpleNamespace(
            formal_stage="global",
            smplx_training_pool_mode="disabled",
            smplx_training_helper_devices="",
            smplx_training_pool_gate_report=None,
            expected_smplx_training_pool_gate_sha256=None,
        )
        formal._validate_smplx_training_pool_device_matrix(
            disabled,
            world_size=1,
            cuda_available=True,
            visible_device_count=1,
        )
        for stage in formal.FORMAL_SMPLX_STAGES:
            with self.subTest(stock_stage=stage):
                stock = copy.copy(disabled)
                stock.formal_stage = stage
                formal._validate_smplx_training_pool_device_matrix(
                    stock,
                    world_size=1,
                    cuda_available=True,
                    visible_device_count=1,
                )
        for field, value, visible in (
            ("smplx_training_helper_devices", "1", 1),
            ("smplx_training_pool_gate_report", "/forbidden.json", 1),
            ("expected_smplx_training_pool_gate_sha256", "a" * 64, 1),
            ("formal_stage", "face", 2),
        ):
            with self.subTest(stock_rejects=field):
                stock = copy.copy(disabled)
                stock.formal_stage = "face"
                setattr(stock, field, value)
                with self.assertRaises(RuntimeError):
                    formal._validate_smplx_training_pool_device_matrix(
                        stock,
                        world_size=1,
                        cuda_available=True,
                        visible_device_count=visible,
                    )
        disabled.smplx_training_pool_gate_report = "/forbidden.json"
        with self.assertRaises(RuntimeError):
            formal._validate_smplx_training_pool_device_matrix(
                disabled,
                world_size=1,
                cuda_available=True,
                visible_device_count=1,
            )

    def test_target_offload_device_matrix_requires_exactly_two_gpus(
        self,
    ) -> None:
        args = SimpleNamespace(
            formal_stage="face",
            smplx_training_pool_mode="target_offload",
            smplx_training_helper_devices="1",
            smplx_training_pool_gate_report="/formal/report.json",
            expected_smplx_training_pool_gate_sha256="a" * 64,
        )
        formal._validate_smplx_training_pool_device_matrix(
            args,
            world_size=1,
            cuda_available=True,
            visible_device_count=2,
        )
        for helpers, visible in (("1,2", 3), ("2", 2), ("1", 3)):
            with self.subTest(helpers=helpers, visible=visible):
                changed = copy.copy(args)
                changed.smplx_training_helper_devices = helpers
                with self.assertRaises(RuntimeError):
                    formal._validate_smplx_training_pool_device_matrix(
                        changed,
                        world_size=1,
                        cuda_available=True,
                        visible_device_count=visible,
                    )

    def test_target_offload_runtime_evidence_is_fail_closed(self) -> None:
        topology = formal._smplx_training_pool_topology(
            "target_offload",
            (1,),
        )
        gate = {
            "mode": "target_offload",
            "sha256": "a" * 64,
            "topology_sha256": formal._payload_sha256(topology),
            "source_binding": {
                "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                "commit": "b" * 40,
                "tree": "c" * 40,
            },
            "topology": topology,
        }
        pool_runtime = {
            "format": "semtalk_smplx_training_pool_v2",
            "mode": "target_offload",
            "primary_device": 0,
            "helper_devices": [1],
            "replica_devices": [0, 1],
            "partition": (
                "primary_stock_full_batch_reconstruction_and_"
                "helper_full_batch_target"
            ),
            "transfer_to_primary": "detached_target_full_outputs_only",
            "completed_forward_pairs": 2,
            "last_forward": {
                "stage": "face",
                "total_rows": 128,
                "clip_length": 64,
                "total_clips": 2,
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
        }
        body = {
            "format": (
                formal.SMPLX_TRAINING_POOL_TARGET_OFFLOAD_RUNTIME_EVIDENCE_FORMAT
            ),
            "formal_stage": "face",
            "mode": "target_offload",
            "gate_report_sha256": gate["sha256"],
            "gate_topology_sha256": gate["topology_sha256"],
            "source_binding": gate["source_binding"],
            "pool_runtime": pool_runtime,
        }
        receipt = {
            **body,
            "receipt_sha256": formal._payload_sha256(body),
        }
        self.assertEqual(
            formal._validate_smplx_training_pool_runtime_evidence(
                receipt,
                formal_stage="face",
                gate_receipt=gate,
            ),
            receipt,
        )
        for validate, runtime_key in (
            (
                build_features.validate_smplx_training_pool_runtime_evidence,
                build_features.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY,
            ),
            (
                base_inference._validate_smplx_training_pool_runtime_evidence,
                base_inference.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY,
            ),
        ):
            with self.subTest(consumer=validate.__module__):
                validate(
                    formal_stage="face",
                    audit={
                        "smplx_training_pool_mode": "target_offload",
                        runtime_key: receipt,
                        "optimizer_updates": 2,
                    },
                    dataset_receipt={},
                    status={
                        "smplx_training_pool_mode": "target_offload",
                        runtime_key: receipt,
                        "optimizer_updates": 2,
                    },
                    gate_receipt=gate,
                    path=Path("/formal/checkpoint.bin"),
                )
        corrupted = copy.deepcopy(receipt)
        corrupted["pool_runtime"]["last_forward"][
            "reconstruction_execution"
        ] = "helper_stream"
        corrupted_body = dict(corrupted)
        corrupted_body.pop("receipt_sha256")
        corrupted["receipt_sha256"] = formal._payload_sha256(
            corrupted_body
        )
        with self.assertRaises(RuntimeError):
            formal._validate_smplx_training_pool_runtime_evidence(
                corrupted,
                formal_stage="face",
                gate_receipt=gate,
            )
        with self.assertRaises(RuntimeError):
            formal._validate_smplx_training_pool_runtime_evidence(
                receipt,
                formal_stage="face",
                gate_receipt=gate,
                expected_optimizer_updates=3,
            )
        count_mismatch = copy.deepcopy(receipt)
        count_mismatch["pool_runtime"]["completed_forward_pairs"] = 3
        count_mismatch_body = dict(count_mismatch)
        count_mismatch_body.pop("receipt_sha256")
        count_mismatch["receipt_sha256"] = formal._payload_sha256(
            count_mismatch_body
        )
        for validate, runtime_key in (
            (
                build_features.validate_smplx_training_pool_runtime_evidence,
                build_features.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY,
            ),
            (
                base_inference._validate_smplx_training_pool_runtime_evidence,
                base_inference.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY,
            ),
        ):
            with self.subTest(counter_consumer=validate.__module__):
                with self.assertRaises(RuntimeError):
                    validate(
                        formal_stage="face",
                        audit={
                            "smplx_training_pool_mode": "target_offload",
                            runtime_key: count_mismatch,
                            "optimizer_updates": 2,
                        },
                        dataset_receipt={},
                        status={
                            "smplx_training_pool_mode": "target_offload",
                            runtime_key: count_mismatch,
                            "optimizer_updates": 2,
                        },
                        gate_receipt=gate,
                        path=Path("/formal/checkpoint.bin"),
                    )

    def test_final_consumers_revalidate_pool_gate_artifacts(self) -> None:
        feature_revalidation = inspect.getsource(
            build_features.revalidate_checkpoint_records
        )
        inference_revalidation = inspect.getsource(
            base_inference._revalidate_frozen_inputs
        )
        for source in (feature_revalidation, inference_revalidation):
            with self.subTest(consumer=source.splitlines()[0]):
                self.assertIn(
                    "SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY",
                    source,
                )
                self.assertIn("pool_receipt", source)
                self.assertIn("sha256", source)

    def test_feature_completion_rejects_changed_pool_gate_artifact(
        self,
    ) -> None:
        def fixture(root: Path) -> tuple[
            dict[str, dict[str, object]],
            dict[str, Path],
        ]:
            records: dict[str, dict[str, object]] = {}
            reports: dict[str, Path] = {}
            for stage in (*build_features.RVQ_NAMES, "global"):
                checkpoint = root / f"{stage}.bin"
                status = root / f"{stage}.json"
                checkpoint.write_bytes(f"{stage}-checkpoint".encode())
                status.write_bytes(f"{stage}-status".encode())
                audit: dict[str, object] = {
                    "smplx_training_pool_mode": "disabled",
                }
                record: dict[str, object] = {
                    "path": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "formal_training_status": str(status),
                    "formal_training_status_sha256": sha256(status),
                    "audit": audit,
                }
                if stage == "upper":
                    audit["smplx_training_pool_mode"] = "target_offload"
                    report = root / f"{stage}-gate.json"
                    report.write_bytes(f"{stage}-gate".encode())
                    reports[stage] = report
                    audit[
                        build_features.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                    ] = {
                        "path": str(report),
                        "sha256": sha256(report),
                    }
                if stage == "global":
                    parity = root / "global-parity.json"
                    parity.write_bytes(b"global-parity")
                    record["global_fastpath_parity"] = {
                        "path": str(parity),
                        "sha256": sha256(parity),
                    }
                records[stage] = record
            return records, reports

        mutations = {
            "deleted": lambda path, root: path.unlink(),
            "tampered": lambda path, root: path.write_bytes(b"tampered"),
            "symlink": lambda path, root: (
                path.unlink(),
                (root / "replacement.json").write_bytes(b"replacement"),
                path.symlink_to(root / "replacement.json"),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                records, reports = fixture(root)
                build_features.revalidate_checkpoint_records(records)
                mutate(reports["upper"], root)
                with self.assertRaises(RuntimeError):
                    build_features.revalidate_checkpoint_records(records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, reports = fixture(root)
            records["global"]["audit"][
                build_features.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
            ] = {"path": "/forbidden", "sha256": "a" * 64}
            with self.assertRaises(RuntimeError):
                build_features.revalidate_checkpoint_records(records)
            stock_root = root / "stock"
            stock_root.mkdir()
            records, _ = fixture(stock_root)
            records["face"]["audit"][
                build_features.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
            ] = records["upper"]["audit"][
                build_features.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
            ]
            with self.assertRaises(RuntimeError):
                build_features.revalidate_checkpoint_records(records)

    def test_inference_completion_rejects_changed_pool_gate_artifact(
        self,
    ) -> None:
        def fixture(root: Path) -> tuple[
            dict[str, object],
            dict[str, dict[str, object]],
            Path,
        ]:
            frozen = root / "frozen.json"
            frozen.write_bytes(b"frozen")
            frozen_sha = sha256(frozen)
            source = {"origin": "test", "commit": "a", "tree": "b"}
            inputs: dict[str, object] = {
                "source": source,
                "canonical_manifest": str(frozen),
                "canonical_manifest_sha256": frozen_sha,
                "canonical_summary_path": str(frozen),
                "canonical_summary_sha256": frozen_sha,
                "canonical_lineage_path": str(frozen),
                "canonical_lineage_sha256": frozen_sha,
                "base_training_summary_path": str(frozen),
                "base_training_summary_sha256": frozen_sha,
                "audio_manifest_sha256": {},
                "audio_summary_sha256": {},
                "audio_lineage_sha256": {},
                "training_lineage_manifest_sha256": {},
                "canonical_by_id": {},
                "audio_by_id": {},
            }
            checkpoint = root / "upper.bin"
            status = root / "upper-status.json"
            report = root / "upper-gate.json"
            checkpoint.write_bytes(b"checkpoint")
            status.write_bytes(b"status")
            report.write_bytes(b"gate")
            records = {
                "upper": {
                    "path": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "formal_training_status": str(status),
                    "formal_training_status_sha256": sha256(status),
                    "audit": {
                        "smplx_training_pool_mode": "target_offload",
                        base_inference.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY: {
                            "path": str(report),
                            "sha256": sha256(report),
                        }
                    },
                }
            }
            return inputs, records, report

        mutations = {
            "deleted": lambda path, root: path.unlink(),
            "tampered": lambda path, root: path.write_bytes(b"tampered"),
            "symlink": lambda path, root: (
                path.unlink(),
                (root / "replacement.json").write_bytes(b"replacement"),
                path.symlink_to(root / "replacement.json"),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, records, report = fixture(root)
                with mock.patch.object(
                    base_inference,
                    "_source_receipt",
                    return_value=inputs["source"],
                ):
                    base_inference._revalidate_frozen_inputs(
                        SimpleNamespace(),
                        inputs,
                        records,
                    )
                    mutate(report, root)
                    with self.assertRaises(
                        (
                            base_inference.InferenceContractError,
                            FileNotFoundError,
                        )
                    ):
                        base_inference._revalidate_frozen_inputs(
                            SimpleNamespace(),
                            inputs,
                            records,
                        )

    def test_trainer_runtime_must_materialize_the_exact_pool(self) -> None:
        args = SimpleNamespace(
            formal_stage="face",
            smplx_training_pool_mode="sharded_local_loss",
            smplx_training_helper_devices="1,2",
        )
        replicas = {
            device: SimpleNamespace(parameters=lambda: ())
            for device in range(3)
        }
        topology = {
            "format": "semtalk_smplx_training_pool_topology_v1",
            "pool_runtime_format": "semtalk_smplx_training_pool_v2",
            "mode": "sharded_local_loss",
            "primary_device": 0,
            "helper_devices": [1, 2],
            "replica_devices": [0, 1, 2],
            "visible_device_count": 3,
            "partition": "balanced_contiguous_whole_clips",
            "transfer_to_primary": (
                "differentiable_scalar_numerators_only"
            ),
        }
        pool = SimpleNamespace(
            mode="sharded_local_loss",
            primary_device=0,
            helper_devices=(1, 2),
            replicas=replicas,
            models=replicas,
            streams={device: object() for device in range(3)},
            gate_replica_modules=lambda: (
                replicas[1],
                replicas[2],
            ),
            runtime_receipt=lambda: {
                "format": "semtalk_smplx_training_pool_v2",
                "mode": "sharded_local_loss",
                "primary_device": 0,
                "helper_devices": [1, 2],
                "replica_devices": [0, 1, 2],
                "partition": "balanced_contiguous_whole_clips",
                "transfer_to_primary": (
                    "differentiable_scalar_numerators_only"
                ),
                "last_forward": None,
            },
        )
        trainer = SimpleNamespace(
            smplx_parallel_mode="sharded_local_loss",
            smplx_parallel_pool=pool,
            smplx_pool=pool,
            smplx=replicas[0],
            model=SimpleNamespace(parameters=lambda: ()),
            opt=SimpleNamespace(param_groups=[]),
        )
        gate_receipt = {
            "formal_stage": "face",
            "mode": "sharded_local_loss",
            "helper_devices": [1, 2],
            "topology": topology,
            "topology_sha256": formal._payload_sha256(topology),
            "runtime": {
                "visible_device_count": 3,
                "device_names": ["NVIDIA H200"] * 3,
                "torch": str(formal.torch.__version__),
                "cuda": "12.8",
                "cudnn": 91002,
            },
        }
        with (
            mock.patch.object(
                formal,
                "_formal_module_cuda_devices",
                side_effect=lambda module: {
                    next(
                        device
                        for device, value in replicas.items()
                        if value is module
                    )
                },
            ),
            mock.patch.object(
                formal.torch.cuda,
                "get_device_name",
                return_value="NVIDIA H200",
            ),
            mock.patch.object(
                formal.torch.cuda,
                "device_count",
                return_value=3,
            ),
            mock.patch.object(
                formal.torch.cuda,
                "current_device",
                return_value=0,
            ),
            mock.patch.object(formal.torch.version, "cuda", "12.8"),
            mock.patch.object(
                formal.torch.backends.cudnn,
                "version",
                return_value=91002,
            ),
        ):
            formal._validate_smplx_training_pool_trainer_runtime(
                args,
                trainer,
                gate_receipt=gate_receipt,
            )
            pool.helper_devices = (1,)
            with self.assertRaises(RuntimeError):
                formal._validate_smplx_training_pool_trainer_runtime(
                    args,
                    trainer,
                    gate_receipt=gate_receipt,
                )

        disabled_args = SimpleNamespace(
            formal_stage="global",
            smplx_training_pool_mode="disabled",
        )
        disabled_trainer = SimpleNamespace()
        formal._validate_smplx_training_pool_trainer_runtime(
            disabled_args,
            disabled_trainer,
            gate_receipt=None,
        )
        for stage in formal.FORMAL_SMPLX_STAGES:
            with self.subTest(stock_stage=stage):
                disabled_vq_args = copy.copy(disabled_args)
                disabled_vq_args.formal_stage = stage
                formal._validate_smplx_training_pool_trainer_runtime(
                    disabled_vq_args,
                    SimpleNamespace(),
                    gate_receipt=None,
                )
        disabled_vq_args = copy.copy(disabled_args)
        disabled_vq_args.formal_stage = "face"
        with self.assertRaises(RuntimeError):
            formal._validate_smplx_training_pool_trainer_runtime(
                disabled_vq_args,
                SimpleNamespace(),
                gate_receipt={},
            )
        disabled_trainer.smplx_pool = object()
        with self.assertRaises(RuntimeError):
            formal._validate_smplx_training_pool_trainer_runtime(
                disabled_args,
                disabled_trainer,
                gate_receipt=None,
            )

    def test_gate_preflight_precedes_nccl_and_trainer_initialization(
        self,
    ) -> None:
        source = inspect.getsource(formal.main)
        preflight = source.index(
            "initial_dataset_receipt = _dataset_receipt("
        )
        nccl = source.index("dist.init_process_group(")
        trainer = source.index(").CustomTrainer(args)")
        runtime = source.index(
            "_validate_smplx_training_pool_trainer_runtime("
        )
        final_receipt = source.index(
            "\n    dataset_receipt = _dataset_receipt(",
            runtime,
        )
        self.assertLess(preflight, nccl)
        self.assertLess(nccl, trainer)
        self.assertLess(trainer, runtime)
        self.assertLess(runtime, final_receipt)

    def test_valid_gate_and_receipt_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FormalPoolGateFixture(Path(directory))
            receipt = fixture.validate()
            self.assertEqual(receipt["children"]["count"], 8)
            self.assertEqual(
                receipt["performance"]["wall"]["pooled_speedup"],
                2.0,
            )
            dataset: dict[str, object] = {}
            formal._attach_smplx_training_pool_gate_receipt(
                dataset,
                receipt,
            )
            resume: dict[str, object] = {}
            formal._attach_smplx_training_pool_gate_receipt(
                resume,
                dataset[formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY],
            )
            formal._verify_smplx_training_pool_gate_resume_receipt(
                resume,
                dataset[formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY],
            )
            dataset.update(
                {
                    "summary_sha256": "6" * 64,
                    "data_mdb_sha256": "7" * 64,
                    "smplx_asset": fixture.smplx,
                }
            )
            trainer = SimpleNamespace(
                model=SimpleNamespace(state_dict=lambda: {}),
                smplx_training_pool_runtime_evidence=(
                    fixture.runtime_evidence(receipt)
                ),
            )
            model_payload = formal._model_payload(
                trainer,
                formal_stage="face",
                config_sha256="8" * 64,
                lineage_sha256="9" * 64,
                dataset_receipt=dataset,
                source_receipt=fixture.source,
                optimizer_updates=2,
                candidate_manifest_receipt=None,
            )
            self.assertEqual(
                model_payload["audit"][
                    formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                ],
                receipt,
            )
            self.assertEqual(
                model_payload["audit"]["smplx_training_pool_mode"],
                receipt["mode"],
            )
            resume[formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY][
                "sha256"
            ] = "0" * 64
            with self.assertRaises(RuntimeError):
                formal._verify_smplx_training_pool_gate_resume_receipt(
                    resume,
                    dataset[
                        formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                    ],
                )

    def test_missing_tampered_and_wrong_bindings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FormalPoolGateFixture(Path(directory))
            original_sha = sha256(fixture.report_path)
            fixture.report["completed_unix"] = 3.0
            fixture.write()
            args = fixture.args()
            args.expected_smplx_training_pool_gate_sha256 = original_sha
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                formal._formal_smplx_training_pool_gate_receipt(
                    args,
                    current_source=fixture.source,
                    representation=fixture.representation,
                    smplx_asset_receipt=fixture.smplx,
                )
        mutations = (
            ("missing semantic", lambda value: value.pop("semantic_common")),
            (
                "missing coordinator",
                lambda value: value["preflight"]["source"].pop(
                    "coordinator"
                ),
            ),
            (
                "wrong stage",
                lambda value: value["scope"].__setitem__("stage", "hands"),
            ),
            (
                "wrong source",
                lambda value: value["preflight"]["source"].__setitem__(
                    "commit", "f" * 40
                ),
            ),
            (
                "wrong representation",
                lambda value: value["semantic_common"][
                    "representation"
                ].__setitem__("data_sha256", "0" * 64),
            ),
            (
                "missing child result",
                lambda value: value["children"][0].pop("result_path"),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = FormalPoolGateFixture(Path(directory))
                    mutate(fixture.report)
                    fixture.write()
                    with self.assertRaises((RuntimeError, KeyError)):
                        fixture.validate()

    def test_missing_report_and_disabled_gate_args_fail_closed(self) -> None:
        args = SimpleNamespace(
            formal_stage="face",
            smplx_training_pool_mode="sharded_local_loss",
            smplx_training_pool_gate_report="/missing/report.json",
            expected_smplx_training_pool_gate_sha256="a" * 64,
        )
        with self.assertRaises(RuntimeError):
            formal._formal_smplx_training_pool_gate_receipt(
                args,
                current_source={},
                representation={},
                smplx_asset_receipt=None,
            )
        args.smplx_training_pool_mode = "disabled"
        args.smplx_training_pool_gate_report = None
        args.expected_smplx_training_pool_gate_sha256 = None
        self.assertIsNone(
            formal._formal_smplx_training_pool_gate_receipt(
                args,
                current_source={},
                representation={},
                smplx_asset_receipt=None,
            )
        )
        args.smplx_training_pool_gate_report = "/forbidden/report.json"
        with self.assertRaises(RuntimeError):
            formal._formal_smplx_training_pool_gate_receipt(
                args,
                current_source={},
                representation={},
                smplx_asset_receipt=None,
            )
        args.formal_stage = "global"
        args.smplx_training_pool_gate_report = None
        args.expected_smplx_training_pool_gate_sha256 = None
        self.assertIsNone(
            formal._formal_smplx_training_pool_gate_receipt(
                args,
                current_source={},
                representation={},
                smplx_asset_receipt=None,
            )
        )

    def test_downstream_consumers_require_the_exact_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FormalPoolGateFixture(Path(directory))
            receipt = fixture.validate()
            dataset = {
                "data_mdb_sha256": fixture.representation["data_sha256"],
                "summary_sha256": fixture.representation["summary_sha256"],
                "lineage_sha256": fixture.representation["lineage_sha256"],
                "smplx_asset": fixture.smplx,
                formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY: receipt,
            }
            audit = {
                "source_receipt": fixture.source,
                "smplx_training_pool_mode": "sharded_local_loss",
                formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY: receipt,
                formal.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY: (
                    fixture.runtime_evidence(receipt)
                ),
            }
            status = {
                "smplx_training_pool_mode": "sharded_local_loss",
                formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY: receipt,
                formal.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY: (
                    fixture.runtime_evidence(receipt)
                ),
            }
            for validate, consumer_module in (
                (
                    build_features.validate_smplx_training_pool_gate_binding,
                    build_features,
                ),
                (
                    base_inference._validate_smplx_training_pool_gate_binding,
                    base_inference,
                ),
            ):
                with (
                    self.subTest(consumer=validate.__module__),
                    mock.patch.dict(
                        consumer_module.SMPLX_TRAINING_POOL_MODE_SPECS[
                            "sharded_local_loss"
                        ],
                        {
                            "harness_sha256": sha256(
                                fixture.harness
                            ),
                            "adapter_sha256": sha256(
                                fixture.adapter
                            ),
                        },
                    ),
                ):
                    validate(
                        formal_stage="face",
                        audit=audit,
                        dataset_receipt=dataset,
                        status=status,
                        path=fixture.report_path,
                    )
                    tampered_status = copy.deepcopy(status)
                    tampered_status[
                        formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
                    ]["sha256"] = "0" * 64
                    with self.assertRaises((RuntimeError, ValueError)):
                        validate(
                            formal_stage="face",
                            audit=audit,
                            dataset_receipt=dataset,
                            status=tampered_status,
                            path=fixture.report_path,
                        )
                    with self.assertRaises((RuntimeError, ValueError)):
                        validate(
                            formal_stage="global",
                            audit={},
                            dataset_receipt=dataset,
                            status={},
                            path=fixture.report_path,
                        )

    def test_downstream_consumers_accept_only_evidence_free_stock_mode(
        self,
    ) -> None:
        runtime_key = formal.SMPLX_TRAINING_POOL_RUNTIME_EVIDENCE_KEY
        gate_key = formal.SMPLX_TRAINING_POOL_GATE_RECEIPT_KEY
        for validate in (
            build_features.validate_smplx_training_pool_gate_binding,
            base_inference._validate_smplx_training_pool_gate_binding,
        ):
            with self.subTest(consumer=validate.__module__):
                audit = {
                    "smplx_training_pool_mode": "disabled",
                    runtime_key: None,
                }
                status = {
                    "smplx_training_pool_mode": "disabled",
                    runtime_key: None,
                }
                validate(
                    formal_stage="face",
                    audit=audit,
                    dataset_receipt={},
                    status=status,
                    path=Path("/formal/stock-face.bin"),
                )
                mutations = (
                    ("audit gate", {**audit, gate_key: {}}, {}, status),
                    ("dataset gate", audit, {gate_key: {}}, status),
                    ("status gate", audit, {}, {**status, gate_key: {}}),
                    (
                        "audit runtime",
                        {**audit, runtime_key: {}},
                        {},
                        status,
                    ),
                    (
                        "dataset runtime",
                        audit,
                        {runtime_key: {}},
                        status,
                    ),
                    (
                        "status runtime",
                        audit,
                        {},
                        {**status, runtime_key: {}},
                    ),
                    (
                        "status mode mismatch",
                        audit,
                        {},
                        {
                            **status,
                            "smplx_training_pool_mode": "target_offload",
                        },
                    ),
                    (
                        "missing audit mode",
                        {runtime_key: None},
                        {},
                        status,
                    ),
                    (
                        "unknown audit mode",
                        {
                            **audit,
                            "smplx_training_pool_mode": "unknown",
                        },
                        {},
                        {
                            **status,
                            "smplx_training_pool_mode": "unknown",
                        },
                    ),
                )
                for (
                    label,
                    changed_audit,
                    changed_dataset,
                    changed_status,
                ) in mutations:
                    with self.subTest(rejects=label):
                        with self.assertRaises((RuntimeError, ValueError)):
                            validate(
                                formal_stage="face",
                                audit=changed_audit,
                                dataset_receipt=changed_dataset,
                                status=changed_status,
                                path=Path("/formal/stock-face.bin"),
                            )


if __name__ == "__main__":
    unittest.main()
