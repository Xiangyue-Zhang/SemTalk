from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "train_base_official_adapt_long.py"
)
LAUNCHER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "run_base_official_adapt_long.sh"
)
FRESH_SCHEDULE = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_fresh_lineage_schedule_20260731.json"
)
TOPOLOGY_GATE_SPEC = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_topology_gate_spec_20260731.json"
)
TOPOLOGY_QUALITY_GATE_SPEC = (
    REPOSITORY
    / "configs"
    / "show_base"
    / "semtalk_base_topology_quality_gate_spec_20260731.json"
)
OFFICIAL_TRAINER = REPOSITORY / "semtalk_base_trainer.py"
SPEC = importlib.util.spec_from_file_location(
    "train_base_official_adapt_long", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ADAPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPT)
SELECTOR_SCRIPT = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "select_base_training_topology.py"
)
SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "select_base_training_topology", SELECTOR_SCRIPT
)
assert SELECTOR_SPEC is not None and SELECTOR_SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SELECTOR_SPEC)
SELECTOR_SPEC.loader.exec_module(SELECTOR)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _activate(mode: str = "validation_gated_w8_l64_g512_empirical_acceleration") -> None:
    ADAPT._activate_topology(argparse.Namespace(topology_mode=mode))


def _probe(seed: str = "a") -> dict[str, object]:
    ranks = []
    for rank in range(ADAPT.WORLD_SIZE):
        rank_hex = format(rank, "x")
        ranks.append(
            {
                "rank": rank,
                "optimizer_updates": ADAPT.TRAJECTORY_PROBE_UPDATES,
                "model_state_tensors": 1790,
                "model_state_schema_sha256": seed * 64,
                "model_state_semantic_sha256": rank_hex * 64,
                "parameter_state_tensors": 1784,
                "parameter_state_schema_sha256": "a" * 64,
                "parameter_state_semantic_sha256": "b" * 64,
                "buffer_state_tensors": 6,
                "buffer_state_schema_sha256": "c" * 64,
                "buffer_state_semantic_sha256": rank_hex * 64,
                "optimizer_state_semantic_sha256": "c" * 64,
                "python_random_state_sha256": rank_hex * 64,
                "numpy_random_state_sha256": "d" * 63 + rank_hex,
                "torch_cpu_rng_state_sha256": "e" * 63 + rank_hex,
                "torch_cuda_rng_state_sha256": "f" * 63 + rank_hex,
                "sample_order_sha256": rank_hex * 63 + "1",
                "sample_count": (
                    ADAPT.TRAJECTORY_PROBE_UPDATES
                    * ADAPT.LOCAL_BATCH_SIZE
                ),
            }
        )
    return ADAPT._assemble_trajectory_probe(
        ranks,
        optimizer_updates=ADAPT.TRAJECTORY_PROBE_UPDATES,
    )


def _gate_report(
    *,
    frozen_sha256: str,
    frozen_compatibility_sha256: str,
    topology_independent_input_sha256: str,
    topology_receipt_sha256: str,
    trajectory_mode: str,
    trajectory_probe: object,
) -> dict[str, object]:
    mode = ADAPT.W8_GLOBAL512_MODE
    _activate(mode)
    spec = ADAPT.TOPOLOGY_SPECS[mode]
    world = int(spec["world_size"])
    probe_epoch_updates = list(
        ADAPT._probe_epoch_update_counts(int(spec["updates_per_epoch"]))
    )
    probe_epoch_samples = [
        updates * int(spec["global_batch_size"])
        for updates in probe_epoch_updates
    ]
    probe_samples = sum(probe_epoch_samples)
    cross_epoch_duplicates = 0 if len(probe_epoch_updates) == 1 else 1
    before = [{"name": "hubert", "running_mean_sha256": "1" * 64}]
    report: dict[str, object] = {
        "format": ADAPT.GATE_FORMAT,
        "status": "pass",
        "topology_mode": mode,
        "topology_classification": spec["classification"],
        "topology_gate_spec_sha256": _sha(TOPOLOGY_GATE_SPEC),
        "topology_independent_input_sha256": (
            topology_independent_input_sha256
        ),
        "frozen_receipt_sha256": frozen_sha256,
        "frozen_gate_compatibility_sha256": (
            frozen_compatibility_sha256
        ),
        "topology_receipt_sha256": topology_receipt_sha256,
        "node_count": spec["node_count"],
        "local_world_size": spec["local_world_size"],
        "world_size": world,
        "local_batch_size": spec["local_batch_size"],
        "global_batch_size": spec["global_batch_size"],
        "updates_per_epoch": spec["updates_per_epoch"],
        "unique_samples_per_epoch": spec["unique_samples_per_epoch"],
        "warmup_updates": ADAPT.THROUGHPUT_WARMUP_UPDATES,
        "timed_updates": ADAPT.THROUGHPUT_TIMED_UPDATES,
        "optimizer_updates": ADAPT.TRAJECTORY_PROBE_UPDATES,
        "trajectory_mode": trajectory_mode,
        "trajectory_probe": trajectory_probe,
        "precision": "bf16",
        "learning_rate": spec["learning_rate"],
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "oom": False,
        "samples_per_second": spec["global_batch_size"] / 0.5,
        "seconds_per_update": 0.5,
        "median_seconds": 0.5,
        "p90_seconds": 0.6,
        "p99_seconds": 0.7,
        "estimated_training_seconds": (
            0.5 * spec["updates_per_epoch"] * ADAPT.TOTAL_EPOCHS
        ),
        "estimated_epochs": ADAPT.TOTAL_EPOCHS,
        "last_metrics": {"total": 1.0},
        "peak_cuda_memory_bytes_all_ranks": [1024] * world,
        "data_wait_seconds": {"median": 0.01, "p99": 0.02},
        "collective_seconds": {
            "probe": "ten_scalar_nccl_all_reduce_calls",
            "median": 0.001,
            "p99": 0.002,
        },
        "batchnorm_inventory": [
            {"rank": rank, "before": before, "after": before}
            for rank in range(world)
        ],
        "rng_inventory": [
            {"rank": rank, "torch_cuda_rng_state_sha256": f"{rank:064x}"}
            for rank in range(world)
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
            "full_epoch_samples": spec["unique_samples_per_epoch"],
            "full_epoch_unique_samples": spec["unique_samples_per_epoch"],
            "dataset_samples": ADAPT.EXPECTED_TRAIN_SAMPLES,
            "dropped_tail_samples": ADAPT.EXPECTED_TRAIN_SAMPLES
            - int(spec["unique_samples_per_epoch"]),
            "full_epoch_sorted_indices_sha256": "8" * 64,
        },
    }
    report["receipt_sha256"] = ADAPT.canonical_json_sha256(report)
    return report


def _base_cli() -> list[str]:
    return [
        "--mode", "throughput_gate",
        "--official-base-checkpoint", "/weights/all/best_semtalk_base.bin",
        "--train-lmdb", "/cache/current-selected/base.lmdb",
        "--dataset-summary", "/cache/current-selected/summary.json",
        "--expected-dataset-summary-sha256", "a" * 64,
        "--lineage-manifest", "/cache/current-selected/lineage.json",
        "--expected-lineage-sha256", "b" * 64,
        "--prerequisite-selection-json", "/selection/current-five.json",
        "--expected-prerequisite-selection-sha256", "c" * 64,
        "--schedule-json", str(FRESH_SCHEDULE),
        "--expected-schedule-sha256", _sha(FRESH_SCHEDULE),
        "--topology-gate-spec", str(TOPOLOGY_GATE_SPEC),
        "--expected-topology-gate-spec-sha256", _sha(TOPOLOGY_GATE_SPEC),
        "--trajectory-mode", ADAPT.FRESH_TRAJECTORY_MODE,
        "--output-root", "/runs/base",
        "--run-name", "selected_all_show",
        "--formal-node-rank", "0",
        "--formal-host-slot", "0",
        "--formal-master-addr", "master.example",
        "--formal-master-port", "29601",
        "--formal-run-id", "formal-run-001",
        "--topology-mode", ADAPT.W8_GLOBAL512_MODE,
        "--local-batch-size", "64",
        "--learning-rate", "0.00003",
    ]


def _frozen_gate_fixture(
    *,
    receipt_sha256: str,
    topology_receipt_sha256: str,
    trajectory_mode: str,
    run_purpose: str,
    target_epochs: list[int],
) -> dict[str, object]:
    return {
        "format": "semtalk_show_base_official_adapt_frozen_inputs_v1",
        "run_purpose": run_purpose,
        "target_epochs": target_epochs,
        "source": {
            "origin": ADAPT.EXPECTED_ORIGIN,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "clean": True,
            "entrypoint_sha256": "3" * 64,
        },
        "official_base": {"sha256": ADAPT.OFFICIAL_BASE_SPEC["sha256"]},
        "speaker_initialization": {"speaker_rows": [0, 1, 2, 3]},
        "dataset": {
            "data_mdb_sha256": "4" * 64,
            "prerequisite_selection": {"sha256": "6" * 64},
            "selected_prerequisite_sha256": {
                "face": "7" * 64,
                "upper": "8" * 64,
                "hands": "9" * 64,
                "lower": "a" * 64,
                "global": "b" * 64,
            },
        },
        "protocol": {
            "forward_contract": {"official": True},
            "loss": {"official": True},
            "precision": "bf16",
            "target_dataset": "SHOW",
            "target_speaker_scope": "All",
            "vq_models_in_training_graph": False,
        },
        "long_contract": {
            "schedule": {"sha256": "5" * 64},
            "trajectory_anchor": {"mode": trajectory_mode},
        },
        "topology": {"receipt_sha256": topology_receipt_sha256},
        "receipt_sha256": receipt_sha256,
    }


def _topology_selection_inputs(
    estimated_training_seconds: dict[str, float],
    *,
    p99_seconds: dict[str, float] | None = None,
    failing_quality_modes: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    modes = list(ADAPT.TOPOLOGY_SPECS)
    if set(estimated_training_seconds) != set(modes):
        raise AssertionError("test ETA fixture must cover all nine modes")
    p99_seconds = p99_seconds or {
        mode: 0.01 + index * 0.001 for index, mode in enumerate(modes)
    }
    failing_quality_modes = failing_quality_modes or set()
    probes: list[dict[str, object]] = []
    quality: list[dict[str, object]] = []
    for index, (mode, spec) in enumerate(ADAPT.TOPOLOGY_SPECS.items()):
        probes.append(
            {
                "mode": mode,
                "status": "pass",
                "report_path": f"/gate/{mode}.json",
                "report_sha256": f"{index + 1:x}" * 64,
                "classification": spec["classification"],
                "precision": spec["precision"],
                "formal_training_eligible": spec[
                    "formal_training_eligible"
                ],
                "topology_independent_input_sha256": "a" * 64,
                "median_seconds": 1.0 + index,
                "p90_seconds": 1.1 + index,
                "p99_seconds": p99_seconds[mode],
                "estimated_training_seconds": (
                    estimated_training_seconds[mode]
                ),
                "samples_per_second": 1000.0,
            }
        )
        fgd = {str(epoch): 0.5 for epoch in SELECTOR.QUALITY_EPOCHS}
        if mode in failing_quality_modes:
            fgd["4"] = 0.7
        quality.append(
            {
                "mode": mode,
                "report_path": f"/quality/{mode}.json",
                "report_sha256": f"{index + 6:x}" * 64,
                "topology_independent_input_sha256": "a" * 64,
                "short_trajectory_receipt": {},
                "val_inputs_receipt": {
                    "path": "/val/inputs.json",
                    "sha256": "c" * 64,
                    "bytes": 10,
                    "receipt_payload_sha256": "d" * 64,
                },
                "pipeline_receipt": {
                    "path": "/val/pipeline.json",
                    "sha256": "e" * 64,
                    "bytes": 10,
                    "receipt_payload_sha256": "f" * 64,
                },
                "candidate_fgd": fgd,
                "candidates": [],
            }
        )
    return probes, quality


def _over_budget_quality_skip(
    probe: dict[str, object],
    *,
    position: int,
) -> dict[str, object]:
    mode = str(probe["mode"])
    return {
        "mode": mode,
        "status": "skipped_over_eta_budget",
        "receipt_path": f"/quality-skips/{mode}.json",
        "receipt_sha256": f"{position + 10:x}" * 64,
        "receipt_payload_sha256": f"{position + 11:x}" * 64,
        "topology_gate_spec_sha256": "b" * 64,
        "quality_gate_spec_sha256": "f" * 64,
        "topology_independent_input_sha256": probe[
            "topology_independent_input_sha256"
        ],
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
        "estimated_training_seconds": probe[
            "estimated_training_seconds"
        ],
        "maximum_estimated_training_seconds": (
            SELECTOR.MAX_TRAINING_SECONDS
        ),
    }


def _validated_throughput_projection(
    selected: dict[str, object],
    **overrides: object,
) -> dict[str, object]:
    projection = {
        "sha256": selected["report_sha256"],
        "topology_mode": selected["mode"],
        "samples_per_second": selected["samples_per_second"],
        "median_seconds": selected["median_seconds"],
        "p90_seconds": selected["p90_seconds"],
        "p99_seconds": selected["p99_seconds"],
        "estimated_training_seconds": selected[
            "estimated_training_seconds"
        ],
    }
    projection.update(overrides)
    return projection


def _consume_topology_selection(
    *,
    selection: dict[str, object],
    path: Path,
    topology_mode: str,
    throughput_gate: dict[str, object],
    verified_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    verified = verified_selection or selection
    reloaded = {
        "probes": copy.deepcopy(verified["probes"]),
        "quality_reports": copy.deepcopy(verified["quality_reports"]),
        "quality_skips": copy.deepcopy(verified["quality_skips"]),
    }
    with mock.patch.object(
        ADAPT,
        "_reload_topology_selection_artifacts",
        return_value=reloaded,
    ):
        return ADAPT.validate_topology_selection(
            argparse.Namespace(
                topology_selection_report=path,
                expected_topology_selection_sha256=_sha(path),
                expected_topology_gate_spec_sha256="b" * 64,
                topology_mode=topology_mode,
            ),
            throughput_gate=throughput_gate,
        )


class OfficialBaseAdaptStaticContracts(unittest.TestCase):
    def test_formal_val_control_is_diffsheg_only_and_topology_exact(self) -> None:
        from scripts.show_base import base_final_authority as final_authority
        from scripts.show_base import base_long_val_contract as base_long
        from scripts.show_base import evaluate_diffsheg_val_fgd as evaluator
        from scripts.show_base import select_base_official_adapt as diffsheg
        from scripts.show_base import select_base_official_adapt_long as long_selector
        from scripts.show_base import talkshow_base_val_contract as talkshow

        self.assertEqual(base_long.TOPOLOGY_SPECS, ADAPT.TOPOLOGY_SPECS)
        # The selected-five inference pipeline remains internally TalkSHOW
        # owned.  The formal long-run selection ABI is DiffSHEG-only.
        self.assertIs(talkshow.validate_pipeline, talkshow.validate_fresh_pipeline)
        self.assertIs(base_long.validate_val_inputs, diffsheg.validate_val_inputs)
        self.assertIs(
            base_long.validate_val_inference_lineage,
            diffsheg.validate_val_inference_lineage,
        )
        self.assertIsNot(
            base_long.validate_diffsheg_report,
            diffsheg.validate_diffsheg_report,
        )
        self.assertIn(
            "scripts/show_base/evaluate_diffsheg_val_fgd.py",
            talkshow.FRESH_PIPELINE_SOURCE_FILES,
        )
        self.assertIs(
            long_selector._profile_values()["validate_diffsheg_report"],
            base_long.validate_diffsheg_report,
        )
        self.assertEqual(
            base_long.VAL_INFERENCE_LINEAGE_FORMAT,
            diffsheg.VAL_INFERENCE_LINEAGE_FORMAT,
        )
        self.assertNotEqual(
            base_long.VAL_INFERENCE_LINEAGE_FORMAT,
            talkshow.VAL_INFERENCE_LINEAGE_FORMAT,
        )
        self.assertIn(
            "diffsheg_clip_manifest_sha256",
            base_long.public_val_coverage(
                {
                    "split": "val",
                    "clip_count": 1_715,
                    "frame_count": 1,
                    "window_count": 1,
                    "uncovered_tail_frames": 0,
                    "clip_ids_sha256": "a" * 64,
                    "diffsheg_clip_manifest_sha256": "b" * 64,
                }
            ),
        )
        self.assertEqual(diffsheg.VAL_METRIC_KEYS, ("fgd",))
        self.assertEqual(
            final_authority.BASE_SELECTION_PROTOCOL,
            "diffsheg_show_validation_fgd_v1",
        )
        self.assertEqual(
            final_authority.BASE_SELECTION_METRIC,
            "validation.diffsheg.metrics.fgd",
        )
        pins = diffsheg.DIFFSHEG_PINNED_RECEIPT
        self.assertEqual(pins["selection_metric"], "fgd")
        self.assertFalse(pins["ba_during_selection"])
        self.assertEqual(
            pins["paspa"]["evaluator_sha256"],
            evaluator.PASPA_EVALUATOR_SHA256,
        )
        self.assertEqual(
            pins["diffsheg_reference_commit"],
            evaluator.DIFFSHEG_REFERENCE_COMMIT,
        )
        self.assertEqual(
            pins["stats_sha256"],
            evaluator.DIFFSHEG_STATS_SHA256,
        )
        self.assertEqual(
            pins["autoencoders"]["fgd"]["sha256"],
            evaluator.DIFFSHEG_GESTURE_AE_SHA256,
        )

        internal_evaluators = {
            "scripts.show_base.evaluate_talkshow_show_metrics",
            "scripts.show_base.replay_released2_primary",
        }
        self.assertTrue(
            {
                module.replace(".", "/") + ".py"
                for module in internal_evaluators
            }.issubset(talkshow.FRESH_PIPELINE_SOURCE_FILES)
        )
        control_modules = set(final_authority._CONTROL_DEPENDENCIES)
        for dependencies in final_authority._CONTROL_DEPENDENCIES.values():
            control_modules.update(dependencies)
        self.assertFalse(
            {
                module.rsplit(".", 1)[-1]
                for module in internal_evaluators
            }
            & control_modules
        )

        formal_validation_files = (
            "scripts/show_base/talkshow_base_val_contract.py",
            "scripts/show_base/base_long_val_contract.py",
            "scripts/show_base/base_final_authority.py",
            "scripts/show_base/run_base_val_inference.py",
            "scripts/show_base/select_base_official_adapt.py",
            "scripts/show_base/select_base_official_adapt_long.py",
            "scripts/show_base/produce_base_val_measurement.py",
            "scripts/show_base/evaluate_diffsheg_val_fgd.py",
        )
        for relative in formal_validation_files:
            source = (REPOSITORY / relative).read_text(encoding="utf-8")
            syntax = ast.parse(source, filename=relative)
            imports: set[str] = set()
            for node in ast.walk(syntax):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.add(module)
                    if module == "scripts.show_base":
                        imports.update(
                            f"{module}.{alias.name}"
                            for alias in node.names
                        )
            self.assertFalse(
                imports & internal_evaluators,
                f"{relative}: internal released2 evaluator became reachable",
            )

    def test_launcher_binds_hash_seed_and_all_gate_topologies(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("export PYTHONHASHSEED=43", source)
        self.assertIn('nproc_per_node=1', source)
        self.assertIn('nproc_per_node=8', source)
        for mode in ADAPT.TOPOLOGY_SPECS:
            self.assertIn(mode, source)
        self.assertIn('"$script_dir/train_base_official_adapt_long.py"', source)

    def test_scratch_entrypoint_is_untouched_by_the_new_entrypoint(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "This is intentionally separate from ``show_base_train.py``",
            source,
        )
        self.assertNotIn("load_pretrained_vq_suite", source)
        self.assertNotIn("from models.rvq", source)
        self.assertNotIn("import models.rvq", source)

    def test_base_objective_cannot_import_or_call_semgate_or_sparse(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT))
        objective = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "audio_conditioned_objective"
        )
        identifiers = {
            node.id.lower()
            for node in ast.walk(objective)
            if isinstance(node, ast.Name)
        }
        self.assertFalse(any("semgate" in name for name in identifiers))
        self.assertFalse(any("sparse" in name for name in identifiers))
        self.assertEqual(len(ADAPT.LOSS_COMPONENTS), 26)

    def test_masked_self_preserves_published_retained_i_bug(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "source_level = RVQ_LEVELS - 1 if masked_self_published_ce",
            source,
        )
        self.assertIn(
            "divisor = float(RVQ_LEVELS if masked_self_published_ce",
            source,
        )
        self.assertIn(
            "repeat_rvq_level_5_six_times_divided_by_6_v1",
            source,
        )

    def test_official_cpu_mask_rng_and_nll_operator_path_are_preserved(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'torch_module.rand(tuple(batch["latent_all"].shape), device="cpu")',
            source,
        )
        self.assertNotIn('torch_module.rand_like(batch["latent_all"])', source)
        self.assertIn("torch_module.nn.functional.log_softmax(", source)
        self.assertIn("dim=2,", source)
        self.assertIn("torch_module.nn.functional.nll_loss(", source)
        self.assertNotIn("torch_module.nn.functional.cross_entropy(", source)

    def test_class_axis_matches_official_logsoftmax_nll_source(self) -> None:
        official_source = OFFICIAL_TRAINER.read_text(encoding="utf-8")
        official_tree = ast.parse(official_source, str(OFFICIAL_TRAINER))
        logsoftmax_initializers = [
            node
            for node in ast.walk(official_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "LogSoftmax"
        ]
        self.assertEqual(len(logsoftmax_initializers), 1)
        self.assertEqual(
            [
                keyword.value.value
                for keyword in logsoftmax_initializers[0].keywords
                if keyword.arg == "dim"
                and isinstance(keyword.value, ast.Constant)
            ],
            [2],
        )
        self.assertIn("self.log_softmax(net_out_val[\"cls_face\"][:,:,:,i])", official_source)
        self.assertIn("self.cls_loss(rec_index_face_val", official_source)

        adapted = ast.parse(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT))
        family = next(
            node
            for node in adapted.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_official_forward_loss_family"
        )
        logsoftmax_calls = [
            node
            for node in ast.walk(family)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log_softmax"
        ]
        self.assertEqual(len(logsoftmax_calls), 1)
        self.assertEqual(
            [
                keyword.value.value
                for keyword in logsoftmax_calls[0].keywords
                if keyword.arg == "dim"
                and isinstance(keyword.value, ast.Constant)
            ],
            [2],
        )

    def test_objective_contains_exactly_three_official_model_forwards(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT))
        objective = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "audio_conditioned_objective"
        )
        calls = [
            node
            for node in ast.walk(objective)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "model"
        ]
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [
                next(
                    (
                        keyword.value.value
                        for keyword in call.keywords
                        if keyword.arg == "use_word"
                        and isinstance(keyword.value, ast.Constant)
                    ),
                    None,
                )
                for call in calls
            ],
            [True, False, True],
        )

    def test_nine_mode_parallelism_candidates_and_gate_lengths(self) -> None:
        self.assertEqual(len(ADAPT.TOPOLOGY_SPECS), 9)
        self.assertEqual(
            {
                (
                    value["world_size"],
                    value["local_batch_size"],
                    value["global_batch_size"],
                    value["updates_per_epoch"],
                    value["learning_rate"],
                )
                for value in ADAPT.TOPOLOGY_SPECS.values()
            },
            {
                (1, 64, 64, 1988, 5e-5),
                (8, 8, 64, 1988, 5e-5),
                (16, 4, 64, 1988, 5e-5),
                (8, 64, 512, 248, 3e-5),
                (16, 32, 512, 248, 3e-5),
                (8, 128, 1024, 124, 3e-5),
                (8, 256, 2048, 62, 3e-5),
                (16, 64, 1024, 124, 3e-5),
                (16, 64, 1024, 124, 6e-5),
            },
        )
        self.assertEqual(
            ADAPT.CANDIDATE_EPOCHS,
            (
                1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80, 100, 120,
                140, 160, 180, 200, 240, 280, 320, 360, 400,
            ),
        )
        self.assertEqual(ADAPT.TOTAL_EPOCHS, 400)
        self.assertEqual(
            ADAPT.RESUME_EPOCHS,
            (40, 80, 120, 160, 200, 240, 280, 320, 360, 400),
        )
        self.assertEqual(ADAPT.THROUGHPUT_WARMUP_UPDATES, 20)
        self.assertEqual(ADAPT.THROUGHPUT_TIMED_UPDATES, 50)
        self.assertEqual(
            ADAPT._probe_epoch_update_counts(124),
            (70,),
        )
        self.assertEqual(
            ADAPT._probe_epoch_update_counts(62),
            (62, 8),
        )
        self.assertEqual(
            ADAPT.CHECKPOINT_FORMAT,
            "semtalk_show_base_official_adapt_checkpoint_v1",
        )

    def test_candidate_ready_publication_is_atomic_and_manifest_is_immutable(
        self,
    ) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("os.link(temporary, path)", source)
        self.assertIn("candidate_manifest_snapshots", source)
        self.assertIn('"immutable_snapshot": True', source)
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "candidate_receipts" / "epoch-0001.json"
            ADAPT._write_new_json(receipt, {"epoch": 1, "status": "ready"})
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8")),
                {"epoch": 1, "status": "ready"},
            )
            self.assertEqual(
                [path.name for path in receipt.parent.iterdir()],
                [receipt.name],
            )
            with self.assertRaises(FileExistsError):
                ADAPT._write_new_json(receipt, {"epoch": 2})
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8"))["epoch"],
                1,
            )

    def test_committed_schedules_and_legacy_anchor_are_pinned(self) -> None:
        schedule = (
            REPOSITORY
            / "configs/show_base/semtalk_base_long_schedule_20260731.json"
        )
        anchor = (
            REPOSITORY
            / "configs/show_base/"
            "semtalk_base_long_trajectory_anchor_20260731.json"
        )
        self.assertEqual(
            _sha(schedule),
            "013f8ade256f20579f9d545ac44da681238ed56af1cb687fc6fe9356c0bbefa3",
        )
        self.assertEqual(
            _sha(anchor),
            "e27c27a0da2793f44608b618d08356df0b60f55ae039434a228c50ff73028cf2",
        )
        self.assertEqual(
            _sha(FRESH_SCHEDULE),
            "6006fe341e5e5e5e1af4e0ae8ef7c67207c213b2d5cf8efb77fcd0054f39e7a6",
        )
        self.assertEqual(
            _sha(TOPOLOGY_GATE_SPEC),
            "1ee9ae31e2ca735265972022c26a82ba2789f7538a128631168af4e86f7808c4",
        )

    def test_protocol_is_three_forward_and_vq_free(self) -> None:
        args = argparse.Namespace(
            learning_rate=5e-5,
            precision="bf16",
            schedule_json="/frozen/schedule.json",
            expected_schedule_sha256="a" * 64,
            trajectory_anchor_json="/frozen/anchor.json",
            expected_trajectory_anchor_sha256="b" * 64,
            trajectory_mode=ADAPT.LEGACY_TRAJECTORY_MODE,
            loader_workers=4,
            seed=43,
            topology_mode=ADAPT.W16_GLOBAL64_MODE,
            formal_host_slot=0,
            formal_master_addr="master.example",
            formal_master_port=29601,
            formal_run_id="formal-run-001",
        )
        protocol = ADAPT.protocol_receipt(
            args,
            contract_receipts={
                "trajectory_anchor": {
                    "mode": ADAPT.LEGACY_TRAJECTORY_MODE,
                    "path": "/frozen/anchor.json",
                    "sha256": "b" * 64,
                }
            },
            topology_gate_spec={
                "path": str(TOPOLOGY_GATE_SPEC),
                "sha256": _sha(TOPOLOGY_GATE_SPEC),
            },
        )
        self.assertEqual(
            protocol["forward_contract"]["forwards_per_optimizer_step"], 3
        )
        self.assertTrue(
            protocol["forward_contract"]["audio_conditioned_main_forward"]
        )
        self.assertTrue(
            protocol["forward_contract"]["masked_self_forward"]
        )
        self.assertTrue(
            protocol["forward_contract"]["word_auxiliary_forward"]
        )
        self.assertEqual(
            protocol["forward_contract"][
                "masked_self_ce_published_source_semantics"
            ],
            "repeat_rvq_level_5_six_times_divided_by_6_v1",
        )
        self.assertFalse(protocol["vq_models_in_training_graph"])
        self.assertEqual(
            protocol["initialization"]["sha256"],
            "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603",
        )
        self.assertEqual(
            protocol["determinism"]["cublas_workspace_config"],
            ":4096:8",
        )
        self.assertEqual(protocol["determinism"]["python_hash_seed"], "43")
        self.assertEqual(
            protocol["distributed_topology"]["nodes"],
            [
                {
                    "node_rank": 0,
                    "host_slot": 0,
                    "hostname": ADAPT.FORMAL_HOST_BY_SLOT[0],
                    "rank_range": list(range(0, 8)),
                },
                {
                    "node_rank": 1,
                    "host_slot": 1,
                    "hostname": ADAPT.FORMAL_HOST_BY_SLOT[1],
                    "rank_range": list(range(8, 16)),
                },
            ],
        )
        self.assertTrue(
            protocol["determinism"][
                "python_hash_seed_required_at_interpreter_start"
            ]
        )
        self.assertFalse(protocol["determinism"]["cudnn_benchmark"])
        self.assertFalse(protocol["determinism"]["matmul_tf32"])
        self.assertEqual(
            protocol["determinism"]["lmdb_worker_binding"],
            "fork_inherited_pinned_data_fd_reverified_at_worker_init_"
            "and_before_and_after_open",
        )

    def test_runtime_source_forbids_benchmark_and_tf32_shortcuts(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("torch_module.use_deterministic_algorithms(True", source)
        self.assertIn("torch_module.backends.cudnn.benchmark = False", source)
        self.assertIn("torch_module.backends.cudnn.deterministic = True", source)
        self.assertIn("torch_module.backends.cuda.matmul.allow_tf32 = False", source)
        self.assertNotIn("torch.backends.cudnn.benchmark = True", source)
        self.assertIn(
            "worker_info.dataset.assert_source_unchanged(full_hash=False)",
            source,
        )

    def test_python_hash_seed_is_fail_closed_at_process_entry(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "PYTHONHASHSEED must be set before interpreter startup",
            ):
                ADAPT._prepare_deterministic_environment(seed=43)
        with mock.patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            clear=True,
        ):
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT._prepare_deterministic_environment(seed=43)
        with mock.patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "43",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            clear=True,
        ):
            ADAPT._prepare_deterministic_environment(seed=43)

    def test_lmdb_open_consumes_pinned_data_inode_during_rename_race(
        self,
    ) -> None:
        """A swap-and-restore cannot redirect the leaf opened by LMDB."""

        show_base_path = REPOSITORY / "dataloaders" / "show_base.py"
        fake_lmdb = types.ModuleType("lmdb")
        fake_lmdb.Environment = object
        fake_numpy = types.ModuleType("numpy")
        fake_torch = types.ModuleType("torch")
        fake_torch.utils = types.SimpleNamespace(
            data=types.SimpleNamespace(Dataset=object)
        )

        with tempfile.TemporaryDirectory() as temporary:
            lmdb_dir = Path(temporary) / "base.lmdb"
            lmdb_dir.mkdir()
            data_path = lmdb_dir / "data.mdb"
            lock_path = lmdb_dir / "lock.mdb"
            original = b"verified-original-data-inode"
            replacement = b"adversarial-replacement-inode"
            data_path.write_bytes(original)
            lock_path.write_bytes(b"verified-lock")
            _, binding = ADAPT._verified_lmdb_receipt(lmdb_dir)
            opened: dict[str, object] = {}

            class _FakeEnvironment:
                def __init__(self) -> None:
                    self.closed = False

                def close(self) -> None:
                    self.closed = True

            fake_environment = _FakeEnvironment()

            def fake_open(path: str, **kwargs: object) -> object:
                saved = lmdb_dir / "data.mdb.verified"
                os.replace(data_path, saved)
                data_path.write_bytes(replacement)
                try:
                    opened["path"] = path
                    descriptor_path = Path(path)
                    if not descriptor_path.exists():
                        descriptor_path = Path(
                            path.replace("/proc/self/fd/", "/dev/fd/")
                        )
                    opened["bytes"] = descriptor_path.read_bytes()
                    opened["kwargs"] = kwargs
                finally:
                    data_path.unlink()
                    os.replace(saved, data_path)
                return fake_environment

            fake_lmdb.open = fake_open
            spec = importlib.util.spec_from_file_location(
                "show_base_pinned_inode_test",
                show_base_path,
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(
                sys.modules,
                {
                    "lmdb": fake_lmdb,
                    "numpy": fake_numpy,
                    "torch": fake_torch,
                },
            ):
                spec.loader.exec_module(module)

            dataset = object.__new__(module._NPZLMDB)
            dataset.path = lmdb_dir
            dataset.required_fields = ()
            dataset.expected_binding = binding
            dataset._dirfd = None
            dataset._datafd = None
            dataset._env = None
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "require fork workers",
                ):
                    dataset.__getstate__()
                try:
                    if Path("/proc/self/fd").is_dir():
                        environment = dataset._open()
                    else:
                        # Production is Linux-only.  On macOS, preserve the
                        # /proc path passed to lmdb.open while the fake opener
                        # dereferences its /dev/fd equivalent above.
                        class _FormalProcPath:
                            def __init__(self, value: str):
                                self.value = value

                            def exists(self) -> bool:
                                return True

                            def __str__(self) -> str:
                                return self.value

                        with mock.patch.object(module, "Path", _FormalProcPath):
                            environment = dataset._open()
                except RuntimeError as error:
                    # Renaming the directory entry normally changes directory
                    # metadata, so the post-open check must fail closed.
                    self.assertIn("directory inode/metadata changed", str(error))
                    self.assertTrue(fake_environment.closed)
                else:
                    # Filesystems with coarser metadata clocks may not expose
                    # the swap, but LMDB still received the pinned leaf inode.
                    environment.close()
                self.assertEqual(opened["bytes"], original)
                self.assertRegex(str(opened["path"]), r"^/proc/self/fd/\d+$")
                self.assertFalse(opened["kwargs"]["subdir"])
                self.assertFalse(opened["kwargs"]["lock"])
                digest, identity = module._sha256_descriptor(dataset._datafd)
                self.assertEqual(
                    digest,
                    binding["files"]["data.mdb"]["sha256"],
                )
                self.assertEqual(
                    (identity["device"], identity["inode"]),
                    (
                        binding["files"]["data.mdb"]["identity"]["device"],
                        binding["files"]["data.mdb"]["identity"]["inode"],
                    ),
                )
            finally:
                dataset.__del__()

    def test_e30_and_speaker2_are_hard_rejected(self) -> None:
        for label in (
            "/weights/e30/best_semtalk_base.bin",
            "/weights/E_30/best_semtalk_base.bin",
            "/weights/speaker2/best_semtalk_base.bin",
            "SPEAKER-2-adaptation",
        ):
            with self.subTest(label=label):
                with self.assertRaises(ADAPT.AdaptationContractError):
                    ADAPT.reject_forbidden_source_labels(label)
        ADAPT.reject_forbidden_source_labels(
            "/weights/all_speakers/best_semtalk_base.bin"
        )
        ADAPT.reject_forbidden_source_labels(
            "/runs/all_speakers/base_epoch_e300.bin"
        )

    def test_argument_contract_rejects_wrong_batch_or_epochs(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_NODE_RANK[0]
            ),
        ):
            ADAPT.validate_args(args)
        args.local_batch_size = 32
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT.validate_args(args)
        args.local_batch_size = 64
        args.epochs = 200
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT.validate_args(args)

    def test_single_node_worker_host_slot_is_audited_and_allowed(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        args.formal_host_slot = 1
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_SLOT[1]
            ),
        ):
            ADAPT.validate_args(args)
            protocol = ADAPT.protocol_receipt(
                args,
                contract_receipts={
                    "trajectory_anchor": {
                        "mode": ADAPT.FRESH_TRAJECTORY_MODE,
                        "path": "/frozen/fresh-lineage.json",
                        "sha256": "f" * 64,
                    }
                },
                topology_gate_spec={
                    "path": str(TOPOLOGY_GATE_SPEC),
                    "sha256": _sha(TOPOLOGY_GATE_SPEC),
                },
            )
        self.assertEqual(
            protocol["distributed_topology"]["nodes"],
            [
                {
                    "node_rank": 0,
                    "host_slot": 1,
                    "hostname": ADAPT.FORMAL_HOST_BY_SLOT[1],
                    "rank_range": list(range(0, 8)),
                }
            ],
        )

    def test_host_slot_and_hostname_mismatch_is_rejected(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_SLOT[1]
            ),
        ):
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "node/host",
            ):
                ADAPT.validate_args(args)

    def test_w16_node_rank_must_equal_physical_host_slot(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        specification = ADAPT.TOPOLOGY_SPECS[ADAPT.W16_GLOBAL512_MODE]
        args.topology_mode = ADAPT.W16_GLOBAL512_MODE
        args.local_batch_size = specification["local_batch_size"]
        args.learning_rate = specification["learning_rate"]
        args.precision = specification["precision"]
        args.formal_node_rank = 1
        args.formal_host_slot = 0
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_SLOT[0]
            ),
        ):
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "node/host",
            ):
                ADAPT.validate_args(args)

    def test_short_quality_requires_gate_not_selection_for_accelerations(
        self,
    ) -> None:
        parser = ADAPT.build_parser()
        for mode in (
            ADAPT.W8_GLOBAL512_MODE,
            ADAPT.W16_GLOBAL512_MODE,
            ADAPT.W8_GLOBAL1024_MODE,
            ADAPT.W8_GLOBAL2048_MODE,
            ADAPT.W16_GLOBAL1024_MODE,
            ADAPT.W16_GLOBAL1024_LR6E5_MODE,
        ):
            with self.subTest(mode=mode):
                specification = ADAPT.TOPOLOGY_SPECS[mode]
                args = parser.parse_args(_base_cli())
                args.mode = ADAPT.SHORT_QUALITY_MODE
                args.epochs = ADAPT.SHORT_QUALITY_TOTAL_EPOCHS
                args.topology_mode = mode
                args.local_batch_size = specification["local_batch_size"]
                args.learning_rate = specification["learning_rate"]
                args.precision = specification["precision"]
                args.throughput_gate_report = "/sealed/throughput.json"
                args.expected_throughput_gate_sha256 = "d" * 64
                args.topology_selection_report = None
                args.expected_topology_selection_sha256 = None
                with mock.patch.object(
                    ADAPT.os,
                    "uname",
                    return_value=types.SimpleNamespace(
                        nodename=ADAPT.FORMAL_HOST_BY_NODE_RANK[0]
                    ),
                ):
                    ADAPT.validate_args(args)
                self.assertEqual(
                    ADAPT._run_purpose(args),
                    ADAPT.RUN_PURPOSE_SHORT_QUALITY,
                )
                self.assertEqual(
                    ADAPT._target_epochs(args),
                    list(ADAPT.SHORT_QUALITY_EPOCHS),
                )

    def test_throughput_probe_passes_sampler_epoch_to_every_update(self) -> None:
        source = inspect.getsource(ADAPT._run_throughput_gate)
        self.assertEqual(
            source.count("epoch=batch_sampler_epoch,"),
            2,
            "warmup and timed throughput updates must both use the sampler epoch",
        )
        self.assertNotIn("precision=args.precision,\n            epoch=0,", source)
        self.assertEqual(
            ADAPT._probe_epoch_update_counts(62),
            (62, 8),
        )

    def test_short_quality_rejects_selection_and_non_e32_target(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        args.mode = ADAPT.SHORT_QUALITY_MODE
        args.epochs = ADAPT.SHORT_QUALITY_TOTAL_EPOCHS
        args.throughput_gate_report = "/sealed/throughput.json"
        args.expected_throughput_gate_sha256 = "d" * 64
        args.topology_selection_report = "/forbidden/selection.json"
        args.expected_topology_selection_sha256 = "e" * 64
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_NODE_RANK[0]
            ),
        ):
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "forbids topology selection",
            ):
                ADAPT.validate_args(args)
            args.topology_selection_report = None
            args.expected_topology_selection_sha256 = None
            args.epochs = ADAPT.TOTAL_EPOCHS
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "exactly 32 epochs",
            ):
                ADAPT.validate_args(args)

    def test_formal_train_without_topology_selection_is_rejected(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        args.mode = "train"
        args.throughput_gate_report = "/sealed/throughput.json"
        args.expected_throughput_gate_sha256 = "d" * 64
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_NODE_RANK[0]
            ),
        ):
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "formal training requires one topology selected",
            ):
                ADAPT.validate_args(args)

    def test_frozen_receipt_declares_run_purpose_and_target_epochs(self) -> None:
        receipt = ADAPT._frozen_receipt(
            source={"source": True},
            official_base={"official": True},
            speaker_initialization={"speaker": True},
            dataset={"dataset": True},
            protocol={"protocol": True},
            long_contract={"contract": True},
            topology={"topology": True},
            run_purpose=ADAPT.RUN_PURPOSE_SHORT_QUALITY,
            target_epochs=ADAPT.SHORT_QUALITY_EPOCHS,
        )
        self.assertEqual(
            receipt["run_purpose"],
            ADAPT.RUN_PURPOSE_SHORT_QUALITY,
        )
        self.assertEqual(
            receipt["target_epochs"],
            list(ADAPT.SHORT_QUALITY_EPOCHS),
        )
        self.assertEqual(
            receipt["receipt_sha256"],
            ADAPT.canonical_json_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_sha256"
                }
            ),
        )
    def test_train_argument_contract_allows_sealed_w1(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        args.mode = "train"
        args.topology_mode = ADAPT.OFFICIAL_W1_REFERENCE_MODE
        args.local_batch_size = 64
        args.learning_rate = 5e-5
        args.precision = "fp32"
        args.topology_selection_report = "/sealed/topology-selection.json"
        args.expected_topology_selection_sha256 = "d" * 64
        args.throughput_gate_report = "/sealed/throughput.json"
        args.expected_throughput_gate_sha256 = "e" * 64
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_NODE_RANK[0]
            ),
        ):
            ADAPT.validate_args(args)

    def test_fresh_selected_vq_args_forbid_external_anchor(self) -> None:
        parser = ADAPT.build_parser()
        args = parser.parse_args(_base_cli())
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_NODE_RANK[0]
            ),
        ):
            ADAPT.validate_args(args)
        args.trajectory_anchor_json = "/frozen/old-anchor.json"
        args.expected_trajectory_anchor_sha256 = "d" * 64
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "requires the hash-pinned five-stage SHOW selection",
        ):
            ADAPT.validate_args(args)


class OfficialBaseTopologyGateContracts(unittest.TestCase):
    def test_topology_independent_authority_excludes_matrix_precision(
        self,
    ) -> None:
        fp32 = _frozen_gate_fixture(
            receipt_sha256="a" * 64,
            topology_receipt_sha256="b" * 64,
            trajectory_mode=ADAPT.FRESH_TRAJECTORY_MODE,
            run_purpose=ADAPT.RUN_PURPOSE_THROUGHPUT,
            target_epochs=[],
        )
        fp32["protocol"]["precision"] = "fp32"
        bf16 = json.loads(json.dumps(fp32))
        bf16["protocol"]["precision"] = "bf16"
        self.assertEqual(
            ADAPT._topology_independent_gate_semantic_sha256(fp32),
            ADAPT._topology_independent_gate_semantic_sha256(bf16),
        )

        changed_dataset = json.loads(json.dumps(fp32))
        changed_dataset["dataset"]["data_mdb_sha256"] = "c" * 64
        self.assertNotEqual(
            ADAPT._topology_independent_gate_semantic_sha256(fp32),
            ADAPT._topology_independent_gate_semantic_sha256(changed_dataset),
        )

    def test_matrix_precision_remains_fail_closed_per_mode(self) -> None:
        args = ADAPT.build_parser().parse_args(_base_cli())
        self.assertEqual(args.topology_mode, ADAPT.W8_GLOBAL512_MODE)
        args.precision = "fp32"
        with mock.patch.object(
            ADAPT.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=ADAPT.FORMAL_HOST_BY_NODE_RANK[0]
            ),
        ):
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "precision differs",
            ):
                ADAPT.validate_args(args)

    def test_gate_spec_is_immutable_and_w1_is_fp32_only(self) -> None:
        receipt = ADAPT.validate_topology_gate_spec(
            argparse.Namespace(
                topology_gate_spec=TOPOLOGY_GATE_SPEC,
                expected_topology_gate_spec_sha256=_sha(
                    TOPOLOGY_GATE_SPEC
                ),
                topology_mode=ADAPT.OFFICIAL_W1_REFERENCE_MODE,
            )
        )
        self.assertEqual(receipt["selected_probe_mode"], ADAPT.OFFICIAL_W1_REFERENCE_MODE)
        self.assertEqual(
            ADAPT.TOPOLOGY_SPECS[ADAPT.OFFICIAL_W1_REFERENCE_MODE][
                "precision"
            ],
            "fp32",
        )
        self.assertTrue(
            ADAPT.TOPOLOGY_SPECS[ADAPT.OFFICIAL_W1_REFERENCE_MODE][
                "formal_training_eligible"
            ]
        )
        self.assertTrue(
            all(
                spec["precision"] == "bf16"
                for mode, spec in ADAPT.TOPOLOGY_SPECS.items()
                if mode != ADAPT.OFFICIAL_W1_REFERENCE_MODE
            )
        )

    def test_every_registered_topology_lr_pair_can_win_by_measured_eta(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        for expected in modes:
            with self.subTest(expected=expected):
                eta = {
                    mode: 10_000.0 + index
                    for index, mode in enumerate(modes)
                }
                eta[expected] = 10.0
                probes, quality = _topology_selection_inputs(eta)
                selection = SELECTOR.select_topology(
                    probes,
                    quality,
                    gate_spec_sha256="b" * 64,
                    quality_gate_spec_sha256="f" * 64,
                )
                self.assertEqual(selection["selected"]["mode"], expected)
                self.assertEqual(selection["candidate_modes"], modes)
                self.assertEqual(
                    selection["selection_decision_branch"],
                    "fastest_quality_safe_finite_under_24h",
                )

    def test_training_consumer_accepts_a_sealed_w1_selection(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 1_000.0 + index for index, mode in enumerate(modes)}
        eta[ADAPT.OFFICIAL_W1_REFERENCE_MODE] = 10.0
        probes, quality = _topology_selection_inputs(eta)
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-w1-selection-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(
                    selection,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            receipt = _consume_topology_selection(
                selection=selection,
                path=path,
                topology_mode=ADAPT.OFFICIAL_W1_REFERENCE_MODE,
                throughput_gate=_validated_throughput_projection(
                    selection["selected"]
                ),
            )
        self.assertEqual(
            receipt["selected"]["mode"],
            ADAPT.OFFICIAL_W1_REFERENCE_MODE,
        )

    def test_training_consumer_replays_cross_mode_validation_authority(
        self,
    ) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 1_000.0 + index for index, mode in enumerate(modes)}
        probes, quality = _topology_selection_inputs(eta)
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        selection["quality_reports"][1]["val_inputs_receipt"][
            "sha256"
        ] = "0" * 64
        selection["receipt_sha256"] = ADAPT.canonical_json_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "receipt_sha256"
            }
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-cross-mode-authority-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(
                    selection,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "missing, forged, or stale",
            ):
                _consume_topology_selection(
                    selection=selection,
                    path=path,
                    topology_mode=str(selection["selected"]["mode"]),
                    throughput_gate=_validated_throughput_projection(
                        selection["selected"]
                    ),
                )

    def test_no_quality_safe_topology_under_24h_fails_closed(self) -> None:
        probes, quality = _topology_selection_inputs(
            {
                mode: SELECTOR.MAX_TRAINING_SECONDS + index + 1.0
                for index, mode in enumerate(ADAPT.TOPOLOGY_SPECS)
            }
        )
        with self.assertRaisesRegex(
            SELECTOR.TopologySelectionError,
            "no quality-safe finite topology meets the 24-hour median and "
            "22-hour p99",
        ):
            SELECTOR.select_topology(
                probes,
                quality,
                gate_spec_sha256="b" * 64,
                quality_gate_spec_sha256="f" * 64,
            )

    def test_over_budget_non_w1_quality_can_be_safely_skipped(self) -> None:
        eta = {
            ADAPT.OFFICIAL_W1_REFERENCE_MODE: 10_000.0,
            ADAPT.W8_GLOBAL64_MODE: 8_000.0,
            ADAPT.W16_GLOBAL64_MODE: 90_000.0,
            ADAPT.W8_GLOBAL512_MODE: 7_000.0,
            ADAPT.W16_GLOBAL512_MODE: 90_001.0,
            ADAPT.W8_GLOBAL1024_MODE: 8_500.0,
            ADAPT.W8_GLOBAL2048_MODE: 8_600.0,
            ADAPT.W16_GLOBAL1024_MODE: 8_400.0,
            ADAPT.W16_GLOBAL1024_LR6E5_MODE: 8_450.0,
        }
        probes, quality = _topology_selection_inputs(eta)
        skip_modes = {
            ADAPT.W16_GLOBAL64_MODE,
            ADAPT.W16_GLOBAL512_MODE,
        }
        skips = [
            _over_budget_quality_skip(probe, position=index)
            for index, probe in enumerate(probes)
            if probe["mode"] in skip_modes
        ]
        measured = [
            report for report in quality if report["mode"] not in skip_modes
        ]
        selection = SELECTOR.select_topology(
            probes,
            measured,
            quality_skips=skips,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(
            selection["selected"]["mode"], ADAPT.W8_GLOBAL512_MODE
        )
        self.assertEqual(
            selection["quality_decisions"][ADAPT.W16_GLOBAL64_MODE][
                "status"
            ],
            "skipped_over_eta_budget",
        )
        self.assertEqual(
            [probe["mode"] for probe in selection["probes"]],
            list(ADAPT.TOPOLOGY_SPECS),
        )

    def test_within_budget_and_w1_quality_cannot_be_skipped(self) -> None:
        eta = {
            mode: 10_000.0 + index
            for index, mode in enumerate(ADAPT.TOPOLOGY_SPECS)
        }
        probes, quality = _topology_selection_inputs(eta)
        candidate = probes[1]
        skip = _over_budget_quality_skip(candidate, position=1)
        measured = [
            report
            for report in quality
            if report["mode"] != candidate["mode"]
        ]
        with self.assertRaisesRegex(
            SELECTOR.TopologySelectionError,
            "not bound to one over-budget probe",
        ):
            SELECTOR.select_topology(
                probes,
                measured,
                quality_skips=[skip],
                gate_spec_sha256="b" * 64,
                quality_gate_spec_sha256="f" * 64,
            )

        eta[ADAPT.OFFICIAL_W1_REFERENCE_MODE] = 90_000.0
        probes, quality = _topology_selection_inputs(eta)
        reference_probe = probes[0]
        reference_skip = _over_budget_quality_skip(
            reference_probe,
            position=0,
        )
        measured = quality[1:]
        with self.assertRaisesRegex(
            SELECTOR.TopologySelectionError,
            "W1 full e1/e2/e4/e8/e16/e32 quality reference is mandatory",
        ):
            SELECTOR.select_topology(
                probes,
                measured,
                quality_skips=[reference_skip],
                gate_spec_sha256="b" * 64,
                quality_gate_spec_sha256="f" * 64,
            )

    def test_training_consumer_accepts_over_budget_skip_receipts(self) -> None:
        eta = {
            ADAPT.OFFICIAL_W1_REFERENCE_MODE: 10_000.0,
            ADAPT.W8_GLOBAL64_MODE: 8_000.0,
            ADAPT.W16_GLOBAL64_MODE: 90_000.0,
            ADAPT.W8_GLOBAL512_MODE: 7_000.0,
            ADAPT.W16_GLOBAL512_MODE: 90_001.0,
            ADAPT.W8_GLOBAL1024_MODE: 8_500.0,
            ADAPT.W8_GLOBAL2048_MODE: 8_600.0,
            ADAPT.W16_GLOBAL1024_MODE: 8_400.0,
            ADAPT.W16_GLOBAL1024_LR6E5_MODE: 8_450.0,
        }
        probes, quality = _topology_selection_inputs(eta)
        skip_modes = {
            ADAPT.W16_GLOBAL64_MODE,
            ADAPT.W16_GLOBAL512_MODE,
        }
        selection = SELECTOR.select_topology(
            probes,
            [
                report
                for report in quality
                if report["mode"] not in skip_modes
            ],
            quality_skips=[
                _over_budget_quality_skip(probe, position=index)
                for index, probe in enumerate(probes)
                if probe["mode"] in skip_modes
            ],
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-eta-skip-selection-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(
                    selection,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            receipt = _consume_topology_selection(
                selection=selection,
                path=path,
                topology_mode=ADAPT.W8_GLOBAL512_MODE,
                throughput_gate=_validated_throughput_projection(
                    selection["selected"]
                ),
            )
        self.assertEqual(
            receipt["selected"]["mode"], ADAPT.W8_GLOBAL512_MODE
        )

    def test_selection_tiebreak_is_p99_then_matrix_order(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 10_000.0 for mode in modes}
        p99 = {mode: 0.05 for mode in modes}
        p99[ADAPT.W16_GLOBAL64_MODE] = 0.01
        probes, quality = _topology_selection_inputs(
            eta, p99_seconds=p99
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(
            selection["selected"]["mode"], ADAPT.W16_GLOBAL64_MODE
        )

        probes, quality = _topology_selection_inputs(
            eta, p99_seconds={mode: 0.01 for mode in modes}
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(
            selection["selected"]["mode"],
            ADAPT.OFFICIAL_W1_REFERENCE_MODE,
        )

    def test_p99_total_updates_is_a_hard_22_hour_gate(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 10_000.0 for mode in modes}
        eta[ADAPT.W8_GLOBAL2048_MODE] = 1.0
        eta[ADAPT.W16_GLOBAL1024_MODE] = 2.0
        p99 = {mode: 0.01 for mode in modes}
        g2048_updates = ADAPT.TOPOLOGY_SPECS[
            ADAPT.W8_GLOBAL2048_MODE
        ]["updates_per_epoch"]
        p99[ADAPT.W8_GLOBAL2048_MODE] = (
            SELECTOR.MAX_P99_TRAINING_SECONDS
            / (g2048_updates * ADAPT.TOTAL_EPOCHS)
            + 0.001
        )
        probes, quality = _topology_selection_inputs(
            eta,
            p99_seconds=p99,
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(
            selection["selected"]["mode"],
            ADAPT.W16_GLOBAL1024_MODE,
        )

    def test_training_consumer_replays_p99_gate_and_selected_rank(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 10_000.0 + index for index, mode in enumerate(modes)}
        p99 = {mode: 0.01 for mode in modes}
        rejected_mode = ADAPT.W16_GLOBAL1024_MODE
        rejected_updates = ADAPT.TOPOLOGY_SPECS[rejected_mode][
            "updates_per_epoch"
        ]
        p99[rejected_mode] = (
            SELECTOR.MAX_P99_TRAINING_SECONDS
            / (rejected_updates * ADAPT.TOTAL_EPOCHS)
            + 0.5
        )
        probes, quality = _topology_selection_inputs(
            eta,
            p99_seconds=p99,
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertNotEqual(selection["selected"]["mode"], rejected_mode)
        selection["selected"] = next(
            probe for probe in selection["probes"] if probe["mode"] == rejected_mode
        )
        selection["receipt_sha256"] = ADAPT.canonical_json_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "receipt_sha256"
            }
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-forged-p99-selection-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(
                    selection,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "missing, forged, or stale",
            ):
                _consume_topology_selection(
                    selection=selection,
                    path=path,
                    topology_mode=rejected_mode,
                    throughput_gate=_validated_throughput_projection(
                        selection["selected"]
                    ),
                )

    def test_training_consumer_rejects_forged_low_p99_projection(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 10_000.0 + index for index, mode in enumerate(modes)}
        selected_mode = ADAPT.W16_GLOBAL1024_MODE
        eta[selected_mode] = 1.0
        probes, quality = _topology_selection_inputs(
            eta,
            p99_seconds={mode: 0.01 for mode in modes},
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(selection["selected"]["mode"], selected_mode)
        updates = ADAPT.TOPOLOGY_SPECS[selected_mode]["updates_per_epoch"]
        raw_p99 = (
            SELECTOR.MAX_P99_TRAINING_SECONDS
            / (updates * ADAPT.TOTAL_EPOCHS)
            + 0.5
        )
        raw_eta = raw_p99 * updates * ADAPT.TOTAL_EPOCHS
        with tempfile.TemporaryDirectory(
            prefix="semtalk-forged-low-p99-projection-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(
                    selection,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "missing, forged, or stale",
            ):
                _consume_topology_selection(
                    selection=selection,
                    path=path,
                    topology_mode=selected_mode,
                    throughput_gate=_validated_throughput_projection(
                        selection["selected"],
                        p99_seconds=raw_p99,
                        estimated_training_seconds=raw_eta,
                    ),
                )

    def test_training_consumer_recomputes_raw_quality_fgd(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        selected_mode = ADAPT.W8_GLOBAL512_MODE
        eta = {
            mode: 10_000.0 + index for index, mode in enumerate(modes)
        }
        eta[selected_mode] = 1.0
        probes, quality = _topology_selection_inputs(
            eta,
            p99_seconds={mode: 0.01 for mode in modes},
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(selection["selected"]["mode"], selected_mode)
        verified_selection = copy.deepcopy(selection)
        selected_quality = next(
            item
            for item in selection["quality_reports"]
            if item["mode"] == selected_mode
        )
        selected_quality["candidate_fgd"]["4"] = 999.0
        selection["receipt_sha256"] = ADAPT.canonical_json_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "receipt_sha256"
            }
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-forged-quality-fgd-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(
                    selection,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "missing, forged, or stale",
            ):
                _consume_topology_selection(
                    selection=selection,
                    verified_selection=verified_selection,
                    path=path,
                    topology_mode=selected_mode,
                    throughput_gate=_validated_throughput_projection(
                        selection["selected"]
                    ),
                )

    def test_training_consumer_rejects_extra_quality_decision_fields(
        self,
    ) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 1_000.0 + index for index, mode in enumerate(modes)}
        selected_mode = ADAPT.W16_GLOBAL1024_MODE
        eta[selected_mode] = 1.0
        probes, quality = _topology_selection_inputs(
            eta,
            p99_seconds={mode: 0.01 for mode in modes},
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        selection["quality_decisions"][selected_mode]["forged_extra"] = True
        selection["receipt_sha256"] = ADAPT.canonical_json_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "receipt_sha256"
            }
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-extra-quality-decision-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(
                    selection,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "missing, forged, or stale",
            ):
                _consume_topology_selection(
                    selection=selection,
                    path=path,
                    topology_mode=selected_mode,
                    throughput_gate=_validated_throughput_projection(
                        selection["selected"]
                    ),
                )

    def test_training_consumer_reloads_every_competitor_probe(self) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        fast = ADAPT.W8_GLOBAL512_MODE
        second = ADAPT.W16_GLOBAL512_MODE
        eta = {mode: 10_000.0 + index for index, mode in enumerate(modes)}
        eta[fast] = 1.0
        eta[second] = 2.0
        probes, quality = _topology_selection_inputs(
            eta,
            p99_seconds={mode: 0.01 for mode in modes},
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        verified_selection = copy.deepcopy(selection)
        fast_probe = next(
            item for item in selection["probes"] if item["mode"] == fast
        )
        fast_probe["estimated_training_seconds"] = 20_000.0
        selection["selected"] = next(
            item for item in selection["probes"] if item["mode"] == second
        )
        selection["receipt_sha256"] = ADAPT.canonical_json_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "receipt_sha256"
            }
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-forged-competitor-probe-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(selection, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "missing, forged, or stale",
            ):
                _consume_topology_selection(
                    selection=selection,
                    verified_selection=verified_selection,
                    path=path,
                    topology_mode=second,
                    throughput_gate=_validated_throughput_projection(
                        selection["selected"]
                    ),
                )

    def test_reload_opens_every_probe_report_and_skip_exactly_once(self) -> None:
        from scripts.show_base import (
            select_base_training_topology as canonical_selector,
        )

        modes = list(ADAPT.TOPOLOGY_SPECS)
        eta = {mode: 1_000.0 + index for index, mode in enumerate(modes)}
        probes, quality = _topology_selection_inputs(eta)
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        probe_by_mode = {row["mode"]: row for row in selection["probes"]}
        quality_by_mode = {
            row["mode"]: row for row in selection["quality_reports"]
        }
        with (
            mock.patch.object(
                canonical_selector,
                "validate_probe",
                side_effect=lambda mode, *_args, **_kwargs: copy.deepcopy(
                    probe_by_mode[mode]
                ),
            ) as probe_validator,
            mock.patch.object(
                canonical_selector,
                "validate_quality_report",
                side_effect=lambda mode, *_args, **_kwargs: copy.deepcopy(
                    quality_by_mode[mode]
                ),
            ) as quality_validator,
            mock.patch.object(
                canonical_selector,
                "validate_quality_skip",
            ) as skip_validator,
        ):
            reloaded = ADAPT._reload_topology_selection_artifacts(
                selection,
                topology_gate_spec_sha256="b" * 64,
            )
        self.assertIsNotNone(reloaded)
        self.assertEqual(probe_validator.call_count, len(modes))
        self.assertEqual(quality_validator.call_count, len(modes))
        self.assertEqual(skip_validator.call_count, 0)

        skip_mode = ADAPT.W16_GLOBAL64_MODE
        skip_eta = dict(eta)
        skip_eta[skip_mode] = SELECTOR.MAX_TRAINING_SECONDS + 1.0
        skip_probes, skip_quality = _topology_selection_inputs(skip_eta)
        skip_receipts = [
            _over_budget_quality_skip(
                next(row for row in skip_probes if row["mode"] == skip_mode),
                position=modes.index(skip_mode),
            )
        ]
        skip_selection = SELECTOR.select_topology(
            skip_probes,
            [row for row in skip_quality if row["mode"] != skip_mode],
            quality_skips=skip_receipts,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        skip_probe_by_mode = {
            row["mode"]: row for row in skip_selection["probes"]
        }
        measured_by_mode = {
            row["mode"]: row for row in skip_selection["quality_reports"]
        }
        skip_by_mode = {
            row["mode"]: row for row in skip_selection["quality_skips"]
        }
        with (
            mock.patch.object(
                canonical_selector,
                "validate_probe",
                side_effect=lambda mode, *_args, **_kwargs: copy.deepcopy(
                    skip_probe_by_mode[mode]
                ),
            ) as probe_validator,
            mock.patch.object(
                canonical_selector,
                "validate_quality_report",
                side_effect=lambda mode, *_args, **_kwargs: copy.deepcopy(
                    measured_by_mode[mode]
                ),
            ) as quality_validator,
            mock.patch.object(
                canonical_selector,
                "validate_quality_skip",
                side_effect=lambda mode, *_args, **_kwargs: copy.deepcopy(
                    skip_by_mode[mode]
                ),
            ) as skip_validator,
        ):
            skip_reloaded = ADAPT._reload_topology_selection_artifacts(
                skip_selection,
                topology_gate_spec_sha256="b" * 64,
            )
        self.assertIsNotNone(skip_reloaded)
        self.assertEqual(probe_validator.call_count, len(modes))
        self.assertEqual(quality_validator.call_count, len(modes) - 1)
        self.assertEqual(skip_validator.call_count, 1)

    def test_training_consumer_reloads_every_competitor_quality_report(
        self,
    ) -> None:
        modes = list(ADAPT.TOPOLOGY_SPECS)
        fast = ADAPT.W8_GLOBAL512_MODE
        second = ADAPT.W16_GLOBAL512_MODE
        eta = {mode: 10_000.0 + index for index, mode in enumerate(modes)}
        eta[fast] = 1.0
        eta[second] = 2.0
        probes, quality = _topology_selection_inputs(
            eta,
            p99_seconds={mode: 0.01 for mode in modes},
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        verified_selection = copy.deepcopy(selection)
        fast_quality = next(
            item
            for item in selection["quality_reports"]
            if item["mode"] == fast
        )
        fast_quality["candidate_fgd"]["4"] = 999.0
        fast_decision = selection["quality_decisions"][fast]
        epoch_four = next(
            item
            for item in fast_decision["comparisons"]
            if item["epoch"] == 4
        )
        epoch_four["candidate_fgd"] = 999.0
        epoch_four["pass"] = False
        fast_decision["all_trajectory_epochs_pass"] = False
        selection["selected"] = next(
            item for item in selection["probes"] if item["mode"] == second
        )
        selection["receipt_sha256"] = ADAPT.canonical_json_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "receipt_sha256"
            }
        )
        with tempfile.TemporaryDirectory(
            prefix="semtalk-forged-competitor-quality-", dir="/private/tmp"
        ) as raw:
            path = Path(raw) / "selection.json"
            path.write_text(
                json.dumps(selection, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "missing, forged, or stale",
            ):
                _consume_topology_selection(
                    selection=selection,
                    verified_selection=verified_selection,
                    path=path,
                    topology_mode=second,
                    throughput_gate=_validated_throughput_projection(
                        selection["selected"]
                    ),
                )

    def test_quality_margin_formula_is_max_not_sum(self) -> None:
        # At low FGD the absolute margin dominates; at high FGD the relative
        # margin dominates.  The two margins are never added together.
        self.assertEqual(SELECTOR.maximum_allowed_fgd(0.1), 0.11)
        self.assertEqual(SELECTOR.maximum_allowed_fgd(1.0), 1.02)
        self.assertNotEqual(SELECTOR.maximum_allowed_fgd(1.0), 1.03)

    def test_acceleration_requires_all_raw_replay_quality_epochs(self) -> None:
        probes, quality = _topology_selection_inputs(
            {
                ADAPT.OFFICIAL_W1_REFERENCE_MODE: 200_000.0,
                ADAPT.W8_GLOBAL64_MODE: 200_001.0,
                ADAPT.W16_GLOBAL64_MODE: 200_002.0,
                ADAPT.W8_GLOBAL512_MODE: 1_000.0,
                ADAPT.W16_GLOBAL512_MODE: 1_001.0,
                ADAPT.W8_GLOBAL1024_MODE: 1_002.0,
                ADAPT.W8_GLOBAL2048_MODE: 1_003.0,
                ADAPT.W16_GLOBAL1024_MODE: 1_004.0,
                ADAPT.W16_GLOBAL1024_LR6E5_MODE: 1_005.0,
            },
            failing_quality_modes={ADAPT.W8_GLOBAL512_MODE},
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(
            selection["selected"]["mode"], ADAPT.W16_GLOBAL512_MODE
        )
        self.assertFalse(
            selection["quality_decisions"][ADAPT.W8_GLOBAL512_MODE][
                "all_trajectory_epochs_pass"
            ]
        )

    def test_fixed_global_batch_ddp_must_match_w1_at_every_quality_epoch(self) -> None:
        probes, quality = _topology_selection_inputs(
            {
                ADAPT.OFFICIAL_W1_REFERENCE_MODE: 10_000.0,
                ADAPT.W8_GLOBAL64_MODE: 1_000.0,
                ADAPT.W16_GLOBAL64_MODE: 2_000.0,
                ADAPT.W8_GLOBAL512_MODE: 90_000.0,
                ADAPT.W16_GLOBAL512_MODE: 90_001.0,
                ADAPT.W8_GLOBAL1024_MODE: 90_002.0,
                ADAPT.W8_GLOBAL2048_MODE: 90_003.0,
                ADAPT.W16_GLOBAL1024_MODE: 90_004.0,
                ADAPT.W16_GLOBAL1024_LR6E5_MODE: 90_005.0,
            },
            failing_quality_modes={
                ADAPT.W8_GLOBAL64_MODE,
                ADAPT.W16_GLOBAL64_MODE,
            },
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(
            selection["selected"]["mode"],
            ADAPT.OFFICIAL_W1_REFERENCE_MODE,
        )
        for mode in (ADAPT.W8_GLOBAL64_MODE, ADAPT.W16_GLOBAL64_MODE):
            decision = selection["quality_decisions"][mode]
            self.assertFalse(decision["all_trajectory_epochs_pass"])
            self.assertEqual(
                [row["epoch"] for row in decision["comparisons"]],
                list(SELECTOR.QUALITY_EPOCHS),
            )

    def test_nonfinite_eta_is_never_ranked(self) -> None:
        probes, quality = _topology_selection_inputs(
            {
                ADAPT.OFFICIAL_W1_REFERENCE_MODE: float("nan"),
                ADAPT.W8_GLOBAL64_MODE: 1_000.0,
                ADAPT.W16_GLOBAL64_MODE: 2_000.0,
                ADAPT.W8_GLOBAL512_MODE: 3_000.0,
                ADAPT.W16_GLOBAL512_MODE: 4_000.0,
                ADAPT.W8_GLOBAL1024_MODE: 5_000.0,
                ADAPT.W8_GLOBAL2048_MODE: 6_000.0,
                ADAPT.W16_GLOBAL1024_MODE: 7_000.0,
                ADAPT.W16_GLOBAL1024_LR6E5_MODE: 8_000.0,
            }
        )
        selection = SELECTOR.select_topology(
            probes,
            quality,
            gate_spec_sha256="b" * 64,
            quality_gate_spec_sha256="f" * 64,
        )
        self.assertEqual(
            selection["selected"]["mode"], ADAPT.W8_GLOBAL64_MODE
        )

    def test_quality_gate_spec_is_hash_pinned_and_preregistered(self) -> None:
        receipt = SELECTOR.validate_quality_gate_spec(
            TOPOLOGY_QUALITY_GATE_SPEC,
            _sha(TOPOLOGY_QUALITY_GATE_SPEC),
        )
        self.assertEqual(
            receipt["payload"]["trajectory_epochs"],
            list(SELECTOR.QUALITY_EPOCHS),
        )
        self.assertEqual(
            receipt["payload"]["format"],
            "semtalk_show_base_topology_quality_gate_spec_v2",
        )
        self.assertNotIn(
            "raw_prediction_replay_required",
            receipt["payload"],
        )
        self.assertTrue(
            receipt["payload"]["diffsheg_validation_measurement_required"]
        )
        self.assertEqual(
            receipt["payload"]["validation_protocol"],
            "diffsheg_show_validation_fgd_v1",
        )
        self.assertEqual(
            receipt["payload"]["primary_metric"],
            "validation.diffsheg.metrics.fgd",
        )
        self.assertEqual(
            receipt["payload"]["candidate_quality_gate"]["modes"],
            list(ADAPT.TOPOLOGY_SPECS),
        )
        self.assertEqual(
            receipt["payload"]["measured_eta_constraint"][
                "maximum_estimated_training_seconds"
            ],
            86_400,
        )
        self.assertEqual(
            receipt["payload"]["measured_eta_constraint"][
                "maximum_p99_training_seconds"
            ],
            79_200,
        )
        self.assertTrue(
            receipt["payload"]["measured_eta_constraint"][
                "p99_total_updates_required"
            ]
        )
        skip_policy = receipt["payload"]["over_eta_budget_quality_skip"]
        self.assertTrue(skip_policy["w1_quality_report_required"])
        self.assertTrue(skip_policy["within_budget_quality_report_required"])
        self.assertEqual(
            skip_policy["receipt_format"], SELECTOR.QUALITY_SKIP_FORMAT
        )

    def test_node_local_inode_differences_are_preserved_not_compared(self) -> None:
        semantic = {
            "format": "semtalk_show_base_selected_feature_dataset_receipt_v1",
            "data_mdb_sha256": "a" * 64,
            "lock_mdb_sha256": "b" * 64,
            "lineage_sha256": "c" * 64,
        }
        nodes = []
        for node_rank, device in enumerate((243, 1_048_582)):
            nodes.append(
                {
                    "dataset": {
                        **semantic,
                        "lmdb_inode_binding": {
                            "format": "semtalk_show_base_lmdb_inode_binding_v1",
                            "directory_identity": {
                                "device": device,
                                "inode": 100 + node_rank,
                            },
                        },
                    }
                }
            )
        receipt = ADAPT._global_dataset_receipt(
            nodes, host_slots=(0, 1)
        )
        self.assertEqual(
            [
                (item["host_slot"], item["hostname"])
                for item in receipt["node_lmdb_inode_bindings"]
            ],
            [
                (0, ADAPT.FORMAL_HOST_BY_SLOT[0]),
                (1, ADAPT.FORMAL_HOST_BY_SLOT[1]),
            ],
        )
        self.assertEqual(
            [
                item["binding"]["directory_identity"]["device"]
                for item in receipt["node_lmdb_inode_bindings"]
            ],
            [243, 1_048_582],
        )
        nodes[1]["dataset"]["data_mdb_sha256"] = "9" * 64
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT._global_dataset_receipt(nodes, host_slots=(0, 1))


class OfficialBaseAdaptReceiptContracts(unittest.TestCase):
    class _FakeTensor:
        shape = (2, 3)
        dtype = "torch.float32"

        def is_floating_point(self) -> bool:
            return False

        def is_complex(self) -> bool:
            return False

    class _FakeTorch:
        envelope: object = None

        @classmethod
        def load(cls, *_: object, **__: object) -> object:
            return cls.envelope

        @staticmethod
        def is_tensor(value: object) -> bool:
            return isinstance(
                value, OfficialBaseAdaptReceiptContracts._FakeTensor
            )

    def _fresh_dataset_receipt(self) -> dict[str, object]:
        return {
            "format": (
                "semtalk_show_base_selected_feature_dataset_receipt_v1"
            ),
            "lmdb": "/cache/current-selected/base.lmdb",
            "entries": ADAPT.EXPECTED_TRAIN_SAMPLES,
            "train_clips": ADAPT.EXPECTED_TRAIN_CLIPS,
            "split": "train",
            "test_visible": False,
            "data_mdb_sha256": "1" * 64,
            "lock_mdb_sha256": "2" * 64,
            "summary": "/cache/current-selected/summary.json",
            "summary_sha256": "3" * 64,
            "lineage": "/cache/current-selected/lineage.json",
            "lineage_sha256": "4" * 64,
            "prerequisite_source": ADAPT.SHOW_VAL_SELECTED_SOURCE,
            "formal_checkpoints": {},
            "vq_models_in_training_graph": False,
            "vq_targets": "precomputed_frozen_lmdb_tensors",
            "prerequisite_selection": {
                "path": "/selection/current-five.json",
                "sha256": "5" * 64,
                "receipt_payload_sha256": "6" * 64,
            },
            "selected_prerequisite_sha256": {
                stage: str(index + 10) * 32
                for index, stage in enumerate(ADAPT.selected_contract.STAGES)
            },
            "global_verified_not_consumed": True,
            "lmdb_inode_binding": {
                "format": "semtalk_show_base_lmdb_inode_binding_v1",
                "directory_identity": {},
                "files": {},
            },
            "canonical_dataset_evidence": {
                "split_counts": dict(ADAPT.EXPECTED_SPLIT_COUNTS),
                "split_disjoint": True,
                "exact_once": True,
                "train_per_clip_ledger_exact": True,
                "test_rows_used_as_training_samples": False,
            },
        }

    def _fresh_long_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            schedule_json=str(FRESH_SCHEDULE),
            expected_schedule_sha256=_sha(FRESH_SCHEDULE),
            trajectory_mode=ADAPT.FRESH_TRAJECTORY_MODE,
            trajectory_anchor_json=None,
            expected_trajectory_anchor_sha256=None,
            expected_prerequisite_selection_sha256="5" * 64,
            expected_dataset_summary_sha256="3" * 64,
            expected_lineage_sha256="4" * 64,
            precision="bf16",
            learning_rate=3e-5,
            seed=43,
            loader_workers=4,
            topology_mode=ADAPT.W16_GLOBAL512_MODE,
        )

    def test_fresh_trajectory_binds_exact_selected_feature_lineage(self) -> None:
        dataset = self._fresh_dataset_receipt()
        receipt = ADAPT.validate_long_contract_receipts(
            self._fresh_long_args(),
            dataset_receipt=dataset,
        )
        trajectory = receipt["trajectory_anchor"]
        self.assertEqual(
            trajectory["mode"],
            ADAPT.FRESH_TRAJECTORY_MODE,
        )
        self.assertIsNone(trajectory["path"])
        self.assertEqual(trajectory["entries"], {})
        self.assertEqual(
            trajectory["feature_lineage_sha256"],
            dataset["lineage_sha256"],
        )
        self.assertEqual(
            trajectory["prerequisite_selection_sha256"],
            dataset["prerequisite_selection"]["sha256"],
        )
        self.assertEqual(
            trajectory["probe_optimizer_updates"],
            ADAPT.TRAJECTORY_PROBE_UPDATES,
        )

    def test_fresh_trajectory_rejects_feature_lineage_mismatch(self) -> None:
        dataset = self._fresh_dataset_receipt()
        dataset["lineage_sha256"] = "7" * 64
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "exact selected SHOW prerequisite/feature lineage",
        ):
            ADAPT.validate_long_contract_receipts(
                self._fresh_long_args(),
                dataset_receipt=dataset,
            )

    def test_fresh_trajectory_rejects_test_visible_features(self) -> None:
        dataset = self._fresh_dataset_receipt()
        dataset["split"] = "test"
        dataset["test_visible"] = True
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "exact selected SHOW prerequisite/feature lineage",
        ):
            ADAPT.validate_long_contract_receipts(
                self._fresh_long_args(),
                dataset_receipt=dataset,
            )

    def test_legacy_anchor_is_rejected_for_selected_show_vqs(self) -> None:
        schedule = (
            REPOSITORY
            / "configs"
            / "show_base"
            / "semtalk_base_long_schedule_20260731.json"
        )
        anchor = (
            REPOSITORY
            / "configs"
            / "show_base"
            / "semtalk_base_long_trajectory_anchor_20260731.json"
        )
        args = self._fresh_long_args()
        args.schedule_json = str(schedule)
        args.expected_schedule_sha256 = _sha(schedule)
        args.trajectory_mode = ADAPT.LEGACY_TRAJECTORY_MODE
        args.trajectory_anchor_json = str(anchor)
        args.expected_trajectory_anchor_sha256 = _sha(anchor)
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "does not match the frozen e400 contract",
        ):
            ADAPT.validate_long_contract_receipts(
                args,
                dataset_receipt=self._fresh_dataset_receipt(),
            )

    def test_official_checkpoint_filename_sha_envelope_and_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best_semtalk_base.bin"
            path.write_bytes(b"fixture-official-base")
            specification = {
                **ADAPT.OFFICIAL_BASE_SPEC,
                "sha256": _sha(path),
            }
            self._FakeTorch.envelope = {
                "epoch": ADAPT.OFFICIAL_BASE_EPOCH,
                "lrs": dict(ADAPT.OFFICIAL_BASE_LRS),
                "model_state": {
                    "module.weight": self._FakeTensor(),
                },
                "opt_state": {
                    "state": {},
                    "param_groups": [
                        {
                            **ADAPT.OFFICIAL_BASE_OPTIMIZER_GROUP,
                            "params": [],
                        }
                    ],
                },
            }
            with (
                mock.patch.object(
                    ADAPT, "OFFICIAL_BASE_OPTIMIZER_STATE_ENTRIES", 0
                ),
                mock.patch.object(
                    ADAPT, "OFFICIAL_BASE_OPTIMIZER_PARAMETERS", 0
                ),
            ):
                state, receipt = ADAPT.read_official_base_checkpoint(
                    path,
                    torch_module=self._FakeTorch,
                    specification=specification,
                )
            self.assertEqual(set(state), {"weight"})
            self.assertEqual(
                receipt["checkpoint_container_schema"],
                ["epoch", "lrs", "model_state", "opt_state"],
            )
            self.assertTrue(receipt["strict_state_dict_load"])
            self._FakeTorch.envelope = {
                "epoch": ADAPT.OFFICIAL_BASE_EPOCH,
                "lrs": dict(ADAPT.OFFICIAL_BASE_LRS),
                "model_state": {"weight": self._FakeTensor()},
                "opt_state": {
                    "state": {},
                    "param_groups": [
                        {
                            **ADAPT.OFFICIAL_BASE_OPTIMIZER_GROUP,
                            "params": [],
                        }
                    ],
                },
                "unexpected": True,
            }
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT.read_official_base_checkpoint(
                    path,
                    torch_module=self._FakeTorch,
                    specification=specification,
                )

    def _dataset_fixture(
        self, root: Path
    ) -> tuple[argparse.Namespace, dict[str, object], dict[str, object]]:
        lmdb = root / "base.lmdb"
        lmdb.mkdir()
        (lmdb / "data.mdb").write_bytes(b"official-base-data")
        (lmdb / "lock.mdb").write_bytes(b"official-base-lock")
        canonical_manifest_path = root / "canonical.jsonl"
        canonical_summary_path = root / "canonical_summary.json"
        canonical_lineage_path = root / "canonical_lineage.json"
        source_receipt = {
            "origin": ADAPT.EXPECTED_ORIGIN,
            "commit": "1" * 40,
            "tree": "2" * 40,
        }
        lineage_contract = {
            "source_receipt": source_receipt,
            "source_audio_sample_rate": 22_000,
            "hubert_target_sample_rate": 16_000,
            "audio_channel_protocol": {"fixture": True},
        }
        lineage_contract_sha = ADAPT._canonical_file_payload_sha256(
            lineage_contract
        )
        rows = []
        per_clip = []
        global_index = 0
        total_raw = 0
        total_usable = 0
        total_dropped = 0
        total_windows = 0
        for split, count in ADAPT.EXPECTED_SPLIT_COUNTS.items():
            for split_index in range(count):
                clip_id = f"{split}-{split_index:05d}"
                if split == "train":
                    if split_index < 2052:
                        frames = 270
                    elif split_index == 2052:
                        frames = 210
                    else:
                        frames = 240
                else:
                    frames = 240
                speaker_id = split_index % 4
                speaker = tuple(ADAPT.SHOW_SPEAKERS)[speaker_id]
                canonical_sha = format(global_index % 16, "x") * 64
                row = {
                    "global_index": global_index,
                    "split": split,
                    "clip_id": clip_id,
                    "speaker": speaker,
                    "speaker_id": speaker_id,
                    "frames": frames,
                    "canonical_npz": f"/canonical/{clip_id}.npz",
                    "canonical_npz_sha256": canonical_sha,
                    "lineage_contract_sha256": lineage_contract_sha,
                }
                rows.append(row)
                if split == "train":
                    usable = (frames // 30) * 30
                    dropped = frames - usable
                    windows = max(
                        0,
                        (usable - ADAPT.POSE_LENGTH) // 20 + 1,
                    )
                    per_clip.append(
                        {
                            "clip_id": clip_id,
                            "canonical_npz": row["canonical_npz"],
                            "canonical_npz_sha256": canonical_sha,
                            "audio_feature_npz": f"/audio/{clip_id}.npz",
                            "audio_feature_npz_sha256": "f" * 64,
                            "raw_frames": frames,
                            "usable_frames": usable,
                            "dropped_tail_frames": dropped,
                            "windows": windows,
                            "speaker_id": speaker_id,
                        }
                    )
                    total_raw += frames
                    total_usable += usable
                    total_dropped += dropped
                    total_windows += windows
                global_index += 1
        self.assertEqual(total_windows, ADAPT.EXPECTED_TRAIN_SAMPLES)
        canonical_manifest_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        canonical_manifest_sha = _sha(canonical_manifest_path)
        canonical_lineage = {
            "final_manifest_sha256": canonical_manifest_sha,
            "lineage_contract_sha256": lineage_contract_sha,
            "lineage_contract": lineage_contract,
        }
        canonical_lineage_path.write_text(
            json.dumps(canonical_lineage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        canonical_summary = {
            "status": "complete",
            "schema_name": "semtalk-show-canonical-motion",
            "schema_version": 1,
            "manifest_sha256": canonical_manifest_sha,
            "split_counts": dict(ADAPT.EXPECTED_SPLIT_COUNTS),
            "clip_count": ADAPT.EXPECTED_CANONICAL_CLIPS,
            "exact_once": True,
            "finite": True,
            "split_disjoint": True,
            "lineage_sha256": ADAPT._canonical_file_payload_sha256(
                canonical_lineage
            ),
            "lineage_contract_sha256": lineage_contract_sha,
            "source_receipt_sha256": ADAPT._canonical_file_payload_sha256(
                source_receipt
            ),
        }
        canonical_summary_path.write_text(
            json.dumps(canonical_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        canonical_receipt = {
            "manifest": str(canonical_manifest_path),
            "manifest_sha256": canonical_manifest_sha,
            "summary": str(canonical_summary_path),
            "summary_sha256": _sha(canonical_summary_path),
            "lineage": str(canonical_lineage_path),
            "lineage_sha256": _sha(canonical_lineage_path),
            "lineage_contract_sha256": lineage_contract_sha,
            "source_receipt": source_receipt,
        }

        lineage_path = root / "lineage.json"
        records = {}
        for stage, specification in ADAPT.OFFICIAL_PREREQUISITE_SPECS.items():
            records[stage] = {
                "path": f"/weights/all/{specification['filename']}",
                "filename": specification["filename"],
                "sha256": specification["sha256"],
                "formal_stage": stage,
                "prerequisite_source": ADAPT.OFFICIAL_BASE_SOURCE,
                "classification": ADAPT.OFFICIAL_BASE_CLASSIFICATION,
                "training_dataset": "BEAT2",
                "speaker_scope": "All-Speakers",
                "show_trained": False,
                "checkpoint_container_schema": ["model_state"],
                "strict_state_dict_load": True,
                "all_model_state_tensors_finite": True,
                "frozen_eval": True,
            }
        lineage: dict[str, object] = {
            "format": "semtalk_show_base_feature_lineage_v1",
            "status": "complete",
            "entries": ADAPT.EXPECTED_TRAIN_SAMPLES,
            "train_clips": ADAPT.EXPECTED_TRAIN_CLIPS,
            "protocol": {
                "scope": "SemTalk Base only",
                "split": "train",
                "speakers": ADAPT.SHOW_SPEAKERS,
                "window_length": 64,
                "stride": 20,
                "in_word": "int64_all_zero_unused_placeholder",
                "forbidden_components": [
                    "ASR",
                    "TextGrid",
                    "vocabulary",
                    "CLIP",
                    "emotion",
                    "semantic",
                    "SemGate",
                    "Sparse",
                ],
                "prerequisite_source": ADAPT.OFFICIAL_BASE_SOURCE,
            },
            "formal_checkpoints": records,
            "canonical_manifest_sha256": {
                str(canonical_manifest_path.resolve()): canonical_manifest_sha,
            },
            "canonical_receipt": canonical_receipt,
            "entry_aggregate_sha256": "a" * 64,
        }
        lineage_path.write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary: dict[str, object] = {
            "format": "semtalk_show_base_lmdb_summary_v1",
            "status": "complete",
            "scope": "SemTalk Base only",
            "entries": ADAPT.EXPECTED_TRAIN_SAMPLES,
            "train_clips": ADAPT.EXPECTED_TRAIN_CLIPS,
            "raw_frames": total_raw,
            "usable_frames": total_usable,
            "dropped_tail_frames": total_dropped,
            "lmdb": str(lmdb),
            "data_mdb_sha256": _sha(lmdb / "data.mdb"),
            "lock_mdb_sha256": _sha(lmdb / "lock.mdb"),
            "lineage_json": str(lineage_path),
            "lineage_json_sha256": _sha(lineage_path),
            "skipped_short_clip_ids": [],
            "entry_aggregate_sha256": "a" * 64,
            "per_clip": per_clip,
        }
        summary_path = root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            dataset_summary=str(summary_path),
            expected_dataset_summary_sha256=_sha(summary_path),
            lineage_manifest=str(lineage_path),
            expected_lineage_sha256=_sha(lineage_path),
            train_lmdb=str(lmdb),
        )
        return args, summary, lineage

    def test_official_all_speakers_dataset_lineage_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = self._dataset_fixture(Path(temporary))
            receipt = ADAPT.validate_dataset_receipts(args)
            self.assertFalse(receipt["vq_models_in_training_graph"])
            self.assertEqual(
                receipt["vq_targets"],
                "precomputed_frozen_lmdb_tensors",
            )
            self.assertEqual(
                set(receipt["formal_checkpoints"]),
                {"face", "hands", "upper", "lower", "global"},
            )
            self.assertTrue(
                receipt["canonical_dataset_evidence"]["split_disjoint"]
            )
            self.assertEqual(
                receipt["canonical_dataset_evidence"]["split_counts"],
                ADAPT.EXPECTED_SPLIT_COUNTS,
            )
            self.assertEqual(
                receipt["lmdb_inode_binding"]["files"]["data.mdb"][
                    "sha256"
                ],
                receipt["data_mdb_sha256"],
            )

    def test_same_descriptor_bytes_are_both_hashed_and_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            original = b'{"version":1}\n'
            replacement = b'{"version":2}\n'
            path.write_bytes(original)
            real_reader = ADAPT._read_regular_file_bytes

            def read_then_replace(
                value: Path, label: str
            ) -> tuple[Path, bytes, dict[str, int]]:
                result = real_reader(value, label)
                path.write_bytes(replacement)
                return result

            with mock.patch.object(
                ADAPT,
                "_read_regular_file_bytes",
                side_effect=read_then_replace,
            ):
                payload, _, _ = ADAPT._load_json_receipt(
                    path,
                    hashlib.sha256(original).hexdigest(),
                    "race fixture",
                )
            self.assertEqual(payload, {"version": 1})
            self.assertEqual(path.read_bytes(), replacement)

    def test_lmdb_leaf_symlink_is_rejected_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real.lmdb"
            target.mkdir()
            (target / "data.mdb").write_bytes(b"data")
            (target / "lock.mdb").write_bytes(b"lock")
            alias = root / "alias.lmdb"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT._verified_lmdb_receipt(alias)

    def test_base_clip_ledger_must_match_canonical_train_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, summary, _ = self._dataset_fixture(Path(temporary))
            summary["per_clip"][0]["clip_id"] = "test-00000"
            summary_path = Path(args.dataset_summary)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.expected_dataset_summary_sha256 = _sha(summary_path)
            with self.assertRaisesRegex(
                ADAPT.AdaptationContractError,
                "ledger",
            ):
                ADAPT.validate_dataset_receipts(args)

    def test_speaker2_or_nonofficial_vq_lineage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, summary, lineage = self._dataset_fixture(root)
            lineage["formal_checkpoints"]["face"]["path"] = (
                "/weights/speaker2/rvq_face_600.bin"
            )
            lineage_path = Path(args.lineage_manifest)
            lineage_path.write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary["lineage_json_sha256"] = _sha(lineage_path)
            summary_path = Path(args.dataset_summary)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.expected_lineage_sha256 = _sha(lineage_path)
            args.expected_dataset_summary_sha256 = _sha(summary_path)
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT.validate_dataset_receipts(args)

    def test_throughput_gate_binds_semantics_across_distinct_run_purposes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_frozen = _frozen_gate_fixture(
                receipt_sha256="a" * 64,
                topology_receipt_sha256="t" * 64,
                trajectory_mode=ADAPT.LEGACY_TRAJECTORY_MODE,
                run_purpose=ADAPT.RUN_PURPOSE_THROUGHPUT,
                target_epochs=[],
            )
            training_frozen = {
                **gate_frozen,
                "receipt_sha256": "b" * 64,
                "run_purpose": ADAPT.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(ADAPT.SHORT_QUALITY_EPOCHS),
            }
            report = _gate_report(
                frozen_sha256="a" * 64,
                frozen_compatibility_sha256=(
                    ADAPT._frozen_gate_compatibility_sha256(gate_frozen)
                ),
                topology_independent_input_sha256=(
                    ADAPT._topology_independent_gate_semantic_sha256(
                        gate_frozen
                    )
                ),
                topology_receipt_sha256="t" * 64,
                trajectory_mode=ADAPT.LEGACY_TRAJECTORY_MODE,
                trajectory_probe=None,
            )
            path = root / "gate.json"
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                throughput_gate_report=str(path),
                expected_throughput_gate_sha256=_sha(path),
                precision="bf16",
                learning_rate=3e-5,
                topology_mode=ADAPT.W8_GLOBAL512_MODE,
                expected_topology_gate_spec_sha256=_sha(
                    TOPOLOGY_GATE_SPEC
                ),
            )
            receipt = ADAPT.validate_throughput_gate(
                args,
                frozen_receipt=training_frozen,
            )
            self.assertEqual(receipt["samples_per_second"], 1024.0)
            changed_receipts = []
            changed = json.loads(json.dumps(training_frozen))
            changed["source"]["commit"] = "9" * 40
            changed_receipts.append(changed)
            changed = json.loads(json.dumps(training_frozen))
            changed["dataset"]["data_mdb_sha256"] = "9" * 64
            changed_receipts.append(changed)
            changed = json.loads(json.dumps(training_frozen))
            changed["dataset"]["selected_prerequisite_sha256"]["face"] = (
                "0" * 64
            )
            changed_receipts.append(changed)
            changed = json.loads(json.dumps(training_frozen))
            changed["topology"]["receipt_sha256"] = "9" * 64
            changed_receipts.append(changed)
            for changed in changed_receipts:
                with self.assertRaises(ADAPT.AdaptationContractError):
                    ADAPT.validate_throughput_gate(
                        args,
                        frozen_receipt=changed,
                    )

    def test_fresh_trajectory_probe_is_byte_exact(self) -> None:
        expected = _probe()
        self.assertEqual(
            ADAPT._require_matching_trajectory_probe(expected, dict(expected)),
            expected,
        )
        changed = json.loads(json.dumps(expected))
        changed["ranks"][3]["sample_order_sha256"] = "9" * 64
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "does not reproduce",
        ):
            ADAPT._require_matching_trajectory_probe(expected, changed)

    def test_probe_eta_must_be_derived_from_measured_median(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_frozen = _frozen_gate_fixture(
                receipt_sha256="a" * 64,
                topology_receipt_sha256="t" * 64,
                trajectory_mode=ADAPT.LEGACY_TRAJECTORY_MODE,
                run_purpose=ADAPT.RUN_PURPOSE_THROUGHPUT,
                target_epochs=[],
            )
            training_frozen = {
                **gate_frozen,
                "receipt_sha256": "b" * 64,
                "run_purpose": ADAPT.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(ADAPT.SHORT_QUALITY_EPOCHS),
            }
            report = _gate_report(
                frozen_sha256="a" * 64,
                frozen_compatibility_sha256=(
                    ADAPT._frozen_gate_compatibility_sha256(gate_frozen)
                ),
                topology_independent_input_sha256=(
                    ADAPT._topology_independent_gate_semantic_sha256(
                        gate_frozen
                    )
                ),
                topology_receipt_sha256="t" * 64,
                trajectory_mode=ADAPT.LEGACY_TRAJECTORY_MODE,
                trajectory_probe=None,
            )
            report["estimated_training_seconds"] = 1.0
            report["receipt_sha256"] = ADAPT.canonical_json_sha256(
                {
                    key: value
                    for key, value in report.items()
                    if key != "receipt_sha256"
                }
            )
            path = root / "forged-eta-gate.json"
            path.write_text(
                json.dumps(report, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(SELECTOR.TopologySelectionError):
                SELECTOR.validate_probe(
                    ADAPT.W8_GLOBAL512_MODE,
                    path,
                    _sha(path),
                    gate_spec_sha256=_sha(TOPOLOGY_GATE_SPEC),
                )
            args = argparse.Namespace(
                throughput_gate_report=str(path),
                expected_throughput_gate_sha256=_sha(path),
                precision="bf16",
                learning_rate=3e-5,
                topology_mode=ADAPT.W8_GLOBAL512_MODE,
                expected_topology_gate_spec_sha256=_sha(
                    TOPOLOGY_GATE_SPEC
                ),
            )
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT.validate_throughput_gate(
                    args,
                    frozen_receipt=training_frozen,
                )

    def test_rank_local_buffers_preserve_parameter_and_adam_consensus(self) -> None:
        probe = _probe()
        self.assertFalse(probe["all_rank_model_state_identical"])
        self.assertTrue(probe["all_rank_parameter_state_identical"])
        self.assertFalse(probe["all_rank_buffer_state_identical"])
        self.assertTrue(probe["all_rank_optimizer_state_identical"])

    def test_all_rank_parameter_and_adam_consensus_is_mandatory(self) -> None:
        probe = _probe()
        ranks = json.loads(json.dumps(probe["ranks"]))
        ranks[7]["parameter_state_semantic_sha256"] = "9" * 64
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT._assemble_trajectory_probe(
                ranks,
                optimizer_updates=ADAPT.TRAJECTORY_PROBE_UPDATES,
            )
        ranks = json.loads(json.dumps(probe["ranks"]))
        ranks[7]["optimizer_state_semantic_sha256"] = "9" * 64
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT._assemble_trajectory_probe(
                ranks,
                optimizer_updates=ADAPT.TRAJECTORY_PROBE_UPDATES,
            )

    def test_selector_rejects_false_buffer_consensus_claim(self) -> None:
        probe = _probe()
        changed = json.loads(json.dumps(probe))
        changed["all_rank_buffer_state_identical"] = True
        with self.assertRaises(SELECTOR.TopologySelectionError):
            SELECTOR._fresh_trajectory_probe(
                ADAPT.W8_GLOBAL512_MODE,
                changed,
                "changed buffer consensus",
            )
        changed = json.loads(json.dumps(probe))
        changed["ranks"][0]["rank"] = False
        with self.assertRaises(SELECTOR.TopologySelectionError):
            SELECTOR._fresh_trajectory_probe(
                ADAPT.W8_GLOBAL512_MODE,
                changed,
                "non-exact rank type",
            )

    def test_trainer_rejects_bool_world_size_and_rank_order(self) -> None:
        _activate(ADAPT.OFFICIAL_W1_REFERENCE_MODE)
        self.addCleanup(_activate, ADAPT.W8_GLOBAL512_MODE)
        probe = _probe()
        changed = json.loads(json.dumps(probe))
        changed["world_size"] = True
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT._validate_trajectory_probe(changed, label="bool world size")
        changed = json.loads(json.dumps(probe))
        changed["rank_order"][0:2] = [False, True]
        with self.assertRaises(ADAPT.AdaptationContractError):
            ADAPT._validate_trajectory_probe(changed, label="bool rank order")

    def test_fresh_throughput_gate_carries_lineage_bound_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _activate(ADAPT.W8_GLOBAL512_MODE)
            probe = _probe()
            gate_frozen = _frozen_gate_fixture(
                receipt_sha256="c" * 64,
                topology_receipt_sha256="u" * 64,
                trajectory_mode=ADAPT.FRESH_TRAJECTORY_MODE,
                run_purpose=ADAPT.RUN_PURPOSE_THROUGHPUT,
                target_epochs=[],
            )
            report = _gate_report(
                frozen_sha256="c" * 64,
                frozen_compatibility_sha256=(
                    ADAPT._frozen_gate_compatibility_sha256(gate_frozen)
                ),
                topology_independent_input_sha256=(
                    ADAPT._topology_independent_gate_semantic_sha256(
                        gate_frozen
                    )
                ),
                topology_receipt_sha256="u" * 64,
                trajectory_mode=ADAPT.FRESH_TRAJECTORY_MODE,
                trajectory_probe=probe,
            )
            path = Path(temporary) / "fresh-gate.json"
            path.write_text(
                json.dumps(report, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                throughput_gate_report=str(path),
                expected_throughput_gate_sha256=_sha(path),
                precision="bf16",
                learning_rate=3e-5,
                topology_mode=ADAPT.W8_GLOBAL512_MODE,
                expected_topology_gate_spec_sha256=_sha(
                    TOPOLOGY_GATE_SPEC
                ),
            )
            receipt = ADAPT.validate_throughput_gate(
                args,
                frozen_receipt=gate_frozen,
            )
            self.assertEqual(receipt["trajectory_probe"], probe)
            report["trajectory_probe"]["ranks"][2][
                "torch_cuda_rng_state_sha256"
            ] = "9" * 64
            report["receipt_sha256"] = ADAPT.canonical_json_sha256(
                {
                    key: value
                    for key, value in report.items()
                    if key != "receipt_sha256"
                }
            )
            changed_path = Path(temporary) / "changed-gate.json"
            changed_path.write_text(
                json.dumps(report, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.throughput_gate_report = str(changed_path)
            args.expected_throughput_gate_sha256 = _sha(changed_path)
            changed = ADAPT.validate_throughput_gate(
                args,
                frozen_receipt=gate_frozen,
            )
            with self.assertRaises(ADAPT.AdaptationContractError):
                ADAPT._require_matching_trajectory_probe(
                    receipt["trajectory_probe"],
                    changed["trajectory_probe"],
                )

    def test_source_receipt_supports_detached_head(self) -> None:
        scripted = [
            subprocess.CompletedProcess([], 0, ADAPT.EXPECTED_ORIGIN + "\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", ""),
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
            subprocess.CompletedProcess([], 0, "b" * 40 + "\n", ""),
        ]
        with mock.patch.object(
            ADAPT.subprocess,
            "run",
            side_effect=scripted,
        ):
            receipt = ADAPT.source_receipt()
        self.assertIsNone(receipt["branch"])
        self.assertTrue(receipt["clean"])


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is optional in the local CPU contract environment",
)
class OfficialBaseAdaptTorchContracts(unittest.TestCase):
    def test_trajectory_probe_partitions_rank_local_batchnorm_buffers(self) -> None:
        import copy
        import torch

        _activate(ADAPT.W8_GLOBAL512_MODE)

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(3, 4)
                self.normalization = torch.nn.BatchNorm1d(4)

        torch.manual_seed(7)
        reference = Tiny()
        reference_optimizer = torch.optim.Adam(
            reference.parameters(), lr=3e-5, betas=(0.5, 0.999)
        )
        reference_optimizer.zero_grad(set_to_none=True)
        synthetic_loss = sum(
            parameter.square().sum() for parameter in reference.parameters()
        )
        synthetic_loss.backward()
        reference_optimizer.step()
        rank_models = [copy.deepcopy(reference) for _ in range(ADAPT.WORLD_SIZE)]
        for rank, model in enumerate(rank_models):
            with torch.no_grad():
                model.normalization.running_mean.fill_(float(rank))
                model.normalization.running_var.fill_(float(rank + 1))
                model.normalization.num_batches_tracked.fill_(rank)
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=3e-5, betas=(0.5, 0.999))
            for model in rank_models
        ]
        for optimizer in optimizers:
            optimizer.load_state_dict(reference_optimizer.state_dict())
        before_model_hashes = [
            ADAPT._model_state_semantic_sha256(model.state_dict())
            for model in rank_models
        ]
        before_optimizer_hashes = [
            ADAPT._state_tree_semantic_sha256(optimizer.state_dict())
            for optimizer in optimizers
        ]
        fixed_rng = {
            "python_random_state_sha256": "1" * 64,
            "numpy_random_state_sha256": "2" * 64,
            "torch_cpu_rng_state_sha256": "3" * 64,
            "torch_cuda_rng_state_sha256": "4" * 64,
        }
        rank_probes = []
        with mock.patch.object(
            ADAPT,
            "_rng_state_hashes",
            return_value=fixed_rng,
        ):
            for rank, (model, optimizer) in enumerate(
                zip(rank_models, optimizers)
            ):
                rank_probes.append(
                    ADAPT._trajectory_rank_probe(
                        model,
                        optimizer,
                        ADAPT.TRAJECTORY_PROBE_UPDATES,
                        rank=rank,
                        device=torch.device("cpu"),
                        sample_order_sha256=f"{rank + 1:064x}",
                        sample_count=(
                            ADAPT.TRAJECTORY_PROBE_UPDATES
                            * ADAPT.LOCAL_BATCH_SIZE
                        ),
                    )
                )

        self.assertEqual(
            before_model_hashes,
            [
                ADAPT._model_state_semantic_sha256(model.state_dict())
                for model in rank_models
            ],
        )
        self.assertEqual(
            before_optimizer_hashes,
            [
                ADAPT._state_tree_semantic_sha256(optimizer.state_dict())
                for optimizer in optimizers
            ],
        )
        probe = ADAPT._assemble_trajectory_probe(
            rank_probes,
            optimizer_updates=ADAPT.TRAJECTORY_PROBE_UPDATES,
        )
        self.assertFalse(probe["all_rank_model_state_identical"])
        self.assertTrue(probe["all_rank_parameter_state_identical"])
        self.assertFalse(probe["all_rank_buffer_state_identical"])
        self.assertTrue(probe["all_rank_optimizer_state_identical"])
        self.assertEqual(rank_probes[0]["parameter_state_tensors"], 4)
        self.assertEqual(rank_probes[0]["buffer_state_tensors"], 3)
        self.assertEqual(
            len(
                {
                    rank_probe["parameter_state_semantic_sha256"]
                    for rank_probe in rank_probes
                }
            ),
            1,
        )
        self.assertEqual(
            len(
                {
                    rank_probe["buffer_state_semantic_sha256"]
                    for rank_probe in rank_probes
                }
            ),
            ADAPT.WORLD_SIZE,
        )

        changed_rank_probes = copy.deepcopy(rank_probes)
        changed_rank_probes[3]["model_state_semantic_sha256"] = "d" * 64
        changed_rank_probes[3]["buffer_state_semantic_sha256"] = "e" * 64
        changed_probe = ADAPT._assemble_trajectory_probe(
            changed_rank_probes,
            optimizer_updates=ADAPT.TRAJECTORY_PROBE_UPDATES,
        )
        with self.assertRaisesRegex(
            ADAPT.AdaptationContractError,
            "does not reproduce",
        ):
            ADAPT._require_matching_trajectory_probe(probe, changed_probe)

    def test_strict_load_and_mean_initializes_only_four_rows(self) -> None:
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spearker_encoder_face = torch.nn.Embedding(25, 768)
                self.spearker_encoder_body = torch.nn.Embedding(25, 768)
                self.other = torch.nn.Linear(3, 2)

        official = Tiny().state_dict()
        official = {
            key: torch.arange(
                value.numel(), dtype=value.dtype
            ).reshape_as(value)
            for key, value in official.items()
        }
        model = Tiny()
        receipt = ADAPT.strict_load_and_initialize_show_speakers(
            model,
            official,
            torch_module=torch,
        )
        state = model.state_dict()
        for key in ADAPT.SPEAKER_EMBEDDING_KEYS:
            mean = official[key].mean(dim=0)
            self.assertTrue(torch.equal(state[key][:4], mean.expand(4, -1)))
            self.assertTrue(torch.equal(state[key][4:], official[key][4:]))
        self.assertTrue(torch.equal(state["other.weight"], official["other.weight"]))
        self.assertTrue(torch.equal(state["other.bias"], official["other.bias"]))
        self.assertTrue(receipt["other_state_unchanged"])

    def test_audio_objective_calls_model_three_times_and_is_finite(self) -> None:
        import torch

        class Fake:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, *_: object, **__: object) -> dict[str, object]:
                self.calls += 1
                output = {}
                for stage in ("face", "upper", "hands", "lower"):
                    output[f"rec_{stage}"] = torch.zeros(1, 6, 1, 16, 256)
                    output[f"cls_{stage}"] = torch.zeros(1, 16, 256, 6)
                output["hubert_cons_loss"] = torch.tensor(0.25)
                output["beat_cons_loss"] = torch.tensor(0.5)
                return output

        batch = {
            "beat": torch.zeros(1, 64, 3),
            "in_word": torch.zeros(1, 64, dtype=torch.int64),
            "tar_id": torch.zeros(1, 64, 1, dtype=torch.int64),
            "latent_all": torch.zeros(1, 64, 337),
            "hubert": torch.zeros(1, 64, 1024),
        }
        for stage in ("face", "upper", "hands", "lower"):
            batch[f"zq_{stage}"] = torch.zeros(1, 6, 1, 16, 256)
            batch[f"tar_index_value_{stage}_top"] = torch.zeros(
                1, 16, 6, dtype=torch.int64
            )
        model = Fake()
        loss, metrics = ADAPT.audio_conditioned_objective(
            model, batch, epoch=0, torch_module=torch
        )
        self.assertEqual(model.calls, 3)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(ADAPT.LOSS_COMPONENTS) - set(metrics), set())


if __name__ == "__main__":
    unittest.main()
