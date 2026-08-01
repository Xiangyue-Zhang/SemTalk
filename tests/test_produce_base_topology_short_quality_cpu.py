from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import produce_base_topology_short_quality as producer
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
    / "semtalk_base_topology_quality_gate_spec_v4_20260801.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return _artifact(path)


def _write_receipt(
    path: Path,
    body: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    payload = dict(body)
    payload["receipt_payload_sha256"] = contract.canonical_json_sha256(
        payload
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        {
            **_artifact(path),
            "receipt_payload_sha256": payload[
                "receipt_payload_sha256"
            ],
        },
        payload,
    )


def _write_contract_receipt(
    path: Path,
    body: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    payload = dict(body)
    payload["receipt_sha256"] = contract.canonical_json_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _artifact(path), payload


def _trajectory_probe(mode: str) -> dict[str, object]:
    specification = contract.TOPOLOGY_SPECS[mode]
    world_size = int(specification["world_size"])
    local_batch_size = int(specification["local_batch_size"])
    ranks: list[dict[str, object]] = []
    for rank in range(world_size):
        ranks.append(
            {
                "rank": rank,
                "optimizer_updates": contract.TRAJECTORY_PROBE_UPDATES,
                "model_state_tensors": 1790,
                "model_state_schema_sha256": "1" * 64,
                "model_state_semantic_sha256": f"{rank + 81:064x}",
                "parameter_state_tensors": 1784,
                "parameter_state_schema_sha256": "2" * 64,
                "parameter_state_semantic_sha256": "3" * 64,
                "buffer_state_tensors": 6,
                "buffer_state_schema_sha256": "4" * 64,
                "buffer_state_semantic_sha256": f"{rank + 97:064x}",
                "optimizer_state_semantic_sha256": "3" * 64,
                "python_random_state_sha256": f"{rank + 1:064x}",
                "numpy_random_state_sha256": f"{rank + 17:064x}",
                "torch_cpu_rng_state_sha256": f"{rank + 33:064x}",
                "torch_cuda_rng_state_sha256": f"{rank + 49:064x}",
                "sample_order_sha256": f"{rank + 65:064x}",
                "sample_count": (
                    contract.TRAJECTORY_PROBE_UPDATES * local_batch_size
                ),
            }
        )
    return {
        "format": contract.TRAJECTORY_PROBE_FORMAT,
        "optimizer_updates": contract.TRAJECTORY_PROBE_UPDATES,
        "world_size": world_size,
        "rank_order": list(range(world_size)),
        "all_rank_model_state_identical": world_size == 1,
        "all_rank_parameter_state_identical": True,
        "all_rank_buffer_state_identical": world_size == 1,
        "all_rank_optimizer_state_identical": True,
        "ranks": ranks,
    }


class _FakeFormalValidation:
    @staticmethod
    def _receipt(
        path: Path,
        expected_sha256: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _sha(path) != expected_sha256:
            raise ValueError("receipt hash mismatch")
        return (
            {
                "path": str(path.resolve()),
                "sha256": expected_sha256,
                "receipt_payload_sha256": payload[
                    "receipt_payload_sha256"
                ],
            },
            payload,
        )

    @staticmethod
    def validate_val_inputs(
        path: Path,
        expected_sha256: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        artifact, payload = _FakeFormalValidation._receipt(
            path, expected_sha256
        )
        if payload.get("split") != "val" or payload.get("test_visible") is not False:
            raise ValueError("val inputs are not val-only")
        return artifact, {
            "clip_count": selector.EXPECTED_VAL_CLIPS,
            "frame_count": 12_345,
            "window_count": 6_789,
            "uncovered_tail_frames": 321,
            "diffsheg_clip_manifest_sha256": "a" * 64,
        }

    @staticmethod
    def validate_pipeline(
        path: Path,
        expected_sha256: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        artifact, payload = _FakeFormalValidation._receipt(
            path, expected_sha256
        )
        if payload.get("split") != "val" or payload.get("test_visible") is not False:
            raise ValueError("pipeline is not val-only")
        return artifact, payload

    @staticmethod
    def validate_val_inference_lineage(
        path: Path,
        expected_sha256: str,
        *,
        epoch: int,
        expected_candidate: dict[str, object],
        val_inputs_artifact: dict[str, object],
        pipeline_artifact: dict[str, object],
        expected_coverage: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        artifact, payload = _FakeFormalValidation._receipt(
            path, expected_sha256
        )
        if (
            payload.get("split") != "val"
            or payload.get("test_visible") is not False
            or payload.get("epoch") != epoch
            or payload.get("candidate_checkpoint") != expected_candidate
            or payload.get("val_inputs_receipt") != val_inputs_artifact
            or payload.get("pipeline_receipt") != pipeline_artifact
        ):
            raise ValueError("lineage is not candidate-bound val-only")
        if expected_coverage.get("clip_count") != selector.EXPECTED_VAL_CLIPS:
            raise ValueError("coverage mismatch")
        return artifact, {
            "prediction_dir": payload["prediction_dir"],
            "ground_truth_dir": payload["ground_truth_dir"],
            "clip_manifest": {"path": payload["clip_manifest"]},
            "coverage": dict(expected_coverage),
        }

    @staticmethod
    def validate_diffsheg_report(
        report: dict[str, object],
        *,
        expected_coverage: dict[str, object],
        inference_lineage: dict[str, object],
        expected_pipeline: dict[str, object],
    ) -> tuple[dict[str, float], dict[str, int]]:
        protocol = report.get("protocol")
        inputs = report.get("inputs")
        metrics = report.get("metrics")
        if (
            report.get("status") != "ok"
            or not isinstance(protocol, dict)
            or protocol.get("selection_split") != "val"
            or protocol.get("test_visible") is not False
            or protocol.get("metric_scope") != "fgd_only"
            or not isinstance(inputs, dict)
            or inputs.get("prediction_dir") != inference_lineage["prediction_dir"]
            or inputs.get("ground_truth_dir")
            != inference_lineage["ground_truth_dir"]
            or inputs.get("clip_manifest")
            != inference_lineage["clip_manifest"]["path"]
            or not isinstance(metrics, dict)
            or set(metrics) != {"fgd"}
            or report.get("provenance") != {"assets": "pinned"}
        ):
            raise ValueError("DiffSHEG report mismatch")
        return {"fgd": float(metrics["fgd"])}, {
            "clip_count": int(expected_coverage["clip_count"]),
            "frame_count": int(expected_coverage["frame_count"]),
            "window_count": int(expected_coverage["window_count"]),
            "uncovered_tail_frames": int(
                expected_coverage["uncovered_tail_frames"]
            ),
        }


class QualityFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.semantic_sha = "9" * 64
        self.topology_spec = TOPOLOGY_SPEC
        self.quality_spec = QUALITY_SPEC
        self.topology_sha = _sha(self.topology_spec)
        self.quality_sha = _sha(self.quality_spec)

        self.val_inputs, _ = _write_receipt(
            root / "val-inputs.json",
            {
                "format": "fixture-val-inputs",
                "split": "val",
                "test_visible": False,
            },
        )
        self.pipeline, _ = _write_receipt(
            root / "val-pipeline.json",
            {
                "format": "fixture-val-pipeline",
                "split": "val",
                "test_visible": False,
            },
        )
        self.checkpoints: dict[int, dict[str, object]] = {}
        self.lineages: dict[int, dict[str, object]] = {}
        self.reports: dict[int, dict[str, object]] = {}
        self.short_status_by_mode: dict[str, dict[str, object]] = {}
        val_artifact = selector._formal_artifact_projection(self.val_inputs)
        pipeline_artifact = selector._formal_artifact_projection(self.pipeline)
        for epoch in selector.CANDIDATE_QUALITY_EPOCHS:
            checkpoint = _write(
                root / f"e{epoch}.bin",
                f"checkpoint-e{epoch}".encode("utf-8"),
            )
            self.checkpoints[epoch] = checkpoint
            prediction_dir = str((root / f"e{epoch}" / "predictions" / "val").resolve())
            ground_truth_dir = str(
                (root / f"e{epoch}" / "ground-truth" / "val").resolve()
            )
            clip_manifest = str(
                (root / f"e{epoch}" / "diffsheg_eval_clip_ids.txt").resolve()
            )
            lineage, _ = _write_receipt(
                root / f"e{epoch}-val-inference-lineage.json",
                {
                    "format": "fixture-val-inference-lineage",
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": epoch,
                    "candidate_checkpoint": checkpoint,
                    "val_inputs_receipt": val_artifact,
                    "pipeline_receipt": pipeline_artifact,
                    "prediction_dir": prediction_dir,
                    "ground_truth_dir": ground_truth_dir,
                    "clip_manifest": clip_manifest,
                },
            )
            self.lineages[epoch] = lineage
            report = {
                "status": "ok",
                "protocol": {
                    "selection_split": "val",
                    "test_visible": False,
                    "metric_scope": "fgd_only",
                },
                "inputs": {
                    "prediction_dir": prediction_dir,
                    "ground_truth_dir": ground_truth_dir,
                    "clip_manifest": clip_manifest,
                },
                "metrics": {"fgd": 1.0 + epoch / 100.0},
                "provenance": {"assets": "pinned"},
            }
            self.reports[epoch] = _write(
                root / f"e{epoch}-diffsheg-val-fgd.json",
                (json.dumps(report, sort_keys=True) + "\n").encode("utf-8"),
            )

        self.frozen, self.frozen_payload = _write_contract_receipt(
            root / "frozen-inputs.json",
            {
                "format": (
                    "semtalk_show_base_official_adapt_frozen_inputs_v1"
                ),
                "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(selector.CANDIDATE_QUALITY_EPOCHS),
                "source": {
                    "origin": contract.EXPECTED_ORIGIN,
                    "commit": "1" * 40,
                    "tree": "2" * 40,
                    "clean": True,
                    "entrypoint_sha256": "3" * 64,
                },
                "official_base": {"sha256": "4" * 64},
                "speaker_initialization": {"rows": [0, 1, 2, 3]},
                "dataset": {"data_mdb_sha256": "5" * 64},
                "protocol": {
                    "format": contract.PROTOCOL_FORMAT,
                    "forward_contract": {"fixture": True},
                    "loss": {"fixture": True},
                    "precision": "bf16",
                    "target_dataset": "SHOW",
                    "target_speaker_scope": "All",
                    "vq_models_in_training_graph": False,
                },
                "long_contract": {
                    "format": (
                        "semtalk_show_base_long_contract_receipts_v1"
                    ),
                    "schedule": {"sha256": "a" * 64},
                    "trajectory_anchor": {
                        "mode": contract.FRESH_TRAJECTORY_MODE,
                        "sha256": "b" * 64,
                    },
                },
                "topology": {"receipt_sha256": "7" * 64},
            },
        )
        self.semantic_sha = (
            contract._topology_independent_gate_semantic_sha256(
                self.frozen_payload
            )
        )
        self.frozen_compatibility_sha = (
            contract._frozen_gate_compatibility_sha256(
                self.frozen_payload
            )
        )

    @staticmethod
    def _append_artifact(
        argv: list[str],
        option: str,
        epoch: int | None,
        artifact: dict[str, object],
    ) -> None:
        argv.append(option)
        if epoch is not None:
            argv.append(str(epoch))
        argv.extend(
            [
                str(artifact["path"]),
                str(artifact["sha256"]),
                str(artifact["bytes"]),
            ]
        )
        if "receipt_payload_sha256" in artifact:
            argv.append(str(artifact["receipt_payload_sha256"]))

    def candidate_ready_receipts(
        self,
        mode: str,
    ) -> dict[int, dict[str, object]]:
        quality_epochs = selector.quality_epochs_for_mode(mode)
        quality_total_epochs = quality_epochs[-1]
        quality_role = selector.quality_role_for_mode(mode)
        reference_only = mode == contract.OFFICIAL_W1_REFERENCE_MODE
        run_root = self.root / f"{mode}-short-quality-v3"
        run_root.mkdir(exist_ok=True)
        for directory in (
            "provisional_candidates",
            "short_quality_candidate_manifest_snapshots",
            "short_quality_candidate_receipts",
        ):
            (run_root / directory).mkdir(exist_ok=True)
        frozen_body = {
            key: value
            for key, value in self.frozen_payload.items()
            if key != "receipt_sha256"
        }
        frozen_body["target_epochs"] = list(quality_epochs)
        frozen, frozen_payload = _write_contract_receipt(
            run_root / "frozen_inputs.json", frozen_body
        )
        semantic_sha = contract._topology_independent_gate_semantic_sha256(
            frozen_payload
        )
        frozen_compatibility_sha = (
            contract._frozen_gate_compatibility_sha256(frozen_payload)
        )
        specification = contract.TOPOLOGY_SPECS[mode]
        updates_per_epoch = int(specification["updates_per_epoch"])
        world_size = int(specification["world_size"])
        probe_epoch_updates = list(
            contract._probe_epoch_update_counts(updates_per_epoch)
        )
        probe_epoch_samples = [
            updates * int(specification["global_batch_size"])
            for updates in probe_epoch_updates
        ]
        probe_samples = sum(probe_epoch_samples)
        cross_epoch_duplicates = 0 if len(probe_epoch_updates) == 1 else 1
        trajectory_probe = _trajectory_probe(mode)
        full_probe_body: dict[str, object] = {
            "format": contract.GATE_FORMAT,
            "status": "pass",
            "topology_mode": mode,
            "topology_classification": specification["classification"],
            "topology_gate_spec_sha256": self.topology_sha,
            "topology_independent_input_sha256": semantic_sha,
            "frozen_receipt_sha256": frozen_payload["receipt_sha256"],
            "frozen_gate_compatibility_sha256": (
                frozen_compatibility_sha
            ),
            "topology_receipt_sha256": "7" * 64,
            "node_count": specification["node_count"],
            "local_world_size": specification["local_world_size"],
            "world_size": world_size,
            "local_batch_size": specification["local_batch_size"],
            "global_batch_size": specification["global_batch_size"],
            "updates_per_epoch": updates_per_epoch,
            "unique_samples_per_epoch": specification[
                "unique_samples_per_epoch"
            ],
            "warmup_updates": contract.THROUGHPUT_WARMUP_UPDATES,
            "timed_updates": contract.THROUGHPUT_TIMED_UPDATES,
            "optimizer_updates": contract.TRAJECTORY_PROBE_UPDATES,
            "trajectory_mode": contract.FRESH_TRAJECTORY_MODE,
            "trajectory_probe": trajectory_probe,
            "precision": specification["precision"],
            "learning_rate": specification["learning_rate"],
            "all_losses_finite": True,
            "all_gradients_finite": True,
            "oom": False,
            "samples_per_second": (
                float(specification["global_batch_size"]) / 0.5
            ),
            "seconds_per_update": 0.5,
            "median_seconds": 0.5,
            "p90_seconds": 0.6,
            "p99_seconds": 0.7,
            "estimated_training_seconds": (
                0.5 * updates_per_epoch * contract.TOTAL_EPOCHS
            ),
            "estimated_epochs": contract.TOTAL_EPOCHS,
            "last_metrics": {"total": 1.0},
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
                    "before": [{"name": "hubert", "sha256": "4" * 64}],
                    "after": [{"name": "hubert", "sha256": "4" * 64}],
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
                "probe_unique_samples": (
                    probe_samples - cross_epoch_duplicates
                ),
                "probe_sampler_epochs": list(
                    range(len(probe_epoch_updates))
                ),
                "probe_epoch_updates": probe_epoch_updates,
                "probe_epoch_samples": probe_epoch_samples,
                "probe_epoch_unique_samples": probe_epoch_samples,
                "probe_cross_epoch_duplicates": cross_epoch_duplicates,
                "full_epoch_samples": specification[
                    "unique_samples_per_epoch"
                ],
                "full_epoch_unique_samples": specification[
                    "unique_samples_per_epoch"
                ],
                "dataset_samples": contract.EXPECTED_TRAIN_SAMPLES,
                "dropped_tail_samples": (
                    contract.EXPECTED_TRAIN_SAMPLES
                    - int(specification["unique_samples_per_epoch"])
                ),
                "full_epoch_sorted_indices_sha256": "8" * 64,
            },
        }
        full_probe, full_probe_payload = _write_contract_receipt(
            self.root / f"{mode}-throughput.json",
            full_probe_body,
        )

        throughput: dict[str, object] = {
            "path": full_probe["path"],
            "sha256": full_probe["sha256"],
            "topology_mode": mode,
            "samples_per_second": full_probe_payload[
                "samples_per_second"
            ],
            "seconds_per_update": full_probe_payload[
                "seconds_per_update"
            ],
            "median_seconds": full_probe_payload["median_seconds"],
            "p90_seconds": full_probe_payload["p90_seconds"],
            "p99_seconds": full_probe_payload["p99_seconds"],
            "estimated_training_seconds": full_probe_payload[
                "estimated_training_seconds"
            ],
            "trajectory_mode": contract.FRESH_TRAJECTORY_MODE,
            "trajectory_probe": trajectory_probe,
            "gate_frozen_receipt_sha256": frozen_payload[
                "receipt_sha256"
            ],
            "frozen_gate_compatibility_sha256": (
                frozen_compatibility_sha
            ),
        }
        val_artifact = selector._formal_artifact_projection(self.val_inputs)
        pipeline_artifact = selector._formal_artifact_projection(self.pipeline)
        self.checkpoints = {}
        self.lineages = {}
        self.reports = {}
        for epoch in quality_epochs:
            checkpoint = _write(
                run_root
                / "provisional_candidates"
                / f"base_official_adapt_short_quality_epoch_{epoch:02d}.bin",
                f"checkpoint-{mode}-e{epoch}".encode("utf-8"),
            )
            self.checkpoints[epoch] = checkpoint
            prediction_dir = str(
                (self.root / mode / f"e{epoch}" / "predictions" / "val").resolve()
            )
            ground_truth_dir = str(
                (self.root / mode / f"e{epoch}" / "ground-truth" / "val").resolve()
            )
            clip_manifest = str(
                (self.root / mode / f"e{epoch}" / "diffsheg_eval_clip_ids.txt").resolve()
            )
            lineage, _ = _write_receipt(
                self.root / f"{mode}-e{epoch}-val-inference-lineage.json",
                {
                    "format": "fixture-val-inference-lineage",
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": epoch,
                    "candidate_checkpoint": checkpoint,
                    "val_inputs_receipt": val_artifact,
                    "pipeline_receipt": pipeline_artifact,
                    "prediction_dir": prediction_dir,
                    "ground_truth_dir": ground_truth_dir,
                    "clip_manifest": clip_manifest,
                },
            )
            self.lineages[epoch] = lineage
            report = {
                "status": "ok",
                "protocol": {
                    "selection_split": "val",
                    "test_visible": False,
                    "metric_scope": "fgd_only",
                },
                "inputs": {
                    "prediction_dir": prediction_dir,
                    "ground_truth_dir": ground_truth_dir,
                    "clip_manifest": clip_manifest,
                },
                "metrics": {"fgd": 1.0 + epoch / 100.0},
                "provenance": {"assets": "pinned"},
            }
            self.reports[epoch] = _write(
                self.root / f"{mode}-e{epoch}-diffsheg-val-fgd.json",
                (json.dumps(report, sort_keys=True) + "\n").encode("utf-8"),
            )
        entries: list[dict[str, object]] = []
        ready: dict[int, dict[str, object]] = {}
        for position, epoch in enumerate(quality_epochs):
            checkpoint = self.checkpoints[epoch]
            relative = (
                "provisional_candidates/"
                f"base_official_adapt_short_quality_epoch_{epoch:02d}.bin"
            )
            model_semantic = f"{position + 1:x}" * 64
            model_schema = f"{position + 5:x}" * 64
            entry = {
                "epoch": epoch,
                "optimizer_updates": epoch * updates_per_epoch,
                "checkpoint": relative,
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_bytes": checkpoint["bytes"],
                "checkpoint_container_schema": ["audit", "model_state"],
                "model_state_tensors": 1790,
                "model_state_schema_sha256": model_schema,
                "model_state_semantic_sha256": model_semantic,
                "all_model_state_tensors_finite": True,
                "frozen_receipt_sha256": frozen_payload[
                    "receipt_sha256"
                ],
                "trajectory_anchor_match": None,
                "trajectory_probe_verified": True,
                "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
            }
            entries.append(entry)
            manifest = {
                "format": contract.SHORT_QUALITY_MANIFEST_FORMAT,
                "status": "running",
                "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(quality_epochs),
                "candidate_epochs": list(quality_epochs),
                "quality_protocol_version": selector.QUALITY_PROTOCOL_VERSION,
                "artifact_root_namespace": (
                    selector.QUALITY_ARTIFACT_ROOT_NAMESPACE
                ),
                "artifact_root": str(run_root.resolve()),
                "quality_role": quality_role,
                "reference_only": reference_only,
                "late_w1_status": "not_measured",
                "w1_tail_equivalence_claimed": False,
                "frozen_receipt_sha256": frozen_payload[
                    "receipt_sha256"
                ],
                "schedule_sha256": "a" * 64,
                "trajectory_anchor_sha256": "b" * 64,
                "throughput_gate": throughput,
                "trajectory_mode": contract.FRESH_TRAJECTORY_MODE,
                "trajectory_probe_verified": True,
                "trajectory_probe": trajectory_probe,
                "entries": list(entries),
                "entries_sha256": contract.canonical_json_sha256(entries),
            }
            manifest_path = (
                run_root
                / "short_quality_candidate_manifest_snapshots"
                / f"epoch-{epoch:04d}.json"
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            ready_body = {
                "format": contract.SHORT_QUALITY_READY_RECEIPT_FORMAT,
                "status": "ready",
                "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(quality_epochs),
                "quality_protocol_version": selector.QUALITY_PROTOCOL_VERSION,
                "artifact_root_namespace": (
                    selector.QUALITY_ARTIFACT_ROOT_NAMESPACE
                ),
                "artifact_root": str(run_root.resolve()),
                "quality_role": quality_role,
                "reference_only": reference_only,
                "late_w1_status": "not_measured",
                "w1_tail_equivalence_claimed": False,
                "selection_eligible": False,
                "test_visible": False,
                "epoch": epoch,
                "optimizer_updates": epoch * updates_per_epoch,
                "candidate_checkpoint": {
                    **checkpoint,
                    "relative_path": relative,
                    "model_state_tensors": 1790,
                    "model_state_schema_sha256": model_schema,
                    "model_state_semantic_sha256": model_semantic,
                },
                "candidate_manifest": {
                    "path": str(manifest_path.resolve()),
                    "sha256_at_ready": _sha(manifest_path),
                    "entries_sha256_at_ready": manifest[
                        "entries_sha256"
                    ],
                    "immutable_snapshot": True,
                    "live_path": str(
                        (
                            run_root
                            / "short_quality_candidate_manifest.json"
                        ).resolve()
                    ),
                },
                "frozen_inputs": {
                    "path": frozen["path"],
                    "sha256": frozen["sha256"],
                    "receipt_payload_sha256": frozen_payload[
                        "receipt_sha256"
                    ],
                },
                "protocol": {
                    "format": contract.PROTOCOL_FORMAT,
                    "payload_sha256": contract.canonical_json_sha256(
                        frozen_payload["protocol"]
                    ),
                },
                "frozen_receipt_sha256": frozen_payload[
                    "receipt_sha256"
                ],
                "schedule_sha256": "a" * 64,
                "trajectory_anchor_sha256": "b" * 64,
                "trajectory_anchor_match": None,
                "published_unix": 1_700_000_000.0 + epoch,
            }
            artifact, _ = _write_receipt(
                run_root
                / "short_quality_candidate_receipts"
                / f"epoch-{epoch:04d}.json",
                ready_body,
            )
            ready[epoch] = artifact
        final_manifest = {
            **manifest,
            "status": "complete",
            "completed_epochs": quality_total_epochs,
            "optimizer_updates": (
                quality_total_epochs * updates_per_epoch
            ),
        }
        final_manifest_path = run_root / "short_quality_candidate_manifest.json"
        final_manifest_path.write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics_path = run_root / "epoch_metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            for epoch in range(1, quality_total_epochs + 1):
                row = {
                    "format": contract.SHORT_QUALITY_EPOCH_METRIC_FORMAT,
                    "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                    "target_epochs": list(quality_epochs),
                    "quality_protocol_version": (
                        selector.QUALITY_PROTOCOL_VERSION
                    ),
                    "artifact_root_namespace": (
                        selector.QUALITY_ARTIFACT_ROOT_NAMESPACE
                    ),
                    "artifact_root": str(run_root.resolve()),
                    "quality_role": quality_role,
                    "reference_only": reference_only,
                    "late_w1_status": "not_measured",
                    "w1_tail_equivalence_claimed": False,
                    "epoch": epoch,
                    "optimizer_updates": epoch * updates_per_epoch,
                    "updates_per_epoch": updates_per_epoch,
                    "learning_rate": specification["learning_rate"],
                    "metrics": {
                        key: 1.0
                        for key in (
                            *contract.LOSS_COMPONENTS,
                            "total",
                            "gradient_norm_preclip",
                        )
                    },
                    "all_finite": True,
                    "completed_unix": 1_700_000_100.0 + epoch,
                }
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        final_manifest_artifact = _artifact(final_manifest_path)
        metrics_artifact = _artifact(metrics_path)
        status_body = {
            "format": contract.SHORT_QUALITY_STATUS_FORMAT,
            "status": "complete",
            "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
            "target_epochs": list(quality_epochs),
            "quality_protocol_version": selector.QUALITY_PROTOCOL_VERSION,
            "artifact_root_namespace": (
                selector.QUALITY_ARTIFACT_ROOT_NAMESPACE
            ),
            "artifact_root": str(run_root.resolve()),
            "quality_role": quality_role,
            "reference_only": reference_only,
            "late_w1_status": "not_measured",
            "w1_tail_equivalence_claimed": False,
            "completed_epochs": quality_total_epochs,
            "optimizer_updates": (
                quality_total_epochs * updates_per_epoch
            ),
            "updates_per_epoch": updates_per_epoch,
            "candidate_manifest_sha256": final_manifest_artifact["sha256"],
            "frozen_receipt_sha256": frozen_payload[
                "receipt_sha256"
            ],
            "throughput_gate": throughput,
            "world_size": specification["world_size"],
            "local_batch_size": specification["local_batch_size"],
            "global_batch_size": specification["global_batch_size"],
            "all_training_state_finite": True,
            "epoch_metrics_jsonl": metrics_artifact["path"],
            "epoch_metrics_sha256": metrics_artifact["sha256"],
            "epoch_metrics_records": quality_total_epochs,
            "schedule_sha256": "a" * 64,
            "trajectory_anchor_sha256": "b" * 64,
            "trajectory_mode": contract.FRESH_TRAJECTORY_MODE,
            "trajectory_probe_verified": True,
            "trajectory_probe": trajectory_probe,
            "started_unix": 1_700_000_000.0,
            "completed_unix": 1_700_000_200.0,
            "candidate_manifest": final_manifest_artifact,
            "epoch_metrics": {
                **metrics_artifact,
                "records": quality_total_epochs,
            },
            "candidate_ready_receipts": [
                ready[epoch] for epoch in quality_epochs
            ],
        }
        status_artifact, _ = _write_receipt(
            run_root / "short_quality_status.json",
            status_body,
        )
        self.short_status_by_mode[mode] = status_artifact
        return ready

    def argv(
        self,
        mode: str,
        output: Path,
        *,
        ready_override: dict[int, dict[str, object]] | None = None,
        lineage_override: dict[int, dict[str, object]] | None = None,
        report_override: dict[int, dict[str, object]] | None = None,
        val_inputs_override: dict[str, object] | None = None,
        pipeline_override: dict[str, object] | None = None,
        status_override: dict[str, object] | None = None,
    ) -> list[str]:
        quality_epochs = selector.quality_epochs_for_mode(mode)
        ready = ready_override or self.candidate_ready_receipts(mode)
        short_output = output.with_name(f"{output.stem}-short.json")
        argv = [
            "--mode",
            mode,
            "--topology-gate-spec",
            str(self.topology_spec),
            "--expected-topology-gate-spec-sha256",
            self.topology_sha,
            "--quality-gate-spec",
            str(self.quality_spec),
            "--expected-quality-gate-spec-sha256",
            self.quality_sha,
        ]
        for epoch in quality_epochs:
            self._append_artifact(
                argv,
                "--candidate-ready-receipt",
                epoch,
                ready[epoch],
            )
        self._append_artifact(
            argv,
            "--short-quality-status",
            None,
            status_override or self.short_status_by_mode[mode],
        )
        argv.extend(
            ["--short-trajectory-output", str(short_output.resolve())]
        )
        self._append_artifact(
            argv,
            "--val-inputs-receipt",
            None,
            val_inputs_override or self.val_inputs,
        )
        self._append_artifact(
            argv,
            "--pipeline-receipt",
            None,
            pipeline_override or self.pipeline,
        )
        sources = (
            (
                "--inference-lineage",
                lineage_override or self.lineages,
            ),
            ("--diffsheg-report", report_override or self.reports),
        )
        for option, artifacts in sources:
            for epoch in quality_epochs:
                self._append_artifact(
                    argv,
                    option,
                    epoch,
                    artifacts[epoch],
                )
        argv.extend(["--output", str(output.resolve())])
        return argv


class ProduceBaseTopologyShortQualityTests(unittest.TestCase):
    def patches(self, fixture: QualityFixture):
        return (
            mock.patch.object(
                selector.formal_validation,
                "validate_val_inputs",
                side_effect=_FakeFormalValidation.validate_val_inputs,
            ),
            mock.patch.object(
                selector.formal_validation,
                "validate_pipeline",
                side_effect=_FakeFormalValidation.validate_pipeline,
            ),
            mock.patch.object(
                selector.formal_validation,
                "validate_val_inference_lineage",
                side_effect=(
                    _FakeFormalValidation.validate_val_inference_lineage
                ),
            ),
            mock.patch.object(
                selector.formal_validation,
                "validate_diffsheg_report",
                side_effect=_FakeFormalValidation.validate_diffsheg_report,
            ),
        )

    def test_positive_cli_is_callable_for_all_nine_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                for index, mode in enumerate(contract.TOPOLOGY_SPECS):
                    quality_epochs = selector.quality_epochs_for_mode(mode)
                    output = fixture.root / f"quality-{index}.json"
                    self.assertEqual(
                        producer.main(fixture.argv(mode, output)),
                        0,
                    )
                    short_output = output.with_name(
                        f"{output.stem}-short.json"
                    )
                    self.assertTrue(short_output.is_file())
                    short_payload = json.loads(
                        short_output.read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        len(short_payload["candidate_ready_receipts"]),
                        len(quality_epochs),
                    )
                    validated = selector.validate_quality_report(
                        mode,
                        output,
                        _sha(output),
                        quality_gate_spec_sha256=fixture.quality_sha,
                        topology_gate_spec_sha256=fixture.topology_sha,
                    )
                    self.assertEqual(
                        validated["candidate_fgd"],
                        {
                            str(epoch): 1.0 + epoch / 100.0
                            for epoch in quality_epochs
                        },
                    )
                    self.assertEqual(
                        [
                            row["provenance"]["format"]
                            for row in validated["candidates"]
                        ],
                        [selector.QUALITY_PROVENANCE_FORMAT]
                        * len(quality_epochs),
                    )
                    report_text = output.read_text(encoding="utf-8")
                    self.assertNotIn("released2", report_text.lower())
                    report_payload = json.loads(report_text)
                    self.assertEqual(
                        set(report_payload["candidates"][0]),
                        {
                            "epoch",
                            "candidate_checkpoint",
                            "inference_lineage",
                            "diffsheg_report",
                            "provenance",
                        },
                    )

    def test_w1_is_exact_four_point_reference_and_copied_tail_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = contract.OFFICIAL_W1_REFERENCE_MODE
            output = fixture.root / "w1-quality.json"
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                self.assertEqual(producer.main(fixture.argv(mode, output)), 0)
                validated = selector.validate_quality_report(
                    mode,
                    output,
                    _sha(output),
                    quality_gate_spec_sha256=fixture.quality_sha,
                    topology_gate_spec_sha256=fixture.topology_sha,
                )
                self.assertEqual(
                    list(validated["candidate_fgd"]),
                    ["1", "2", "4", "8"],
                )
                self.assertEqual(validated["quality_role"], "w1_reference_only")
                self.assertIs(validated["reference_only"], True)
                self.assertEqual(validated["late_w1_status"], "not_measured")
                self.assertIs(
                    validated["w1_tail_equivalence_claimed"], False
                )

                copied = json.loads(output.read_text(encoding="utf-8"))
                copied["trajectory_epochs"] = [1, 2, 4, 8, 16, 32]
                epoch_eight = copied["candidates"][-1]
                for epoch in (16, 32):
                    copied_row = json.loads(json.dumps(epoch_eight))
                    copied_row["epoch"] = epoch
                    copied["candidates"].append(copied_row)
                copied["receipt_sha256"] = contract.canonical_json_sha256(
                    {
                        key: value
                        for key, value in copied.items()
                        if key != "receipt_sha256"
                    }
                )
                copied_path = fixture.root / "w1-copied-e8-tail.json"
                copied_path.write_text(
                    json.dumps(copied, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "forged or stale",
                ):
                    selector.validate_quality_report(
                        mode,
                        copied_path,
                        _sha(copied_path),
                        quality_gate_spec_sha256=fixture.quality_sha,
                        topology_gate_spec_sha256=fixture.topology_sha,
                    )

    def test_old_v2_report_and_role_or_root_swaps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = contract.OFFICIAL_W1_REFERENCE_MODE
            output = fixture.root / "w1-quality.json"
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                self.assertEqual(producer.main(fixture.argv(mode, output)), 0)
                original = json.loads(output.read_text(encoding="utf-8"))
                attacks = {
                    "old-v2": {
                        "format": "semtalk_show_base_topology_quality_report_v2",
                        "quality_protocol_version": 2,
                    },
                    "role-swap": {
                        "quality_role": "candidate_quality",
                        "reference_only": False,
                    },
                    "root-swap": {
                        "artifact_root": str(
                            (fixture.root / "foreign-short-quality-v3").resolve()
                        ),
                    },
                }
                for label, changes in attacks.items():
                    with self.subTest(attack=label):
                        changed = json.loads(json.dumps(original))
                        changed.update(changes)
                        changed["receipt_sha256"] = (
                            contract.canonical_json_sha256(
                                {
                                    key: value
                                    for key, value in changed.items()
                                    if key != "receipt_sha256"
                                }
                            )
                        )
                        path = fixture.root / f"{label}.json"
                        path.write_text(
                            json.dumps(changed, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        with self.assertRaises(selector.TopologySelectionError):
                            selector.validate_quality_report(
                                mode,
                                path,
                                _sha(path),
                                quality_gate_spec_sha256=fixture.quality_sha,
                                topology_gate_spec_sha256=fixture.topology_sha,
                            )

    def test_provisional_bundle_cannot_masquerade_as_final_training(self) -> None:
        from scripts.show_base import base_long_val_contract as final_contract

        self.assertEqual(
            final_contract.CANDIDATE_MANIFEST_FORMAT,
            contract.MANIFEST_FORMAT,
        )
        self.assertNotEqual(
            contract.SHORT_QUALITY_MANIFEST_FORMAT,
            final_contract.CANDIDATE_MANIFEST_FORMAT,
        )
        self.assertNotEqual(
            contract.SHORT_QUALITY_READY_RECEIPT_FORMAT,
            contract.READY_RECEIPT_FORMAT,
        )
        self.assertNotEqual(
            contract.SHORT_QUALITY_CHECKPOINT_FORMAT,
            contract.CHECKPOINT_FORMAT,
        )

    def test_producer_rejects_old_e400_training_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = list(contract.TOPOLOGY_SPECS)[1]
            fixture.candidate_ready_receipts(mode)
            original = json.loads(
                Path(
                    str(fixture.short_status_by_mode[mode]["path"])
                ).read_text(encoding="utf-8")
            )
            original.pop("receipt_payload_sha256")
            original["format"] = contract.STATUS_FORMAT
            original["run_purpose"] = contract.RUN_PURPOSE_FORMAL_TRAINING
            original["target_epochs"] = list(contract.CANDIDATE_EPOCHS)
            original["completed_epochs"] = contract.TOTAL_EPOCHS
            old_status, _ = _write_receipt(
                fixture.root / "old-e400-status.json",
                original,
            )
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "provisional short-quality status changed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "old-e400-quality.json",
                            status_override=old_status,
                        )
                    )

    def test_provisional_epoch_set_is_exact_e1_e2_e4_e8_e16_e32(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = list(contract.TOPOLOGY_SPECS)[1]
            fixture.candidate_ready_receipts(mode)
            original = json.loads(
                Path(
                    str(fixture.short_status_by_mode[mode]["path"])
                ).read_text(encoding="utf-8")
            )
            original.pop("receipt_payload_sha256")
            original["target_epochs"] = [1, 2, 4, 8, 16]
            changed_status, _ = _write_receipt(
                fixture.root / "wrong-epochs-status.json",
                original,
            )
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "provisional short-quality status changed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "wrong-epochs-quality.json",
                            status_override=changed_status,
                        )
                    )

    def test_checkpoint_swap_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            swapped = fixture.candidate_ready_receipts(mode)
            swapped[1], swapped[2] = swapped[2], swapped[1]
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "candidate-ready protocol changed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "checkpoint-swap.json",
                            ready_override=swapped,
                        )
                    )

    def test_inference_lineage_swap_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            swapped = dict(fixture.lineages)
            swapped[1] = fixture.lineages[2]
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "inference lineage validation failed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "lineage-swap.json",
                            lineage_override=swapped,
                        )
                    )

    def test_candidate_ready_omission_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            argv = fixture.argv(mode, fixture.root / "omission.json")
            option = "--candidate-ready-receipt"
            starts = [index for index, value in enumerate(argv) if value == option]
            del argv[starts[-1] : starts[-1] + 6]
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "must name .* exactly in order",
                ):
                    producer.main(argv)

    def test_candidate_ready_reuse_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            reused = fixture.candidate_ready_receipts(mode)
            reused[2] = reused[1]
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "receipt was reused|protocol changed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "reuse.json",
                            ready_override=reused,
                        )
                    )

    def test_create_new_outputs_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            output = fixture.root / "immutable-output.json"
            argv = fixture.argv(mode, output)
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                self.assertEqual(producer.main(argv), 0)
                with self.assertRaisesRegex(
                    FileExistsError,
                    "refusing to overwrite immutable quality output",
                ):
                    producer.main(argv)

    def test_diffsheg_report_swap_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            reports = dict(fixture.reports)
            reports[1] = fixture.reports[2]
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "DiffSHEG report validation failed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "diffsheg-report-swap.json",
                            report_override=reports,
                        )
                    )

    def test_common_val_input_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            alternate, _ = _write_receipt(
                fixture.root / "alternate-val-inputs.json",
                {
                    "format": "fixture-val-inputs",
                    "split": "val",
                    "test_visible": False,
                    "changed": True,
                },
            )
            patches = self.patches(fixture)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "inference lineage validation failed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "val-input-mismatch.json",
                            val_inputs_override=alternate,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
