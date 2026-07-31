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
    / "semtalk_base_topology_quality_gate_spec_20260731.json"
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
                "model_state_semantic_sha256": "2" * 64,
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
        "all_rank_model_state_identical": True,
        "all_rank_optimizer_state_identical": True,
        "ranks": ranks,
    }


class _FakeReplication:
    @staticmethod
    def load_gate(
        path: Path,
        expected_sha256: str,
        *,
        expected_scope: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _sha(path) != expected_sha256:
            raise ValueError("gate hash mismatch")
        if payload["scope"] != expected_scope:
            raise ValueError("gate scope mismatch")
        return (
            {
                **_artifact(path),
                "receipt_payload_sha256": payload[
                    "receipt_payload_sha256"
                ],
            },
            payload,
        )


class _FakeMetrics:
    @staticmethod
    def validate_released2_primary_screen_receipt(
        value: dict[str, object],
        **expected: object,
    ) -> dict[str, object]:
        payload = json.loads(
            Path(str(value["path"])).read_text(encoding="utf-8")
        )
        if payload["prediction_manifest"] != expected[
            "expected_prediction_manifest"
        ]:
            raise ValueError("screen prediction mismatch")
        if payload["distribution_receipt"] != expected[
            "expected_distribution_receipt"
        ]:
            raise ValueError("screen distribution mismatch")
        if payload["canonical_manifest"] != expected[
            "expected_canonical_manifest"
        ]:
            raise ValueError("screen canonical mismatch")
        if payload["real_feature_cache"] != expected[
            "expected_real_feature_cache"
        ]:
            raise ValueError("screen cache mismatch")
        return {
            "artifact": dict(value),
            "primary_metric": payload["primary_metric"],
        }

    @staticmethod
    def validate_released2_primary_screen_replay_receipt(
        value: dict[str, object],
        **expected: object,
    ) -> dict[str, object]:
        payload = json.loads(
            Path(str(value["path"])).read_text(encoding="utf-8")
        )
        screen = expected["expected_screen"]
        if expected["screen_validation"]["artifact"] != expected[
            "expected_screen_artifact"
        ]:
            raise ValueError("screen artifact mismatch")
        if payload["report_payload_sha256"] != screen[
            "receipt_payload_sha256"
        ]:
            raise ValueError("replay screen mismatch")
        if payload["prediction_manifest"] != expected[
            "expected_prediction_manifest"
        ]:
            raise ValueError("replay prediction mismatch")
        if payload["primary_metric"] != expected["screen_validation"][
            "primary_metric"
        ]:
            raise ValueError("replay metric mismatch")
        return {
            "artifact": dict(value),
            "primary_metric": payload["primary_metric"],
        }


class QualityFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.semantic_sha = "9" * 64
        self.topology_spec = TOPOLOGY_SPEC
        self.quality_spec = QUALITY_SPEC
        self.topology_sha = _sha(self.topology_spec)
        self.quality_sha = _sha(self.quality_spec)

        canonical_path = root / "canonical.jsonl"
        canonical_path.write_text('{"clip_id":"fixture"}\n', encoding="utf-8")
        self.canonical = {
            **_artifact(canonical_path),
            "rows": selector.EXPECTED_VAL_CLIPS,
            "selected_rows": selector.EXPECTED_VAL_CLIPS,
        }
        self.cache, _ = _write_receipt(
            root / "real-cache.json",
            {"format": "fixture-real-cache"},
        )
        self.checkpoints: dict[int, dict[str, object]] = {}
        self.predictions: dict[int, dict[str, object]] = {}
        self.distributions: dict[int, dict[str, object]] = {}
        self.screens: dict[int, dict[str, object]] = {}
        self.replays: dict[int, dict[str, object]] = {}
        self.short_status_by_mode: dict[str, dict[str, object]] = {}
        for position, epoch in enumerate(selector.QUALITY_EPOCHS):
            checkpoint = _write(
                root / f"e{epoch}.bin",
                f"checkpoint-e{epoch}".encode("utf-8"),
            )
            self.checkpoints[epoch] = checkpoint
            prediction_path = root / f"e{epoch}-prediction.jsonl"
            with prediction_path.open("w", encoding="utf-8") as handle:
                for index in range(selector.EXPECTED_VAL_CLIPS):
                    row = {
                        "global_index": index,
                        "split": "val",
                        "source_clip_id": f"source-{index:04d}",
                        "canonical_clip_id": f"clip-{index:04d}",
                        "frames": 32,
                        "epoch": epoch,
                        "candidate_checkpoint_sha256": checkpoint[
                            "sha256"
                        ],
                        "prediction": {
                            "path": str(
                                (root / f"e{epoch}-res-{index:04d}.npz")
                                .resolve()
                            ),
                            "sha256": f"{(index + position) % 16:x}" * 64,
                            "bytes": 1,
                        },
                        "ground_truth": {
                            "path": str(
                                (root / f"gt-{index:04d}.npz").resolve()
                            ),
                            "sha256": f"{(index + position + 1) % 16:x}"
                            * 64,
                            "bytes": 1,
                        },
                    }
                    handle.write(
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            prediction = _artifact(prediction_path)
            self.predictions[epoch] = prediction
            gate, _gate_payload = _write_receipt(
                root / f"e{epoch}-gate.json",
                {
                    "format": "fixture-gate",
                    "status": "pass",
                    "scope": "validation_candidate_family",
                    "split": "val",
                    "test_visible": False,
                    "model_bundle": {
                        "checkpoints": {"base": checkpoint}
                    },
                },
            )
            distribution, distribution_payload = _write_receipt(
                root / f"e{epoch}-distribution.json",
                {
                    "format": "fixture-distribution",
                    "prediction_manifest": prediction,
                    "validation_gate": gate,
                },
            )
            self.distributions[epoch] = distribution
            metric = 1.0 + epoch / 100.0
            screen, screen_payload = _write_receipt(
                root / f"e{epoch}-screen.json",
                {
                    "format": "fixture-screen",
                    "split": "val",
                    "test_visible": False,
                    "clip_count": selector.EXPECTED_VAL_CLIPS,
                    "canonical_manifest": self.canonical,
                    "prediction_manifest": prediction,
                    "distribution_receipt": distribution,
                    "real_feature_cache": self.cache,
                    "primary_metric": metric,
                },
            )
            self.screens[epoch] = screen
            replay, _ = _write_receipt(
                root / f"e{epoch}-replay.json",
                {
                    "format": "fixture-replay",
                    "split": "val",
                    "clip_count": selector.EXPECTED_VAL_CLIPS,
                    "report_payload_sha256": screen_payload[
                        "receipt_payload_sha256"
                    ],
                    "canonical_manifest": self.canonical,
                    "prediction_manifest": prediction,
                    "distribution_receipt_payload_sha256": (
                        distribution_payload["receipt_payload_sha256"]
                    ),
                    "real_feature_cache": self.cache,
                    "primary_metric": metric,
                },
            )
            self.replays[epoch] = replay

        self.frozen, self.frozen_payload = _write_contract_receipt(
            root / "frozen-inputs.json",
            {
                "format": (
                    "semtalk_show_base_official_adapt_frozen_inputs_v1"
                ),
                "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(selector.QUALITY_EPOCHS),
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
        specification = contract.TOPOLOGY_SPECS[mode]
        updates_per_epoch = int(specification["updates_per_epoch"])
        world_size = int(specification["world_size"])
        trajectory_probe = _trajectory_probe(mode)
        full_probe_body: dict[str, object] = {
            "format": contract.GATE_FORMAT,
            "status": "pass",
            "topology_mode": mode,
            "topology_classification": specification["classification"],
            "topology_gate_spec_sha256": self.topology_sha,
            "topology_independent_input_sha256": self.semantic_sha,
            "frozen_receipt_sha256": "6" * 64,
            "frozen_gate_compatibility_sha256": (
                self.frozen_compatibility_sha
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
            "samples_per_second": 1024.0,
            "seconds_per_update": 0.5,
            "median_seconds": 0.5,
            "p90_seconds": 0.6,
            "p99_seconds": 0.7,
            "estimated_training_seconds": 100.0,
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
                "probe_samples": (
                    contract.TRAJECTORY_PROBE_UPDATES
                    * int(specification["global_batch_size"])
                ),
                "probe_unique_samples": (
                    contract.TRAJECTORY_PROBE_UPDATES
                    * int(specification["global_batch_size"])
                ),
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
            "gate_frozen_receipt_sha256": "6" * 64,
            "frozen_gate_compatibility_sha256": (
                self.frozen_compatibility_sha
            ),
        }
        entries: list[dict[str, object]] = []
        ready: dict[int, dict[str, object]] = {}
        for position, epoch in enumerate(selector.QUALITY_EPOCHS):
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
                "frozen_receipt_sha256": self.frozen_payload[
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
                "target_epochs": list(selector.QUALITY_EPOCHS),
                "candidate_epochs": list(selector.QUALITY_EPOCHS),
                "frozen_receipt_sha256": self.frozen_payload[
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
                self.root / f"{mode}-e{epoch}-manifest.json"
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            ready_body = {
                "format": contract.SHORT_QUALITY_READY_RECEIPT_FORMAT,
                "status": "ready",
                "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(selector.QUALITY_EPOCHS),
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
                        (self.root / f"{mode}-live-manifest.json").resolve()
                    ),
                },
                "frozen_inputs": {
                    "path": self.frozen["path"],
                    "sha256": self.frozen["sha256"],
                    "receipt_payload_sha256": self.frozen_payload[
                        "receipt_sha256"
                    ],
                },
                "protocol": {
                    "format": contract.PROTOCOL_FORMAT,
                    "payload_sha256": contract.canonical_json_sha256(
                        self.frozen_payload["protocol"]
                    ),
                },
                "frozen_receipt_sha256": self.frozen_payload[
                    "receipt_sha256"
                ],
                "schedule_sha256": "a" * 64,
                "trajectory_anchor_sha256": "b" * 64,
                "trajectory_anchor_match": None,
                "published_unix": 1_700_000_000.0 + epoch,
            }
            artifact, _ = _write_receipt(
                self.root / f"{mode}-e{epoch}-ready.json",
                ready_body,
            )
            ready[epoch] = artifact
        final_manifest = {
            **manifest,
            "status": "complete",
            "completed_epochs": contract.SHORT_QUALITY_TOTAL_EPOCHS,
            "optimizer_updates": (
                contract.SHORT_QUALITY_TOTAL_EPOCHS * updates_per_epoch
            ),
        }
        final_manifest_path = self.root / f"{mode}-short-final-manifest.json"
        final_manifest_path.write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics_path = self.root / f"{mode}-short-metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            for epoch in range(1, contract.SHORT_QUALITY_TOTAL_EPOCHS + 1):
                row = {
                    "format": contract.SHORT_QUALITY_EPOCH_METRIC_FORMAT,
                    "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                    "target_epochs": list(selector.QUALITY_EPOCHS),
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
            "target_epochs": list(selector.QUALITY_EPOCHS),
            "completed_epochs": contract.SHORT_QUALITY_TOTAL_EPOCHS,
            "optimizer_updates": (
                contract.SHORT_QUALITY_TOTAL_EPOCHS * updates_per_epoch
            ),
            "updates_per_epoch": updates_per_epoch,
            "candidate_manifest_sha256": final_manifest_artifact["sha256"],
            "frozen_receipt_sha256": self.frozen_payload[
                "receipt_sha256"
            ],
            "throughput_gate": throughput,
            "world_size": specification["world_size"],
            "local_batch_size": specification["local_batch_size"],
            "global_batch_size": specification["global_batch_size"],
            "all_training_state_finite": True,
            "epoch_metrics_jsonl": metrics_artifact["path"],
            "epoch_metrics_sha256": metrics_artifact["sha256"],
            "epoch_metrics_records": contract.SHORT_QUALITY_TOTAL_EPOCHS,
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
                "records": contract.SHORT_QUALITY_TOTAL_EPOCHS,
            },
            "candidate_ready_receipts": [
                ready[epoch] for epoch in selector.QUALITY_EPOCHS
            ],
        }
        status_artifact, _ = _write_receipt(
            self.root / f"{mode}-short-status.json",
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
        prediction_override: dict[int, dict[str, object]] | None = None,
        screen_override: dict[int, dict[str, object]] | None = None,
        replay_override: dict[int, dict[str, object]] | None = None,
        status_override: dict[str, object] | None = None,
    ) -> list[str]:
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
        for epoch in selector.QUALITY_EPOCHS:
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
        argv.extend(
            [
                "--canonical-manifest",
                str(self.canonical["path"]),
                str(self.canonical["sha256"]),
                str(self.canonical["bytes"]),
                str(self.canonical["rows"]),
                str(self.canonical["selected_rows"]),
            ]
        )
        self._append_artifact(
            argv, "--real-feature-cache", None, self.cache
        )
        sources = (
            (
                "--prediction-manifest",
                prediction_override or self.predictions,
            ),
            ("--distribution-receipt", self.distributions),
            ("--primary-screen-receipt", screen_override or self.screens),
            ("--primary-replay-receipt", replay_override or self.replays),
        )
        for option, artifacts in sources:
            for epoch in selector.QUALITY_EPOCHS:
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
                selector,
                "_replication_module",
                return_value=_FakeReplication,
            ),
            mock.patch.object(
                selector,
                "_metrics_module",
                return_value=_FakeMetrics,
            ),
        )

    def test_positive_cli_is_callable_for_all_five_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            patches = self.patches(fixture)
            with patches[0], patches[1]:
                for index, mode in enumerate(contract.TOPOLOGY_SPECS):
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
                        4,
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
                        {"1": 1.01, "2": 1.02, "4": 1.04, "8": 1.08},
                    )
                    self.assertEqual(
                        [
                            row["provenance"]["format"]
                            for row in validated["candidates"]
                        ],
                        [selector.QUALITY_PROVENANCE_FORMAT] * 4,
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
            mode = next(iter(contract.TOPOLOGY_SPECS))
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
            with patches[0], patches[1]:
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

    def test_provisional_epoch_set_is_exact_e1_e2_e4_e8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
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
            with patches[0], patches[1]:
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
            with patches[0], patches[1]:
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

    def test_prediction_swap_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            swapped = dict(fixture.predictions)
            swapped[1] = fixture.predictions[2]
            patches = self.patches(fixture)
            with patches[0], patches[1]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "prediction row 0 authority changed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "prediction-swap.json",
                            prediction_override=swapped,
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
            with patches[0], patches[1]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "e1/e2/e4/e8 exactly in order",
                ):
                    producer.main(argv)

    def test_candidate_ready_reuse_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            reused = fixture.candidate_ready_receipts(mode)
            reused[2] = reused[1]
            patches = self.patches(fixture)
            with patches[0], patches[1]:
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
            with patches[0], patches[1]:
                self.assertEqual(producer.main(argv), 0)
                with self.assertRaisesRegex(
                    FileExistsError,
                    "refusing to overwrite immutable quality output",
                ):
                    producer.main(argv)

    def test_metric_receipt_swap_attack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualityFixture(Path(directory))
            mode = next(iter(contract.TOPOLOGY_SPECS))
            screens = dict(fixture.screens)
            replays = dict(fixture.replays)
            screens[1] = fixture.screens[2]
            replays[1] = fixture.replays[2]
            patches = self.patches(fixture)
            with patches[0], patches[1]:
                with self.assertRaisesRegex(
                    selector.TopologySelectionError,
                    "chain changed",
                ):
                    producer.main(
                        fixture.argv(
                            mode,
                            fixture.root / "metric-swap.json",
                            screen_override=screens,
                            replay_override=replays,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
