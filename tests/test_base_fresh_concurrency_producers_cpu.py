from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.show_base import base_fresh_probe_producer as PRODUCER
from scripts.show_base import base_fresh_val_orchestrator as ORCHESTRATOR
from scripts.show_base import published_test_winner_claim as AUTHORITY


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _write_json(path: Path, value: object) -> dict[str, object]:
    return _write_bytes(path, _canonical_bytes(value))


def _write_jsonl(
    path: Path, rows: list[dict[str, object]]
) -> dict[str, object]:
    return _write_bytes(
        path, b"".join(_canonical_bytes(row) for row in rows)
    )


def _write_npy(path: Path, value: np.ndarray) -> dict[str, object]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return _write_bytes(path, buffer.getvalue())


def _write_receipt(
    path: Path, value: dict[str, object]
) -> dict[str, object]:
    receipt = copy.deepcopy(value)
    receipt.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = AUTHORITY.canonical_json_sha256(
        receipt
    )
    artifact = _write_json(path, receipt)
    artifact["receipt_payload_sha256"] = receipt[
        "receipt_payload_sha256"
    ]
    return artifact


def _argv_sha(argv: list[str]) -> str:
    return hashlib.sha256(
        b"\0".join(os.fsencode(item) for item in argv) + b"\0"
    ).hexdigest()


class ProducerFixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        (self.root / "runs").mkdir()
        self.spec_registry: dict[
            str,
            tuple[
                dict[str, object],
                dict[str, object],
                dict[str, object],
            ],
        ] = {}
        self.behavior: dict[str, object] = {}
        self.elapsed_seconds = 120
        self.fake_pid = 41000
        self.source = {
            "origin": AUTHORITY.EXPECTED_ORIGIN,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        self.pipeline = _write_receipt(
            self.root / "pipeline.json",
            {
                "format": "semtalk_show_base_probe_pipeline_fixture_v1",
                "status": "frozen",
                "split": "val",
                "test_visible": False,
                "generator_module": ORCHESTRATOR.GENERATOR_MODULE,
                "source_closure": {},
            },
        )
        self.prerequisite = _write_receipt(
            self.root / "selected-five.json",
            {
                "format": "semtalk_show_selected_five_fixture_v1",
                "status": "selected",
                "target_dataset": "SHOW",
                "target_speaker_scope": AUTHORITY.EXPECTED_SCOPE,
                "split": "val",
                "test_visible": False,
            },
        )
        self.val_inputs = _write_receipt(
            self.root / "val-inputs.json",
            {
                "format": "semtalk_show_base_val_inputs_fixture_v1",
                "status": "frozen",
                "dataset": "SHOW",
                "target_speaker_scope": AUTHORITY.EXPECTED_SCOPE,
                "split": "val",
                "test_visible": False,
            },
        )
        self.subset_rows = [
            {
                "canonical_clip_id": f"all-speakers-val-{index:03d}",
                "split": "val",
            }
            for index in range(ORCHESTRATOR.PROBE_CLIPS_PER_CANDIDATE)
        ]
        self.subset = _write_jsonl(
            self.root / "probe-subset.jsonl", self.subset_rows
        )
        self.checkpoints = {
            epoch: _write_bytes(
                self.root / f"base-e{epoch}.pth",
                f"official SemTalk Base SHOW e{epoch}\n".encode("ascii"),
            )
            for epoch in ORCHESTRATOR.PROBE_EPOCHS
        }
        self.binding = {
            "source": copy.deepcopy(self.source),
            "pipeline": copy.deepcopy(self.pipeline),
            "prerequisite_selection": copy.deepcopy(self.prerequisite),
            "val_inputs": copy.deepcopy(self.val_inputs),
            "candidate_checkpoints": [
                {
                    "epoch": epoch,
                    "candidate_checkpoint": copy.deepcopy(
                        self.checkpoints[epoch]
                    ),
                }
                for epoch in ORCHESTRATOR.PROBE_EPOCHS
            ],
            "subset_manifest": copy.deepcopy(self.subset),
            "seed": 20260731,
            "clips_per_candidate": (
                ORCHESTRATOR.PROBE_CLIPS_PER_CANDIDATE
            ),
            "shards_per_candidate": AUTHORITY.EXPECTED_SHARDS,
        }
        self.executables = {}
        for name in ("python", "quality-producer", "runner", "nvidia-smi"):
            path = self.root / "bin" / name
            artifact = _write_bytes(path, b"#!/bin/sh\nexit 0\n")
            path.chmod(0o700)
            self.executables[name] = artifact
        self.lmdb = {
            role: _write_bytes(
                self.root / "lmdb" / f"{role}.bin",
                f"SHOW LMDB {role}\n".encode("ascii"),
            )
            for role in ("summary", "data", "lock")
        }
        self.candidate_bundle = {
            role: _write_bytes(
                self.root / "candidate-bundle" / f"{role}.json",
                _canonical_bytes({"fixture_role": role}),
            )
            for role in ("manifest", "status", "frozen_inputs")
        }
        self.training_metric_rows = [
            {
                "format": "semtalk_show_base_long_epoch_metric_v1",
                "epoch": epoch,
                "optimizer_updates": epoch * 248,
                "updates_per_epoch": 248,
                "metrics": {"total": 1.0 / float(epoch + 1)},
                "all_finite": True,
            }
            for epoch in range(1, 401)
        ]
        self.training_metrics = _write_jsonl(
            self.root / "training" / "epoch-metrics.jsonl",
            self.training_metric_rows,
        )
        self.candidate_bundle["status"] = _write_json(
            Path(self.candidate_bundle["status"]["path"]),
            {
                "status": "complete",
                "epoch_metrics_jsonl": self.training_metrics["path"],
                "epoch_metrics_sha256": self.training_metrics["sha256"],
                "epoch_metrics_records": 400,
            },
        )
        self.feature_extractor = _write_bytes(
            self.root / "metric-assets" / "feature-extractor.pth",
            b"CPU fixture TalkSHOW feature extractor\n",
        )
        self.smplx_asset = _write_bytes(
            self.root / "metric-assets" / "SMPLX_NEUTRAL_2020.npz",
            b"CPU fixture SMPL-X asset\n",
        )
        self.ground_truth = _write_bytes(
            self.root / "ground-truth" / "shared-val-ground-truth.npz",
            b"CPU fixture canonical SHOW validation ground truth\n",
        )
        self.metric_assets = {
            "talkshow_metric_root": str(
                (self.root / "metric-assets" / "talkshow").resolve()
            ),
            "talkshow_source": {},
            "feature_extractor": self.feature_extractor,
            "smplx_asset": self.smplx_asset,
        }

    def execution_spec(
        self,
        tag: str,
        *,
        host: str,
        candidates_per_wave: int,
        execution_mode: str,
        representation_lmdb: dict[str, object] | None = None,
    ) -> dict[str, object]:
        quality = _write_receipt(
            self.root / "specs" / f"{tag}-quality.json",
            {
                "format": PRODUCER.QUALITY_INPUT_FORMAT,
                "status": "frozen",
                "dataset": "SHOW",
                "target_speaker_scope": AUTHORITY.EXPECTED_SCOPE,
                "split": "val",
                "test_visible": False,
                "source": self.binding["source"],
                "pipeline": self.binding["pipeline"],
                "selected_five_authority": self.binding[
                    "prerequisite_selection"
                ],
                "candidate_bundle": self.candidate_bundle,
                "val_inputs": self.binding["val_inputs"],
                "representation_lmdb": representation_lmdb or self.lmdb,
                "training_metrics": self.training_metrics,
                "metric_assets": self.metric_assets,
                "topology": {
                    "world_size": 8,
                    "physical_gpus": list(PRODUCER.EXPECTED_GPUS),
                    "candidates_per_wave": candidates_per_wave,
                    "execution_mode": execution_mode,
                    "seed": self.binding["seed"],
                    "clips_per_candidate": self.binding[
                        "clips_per_candidate"
                    ],
                    "shards_per_candidate": self.binding[
                        "shards_per_candidate"
                    ],
                },
            },
        )
        spec_artifact = _write_receipt(
            self.root / "specs" / f"{tag}-execution.json",
            {
                "format": PRODUCER.EXECUTION_SPEC_FORMAT,
                "status": "frozen",
                "split": "val",
                "test_visible": False,
                "formal_host": host,
                "candidates_per_wave": candidates_per_wave,
                "execution_mode": execution_mode,
                "probe_binding": self.binding,
                "quality_input": quality,
                "source_root": str(self.root),
                "python": self.executables["python"],
                "quality_producer": self.executables["quality-producer"],
                "runner": {
                    **self.executables["runner"],
                    "path": str(PRODUCER.EXPECTED_RUNNER),
                },
                "nvidia_smi": self.executables["nvidia-smi"],
            },
        )
        _artifact, spec = PRODUCER._payload_artifact(
            spec_artifact, "CPU execution spec fixture"
        )
        self.spec_registry[str(spec_artifact["path"])] = (
            copy.deepcopy(spec_artifact),
            copy.deepcopy(spec),
            copy.deepcopy(self.binding),
        )
        return spec_artifact

    def validate_execution_spec(
        self, artifact: dict[str, object]
    ) -> tuple[
        dict[str, object], dict[str, object], dict[str, object]
    ]:
        normalized, _value = PRODUCER._payload_artifact(
            artifact, "CPU execution spec fixture"
        )
        registered = self.spec_registry.get(normalized["path"])
        if registered is None or registered[0] != normalized:
            raise PRODUCER.ProbeProducerError(
                "unregistered CPU execution spec fixture"
            )
        PRODUCER._payload_artifact(
            registered[1]["quality_input"], "CPU quality input fixture"
        )
        return copy.deepcopy(registered)

    def replay_context(self):
        stack = contextlib.ExitStack()
        stack.enter_context(
            mock.patch.object(
                PRODUCER,
                "validate_execution_spec",
                side_effect=self.validate_execution_spec,
            )
        )
        stack.enter_context(
            mock.patch.object(
                PRODUCER, "_execution_input_closure", return_value=[]
            )
        )
        stack.enter_context(
            mock.patch.object(
                PRODUCER,
                "_validate_full_inference_lineage",
                side_effect=self.validate_full_inference_lineage,
            )
        )
        stack.enter_context(
            mock.patch.object(
                PRODUCER,
                "_validate_workload_execution",
                side_effect=self.validate_workload_execution,
            )
        )
        return stack

    def validate_full_inference_lineage(
        self,
        *,
        lineage_artifact: dict[str, object],
        lineage: dict[str, object],
        epoch: int,
        checkpoint: dict[str, object],
        binding: dict[str, object],
        workload_root: Path,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        del lineage_artifact, checkpoint, binding
        reference = lineage["final_manifest"]
        full_manifest = PRODUCER._output_artifact(
            Path(reference["path"]),
            f"CPU Base e{epoch} full prediction manifest",
        )
        if any(
            full_manifest[key] != reference[key]
            for key in ("path", "sha256", "bytes")
        ):
            raise PRODUCER.ProbeProducerError(
                "CPU full prediction manifest reference changed"
            )
        Path(full_manifest["path"]).relative_to(workload_root)
        rows = AUTHORITY._strict_jsonl_bytes(
            Path(full_manifest["path"]).read_bytes(),
            f"CPU Base e{epoch} full prediction manifest",
        )
        if len(rows) != ORCHESTRATOR.EXPECTED_CLIPS:
            raise PRODUCER.ProbeProducerError(
                "CPU full prediction coverage changed"
            )
        return full_manifest, rows

    def validate_workload_execution(
        self,
        value: dict[str, object],
        *,
        execution_spec: dict[str, object],
        binding: dict[str, object],
        workload_root: Path,
    ) -> tuple[
        dict[str, object],
        dict[int, dict[str, dict[str, object]]],
        set[str],
    ]:
        artifact, receipt = PRODUCER._payload_artifact(
            value, "CPU probe workload execution"
        )
        if (
            receipt.get("format") != PRODUCER.WORKLOAD_EXECUTION_FORMAT
            or receipt.get("status") != "complete"
            or receipt.get("split") != "val"
            or receipt.get("test_visible") is not False
            or receipt.get("formal_host")
            != execution_spec["formal_host"]
            or receipt.get("candidates_per_wave")
            != execution_spec["candidates_per_wave"]
            or receipt.get("execution_mode")
            != execution_spec["execution_mode"]
            or receipt.get("probe_binding_sha256")
            != AUTHORITY.canonical_json_sha256(binding)
            or receipt.get("quality_input")
            != execution_spec["quality_input"]
            or artifact["path"]
            != str(workload_root / "workload-execution.json")
        ):
            raise PRODUCER.ProbeProducerError(
                "CPU workload execution identity changed"
            )
        candidates = receipt.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(
            ORCHESTRATOR.PROBE_EPOCHS
        ):
            raise PRODUCER.ProbeProducerError(
                "CPU workload execution coverage changed"
            )
        by_epoch: dict[int, dict[str, dict[str, object]]] = {}
        for expected_epoch, row in zip(
            ORCHESTRATOR.PROBE_EPOCHS, candidates
        ):
            if row.get("epoch") != expected_epoch:
                raise PRODUCER.ProbeProducerError(
                    "CPU workload execution order changed"
                )
            lineage_artifact, _lineage = PRODUCER._payload_artifact(
                row["inference_lineage"],
                f"CPU Base e{expected_epoch} execution lineage",
            )
            full_manifest = PRODUCER._output_artifact(
                Path(row["full_prediction_manifest"]["path"]),
                f"CPU Base e{expected_epoch} execution full manifest",
            )
            if full_manifest != row["full_prediction_manifest"]:
                raise PRODUCER.ProbeProducerError(
                    "CPU workload execution full manifest changed"
                )
            by_epoch[expected_epoch] = {
                "inference_lineage": lineage_artifact,
                "full_prediction_manifest": full_manifest,
            }
        return artifact, by_epoch, {artifact["path"]}

    def _candidate_ready(
        self,
        *,
        root: Path,
        spec: dict[str, object],
    ) -> dict[str, object]:
        candidates = []
        executed_candidates = []
        real = np.arange(64 * 6, dtype=np.float64).reshape(64, 6) / 100.0
        checkpoints = {
            row["epoch"]: row["candidate_checkpoint"]
            for row in self.binding["candidate_checkpoints"]
        }
        for epoch in ORCHESTRATOR.PROBE_EPOCHS:
            prediction_rows = []
            for index, subset_row in enumerate(self.subset_rows):
                prediction = _write_bytes(
                    (
                        self.root / "external-predictions"
                        if self.behavior.get("external_prediction")
                        else root / f"e{epoch}" / "predictions"
                    )
                    / f"{epoch}-{index:03d}.npy",
                    f"Base-e{epoch}-clip-{index:03d}\n".encode("ascii"),
                )
                prediction_rows.append(
                    {
                        "canonical_clip_id": subset_row[
                            "canonical_clip_id"
                        ],
                        "prediction": prediction,
                    }
                )
            prediction_manifest = _write_jsonl(
                root / f"e{epoch}" / "prediction-manifest.jsonl",
                prediction_rows,
            )
            full_prediction_rows = [
                {
                    "global_index": index,
                    "canonical_clip_id": row["canonical_clip_id"],
                    "frames": 120,
                    "epoch": epoch,
                    "candidate_checkpoint_sha256": checkpoints[epoch][
                        "sha256"
                    ],
                    "prediction": row["prediction"],
                    "ground_truth": self.ground_truth,
                }
                for index, row in enumerate(prediction_rows)
            ]
            full_prediction_rows.extend(
                {
                    "global_index": index,
                    "canonical_clip_id": (
                        f"all-speakers-val-{index:04d}"
                    ),
                    "frames": 120,
                    "epoch": epoch,
                    "candidate_checkpoint_sha256": checkpoints[epoch][
                        "sha256"
                    ],
                    "prediction": prediction_rows[-1]["prediction"],
                    "ground_truth": self.ground_truth,
                }
                for index in range(
                    ORCHESTRATOR.PROBE_CLIPS_PER_CANDIDATE,
                    ORCHESTRATOR.EXPECTED_CLIPS,
                )
            )
            full_prediction_manifest = _write_jsonl(
                root / f"e{epoch}" / "final_manifest.jsonl",
                full_prediction_rows,
            )
            real_features = _write_npy(
                root / f"e{epoch}" / "real-features.npy", real
            )
            generated_features = _write_npy(
                root / f"e{epoch}" / "generated-features.npy",
                np.repeat(
                    real + float(epoch) / 1000.0,
                    2,
                    axis=0,
                ),
            )
            feature_manifest = _write_jsonl(
                root / f"e{epoch}" / "feature-manifest.jsonl",
                [
                    {
                        "canonical_clip_id": row["canonical_clip_id"],
                        "real_start": index,
                        "real_rows": 1,
                        "generated_start": 2 * index,
                        "generated_rows": 2,
                    }
                    for index, row in enumerate(self.subset_rows)
                ],
            )
            trajectory = _write_jsonl(
                root / f"e{epoch}" / "training-trajectory.jsonl",
                [
                    {
                        "optimizer_update": row["optimizer_updates"],
                        "loss": row["metrics"]["total"],
                    }
                    for row in self.training_metric_rows[:epoch]
                ],
            )
            inference_lineage = _write_receipt(
                root / f"e{epoch}" / "inference-lineage.json",
                {
                    "format": (
                        ORCHESTRATOR.val_contract.VAL_INFERENCE_LINEAGE_FORMAT
                    ),
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": epoch,
                    "candidate_checkpoint": {
                        "path": checkpoints[epoch]["path"],
                        "sha256": checkpoints[epoch]["sha256"],
                    },
                    "val_inputs_receipt": {
                        key: self.binding["val_inputs"][key]
                        for key in (
                            "path",
                            "sha256",
                            "receipt_payload_sha256",
                        )
                    },
                    "pipeline_receipt": {
                        key: self.binding["pipeline"][key]
                        for key in (
                            "path",
                            "sha256",
                            "receipt_payload_sha256",
                        )
                    },
                    "clip_count": ORCHESTRATOR.EXPECTED_CLIPS,
                    "prediction_files": ORCHESTRATOR.EXPECTED_CLIPS,
                    "ground_truth_files": ORCHESTRATOR.EXPECTED_CLIPS,
                    "exact_once": True,
                    "finite": True,
                    "final_manifest": full_prediction_manifest,
                },
            )
            native = _write_receipt(
                root / f"e{epoch}" / "trainer-native.json",
                {
                    "format": PRODUCER.TRAINER_NATIVE_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": epoch,
                    "candidate_checkpoint": checkpoints[epoch],
                    "subset_manifest": self.binding["subset_manifest"],
                    "prediction_manifest": prediction_manifest,
                    "quality_input": spec["quality_input"],
                    "inference_lineage": inference_lineage,
                    "real_features": real_features,
                    "generated_features": generated_features,
                    "feature_manifest": feature_manifest,
                    "training_trajectory": trajectory,
                },
            )
            candidates.append(native)
            executed_candidates.append(
                {
                    "epoch": epoch,
                    "inference_lineage": inference_lineage,
                    "full_prediction_manifest": full_prediction_manifest,
                }
            )
        workload_execution = _write_receipt(
            root / "workload-execution.json",
            {
                "format": PRODUCER.WORKLOAD_EXECUTION_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "formal_host": spec["formal_host"],
                "candidates_per_wave": spec["candidates_per_wave"],
                "execution_mode": spec["execution_mode"],
                "probe_binding_sha256": AUTHORITY.canonical_json_sha256(
                    self.binding
                ),
                "quality_input": spec["quality_input"],
                "prepare": {},
                "candidates": executed_candidates,
            },
        )
        return _write_receipt(
            root / "candidate-ready.json",
            {
                "format": PRODUCER.CANDIDATE_READY_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "formal_host": spec["formal_host"],
                "candidates_per_wave": spec["candidates_per_wave"],
                "execution_mode": spec["execution_mode"],
                "probe_binding_sha256": AUTHORITY.canonical_json_sha256(
                    self.binding
                ),
                "quality_input": spec["quality_input"],
                "workload_execution": workload_execution,
                "candidates": candidates,
            },
        )

    def fake_execute(self, **kwargs: object) -> PRODUCER.ObservedExecution:
        runner_argv = list(kwargs["runner_argv"])
        workload_argv = list(kwargs["workload_argv"])
        root = Path(kwargs["root"])
        spec_sha = workload_argv[
            workload_argv.index("--execution-spec-sha256") + 1
        ]
        public_spec_path = next(
            path
            for path, (artifact, spec, _binding) in self.spec_registry.items()
            if artifact["sha256"] == spec_sha
            and spec["formal_host"]
            == workload_argv[workload_argv.index("--formal-host") + 1]
            and str(spec["candidates_per_wave"])
            == workload_argv[
                workload_argv.index("--candidates-per-wave") + 1
            ]
            and spec["execution_mode"]
            == workload_argv[workload_argv.index("--execution-mode") + 1]
        )
        spec = self.spec_registry[public_spec_path][1]
        workload_root = root / "workload"
        self._candidate_ready(root=workload_root, spec=spec)

        self.fake_pid += 100
        runner_pid = self.fake_pid
        child_pid = runner_pid + 1
        restored = {
            str(index): runner_pid + 10 + index
            for index in PRODUCER.EXPECTED_GPUS
        }
        return_code = int(self.behavior.get("runner_rc", 0))
        status = {
            "state": "finished",
            "return_code": return_code,
            "error": None,
            "cleanup_error": None,
            "restore_error": None,
            "received_signal": None,
            "command": workload_argv,
            "wrapper_pid": runner_pid,
            "child_pid": child_pid,
            "restored_guards": restored,
        }
        if not self.behavior.get("missing_status"):
            _write_json(root / "runner-status.json", status)
        log_payload = (
            b"CUDA out of memory\n"
            if self.behavior.get("oom_log")
            else b"guarded probe complete\n"
        )
        _write_bytes(root / "runner.log", log_payload)
        if not self.behavior.get("missing_stdout"):
            _write_bytes(root / "runner.stdout", b"quality probe complete\n")
        _write_bytes(root / "runner.stderr", b"")
        started = 1_000_000_000
        finished = started + self.elapsed_seconds * 1_000_000_000
        if self.behavior.get("bad_timing"):
            finished = started
        peaks = [70] * len(PRODUCER.EXPECTED_GPUS)
        totals = [100] * len(PRODUCER.EXPECTED_GPUS)
        if self.behavior.get("bad_memory"):
            peaks[0] = 101
        sample_stamp = started + 1 if finished > started else started
        telemetry = ORCHESTRATOR._with_payload_sha(
            {
                "format": PRODUCER.MEMORY_TELEMETRY_FORMAT,
                "status": "complete",
                "gpu_indices": list(PRODUCER.EXPECTED_GPUS),
                "samples": [
                    {
                        "monotonic_ns": sample_stamp,
                        "used_bytes": peaks,
                    }
                ],
                "peak_memory_bytes": peaks,
                "total_memory_bytes": totals,
            }
        )
        telemetry_artifact = ORCHESTRATOR._write_new(
            root / "memory-telemetry.json", telemetry
        )
        runner_identity = {
            "pid": runner_pid,
            "ppid": 40000,
            "pgid": runner_pid,
            "sid": 40000,
            "starttime_ticks": 1234,
            "argv": runner_argv,
            "argv_sha256": _argv_sha(runner_argv),
        }
        child_identity = {
            "pid": child_pid,
            "ppid": runner_pid,
            "pgid": runner_pid,
            "sid": 40000,
            "starttime_ticks": 1235,
            "argv": workload_argv,
            "argv_sha256": _argv_sha(workload_argv),
        }
        guard_identities = {}
        for index in PRODUCER.EXPECTED_GPUS:
            argv = [
                "python",
                "globaldiff_gpu_guard_cnn",
                "torchvision",
                "resnet18",
                str(index),
            ]
            if self.behavior.get("fake_guard") and index == 0:
                argv = ["python", "fake_guard", "0"]
            guard_identities[str(index)] = {
                "pid": restored[str(index)],
                "ppid": runner_identity["ppid"],
                "pgid": restored[str(index)],
                "sid": restored[str(index)],
                "starttime_ticks": 2000 + index,
                "argv": argv,
                "argv_sha256": _argv_sha(argv),
            }
        guard_verification = {
            "restored_guards": restored,
            "guard_identities": guard_identities,
            "guard_executables": {
                str(index): (
                    self.executables["quality-producer"]
                    if self.behavior.get("fake_guard_executable")
                    and index == 0
                    else spec["python"]
                )
                for index in PRODUCER.EXPECTED_GPUS
            },
            "nvidia_smi_compute_rows_sha256": "a" * 64,
            "nvidia_smi_uuid_rows_sha256": "b" * 64,
        }
        if self.behavior.get("swap_input"):
            replacement = Path(public_spec_path).with_suffix(".replacement")
            replacement.write_bytes(b"{}\n")
            os.replace(replacement, public_spec_path)
        if self.behavior.get("swap_root"):
            moved = root.with_name(root.name + ".moved")
            root.rename(moved)
            root.mkdir()
        return PRODUCER.ObservedExecution(
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            gpu_peak_memory_bytes=peaks,
            gpu_total_memory_bytes=totals,
            runner_return_code=return_code,
            oom_observed=bool(self.behavior.get("fake_oom", False)),
            descendants_exited=not bool(
                self.behavior.get("live_descendant", False)
            ),
            runner_identity=runner_identity,
            observed_descendants=[
                child_identity,
                *guard_identities.values(),
            ],
            status_artifact=PRODUCER._output_artifact(
                root / "runner-status.json", "fake runner status"
            ),
            log_artifact=PRODUCER._output_artifact(
                root / "runner.log", "fake runner log"
            ),
            stdout_artifact=PRODUCER._output_artifact(
                root / "runner.stdout", "fake runner stdout"
            ),
            stderr_artifact=PRODUCER._output_artifact(
                root / "runner.stderr",
                "fake runner stderr",
                allow_empty=True,
            ),
            telemetry_artifact=telemetry_artifact,
            status=status,
            guard_verification=guard_verification,
        )

    def run(
        self,
        tag: str,
        *,
        host: str,
        candidates_per_wave: int,
        execution_mode: str,
        elapsed_seconds: int,
        behavior: dict[str, object] | None = None,
        representation_lmdb: dict[str, object] | None = None,
    ) -> dict[str, object]:
        spec = self.execution_spec(
            tag,
            host=host,
            candidates_per_wave=candidates_per_wave,
            execution_mode=execution_mode,
            representation_lmdb=representation_lmdb,
        )
        output_root = self.root / "runs" / tag
        self.behavior = behavior or {}
        self.elapsed_seconds = elapsed_seconds
        try:
            with self.replay_context(), mock.patch.object(
                PRODUCER.FormalExecutionBackend,
                "execute",
                side_effect=self.fake_execute,
            ):
                run = PRODUCER.run_and_seal_probe(
                    spec, output_root=output_root
                )
        finally:
            self.behavior = {}
        return _write_receipt(output_root / "probe-run.json", run)

    def comparison_matrix(self) -> list[dict[str, object]]:
        comparisons = []
        elapsed = {1: 90, 2: 55, 4: 40}
        for host_index, host in enumerate(
            ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values()
        ):
            serial = self.run(
                f"host{host_index}-serial",
                host=host,
                candidates_per_wave=1,
                execution_mode="serial",
                elapsed_seconds=120,
            )
            for mode in ORCHESTRATOR.CONCURRENCY_SELECTION_ORDER:
                concurrent = self.run(
                    f"host{host_index}-c{mode}",
                    host=host,
                    candidates_per_wave=mode,
                    execution_mode="concurrent",
                    elapsed_seconds=elapsed[mode] + host_index,
                )
                with self.replay_context():
                    comparison = (
                        ORCHESTRATOR.build_multicandidate_comparison(
                            serial_run=serial,
                            concurrent_run=concurrent,
                        )
                    )
                comparisons.append(
                    _write_receipt(
                        self.root
                        / "comparisons"
                        / f"host{host_index}-c{mode}.json",
                        comparison,
                    )
                )
        return comparisons


class BaseFreshConcurrencyProducerCpuTests(unittest.TestCase):
    def test_lineage_authority_requires_exact_compact_receipt(self) -> None:
        full = {
            "path": "/authority.json",
            "sha256": "1" * 64,
            "bytes": 17,
            "receipt_payload_sha256": "2" * 64,
        }
        compact = {
            key: full[key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        }
        self.assertTrue(
            PRODUCER._same_lineage_authority_receipt(compact, full)
        )
        self.assertFalse(
            PRODUCER._same_lineage_authority_receipt(full, full)
        )
        attacked = dict(compact)
        attacked["sha256"] = "3" * 64
        self.assertFalse(
            PRODUCER._same_lineage_authority_receipt(attacked, full)
        )

    def test_common_valid_alternate_lmdb_and_selection_fail_training_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            selected = _write_receipt(
                root / "selected-five.json",
                {"format": "selected-five", "status": "selected"},
            )
            lmdb_root = root / "base.lmdb"
            data = _write_bytes(lmdb_root / "data.mdb", b"selected data\n")
            lock = _write_bytes(lmdb_root / "lock.mdb", b"selected lock\n")
            lineage = _write_bytes(root / "lineage.json", b"{}\n")
            summary_body = {
                "format": "semtalk_show_base_lmdb_summary_v1",
                "status": "complete",
                "scope": "SemTalk Base only",
                "entries": 127_286,
                "train_clips": 13_687,
                "lmdb": str(lmdb_root),
                "data_mdb_sha256": data["sha256"],
                "lock_mdb_sha256": lock["sha256"],
                "lineage_json": lineage["path"],
                "lineage_json_sha256": lineage["sha256"],
            }
            summary = _write_json(root / "summary.json", summary_body)
            selected_compact = {
                key: selected[key]
                for key in ("path", "sha256", "receipt_payload_sha256")
            }
            selected_dataset = {
                "format": (
                    "semtalk_show_base_selected_feature_dataset_receipt_v1"
                ),
                "lmdb": str(lmdb_root),
                "summary": summary["path"],
                "summary_sha256": summary["sha256"],
                "lineage": lineage["path"],
                "lineage_sha256": lineage["sha256"],
                "entries": 127_286,
                "train_clips": 13_687,
                "split": "train",
                "test_visible": False,
                "data_mdb_sha256": data["sha256"],
                "lock_mdb_sha256": lock["sha256"],
                "prerequisite_selection": selected_compact,
                "selected_prerequisite_sha256": {
                    stage: f"{index + 1:x}" * 64
                    for index, stage in enumerate(AUTHORITY.STAGES)
                },
                "lmdb_binding_scope": (
                    "ordered_node_local_inode_bindings_with_global_content_sha256"
                ),
                "node_lmdb_inode_bindings": [],
            }
            normalized = PRODUCER._validate_representation_lmdb_authority(
                {"summary": summary, "data": data, "lock": lock},
                selected_authority=selected,
                selected_dataset=selected_dataset,
            )
            self.assertEqual(normalized["data"], data)

            alternate_selected = _write_receipt(
                root / "alternate-selected-five.json",
                {"format": "selected-five", "status": "selected"},
            )
            alternate_root = root / "alternate-base.lmdb"
            alternate_data = _write_bytes(
                alternate_root / "data.mdb", b"alternate valid data\n"
            )
            alternate_lock = _write_bytes(
                alternate_root / "lock.mdb", b"alternate valid lock\n"
            )
            alternate_summary = _write_json(
                root / "alternate-summary.json",
                {
                    **summary_body,
                    "lmdb": str(alternate_root),
                    "data_mdb_sha256": alternate_data["sha256"],
                    "lock_mdb_sha256": alternate_lock["sha256"],
                },
            )
            with self.assertRaises(PRODUCER.ProbeProducerError):
                PRODUCER._validate_representation_lmdb_authority(
                    {
                        "summary": alternate_summary,
                        "data": alternate_data,
                        "lock": alternate_lock,
                    },
                    selected_authority=alternate_selected,
                    selected_dataset=selected_dataset,
                )

    def test_caller_scalar_and_capture_producers_are_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProducerFixture(Path(raw))
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_probe_metric(
                    epoch=1,
                    checkpoint=fixture.checkpoints[1],
                    subset_manifest=fixture.subset,
                    prediction_manifest=fixture.subset,
                    metric_values={"body.released2.metrics.FGD": 0.0},
                )
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_probe_run({})
            with self.assertRaises(SystemExit):
                ORCHESTRATOR._parse_args(["build-probe-metric"])
            with self.assertRaises(SystemExit):
                ORCHESTRATOR._parse_args(["build-probe-run"])

    def test_sole_formal_cli_runs_create_new_and_replays_standalone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProducerFixture(Path(raw))
            spec = fixture.execution_spec(
                "formal-cli",
                host=next(
                    iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
                ),
                candidates_per_wave=2,
                execution_mode="concurrent",
            )
            output_root = fixture.root / "runs" / "formal-cli"
            output_json = output_root / "probe-run.json"
            argv = [
                "run-and-seal-probe",
                "--execution-spec-path",
                str(spec["path"]),
                "--execution-spec-sha256",
                str(spec["sha256"]),
                "--execution-spec-payload-sha256",
                str(spec["receipt_payload_sha256"]),
                "--output-root",
                str(output_root),
                "--output-json",
                str(output_json),
            ]
            with fixture.replay_context(), mock.patch.object(
                PRODUCER.FormalExecutionBackend,
                "execute",
                side_effect=fixture.fake_execute,
            ):
                self.assertEqual(PRODUCER.main(argv), 0)
                with self.assertRaises(PRODUCER.ProbeProducerError):
                    PRODUCER.main(argv)
            run_artifact, _run = PRODUCER._payload_artifact(
                {
                    "path": str(output_json),
                    "sha256": hashlib.sha256(output_json.read_bytes()).hexdigest(),
                    "bytes": output_json.stat().st_size,
                    "receipt_payload_sha256": json.loads(
                        output_json.read_text(encoding="utf-8")
                    )["receipt_payload_sha256"],
                },
                "formal CLI probe run",
            )
            replay_argv = [
                "replay-probe-run",
                "--probe-run-path",
                str(run_artifact["path"]),
                "--probe-run-sha256",
                str(run_artifact["sha256"]),
                "--probe-run-payload-sha256",
                str(run_artifact["receipt_payload_sha256"]),
            ]
            with fixture.replay_context():
                self.assertEqual(PRODUCER.main(replay_argv), 0)

    def test_normal_two_host_serial_c1_c2_c4_matrix_and_gate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProducerFixture(Path(raw))
            comparisons = fixture.comparison_matrix()
            self.assertEqual(len(comparisons), 6)
            with fixture.replay_context():
                gate = ORCHESTRATOR.build_multicandidate_gate(comparisons)
                self.assertEqual(gate["status"], "pass")
                self.assertEqual(gate["selected_candidates_per_wave"], 4)
                for comparison in comparisons:
                    _artifact, payload = AUTHORITY._verify_compact_receipt(
                        comparison, "CPU comparison"
                    )
                    self.assertTrue(
                        payload["equivalence"][
                            "deterministic_semantics_equal"
                        ]
                    )

    def test_serial_concurrent_pair_cannot_switch_representation_lmdb(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProducerFixture(Path(raw))
            host = next(
                iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
            )
            serial = fixture.run(
                "lmdb-serial",
                host=host,
                candidates_per_wave=1,
                execution_mode="serial",
                elapsed_seconds=120,
            )
            alternate_lmdb = {
                role: _write_bytes(
                    fixture.root / "alternate-lmdb" / f"{role}.bin",
                    f"different SHOW LMDB {role}\n".encode("ascii"),
                )
                for role in ("summary", "data", "lock")
            }
            concurrent = fixture.run(
                "lmdb-concurrent",
                host=host,
                candidates_per_wave=2,
                execution_mode="concurrent",
                elapsed_seconds=55,
                representation_lmdb=alternate_lmdb,
            )
            with fixture.replay_context(), self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_multicandidate_comparison(
                    serial_run=serial,
                    concurrent_run=concurrent,
                )

    def test_fake_runtime_evidence_and_path_swap_fail_closed(self) -> None:
        attacks = {
            "fake-rc": {"runner_rc": 7},
            "fake-oom": {"fake_oom": True},
            "owned-oom-log": {"oom_log": True},
            "fake-memory": {"bad_memory": True},
            "fake-timing": {"bad_timing": True},
            "fake-guard": {"fake_guard": True},
            "live-descendant": {"live_descendant": True},
            "missing-status": {"missing_status": True},
            "missing-stdout": {"missing_stdout": True},
            "input-path-swap": {"swap_input": True},
            "run-root-swap": {"swap_root": True},
            "external-prediction": {"external_prediction": True},
            "fake-guard-executable": {"fake_guard_executable": True},
        }
        for tag, behavior in attacks.items():
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as raw:
                fixture = ProducerFixture(Path(raw))
                with self.assertRaises(Exception):
                    fixture.run(
                        tag,
                        host=next(
                            iter(
                                ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values()
                            )
                        ),
                        candidates_per_wave=2,
                        execution_mode="concurrent",
                        elapsed_seconds=60,
                        behavior=behavior,
                    )

    def test_replay_rejects_scalar_old_cross_mode_and_authority_attacks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProducerFixture(Path(raw))
            host = next(
                iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
            )
            run_c2 = fixture.run(
                "replay-c2",
                host=host,
                candidates_per_wave=2,
                execution_mode="concurrent",
                elapsed_seconds=55,
            )
            run_c4 = fixture.run(
                "replay-c4",
                host=host,
                candidates_per_wave=4,
                execution_mode="concurrent",
                elapsed_seconds=40,
            )
            _artifact, base_run = AUTHORITY._verify_compact_receipt(
                run_c2, "base run"
            )
            _artifact, evidence = AUTHORITY._verify_compact_receipt(
                base_run["producer_evidence"], "base producer evidence"
            )

            metric_artifact = base_run["candidate_outputs"][0][
                "metric_receipt"
            ]
            _artifact, metric = AUTHORITY._verify_compact_receipt(
                metric_artifact, "base derived metric"
            )
            metric["metric_values"]["body.released2.metrics.FGD"] += 1.0
            fake_metric = _write_receipt(
                fixture.root / "attacks" / "fake-metric.json", metric
            )
            scalar_evidence = copy.deepcopy(evidence)
            scalar_evidence["derived_metrics"][0] = fake_metric
            scalar_evidence_artifact = _write_receipt(
                fixture.root / "attacks" / "scalar-evidence.json",
                scalar_evidence,
            )
            scalar_run = copy.deepcopy(base_run)
            scalar_run["producer_evidence"] = scalar_evidence_artifact
            scalar_run["candidate_outputs"][0]["metric_receipt"] = (
                fake_metric
            )
            attacks = [
                _write_receipt(
                    fixture.root / "attacks" / "scalar-run.json",
                    scalar_run,
                )
            ]

            old_run = copy.deepcopy(base_run)
            old_run["format"] = (
                "semtalk_show_base_fresh_val_multicandidate_probe_run_v1"
            )
            attacks.append(
                _write_receipt(
                    fixture.root / "attacks" / "old-run.json", old_run
                )
            )

            _artifact, c4_payload = AUTHORITY._verify_compact_receipt(
                run_c4, "C4 run"
            )
            cross_mode = copy.deepcopy(base_run)
            cross_mode["producer_evidence"] = c4_payload[
                "producer_evidence"
            ]
            attacks.append(
                _write_receipt(
                    fixture.root / "attacks" / "cross-mode.json",
                    cross_mode,
                )
            )

            wrong_authority = copy.deepcopy(base_run)
            wrong_authority["probe_binding"]["source"]["commit"] = "f" * 40
            attacks.append(
                _write_receipt(
                    fixture.root / "attacks" / "wrong-authority.json",
                    wrong_authority,
                )
            )
            with fixture.replay_context():
                for attack in attacks:
                    with self.subTest(path=attack["path"]), self.assertRaises(
                        ORCHESTRATOR.BaseFreshValOrchestratorError
                    ):
                        ORCHESTRATOR._replay_probe_run(attack)

    def test_missing_owned_artifact_and_raw_trajectory_fail_replay(self) -> None:
        for target in ("runner_stdout", "runner_status", "trajectory"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                fixture = ProducerFixture(Path(raw))
                run = fixture.run(
                    f"missing-{target}",
                    host=next(
                        iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
                    ),
                    candidates_per_wave=1,
                    execution_mode="serial",
                    elapsed_seconds=120,
                )
                _artifact, payload = AUTHORITY._verify_compact_receipt(
                    run, "missing artifact run"
                )
                _artifact, evidence = AUTHORITY._verify_compact_receipt(
                    payload["producer_evidence"], "producer evidence"
                )
                if target == "trajectory":
                    _artifact, ready = AUTHORITY._verify_compact_receipt(
                        evidence["candidate_ready"], "candidate ready"
                    )
                    _artifact, native = AUTHORITY._verify_compact_receipt(
                        ready["candidates"][0], "trainer native"
                    )
                    path = Path(native["training_trajectory"]["path"])
                else:
                    path = Path(evidence[target]["path"])
                path.unlink()
                with fixture.replay_context(), self.assertRaises(
                    ORCHESTRATOR.BaseFreshValOrchestratorError
                ):
                    ORCHESTRATOR._replay_probe_run(run)

    def test_output_root_is_create_new_and_nonreusable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProducerFixture(Path(raw))
            host = next(
                iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
            )
            fixture.run(
                "nonreuse",
                host=host,
                candidates_per_wave=1,
                execution_mode="serial",
                elapsed_seconds=120,
            )
            spec = fixture.execution_spec(
                "nonreuse-second",
                host=host,
                candidates_per_wave=1,
                execution_mode="serial",
            )
            with fixture.replay_context(), self.assertRaises(
                PRODUCER.ProbeProducerError
            ):
                PRODUCER.run_and_seal_probe(
                    spec, output_root=fixture.root / "runs" / "nonreuse"
                )


if __name__ == "__main__":
    unittest.main()
