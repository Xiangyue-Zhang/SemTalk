from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "show_base"
    / "select_base_official_adapt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_base_official_adapt_under_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)
FEATURE_SCRIPT = ROOT / "scripts" / "show_base" / "build_base_features.py"
FEATURE_SPEC = importlib.util.spec_from_file_location(
    "build_base_features_under_test",
    FEATURE_SCRIPT,
)
assert FEATURE_SPEC is not None and FEATURE_SPEC.loader is not None
FEATURES = importlib.util.module_from_spec(FEATURE_SPEC)
FEATURE_SPEC.loader.exec_module(FEATURES)


def _canonical_temporary_directory() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(
        dir=Path(tempfile.gettempdir()).resolve()
    )


def _write_json(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_payload_hash(value: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(value)
    result["receipt_payload_sha256"] = (
        SELECTOR.canonical_json_sha256(result)
    )
    return result


def _artifact(path: Path, *, payload_hash: str | None = None) -> dict[str, str]:
    result = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if payload_hash is not None:
        result["receipt_payload_sha256"] = payload_hash
    return result


class SelectionFixture:
    def __init__(
        self,
        root: Path,
        *,
        materialize_inference_outputs: bool = False,
    ) -> None:
        self.root = root
        self.materialize_inference_outputs = (
            materialize_inference_outputs
        )
        self.run = root / "adapt"
        self.run.mkdir()
        (self.run / "candidates").mkdir()
        self.candidate_paths: dict[int, Path] = {}
        self.candidate_sha: dict[int, str] = {}
        self._write_candidate_bundle()
        self._write_val_inputs()
        self._write_pipeline()
        self.measurement_paths: list[Path] = []
        self.measurement_hashes: list[str] = []
        self._write_measurements()

    def _write_candidate_bundle(self) -> None:
        frozen = {
            "format": SELECTOR.FROZEN_INPUTS_FORMAT,
            "source": {
                "origin": SELECTOR.BASE_PRODUCER_SOURCE["origin"],
                "commit": SELECTOR.BASE_PRODUCER_SOURCE["commit"],
                "tree": SELECTOR.BASE_PRODUCER_SOURCE["tree"],
                "branch": None,
                "clean": True,
                "entrypoint": "/source/train_base_official_adapt.py",
                "entrypoint_sha256": SELECTOR.BASE_PRODUCER_SOURCE[
                    "entrypoint_sha256"
                ],
            },
            "official_base": {
                "source": "released_all_speakers_v1",
                "path": "/official/best_semtalk_base.bin",
                "filename": "best_semtalk_base.bin",
                "sha256": SELECTOR.OFFICIAL_BASE_CHECKPOINT["sha256"],
                "speaker_scope": "All-Speakers",
                "all_model_state_tensors_finite": True,
                "strict_state_dict_load": True,
            },
            "speaker_initialization": {"speaker_rows": [0, 1, 2, 3]},
            "dataset": {
                "entries": 127_286,
                "train_clips": 13_687,
                "vq_models_in_training_graph": False,
            },
            "protocol": {
                "format": (
                    "semtalk_show_base_official_adapt_protocol_v1"
                ),
                "target_dataset": "SHOW",
                "target_speaker_scope": "All",
                "candidate_epochs": list(
                    SELECTOR.EXPECTED_CANDIDATE_EPOCHS
                ),
                "epochs": 40,
                "expected_updates_per_epoch": 248,
                "vq_models_in_training_graph": False,
                "precision": "bf16",
                "optimizer": {
                    "name": "Adam",
                    "learning_rate": 3e-5,
                },
                "throughput_gate": {
                    "warmup_updates": 20,
                    "timed_updates": 50,
                },
                "initialization": {
                    "forbidden_sources": ["e30", "Speaker2"],
                },
            },
        }
        frozen["receipt_sha256"] = SELECTOR.canonical_json_sha256(frozen)
        self.frozen_path = self.run / "frozen_inputs.json"
        self.frozen_sha = _write_json(self.frozen_path, frozen)
        gate_report = {
            "format": SELECTOR.THROUGHPUT_GATE_FORMAT,
            "status": "pass",
            "frozen_receipt_sha256": frozen["receipt_sha256"],
            "world_size": 8,
            "local_batch_size": 64,
            "global_batch_size": 512,
            "warmup_updates": 20,
            "timed_updates": 50,
            "precision": "bf16",
            "learning_rate": 3e-5,
            "elapsed_seconds": 10.0,
            "seconds_per_update": 0.2,
            "samples_per_second": 2_560.0,
            "estimated_40_epoch_training_seconds": 1_984.0,
            "last_metrics": {"loss": 1.0},
            "all_losses_finite": True,
            "optimizer_updates": 70,
            "peak_cuda_memory_bytes_rank0": 1,
            "completed_unix": 1.0,
        }
        gate_report["receipt_sha256"] = SELECTOR.canonical_json_sha256(
            gate_report
        )
        gate_path = self.run / "throughput_gate.json"
        gate_sha = _write_json(gate_path, gate_report)
        throughput_gate = {
            "path": str(gate_path.resolve()),
            "sha256": gate_sha,
            "samples_per_second": 2_560.0,
            "seconds_per_update": 0.2,
        }

        entries = []
        for epoch in SELECTOR.EXPECTED_CANDIDATE_EPOCHS:
            path = (
                self.run
                / "candidates"
                / f"base_official_adapt_epoch_{epoch:02d}.bin"
            )
            path.write_bytes(f"candidate-{epoch}".encode("ascii"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.candidate_paths[epoch] = path
            self.candidate_sha[epoch] = digest
            entries.append(
                {
                    "epoch": epoch,
                    "optimizer_updates": epoch * 248,
                    "checkpoint": (
                        "candidates/"
                        f"base_official_adapt_epoch_{epoch:02d}.bin"
                    ),
                    "checkpoint_sha256": digest,
                    "checkpoint_bytes": path.stat().st_size,
                    "checkpoint_container_schema": [
                        "audit",
                        "model_state",
                    ],
                    "model_state_tensors": 1,
                    "model_state_schema_sha256": "a" * 64,
                    "all_model_state_tensors_finite": True,
                    "frozen_receipt_sha256": frozen[
                        "receipt_sha256"
                    ],
                }
            )
        manifest = {
            "format": SELECTOR.CANDIDATE_MANIFEST_FORMAT,
            "status": "complete",
            "candidate_epochs": list(
                SELECTOR.EXPECTED_CANDIDATE_EPOCHS
            ),
            "frozen_receipt_sha256": frozen["receipt_sha256"],
            "throughput_gate": throughput_gate,
            "entries": entries,
            "entries_sha256": SELECTOR.canonical_json_sha256(entries),
            "completed_epochs": 40,
            "optimizer_updates": 40 * 248,
        }
        self.manifest_path = self.run / "candidate_manifest.json"
        self.manifest_sha = _write_json(self.manifest_path, manifest)
        status = {
            "format": SELECTOR.CANDIDATE_STATUS_FORMAT,
            "status": "complete",
            "completed_epochs": 40,
            "optimizer_updates": 40 * 248,
            "updates_per_epoch": 248,
            "candidate_manifest_sha256": self.manifest_sha,
            "frozen_receipt_sha256": frozen["receipt_sha256"],
            "throughput_gate": throughput_gate,
            "world_size": 8,
            "local_batch_size": 64,
            "global_batch_size": 512,
            "all_training_state_finite": True,
            "started_unix": 1.0,
            "completed_unix": 2.0,
        }
        self.status_path = self.run / "status.json"
        self.status_sha = _write_json(self.status_path, status)

    def _canonical_rows(self) -> list[dict[str, object]]:
        speakers = tuple(SELECTOR.SHOW_SPEAKER_IDS)
        return [
            {
                "global_index": index,
                "clip_id": (
                    f"{speakers[index % len(speakers)]}/"
                    f"video-{index}/sequence-{index}"
                ),
                "split": "val",
                "frames": 88 + (88 if index % 3 == 0 else 0),
            }
            for index in range(SELECTOR.EXPECTED_VAL_CLIPS)
        ]

    def _write_val_inputs(self) -> None:
        inputs_root = self.root / "val-inputs"
        inputs_root.mkdir()
        self.canonical_path = inputs_root / "canonical-val.jsonl"
        self.canonical_rows = self._canonical_rows()
        canonical_manifest_sha = _write_jsonl(
            self.canonical_path,
            self.canonical_rows,
        )
        canonical_summary = inputs_root / "canonical-summary.json"
        canonical_lineage = inputs_root / "canonical-lineage.json"
        self.canonical_summary_path = canonical_summary
        self.canonical_lineage_path = canonical_lineage
        lineage = _with_payload_hash(
            {
                "format": SELECTOR.VAL_CANONICAL_LINEAGE_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "clip_count": SELECTOR.EXPECTED_VAL_CLIPS,
                "manifest_sha256": canonical_manifest_sha,
                "lineage_contract_sha256": "9" * 64,
                "projection": {
                    "operation": "filter_exact_split",
                    "split": "val",
                    "test_rows_materialized": False,
                },
                "source_receipt": {
                    "origin": SELECTOR.BASE_PRODUCER_SOURCE["origin"],
                    "commit": "4" * 40,
                    "tree": "5" * 40,
                    "full_manifest_sha256": "6" * 64,
                    "full_summary_sha256": "7" * 64,
                    "full_lineage_sha256": "8" * 64,
                },
            }
        )
        lineage_sha = _write_json(canonical_lineage, lineage)
        self.canonical_lineage_sha = lineage_sha
        summary = _with_payload_hash(
            {
                "format": SELECTOR.VAL_CANONICAL_SUMMARY_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "clip_count": SELECTOR.EXPECTED_VAL_CLIPS,
                "manifest_sha256": canonical_manifest_sha,
                "lineage_sha256": lineage_sha,
                "lineage_contract_sha256": "9" * 64,
            }
        )
        self.canonical_summary_sha = _write_json(
            canonical_summary,
            summary,
        )
        canonical_ids, coverage = SELECTOR._canonical_coverage(
            self.canonical_path.read_bytes(),
            str(self.canonical_path),
        )
        assert len(canonical_ids) == SELECTOR.EXPECTED_VAL_CLIPS

        audio_manifests = []
        audio_summaries = []
        audio_lineages = []
        audio_root = inputs_root / "audio-features"
        audio_root.mkdir()
        audio_receipts: dict[str, tuple[Path, str]] = {}
        for row in self.canonical_rows:
            clip_id = str(row["clip_id"])
            audio_path = (
                audio_root
                / f"{hashlib.sha256(clip_id.encode('utf-8')).hexdigest()}.npz"
            )
            audio_path.write_bytes(f"audio:{clip_id}".encode("utf-8"))
            audio_receipts[clip_id] = (
                audio_path,
                hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            )
        for shard in range(SELECTOR.EXPECTED_AUDIO_SHARDS):
            rows = [
                {
                    "format": "semtalk_show_audio_clip_v1",
                    "split": "val",
                    "clip_id": row["clip_id"],
                    "shard_id": shard,
                    "num_shards": SELECTOR.EXPECTED_AUDIO_SHARDS,
                    "audio_feature_npz": str(
                        audio_receipts[str(row["clip_id"])][0].resolve()
                    ),
                    "audio_feature_npz_sha256": audio_receipts[
                        str(row["clip_id"])
                    ][1],
                    "frames": row["frames"],
                    "beat_shape": [row["frames"], 3],
                    "hubert_shape": [row["frames"], 1024],
                }
                for index, row in enumerate(self.canonical_rows)
                if index % SELECTOR.EXPECTED_AUDIO_SHARDS == shard
            ]
            manifest = inputs_root / f"audio-{shard}.jsonl"
            manifest_sha = _write_jsonl(manifest, rows)
            summary = inputs_root / f"audio-{shard}.summary.json"
            _write_json(
                summary,
                {
                    "format": "semtalk_show_audio_summary_v1",
                    "status": "complete",
                    "shard_id": shard,
                    "num_shards": SELECTOR.EXPECTED_AUDIO_SHARDS,
                    "full_split_clips": SELECTOR.EXPECTED_VAL_CLIPS,
                    "shard_clips": len(rows),
                    "output_manifest_sha256": manifest_sha,
                },
            )
            lineage = inputs_root / f"audio-{shard}.lineage.json"
            _write_json(
                lineage,
                {
                    "format": "semtalk_show_audio_lineage_v1",
                    "status": "complete",
                    "shard_id": shard,
                    "num_shards": SELECTOR.EXPECTED_AUDIO_SHARDS,
                    "full_split_clips": SELECTOR.EXPECTED_VAL_CLIPS,
                    "shard_clips": len(rows),
                    "output_manifest_sha256": manifest_sha,
                    "protocol": {"split": "val"},
                },
            )
            audio_manifests.append(_artifact(manifest))
            audio_summaries.append(_artifact(summary))
            audio_lineages.append(_artifact(lineage))

        val_inputs = _with_payload_hash(
            {
                "format": SELECTOR.VAL_INPUTS_FORMAT,
                "status": "frozen",
                "split": "val",
                "test_visible": False,
                "expected_clip_count": SELECTOR.EXPECTED_VAL_CLIPS,
                "canonical_manifest": _artifact(self.canonical_path),
                "canonical_summary": _artifact(canonical_summary),
                "canonical_lineage": _artifact(canonical_lineage),
                "audio_manifests": audio_manifests,
                "audio_summaries": audio_summaries,
                "audio_lineages": audio_lineages,
                "clip_ids_sha256": coverage["clip_ids_sha256"],
                "diffsheg_clip_manifest_sha256": coverage[
                    "diffsheg_clip_manifest_sha256"
                ],
            }
        )
        self.val_inputs_path = inputs_root / "val-inputs.json"
        self.val_inputs_sha = _write_json(
            self.val_inputs_path,
            val_inputs,
        )
        self.val_inputs_payload_sha = val_inputs[
            "receipt_payload_sha256"
        ]
        self.coverage = coverage

    def _write_pipeline(self) -> None:
        pinned_entrypoint = (
            self.root
            / "pinned-78412"
            / SELECTOR.VAL_INFERENCE_SOURCE["entrypoint"]
        )
        pinned_entrypoint.parent.mkdir()
        pinned_bytes = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                (
                    f"{SELECTOR.VAL_INFERENCE_SOURCE['commit']}:"
                    "scripts/show_base/run_base_inference.py"
                ),
            ],
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(pinned_bytes).hexdigest() != (
            SELECTOR.VAL_INFERENCE_SOURCE["entrypoint_sha256"]
        ):
            raise RuntimeError("pinned validation inference helper changed")
        pinned_entrypoint.write_bytes(pinned_bytes)
        pipeline = _with_payload_hash(
            {
                "format": SELECTOR.PIPELINE_FORMAT,
                "status": "frozen",
                "split": "val",
                "test_visible": False,
                "mode": "official_show_adapt_v1",
                "base_candidate_variable_only": True,
                "source": SELECTOR.VAL_INFERENCE_SOURCE,
                "inference_entrypoint": {
                    "path": str(pinned_entrypoint.resolve()),
                    "sha256": hashlib.sha256(pinned_bytes).hexdigest(),
                },
                "inference_helpers": list(SELECTOR.INFERENCE_HELPERS),
                "fixed_checkpoints": {
                    "face": {
                        "path": "/fixed/best_face_transfer.bin",
                        "sha256": "b" * 64,
                        "selection_split": "val",
                        "test_visible": False,
                    },
                    "global": {
                        "path": "/fixed/best_global_transfer.bin",
                        "sha256": "c" * 64,
                        "selection_split": "val",
                        "test_visible": False,
                    },
                    **{
                        stage: {
                            "path": (
                                "/official/"
                                f"{specification['filename']}"
                            ),
                            "sha256": specification["sha256"],
                            "source": "released_all_speakers_v1",
                        }
                        for stage, specification in (
                            SELECTOR.FIXED_OFFICIAL_CHECKPOINTS.items()
                        )
                    },
                },
                "diffsheg": SELECTOR.DIFFSHEG_PINNED_RECEIPT,
            }
        )
        self.pipeline_path = self.root / "pipeline.json"
        self.pipeline_sha = _write_json(self.pipeline_path, pipeline)
        self.pipeline_payload_sha = pipeline["receipt_payload_sha256"]

    def report(
        self,
        epoch: int,
        inference: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if inference is None:
            inference = self.inference_by_epoch[epoch]
        pins = SELECTOR.DIFFSHEG_PINNED_RECEIPT
        fgd = {
            1: 0.7,
            2: 0.6,
            4: 0.4,
            8: 0.3,
            16: 0.3,
            32: 0.5,
            40: 0.8,
        }[epoch]
        return {
            "status": "ok",
            "protocol": {
                "name": pins["protocol"],
                "version": pins["protocol_version"],
                "status": "reconstructed_from_public_components",
                "diffsheg_reference_commit": pins[
                    "diffsheg_reference_commit"
                ],
                "window_length": SELECTOR.DIFFSHEG_WINDOW,
                "window_stride": SELECTOR.DIFFSHEG_STRIDE,
                "precision": pins["precision"],
                "ba": None,
                "clip_order": (
                    f"explicit_manifest:{inference['clip_manifest_path']}"
                ),
                "selection_split": "val",
                "test_visible": False,
                "metric_scope": "fgd_only",
                "parameter_order": (
                    "PASPA/BEAT2 poses[165] -> DiffSHEG "
                    "ShowDataset.extract_pose gesture[129]"
                ),
                "normalization": "DiffSHEG talkshow_mean_std.npy",
                "tail_policy": "drop_incomplete_tail",
            },
            "inputs": {
                "prediction_dir": inference["prediction_dir"],
                "ground_truth_dir": inference["ground_truth_dir"],
                "clip_manifest": inference["clip_manifest_path"],
                "weights_dir": "/assets/ae_weights",
                "clip_count": SELECTOR.EXPECTED_VAL_CLIPS,
                "frame_count": self.coverage["frame_count"],
                "window_count": self.coverage["window_count"],
                "uncovered_tail_frames": self.coverage[
                    "uncovered_tail_frames"
                ],
                "clip_manifest_sha256": self.coverage[
                    "diffsheg_clip_manifest_sha256"
                ],
                "clip_manifest_file_sha256": inference[
                    "clip_manifest_sha256"
                ],
                "stats": {
                    "path": "/assets/talkshow_mean_std.npy",
                    "sha256": pins["stats_sha256"],
                },
                "checkpoint_paths": {
                    "fgd": "/assets/ae_weights/gesture.pth.tar",
                },
            },
            "metrics": {
                "fgd": fgd,
            },
            "diagnostics": {
                "gesture_feature_count": self.coverage["window_count"],
                "gesture_feature_dim": 300,
            },
            "provenance": {
                "evaluator": {
                    "path": "/evaluator/scripts/diffsheg_show_eval.py",
                    "repository_root": "/evaluator",
                    "sha256": pins["paspa"]["evaluator_sha256"],
                    "repository_git_head": pins["paspa"]["commit"],
                    "repository_git_tree": pins["paspa"]["tree"],
                    "repository_origin": pins["paspa"]["origin"],
                },
                "diffsheg_root": {
                    "path": "/diffsheg",
                    "git_head": pins["diffsheg_reference_commit"],
                },
                "autoencoders": {
                    name: {
                        "path": (
                            "/assets/ae_weights/"
                            f"{specification['filename']}"
                        ),
                        "sha256": specification["sha256"],
                        "input_dim": specification["input_dim"],
                        "latent_dim": specification["latent_dim"],
                        "state_container": specification[
                            "state_container"
                        ],
                        "load_mode": specification["load_mode"],
                        "feature_count": self.coverage["window_count"],
                    }
                    for name, specification in pins[
                        "autoencoders"
                    ].items()
                },
                "adapter": {
                    "path": (
                        "/semtalk/scripts/show_base/"
                        "evaluate_diffsheg_val_fgd.py"
                    ),
                    "sha256": "2" * 64,
                    "repository_root": "/semtalk",
                    "repository_git_head": "3" * 40,
                },
                "device": "cuda:0",
                "runtime_versions": {
                    "python": "3.11.0",
                    "numpy": "2.0.0",
                    "torch": "2.6.0",
                    "scipy": "1.15.0",
                },
            },
        }

    def _write_inference_lineage(
        self,
        metrics_root: Path,
        epoch: int,
    ) -> dict[str, object]:
        output_root = metrics_root / f"epoch-{epoch}-outputs"
        prediction_dir = output_root / "predictions" / "val"
        ground_truth_dir = output_root / "ground-truth" / "val"
        prediction_dir.mkdir(parents=True)
        ground_truth_dir.mkdir(parents=True)
        final_rows = []
        prediction_template = output_root / ".prediction-template.bin"
        ground_truth_template = output_root / ".ground-truth-template.bin"
        prediction_payload = f"prediction:{epoch}".encode("utf-8")
        ground_truth_payload = b"ground-truth"
        if self.materialize_inference_outputs:
            prediction_template.write_bytes(prediction_payload)
            ground_truth_template.write_bytes(ground_truth_payload)
        for canonical in self.coverage["_ordered_clips"]:
            output_id = str(canonical["canonical_clip_id"])
            prediction_path = (
                prediction_dir / f"res_{output_id}.npz"
            )
            ground_truth_path = (
                ground_truth_dir / f"gt_{output_id}.npz"
            )
            if self.materialize_inference_outputs:
                os.link(prediction_template, prediction_path)
                os.link(ground_truth_template, ground_truth_path)
            final_rows.append(
                {
                    "global_index": canonical["global_index"],
                    "split": "val",
                    "source_clip_id": canonical["source_clip_id"],
                    "canonical_clip_id": output_id,
                    "frames": canonical["frames"],
                    "epoch": epoch,
                    "candidate_checkpoint_sha256": self.candidate_sha[
                        epoch
                    ],
                    "prediction": {
                        "path": str(prediction_path.resolve()),
                        "sha256": hashlib.sha256(
                            prediction_payload
                        ).hexdigest(),
                        "bytes": len(prediction_payload),
                    },
                    "ground_truth": {
                        "path": str(ground_truth_path.resolve()),
                        "sha256": hashlib.sha256(
                            ground_truth_payload
                        ).hexdigest(),
                        "bytes": len(ground_truth_payload),
                    },
                }
            )
        final_manifest = output_root / "final_manifest.jsonl"
        _write_jsonl(final_manifest, final_rows)
        clip_manifest = output_root / "diffsheg_eval_clip_ids.txt"
        clip_manifest.write_text(
            "".join(
                f"{row['canonical_clip_id']}\n" for row in final_rows
            ),
            encoding="utf-8",
        )
        clip_manifest_sha = hashlib.sha256(
            clip_manifest.read_bytes()
        ).hexdigest()
        lineage = _with_payload_hash(
            {
                "format": SELECTOR.VAL_INFERENCE_LINEAGE_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "epoch": epoch,
                "candidate_checkpoint": {
                    "path": str(self.candidate_paths[epoch].resolve()),
                    "sha256": self.candidate_sha[epoch],
                },
                "val_inputs_receipt": _artifact(
                    self.val_inputs_path,
                    payload_hash=self.val_inputs_payload_sha,
                ),
                "pipeline_receipt": _artifact(
                    self.pipeline_path,
                    payload_hash=self.pipeline_payload_sha,
                ),
                "prediction_dir": str(prediction_dir.resolve()),
                "ground_truth_dir": str(ground_truth_dir.resolve()),
                "final_manifest": _artifact(final_manifest),
                "clip_manifest": _artifact(clip_manifest),
                "clip_count": SELECTOR.EXPECTED_VAL_CLIPS,
                "frame_count": self.coverage["frame_count"],
                "window_count": self.coverage["window_count"],
                "uncovered_tail_frames": self.coverage[
                    "uncovered_tail_frames"
                ],
                "clip_ids_sha256": self.coverage["clip_ids_sha256"],
                "diffsheg_clip_manifest_sha256": self.coverage[
                    "diffsheg_clip_manifest_sha256"
                ],
                "prediction_files": SELECTOR.EXPECTED_VAL_CLIPS,
                "ground_truth_files": SELECTOR.EXPECTED_VAL_CLIPS,
                "exact_once": True,
                "finite": True,
            }
        )
        lineage_path = output_root / "val-inference-lineage.json"
        _write_json(lineage_path, lineage)
        return {
            "lineage_path": lineage_path,
            "lineage_payload_sha256": lineage["receipt_payload_sha256"],
            "prediction_dir": str(prediction_dir.resolve()),
            "ground_truth_dir": str(ground_truth_dir.resolve()),
            "clip_manifest_path": str(clip_manifest.resolve()),
            "clip_manifest_sha256": clip_manifest_sha,
        }

    def _write_measurements(self) -> None:
        metrics_root = self.root / "measurements"
        metrics_root.mkdir()
        self.inference_by_epoch: dict[int, dict[str, object]] = {}
        for epoch in SELECTOR.EXPECTED_CANDIDATE_EPOCHS:
            inference = self._write_inference_lineage(
                metrics_root,
                epoch,
            )
            self.inference_by_epoch[epoch] = inference
            report_path = metrics_root / f"epoch-{epoch}.diffsheg.json"
            _write_json(report_path, self.report(epoch, inference))
            measurement = _with_payload_hash(
                {
                    "format": SELECTOR.MEASUREMENT_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "selection_eligible": True,
                    "epoch": epoch,
                    "candidate_checkpoint": {
                        "path": str(
                            self.candidate_paths[epoch].resolve()
                        ),
                        "sha256": self.candidate_sha[epoch],
                    },
                    "val_inputs_receipt": _artifact(
                        self.val_inputs_path,
                        payload_hash=self.val_inputs_payload_sha,
                    ),
                    "pipeline_receipt": _artifact(
                        self.pipeline_path,
                        payload_hash=self.pipeline_payload_sha,
                    ),
                    "inference_lineage": _artifact(
                        Path(inference["lineage_path"]),
                        payload_hash=str(
                            inference["lineage_payload_sha256"]
                        ),
                    ),
                    "diffsheg_report": _artifact(report_path),
                }
            )
            measurement_path = metrics_root / f"epoch-{epoch}.json"
            measurement_sha = _write_json(
                measurement_path,
                measurement,
            )
            self.measurement_paths.append(measurement_path)
            self.measurement_hashes.append(measurement_sha)

    def bundle(self) -> dict[str, object]:
        return SELECTOR.validate_candidate_bundle(
            manifest_path=self.manifest_path,
            expected_manifest_sha256=self.manifest_sha,
            status_path=self.status_path,
            expected_status_sha256=self.status_sha,
            frozen_inputs_path=self.frozen_path,
            expected_frozen_inputs_sha256=self.frozen_sha,
        )

    def argv(self, output: Path) -> list[str]:
        result = [
            "--base-candidate-manifest",
            str(self.manifest_path),
            "--expected-base-candidate-manifest-sha256",
            self.manifest_sha,
            "--base-status-json",
            str(self.status_path),
            "--expected-base-formal-status-sha256",
            self.status_sha,
            "--base-frozen-inputs-json",
            str(self.frozen_path),
            "--expected-base-frozen-inputs-sha256",
            self.frozen_sha,
        ]
        for path in self.measurement_paths:
            result.extend(["--measurement-json", str(path)])
        for digest in self.measurement_hashes:
            result.extend(["--expected-measurement-sha256", digest])
        result.extend(["--output-json", str(output)])
        return result


class BaseValSelectorStaticContracts(unittest.TestCase):
    def test_exact_candidate_epochs_and_pinned_diffsheg_assets(self) -> None:
        self.assertEqual(
            SELECTOR.EXPECTED_CANDIDATE_EPOCHS,
            (1, 2, 4, 8, 16, 32, 40),
        )
        self.assertEqual(SELECTOR.VAL_METRIC_KEYS, ("fgd",))
        self.assertEqual(SELECTOR.EXPECTED_VAL_CLIPS, 1_715)
        self.assertEqual(
            SELECTOR.DIFFSHEG_PINNED_RECEIPT["paspa"]["commit"],
            "0df27e6cab4b5ced19cc923afe352f77d547924b",
        )
        self.assertEqual(
            SELECTOR.DIFFSHEG_PINNED_RECEIPT["paspa"]["tree"],
            "574662eaf3122beb5631c02c847456e752d77f0b",
        )
        self.assertEqual(
            set(
                SELECTOR.DIFFSHEG_PINNED_RECEIPT[
                    "autoencoders"
                ]
            ),
            {"fgd"},
        )

    def test_selector_cli_has_no_split_or_test_input(self) -> None:
        parser = SELECTOR.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--split", option_strings)
        self.assertFalse(
            any("test" in option.casefold() for option in option_strings)
        )

    def test_e30_and_speaker2_are_hard_rejected(self) -> None:
        for value in (
            "/weights/e30/base.bin",
            "/weights/E_30/base.bin",
            "/weights/speaker2/base.bin",
            "SPEAKER-2-adaptation",
        ):
            with self.subTest(value=value):
                with self.assertRaises(
                    SELECTOR.SelectionContractError
                ):
                    SELECTOR.reject_forbidden_source_labels(value)

    def test_real_latest_clip_path_is_not_test_labeled(self) -> None:
        real_validation_path = Path(
            "/local-ssd/xiangyuezhang/"
            "semtalk_show_base_canonical_cache_aa8e519_20260729_v1/"
            "clips/val/conan/"
            "Conan_On_Trump_s_Latest_Portrait_Faux_Pas_-_CONAN_on_TBS-"
            "WIVVOCz9r50.webm/115448-00_03_58-00_04_08.npz"
        )
        SELECTOR.reject_test_path(
            real_validation_path,
            "validation prediction",
        )
        SELECTOR.reject_test_path(
            Path("/frozen/val/contest/Latest/results.npz"),
            "validation prediction",
        )
        for value in (
            "/frozen/val/testimonial/results.npz",
            "/frozen/val/testament/results.npz",
        ):
            with self.subTest(value=value):
                SELECTOR.reject_test_path(
                    Path(value),
                    "validation prediction",
                )

    def test_actual_test_path_labels_are_hard_rejected(self) -> None:
        for value in (
            "/frozen/test/clip.npz",
            "/frozen/tests/clip.npz",
            "/frozen/test_predictions/clip.npz",
            "/frozen/testset/clip.npz",
            "/frozen/testsets/clip.npz",
            "/frozen/actual-test/clip.npz",
            "/frozen/beat2_semtalk_test.pkl",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    SELECTOR.SelectionContractError,
                    "test-labeled",
                ):
                    SELECTOR.reject_test_path(
                        Path(value),
                        "validation prediction",
                    )

    def test_parent_symlink_cannot_hide_resolved_test_artifacts(self) -> None:
        with _canonical_temporary_directory() as temporary:
            root = Path(temporary)
            actual_parent = root / "actual-test"
            actual_val = actual_parent / "val"
            actual_val.mkdir(parents=True)
            output = actual_val / "res_clip.npz"
            output.write_bytes(b"output")
            safe_parent = root / "safe-parent"
            safe_parent.symlink_to(actual_parent, target_is_directory=True)
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "canonical with no symlink ancestor",
            ):
                SELECTOR.require_directory(
                    str(safe_parent / "val"),
                    "validation output",
                )
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "canonical with no symlink ancestor",
            ):
                SELECTOR._validate_output_file_receipt(
                    {
                        "path": str(safe_parent / "val" / output.name),
                        "sha256": hashlib.sha256(
                            output.read_bytes()
                        ).hexdigest(),
                        "bytes": output.stat().st_size,
                    },
                    expected_directory=actual_val.resolve(),
                    expected_filename=output.name,
                    label="validation prediction",
                )

    def test_minimum_fgd_tie_breaks_to_earliest_epoch(self) -> None:
        rows = [
            {"epoch": epoch, "metrics": {"fgd": 0.1}}
            for epoch in SELECTOR.EXPECTED_CANDIDATE_EPOCHS
        ]
        selected = SELECTOR.select_minimum_fgd(rows)
        self.assertEqual(selected["epoch"], 1)
        rows[0]["metrics"]["fgd"] = 0.2
        selected = SELECTOR.select_minimum_fgd(rows)
        self.assertEqual(selected["epoch"], 2)

    def test_selection_rows_reject_non_fgd_metrics(self) -> None:
        rows = [
            {"epoch": epoch, "metrics": {"fgd": 0.1}}
            for epoch in SELECTOR.EXPECTED_CANDIDATE_EPOCHS
        ]
        rows[0]["metrics"]["fmd"] = 1.0
        with self.assertRaisesRegex(
            SELECTOR.SelectionContractError,
            "selection row metrics schema mismatch",
        ):
            SELECTOR.select_minimum_fgd(rows)

    def test_e30_is_not_a_candidate_epoch(self) -> None:
        rows = [
            {"epoch": epoch, "metrics": {"fgd": 1.0}}
            for epoch in (1, 2, 4, 8, 16, 30, 40)
        ]
        with self.assertRaisesRegex(
            SELECTOR.SelectionContractError,
            "exactly cover epochs",
        ):
            SELECTOR.select_minimum_fgd(rows)

    def test_audio_builder_adds_val_without_removing_train_or_test(self) -> None:
        source = (
            ROOT / "scripts/show_base/build_base_features.py"
        ).read_text(encoding="utf-8")
        launcher = (
            ROOT / "scripts/show_base/run_audio_cache_8shard.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"train": 13_687', source)
        self.assertIn('"val": 1_715', source)
        self.assertIn('"test": 1_708', source)
        self.assertIn('choices=("train", "val", "test")', source)
        self.assertIn("train) formal_expected_clips=13687", launcher)
        self.assertIn("val) formal_expected_clips=1715", launcher)
        self.assertIn("test) formal_expected_clips=1708", launcher)

    def test_audio_launcher_rejects_unguarded_direct_invocation(self) -> None:
        launcher = ROOT / "scripts/show_base/run_audio_cache_8shard.sh"
        arguments = [
            "/missing/repo",
            "/missing/python",
            "/missing/manifest",
            "/missing/summary",
            "/missing/lineage",
            "/missing/hubert",
            "a" * 64,
            "/missing/output",
            "val",
            "1708",
            "1" * 40,
            "2" * 40,
            "3" * 40,
            "4" * 40,
            "b" * 64,
            "c" * 64,
            "d" * 64,
        ]
        result = subprocess.run(
            ["/bin/bash", str(launcher), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertRegex(
            result.stderr,
            (
                "guarded runner process is unavailable|"
                "requires direct parent /tmp/globaldiff_guarded_runner.py"
            ),
        )
        launcher_source = launcher.read_text(encoding="utf-8")
        self.assertLess(
            launcher_source.index(
                "semtalk_require_exact_guarded_runner_all_gpus"
            ),
            launcher_source.index("formal_expected_clips=1715"),
        )

    def test_audio_builder_accepts_only_authenticated_val_view_receipt(
        self,
    ) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            manifest_sha = hashlib.sha256(
                fixture.canonical_path.read_bytes()
            ).hexdigest()
            receipt = FEATURES.load_canonical_receipt(
                manifest_paths=[fixture.canonical_path],
                manifest_hashes={
                    str(fixture.canonical_path.resolve()): manifest_sha
                },
                summary_path=fixture.canonical_summary_path,
                lineage_path=fixture.canonical_lineage_path,
                expected_manifest_sha256=manifest_sha,
                expected_summary_sha256=fixture.canonical_summary_sha,
                expected_lineage_sha256=fixture.canonical_lineage_sha,
                expected_canonical_source_commit="4" * 40,
                expected_canonical_source_tree="5" * 40,
                split="val",
            )
            self.assertEqual(receipt["split"], "val")
            self.assertFalse(receipt["test_visible"])

            mixed = Path(temporary) / "mixed.jsonl"
            mixed_rows = [
                {"split": "val"} for _ in range(SELECTOR.EXPECTED_VAL_CLIPS)
            ]
            mixed_rows.append({"split": "test"})
            mixed_sha = _write_jsonl(mixed, mixed_rows)
            with self.assertRaisesRegex(RuntimeError, "val-only"):
                FEATURES.canonical_split_rows(
                    [mixed],
                    "val",
                    expected_manifest_sha256=mixed_sha,
                )


class BaseValSelectorReceiptContracts(unittest.TestCase):
    def test_complete_fixture_selects_minimum_fgd_with_earliest_tie(
        self,
    ) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(
                Path(temporary),
                materialize_inference_outputs=True,
            )
            selection = SELECTOR.build_selection(
                candidate_bundle=fixture.bundle(),
                measurement_paths=fixture.measurement_paths,
                expected_measurement_sha256=fixture.measurement_hashes,
            )
            first_lineage = json.loads(
                Path(
                    json.loads(
                        fixture.measurement_paths[0].read_text(
                            encoding="utf-8"
                        )
                    )["inference_lineage"]["path"]
                ).read_text(encoding="utf-8")
            )
            first_output = json.loads(
                Path(
                    first_lineage["final_manifest"]["path"]
                ).read_text(encoding="utf-8").splitlines()[0]
            )["prediction"]["path"]
            Path(first_output).write_bytes(b"tampered-output")
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "bound output artifact",
            ):
                SELECTOR.build_selection(
                    candidate_bundle=fixture.bundle(),
                    measurement_paths=fixture.measurement_paths,
                    expected_measurement_sha256=fixture.measurement_hashes,
                )
        self.assertEqual(selection["selected"]["epoch"], 8)
        self.assertEqual(selection["selected"]["fgd"], 0.3)
        self.assertEqual(selection["split"], "val")
        self.assertFalse(selection["test_visible"])
        self.assertFalse(
            selection["selection_policy"][
                "test_feedback_into_selection"
            ]
        )
        self.assertEqual(
            selection["test_policy"]["authorized_evaluations"],
            1,
        )
        self.assertEqual(
            selection["receipt_payload_sha256"],
            SELECTOR.canonical_json_sha256(
                {
                    key: value
                    for key, value in selection.items()
                    if key != "receipt_payload_sha256"
                }
            ),
        )

    def test_main_atomically_creates_and_never_overwrites_selection(
        self,
    ) -> None:
        with _canonical_temporary_directory() as temporary:
            root = Path(temporary)
            fixture = SelectionFixture(
                root,
                materialize_inference_outputs=True,
            )
            output = root / "selected.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(SELECTOR.main(fixture.argv(output)), 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["selected"]["epoch"], 8)
            with (
                redirect_stdout(io.StringIO()),
                self.assertRaises(SELECTOR.SelectionContractError),
            ):
                SELECTOR.main(fixture.argv(output))

    def test_main_rejects_e30_labeled_output_before_publication(self) -> None:
        with _canonical_temporary_directory() as temporary:
            root = Path(temporary)
            fixture = SelectionFixture(root)
            output = root / "e30" / "selected.json"
            with self.assertRaises(
                SELECTOR.SelectionContractError
            ):
                SELECTOR.main(fixture.argv(output))
            self.assertFalse(output.exists())

    def test_wrong_candidate_epoch_order_is_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(
                Path(temporary),
                materialize_inference_outputs=True,
            )
            paths = list(fixture.measurement_paths)
            hashes = list(fixture.measurement_hashes)
            paths[0], paths[1] = paths[1], paths[0]
            hashes[0], hashes[1] = hashes[1], hashes[0]
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "exactly cover epochs",
            ):
                SELECTOR.build_selection(
                    candidate_bundle=fixture.bundle(),
                    measurement_paths=paths,
                    expected_measurement_sha256=hashes,
                )

    def test_unlisted_epoch_30_candidate_file_is_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            (
                fixture.run
                / "candidates"
                / "base_official_adapt_epoch_30.bin"
            ).write_bytes(b"withdrawn")
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "exact seven-epoch cover",
            ):
                fixture.bundle()

    def test_boolean_candidate_epoch_is_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            manifest = json.loads(
                fixture.manifest_path.read_text(encoding="utf-8")
            )
            manifest["candidate_epochs"][0] = True
            manifest["entries"][0]["epoch"] = True
            manifest["entries_sha256"] = SELECTOR.canonical_json_sha256(
                manifest["entries"]
            )
            fixture.manifest_sha = _write_json(
                fixture.manifest_path,
                manifest,
            )
            status = json.loads(
                fixture.status_path.read_text(encoding="utf-8")
            )
            status["candidate_manifest_sha256"] = fixture.manifest_sha
            fixture.status_sha = _write_json(fixture.status_path, status)
            with self.assertRaises(SELECTOR.SelectionContractError):
                fixture.bundle()

    def test_frozen_source_with_e30_path_is_hard_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            frozen = json.loads(
                fixture.frozen_path.read_text(encoding="utf-8")
            )
            frozen["source"]["entrypoint"] = (
                "/withdrawn/e30/train_base_official_adapt.py"
            )
            frozen.pop("receipt_sha256")
            frozen["receipt_sha256"] = SELECTOR.canonical_json_sha256(
                frozen
            )
            with self.assertRaises(SELECTOR.SelectionContractError):
                SELECTOR._validate_frozen_inputs(frozen)

    def test_nonfinite_fgd_and_inexact_coverage_are_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            report = fixture.report(1)
            report["metrics"]["fgd"] = float("inf")
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "finite",
            ):
                SELECTOR.validate_diffsheg_report(
                    report,
                    expected_coverage=fixture.coverage,
                )
            for key in (
                "frame_count",
                "window_count",
                "uncovered_tail_frames",
            ):
                report = fixture.report(1)
                report["inputs"][key] += 1
                with self.subTest(key=key), self.assertRaisesRegex(
                    SELECTOR.SelectionContractError,
                    "exactly cover",
                ):
                    SELECTOR.validate_diffsheg_report(
                        report,
                        expected_coverage=fixture.coverage,
                    )
            report = fixture.report(1)
            report["inputs"]["clip_count"] = 1_708
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "exactly cover",
            ):
                SELECTOR.validate_diffsheg_report(
                    report,
                    expected_coverage=fixture.coverage,
                )

    def test_validation_report_requires_exact_fgd_only_metrics(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            report = fixture.report(1)
            self.assertEqual(set(report["metrics"]), {"fgd"})
            metrics, _ = SELECTOR.validate_diffsheg_report(
                report,
                expected_coverage=fixture.coverage,
            )
            self.assertEqual(metrics, {"fgd": 0.7})

            for extra_key in (
                "fmd",
                "fed",
                "expression_diversity",
                "pcm",
                "gesture_diversity",
                "BA",
            ):
                with self.subTest(extra_key=extra_key):
                    non_fgd_only = fixture.report(1)
                    non_fgd_only["metrics"][extra_key] = 0.0
                    with self.assertRaisesRegex(
                        SELECTOR.SelectionContractError,
                        "exactly FGD-only",
                    ):
                        SELECTOR.validate_diffsheg_report(
                            non_fgd_only,
                            expected_coverage=fixture.coverage,
                        )

    def test_validation_report_adapter_is_bound_to_fresh_pipeline_source(
        self,
    ) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            report = fixture.report(1)
            adapter = report["provenance"]["adapter"]
            relative = (
                "scripts/show_base/evaluate_diffsheg_val_fgd.py"
            )
            pipeline = {
                "source": {
                    "origin": (
                        "git@github.com:Xiangyue-Zhang/SemTalk.git"
                    ),
                    "source_root": adapter["repository_root"],
                    "commit": adapter["repository_git_head"],
                    "tree": "4" * 40,
                    "clean": True,
                    "detached": True,
                    "local_branches_at_commit": [],
                },
                "source_closure": {
                    relative: {
                        "path": adapter["path"],
                        "sha256": adapter["sha256"],
                        "bytes": 1,
                        "git_mode": "100644",
                        "git_blob_sha1": "5" * 40,
                    }
                },
            }
            SELECTOR.validate_diffsheg_report(
                report,
                expected_coverage=fixture.coverage,
                expected_pipeline=pipeline,
            )
            for field, changed in (
                ("sha256", "0" * 64),
                ("repository_git_head", "1" * 40),
                ("path", "/semtalk/scripts/show_base/forged.py"),
                ("repository_root", "/another/semtalk"),
            ):
                forged = fixture.report(1)
                forged["provenance"]["adapter"][field] = changed
                with self.subTest(field=field), self.assertRaisesRegex(
                    SELECTOR.SelectionContractError,
                    "frozen pipeline source",
                ):
                    SELECTOR.validate_diffsheg_report(
                        forged,
                        expected_coverage=fixture.coverage,
                        expected_pipeline=pipeline,
                    )

    def test_validation_report_protocol_is_strict_val_only(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            for field, changed in (
                ("selection_split", "test"),
                ("test_visible", True),
                ("metric_scope", "all_metrics"),
            ):
                report = fixture.report(1)
                report["protocol"][field] = changed
                with self.subTest(field=field), self.assertRaisesRegex(
                    SELECTOR.SelectionContractError,
                    "protocol mismatch",
                ):
                    SELECTOR.validate_diffsheg_report(
                        report,
                        expected_coverage=fixture.coverage,
                    )

    def test_validation_report_rejects_unknown_test_fields(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            mutations = (
                ("top", "test_metrics", True),
                ("protocol", "test_metric_used", True),
                (
                    "inputs",
                    "test_ground_truth_dir",
                    "/secret/test/ground-truth",
                ),
                ("provenance", "test_source", "/secret/test"),
            )
            for section, field, value in mutations:
                report = fixture.report(1)
                if section == "top":
                    report[field] = value
                else:
                    report[section][field] = value
                with self.subTest(section=section, field=field):
                    with self.assertRaisesRegex(
                        SELECTOR.SelectionContractError,
                        "schema mismatch",
                    ):
                        SELECTOR.validate_diffsheg_report(
                            report,
                            expected_coverage=fixture.coverage,
                        )

    def test_pinned_evaluator_or_autoencoder_mismatch_is_rejected(
        self,
    ) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            report = fixture.report(1)
            report["provenance"]["evaluator"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "evaluator provenance",
            ):
                SELECTOR.validate_diffsheg_report(
                    report,
                    expected_coverage=fixture.coverage,
                )
            report = fixture.report(1)
            report["provenance"]["autoencoders"]["fgd"][
                "sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "fgd autoencoder",
            ):
                SELECTOR.validate_diffsheg_report(
                    report,
                    expected_coverage=fixture.coverage,
                )
            for field, value in (
                ("state_container", "model_state"),
                ("load_mode", "encoder_only"),
                ("latent_dim", 299),
            ):
                report = fixture.report(1)
                report["provenance"]["autoencoders"]["fgd"][field] = value
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        SELECTOR.SelectionContractError,
                        "fgd autoencoder",
                    ):
                        SELECTOR.validate_diffsheg_report(
                            report,
                            expected_coverage=fixture.coverage,
                        )
            report = fixture.report(1)
            report["provenance"]["autoencoders"]["fmd"] = {
                "path": "/assets/gesture_expression.pth.tar",
                "sha256": "1" * 64,
                "input_dim": 232,
            }
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "evaluator provenance",
            ):
                SELECTOR.validate_diffsheg_report(
                    report,
                    expected_coverage=fixture.coverage,
                )

    def test_diffsheg_report_rejects_e30_labeled_input_path(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            report = fixture.report(1)
            report["inputs"]["prediction_dir"] = "/withdrawn/e30/val"
            with self.assertRaises(
                SELECTOR.SelectionContractError
            ):
                SELECTOR.validate_diffsheg_report(
                    report,
                    expected_coverage=fixture.coverage,
                )
            report = fixture.report(1)
            report["inputs"]["prediction_dir"] = "/frozen/test/val"
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "test-labeled",
            ):
                SELECTOR.validate_diffsheg_report(
                    report,
                    expected_coverage=fixture.coverage,
                )

    def test_candidate_report_cannot_be_reused_for_another_epoch(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(
                Path(temporary),
                materialize_inference_outputs=True,
            )
            first_report = Path(
                json.loads(
                    fixture.measurement_paths[0].read_text(
                        encoding="utf-8"
                    )
                )["diffsheg_report"]["path"]
            )
            second_measurement_path = fixture.measurement_paths[1]
            second = json.loads(
                second_measurement_path.read_text(encoding="utf-8")
            )
            second["diffsheg_report"] = _artifact(first_report)
            second.pop("receipt_payload_sha256")
            second = _with_payload_hash(second)
            fixture.measurement_hashes[1] = _write_json(
                second_measurement_path,
                second,
            )
            with self.assertRaises(SELECTOR.SelectionContractError):
                SELECTOR.build_selection(
                    candidate_bundle=fixture.bundle(),
                    measurement_paths=fixture.measurement_paths,
                    expected_measurement_sha256=fixture.measurement_hashes,
                )

    def test_inference_lineage_must_bind_its_candidate_sha_and_epoch(
        self,
    ) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            measurement_path = fixture.measurement_paths[0]
            measurement = json.loads(
                measurement_path.read_text(encoding="utf-8")
            )
            lineage_path = Path(measurement["inference_lineage"]["path"])
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            lineage["epoch"] = 2
            lineage["candidate_checkpoint"] = {
                "path": str(fixture.candidate_paths[2].resolve()),
                "sha256": fixture.candidate_sha[2],
            }
            lineage.pop("receipt_payload_sha256")
            lineage = _with_payload_hash(lineage)
            _write_json(lineage_path, lineage)
            measurement["inference_lineage"] = _artifact(
                lineage_path,
                payload_hash=str(lineage["receipt_payload_sha256"]),
            )
            measurement.pop("receipt_payload_sha256")
            measurement = _with_payload_hash(measurement)
            fixture.measurement_hashes[0] = _write_json(
                measurement_path,
                measurement,
            )
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "candidate-bound",
            ):
                SELECTOR.build_selection(
                    candidate_bundle=fixture.bundle(),
                    measurement_paths=fixture.measurement_paths,
                    expected_measurement_sha256=fixture.measurement_hashes,
                )

    def test_canonical_output_id_collision_is_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            rows = copy.deepcopy(fixture.canonical_rows)
            first_sequence = str(rows[0]["clip_id"]).split("/")[-1]
            rows[4]["clip_id"] = f"oliver/other-video/{first_sequence}"
            payload = "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ).encode("utf-8")
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "output-ID collision",
            ):
                SELECTOR._canonical_coverage(payload, "collision fixture")

    def test_audio_feature_file_hash_is_reopened(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            val_inputs = json.loads(
                fixture.val_inputs_path.read_text(encoding="utf-8")
            )
            manifest_path = Path(val_inputs["audio_manifests"][0]["path"])
            first_row = json.loads(
                manifest_path.read_text(encoding="utf-8").splitlines()[0]
            )
            Path(first_row["audio_feature_npz"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "audio feature receipt mismatch",
            ):
                SELECTOR.validate_val_inputs(
                    fixture.val_inputs_path,
                    fixture.val_inputs_sha,
                )

    def test_any_canonical_test_row_is_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            rows = list(fixture.canonical_rows)
            rows[0] = {**rows[0], "split": "test"}
            _write_jsonl(fixture.canonical_path, rows)
            val_inputs = json.loads(
                fixture.val_inputs_path.read_text(encoding="utf-8")
            )
            val_inputs["canonical_manifest"] = _artifact(
                fixture.canonical_path
            )
            val_inputs.pop("receipt_payload_sha256")
            val_inputs = _with_payload_hash(val_inputs)
            val_sha = _write_json(fixture.val_inputs_path, val_inputs)
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "val rows only",
            ):
                SELECTOR.validate_val_inputs(
                    fixture.val_inputs_path,
                    val_sha,
                )

    def test_any_audio_test_row_is_rejected(self) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            val_inputs = json.loads(
                fixture.val_inputs_path.read_text(encoding="utf-8")
            )
            first = Path(val_inputs["audio_manifests"][0]["path"])
            rows = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["split"] = "test"
            _write_jsonl(first, rows)
            val_inputs["audio_manifests"][0] = _artifact(first)
            val_inputs.pop("receipt_payload_sha256")
            val_inputs = _with_payload_hash(val_inputs)
            val_sha = _write_json(fixture.val_inputs_path, val_inputs)
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "invalid validation audio row",
            ):
                SELECTOR.validate_val_inputs(
                    fixture.val_inputs_path,
                    val_sha,
                )

    def test_test_visible_measurement_is_rejected_before_selection(
        self,
    ) -> None:
        with _canonical_temporary_directory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            first = fixture.measurement_paths[0]
            measurement = json.loads(first.read_text(encoding="utf-8"))
            measurement["test_visible"] = True
            measurement.pop("receipt_payload_sha256")
            measurement = _with_payload_hash(measurement)
            fixture.measurement_hashes[0] = _write_json(
                first,
                measurement,
            )
            with self.assertRaisesRegex(
                SELECTOR.SelectionContractError,
                "validation-only",
            ):
                SELECTOR.build_selection(
                    candidate_bundle=fixture.bundle(),
                    measurement_paths=fixture.measurement_paths,
                    expected_measurement_sha256=(
                        fixture.measurement_hashes
                    ),
                )

    def test_strict_json_rejects_nonfinite_constants(self) -> None:
        with self.assertRaises(SELECTOR.SelectionContractError):
            SELECTOR._strict_json_bytes(
                b'{"fgd": NaN}',
                "fixture",
            )


if __name__ == "__main__":
    unittest.main()
