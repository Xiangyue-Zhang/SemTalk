from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.show_base import base_fresh_val_orchestrator as ORCHESTRATOR
from scripts.show_base import published_test_winner_claim as AUTHORITY
from tests.test_published_test_winner_claim_cpu import ClaimFixture

with mock.patch.dict(sys.modules, {"numpy": mock.MagicMock()}):
    from tests.test_select_base_official_adapt_cpu import (
        SELECTOR as LEGACY_VAL_CONTRACT,
        SelectionFixture as LegacySelectionFixture,
    )


def _write_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _write_json(path: Path, value: object) -> dict[str, object]:
    return _write_bytes(
        path,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_receipt(path: Path, value: dict[str, object]) -> dict[str, object]:
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


class AuditedFrozenValFixture(LegacySelectionFixture):
    """Materialize the real audited 1,715-clip val contract for fresh tests."""

    def _canonical_rows(self) -> list[dict[str, object]]:
        speakers = tuple(LEGACY_VAL_CONTRACT.SHOW_SPEAKER_IDS)
        return [
            {
                "global_index": (
                    AUTHORITY.EXPECTED_VAL_GLOBAL_INDEX_START + index
                ),
                "clip_id": (
                    f"{speakers[index % len(speakers)]}/"
                    f"video-{index}/sequence-{index}"
                ),
                "split": "val",
                "frames": 88 + (88 if index % 3 == 0 else 0),
            }
            for index in range(LEGACY_VAL_CONTRACT.EXPECTED_VAL_CLIPS)
        ]


class CandidateTransactionFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.authority = ClaimFixture(root / "authority")
        self.epoch = AUTHORITY.BASE_CANDIDATE_EPOCHS[0]
        self.updates = self.epoch * AUTHORITY.BASE_UPDATES_PER_EPOCH
        self.checkpoint = _write_bytes(
            root / "base-e1.pth", b"fresh semtalk base checkpoint"
        )
        audited_val_root = root / "audited-val"
        audited_val_root.mkdir()
        self.frozen_val = AuditedFrozenValFixture(audited_val_root)
        self.val_inputs = {
            "path": str(self.frozen_val.val_inputs_path.resolve()),
            "sha256": self.frozen_val.val_inputs_sha,
            "bytes": self.frozen_val.val_inputs_path.stat().st_size,
            "receipt_payload_sha256": (
                self.frozen_val.val_inputs_payload_sha
            ),
        }
        selected_by_stage = {
            row["stage"]: row
            for row in self.authority.prerequisite_payload["stages"]
        }
        self.fixed_pipeline_checkpoints = {}
        for stage in AUTHORITY.STAGES:
            selected = selected_by_stage[stage]
            checkpoint = self.authority.fixed_checkpoints[stage]
            self.fixed_pipeline_checkpoints[stage] = {
                "stage": stage,
                **checkpoint,
                "source": "show_val_selected_v1",
                "selection_split": "val",
                "test_visible": False,
                "epoch": selected["epoch"],
                "optimizer_updates": selected["optimizer_updates"],
                "updates_per_epoch": (
                    AUTHORITY.PREREQUISITE_UPDATES_PER_EPOCH_BY_STAGE[
                        stage
                    ]
                ),
                "candidate_audit_sha256": hashlib.sha256(
                    f"{stage}-candidate-audit".encode("ascii")
                ).hexdigest(),
                "selection_metric": selected["selection_metric"],
                "measurement_receipt": copy.deepcopy(
                    selected["measurement_receipt"]
                ),
            }
        pipeline_source_root = (root / "fresh-source").resolve()
        self.pipeline_payload = {
            "format": "semtalk_show_base_fresh_val_pipeline_v2",
            "status": "frozen",
            "split": "val",
            "test_visible": False,
            "mode": "show_val_selected_five_prerequisites_v2",
            "base_candidate_variable_only": True,
            "source": {
                "origin": AUTHORITY.EXPECTED_ORIGIN,
                "source_root": str(pipeline_source_root),
                "commit": "1" * 40,
                "tree": "2" * 40,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            },
            "source_closure": {},
            "inference_entrypoint": {
                "path": str(
                    pipeline_source_root
                    / "scripts/show_base/run_base_val_inference.py"
                ),
                "sha256": "3" * 64,
                "bytes": 1,
                "git_mode": "100644",
                "git_blob_sha1": "4" * 40,
            },
            "inference_helper": {
                "path": str(
                    pipeline_source_root
                    / "scripts/show_base/run_base_inference.py"
                ),
                "sha256": "5" * 64,
                "bytes": 1,
                "git_mode": "100644",
                "git_blob_sha1": "6" * 40,
            },
            "generator_module": "models.semtalk.semtalk_base",
            "prerequisite_consumption": copy.deepcopy(
                AUTHORITY.FRESH_PREREQUISITE_CONSUMPTION
            ),
            "prerequisite_selection": copy.deepcopy(
                self.authority.prerequisite_artifact
            ),
            "fixed_checkpoints": copy.deepcopy(
                self.fixed_pipeline_checkpoints
            ),
        }
        self.pipeline = _write_receipt(
            root / "fresh-pipeline.json", self.pipeline_payload
        )
        self.preflight = _write_receipt(
            root / "preflight.json",
            {
                "format": AUTHORITY.FRESH_VAL_PREFLIGHT_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "candidate_epochs": list(AUTHORITY.BASE_CANDIDATE_EPOCHS),
                "candidate_bundle": {
                    "updates_per_epoch": AUTHORITY.BASE_UPDATES_PER_EPOCH,
                    "selected_topology": {
                        "mode": (
                            "validation_gated_w8_l64_g512_empirical_acceleration"
                        ),
                        "updates_per_epoch": AUTHORITY.BASE_UPDATES_PER_EPOCH,
                    },
                    "producer_source": {
                        "origin": AUTHORITY.EXPECTED_ORIGIN,
                        "commit": "1" * 40,
                        "tree": "2" * 40,
                        "clean": True,
                        "detached": True,
                        "local_branches_at_commit": [],
                    },
                    "frozen_inputs": {
                        "receipt_sha256": "f" * 64,
                    },
                    "candidates": {str(self.epoch): self.checkpoint},
                },
                "val_inputs_receipt": {
                    key: self.val_inputs[key]
                    for key in (
                        "path",
                        "sha256",
                        "receipt_payload_sha256",
                    )
                },
                "pipeline_receipt": {
                    key: self.pipeline[key]
                    for key in (
                        "path",
                        "sha256",
                        "receipt_payload_sha256",
                    )
                },
                "coverage": {
                    "clip_count": AUTHORITY.EXPECTED_VAL_CLIPS
                },
            },
        )
        self.final_manifest = self._final_manifest()
        self.lineage = _write_receipt(
            root / "val-inference-lineage.json",
            {
                "format": AUTHORITY.FORMAL_VAL_LINEAGE_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "epoch": self.epoch,
                "candidate_checkpoint": {
                    "path": self.checkpoint["path"],
                    "sha256": self.checkpoint["sha256"],
                },
                "val_inputs_receipt": {
                    key: self.val_inputs[key]
                    for key in (
                        "path",
                        "sha256",
                        "receipt_payload_sha256",
                    )
                },
                "pipeline_receipt": {
                    key: self.pipeline[key]
                    for key in (
                        "path",
                        "sha256",
                        "receipt_payload_sha256",
                    )
                },
                "prediction_dir": str((root / "predictions").resolve()),
                "ground_truth_dir": str((root / "ground-truth").resolve()),
                "final_manifest": {
                    "path": self.final_manifest["path"],
                    "sha256": self.final_manifest["sha256"],
                },
                "clip_manifest": {
                    "path": str((root / "clip-ids.txt").resolve()),
                    "sha256": "3" * 64,
                },
                "clip_count": AUTHORITY.EXPECTED_VAL_CLIPS,
                "frame_count": self.frozen_val.coverage["frame_count"],
                "window_count": self.frozen_val.coverage["window_count"],
                "uncovered_tail_frames": self.frozen_val.coverage[
                    "uncovered_tail_frames"
                ],
                "clip_ids_sha256": self.frozen_val.coverage[
                    "clip_ids_sha256"
                ],
                "talkshow_window_manifest_sha256": self.frozen_val.coverage[
                    "diffsheg_clip_manifest_sha256"
                ],
                "prediction_files": AUTHORITY.EXPECTED_VAL_CLIPS,
                "ground_truth_files": AUTHORITY.EXPECTED_VAL_CLIPS,
                "exact_once": True,
                "finite": True,
            },
        )
        self.gate = _write_receipt(
            root / "gate.json",
            {
                "format": "semtalk_show_deterministic_replication_gate_v2",
                "payload_hash_algorithm": AUTHORITY.PAYLOAD_HASH_ALGORITHM,
                "status": "pass",
                "split": "val",
                "test_visible": False,
                "scope": "validation_candidate_family",
                "coverage_mode": "full_frozen_val_1715",
                "gate_protocol": (
                    "deterministic_replication_of_single_prediction_v1"
                ),
                "seeds": [0, 15],
                "seed_runs": [],
                "subset_manifest": {},
                "source_closure": {
                    "origin": AUTHORITY.EXPECTED_ORIGIN,
                },
                "model_bundle": {
                    "checkpoints": {"base": self.checkpoint},
                },
                "proof": {"clip_count": AUTHORITY.EXPECTED_VAL_CLIPS},
                "replication_authorization": {},
            },
        )
        self.distribution = _write_receipt(
            root / "distribution.json",
            {
                "format": AUTHORITY.DISTRIBUTION_FORMAT,
                "payload_hash_algorithm": AUTHORITY.PAYLOAD_HASH_ALGORITHM,
                "protocol": (
                    "deterministic_replication_of_single_prediction_v1"
                ),
                "physical_samples_per_clip": 1,
                "independent_samples": False,
                "deterministic_delta_distribution": True,
                "seed_consumed": False,
                "logical_slots": list(range(16)),
                "released2_slots": [0, 1],
                "paper16_slots": list(range(16)),
                "face_slot": 0,
                "prediction_manifest": self.final_manifest,
                "validation_gate": self.gate,
                "variation_policy": {
                    "format": (
                        "raw_primitive_with_deterministic_delta_integrity_v1"
                    ),
                    "reported_statistic": "raw_metric_primitive_v1",
                    "reported_value_path_template": (
                        "body.<protocol>.metrics.Variation"
                    ),
                    "exact_zero_claim": False,
                    "delta_integrity_check": (
                        "variation_sum_lte_integrity_tolerance_sum_v1"
                    ),
                    "integrity_tolerance_source": (
                        "metric_report_float64_roundoff_bound_v1"
                    ),
                    "public_value_transform": "identity_no_clamp_no_round_v1",
                },
            },
        )
        self.failure = _write_receipt(
            root / "failure.json",
            ORCHESTRATOR.build_failure_manifest(self.epoch),
        )
        self.shards, self.model_sha, self.runtime_sha = self._shards()
        self.transaction_payload = self._transaction()
        self.transaction = _write_receipt(
            root / "candidate-transaction.json", self.transaction_payload
        )

    def _final_manifest(self) -> dict[str, object]:
        rows = []
        for position, canonical_row in enumerate(
            self.frozen_val.coverage["_ordered_clips"]
        ):
            canonical_clip_id = canonical_row["canonical_clip_id"]
            rows.append(
                {
                    "global_index": canonical_row["global_index"],
                    "split": "val",
                    "source_clip_id": canonical_row["source_clip_id"],
                    "canonical_clip_id": canonical_clip_id,
                    "frames": canonical_row["frames"],
                    "epoch": self.epoch,
                    "candidate_checkpoint_sha256": self.checkpoint[
                        "sha256"
                    ],
                    "prediction": {
                        "path": str(
                            (
                                self.root
                                / "predictions"
                                / f"res_{canonical_clip_id}.npz"
                            )
                            .resolve()
                        ),
                        "sha256": hashlib.sha256(
                            f"prediction-{position}".encode()
                        ).hexdigest(),
                        "bytes": 1,
                    },
                    "ground_truth": {
                        "path": str(
                            (
                                self.root
                                / "ground-truth"
                                / f"gt_{canonical_clip_id}.npz"
                            )
                            .resolve()
                        ),
                        "sha256": hashlib.sha256(
                            f"ground-truth-{position}".encode()
                        ).hexdigest(),
                        "bytes": 1,
                    },
                }
            )
        self.final_rows = rows
        payload = b"".join(
            (
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            for row in rows
        )
        return _write_bytes(self.root / "final_manifest.jsonl", payload)

    def _shards(
        self,
    ) -> tuple[list[dict[str, object]], str, str]:
        model_receipts: dict[str, dict[str, object]] = {
            "base": {
                **self.checkpoint,
                "stage": "base",
                "candidate_epoch": self.epoch,
                "optimizer_updates": self.updates,
                "updates_per_epoch": AUTHORITY.BASE_UPDATES_PER_EPOCH,
                "frozen_receipt_sha256": "f" * 64,
                "strict_state_dict_load": True,
                "all_model_state_tensors_finite": True,
                "frozen_eval": True,
            },
        }
        for stage in AUTHORITY.STAGES:
            model_receipts[stage] = {
                **copy.deepcopy(self.fixed_pipeline_checkpoints[stage]),
                "prerequisite_selection": copy.deepcopy(
                    self.authority.prerequisite_artifact
                ),
                "model_state_tensors": 1,
                "model_state_schema_sha256": hashlib.sha256(
                    f"{stage}-model-state-schema".encode("ascii")
                ).hexdigest(),
                "strict_state_dict_load": True,
                "all_model_state_tensors_finite": True,
                "frozen_eval": True,
            }
        runtime = {
            "python": "3.fixture",
            "torch": "fixture",
            "deterministic_algorithms": True,
            "seed": 20260731,
        }
        model_sha = AUTHORITY.canonical_json_sha256(model_receipts)
        runtime_sha = AUTHORITY.canonical_json_sha256(runtime)
        compact_preflight = {
            key: self.preflight[key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        }
        result = []
        for shard_id in range(AUTHORITY.EXPECTED_SHARDS):
            positions = list(
                range(
                    shard_id,
                    AUTHORITY.EXPECTED_VAL_CLIPS,
                    AUTHORITY.EXPECTED_SHARDS,
                )
            )
            manifest = _write_bytes(
                self.root / f"shard-{shard_id}.jsonl",
                b"".join(
                    (
                        json.dumps(
                            {
                                **self.final_rows[position],
                                "canonical_position": position,
                                "prediction": {
                                    **self.final_rows[position]["prediction"],
                                    "path": str(
                                        (
                                            self.root
                                            / f"shard-{shard_id}"
                                            / "predictions"
                                            / "val"
                                            / (
                                                "res_"
                                                + self.final_rows[position][
                                                    "canonical_clip_id"
                                                ]
                                                + ".npz"
                                            )
                                        ).resolve()
                                    ),
                                },
                                "ground_truth": {
                                    **self.final_rows[position][
                                        "ground_truth"
                                    ],
                                    "path": str(
                                        (
                                            self.root
                                            / f"shard-{shard_id}"
                                            / "ground-truth"
                                            / "val"
                                            / (
                                                "gt_"
                                                + self.final_rows[position][
                                                    "canonical_clip_id"
                                                ]
                                                + ".npz"
                                            )
                                        ).resolve()
                                    ),
                                },
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                    for position in positions
                ),
            )
            receipt = _write_receipt(
                self.root / f"shard-{shard_id}-receipt.json",
                {
                    "format": AUTHORITY.FRESH_VAL_SHARD_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": self.epoch,
                    "candidate_checkpoint": {
                        "path": self.checkpoint["path"],
                        "sha256": self.checkpoint["sha256"],
                    },
                    "preflight_receipt": compact_preflight,
                    "assignment": AUTHORITY.FRESH_VAL_ASSIGNMENT,
                    "shard_id": shard_id,
                    "num_shards": AUTHORITY.EXPECTED_SHARDS,
                    "clip_count": len(positions),
                    "frame_count": len(positions) * 65,
                    "prediction_files": len(positions),
                    "ground_truth_files": len(positions),
                    "manifest": {
                        "path": manifest["path"],
                        "sha256": manifest["sha256"],
                    },
                    "model_receipts": model_receipts,
                    "model_receipts_sha256": model_sha,
                    "runtime_contract": runtime,
                    "runtime_contract_sha256": runtime_sha,
                    "device": {
                        "device": f"cuda:{shard_id}",
                        "name": "NVIDIA H200",
                    },
                    "exact_once": True,
                    "finite": True,
                },
            )
            result.append(
                {
                    "shard_id": shard_id,
                    "receipt": receipt,
                    "manifest": manifest,
                }
            )
        return result, model_sha, runtime_sha

    def _transaction(self) -> dict[str, object]:
        return {
            "format": AUTHORITY.FRESH_VAL_TRANSACTION_FORMAT,
            "payload_hash_algorithm": AUTHORITY.PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "generator": "SemTalk Base Motion Generation",
            "generator_module": AUTHORITY.FRESH_VAL_GENERATOR_MODULE,
            "dataset": "SHOW",
            "target_speaker_scope": AUTHORITY.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "formal_host": AUTHORITY.FRESH_VAL_FORMAL_HOSTS[0],
            "epoch": self.epoch,
            "optimizer_updates": self.updates,
            "candidate_checkpoint": self.checkpoint,
            "prerequisite_selection": self.authority.prerequisite_artifact,
            "continuation_decision": self.authority.continuation_artifact,
            "preflight_receipt": self.preflight,
            "val_inputs_receipt": self.val_inputs,
            "pipeline_receipt": self.pipeline,
            "prediction_manifest": self.final_manifest,
            "inference_lineage": self.lineage,
            "distribution_receipt": self.distribution,
            "validation_gate": self.gate,
            "shards": self.shards,
            "failure_manifest": self.failure,
            "source_runtime_input_pins": {
                "source": {
                    "origin": AUTHORITY.EXPECTED_ORIGIN,
                    "commit": "1" * 40,
                    "tree": "2" * 40,
                    "clean": True,
                    "detached": True,
                    "local_branches_at_commit": [],
                },
                "preflight_receipt_payload_sha256": self.preflight[
                    "receipt_payload_sha256"
                ],
                "val_inputs_receipt": self.val_inputs,
                "pipeline_receipt": self.pipeline,
                "model_receipts_sha256": self.model_sha,
                "runtime_contract_sha256": self.runtime_sha,
                "prediction_manifest_sha256": self.final_manifest["sha256"],
                "distribution_receipt_payload_sha256": self.distribution[
                    "receipt_payload_sha256"
                ],
            },
            "coverage": {
                "clip_count": AUTHORITY.EXPECTED_VAL_CLIPS,
                "num_shards": AUTHORITY.EXPECTED_SHARDS,
                "exact_once": True,
                "all_finite": True,
                "failure_count": 0,
            },
        }

    def validate(self) -> tuple[dict[str, object], ...]:
        distribution_artifact, distribution = AUTHORITY._validate_distribution(
            self.distribution,
            prediction_manifest=self.final_manifest,
        )
        def validate_pipeline(
            path: Path,
            expected_sha256: str,
            *,
            expected_prerequisite_selection: object,
        ) -> tuple[dict[str, object], dict[str, object]]:
            if (
                path != Path(self.pipeline["path"])
                or expected_sha256 != self.pipeline["sha256"]
                or expected_prerequisite_selection
                != self.authority.prerequisite_artifact
            ):
                raise RuntimeError("fresh pipeline fixture binding changed")
            normalized, payload = AUTHORITY._verify_compact_receipt(
                self.pipeline, "fresh pipeline fixture"
            )
            return normalized, payload

        contract = SimpleNamespace(
            validate_val_inputs=LEGACY_VAL_CONTRACT.validate_val_inputs,
            validate_pipeline=validate_pipeline,
        )
        with mock.patch.object(
            AUTHORITY,
            "_fresh_local_module",
            side_effect=lambda name: (
                contract
                if name == "base_long_val_contract"
                else self.authority.source_module(
                    name,
                    self.authority.adapter(),
                )
            ),
        ):
            return AUTHORITY._validate_candidate_transaction(
                self.transaction,
                epoch=self.epoch,
                updates=self.updates,
                checkpoint=self.checkpoint,
                prediction_manifest=self.final_manifest,
                distribution_artifact=distribution_artifact,
                distribution=distribution,
                prerequisite_artifact=self.authority.prerequisite_artifact,
                continuation_artifact=self.authority.continuation_artifact,
            )


class BaseFreshValOrchestratorCpuTests(unittest.TestCase):
    def test_create_new_run_root_is_dirfd_bound_and_nonreusable(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk-create-root-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            run_root = root / "formal root with spaces"
            observed, device, inode = ORCHESTRATOR._create_new_directory_tree(
                run_root,
                subdirectories=("logs", "candidates", "seals"),
            )
            identity = ORCHESTRATOR.os.stat(
                run_root, follow_symlinks=False
            )
            self.assertEqual(observed, run_root)
            self.assertEqual((device, inode), (identity.st_dev, identity.st_ino))
            self.assertEqual(
                sorted(path.name for path in run_root.iterdir()),
                ["candidates", "logs", "seals"],
            )
            with self.assertRaises(FileExistsError):
                ORCHESTRATOR._create_new_directory_tree(run_root)

    def test_create_new_run_root_rejects_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk-create-root-link-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(
                ORCHESTRATOR.BaseFreshValOrchestratorError,
                "canonical non-symlink parent",
            ):
                ORCHESTRATOR._create_new_directory_tree(
                    linked_parent / "formal"
                )

    def test_safe_snapshot_rejects_rename_after_same_fd_open(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk-snapshot-rename-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            target = root / "artifact.json"
            moved = root / "opened-artifact.json"
            target.write_bytes(b"a" * (2 * 1024 * 1024))
            original_read = ORCHESTRATOR.os.read
            swapped = False

            def racing_read(file_fd: int, count: int) -> bytes:
                nonlocal swapped
                payload = original_read(file_fd, count)
                if payload and not swapped:
                    swapped = True
                    target.rename(moved)
                    target.write_bytes(b"replacement")
                return payload

            with mock.patch.object(
                ORCHESTRATOR.os, "read", side_effect=racing_read
            ):
                with self.assertRaisesRegex(
                    ORCHESTRATOR.BaseFreshValOrchestratorError,
                    "changed while it was read",
                ):
                    ORCHESTRATOR._safe_snapshot(target, "racing artifact")

    def test_safe_snapshot_rejects_symlink_swap_before_fd_open(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk-snapshot-symlink-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            target = root / "artifact.json"
            moved = root / "original-artifact.json"
            decoy = root / "decoy.json"
            target.write_text("original\n", encoding="utf-8")
            decoy.write_text("decoy\n", encoding="utf-8")
            original_open = ORCHESTRATOR.os.open
            swapped = False

            def racing_open(
                path: object, flags: int, *args: object, **kwargs: object
            ) -> int:
                nonlocal swapped
                if (
                    path == target.name
                    and kwargs.get("dir_fd") is not None
                    and not swapped
                ):
                    swapped = True
                    target.rename(moved)
                    target.symlink_to(decoy)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(
                ORCHESTRATOR.os, "open", side_effect=racing_open
            ):
                with self.assertRaisesRegex(
                    ORCHESTRATOR.BaseFreshValOrchestratorError,
                    "cannot safely read",
                ):
                    ORCHESTRATOR._safe_snapshot(target, "racing artifact")

    def test_safe_snapshot_rejects_regular_swap_before_fd_open(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk-snapshot-regular-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            target = root / "artifact.json"
            moved = root / "original-artifact.json"
            target.write_text("original\n", encoding="utf-8")
            original_open = ORCHESTRATOR.os.open
            swapped = False

            def racing_open(
                path: object, flags: int, *args: object, **kwargs: object
            ) -> int:
                nonlocal swapped
                if (
                    path == target.name
                    and kwargs.get("dir_fd") is not None
                    and not swapped
                ):
                    swapped = True
                    target.rename(moved)
                    target.write_text("regular replacement\n", encoding="utf-8")
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(
                ORCHESTRATOR.os, "open", side_effect=racing_open
            ):
                with self.assertRaisesRegex(
                    ORCHESTRATOR.BaseFreshValOrchestratorError,
                    "changed before it was opened",
                ):
                    ORCHESTRATOR._safe_snapshot(target, "racing artifact")

    def test_guarded_launcher_and_cpu_union_are_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root / "scripts/show_base/run_base_fresh_val_8shard.sh"
        ).read_text(encoding="utf-8")
        finalizer = (
            root
            / "scripts/show_base/finalize_base_fresh_val_partitions.sh"
        ).read_text(encoding="utf-8")
        winner_full = (
            root / "scripts/show_base/run_base_fresh_val_winner_full.sh"
        ).read_text(encoding="utf-8")
        publisher = (
            root
            / "scripts/show_base/publish_base_fresh_val_final_claim.sh"
        ).read_text(encoding="utf-8")
        orchestrator_source = (
            root / "scripts/show_base/base_fresh_val_orchestrator.py"
        ).read_text(encoding="utf-8")
        self.assertIn("/tmp/globaldiff_guarded_runner.py", launcher)
        self.assertIn(
            '. "$launcher_dir/guarded_runner_contract.sh"', launcher
        )
        self.assertIn(
            "semtalk_require_exact_guarded_runner_all_gpus", launcher
        )
        self.assertIn("--num-shards 8", launcher)
        self.assertIn("expected_argv", launcher)
        self.assertIn("argv == expected_argv", launcher)
        self.assertIn("_partition_epochs", launcher)
        self.assertIn(
            "candidate_index_modulo_partition_count", orchestrator_source
        )
        self.assertNotIn("pgrep", launcher)
        self.assertNotIn("pkill", launcher)
        self.assertIn("union-partitions", finalizer)
        self.assertIn("publish-evidence", finalizer)
        self.assertNotIn("--device", finalizer)
        for script in (launcher, finalizer, winner_full, publisher):
            self.assertIn("create-run-root", script)
            self.assertIn("printf '%s\\0%s\\0%s\\0'", script)
        self.assertIn("expected_formal_host=${context[35]}", winner_full)
        self.assertIn('"$(hostname)" != "$expected_formal_host"', winner_full)

    def test_two_candidate_seals_bind_each_epoch_screen_in_real_argv(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher_path = (
            root / "scripts/show_base/run_base_fresh_val_8shard.sh"
        )
        launcher = launcher_path.read_text(encoding="utf-8")
        start = "    # BEGIN CANDIDATE_SEAL_ARGV_DATAFLOW\n"
        end = "    # END CANDIDATE_SEAL_ARGV_DATAFLOW\n"
        self.assertEqual(launcher.count(start), 1)
        self.assertEqual(launcher.count(end), 1)
        production_block = launcher.split(start, 1)[1].split(end, 1)[0]

        with tempfile.TemporaryDirectory(
            prefix="semtalk-seal-argv-", dir="/private/tmp"
        ) as raw:
            temp = Path(raw)
            argv_log = temp / "argv.jsonl"
            orchestrator = temp / "orchestrator_stub.py"
            orchestrator.write_text(
                "import json, os, sys\n"
                "with open(os.environ['SEAL_ARGV_LOG'], 'a', "
                "encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            run_root = temp / "run"
            prefix = "\n".join(
                (
                    "set -euo pipefail",
                    f"python_bin={shlex.quote(sys.executable)}",
                    f"orchestrator={shlex.quote(str(orchestrator))}",
                    f"run_root={shlex.quote(str(run_root))}",
                    'mkdir -p "$run_root/logs" "$run_root/seals"',
                    "wave_epochs=(25 50)",
                    "declare -a replay_by_epoch=()",
                    "declare -a replay_sha_by_epoch=()",
                    "declare -a replay_payload_by_epoch=()",
                    "declare -a screen_by_epoch=()",
                    "declare -a screen_sha_by_epoch=()",
                    "declare -a screen_payload_by_epoch=()",
                    "declare -a transaction_by_epoch=()",
                    "declare -a transaction_sha_by_epoch=()",
                    "declare -a transaction_payload_by_epoch=()",
                    "for epoch in 25 50; do",
                    '  replay_by_epoch[$epoch]="/artifacts/e${epoch}-replay.json"',
                    '  screen_by_epoch[$epoch]="/artifacts/e${epoch}-screen.json"',
                    '  screen_sha_by_epoch[$epoch]="${epoch}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
                    '  screen_payload_by_epoch[$epoch]="${epoch}bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"',
                    '  transaction_by_epoch[$epoch]="/artifacts/e${epoch}-transaction.json"',
                    '  transaction_sha_by_epoch[$epoch]="${epoch}cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"',
                    '  transaction_payload_by_epoch[$epoch]="${epoch}dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"',
                    "done",
                    'prerequisite="/artifacts/prerequisite.json"',
                    'prerequisite_sha="e"',
                    'prerequisite_payload="f"',
                    'continuation="/artifacts/continuation.json"',
                    'continuation_sha="1"',
                    'continuation_payload="2"',
                    'real_cache="/artifacts/real-cache.json"',
                    'real_cache_sha="3"',
                    'real_cache_payload="4"',
                    "seal_paths=()",
                    "seal_shas=()",
                    "seal_payloads=()",
                    "artifact_result=()",
                    "artifact_fields() {",
                    '  artifact_result=("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" "1" "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")',
                    "}",
                    "# Reproduce the stale scalar left by the preceding replay loop.",
                    'screen="${screen_by_epoch[50]}"',
                    "",
                )
            )
            harness = temp / "harness.sh"
            harness.write_text(
                prefix + production_block + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["bash", str(harness)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "SEAL_ARGV_LOG": str(argv_log)},
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            calls = [
                json.loads(line)
                for line in argv_log.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(calls), 2)
        for epoch, argv in zip((25, 50), calls):
            self.assertEqual(argv[0], "candidate-seal")
            screen_index = argv.index("--primary-screen-path") + 1
            output_index = argv.index("--output-json") + 1
            self.assertEqual(
                argv[screen_index], f"/artifacts/e{epoch}-screen.json"
            )
            self.assertTrue(argv[output_index].endswith(f"/seals/e{epoch}.json"))
        self.assertNotEqual(
            calls[0][calls[0].index("--primary-screen-path") + 1],
            calls[1][calls[1].index("--primary-screen-path") + 1],
        )

    def test_direct_gpu_launcher_invocation_is_rejected(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = root / "scripts/show_base/run_base_fresh_val_8shard.sh"
        with tempfile.TemporaryDirectory(
            prefix="semtalk-direct-launch-", dir="/private/tmp"
        ) as raw:
            temp = Path(raw)
            spec = temp / "spec.json"
            spec.write_text("{}\n", encoding="utf-8")
            run_root = temp / "formal-output"
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            tree = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
                text=True,
            ).strip()
            result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    str(root),
                    sys.executable,
                    str(spec),
                    "a" * 64,
                    "b" * 64,
                    "0",
                    "2",
                    str(run_root),
                    commit,
                    tree,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stderr, r"guarded[- ]runner")
            self.assertFalse(run_root.exists())

    def test_unmocked_1715_clip_eight_shard_transaction_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-fresh-val-") as raw:
            fixture = CandidateTransactionFixture(Path(raw))
            artifact, payload, lineage, canonical = fixture.validate()
            self.assertEqual(artifact, fixture.transaction)
            self.assertEqual(payload["coverage"]["clip_count"], 1715)
            self.assertEqual(len(payload["shards"]), 8)
            self.assertEqual(lineage, fixture.lineage)
            self.assertEqual(canonical["rows"], 1715)
            self.assertEqual(canonical["selected_rows"], 1715)
            self.assertEqual(
                canonical["sha256"],
                hashlib.sha256(
                    fixture.frozen_val.canonical_path.read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                payload["generator_module"],
                "models.semtalk.semtalk_base",
            )

    def test_rehashed_metric_canonical_fork_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-fresh-val-") as raw:
            fixture = CandidateTransactionFixture(Path(raw))
            _artifact, _payload, lineage, canonical = fixture.validate()
            distribution_artifact, distribution = (
                AUTHORITY._validate_distribution(
                    fixture.distribution,
                    prediction_manifest=fixture.final_manifest,
                )
            )
            del distribution_artifact
            forked_manifest = _write_bytes(
                Path(raw) / "forked-canonical.jsonl",
                b'{"self_signed":"not_frozen_val"}\n',
            )
            forked_canonical = {
                **forked_manifest,
                "rows": AUTHORITY.EXPECTED_VAL_CLIPS,
                "selected_rows": AUTHORITY.EXPECTED_VAL_CLIPS,
            }
            report = {
                "format": AUTHORITY.METRIC_REPORT_FORMAT,
                "report_payload_hash_algorithm": (
                    AUTHORITY.METRIC_REPORT_HASH_ALGORITHM
                ),
                "status": "complete",
                "generator": "SemTalk Base-only",
                "dataset": AUTHORITY.EXPECTED_DATASET,
                "split": "val",
                "selection_protocol": {
                    "primary_metric": AUTHORITY.PRIMARY_METRIC,
                    "mode": "min",
                    "validation_only_for_selection": True,
                    "test_evaluations": 0,
                },
                "distribution_receipt": distribution,
                "inputs": {
                    "canonical_manifest": forked_canonical,
                    "prediction_manifest": {
                        **fixture.final_manifest,
                        "rows": AUTHORITY.EXPECTED_VAL_CLIPS,
                    },
                    "prediction_lineage": {
                        "path": lineage["path"],
                        "sha256": lineage["sha256"],
                        "bytes": lineage["bytes"],
                        "payload_sha256": lineage[
                            "receipt_payload_sha256"
                        ],
                    },
                },
                "metric_assets": {},
                "counts": {},
                "body": {},
                "face": {},
                "rs": {},
                "runtime": {},
                "formal_mode": True,
                "test_only_mode": False,
            }
            report["report_payload_sha256"] = (
                AUTHORITY.canonical_json_sha256(report, newline=True)
            )
            report_artifact = _write_json(
                Path(raw) / "forked-report.json", report
            )
            with self.assertRaisesRegex(
                AUTHORITY.PublishedWinnerClaimError,
                "canonical manifest differs from audited val inputs",
            ):
                AUTHORITY._validate_metric_report(
                    report_artifact,
                    prediction_manifest=fixture.final_manifest,
                    lineage_artifact=lineage,
                    distribution=distribution,
                    expected_canonical_manifest=canonical,
                )

    def test_rehashed_real_cache_canonical_fork_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-fresh-val-") as raw:
            fixture = CandidateTransactionFixture(Path(raw))
            _artifact, _payload, _lineage, canonical = fixture.validate()
            forked_manifest = _write_bytes(
                Path(raw) / "forked-cache-canonical.jsonl",
                b'{"self_signed":"not_frozen_val"}\n',
            )
            forked_canonical = {
                **forked_manifest,
                "rows": AUTHORITY.EXPECTED_VAL_CLIPS,
                "selected_rows": AUTHORITY.EXPECTED_VAL_CLIPS,
            }
            cache = _write_receipt(
                Path(raw) / "forked-cache.json",
                {
                    "format": "semtalk_show_released2_real_feature_cache_v1",
                    "status": "complete",
                    "split": "val",
                    "clip_count": AUTHORITY.EXPECTED_VAL_CLIPS,
                    "canonical_manifest": forked_canonical,
                    "metric_assets": {},
                    "runtime": {},
                    "real_feature_statistics": {},
                    "formal_mode": True,
                    "test_only_mode": False,
                },
            )
            with self.assertRaisesRegex(
                AUTHORITY.PublishedWinnerClaimError,
                "real-feature cache differs from audited val inputs",
            ):
                AUTHORITY._validate_primary_replay_receipt(
                    {},
                    report={},
                    report_validation={},
                    prediction_manifest=fixture.final_manifest,
                    distribution={},
                    expected_real_feature_cache=cache,
                    expected_canonical_manifest=canonical,
                )

    def test_fresh_v2_winner_to_claim_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-fresh-claim-") as raw:
            root = Path(raw)
            fixture = ClaimFixture(root / "legacy-fixture")
            canonical = {"fixture": True}
            cache = _write_receipt(
                root / "released2-real-feature-cache.json",
                {
                    "format": "semtalk_show_released2_real_feature_cache_v1",
                    "status": "complete",
                    "split": "val",
                    "clip_count": AUTHORITY.EXPECTED_VAL_CLIPS,
                    "canonical_manifest": canonical,
                },
            )
            rows = copy.deepcopy(fixture.winner_payload["candidates"])
            transactions: dict[int, dict[str, object]] = {}
            screens: dict[int, dict[str, object]] = {}
            replays: dict[int, dict[str, object]] = {}
            by_epoch: dict[int, dict[str, object]] = {}
            for row in rows:
                replay_path = Path(row["primary_replay_receipt"]["path"])
                fgd = float(row["body_released2_fgd"])
                screen = _write_receipt(
                    replay_path.parent / "released2-primary-screen.json",
                    {
                        "format": AUTHORITY.PRIMARY_SCREEN_FORMAT,
                        "status": "pass",
                        "split": "val",
                        "test_visible": False,
                        "primary_metric": fgd,
                    },
                )
                row.pop("talkshow_metric_report")
                row.pop("primary_replay_receipt")
                row["primary_screen_receipt"] = screen
                epoch = int(row["epoch"])
                transaction = _write_receipt(
                    replay_path.parent / "candidate-transaction.json",
                    {
                        "format": AUTHORITY.FRESH_VAL_TRANSACTION_FORMAT,
                        "status": "complete",
                        "epoch": epoch,
                        "inference_lineage": row["inference_lineage"],
                    },
                )
                row["candidate_transaction"] = transaction
                replay = _write_receipt(
                    replay_path.parent / "released2-primary-fresh-replay.json",
                    {
                        "format": (
                            "semtalk_show_released2_primary_fresh_replay_v1"
                        ),
                        "status": "complete",
                        "split": "val",
                        "test_visible": False,
                        "primary_metric": fgd,
                    },
                )
                row["primary_replay_receipt"] = replay
                transactions[epoch] = transaction
                screens[epoch] = screen
                replays[epoch] = replay
                by_epoch[epoch] = row
            selected = min(
                rows,
                key=lambda row: (
                    row["body_released2_fgd"],
                    row["epoch"],
                    row["optimizer_updates"],
                ),
            )
            selection_payload = copy.deepcopy(fixture.winner_payload)
            selection_payload["format"] = (
                AUTHORITY.FRESH_WINNER_SELECTION_FORMAT
            )
            selection_payload["real_feature_cache"] = cache
            selection_payload["candidates"] = rows
            selection_payload["selected"] = selected
            selection_payload.pop("receipt_payload_sha256", None)
            selection_artifact = _write_receipt(
                root / "fresh-winner-selection.json",
                selection_payload,
            )

            def replay_transaction(
                value: dict[str, object], **kwargs: object
            ) -> tuple[dict[str, object], ...]:
                epoch = int(kwargs["epoch"])
                normalized, payload = AUTHORITY._verify_compact_receipt(
                    value, f"Base e{epoch} candidate transaction"
                )
                if normalized != transactions[epoch]:
                    raise AssertionError("unexpected fresh transaction")
                return (
                    normalized,
                    payload,
                    by_epoch[epoch]["inference_lineage"],
                    canonical,
                )

            adapter = fixture.adapter()

            def replay_screen(
                value: dict[str, object], **_kwargs: object
            ) -> tuple[dict[str, object], float, dict[str, object]]:
                normalized, payload = AUTHORITY._verify_compact_receipt(
                    value, "fresh primary screen fixture"
                )
                epoch = next(
                    epoch
                    for epoch, expected in screens.items()
                    if expected == normalized
                )
                return (
                    normalized,
                    float(by_epoch[epoch]["body_released2_fgd"]),
                    {"fixture": True},
                )

            winner_full_closure = _write_receipt(
                root / "winner-full-metric-closure.json",
                {
                    "format": AUTHORITY.WINNER_FULL_CLOSURE_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                },
            )

            def replay_winner_full_closure(
                value: dict[str, object], **_kwargs: object
            ) -> tuple[dict[str, object], dict[str, object]]:
                normalized, payload = AUTHORITY._verify_compact_receipt(
                    value, "winner full closure fixture"
                )
                if normalized != winner_full_closure:
                    raise AssertionError("unexpected winner full closure")
                return normalized, payload

            def replay_primary_screen_replay(
                value: dict[str, object], **_kwargs: object
            ) -> tuple[dict[str, object], float]:
                normalized, _payload = AUTHORITY._verify_compact_receipt(
                    value, "released2 primary screen raw replay fixture"
                )
                epoch = next(
                    epoch
                    for epoch, expected in replays.items()
                    if normalized == expected
                )
                return normalized, float(by_epoch[epoch]["body_released2_fgd"])

            with (
                mock.patch.object(
                    AUTHORITY,
                    "_fresh_local_module",
                    side_effect=lambda name: fixture.source_module(
                        name, adapter
                    ),
                ),
                mock.patch.object(
                    AUTHORITY,
                    "_validate_candidate_transaction",
                    side_effect=replay_transaction,
                ),
                mock.patch.object(
                    AUTHORITY,
                    "_validate_primary_screen_receipt",
                    side_effect=replay_screen,
                ),
                mock.patch.object(
                    AUTHORITY,
                    "_validate_primary_screen_replay_receipt",
                    side_effect=replay_primary_screen_replay,
                ),
                mock.patch.object(
                    AUTHORITY,
                    "_validate_winner_full_metric_closure",
                    side_effect=replay_winner_full_closure,
                ),
            ):
                claim_payload = (
                    AUTHORITY.build_fresh_published_test_winner_claim(
                        winner_selection=selection_artifact,
                        prerequisite_selection=(
                            fixture.prerequisite_artifact
                        ),
                        continuation_decision=(
                            fixture.continuation_artifact
                        ),
                        continuation_waves=[],
                        real_feature_cache=cache,
                        winner_full_metric_closure=winner_full_closure,
                        expected_output_root=root / "one-shot-test-output",
                    )
                )
                claim_artifact = _write_receipt(
                    root / "fresh-published-claim.json", claim_payload
                )
                validated = (
                    AUTHORITY.validate_published_test_winner_claim(
                        claim_artifact["path"],
                        expected_claim_sha256=claim_artifact["sha256"],
                        expected_claim_bytes=claim_artifact["bytes"],
                        expected_claim_payload_sha256=claim_artifact[
                            "receipt_payload_sha256"
                        ],
                        expected_output_root=(
                            root / "one-shot-test-output"
                        ),
                        prerequisite_selection=(
                            fixture.prerequisite_artifact
                        ),
                        continuation_decision=(
                            fixture.continuation_artifact
                        ),
                        continuation_waves=[],
                        winner_full_metric_closure=winner_full_closure,
                        expected_claim_format=AUTHORITY.FRESH_CLAIM_FORMAT,
                    )
                )
            self.assertEqual(
                claim_payload["format"], AUTHORITY.FRESH_CLAIM_FORMAT
            )
            self.assertEqual(
                validated["winner_selection"], selection_artifact
            )
            self.assertEqual(
                validated["selected_base_checkpoint"],
                selected["candidate_checkpoint"],
            )
            self.assertEqual(
                len(selection_payload["candidates"]),
                len(AUTHORITY.BASE_CANDIDATE_EPOCHS),
            )

            missing_transaction = copy.deepcopy(selection_payload)
            missing_transaction["candidates"][0].pop(
                "candidate_transaction"
            )
            missing_transaction.pop("receipt_payload_sha256", None)
            missing_artifact = _write_receipt(
                root / "fresh-selection-missing-transaction.json",
                missing_transaction,
            )
            with (
                mock.patch.object(
                    AUTHORITY,
                    "_fresh_local_module",
                    side_effect=lambda name: fixture.source_module(
                        name, adapter
                    ),
                ),
                mock.patch.object(
                    AUTHORITY,
                    "_validate_candidate_transaction",
                    side_effect=replay_transaction,
                ),
                mock.patch.object(
                    AUTHORITY,
                    "_validate_primary_screen_receipt",
                    side_effect=replay_screen,
                ),
                self.assertRaisesRegex(
                    AUTHORITY.PublishedWinnerClaimError,
                    "fresh candidate transaction is mandatory",
                ),
            ):
                AUTHORITY.build_fresh_published_test_winner_claim(
                    winner_selection=missing_artifact,
                    prerequisite_selection=fixture.prerequisite_artifact,
                    continuation_decision=fixture.continuation_artifact,
                    continuation_waves=[],
                    real_feature_cache=cache,
                    winner_full_metric_closure=winner_full_closure,
                    expected_output_root=root / "missing-transaction-output",
                )

    def test_shard_runtime_pin_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-fresh-val-") as raw:
            fixture = CandidateTransactionFixture(Path(raw))
            transaction = copy.deepcopy(fixture.transaction_payload)
            transaction["source_runtime_input_pins"][
                "runtime_contract_sha256"
            ] = "f" * 64
            fixture.transaction = _write_receipt(
                Path(raw) / "tampered-transaction.json", transaction
            )
            with self.assertRaisesRegex(
                AUTHORITY.PublishedWinnerClaimError,
                "transaction closure changed",
            ):
                fixture.validate()

    def test_rehashed_shard_and_final_manifest_divergence_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-fresh-val-") as raw:
            root = Path(raw)
            fixture = CandidateTransactionFixture(root)
            shard_manifest_path = Path(fixture.shards[0]["manifest"]["path"])
            shard_rows = [
                json.loads(line)
                for line in shard_manifest_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            shard_rows[0]["prediction"]["sha256"] = "f" * 64
            tampered_manifest = _write_bytes(
                root / "rehashed-shard-0.jsonl",
                b"".join(
                    (
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                    for row in shard_rows
                ),
            )
            receipt_payload = json.loads(
                Path(fixture.shards[0]["receipt"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            receipt_payload["manifest"] = {
                "path": tampered_manifest["path"],
                "sha256": tampered_manifest["sha256"],
            }
            tampered_receipt = _write_receipt(
                root / "rehashed-shard-0-receipt.json", receipt_payload
            )
            transaction = copy.deepcopy(fixture.transaction_payload)
            transaction["shards"][0] = {
                "shard_id": 0,
                "receipt": tampered_receipt,
                "manifest": tampered_manifest,
            }
            fixture.transaction = _write_receipt(
                root / "rehashed-divergent-transaction.json", transaction
            )
            with self.assertRaisesRegex(
                AUTHORITY.PublishedWinnerClaimError,
                "shard/final manifest diverged",
            ):
                fixture.validate()

    def test_failure_manifest_is_create_new_and_val_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-fresh-val-") as raw:
            root = Path(raw)
            payload = ORCHESTRATOR.build_failure_manifest(1)
            output = root / "failures.json"
            artifact = ORCHESTRATOR._write_new(output.resolve(), payload)
            self.assertEqual(artifact["receipt_payload_sha256"], payload[
                "receipt_payload_sha256"
            ])
            self.assertFalse(payload["test_visible"])
            with self.assertRaises(FileExistsError):
                ORCHESTRATOR._write_new(output.resolve(), payload)

    def test_unmocked_two_worker_static_partition_exact_union(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-partition-") as raw:
            root = Path(raw)
            common = {
                "preflight_receipt": {
                    "path": str((root / "preflight.json").resolve()),
                    "sha256": "1" * 64,
                    "bytes": 1,
                    "receipt_payload_sha256": "2" * 64,
                },
                "prerequisite_selection": {
                    "path": str((root / "prerequisite.json").resolve()),
                    "sha256": "3" * 64,
                    "bytes": 1,
                    "receipt_payload_sha256": "4" * 64,
                },
                "continuation_decision": {
                    "path": str((root / "continuation.json").resolve()),
                    "sha256": "5" * 64,
                    "bytes": 1,
                    "receipt_payload_sha256": "6" * 64,
                },
                "source": {
                    "origin": AUTHORITY.EXPECTED_ORIGIN,
                    "commit": "7" * 40,
                    "tree": "8" * 40,
                },
                "val_inputs_receipt": {
                    "path": str((root / "val.json").resolve()),
                    "sha256": "9" * 64,
                    "bytes": 1,
                    "receipt_payload_sha256": "a" * 64,
                },
                "pipeline_receipt": {
                    "path": str((root / "pipeline.json").resolve()),
                    "sha256": "b" * 64,
                    "bytes": 1,
                    "receipt_payload_sha256": "c" * 64,
                },
                "runtime_contract_sha256": "d" * 64,
            }
            seals: dict[int, dict[str, object]] = {}
            for candidate_index, epoch in enumerate(
                AUTHORITY.BASE_CANDIDATE_EPOCHS
            ):
                formal_host = AUTHORITY.FRESH_VAL_FORMAL_HOSTS[
                    candidate_index % 2
                ]
                transaction = _write_receipt(
                    root / f"transaction-e{epoch}.json",
                    {
                        "format": AUTHORITY.FRESH_VAL_TRANSACTION_FORMAT,
                        "generator_module": AUTHORITY.FRESH_VAL_GENERATOR_MODULE,
                        "split": "val",
                        "test_visible": False,
                        "formal_host": formal_host,
                        "epoch": epoch,
                        "optimizer_updates": (
                            epoch * AUTHORITY.BASE_UPDATES_PER_EPOCH
                        ),
                        "preflight_receipt": common["preflight_receipt"],
                        "prerequisite_selection": common[
                            "prerequisite_selection"
                        ],
                        "continuation_decision": common[
                            "continuation_decision"
                        ],
                        "inference_lineage": {
                            "path": str(
                                (root / f"lineage-e{epoch}.json").resolve()
                            ),
                            "sha256": hashlib.sha256(
                                f"lineage-{epoch}".encode()
                            ).hexdigest(),
                            "bytes": 1,
                            "receipt_payload_sha256": hashlib.sha256(
                                f"lineage-payload-{epoch}".encode()
                            ).hexdigest(),
                        },
                        "prediction_manifest": {
                            "path": str(
                                (root / f"manifest-e{epoch}.jsonl").resolve()
                            ),
                            "sha256": hashlib.sha256(
                                f"manifest-{epoch}".encode()
                            ).hexdigest(),
                            "bytes": 1,
                        },
                        "distribution_receipt": {
                            "path": str(
                                (root / f"distribution-e{epoch}.json").resolve()
                            ),
                            "sha256": hashlib.sha256(
                                f"distribution-{epoch}".encode()
                            ).hexdigest(),
                            "bytes": 1,
                            "receipt_payload_sha256": hashlib.sha256(
                                f"distribution-payload-{epoch}".encode()
                            ).hexdigest(),
                        },
                        "source_runtime_input_pins": {
                            "source": common["source"],
                            "val_inputs_receipt": common[
                                "val_inputs_receipt"
                            ],
                            "pipeline_receipt": common[
                                "pipeline_receipt"
                            ],
                            "runtime_contract_sha256": common[
                                "runtime_contract_sha256"
                            ],
                        },
                        "coverage": {
                            "clip_count": AUTHORITY.EXPECTED_VAL_CLIPS,
                            "num_shards": AUTHORITY.EXPECTED_SHARDS,
                            "exact_once": True,
                            "all_finite": True,
                            "failure_count": 0,
                        },
                        "shards": [
                            {"shard_id": shard_id}
                            for shard_id in range(AUTHORITY.EXPECTED_SHARDS)
                        ],
                    },
                )
                seals[epoch] = _write_receipt(
                    root / f"seal-e{epoch}.json",
                    {
                        "format": ORCHESTRATOR.CANDIDATE_SEAL_FORMAT,
                        "payload_hash_algorithm": (
                            AUTHORITY.PAYLOAD_HASH_ALGORITHM
                        ),
                        "status": "complete",
                        "split": "val",
                        "test_visible": False,
                        "row": {
                            "epoch": epoch,
                            "optimizer_updates": (
                                epoch * AUTHORITY.BASE_UPDATES_PER_EPOCH
                            ),
                            "candidate_transaction": transaction,
                        },
                        "body_released2_fgd": float(epoch),
                    },
                )
            partition_artifacts = []
            for partition_id in range(2):
                epochs = ORCHESTRATOR._partition_epochs(partition_id, 2)
                receipt = ORCHESTRATOR.build_partition_receipt(
                    partition_id=partition_id,
                    partition_count=2,
                    candidate_seals=[seals[epoch] for epoch in epochs],
                )
                partition_artifacts.append(
                    _write_receipt(
                        root / f"partition-{partition_id}.json", receipt
                    )
                )
            union = ORCHESTRATOR.build_partition_union(
                partition_count=2,
                partition_receipts=partition_artifacts,
            )
            self.assertEqual(union["candidate_epochs"], list(
                AUTHORITY.BASE_CANDIDATE_EPOCHS
            ))
            self.assertEqual(union["coverage"], {
                "candidate_count": 22,
                "shard_count": 176,
                "clip_evaluations": 37_730,
                "exact_once": True,
                "all_finite": True,
                "failure_count": 0,
            })
            duplicate = copy.deepcopy(partition_artifacts)
            duplicate[1] = duplicate[0]
            with self.assertRaises(ORCHESTRATOR.BaseFreshValOrchestratorError):
                ORCHESTRATOR.build_partition_union(
                    partition_count=2,
                    partition_receipts=duplicate,
                )


if __name__ == "__main__":
    unittest.main()
