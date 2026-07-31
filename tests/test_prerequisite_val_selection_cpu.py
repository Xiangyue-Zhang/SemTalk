from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


from scripts.show_base import merge_prerequisite_val_shards as merger
from scripts.show_base import build_prerequisite_candidate_index as builder
from scripts.show_base import build_val_canonical_view as val_view_builder
from scripts.show_base import prerequisite_val_contract as contract
from scripts.show_base import select_prerequisite_candidates as selector


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrerequisiteValidationSelectionTest(unittest.TestCase):
    def source_receipt(
        self,
        source_root: Path,
        name: str,
    ) -> dict[str, object]:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        return {
            "source_root": str(source_root.resolve()),
            "origin": contract.EXPECTED_ORIGIN,
            "commit": "a" * 40,
            "tree": "b" * 40,
            "clean": True,
            "script": str(path.resolve()),
            "script_relative": name,
            "script_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def training_source_receipt(
        self,
        source_root: Path,
        name: str,
    ) -> dict[str, object]:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        training_audit = {
            "commit": "e" * 40,
            "tree": "f" * 40,
            "origin": contract.EXPECTED_ORIGIN,
            "entrypoint": str(path.resolve()),
            "entrypoint_sha256": digest,
        }
        return contract.receipt_payload(
            {
                "format": contract.TRAINING_SOURCE_FREEZE_FORMAT,
                "training_audit": training_audit,
                "source_root": str(source_root.resolve()),
                "portable_identity": {
                    "origin": contract.EXPECTED_ORIGIN,
                    "commit": "e" * 40,
                    "tree": "f" * 40,
                    "script_relative": name,
                    "script_sha256": digest,
                },
                "clean": True,
                "detached": True,
                "local_branch_count": 0,
                "official_baseline_commit": (
                    contract.OFFICIAL_BASELINE_COMMIT
                ),
                "official_baseline_is_ancestor": True,
            }
        )

    def canonical_fixture(
        self,
        root: Path,
    ) -> tuple[
        list[dict[str, object]],
        Path,
        str,
        Path,
        str,
        Path,
        str,
    ]:
        self.assertEqual(
            contract.EXPECTED_SHOW_SPLIT_COUNTS,
            val_view_builder.EXPECTED_SPLIT_COUNTS,
        )
        self.assertEqual(
            contract.EXPECTED_VAL_GLOBAL_INDEX_START,
            val_view_builder.EXPECTED_VAL_GLOBAL_INDEX_START,
        )
        canonical_root = root / "canonical_npz"
        canonical_root.mkdir(parents=True)
        speakers = tuple(contract.SHOW_SPEAKERS)
        val_rows: list[dict[str, object]] = []
        for ordinal, global_index in enumerate(
            range(
                contract.EXPECTED_VAL_GLOBAL_INDEX_START,
                contract.EXPECTED_VAL_GLOBAL_INDEX_STOP,
            )
        ):
            speaker = speakers[ordinal % len(speakers)]
            canonical = canonical_root / f"clip_{global_index:05d}.npz"
            canonical.write_bytes(f"canonical:{global_index}\n".encode())
            val_rows.append(
                {
                    "global_index": global_index,
                    "clip_id": (
                        f"{speaker}/video-{global_index}/"
                        f"sequence-{global_index}"
                    ),
                    "split": "val",
                    "speaker": speaker,
                    "speaker_id": contract.SHOW_SPEAKERS[speaker],
                    "frames": 90,
                    "canonical_npz": str(canonical.resolve()),
                    "canonical_npz_sha256": hashlib.sha256(
                        canonical.read_bytes()
                    ).hexdigest(),
                    "lineage_contract_sha256": "1" * 64,
                }
            )
        full_rows: list[dict[str, object]] = []
        val_by_index = {
            int(row["global_index"]): row for row in val_rows
        }
        global_index = 0
        for split, count in val_view_builder.EXPECTED_SPLIT_COUNTS.items():
            for ordinal in range(count):
                if split == "val":
                    row = val_by_index[global_index]
                else:
                    row = {
                        "global_index": global_index,
                        "clip_id": (
                            f"oliver/{split}-video-{ordinal}/"
                            f"sequence-{ordinal}"
                        ),
                        "split": split,
                        "lineage_contract_sha256": "1" * 64,
                    }
                full_rows.append(row)
                global_index += 1
        full_manifest = root / "full_canonical" / "manifest.jsonl"
        full_manifest_sha = write_jsonl(full_manifest, full_rows)
        full_lineage = root / "full_canonical" / "lineage.json"
        full_lineage_sha = write_json(
            full_lineage,
            {
                "lineage_contract_sha256": "1" * 64,
                "lineage_contract": {
                    "source_receipt": {
                        "origin": contract.EXPECTED_ORIGIN,
                        "commit": "2" * 40,
                        "tree": "3" * 40,
                    }
                },
            },
        )
        full_summary = root / "full_canonical" / "summary.json"
        full_summary_sha = write_json(
            full_summary,
            {
                "status": "complete",
                "manifest_sha256": full_manifest_sha,
                "lineage_sha256": full_lineage_sha,
                "split_counts": val_view_builder.EXPECTED_SPLIT_COUNTS,
            },
        )
        output = val_view_builder.build(
            val_view_builder.parser().parse_args(
                [
                    "--full-manifest",
                    str(full_manifest),
                    "--full-summary",
                    str(full_summary),
                    "--full-lineage",
                    str(full_lineage),
                    "--expected-full-manifest-sha256",
                    full_manifest_sha,
                    "--expected-full-summary-sha256",
                    full_summary_sha,
                    "--expected-full-lineage-sha256",
                    full_lineage_sha,
                    "--expected-source-commit",
                    "2" * 40,
                    "--expected-source-tree",
                    "3" * 40,
                    "--output-root",
                    str(root / "canonical"),
                ]
            )
        )
        manifest = Path(str(output["manifest"]))
        summary = Path(str(output["summary"]))
        lineage = Path(str(output["lineage"]))
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ]
        return (
            rows,
            manifest,
            str(output["manifest_sha256"]),
            summary,
            str(output["summary_sha256"]),
            lineage,
            str(output["lineage_sha256"]),
        )

    def rewrite_canonical_receipts(
        self,
        rows: list[dict[str, object]],
        manifest: Path,
        summary: Path,
        lineage: Path,
    ) -> tuple[str, str, str]:
        indices = [int(row["global_index"]) for row in rows]
        indices_sha = contract.canonical_payload_sha256(indices)
        manifest_sha = write_jsonl(manifest, rows)
        lineage_payload = json.loads(lineage.read_text())
        lineage_payload.pop("receipt_payload_sha256")
        lineage_payload["manifest_sha256"] = manifest_sha
        lineage_payload["global_index_start"] = indices[0]
        lineage_payload["global_index_stop_exclusive"] = indices[-1] + 1
        lineage_payload["global_indices_sha256"] = indices_sha
        lineage_payload = contract.receipt_payload(lineage_payload)
        lineage_sha = write_json(lineage, lineage_payload)
        summary_payload = json.loads(summary.read_text())
        summary_payload.pop("receipt_payload_sha256")
        summary_payload["manifest_sha256"] = manifest_sha
        summary_payload["lineage_sha256"] = lineage_sha
        summary_payload["global_index_start"] = indices[0]
        summary_payload["global_index_stop_exclusive"] = indices[-1] + 1
        summary_payload["global_indices_sha256"] = indices_sha
        summary_payload = contract.receipt_payload(summary_payload)
        summary_sha = write_json(summary, summary_payload)
        return manifest_sha, summary_sha, lineage_sha

    def candidate_fixture(
        self,
        root: Path,
        source: dict[str, object],
    ) -> tuple[Path, str, dict[str, object]]:
        stages: dict[str, list[dict[str, object]]] = {}
        source_receipts = {
            stage: dict(source) for stage in contract.STAGES
        }
        for stage in contract.STAGES:
            entries = []
            for epoch in contract.EXPECTED_CANDIDATE_EPOCHS:
                path = (
                    root
                    / "candidate_files"
                    / stage
                    / f"{stage}_epoch_{epoch:04d}.bin"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{stage}:{epoch}\n".encode())
                entries.append(
                    {
                        "epoch": epoch,
                        "optimizer_updates": (
                            epoch * contract.updates_per_epoch(stage)
                        ),
                        "checkpoint": str(path.resolve()),
                        "checkpoint_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        "checkpoint_bytes": path.stat().st_size,
                        "checkpoint_audit_sha256": hashlib.sha256(
                            f"audit:{stage}:{epoch}".encode()
                        ).hexdigest(),
                    }
                )
            stages[stage] = entries
        payload = contract.receipt_payload(
            {
                "format": contract.CANDIDATE_INDEX_FORMAT,
                "status": "complete",
                "target_dataset": "SHOW",
                "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
                "selection_split": "val",
                "test_visible": False,
                "candidate_epochs": list(
                    contract.EXPECTED_CANDIDATE_EPOCHS
                ),
                "updates_per_epoch": contract.updates_per_epoch_map(
                    contract.STAGES
                ),
                "source_policy": contract.build_source_policy(
                    source_receipts,
                    stages=contract.STAGES,
                    reprove_ancestry=False,
                ),
                "source_receipts": source_receipts,
                "config_sha256": {
                    stage: hashlib.sha256(stage.encode()).hexdigest()
                    for stage in contract.STAGES
                },
                "dataset_receipt_sha256": {
                    stage: hashlib.sha256(
                        f"dataset:{stage}".encode()
                    ).hexdigest()
                    for stage in contract.STAGES
                },
                "formal_training_status": {
                    stage: {
                        "path": str(
                            (
                                root
                                / "status"
                                / f"{stage}.json"
                            ).resolve()
                        ),
                        "sha256": hashlib.sha256(
                            f"status:{stage}".encode()
                        ).hexdigest(),
                    }
                    for stage in contract.STAGES
                },
                "stages": stages,
            }
        )
        path = root / "candidate_index.json"
        digest = write_json(path, payload)
        contract.validate_candidate_index(payload)
        return path, digest, payload

    @staticmethod
    def candidate_error(stage: str, epoch: int) -> float:
        targets = {
            "face": {40, 60},
            "hands": {80},
            "upper": {100},
            "lower": {120},
            "global": {140},
        }
        distance = min(abs(epoch - target) for target in targets[stage])
        return 0.125 + distance / 160.0

    def shard_fixture(
        self,
        root: Path,
        rows: list[dict[str, object]],
        canonical_receipt: dict[str, object],
        candidate_index: dict[str, object],
        evaluator_source: dict[str, object],
    ) -> Path:
        shard_root = root / "shard_root"
        for stage in contract.STAGES:
            for epoch in contract.EXPECTED_CANDIDATE_EPOCHS:
                candidate = contract.candidate_lookup(
                    candidate_index,
                    stage,
                    epoch,
                )
                error = self.candidate_error(stage, epoch)
                for shard_index in range(contract.EXPECTED_SHARDS):
                    shard_rows = [
                        row
                        for row in rows
                        if int(row["global_index"])
                        % contract.EXPECTED_SHARDS
                        == shard_index
                    ]
                    clips = [str(row["clip_id"]) for row in shard_rows]
                    windows, records_sha = merger._expected_window_evidence(
                        shard_rows
                    )
                    counts = merger._expected_accumulator_counts(
                        stage,
                        windows,
                    )
                    accumulators = {
                        name: {
                            "count": count,
                            "sum_abs": error * count,
                            "sum_squared": error * error * count,
                            "max_abs": error,
                        }
                        for name, count in counts.items()
                    }
                    histograms = None
                    if stage in contract.RVQ_STAGES:
                        tokens = windows * (contract.WINDOW_LENGTH // 4)
                        histograms = [
                            [tokens] + [0] * (contract.CODEBOOK_SIZE - 1)
                            for _ in range(contract.RVQ_LEVELS)
                        ]
                    payload = contract.receipt_payload(
                        {
                            "format": contract.SHARD_FORMAT,
                            "status": "complete",
                            "stage": stage,
                            "split": "val",
                            "test_visible": False,
                            "epoch": epoch,
                            "optimizer_updates": candidate[
                                "optimizer_updates"
                            ],
                            "checkpoint": {
                                "path": candidate["checkpoint"],
                                "sha256": candidate[
                                    "checkpoint_sha256"
                                ],
                                "bytes": candidate["checkpoint_bytes"],
                            },
                            "checkpoint_audit_sha256": candidate[
                                "checkpoint_audit_sha256"
                            ],
                            "canonical_receipt": canonical_receipt,
                            "producer_source": evaluator_source,
                            "protocol": {
                                "name": contract.SELECTION_METRICS[stage],
                                "stage_independent": True,
                                "candidate_variable_only": True,
                                "window_length": contract.WINDOW_LENGTH,
                                "window_stride": contract.WINDOW_STRIDE,
                                "selection_split": "val",
                                "test_visible": False,
                                "full_base_fgd_used": False,
                                "inference_only_exclusions": [
                                    "quantizer_embedding_loss",
                                    "smplx_vertex_loss",
                                ],
                            },
                            "shard": {
                                "index": shard_index,
                                "count": contract.EXPECTED_SHARDS,
                                "clip_count": len(shard_rows),
                                "clip_ids": clips,
                                "clip_ids_sha256": (
                                    contract.canonical_payload_sha256(clips)
                                ),
                                "window_count": windows,
                                "window_records_sha256": records_sha,
                            },
                            "finite": True,
                            "exact_once_within_shard": True,
                            "determinism": {
                                "seed": 20260731,
                                "torch_deterministic_algorithms": True,
                                "tf32": False,
                                "first_batch_exact_replay": True,
                                "output_digest_sha256": "d" * 64,
                            },
                            "accumulators": accumulators,
                            "codebook_histograms": histograms,
                            "runtime": {
                                "python": "3.11.9",
                                "torch": "2.7.0",
                                "numpy": "2.1.0",
                                "device": "cuda:0",
                                "batch_size": 32,
                                "batches": math.ceil(windows / 32),
                            },
                        }
                    )
                    path = (
                        shard_root
                        / "shards"
                        / stage
                        / f"epoch_{epoch:04d}"
                        / f"shard_{shard_index:02d}.json"
                    )
                    write_json(path, payload)
        return shard_root

    def build_full_fixture(self, root: Path) -> dict[str, object]:
        source_root = root / "source"
        evaluator_source = self.source_receipt(
            source_root / "host0",
            "evaluate_prerequisite_val_shard.py",
        )
        evaluator_source_host1 = self.source_receipt(
            source_root / "host1",
            "evaluate_prerequisite_val_shard.py",
        )
        merge_source = self.source_receipt(
            source_root / "merge",
            "merge_prerequisite_val_shards.py",
        )
        selector_source = self.source_receipt(
            source_root / "selector",
            "select_prerequisite_candidates.py",
        )
        training_source = self.training_source_receipt(
            source_root / "training",
            "show_base_train.py",
        )
        (
            rows,
            manifest,
            manifest_sha,
            summary,
            summary_sha,
            lineage,
            lineage_sha,
        ) = self.canonical_fixture(root)
        loaded_rows, canonical_receipt = contract.load_val_canonical(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            summary_path=summary,
            summary_sha256=summary_sha,
            lineage_path=lineage,
            lineage_sha256=lineage_sha,
        )
        self.assertEqual(rows, loaded_rows)
        candidate_path, candidate_sha, candidate_index = (
            self.candidate_fixture(root, training_source)
        )
        shard_root = self.shard_fixture(
            root,
            rows,
            canonical_receipt,
            candidate_index,
            evaluator_source,
        )
        shard_roots = [root / "shard_partition_0", root / "shard_partition_1"]
        for partition_root in shard_roots:
            (partition_root / "shards").mkdir(parents=True)
        plan_index = 0
        for stage in contract.STAGES:
            for epoch in contract.EXPECTED_CANDIDATE_EPOCHS:
                source = (
                    shard_root
                    / "shards"
                    / stage
                    / f"epoch_{epoch:04d}"
                )
                destination_parent = (
                    shard_roots[plan_index % 2] / "shards" / stage
                )
                destination_parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination_parent / source.name))
                plan_index += 1
        shutil.rmtree(shard_root)
        for shard_path in (
            shard_roots[1] / "shards"
        ).glob("*/*/shard_*.json"):
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            payload.pop("receipt_payload_sha256")
            payload["producer_source"] = evaluator_source_host1
            write_json(shard_path, contract.receipt_payload(payload))
        return {
            "rows": rows,
            "manifest": manifest,
            "manifest_sha": manifest_sha,
            "summary": summary,
            "summary_sha": summary_sha,
            "lineage": lineage,
            "lineage_sha": lineage_sha,
            "candidate_path": candidate_path,
            "candidate_sha": candidate_sha,
            "candidate_index": candidate_index,
            "shard_roots": shard_roots,
            "merge_source": merge_source,
            "selector_source": selector_source,
        }

    def test_full_merge_and_exact_selection_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_full_fixture(root)
            measurement_root = root / "measurements"
            measurement = merger.merge(
                candidate_index_path=fixture["candidate_path"],
                candidate_index_sha256=fixture["candidate_sha"],
                canonical_manifest=fixture["manifest"],
                canonical_manifest_sha256=fixture["manifest_sha"],
                canonical_summary=fixture["summary"],
                canonical_summary_sha256=fixture["summary_sha"],
                canonical_lineage=fixture["lineage"],
                canonical_lineage_sha256=fixture["lineage_sha"],
                shard_roots=fixture["shard_roots"],
                output_root=measurement_root,
                merge_source=fixture["merge_source"],
            )
            self.assertEqual(measurement["coverage"]["total_shard_jobs"], 400)
            self.assertEqual(
                set(path.name for path in measurement_root.iterdir()),
                {
                    "face_measurements.json",
                    "hands_measurements.json",
                    "upper_measurements.json",
                    "lower_measurements.json",
                    "global_measurements.json",
                    "measurement_index.json",
                },
            )
            measurement_index = measurement_root / "measurement_index.json"
            measurement_sha = hashlib.sha256(
                measurement_index.read_bytes()
            ).hexdigest()
            output = root / "selection.json"
            selected = selector.select(
                candidate_index_path=fixture["candidate_path"],
                candidate_index_sha256=fixture["candidate_sha"],
                measurement_index_path=measurement_index,
                measurement_index_sha256=measurement_sha,
                output_json=output,
                selector_source=fixture["selector_source"],
            )
            self.assertEqual(
                set(selected),
                {
                    "format",
                    "status",
                    "target_dataset",
                    "target_speaker_scope",
                    "split",
                    "test_visible",
                    "protocol",
                    "selection_policy",
                    "canonical_view",
                    "producer_sources",
                    "training_sources",
                    "config_sha256",
                    "candidate_index_receipt",
                    "measurement_index_receipt",
                    "stages",
                    "receipt_payload_sha256",
                },
            )
            self.assertEqual(
                {stage["stage"]: stage["epoch"] for stage in selected["stages"]},
                {
                    "face": 40,
                    "hands": 80,
                    "upper": 100,
                    "lower": 120,
                    "global": 140,
                },
            )
            face = selected["stages"][0]
            self.assertEqual(face["candidate_index"], 1)
            self.assertEqual(
                set(face["measurement_receipt"]),
                {"path", "sha256", "receipt_payload_sha256"},
            )
            self.assertFalse(selected["test_visible"])
            self.assertEqual(
                selected["target_speaker_scope"],
                "all_speakers_0_1_2_3",
            )
            self.assertFalse(selected["protocol"]["full_base_fgd_used"])
            self.assertEqual(
                selected["selection_policy"]["ordering"],
                [
                    "selection_score",
                    "epoch",
                    "optimizer_updates",
                    "checkpoint_sha256",
                ],
            )
            self.assertTrue(output.is_file())
            contract.validate_selection_receipt(selected)
            tampered_cases = []
            tampered = copy.deepcopy(selected)
            tampered["status"] = "tampered"
            tampered_cases.append(tampered)
            tampered = copy.deepcopy(selected)
            tampered["stages"][0]["epoch"] = 200
            tampered_cases.append(tampered)
            tampered = copy.deepcopy(selected)
            tampered["producer_sources"]["selector"]["tree"] = "0" * 40
            tampered_cases.append(tampered)
            for tampered in tampered_cases:
                with self.assertRaisesRegex(
                    contract.ContractError,
                    "receipt payload mismatch",
                ):
                    contract.validate_selection_receipt(tampered)

            mismatched_selector = copy.deepcopy(
                fixture["selector_source"]
            )
            mismatched_selector["commit"] = "c" * 40
            with self.assertRaisesRegex(
                contract.ContractError,
                "repository identities differ",
            ):
                selector.select(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    measurement_index_path=measurement_index,
                    measurement_index_sha256=measurement_sha,
                    output_json=root / "mismatched_selector.json",
                    selector_source=mismatched_selector,
                )

            mismatched_shard = (
                fixture["shard_roots"][1]
                / "shards"
                / "face"
                / "epoch_0040"
                / "shard_00.json"
            )
            original_shard_bytes = mismatched_shard.read_bytes()
            mismatched_payload = json.loads(
                original_shard_bytes.decode("utf-8")
            )
            mismatched_payload.pop("receipt_payload_sha256")
            mismatched_payload["producer_source"]["commit"] = "c" * 40
            write_json(
                mismatched_shard,
                contract.receipt_payload(mismatched_payload),
            )
            with self.assertRaisesRegex(
                contract.ContractError,
                "portable source changed",
            ):
                merger.merge(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    canonical_manifest=fixture["manifest"],
                    canonical_manifest_sha256=fixture["manifest_sha"],
                    canonical_summary=fixture["summary"],
                    canonical_summary_sha256=fixture["summary_sha"],
                    canonical_lineage=fixture["lineage"],
                    canonical_lineage_sha256=fixture["lineage_sha"],
                    shard_roots=fixture["shard_roots"],
                    output_root=root / "source_mismatch_measurements",
                    merge_source=fixture["merge_source"],
                )
            mismatched_shard.write_bytes(original_shard_bytes)

            shard_tree = fixture["shard_roots"][0] / "shards"
            extra_stage = shard_tree / "extra_stage"
            extra_stage.mkdir()
            with self.assertRaisesRegex(
                contract.ContractError,
                "unexpected shard directory",
            ):
                merger.merge(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    canonical_manifest=fixture["manifest"],
                    canonical_manifest_sha256=fixture["manifest_sha"],
                    canonical_summary=fixture["summary"],
                    canonical_summary_sha256=fixture["summary_sha"],
                    canonical_lineage=fixture["lineage"],
                    canonical_lineage_sha256=fixture["lineage_sha"],
                    shard_roots=fixture["shard_roots"],
                    output_root=root / "extra_stage_measurements",
                    merge_source=fixture["merge_source"],
                )
            extra_stage.rmdir()

            extra_epoch = shard_tree / "face" / "epoch_9999"
            extra_epoch.mkdir()
            with self.assertRaisesRegex(
                contract.ContractError,
                "unexpected shard directory",
            ):
                merger.merge(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    canonical_manifest=fixture["manifest"],
                    canonical_manifest_sha256=fixture["manifest_sha"],
                    canonical_summary=fixture["summary"],
                    canonical_summary_sha256=fixture["summary_sha"],
                    canonical_lineage=fixture["lineage"],
                    canonical_lineage_sha256=fixture["lineage_sha"],
                    shard_roots=fixture["shard_roots"],
                    output_root=root / "extra_epoch_measurements",
                    merge_source=fixture["merge_source"],
                )
            extra_epoch.rmdir()

            shard08 = (
                shard_tree
                / "face"
                / "epoch_0020"
                / "shard_08.json"
            )
            shutil.copyfile(
                shard08.with_name("shard_00.json"),
                shard08,
            )
            with self.assertRaisesRegex(
                contract.ContractError,
                "unexpected shard file",
            ):
                merger.merge(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    canonical_manifest=fixture["manifest"],
                    canonical_manifest_sha256=fixture["manifest_sha"],
                    canonical_summary=fixture["summary"],
                    canonical_summary_sha256=fixture["summary_sha"],
                    canonical_lineage=fixture["lineage"],
                    canonical_lineage_sha256=fixture["lineage_sha"],
                    shard_roots=fixture["shard_roots"],
                    output_root=root / "shard08_measurements",
                    merge_source=fixture["merge_source"],
                )
            shard08.unlink()

            shard_symlink = (
                shard_tree
                / "face"
                / "epoch_0020"
                / "shard_alias.json"
            )
            shard_symlink.symlink_to("shard_00.json")
            with self.assertRaisesRegex(
                contract.ContractError,
                "contains a symlink",
            ):
                merger.merge(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    canonical_manifest=fixture["manifest"],
                    canonical_manifest_sha256=fixture["manifest_sha"],
                    canonical_summary=fixture["summary"],
                    canonical_summary_sha256=fixture["summary_sha"],
                    canonical_lineage=fixture["lineage"],
                    canonical_lineage_sha256=fixture["lineage_sha"],
                    shard_roots=fixture["shard_roots"],
                    output_root=root / "symlink_measurements",
                    merge_source=fixture["merge_source"],
                )
            shard_symlink.unlink()

            duplicate_source = (
                fixture["shard_roots"][0]
                / "shards"
                / "face"
                / "epoch_0020"
                / "shard_00.json"
            )
            duplicate_target = (
                fixture["shard_roots"][1]
                / "shards"
                / "face"
                / "epoch_0020"
                / "shard_00.json"
            )
            duplicate_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(duplicate_source, duplicate_target)
            with self.assertRaisesRegex(
                contract.ContractError,
                "duplicate shard path",
            ):
                merger.merge(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    canonical_manifest=fixture["manifest"],
                    canonical_manifest_sha256=fixture["manifest_sha"],
                    canonical_summary=fixture["summary"],
                    canonical_summary_sha256=fixture["summary_sha"],
                    canonical_lineage=fixture["lineage"],
                    canonical_lineage_sha256=fixture["lineage_sha"],
                    shard_roots=fixture["shard_roots"],
                    output_root=root / "duplicate_measurements",
                    merge_source=fixture["merge_source"],
                )
            duplicate_target.unlink()

            missing = (
                fixture["shard_roots"][1]
                / "shards"
                / "global"
                / "epoch_0200"
                / "shard_07.json"
            )
            missing.unlink()
            with self.assertRaisesRegex(
                contract.ContractError,
                "inventory is not the exact expected 400",
            ):
                merger.merge(
                    candidate_index_path=fixture["candidate_path"],
                    candidate_index_sha256=fixture["candidate_sha"],
                    canonical_manifest=fixture["manifest"],
                    canonical_manifest_sha256=fixture["manifest_sha"],
                    canonical_summary=fixture["summary"],
                    canonical_summary_sha256=fixture["summary_sha"],
                    canonical_lineage=fixture["lineage"],
                    canonical_lineage_sha256=fixture["lineage_sha"],
                    shard_roots=fixture["shard_roots"],
                    output_root=root / "missing_measurements",
                    merge_source=fixture["merge_source"],
                )

    def test_strict_json_and_forbidden_labels(self) -> None:
        compact = json.dumps(
            {"a": 1},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(
            contract.canonical_payload_sha256({"a": 1}),
            hashlib.sha256(compact).hexdigest(),
        )
        self.assertNotEqual(
            contract.canonical_payload_sha256({"a": 1}),
            hashlib.sha256(compact + b"\n").hexdigest(),
        )
        with self.assertRaisesRegex(contract.ContractError, "duplicate JSON key"):
            contract.strict_json_bytes(b'{"a":1,"a":2}', "duplicate")
        with self.assertRaisesRegex(contract.ContractError, "non-finite"):
            contract.strict_json_bytes(b'{"a":NaN}', "nan")
        for value in (
            "/frozen/test/results.json",
            "/weights/e30/checkpoint.bin",
            "/weights/Speaker2/checkpoint.bin",
        ):
            with self.assertRaises(contract.ContractError):
                contract.reject_forbidden_label(value, "forbidden")

    def test_candidate_schedule_is_epochs_20_through_200(self) -> None:
        self.assertEqual(
            contract.EXPECTED_CANDIDATE_EPOCHS,
            (20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
        )
        self.assertEqual(
            contract.validate_candidate_epochs(
                [*contract.EXPECTED_CANDIDATE_EPOCHS, 220, 240]
            ),
            (*contract.EXPECTED_CANDIDATE_EPOCHS, 220, 240),
        )
        for invalid in (
            list(contract.EXPECTED_CANDIDATE_EPOCHS[:-1]),
            [*contract.EXPECTED_CANDIDATE_EPOCHS, 200],
            [*contract.EXPECTED_CANDIDATE_EPOCHS, 221],
            [*contract.EXPECTED_CANDIDATE_EPOCHS, 240],
            [*contract.EXPECTED_CANDIDATE_EPOCHS, 220, 260],
            [*contract.EXPECTED_CANDIDATE_EPOCHS, 240, 220],
        ):
            with self.assertRaises(contract.ContractError):
                contract.validate_candidate_epochs(invalid)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = self.training_source_receipt(
                root / "source",
                "show_base_train.py",
            )
            _, _, payload = self.candidate_fixture(root, source)
            partial = {
                key: copy.deepcopy(value)
                for key, value in payload.items()
                if key != "receipt_payload_sha256"
            }
            partial["format"] = contract.PARTIAL_CANDIDATE_INDEX_FORMAT
            for key in (
                "source_receipts",
                "config_sha256",
                "dataset_receipt_sha256",
                "formal_training_status",
                "stages",
            ):
                partial[key].pop("global")
            partial["updates_per_epoch"].pop("global")
            partial["source_policy"] = contract.build_source_policy(
                partial["source_receipts"],
                stages=contract.STAGES[:-1],
                reprove_ancestry=False,
            )
            partial = contract.receipt_payload(partial)
            contract.validate_candidate_index(
                partial,
                allow_partial=True,
            )
            with self.assertRaisesRegex(
                contract.ContractError,
                "protocol",
            ):
                contract.validate_candidate_index(partial)
            for missing_stage in contract.STAGES[:-1]:
                broken_partial = copy.deepcopy(partial)
                broken_partial.pop("receipt_payload_sha256")
                broken_partial["stages"].pop(missing_stage)
                broken_partial = contract.receipt_payload(broken_partial)
                with self.assertRaisesRegex(
                    contract.ContractError,
                    "stage coverage",
                ):
                    contract.validate_candidate_index(
                        broken_partial,
                        allow_partial=True,
                    )
            broken = dict(payload)
            broken.pop("receipt_payload_sha256")
            broken["stages"] = dict(broken["stages"])
            broken["stages"].pop("global")
            broken = contract.receipt_payload(broken)
            with self.assertRaisesRegex(
                contract.ContractError,
                "stage coverage",
            ):
                contract.validate_candidate_index(broken)

    def test_candidate_audit_exact_schema_and_status_final_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stage = "face"
            epoch = 200
            source = {
                "commit": "a" * 40,
                "tree": "b" * 40,
                "origin": contract.EXPECTED_ORIGIN,
                "entrypoint": str((root / "show_base_train.py").resolve()),
                "entrypoint_sha256": "c" * 64,
            }
            initialization_spec = contract.OFFICIAL_INITIALIZATION[stage]
            initialization = {
                "stage": stage,
                "path": str(
                    (root / initialization_spec["filename"]).resolve()
                ),
                "filename": initialization_spec["filename"],
                "sha256": initialization_spec["sha256"],
                "official_all_speakers": True,
                "withdrawn_e30_allowed": False,
                "model_state_sha256": "d" * 64,
            }
            sampler = {
                "class": (
                    "torch.utils.data.distributed.DistributedSampler"
                ),
                "shuffle": True,
                "seed": 43,
                "drop_last": True,
                "set_epoch": "before every epoch",
            }
            rvq_ema = {
                "enabled": True,
                "assignment": (
                    "rank-local Gumbel samples from seed + global rank"
                ),
                "statistics": ["code_count", "code_sum"],
                "collective": "all_reduce SUM",
                "initialization": (
                    "global rank-ordered prefix; rank0 broadcast"
                ),
                "dead_code_reset": (
                    "global rank-ordered prefix on demand; rank0 broadcast"
                ),
                "perplexity": "global code_count all_reduce SUM",
                "rank_state": (
                    "exact SHA-256 agreement at save/resume/finalize"
                ),
            }
            stage_updates = contract.updates_per_epoch(stage)
            stage_batch = 256 if stage in contract.RVQ_STAGES else 64
            distributed_unsigned = {
                "format": "semtalk_show_representation_ddp_v1",
                "formal_stage": stage,
                "world_size": 2 if stage in contract.RVQ_STAGES else 1,
                "local_batch_size": (
                    128 if stage in contract.RVQ_STAGES else 64
                ),
                "global_batch_size": stage_batch,
                "train_samples": 127_286,
                "available_train_samples": 127_286,
                "consumed_samples_per_epoch": stage_updates * stage_batch,
                "dropped_samples_per_epoch": (
                    127_286 - stage_updates * stage_batch
                ),
                "padding_or_duplicate_samples_per_epoch": 0,
                "updates_per_epoch": stage_updates,
                "loader_drop_last": True,
                "sampler": sampler,
                "rvq_ema": rvq_ema,
            }
            distributed = {
                **distributed_unsigned,
                "receipt_sha256": contract.canonical_payload_sha256(
                    distributed_unsigned
                ),
            }
            rvq_prior = {
                "format": "semtalk_show_official_rvq_ema_prior_v2",
                "layers": [
                    {
                        "name": f"module.quantizer.layers.{index}",
                        "ema_decay": 0.99,
                        "prior_count": 100.0,
                    }
                    for index in range(contract.RVQ_LEVELS)
                ],
                "init": True,
                "code_sum": (
                    "loaded codebook multiplied by decay-aware prior_count"
                ),
                "code_count": "1 / (1 - ema_decay) per code",
                "first_forward_codebook_reset": False,
                "unused_code_grace": (
                    "legacy reset only after the decay-aware prior falls "
                    "below one"
                ),
            }
            rvq_rank = {
                "format": "semtalk_show_rvq_rank_state_v1",
                "world_size": 2,
                "state_sha256": "e" * 64,
                "all_ranks_exact": True,
            }
            audit = {
                "format": "semtalk_show_representation_candidate_v1",
                "formal_stage": stage,
                "completed_epochs": epoch,
                "optimizer_updates": 99_400,
                "config_sha256": "1" * 64,
                "lineage_manifest_sha256": "2" * 64,
                "dataset_receipt_sha256": "3" * 64,
                "source_receipt": source,
                "source_receipt_sha256": (
                    contract.canonical_payload_sha256(source)
                ),
                "initialization_receipt": initialization,
                "rvq_ema_prior_receipt": rvq_prior,
                "distributed_training_receipt": distributed,
                "rvq_rank_state_receipt": rvq_rank,
                "selection_status": "offline_validation_pending",
            }
            validated = contract.validate_representation_candidate_audit(
                audit,
                stage=stage,
                epoch=epoch,
                label="fixture candidate",
                reprove_paths=False,
            )
            self.assertEqual(validated, audit)
            for key, invalid in (
                ("init", False),
                ("code_sum", "tampered"),
                ("code_count", "tampered"),
                ("first_forward_codebook_reset", True),
                ("unused_code_grace", "tampered"),
            ):
                broken = copy.deepcopy(audit)
                broken["rvq_ema_prior_receipt"][key] = invalid
                with self.assertRaisesRegex(
                    contract.ContractError,
                    "RVQ EMA-prior schema mismatch",
                ):
                    contract.validate_representation_candidate_audit(
                        broken,
                        stage=stage,
                        epoch=epoch,
                        label="fixture candidate",
                        reprove_paths=False,
                    )
            for mutation in ("extra", "missing"):
                broken = copy.deepcopy(audit)
                receipt = broken["rvq_ema_prior_receipt"]
                if mutation == "extra":
                    receipt["unexpected"] = True
                else:
                    receipt.pop("unused_code_grace")
                with self.assertRaisesRegex(
                    contract.ContractError,
                    "schema mismatch",
                ):
                    contract.validate_representation_candidate_audit(
                        broken,
                        stage=stage,
                        epoch=epoch,
                        label="fixture candidate",
                        reprove_paths=False,
                    )
            reordered_prior = copy.deepcopy(audit)
            reordered_prior["rvq_ema_prior_receipt"]["layers"][0][
                "name"
            ], reordered_prior["rvq_ema_prior_receipt"]["layers"][1][
                "name"
            ] = (
                reordered_prior["rvq_ema_prior_receipt"]["layers"][1][
                    "name"
                ],
                reordered_prior["rvq_ema_prior_receipt"]["layers"][0][
                    "name"
                ],
            )
            with self.assertRaisesRegex(
                contract.ContractError,
                "invalid RVQ EMA-prior layer",
            ):
                contract.validate_representation_candidate_audit(
                    reordered_prior,
                    stage=stage,
                    epoch=epoch,
                    label="fixture candidate",
                    reprove_paths=False,
                )
            boolean_prior = copy.deepcopy(audit)
            boolean_prior["rvq_ema_prior_receipt"]["layers"][0][
                "prior_count"
            ] = True
            with self.assertRaisesRegex(
                contract.ContractError,
                "must be a JSON number",
            ):
                contract.validate_representation_candidate_audit(
                    boolean_prior,
                    stage=stage,
                    epoch=epoch,
                    label="fixture candidate",
                    reprove_paths=False,
                )
            for mutation in ("extra", "missing"):
                broken = copy.deepcopy(audit)
                if mutation == "extra":
                    broken["unexpected"] = True
                else:
                    broken.pop("dataset_receipt_sha256")
                with self.assertRaisesRegex(
                    contract.ContractError,
                    "schema mismatch",
                ):
                    contract.validate_representation_candidate_audit(
                        broken,
                        stage=stage,
                        epoch=epoch,
                        label="fixture candidate",
                        reprove_paths=False,
                    )
            expected_common = {
                key: audit[key]
                for key in (
                    "source_receipt",
                    "source_receipt_sha256",
                    "config_sha256",
                    "lineage_manifest_sha256",
                    "dataset_receipt_sha256",
                    "initialization_receipt",
                    "rvq_ema_prior_receipt",
                    "distributed_training_receipt",
                )
            }
            for key in expected_common:
                broken = copy.deepcopy(audit)
                broken[key] = None
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"{key} binding mismatch",
                ):
                    builder._require_common_audit_binding(
                        broken,
                        expected_common,
                        "candidate/status/final",
                    )

            candidate = {
                "checkpoint": str((root / "face_e200.bin").resolve()),
                "checkpoint_sha256": "f" * 64,
            }
            latest = {
                "path": candidate["checkpoint"],
                "sha256": candidate["checkpoint_sha256"],
                "completed_epochs": 200,
                "optimizer_updates": 99_400,
                "selection_status": "offline_validation_pending",
            }
            builder._validate_latest_candidate_receipt(
                latest,
                stage=stage,
                candidate=candidate,
                label="latest",
            )
            broken_latest = dict(latest)
            broken_latest["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "latest"):
                builder._validate_latest_candidate_receipt(
                    broken_latest,
                    stage=stage,
                    candidate=candidate,
                    label="latest",
                )

            class FakeTensor:
                dtype = "float32"
                shape = (1,)

                def __init__(self, value: float):
                    self.value = value

                def detach(self) -> "FakeTensor":
                    return self

                def cpu(self) -> "FakeTensor":
                    return self

            class FakeTorch:
                @staticmethod
                def is_tensor(value: object) -> bool:
                    return isinstance(value, FakeTensor)

                @staticmethod
                def equal(left: FakeTensor, right: FakeTensor) -> bool:
                    return left.value == right.value

            builder._assert_tensor_states_equal(
                FakeTorch,
                {"weight": FakeTensor(1.0)},
                {"weight": FakeTensor(1.0)},
                "e200/final",
            )
            with self.assertRaisesRegex(RuntimeError, "tensor 'weight'"):
                builder._assert_tensor_states_equal(
                    FakeTorch,
                    {"weight": FakeTensor(1.0)},
                    {"weight": FakeTensor(2.0)},
                    "e200/final",
                )

    def test_portable_source_identity_allows_different_absolute_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = self.training_source_receipt(
                root / "host0",
                "show_base_train.py",
            )
            second = self.training_source_receipt(
                root / "host1",
                "show_base_train.py",
            )
            self.assertNotEqual(first["source_root"], second["source_root"])
            self.assertEqual(
                first["portable_identity"],
                second["portable_identity"],
            )
            contract.validate_frozen_training_source(
                first,
                "host0",
                reprove_checkout=False,
            )
            contract.validate_frozen_training_source(
                second,
                "host1",
                reprove_checkout=False,
            )
            _, _, candidate_index = self.candidate_fixture(root, first)
            unsigned = dict(candidate_index)
            unsigned.pop("receipt_payload_sha256")
            unsigned["source_receipts"] = {
                stage: (first if index % 2 == 0 else second)
                for index, stage in enumerate(contract.STAGES)
            }
            portable_index = contract.receipt_payload(unsigned)
            contract.validate_candidate_index(portable_index)

    def test_canonical_duplicate_clip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (
                rows,
                manifest,
                _,
                summary,
                _,
                lineage,
                _,
            ) = self.canonical_fixture(root)
            rows[1]["clip_id"] = rows[0]["clip_id"]
            manifest_sha = write_jsonl(manifest, rows)
            lineage_payload = json.loads(lineage.read_text())
            lineage_payload.pop("receipt_payload_sha256")
            lineage_payload["manifest_sha256"] = manifest_sha
            lineage_payload = contract.receipt_payload(lineage_payload)
            lineage_sha = write_json(lineage, lineage_payload)
            summary_payload = json.loads(summary.read_text())
            summary_payload.pop("receipt_payload_sha256")
            summary_payload["manifest_sha256"] = manifest_sha
            summary_payload["lineage_sha256"] = lineage_sha
            summary_payload = contract.receipt_payload(summary_payload)
            summary_sha = write_json(summary, summary_payload)
            with self.assertRaisesRegex(
                contract.ContractError,
                "not exact-once",
            ):
                contract.load_val_canonical(
                    manifest_path=manifest,
                    manifest_sha256=manifest_sha,
                    summary_path=summary,
                    summary_sha256=summary_sha,
                    lineage_path=lineage,
                    lineage_sha256=lineage_sha,
                )

    def test_canonical_global_indices_preserve_official_full_manifest_range(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (
                rows,
                manifest,
                manifest_sha,
                summary,
                summary_sha,
                lineage,
                lineage_sha,
            ) = self.canonical_fixture(root)
            loaded, _ = contract.load_val_canonical(
                manifest_path=manifest,
                manifest_sha256=manifest_sha,
                summary_path=summary,
                summary_sha256=summary_sha,
                lineage_path=lineage,
                lineage_sha256=lineage_sha,
            )
            indices = [int(row["global_index"]) for row in loaded]
            self.assertEqual(
                indices,
                list(
                    range(
                        contract.EXPECTED_VAL_GLOBAL_INDEX_START,
                        contract.EXPECTED_VAL_GLOBAL_INDEX_STOP,
                    )
                ),
            )
            partitions = [
                [
                    index
                    for index in indices
                    if index % contract.EXPECTED_SHARDS == shard
                ]
                for shard in range(contract.EXPECTED_SHARDS)
            ]
            self.assertEqual(
                [len(partition) for partition in partitions],
                [215, 215, 214, 214, 214, 214, 214, 215],
            )
            self.assertEqual(
                sorted(
                    index
                    for partition in partitions
                    for index in partition
                ),
                indices,
            )

            rebased = copy.deepcopy(rows)
            for index, row in enumerate(rebased):
                row["global_index"] = index
            manifest_sha, summary_sha, lineage_sha = (
                self.rewrite_canonical_receipts(
                    rebased,
                    manifest,
                    summary,
                    lineage,
                )
            )
            with self.assertRaisesRegex(
                contract.ContractError,
                "receipt binding mismatch",
            ):
                contract.load_val_canonical(
                    manifest_path=manifest,
                    manifest_sha256=manifest_sha,
                    summary_path=summary,
                    summary_sha256=summary_sha,
                    lineage_path=lineage,
                    lineage_sha256=lineage_sha,
                )

    def test_canonical_npz_paths_and_row_windows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (
                rows,
                manifest,
                _,
                summary,
                _,
                lineage,
                _,
            ) = self.canonical_fixture(root)

            def reject(
                mutated_rows: list[dict[str, object]],
                pattern: str,
            ) -> None:
                manifest_sha, summary_sha, lineage_sha = (
                    self.rewrite_canonical_receipts(
                        mutated_rows,
                        manifest,
                        summary,
                        lineage,
                    )
                )
                with self.assertRaisesRegex(contract.ContractError, pattern):
                    contract.load_val_canonical(
                        manifest_path=manifest,
                        manifest_sha256=manifest_sha,
                        summary_path=summary,
                        summary_sha256=summary_sha,
                        lineage_path=lineage,
                        lineage_sha256=lineage_sha,
                    )

            for forbidden in ("test", "Speaker2", "e30"):
                mutated = copy.deepcopy(rows)
                mutated[0]["canonical_npz"] = str(
                    root / forbidden / "must_not_be_opened.npz"
                )
                reject(mutated, "forbidden|test-labelled|Speaker2|e30")

            for index, forbidden in enumerate(("test", "Speaker2", "e30")):
                forbidden_target = root / f"{forbidden}_target"
                forbidden_target.mkdir()
                resolved_file = forbidden_target / "clip.npz"
                resolved_file.write_bytes(b"resolved forbidden\n")
                safe_alias = root / f"safe_alias_{index}"
                safe_alias.symlink_to(
                    forbidden_target,
                    target_is_directory=True,
                )
                mutated = copy.deepcopy(rows)
                mutated[0]["canonical_npz"] = str(
                    safe_alias / resolved_file.name
                )
                reject(
                    mutated,
                    "forbidden|test-labelled|Speaker2|e30",
                )

            safe_file_alias = root / "safe_file_alias.npz"
            safe_file_alias.symlink_to(Path(rows[0]["canonical_npz"]))
            mutated = copy.deepcopy(rows)
            mutated[0]["canonical_npz"] = str(safe_file_alias)
            reject(mutated, "regular non-symlink")

            mutated = copy.deepcopy(rows)
            mutated[0]["canonical_npz"] = "relative/canonical.npz"
            reject(mutated, "must be absolute")

            mutated = copy.deepcopy(rows)
            mutated[0]["frames"] = 30
            reject(mutated, "not val-only")

            mutated = copy.deepcopy(rows)
            mutated[0].pop("canonical_npz")
            reject(mutated, "schema is incomplete")

    def test_limited_training_source_is_enriched_at_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "source"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "remote",
                    "add",
                    "origin",
                    contract.EXPECTED_ORIGIN,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "config",
                    "user.name",
                    "Xiangyue-Zhang",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "config",
                    "user.email",
                    "85532891+Xiangyue-Zhang@users.noreply.github.com",
                ],
                check=True,
            )
            entrypoint = root / "show_base_train.py"
            entrypoint.write_text("# frozen\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(root), "add", "show_base_train.py"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            tree = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            branch = subprocess.run(
                ["git", "-C", str(root), "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "-C", str(root), "checkout", "-q", "--detach", commit],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "branch", "-D", branch],
                check=True,
                capture_output=True,
            )
            raw = {
                "commit": commit,
                "tree": tree,
                "origin": contract.EXPECTED_ORIGIN,
                "entrypoint": str(entrypoint.resolve()),
                "entrypoint_sha256": hashlib.sha256(
                    entrypoint.read_bytes()
                ).hexdigest(),
            }
            with mock.patch.object(
                contract,
                "OFFICIAL_BASELINE_COMMIT",
                commit,
            ):
                frozen = contract.freeze_training_audit_source(
                    raw,
                    "fixture",
                )
                self.assertTrue(frozen["clean"])
                self.assertTrue(frozen["detached"])
                self.assertEqual(frozen["local_branch_count"], 0)
                self.assertEqual(
                    frozen["official_baseline_commit"],
                    commit,
                )
                self.assertTrue(
                    frozen["official_baseline_is_ancestor"]
                )
                self.assertEqual(set(frozen["training_audit"]), set(raw))
                self.assertEqual(
                    set(frozen["portable_identity"]),
                    {
                        "origin",
                        "commit",
                        "tree",
                        "script_relative",
                        "script_sha256",
                    },
                )
                contract.validate_frozen_training_source(
                    frozen,
                    "fixture",
                    reprove_checkout=True,
                )
                entrypoint.write_text("# dirty\n", encoding="utf-8")
                with self.assertRaises(contract.ContractError):
                    contract.freeze_training_audit_source(
                        raw,
                        "fixture",
                    )

    def test_frozen_training_source_rejects_wrong_official_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frozen = self.training_source_receipt(
                root,
                "show_base_train.py",
            )
            frozen["official_baseline_commit"] = "9" * 40
            frozen.pop("receipt_payload_sha256")
            frozen = contract.receipt_payload(frozen)
            with self.assertRaisesRegex(
                contract.ContractError,
                "freeze binding mismatch",
            ):
                contract.validate_frozen_training_source(
                    frozen,
                    "attacked",
                    reprove_checkout=False,
                )

    def test_launcher_has_disjoint_two_host_partition_contract(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "show_base"
            / "run_prerequisite_val_8shard.sh"
        )
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        source = launcher.read_text(encoding="utf-8")
        for token in (
            "SEMTALK_PREREQ_VAL_PARTITION",
            "0of2",
            "1of2",
            "current_plan_index % partition_modulus",
            "SEMTALK_PREREQ_VAL_CANDIDATES_PER_WAVE",
            "SEMTALK_PREREQ_VAL_MULTICANDIDATE_GATE_RECEIPT",
            "nonglobal|global|gate",
            "PARTIAL_CANDIDATE_INDEX_FORMAT",
            'if [[ "$partition" != all ]]',
            "--shard-root",
        ):
            self.assertIn(token, source)
        self.assertNotIn("git push", source)
        self.assertNotIn("git checkout -b", source)
        self.assertNotIn(
            "semtalk_show_prerequisite_nonglobal_candidate_index_v1",
            source,
        )
        self.assertIn(
            "partial_candidate_index_format=${candidate_index_formats[1]}",
            source,
        )

    def test_launcher_distinguishes_legacy_and_segmented_plan_shape_policy(
        self,
    ) -> None:
        helper = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "show_base"
            / "prerequisite_val_launcher_contract.sh"
        )
        subprocess.run(["bash", "-n", str(helper)], check=True)

        def authority(actual: str) -> str:
            command = (
                f"source {shlex.quote(str(helper))}; "
                "semtalk_prereq_candidate_index_authority "
                f"{shlex.quote(actual)} legacy-partial segmented-partial "
                "legacy-complete segmented-complete"
            )
            return subprocess.run(
                ["bash", "-c", command],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertEqual(
            authority("legacy-partial"),
            "partial\t40\t4\tface",
        )
        self.assertEqual(
            authority("segmented-partial"),
            "partial\t40\t0\tface",
        )
        self.assertEqual(
            authority("legacy-complete"),
            "complete\t50\t5\tglobal",
        )
        self.assertEqual(
            authority("segmented-complete"),
            "complete\t50\t0\tglobal",
        )

    def test_launcher_accepts_segmented_heterogeneous_plan_counts(
        self,
    ) -> None:
        helper = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "show_base"
            / "prerequisite_val_launcher_contract.sh"
        )

        def plan_is_accepted(actual: str, jobs: int) -> bool:
            policy = (
                "semtalk_prereq_candidate_index_authority "
                f"{shlex.quote(actual)} legacy-partial segmented-partial "
                "legacy-complete segmented-complete"
            )
            command = (
                f"source {shlex.quote(str(helper))}; "
                "IFS=$'\\t' read -r _ minimum modulus _ "
                f"<<<\"$({policy})\"; "
                "semtalk_prereq_plan_job_count_valid "
                f"{jobs} \"$minimum\" \"$modulus\""
            )
            return (
                subprocess.run(
                    ["bash", "-c", command],
                    check=False,
                    capture_output=True,
                    text=True,
                ).returncode
                == 0
            )

        self.assertFalse(plan_is_accepted("legacy-partial", 41))
        self.assertTrue(plan_is_accepted("segmented-partial", 41))
        self.assertFalse(plan_is_accepted("legacy-complete", 51))
        self.assertTrue(plan_is_accepted("segmented-complete", 51))

    def test_source_policy_accepts_only_global_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve() / "source"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Fixture"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "config",
                    "user.email",
                    "fixture@example.invalid",
                ],
                check=True,
            )
            entrypoint = repo / "show_base_train.py"
            entrypoint.write_text("# rvq\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "show_base_train.py"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "rvq"], check=True)
            rvq_commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            rvq_tree = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True
            ).strip()
            entrypoint.write_text("# global fix\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "show_base_train.py"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "global"], check=True)
            global_commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            global_tree = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True
            ).strip()

            def frozen(commit: str, tree: str) -> dict[str, object]:
                identity = {
                    "origin": contract.EXPECTED_ORIGIN,
                    "commit": commit,
                    "tree": tree,
                    "script_relative": "show_base_train.py",
                    "script_sha256": "1" * 64,
                }
                return {
                    "source_root": str(repo),
                    "portable_identity": identity,
                }

            sources = {
                stage: frozen(rvq_commit, rvq_tree)
                for stage in contract.RVQ_STAGES
            }
            sources["global"] = frozen(global_commit, global_tree)
            policy = contract.build_source_policy(
                sources,
                stages=contract.STAGES,
                reprove_ancestry=True,
            )
            self.assertTrue(policy["global_descends_from_rvq"])

            reverse = {
                stage: frozen(global_commit, global_tree)
                for stage in contract.RVQ_STAGES
            }
            reverse["global"] = frozen(rvq_commit, rvq_tree)
            with self.assertRaisesRegex(
                contract.ContractError,
                "not an RVQ-source descendant",
            ):
                contract.build_source_policy(
                    reverse,
                    stages=contract.STAGES,
                    reprove_ancestry=True,
                )

            mixed = copy.deepcopy(sources)
            mixed["hands"] = frozen(global_commit, global_tree)
            with self.assertRaisesRegex(
                contract.ContractError,
                "RVQ identities differ",
            ):
                contract.build_source_policy(
                    mixed,
                    stages=contract.STAGES,
                    reprove_ancestry=False,
                )

    def test_optimizer_runtime_receipt_is_stage_specific(self) -> None:
        payload = {
            "format": "semtalk_show_optimizer_runtime_v1",
            "formal_stage": "global",
            "class": "torch.optim.Adam",
            "base_learning_rate": 1.5e-4,
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "eps": 1e-8,
            "amsgrad": False,
            "parameter_groups": 1,
            "trained_parameter_tensors": 12,
        }
        payload["receipt_sha256"] = contract.canonical_payload_sha256(
            payload
        )
        self.assertEqual(
            contract.validate_optimizer_runtime_receipt(
                payload,
                stage="global",
                label="global optimizer",
                required=True,
            ),
            payload,
        )
        wrong_lr = dict(payload)
        wrong_lr["base_learning_rate"] = 6e-4
        wrong_lr.pop("receipt_sha256")
        wrong_lr["receipt_sha256"] = contract.canonical_payload_sha256(
            wrong_lr
        )
        with self.assertRaisesRegex(
            contract.ContractError,
            "optimizer protocol mismatch",
        ):
            contract.validate_optimizer_runtime_receipt(
                wrong_lr,
                stage="global",
                label="global optimizer",
                required=True,
            )
        with self.assertRaisesRegex(contract.ContractError, "is required"):
            contract.validate_optimizer_runtime_receipt(
                None,
                stage="global",
                label="global optimizer",
                required=True,
            )
        self.assertIsNone(
            contract.validate_optimizer_runtime_receipt(
                None,
                stage="face",
                label="legacy RVQ optimizer",
                required=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
