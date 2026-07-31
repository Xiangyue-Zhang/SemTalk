from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


from scripts.show_base import merge_prerequisite_val_shards as merger
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
                "origin": contract.EXPECTED_ORIGIN,
                "commit": "e" * 40,
                "tree": "f" * 40,
                "clean": True,
                "detached": True,
                "local_branch_count": 0,
                "entrypoint_relative": name,
                "entrypoint_sha256": digest,
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
        rows: list[dict[str, object]] = [
            {
                "global_index": index,
                "clip_id": f"oliver/video-{index}/sequence-{index}",
                "split": "val",
                "frames": 90,
                "canonical_npz_sha256": f"{index:064x}"[-64:],
            }
            for index in range(contract.EXPECTED_VAL_CLIPS)
        ]
        manifest = root / "canonical" / "manifest.jsonl"
        manifest_sha = write_jsonl(manifest, rows)
        lineage = root / "canonical" / "lineage.json"
        lineage_payload = contract.receipt_payload(
            {
                "format": contract.VAL_CANONICAL_LINEAGE_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "clip_count": contract.EXPECTED_VAL_CLIPS,
                "manifest_sha256": manifest_sha,
            }
        )
        lineage_sha = write_json(lineage, lineage_payload)
        summary = root / "canonical" / "summary.json"
        summary_payload = contract.receipt_payload(
            {
                "format": contract.VAL_CANONICAL_SUMMARY_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "clip_count": contract.EXPECTED_VAL_CLIPS,
                "manifest_sha256": manifest_sha,
                "lineage_sha256": lineage_sha,
            }
        )
        summary_sha = write_json(summary, summary_payload)
        return (
            rows,
            manifest,
            manifest_sha,
            summary,
            summary_sha,
            lineage,
            lineage_sha,
        )

    def candidate_fixture(
        self,
        root: Path,
        source: dict[str, object],
    ) -> tuple[Path, str, dict[str, object]]:
        stages: dict[str, list[dict[str, object]]] = {}
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
                            epoch * contract.EXPECTED_UPDATES_PER_EPOCH
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
                "updates_per_epoch": contract.EXPECTED_UPDATES_PER_EPOCH,
                "source_receipts": {
                    stage: dict(source) for stage in contract.STAGES
                },
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
            source_root,
            "evaluate_prerequisite_val_shard.py",
        )
        merge_source = self.source_receipt(
            source_root,
            "merge_prerequisite_val_shards.py",
        )
        selector_source = self.source_receipt(
            source_root,
            "select_prerequisite_candidates.py",
        )
        training_source = self.training_source_receipt(
            source_root,
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
            root = Path(temporary)
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
                "exactly one shard root",
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
                "exactly one shard root",
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.training_source_receipt(
                root / "source",
                "show_base_train.py",
            )
            _, _, payload = self.candidate_fixture(root, source)
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

    def test_canonical_duplicate_clip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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

    def test_limited_training_source_is_enriched_at_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
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
            frozen = contract.freeze_training_audit_source(raw, "fixture")
            self.assertTrue(frozen["clean"])
            self.assertTrue(frozen["detached"])
            self.assertEqual(frozen["local_branch_count"], 0)
            self.assertEqual(set(frozen["training_audit"]), set(raw))
            contract.validate_frozen_training_source(
                frozen,
                "fixture",
                reprove_checkout=True,
            )
            entrypoint.write_text("# dirty\n", encoding="utf-8")
            with self.assertRaises(contract.ContractError):
                contract.freeze_training_audit_source(raw, "fixture")

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
            'if [[ "$partition" != all ]]',
            "--shard-root",
        ):
            self.assertIn(token, source)
        self.assertNotIn("git push", source)
        self.assertNotIn("git checkout -b", source)


if __name__ == "__main__":
    unittest.main()
