from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.show_base import build_base_val_receipts as builder
from scripts.show_base import talkshow_base_val_contract as selector
from utils import show_official_transfer as transfer_contract


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    encoded = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _with_payload_hash(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["receipt_payload_sha256"] = selector.canonical_json_sha256(result)
    return result


class BaseValReceiptBuilderTests(unittest.TestCase):
    def _build_fresh_source_fixture(self, root: Path) -> Path:
        source_root = (root / "fresh-source").resolve()
        for relative in selector.FRESH_PIPELINE_SOURCE_FILES:
            source = ROOT / relative
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

        def git(*arguments: str) -> str:
            process = subprocess.run(
                ["git", "-C", str(source_root), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return process.stdout.strip()

        git("init", "--quiet")
        git(
            "remote",
            "add",
            "origin",
            "git@github.com:Xiangyue-Zhang/SemTalk.git",
        )
        git("add", "--", *selector.FRESH_PIPELINE_SOURCE_FILES)
        git(
            "-c",
            "user.name=Xiangyue-Zhang",
            "-c",
            "user.email=85532891+Xiangyue-Zhang@users.noreply.github.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fresh source fixture",
        )
        branch = git("symbolic-ref", "--short", "HEAD")
        commit = git("rev-parse", "HEAD")
        git("checkout", "--quiet", "--detach", commit)
        git("branch", "-D", branch)
        return source_root

    def _build_input_fixture(
        self,
        root: Path,
    ) -> argparse.Namespace:
        canonical_rows: list[dict[str, object]] = []
        speakers = tuple(selector.SHOW_SPEAKER_IDS)
        for index in range(selector.EXPECTED_VAL_CLIPS):
            canonical_rows.append(
                {
                    "global_index": index,
                    "clip_id": (
                        f"{speakers[index % len(speakers)]}/"
                        f"video-{index}/sequence-{index}"
                    ),
                    "split": "val",
                    "frames": 88,
                }
            )
        canonical = root / "canonical-val.jsonl"
        canonical_sha = _write_jsonl(canonical, canonical_rows)
        lineage = root / "canonical-lineage.json"
        lineage_sha = _write_json(
            lineage,
            _with_payload_hash(
                {
                    "format": selector.VAL_CANONICAL_LINEAGE_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "clip_count": selector.EXPECTED_VAL_CLIPS,
                    "manifest_sha256": canonical_sha,
                    "lineage_contract_sha256": "9" * 64,
                    "projection": {
                        "operation": "filter_exact_split",
                        "split": "val",
                        "test_rows_materialized": False,
                    },
                    "source_receipt": {
                        "origin": selector.BASE_PRODUCER_SOURCE["origin"],
                        "commit": "4" * 40,
                        "tree": "5" * 40,
                        "full_manifest_sha256": "6" * 64,
                        "full_summary_sha256": "7" * 64,
                        "full_lineage_sha256": "8" * 64,
                    },
                }
            ),
        )
        summary = root / "canonical-summary.json"
        _write_json(
            summary,
            _with_payload_hash(
                {
                    "format": selector.VAL_CANONICAL_SUMMARY_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "clip_count": selector.EXPECTED_VAL_CLIPS,
                    "manifest_sha256": canonical_sha,
                    "lineage_sha256": lineage_sha,
                    "lineage_contract_sha256": "9" * 64,
                }
            ),
        )
        audio_root = root / "audio"
        audio_root.mkdir()
        audio_artifacts: dict[str, tuple[Path, str]] = {}
        for row in canonical_rows:
            clip_id = str(row["clip_id"])
            audio_path = (
                audio_root
                / f"{hashlib.sha256(clip_id.encode()).hexdigest()}.npz"
            )
            audio_path.write_bytes(clip_id.encode())
            audio_artifacts[clip_id] = (
                audio_path,
                builder._sha256_file(audio_path),
            )
        manifests: list[Path] = []
        summaries: list[Path] = []
        lineages: list[Path] = []
        for shard in range(selector.EXPECTED_AUDIO_SHARDS):
            rows = []
            for index, canonical_row in enumerate(canonical_rows):
                if index % selector.EXPECTED_AUDIO_SHARDS != shard:
                    continue
                clip_id = str(canonical_row["clip_id"])
                audio_path, audio_sha = audio_artifacts[clip_id]
                rows.append(
                    {
                        "format": "semtalk_show_audio_clip_v1",
                        "split": "val",
                        "clip_id": clip_id,
                        "shard_id": shard,
                        "num_shards": selector.EXPECTED_AUDIO_SHARDS,
                        "audio_feature_npz": str(audio_path),
                        "audio_feature_npz_sha256": audio_sha,
                        "frames": 88,
                        "beat_shape": [88, 3],
                        "hubert_shape": [88, 1024],
                    }
                )
            manifest = root / f"audio-{shard}.jsonl"
            manifest_sha = _write_jsonl(manifest, rows)
            shard_summary = root / f"audio-{shard}.summary.json"
            _write_json(
                shard_summary,
                {
                    "format": "semtalk_show_audio_summary_v1",
                    "status": "complete",
                    "shard_id": shard,
                    "num_shards": selector.EXPECTED_AUDIO_SHARDS,
                    "full_split_clips": selector.EXPECTED_VAL_CLIPS,
                    "shard_clips": len(rows),
                    "output_manifest_sha256": manifest_sha,
                },
            )
            shard_lineage = root / f"audio-{shard}.lineage.json"
            _write_json(
                shard_lineage,
                {
                    "format": "semtalk_show_audio_lineage_v1",
                    "status": "complete",
                    "shard_id": shard,
                    "num_shards": selector.EXPECTED_AUDIO_SHARDS,
                    "full_split_clips": selector.EXPECTED_VAL_CLIPS,
                    "shard_clips": len(rows),
                    "output_manifest_sha256": manifest_sha,
                    "protocol": {"split": "val"},
                },
            )
            manifests.append(manifest)
            summaries.append(shard_summary)
            lineages.append(shard_lineage)
        return argparse.Namespace(
            output=root / "val-inputs.json",
            canonical_manifest=canonical,
            canonical_summary=summary,
            canonical_lineage=lineage,
            audio_manifest=manifests,
            audio_summary=summaries,
            audio_lineage=lineages,
        )

    def test_inputs_builds_exact_receipt_and_revalidates_immediately(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            args = self._build_input_fixture(root)
            report = builder.build_inputs(args)
            self.assertEqual(report["clip_count"], 1_715)
            self.assertEqual(report["audio_shards"], 8)
            artifact, coverage = selector.validate_val_inputs(
                args.output,
                report["sha256"],
            )
            self.assertEqual(artifact["path"], str(args.output))
            self.assertEqual(coverage["audio"]["clip_count"], 1_715)
            receipt = json.loads(args.output.read_text())
            self.assertFalse(receipt["test_visible"])
            self.assertEqual(
                [Path(item["path"]).name for item in receipt["audio_manifests"]],
                [f"audio-{shard}.jsonl" for shard in range(8)],
            )
            with self.assertRaises(FileExistsError):
                builder.build_inputs(args)

    def _write_transfer(
        self,
        root: Path,
        *,
        stage: str,
        totals: list[float],
    ) -> tuple[Path, Path]:
        metrics_path = root / f"{stage}-metrics.jsonl"
        rows = [
            {
                "epoch": epoch,
                "train": {"total": total + 0.2},
                "val": {"total": total},
                "elapsed_seconds": float(epoch),
                "rank_count": 8,
                "finite": True,
            }
            for epoch, total in enumerate(totals, 1)
        ]
        metrics_sha = _write_jsonl(metrics_path, rows)
        best_total, best_epoch = min(
            (total, epoch)
            for epoch, total in enumerate(totals, 1)
        )
        source_path = ROOT / "scripts/show_base/train_official_transfer.py"
        source = {
            "origin": builder.EXPECTED_TRANSFER_SOURCE["origin"],
            "commit": builder.EXPECTED_TRANSFER_SOURCE["commit"],
            "tree": builder.EXPECTED_TRANSFER_SOURCE["tree"],
            "branch": None,
            "clean": True,
            "entrypoint": str(source_path),
            "entrypoint_sha256": builder.EXPECTED_TRANSFER_SOURCE[
                "entrypoint_sha256"
            ],
        }
        protocol = {
            "stage": stage,
            "selection_split": "val",
            "test_visible": False,
        }
        shared = {
            "official_initialization": {"stage": stage},
            "source_receipt": source,
            "cache_receipt": {"stage": stage},
            "trainable_policy": {"stage": stage},
            "protocol": protocol,
            "frozen_encoder_quantizer_sha256": "c" * 64,
        }
        transfer = {
            "format": builder.TRANSFER_FORMAT,
            "stage": stage,
            "epoch": best_epoch,
            **shared,
            "validation": {"total": best_total},
            "model_state_sha256": "d" * 64,
            "withdrawn_e30_allowed": False,
            "test_visible": False,
        }
        transfer["receipt_sha256"] = (
            transfer_contract.canonical_payload_sha256(transfer)
        )
        checkpoint = root / f"best_{stage}_transfer.bin"
        checkpoint.write_bytes(
            pickle.dumps(
                {
                    "model_state": {},
                    "optimizer_state": {},
                    "epoch": best_epoch,
                    "transfer_receipt": transfer,
                }
            )
        )
        summary = {
            "format": builder.TRANSFER_FORMAT,
            "status": "complete",
            "stage": stage,
            "epochs": len(totals),
            "best_epoch": best_epoch,
            "best_validation_total": best_total,
            **shared,
            "withdrawn_e30_allowed": False,
            "test_visible": False,
            "metrics_jsonl": str(metrics_path),
            "metrics_jsonl_sha256": metrics_sha,
            "elapsed_seconds": float(len(totals)),
        }
        summary["receipt_sha256"] = (
            transfer_contract.canonical_payload_sha256(summary)
        )
        summary_path = root / f"{stage}-summary.json"
        _write_json(summary_path, summary)
        return checkpoint, summary_path

    def test_historical_pipeline_builder_is_retired(self) -> None:
        with self.assertRaisesRegex(
            builder.ReceiptBuildError,
            "historical pipeline builder is retired",
        ):
            builder.build_pipeline(argparse.Namespace())

    def test_fresh_pipeline_source_accepts_argparse_path(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            source_root = self._build_fresh_source_fixture(Path(directory))
            parsed = builder.build_parser().parse_args(
                [
                    "fresh-pipeline",
                    "--output",
                    str(Path(directory) / "fresh-pipeline.json"),
                    "--source-root",
                    str(source_root),
                    "--prerequisite-selection",
                    str(Path(directory) / "selection.json"),
                    "--expected-prerequisite-selection-sha256",
                    "a" * 64,
                ]
            )
            self.assertIsInstance(parsed.source_root, Path)
            receipt = selector.build_fresh_pipeline_source_receipt(
                parsed.source_root
            )
            self.assertEqual(receipt["source_root"], str(source_root))
            self.assertEqual(
                set(receipt["files"]),
                set(selector.FRESH_PIPELINE_SOURCE_FILES),
            )


    def test_rejects_symlink_and_forbidden_labels(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            target = root / "canonical-val.jsonl"
            target.write_bytes(b"")
            symlink = root / "canonical-link.jsonl"
            symlink.symlink_to(target)
            with self.assertRaises(selector.SelectionContractError):
                builder._artifact(symlink, "canonical")
            forbidden = root / "speaker2"
            forbidden.mkdir()
            with self.assertRaises(selector.SelectionContractError):
                builder._prepare_new_output(
                    forbidden / "receipt.json",
                    "receipt",
                )
            withdrawn = root / "e30"
            withdrawn.mkdir()
            with self.assertRaises(selector.SelectionContractError):
                builder._prepare_new_output(
                    withdrawn / "receipt.json",
                    "receipt",
                )


if __name__ == "__main__":
    unittest.main()
