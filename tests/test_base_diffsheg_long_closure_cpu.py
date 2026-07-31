from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_long_val_contract as contract
from scripts.show_base import build_base_diffsheg_val_inputs as inputs
from scripts.show_base import produce_base_val_measurement as producer
from scripts.show_base import select_base_official_adapt as legacy
from scripts.show_base import select_base_official_adapt_long as selector


class BaseDiffSHEGLongClosureTests(unittest.TestCase):
    def test_long_profile_is_one_coherent_diffsheg_producer_abi(self) -> None:
        profile = selector._profile_values()
        self.assertIs(profile["validate_val_inputs"], contract.validate_val_inputs)
        self.assertIs(profile["validate_pipeline"], contract.validate_pipeline)
        self.assertIs(
            profile["validate_val_inference_lineage"],
            contract.validate_val_inference_lineage,
        )
        self.assertEqual(
            profile["VAL_INFERENCE_LINEAGE_FORMAT"],
            legacy.VAL_INFERENCE_LINEAGE_FORMAT,
        )

    def test_measurement_producer_is_val_only_and_binds_one_candidate(
        self,
    ) -> None:
        epoch = 200
        checkpoint = {
            "path": "/validation/base-e0200.bin",
            "sha256": "a" * 64,
            "bytes": 17,
        }
        artifacts = {
            role: {
                "path": f"/validation/{role}.json",
                "sha256": character * 64,
                "receipt_payload_sha256": character.upper().lower() * 64,
            }
            for role, character in (
                ("val", "b"),
                ("pipeline", "c"),
                ("lineage", "d"),
                ("report", "e"),
            )
        }
        artifacts["report"].pop("receipt_payload_sha256")
        measurement = producer.build_measurement(
            epoch=epoch,
            candidate_bundle={"candidates": {epoch: checkpoint}},
            val_inputs_artifact=artifacts["val"],
            pipeline_artifact=artifacts["pipeline"],
            inference_lineage_artifact=artifacts["lineage"],
            diffsheg_report_artifact=artifacts["report"],
        )
        claimed = measurement.pop("receipt_payload_sha256")
        self.assertEqual(legacy.canonical_json_sha256(measurement), claimed)
        self.assertEqual(measurement["split"], "val")
        self.assertFalse(measurement["test_visible"])
        self.assertTrue(measurement["selection_eligible"])
        self.assertEqual(
            measurement["candidate_checkpoint"],
            {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
        )
        self.assertNotIn("released2", str(measurement).casefold())

    def test_withdrawn_e30_is_not_a_producer_epoch(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            producer._epoch("30")

    def test_diffsheg_input_builder_freezes_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            canonical = root / "canonical.jsonl"
            canonical.write_bytes(b"canonical")
            args = argparse.Namespace(
                canonical_manifest=canonical,
                canonical_summary=root / "summary.json",
                canonical_lineage=root / "lineage.json",
                audio_manifest=[root / f"audio-{i}.jsonl" for i in range(8)],
                audio_summary=[root / f"audio-{i}-summary.json" for i in range(8)],
                audio_lineage=[root / f"audio-{i}-lineage.json" for i in range(8)],
                output=root / "diffsheg-val-inputs.json",
            )
            artifact_counter = 0

            def fake_artifact(path: Path, _label: str):
                nonlocal artifact_counter
                artifact_counter += 1
                return (
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(str(path).encode()).hexdigest(),
                    },
                    path,
                    b"canonical" if path == canonical else b"receipt",
                )

            captured: dict[str, object] = {}

            def fake_atomic(path, receipt, *, label, validator):
                captured.update(receipt)
                self.assertEqual(label, "DiffSHEG validation inputs receipt")
                self.assertIs(validator, legacy.validate_val_inputs)
                return path.resolve(), "f" * 64

            with (
                mock.patch.object(inputs.common, "_artifact", side_effect=fake_artifact),
                mock.patch.object(
                    legacy,
                    "_canonical_coverage",
                    return_value=(
                        {f"clip-{i}" for i in range(legacy.EXPECTED_VAL_CLIPS)},
                        {
                            "clip_count": legacy.EXPECTED_VAL_CLIPS,
                            "clip_ids_sha256": "1" * 64,
                            "diffsheg_clip_manifest_sha256": "2" * 64,
                        },
                    ),
                ),
                mock.patch.object(
                    legacy,
                    "_validate_val_canonical_receipts",
                    return_value=({"summary": True}, {"lineage": True}),
                ),
                mock.patch.object(
                    legacy,
                    "_audio_coverage",
                    return_value={
                        "manifests": [{"shard": i} for i in range(8)],
                        "summaries": [{"shard": i} for i in range(8)],
                        "lineages": [{"shard": i} for i in range(8)],
                        "num_shards": 8,
                    },
                ),
                mock.patch.object(inputs.common, "_atomic_new_json", side_effect=fake_atomic),
            ):
                result = inputs.build_inputs(args)
            self.assertEqual(artifact_counter, 27)
            self.assertEqual(captured["format"], legacy.VAL_INPUTS_FORMAT)
            self.assertEqual(captured["split"], "val")
            self.assertFalse(captured["test_visible"])
            self.assertEqual(
                captured["diffsheg_clip_manifest_sha256"],
                "2" * 64,
            )
            self.assertEqual(result["audio_shards"], 8)


if __name__ == "__main__":
    unittest.main()
