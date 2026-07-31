from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import build_prerequisite_continuation_stage_plans as builder
from scripts.show_base import prerequisite_continuation_runtime as runtime
from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract
from tests.test_prerequisite_continuation_wave_cpu import authorize, fixture


class ContinuationStagePlanBuilderTests(unittest.TestCase):
    def test_recursive_old_plan_is_derived_from_predecessor(self) -> None:
        previous = authorize(fixture(boundary=200))
        binding = {
            "path": "/formal/waves/e220.json",
            "sha256": "a" * 64,
            "receipt_payload_sha256": previous["receipt_payload_sha256"],
        }
        old = builder._later_wave_old(
            stage="face",
            boundary=220,
            predecessor=previous,
            predecessor_binding=binding,
        )
        self.assertEqual(old["old_run_path"], "/formal/new/e220/face")
        self.assertEqual(
            [segment["candidate_epochs"] for segment in old["candidate_segment_chain"]],
            [list(range(20, 201, 20)), [220]],
        )
        self.assertEqual(old["predecessor_wave"], binding)
        with self.assertRaisesRegex(builder.StagePlanBuildError, "boundary"):
            builder._later_wave_old(
                stage="face",
                boundary=240,
                predecessor=previous,
                predecessor_binding=binding,
            )

    def test_printed_config_receipt_is_recomputed_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "config.json"
            snapshot = {
                "epochs": 220,
                "run_name": "continuation_global",
                "final_ckpt_name": "show_ft_global_220.bin",
                "batch_size": 64,
            }
            source = {"origin": contract.EXPECTED_ORIGIN, "marker": "source"}
            value = {
                "format": runtime.CONFIG_RECEIPT_FORMAT,
                "formal_stage": "global",
                "hostname": sorted(wave.FORMAL_HOSTS)[0],
                "world_size": 1,
                "smplx_asset": None,
                "config_snapshot": snapshot,
                "config_sha256": hashlib.sha256(
                    json.dumps(
                        snapshot,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode()
                ).hexdigest(),
                "config_semantic_sha256": runtime.config_semantic_sha256(snapshot),
                "source_receipt": source,
                "source_receipt_sha256": contract.canonical_payload_sha256(source),
            }
            value["receipt_payload_sha256"] = contract.canonical_payload_sha256(value)
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            digest = contract.sha256_file(path)
            with mock.patch.object(
                contract,
                "validate_training_audit_source",
                return_value=source,
            ):
                loaded = builder._load_config(
                    path, digest, stage="global", label="global config"
                )
            self.assertEqual(loaded["config_snapshot"], snapshot)

            value["config_sha256"] = "f" * 64
            value["receipt_payload_sha256"] = contract.canonical_payload_sha256(
                {key: child for key, child in value.items() if key != "receipt_payload_sha256"}
            )
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            with mock.patch.object(
                contract,
                "validate_training_audit_source",
                return_value=source,
            ), self.assertRaisesRegex(builder.StagePlanBuildError, "payload mismatch"):
                builder._load_config(
                    path,
                    contract.sha256_file(path),
                    stage="global",
                    label="global config",
                )


if __name__ == "__main__":
    unittest.main()
