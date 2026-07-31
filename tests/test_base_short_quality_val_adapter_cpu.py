from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.show_base import base_short_quality_val_adapter as ADAPTER
from scripts.show_base import run_base_val_inference as FORMAL


class BaseShortQualityValAdapterCpuTest(unittest.TestCase):
    def test_cli_is_val_only_and_exactly_e1_e2_e4_e8(self) -> None:
        common = [
            "shard",
            "--split",
            "val",
            "--preflight",
            "/frozen/val/preflight.json",
            "--expected-preflight-sha256",
            "a" * 64,
            "--epoch",
            "1",
            "--output-root",
            "/formal/val/e1",
            "--num-shards",
            "8",
            "--shard-id",
            "0",
            "--device",
            "cuda:0",
        ]
        self.assertEqual(ADAPTER.parse_args(common).epoch, 1)
        for epoch in ("2", "4", "8"):
            argv = list(common)
            argv[argv.index("1")] = epoch
            self.assertEqual(ADAPTER.parse_args(argv).epoch, int(epoch))
        for epoch in ("3", "25", "400"):
            argv = list(common)
            argv[argv.index("1")] = epoch
            with self.assertRaises(SystemExit):
                ADAPTER.parse_args(argv)
        argv = list(common)
        argv[argv.index("val")] = "test"
        with self.assertRaises(SystemExit):
            ADAPTER.parse_args(argv)

    def test_ready_receipts_require_exact_order_and_lowercase_sha(self) -> None:
        values = []
        with mock.patch.object(Path, "read_bytes") as read_bytes:
            read_bytes.return_value = json.dumps(
                {"receipt_payload_sha256": "b" * 64}
            ).encode()
            for epoch in ADAPTER.QUALITY_EPOCHS:
                values.append((str(epoch), f"/frozen/val/e{epoch}.json", "a" * 64))
            result = ADAPTER._normalize_ready(values)
            self.assertEqual(len(result), 4)
            with self.assertRaises(ADAPTER.ShortQualityValError):
                ADAPTER._normalize_ready(list(reversed(values)))
            bad = list(values)
            bad[0] = ("1", "/frozen/val/e1.json", "A" * 64)
            with self.assertRaises(ADAPTER.ShortQualityValError):
                ADAPTER._normalize_ready(bad)

    def test_selected_five_and_selection_authority_are_exact(self) -> None:
        stages = ("face", "hands", "upper", "lower", "global")
        selected = {stage: str(index + 1) * 64 for index, stage in enumerate(stages)}
        pipeline = {
            "fixed_checkpoints": {
                stage: {"sha256": selected[stage]} for stage in stages
            },
            "prerequisite_selection": {"sha256": "f" * 64},
        }
        frozen = {
            "dataset": {
                "selected_prerequisite_sha256": selected,
                "prerequisite_selection": {"sha256": "f" * 64},
            }
        }
        self.assertEqual(ADAPTER._selected_five(frozen, pipeline), selected)
        frozen["dataset"]["selected_prerequisite_sha256"] = {
            **selected,
            "face": "0" * 64,
        }
        with self.assertRaises(ADAPTER.ShortQualityValError):
            ADAPTER._selected_five(frozen, pipeline)

    def test_engine_adapter_is_process_local_and_always_restored(self) -> None:
        original = FORMAL._preflight_artifact
        with ADAPTER._engine_adapter():
            self.assertIs(FORMAL._preflight_artifact, ADAPTER._preflight_artifact)
        self.assertIs(FORMAL._preflight_artifact, original)
        with self.assertRaisesRegex(RuntimeError, "boom"), ADAPTER._engine_adapter():
            raise RuntimeError("boom")
        self.assertIs(FORMAL._preflight_artifact, original)

    def test_distribution_rejects_gate_fixed_five_or_source_mismatch(self) -> None:
        checkpoint = {
            "path": "/frozen/val/base.bin",
            "sha256": "a" * 64,
            "bytes": 10,
        }
        stages = ("face", "hands", "upper", "lower", "global")
        pipeline = {
            "source": {"commit": "1" * 40, "tree": "2" * 40},
            "fixed_checkpoints": {
                stage: {
                    "path": f"/frozen/val/{stage}.bin",
                    "sha256": str(index + 1) * 64,
                    "bytes": index + 1,
                }
                for index, stage in enumerate(stages)
            },
        }
        expected = {
            "base": checkpoint,
            **{stage: dict(pipeline["fixed_checkpoints"][stage]) for stage in stages},
        }
        gate = {
            "model_bundle": {
                "checkpoints": {
                    **expected,
                    "face": {**expected["face"], "sha256": "0" * 64},
                }
            },
            "source_closure": {"commit": "1" * 40, "tree": "2" * 40},
        }
        args = SimpleNamespace(
            preflight=Path("/frozen/val/preflight.json"),
            expected_preflight_sha256="b" * 64,
            epoch=1,
            validation_gate=Path("/frozen/val/gate.json"),
            expected_validation_gate_sha256="c" * 64,
            lineage=Path("/does/not/get/read.json"),
        )
        preflight = {
            "candidate_bundle": {"candidates": {"1": checkpoint}},
            "pipeline_receipt": {
                "path": "/frozen/val/pipeline.json",
                "sha256": "d" * 64,
            },
        }
        with (
            mock.patch.object(
                ADAPTER,
                "_preflight_artifact",
                return_value=({}, preflight),
            ),
            mock.patch.object(
                ADAPTER.replication,
                "load_gate",
                return_value=(
                    {"path": "/frozen/val/gate.json", "sha256": "c" * 64},
                    gate,
                ),
            ),
            mock.patch.object(
                ADAPTER.engine.selector,
                "validate_pipeline",
                return_value=({}, pipeline),
            ),
            self.assertRaisesRegex(
                ADAPTER.ShortQualityValError,
                "Base plus selected five/source",
            ),
        ):
            ADAPTER.distribution(args)

    def test_prepare_schema_is_isolated_from_formal_preflight(self) -> None:
        body = {
            "format": ADAPTER.FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "candidate_epochs": [1, 2, 4, 8],
            "candidate_bundle": {},
            "val_inputs_receipt": {},
            "pipeline_receipt": {},
            "pipeline_source": {},
            "inference_entrypoint": {},
            "coverage": {},
            "short_quality_authority": {},
            "adapter_source": {},
            "quality_input_binding_sha256": "a" * 64,
        }
        payload = ADAPTER._with_payload_sha(body)
        with (
            mock.patch.object(
                FORMAL,
                "_regular_file",
                return_value=Path("/frozen/val/preflight.json"),
            ),
            mock.patch.object(FORMAL, "_verified_json", return_value=payload),
            self.assertRaises(FORMAL.ValInferenceContractError),
        ):
            FORMAL._preflight_artifact(Path("/frozen/val/preflight.json"), "b" * 64)

    def test_dedicated_preflight_freshly_replays_all_authorities(self) -> None:
        body = {
            "format": ADAPTER.FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "candidate_epochs": [1, 2, 4, 8],
            "candidate_bundle": {"candidates": {str(e): {} for e in (1, 2, 4, 8)}},
            "val_inputs_receipt": {
                "path": "/frozen/val/inputs.json",
                "sha256": "1" * 64,
            },
            "pipeline_receipt": {
                "path": "/frozen/val/pipeline.json",
                "sha256": "2" * 64,
            },
            "pipeline_source": {},
            "inference_entrypoint": {},
            "coverage": {},
            "short_quality_authority": {
                "topology_mode": "w8_b8",
                "topology_gate_spec": {
                    "path": "/frozen/val/topology.json",
                    "sha256": "3" * 64,
                },
                "quality_gate_spec": {
                    "path": "/frozen/val/quality.json",
                    "sha256": "4" * 64,
                },
                "short_quality_status": {"path": "/frozen/val/status.json"},
                "candidate_ready_receipts": [
                    {"path": f"/frozen/val/e{e}.json"} for e in (1, 2, 4, 8)
                ],
            },
            "adapter_source": {},
            "quality_input_binding_sha256": "5" * 64,
        }
        payload = ADAPTER._with_payload_sha(body)
        rebuilt = dict(body)
        with (
            mock.patch.object(
                ADAPTER.engine,
                "_regular_file",
                return_value=Path("/frozen/val/preflight.json"),
            ),
            mock.patch.object(ADAPTER.engine, "_verified_json", return_value=payload),
            mock.patch.object(
                ADAPTER, "_build_bound_payload", return_value=rebuilt
            ) as fresh,
        ):
            artifact, observed = ADAPTER._preflight_artifact(
                Path("/frozen/val/preflight.json"), "a" * 64
            )
        self.assertEqual(observed, payload)
        self.assertEqual(artifact["sha256"], "a" * 64)
        kwargs = fresh.call_args.kwargs
        self.assertEqual(kwargs["mode"], "w8_b8")
        self.assertEqual(len(kwargs["ready_artifacts"]), 4)
        self.assertEqual(kwargs["topology_gate_sha"], "3" * 64)
        self.assertEqual(kwargs["quality_gate_sha"], "4" * 64)

    def test_launcher_is_guarded_exact_8shard_and_avoids_broad_process_tools(
        self,
    ) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/show_base/base_short_quality_val_8shard.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('. "$launcher_dir/guarded_runner_contract.sh"', source)
        self.assertIn("semtalk_require_exact_guarded_runner_all_gpus", source)
        self.assertIn("for shard_id in 0 1 2 3 4 5 6 7", source)
        self.assertIn("--num-shards 8", source)
        self.assertIn("for epoch in 1 2 4 8", source)
        self.assertIn("--split val", source)
        for forbidden in ("pgrep", "pkill", "killall", "--split test"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
