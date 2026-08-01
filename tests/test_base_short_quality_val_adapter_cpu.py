from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.show_base import base_short_quality_val_adapter as ADAPTER
from scripts.show_base import base_long_val_contract as LONG
from scripts.show_base import run_base_val_inference as FORMAL


class BaseShortQualityValAdapterCpuTest(unittest.TestCase):
    def test_cli_is_val_only_and_accepts_the_candidate_epoch_union(self) -> None:
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
        for epoch in ("2", "4", "8", "16", "32"):
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
        candidate_mode = ADAPTER.training.W8_GLOBAL64_MODE
        candidate_epochs = ADAPTER.topology.quality_epochs_for_mode(
            candidate_mode
        )
        reference_mode = ADAPTER.training.OFFICIAL_W1_REFERENCE_MODE
        reference_epochs = ADAPTER.topology.quality_epochs_for_mode(
            reference_mode
        )
        with mock.patch.object(Path, "read_bytes") as read_bytes:
            read_bytes.return_value = json.dumps(
                {"receipt_payload_sha256": "b" * 64}
            ).encode()
            values = [
                (str(epoch), f"/frozen/val/e{epoch}.json", "a" * 64)
                for epoch in candidate_epochs
            ]
            result = ADAPTER._normalize_ready(values, candidate_mode)
            self.assertEqual(len(result), len(candidate_epochs))
            with self.assertRaises(ADAPTER.ShortQualityValError):
                ADAPTER._normalize_ready(list(reversed(values)), candidate_mode)
            bad = list(values)
            bad[0] = ("1", "/frozen/val/e1.json", "A" * 64)
            with self.assertRaises(ADAPTER.ShortQualityValError):
                ADAPTER._normalize_ready(bad, candidate_mode)

            reference_values = values[: len(reference_epochs)]
            reference = ADAPTER._normalize_ready(
                reference_values, reference_mode
            )
            self.assertEqual(
                [Path(item["path"]).stem for item in reference],
                [f"e{epoch}" for epoch in reference_epochs],
            )
            with self.assertRaises(ADAPTER.ShortQualityValError):
                ADAPTER._normalize_ready(values, reference_mode)

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

    def test_adapter_reuses_formal_diffsheg_abi_without_distribution(self) -> None:
        self.assertIs(
            ADAPTER.formal_validation.validate_val_inference_lineage,
            LONG.validate_val_inference_lineage,
        )
        self.assertIs(
            ADAPTER.formal_validation.validate_diffsheg_report,
            LONG.validate_diffsheg_report,
        )
        self.assertIs(
            FORMAL.selector.validate_val_inference_lineage,
            LONG.validate_val_inference_lineage,
        )
        self.assertIn(
            "scripts/show_base/evaluate_diffsheg_val_fgd.py",
            ADAPTER.SOURCE_FILES,
        )
        self.assertIn(
            "scripts/show_base/select_base_training_topology.py",
            ADAPTER.SOURCE_FILES,
        )
        self.assertIn(
            "scripts/show_base/train_base_official_adapt_long.py",
            ADAPTER.SOURCE_FILES,
        )
        self.assertEqual(
            FORMAL.SHORT_QUALITY_CHECKPOINT_FORMAT,
            ADAPTER.training.SHORT_QUALITY_CHECKPOINT_FORMAT,
        )
        self.assertEqual(
            FORMAL.SHORT_QUALITY_PROTOCOL_VERSION,
            ADAPTER.topology.QUALITY_PROTOCOL_VERSION,
        )
        self.assertEqual(
            FORMAL.SHORT_QUALITY_ARTIFACT_ROOT_NAMESPACE,
            ADAPTER.topology.QUALITY_ARTIFACT_ROOT_NAMESPACE,
        )
        self.assertNotIn(
            "scripts/show_base/replay_released2_primary.py",
            ADAPTER.SOURCE_FILES,
        )
        with self.assertRaises(SystemExit):
            ADAPTER.parse_args(["distribution"])

    def test_prepare_schema_is_isolated_from_formal_preflight(self) -> None:
        candidate_mode = ADAPTER.training.W8_GLOBAL64_MODE
        candidate_epochs = list(
            ADAPTER.topology.quality_epochs_for_mode(candidate_mode)
        )
        body = {
            "format": ADAPTER.FORMAT,
            "status": "complete",
            "quality_protocol_version": ADAPTER.topology.QUALITY_PROTOCOL_VERSION,
            "artifact_root_namespace": (
                ADAPTER.topology.QUALITY_ARTIFACT_ROOT_NAMESPACE
            ),
            "artifact_root": "/frozen/quality/candidate",
            "quality_role": "candidate_quality",
            "reference_only": False,
            "late_w1_status": "not_measured",
            "w1_tail_equivalence_claimed": False,
            "split": "val",
            "test_visible": False,
            "candidate_epochs": candidate_epochs,
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
        mode = ADAPTER.training.W8_GLOBAL64_MODE
        epochs = ADAPTER.topology.quality_epochs_for_mode(mode)
        with tempfile.TemporaryDirectory(
            prefix="semtalk_short_quality_preflight_"
        ) as raw:
            artifact_root = Path(raw).resolve()
            metadata = {
                "quality_protocol_version": (
                    ADAPTER.topology.QUALITY_PROTOCOL_VERSION
                ),
                "artifact_root_namespace": (
                    ADAPTER.topology.QUALITY_ARTIFACT_ROOT_NAMESPACE
                ),
                "artifact_root": str(artifact_root),
                "quality_role": "candidate_quality",
                "reference_only": False,
                "late_w1_status": "not_measured",
                "w1_tail_equivalence_claimed": False,
            }
            authority = {
                "topology_mode": mode,
                "topology_gate_spec": {
                    "path": "/frozen/val/topology.json",
                    "sha256": "3" * 64,
                },
                "quality_gate_spec": {
                    "path": "/frozen/val/quality.json",
                    "sha256": "4" * 64,
                },
                "short_quality_status": {
                    "path": "/frozen/val/status.json"
                },
                "candidate_ready_receipts": [
                    {"path": f"/frozen/val/e{epoch}.json"}
                    for epoch in epochs
                ],
                **metadata,
            }
            body = {
                "format": ADAPTER.FORMAT,
                "status": "complete",
                **metadata,
                "split": "val",
                "test_visible": False,
                "candidate_epochs": list(epochs),
                "candidate_bundle": {
                    "candidates": {str(epoch): {} for epoch in epochs}
                },
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
                "short_quality_authority": authority,
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
                mock.patch.object(
                    ADAPTER.engine, "_verified_json", return_value=payload
                ),
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
            self.assertEqual(kwargs["mode"], mode)
            self.assertEqual(len(kwargs["ready_artifacts"]), len(epochs))
            self.assertEqual(kwargs["topology_gate_sha"], "3" * 64)
            self.assertEqual(kwargs["quality_gate_sha"], "4" * 64)

    def test_w1_preflight_rejects_unmeasured_tail_before_engine_entry(self) -> None:
        preflight = {
            "candidate_epochs": list(
                ADAPTER.topology.W1_REFERENCE_EPOCHS
            ),
            "short_quality_authority": {
                "topology_mode": ADAPTER.training.OFFICIAL_W1_REFERENCE_MODE
            },
        }
        args = ADAPTER.parse_args(
            [
                "shard",
                "--split",
                "val",
                "--preflight",
                "/frozen/val/preflight.json",
                "--expected-preflight-sha256",
                "a" * 64,
                "--epoch",
                "16",
                "--output-root",
                "/frozen/val/e16",
                "--num-shards",
                "8",
                "--shard-id",
                "0",
                "--device",
                "cuda:0",
            ]
        )
        with (
            mock.patch.object(
                ADAPTER,
                "_preflight_artifact",
                return_value=({}, preflight),
            ),
            mock.patch.object(ADAPTER.engine, "run_shard") as engine,
            self.assertRaisesRegex(
                ADAPTER.ShortQualityValError,
                "not measured",
            ),
        ):
            ADAPTER.run_shard(args)
        engine.assert_not_called()

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
        self.assertIn("mapfile -t quality_epochs", source)
        self.assertIn("adapter._preflight_artifact", source)
        self.assertIn('for epoch in "${quality_epochs[@]}"', source)
        self.assertNotIn("for epoch in 1 2 4 8 16 32", source)
        self.assertIn("--split val", source)
        for required in (
            "evaluate_diffsheg_val_fgd.py",
            "final/predictions/val",
            "final/ground-truth/val",
            "diffsheg_eval_clip_ids.txt",
            "--paspa-root",
            "--diffsheg-root",
            "semtalk_show_base_short_quality_val_completion_v3",
            '"inference_lineage"',
            '"diffsheg_report"',
        ):
            self.assertIn(required, source)
        for forbidden in (
            "pgrep",
            "pkill",
            "killall",
            "--split test",
            "released2",
            "replay_released2_primary.py",
            "distribution.json",
            "real-feature-cache",
            "talkshow-metric-root",
            "feature-extractor",
            "smplx-asset",
        ):
            self.assertNotIn(forbidden, source)
        repository = Path(__file__).resolve().parents[1]
        for relative in (
            "scripts/show_base/base_short_quality_val_adapter.py",
            "scripts/show_base/produce_base_topology_short_quality.py",
            "scripts/show_base/select_base_training_topology.py",
        ):
            migrated = (repository / relative).read_text(encoding="utf-8")
            self.assertNotIn("released2", migrated.lower(), relative)


if __name__ == "__main__":
    unittest.main()
