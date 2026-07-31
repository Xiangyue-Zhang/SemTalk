from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts.show_base import base_final_authority as authority
from scripts.show_base import build_segmented_prerequisite_candidate_index as builder
from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract
from scripts.show_base import published_test_winner_claim as published


STAGES = contract.STAGES
CAPS = {"face": 600, "hands": 500, "upper": 500, "lower": 600, "global": 1700}


def binding(name: str) -> dict[str, object]:
    return {
        "path": f"/formal/{name}.json",
        "sha256": (name[0] if name[0] in "abcdef" else "a") * 64,
        "bytes": 123,
        "receipt_payload_sha256": (name[-1] if name[-1] in "abcdef" else "b") * 64,
    }


def wave_receipt(
    name: str,
    active: tuple[str, ...],
    boundary: int,
    predecessors: dict[str, dict[str, object] | None],
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = binding(name)
    receipt = {
        "format": wave.PER_STAGE_FORMAT,
        "status": "authorized",
        "test_visible": False,
        "decision": binding(f"decision-{name}"),
        "trigger_stages": list(active),
        "stages": [
            {
                "stage": stage,
                "boundary_epoch": boundary,
                "target_epoch": boundary + 20,
                "cap_epoch": CAPS[stage],
                "old_segment": {
                    "predecessor_wave": copy.deepcopy(predecessors[stage])
                },
            }
            for stage in active
        ],
    }
    receipt["receipt_payload_sha256"] = artifact["receipt_payload_sha256"]
    return artifact, receipt


class PerStageConsumerTests(unittest.TestCase):
    def _mixed_waves(self):
        predecessor = {stage: None for stage in STAGES}
        first, first_receipt = wave_receipt(
            "first", STAGES, 200, predecessor
        )
        first_ref = {key: first[key] for key in ("path", "sha256", "receipt_payload_sha256")}
        predecessor = {stage: first_ref for stage in STAGES}
        second_active = ("face", "lower", "global")
        second, second_receipt = wave_receipt(
            "second", second_active, 220, predecessor
        )
        return [first, second], {
            first["path"]: first_receipt,
            second["path"]: second_receipt,
        }

    def test_mixed_continue_freeze_schedules_replay_exactly(self) -> None:
        artifacts, receipts = self._mixed_waves()
        schedules = {
            stage: [*range(20, 201, 20), 220]
            for stage in STAGES
        }
        for stage in ("face", "lower", "global"):
            schedules[stage].append(240)
        module = types.SimpleNamespace(
            replay_wave_file=lambda path, _sha: copy.deepcopy(receipts[str(path)])
        )
        with mock.patch.object(
            authority, "_payload_artifact_from_binding", side_effect=lambda value, _label: dict(value)
        ), mock.patch.object(authority, "_control_module", return_value=module):
            normalized = authority._replay_continuation_waves(
                artifacts, candidate_epochs_by_stage=schedules
            )
        self.assertEqual(normalized[1]["trigger_stages"], ["face", "lower", "global"])
        self.assertEqual(
            [item["target_epoch"] for item in normalized[1]["stage_transitions"]],
            [240, 240, 240],
        )

        with mock.patch.object(
            published,
            "_normalize_artifact",
            side_effect=lambda value, _label, with_payload: (dict(value), {}),
        ), mock.patch.object(
            published,
            "_replay_continuation_wave_file",
            side_effect=lambda value: copy.deepcopy(receipts[value["path"]]),
        ), mock.patch.object(
            published,
            "_fresh_local_module",
            return_value=contract,
        ):
            candidate_epochs = sorted(
                {epoch for stage_epochs in schedules.values() for epoch in stage_epochs}
            )
            result = published._validate_continuation_waves(
                artifacts,
                prerequisite_selection={
                    "format": published.PREREQUISITE_SELECTION_FORMAT,
                    "protocol": {
                        "name": "five_independent_show_prerequisite_validation_v2",
                        "candidate_epochs": candidate_epochs,
                        "candidate_epochs_by_stage": schedules,
                        "candidates_per_stage": {
                            stage: len(schedules[stage]) for stage in STAGES
                        },
                        "clips_per_candidate": published.EXPECTED_VAL_CLIPS,
                        "shards_per_candidate": published.EXPECTED_SHARDS,
                        "window_length": 64,
                        "window_stride": 20,
                        "full_base_fgd_used": False,
                    },
                },
            )
        self.assertEqual(result, artifacts)

    def test_frozen_stage_cannot_reenter(self) -> None:
        artifacts, receipts = self._mixed_waves()
        second_ref = {
            key: artifacts[1][key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        }
        third, third_receipt = wave_receipt(
            "third", ("hands",), 240, {stage: second_ref for stage in STAGES}
        )
        artifacts.append(third)
        receipts[third["path"]] = third_receipt
        schedules = {stage: [*range(20, 261, 20)] for stage in STAGES}
        module = types.SimpleNamespace(
            replay_wave_file=lambda path, _sha: copy.deepcopy(receipts[str(path)])
        )
        with mock.patch.object(
            authority, "_payload_artifact_from_binding", side_effect=lambda value, _label: dict(value)
        ), mock.patch.object(authority, "_control_module", return_value=module):
            with self.assertRaisesRegex(
                authority.BaseFinalAuthorityError, "reactivates a frozen stage"
            ):
                authority._replay_continuation_waves(
                    artifacts, candidate_epochs_by_stage=schedules
                )

    def test_candidate_index_stage_helpers_are_independent(self) -> None:
        common = list(range(20, 201, 20))
        index = {
            "candidate_epochs": [*common, 220, 240],
            "candidate_epochs_by_stage": {
                "face": [*common, 220, 240],
                "hands": common,
                "upper": [*common, 220],
                "lower": common,
                "global": [*common, 220],
            },
            "stages": {stage: [] for stage in STAGES},
        }
        self.assertEqual(contract.candidate_epochs_for_stage(index, "hands")[-1], 200)
        self.assertEqual(contract.candidate_epochs_for_stage(index, "face")[-1], 240)
        with self.assertRaisesRegex(contract.ContractError, "frozen schedule"):
            contract.candidate_lookup(index, "hands", 220)

    def test_segmented_builder_never_appends_frozen_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            common = list(range(20, 201, 20))
            prior = {
                "format": contract.CANDIDATE_INDEX_FORMAT,
                "candidate_epochs": common,
                "receipt_payload_sha256": "1" * 64,
                "source_receipts": {stage: {"stage": stage, "version": 1} for stage in STAGES},
                "config_sha256": {stage: "2" * 64 for stage in STAGES},
                "dataset_receipt_sha256": {stage: "3" * 64 for stage in STAGES},
                "formal_training_status": {stage: {"stage": stage, "version": 1} for stage in STAGES},
                "stages": {
                    stage: [
                        {
                            "epoch": epoch,
                            "checkpoint": str(root / stage / "prefix" / "representation_candidates" / f"e{epoch}.bin"),
                        }
                        for epoch in common
                    ]
                    for stage in STAGES
                },
            }
            prior_artifact = {
                "path": str(root / "prior.json"),
                "sha256": "4" * 64,
                "bytes": 100,
                "receipt_payload_sha256": prior["receipt_payload_sha256"],
            }
            wave_artifact = {
                "path": str(root / "wave.json"),
                "sha256": "5" * 64,
                "bytes": 100,
                "receipt_payload_sha256": "6" * 64,
            }
            entries = []
            for stage in ("face", "global"):
                initial = wave._make_chain_segment(
                    run_path=str(root / stage / "prefix"),
                    start_epoch=20,
                    end_epoch=200,
                    predecessor_segment_id=None,
                )
                new_run = root / stage / "next"
                appended = wave._make_chain_segment(
                    run_path=str(new_run),
                    start_epoch=220,
                    end_epoch=220,
                    predecessor_segment_id=initial["segment_id"],
                )
                entries.append(
                    {
                        "stage": stage,
                        "boundary_epoch": 200,
                        "target_epoch": 220,
                        "old_segment": {
                            "candidate_catalog_receipt": {
                                key: prior_artifact[key]
                                for key in ("path", "sha256", "receipt_payload_sha256")
                            },
                            "candidate_segment_chain": [initial],
                        },
                        "new_segment": {
                            "run_path": str(new_run),
                            "authorized_candidate_epochs": [220],
                            "predecessor_segment_id": initial["segment_id"],
                            "chain_segment_id": appended["segment_id"],
                        },
                    }
                )
            receipt = {
                "format": wave.PER_STAGE_FORMAT,
                "trigger_stages": ["face", "global"],
                "stages": entries,
                "receipt_payload_sha256": wave_artifact["receipt_payload_sha256"],
            }
            (root / "face" / "next").mkdir(parents=True)
            (root / "global" / "next").mkdir(parents=True)
            published_payload = {}

            def load(path, _sha, **_kwargs):
                return (published_payload if Path(path).name == "out.json" else prior), {}

            def build_stage(_torch, *, stage, target, **_kwargs):
                return (
                    {"epoch": target, "checkpoint": str(root / stage / "next" / "representation_candidates" / f"e{target}.bin")},
                    {"stage": stage, "version": 2},
                    "7" * 64,
                    "8" * 64,
                    {"stage": stage, "version": 2},
                )

            args = types.SimpleNamespace(
                stage_scope="full",
                prior_candidate_index=Path(prior_artifact["path"]),
                expected_prior_candidate_index_sha256=prior_artifact["sha256"],
                continuation_wave=Path(wave_artifact["path"]),
                expected_continuation_wave_sha256=wave_artifact["sha256"],
                stage_run=[f"face={root / 'face' / 'next'}", f"global={root / 'global' / 'next'}"],
                output_json=root / "out.json",
            )
            with mock.patch.dict("sys.modules", {"torch": types.ModuleType("torch")}), mock.patch.object(builder.contract, "load_candidate_index", side_effect=load), mock.patch.object(
                builder, "_artifact", side_effect=[prior_artifact, wave_artifact]
            ), mock.patch.object(builder.wave, "replay_wave_file", return_value=receipt), mock.patch.object(
                builder, "_build_stage", side_effect=build_stage
            ), mock.patch.object(builder.contract, "validate_candidate_index", side_effect=lambda value, **_kwargs: value), mock.patch.object(
                builder.contract, "build_source_policy", return_value={"format": "fixture"}
            ), mock.patch.object(
                builder.contract,
                "atomic_json_new",
                side_effect=lambda _path, value: published_payload.update(copy.deepcopy(value)) or "9" * 64,
            ):
                result = builder.build(args)
            self.assertEqual(result["candidate_epochs_by_stage"]["face"][-1], 220)
            self.assertEqual(result["candidate_epochs_by_stage"]["global"][-1], 220)
            for stage in ("hands", "upper", "lower"):
                self.assertEqual(result["candidate_epochs_by_stage"][stage][-1], 200)
                self.assertEqual(result["stages"][stage], prior["stages"][stage])
                self.assertEqual(result["source_receipts"][stage], prior["source_receipts"][stage])


if __name__ == "__main__":
    unittest.main()
