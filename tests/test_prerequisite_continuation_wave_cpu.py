from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.show_base import prerequisite_continuation_wave as wave


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
OID_A = "1" * 40
OID_B = "2" * 40
OID_C = "3" * 40
OID_D = "4" * 40
HOSTS = tuple(sorted(wave.FORMAL_HOSTS))


def smplx_asset(stage: str, host: str) -> dict | None:
    if stage == "global":
        return None
    return {
        "format": "semtalk_show_smplx_asset_v1",
        "filename": wave.FORMAL_SMPLX_FILENAME,
        "path": (
            f"/formal/assets/{host}/{wave.FORMAL_SMPLX_FILENAME}"
        ),
        "sha256": wave.FORMAL_SMPLX_SHA256,
        "bytes": wave.FORMAL_SMPLX_BYTES,
        "regular_file": True,
        "symlink": False,
    }


def source_ancestry(stage: str) -> dict:
    proof = {
        "format": wave.SOURCE_ANCESTRY_FORMAT,
        "repository_path": f"/formal/source/new/{stage}",
        "origin": wave.val_contract.EXPECTED_ORIGIN,
        "head_commit": OID_C,
        "head_tree": OID_D,
        "old_commit": OID_A,
        "baseline_commit": wave.OFFICIAL_BASELINE_COMMIT,
        "clean": True,
        "detached_head": True,
        "local_branch_ref_count": 0,
        "old_is_ancestor": True,
        "baseline_is_ancestor": True,
    }
    proof["receipt_payload_sha256"] = wave.canonical_json_sha256(proof)
    return proof


def checkpoint(stage: str, epoch: int) -> dict:
    return {
        "path": (
            f"/formal/old/{stage}/representation_candidates/"
            f"{stage}_epoch_{epoch:04d}.bin"
        ),
        "sha256": hashlib.sha256(
            f"checkpoint:{stage}:{epoch}".encode()
        ).hexdigest(),
        "bytes": 1_000_000 + epoch,
    }


def candidate(stage: str, epoch: int) -> dict:
    return {
        "epoch": epoch,
        "optimizer_updates": epoch * wave.EXPECTED_UPDATES_PER_EPOCH,
        "checkpoint": checkpoint(stage, epoch),
        "checkpoint_audit_sha256": hashlib.sha256(
            f"audit:{stage}:{epoch}".encode()
        ).hexdigest(),
    }


def file_binding(stage: str, filename: str, marker: str) -> dict:
    return {
        "path": f"/formal/old/{stage}/{filename}",
        "sha256": hashlib.sha256(
            f"{marker}:{stage}".encode()
        ).hexdigest(),
        "bytes": 1_000 + len(stage),
    }


def boundary_state_proof(stage: str, boundary: int) -> dict:
    world = 4 if stage in wave.RVQ_STAGES else 1
    proof = {
        "format": wave.boundary_state.FORMAT,
        "stage": stage,
        "boundary_epoch": boundary,
        "optimizer_updates": (
            boundary * wave.EXPECTED_UPDATES_PER_EPOCH
        ),
        "world_size": world,
        "model_state_sha256": SHA_A,
        "optimizer_state_sha256": SHA_B,
        "scheduler_state_sha256": SHA_C,
        "rvq_ema_state_sha256": SHA_D,
        "rng_states_sha256": hashlib.sha256(
            f"rng:{stage}:{boundary}".encode()
        ).hexdigest(),
        "trained_parameter_count": 17,
        "adam_state_count": 17,
        "adam_step": boundary * wave.EXPECTED_UPDATES_PER_EPOCH,
        "scheduler_epoch": boundary - 1,
        "scheduler_lrs": [0.000123],
        "rvq_ema_layers": 6 if stage in wave.RVQ_STAGES else 0,
        "rng_rank_count": world,
    }
    proof["receipt_payload_sha256"] = (
        wave.boundary_state.canonical_sha256(proof)
    )
    return proof


def fixture(
    *,
    triggers: tuple[str, ...] = ("face",),
    boundary: int = 200,
) -> dict:
    decision_stages = []
    selected = {}
    catalogs = {}
    plans = {}
    for index, stage in enumerate(wave.STAGES):
        old_host = HOSTS[index % len(HOSTS)]
        new_host = HOSTS[(index + 1) % len(HOSTS)]
        decision_stages.append(
            {
                "stage": stage,
                "latest_epoch": boundary,
                "requests_continuation": stage in triggers,
            }
        )
        catalog = [
            candidate(stage, epoch)
            for epoch in range(
                wave.INTERVAL_EPOCHS,
                boundary + 1,
                wave.INTERVAL_EPOCHS,
            )
        ]
        catalogs[stage] = catalog
        # The independently selected winner is intentionally older.
        selected[stage] = copy.deepcopy(
            next(item for item in catalog if item["epoch"] == 120)
        )
        plans[stage] = {
            "stage": stage,
            "old_run_path": f"/formal/old/{stage}",
            "new_run_path": f"/formal/new/e{boundary + 20}/{stage}",
            "old_source": {
                "commit": OID_A,
                "tree": OID_B,
                "source_receipt_sha256": SHA_A,
            },
            "new_source": {
                "commit": OID_C,
                "tree": OID_D,
                "source_receipt_sha256": SHA_B,
            },
            "source_ancestry": source_ancestry(stage),
            "old_host": old_host,
            "new_host": new_host,
            "old_smplx_asset": smplx_asset(stage, old_host),
            "new_smplx_asset": smplx_asset(stage, new_host),
            "old_config_sha256": SHA_A,
            "new_config_sha256": SHA_B,
            "old_source_audit_sha256": SHA_A,
            "old_dataset_receipt_sha256": SHA_B,
            "old_config_semantic_sha256": SHA_D,
            "new_config_semantic_sha256": SHA_D,
            "old_dataset_semantic_sha256": SHA_C,
            "new_dataset_semantic_sha256": SHA_C,
            "candidate_segment_chain": [
                wave._make_chain_segment(
                    run_path=f"/formal/old/{stage}",
                    start_epoch=wave.INTERVAL_EPOCHS,
                    end_epoch=boundary,
                    predecessor_segment_id=None,
                )
            ],
            "predecessor_wave": None,
            "candidate_catalog_receipt": {
                "path": f"/formal/receipts/{stage}.json",
                "sha256": SHA_D,
                "receipt_payload_sha256": SHA_A,
            },
            "formal_status": file_binding(
                stage,
                "formal_training_status.json",
                "status",
            ),
            "boundary_resume": file_binding(
                stage,
                "latest_resume.pt",
                "resume",
            ),
            "final_checkpoint": file_binding(
                stage,
                f"{stage}_final.bin",
                "final",
            ),
            "boundary_state": boundary_state_proof(stage, boundary),
        }
    decision_receipt = {
        "status": "complete",
        "decision": "continue",
        "test_visible": False,
        "stages": decision_stages,
    }
    decision_receipt["receipt_payload_sha256"] = (
        wave.canonical_json_sha256(decision_receipt)
    )
    return {
        "decision_receipt": decision_receipt,
        "decision_binding": {
            "path": "/formal/receipts/decision.json",
            "sha256": SHA_A,
            "receipt_payload_sha256": decision_receipt[
                "receipt_payload_sha256"
            ],
        },
        "selected_by_stage": selected,
        "candidate_catalog_by_stage": catalogs,
        "stage_plans": plans,
    }


def authorize(data: dict) -> dict:
    return wave.authorize_wave(**data)


def recursive_fixture(previous: dict) -> dict:
    data = fixture(boundary=220)
    predecessor_binding = {
        "path": "/formal/receipts/wave_e220.json",
        "sha256": SHA_C,
        "receipt_payload_sha256": previous["receipt_payload_sha256"],
    }
    previous_by_stage = {
        entry["stage"]: entry for entry in previous["stages"]
    }
    for stage in wave.STAGES:
        previous_entry = previous_by_stage[stage]
        previous_old = previous_entry["old_segment"]
        previous_new = previous_entry["new_segment"]
        latest_run = previous_new["run_path"]
        next_run = f"/formal/new/e240/{stage}"
        appended = wave._make_chain_segment(
            run_path=latest_run,
            start_epoch=220,
            end_epoch=220,
            predecessor_segment_id=previous_new[
                "predecessor_segment_id"
            ],
        )
        assert appended["segment_id"] == previous_new[
            "chain_segment_id"
        ]
        plan = data["stage_plans"][stage]
        plan["old_run_path"] = latest_run
        plan["new_run_path"] = next_run
        plan["candidate_segment_chain"] = (
            copy.deepcopy(previous_old["candidate_segment_chain"])
            + [appended]
        )
        plan["predecessor_wave"] = copy.deepcopy(predecessor_binding)
        plan["old_source"] = copy.deepcopy(previous_new["source"])
        plan["new_source"] = copy.deepcopy(previous_new["source"])
        ancestry = copy.deepcopy(previous_new["source_ancestry"])
        ancestry["old_commit"] = previous_new["source"]["commit"]
        unsigned = dict(ancestry)
        unsigned.pop("receipt_payload_sha256")
        ancestry["receipt_payload_sha256"] = wave.canonical_json_sha256(
            unsigned
        )
        plan["source_ancestry"] = ancestry
        plan["old_host"] = previous_new["host"]
        plan["old_smplx_asset"] = copy.deepcopy(
            previous_new["smplx_asset"]
        )
        for key, filename in (
            ("formal_status", "formal_training_status.json"),
            ("boundary_resume", "latest_resume.pt"),
            ("final_checkpoint", f"{stage}_final.bin"),
        ):
            plan[key]["path"] = f"{latest_run}/{filename}"
        plan["boundary_state"] = boundary_state_proof(stage, 220)
        catalog = data["candidate_catalog_by_stage"][stage]
        for candidate_value in catalog:
            epoch = candidate_value["epoch"]
            run = (
                previous_old["run_path"]
                if epoch <= 200
                else latest_run
            )
            candidate_value["checkpoint"]["path"] = (
                f"{run}/representation_candidates/"
                f"{stage}_epoch_{epoch:04d}.bin"
            )
        data["selected_by_stage"][stage] = copy.deepcopy(
            next(item for item in catalog if item["epoch"] == 120)
        )
    return data


def resign(receipt: dict) -> dict:
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = wave.canonical_json_sha256(unsigned)
    return receipt


class ContinuationWaveTests(unittest.TestCase):
    def test_one_trigger_authorizes_all_five(self) -> None:
        data = fixture(triggers=("face",))
        receipt = authorize(data)
        self.assertEqual(receipt["trigger_stages"], ["face"])
        self.assertEqual(len(receipt["stages"]), 5)
        self.assertTrue(receipt["stages"][0]["triggered"])
        self.assertTrue(
            all(
                stage["new_segment"]["authorized_candidate_epochs"]
                == [220]
                for stage in receipt["stages"]
            )
        )
        self.assertTrue(
            all(
                stage["resume_boundary_candidate"]["epoch"] == 200
                for stage in receipt["stages"]
            )
        )

    def test_three_triggers_are_exact_and_ordered(self) -> None:
        data = fixture(triggers=("face", "upper", "global"))
        receipt = authorize(data)
        self.assertEqual(
            receipt["trigger_stages"],
            ["face", "upper", "global"],
        )
        self.assertEqual(
            [stage["triggered"] for stage in receipt["stages"]],
            [True, False, True, False, True],
        )

    def test_independent_winner_can_precede_boundary(self) -> None:
        receipt = authorize(fixture())
        for stage in receipt["stages"]:
            self.assertEqual(
                stage["independent_selected_candidate"]["epoch"],
                120,
            )
            self.assertEqual(
                stage["resume_boundary_candidate"]["epoch"],
                200,
            )

    def test_stop_decision_is_rejected(self) -> None:
        data = fixture()
        data["decision_receipt"]["decision"] = "stop"
        resign(data["decision_receipt"])
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "does not authorize",
        ):
            authorize(data)

    def test_continue_without_trigger_is_rejected(self) -> None:
        data = fixture(triggers=())
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "no triggering stage",
        ):
            authorize(data)

    def test_unequal_stage_boundaries_are_rejected(self) -> None:
        data = fixture()
        data["decision_receipt"]["stages"][2]["latest_epoch"] = 220
        resign(data["decision_receipt"])
        data["decision_binding"]["receipt_payload_sha256"] = (
            data["decision_receipt"]["receipt_payload_sha256"]
        )
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "share one resume boundary",
        ):
            authorize(data)

    def test_missing_stage_input_is_rejected(self) -> None:
        data = fixture()
        data["stage_plans"].pop("global")
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "exactly cover all five",
        ):
            authorize(data)

    def test_duplicate_decision_stage_is_rejected(self) -> None:
        data = fixture()
        data["decision_receipt"]["stages"][4]["stage"] = "lower"
        resign(data["decision_receipt"])
        data["decision_binding"]["receipt_payload_sha256"] = (
            data["decision_receipt"]["receipt_payload_sha256"]
        )
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "stage order changed",
        ):
            authorize(data)

    def test_selected_checkpoint_swap_is_rejected(self) -> None:
        data = fixture()
        data["selected_by_stage"]["face"]["checkpoint"]["sha256"] = SHA_D
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "differs from catalog",
        ):
            authorize(data)

    def test_candidate_catalog_hole_is_rejected(self) -> None:
        data = fixture()
        del data["candidate_catalog_by_stage"]["face"][4]
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "exactly cover e20..boundary",
        ):
            authorize(data)

    def test_dataset_semantic_drift_is_rejected(self) -> None:
        data = fixture()
        data["stage_plans"]["upper"][
            "new_dataset_semantic_sha256"
        ] = SHA_D
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "training semantics changed",
        ):
            authorize(data)

    def test_config_semantic_drift_is_rejected(self) -> None:
        data = fixture()
        data["stage_plans"]["lower"][
            "new_config_semantic_sha256"
        ] = SHA_A
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "training semantics changed",
        ):
            authorize(data)

    def test_new_source_divergence_is_rejected(self) -> None:
        data = fixture()
        data["stage_plans"]["global"]["new_source"]["commit"] = OID_A
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "source ancestry|one new source",
        ):
            authorize(data)

    def test_unknown_hostname_is_rejected(self) -> None:
        data = fixture()
        data["stage_plans"]["face"]["new_host"] = "worker-alias"
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "two exact formal hosts",
        ):
            authorize(data)

    def test_one_host_only_composite_is_rejected(self) -> None:
        data = fixture()
        for plan in data["stage_plans"].values():
            plan["old_host"] = HOSTS[0]
            plan["new_host"] = HOSTS[0]
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "bind both formal hosts",
        ):
            authorize(data)

    def test_smplx_sha_or_byte_drift_is_rejected(self) -> None:
        attacks = (("sha256", SHA_A), ("bytes", wave.FORMAL_SMPLX_BYTES - 1))
        for key, replacement in attacks:
            with self.subTest(key=key):
                data = fixture()
                data["stage_plans"]["upper"]["new_smplx_asset"][
                    key
                ] = replacement
                with self.assertRaisesRegex(
                    wave.ContinuationWaveError,
                    "pinned SMPL-X",
                ):
                    authorize(data)

    def test_global_must_not_bind_smplx(self) -> None:
        data = fixture()
        host = data["stage_plans"]["global"]["new_host"]
        data["stage_plans"]["global"]["new_smplx_asset"] = smplx_asset(
            "face",
            host,
        )
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "null for the Global",
        ):
            authorize(data)

    def test_resigned_false_source_ancestry_is_rejected(self) -> None:
        data = fixture()
        proof = data["stage_plans"]["hands"]["source_ancestry"]
        proof["old_is_ancestor"] = False
        unsigned = dict(proof)
        unsigned.pop("receipt_payload_sha256")
        proof["receipt_payload_sha256"] = wave.canonical_json_sha256(
            unsigned
        )
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "exact source ancestry",
        ):
            authorize(data)

    def test_wrong_source_origin_or_baseline_is_rejected(self) -> None:
        attacks = (
            ("origin", "git@github.com:other/SemTalk.git"),
            ("baseline_commit", OID_B),
        )
        for key, replacement in attacks:
            with self.subTest(key=key):
                data = fixture()
                proof = data["stage_plans"]["lower"][
                    "source_ancestry"
                ]
                proof[key] = replacement
                unsigned = dict(proof)
                unsigned.pop("receipt_payload_sha256")
                proof["receipt_payload_sha256"] = (
                    wave.canonical_json_sha256(unsigned)
                )
                with self.assertRaisesRegex(
                    wave.ContinuationWaveError,
                    "exact source ancestry",
                ):
                    authorize(data)

    def test_boundary_checkpoint_path_swap_is_rejected_on_replay(self) -> None:
        data = fixture()
        receipt = authorize(data)
        attacked = copy.deepcopy(receipt)
        attacked["stages"][0]["resume_boundary_candidate"][
            "checkpoint"
        ]["path"] = data["candidate_catalog_by_stage"]["hands"][-1][
            "checkpoint"
        ]["path"]
        resign(attacked)
        with self.assertRaises(wave.ContinuationWaveError):
            wave.replay_wave_receipt(attacked, **data)

    def test_boundary_checkpoint_byte_swap_is_rejected_on_replay(self) -> None:
        data = fixture()
        receipt = authorize(data)
        attacked = copy.deepcopy(receipt)
        attacked["stages"][1]["resume_boundary_candidate"][
            "checkpoint"
        ]["bytes"] += 1
        resign(attacked)
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "differs from fresh replay",
        ):
            wave.replay_wave_receipt(attacked, **data)

    def test_resigned_target_plus_40_is_rejected(self) -> None:
        receipt = authorize(fixture())
        attacked = copy.deepcopy(receipt)
        attacked["target_epoch"] = 240
        for stage in attacked["stages"]:
            stage["new_segment"]["target_epoch"] = 240
            stage["new_segment"]["authorized_candidate_epochs"] = [240]
            stage["new_segment"]["segment_id"] = wave._segment_id(
                stage["new_segment"],
                id_key="segment_id",
            )
        resign(attacked)
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "not exact \\+20",
        ):
            wave.validate_wave_schema(attacked)

    def test_copied_prefix_field_is_rejected(self) -> None:
        receipt = authorize(fixture())
        attacked = copy.deepcopy(receipt)
        attacked["stages"][0]["new_segment"]["candidate_prefix"] = [
            20,
            40,
            60,
        ]
        resign(attacked)
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "schema mismatch",
        ):
            wave.validate_wave_schema(attacked)

    def test_new_segment_reusing_old_run_is_rejected(self) -> None:
        data = fixture()
        data["stage_plans"]["face"]["new_run_path"] = (
            data["stage_plans"]["face"]["old_run_path"]
        )
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "disjoint",
        ):
            authorize(data)

    def test_cross_stage_old_new_run_swap_is_rejected(self) -> None:
        data = fixture()
        data["stage_plans"]["face"]["new_run_path"] = (
            data["stage_plans"]["hands"]["old_run_path"]
        )
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "globally disjoint",
        ):
            authorize(data)

    def test_resigned_decision_trigger_attack_breaks_binding(self) -> None:
        data = fixture()
        data["decision_receipt"]["stages"][0][
            "requests_continuation"
        ] = False
        data["decision_receipt"]["stages"][1][
            "requests_continuation"
        ] = True
        resign(data["decision_receipt"])
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "binding payload mismatch",
        ):
            authorize(data)

    def test_receipt_replay_is_exact(self) -> None:
        data = fixture(triggers=("hands",))
        receipt = authorize(data)
        replayed = wave.replay_wave_receipt(receipt, **data)
        self.assertEqual(replayed, receipt)

    def test_recursive_e200_e220_e240_segment_chain_is_exact(self) -> None:
        first = authorize(fixture())
        second_data = recursive_fixture(first)
        second = authorize(second_data)
        self.assertEqual(second["boundary_epoch"], 220)
        self.assertEqual(second["target_epoch"], 240)
        for entry in second["stages"]:
            chain = entry["old_segment"]["candidate_segment_chain"]
            self.assertEqual(len(chain), 2)
            self.assertEqual(chain[0]["candidate_epochs"], list(range(20, 201, 20)))
            self.assertEqual(chain[1]["candidate_epochs"], [220])
            self.assertEqual(
                entry["independent_selected_candidate"]["epoch"],
                120,
            )
            self.assertEqual(
                entry["resume_boundary_candidate"]["epoch"],
                220,
            )
            self.assertEqual(
                entry["new_segment"]["predecessor_segment_id"],
                chain[-1]["segment_id"],
            )
        wave.replay_wave_receipt(second, **second_data)

    def test_recursive_chain_rejects_copied_prefix_into_e220(self) -> None:
        first = authorize(fixture())
        data = recursive_fixture(first)
        copied = wave._make_chain_segment(
            run_path=data["stage_plans"]["face"]["old_run_path"],
            start_epoch=20,
            end_epoch=220,
            predecessor_segment_id=None,
        )
        data["stage_plans"]["face"]["candidate_segment_chain"] = [
            copied
        ]
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "chain/predecessor|old run/predecessor",
        ):
            authorize(data)

    def test_recursive_chain_rejects_candidate_in_wrong_segment(self) -> None:
        first = authorize(fixture())
        data = recursive_fixture(first)
        face_catalog = data["candidate_catalog_by_stage"]["face"]
        e220 = next(row for row in face_catalog if row["epoch"] == 220)
        e220["checkpoint"]["path"] = (
            "/formal/old/face/representation_candidates/"
            "face_epoch_0220.bin"
        )
        with self.assertRaisesRegex(
            wave.ContinuationWaveError,
            "outside its immutable segment",
        ):
            authorize(data)


def adapter_fixture(
    root: Path,
    *,
    data: dict | None = None,
    predecessor: dict | None = None,
) -> dict:
    data = fixture(triggers=("upper",)) if data is None else data
    selection_binding = {
        "path": str((root / "selection.json").resolve()),
        "sha256": SHA_A,
        "receipt_payload_sha256": SHA_B,
    }
    candidate_index_binding = {
        "path": str((root / "candidate_index.json").resolve()),
        "sha256": SHA_C,
        "receipt_payload_sha256": SHA_D,
    }
    decision = copy.deepcopy(data["decision_receipt"])
    decision["inputs"] = {"selection": selection_binding}
    resign(decision)

    index_stages = {}
    bridge_selected = {}
    for stage in wave.STAGES:
        index_stages[stage] = [
            {
                "epoch": candidate_value["epoch"],
                "optimizer_updates": candidate_value[
                    "optimizer_updates"
                ],
                "checkpoint": candidate_value["checkpoint"]["path"],
                "checkpoint_sha256": candidate_value["checkpoint"][
                    "sha256"
                ],
                "checkpoint_bytes": candidate_value["checkpoint"][
                    "bytes"
                ],
                "checkpoint_audit_sha256": candidate_value[
                    "checkpoint_audit_sha256"
                ],
            }
            for candidate_value in data["candidate_catalog_by_stage"][stage]
        ]
        selected = data["selected_by_stage"][stage]
        bridge_selected[stage] = {
            "epoch": selected["epoch"],
            "optimizer_updates": selected["optimizer_updates"],
            "candidate_checkpoint": copy.deepcopy(
                selected["checkpoint"]
            ),
            "candidate_audit_sha256": selected[
                "checkpoint_audit_sha256"
            ],
        }
    candidate_index = {
        "receipt_payload_sha256": SHA_D,
        "stages": index_stages,
        "source_receipts": {
            stage: {
                "portable_identity": {
                    "commit": data["stage_plans"][stage]["old_source"][
                        "commit"
                    ],
                    "tree": data["stage_plans"][stage]["old_source"][
                        "tree"
                    ],
                },
                "receipt_payload_sha256": data["stage_plans"][stage][
                    "old_source"
                ]["source_receipt_sha256"],
            }
            for stage in wave.STAGES
        },
        "config_sha256": {
            stage: SHA_A for stage in wave.STAGES
        },
        "dataset_receipt_sha256": {
            stage: hashlib.sha256(
                f"dataset-receipt:{stage}".encode()
            ).hexdigest()
            for stage in wave.STAGES
        },
        "formal_training_status": {
            stage: {
                "path": f"/formal/old/{stage}/formal_training_status.json",
                "sha256": hashlib.sha256(
                    f"status:{stage}".encode()
                ).hexdigest(),
            }
            for stage in wave.STAGES
        },
    }
    bridge = {
        "selection": selection_binding,
        "candidate_index_receipt": candidate_index_binding,
        "selected": bridge_selected,
    }
    adapter_plans = {
        stage: {
            key: copy.deepcopy(value)
            for key, value in data["stage_plans"][stage].items()
            if key in wave.ADAPTER_STAGE_PLAN_KEYS
        }
        for stage in wave.STAGES
    }
    for stage in wave.STAGES:
        adapter_plans[stage]["new_source_repository"] = (
            data["stage_plans"][stage]["source_ancestry"][
                "repository_path"
            ]
        )
    return {
        "decision": decision,
        "bridge": bridge,
        "candidate_index": candidate_index,
        "candidate_index_artifact": {
            "path": candidate_index_binding["path"],
            "sha256": candidate_index_binding["sha256"],
        },
        "stage_plans": adapter_plans,
        "runtime_evidence": {
            stage: {
                key: copy.deepcopy(data["stage_plans"][stage][key])
                for key in (
                    "old_host",
                    "old_smplx_asset",
                    "old_source_audit_sha256",
                    "old_dataset_receipt_sha256",
                    "formal_status",
                    "boundary_resume",
                    "final_checkpoint",
                    "boundary_state",
                )
            }
            for stage in wave.STAGES
        },
        "source_ancestry": {
            stage: copy.deepcopy(
                data["stage_plans"][stage]["source_ancestry"]
            )
            for stage in wave.STAGES
        },
        "predecessor": predecessor,
    }


class ContinuationWaveAdapterTests(unittest.TestCase):
    def run_adapter(self, values: dict, root: Path) -> dict:
        decision_path = root / "decision.json"
        decision_path.write_text("{}\n")
        with (
            mock.patch.object(
                wave.continuation_decision,
                "replay_decision",
                return_value=values["decision"],
            ),
            mock.patch.object(
                wave.selected_contract,
                "load_selected_prerequisites",
                return_value=values["bridge"],
            ),
            mock.patch.object(
                wave.val_contract,
                "load_candidate_index",
                return_value=(
                    values["candidate_index"],
                    values["candidate_index_artifact"],
                ),
            ),
            mock.patch.object(
                wave,
                "_load_old_runtime_evidence",
                side_effect=lambda **kwargs: copy.deepcopy(
                    values["runtime_evidence"][kwargs["stage"]]
                ),
            ),
            mock.patch.object(
                wave,
                "prove_source_ancestry",
                side_effect=lambda repository_path, **kwargs: copy.deepcopy(
                    values["source_ancestry"][
                        Path(repository_path).name
                    ]
                ),
            ),
            mock.patch.object(
                wave,
                "replay_wave_file",
                return_value=values["predecessor"],
            ),
        ):
            return wave.authorize_wave_from_replayed_inputs(
                decision_path=decision_path,
                expected_decision_sha256=SHA_A,
                stage_plans=values["stage_plans"],
            )

    def test_real_adapter_normalizes_decision_bridge_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            values = adapter_fixture(root)
            receipt = self.run_adapter(values, root)
        self.assertEqual(receipt["trigger_stages"], ["upper"])
        self.assertEqual(
            [stage["stage"] for stage in receipt["stages"]],
            list(wave.STAGES),
        )
        self.assertTrue(
            all(
                stage["old_segment"]["candidate_catalog_receipt"]
                == values["bridge"]["candidate_index_receipt"]
                for stage in receipt["stages"]
            )
        )

    def test_adapter_rejects_selected_bridge_checkpoint_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            values = adapter_fixture(root)
            values["bridge"]["selected"]["hands"][
                "candidate_checkpoint"
            ]["sha256"] = SHA_D
            with self.assertRaisesRegex(
                wave.ContinuationWaveError,
                "differs from catalog",
            ):
                self.run_adapter(values, root)

    def test_adapter_rejects_candidate_index_payload_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            values = adapter_fixture(root)
            values["candidate_index"]["receipt_payload_sha256"] = SHA_A
            with self.assertRaisesRegex(
                wave.ContinuationWaveError,
                "binding changed",
            ):
                self.run_adapter(values, root)

    def test_adapter_rejects_old_source_or_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            values = adapter_fixture(root)
            values["stage_plans"]["global"]["old_config_sha256"] = SHA_B
            with self.assertRaisesRegex(
                wave.ContinuationWaveError,
                "differs from candidate index",
            ):
                self.run_adapter(values, root)

    def test_adapter_recursively_replays_e220_before_e240(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            audit = {
                "format": "test_training_audit_v1",
                "portable_identity": {
                    "commit": OID_C,
                    "tree": OID_D,
                },
            }
            audit_sha = wave.val_contract.canonical_payload_sha256(audit)
            first_data = fixture()
            for stage in wave.STAGES:
                first_data["stage_plans"][stage]["new_source"][
                    "source_receipt_sha256"
                ] = audit_sha
            first = authorize(first_data)
            recursive = recursive_fixture(first)
            values = adapter_fixture(
                root,
                data=recursive,
                predecessor=first,
            )
            values["candidate_index"]["segmented_union"] = {
                "format": "test_segmented_union_v1"
            }
            for stage in wave.STAGES:
                source = values["candidate_index"]["source_receipts"][
                    stage
                ]
                source["training_audit"] = copy.deepcopy(audit)
                # The union wrapper receipt is intentionally distinct from
                # the preceding wave's new-source training-audit receipt.
                source["receipt_payload_sha256"] = SHA_D
            receipt = self.run_adapter(values, root)
            drifted = copy.deepcopy(values)
            drifted["stage_plans"]["face"]["old_source"][
                "source_receipt_sha256"
            ] = SHA_D
            with self.assertRaisesRegex(
                wave.ContinuationWaveError,
                "differs from candidate index",
            ):
                self.run_adapter(drifted, root)
        self.assertEqual(receipt["boundary_epoch"], 220)
        self.assertEqual(receipt["target_epoch"], 240)
        self.assertTrue(
            all(
                len(stage["old_segment"]["candidate_segment_chain"])
                == 2
                for stage in receipt["stages"]
            )
        )


class SourceAncestryProofTests(unittest.TestCase):
    def git(self, root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_real_git_proof_requires_clean_detached_branchless_ancestry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.git(root, "init", "-q")
            self.git(
                root,
                "config",
                "user.name",
                "Xiangyue-Zhang",
            )
            self.git(
                root,
                "config",
                "user.email",
                "85532891+Xiangyue-Zhang@users.noreply.github.com",
            )
            self.git(
                root,
                "remote",
                "add",
                "origin",
                wave.val_contract.EXPECTED_ORIGIN,
            )
            source_file = root / "source.txt"
            commits = []
            trees = []
            for marker in ("baseline", "old", "new"):
                source_file.write_text(marker + "\n")
                self.git(root, "add", "source.txt")
                self.git(root, "commit", "-q", "-m", marker)
                commits.append(self.git(root, "rev-parse", "HEAD"))
                trees.append(self.git(root, "rev-parse", "HEAD^{tree}"))
            branch = self.git(
                root,
                "symbolic-ref",
                "--short",
                "HEAD",
            )
            self.git(root, "checkout", "-q", "--detach", commits[-1])
            self.git(root, "branch", "-D", branch)
            old_source = {
                "commit": commits[1],
                "tree": trees[1],
                "source_receipt_sha256": SHA_A,
            }
            new_source = {
                "commit": commits[2],
                "tree": trees[2],
                "source_receipt_sha256": SHA_B,
            }
            with mock.patch.object(
                wave,
                "OFFICIAL_BASELINE_COMMIT",
                commits[0],
            ):
                proof = wave.prove_source_ancestry(
                    root,
                    old_source=old_source,
                    new_source=new_source,
                )
            self.assertTrue(proof["clean"])
            self.assertTrue(proof["detached_head"])
            self.assertEqual(proof["local_branch_ref_count"], 0)
            self.assertTrue(proof["old_is_ancestor"])
            self.assertTrue(proof["baseline_is_ancestor"])


if __name__ == "__main__":
    unittest.main()
