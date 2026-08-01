from __future__ import annotations

import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts.show_base import base_v14_formal_contract as contract
from scripts.show_base import train_base_official_adapt_long as training


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = (
    ROOT
    / "configs"
    / "show_base"
    / "semtalk_base_v14_formal_schedule_20260802.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _sha(path)


class V14FormalContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.schedule = self.root / "schedule.json"
        self.schedule.write_bytes(SCHEDULE.read_bytes())
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text("{}\n", encoding="utf-8")
        self.gate = self.root / "gate.json"
        self.gate.write_text("{}\n", encoding="utf-8")
        self.report_artifacts = [
            {
                "mode": mode,
                "path": str((self.root / f"{mode}.json").resolve()),
                "sha256": f"{index + 1:064x}",
                "bytes": 100 + index,
                "payload_sha256": f"{index + 11:064x}",
            }
            for index, mode in enumerate(sorted(contract.MODES))
        ]
        self.pipeline_source = {
            "origin": contract.ORIGIN,
            "source_root": "/pinned/validation/source",
            "commit": contract.VALIDATION_SOURCE_COMMIT,
            "tree": contract.VALIDATION_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        self.validation_authority = {
            "origin": contract.ORIGIN,
            "commit": contract.VALIDATION_SOURCE_COMMIT,
            "tree": contract.VALIDATION_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
            "selector_sha256": "a" * 64,
            "training_contract_sha256": "b" * 64,
            "validation_contract_sha256": "c" * 64,
        }
        self.validated = []
        results = []
        for mode_index, mode in enumerate(contract.MODES):
            candidates = []
            for epoch_index, epoch in enumerate(contract.CANDIDATE_EPOCHS):
                value = 0.03 + mode_index * 0.01 + epoch_index * 0.001
                if mode == contract.MODE_P2 and epoch == 8:
                    value = 0.01
                checkpoint = {"sha256": f"{100 + mode_index * 10 + epoch_index:064x}"}
                candidates.append(
                    {
                        "epoch": epoch,
                        "diffsheg_fgd": value,
                        "candidate_checkpoint": checkpoint,
                    }
                )
                results.append(
                    {
                        "topology_id": contract.MODE_IDS[mode],
                        "topology_mode": mode,
                        "epoch": epoch,
                        "validation_diffsheg_fgd": value,
                        "checkpoint_sha256": checkpoint["sha256"],
                    }
                )
            artifact = next(
                row for row in self.report_artifacts if row["mode"] == mode
            )
            self.validated.append(
                {
                    "mode": mode,
                    "report_sha256": artifact["sha256"],
                    "topology_independent_input_sha256": "d" * 64,
                    "candidates": candidates,
                }
            )
        winner = min(
            results,
            key=lambda row: (
                row["validation_diffsheg_fgd"],
                row["epoch"],
                row["topology_mode"],
            ),
        )
        self.protocol_artifact = {
            "path": str(self.protocol.resolve()),
            "sha256": contract.SELECTION_PROTOCOL_SHA256,
            "bytes": self.protocol.stat().st_size,
        }
        self.audit = {
            "format": contract.AUDIT_FORMAT,
            "status": "complete",
            "selection_protocol": self.protocol_artifact,
            "selection_tuple": list(contract.SELECTION_TUPLE),
            "validation_split": "val",
            "test_visible": False,
            "test_measurements_authorized": 0,
            "training_source_authority": {
                "origin": contract.ORIGIN,
                "commit": contract.TRAINING_SOURCE_COMMIT,
                "tree": contract.TRAINING_SOURCE_TREE,
            },
            "validation_source_authority": self.validation_authority,
            "validation_pipeline_source": self.pipeline_source,
            "quality_reports": self.report_artifacts,
            "quality_report_sha256": {
                row["mode"]: row["sha256"] for row in self.report_artifacts
            },
            "topology_independent_input_sha256": "d" * 64,
            "common_authority": {
                "source_sha256": "1" * 64,
                "data_vq_sha256": "2" * 64,
                "initialization_sha256": "3" * 64,
                "runtime_objective_sha256": "4" * 64,
            },
            "results": results,
            "winner": winner,
            "trainer_native_selection": {
                "role": "audit_only",
                "authoritative_for_winner": False,
                "status": "unavailable_not_authoritative",
                "reason_code": "trainer_native_audit_not_provided",
                "evidence_receipt": None,
                "probe_modes": [],
                "blocking_mode": None,
            },
            "formal_training": {
                "target_epochs": 400,
                "fresh": True,
                "restart_from_same_frozen_initialization": True,
                "selected_quality_checkpoint_is_evidence_only": True,
                "checkpoint_selection_split": "val",
                "final_test_runs_after_formal_selection": 1,
            },
        }
        self._write_audit(self.audit)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_audit(self, body: dict[str, object]) -> None:
        body.pop("receipt_sha256", None)
        body["receipt_sha256"] = contract.canonical_json_sha256(body)
        self.audit_path = self.root / "audit.json"
        self.audit_sha = _write_json(self.audit_path, body)

    def _call(
        self,
        *,
        gate_source_commit: str = "f" * 40,
        gate_run_id: str = "gate-run-0001",
        gate_port: int = 22101,
    ):
        project = {
            "origin": contract.ORIGIN,
            "commit": "f" * 40,
            "tree": "a" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        selector = mock.Mock()
        selector.validate_probe.return_value = {
            "mode": contract.MODE_P2,
            "status": "pass",
            "report_sha256": _sha(self.gate),
        }
        v14_selector = mock.Mock()
        v14_selector.TRAINER_NATIVE_NOT_PROVIDED_REASON = (
            "trainer_native_audit_not_provided"
        )
        v14_selector.TRAINER_NATIVE_UNAVAILABLE_REASON = (
            "w1_reference_quality_report_unavailable"
        )
        v14_selector.OFFICIAL_SELECTOR_SHA256 = "a" * 64
        v14_selector.OFFICIAL_TRAIN_CONTRACT_SHA256 = "b" * 64
        v14_selector.OFFICIAL_VALIDATION_CONTRACT_SHA256 = "c" * 64
        frozen_by_mode = {
            mode: {
                "source": {
                    "origin": contract.ORIGIN,
                    "commit": contract.TRAINING_SOURCE_COMMIT,
                    "tree": contract.TRAINING_SOURCE_TREE,
                    "clean": True,
                },
                "run_purpose": "topology_short_quality",
                "target_epochs": list(contract.CANDIDATE_EPOCHS),
                "protocol": {
                    "distributed_topology": {
                        "mode": mode,
                        "formal_run_id": f"quality-run-{index + 1:04d}",
                        "master_port": 21001 + index,
                    }
                },
            }
            for index, mode in enumerate(contract.MODES)
        }
        v14_selector._load_mode_frozen.side_effect = (
            lambda _validated, mode: frozen_by_mode[mode]
        )
        v14_selector._authority_projection.side_effect = (
            lambda _frozen, _mode, _training: copy.deepcopy(
                self.audit["common_authority"]
            )
        )
        v14_selector._validation_pipeline_authority.side_effect = (
            lambda _validated, _mode: copy.deepcopy(self.pipeline_source)
        )
        with (
            mock.patch.object(
                contract,
                "validate_protocol",
                return_value=({}, self.protocol_artifact),
            ),
            mock.patch.object(
                contract,
                "_validated_quality_reports",
                return_value=(self.validated, self.report_artifacts),
            ),
            mock.patch.object(
                contract,
                "current_project_authority",
                return_value=project,
            ),
            mock.patch.object(
                contract,
                "_gate_source_and_identity",
                return_value=(
                    {
                        "origin": contract.ORIGIN,
                        "commit": gate_source_commit,
                        "tree": project["tree"],
                        "clean": True,
                    },
                    gate_run_id,
                    gate_port,
                ),
            ),
        ):
            return contract.validate_control_plane(
                project_root=ROOT,
                schedule_path=self.schedule,
                expected_schedule_sha256=_sha(self.schedule),
                protocol_path=self.protocol,
                expected_protocol_sha256=contract.SELECTION_PROTOCOL_SHA256,
                audit_path=self.audit_path,
                expected_audit_sha256=self.audit_sha,
                throughput_gate_path=self.gate,
                expected_throughput_gate_sha256=_sha(self.gate),
                topology_mode=contract.MODE_P2,
                formal_run_id="formal-run-0001",
                formal_master_port=22202,
                selector=selector,
                v14_selector=v14_selector,
                training_contract=training,
                topology_specs=training.TOPOLOGY_SPECS,
            )

    def test_accepts_v14_winner_when_native_selection_is_unavailable(self) -> None:
        receipt = self._call()
        self.assertEqual(receipt["selected_mode"], contract.MODE_P2)
        self.assertEqual(receipt["winner_epoch"], 8)
        self.assertTrue(receipt["native_nine_mode_is_not_v14_authority"])

    def test_rejects_nonminimum_winner_after_valid_self_hash(self) -> None:
        forged = copy.deepcopy(self.audit)
        forged["winner"] = forged["results"][0]
        self._write_audit(forged)
        with self.assertRaisesRegex(contract.V14ContractError, "unique minimum"):
            self._call()

    def test_rejects_authoritative_native_claim_after_reseal(self) -> None:
        forged = copy.deepcopy(self.audit)
        forged["trainer_native_selection"][
            "authoritative_for_winner"
        ] = True
        self._write_audit(forged)
        with self.assertRaisesRegex(
            contract.V14ContractError,
            "native audit schema",
        ):
            self._call()

    def test_rejects_forged_common_authority_after_reseal(self) -> None:
        forged = copy.deepcopy(self.audit)
        forged["common_authority"]["source_sha256"] = "9" * 64
        self._write_audit(forged)
        with self.assertRaisesRegex(contract.V14ContractError, "validation source"):
            self._call()

    def test_rejects_gate_from_old_source(self) -> None:
        with self.assertRaisesRegex(contract.V14ContractError, "current source"):
            self._call(gate_source_commit=contract.TRAINING_SOURCE_COMMIT)

    def test_rejects_reused_gate_execution_identity(self) -> None:
        with self.assertRaisesRegex(contract.V14ContractError, "distinct run ID/port"):
            self._call(gate_run_id="formal-run-0001")
        with self.assertRaisesRegex(contract.V14ContractError, "distinct run ID/port"):
            self._call(gate_port=22202)
        with self.assertRaisesRegex(contract.V14ContractError, "distinct run ID/port"):
            self._call(gate_run_id="quality-run-0001")
        with self.assertRaisesRegex(contract.V14ContractError, "distinct run ID/port"):
            self._call(gate_port=21002)

    def test_schedule_rejects_nine_mode_label_masquerading_as_v14(self) -> None:
        payload = json.loads(self.schedule.read_text(encoding="utf-8"))
        payload["training"]["topology_source"] = "sealed_nine_mode_topology_gate_v1"
        with self.assertRaisesRegex(contract.V14ContractError, "truthful"):
            contract.validate_schedule_payload(payload)

    def test_control_plane_rejects_resealed_schedule(self) -> None:
        payload = json.loads(self.schedule.read_text(encoding="utf-8"))
        payload["operator_note"] = "not part of the frozen V14 schedule"
        _write_json(self.schedule, payload)
        with self.assertRaisesRegex(contract.V14ContractError, "schedule SHA-256"):
            self._call()

    def test_trainer_accepts_truthful_v14_schedule_without_changing_recipe(self) -> None:
        dataset = {
            "format": "semtalk_show_base_selected_feature_dataset_receipt_v1",
            "split": "train",
            "test_visible": False,
            "data_mdb_sha256": "1" * 64,
            "lock_mdb_sha256": "2" * 64,
            "summary_sha256": "3" * 64,
            "lineage_sha256": "4" * 64,
            "prerequisite_source": training.SHOW_VAL_SELECTED_SOURCE,
            "prerequisite_selection": {
                "sha256": "5" * 64,
            },
            "selected_prerequisite_sha256": {
                stage: f"{index + 10:02x}" * 32
                for index, stage in enumerate(
                    training.selected_contract.STAGES
                )
            },
            "global_verified_not_consumed": True,
            "lmdb_inode_binding": {
                "format": "semtalk_show_base_lmdb_inode_binding_v1",
            },
            "canonical_dataset_evidence": {
                "split_counts": dict(training.EXPECTED_SPLIT_COUNTS),
                "split_disjoint": True,
                "exact_once": True,
                "train_per_clip_ledger_exact": True,
                "test_rows_used_as_training_samples": False,
            },
        }
        args = argparse.Namespace(
            schedule_json=str(self.schedule),
            expected_schedule_sha256=_sha(self.schedule),
            trajectory_mode=training.FRESH_TRAJECTORY_MODE,
            expected_prerequisite_selection_sha256="5" * 64,
            expected_dataset_summary_sha256="3" * 64,
            expected_lineage_sha256="4" * 64,
            precision="bf16",
            learning_rate=3e-5,
            seed=43,
            loader_workers=4,
            topology_mode=contract.MODE_P2,
        )
        receipt = training.validate_long_contract_receipts(
            args, dataset_receipt=dataset
        )
        self.assertEqual(
            receipt["schedule"]["topology_source"],
            contract.TOPOLOGY_SOURCE,
        )
        self.assertEqual(
            receipt["trajectory_anchor"]["probe_optimizer_updates"], 70
        )

    def test_trainer_args_do_not_require_absent_native_selection_for_v14(
        self,
    ) -> None:
        from tests.test_train_base_official_adapt_long_cpu import _base_cli

        args = training.build_parser().parse_args(_base_cli())
        specification = training.TOPOLOGY_SPECS[contract.MODE_P2]
        args.mode = "train"
        args.schedule_json = str(SCHEDULE)
        args.expected_schedule_sha256 = _sha(SCHEDULE)
        args.topology_mode = contract.MODE_P2
        args.local_batch_size = specification["local_batch_size"]
        args.learning_rate = specification["learning_rate"]
        args.precision = specification["precision"]
        args.throughput_gate_report = "/sealed/fresh-gate.json"
        args.expected_throughput_gate_sha256 = "d" * 64
        args.topology_selection_report = None
        args.expected_topology_selection_sha256 = None
        args.v14_selection_protocol = "/sealed/v14-protocol.json"
        args.expected_v14_selection_protocol_sha256 = (
            contract.SELECTION_PROTOCOL_SHA256
        )
        args.v14_selection_audit = "/sealed/v14-audit.json"
        args.expected_v14_selection_audit_sha256 = "e" * 64
        with mock.patch.object(
            training.os,
            "uname",
            return_value=types.SimpleNamespace(
                nodename=training.FORMAL_HOST_BY_NODE_RANK[0]
            ),
        ):
            training.validate_args(args)

    def test_cli_is_cpu_only_and_callable(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.show_base.base_v14_formal_contract",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--selection-audit", result.stdout)
        self.assertNotIn("--trainer-native-selection", result.stdout)


class TrainingSemanticsDiffTest(unittest.TestCase):
    def test_v14_control_plane_does_not_change_training_semantic_functions(self) -> None:
        path = "scripts/show_base/train_base_official_adapt_long.py"
        baseline = subprocess.run(
            ["git", "show", f"{contract.TRAINING_SOURCE_COMMIT}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        current = (ROOT / path).read_text(encoding="utf-8")
        names = {
            "_model_args",
            "read_official_base_checkpoint",
            "strict_load_and_initialize_show_speakers",
            "_coerce_target_zq",
            "_official_forward_loss_family",
            "audio_conditioned_objective",
            "_move_batch",
            "one_optimizer_update",
            "_create_dataloader",
            "_run_throughput_gate",
            "_run_training",
        }

        def projection(source: str) -> dict[str, str]:
            tree = ast.parse(source)
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            }

        baseline_projection = projection(baseline)
        current_projection = projection(current)
        self.assertEqual(set(baseline_projection), names)
        self.assertEqual(current_projection, baseline_projection)


if __name__ == "__main__":
    unittest.main()
