from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import finalize_prerequisite_val_selection as finalizer
from scripts.show_base import prerequisite_val_contract as contract


class FinalizerAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        schedule = list(contract.REQUIRED_CANDIDATE_EPOCHS)
        self.full = {
            "format": contract.CANDIDATE_INDEX_FORMAT,
            "candidate_epochs": schedule,
        }
        for key in (
            "source_receipts",
            "config_sha256",
            "dataset_receipt_sha256",
            "formal_training_status",
            "stages",
        ):
            self.full[key] = {
                stage: {
                    "binding": f"{key}:{stage}",
                    "schedule": schedule,
                }
                for stage in contract.STAGES
            }
        self.partial = {
            "format": contract.PARTIAL_CANDIDATE_INDEX_FORMAT,
            "candidate_epochs": schedule,
        }
        for key in (
            "source_receipts",
            "config_sha256",
            "dataset_receipt_sha256",
            "formal_training_status",
            "stages",
        ):
            self.partial[key] = {
                stage: copy.deepcopy(self.full[key][stage])
                for stage in contract.STAGES[:-1]
            }

    def test_exact_partial_subset_is_accepted(self) -> None:
        finalizer.validate_partial_full_authority(
            self.partial,
            self.full,
        )

    def test_omission_substitution_schedule_and_sha_drift_fail(self) -> None:
        attacks = []
        omitted = copy.deepcopy(self.partial)
        omitted["stages"].pop("upper")
        attacks.append(omitted)
        substituted = copy.deepcopy(self.partial)
        substituted["stages"]["global"] = substituted["stages"].pop(
            "lower"
        )
        attacks.append(substituted)
        schedule = copy.deepcopy(self.partial)
        schedule["candidate_epochs"] = schedule["candidate_epochs"][:-1]
        attacks.append(schedule)
        drift = copy.deepcopy(self.partial)
        drift["config_sha256"]["hands"]["binding"] = "forged"
        attacks.append(drift)
        for attack in attacks:
            with self.assertRaises(contract.ContractError):
                finalizer.validate_partial_full_authority(
                    attack,
                    self.full,
                )

    def test_raw_root_symlink_fails_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            other = root / "other"
            real.mkdir()
            other.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                contract.ContractError,
                "absolute regular",
            ):
                finalizer._validate_partition_union(
                    roots=[link, other],
                    candidate_index=self.full,
                    candidate_index_path=root / "full.json",
                    candidate_index_sha256="1" * 64,
                    nonglobal_candidate_index=self.partial,
                    nonglobal_candidate_index_path=root / "partial.json",
                    nonglobal_candidate_index_sha256="2" * 64,
                    manifest_sha256="3" * 64,
                    summary_sha256="4" * 64,
                    lineage_sha256="5" * 64,
                    source_commit="6" * 40,
                    source_tree="7" * 40,
                )

    def test_partition_jobs_and_shards_form_exact_disjoint_union(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            full_path = root / "full.json"
            partial_path = root / "partial.json"
            full_path.write_text("{}\n")
            partial_path.write_text("{}\n")
            schedule = list(contract.REQUIRED_CANDIDATE_EPOCHS)
            full = {
                "format": contract.CANDIDATE_INDEX_FORMAT,
                "candidate_epochs": schedule,
                "stages": {},
            }
            for stage in contract.STAGES:
                full["stages"][stage] = [
                    {
                        "epoch": epoch,
                        "optimizer_updates": epoch * 497,
                        "checkpoint": str(
                            (root / f"{stage}-{epoch}.bin").resolve()
                        ),
                        "checkpoint_sha256": f"{epoch:064x}",
                        "checkpoint_bytes": 1,
                        "checkpoint_audit_sha256": f"{epoch + 1:064x}",
                    }
                    for epoch in schedule
                ]
            partial = {
                "format": contract.PARTIAL_CANDIDATE_INDEX_FORMAT,
                "candidate_epochs": schedule,
                "stages": {
                    stage: full["stages"][stage]
                    for stage in contract.STAGES[:-1]
                },
            }
            roots = []
            common = {
                "canonical_manifest_sha256": "3" * 64,
                "canonical_summary_sha256": "4" * 64,
                "canonical_lineage_sha256": "5" * 64,
                "source_commit": "6" * 40,
                "source_tree": "7" * 40,
            }
            for partition, stages, authority, authority_sha in (
                (
                    "nonglobal",
                    contract.STAGES[:-1],
                    partial_path,
                    "2" * 64,
                ),
                (
                    "global",
                    ("global",),
                    full_path,
                    "1" * 64,
                ),
            ):
                partition_root = root / partition
                partition_root.mkdir()
                jobs = finalizer._expected_jobs(full, stages)
                for job in jobs:
                    directory = (
                        partition_root
                        / "shards"
                        / job["stage"]
                        / f"epoch_{job['epoch']:04d}"
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    for shard in range(8):
                        (directory / f"shard_{shard:02d}.json").write_text(
                            "{}\n"
                        )
                receipt = contract.receipt_payload(
                    {
                        "format": finalizer.PARTITION_FORMAT,
                        "status": "complete",
                        "partition": partition,
                        "test_visible": False,
                        "protocol": {
                            "candidates_per_wave": 4,
                            "shards_per_candidate": 8,
                        },
                        "timing": {},
                        "inputs": {
                            "candidate_index": str(authority.resolve()),
                            "candidate_index_sha256": authority_sha,
                            **common,
                            "multi_candidate_gate": {
                                "path": str((root / f"{partition}-gate.json").resolve()),
                                "sha256": "8" * 64,
                            },
                        },
                        "jobs": jobs,
                        "coverage": {
                            "candidate_jobs": len(jobs),
                            "shard_jobs": len(jobs) * 8,
                            "exact_once": True,
                        },
                    }
                )
                (partition_root / "partition_receipt.json").write_text(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                roots.append(partition_root)

            def replay(path: Path, digest: str) -> dict[str, object]:
                authority = partial_path if "nonglobal" in path.name else full_path
                authority_sha = "2" * 64 if authority == partial_path else "1" * 64
                return {
                    "inputs": {
                        "common": {
                            "candidate_index": str(authority.resolve()),
                            "candidate_index_sha256": authority_sha,
                            **common,
                            "multi_candidate_gate": None,
                        }
                    }
                }

            arguments = {
                "roots": roots,
                "candidate_index": full,
                "candidate_index_path": full_path,
                "candidate_index_sha256": "1" * 64,
                "nonglobal_candidate_index": partial,
                "nonglobal_candidate_index_path": partial_path,
                "nonglobal_candidate_index_sha256": "2" * 64,
                "manifest_sha256": common["canonical_manifest_sha256"],
                "summary_sha256": common["canonical_summary_sha256"],
                "lineage_sha256": common["canonical_lineage_sha256"],
                "source_commit": common["source_commit"],
                "source_tree": common["source_tree"],
            }
            with mock.patch.object(
                finalizer.multicandidate_gate,
                "replay_gate",
                side_effect=replay,
            ):
                self.assertEqual(
                    finalizer._validate_partition_union(**arguments),
                    roots,
                )
                missing = roots[1] / "shards/global/epoch_0200/shard_07.json"
                missing.unlink()
                with self.assertRaisesRegex(
                    contract.ContractError,
                    "inventory",
                ):
                    finalizer._validate_partition_union(**arguments)


if __name__ == "__main__":
    unittest.main()
