from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract


class SegmentedCandidateIndexContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.prior = self.root / "prior.json"
        self.prior.write_bytes(b"prior\n")
        self.wave = self.root / "wave.json"
        self.wave.write_bytes(b"wave\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _binding(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "receipt_payload_sha256": "a" * 64,
        }

    def _payload(self) -> dict[str, object]:
        epochs = list(range(20, 221, 20))
        stages = {}
        chains = {}
        for stage in contract.STAGES:
            prefix = self.root / stage / "prefix"
            extension = self.root / stage / "extension"
            (prefix / "representation_candidates").mkdir(parents=True, exist_ok=True)
            (extension / "representation_candidates").mkdir(parents=True, exist_ok=True)
            entries = []
            for epoch in epochs:
                run = prefix if epoch <= 200 else extension
                checkpoint = run / "representation_candidates" / f"e{epoch}.bin"
                checkpoint.write_bytes(f"{stage}:{epoch}".encode())
                entries.append(
                    {
                        "epoch": epoch,
                        "optimizer_updates": epoch * contract.EXPECTED_UPDATES_PER_EPOCH,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": contract.sha256_file(checkpoint),
                        "checkpoint_bytes": checkpoint.stat().st_size,
                        "checkpoint_audit_sha256": "b" * 64,
                    }
                )
            first = wave._make_chain_segment(
                run_path=str(prefix),
                start_epoch=20,
                end_epoch=200,
                predecessor_segment_id=None,
            )
            second = wave._make_chain_segment(
                run_path=str(extension),
                start_epoch=220,
                end_epoch=220,
                predecessor_segment_id=first["segment_id"],
            )
            stages[stage] = entries
            chains[stage] = [first, second]
        segmented = {
            "format": contract.SEGMENTED_UNION_FORMAT,
            "status": "complete",
            "prior_candidate_index": self._binding(self.prior),
            "continuation_waves": [self._binding(self.wave)],
            "candidate_segment_chain": chains,
        }
        segmented["receipt_payload_sha256"] = contract.canonical_payload_sha256(segmented)
        payload = {
            "format": contract.CANDIDATE_INDEX_FORMAT,
            "status": "complete",
            "target_dataset": "SHOW",
            "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
            "selection_split": "val",
            "test_visible": False,
            "candidate_epochs": epochs,
            "updates_per_epoch": contract.EXPECTED_UPDATES_PER_EPOCH,
            "source_receipts": {
                stage: {"portable_identity": {"stage": stage}}
                for stage in contract.STAGES
            },
            "config_sha256": {stage: "c" * 64 for stage in contract.STAGES},
            "dataset_receipt_sha256": {stage: "d" * 64 for stage in contract.STAGES},
            "formal_training_status": {stage: {} for stage in contract.STAGES},
            "stages": stages,
            "segmented_union": segmented,
        }
        payload["receipt_payload_sha256"] = contract.canonical_payload_sha256(payload)
        return payload

    def _validate(self, payload: dict[str, object]) -> None:
        with mock.patch.object(
            contract,
            "validate_frozen_training_source",
            side_effect=lambda value, *_args, **_kwargs: value,
        ), mock.patch.object(
            contract,
            "portable_training_source_identity",
            return_value={"same": True},
        ):
            contract.validate_candidate_index(payload)

    def test_exact_segmented_union_is_accepted(self) -> None:
        self._validate(self._payload())

    def test_candidate_cannot_escape_its_segment(self) -> None:
        payload = self._payload()
        rogue = self.root / "rogue" / "representation_candidates" / "e220.bin"
        rogue.parent.mkdir(parents=True)
        rogue.write_bytes(b"face:220")
        payload["stages"]["face"][-1]["checkpoint"] = str(rogue)
        payload["stages"]["face"][-1]["checkpoint_sha256"] = contract.sha256_file(rogue)
        payload["stages"]["face"][-1]["checkpoint_bytes"] = rogue.stat().st_size
        payload["receipt_payload_sha256"] = contract.canonical_payload_sha256(
            {key: value for key, value in payload.items() if key != "receipt_payload_sha256"}
        )
        with self.assertRaisesRegex(contract.ContractError, "escapes"):
            self._validate(payload)

    def test_segment_chain_and_bound_artifact_tampering_fail(self) -> None:
        payload = self._payload()
        payload["segmented_union"]["candidate_segment_chain"]["face"][1]["start_epoch"] = 240
        unsigned = dict(payload["segmented_union"])
        unsigned.pop("receipt_payload_sha256")
        payload["segmented_union"]["receipt_payload_sha256"] = contract.canonical_payload_sha256(unsigned)
        payload["receipt_payload_sha256"] = contract.canonical_payload_sha256(
            {key: value for key, value in payload.items() if key != "receipt_payload_sha256"}
        )
        with self.assertRaisesRegex(contract.ContractError, "segmented chain"):
            self._validate(payload)

        payload = self._payload()
        self.wave.write_bytes(b"changed\n")
        with self.assertRaisesRegex(contract.ContractError, "changed"):
            self._validate(payload)


if __name__ == "__main__":
    unittest.main()
