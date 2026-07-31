from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import (
    build_segmented_prerequisite_candidate_index as builder,
)
from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract


class SegmentedCandidateIndexContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.prior = self.root / "prior.json"
        self._write_receipt(
            self.prior,
            {
                "format": contract.CANDIDATE_INDEX_FORMAT,
                "kind": "prior",
            },
        )
        self.wave = self.root / "wave.json"
        self._write_receipt(
            self.wave,
            {
                "format": contract.CONTINUATION_WAVE_FORMAT,
                "kind": "wave",
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_receipt(path: Path, value: dict[str, object]) -> None:
        payload = contract.receipt_payload(value)
        path.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _binding(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        receipt = contract.verify_receipt_payload(
            contract.strict_json_bytes(payload, str(path)), str(path)
        )
        return {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
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
                        "optimizer_updates": (
                            epoch * contract.updates_per_epoch(stage)
                        ),
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
        portable_identity = {
            "origin": contract.EXPECTED_ORIGIN,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "script_relative": "show_base_train.py",
            "script_sha256": "3" * 64,
        }
        source_receipts = {
            stage: {"portable_identity": dict(portable_identity)}
            for stage in contract.STAGES
        }
        payload = {
            "format": contract.CANDIDATE_INDEX_FORMAT,
            "status": "complete",
            "target_dataset": "SHOW",
            "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
            "selection_split": "val",
            "test_visible": False,
            "candidate_epochs": epochs,
            "updates_per_epoch": contract.updates_per_epoch_map(
                contract.STAGES
            ),
            "source_policy": contract.build_source_policy(
                source_receipts,
                stages=contract.STAGES,
                reprove_ancestry=False,
            ),
            "source_receipts": source_receipts,
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
            "_validate_bound_prior_candidate_index",
            side_effect=lambda value, **_kwargs: value,
        ), mock.patch.object(
            contract,
            "_replay_bound_continuation_wave",
            side_effect=lambda value, **_kwargs: value,
        ):
            contract.validate_candidate_index(payload)

    @staticmethod
    def _resign(payload: dict[str, object]) -> None:
        segmented_unsigned = dict(payload["segmented_union"])
        segmented_unsigned.pop("receipt_payload_sha256")
        payload["segmented_union"][
            "receipt_payload_sha256"
        ] = contract.canonical_payload_sha256(segmented_unsigned)
        payload["receipt_payload_sha256"] = contract.canonical_payload_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_payload_sha256"
            }
        )

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
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_payload_sha256"
            }
        )
        with self.assertRaisesRegex(contract.ContractError, "escapes"):
            self._validate(payload)

    def test_segment_chain_and_bound_artifact_tampering_fail(self) -> None:
        payload = self._payload()
        payload["segmented_union"]["candidate_segment_chain"]["face"][1][
            "start_epoch"
        ] = 240
        unsigned = dict(payload["segmented_union"])
        unsigned.pop("receipt_payload_sha256")
        payload["segmented_union"][
            "receipt_payload_sha256"
        ] = contract.canonical_payload_sha256(unsigned)
        payload["receipt_payload_sha256"] = contract.canonical_payload_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_payload_sha256"
            }
        )
        with self.assertRaisesRegex(contract.ContractError, "segmented chain"):
            self._validate(payload)

        payload = self._payload()
        self.wave.write_bytes(b"changed\n")
        with self.assertRaisesRegex(contract.ContractError, "SHA-256 mismatch"):
            self._validate(payload)

    def test_bound_artifact_payload_hash_and_format_are_recomputed(self) -> None:
        for label in ("prior", "wave"):
            with self.subTest(label=label):
                payload = self._payload()
                binding = (
                    payload["segmented_union"]["prior_candidate_index"]
                    if label == "prior"
                    else payload["segmented_union"]["continuation_waves"][0]
                )
                binding["receipt_payload_sha256"] = "f" * 64
                self._resign(payload)
                with self.assertRaisesRegex(
                    contract.ContractError, "payload SHA mismatch"
                ):
                    self._validate(payload)

        payload = self._payload()
        self._write_receipt(
            self.wave,
            {
                "format": contract.CANDIDATE_INDEX_FORMAT,
                "kind": "wrong-artifact-type",
            },
        )
        payload["segmented_union"]["continuation_waves"][0] = self._binding(
            self.wave
        )
        self._resign(payload)
        with self.assertRaisesRegex(contract.ContractError, "format mismatch"):
            self._validate(payload)
        self._write_receipt(
            self.wave,
            {
                "format": contract.CONTINUATION_WAVE_FORMAT,
                "kind": "wave",
            },
        )

    def test_rehashed_bound_artifacts_still_require_full_semantics(self) -> None:
        prior_payload = self.prior.read_bytes()
        prior_value = contract.strict_json_bytes(
            prior_payload, str(self.prior)
        )
        with self.assertRaisesRegex(
            contract.ContractError, "candidate index protocol mismatch"
        ):
            contract._validate_bound_prior_candidate_index(
                prior_value,
                path=self.prior,
                artifact_stack=frozenset(),
            )

        wave_payload = self.wave.read_bytes()
        wave_value = contract.strict_json_bytes(wave_payload, str(self.wave))
        with self.assertRaisesRegex(
            contract.ContractError, "semantic replay failed"
        ):
            contract._replay_bound_continuation_wave(
                wave_value,
                path=self.wave,
                expected_sha256=hashlib.sha256(wave_payload).hexdigest(),
                artifact_stack=frozenset(),
            )

    def test_recursive_artifact_cycles_are_rejected(self) -> None:
        prior_value = contract.strict_json_bytes(
            self.prior.read_bytes(), str(self.prior)
        )
        with self.assertRaisesRegex(contract.ContractError, "cycle detected"):
            contract._validate_bound_prior_candidate_index(
                prior_value,
                path=self.prior,
                artifact_stack=frozenset({self.prior}),
            )

        wave_payload = self.wave.read_bytes()
        wave_sha = hashlib.sha256(wave_payload).hexdigest()
        with self.assertRaisesRegex(
            wave.ContinuationWaveError, "cycle detected"
        ):
            wave.replay_wave_file(
                self.wave,
                wave_sha,
                _wave_stack=frozenset({self.wave}),
            )

    def test_bound_wave_rejects_leaf_replacement_after_open(self) -> None:
        payload = self._payload()
        original = self.wave.read_bytes()
        replacement = original.replace(b'"kind": "wave"', b'"kind": "swap"')
        archived = self.root / "opened-wave.json"
        real_open = os.open
        swapped = False

        def racing_open(path, flags, *args, **kwargs):
            nonlocal swapped
            descriptor = real_open(path, flags, *args, **kwargs)
            if path == self.wave.name and not swapped:
                swapped = True
                self.wave.rename(archived)
                self.wave.write_bytes(replacement)
            return descriptor

        with mock.patch.object(
            contract.os, "open", side_effect=racing_open
        ), self.assertRaisesRegex(contract.ContractError, "path changed"):
            self._validate(payload)
        self.assertTrue(swapped)
        self.assertEqual(archived.read_bytes(), original)
        self.assertEqual(self.wave.read_bytes(), replacement)

    def test_candidate_rejects_leaf_replacement_after_open(self) -> None:
        payload = self._payload()
        target = (
            self.root
            / "face"
            / "prefix"
            / "representation_candidates"
            / "e20.bin"
        )
        original = target.read_bytes()
        replacement = b"evil:20"
        self.assertEqual(len(replacement), len(original))
        archived = target.with_name("opened-e20.bin")
        real_open = os.open
        swapped = False

        def racing_open(path, flags, *args, **kwargs):
            nonlocal swapped
            descriptor = real_open(path, flags, *args, **kwargs)
            if path == target.name and not swapped:
                swapped = True
                target.rename(archived)
                target.write_bytes(replacement)
            return descriptor

        with mock.patch.object(
            contract.os, "open", side_effect=racing_open
        ), self.assertRaisesRegex(contract.ContractError, "path changed"):
            self._validate(payload)
        self.assertTrue(swapped)
        self.assertEqual(archived.read_bytes(), original)
        self.assertEqual(target.read_bytes(), replacement)

    def test_status_snapshot_rejects_leaf_replacement_after_open(self) -> None:
        run = self.root / "status-run"
        run.mkdir()
        status_path = run / "formal_training_status.json"
        original = b'{"marker":"original"}\n'
        replacement = b'{"marker":"replacement"}\n'
        status_path.write_bytes(original)
        archived = run / "opened-status.json"
        real_open = os.open
        swapped = False

        def racing_open(path, flags, *args, **kwargs):
            nonlocal swapped
            descriptor = real_open(path, flags, *args, **kwargs)
            if path == status_path.name and not swapped:
                swapped = True
                status_path.rename(archived)
                status_path.write_bytes(replacement)
            return descriptor

        with mock.patch.object(
            contract.os, "open", side_effect=racing_open
        ), self.assertRaisesRegex(contract.ContractError, "path changed"):
            builder._read_status_snapshot(run, "face")
        self.assertTrue(swapped)
        self.assertEqual(status_path.read_bytes(), replacement)
        self.assertEqual(archived.read_bytes(), original)

    def test_status_snapshot_rejects_symlink_leaf(self) -> None:
        run = self.root / "symlink-status-run"
        run.mkdir()
        target = run / "real-status.json"
        target.write_text('{"status":"complete"}\n', encoding="utf-8")
        (run / "formal_training_status.json").symlink_to(target)
        with self.assertRaisesRegex(contract.ContractError, "non-symlink"):
            builder._read_status_snapshot(run, "face")


if __name__ == "__main__":
    unittest.main()
