from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_long_val_contract as long_contract
from scripts.show_base import validate_base_long_test_winner as validator


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class LongBaseTestWinnerContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="semtalk-long-winner-"
        )
        self.root = Path(self.temporary.name).resolve()
        self.bundle, self.selection_path = self._fixture()
        self.selection_sha = _sha(self.selection_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture(self) -> tuple[dict[str, object], Path]:
        candidates: dict[int, dict[str, object]] = {}
        rows = []
        measurements = []
        for index, epoch in enumerate(
            long_contract.EXPECTED_CANDIDATE_EPOCHS
        ):
            checkpoint = self.root / f"candidate-e{epoch:04d}.bin"
            checkpoint.write_bytes(f"candidate-{epoch}".encode("ascii"))
            checkpoint_binding = {
                "path": str(checkpoint),
                "sha256": _sha(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
            candidates[epoch] = checkpoint_binding
            inference = self.root / f"val-lineage-e{epoch:04d}.json"
            report = self.root / f"val-diffsheg-e{epoch:04d}.json"
            _write_json(inference, {"epoch": epoch, "split": "val"})
            # e200 is the unique validation winner.
            fgd = 0.125 if epoch == 200 else index + 1.0
            _write_json(report, {"epoch": epoch, "fgd": fgd})
            measurement = self.root / f"measurement-e{epoch:04d}.json"
            _write_json(
                measurement,
                {
                    "epoch": epoch,
                    "diffsheg_report": {
                        "path": str(report),
                        "sha256": _sha(report),
                    },
                },
            )
            measurements.append(
                {"path": str(measurement), "sha256": _sha(measurement)}
            )
            rows.append(
                {
                    "epoch": epoch,
                    "metrics": {"fgd": fgd},
                    "candidate_checkpoint": checkpoint_binding,
                    "inference_lineage": {
                        "path": str(inference),
                        "sha256": _sha(inference),
                    },
                    "diffsheg_report": {
                        "path": str(report),
                        "sha256": _sha(report),
                    },
                }
            )
        bundle: dict[str, object] = {
            "manifest": {
                "path": str(self.root / "manifest.json"),
                "sha256": "1" * 64,
            },
            "status": {
                "path": str(self.root / "status.json"),
                "sha256": "2" * 64,
            },
            "frozen_inputs": {
                "path": str(self.root / "frozen.json"),
                "sha256": "3" * 64,
                "receipt_sha256": "4" * 64,
            },
            "candidates": candidates,
        }
        winner = next(row for row in rows if row["epoch"] == 200)
        selection = {
            "format": long_contract.SELECTION_FORMAT,
            "status": "selected",
            "split": "val",
            "test_visible": False,
            "selection_eligible": True,
            "selection_policy": {
                "candidate_epochs": list(
                    long_contract.EXPECTED_CANDIDATE_EPOCHS
                ),
                "metric": "FGD",
                "metric_report_key": "fgd",
                "operator": "min",
                "tie_break": "lowest_epoch",
                "ordering": ["fgd", "epoch"],
                "test_feedback_into_selection": False,
            },
            "candidate_bundle": {
                key: bundle[key]
                for key in ("manifest", "status", "frozen_inputs")
            },
            "candidate_metrics": rows,
            "measurement_receipts": measurements,
            "selected": {
                "epoch": 200,
                "candidate_checkpoint": winner["candidate_checkpoint"],
                "fgd": 0.125,
                "inference_lineage": winner["inference_lineage"],
                "diffsheg_report": winner["diffsheg_report"],
            },
            "test_policy": {
                "authorized_evaluations": 1,
                "one_shot_claim_required": True,
                "selection_feedback": False,
            },
        }
        selection["receipt_payload_sha256"] = (
            long_contract.canonical_json_sha256(selection)
        )
        selection_path = self.root / "validation-selection.json"
        _write_json(selection_path, selection)
        return bundle, selection_path

    def _authorize(
        self,
        *,
        epoch: int = 200,
        selection_path: Path | None = None,
    ) -> dict[str, object]:
        checkpoint = self.bundle["candidates"][epoch]
        active_selection = selection_path or self.selection_path
        with mock.patch.object(
            validator.long_selector,
            "build_selection",
            side_effect=lambda **kwargs: self._replay_selection(
                active_selection,
                **kwargs,
            ),
        ):
            return validator.validate_test_winner(
                selection_path=active_selection,
                expected_selection_sha256=_sha(active_selection),
                checkpoint_path=Path(checkpoint["path"]),
                expected_checkpoint_sha256=checkpoint["sha256"],
                candidate_bundle=self.bundle,
            )

    def _replay_selection(
        self,
        selection_path: Path,
        *,
        candidate_bundle: object,
        measurement_paths: object,
        expected_measurement_sha256: object,
    ) -> dict[str, object]:
        self.assertEqual(candidate_bundle, self.bundle)
        selection = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
        rows = selection["candidate_metrics"]
        self.assertEqual(len(measurement_paths), len(rows))
        self.assertEqual(len(expected_measurement_sha256), len(rows))
        for row, path, expected_sha in zip(
            rows,
            measurement_paths,
            expected_measurement_sha256,
        ):
            self.assertEqual(_sha(path), expected_sha)
            measurement = json.loads(path.read_text(encoding="utf-8"))
            report_receipt = measurement["diffsheg_report"]
            report_path = Path(report_receipt["path"])
            self.assertEqual(_sha(report_path), report_receipt["sha256"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["epoch"], row["epoch"])
            row["metrics"] = {"fgd": report["fgd"]}
        winner = min(
            rows,
            key=lambda row: (row["metrics"]["fgd"], row["epoch"]),
        )
        selection["selected"] = {
            "epoch": winner["epoch"],
            "candidate_checkpoint": winner["candidate_checkpoint"],
            "fgd": winner["metrics"]["fgd"],
            "inference_lineage": winner["inference_lineage"],
            "diffsheg_report": winner["diffsheg_report"],
        }
        selection.pop("receipt_payload_sha256")
        selection["receipt_payload_sha256"] = (
            long_contract.canonical_json_sha256(selection)
        )
        return selection

    def test_only_recomputed_twenty_two_way_val_winner_is_authorized(
        self,
    ) -> None:
        receipt = self._authorize()
        self.assertEqual(receipt["selected_epoch"], 200)
        self.assertEqual(receipt["candidate_count"], 22)
        self.assertFalse(receipt["test_visible_during_selection"])
        self.assertEqual(receipt["status"], "validated")
        self.assertEqual(receipt["authorized_test_evaluations"], 0)

    def test_nonwinner_checkpoint_is_rejected(self) -> None:
        with self.assertRaises(validator.TestWinnerContractError):
            self._authorize(epoch=180)

    def test_declared_winner_must_equal_recomputed_minimum(self) -> None:
        selection = json.loads(
            self.selection_path.read_text(encoding="utf-8")
        )
        selection["selected"]["epoch"] = 180
        selection["receipt_payload_sha256"] = (
            long_contract.canonical_json_sha256(
                {
                    key: value
                    for key, value in selection.items()
                    if key != "receipt_payload_sha256"
                }
            )
        )
        altered = self.root / "altered-validation-selection.json"
        _write_json(altered, selection)
        with self.assertRaises(validator.TestWinnerContractError):
            self._authorize(selection_path=altered)

    def test_selection_file_with_test_label_is_rejected(self) -> None:
        contaminated = self.root / "actual-test" / "selection.json"
        contaminated.parent.mkdir()
        contaminated.write_bytes(self.selection_path.read_bytes())
        with self.assertRaises(long_contract.SelectionContractError):
            self._authorize(selection_path=contaminated)

    def test_embedded_fgd_must_match_replayed_report(self) -> None:
        selection = json.loads(
            self.selection_path.read_text(encoding="utf-8")
        )
        winner = next(
            row
            for row in selection["candidate_metrics"]
            if row["epoch"] == 200
        )
        report = Path(winner["diffsheg_report"]["path"])
        _write_json(report, {"epoch": 200, "fgd": 99.0})
        winner["diffsheg_report"]["sha256"] = _sha(report)
        measurement = next(
            Path(receipt["path"])
            for receipt in selection["measurement_receipts"]
            if json.loads(
                Path(receipt["path"]).read_text(encoding="utf-8")
            )["epoch"]
            == 200
        )
        measurement_payload = json.loads(
            measurement.read_text(encoding="utf-8")
        )
        measurement_payload["diffsheg_report"]["sha256"] = _sha(report)
        _write_json(measurement, measurement_payload)
        for receipt in selection["measurement_receipts"]:
            if receipt["path"] == str(measurement):
                receipt["sha256"] = _sha(measurement)
        selection.pop("receipt_payload_sha256")
        selection["receipt_payload_sha256"] = (
            long_contract.canonical_json_sha256(selection)
        )
        inconsistent = self.root / "inconsistent-validation-selection.json"
        _write_json(inconsistent, selection)
        with self.assertRaisesRegex(
            validator.TestWinnerContractError,
            "differs from replayed",
        ):
            self._authorize(selection_path=inconsistent)

    def test_artifact_receipts_reject_extra_keys(self) -> None:
        selection = json.loads(
            self.selection_path.read_text(encoding="utf-8")
        )
        selection["candidate_metrics"][0]["inference_lineage"][
            "unexpected"
        ] = True
        selection.pop("receipt_payload_sha256")
        selection["receipt_payload_sha256"] = (
            long_contract.canonical_json_sha256(selection)
        )
        altered = self.root / "extra-artifact-key-selection.json"
        _write_json(altered, selection)
        with self.assertRaisesRegex(
            validator.TestWinnerContractError,
            "artifact schema mismatch",
        ):
            self._authorize(selection_path=altered)

    def test_one_shot_claim_has_one_canonical_exclusive_slot(self) -> None:
        validation = self._authorize()
        claimed = validator.publish_test_winner_claim(
            validation,
            selection_path=self.selection_path,
        )
        claim_path = Path(claimed["claim"]["path"])
        self.assertEqual(
            claim_path,
            self.selection_path.with_name(
                f"{self.selection_path.name}.test-winner-claim.json"
            ),
        )
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
        claimed_hash = payload.pop("receipt_payload_sha256")
        self.assertEqual(
            long_contract.canonical_json_sha256(payload),
            claimed_hash,
        )
        self.assertEqual(
            payload["candidate_bundle"],
            {
                key: self.bundle[key]
                for key in ("manifest", "status", "frozen_inputs")
            },
        )
        self.assertEqual(payload["selected_checkpoint"]["bytes"], 13)
        with self.assertRaisesRegex(
            validator.TestWinnerContractError,
            "already claimed",
        ):
            validator.publish_test_winner_claim(
                validation,
                selection_path=self.selection_path,
            )


if __name__ == "__main__":
    unittest.main()
