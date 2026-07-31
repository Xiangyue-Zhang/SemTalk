from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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
            _write_json(report, {"epoch": epoch, "fgd": index + 1.0})
            # e200 is the unique validation winner.
            fgd = 0.125 if epoch == 200 else index + 1.0
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
        return validator.validate_test_winner(
            selection_path=selection_path or self.selection_path,
            expected_selection_sha256=_sha(
                selection_path or self.selection_path
            ),
            checkpoint_path=Path(checkpoint["path"]),
            expected_checkpoint_sha256=checkpoint["sha256"],
            candidate_bundle=self.bundle,
        )

    def test_only_recomputed_twenty_two_way_val_winner_is_authorized(
        self,
    ) -> None:
        receipt = self._authorize()
        self.assertEqual(receipt["selected_epoch"], 200)
        self.assertEqual(receipt["candidate_count"], 22)
        self.assertFalse(receipt["test_visible_during_selection"])
        self.assertEqual(receipt["authorized_test_evaluations"], 1)

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


if __name__ == "__main__":
    unittest.main()
