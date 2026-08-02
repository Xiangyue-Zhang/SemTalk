from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import adapt_base_v14_live_selection_for_final as ADAPT
from scripts.show_base import base_long_val_contract as LONG


def write_json(path: Path, value: object) -> None:
    path.write_bytes(ADAPT.canonical_json_bytes(value))


def artifact(path: Path, *, payload: bool = False) -> dict[str, object]:
    raw = path.read_bytes()
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    if payload:
        result["receipt_payload_sha256"] = __import__("json").loads(raw)[
            "receipt_payload_sha256"
        ]
    return result


class LiveSelectionHandoffCpuTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, object]:
        epochs = tuple(LONG.EXPECTED_CANDIDATE_EPOCHS)
        candidate_artifacts: dict[str, dict[str, object]] = {}
        for role in ("manifest", "status", "frozen_inputs"):
            path = root / f"{role}.json"
            write_json(path, {"role": role})
            candidate_artifacts[role] = artifact(path)
        checkpoints: dict[int, dict[str, object]] = {}
        for epoch in epochs:
            path = root / f"checkpoint-e{epoch:04d}.bin"
            path.write_bytes(f"checkpoint-{epoch}".encode())
            checkpoints[epoch] = artifact(path)
        candidate_bundle = {
            "candidate_epochs": list(epochs),
            "candidates": checkpoints,
        }
        measurements: list[dict[str, object]] = []
        for epoch in epochs:
            path = root / f"measurement-e{epoch:04d}.json"
            unsigned = {"epoch": epoch, "split": "val"}
            write_json(
                path,
                {
                    **unsigned,
                    "receipt_payload_sha256": ADAPT.canonical_json_sha256(
                        unsigned
                    ),
                },
            )
            measurements.append(artifact(path))
        rows = []
        for epoch in epochs:
            fgd = 0.1 if epoch == 80 else 1.0 + epoch / 1000.0
            rows.append(
                {
                    "epoch": epoch,
                    "candidate_checkpoint": checkpoints[epoch],
                    "metrics": {LONG.PRIMARY_SELECTION_REPORT_KEY: fgd},
                }
            )
        selected = {
            "epoch": 80,
            "candidate_checkpoint": checkpoints[80],
            "fgd": 0.1,
        }
        formal_unsigned = {
            "format": LONG.SELECTION_FORMAT,
            "status": "selected",
            "split": "val",
            "test_visible": False,
            "selection_eligible": True,
            "candidate_metrics": rows,
            "selected": selected,
            "selection_policy": {
                "metric": "FGD",
                "operator": "min",
                "tie_break": "lowest_epoch",
            },
        }
        formal = {
            **formal_unsigned,
            "receipt_payload_sha256": ADAPT.canonical_json_sha256(
                formal_unsigned
            ),
        }
        reconciliation_path = root / "reconciliation.json"
        reconciliation_unsigned = {
            "format": ADAPT.live_bridge.RECONCILIATION_FORMAT,
            "status": "complete",
        }
        reconciliation = {
            **reconciliation_unsigned,
            "receipt_payload_sha256": ADAPT.canonical_json_sha256(
                reconciliation_unsigned
            ),
        }
        write_json(reconciliation_path, reconciliation)
        reconciliation_artifact = artifact(
            reconciliation_path, payload=True
        )
        reconciliation_value = {
            "producer_manifest": candidate_artifacts["manifest"],
            "producer_status": candidate_artifacts["status"],
            "frozen_inputs": {
                **candidate_artifacts["frozen_inputs"],
                "receipt_payload_sha256": "f" * 64,
            },
        }
        live = []
        for epoch in epochs:
            live.append(
                {
                    "path": str(root / f"live-e{epoch:04d}.json"),
                    "sha256": f"{epoch:064x}",
                    "bytes": epoch + 1,
                    "receipt_payload_sha256": f"{epoch + 1000:064x}",
                }
            )
        outer_unsigned = {
            "format": ADAPT.OUTER_SELECTION_FORMAT,
            "status": "selected",
            "split": "val",
            "test_visible": False,
            "selection_eligible": True,
            "candidate_epochs": list(epochs),
            "reconciliation_receipt": reconciliation_artifact,
            "live_measurements": live,
            "reconciled_measurements": measurements,
            "formal_selection": formal,
            "selected": selected,
            "test_evaluations_observed": 0,
        }
        outer = {
            **outer_unsigned,
            "receipt_payload_sha256": ADAPT.canonical_json_sha256(
                outer_unsigned
            ),
        }
        outer_path = root / "outer-selection.json"
        write_json(outer_path, outer)
        return {
            "candidate_artifacts": candidate_artifacts,
            "candidate_bundle": candidate_bundle,
            "measurements": measurements,
            "formal": formal,
            "outer": outer,
            "outer_artifact": artifact(outer_path, payload=True),
            "reconciliation_artifact": reconciliation_artifact,
            "reconciliation_value": reconciliation_value,
        }

    def patches(self, fixture: dict[str, object]):
        def validate_measurement(
            *, measurement_path: Path, expected_measurement_sha256: str,
            candidate_bundle: dict[str, object],
        ):
            epoch = int(measurement_path.stem.rsplit("e", 1)[1])
            self.assertEqual(
                hashlib.sha256(measurement_path.read_bytes()).hexdigest(),
                expected_measurement_sha256,
            )
            return (
                {
                    "path": str(measurement_path),
                    "sha256": expected_measurement_sha256,
                },
                {"epoch": epoch},
            )

        return (
            mock.patch.object(
                ADAPT.long_contract,
                "validate_candidate_bundle",
                return_value=fixture["candidate_bundle"],
            ),
            mock.patch.object(
                ADAPT.live_bridge,
                "_validate_reconciliation",
                return_value=(
                    fixture["reconciliation_artifact"],
                    fixture["reconciliation_value"],
                ),
            ),
            mock.patch.object(
                ADAPT.long_selector,
                "validate_measurement",
                side_effect=validate_measurement,
            ),
            mock.patch.object(
                ADAPT.long_selector,
                "build_selection",
                return_value=fixture["formal"],
            ),
        )

    def test_publishes_pure_official_selection_and_replayable_handoff(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as raw:
            root = Path(raw).resolve()
            fixture = self.fixture(root)
            selection_path = root / "formal-selection.json"
            handoff_path = root / "handoff.json"
            first, second, third, fourth = self.patches(fixture)
            with first, second, third, fourth:
                result = ADAPT.publish_handoff(
                    outer_artifact=fixture["outer_artifact"],
                    candidate_artifacts=fixture["candidate_artifacts"],
                    formal_selection_output=selection_path,
                    handoff_receipt_output=handoff_path,
                )
                replayed = ADAPT.validate_handoff(
                    handoff_artifact=result["handoff_receipt"],
                    winner_selection=result["formal_selection"],
                    candidate_artifacts=fixture["candidate_artifacts"],
                )
            self.assertEqual(
                __import__("json").loads(selection_path.read_bytes()),
                fixture["formal"],
            )
            self.assertEqual(replayed["formal_selection"], fixture["formal"])
            handoff = __import__("json").loads(handoff_path.read_bytes())
            self.assertEqual(handoff["format"], ADAPT.HANDOFF_FORMAT)
            self.assertEqual(len(handoff["reconciled_measurements"]), 22)
            self.assertEqual(handoff["selected"]["epoch"], 80)
            self.assertEqual(
                handoff["formal_selection_output"],
                result["formal_selection"],
            )
            self.assertNotIn("outer_selection", fixture["formal"])
            manual = root / "manually-extracted-selection.json"
            manual.write_bytes(selection_path.read_bytes())
            with self.assertRaisesRegex(
                ADAPT.LiveSelectionHandoffError,
                "not the handoff formal output",
            ):
                ADAPT.validate_handoff(
                    handoff_artifact=result["handoff_receipt"],
                    winner_selection=artifact(manual, payload=True),
                    candidate_artifacts=fixture["candidate_artifacts"],
                )

    def test_rejects_nested_selection_divergence_and_manual_output_copy(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as raw:
            root = Path(raw).resolve()
            fixture = self.fixture(root)
            attacked = copy.deepcopy(fixture["formal"])
            attacked["selected"]["epoch"] = 100
            fixture["outer"]["formal_selection"] = attacked
            outer_unsigned = dict(fixture["outer"])
            outer_unsigned.pop("receipt_payload_sha256")
            fixture["outer"] = {
                **outer_unsigned,
                "receipt_payload_sha256": ADAPT.canonical_json_sha256(
                    outer_unsigned
                ),
            }
            outer_path = Path(fixture["outer_artifact"]["path"])
            write_json(outer_path, fixture["outer"])
            fixture["outer_artifact"] = artifact(outer_path, payload=True)
            first, second, third, fourth = self.patches(fixture)
            with first, second, third, fourth, self.assertRaisesRegex(
                ADAPT.LiveSelectionHandoffError, "nested official selection"
            ):
                ADAPT.publish_handoff(
                    outer_artifact=fixture["outer_artifact"],
                    candidate_artifacts=fixture["candidate_artifacts"],
                    formal_selection_output=root / "selection.json",
                    handoff_receipt_output=root / "handoff.json",
                )
            self.assertFalse((root / "selection.json").exists())

    def test_create_new_outputs_refuse_reuse(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as raw:
            root = Path(raw).resolve()
            fixture = self.fixture(root)
            selection_path = root / "selection.json"
            handoff_path = root / "handoff.json"
            first, second, third, fourth = self.patches(fixture)
            with first, second, third, fourth:
                ADAPT.publish_handoff(
                    outer_artifact=fixture["outer_artifact"],
                    candidate_artifacts=fixture["candidate_artifacts"],
                    formal_selection_output=selection_path,
                    handoff_receipt_output=handoff_path,
                )
            with self.assertRaisesRegex(
                ADAPT.LiveSelectionHandoffError, "canonical and absent"
            ):
                ADAPT.publish_handoff(
                    outer_artifact=fixture["outer_artifact"],
                    candidate_artifacts=fixture["candidate_artifacts"],
                    formal_selection_output=selection_path,
                    handoff_receipt_output=root / "second-handoff.json",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
