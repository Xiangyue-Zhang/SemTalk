from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import publish_prerequisite_continuation_wave as publish


SHA_A = "a" * 64


def stage_plan(stage: str) -> dict:
    return {
        "stage": stage,
        "marker": f"plan:{stage}",
    }


def write_plans(path: Path, *, stages: list[str] | None = None) -> str:
    ordered = list(wave.STAGES) if stages is None else stages
    value = {
        "format": publish.FORMAT,
        "status": "complete",
        "test_visible": False,
        "stages": {stage: stage_plan(stage) for stage in ordered},
    }
    value["receipt_payload_sha256"] = wave.canonical_json_sha256(value)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode() + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def wave_receipt() -> dict:
    value = {
        "format": wave.FORMAT,
        "status": "complete",
        "boundary_epoch": 200,
        "target_epoch": 220,
    }
    value["receipt_payload_sha256"] = wave.canonical_json_sha256(value)
    return value


class PublishContinuationWaveTests(unittest.TestCase):
    def test_load_accepts_only_nonempty_known_active_stage_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "plans.json"
            digest = write_plans(
                path,
                stages=["face", "hands", "upper", "lower"],
            )
            plans, _ = publish._load_stage_plans(path, digest)
            self.assertEqual(
                list(plans),
                ["face", "hands", "upper", "lower"],
            )

            bad_path = root / "bad-plans.json"
            bad_digest = write_plans(bad_path, stages=["face", "unknown"])
            with self.assertRaisesRegex(
                publish.PublishContinuationWaveError,
                "inventory is invalid",
            ):
                publish._load_stage_plans(bad_path, bad_digest)

    def test_load_rejects_file_and_payload_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "plans.json"
            digest = write_plans(path)
            with self.assertRaisesRegex(
                publish.PublishContinuationWaveError,
                "SHA-256 mismatch",
            ):
                publish._load_stage_plans(path, SHA_A)

            value = json.loads(path.read_text())
            value["stages"]["face"]["marker"] = "changed"
            path.write_text(json.dumps(value, sort_keys=True) + "\n")
            changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                publish.PublishContinuationWaveError,
                "payload SHA-256 mismatch",
            ):
                publish._load_stage_plans(path, changed_digest)

    def test_publish_is_new_only_and_freshly_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plans_path = root / "plans.json"
            plans_sha = write_plans(plans_path)
            decision_path = root / "decision.json"
            decision_path.write_text("{}\n")
            output = root / "wave.json"
            receipt = wave_receipt()

            with (
                mock.patch.object(
                    wave,
                    "authorize_wave_from_replayed_inputs",
                    return_value=copy.deepcopy(receipt),
                ) as authorize,
                mock.patch.object(
                    wave,
                    "replay_wave_file",
                    return_value=copy.deepcopy(receipt),
                ) as replay,
            ):
                observed, file_sha = publish.publish_wave(
                    decision_path=decision_path,
                    expected_decision_sha256=SHA_A,
                    stage_plans_path=plans_path,
                    expected_stage_plans_sha256=plans_sha,
                    output_json=output,
                )
                self.assertEqual(observed, receipt)
                self.assertEqual(
                    file_sha,
                    hashlib.sha256(output.read_bytes()).hexdigest(),
                )
                authorize.assert_called_once()
                replay.assert_called_once_with(output, file_sha)

                with self.assertRaisesRegex(
                    publish.PublishContinuationWaveError,
                    "overwrite",
                ):
                    publish.publish_wave(
                        decision_path=decision_path,
                        expected_decision_sha256=SHA_A,
                        stage_plans_path=plans_path,
                        expected_stage_plans_sha256=plans_sha,
                        output_json=output,
                    )


if __name__ == "__main__":
    unittest.main()
