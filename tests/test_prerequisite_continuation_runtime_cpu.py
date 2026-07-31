from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import prerequisite_continuation_runtime as runtime
from scripts.show_base import prerequisite_continuation_wave as wave
from tests import test_prerequisite_continuation_wave_cpu as wave_fixture


SHA_A = "a" * 64


def build_runtime_fixture(root: Path, stage: str = "hands") -> dict:
    values = wave_fixture.fixture(triggers=("face",))
    boundary = 200
    target = 220
    for current in wave.STAGES:
        old = root / "old" / current
        new = root / "new" / f"e{target}" / current
        values["stage_plans"][current]["old_run_path"] = str(old)
        values["stage_plans"][current]["new_run_path"] = str(new)
        values["stage_plans"][current]["candidate_segment_chain"] = [
            wave._make_chain_segment(
                run_path=str(old),
                start_epoch=wave.INTERVAL_EPOCHS,
                end_epoch=boundary,
                predecessor_segment_id=None,
            )
        ]
        for candidate in values["candidate_catalog_by_stage"][current]:
            epoch = candidate["epoch"]
            candidate["checkpoint"]["path"] = str(
                old
                / "representation_candidates"
                / f"{current}_epoch_{epoch:04d}.bin"
            )
        values["selected_by_stage"][current] = copy.deepcopy(
            next(
                candidate
                for candidate in values[
                    "candidate_catalog_by_stage"
                ][current]
                if candidate["epoch"] == 120
            )
        )
        status = old / "formal_training_status.json"
        resume = old / "latest_resume.pt"
        final = old / f"{current}_final.bin"
        values["stage_plans"][current]["formal_status"]["path"] = str(
            status
        )
        values["stage_plans"][current]["boundary_resume"]["path"] = str(
            resume
        )
        values["stage_plans"][current]["final_checkpoint"]["path"] = str(
            final
        )
        if current == stage:
            old.mkdir(parents=True)
            payload = f"resume:{current}:{boundary}".encode()
            resume.write_bytes(payload)
            values["stage_plans"][current]["boundary_resume"].update(
                {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
    receipt = wave.authorize_wave(**values)
    entry = next(
        value for value in receipt["stages"] if value["stage"] == stage
    )
    wave_path = root / "wave.json"
    wave_path.write_text("{}\n")
    resume_binding = copy.deepcopy(
        entry["old_segment"]["boundary_resume"]
    )
    old_snapshot = {
        "format": runtime.OLD_SNAPSHOT_FORMAT,
        "stage": stage,
        "bindings": {
            label: copy.deepcopy(resume_binding)
            for label in runtime.OLD_SNAPSHOT_LABELS
        },
    }
    old_snapshot["receipt_payload_sha256"] = (
        wave.canonical_json_sha256(old_snapshot)
    )
    return {
        "receipt": receipt,
        "entry": entry,
        "stage": stage,
        "wave_path": wave_path,
        "expected_wave_sha256": SHA_A,
        "target_epoch": target,
        "current_new_run": Path(entry["new_segment"]["run_path"]),
        "resume_path": Path(
            entry["old_segment"]["boundary_resume"]["path"]
        ),
        "current_source": copy.deepcopy(entry["new_segment"]["source"]),
        "current_host": entry["new_segment"]["host"],
        "current_smplx_asset": copy.deepcopy(
            entry["new_segment"]["smplx_asset"]
        ),
        "current_config_sha256": entry["new_segment"][
            "config_sha256"
        ],
        "current_config_semantic_sha256": entry["new_segment"][
            "config_semantic_sha256"
        ],
        "current_dataset_semantic_sha256": entry["new_segment"][
            "dataset_semantic_sha256"
        ],
        "world_size": entry["training_topology"]["world_size"],
        "old_snapshot": old_snapshot,
    }


class ContinuationRuntimeTests(unittest.TestCase):
    def verify(self, values: dict) -> dict:
        kwargs = {
            key: value
            for key, value in values.items()
            if key
            not in {
                "receipt",
                "entry",
                "stage",
                "old_snapshot",
            }
        }
        kwargs["stage"] = values["stage"]
        with (
            mock.patch.object(
                runtime.wave,
                "replay_wave_file",
                return_value=values["receipt"],
            ),
            mock.patch.object(
                runtime,
                "_load_resume_proof",
                return_value=copy.deepcopy(
                    values["entry"]["old_segment"]["boundary_state"]
                ),
            ),
            mock.patch.object(
                runtime,
                "_verify_current_smplx_asset",
                return_value=copy.deepcopy(
                    values["current_smplx_asset"]
                ),
            ),
            mock.patch.object(
                runtime,
                "_build_old_segment_snapshot",
                return_value=copy.deepcopy(values["old_snapshot"]),
            ),
        ):
            return runtime.verify_runtime_wave_stage(**kwargs)

    def test_exact_old_boundary_resume_starts_empty_new_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            receipt = self.verify(values)
            self.assertEqual(receipt["boundary_epoch"], 200)
            self.assertEqual(receipt["target_epoch"], 220)
            self.assertEqual(
                values["entry"]["independent_selected_candidate"][
                    "epoch"
                ],
                120,
            )
            self.assertEqual(
                values["entry"]["resume_boundary_candidate"]["epoch"],
                200,
            )
            self.assertFalse(
                (
                    values["current_new_run"]
                    / "representation_candidates"
                ).exists()
            )

    def test_old_segment_pre_post_sha_is_exact_and_mutation_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            self.assertFalse(values["current_new_run"].exists())
            receipt = self.verify(values)
            proof = runtime.verify_old_segment_unchanged(receipt)
            self.assertTrue(proof["unchanged"])
            self.assertEqual(proof["before"], proof["after"])
            values["resume_path"].write_bytes(b"postflight mutation")
            with self.assertRaisesRegex(
                runtime.ContinuationRuntimeError,
                "binding changed",
            ):
                runtime.verify_old_segment_unchanged(receipt)

    def test_copied_prefix_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            candidate_dir = (
                values["current_new_run"]
                / "representation_candidates"
            )
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "hands_epoch_0120.bin").write_bytes(b"copy")
            with self.assertRaisesRegex(
                runtime.ContinuationRuntimeError,
                "already contains representation candidates",
            ):
                self.verify(values)

    def test_reused_new_run_resume_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            values["current_new_run"].mkdir(parents=True)
            (
                values["current_new_run"] / "latest_resume.pt"
            ).write_bytes(b"reuse")
            with self.assertRaisesRegex(
                runtime.ContinuationRuntimeError,
                "already contains latest_resume",
            ):
                self.verify(values)

    def test_existing_target_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            candidate_dir = (
                values["current_new_run"]
                / "representation_candidates"
            )
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "hands_epoch_0220.bin").write_bytes(b"target")
            with self.assertRaises(
                runtime.ContinuationRuntimeError,
            ):
                self.verify(values)

    def test_resume_path_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            values["resume_path"] = (
                values["resume_path"].parent / "other_resume.pt"
            )
            with self.assertRaisesRegex(
                runtime.ContinuationRuntimeError,
                "differs from continuation authorization",
            ):
                self.verify(values)

    def test_old_resume_byte_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            values["resume_path"].write_bytes(b"changed")
            with self.assertRaisesRegex(
                runtime.ContinuationRuntimeError,
                "binding changed",
            ):
                self.verify(values)

    def test_wrong_target_world_or_source_is_rejected(self) -> None:
        attacks = (
            ("target_epoch", 240),
            ("world_size", 1),
            ("current_host", "not-a-formal-host"),
            (
                "current_source",
                {
                    "commit": "9" * 40,
                    "tree": "4" * 40,
                    "source_receipt_sha256": "b" * 64,
                },
            ),
        )
        for key, replacement in attacks:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                values = build_runtime_fixture(Path(temporary).resolve())
                values[key] = replacement
                with self.assertRaisesRegex(
                    runtime.ContinuationRuntimeError,
                    "differs from continuation authorization|formal hosts",
                ):
                    self.verify(values)

    def test_boundary_state_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            kwargs = {
                key: value
                for key, value in values.items()
                if key not in {"receipt", "entry", "stage"}
            }
            kwargs.pop("old_snapshot")
            kwargs["stage"] = values["stage"]
            attacked = copy.deepcopy(
                values["entry"]["old_segment"]["boundary_state"]
            )
            attacked["model_state_sha256"] = "f" * 64
            with (
                mock.patch.object(
                    runtime.wave,
                    "replay_wave_file",
                    return_value=values["receipt"],
                ),
                mock.patch.object(
                    runtime,
                    "_load_resume_proof",
                    return_value=attacked,
                ),
                mock.patch.object(
                    runtime,
                    "_verify_current_smplx_asset",
                    return_value=copy.deepcopy(
                        values["current_smplx_asset"]
                    ),
                ),
                mock.patch.object(
                    runtime,
                    "_build_old_segment_snapshot",
                    return_value=copy.deepcopy(
                        values["old_snapshot"]
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    runtime.ContinuationRuntimeError,
                    "boundary state differs",
                ):
                    runtime.verify_runtime_wave_stage(**kwargs)

    def test_runtime_receipt_resigning_does_not_hide_accounting_attack(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = build_runtime_fixture(Path(temporary).resolve())
            receipt = self.verify(values)
            attacked = copy.deepcopy(receipt)
            attacked["target_epoch"] = 240
            unsigned = dict(attacked)
            unsigned.pop("receipt_payload_sha256")
            attacked["receipt_payload_sha256"] = (
                wave.canonical_json_sha256(unsigned)
            )
            with self.assertRaisesRegex(
                runtime.ContinuationRuntimeError,
                "accounting mismatch",
            ):
                runtime.validate_runtime_receipt(attacked)


if __name__ == "__main__":
    unittest.main()
