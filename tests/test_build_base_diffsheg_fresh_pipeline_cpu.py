from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.show_base import build_base_diffsheg_fresh_pipeline as builder


class BuildBaseDiffSHEGFreshPipelineTests(unittest.TestCase):
    def test_absolute_cli_works_from_a_foreign_cwd(self) -> None:
        script = Path(builder.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=directory,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("DiffSHEG-primary", process.stdout)

    def test_builder_uses_only_the_primary_authority_and_validator(self) -> None:
        payload = {
            "receipt_payload_sha256": "1" * 64,
            "source": {"commit": "2" * 40},
            "prerequisite_selection": {"sha256": "3" * 64},
            "fixed_checkpoints": {
                stage: {"sha256": character * 64}
                for stage, character in zip(
                    ("face", "hands", "upper", "lower", "global"),
                    "45678",
                )
            },
        }
        args = argparse.Namespace(
            output=Path("/out/pipeline.json"),
            source_root=Path("/source"),
            prerequisite_selection=Path("/selection.json"),
            expected_prerequisite_selection_sha256="9" * 64,
        )

        def fake_atomic(path, value, *, label, validator):
            self.assertEqual(path, args.output)
            self.assertIs(value, payload)
            self.assertEqual(
                label,
                "DiffSHEG-primary fresh validation pipeline receipt",
            )
            self.assertIs(validator, builder.primary.validate_fresh_pipeline)
            return path, "a" * 64

        with (
            mock.patch.object(
                builder.primary,
                "build_fresh_pipeline_payload",
                return_value=payload,
            ) as build,
            mock.patch.object(
                builder.common,
                "_atomic_new_json",
                side_effect=fake_atomic,
            ),
        ):
            report = builder.build_pipeline(args)
        build.assert_called_once_with(
            source_root=args.source_root,
            prerequisite_selection=args.prerequisite_selection,
            expected_prerequisite_selection_sha256=(
                args.expected_prerequisite_selection_sha256
            ),
        )
        self.assertEqual(report["kind"], "diffsheg-fresh-pipeline")
        self.assertEqual(report["split"], "val")
        self.assertFalse(report["test_visible"])
        self.assertEqual(report["sha256"], "a" * 64)
        self.assertEqual(
            set(report["fixed_checkpoint_sha256"]),
            {"face", "hands", "upper", "lower", "global"},
        )


if __name__ == "__main__":
    unittest.main()
