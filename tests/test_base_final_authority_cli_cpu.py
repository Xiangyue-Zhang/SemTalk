from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_final_authority as authority


def _write_inputs(path: Path, *, test_evaluations: int = 1) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "format": authority.INPUTS_FORMAT,
        "status": "ready",
        "selection_protocol": authority.BASE_SELECTION_PROTOCOL,
        "test_policy": {
            "test_evaluations": test_evaluations,
            "test_feedback_into_selection": False,
        },
        "expected_output_root": str(path.parent / "formal-output"),
        "canonical_manifest": {},
        "canonical_summary": {},
        "canonical_lineage": {},
        "canonical_root_receipt": {},
        "audio_authorities": [],
        "base_long_candidate_artifacts": {},
        "winner_selection": {},
        "continuation_decision": {},
        "continuation_waves": [],
        "winner_validation_metric_closure": {},
        "test_claim": {},
        "inference_source": {},
        "checkpoints": {},
    }
    value = {
        **unsigned,
        "receipt_payload_sha256": authority.canonical_json_sha256(unsigned),
    }
    path.write_bytes(authority.canonical_json_bytes(value))
    return value


class BaseFinalAuthorityCliTests(unittest.TestCase):
    def test_inputs_require_external_hash_and_exact_one_test_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "inputs.json"
            expected = _write_inputs(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                authority._load_authority_inputs(
                    path,
                    expected_file_sha256=digest,
                ),
                expected,
            )
            with self.assertRaises(authority.BaseFinalAuthorityError):
                authority._load_authority_inputs(
                    path,
                    expected_file_sha256="0" * 64,
                )
            _write_inputs(path, test_evaluations=2)
            with self.assertRaisesRegex(
                authority.BaseFinalAuthorityError,
                "identity/policy",
            ):
                authority._load_authority_inputs(
                    path,
                    expected_file_sha256=hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                )

    def test_cli_creates_and_revalidates_one_new_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs_path = root / "inputs.json"
            _write_inputs(inputs_path)
            output = root / "authority.json"
            digest = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
            built = {
                "format": authority.FORMAT,
                "status": "authorized_pre_inference",
                "receipt_payload_sha256": "a" * 64,
            }
            with (
                mock.patch.object(
                    authority,
                    "build_test_authority",
                    return_value=built,
                ) as build,
                mock.patch.object(
                    authority,
                    "validate_test_authority",
                    return_value=built,
                ) as validate,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(
                    authority.main(
                        [
                            "--inputs-json",
                            str(inputs_path),
                            "--expected-inputs-sha256",
                            digest,
                            "--output-json",
                            str(output),
                        ]
                    ),
                    0,
                )
            self.assertEqual(json.loads(output.read_text()), built)
            build.assert_called_once()
            validate.assert_called_once()
            with self.assertRaisesRegex(
                authority.BaseFinalAuthorityError,
                "canonical and absent",
            ):
                authority.main(
                    [
                        "--inputs-json",
                        str(inputs_path),
                        "--expected-inputs-sha256",
                        digest,
                        "--output-json",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
