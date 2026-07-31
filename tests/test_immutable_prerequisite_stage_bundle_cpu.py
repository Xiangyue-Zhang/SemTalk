from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import tempfile
import unittest

from scripts.show_base import immutable_prerequisite_stage_bundle as bundle


class ImmutablePrerequisiteStageBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source-face"
        self.source.mkdir()
        (self.source / "nested").mkdir()
        (self.source / "formal_training_status.json").write_text(
            '{"status":"complete"}\n', encoding="utf-8"
        )
        (self.source / "nested" / "checkpoint.bin").write_bytes(b"weights")
        self.destination = self.root / "canonical" / "face"
        self.destination.parent.mkdir()
        self.bundle_root = self.root / "bundle"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _pack(self) -> dict[str, object]:
        return bundle.pack(
            argparse.Namespace(
                source_run=[f"face={self.source}"],
                destination_run=[f"face={self.destination}"],
                bundle_root=self.bundle_root,
                source_host=socket.gethostname(),
                destination_host=socket.gethostname(),
            )
        )

    def test_pack_and_stage_are_byte_exact_and_new_only(self) -> None:
        artifact = self._pack()
        result = bundle.stage(
            argparse.Namespace(
                bundle_manifest=Path(artifact["path"]),
                expected_bundle_sha256=artifact["sha256"],
                expected_bundle_payload_sha256=artifact[
                    "receipt_payload_sha256"
                ],
                destination_host=socket.gethostname(),
                stage="face",
                destination_run=self.destination,
            )
        )
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(
            (self.destination / "nested" / "checkpoint.bin").read_bytes(),
            b"weights",
        )
        self.assertFalse((self.destination / bundle.INCOMPLETE).exists())
        with self.assertRaisesRegex(bundle.BundleError, "refusing to reuse"):
            bundle.stage(
                argparse.Namespace(
                    bundle_manifest=Path(artifact["path"]),
                    expected_bundle_sha256=artifact["sha256"],
                    expected_bundle_payload_sha256=artifact[
                        "receipt_payload_sha256"
                    ],
                    destination_host=socket.gethostname(),
                    stage="face",
                    destination_run=self.destination,
                )
            )

    def test_tampered_bundle_and_wrong_destination_fail(self) -> None:
        artifact = self._pack()
        checkpoint = self.bundle_root / "files" / "face" / "nested" / "checkpoint.bin"
        checkpoint.write_bytes(b"tampered")
        with self.assertRaisesRegex(bundle.BundleError, "bundle bytes changed"):
            bundle.stage(
                argparse.Namespace(
                    bundle_manifest=Path(artifact["path"]),
                    expected_bundle_sha256=artifact["sha256"],
                    expected_bundle_payload_sha256=artifact[
                        "receipt_payload_sha256"
                    ],
                    destination_host=socket.gethostname(),
                    stage="face",
                    destination_run=self.destination,
                )
            )
        manifest = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["stages"]["face"]["destination_run"], str(self.destination))

    def test_symlink_in_source_is_rejected(self) -> None:
        (self.source / "link").symlink_to(self.source / "nested" / "checkpoint.bin")
        with self.assertRaisesRegex(bundle.BundleError, "symlink"):
            self._pack()
        self.assertTrue((self.bundle_root / bundle.INCOMPLETE).exists())

    def test_declared_hostname_must_match_the_executing_node(self) -> None:
        with self.assertRaisesRegex(bundle.BundleError, "source hostname"):
            bundle.pack(
                argparse.Namespace(
                    source_run=[f"face={self.source}"],
                    destination_run=[f"face={self.destination}"],
                    bundle_root=self.bundle_root,
                    source_host="not-this-host",
                    destination_host=socket.gethostname(),
                )
            )
        artifact = self._pack()
        with self.assertRaisesRegex(bundle.BundleError, "destination hostname"):
            bundle.stage(
                argparse.Namespace(
                    bundle_manifest=Path(artifact["path"]),
                    expected_bundle_sha256=artifact["sha256"],
                    expected_bundle_payload_sha256=artifact[
                        "receipt_payload_sha256"
                    ],
                    destination_host="not-this-host",
                    stage="face",
                    destination_run=self.destination,
                )
            )


if __name__ == "__main__":
    unittest.main()
