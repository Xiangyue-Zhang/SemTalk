from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


from scripts.show_base import build_talkshow_metric_root as BUILDER
from scripts.show_base import evaluate_talkshow_show_metrics as METRICS


ROOT = Path(__file__).resolve().parents[1]
LOCAL_UPSTREAM = Path(
    "/private/tmp/talkshow_upstream_9aef82d_20260731"
)


class TalkShowMetricRootBuilderTest(unittest.TestCase):
    def test_patch_bytes_marker_and_source_have_no_legacy_preimage(self) -> None:
        expected = b'"""Metric-only TalkSHOW package."""\n'
        self.assertEqual(METRICS.TALKSHOW_METRIC_ONLY_INIT, expected)
        self.assertEqual(len(expected), 36)
        self.assertEqual(
            hashlib.sha256(expected).hexdigest(),
            "b511f4f4ee1421ddd212861e9d5817795a69dac74c493461a60424f19e2d36d1",
        )
        self.assertEqual(
            METRICS.TALKSHOW_PATCH_MARKER,
            {
                "upstream_origin": (
                    "https://github.com/yhw-yhw/TalkSHOW.git"
                ),
                "upstream_commit": (
                    "9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
                ),
                "upstream_tree": (
                    "d993229539e63442a1f1327bae6d80d35f97c521"
                ),
                "patch_version": (
                    "semtalk-metric-only-no-eager-import-v2"
                ),
                "scope": "released-show-body-feature-extractor-only",
                "source_file": "nets/__init__.py",
                "patch_operation": "replace_with_exact_bytes_v1",
                "source_sha256": (
                    "d12f3ebd1b8f4b251085061404d72530df58bc972272995a3810236aefdd4d21"
                ),
                "patched_bytes": 36,
                "patched_sha256": (
                    "b511f4f4ee1421ddd212861e9d5817795a69dac74c493461a60424f19e2d36d1"
                ),
            },
        )
        for relative in (
            "scripts/show_base/build_talkshow_metric_root.py",
            "scripts/show_base/evaluate_talkshow_show_metrics.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("e0e277d46475c203", source)
            self.assertNotIn("globaldiff-show-metric-v1", source)

    def test_output_claim_is_create_new_and_preserves_existing(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="talkshow-builder-claim-",
            dir="/private/tmp",
        ) as raw:
            output = Path(raw) / "metric-root"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                BUILDER._claim_new_directory(output)
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_cli_supports_direct_execution(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "show_base"
                    / "build_talkshow_metric_root.py"
                ),
                "--help",
            ],
            cwd="/",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--upstream-root", completed.stdout)
        self.assertIn("--output-root", completed.stdout)

    def test_real_upstream_build_proves_exact_fifteen_file_closure(self) -> None:
        if not LOCAL_UPSTREAM.is_dir():
            self.skipTest("local pinned TalkSHOW upstream is unavailable")
        with tempfile.TemporaryDirectory(
            prefix="talkshow-builder-real-",
            dir="/private/tmp",
        ) as raw:
            output = Path(raw) / "metric-root"
            report = BUILDER.build_metric_root(LOCAL_UPSTREAM, output)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["patched_init_bytes"], 36)
            self.assertEqual(
                report["patched_init_sha256"],
                METRICS.TALKSHOW_PATCH_MARKER["patched_sha256"],
            )
            self.assertEqual(
                report["import_closure"],
                list(METRICS.TALKSHOW_FGD_SOURCE_FILES),
            )
            self.assertEqual(
                (output / "nets" / "__init__.py").read_bytes(),
                METRICS.TALKSHOW_METRIC_ONLY_INIT,
            )
            self.assertEqual(
                (output / ".paspa_talkshow_patch.json").read_bytes(),
                METRICS.canonical_json_bytes(METRICS.TALKSHOW_PATCH_MARKER),
            )
            receipt = METRICS.validate_talkshow_metric_root(output)
            self.assertEqual(
                tuple(receipt["files"]),
                METRICS.TALKSHOW_FGD_SOURCE_FILES,
            )
            dirty = subprocess.run(
                [
                    "git",
                    "-C",
                    str(output),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout.decode("utf-8").splitlines()
            self.assertEqual(set(dirty), BUILDER.EXPECTED_DIRTY)
            with self.assertRaises(FileExistsError):
                BUILDER.build_metric_root(LOCAL_UPSTREAM, output)
            self.assertEqual(
                json.loads(
                    (output / ".paspa_talkshow_patch.json").read_bytes()
                ),
                METRICS.TALKSHOW_PATCH_MARKER,
            )


if __name__ == "__main__":
    unittest.main()
