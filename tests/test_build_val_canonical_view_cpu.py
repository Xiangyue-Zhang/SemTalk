from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "show_base" / "build_val_canonical_view.py"
SPEC = importlib.util.spec_from_file_location("val_view_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEW)


def write_json(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildValCanonicalViewTest(unittest.TestCase):
    def fixture(self, root: Path):
        manifest = root / "full.jsonl"
        rows = []
        global_index = 0
        for split, count in VIEW.EXPECTED_SPLIT_COUNTS.items():
            for ordinal in range(count):
                rows.append(
                    {
                        "global_index": global_index,
                        "clip_id": (
                            f"oliver/{split}-video-{ordinal}/"
                            f"sequence-{ordinal}"
                        ),
                        "split": split,
                        "lineage_contract_sha256": "a" * 64,
                    }
                )
                global_index += 1
        manifest.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        lineage = root / "lineage.json"
        lineage_sha = write_json(
            lineage,
            {
                "lineage_contract_sha256": "a" * 64,
                "lineage_contract": {
                    "source_receipt": {
                        "origin": VIEW.EXPECTED_ORIGIN,
                        "commit": "b" * 40,
                        "tree": "c" * 40,
                    }
                },
            },
        )
        summary = root / "summary.json"
        summary_sha = write_json(
            summary,
            {
                "status": "complete",
                "manifest_sha256": manifest_sha,
                "lineage_sha256": lineage_sha,
                "split_counts": VIEW.EXPECTED_SPLIT_COUNTS,
            },
        )
        args = VIEW.parser().parse_args(
            [
                "--full-manifest",
                str(manifest),
                "--full-summary",
                str(summary),
                "--full-lineage",
                str(lineage),
                "--expected-full-manifest-sha256",
                manifest_sha,
                "--expected-full-summary-sha256",
                summary_sha,
                "--expected-full-lineage-sha256",
                lineage_sha,
                "--expected-source-commit",
                "b" * 40,
                "--expected-source-tree",
                "c" * 40,
                "--output-root",
                str(root / "val-view"),
            ]
        )
        return args

    def test_builds_exact_validation_only_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = VIEW.build(self.fixture(root))
            output = root / "val-view"
            self.assertEqual(
                set(path.name for path in output.iterdir()),
                {"manifest.jsonl", "summary.json", "lineage.json"},
            )
            rows = [
                json.loads(line)
                for line in (output / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), VIEW.EXPECTED_VAL_CLIPS)
            self.assertTrue(all(row["split"] == "val" for row in rows))
            self.assertEqual(
                [row["global_index"] for row in rows],
                list(
                    range(
                        VIEW.EXPECTED_VAL_GLOBAL_INDEX_START,
                        VIEW.EXPECTED_VAL_GLOBAL_INDEX_STOP,
                    )
                ),
            )
            summary = json.loads((output / "summary.json").read_text())
            lineage = json.loads((output / "lineage.json").read_text())
            self.assertFalse(summary["test_visible"])
            self.assertFalse(lineage["test_visible"])
            self.assertFalse(
                lineage["projection"]["test_rows_materialized"]
            )
            self.assertEqual(
                summary["global_index_start"],
                VIEW.EXPECTED_VAL_GLOBAL_INDEX_START,
            )
            self.assertEqual(
                summary["global_index_stop_exclusive"],
                VIEW.EXPECTED_VAL_GLOBAL_INDEX_STOP,
            )
            self.assertEqual(
                summary["global_indices_sha256"],
                lineage["global_indices_sha256"],
            )
            self.assertEqual(
                receipt["manifest_sha256"],
                hashlib.sha256(
                    (output / "manifest.jsonl").read_bytes()
                ).hexdigest(),
            )

    def test_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            (root / "val-view").mkdir()
            with self.assertRaises(FileExistsError):
                VIEW.build(args)

    def test_refuses_wrong_frozen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            args.expected_full_manifest_sha256 = "d" * 64
            with self.assertRaisesRegex(VIEW.ValViewError, "mismatch"):
                VIEW.build(args)

    def test_refuses_non_val_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            rows = [
                json.loads(line)
                for line in args.full_manifest.read_text().splitlines()
            ]
            removed = False
            kept = []
            for row in rows:
                if row["split"] == "val" and not removed:
                    removed = True
                    continue
                kept.append(row)
            args.full_manifest.write_text(
                "".join(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                    for row in kept
                ),
                encoding="utf-8",
            )
            args.expected_full_manifest_sha256 = hashlib.sha256(
                args.full_manifest.read_bytes()
            ).hexdigest()
            summary = json.loads(args.full_summary.read_text())
            summary["manifest_sha256"] = args.expected_full_manifest_sha256
            args.expected_full_summary_sha256 = write_json(
                args.full_summary,
                summary,
            )
            with self.assertRaisesRegex(VIEW.ValViewError, "val row count"):
                VIEW.build(args)

    def test_refuses_rebased_validation_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            rows = [
                json.loads(line)
                for line in args.full_manifest.read_text().splitlines()
            ]
            rebased = 0
            for row in rows:
                if row["split"] == "val":
                    row["global_index"] = rebased
                    rebased += 1
            args.full_manifest.write_text(
                "".join(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            args.expected_full_manifest_sha256 = hashlib.sha256(
                args.full_manifest.read_bytes()
            ).hexdigest()
            summary = json.loads(args.full_summary.read_text())
            summary["manifest_sha256"] = args.expected_full_manifest_sha256
            args.expected_full_summary_sha256 = write_json(
                args.full_summary,
                summary,
            )
            with self.assertRaisesRegex(
                VIEW.ValViewError,
                "official global-index range",
            ):
                VIEW.build(args)


if __name__ == "__main__":
    unittest.main()
