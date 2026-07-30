from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.show_base import train_official_transfer as transfer


class OfficialTransferCacheAuditTests(unittest.TestCase):
    def _cache_root(
        self,
        parent: Path,
        *,
        shard_index: int,
        key: str,
    ) -> Path:
        root = parent / f"shard_{shard_index}"
        root.mkdir()
        row = {
            "split": "train",
            "key": key,
            "payload_sha256": "b" * 64,
        }
        records = (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        records_path = root / "records.jsonl"
        records_path.write_bytes(records)
        (root / "data.lmdb").write_bytes(b"test-only-placeholder")
        summary = {
            "format": transfer.CACHE_FORMAT,
            "status": "complete",
            "stage": "face",
            "test_visible": False,
            "splits": list(transfer.TRAINING_SPLITS),
            "coverage": {
                "record_count": 1,
                "records_jsonl_sha256": hashlib.sha256(records).hexdigest(),
            },
            "official_initialization": {
                "withdrawn_e30_allowed": False,
                "face": {"sha256": "a" * 64},
            },
            "canonical_receipt": {"receipt_sha256": "c" * 64},
            "shard": {"index": shard_index, "count": 2},
        }
        summary["receipt_sha256"] = transfer.canonical_payload_sha256(summary)
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return root

    def test_cache_keys_are_exact_once_across_distinct_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            roots = [
                self._cache_root(parent, shard_index=0, key="shared-key"),
                self._cache_root(parent, shard_index=1, key="shared-key"),
            ]
            with self.assertRaisesRegex(
                RuntimeError,
                "duplicate cache record 'shared-key'",
            ):
                transfer._read_cache_roots(
                    roots,
                    stage="face",
                    official_receipts={
                        "face": {"sha256": "a" * 64},
                    },
                )

    def test_distinct_cache_keys_cover_every_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            roots = [
                self._cache_root(parent, shard_index=0, key="key-0"),
                self._cache_root(parent, shard_index=1, key="key-1"),
            ]
            records, receipt = transfer._read_cache_roots(
                roots,
                stage="face",
                official_receipts={"face": {"sha256": "a" * 64}},
            )
            self.assertEqual([row["key"] for row in records], ["key-0", "key-1"])
            self.assertEqual(receipt["record_count"], 2)
            self.assertEqual(receipt["shards"], [0, 1])


class OfficialTransferDistributedIndexTests(unittest.TestCase):
    def test_validation_indices_are_exact_once_with_unequal_rank_lengths(self) -> None:
        per_rank = [
            transfer._rank_indices(
                1_715,
                rank=rank,
                world_size=8,
                seed=43,
                epoch=1,
                shuffle=False,
                equal=False,
            )
            for rank in range(8)
        ]
        self.assertEqual([len(values) for values in per_rank], [215] * 3 + [214] * 5)
        flattened = [value for values in per_rank for value in values]
        self.assertEqual(sorted(flattened), list(range(1_715)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_training_indices_are_equal_length_and_deterministically_truncated(
        self,
    ) -> None:
        per_rank = [
            transfer._rank_indices(
                127_286,
                rank=rank,
                world_size=8,
                seed=43,
                epoch=1,
                shuffle=False,
                equal=True,
            )
            for rank in range(8)
        ]
        self.assertEqual([len(values) for values in per_rank], [15_910] * 8)
        flattened = [value for values in per_rank for value in values]
        self.assertEqual(sorted(flattened), list(range(127_280)))
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()
