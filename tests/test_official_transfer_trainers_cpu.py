from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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


class OfficialTransferFaceAutogradTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is unavailable in the lightweight CPU test environment",
    )
    def test_verified_cached_zq_is_normal_tensor_for_decoder_backward(self) -> None:
        import torch

        cached_zq = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        with torch.inference_mode():
            inference_zq = cached_zq.clone()
        self.assertTrue(inference_zq.is_inference())
        self.assertFalse(cached_zq.is_inference())

        class Quantizer:
            @staticmethod
            def get_codebook_entry(code_ids):
                self.assertEqual(tuple(code_ids.shape), (2, 4))
                return inference_zq

        class Model:
            quantizer = Quantizer()

        class RecordingConv(torch.nn.Conv1d):
            observed_inference_input = None

            def forward(self, tensor):
                self.observed_inference_input = tensor.is_inference()
                return super().forward(tensor)

        decoder = RecordingConv(3, 3, kernel_size=1)
        optimizer = torch.optim.SGD(decoder.parameters(), lr=1e-3)
        loader = [
            {
                "code_ids": torch.zeros((2, 4), dtype=torch.long),
                "zq": cached_zq,
                "target": torch.zeros((2, 3, 4), dtype=torch.float32),
            }
        ]

        def losses(torch_module, prediction, target, **_weights):
            total = (prediction - target).square().mean()
            return {
                "total": total,
                "jaw_geodesic": total,
                "expression_l1": total,
                "velocity": total,
                "acceleration": total,
            }

        with mock.patch.object(transfer, "face_task_losses", losses):
            metrics, samples = transfer._run_face_epoch(
                torch,
                model=Model(),
                decoder=decoder,
                loader=loader,
                optimizer=optimizer,
                device=torch.device("cpu"),
                weights={
                    "jaw": 1.0,
                    "expression": 1.0,
                    "velocity": 1.0,
                    "acceleration": 1.0,
                },
            )

        self.assertEqual(samples, 2)
        self.assertFalse(decoder.observed_inference_input)
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(value))
                for value in metrics.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
