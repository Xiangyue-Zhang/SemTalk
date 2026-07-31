from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.show_base import check_prerequisite_val_multicandidate_gate as gate
from scripts.show_base import prerequisite_val_contract as contract


class MultiCandidateGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.common = {
            "candidate_index": str((self.root / "candidate.json").resolve()),
            "candidate_index_sha256": "1" * 64,
            "canonical_manifest_sha256": "2" * 64,
            "canonical_summary_sha256": "3" * 64,
            "canonical_lineage_sha256": "4" * 64,
            "source_commit": "5" * 40,
            "source_tree": "6" * 40,
            "multi_candidate_gate": None,
        }
        (self.root / "candidate.json").write_text("candidate\n")
        self.jobs = [
            {
                "stage": "face",
                "epoch": epoch,
                "optimizer_updates": epoch * 497,
                "checkpoint": str((self.root / f"e{epoch}.pt").resolve()),
                "checkpoint_sha256": hashlib.sha256(str(epoch).encode()).hexdigest(),
            }
            for epoch in (20, 40, 60, 80)
        ]
        self.serial = self._make_root("serial", 1, 40)
        self.concurrent = self._make_root("concurrent", 4, 12)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make_root(self, name: str, concurrency: int, elapsed: int) -> Path:
        root = self.root / name
        (root / "shards").mkdir(parents=True)
        for job in self.jobs:
            epoch_dir = root / "shards" / "face" / f"epoch_{job['epoch']:04d}"
            epoch_dir.mkdir(parents=True)
            for shard in range(8):
                payload = contract.receipt_payload(
                    {
                        "format": contract.SHARD_FORMAT,
                        "status": "complete",
                        "stage": "face",
                        "epoch": job["epoch"],
                        "shard": shard,
                        "numeric": {"sum": job["epoch"] + shard, "count": 17},
                    }
                )
                (epoch_dir / f"shard_{shard:02d}.json").write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
                )
        receipt = contract.receipt_payload(
            {
                "format": gate.PARTITION_FORMAT,
                "status": "complete",
                "partition": "gate",
                "test_visible": False,
                "protocol": {
                    "candidates_per_wave": concurrency,
                    "shards_per_candidate": 8,
                    "batch_size": 32,
                    "candidate_jobs": 4,
                    "waves": 4 // concurrency,
                },
                "timing": {
                    "started_unix": 100,
                    "ended_unix": 100 + elapsed,
                    "elapsed_seconds": elapsed,
                },
                "inputs": self.common,
                "jobs": self.jobs,
                "coverage": {
                    "candidate_jobs": 4,
                    "shard_jobs": 32,
                    "exact_once": True,
                },
            }
        )
        (root / "partition_receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        return root

    def test_exact_gate_and_fresh_replay(self) -> None:
        output = self.root / "gate.json"
        result = gate.check(
            serial_root=self.serial,
            concurrent_root=self.concurrent,
            output_json=output,
        )
        self.assertTrue(result["equivalence"]["byte_exact"])
        self.assertLess(result["throughput"]["concurrent_over_serial_ratio"], 1.0)
        replayed = gate.replay_gate(
            output,
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )
        self.assertEqual(replayed, result)

    def test_changed_missing_extra_and_slow_shards_fail(self) -> None:
        changed = self.concurrent / "shards/face/epoch_0020/shard_00.json"
        changed.write_text(changed.read_text() + " ")
        with self.assertRaisesRegex(contract.ContractError, "differs"):
            gate.check(
                serial_root=self.serial,
                concurrent_root=self.concurrent,
                output_json=self.root / "changed.json",
            )
        changed.write_bytes(
            (self.serial / "shards/face/epoch_0020/shard_00.json").read_bytes()
        )
        missing = self.concurrent / "shards/face/epoch_0040/shard_01.json"
        missing.unlink()
        with self.assertRaisesRegex(contract.ContractError, "exactly 32"):
            gate.check(
                serial_root=self.serial,
                concurrent_root=self.concurrent,
                output_json=self.root / "missing.json",
            )
        missing.write_bytes(
            (self.serial / "shards/face/epoch_0040/shard_01.json").read_bytes()
        )
        extra = self.concurrent / "shards/extra.json"
        extra.write_text("{}\n")
        with self.assertRaisesRegex(contract.ContractError, "exactly 32"):
            gate.check(
                serial_root=self.serial,
                concurrent_root=self.concurrent,
                output_json=self.root / "extra.json",
            )
        extra.unlink()
        receipt_path = self.concurrent / "partition_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["timing"] = {
            "started_unix": 100,
            "ended_unix": 200,
            "elapsed_seconds": 100,
        }
        receipt.pop("receipt_payload_sha256")
        receipt = contract.receipt_payload(receipt)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with self.assertRaisesRegex(contract.ContractError, "throughput"):
            gate.check(
                serial_root=self.serial,
                concurrent_root=self.concurrent,
                output_json=self.root / "slow.json",
            )

    def test_root_receipt_and_shard_symlinks_fail_closed(self) -> None:
        root_link = self.root / "serial-link"
        root_link.symlink_to(self.serial, target_is_directory=True)
        with self.assertRaisesRegex(contract.ContractError, "root"):
            gate.check(
                serial_root=root_link,
                concurrent_root=self.concurrent,
                output_json=self.root / "root-link.json",
            )
        receipt = self.serial / "partition_receipt.json"
        receipt_real = self.serial / "partition_receipt.real.json"
        receipt.rename(receipt_real)
        receipt.symlink_to(receipt_real)
        with self.assertRaisesRegex(contract.ContractError, "receipt"):
            gate.check(
                serial_root=self.serial,
                concurrent_root=self.concurrent,
                output_json=self.root / "receipt-link.json",
            )
        receipt.unlink()
        receipt_real.rename(receipt)
        shard = self.serial / "shards/face/epoch_0020/shard_00.json"
        shard_real = shard.with_suffix(".real.json")
        shard.rename(shard_real)
        shard.symlink_to(shard_real)
        with self.assertRaisesRegex(contract.ContractError, "symlink"):
            gate.check(
                serial_root=self.serial,
                concurrent_root=self.concurrent,
                output_json=self.root / "shard-link.json",
            )


if __name__ == "__main__":
    unittest.main()
