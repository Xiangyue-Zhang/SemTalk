from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.show_base import base_fresh_probe_producer as PRODUCER
from scripts.show_base import base_fresh_probe_workload as WORKLOAD
from scripts.show_base import base_fresh_val_orchestrator as ORCHESTRATOR
from scripts.show_base import published_test_winner_claim as AUTHORITY


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


class WorkloadCpuTests(unittest.TestCase):
    def test_regular_pinned_receipt_replays_exact_payload(self) -> None:
        if sys.platform == "linux":
            self.skipTest("formal Linux requires a producer-owned sealed memfd")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            body = {"format": "fixture", "status": "frozen"}
            receipt = dict(body)
            receipt["receipt_payload_sha256"] = (
                AUTHORITY.canonical_json_sha256(body)
            )
            payload = _canonical_bytes(receipt)
            path = root / "pinned.json"
            path.write_bytes(payload)
            self.assertEqual(
                WORKLOAD._read_pinned_receipt(
                    path,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_payload_sha256=receipt[
                        "receipt_payload_sha256"
                    ],
                    label="fixture pin",
                ),
                receipt,
            )

    def test_regular_pinned_receipt_rejects_duplicate_key(self) -> None:
        if sys.platform == "linux":
            self.skipTest("formal Linux requires a producer-owned sealed memfd")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "duplicate.json"
            payload = b'{"format":"a","format":"b","receipt_payload_sha256":"' + b"0" * 64 + b'"}\n'
            path.write_bytes(payload)
            with self.assertRaises(WORKLOAD.ProbeWorkloadError):
                WORKLOAD._read_pinned_receipt(
                    path,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_payload_sha256="0" * 64,
                    label="duplicate pin",
                )

    def test_projection_is_exact_first_prefix_and_create_new(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint = {"sha256": "a" * 64}
            full_rows = []
            subset_rows = []
            for index in range(3):
                clip_id = f"clip-{index}"
                prediction = _write(
                    root / f"prediction-{index}.npz",
                    f"prediction {index}\n".encode("ascii"),
                )
                subset_rows.append({"canonical_clip_id": clip_id})
                full_rows.append(
                    {
                        "canonical_clip_id": clip_id,
                        "epoch": 4,
                        "candidate_checkpoint_sha256": checkpoint["sha256"],
                        "prediction": prediction,
                    }
                )
            output = root / "projection.jsonl"
            artifact, projected = WORKLOAD._project_predictions(
                rows=full_rows,
                subset_rows=subset_rows,
                epoch=4,
                checkpoint=checkpoint,
                output=output,
            )
            self.assertEqual(artifact["path"], str(output))
            self.assertEqual(len(projected), 3)
            with self.assertRaises(FileExistsError):
                WORKLOAD._project_predictions(
                    rows=full_rows,
                    subset_rows=subset_rows,
                    epoch=4,
                    checkpoint=checkpoint,
                    output=output,
                )
            attacked = copy.deepcopy(full_rows)
            attacked[1]["candidate_checkpoint_sha256"] = "b" * 64
            with self.assertRaises(WORKLOAD.ProbeWorkloadError):
                WORKLOAD._project_predictions(
                    rows=attacked,
                    subset_rows=subset_rows,
                    epoch=4,
                    checkpoint=checkpoint,
                    output=root / "attacked.jsonl",
                )

    def test_command_record_comes_from_real_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            argv = [sys.executable, "-c", "print('real-command')"]
            record = WORKLOAD._run_one(
                argv,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
            )
            self.assertEqual(record["return_code"], 0)
            self.assertEqual(record["argv"], argv)
            self.assertEqual((root / "stdout.log").read_text(), "real-command\n")
            self.assertEqual(
                record["argv_sha256"], AUTHORITY.canonical_json_sha256(argv)
            )

    def test_full_lineage_semantics_ignore_only_paths(self) -> None:
        checkpoint = {"sha256": "c" * 64}

        def rows(prefix: str) -> list[dict[str, object]]:
            return [
                {
                    "global_index": index,
                    "canonical_clip_id": f"clip-{index:04d}",
                    "frames": 64,
                    "epoch": 8,
                    "candidate_checkpoint_sha256": checkpoint["sha256"],
                    "prediction": {
                        "path": f"/{prefix}/prediction-{index}.npz",
                        "sha256": f"{index:064x}",
                        "bytes": 100 + index,
                    },
                    "ground_truth": {
                        "path": f"/{prefix}/ground-truth-{index}.npz",
                        "sha256": f"{index + 1:064x}",
                        "bytes": 200 + index,
                    },
                }
                for index in range(ORCHESTRATOR.EXPECTED_CLIPS)
            ]

        serial = rows("serial")
        concurrent = rows("concurrent")
        self.assertEqual(
            PRODUCER._full_lineage_values_fingerprint(
                serial, epoch=8, checkpoint=checkpoint
            ),
            PRODUCER._full_lineage_values_fingerprint(
                concurrent, epoch=8, checkpoint=checkpoint
            ),
        )
        concurrent[0]["prediction"]["sha256"] = "f" * 64
        self.assertNotEqual(
            PRODUCER._full_lineage_values_fingerprint(
                serial, epoch=8, checkpoint=checkpoint
            ),
            PRODUCER._full_lineage_values_fingerprint(
                concurrent, epoch=8, checkpoint=checkpoint
            ),
        )


if __name__ == "__main__":
    unittest.main()
