#!/usr/bin/env python3
"""Prove 4-candidate SHOW prerequisite validation is lossless and faster."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Sequence

from scripts.show_base import prerequisite_val_contract as contract


FORMAT = "semtalk_show_prerequisite_val_multicandidate_gate_v1"
PARTITION_FORMAT = "semtalk_show_prerequisite_val_partition_v1"
MAX_ELAPSED_RATIO = 1.10


def _load_partition(root: Path, expected_concurrency: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise contract.ContractError(
            "gate root must be an absolute regular directory"
        )
    root = root.resolve(strict=True)
    path = root / "partition_receipt.json"
    if path.is_symlink() or not path.is_file():
        raise contract.ContractError("gate partition receipt must be regular")
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    receipt = contract.verify_receipt_payload(
        contract.strict_json_bytes(payload, str(path)),
        "gate partition receipt",
    )
    receipt = contract.exact_keys(
        receipt,
        (
            "format",
            "status",
            "partition",
            "test_visible",
            "protocol",
            "timing",
            "inputs",
            "jobs",
            "coverage",
            "receipt_payload_sha256",
        ),
        "gate partition receipt",
    )
    protocol = receipt["protocol"]
    timing = receipt["timing"]
    coverage = receipt["coverage"]
    if (
        receipt["format"] != PARTITION_FORMAT
        or receipt["status"] != "complete"
        or receipt["partition"] != "gate"
        or receipt["test_visible"] is not False
        or not isinstance(protocol, dict)
        or protocol.get("candidates_per_wave") != expected_concurrency
        or protocol.get("shards_per_candidate") != contract.EXPECTED_SHARDS
        or protocol.get("candidate_jobs") != 4
        or protocol.get("waves") != 4 // expected_concurrency
        or not isinstance(timing, dict)
        or not isinstance(timing.get("elapsed_seconds"), int)
        or timing["elapsed_seconds"] <= 0
        or coverage
        != {
            "candidate_jobs": 4,
            "shard_jobs": 32,
            "exact_once": True,
        }
    ):
        raise contract.ContractError("gate partition protocol mismatch")
    jobs = receipt["jobs"]
    inputs = receipt["inputs"]
    if (
        not isinstance(inputs, dict)
        or set(inputs)
        != {
            "candidate_index",
            "candidate_index_sha256",
            "canonical_manifest_sha256",
            "canonical_summary_sha256",
            "canonical_lineage_sha256",
            "source_commit",
            "source_tree",
            "multi_candidate_gate",
        }
        or inputs["multi_candidate_gate"] is not None
        or not isinstance(jobs, list)
        or len(jobs) != 4
    ):
        raise contract.ContractError("gate partition job coverage mismatch")
    return receipt, {
        "path": str(path.resolve(strict=True)),
        "sha256": sha,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }


def _shards(root: Path) -> dict[Path, bytes]:
    subtree = root / "shards"
    if subtree.is_symlink() or not subtree.is_dir():
        raise contract.ContractError("gate shard subtree is invalid")
    result: dict[Path, bytes] = {}
    for path in sorted(subtree.rglob("*")):
        if path.is_symlink():
            raise contract.ContractError("gate shard subtree contains symlink")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix != ".json":
            raise contract.ContractError("gate shard subtree has extra file")
        relative = path.relative_to(root)
        result[relative] = path.read_bytes()
    if len(result) != 32:
        raise contract.ContractError("gate requires exactly 32 shard receipts")
    return result


def check(
    *,
    serial_root: Path,
    concurrent_root: Path,
    output_json: Path | None,
) -> dict[str, Any]:
    serial_receipt, serial_binding = _load_partition(serial_root, 1)
    concurrent_receipt, concurrent_binding = _load_partition(concurrent_root, 4)
    for key in ("inputs", "jobs"):
        if serial_receipt[key] != concurrent_receipt[key]:
            raise contract.ContractError(f"gate {key} changed across runs")
    if serial_receipt["protocol"]["batch_size"] != concurrent_receipt["protocol"]["batch_size"]:
        raise contract.ContractError("gate batch size changed across runs")
    serial_shards = _shards(serial_root.resolve(strict=True))
    concurrent_shards = _shards(concurrent_root.resolve(strict=True))
    if set(serial_shards) != set(concurrent_shards):
        raise contract.ContractError("gate shard inventories differ")
    for relative in sorted(serial_shards):
        if serial_shards[relative] != concurrent_shards[relative]:
            raise contract.ContractError(
                f"gate shard payload differs: {relative}"
            )
    serial_elapsed = serial_receipt["timing"]["elapsed_seconds"]
    concurrent_elapsed = concurrent_receipt["timing"]["elapsed_seconds"]
    ratio = concurrent_elapsed / serial_elapsed
    if ratio > MAX_ELAPSED_RATIO:
        raise contract.ContractError("4-candidate gate has a throughput regression")
    tree_hash = hashlib.sha256()
    for relative in sorted(serial_shards):
        tree_hash.update(str(relative).encode("utf-8"))
        tree_hash.update(b"\0")
        tree_hash.update(serial_shards[relative])
    result = contract.receipt_payload(
        {
            "format": FORMAT,
            "status": "pass",
            "test_visible": False,
            "protocol": {
                "serial_candidates_per_wave": 1,
                "concurrent_candidates_per_wave": 4,
                "candidates": 4,
                "shards_per_candidate": contract.EXPECTED_SHARDS,
                "batch_size": serial_receipt["protocol"]["batch_size"],
                "numeric_equivalence": "byte_exact_receipt_payloads",
                "maximum_elapsed_ratio": MAX_ELAPSED_RATIO,
            },
            "inputs": {
                "serial_partition": serial_binding,
                "concurrent_partition": concurrent_binding,
                "common": serial_receipt["inputs"],
                "jobs": serial_receipt["jobs"],
            },
            "equivalence": {
                "shard_receipts": len(serial_shards),
                "shard_tree_sha256": tree_hash.hexdigest(),
                "byte_exact": True,
            },
            "throughput": {
                "serial_elapsed_seconds": serial_elapsed,
                "concurrent_elapsed_seconds": concurrent_elapsed,
                "concurrent_over_serial_ratio": ratio,
                "pass": True,
            },
        }
    )
    if output_json is not None:
        contract.atomic_json_new(output_json, result)
    return result


def replay_gate(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved, payload, _ = contract.read_verified_file(
        path,
        expected_sha256,
        "multi-candidate gate",
    )
    observed = contract.verify_receipt_payload(
        contract.strict_json_bytes(payload, str(resolved)),
        "multi-candidate gate",
    )
    observed = contract.exact_keys(
        observed,
        (
            "format",
            "status",
            "test_visible",
            "protocol",
            "inputs",
            "equivalence",
            "throughput",
            "receipt_payload_sha256",
        ),
        "multi-candidate gate",
    )
    inputs = observed.get("inputs")
    if (
        observed["format"] != FORMAT
        or observed["status"] != "pass"
        or observed["test_visible"] is not False
        or not isinstance(inputs, dict)
        or not isinstance(inputs.get("serial_partition"), dict)
        or not isinstance(inputs.get("concurrent_partition"), dict)
    ):
        raise contract.ContractError("multi-candidate gate protocol mismatch")
    expected = check(
        serial_root=Path(inputs["serial_partition"]["path"]).parent,
        concurrent_root=Path(inputs["concurrent_partition"]["path"]).parent,
        output_json=None,
    )
    if contract.canonical_json_bytes(observed) != contract.canonical_json_bytes(expected):
        raise contract.ContractError("multi-candidate gate differs from fresh replay")
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--serial-root", type=Path, required=True)
    parser.add_argument("--concurrent-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check(
        serial_root=args.serial_root,
        concurrent_root=args.concurrent_root,
        output_json=args.output_json,
    )
    print(result["receipt_payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
