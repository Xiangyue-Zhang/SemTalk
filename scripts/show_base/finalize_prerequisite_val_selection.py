#!/usr/bin/env python3
"""CPU-only exact-union finalizer for split SHOW prerequisite validation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Sequence

from scripts.show_base import check_prerequisite_val_multicandidate_gate as multicandidate_gate
from scripts.show_base import gate_released_all_speakers_on_show as source_gate
from scripts.show_base import merge_prerequisite_val_shards as merger
from scripts.show_base import prerequisite_val_contract as contract
from scripts.show_base import select_prerequisite_candidates as selector


PARTITION_FORMAT = "semtalk_show_prerequisite_val_partition_v1"


def _expected_jobs(index: dict[str, Any], stages: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "stage": stage,
            "epoch": epoch,
            "optimizer_updates": candidate["optimizer_updates"],
            "checkpoint": candidate["checkpoint"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
        }
        for stage in stages
        for epoch in contract.candidate_epochs_for_stage(index, stage)
        for candidate in [contract.candidate_lookup(index, stage, epoch)]
    ]


def validate_partial_full_authority(
    nonglobal_candidate_index: dict[str, Any],
    candidate_index: dict[str, Any],
) -> None:
    if (
        nonglobal_candidate_index["format"]
        not in {
            contract.PARTIAL_CANDIDATE_INDEX_FORMAT,
            contract.SEGMENTED_PARTIAL_CANDIDATE_INDEX_FORMAT,
        }
        or candidate_index["format"]
        not in {
            contract.CANDIDATE_INDEX_FORMAT,
            contract.SEGMENTED_CANDIDATE_INDEX_FORMAT,
        }
    ):
        raise contract.ContractError(
            "partial/full candidate schedules differ"
        )
    for stage in contract.STAGES[:-1]:
        if contract.candidate_epochs_for_stage(
            nonglobal_candidate_index, stage
        ) != contract.candidate_epochs_for_stage(candidate_index, stage):
            raise contract.ContractError(
                "partial/full candidate schedules differ"
            )
    for key in (
        "source_receipts",
        "config_sha256",
        "dataset_receipt_sha256",
        "formal_training_status",
        "stages",
    ):
        full_subset = {
            stage: candidate_index[key][stage]
            for stage in contract.STAGES[:-1]
        }
        if nonglobal_candidate_index[key] != full_subset:
            raise contract.ContractError(
                f"partial/full candidate {key} subset differs"
            )


def _load_partition(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    receipt = contract.verify_receipt_payload(
        contract.strict_json_bytes(payload, str(path)),
        "formal partition receipt",
    )
    return contract.exact_keys(
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
        "formal partition receipt",
    )


def _validate_partition_union(
    *,
    roots: Sequence[Path],
    candidate_index: dict[str, Any],
    candidate_index_path: Path,
    candidate_index_sha256: str,
    nonglobal_candidate_index: dict[str, Any],
    nonglobal_candidate_index_path: Path,
    nonglobal_candidate_index_sha256: str,
    manifest_sha256: str,
    summary_sha256: str,
    lineage_sha256: str,
    source_commit: str,
    source_tree: str,
) -> list[Path]:
    if len(roots) != 2:
        raise contract.ContractError(
            "finalizer requires exactly nonglobal and global shard roots"
        )
    resolved: list[Path] = []
    receipts: dict[str, dict[str, Any]] = {}
    gate_bindings: dict[str, dict[str, str]] = {}
    for raw_root in roots:
        if (
            not raw_root.is_absolute()
            or raw_root.is_symlink()
            or not raw_root.is_dir()
        ):
            raise contract.ContractError(
                "formal shard roots must be absolute regular directories"
            )
        root = raw_root.resolve(strict=True)
        if root in resolved:
            raise contract.ContractError("formal shard roots must be distinct regular directories")
        resolved.append(root)
        receipt_path = root / "partition_receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise contract.ContractError("formal partition receipt missing")
        receipt = _load_partition(receipt_path)
        partition = receipt["partition"]
        if (
            receipt["format"] != PARTITION_FORMAT
            or receipt["status"] != "complete"
            or receipt["test_visible"] is not False
            or partition not in {"nonglobal", "global"}
            or partition in receipts
        ):
            raise contract.ContractError("formal partition coverage mismatch")
        stages = contract.STAGES[:-1] if partition == "nonglobal" else ("global",)
        expected_jobs = _expected_jobs(candidate_index, stages)
        authority_path = (
            nonglobal_candidate_index_path
            if partition == "nonglobal"
            else candidate_index_path
        )
        authority_sha = (
            nonglobal_candidate_index_sha256
            if partition == "nonglobal"
            else candidate_index_sha256
        )
        expected_inputs = {
            "candidate_index": str(authority_path.resolve(strict=True)),
            "candidate_index_sha256": authority_sha,
            "canonical_manifest_sha256": manifest_sha256,
            "canonical_summary_sha256": summary_sha256,
            "canonical_lineage_sha256": lineage_sha256,
            "source_commit": source_commit,
            "source_tree": source_tree,
        }
        inputs = receipt["inputs"]
        if not isinstance(inputs, dict):
            raise contract.ContractError("formal partition inputs missing")
        current_gate = inputs.get("multi_candidate_gate")
        if (
            {key: inputs.get(key) for key in expected_inputs} != expected_inputs
            or not isinstance(current_gate, dict)
            or set(current_gate) != {"path", "sha256"}
            or receipt["jobs"] != expected_jobs
            or receipt["coverage"]
            != {
                "candidate_jobs": len(expected_jobs),
                "shard_jobs": len(expected_jobs) * contract.EXPECTED_SHARDS,
                "exact_once": True,
            }
            or receipt["protocol"].get("candidates_per_wave") != 4
            or receipt["protocol"].get("shards_per_candidate")
            != contract.EXPECTED_SHARDS
        ):
            raise contract.ContractError("formal partition binding mismatch")
        expected_relative = {
            Path("shards")
            / row["stage"]
            / f"epoch_{row['epoch']:04d}"
            / f"shard_{shard:02d}.json"
            for row in expected_jobs
            for shard in range(contract.EXPECTED_SHARDS)
        }
        observed_relative: set[Path] = set()
        shard_subtree = root / "shards"
        if shard_subtree.is_symlink() or not shard_subtree.is_dir():
            raise contract.ContractError("formal shard subtree missing")
        for child in shard_subtree.rglob("*"):
            if child.is_symlink():
                raise contract.ContractError("formal shard subtree contains symlink")
            if child.is_dir():
                continue
            if not child.is_file():
                raise contract.ContractError("formal shard subtree entry invalid")
            observed_relative.add(child.relative_to(root))
        if observed_relative != expected_relative:
            raise contract.ContractError(
                "formal partition shard inventory does not match its jobs"
            )
        gate_bindings[partition] = current_gate
        receipts[partition] = receipt
    if set(receipts) != {"nonglobal", "global"}:
        raise contract.ContractError("formal partition union is incomplete")
    for partition, binding in gate_bindings.items():
        gate_receipt = multicandidate_gate.replay_gate(
            Path(binding["path"]),
            binding["sha256"],
        )
        authority_path = (
            nonglobal_candidate_index_path
            if partition == "nonglobal"
            else candidate_index_path
        )
        authority_sha = (
            nonglobal_candidate_index_sha256
            if partition == "nonglobal"
            else candidate_index_sha256
        )
        expected_gate_inputs = {
            "candidate_index": str(authority_path.resolve(strict=True)),
            "candidate_index_sha256": authority_sha,
            "canonical_manifest_sha256": manifest_sha256,
            "canonical_summary_sha256": summary_sha256,
            "canonical_lineage_sha256": lineage_sha256,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "multi_candidate_gate": None,
        }
        if gate_receipt["inputs"]["common"] != expected_gate_inputs:
            raise contract.ContractError(
                "formal partition gate binding mismatch"
            )
    combined = receipts["nonglobal"]["jobs"] + receipts["global"]["jobs"]
    expected_all = _expected_jobs(candidate_index, contract.STAGES)
    identities = [(row["stage"], row["epoch"]) for row in combined]
    if (
        combined != expected_all
        or len(identities) != len(set(identities))
        or len(combined)
        != sum(
            len(contract.candidate_epochs_for_stage(candidate_index, stage))
            for stage in contract.STAGES
        )
    ):
        raise contract.ContractError("formal partition jobs are not an exact disjoint union")
    return resolved


def finalize(
    *,
    candidate_index_path: Path,
    candidate_index_sha256: str,
    nonglobal_candidate_index_path: Path,
    nonglobal_candidate_index_sha256: str,
    canonical_manifest: Path,
    manifest_sha256: str,
    canonical_summary: Path,
    summary_sha256: str,
    canonical_lineage: Path,
    lineage_sha256: str,
    shard_roots: Sequence[Path],
    measurement_root: Path,
    selection_json: Path,
    source_commit: str,
    source_tree: str,
) -> dict[str, Any]:
    candidate_index, _ = contract.load_candidate_index(
        candidate_index_path,
        candidate_index_sha256,
    )
    nonglobal_candidate_index, _ = contract.load_candidate_index(
        nonglobal_candidate_index_path,
        nonglobal_candidate_index_sha256,
        allow_partial=True,
    )
    validate_partial_full_authority(
        nonglobal_candidate_index,
        candidate_index,
    )
    roots = _validate_partition_union(
        roots=shard_roots,
        candidate_index=candidate_index,
        candidate_index_path=candidate_index_path,
        candidate_index_sha256=candidate_index_sha256,
        nonglobal_candidate_index=nonglobal_candidate_index,
        nonglobal_candidate_index_path=nonglobal_candidate_index_path,
        nonglobal_candidate_index_sha256=nonglobal_candidate_index_sha256,
        manifest_sha256=manifest_sha256,
        summary_sha256=summary_sha256,
        lineage_sha256=lineage_sha256,
        source_commit=source_commit,
        source_tree=source_tree,
    )
    merge_source = source_gate.git_source_receipt(
        source_commit,
        source_tree,
        script=Path(merger.__file__).resolve(),
    )
    merger.merge(
        candidate_index_path=candidate_index_path,
        candidate_index_sha256=candidate_index_sha256,
        canonical_manifest=canonical_manifest,
        canonical_manifest_sha256=manifest_sha256,
        canonical_summary=canonical_summary,
        canonical_summary_sha256=summary_sha256,
        canonical_lineage=canonical_lineage,
        canonical_lineage_sha256=lineage_sha256,
        shard_roots=roots,
        output_root=measurement_root,
        merge_source=merge_source,
    )
    measurement_index = measurement_root / "measurement_index.json"
    measurement_sha = hashlib.sha256(measurement_index.read_bytes()).hexdigest()
    selector_source = source_gate.git_source_receipt(
        source_commit,
        source_tree,
        script=Path(selector.__file__).resolve(),
    )
    result = selector.select(
        candidate_index_path=candidate_index_path,
        candidate_index_sha256=candidate_index_sha256,
        measurement_index_path=measurement_index,
        measurement_index_sha256=measurement_sha,
        output_json=selection_json,
        selector_source=selector_source,
    )
    selection_sha = hashlib.sha256(selection_json.read_bytes()).hexdigest()
    replayed = selector.replay_selection(
        selection_path=selection_json,
        expected_selection_sha256=selection_sha,
    )
    if replayed != result:
        raise contract.ContractError("final selected-five fresh replay changed")
    return {
        "selection": result,
        "selection_sha256": selection_sha,
        "measurement_index_sha256": measurement_sha,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--expected-candidate-index-sha256", required=True)
    parser.add_argument(
        "--nonglobal-candidate-index",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-nonglobal-candidate-index-sha256",
        required=True,
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = finalize(
        candidate_index_path=args.candidate_index,
        candidate_index_sha256=args.expected_candidate_index_sha256,
        nonglobal_candidate_index_path=args.nonglobal_candidate_index,
        nonglobal_candidate_index_sha256=(
            args.expected_nonglobal_candidate_index_sha256
        ),
        canonical_manifest=args.canonical_manifest,
        manifest_sha256=args.expected_manifest_sha256,
        canonical_summary=args.canonical_summary,
        summary_sha256=args.expected_summary_sha256,
        canonical_lineage=args.canonical_lineage,
        lineage_sha256=args.expected_lineage_sha256,
        shard_roots=args.shard_root,
        measurement_root=args.measurement_root,
        selection_json=args.selection_json,
        source_commit=args.expected_source_commit,
        source_tree=args.expected_source_tree,
    )
    print(result["selection_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
