#!/usr/bin/env python3
"""Append one immutable continuation segment to a prerequisite index.

The initial e20..e200 index and every prior segment remain immutable.  This
publisher replays the authorized continuation wave, proves each new run ended
at the one authorized target epoch, validates the target checkpoint/final
model/resume/status, and emits a new union index whose segment chain maps every
candidate to its original run path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.show_base import build_prerequisite_candidate_index as initial
from scripts.show_base import prerequisite_continuation_runtime as runtime
from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract


class SegmentedIndexError(RuntimeError):
    """Raised when a continuation segment cannot join the frozen index."""


def _artifact(path: Path, sha: str, payload_sha: str) -> dict[str, Any]:
    resolved, payload, observed = contract.read_verified_file(
        path, sha, "segmented union input", val_only=False
    )
    if observed != sha:
        raise SegmentedIndexError("segmented union input changed")
    contract.require_sha256(payload_sha, "segmented union input payload SHA")
    return {
        "path": str(resolved),
        "sha256": observed,
        "bytes": len(payload),
        "receipt_payload_sha256": payload_sha,
    }


def _parse_runs(values: Sequence[str], stages: Sequence[str]) -> dict[str, Path]:
    parsed = initial._parse_stage_runs(values, stages)
    return parsed


def _wave_entry(receipt: Mapping[str, Any], stage: str) -> dict[str, Any]:
    matches = [value for value in receipt["stages"] if value["stage"] == stage]
    if len(matches) != 1:
        raise SegmentedIndexError(f"{stage} wave entry is not exact-once")
    return matches[0]


def _old_read_only_proof(value: Any, *, stage: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != runtime.OLD_BEFORE_AFTER_KEYS
        or value.get("format") != runtime.OLD_BEFORE_AFTER_FORMAT
        or value.get("stage") != stage
        or value.get("unchanged") is not True
    ):
        raise SegmentedIndexError(f"{stage} old-segment proof mismatch")
    before = runtime.validate_old_segment_snapshot(value["before"])
    after = runtime.validate_old_segment_snapshot(value["after"])
    if before != after:
        raise SegmentedIndexError(f"{stage} old segment changed")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_payload_sha256")
    if claimed != wave.canonical_json_sha256(unsigned):
        raise SegmentedIndexError(f"{stage} old-segment proof hash mismatch")


def _build_stage(
    torch: Any,
    *,
    stage: str,
    run: Path,
    target: int,
    wave_entry: Mapping[str, Any],
    wave_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str, dict[str, Any]]:
    new = wave_entry["new_segment"]
    if new["run_path"] != str(run) or new["authorized_candidate_epochs"] != [target]:
        raise SegmentedIndexError(f"{stage} run differs from wave authority")
    status_path = contract.regular_file(
        run / "formal_training_status.json", f"{stage} continuation status"
    )
    status_bytes = status_path.read_bytes()
    status = contract.strict_json_bytes(status_bytes, str(status_path))
    if (
        not isinstance(status, dict)
        or status.get("status") != "complete"
        or status.get("formal_stage") != stage
        or status.get("completed_epochs") != target
        or status.get("optimizer_updates")
        != target * contract.EXPECTED_UPDATES_PER_EPOCH
        or status.get("updates_per_epoch") != contract.EXPECTED_UPDATES_PER_EPOCH
        or status.get("continuation_wave_receipt") != wave_binding
        or status.get("hostname") != new["host"]
        or status.get("config_sha256") != new["config_sha256"]
    ):
        raise SegmentedIndexError(f"{stage} continuation status mismatch")
    _old_read_only_proof(
        status.get("continuation_old_segment_read_only"), stage=stage
    )
    source = contract.validate_training_audit_source(
        status.get("source_receipt"),
        f"{stage} continuation source",
        reprove_entrypoint=True,
    )
    source_sha = contract.canonical_payload_sha256(source)
    if (
        status.get("source_receipt_sha256") != source_sha
        or {
            "commit": source.get("commit"),
            "tree": source.get("tree"),
            "source_receipt_sha256": source_sha,
        }
        != new["source"]
    ):
        raise SegmentedIndexError(f"{stage} continuation source mismatch")
    dataset = status.get("dataset_receipt")
    if (
        not isinstance(dataset, dict)
        or runtime.dataset_semantic_sha256(dataset)
        != new["dataset_semantic_sha256"]
        or status.get("smplx_asset_receipt") != new["smplx_asset"]
    ):
        raise SegmentedIndexError(f"{stage} continuation dataset mismatch")
    dataset_sha = contract.canonical_payload_sha256(dataset)
    initialization = contract.validate_initialization_receipt(
        status.get("initialization_receipt"),
        stage=stage,
        label=f"{stage} continuation initialization",
        reprove_path=True,
    )
    distributed = contract.validate_distributed_training_receipt(
        status.get("distributed_training_receipt"),
        stage=stage,
        label=f"{stage} continuation distributed receipt",
    )
    prior = contract.validate_rvq_ema_prior_receipt(
        status.get("rvq_ema_prior_receipt"),
        stage=stage,
        label=f"{stage} continuation RVQ prior",
    )
    rank_state = contract.validate_rvq_rank_state_receipt(
        status.get("rvq_rank_state_receipt"),
        stage=stage,
        distributed=distributed,
        label=f"{stage} continuation RVQ rank state",
    )
    lineage_sha = contract.require_sha256(
        status.get("lineage_manifest_sha256"), f"{stage} lineage SHA"
    )
    expected_common = {
        "source_receipt": source,
        "source_receipt_sha256": source_sha,
        "config_sha256": new["config_sha256"],
        "lineage_manifest_sha256": lineage_sha,
        "dataset_receipt_sha256": dataset_sha,
        "initialization_receipt": initialization,
        "rvq_ema_prior_receipt": prior,
        "distributed_training_receipt": distributed,
    }
    final_evidence, final_audit, final_state = initial._audit_final_checkpoint(
        torch, status=status, run=run, stage=stage, final_epoch=target
    )
    initial._require_common_audit_binding(
        final_audit, expected_common, f"{stage} continuation final/status"
    )
    if (
        final_audit.get("rvq_rank_state_receipt") != rank_state
        or final_audit.get("continuation_wave_receipt") != wave_binding
    ):
        raise SegmentedIndexError(f"{stage} final continuation binding mismatch")
    candidate_dir = run / "representation_candidates"
    expected_path = candidate_dir / (
        f"{stage}_epoch_{target:04d}_step_"
        f"{target * contract.EXPECTED_UPDATES_PER_EPOCH:09d}.bin"
    )
    if (
        candidate_dir.is_symlink()
        or not candidate_dir.is_dir()
        or {path.name for path in candidate_dir.iterdir()} != {expected_path.name}
    ):
        raise SegmentedIndexError(f"{stage} target candidate inventory is not exact")
    candidate_path = contract.regular_file(
        expected_path, f"{stage} e{target} continuation candidate"
    )
    audit, checkpoint_sha, candidate_state = initial._checkpoint_audit(
        torch, candidate_path, stage=stage, epoch=target
    )
    initial._require_common_audit_binding(
        audit, expected_common, f"{stage} continuation candidate/status"
    )
    if (
        audit.get("rvq_rank_state_receipt") != rank_state
        or audit.get("continuation_wave_receipt") != wave_binding
    ):
        raise SegmentedIndexError(f"{stage} candidate wave binding mismatch")
    entry = {
        "epoch": target,
        "optimizer_updates": target * contract.EXPECTED_UPDATES_PER_EPOCH,
        "checkpoint": str(candidate_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": candidate_path.stat().st_size,
        "checkpoint_audit_sha256": contract.canonical_payload_sha256(audit),
    }
    initial._validate_latest_candidate_receipt(
        status.get("latest_representation_candidate"),
        stage=stage,
        candidate=entry,
        final_epoch=target,
        label=f"{stage} continuation latest candidate",
    )
    initial._validate_latest_candidate_receipt(
        final_audit.get("latest_representation_candidate"),
        stage=stage,
        candidate=entry,
        final_epoch=target,
        label=f"{stage} continuation final latest candidate",
    )
    initial._assert_tensor_states_equal(
        torch, candidate_state, final_state, f"{stage} target/final model state"
    )
    resume_path = contract.regular_file(run / "latest_resume.pt", f"{stage} resume")
    resume_sha = contract.require_sha256(
        status.get("latest_resume_sha256"), f"{stage} resume SHA"
    )
    if contract.sha256_file(resume_path) != resume_sha:
        raise SegmentedIndexError(f"{stage} continuation resume changed")
    status_artifact = {
        "path": str(status_path),
        "sha256": hashlib.sha256(status_bytes).hexdigest(),
        "finite_evidence": {
            "final_checkpoint": final_evidence,
            "latest_resume": {
                "path": str(resume_path),
                "sha256": resume_sha,
                "bytes": resume_path.stat().st_size,
            },
            "last_metrics": initial._finite_metrics(
                status.get("last_metrics"), f"{stage} continuation status"
            ),
            "all_candidate_model_tensors_finite": True,
        },
    }
    frozen_source = contract.freeze_training_audit_source(
        source, f"{stage} continuation source"
    )
    return entry, frozen_source, new["config_sha256"], dataset_sha, status_artifact


def build(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    scope = args.stage_scope
    stages = contract.STAGES if scope == "full" else contract.STAGES[:-1]
    prior, _ = contract.load_candidate_index(
        args.prior_candidate_index,
        args.expected_prior_candidate_index_sha256,
        allow_partial=False,
    )
    expected_format = (
        contract.CANDIDATE_INDEX_FORMAT
        if scope == "full"
        else contract.PARTIAL_CANDIDATE_INDEX_FORMAT
    )
    if prior["format"] != contract.CANDIDATE_INDEX_FORMAT:
        raise SegmentedIndexError(
            "the wave-bound full prior index is required for either output scope"
        )
    prior_artifact = _artifact(
        args.prior_candidate_index,
        args.expected_prior_candidate_index_sha256,
        prior["receipt_payload_sha256"],
    )
    receipt = wave.replay_wave_file(
        args.continuation_wave, args.expected_continuation_wave_sha256
    )
    wave_artifact = _artifact(
        args.continuation_wave,
        args.expected_continuation_wave_sha256,
        receipt["receipt_payload_sha256"],
    )
    target = receipt["target_epoch"]
    if tuple(prior["candidate_epochs"]) != tuple(range(20, target, 20)):
        raise SegmentedIndexError("prior index does not end at wave boundary")
    runs = _parse_runs(args.stage_run, stages)
    wave_binding = {
        key: wave_artifact[key]
        for key in ("path", "sha256", "receipt_payload_sha256")
    }
    new_entries: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] = {}
    configs: dict[str, str] = {}
    datasets: dict[str, str] = {}
    statuses: dict[str, Any] = {}
    chains: dict[str, Any] = {}
    for stage in stages:
        wave_entry = _wave_entry(receipt, stage)
        old = wave_entry["old_segment"]
        catalog = old["candidate_catalog_receipt"]
        if (
            catalog["path"] != prior_artifact["path"]
            or catalog["sha256"] != prior_artifact["sha256"]
            or catalog["receipt_payload_sha256"]
            != prior_artifact["receipt_payload_sha256"]
        ):
            raise SegmentedIndexError(f"{stage} wave binds a different prior index")
        entry, source, config_sha, dataset_sha, status = _build_stage(
            torch,
            stage=stage,
            run=runs[stage],
            target=target,
            wave_entry=wave_entry,
            wave_binding=wave_binding,
        )
        appended = wave._make_chain_segment(
            run_path=wave_entry["new_segment"]["run_path"],
            start_epoch=target,
            end_epoch=target,
            predecessor_segment_id=wave_entry["new_segment"][
                "predecessor_segment_id"
            ],
        )
        if appended["segment_id"] != wave_entry["new_segment"]["chain_segment_id"]:
            raise SegmentedIndexError(f"{stage} new segment identity changed")
        chain = [*old["candidate_segment_chain"], appended]
        if "segmented_union" in prior and (
            prior["segmented_union"]["candidate_segment_chain"][stage]
            != old["candidate_segment_chain"]
        ):
            raise SegmentedIndexError(f"{stage} prior segmented chain changed")
        new_entries[stage] = entry
        sources[stage] = source
        configs[stage] = config_sha
        datasets[stage] = dataset_sha
        statuses[stage] = status
        chains[stage] = chain
    portable = {
        contract.canonical_payload_sha256(
            contract.portable_training_source_identity(
                source["portable_identity"], "terminal portable source"
            )
        )
        for source in sources.values()
    }
    if len(portable) != 1:
        raise SegmentedIndexError("terminal continuation sources differ")
    prior_waves = (
        list(prior["segmented_union"]["continuation_waves"])
        if "segmented_union" in prior
        else []
    )
    segmented = {
        "format": contract.SEGMENTED_UNION_FORMAT,
        "status": "complete",
        "prior_candidate_index": prior_artifact,
        "continuation_waves": [*prior_waves, wave_artifact],
        "candidate_segment_chain": chains,
    }
    segmented["receipt_payload_sha256"] = contract.canonical_payload_sha256(
        segmented
    )
    payload = contract.receipt_payload(
        {
            "format": expected_format,
            "status": "complete",
            "target_dataset": "SHOW",
            "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
            "selection_split": "val",
            "test_visible": False,
            "candidate_epochs": [*prior["candidate_epochs"], target],
            "updates_per_epoch": contract.EXPECTED_UPDATES_PER_EPOCH,
            "source_receipts": sources,
            "config_sha256": configs,
            "dataset_receipt_sha256": datasets,
            "formal_training_status": statuses,
            "stages": {
                stage: [*prior["stages"][stage], new_entries[stage]]
                for stage in stages
            },
            "segmented_union": segmented,
        }
    )
    contract.validate_candidate_index(
        payload, allow_partial=scope == "nonglobal"
    )
    file_sha = contract.atomic_json_new(args.output_json, payload)
    replayed, _artifact_value = contract.load_candidate_index(
        args.output_json, file_sha, allow_partial=scope == "nonglobal"
    )
    if replayed != payload:
        raise SegmentedIndexError("published segmented index differs from replay")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage-scope", choices=("full", "nonglobal"), default="full")
    parser.add_argument("--prior-candidate-index", type=Path, required=True)
    parser.add_argument("--expected-prior-candidate-index-sha256", required=True)
    parser.add_argument("--continuation-wave", type=Path, required=True)
    parser.add_argument("--expected-continuation-wave-sha256", required=True)
    parser.add_argument("--stage-run", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build(_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal_epoch": result["candidate_epochs"][-1],
                "stages": list(result["stages"]),
                "receipt_payload_sha256": result["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
