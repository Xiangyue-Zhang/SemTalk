#!/usr/bin/env python3
"""Freeze the five completed prerequisite candidate sets into one index."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Sequence


from scripts.show_base import prerequisite_val_contract as contract


def _parse_stage_runs(
    values: Sequence[str],
    expected_stages: Sequence[str] = contract.STAGES,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("--stage-run must be STAGE=/absolute/run/path")
        stage, path_value = raw.split("=", 1)
        if stage not in expected_stages or stage in result:
            raise ValueError(f"invalid or duplicate stage-run {stage!r}")
        path = Path(path_value)
        if not path.is_absolute():
            raise ValueError("stage run paths must be absolute")
        contract.reject_forbidden_label(path, f"{stage} stage run")
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"invalid {stage} stage run directory")
        result[stage] = path.resolve(strict=True)
    if set(result) != set(expected_stages):
        raise ValueError(
            f"exactly {len(expected_stages)} requested stage runs are required"
        )
    return result


def _checkpoint_audit(
    torch: Any,
    path: Path,
    *,
    stage: str,
    epoch: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    payload_bytes = path.read_bytes()
    checkpoint = torch.load(
        io.BytesIO(payload_bytes),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model_state",
        "audit",
    }:
        raise RuntimeError(f"{path}: invalid representation candidate")
    audit = contract.validate_representation_candidate_audit(
        checkpoint["audit"],
        stage=stage,
        epoch=epoch,
        label=f"{path} candidate audit",
        reprove_paths=False,
    )
    model_state = checkpoint["model_state"]
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError(f"{path}: empty candidate model state")
    for name, tensor in model_state.items():
        if (
            not isinstance(name, str)
            or not torch.is_tensor(tensor)
            or (
                (tensor.is_floating_point() or tensor.is_complex())
                and not bool(tensor.isfinite().all().item())
            )
        ):
            raise RuntimeError(f"{path}: invalid/non-finite model tensor")
    return audit, contract.sha256_file(path), dict(model_state)


def _finite_metrics(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"{label}: final metrics are missing")
    result = {}
    for name, metric in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(metric, dict)
            or set(metric) != {"avg", "count"}
            or isinstance(metric["avg"], bool)
            or type(metric["avg"]) not in {int, float}
            or not math.isfinite(float(metric["avg"]))
            or isinstance(metric["count"], bool)
            or not isinstance(metric["count"], int)
            or metric["count"] <= 0
        ):
            raise RuntimeError(f"{label}: invalid metric {name!r}")
        result[name] = dict(metric)
    return result


def _audit_final_checkpoint(
    torch: Any,
    *,
    status: dict[str, Any],
    run: Path,
    stage: str,
    final_epoch: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final_path = contract.regular_file(
        status.get("final_checkpoint"),
        f"{stage} final checkpoint",
    )
    if final_path.parent != run:
        raise RuntimeError(f"{stage} final checkpoint escapes stage run")
    expected_sha = contract.require_sha256(
        status.get("final_checkpoint_sha256"),
        f"{stage} final checkpoint SHA-256",
    )
    payload = final_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise RuntimeError(f"{stage} final checkpoint changed")
    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model_state",
        "audit",
    }:
        raise RuntimeError(f"{stage} final checkpoint schema mismatch")
    audit = checkpoint["audit"]
    if (
        not isinstance(audit, dict)
        or audit.get("format") != "semtalk_show_model_v2"
        or audit.get("formal_stage") != stage
        or audit.get("optimizer_updates")
        != final_epoch * contract.updates_per_epoch(stage)
    ):
        raise RuntimeError(f"{stage} final checkpoint audit mismatch")
    source = contract.validate_training_audit_source(
        audit.get("source_receipt"),
        f"{stage} final checkpoint source",
        reprove_entrypoint=False,
    )
    if (
        contract.require_sha256(
            audit.get("source_receipt_sha256"),
            f"{stage} final source receipt SHA-256",
        )
        != contract.canonical_payload_sha256(source)
    ):
        raise RuntimeError(f"{stage} final source payload hash mismatch")
    for key in (
        "config_sha256",
        "lineage_manifest_sha256",
        "dataset_summary_sha256",
        "data_mdb_sha256",
        "dataset_receipt_sha256",
    ):
        contract.require_sha256(
            audit.get(key),
            f"{stage} final {key}",
        )
    contract.validate_initialization_receipt(
        audit.get("initialization_receipt"),
        stage=stage,
        label=f"{stage} final initialization receipt",
        reprove_path=False,
    )
    distributed = contract.validate_distributed_training_receipt(
        audit.get("distributed_training_receipt"),
        stage=stage,
        label=f"{stage} final distributed training receipt",
    )
    contract.validate_optimizer_runtime_receipt(
        audit.get("optimizer_runtime_receipt"),
        stage=stage,
        label=f"{stage} final optimizer runtime receipt",
        required=stage == "global",
    )
    contract.validate_rvq_ema_prior_receipt(
        audit.get("rvq_ema_prior_receipt"),
        stage=stage,
        label=f"{stage} final RVQ EMA-prior receipt",
    )
    contract.validate_rvq_rank_state_receipt(
        audit.get("rvq_rank_state_receipt"),
        stage=stage,
        distributed=distributed,
        label=f"{stage} final RVQ rank-state receipt",
    )
    state = checkpoint["model_state"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{stage} final checkpoint model state is empty")
    tensor_count = 0
    element_count = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise RuntimeError(f"{stage} final model entry is invalid")
        if (
            (tensor.is_floating_point() or tensor.is_complex())
            and not bool(tensor.isfinite().all().item())
        ):
            raise RuntimeError(f"{stage} final tensor {name!r} is non-finite")
        tensor_count += 1
        element_count += tensor.numel()
    return (
        {
            "path": str(final_path),
            "sha256": expected_sha,
            "bytes": len(payload),
            "tensor_count": tensor_count,
            "element_count": element_count,
            "all_model_tensors_finite": True,
        },
        dict(audit),
        dict(state),
    )


def _assert_tensor_states_equal(
    torch: Any,
    left: dict[str, Any],
    right: dict[str, Any],
    label: str,
) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"{label}: tensor key sets differ")
    for name in sorted(left):
        left_value = left[name]
        right_value = right[name]
        if (
            not torch.is_tensor(left_value)
            or not torch.is_tensor(right_value)
            or left_value.dtype != right_value.dtype
            or tuple(left_value.shape) != tuple(right_value.shape)
            or not bool(
                torch.equal(
                    left_value.detach().cpu(),
                    right_value.detach().cpu(),
                )
            )
        ):
            raise RuntimeError(f"{label}: tensor {name!r} differs")


def _validate_latest_candidate_receipt(
    value: Any,
    *,
    stage: str,
    candidate: dict[str, Any],
    label: str,
    final_epoch: int | None = None,
) -> dict[str, Any]:
    if final_epoch is None:
        # Backward-compatible helper default for the mandatory frozen prefix.
        # Dynamic/continued schedules always pass their explicit boundary.
        final_epoch = contract.REQUIRED_CANDIDATE_EPOCHS[-1]
    value = contract.exact_keys(
        value,
        (
            "path",
            "sha256",
            "completed_epochs",
            "optimizer_updates",
            "selection_status",
        ),
        label,
    )
    if value != {
        "path": candidate["checkpoint"],
        "sha256": candidate["checkpoint_sha256"],
        "completed_epochs": final_epoch,
        "optimizer_updates": (
            final_epoch * contract.updates_per_epoch(stage)
        ),
        "selection_status": "offline_validation_pending",
    }:
        raise RuntimeError(f"{stage} latest representation candidate mismatch")
    return dict(value)


def _require_common_audit_binding(
    value: dict[str, Any],
    expected: dict[str, Any],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"{label}: {key} binding mismatch")


def build(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    scope = getattr(args, "stage_scope", "full")
    if scope not in {"full", "nonglobal"}:
        raise ValueError("stage scope must be full or nonglobal")
    expected_stages = (
        contract.STAGES if scope == "full" else contract.STAGES[:-1]
    )
    runs = _parse_stage_runs(args.stage_run, expected_stages)
    requested_epochs = getattr(args, "candidate_epoch", None)
    schedule = contract.validate_candidate_epochs(
        list(requested_epochs)
        if requested_epochs
        else list(contract.REQUIRED_CANDIDATE_EPOCHS)
    )
    final_epoch = schedule[-1]
    stages: dict[str, list[dict[str, Any]]] = {}
    source_receipts: dict[str, Any] = {}
    config_sha256: dict[str, str] = {}
    dataset_receipt_sha256: dict[str, str] = {}
    status_receipts: dict[str, dict[str, Any]] = {}
    for stage in expected_stages:
        run = runs[stage]
        status_path = contract.regular_file(
            run / "formal_training_status.json",
            f"{stage} formal training status",
        )
        status_bytes = status_path.read_bytes()
        status = contract.strict_json_bytes(status_bytes, str(status_path))
        if (
            not isinstance(status, dict)
            or status.get("status") != "complete"
            or status.get("formal_stage") != stage
            or status.get("completed_epochs") != final_epoch
            or status.get("updates_per_epoch")
            != contract.updates_per_epoch(stage)
            or status.get("optimizer_updates")
            != final_epoch * contract.updates_per_epoch(stage)
        ):
            raise RuntimeError(f"{stage} formal training is not complete")
        status_source = contract.validate_training_audit_source(
            status.get("source_receipt"),
            f"{stage} status training source",
            reprove_entrypoint=True,
        )
        status_source_sha = contract.canonical_payload_sha256(status_source)
        status_config_sha = contract.require_sha256(
            status.get("config_sha256"),
            f"{stage} status config SHA-256",
        )
        status_lineage_sha = contract.require_sha256(
            status.get("lineage_manifest_sha256"),
            f"{stage} status lineage SHA-256",
        )
        status_dataset_receipt = status.get("dataset_receipt")
        if (
            not isinstance(status_dataset_receipt, dict)
            or not status_dataset_receipt
        ):
            raise RuntimeError(f"{stage} status dataset receipt is missing")
        dataset_source_binding = contract.exact_keys(
            status_dataset_receipt.get("source_binding"),
            ("origin", "commit", "tree"),
            f"{stage} status dataset source binding",
        )
        if dataset_source_binding != {
            key: status_source[key]
            for key in ("origin", "commit", "tree")
        }:
            raise RuntimeError(f"{stage} dataset/source binding mismatch")
        if (
            status_dataset_receipt.get("entries") != 127_286
            or status_dataset_receipt.get("train_clips") != 13_687
            or status_dataset_receipt.get("split_label")
            != "SHOW available frozen subset"
        ):
            raise RuntimeError(f"{stage} dataset accounting mismatch")
        for key in (
            "summary_sha256",
            "lineage_sha256",
            "data_mdb_sha256",
        ):
            contract.require_sha256(
                status_dataset_receipt.get(key),
                f"{stage} status dataset {key}",
            )
        status_dataset_sha = contract.canonical_payload_sha256(
            status_dataset_receipt
        )
        status_initialization = contract.validate_initialization_receipt(
            status.get("initialization_receipt"),
            stage=stage,
            label=f"{stage} status initialization receipt",
            reprove_path=True,
        )
        status_distributed = (
            contract.validate_distributed_training_receipt(
                status.get("distributed_training_receipt"),
                stage=stage,
                label=f"{stage} status distributed training receipt",
            )
        )
        status_optimizer_runtime = (
            contract.validate_optimizer_runtime_receipt(
                status.get("optimizer_runtime_receipt"),
                stage=stage,
                label=f"{stage} status optimizer runtime receipt",
                required=stage == "global",
            )
        )
        status_rvq_prior = contract.validate_rvq_ema_prior_receipt(
            status.get("rvq_ema_prior_receipt"),
            stage=stage,
            label=f"{stage} status RVQ EMA-prior receipt",
        )
        status_rvq_rank = contract.validate_rvq_rank_state_receipt(
            status.get("rvq_rank_state_receipt"),
            stage=stage,
            distributed=status_distributed,
            label=f"{stage} status RVQ rank-state receipt",
        )
        if (
            status.get("source_receipt_sha256") != status_source_sha
        ):
            raise RuntimeError(f"{stage} status source/config binding mismatch")
        final_metrics = _finite_metrics(
            status.get("last_metrics"),
            f"{stage} status",
        )
        (
            final_checkpoint_evidence,
            final_audit,
            final_model_state,
        ) = _audit_final_checkpoint(
            torch,
            status=status,
            run=run,
            stage=stage,
            final_epoch=final_epoch,
        )
        expected_common = {
            "source_receipt": status_source,
            "source_receipt_sha256": status_source_sha,
            "config_sha256": status_config_sha,
            "lineage_manifest_sha256": status_lineage_sha,
            "dataset_receipt_sha256": status_dataset_sha,
            "initialization_receipt": status_initialization,
            "rvq_ema_prior_receipt": status_rvq_prior,
            "distributed_training_receipt": status_distributed,
            "optimizer_runtime_receipt": status_optimizer_runtime,
        }
        _require_common_audit_binding(
            final_audit,
            expected_common,
            f"{stage} final/status",
        )
        if (
            final_audit.get("rvq_rank_state_receipt") != status_rvq_rank
            or final_audit.get("dataset_summary_sha256")
            != status_dataset_receipt.get("summary_sha256")
            or final_audit.get("data_mdb_sha256")
            != status_dataset_receipt.get("data_mdb_sha256")
            or final_audit.get("smplx_asset_receipt")
            != status_dataset_receipt.get("smplx_asset")
            or status.get("smplx_asset_receipt")
            != status_dataset_receipt.get("smplx_asset")
        ):
            raise RuntimeError(
                f"{stage} final/status dataset or RVQ binding mismatch"
            )
        resume_path = contract.regular_file(
            run / "latest_resume.pt",
            f"{stage} latest resume",
        )
        resume_sha = contract.require_sha256(
            status.get("latest_resume_sha256"),
            f"{stage} latest resume SHA-256",
        )
        if contract.sha256_file(resume_path) != resume_sha:
            raise RuntimeError(f"{stage} latest resume changed")
        status_sha = contract.sha256_file(status_path)
        status_receipts[stage] = {
            "path": str(status_path),
            "sha256": status_sha,
            "finite_evidence": {
                "final_checkpoint": final_checkpoint_evidence,
                "latest_resume": {
                    "path": str(resume_path),
                    "sha256": resume_sha,
                    "bytes": resume_path.stat().st_size,
                },
                "last_metrics": final_metrics,
                "all_candidate_model_tensors_finite": True,
            },
        }
        candidate_dir = run / "representation_candidates"
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            raise RuntimeError(f"{stage} candidate directory is invalid")
        expected_paths = [
            candidate_dir
            / (
                f"{stage}_epoch_{epoch:04d}_step_"
                f"{epoch * contract.updates_per_epoch(stage):09d}.bin"
            )
            for epoch in schedule
        ]
        actual_paths = list(candidate_dir.iterdir())
        if (
            {path.name for path in actual_paths}
            != {path.name for path in expected_paths}
            or len(actual_paths) != len(expected_paths)
            or any(
                path.is_symlink() or not path.is_file()
                for path in actual_paths
            )
        ):
            raise RuntimeError(f"{stage} candidate file set is not exact")
        entries = []
        stage_audit: dict[str, Any] | None = None
        final_candidate_model_state: dict[str, Any] | None = None
        final_candidate_audit: dict[str, Any] | None = None
        for epoch, path in zip(
            schedule,
            expected_paths,
        ):
            resolved = contract.regular_file(
                path,
                f"{stage} epoch {epoch} candidate",
            )
            audit, digest, candidate_model_state = _checkpoint_audit(
                torch,
                resolved,
                stage=stage,
                epoch=epoch,
            )
            _require_common_audit_binding(
                audit,
                expected_common,
                (
                    f"{stage} epoch {epoch} "
                    "candidate/status/final"
                ),
            )
            if stage_audit is None:
                stage_audit = audit
            else:
                for key in (
                    "config_sha256",
                    "lineage_manifest_sha256",
                    "dataset_receipt_sha256",
                    "source_receipt",
                    "source_receipt_sha256",
                    "initialization_receipt",
                    "rvq_ema_prior_receipt",
                    "distributed_training_receipt",
                    "optimizer_runtime_receipt",
                ):
                    if audit.get(key) != stage_audit.get(key):
                        raise RuntimeError(
                            f"{stage} candidate audit {key} changed"
                        )
            entries.append(
                {
                    "epoch": epoch,
                    "optimizer_updates": (
                        epoch * contract.updates_per_epoch(stage)
                    ),
                    "checkpoint": str(resolved),
                    "checkpoint_sha256": digest,
                    "checkpoint_bytes": resolved.stat().st_size,
                    "checkpoint_audit_sha256": (
                        contract.canonical_payload_sha256(audit)
                    ),
                }
            )
            if epoch == final_epoch:
                final_candidate_model_state = candidate_model_state
                final_candidate_audit = audit
        assert stage_audit is not None
        assert final_candidate_model_state is not None
        assert final_candidate_audit is not None
        if stage_audit["source_receipt"] != status_source:
            raise RuntimeError(f"{stage} candidate/status source mismatch")
        final_candidate = entries[-1]
        _validate_latest_candidate_receipt(
            status.get("latest_representation_candidate"),
            stage=stage,
            candidate=final_candidate,
            final_epoch=final_epoch,
            label=f"{stage} status latest representation candidate",
        )
        _validate_latest_candidate_receipt(
            final_audit.get("latest_representation_candidate"),
            stage=stage,
            candidate=final_candidate,
            final_epoch=final_epoch,
            label=f"{stage} final latest representation candidate",
        )
        if (
            final_candidate_audit.get("rvq_rank_state_receipt")
            != status_rvq_rank
        ):
            raise RuntimeError(
                f"{stage} epoch {final_epoch}/status/final RVQ rank "
                "binding mismatch"
            )
        _assert_tensor_states_equal(
            torch,
            final_candidate_model_state,
            final_model_state,
            f"{stage} epoch {final_epoch}/final model state",
        )
        source_receipts[stage] = contract.freeze_training_audit_source(
            stage_audit["source_receipt"],
            f"{stage} training source",
        )
        config_sha256[stage] = contract.require_sha256(
            stage_audit.get("config_sha256"),
            f"{stage} config SHA-256",
        )
        dataset_receipt_sha256[stage] = contract.require_sha256(
            stage_audit.get("dataset_receipt_sha256"),
            f"{stage} dataset receipt SHA-256",
        )
        stages[stage] = entries
    rvq_source_core = {
        contract.canonical_payload_sha256(
            contract.portable_training_source_identity(
                value["portable_identity"],
                "portable training source",
            )
        )
        for stage, value in source_receipts.items()
        if stage in contract.RVQ_STAGES
    }
    if len(rvq_source_core) != 1:
        raise RuntimeError("prerequisite RVQ candidate sources differ")
    payload = contract.receipt_payload(
        {
            "format": (
                contract.CANDIDATE_INDEX_FORMAT
                if scope == "full"
                else contract.PARTIAL_CANDIDATE_INDEX_FORMAT
            ),
            "status": "complete",
            "target_dataset": "SHOW",
            "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
            "selection_split": "val",
            "test_visible": False,
            "candidate_epochs": list(schedule),
            "updates_per_epoch": contract.updates_per_epoch_map(
                expected_stages
            ),
            "source_policy": contract.build_source_policy(
                source_receipts,
                stages=expected_stages,
                reprove_ancestry=True,
            ),
            "source_receipts": source_receipts,
            "config_sha256": config_sha256,
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "formal_training_status": status_receipts,
            "stages": stages,
        }
    )
    contract.validate_candidate_index(
        payload,
        allow_partial=scope == "nonglobal",
        reprove_source_ancestry=True,
    )
    contract.atomic_json_new(args.output_json, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable five-prerequisite candidate index"
    )
    parser.add_argument(
        "--stage-scope",
        choices=("full", "nonglobal"),
        default="full",
        help="freeze all five stages or the four RVQ stages only",
    )
    parser.add_argument(
        "--stage-run",
        action="append",
        required=True,
        help="repeat for every stage in --stage-scope",
    )
    parser.add_argument(
        "--candidate-epoch",
        action="append",
        type=int,
        help=(
            "optional append-only formal inventory; repeat for every epoch. "
            "It must contain the mandatory e20..e200 prefix"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "stages": list(result["stages"]),
                "candidates_per_stage": len(
                    result["candidate_epochs"]
                ),
                "receipt_payload_sha256": result[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
