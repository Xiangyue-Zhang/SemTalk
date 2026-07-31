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


def _parse_stage_runs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("--stage-run must be STAGE=/absolute/run/path")
        stage, path_value = raw.split("=", 1)
        if stage not in contract.STAGES or stage in result:
            raise ValueError(f"invalid or duplicate stage-run {stage!r}")
        path = Path(path_value)
        if not path.is_absolute():
            raise ValueError("stage run paths must be absolute")
        contract.reject_forbidden_label(path, f"{stage} stage run")
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"invalid {stage} stage run directory")
        result[stage] = path.resolve(strict=True)
    if set(result) != set(contract.STAGES):
        raise ValueError("exactly five stage runs are required")
    return result


def _checkpoint_audit(
    torch: Any,
    path: Path,
    *,
    stage: str,
    epoch: int,
) -> tuple[dict[str, Any], str]:
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
    audit = checkpoint["audit"]
    updates = epoch * contract.EXPECTED_UPDATES_PER_EPOCH
    if (
        not isinstance(audit, dict)
        or audit.get("format") != "semtalk_show_representation_candidate_v1"
        or audit.get("formal_stage") != stage
        or audit.get("completed_epochs") != epoch
        or audit.get("optimizer_updates") != updates
        or audit.get("selection_status") != "offline_validation_pending"
    ):
        raise RuntimeError(f"{path}: candidate audit mismatch")
    source = audit.get("source_receipt")
    source = contract.validate_training_audit_source(
        source,
        f"{path} candidate source",
    )
    source_sha = hashlib.sha256(
        json.dumps(
            source,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if audit.get("source_receipt_sha256") != source_sha:
        raise RuntimeError(f"{path}: candidate source payload hash mismatch")
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
    return audit, contract.sha256_file(path)


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
    training_source: dict[str, Any],
) -> dict[str, Any]:
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
        != 200 * contract.EXPECTED_UPDATES_PER_EPOCH
        or audit.get("source_receipt") != training_source
        or audit.get("config_sha256") != status.get("config_sha256")
    ):
        raise RuntimeError(f"{stage} final checkpoint audit mismatch")
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
    return {
        "path": str(final_path),
        "sha256": expected_sha,
        "bytes": len(payload),
        "tensor_count": tensor_count,
        "element_count": element_count,
        "all_model_tensors_finite": True,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    runs = _parse_stage_runs(args.stage_run)
    stages: dict[str, list[dict[str, Any]]] = {}
    source_receipts: dict[str, Any] = {}
    config_sha256: dict[str, str] = {}
    dataset_receipt_sha256: dict[str, str] = {}
    status_receipts: dict[str, dict[str, Any]] = {}
    for stage in contract.STAGES:
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
            or status.get("completed_epochs") != 200
            or status.get("updates_per_epoch")
            != contract.EXPECTED_UPDATES_PER_EPOCH
            or status.get("optimizer_updates")
            != 200 * contract.EXPECTED_UPDATES_PER_EPOCH
        ):
            raise RuntimeError(f"{stage} formal training is not complete")
        status_source = contract.validate_training_audit_source(
            status.get("source_receipt"),
            f"{stage} status training source",
        )
        status_source_sha = hashlib.sha256(
            json.dumps(
                status_source,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if (
            status.get("source_receipt_sha256") != status_source_sha
            or contract.require_sha256(
                status.get("config_sha256"),
                f"{stage} status config SHA-256",
            )
            != status.get("config_sha256")
            or contract.require_sha256(
                status.get("lineage_manifest_sha256"),
                f"{stage} status lineage SHA-256",
            )
            != status.get("lineage_manifest_sha256")
        ):
            raise RuntimeError(f"{stage} status source/config binding mismatch")
        final_metrics = _finite_metrics(
            status.get("last_metrics"),
            f"{stage} status",
        )
        final_checkpoint_evidence = _audit_final_checkpoint(
            torch,
            status=status,
            run=run,
            stage=stage,
            training_source=status_source,
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
                f"{epoch * contract.EXPECTED_UPDATES_PER_EPOCH:09d}.bin"
            )
            for epoch in contract.EXPECTED_CANDIDATE_EPOCHS
        ]
        actual = {path.resolve() for path in candidate_dir.iterdir()}
        if actual != {path.resolve() for path in expected_paths}:
            raise RuntimeError(f"{stage} candidate file set is not exact")
        entries = []
        stage_audit: dict[str, Any] | None = None
        for epoch, path in zip(
            contract.EXPECTED_CANDIDATE_EPOCHS,
            expected_paths,
        ):
            resolved = contract.regular_file(
                path,
                f"{stage} epoch {epoch} candidate",
            )
            audit, digest = _checkpoint_audit(
                torch,
                resolved,
                stage=stage,
                epoch=epoch,
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
                    "distributed_training_receipt",
                ):
                    if audit.get(key) != stage_audit.get(key):
                        raise RuntimeError(
                            f"{stage} candidate audit {key} changed"
                        )
            entries.append(
                {
                    "epoch": epoch,
                    "optimizer_updates": (
                        epoch * contract.EXPECTED_UPDATES_PER_EPOCH
                    ),
                    "checkpoint": str(resolved),
                    "checkpoint_sha256": digest,
                    "checkpoint_bytes": resolved.stat().st_size,
                    "checkpoint_audit_sha256": (
                        contract.canonical_payload_sha256(audit)
                    ),
                }
            )
        assert stage_audit is not None
        if stage_audit["source_receipt"] != status_source:
            raise RuntimeError(f"{stage} candidate/status source mismatch")
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
    source_core = {
        (
            value["origin"],
            value["commit"],
            value["tree"],
            value["clean"],
            value["detached"],
            value["local_branch_count"],
        )
        for value in source_receipts.values()
    }
    if len(source_core) != 1:
        raise RuntimeError("five prerequisite candidate sources differ")
    payload = contract.receipt_payload(
        {
            "format": contract.CANDIDATE_INDEX_FORMAT,
            "status": "complete",
            "target_dataset": "SHOW",
            "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
            "selection_split": "val",
            "test_visible": False,
            "candidate_epochs": list(contract.EXPECTED_CANDIDATE_EPOCHS),
            "updates_per_epoch": contract.EXPECTED_UPDATES_PER_EPOCH,
            "source_receipts": source_receipts,
            "config_sha256": config_sha256,
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "formal_training_status": status_receipts,
            "stages": stages,
        }
    )
    contract.validate_candidate_index(payload)
    contract.atomic_json_new(args.output_json, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable five-prerequisite candidate index"
    )
    parser.add_argument(
        "--stage-run",
        action="append",
        required=True,
        help="repeat exactly five times as STAGE=/absolute/run/path",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "stages": list(contract.STAGES),
                "candidates_per_stage": len(
                    contract.EXPECTED_CANDIDATE_EPOCHS
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
