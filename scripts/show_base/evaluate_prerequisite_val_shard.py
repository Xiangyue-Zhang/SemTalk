#!/usr/bin/env python3
"""Evaluate one prerequisite candidate on one frozen SHOW validation shard.

This is the only GPU entrypoint in the prerequisite-selection chain.  It
evaluates one stage independently, emits mergeable sufficient statistics, and
never opens a test-labelled input.  Eight invocations with shard indices 0..7
cover each of the 1,715 validation clips exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import gate_released_all_speakers_on_show as gate
from scripts.show_base import prerequisite_val_contract as contract


def _update_accumulator(accumulator: dict[str, Any], difference: Any) -> None:
    value = difference.detach().double()
    if not bool(value.isfinite().all().item()):
        raise RuntimeError("validation difference contains NaN/Inf")
    absolute = value.abs()
    accumulator["count"] += int(value.numel())
    accumulator["sum_abs"] += float(absolute.sum().item())
    accumulator["sum_squared"] += float(value.square().sum().item())
    accumulator["max_abs"] = max(
        accumulator["max_abs"],
        float(absolute.max().item()),
    )


def _fresh_accumulators(stage: str) -> dict[str, dict[str, Any]]:
    return {
        name: contract.error_accumulator()
        for name in contract.ACCUMULATOR_KEYS[stage]
    }


def _checkpoint_payload(
    torch: Any,
    path: Path,
    expected_sha256: str,
    *,
    stage: str,
    epoch: int,
    optimizer_updates: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved, payload, observed_sha = contract.read_verified_file(
        path,
        expected_sha256,
        f"{stage} candidate checkpoint",
    )
    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model_state",
        "audit",
    }:
        raise RuntimeError("candidate checkpoint container schema mismatch")
    audit = contract.validate_representation_candidate_audit(
        checkpoint["audit"],
        stage=stage,
        epoch=epoch,
        label="candidate checkpoint audit",
        reprove_paths=False,
    )
    state: dict[str, Any] = {}
    raw_state = checkpoint["model_state"]
    if not isinstance(raw_state, dict) or not raw_state:
        raise RuntimeError("candidate model_state is empty")
    for raw_name, value in raw_state.items():
        if not isinstance(raw_name, str) or not torch.is_tensor(value):
            raise RuntimeError("candidate model_state entry is invalid")
        name = raw_name[7:] if raw_name.startswith("module.") else raw_name
        if name in state:
            raise RuntimeError("candidate normalized state key is duplicated")
        if (
            (value.is_floating_point() or value.is_complex())
            and not bool(value.isfinite().all().item())
        ):
            raise RuntimeError(f"candidate tensor {name!r} contains NaN/Inf")
        state[name] = value
    return state, audit, {
        "path": str(resolved),
        "sha256": observed_sha,
        "bytes": len(payload),
    }


def _load_model(
    torch: Any,
    *,
    stage: str,
    state: Mapping[str, Any],
    device: Any,
) -> Any:
    from models.motion_representation import VAEConvZero
    from models.rvq import RVQVAE

    spec = gate.WEIGHT_SPECS[stage]
    namespace = SimpleNamespace(
        vae_test_dim=spec["dimension"],
        vae_layer=spec["vae_layer"],
        vae_length=256,
    )
    model = (
        RVQVAE(namespace)
        if stage in contract.RVQ_STAGES
        else VAEConvZero(namespace)
    )
    expected = set(model.state_dict())
    if expected != set(state):
        raise RuntimeError(
            f"{stage} state schema mismatch; "
            f"missing={sorted(expected - set(state))[:3]} "
            f"extra={sorted(set(state) - expected)[:3]}"
        )
    model.load_state_dict(dict(state), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _rvq_metrics(
    torch: Any,
    rc: Any,
    *,
    stage: str,
    model: Any,
    features: Mapping[str, Any],
    accumulators: dict[str, dict[str, Any]],
    histograms: list[list[int]],
) -> tuple[Any, Any]:
    value = features[stage]
    indices, reconstruction = gate._decode_rvq_checked(
        torch,
        model,
        value,
        stage,
    )
    rotation_width = {
        "face": 6,
        "hands": 180,
        "upper": 78,
        "lower": 54,
    }[stage]
    target_rotation = rc.rotation_6d_to_matrix(
        value[..., :rotation_width].reshape(
            *value.shape[:2],
            rotation_width // 6,
            6,
        )
    )
    reconstructed_rotation = rc.rotation_6d_to_matrix(
        reconstruction[..., :rotation_width].reshape(
            *reconstruction.shape[:2],
            rotation_width // 6,
            6,
        )
    )
    _update_accumulator(
        accumulators["rotation_geodesic"],
        gate._rotation_geodesic_error(
            torch,
            rc,
            value[..., :rotation_width].reshape(
                *value.shape[:2],
                rotation_width // 6,
                6,
            ),
            reconstruction[..., :rotation_width].reshape(
                *reconstruction.shape[:2],
                rotation_width // 6,
                6,
            ),
        ),
    )
    _update_accumulator(
        accumulators["rotation_velocity"],
        (reconstructed_rotation[:, 1:] - reconstructed_rotation[:, :-1])
        - (target_rotation[:, 1:] - target_rotation[:, :-1]),
    )
    _update_accumulator(
        accumulators["rotation_acceleration"],
        (
            reconstructed_rotation[:, 2:]
            + reconstructed_rotation[:, :-2]
            - 2.0 * reconstructed_rotation[:, 1:-1]
        )
        - (
            target_rotation[:, 2:]
            + target_rotation[:, :-2]
            - 2.0 * target_rotation[:, 1:-1]
        ),
    )
    if stage == "face":
        target = value[..., 6:]
        reconstructed = reconstruction[..., 6:]
        _update_accumulator(accumulators["expression"], reconstructed - target)
        _update_accumulator(
            accumulators["expression_velocity"],
            (reconstructed[:, 1:] - reconstructed[:, :-1])
            - (target[:, 1:] - target[:, :-1]),
        )
        _update_accumulator(
            accumulators["expression_acceleration"],
            (
                reconstructed[:, 2:]
                + reconstructed[:, :-2]
                - 2.0 * reconstructed[:, 1:-1]
            )
            - (
                target[:, 2:]
                + target[:, :-2]
                - 2.0 * target[:, 1:-1]
            ),
        )
    elif stage == "lower":
        _update_accumulator(
            accumulators["translation"],
            reconstruction[..., 54:57] - value[..., 54:57],
        )
        _update_accumulator(
            accumulators["contact"],
            reconstruction[..., 57:61] - value[..., 57:61],
        )
    flattened = indices.detach().cpu().reshape(-1, gate.RVQ_LEVELS)
    for level in range(gate.RVQ_LEVELS):
        counts = torch.bincount(
            flattened[:, level],
            minlength=gate.CODEBOOK_SIZE,
        ).tolist()
        for index, count in enumerate(counts):
            histograms[level][index] += int(count)
    return indices, reconstruction


def _global_metrics(
    torch: Any,
    *,
    model: Any,
    features: Mapping[str, Any],
    accumulators: dict[str, dict[str, Any]],
) -> Any:
    model_input = features["lower"].clone()
    model_input[..., 54:61] = 0.0
    output = model(model_input).get("rec_pose")
    if (
        output is None
        or tuple(output.shape) != tuple(model_input.shape)
        or not bool(output.isfinite().all().item())
    ):
        raise RuntimeError("global candidate output is invalid")
    target_translation = features["translation"]
    target_contact = features["contact"]
    target_velocity = gate._central_velocity(torch, target_translation)
    predicted_channels = output[..., 54:57]
    reconstructed_translation = gate._integrate_xz(
        torch,
        predicted_channels,
        target_translation[:, 0],
    )
    _update_accumulator(
        accumulators["contact"],
        output[..., 57:61] - target_contact,
    )
    _update_accumulator(
        accumulators["velocity_x"],
        predicted_channels[..., 0] - target_velocity[..., 0],
    )
    _update_accumulator(
        accumulators["velocity_z"],
        predicted_channels[..., 2] - target_velocity[..., 2],
    )
    for name, channel in (("x", 0), ("z", 2)):
        prediction = predicted_channels[..., channel]
        target = target_velocity[..., channel]
        _update_accumulator(
            accumulators[f"velocity_delta_{name}"],
            (prediction[:, 1:] - prediction[:, :-1])
            - (target[:, 1:] - target[:, :-1]),
        )
        _update_accumulator(
            accumulators[f"velocity_acceleration_{name}"],
            (
                prediction[:, 2:]
                + prediction[:, :-2]
                - 2.0 * prediction[:, 1:-1]
            )
            - (
                target[:, 2:]
                + target[:, :-2]
                - 2.0 * target[:, 1:-1]
            ),
        )
    _update_accumulator(
        accumulators["integrated_translation_velocity"],
        (
            reconstructed_translation[:, 1:]
            - reconstructed_translation[:, :-1]
        )
        - (target_translation[:, 1:] - target_translation[:, :-1]),
    )
    _update_accumulator(
        accumulators["integrated_translation_acceleration"],
        (
            reconstructed_translation[:, 2:]
            + reconstructed_translation[:, :-2]
            - 2.0 * reconstructed_translation[:, 1:-1]
        )
        - (
            target_translation[:, 2:]
            + target_translation[:, :-2]
            - 2.0 * target_translation[:, 1:-1]
        ),
    )
    _update_accumulator(
        accumulators["integrated_translation"],
        reconstructed_translation - target_translation,
    )
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from utils import rotation_conversions as rc

    if args.stage not in contract.STAGES:
        raise ValueError("--stage is invalid")
    if args.epoch not in contract.EXPECTED_CANDIDATE_EPOCHS:
        raise ValueError("--epoch is outside the frozen schedule")
    if (
        args.optimizer_updates
        != args.epoch * contract.EXPECTED_UPDATES_PER_EPOCH
    ):
        raise ValueError("--optimizer-updates is inconsistent")
    if (
        args.shard_count != contract.EXPECTED_SHARDS
        or not 0 <= args.shard_index < args.shard_count
        or args.batch_size <= 0
    ):
        raise ValueError("invalid shard or batch configuration")

    source = gate.git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
        script=Path(__file__).resolve(),
    )
    rows, canonical_receipt = contract.load_val_canonical(
        manifest_path=args.canonical_manifest,
        manifest_sha256=args.expected_manifest_sha256,
        summary_path=args.canonical_summary,
        summary_sha256=args.expected_summary_sha256,
        lineage_path=args.canonical_lineage,
        lineage_sha256=args.expected_lineage_sha256,
    )
    shard_rows = [
        row
        for row in rows
        if int(row["global_index"]) % args.shard_count == args.shard_index
    ]
    if not shard_rows:
        raise RuntimeError("validation shard is empty")
    expected_clip_ids = [str(row["clip_id"]) for row in shard_rows]
    state, audit, checkpoint_receipt = _checkpoint_payload(
        torch,
        args.checkpoint,
        args.expected_checkpoint_sha256,
        stage=args.stage,
        epoch=args.epoch,
        optimizer_updates=args.optimizer_updates,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal prerequisite validation requires CUDA")
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError("unsupported deterministic CUBLAS workspace")
    if device.index is not None:
        torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model = _load_model(
        torch,
        stage=args.stage,
        state=state,
        device=device,
    )
    accumulators = _fresh_accumulators(args.stage)
    histograms = (
        [
            [0 for _ in range(gate.CODEBOOK_SIZE)]
            for _ in range(gate.RVQ_LEVELS)
        ]
        if args.stage in contract.RVQ_STAGES
        else None
    )
    seen: set[tuple[int, int]] = set()
    records_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    batches = 0
    replay_checked = False
    with torch.inference_mode():
        for records, batch in gate._batch_iterator(
            np,
            shard_rows,
            args.batch_size,
        ):
            batches += 1
            for record in records:
                key = (int(record["global_index"]), int(record["start"]))
                if key in seen:
                    raise RuntimeError("duplicate window inside shard")
                seen.add(key)
                records_digest.update(gate.canonical_json_bytes(record))
            features = gate._build_features(torch, rc, batch, device)
            if args.stage in contract.RVQ_STAGES:
                assert histograms is not None
                indices, output = _rvq_metrics(
                    torch,
                    rc,
                    stage=args.stage,
                    model=model,
                    features=features,
                    accumulators=accumulators,
                    histograms=histograms,
                )
                output_digest.update(
                    indices.detach().cpu().contiguous().numpy().tobytes()
                )
                if not replay_checked:
                    replay_indices, replay_output = gate._decode_rvq_checked(
                        torch,
                        model,
                        features[args.stage],
                        args.stage,
                    )
                    if not torch.equal(indices, replay_indices) or not torch.equal(
                        output,
                        replay_output,
                    ):
                        raise RuntimeError("RVQ deterministic replay mismatch")
                    replay_checked = True
            else:
                output = _global_metrics(
                    torch,
                    model=model,
                    features=features,
                    accumulators=accumulators,
                )
                output_digest.update(
                    output.detach().cpu().contiguous().numpy().tobytes()
                )
                if not replay_checked:
                    model_input = features["lower"].clone()
                    model_input[..., 54:61] = 0.0
                    replay = model(model_input).get("rec_pose")
                    if replay is None or not torch.equal(output, replay):
                        raise RuntimeError("Global deterministic replay mismatch")
                    replay_checked = True
    expected_windows = sum(gate.window_count(row["frames"]) for row in shard_rows)
    if len(seen) != expected_windows or batches <= 0 or not replay_checked:
        raise RuntimeError("validation shard window coverage mismatch")
    for name, value in accumulators.items():
        contract.validate_accumulator(value, f"{args.stage}.{name}")

    result = contract.receipt_payload(
        {
            "format": contract.SHARD_FORMAT,
            "status": "complete",
            "stage": args.stage,
            "split": "val",
            "test_visible": False,
            "epoch": args.epoch,
            "optimizer_updates": args.optimizer_updates,
            "checkpoint": checkpoint_receipt,
            "checkpoint_audit_sha256": contract.canonical_payload_sha256(audit),
            "canonical_receipt": canonical_receipt,
            "producer_source": source,
            "protocol": {
                "name": contract.SELECTION_METRICS[args.stage],
                "stage_independent": True,
                "candidate_variable_only": True,
                "window_length": gate.WINDOW_LENGTH,
                "window_stride": gate.WINDOW_STRIDE,
                "selection_split": "val",
                "test_visible": False,
                "full_base_fgd_used": False,
                "inference_only_exclusions": [
                    "quantizer_embedding_loss",
                    "smplx_vertex_loss",
                ],
            },
            "shard": {
                "index": args.shard_index,
                "count": args.shard_count,
                "clip_count": len(shard_rows),
                "clip_ids": expected_clip_ids,
                "clip_ids_sha256": contract.canonical_payload_sha256(
                    expected_clip_ids
                ),
                "window_count": expected_windows,
                "window_records_sha256": records_digest.hexdigest(),
            },
            "finite": True,
            "exact_once_within_shard": True,
            "determinism": {
                "seed": args.seed,
                "torch_deterministic_algorithms": True,
                "tf32": False,
                "first_batch_exact_replay": True,
                "output_digest_sha256": output_digest.hexdigest(),
            },
            "accumulators": accumulators,
            "codebook_histograms": histograms,
            "runtime": {
                "python": os.sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "batch_size": args.batch_size,
                "batches": batches,
            },
        }
    )
    contract.atomic_json_new(args.output_json, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one SHOW prerequisite validation shard"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--optimizer-updates", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "epoch": result["epoch"],
                "shard": result["shard"]["index"],
                "clips": result["shard"]["clip_count"],
                "windows": result["shard"]["window_count"],
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
