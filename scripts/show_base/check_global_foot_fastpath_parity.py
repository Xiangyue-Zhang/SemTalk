#!/usr/bin/env python3
"""Strict legacy-vs-fast parity gate for the SHOW Global VAE geometry loss.

This gate intentionally runs the expensive legacy SMPL-X path once.  It checks
the immutable lower-foot sidecar, every replaced loss component, gradients, and
one Adam update against the algebraic fast path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base.build_show_cache import (  # noqa: E402
    FOOT_JOINTS,
    GLOBAL_FOOT_FASTPATH_CONTRACT,
    LOWER_JOINT_INDICES,
    resolve_smplx_asset,
)
from utils import rotation_conversions as rc  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(
    label: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    left = left.detach()
    right = right.detach()
    if left.shape != right.shape:
        raise AssertionError(f"{label}: shape {left.shape} != {right.shape}")
    delta = (left - right).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    denominator = torch.maximum(left.abs(), right.abs()).clamp_min(1e-12)
    max_rel = (
        float((delta / denominator).max().item()) if delta.numel() else 0.0
    )
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{label}: max_abs={max_abs:.9g}, max_rel={max_rel:.9g}, "
            f"atol={atol}, rtol={rtol}"
        )
    return {"max_abs": max_abs, "max_rel": max_rel}


def trainer_lower_only_pose(pose: torch.Tensor) -> torch.Tensor:
    selected = pose.reshape(-1, 55, 3)[:, LOWER_JOINT_INDICES, :]
    selected = rc.axis_angle_to_matrix(selected)
    selected = rc.matrix_to_rotation_6d(selected)
    selected = rc.rotation_6d_to_matrix(selected)
    selected = rc.matrix_to_axis_angle(selected)
    result = torch.zeros_like(pose).reshape(-1, 55, 3)
    result[:, LOWER_JOINT_INDICES, :] = selected
    return result.reshape(-1, 165)


def smplx_forward(
    model: torch.nn.Module,
    pose: torch.Tensor,
    beta: torch.Tensor,
    trans: torch.Tensor,
) -> Any:
    zeros_expression = torch.zeros(
        (pose.shape[0], 100), dtype=pose.dtype, device=pose.device
    )
    return model(
        betas=beta,
        transl=trans,
        expression=zeros_expression,
        global_orient=pose[:, 0:3],
        body_pose=pose[:, 3:66],
        jaw_pose=pose[:, 66:69],
        leye_pose=pose[:, 69:72],
        reye_pose=pose[:, 72:75],
        left_hand_pose=pose[:, 75:120],
        right_hand_pose=pose[:, 120:165],
        return_verts=True,
        return_joints=True,
    )


def foot_loss(
    feet: torch.Tensor,
    contact: torch.Tensor,
) -> torch.Tensor:
    velocity = torch.zeros_like(feet)
    velocity[:, :-1] = feet[:, 1:] - feet[:, :-1]
    velocity[~(contact > 0.95)] = 0
    return F.l1_loss(velocity, torch.zeros_like(velocity))


def geometry_legacy(
    model: torch.nn.Module,
    lower_pose: torch.Tensor,
    beta: torch.Tensor,
    target_trans: torch.Tensor,
    state: torch.Tensor,
    *,
    rec_weight: float,
    rec_pos_weight: float,
    rec_ver_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    frames = target_trans.shape[1]
    rec_trans = state[:, :, :3]
    contact = state[:, :, 3:7]
    rec = smplx_forward(
        model,
        lower_pose,
        beta,
        rec_trans.reshape(frames, 3),
    )
    with torch.no_grad():
        target = smplx_forward(
            model,
            lower_pose,
            beta,
            target_trans.reshape(frames, 3),
        )
    vertex = F.mse_loss(rec.vertices, target.vertices)
    vertex_velocity = F.mse_loss(
        rec.vertices[:, 1:] - rec.vertices[:, :-1],
        target.vertices[:, 1:] - target.vertices[:, :-1],
    )
    vertex_acceleration = F.mse_loss(
        rec.vertices[:, 2:]
        + rec.vertices[:, :-2]
        - 2 * rec.vertices[:, 1:-1],
        target.vertices[:, 2:]
        + target.vertices[:, :-2]
        - 2 * target.vertices[:, 1:-1],
    )
    feet = rec.joints[:, FOOT_JOINTS, :].reshape(1, frames, 4, 3)
    foot = foot_loss(feet, contact)
    target_contact = (contact.detach() > 0.95).to(contact.dtype)
    contact_mse = F.mse_loss(contact, target_contact)
    total = (
        contact_mse * rec_weight * rec_pos_weight
        + (vertex + 5 * vertex_velocity + 5 * vertex_acceleration)
        * rec_weight
        * rec_ver_weight
        + foot * rec_weight * rec_ver_weight * 20
    )
    return total, {
        "contact": contact_mse,
        "vertex": vertex,
        "vertex_velocity_wrong_axis": vertex_velocity,
        "vertex_acceleration_wrong_axis": vertex_acceleration,
        "foot": foot,
        "total": total,
    }


def geometry_fast(
    lower_foot_local: torch.Tensor,
    target_trans: torch.Tensor,
    state: torch.Tensor,
    *,
    rec_weight: float,
    rec_pos_weight: float,
    rec_ver_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rec_trans = state[:, :, :3]
    contact = state[:, :, 3:7]
    vertex = F.mse_loss(rec_trans, target_trans)
    graph_zero = rec_trans.sum() * 0.0
    feet = lower_foot_local + rec_trans.unsqueeze(2)
    foot = foot_loss(feet, contact)
    target_contact = (contact.detach() > 0.95).to(contact.dtype)
    contact_mse = F.mse_loss(contact, target_contact)
    total = (
        contact_mse * rec_weight * rec_pos_weight
        + (vertex + 5 * graph_zero + 5 * graph_zero)
        * rec_weight
        * rec_ver_weight
        + foot * rec_weight * rec_ver_weight * 20
    )
    return total, {
        "contact": contact_mse,
        "vertex": vertex,
        "vertex_velocity_wrong_axis": graph_zero,
        "vertex_acceleration_wrong_axis": graph_zero,
        "foot": foot,
        "total": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-npz", type=Path, required=True)
    parser.add_argument("--lower-foot-local", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--loss-atol", type=float, default=2e-6)
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--gradient-atol", type=float, default=2e-6)
    parser.add_argument("--gradient-rtol", type=float, default=1e-5)
    parser.add_argument("--cache-atol", type=float, default=1e-6)
    parser.add_argument("--cache-rtol", type=float, default=1e-6)
    parser.add_argument("--wrong-axis-zero-atol", type=float, default=2e-7)
    # create_optimizer scales the released prerequisite lr_base=3e-4 by
    # batch_size/128 = 64/128, so the effective formal Adam LR is 1.5e-4.
    parser.add_argument("--adam-lr", type=float, default=1.5e-4)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 3 or args.start_frame < 0:
        raise ValueError("--frames must be >=3 and --start-frame must be >=0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal parity gate requires CUDA")
    for path, label in (
        (args.canonical_npz, "canonical NPZ"),
        (args.lower_foot_local, "lower-foot sidecar"),
        (args.smplx_model, "SMPL-X input"),
    ):
        if path.is_symlink():
            raise RuntimeError(f"{label} must not be a symlink: {path}")
    canonical = args.canonical_npz.resolve()
    foot_path = args.lower_foot_local.resolve()
    with np.load(canonical, allow_pickle=False) as archive:
        pose_np = np.asarray(archive["pose"], dtype=np.float32)
        beta_np = np.asarray(archive["beta"], dtype=np.float32)
        trans_np = np.asarray(archive["trans"], dtype=np.float32)
        contact_np = np.asarray(archive["contact"], dtype=np.float32)
    foot_np = np.load(foot_path, allow_pickle=False)
    end = args.start_frame + args.frames
    if end > pose_np.shape[0] or foot_np.shape != (pose_np.shape[0], 4, 3):
        raise RuntimeError("requested parity window or foot sidecar shape is invalid")

    pose = torch.from_numpy(pose_np[args.start_frame:end]).to(device)
    beta = torch.from_numpy(beta_np[args.start_frame:end]).to(device)
    target_trans = torch.from_numpy(
        trans_np[args.start_frame:end]
    ).to(device).unsqueeze(0)
    lower_foot_local = torch.from_numpy(
        np.asarray(foot_np[args.start_frame:end], dtype=np.float32)
    ).to(device).unsqueeze(0)
    lower_pose = trainer_lower_only_pose(pose)

    _, model_root, gender = resolve_smplx_asset(args.smplx_model)
    import smplx

    model = smplx.create(
        model_path=str(model_root),
        model_type="smplx",
        gender=gender,
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        dtype=torch.float32,
    ).to(device).eval()
    model.requires_grad_(False)

    with torch.no_grad():
        local_output = smplx_forward(
            model,
            lower_pose,
            beta,
            torch.zeros((args.frames, 3), dtype=torch.float32, device=device),
        )
        observed_local = local_output.joints[
            :, FOOT_JOINTS, :
        ].reshape(1, args.frames, 4, 3)
    checks: dict[str, Any] = {
        "cache": assert_close(
            "lower_foot_local",
            observed_local,
            lower_foot_local,
            atol=args.cache_atol,
            rtol=args.cache_rtol,
        )
    }

    frame_axis = torch.arange(
        args.frames, dtype=torch.float32, device=device
    ).reshape(1, args.frames, 1)
    translation_offset = torch.cat(
        (
            torch.sin(frame_axis * 0.17) * 0.015,
            torch.cos(frame_axis * 0.11) * 0.008,
            torch.sin(frame_axis * 0.07 + 0.3) * 0.012,
        ),
        dim=-1,
    )
    target_contact = torch.from_numpy(
        contact_np[args.start_frame:end]
    ).to(device).unsqueeze(0)
    initial_contact = torch.where(
        target_contact > 0.5,
        torch.full_like(target_contact, 0.97),
        torch.full_like(target_contact, 0.93),
    )
    initial = torch.cat(
        (target_trans + translation_offset, initial_contact), dim=-1
    )
    legacy_state = torch.nn.Parameter(initial.clone())
    fast_state = torch.nn.Parameter(initial.clone())
    legacy_optimizer = torch.optim.Adam(
        [legacy_state], lr=args.adam_lr, betas=(0.5, 0.999)
    )
    fast_optimizer = torch.optim.Adam(
        [fast_state], lr=args.adam_lr, betas=(0.5, 0.999)
    )

    legacy_total, legacy_parts = geometry_legacy(
        model,
        lower_pose,
        beta,
        target_trans,
        legacy_state,
        rec_weight=1.0,
        rec_pos_weight=1.0,
        rec_ver_weight=1.0,
    )
    fast_total, fast_parts = geometry_fast(
        lower_foot_local,
        target_trans,
        fast_state,
        rec_weight=1.0,
        rec_pos_weight=1.0,
        rec_ver_weight=1.0,
    )
    legacy_total.backward()
    fast_total.backward()

    wrong_axis = max(
        float(legacy_parts["vertex_velocity_wrong_axis"].item()),
        float(legacy_parts["vertex_acceleration_wrong_axis"].item()),
    )
    if wrong_axis > args.wrong_axis_zero_atol:
        raise AssertionError(
            "legacy wrong-axis vertex derivative is not numerically zero: "
            f"{wrong_axis} > {args.wrong_axis_zero_atol}"
        )
    checks["losses"] = {
        name: assert_close(
            name,
            legacy_parts[name],
            fast_parts[name],
            atol=args.loss_atol,
            rtol=args.loss_rtol,
        )
        for name in legacy_parts
    }
    checks["gradient"] = assert_close(
        "state_gradient",
        legacy_state.grad,
        fast_state.grad,
        atol=args.gradient_atol,
        rtol=args.gradient_rtol,
    )

    legacy_optimizer.step()
    fast_optimizer.step()
    checks["adam_parameter"] = assert_close(
        "adam_parameter",
        legacy_state,
        fast_state,
        atol=args.gradient_atol,
        rtol=args.gradient_rtol,
    )
    legacy_adam = legacy_optimizer.state[legacy_state]
    fast_adam = fast_optimizer.state[fast_state]
    checks["adam_exp_avg"] = assert_close(
        "adam_exp_avg",
        legacy_adam["exp_avg"],
        fast_adam["exp_avg"],
        atol=args.gradient_atol,
        rtol=args.gradient_rtol,
    )
    checks["adam_exp_avg_sq"] = assert_close(
        "adam_exp_avg_sq",
        legacy_adam["exp_avg_sq"],
        fast_adam["exp_avg_sq"],
        atol=args.gradient_atol,
        rtol=args.gradient_rtol,
    )
    if float(legacy_adam["step"].item()) != float(fast_adam["step"].item()):
        raise AssertionError("Adam step counter mismatch")

    report = {
        "status": "pass",
        "contract": GLOBAL_FOOT_FASTPATH_CONTRACT,
        "canonical_npz": str(canonical),
        "canonical_npz_sha256": sha256(canonical),
        "lower_foot_local": str(foot_path),
        "lower_foot_local_sha256": sha256(foot_path),
        "smplx_asset": str(resolve_smplx_asset(args.smplx_model)[0]),
        "smplx_asset_sha256": sha256(
            resolve_smplx_asset(args.smplx_model)[0]
        ),
        "window": {
            "start_frame": args.start_frame,
            "frames": args.frames,
        },
        "thresholds": {
            "contact": 0.95,
            "loss_atol": args.loss_atol,
            "loss_rtol": args.loss_rtol,
            "gradient_atol": args.gradient_atol,
            "gradient_rtol": args.gradient_rtol,
            "cache_atol": args.cache_atol,
            "cache_rtol": args.cache_rtol,
            "wrong_axis_zero_atol": args.wrong_axis_zero_atol,
        },
        "optimizer": {
            "name": "Adam",
            "lr": args.adam_lr,
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "steps": 1,
        },
        "legacy_wrong_axis_max_loss": wrong_axis,
        "checks": checks,
        "completed_unix": time.time(),
        "argv": sys.argv,
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report_json is not None:
        destination = args.report_json.resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}"
        )
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    print(payload, end="")


if __name__ == "__main__":
    main()
