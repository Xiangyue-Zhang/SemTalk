"""Contracts and tensor operations for official-initialized SHOW transfer.

This module is intentionally independent from :mod:`show_base_train`.  The
formal scratch-training entrypoint keeps its original contract; transfer
learning has a separate, fail-closed trust root.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


TRANSFER_FORMAT = "semtalk_show_official_transfer_v1"
CACHE_FORMAT = "semtalk_show_official_transfer_cache_v1"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
FPS = 30
WINDOW_LENGTH = 64
WINDOW_STRIDE = 20
PRE_FRAMES = 4
RVQ_LEVELS = 6
CODEBOOK_SIZE = 256

OFFICIAL_CHECKPOINTS: dict[str, dict[str, str]] = {
    "face": {
        "filename": "rvq_face_600.bin",
        "sha256": (
            "31b04c88456a25f4d57841c0cb507b4c856daccb3875878d06545110a6152127"
        ),
    },
    "hands": {
        "filename": "rvq_hands_500.bin",
        "sha256": (
            "08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a539b03e436"
        ),
    },
    "upper": {
        "filename": "rvq_upper_500.bin",
        "sha256": (
            "05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d88d08ac17"
        ),
    },
    "lower": {
        "filename": "rvq_lower_600.bin",
        "sha256": (
            "2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c171ae4efce8"
        ),
    },
    "global": {
        "filename": "last_1700_foot.bin",
        "sha256": (
            "6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e127e19e0144ee12"
        ),
    },
}

_E30_PATTERN = re.compile(
    r"(?:^|[/_.-])(?:e|epoch)[_-]?0*30(?:$|[/_.-])",
    flags=re.IGNORECASE,
)


def canonical_json_bytes(value: Any) -> bytes:
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


def canonical_payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def reject_e30_reference(value: str | Path, label: str) -> None:
    """Reject every known spelling of the withdrawn epoch-30 lineage.

    Exact official SHA allowlisting below is the primary trust boundary.  This
    explicit path check additionally ensures that an official payload copied
    underneath a withdrawn/e30 directory cannot accidentally inherit that
    lineage.
    """

    text = str(value)
    lowered = text.lower()
    if "e30" in lowered or _E30_PATTERN.search(text):
        raise RuntimeError(f"{label} references forbidden withdrawn e30 lineage: {text}")


def validate_official_digest(stage: str, actual_sha256: str) -> dict[str, str]:
    if stage not in OFFICIAL_CHECKPOINTS:
        raise ValueError(f"unsupported official checkpoint stage {stage!r}")
    require_sha256(actual_sha256, f"{stage} checkpoint SHA-256")
    expected = OFFICIAL_CHECKPOINTS[stage]
    if actual_sha256 != expected["sha256"]:
        raise RuntimeError(
            f"{stage} checkpoint is not the official All-Speakers payload: "
            f"{actual_sha256} != {expected['sha256']}"
        )
    return dict(expected)


def verify_official_checkpoint(stage: str, path: Path) -> dict[str, str]:
    reject_e30_reference(path, f"{stage} checkpoint")
    if path.is_symlink():
        raise RuntimeError(f"{stage} checkpoint must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    spec = OFFICIAL_CHECKPOINTS.get(stage)
    if spec is None:
        raise ValueError(f"unsupported official checkpoint stage {stage!r}")
    if resolved.name != spec["filename"]:
        raise RuntimeError(
            f"{stage} checkpoint filename {resolved.name!r} != "
            f"official {spec['filename']!r}"
        )
    actual = sha256_file(resolved)
    validate_official_digest(stage, actual)
    return {
        "stage": stage,
        "path": str(resolved),
        "filename": spec["filename"],
        "sha256": actual,
        "official_all_speakers": True,
        "withdrawn_e30_allowed": False,
    }


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash a tensor state dict without depending on torch serialization."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not hasattr(value, "detach"):
            raise TypeError(f"state entry {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def normalized_model_state(torch: Any, payload: bytes, label: str) -> dict[str, Any]:
    """Load only a checkpoint model state and normalize one optional DDP prefix."""

    import io

    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state"), dict
    ):
        raise RuntimeError(f"{label} checkpoint lacks model_state")
    result: dict[str, Any] = {}
    for raw_key, value in checkpoint["model_state"].items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            raise RuntimeError(f"{label} has invalid model_state entry")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in result:
            raise RuntimeError(f"{label} has duplicate normalized key {key!r}")
        if (
            (value.is_floating_point() or value.is_complex())
            and not bool(value.isfinite().all().item())
        ):
            raise RuntimeError(f"{label} tensor {key!r} contains NaN/Inf")
        result[key] = value
    return result


def load_official_model_state(
    torch: Any,
    *,
    stage: str,
    checkpoint: Path,
    model: Any,
) -> dict[str, str]:
    receipt = verify_official_checkpoint(stage, checkpoint)
    payload = Path(receipt["path"]).read_bytes()
    state = normalized_model_state(
        torch,
        payload,
        f"official All-Speakers {stage}",
    )
    expected_keys = set(model.state_dict())
    actual_keys = set(state)
    if expected_keys != actual_keys:
        raise RuntimeError(
            f"{stage} state schema mismatch; "
            f"missing={sorted(expected_keys - actual_keys)[:5]}, "
            f"extra={sorted(actual_keys - expected_keys)[:5]}"
        )
    model.load_state_dict(state, strict=True)
    receipt["model_state_sha256"] = state_dict_sha256(state)
    return receipt


def freeze_face_for_decoder_transfer(model: Any) -> dict[str, Any]:
    """Freeze encoder and RVQ/EMA, leaving exactly the decoder trainable."""

    if not all(hasattr(model, name) for name in ("encoder", "quantizer", "decoder")):
        raise TypeError("face model must expose encoder, quantizer, and decoder")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    model.encoder.eval()
    model.quantizer.eval()
    model.decoder.train()
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any(not name.startswith("decoder.") for name in trainable):
        raise RuntimeError("face transfer must train decoder parameters only")
    return {
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_encoder": True,
        "frozen_quantizer": True,
        "frozen_rvq_ema": True,
    }


def frozen_face_state(model: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("encoder.") or key.startswith("quantizer.")
    }


def configure_global_transfer_policy(
    model: Any,
    *,
    policy: str,
    authorize_full_model_fallback: bool,
) -> dict[str, Any]:
    """Configure the predeclared Global adaptation boundary.

    Decoder-only is the default and freezes the official encoder.  A full
    model fallback is intentionally awkward to enable: it must be explicitly
    selected *and* separately authorized after a frozen decoder-only gate
    fails.  No code path silently broadens trainability.
    """

    if not all(hasattr(model, name) for name in ("encoder", "decoder")):
        raise TypeError("Global model must expose encoder and decoder")
    if policy not in {"decoder_only", "full_model_fallback"}:
        raise ValueError(f"unsupported Global transfer policy {policy!r}")
    if policy == "full_model_fallback" and not authorize_full_model_fallback:
        raise RuntimeError(
            "full_model_fallback requires explicit "
            "--authorize-full-global-fallback"
        )
    if policy == "decoder_only" and authorize_full_model_fallback:
        raise RuntimeError(
            "fallback authorization is invalid with decoder_only policy"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(policy == "full_model_fallback")
    if policy == "decoder_only":
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)
        model.eval()
        model.encoder.eval()
        model.decoder.train()
    else:
        model.train()
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Global transfer has no trainable parameters")
    if policy == "decoder_only" and any(
        not name.startswith("decoder.") for name in trainable
    ):
        raise RuntimeError("decoder_only Global policy leaked trainable state")
    return {
        "policy": policy,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_encoder": policy == "decoder_only",
        "full_model_fallback_authorized": bool(authorize_full_model_fallback),
        "fallback_requires_frozen_decoder_gate_failure": (
            policy == "full_model_fallback"
        ),
    }


def frozen_global_state(model: Any, *, policy: str) -> dict[str, Any]:
    if policy == "decoder_only":
        return {
            key: value
            for key, value in model.state_dict().items()
            if key.startswith("encoder.")
        }
    if policy == "full_model_fallback":
        return {}
    raise ValueError(policy)


def _matrix_geodesic(torch: Any, relative: Any) -> Any:
    """Return a finite-gradient SO(3) angle for relative rotation matrices.

    ``acos(trace)`` has an infinite derivative at identity and can turn the
    first perfectly aligned temporal delta into NaN gradients.  For a valid
    rotation matrix, the norm of the skew-vector is ``abs(sin(theta))``.
    ``atan2(abs(sin(theta)), cos(theta))`` therefore returns the same angle in
    ``[0, pi]`` while PyTorch's zero subgradient for ``vector_norm(0)`` keeps
    exact identity and pi rotations finite.
    """

    if relative.shape[-2:] != (3, 3):
        raise ValueError("relative rotations must end in a 3x3 matrix")
    cosine = (
        relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0
    ) / 2.0
    skew_vector = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine_magnitude = torch.linalg.vector_norm(skew_vector, dim=-1)
    return torch.atan2(
        sine_magnitude,
        cosine.clamp(-1.0, 1.0),
    )


def rotation_geodesic(torch: Any, target: Any, prediction: Any) -> Any:
    """Return SO(3) geodesic distance for matching ``(..., 6)`` tensors."""

    from utils import rotation_conversions as rc

    if target.shape != prediction.shape or target.shape[-1] != 6:
        raise ValueError("rotation tensors must have matching (..., 6) shape")
    target_matrix = rc.rotation_6d_to_matrix(target)
    prediction_matrix = rc.rotation_6d_to_matrix(prediction)
    relative = prediction_matrix.transpose(-1, -2) @ target_matrix
    return _matrix_geodesic(torch, relative)


def _rotation_delta_geodesic(torch: Any, target: Any, prediction: Any) -> Any:
    from utils import rotation_conversions as rc

    target_matrix = rc.rotation_6d_to_matrix(target)
    prediction_matrix = rc.rotation_6d_to_matrix(prediction)
    target_delta = target_matrix[:, :-1].transpose(-1, -2) @ target_matrix[:, 1:]
    prediction_delta = (
        prediction_matrix[:, :-1].transpose(-1, -2) @ prediction_matrix[:, 1:]
    )
    relative = prediction_delta.transpose(-1, -2) @ target_delta
    return _matrix_geodesic(torch, relative)


def face_task_losses(
    torch: Any,
    prediction: Any,
    target: Any,
    *,
    jaw_weight: float,
    expression_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
) -> dict[str, Any]:
    """Task-space jaw/expression reconstruction and temporal losses.

    No SMPL-X call is involved.  Jaw rotation is compared on SO(3);
    expression and temporal expression terms are L1.  Jaw velocity compares
    relative rotations, and jaw acceleration compares adjacent relative
    rotation matrices in task space.
    """

    from utils import rotation_conversions as rc

    if prediction.shape != target.shape or prediction.shape[-1] != 106:
        raise ValueError("face tensors must have matching [B,T,106] shape")
    if prediction.shape[1] < 3:
        raise ValueError("face task losses require at least three frames")
    if not bool(prediction.isfinite().all().item()) or not bool(
        target.isfinite().all().item()
    ):
        raise RuntimeError("face tensors contain NaN/Inf")

    jaw_target = target[..., :6]
    jaw_prediction = prediction[..., :6]
    expression_target = target[..., 6:]
    expression_prediction = prediction[..., 6:]

    jaw = rotation_geodesic(torch, jaw_target, jaw_prediction).mean()
    expression = torch.nn.functional.l1_loss(
        expression_prediction,
        expression_target,
    )
    jaw_velocity = _rotation_delta_geodesic(
        torch,
        jaw_target,
        jaw_prediction,
    ).mean()
    expression_velocity = torch.nn.functional.l1_loss(
        expression_prediction[:, 1:] - expression_prediction[:, :-1],
        expression_target[:, 1:] - expression_target[:, :-1],
    )

    target_matrix = rc.rotation_6d_to_matrix(jaw_target)
    prediction_matrix = rc.rotation_6d_to_matrix(jaw_prediction)
    jaw_acceleration = torch.nn.functional.l1_loss(
        prediction_matrix[:, 2:]
        + prediction_matrix[:, :-2]
        - 2.0 * prediction_matrix[:, 1:-1],
        target_matrix[:, 2:] + target_matrix[:, :-2] - 2.0 * target_matrix[:, 1:-1],
    )
    expression_acceleration = torch.nn.functional.l1_loss(
        expression_prediction[:, 2:]
        + expression_prediction[:, :-2]
        - 2.0 * expression_prediction[:, 1:-1],
        expression_target[:, 2:]
        + expression_target[:, :-2]
        - 2.0 * expression_target[:, 1:-1],
    )
    velocity = jaw_velocity + expression_velocity
    acceleration = jaw_acceleration + expression_acceleration
    total = (
        float(jaw_weight) * jaw
        + float(expression_weight) * expression
        + float(velocity_weight) * velocity
        + float(acceleration_weight) * acceleration
    )
    result = {
        "total": total,
        "jaw_geodesic": jaw,
        "expression_l1": expression,
        "velocity": velocity,
        "jaw_velocity_geodesic": jaw_velocity,
        "expression_velocity_l1": expression_velocity,
        "acceleration": acceleration,
        "jaw_acceleration_matrix_l1": jaw_acceleration,
        "expression_acceleration_l1": expression_acceleration,
    }
    if not all(bool(value.isfinite().item()) for value in result.values()):
        raise RuntimeError("face task loss contains NaN/Inf")
    return result


def central_velocity(torch: Any, translation: Any, *, fps: int = FPS) -> Any:
    if translation.ndim != 3 or translation.shape[-1] != 3:
        raise ValueError("translation must have [B,T,3] shape")
    if translation.shape[1] < 2:
        raise ValueError("translation requires at least two frames")
    dt = 1.0 / float(fps)
    velocity = torch.zeros_like(translation)
    velocity[:, 1:-1] = (translation[:, 2:] - translation[:, :-2]) / (2.0 * dt)
    velocity[:, 0] = (translation[:, 1] - translation[:, 0]) / dt
    velocity[:, -1] = (translation[:, -1] - translation[:, -2]) / dt
    return velocity


def root_channels_from_translation(torch: Any, translation: Any) -> Any:
    velocity = central_velocity(torch, translation)
    return torch.stack(
        [
            velocity[..., 0],
            translation[..., 1],
            velocity[..., 2],
        ],
        dim=-1,
    )


def integrate_root_channels(torch: Any, channels: Any, anchor: Any) -> Any:
    """Integrate x/z velocity with a shared absolute anchor; y stays absolute."""

    if channels.ndim != 3 or channels.shape[-1] != 3:
        raise ValueError("root channels must have [B,T,3] shape")
    if anchor.ndim != 2 or anchor.shape != (channels.shape[0], 3):
        raise ValueError("anchor must have [B,3] shape")
    result = torch.zeros_like(channels)
    result[..., 1] = channels[..., 1]
    result[:, 0, 0] = anchor[:, 0]
    result[:, 0, 2] = anchor[:, 2]
    dt = 1.0 / float(FPS)
    for frame in range(1, channels.shape[1]):
        result[:, frame, 0] = (
            result[:, frame - 1, 0] + channels[:, frame - 1, 0] * dt
        )
        result[:, frame, 2] = (
            result[:, frame - 1, 2] + channels[:, frame - 1, 2] * dt
        )
    return result


def compose_global_input(
    torch: Any,
    decoded_lower: Any,
    observed_contact: Any,
    *,
    pre_frames: int = PRE_FRAMES,
) -> Any:
    """Build the actual Base→Lower→Global interface used for SHOW transfer.

    Rotations always come from the frozen official Lower decoder.  Lower
    translation channels are forced to zero for every frame and cannot carry a
    world-root seed.  The observed prefix uses canonical contact labels; all
    generated frames preserve the decoded Lower contacts.
    """

    if decoded_lower.ndim != 3 or decoded_lower.shape[-1] != 61:
        raise ValueError("decoded_lower must have [B,T,61] shape")
    if observed_contact.shape != (*decoded_lower.shape[:2], 4):
        raise ValueError("observed_contact must have [B,T,4] shape")
    if not 0 <= pre_frames <= decoded_lower.shape[1]:
        raise ValueError("pre_frames is out of range")
    from utils import rotation_conversions as rc

    result = decoded_lower.clone()
    rotations = result[..., :54].reshape(*result.shape[:2], 9, 6)
    result[..., :54] = rc.matrix_to_rotation_6d(
        rc.rotation_6d_to_matrix(rotations)
    ).reshape(*result.shape[:2], 54)
    result[..., 54:57] = 0.0
    if pre_frames:
        result[:, :pre_frames, 57:61] = observed_contact[:, :pre_frames]
    if not bool(result.isfinite().all().item()):
        raise RuntimeError("composed Global input contains NaN/Inf")
    if not bool(torch.equal(result[..., 54:57], torch.zeros_like(result[..., 54:57]))):
        raise RuntimeError("Global input lower translation channels are not exact zero")
    return result


def global_root_losses(
    torch: Any,
    prediction: Any,
    target_root_channels: Any,
    target_translation: Any,
    anchor: Any,
    *,
    channel_weight: float,
    integrated_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
) -> dict[str, Any]:
    if prediction.ndim != 3 or prediction.shape[-1] != 61:
        raise ValueError("Global prediction must have [B,T,61] shape")
    if target_root_channels.shape != (*prediction.shape[:2], 3):
        raise ValueError("target_root_channels shape mismatch")
    if target_translation.shape != (*prediction.shape[:2], 3):
        raise ValueError("target_translation shape mismatch")
    predicted_root = prediction[..., 54:57]
    integrated = integrate_root_channels(torch, predicted_root, anchor)
    channel = torch.nn.functional.l1_loss(predicted_root, target_root_channels)
    integrated_error = torch.nn.functional.l1_loss(integrated, target_translation)
    velocity = torch.nn.functional.l1_loss(
        predicted_root[:, 1:] - predicted_root[:, :-1],
        target_root_channels[:, 1:] - target_root_channels[:, :-1],
    )
    acceleration = torch.nn.functional.l1_loss(
        predicted_root[:, 2:]
        + predicted_root[:, :-2]
        - 2.0 * predicted_root[:, 1:-1],
        target_root_channels[:, 2:]
        + target_root_channels[:, :-2]
        - 2.0 * target_root_channels[:, 1:-1],
    )
    total = (
        float(channel_weight) * channel
        + float(integrated_weight) * integrated_error
        + float(velocity_weight) * velocity
        + float(acceleration_weight) * acceleration
    )
    result = {
        "total": total,
        "root_channel_l1": channel,
        "integrated_root_l1": integrated_error,
        "root_velocity_l1": velocity,
        "root_acceleration_l1": acceleration,
    }
    if not all(bool(value.isfinite().item()) for value in result.values()):
        raise RuntimeError("Global root loss contains NaN/Inf")
    return result


def validate_loss_weights(values: Mapping[str, float], label: str) -> None:
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values.values()
    ):
        raise ValueError(f"{label} weights must be finite non-negative numbers")
    if sum(float(value) for value in values.values()) <= 0.0:
        raise ValueError(f"{label} requires at least one positive weight")
