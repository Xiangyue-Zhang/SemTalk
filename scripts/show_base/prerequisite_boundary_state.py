#!/usr/bin/env python3
"""Structural non-initialization proof for prerequisite resume state.

The proof is CPU-only and independent of model construction.  It validates
the serialized optimizer parameter IDs, Adam moments and step counters,
epoch scheduler position, RVQ EMA accumulators, and rank RNG inventory before
a continuation process may load the state into live objects.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


FORMAT = "semtalk_show_prerequisite_boundary_state_v1"
STAGES = ("face", "hands", "upper", "lower", "global")
RVQ_STAGES = frozenset(("face", "hands", "upper", "lower"))
UPDATES_PER_EPOCH = 497
GLOBAL_UPDATES_PER_EPOCH = 1_988
RNG_KEYS = {"python", "numpy", "torch_cpu", "torch_cuda"}
PROOF_KEYS = {
    "format",
    "stage",
    "boundary_epoch",
    "optimizer_updates",
    "world_size",
    "model_state_sha256",
    "optimizer_state_sha256",
    "scheduler_state_sha256",
    "rvq_ema_state_sha256",
    "rng_states_sha256",
    "trained_parameter_count",
    "adam_state_count",
    "adam_step",
    "scheduler_epoch",
    "scheduler_lrs",
    "rvq_ema_layers",
    "rng_rank_count",
    "receipt_payload_sha256",
}


class BoundaryStateError(RuntimeError):
    """Raised when resume state is empty, reset, or structurally ambiguous."""


def updates_per_epoch(stage: str) -> int:
    if stage in RVQ_STAGES:
        return UPDATES_PER_EPOCH
    if stage == "global":
        return GLOBAL_UPDATES_PER_EPOCH
    raise BoundaryStateError(f"unknown stage {stage!r}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BoundaryStateError("state proof is not finite JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BoundaryStateError(f"{label} must be an exact integer")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise BoundaryStateError(f"{label} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise BoundaryStateError(f"{label} must be finite")
    return number


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BoundaryStateError(f"{label} must be a lowercase SHA-256")
    return value


def _step_number(value: Any, label: str) -> int:
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    number = _require_finite(value, label)
    if not number.is_integer():
        raise BoundaryStateError(f"{label} must be an exact integer step")
    return int(number)


def _tensor_info(value: Any, label: str) -> dict[str, Any]:
    """Return finite tensor statistics without importing torch at module load."""

    if getattr(value, "_continuation_test_tensor", False):
        shape = tuple(value.shape)
        values = tuple(float(item) for item in value.values)
        if not values or math.prod(shape) != len(values):
            raise BoundaryStateError(f"{label} test tensor is empty")
        if not all(math.isfinite(item) for item in values):
            raise BoundaryStateError(f"{label} contains non-finite values")
        return {
            "shape": shape,
            "numel": len(values),
            "nonzero": any(item != 0.0 for item in values),
            "sum": float(sum(values)),
            "bytes": value.bytes,
            "dtype": value.dtype,
        }

    module = type(value).__module__
    name = type(value).__name__
    if not (module.startswith("torch") and name == "Tensor"):
        raise BoundaryStateError(f"{label} must be a torch Tensor")
    import torch

    tensor = value.detach().cpu()
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    tensor = tensor.contiguous()
    if tensor.numel() <= 0:
        raise BoundaryStateError(f"{label} tensor is empty")
    if (
        (tensor.is_floating_point() or tensor.is_complex())
        and not bool(torch.isfinite(tensor).all().item())
    ):
        raise BoundaryStateError(f"{label} contains non-finite values")
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": tuple(tensor.shape),
        "numel": int(tensor.numel()),
        "nonzero": bool(torch.count_nonzero(tensor).item()),
        "sum": float(tensor.double().sum().item()),
        "bytes": raw,
        "dtype": str(tensor.dtype),
    }


def _semantic_update(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii") + b";")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise BoundaryStateError("state contains a non-finite scalar")
        digest.update(b"F" + value.hex().encode("ascii") + b";")
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + str(len(encoded)).encode("ascii") + b":")
        digest.update(encoded)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        digest.update(b"Y" + str(len(encoded)).encode("ascii") + b":")
        digest.update(encoded)
    elif isinstance(value, tuple):
        digest.update(b"(" + str(len(value)).encode("ascii") + b":")
        for item in value:
            _semantic_update(digest, item)
    elif isinstance(value, list):
        digest.update(b"[" + str(len(value)).encode("ascii") + b":")
        for item in value:
            _semantic_update(digest, item)
    elif isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            key_digest = hashlib.sha256()
            _semantic_update(key_digest, key)
            items.append((key_digest.hexdigest(), key, item))
        digest.update(b"{" + str(len(items)).encode("ascii") + b":")
        for _, key, item in sorted(items, key=lambda entry: entry[0]):
            _semantic_update(digest, key)
            _semantic_update(digest, item)
    elif (
        getattr(value, "_continuation_test_tensor", False)
        or (
            type(value).__module__.startswith("torch")
            and type(value).__name__ == "Tensor"
        )
    ):
        info = _tensor_info(value, "semantic tensor")
        descriptor = _canonical_bytes(
            {
                "dtype": info["dtype"],
                "shape": list(info["shape"]),
            }
        )
        digest.update(b"T" + str(len(descriptor)).encode("ascii") + b":")
        digest.update(descriptor)
        digest.update(info["bytes"])
    elif (
        type(value).__module__.startswith("numpy")
        and hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "tobytes")
    ):
        import numpy as np

        array = np.asarray(value)
        if array.size <= 0:
            raise BoundaryStateError("state contains an empty NumPy array")
        if array.dtype.kind in {"f", "c"} and not bool(
            np.isfinite(array).all()
        ):
            raise BoundaryStateError(
                "state contains a non-finite NumPy array"
            )
        contiguous = np.ascontiguousarray(array)
        descriptor = _canonical_bytes(
            {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            }
        )
        digest.update(b"A" + str(len(descriptor)).encode("ascii") + b":")
        digest.update(descriptor)
        digest.update(contiguous.tobytes(order="C"))
    elif type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        _semantic_update(digest, value.item())
    else:
        raise BoundaryStateError(
            f"unsupported state type {type(value)!r}"
        )


def semantic_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _semantic_update(digest, value)
    return digest.hexdigest()


def _validate_model_state(value: Any) -> tuple[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise BoundaryStateError("model state is empty")
    nonzero = False
    for name, tensor in value.items():
        if not isinstance(name, str) or not name:
            raise BoundaryStateError("model state key is invalid")
        info = _tensor_info(tensor, f"model state {name}")
        nonzero = nonzero or info["nonzero"]
    if not nonzero:
        raise BoundaryStateError("model state is all zero")
    return semantic_sha256(value), nonzero


def _validate_optimizer_state(
    value: Any,
    *,
    expected_updates: int,
) -> tuple[str, int, int]:
    if not isinstance(value, dict) or set(value) != {
        "state",
        "param_groups",
    }:
        raise BoundaryStateError("optimizer state schema mismatch")
    state = value["state"]
    groups = value["param_groups"]
    if not isinstance(state, dict) or not state:
        raise BoundaryStateError("optimizer state is empty")
    if not isinstance(groups, list) or not groups:
        raise BoundaryStateError("optimizer param groups are empty")
    parameter_ids: list[int] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise BoundaryStateError(
                f"optimizer param group {group_index} is invalid"
            )
        params = group.get("params")
        if not isinstance(params, list) or not params:
            raise BoundaryStateError(
                f"optimizer param group {group_index} is empty"
            )
        for parameter in params:
            parameter_ids.append(
                _require_int(
                    parameter,
                    f"optimizer param group {group_index} parameter",
                )
            )
        _require_finite(
            group.get("lr"),
            f"optimizer param group {group_index} lr",
        )
    if (
        len(parameter_ids) != len(set(parameter_ids))
        or set(state) != set(parameter_ids)
    ):
        raise BoundaryStateError(
            "optimizer state does not exactly cover trained parameters"
        )

    any_first_moment = False
    any_second_moment = False
    for parameter_id in parameter_ids:
        parameter_state = state[parameter_id]
        if not isinstance(parameter_state, dict):
            raise BoundaryStateError(
                f"optimizer state {parameter_id} is invalid"
            )
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not required <= set(parameter_state):
            raise BoundaryStateError(
                f"optimizer state {parameter_id} is missing Adam moments"
            )
        step = _step_number(
            parameter_state["step"],
            f"optimizer state {parameter_id} step",
        )
        if step != expected_updates:
            raise BoundaryStateError(
                f"optimizer state {parameter_id} step mismatch"
            )
        first = _tensor_info(
            parameter_state["exp_avg"],
            f"optimizer state {parameter_id} exp_avg",
        )
        second = _tensor_info(
            parameter_state["exp_avg_sq"],
            f"optimizer state {parameter_id} exp_avg_sq",
        )
        if first["shape"] != second["shape"]:
            raise BoundaryStateError(
                f"optimizer state {parameter_id} moment shape mismatch"
            )
        any_first_moment = any_first_moment or first["nonzero"]
        any_second_moment = any_second_moment or second["nonzero"]
    if not any_first_moment or not any_second_moment:
        raise BoundaryStateError("optimizer Adam moments are all zero")
    return (
        semantic_sha256(value),
        len(parameter_ids),
        len(state),
    )


def _cosine_epoch_values(
    scheduler: Mapping[str, Any],
    epoch: int,
) -> list[float]:
    base_values = scheduler.get("base_values")
    if not isinstance(base_values, list) or not base_values:
        raise BoundaryStateError("scheduler base_values are unavailable")
    base = [
        _require_finite(value, "scheduler base value")
        for value in base_values
    ]
    t_initial = _require_int(
        scheduler.get("t_initial"),
        "scheduler t_initial",
    )
    if t_initial <= 0:
        raise BoundaryStateError("scheduler t_initial must be positive")
    t_mul = _require_finite(scheduler.get("t_mul"), "scheduler t_mul")
    lr_min = _require_finite(
        scheduler.get("lr_min"),
        "scheduler lr_min",
    )
    decay_rate = _require_finite(
        scheduler.get("decay_rate"),
        "scheduler decay_rate",
    )
    cycle_limit = _require_int(
        scheduler.get("cycle_limit"),
        "scheduler cycle_limit",
    )
    warmup_t = _require_int(
        scheduler.get("warmup_t"),
        "scheduler warmup_t",
    )
    warmup_lr = _require_finite(
        scheduler.get("warmup_lr_init"),
        "scheduler warmup_lr_init",
    )
    if scheduler.get("noise_range_t") is not None:
        raise BoundaryStateError(
            "scheduler noise requires a separate deterministic proof"
        )
    if scheduler.get("t_in_epochs") is not True:
        raise BoundaryStateError("scheduler is not epoch based")
    if epoch < warmup_t:
        steps = scheduler.get("warmup_steps")
        if not isinstance(steps, list) or len(steps) != len(base):
            raise BoundaryStateError("scheduler warmup steps mismatch")
        return [
            warmup_lr
            + epoch * _require_finite(step, "scheduler warmup step")
            for step in steps
        ]
    current = epoch - warmup_t if scheduler.get("warmup_prefix") else epoch
    if t_mul == 1.0:
        cycle = current // t_initial
        cycle_length = float(t_initial)
        cycle_position = current - t_initial * cycle
    else:
        argument = 1.0 - current / t_initial * (1.0 - t_mul)
        if argument <= 0.0 or t_mul <= 0.0:
            raise BoundaryStateError("scheduler cycle is invalid")
        cycle = math.floor(math.log(argument, t_mul))
        cycle_length = t_initial * (t_mul ** cycle)
        cycle_position = current - (
            (1.0 - t_mul ** cycle) / (1.0 - t_mul) * t_initial
        )
    gamma = decay_rate ** cycle
    minimum = lr_min * gamma
    if cycle_limit != 0 and cycle >= cycle_limit:
        return [lr_min for _ in base]
    return [
        minimum
        + 0.5
        * (maximum * gamma - minimum)
        * (1.0 + math.cos(math.pi * cycle_position / cycle_length))
        for maximum in base
    ]


def _step_epoch_values(
    scheduler: Mapping[str, Any],
    epoch: int,
) -> list[float]:
    base_values = scheduler.get("base_values")
    if not isinstance(base_values, list) or not base_values:
        raise BoundaryStateError("scheduler base_values are unavailable")
    base = [
        _require_finite(value, "scheduler base value")
        for value in base_values
    ]
    decay_t = _require_int(
        scheduler.get("decay_t"),
        "scheduler decay_t",
    )
    decay_rate = _require_finite(
        scheduler.get("decay_rate"),
        "scheduler decay_rate",
    )
    warmup_t = _require_int(
        scheduler.get("warmup_t"),
        "scheduler warmup_t",
    )
    warmup_lr = _require_finite(
        scheduler.get("warmup_lr_init"),
        "scheduler warmup_lr_init",
    )
    if decay_t <= 0 or decay_rate <= 0.0 or warmup_t < 0:
        raise BoundaryStateError("step scheduler envelope is invalid")
    if scheduler.get("noise_range_t") is not None:
        raise BoundaryStateError(
            "scheduler noise requires a separate deterministic proof"
        )
    if scheduler.get("t_in_epochs") is not True:
        raise BoundaryStateError("scheduler is not epoch based")
    if epoch < warmup_t:
        steps = scheduler.get("warmup_steps")
        if not isinstance(steps, list) or len(steps) != len(base):
            raise BoundaryStateError("scheduler warmup steps mismatch")
        return [
            warmup_lr
            + epoch * _require_finite(step, "scheduler warmup step")
            for step in steps
        ]
    return [
        value * (decay_rate ** (epoch // decay_t))
        for value in base
    ]


def _validate_scheduler_state(
    value: Any,
    *,
    optimizer_state: Mapping[str, Any],
    boundary_epoch: int,
) -> tuple[str, list[float]]:
    if not isinstance(value, dict) or not value:
        raise BoundaryStateError("scheduler state is empty")
    has_step = "decay_t" in value
    has_cosine = "t_initial" in value
    if has_step == has_cosine:
        raise BoundaryStateError(
            "scheduler state kind is ambiguous or unsupported"
        )
    expected_lrs = (
        _step_epoch_values(value, boundary_epoch - 1)
        if has_step
        else _cosine_epoch_values(value, boundary_epoch - 1)
    )
    groups = optimizer_state["param_groups"]
    observed_lrs = [
        _require_finite(group.get("lr"), "optimizer current lr")
        for group in groups
    ]
    if len(observed_lrs) != len(expected_lrs) or any(
        not math.isclose(
            observed,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        for observed, expected in zip(observed_lrs, expected_lrs)
    ):
        raise BoundaryStateError(
            "scheduler progress differs from boundary epoch"
        )
    return semantic_sha256(value), observed_lrs


def _validate_rvq_ema(
    value: Any,
    *,
    stage: str,
) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise BoundaryStateError("RVQ EMA state must be an object")
    if stage == "global":
        if value:
            raise BoundaryStateError("Global must not contain RVQ EMA state")
        return semantic_sha256(value), 0
    if len(value) != 6:
        raise BoundaryStateError("RVQ state must contain exactly six layers")
    for name, item in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(item, dict)
            or set(item) != {"init", "code_sum", "code_count"}
            or item["init"] is not True
        ):
            raise BoundaryStateError(f"RVQ EMA layer {name!r} is uninitialized")
        code_sum = _tensor_info(
            item["code_sum"],
            f"RVQ EMA {name} code_sum",
        )
        code_count = _tensor_info(
            item["code_count"],
            f"RVQ EMA {name} code_count",
        )
        if (
            len(code_count["shape"]) != 1
            or not code_count["nonzero"]
            or code_count["sum"] <= 0.0
            or not code_sum["nonzero"]
            or not code_sum["shape"]
            or code_sum["shape"][0] != code_count["shape"][0]
        ):
            raise BoundaryStateError(
                f"RVQ EMA layer {name!r} accumulators are invalid"
            )
    return semantic_sha256(value), len(value)


def _validate_rng_states(
    value: Any,
    *,
    world_size: int,
) -> tuple[str, int]:
    if not isinstance(value, list) or len(value) != world_size:
        raise BoundaryStateError(
            "RNG state count does not match exact world size"
        )
    for rank, state in enumerate(value):
        if not isinstance(state, dict) or set(state) != RNG_KEYS:
            raise BoundaryStateError(f"RNG rank {rank} schema mismatch")
        if (
            not isinstance(state["python"], tuple)
            or not state["python"]
            or not isinstance(state["numpy"], tuple)
            or len(state["numpy"]) != 5
        ):
            raise BoundaryStateError(f"RNG rank {rank} host state is empty")
        _tensor_info(state["torch_cpu"], f"RNG rank {rank} torch CPU")
        _tensor_info(state["torch_cuda"], f"RNG rank {rank} torch CUDA")
    return semantic_sha256(value), len(value)


def build_boundary_state_proof(
    resume: Mapping[str, Any],
    *,
    stage: str,
    boundary_epoch: int,
    world_size: int,
) -> dict[str, Any]:
    """Validate one old resume and return its immutable state proof."""

    if stage not in STAGES:
        raise BoundaryStateError(f"unknown stage {stage!r}")
    boundary = _require_int(boundary_epoch, "boundary epoch")
    world = _require_int(world_size, "world size")
    if boundary < 200 or boundary % 20 or world <= 0:
        raise BoundaryStateError("boundary/world protocol mismatch")
    expected_world = 4 if stage in RVQ_STAGES else 1
    stage_updates = updates_per_epoch(stage)
    expected_updates = boundary * stage_updates
    if (
        resume.get("format") != "semtalk_show_train_resume_v5"
        or resume.get("completed_epochs") != boundary
        or resume.get("optimizer_updates") != expected_updates
        or resume.get("updates_per_epoch") != stage_updates
        or resume.get("world_size") != world
        or world != expected_world
    ):
        raise BoundaryStateError("resume boundary accounting mismatch")

    model_sha, _ = _validate_model_state(resume.get("model_state"))
    optimizer = resume.get("optimizer_state")
    optimizer_sha, parameter_count, adam_count = (
        _validate_optimizer_state(
            optimizer,
            expected_updates=expected_updates,
        )
    )
    scheduler_sha, scheduler_lrs = _validate_scheduler_state(
        resume.get("scheduler_state"),
        optimizer_state=optimizer,
        boundary_epoch=boundary,
    )
    rvq_sha, rvq_layers = _validate_rvq_ema(
        resume.get("rvq_ema_state"),
        stage=stage,
    )
    rng_sha, rng_count = _validate_rng_states(
        resume.get("rng_states"),
        world_size=world,
    )
    proof = {
        "format": FORMAT,
        "stage": stage,
        "boundary_epoch": boundary,
        "optimizer_updates": expected_updates,
        "world_size": world,
        "model_state_sha256": model_sha,
        "optimizer_state_sha256": optimizer_sha,
        "scheduler_state_sha256": scheduler_sha,
        "rvq_ema_state_sha256": rvq_sha,
        "rng_states_sha256": rng_sha,
        "trained_parameter_count": parameter_count,
        "adam_state_count": adam_count,
        "adam_step": expected_updates,
        "scheduler_epoch": boundary - 1,
        "scheduler_lrs": scheduler_lrs,
        "rvq_ema_layers": rvq_layers,
        "rng_rank_count": rng_count,
    }
    proof["receipt_payload_sha256"] = canonical_sha256(proof)
    validate_boundary_state_proof(proof)
    return proof


def validate_boundary_state_proof(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PROOF_KEYS:
        raise BoundaryStateError("boundary state proof schema mismatch")
    if value["format"] != FORMAT or value["stage"] not in STAGES:
        raise BoundaryStateError("boundary state proof protocol mismatch")
    boundary = _require_int(value["boundary_epoch"], "proof boundary")
    updates = _require_int(value["optimizer_updates"], "proof updates")
    world = _require_int(value["world_size"], "proof world size")
    expected_world = 4 if value["stage"] in RVQ_STAGES else 1
    if (
        boundary < 200
        or boundary % 20
        or updates != boundary * updates_per_epoch(value["stage"])
        or world != expected_world
        or value["adam_step"] != updates
        or value["scheduler_epoch"] != boundary - 1
        or value["trained_parameter_count"] <= 0
        or value["adam_state_count"] != value["trained_parameter_count"]
        or value["rng_rank_count"] != world
        or value["rvq_ema_layers"]
        != (6 if value["stage"] in RVQ_STAGES else 0)
    ):
        raise BoundaryStateError("boundary state proof accounting mismatch")
    for key in (
        "model_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "rvq_ema_state_sha256",
        "rng_states_sha256",
        "receipt_payload_sha256",
    ):
        _require_sha256(value[key], f"proof {key}")
    lrs = value["scheduler_lrs"]
    if not isinstance(lrs, list) or not lrs:
        raise BoundaryStateError("proof scheduler LRs are empty")
    for index, lr in enumerate(lrs):
        _require_finite(lr, f"proof scheduler LR {index}")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_payload_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise BoundaryStateError("boundary state proof payload mismatch")
    return dict(value)
