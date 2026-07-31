"""Small invariants for differentiable SMPL-X training losses.

The RVQ/VAE itself deliberately remains on logical CUDA device 0.  The
optional :class:`SmplxTrainingPool` only replicates the frozen SMPL-X loss
module.  ``target_offload`` preserves the stock reconstruction forward and
autograd graph on logical device 0 while overlapping the detached full-batch
target forward on logical device 1.  ``full_batch_dual`` preserves the
original two-helper reference path.  ``sharded_local_loss`` partitions
flattened batches by whole clips, runs reconstruction and detached target
SMPL-X on every helper, and returns only differentiable scalar numerators to
logical device 0.  The RVQ/VAE, optimizer, update order, and training RNG
remain on logical device 0.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, TypeVar

import numpy as np
import torch


_Module = TypeVar("_Module")
_POOL_MODES = frozenset({
    "disabled",
    "target_offload",
    "full_batch_dual",
    "sharded_local_loss",
})
_LOCAL_LOSS_STAGES = frozenset({"face", "hands", "upper", "lower"})


def freeze_smplx_for_training(model: _Module) -> _Module:
    """Freeze SMPL-X parameters without disabling gradients for its inputs."""
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("SMPL-X parameters must be frozen during training")
    return model


def smplx_target_forward(model: _Module, **kwargs):
    """Compute a detached target and skip the unused shaped-vertex output."""
    if "return_shaped" in kwargs:
        raise TypeError("return_shaped is fixed to False for target forwards")
    with torch.no_grad():
        return model(return_shaped=False, **kwargs)


def parse_smplx_helper_devices(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse an explicit, duplicate-free list of logical helper devices."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        pieces: Iterable[object] = stripped.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        pieces = value
    else:
        raise TypeError(
            "smplx_training_helper_devices must be a comma-separated string "
            "or an integer sequence"
        )

    devices: list[int] = []
    for piece in pieces:
        if isinstance(piece, bool):
            raise TypeError("SMPL-X helper device indices must be exact integers")
        if isinstance(piece, int):
            device = piece
        elif isinstance(piece, str):
            token = piece.strip()
            if not token or not token.isascii() or not token.isdecimal():
                raise ValueError(
                    "SMPL-X helper devices must contain decimal logical indices"
                )
            device = int(token)
        else:
            raise TypeError("SMPL-X helper device indices must be exact integers")
        if device <= 0:
            raise ValueError(
                "logical CUDA device 0 is the fixed primary and cannot be a helper"
            )
        devices.append(device)
    if len(set(devices)) != len(devices):
        raise ValueError("SMPL-X helper device indices must be unique")
    return tuple(devices)


def clip_aligned_spans(
    total_rows: int,
    clip_length: int,
    shard_count: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return ``(row_start,row_end,clip_start,clip_end)`` shards."""
    for name, value in (
        ("total_rows", total_rows),
        ("clip_length", clip_length),
        ("shard_count", shard_count),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive exact integer")
    if total_rows % clip_length != 0:
        raise ValueError("flattened SMPL-X rows are not whole clips")
    clip_count = total_rows // clip_length
    if clip_count < shard_count:
        raise ValueError("every helper must receive at least one whole clip")
    base, remainder = divmod(clip_count, shard_count)
    spans: list[tuple[int, int, int, int]] = []
    clip_start = 0
    for shard_index in range(shard_count):
        local_clips = base + (1 if shard_index < remainder else 0)
        clip_end = clip_start + local_clips
        spans.append(
            (
                clip_start * clip_length,
                clip_end * clip_length,
                clip_start,
                clip_end,
            )
        )
        clip_start = clip_end
    if clip_start != clip_count:
        raise AssertionError("clip-aligned SMPL-X shards lost input clips")
    return tuple(spans)


def smplx_local_loss_numerators(
    stage: str,
    rec_output: Mapping[str, torch.Tensor],
    target_output: Mapping[str, torch.Tensor],
    *,
    clip_length: int,
    static_mask: torch.Tensor | None = None,
) -> dict[str, tuple[torch.Tensor, int]]:
    """Compute stage-exact local sums and element counts.

    No reduction crosses a helper boundary.  The caller combines every sum
    using its exact element count, which recovers the original global-mean
    weighting without transferring vertices or joints to logical0.
    """
    if stage not in _LOCAL_LOSS_STAGES:
        raise ValueError(f"unsupported local SMPL-X loss stage: {stage!r}")
    if type(clip_length) is not int or clip_length <= 0:
        raise ValueError("clip_length must be a positive exact integer")

    output_key = "joints" if stage == "lower" else "vertices"
    rec = rec_output[output_key]
    target = target_output[output_key]
    if (
        rec.shape != target.shape
        or rec.ndim != 3
        or rec.shape[0] <= 0
        or rec.shape[0] % clip_length != 0
        or not rec.requires_grad
        or target.requires_grad
    ):
        raise RuntimeError("local SMPL-X outputs violate the loss contract")

    squared = (rec - target).square()
    components: dict[str, tuple[torch.Tensor, int]] = {
        "ver": (squared.sum(), squared.numel())
    }
    if stage == "face":
        velocity_delta = (
            rec[:, 1:] - rec[:, :-1]
            - target[:, 1:] + target[:, :-1]
        )
        acceleration_delta = (
            rec[:, 2:] + rec[:, :-2] - 2 * rec[:, 1:-1]
            - target[:, 2:] - target[:, :-2] + 2 * target[:, 1:-1]
        )
        components["ver_vel"] = (
            velocity_delta.square().sum(),
            velocity_delta.numel(),
        )
        components["ver_acc"] = (
            acceleration_delta.square().sum(),
            acceleration_delta.numel(),
        )
    elif stage in {"hands", "upper"}:
        velocity_delta = (
            rec[:, 1:] - rec[:, :-1]
            - target[:, 1:] + target[:, :-1]
        )
        acceleration_delta = (
            rec[:, 2:] + rec[:, :-2] - 2 * rec[:, 1:-1]
            - target[:, 2:] - target[:, :-2] + 2 * target[:, 1:-1]
        )
        components["ver_vel"] = (
            velocity_delta.abs().sum(),
            velocity_delta.numel(),
        )
        components["ver_acc"] = (
            acceleration_delta.abs().sum(),
            acceleration_delta.numel(),
        )
    else:
        local_clips = rec.shape[0] // clip_length
        expected_mask_shape = (local_clips, clip_length, 4)
        if (
            static_mask is None
            or static_mask.dtype != torch.bool
            or tuple(static_mask.shape) != expected_mask_shape
            or static_mask.device != rec.device
            or static_mask.requires_grad
        ):
            raise RuntimeError("lower local loss requires an exact static mask")
        joints_rec = rec.reshape(local_clips, clip_length, -1, 3)
        model_feet = joints_rec[:, :, (7, 8, 10, 11)]
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:] - model_feet[:, :-1]
        )
        model_foot_v[~static_mask] = 0
        components["foot"] = (
            model_foot_v.abs().sum(),
            model_foot_v.numel(),
        )
    if any(
        numerator.ndim != 0
        or not numerator.requires_grad
        or type(count) is not int
        or count <= 0
        for numerator, count in components.values()
    ):
        raise RuntimeError("local SMPL-X loss did not produce scalar sums")
    return components


def combine_local_loss_parts(
    parts: Sequence[Mapping[str, tuple[torch.Tensor, int]]],
) -> dict[str, torch.Tensor]:
    """Combine tiny differentiable numerators with exact global counts."""
    if not parts:
        raise ValueError("local SMPL-X loss aggregation requires shards")
    component_names = tuple(parts[0])
    if not component_names or any(
        tuple(part) != component_names for part in parts
    ):
        raise ValueError("local SMPL-X shards have different components")
    combined: dict[str, torch.Tensor] = {}
    for name in component_names:
        numerators = [part[name][0] for part in parts]
        counts = [part[name][1] for part in parts]
        if any(
            numerator.ndim != 0
            or not numerator.requires_grad
            or type(count) is not int
            or count <= 0
            for numerator, count in zip(numerators, counts)
        ):
            raise RuntimeError("invalid local SMPL-X numerator/count")
        total = numerators[0]
        for numerator in numerators[1:]:
            total = total + numerator
        combined[name] = total / sum(counts)
    return combined


def _clone_numpy_rng_state(state):
    return (state[0], state[1].copy(), state[2], state[3], state[4])


@dataclass(frozen=True)
class _RngState:
    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _TargetOffload:
    outputs: dict[str, torch.Tensor]
    retained_primary_inputs: dict[str, object]
    retained_helper_inputs: dict[str, object]
    retained_helper_outputs: dict[str, torch.Tensor]
    ready: torch.cuda.Event


def _capture_rng_state() -> _RngState:
    return _RngState(
        python=random.getstate(),
        numpy=_clone_numpy_rng_state(np.random.get_state()),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=tuple(
            torch.cuda.get_rng_state(device).clone()
            for device in range(torch.cuda.device_count())
        ),
    )


def _restore_rng_state(state: _RngState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    for device, cuda_state in enumerate(state.torch_cuda):
        torch.cuda.set_rng_state(cuda_state, device=device)


def _rng_state_equal(left: _RngState, right: _RngState) -> bool:
    return (
        left.python == right.python
        and left.numpy[0] == right.numpy[0]
        and np.array_equal(left.numpy[1], right.numpy[1])
        and left.numpy[2:] == right.numpy[2:]
        and torch.equal(left.torch_cpu, right.torch_cpu)
        and len(left.torch_cuda) == len(right.torch_cuda)
        and all(
            torch.equal(left_value, right_value)
            for left_value, right_value in zip(
                left.torch_cuda,
                right.torch_cuda,
            )
        )
    )


def _module_cuda_devices(model: torch.nn.Module) -> set[int]:
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if not tensors:
        raise RuntimeError("SMPL-X pool device audit requires module tensors")
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("SMPL-X pool modules must contain only CUDA tensors")
    return {int(tensor.device.index) for tensor in tensors}


def _assert_replicas_exact(
    primary: torch.nn.Module,
    replica: torch.nn.Module,
    *,
    replica_device: int,
) -> None:
    primary_state = primary.state_dict()
    replica_state = replica.state_dict()
    if set(primary_state) != set(replica_state):
        raise RuntimeError("SMPL-X replica state keys differ from the primary")
    for name, primary_value in primary_state.items():
        replica_value = replica_state[name]
        if (
            primary_value.dtype != replica_value.dtype
            or tuple(primary_value.shape) != tuple(replica_value.shape)
            or not torch.equal(
                primary_value.detach().cpu(),
                replica_value.detach().cpu(),
            )
        ):
            raise RuntimeError(
                f"SMPL-X replica on logical{replica_device} differs at {name}"
            )


class SmplxTrainingPool:
    """Frozen SMPL-X replicas used only by the differentiable loss path."""

    def __init__(
        self,
        primary_model: torch.nn.Module,
        *,
        mode: str,
        helper_devices: str | Sequence[int],
        optimizer: torch.optim.Optimizer,
        training_model: torch.nn.Module,
    ) -> None:
        if mode not in {
            "target_offload",
            "full_batch_dual",
            "sharded_local_loss",
        }:
            raise ValueError(
                "SMPL-X pool mode must be target_offload, full_batch_dual, "
                "or sharded_local_loss"
            )
        self.mode = mode
        self.primary_device = 0
        self.helper_devices = parse_smplx_helper_devices(helper_devices)
        if not self.helper_devices:
            raise ValueError("enabled SMPL-X pooling requires explicit helpers")
        if mode == "target_offload":
            if self.helper_devices != (1,):
                raise ValueError(
                    "target_offload requires exact logical helper (1,)"
                )
        elif mode == "full_batch_dual":
            if self.helper_devices != (1, 2):
                raise ValueError(
                    "full_batch_dual requires exact logical helpers (1, 2)"
                )
        elif (
            len(self.helper_devices) < 2
            or self.helper_devices
            != tuple(range(1, len(self.helper_devices) + 1))
        ):
            raise ValueError(
                "sharded_local_loss requires at least two contiguous logical "
                "helpers starting at 1"
            )
        if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
            raise RuntimeError(
                "SMPL-X pooling requires logical CUDA device 0 as primary"
            )
        visible_devices = torch.cuda.device_count()
        if visible_devices != len(self.helper_devices) + 1:
            raise RuntimeError(
                "SMPL-X pooling requires exactly primary+helpers visible; "
                "extra visible devices would be uncontrolled or unused"
            )

        pool_devices = (self.primary_device,) + self.helper_devices
        reference_name = torch.cuda.get_device_name(self.primary_device)
        reference_capability = torch.cuda.get_device_capability(
            self.primary_device
        )
        for device in pool_devices:
            if (
                torch.cuda.get_device_name(device) != reference_name
                or torch.cuda.get_device_capability(device)
                != reference_capability
            ):
                raise RuntimeError(
                    "SMPL-X pool devices must have identical name/capability"
                )
        for source in pool_devices:
            for destination in pool_devices:
                if (
                    source != destination
                    and not torch.cuda.can_device_access_peer(
                        source,
                        destination,
                    )
                ):
                    raise RuntimeError(
                        "SMPL-X pool requires full peer access between all "
                        "selected devices"
                    )

        primary_model = freeze_smplx_for_training(primary_model)
        if _module_cuda_devices(primary_model) != {self.primary_device}:
            raise RuntimeError("primary SMPL-X must remain on logical0")
        if isinstance(
            training_model,
            torch.nn.parallel.DistributedDataParallel,
        ):
            raise RuntimeError(
                "SMPL-X pooling forbids DistributedDataParallel for RVQ/VAE"
            )
        if (
            isinstance(training_model, torch.nn.DataParallel)
            and tuple(training_model.device_ids) != (self.primary_device,)
        ):
            raise RuntimeError(
                "SMPL-X pooling requires DataParallel device_ids == [0]"
            )
        if _module_cuda_devices(training_model) != {self.primary_device}:
            raise RuntimeError("RVQ/VAE training model must remain on logical0")

        rng_before = _capture_rng_state()
        current_device = torch.cuda.current_device()
        replicas: dict[int, torch.nn.Module] = {
            self.primary_device: primary_model
        }
        streams: dict[int, torch.cuda.Stream] = {}
        try:
            for device in pool_devices:
                with torch.cuda.device(device):
                    streams[device] = torch.cuda.Stream(device=device)
                    if device == self.primary_device:
                        continue
                    replica = copy.deepcopy(primary_model).to(
                        torch.device("cuda", device)
                    )
                    replicas[device] = freeze_smplx_for_training(replica)
        finally:
            _restore_rng_state(rng_before)
            torch.cuda.set_device(current_device)
        if current_device != 0 or torch.cuda.current_device() != 0:
            raise RuntimeError("SMPL-X pool initialization changed primary device")
        if not _rng_state_equal(rng_before, _capture_rng_state()):
            raise RuntimeError("SMPL-X pool initialization changed RNG state")

        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        training_parameter_ids = {
            id(parameter) for parameter in training_model.parameters()
        }
        for device, replica in replicas.items():
            if _module_cuda_devices(replica) != {device}:
                raise RuntimeError(
                    f"SMPL-X replica is not isolated on logical{device}"
                )
            if any(parameter.requires_grad for parameter in replica.parameters()):
                raise RuntimeError("SMPL-X replicas must remain frozen")
            replica_parameter_ids = {
                id(parameter) for parameter in replica.parameters()
            }
            if (
                replica_parameter_ids & optimizer_parameter_ids
                or replica_parameter_ids & training_parameter_ids
            ):
                raise RuntimeError(
                    "SMPL-X replica leaked into the optimizer/training model"
                )
            if device != self.primary_device:
                _assert_replicas_exact(
                    primary_model,
                    replica,
                    replica_device=device,
                )

        self.replicas = replicas
        self.models = replicas
        self.streams = streams
        self._forward_rng_audited = False
        self.last_forward_receipt: dict[str, object] | None = None
        self.completed_forward_pairs = 0

    def gate_replica_modules(self) -> tuple[torch.nn.Module, ...]:
        """Expose helper replicas in explicit device order for audit gates."""
        return tuple(
            self.replicas[device] for device in self.helper_devices
        )

    def runtime_receipt(self) -> dict[str, object]:
        """Return synchronization-free topology and last-call audit data."""
        receipt = {
            "format": "semtalk_smplx_training_pool_v2",
            "mode": self.mode,
            "primary_device": self.primary_device,
            "helper_devices": list(self.helper_devices),
            "replica_devices": sorted(self.replicas),
            "partition": {
                "target_offload": (
                    "primary_stock_full_batch_reconstruction_and_"
                    "helper_full_batch_target"
                ),
                "full_batch_dual": "full_batch_dual",
                "sharded_local_loss": "balanced_contiguous_whole_clips",
            }[self.mode],
            "transfer_to_primary": {
                "target_offload": "detached_target_full_outputs_only",
                "full_batch_dual": "full_outputs",
                "sharded_local_loss": (
                    "differentiable_scalar_numerators_only"
                ),
            }[self.mode],
            "last_forward": copy.deepcopy(self.last_forward_receipt),
        }
        if self.mode == "target_offload":
            receipt["completed_forward_pairs"] = (
                self.completed_forward_pairs
            )
        return receipt

    def restore_completed_forward_pairs(self, value: int) -> None:
        """Restore the cumulative target-forward count from an audited resume."""
        if self.mode != "target_offload":
            raise RuntimeError(
                "only target_offload restores completed forward pairs"
            )
        if (
            type(value) is not int
            or value < 0
            or self.completed_forward_pairs != 0
            or self.last_forward_receipt is not None
        ):
            raise RuntimeError(
                "invalid target_offload completed-forward restore"
            )
        self.completed_forward_pairs = value

    @staticmethod
    def _validate_kwargs(
        rec_kwargs: Mapping[str, object],
        target_kwargs: Mapping[str, object],
    ) -> int:
        rec_tensors = {
            name: value
            for name, value in rec_kwargs.items()
            if torch.is_tensor(value)
        }
        target_tensors = {
            name: value
            for name, value in target_kwargs.items()
            if torch.is_tensor(value)
        }
        if not rec_tensors or not target_tensors:
            raise ValueError("SMPL-X pool requires tensor rec/target inputs")
        row_counts = {
            int(value.shape[0])
            for value in (*rec_tensors.values(), *target_tensors.values())
        }
        if len(row_counts) != 1:
            raise ValueError("all SMPL-X pool tensors must share row count")
        total_rows = row_counts.pop()
        if total_rows <= 0:
            raise ValueError("SMPL-X pool cannot process an empty batch")
        if any(
            value.device.type != "cuda" or value.device.index != 0
            for value in (*rec_tensors.values(), *target_tensors.values())
        ):
            raise RuntimeError("all SMPL-X pool inputs must start on logical0")
        if not any(value.requires_grad for value in rec_tensors.values()):
            raise RuntimeError(
                "reconstruction SMPL-X inputs must preserve an autograd path"
            )
        if any(value.requires_grad for value in target_tensors.values()):
            raise RuntimeError("target SMPL-X inputs must not require gradients")
        return total_rows

    @staticmethod
    def _copy_full_batch_kwargs(
        kwargs: Mapping[str, object],
        *,
        device: int,
    ) -> dict[str, object]:
        destination = torch.device("cuda", device)
        return {
            name: (
                value.to(destination, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for name, value in kwargs.items()
        }

    @staticmethod
    def _copy_shard_kwargs(
        kwargs: Mapping[str, object],
        *,
        row_start: int,
        row_end: int,
        device: int,
    ) -> dict[str, object]:
        destination = torch.device("cuda", device)
        return {
            name: (
                value[row_start:row_end].to(
                    destination,
                    non_blocking=True,
                )
                if torch.is_tensor(value)
                else value
            )
            for name, value in kwargs.items()
        }

    def _schedule(
        self,
        *,
        device: int,
        kwargs: Mapping[str, object],
        expected_rows: int,
        target: bool,
        output_keys: tuple[str, ...],
        inputs_ready: torch.cuda.Event,
    ) -> tuple[dict[str, torch.Tensor], torch.cuda.Event]:
        stream = self.streams[device]
        model = self.models[device]
        with torch.cuda.device(device), torch.cuda.stream(stream):
            stream.wait_event(inputs_ready)
            local_kwargs = self._copy_full_batch_kwargs(
                kwargs,
                device=device,
            )
            if target:
                output = smplx_target_forward(model, **local_kwargs)
            else:
                output = model(**local_kwargs)
            gathered: dict[str, torch.Tensor] = {}
            for key in output_keys:
                value = output[key]
                if value.shape[0] != expected_rows:
                    raise RuntimeError(
                        f"SMPL-X output {key!r} violates row ordering contract"
                    )
                gathered[key] = value.to(
                    torch.device("cuda", self.primary_device),
                    non_blocking=True,
                )
                if target and gathered[key].requires_grad:
                    raise RuntimeError("target SMPL-X output retained autograd")
                if not target and not gathered[key].requires_grad:
                    raise RuntimeError(
                        "reconstruction SMPL-X output lost autograd"
                    )
            ready = torch.cuda.Event(
                enable_timing=False,
                blocking=False,
                interprocess=False,
            )
            ready.record(stream)
        return gathered, ready

    def _schedule_target_offload(
        self,
        *,
        device: int,
        kwargs: Mapping[str, object],
        expected_rows: int,
        output_keys: tuple[str, ...],
        inputs_ready: torch.cuda.Event,
    ) -> _TargetOffload:
        """Enqueue target copies/compute on auxiliary streams only."""
        transfer_stream = self.streams[self.primary_device]
        helper_stream = self.streams[device]
        model = self.models[device]
        retained_primary_inputs = dict(kwargs)
        with (
            torch.cuda.device(self.primary_device),
            torch.cuda.stream(transfer_stream),
        ):
            transfer_stream.wait_event(inputs_ready)
            for value in retained_primary_inputs.values():
                if torch.is_tensor(value):
                    value.record_stream(transfer_stream)
            input_handoff = torch.cuda.Event(
                enable_timing=False,
                blocking=False,
                interprocess=False,
            )
            input_handoff.record(transfer_stream)
            with (
                torch.cuda.device(device),
                torch.cuda.stream(helper_stream),
            ):
                helper_stream.wait_event(input_handoff)
                helper_destination = torch.device("cuda", device)
                local_kwargs = {
                    name: (
                        value.to(
                            helper_destination,
                            non_blocking=True,
                        )
                        if torch.is_tensor(value)
                        else value
                    )
                    for name, value in retained_primary_inputs.items()
                }
                for value in local_kwargs.values():
                    if torch.is_tensor(value):
                        value.record_stream(helper_stream)
                output = smplx_target_forward(model, **local_kwargs)
                selected: dict[str, torch.Tensor] = {}
                for key in output_keys:
                    value = output[key]
                    if (
                        value.shape[0] != expected_rows
                        or value.device.type != "cuda"
                        or value.device.index != device
                        or value.requires_grad
                    ):
                        raise RuntimeError(
                            f"SMPL-X target output {key!r} violates helper "
                            "compute contract"
                        )
                    value.record_stream(helper_stream)
                    selected[key] = value
                compute_ready = torch.cuda.Event(
                    enable_timing=False,
                    blocking=False,
                    interprocess=False,
                )
                compute_ready.record(helper_stream)
                with (
                    torch.cuda.device(self.primary_device),
                    torch.cuda.stream(transfer_stream),
                ):
                    transfer_stream.wait_event(compute_ready)
                    primary_destination = torch.device(
                        "cuda",
                        self.primary_device,
                    )
                    gathered = {
                        key: value.to(
                            primary_destination,
                            non_blocking=True,
                        )
                        for key, value in selected.items()
                    }
                    for value in gathered.values():
                        value.record_stream(transfer_stream)
                    if any(
                        value.requires_grad
                        for value in gathered.values()
                    ):
                        raise RuntimeError(
                            "target SMPL-X output retained autograd"
                        )
                    return_ready = torch.cuda.Event(
                        enable_timing=False,
                        blocking=False,
                        interprocess=False,
                    )
                    return_ready.record(transfer_stream)
        return _TargetOffload(
            outputs=gathered,
            retained_primary_inputs=retained_primary_inputs,
            retained_helper_inputs=local_kwargs,
            retained_helper_outputs=selected,
            ready=return_ready,
        )

    @staticmethod
    def _collect_target_offload(
        scheduled: _TargetOffload,
    ) -> dict[str, torch.Tensor]:
        """Join target transfer after stock rec was completely enqueued."""
        primary_stream = torch.cuda.current_stream(device=0)
        primary_stream.wait_event(scheduled.ready)
        for value in scheduled.outputs.values():
            value.record_stream(primary_stream)
        # The scheduled receipt deliberately owns source inputs, helper
        # inputs, and helper outputs through the destination dependency.
        return scheduled.outputs

    @staticmethod
    def _collect_full_batch(
        scheduled: tuple[dict[str, torch.Tensor], torch.cuda.Event],
    ) -> dict[str, torch.Tensor]:
        primary_stream = torch.cuda.current_stream(device=0)
        outputs, ready = scheduled
        primary_stream.wait_event(ready)
        return outputs

    def _schedule_local_loss(
        self,
        *,
        device: int,
        stage: str,
        rec_kwargs: Mapping[str, object],
        target_kwargs: Mapping[str, object],
        row_start: int,
        row_end: int,
        clip_start: int,
        clip_end: int,
        clip_length: int,
        static_mask: torch.Tensor | None,
        inputs_ready: torch.cuda.Event,
    ) -> tuple[
        dict[str, tuple[torch.Tensor, int]],
        torch.cuda.Event,
    ]:
        stream = self.streams[device]
        model = self.models[device]
        with torch.cuda.device(device), torch.cuda.stream(stream):
            stream.wait_event(inputs_ready)
            local_rec_kwargs = self._copy_shard_kwargs(
                rec_kwargs,
                row_start=row_start,
                row_end=row_end,
                device=device,
            )
            local_target_kwargs = self._copy_shard_kwargs(
                target_kwargs,
                row_start=row_start,
                row_end=row_end,
                device=device,
            )
            rec_output = model(**local_rec_kwargs)
            target_output = smplx_target_forward(
                model,
                **local_target_kwargs,
            )
            local_static_mask = (
                None
                if static_mask is None
                else static_mask[clip_start:clip_end].to(
                    torch.device("cuda", device),
                    non_blocking=True,
                )
            )
            local_parts = smplx_local_loss_numerators(
                stage,
                rec_output,
                target_output,
                clip_length=clip_length,
                static_mask=local_static_mask,
            )
            primary_parts = {
                name: (
                    numerator.to(
                        torch.device("cuda", self.primary_device),
                        non_blocking=True,
                    ),
                    count,
                )
                for name, (numerator, count) in local_parts.items()
            }
            if any(
                numerator.ndim != 0
                or not numerator.requires_grad
                or numerator.device.type != "cuda"
                or numerator.device.index != self.primary_device
                for numerator, _ in primary_parts.values()
            ):
                raise RuntimeError(
                    "local SMPL-X loss transferred non-scalar payload"
                )
            ready = torch.cuda.Event(
                enable_timing=False,
                blocking=False,
                interprocess=False,
            )
            ready.record(stream)
        return primary_parts, ready

    def forward_pair(
        self,
        *,
        rec_kwargs: Mapping[str, object],
        target_kwargs: Mapping[str, object],
        clip_length: int,
        output_keys: Sequence[str],
        stage: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run paired forwards; only target offload requires stage evidence."""
        if self.mode not in {"target_offload", "full_batch_dual"}:
            raise RuntimeError(
                "forward_pair requires target_offload or full_batch_dual mode"
            )
        if (
            self.mode == "target_offload"
            and stage not in _LOCAL_LOSS_STAGES
        ):
            raise ValueError(
                "target_offload requires an explicit supported SMPL-X stage"
            )
        if (
            self.mode == "full_batch_dual"
            and stage is not None
            and stage not in _LOCAL_LOSS_STAGES
        ):
            raise ValueError(
                "full_batch_dual received an unsupported SMPL-X stage"
            )
        if (
            type(clip_length) is not int
            or clip_length <= 0
            or not output_keys
            or any(type(key) is not str or not key for key in output_keys)
            or len(set(output_keys)) != len(output_keys)
        ):
            raise ValueError("invalid SMPL-X pool forward contract")
        keys = tuple(output_keys)
        total_rows = self._validate_kwargs(rec_kwargs, target_kwargs)
        if total_rows % clip_length != 0:
            raise ValueError("flattened SMPL-X rows are not whole clips")
        rng_before = (
            _capture_rng_state() if not self._forward_rng_audited else None
        )
        primary_stream = torch.cuda.current_stream(device=0)
        inputs_ready = torch.cuda.Event(
            enable_timing=False,
            blocking=False,
            interprocess=False,
        )
        inputs_ready.record(primary_stream)

        if self.mode == "target_offload":
            target_device = self.helper_devices[0]
            target_scheduled = self._schedule_target_offload(
                device=target_device,
                kwargs=target_kwargs,
                expected_rows=total_rows,
                output_keys=keys,
                inputs_ready=inputs_ready,
            )
            # This is intentionally the same module, input tensors, full-batch
            # shape, device, and current logical0 stream as the stock path.
            # Only the detached target forward is offloaded.
            primary_output = self.models[self.primary_device](**rec_kwargs)
            rec_output = {key: primary_output[key] for key in keys}
            target_output = self._collect_target_offload(
                target_scheduled
            )
            self.last_forward_receipt = {
                "stage": stage,
                "total_rows": total_rows,
                "clip_length": clip_length,
                "total_clips": total_rows // clip_length,
                "helpers": [target_device],
                "reconstruction_device": self.primary_device,
                "reconstruction_execution": (
                    "stock_full_batch_primary_current_stream"
                ),
                "target_device": target_device,
                "target_execution": "detached_full_batch_helper",
                "transfer_to_primary": "detached_target_full_outputs_only",
                "output_keys": list(keys),
            }
        else:
            rec_device, target_device = self.helper_devices
            rec_scheduled = self._schedule(
                device=rec_device,
                kwargs=rec_kwargs,
                expected_rows=total_rows,
                target=False,
                output_keys=keys,
                inputs_ready=inputs_ready,
            )
            target_scheduled = self._schedule(
                device=target_device,
                kwargs=target_kwargs,
                expected_rows=total_rows,
                target=True,
                output_keys=keys,
                inputs_ready=inputs_ready,
            )
            rec_output = self._collect_full_batch(rec_scheduled)
            target_output = self._collect_full_batch(target_scheduled)
        if any(
            rec_output[key].shape[0] != total_rows
            or target_output[key].shape[0] != total_rows
            for key in keys
        ):
            raise RuntimeError("ordered SMPL-X gather lost or duplicated rows")
        if any(not rec_output[key].requires_grad for key in keys):
            raise RuntimeError("gathered reconstruction output lost autograd")
        if any(target_output[key].requires_grad for key in keys):
            raise RuntimeError("gathered target output retained autograd")

        if rng_before is not None:
            rng_after = _capture_rng_state()
            if not _rng_state_equal(rng_before, rng_after):
                raise RuntimeError("SMPL-X pool forward changed RNG state")
            self._forward_rng_audited = True
        if self.mode == "target_offload":
            self.completed_forward_pairs += 1
        return rec_output, target_output

    def forward_local_losses(
        self,
        *,
        stage: str,
        rec_kwargs: Mapping[str, object],
        target_kwargs: Mapping[str, object],
        clip_length: int,
        static_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute sharded stage losses and return only global means."""
        if self.mode != "sharded_local_loss":
            raise RuntimeError(
                "forward_local_losses requires sharded_local_loss mode"
            )
        if stage not in _LOCAL_LOSS_STAGES:
            raise ValueError(f"unsupported local loss stage: {stage!r}")
        if type(clip_length) is not int or clip_length <= 0:
            raise ValueError("clip_length must be a positive exact integer")
        total_rows = self._validate_kwargs(rec_kwargs, target_kwargs)
        spans = clip_aligned_spans(
            total_rows,
            clip_length,
            len(self.helper_devices),
        )
        total_clips = total_rows // clip_length
        if stage == "lower":
            if (
                static_mask is None
                or static_mask.dtype != torch.bool
                or tuple(static_mask.shape) != (total_clips, clip_length, 4)
                or static_mask.device.type != "cuda"
                or static_mask.device.index != self.primary_device
                or static_mask.requires_grad
            ):
                raise RuntimeError(
                    "lower sharded local loss requires a logical0 bool mask"
                )
        elif static_mask is not None:
            raise RuntimeError("only lower local loss accepts static_mask")

        rng_before = (
            _capture_rng_state() if not self._forward_rng_audited else None
        )
        primary_stream = torch.cuda.current_stream(device=0)
        inputs_ready = torch.cuda.Event(
            enable_timing=False,
            blocking=False,
            interprocess=False,
        )
        inputs_ready.record(primary_stream)
        scheduled = tuple(
            self._schedule_local_loss(
                device=device,
                stage=stage,
                rec_kwargs=rec_kwargs,
                target_kwargs=target_kwargs,
                row_start=row_start,
                row_end=row_end,
                clip_start=clip_start,
                clip_end=clip_end,
                clip_length=clip_length,
                static_mask=static_mask,
                inputs_ready=inputs_ready,
            )
            for device, (
                row_start,
                row_end,
                clip_start,
                clip_end,
            ) in zip(self.helper_devices, spans)
        )
        for _, ready in scheduled:
            primary_stream.wait_event(ready)
        parts = tuple(part for part, _ in scheduled)
        combined = combine_local_loss_parts(parts)
        if any(
            scalar.ndim != 0
            or not scalar.requires_grad
            or scalar.device.type != "cuda"
            or scalar.device.index != self.primary_device
            for scalar in combined.values()
        ):
            raise RuntimeError(
                "sharded local SMPL-X loss did not return primary scalars"
            )

        self.last_forward_receipt = {
            "stage": stage,
            "total_rows": total_rows,
            "clip_length": clip_length,
            "total_clips": total_clips,
            "helpers": list(self.helper_devices),
            "spans": [
                {
                    "device": device,
                    "row_start": row_start,
                    "row_end": row_end,
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                }
                for device, (
                    row_start,
                    row_end,
                    clip_start,
                    clip_end,
                ) in zip(self.helper_devices, spans)
            ],
            "component_counts": {
                name: [part[name][1] for part in parts]
                for name in combined
            },
            "aggregation": "ordered_numerator_sum_over_exact_global_count",
        }
        if rng_before is not None:
            rng_after = _capture_rng_state()
            if not _rng_state_equal(rng_before, rng_after):
                raise RuntimeError(
                    "sharded local SMPL-X forward changed RNG state"
                )
            self._forward_rng_audited = True
        return combined


def build_smplx_training_pool(
    args,
    primary_model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    training_model: torch.nn.Module,
) -> SmplxTrainingPool | None:
    """Build the optional pool and reject ambiguous disabled configuration."""
    mode = getattr(args, "smplx_training_pool_mode", "disabled")
    helpers = getattr(args, "smplx_training_helper_devices", "")
    if type(mode) is not str or mode not in _POOL_MODES:
        raise ValueError(f"unsupported SMPL-X training pool mode: {mode!r}")
    parsed_helpers = parse_smplx_helper_devices(helpers)
    if mode == "disabled":
        if parsed_helpers:
            raise ValueError(
                "SMPL-X helpers were supplied while pooling is disabled"
            )
        return None
    if not getattr(args, "train_only", False):
        raise RuntimeError("SMPL-X training pooling is training-only")
    if getattr(args, "rec_ver_weight", 0) <= 0:
        raise RuntimeError("SMPL-X training pooling requires vertex loss")
    return SmplxTrainingPool(
        primary_model,
        mode=mode,
        helper_devices=parsed_helpers,
        optimizer=optimizer,
        training_model=training_model,
    )
