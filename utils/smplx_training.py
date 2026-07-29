"""Small invariants for differentiable SMPL-X training losses."""

from __future__ import annotations

from typing import TypeVar

import torch


_Module = TypeVar("_Module")


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
