"""Small tensor operations used by the formal SHOW Base trainers."""

from __future__ import annotations

import numpy as np
import torch


def assert_all_finite_async(
    tensor: torch.Tensor,
    *,
    field_name: str,
) -> None:
    """Enqueue a finite-value assertion without synchronizing CUDA to host.

    ``torch._assert_async`` is available in the pinned Torch 2.6 runtime.  On
    CUDA it checks the scalar condition on the current stream; on CPU it
    raises ``RuntimeError`` immediately, which keeps the contract directly
    testable without a GPU.
    """

    torch._assert_async(
        torch.isfinite(tensor).all(),
        f"{field_name} contains non-finite values",
    )


def inverse_selection_tensor(
    filtered: torch.Tensor,
    selection: np.ndarray,
    rows: int,
    *,
    output_size: int = 165,
) -> torch.Tensor:
    """Restore selected pose columns without a per-row CUDA launch.

    The legacy trainers copied every row independently in Python.  Advanced
    indexing performs the same zero-fill and column copies in one operation,
    while preserving autograd from ``filtered``.
    """

    if filtered.ndim != 2:
        raise ValueError(
            f"filtered tensor must be rank two, got {tuple(filtered.shape)}"
        )
    if rows != filtered.shape[0]:
        raise ValueError(
            f"row count mismatch: requested={rows}, tensor={filtered.shape[0]}"
        )
    selection_array = np.asarray(selection)
    if selection_array.ndim != 1 or selection_array.size != output_size:
        raise ValueError(
            "selection mask must be one-dimensional with "
            f"{output_size} entries, got {selection_array.shape}"
        )
    if filtered.dtype != torch.float32:
        raise TypeError(
            "formal SHOW inverse selection requires float32, "
            f"got {filtered.dtype}"
        )

    selection_tensor = torch.from_numpy(selection_array).to(
        device=filtered.device
    )
    selected_indices = torch.where(selection_tensor == 1)[0]
    if filtered.shape[1] != selected_indices.numel():
        raise ValueError(
            "filtered width does not match selected columns: "
            f"{filtered.shape[1]} != {selected_indices.numel()}"
        )
    restored = torch.zeros(
        (rows, output_size),
        dtype=torch.float32,
        device=filtered.device,
    )
    restored[:, selected_indices] = filtered
    return restored
