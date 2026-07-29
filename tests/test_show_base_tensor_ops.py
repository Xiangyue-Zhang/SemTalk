from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - minimal local environments
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class ShowBaseTensorOpsTests(unittest.TestCase):
    def _legacy(self, filtered, selection):
        selection_tensor = torch.from_numpy(selection)
        restored = torch.zeros((filtered.shape[0], selection.size))
        selected_indices = torch.where(selection_tensor == 1)[0]
        for row in range(filtered.shape[0]):
            restored[row, selected_indices] = filtered[row]
        return restored

    def test_forward_and_gradient_match_legacy_exactly(self):
        from utils.show_base_tensor_ops import inverse_selection_tensor

        selection = np.zeros(165, dtype=np.int64)
        selection[[0, 2, 66, 75, 119, 164]] = 1
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        legacy_input = source.clone().requires_grad_(True)
        optimized_input = source.clone().requires_grad_(True)

        legacy = self._legacy(legacy_input, selection)
        optimized = inverse_selection_tensor(
            optimized_input,
            selection,
            rows=4,
        )
        self.assertTrue(torch.equal(legacy, optimized))

        weights = torch.arange(4 * 165, dtype=torch.float32).reshape(4, 165)
        (legacy * weights).sum().backward()
        (optimized * weights).sum().backward()
        self.assertTrue(
            torch.equal(legacy_input.grad, optimized_input.grad)
        )

    def test_contract_rejects_bad_shapes_and_dtype(self):
        from utils.show_base_tensor_ops import inverse_selection_tensor

        selection = np.zeros(165, dtype=np.int64)
        selection[:3] = 1
        with self.assertRaises(ValueError):
            inverse_selection_tensor(
                torch.zeros(2, 3, dtype=torch.float32),
                selection,
                rows=3,
            )
        with self.assertRaises(TypeError):
            inverse_selection_tensor(
                torch.zeros(2, 3, dtype=torch.float64),
                selection,
                rows=2,
            )
        with self.assertRaises(ValueError):
            inverse_selection_tensor(
                torch.zeros(2, 4, dtype=torch.float32),
                selection,
                rows=2,
            )
