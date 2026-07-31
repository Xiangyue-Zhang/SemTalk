from __future__ import annotations

import unittest

class ResidualVQAutogradContractTest(unittest.TestCase):
    def test_residual_updates_preserve_saved_tensors_for_backward(self) -> None:
        try:
            import torch
            from torch import nn

            from models.residual_vq import ResidualVQ
        except ModuleNotFoundError as error:
            self.skipTest(f"optional torch test dependency unavailable: {error}")

        class DifferentiableDummyQuantizer(nn.Module):
            """Exercise the saved-tensor lifetime of an EMA layer."""

            def forward(
                self,
                residual: torch.Tensor,
                *,
                return_idx: bool,
                temperature: float | None,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                del temperature
                if not return_idx:
                    raise AssertionError("ResidualVQ must request indices")
                batch, _, frames = residual.shape
                quantized = residual * 0.25
                indices = torch.zeros(
                    (batch, frames),
                    dtype=torch.long,
                    device=residual.device,
                )
                commitment = residual.square().mean()
                perplexity = residual.new_tensor(1.0)
                return quantized, indices, commitment, perplexity

        quantizer = ResidualVQ.__new__(ResidualVQ)
        nn.Module.__init__(quantizer)
        quantizer.num_quantizers = 3
        quantizer.quantize_dropout_prob = 0.0
        quantizer.quantize_dropout_cutoff_index = 0
        quantizer.layers = nn.ModuleList(
            [DifferentiableDummyQuantizer() for _ in range(3)]
        )
        # Keep this legacy-style fixture free of fields introduced after the
        # original ResidualVQ contract.  Missing diagnostics must default off.
        self.assertFalse(hasattr(quantizer, "check_finite_every_step"))

        leaf = torch.randn(2, 8, 16, requires_grad=True)
        encoded = leaf * 1.0
        reconstruction, _, commitment, _ = quantizer(encoded)
        objective = reconstruction.square().mean() + commitment
        objective.backward()

        self.assertIsNotNone(leaf.grad)
        self.assertTrue(torch.isfinite(leaf.grad).all())


if __name__ == "__main__":
    unittest.main()
