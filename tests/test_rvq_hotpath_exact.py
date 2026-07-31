import copy
import unittest
from types import SimpleNamespace

import torch


class RVQFiniteDiagnosticExactTest(unittest.TestCase):
    def test_default_fast_path_matches_diagnostic_path_exactly(self):
        # The upstream quantizer constructs its initial codebook with
        # Tensor.cuda().  This test-only shim exercises the same mathematics
        # in the CPU/Gloo test environment without changing production code.
        original_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
        try:
            from models.rvq import RVQVAE

            args = SimpleNamespace(
                vae_test_dim=9,
                rvq_check_finite_every_step=False,
            )
            torch.manual_seed(11)
            fast = RVQVAE(
                args,
                nb_code=32,
                code_dim=16,
                output_emb_width=16,
                down_t=2,
                stride_t=2,
                width=32,
                depth=1,
            )
            checked = copy.deepcopy(fast)
            checked.check_finite_every_step = True
            checked.quantizer.check_finite_every_step = True
            fast.train()
            checked.train()

            fast_input = torch.randn(4, 64, 9, requires_grad=True)
            checked_input = fast_input.detach().clone().requires_grad_(True)
            torch.manual_seed(23)
            fast_output = fast(fast_input)
            fast_loss = (
                fast_output["rec_pose"].square().mean()
                + fast_output["embedding_loss"]
            )
            fast_loss.backward()
            torch.manual_seed(23)
            checked_output = checked(checked_input)
            checked_loss = (
                checked_output["rec_pose"].square().mean()
                + checked_output["embedding_loss"]
            )
            checked_loss.backward()

            self.assertTrue(
                torch.equal(
                    fast_output["rec_pose"],
                    checked_output["rec_pose"],
                )
            )
            self.assertTrue(
                torch.equal(
                    fast_output["embedding_loss"],
                    checked_output["embedding_loss"],
                )
            )
            self.assertTrue(torch.equal(fast_input.grad, checked_input.grad))
            for (fast_name, fast_param), (
                checked_name,
                checked_param,
            ) in zip(fast.named_parameters(), checked.named_parameters()):
                self.assertEqual(fast_name, checked_name)
                self.assertTrue(
                    torch.equal(fast_param.grad, checked_param.grad),
                    fast_name,
                )
            for (fast_name, fast_state), (
                checked_name,
                checked_state,
            ) in zip(fast.state_dict().items(), checked.state_dict().items()):
                self.assertEqual(fast_name, checked_name)
                self.assertTrue(
                    torch.equal(fast_state, checked_state),
                    fast_name,
                )
            for fast_layer, checked_layer in zip(
                fast.quantizer.layers,
                checked.quantizer.layers,
            ):
                self.assertTrue(
                    torch.equal(
                        fast_layer.code_sum,
                        checked_layer.code_sum,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        fast_layer.code_count,
                        checked_layer.code_count,
                    )
                )
        finally:
            torch.Tensor.cuda = original_cuda


if __name__ == "__main__":
    unittest.main()
