from __future__ import annotations

import os
from pathlib import Path
import unittest


try:
    import smplx
    import torch
except ImportError:  # pragma: no cover - lightweight local environment
    smplx = None
    torch = None


MODEL_ROOT = os.environ.get("SEMTALK_TEST_SMPLX_MODEL_ROOT")


@unittest.skipIf(
    torch is None or smplx is None or not MODEL_ROOT,
    "torch, smplx, and SEMTALK_TEST_SMPLX_MODEL_ROOT are required",
)
class ActualSMPLXFastpathTests(unittest.TestCase):
    def test_outputs_and_input_gradients_are_byte_exact(self):
        from utils.smplx_training import (
            freeze_smplx_for_training,
            smplx_target_forward,
        )

        model_root = Path(MODEL_ROOT)
        model = smplx.create(
            str(model_root),
            model_type="smplx",
            gender="NEUTRAL_2020",
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext="npz",
            use_pca=False,
        ).cpu().eval()
        batch = 1

        constant = {
            "betas": torch.linspace(-0.01, 0.01, 300).reshape(batch, 300),
            "transl": torch.tensor([[0.1, -0.2, 0.3]]),
            "leye_pose": torch.tensor([[0.01, 0.02, -0.03]]),
            "reye_pose": torch.tensor([[-0.02, 0.01, 0.03]]),
        }
        reference_leaves = {
            "global_orient": torch.tensor(
                [[0.02, -0.03, 0.01]], requires_grad=True
            ),
            "body_pose": torch.linspace(-0.04, 0.04, 63)
            .reshape(batch, 63)
            .requires_grad_(True),
            "jaw_pose": torch.tensor(
                [[0.03, -0.02, 0.01]], requires_grad=True
            ),
            "left_hand_pose": torch.linspace(-0.03, 0.03, 45)
            .reshape(batch, 45)
            .requires_grad_(True),
            "right_hand_pose": torch.linspace(0.03, -0.03, 45)
            .reshape(batch, 45)
            .requires_grad_(True),
            "expression": torch.linspace(-0.02, 0.02, 100)
            .reshape(batch, 100)
            .requires_grad_(True),
        }
        candidate_leaves = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in reference_leaves.items()
        }
        target_kwargs = {
            **constant,
            **{
                name: value.detach().clone().mul_(0.5)
                for name, value in reference_leaves.items()
            },
            "return_verts": True,
            "return_joints": True,
        }

        reference_rec = model(
            **constant,
            **reference_leaves,
            return_verts=True,
            return_joints=True,
            return_shaped=True,
        )
        reference_target = model(**target_kwargs, return_shaped=True)
        reference_loss = torch.nn.functional.mse_loss(
            reference_rec.vertices,
            reference_target.vertices,
        ) + torch.nn.functional.mse_loss(
            reference_rec.joints,
            reference_target.joints,
        )
        reference_loss.backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))

        model = freeze_smplx_for_training(model)
        candidate_rec = model(
            **constant,
            **candidate_leaves,
            return_verts=True,
            return_joints=True,
            return_shaped=False,
        )
        candidate_target = smplx_target_forward(model, **target_kwargs)
        candidate_loss = torch.nn.functional.mse_loss(
            candidate_rec.vertices,
            candidate_target.vertices,
        ) + torch.nn.functional.mse_loss(
            candidate_rec.joints,
            candidate_target.joints,
        )
        candidate_loss.backward()

        self.assertTrue(
            torch.equal(reference_rec.vertices, candidate_rec.vertices)
        )
        self.assertTrue(torch.equal(reference_rec.joints, candidate_rec.joints))
        self.assertTrue(
            torch.equal(reference_target.vertices, candidate_target.vertices)
        )
        self.assertTrue(
            torch.equal(reference_target.joints, candidate_target.joints)
        )
        self.assertTrue(torch.equal(reference_loss, candidate_loss))
        for name in reference_leaves:
            self.assertTrue(
                torch.equal(
                    reference_leaves[name].grad,
                    candidate_leaves[name].grad,
                ),
                name,
            )
        self.assertIsNotNone(reference_rec.v_shaped)
        self.assertIsNone(candidate_rec.v_shaped)
        self.assertIsNone(candidate_target.v_shaped)
        self.assertFalse(candidate_target.vertices.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
