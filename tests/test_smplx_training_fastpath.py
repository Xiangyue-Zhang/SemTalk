from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAINERS = (
    ROOT / "aeface_trainer.py",
    ROOT / "ae_trainer.py",
    ROOT / "aelower_trainer.py",
)
SMPLX_PARAMETER_INPUTS = {
    "betas",
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "transl",
    "expression",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
}


class SMPLXTrainerStaticContractTests(unittest.TestCase):
    def test_only_differentiable_calls_are_direct_and_skip_v_shaped(self):
        for path in TRAINERS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            direct_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "smplx"
            ]
            self.assertEqual(len(direct_calls), 2, path.name)
            for call in direct_calls:
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                self.assertTrue(
                    SMPLX_PARAMETER_INPUTS.issubset(keywords),
                    path.name,
                )
                self.assertIn("return_shaped", keywords, path.name)
                self.assertIsInstance(keywords["return_shaped"], ast.Constant)
                self.assertIs(keywords["return_shaped"].value, False)

    def test_target_calls_use_the_detached_helper(self):
        for path in TRAINERS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            target_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "smplx_target_forward"
            ]
            self.assertEqual(len(target_calls), 2, path.name)
            for call in target_calls:
                self.assertGreaterEqual(len(call.args), 1)
                model = call.args[0]
                self.assertIsInstance(model, ast.Attribute)
                self.assertEqual(model.attr, "smplx")
                keywords = {keyword.arg for keyword in call.keywords}
                self.assertTrue(
                    SMPLX_PARAMETER_INPUTS.issubset(keywords),
                    path.name,
                )

    def test_all_formal_trainers_freeze_smplx(self):
        for path in TRAINERS:
            source = path.read_text(encoding="utf-8")
            self.assertIn("freeze_smplx_for_training(", source, path.name)


try:
    import torch
except ImportError:  # pragma: no cover - local lightweight test environment
    torch = None


@unittest.skipIf(torch is None, "torch is required for the autograd contract")
class SMPLXTrainingAutogradContractTests(unittest.TestCase):
    def test_freezing_preserves_input_gradient_and_removes_parameter_grads(self):
        from utils.smplx_training import freeze_smplx_for_training

        class ToySMPLX(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unused_pose = torch.nn.Parameter(torch.tensor([7.0]))
                self.register_buffer("scale", torch.tensor([3.0]))

            def forward(self, pose, return_shaped=True):
                primary = pose.square() * self.scale
                shaped = pose + 1 if return_shaped else None
                return {"primary": primary, "v_shaped": shaped}

        reference = ToySMPLX()
        candidate = ToySMPLX()
        candidate.load_state_dict(reference.state_dict())
        pose_reference = torch.tensor([2.0, -4.0], requires_grad=True)
        pose_candidate = pose_reference.detach().clone().requires_grad_(True)

        reference_output = reference(pose_reference, return_shaped=False)
        candidate = freeze_smplx_for_training(candidate)
        candidate_output = candidate(pose_candidate, return_shaped=False)
        self.assertTrue(
            torch.equal(
                reference_output["primary"],
                candidate_output["primary"],
            )
        )

        reference_output["primary"].sum().backward()
        candidate_output["primary"].sum().backward()
        self.assertTrue(torch.equal(pose_reference.grad, pose_candidate.grad))
        self.assertIsNone(reference.unused_pose.grad)
        self.assertIsNone(candidate.unused_pose.grad)
        self.assertFalse(candidate.training)
        self.assertFalse(any(p.requires_grad for p in candidate.parameters()))

    def test_target_forward_is_detached_and_omits_v_shaped(self):
        from utils.smplx_training import smplx_target_forward

        class ToySMPLX(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.return_shaped_values = []

            def forward(self, pose, return_shaped=True):
                self.return_shaped_values.append(return_shaped)
                return {
                    "primary": pose * 2,
                    "v_shaped": pose + 1 if return_shaped else None,
                }

        model = ToySMPLX()
        target = torch.tensor([1.0], requires_grad=True)
        output = smplx_target_forward(model, pose=target)
        self.assertEqual(model.return_shaped_values, [False])
        self.assertFalse(output["primary"].requires_grad)
        self.assertIsNone(output["primary"].grad_fn)
        self.assertIsNone(output["v_shaped"])


if __name__ == "__main__":
    unittest.main()
