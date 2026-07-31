from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SEMTALK = REPOSITORY / "models" / "semtalk.py"

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised by minimal CPU-only hosts
    torch = None
    F = None


def _original_rhythmic_loss(
    facial_features,
    audio_features,
    *,
    temperature: float,
):
    """Replay the pre-fix [B, C=T, T] cross-entropy semantics exactly."""

    pooled_audio = F.avg_pool1d(
        audio_features.permute(0, 2, 1), kernel_size=4
    ).permute(0, 2, 1)
    normalized_facial = F.normalize(facial_features, p=2, dim=-1)
    normalized_audio = F.normalize(pooled_audio, p=2, dim=-1)
    similarity = torch.matmul(
        normalized_facial, normalized_audio.transpose(-1, -2)
    ) / temperature
    batch_size, num_frames, _ = facial_features.shape
    labels = (
        torch.arange(num_frames, device=facial_features.device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    return F.cross_entropy(similarity, labels)


class RhythmicIdentificationLossStaticContractTest(unittest.TestCase):
    def test_cross_entropy_is_2d_and_preserves_original_dim1_class_axis(self) -> None:
        tree = ast.parse(SEMTALK.read_text(encoding="utf-8"), str(SEMTALK))
        rhythmic = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "RhythmicIdentificationLoss"
        )
        forward = next(
            node
            for node in rhythmic.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )

        assignments = {
            target.id: node.value
            for node in forward.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        flattened = assignments["similarity_matrix_2d"]
        self.assertIsInstance(flattened, ast.Call)
        self.assertIsInstance(flattened.func, ast.Attribute)
        self.assertEqual(flattened.func.attr, "reshape")
        self.assertEqual(len(flattened.args), 2)
        self.assertIsInstance(flattened.args[0], ast.UnaryOp)
        self.assertIsInstance(flattened.args[0].op, ast.USub)
        self.assertEqual(flattened.args[0].operand.value, 1)
        self.assertIsInstance(flattened.args[1], ast.Name)
        self.assertEqual(flattened.args[1].id, "num_frames")

        transpose = flattened.func.value
        self.assertIsInstance(transpose, ast.Call)
        self.assertIsInstance(transpose.func, ast.Attribute)
        self.assertEqual(transpose.func.attr, "transpose")
        self.assertIsInstance(transpose.func.value, ast.Name)
        self.assertEqual(transpose.func.value.id, "similarity_matrix")
        self.assertEqual([argument.value for argument in transpose.args], [1, 2])

        flattened_labels = assignments["labels_1d"]
        self.assertIsInstance(flattened_labels, ast.Call)
        self.assertIsInstance(flattened_labels.func, ast.Attribute)
        self.assertEqual(flattened_labels.func.attr, "reshape")
        self.assertIsInstance(flattened_labels.func.value, ast.Name)
        self.assertEqual(flattened_labels.func.value.id, "labels")

        calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "F"
            and node.func.attr == "cross_entropy"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [argument.id for argument in calls[0].args],
            ["similarity_matrix_2d", "labels_1d"],
        )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class RhythmicIdentificationLossEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from models.semtalk import RhythmicIdentificationLoss
        except ImportError as error:
            raise unittest.SkipTest(
                f"SemTalk model dependencies are unavailable: {error}"
            ) from error
        cls.loss_type = RhythmicIdentificationLoss

    def test_forward_and_both_input_gradients_match_original_semantics(self) -> None:
        cases = (
            (1, 2, 3, torch.float32),
            (2, 5, 7, torch.float32),
            (1, 3, 4, torch.float64),
            (3, 7, 5, torch.float64),
        )
        for batch_size, num_frames, feature_dim, dtype in cases:
            with self.subTest(
                batch_size=batch_size,
                num_frames=num_frames,
                feature_dim=feature_dim,
                dtype=str(dtype),
            ):
                generator = torch.Generator(device="cpu").manual_seed(
                    9137 + batch_size * 101 + num_frames * 11 + feature_dim
                )
                facial_seed = torch.randn(
                    batch_size,
                    num_frames,
                    feature_dim,
                    dtype=dtype,
                    generator=generator,
                )
                audio_seed = torch.randn(
                    batch_size,
                    num_frames * 4,
                    feature_dim,
                    dtype=dtype,
                    generator=generator,
                )
                reference_facial = facial_seed.clone().requires_grad_(True)
                reference_audio = audio_seed.clone().requires_grad_(True)
                candidate_facial = facial_seed.clone().requires_grad_(True)
                candidate_audio = audio_seed.clone().requires_grad_(True)

                reference = _original_rhythmic_loss(
                    reference_facial,
                    reference_audio,
                    temperature=0.13,
                )
                candidate = self.loss_type(temperature=0.13)(
                    candidate_facial,
                    candidate_audio,
                )
                reference_gradients = torch.autograd.grad(
                    reference, (reference_facial, reference_audio)
                )
                candidate_gradients = torch.autograd.grad(
                    candidate, (candidate_facial, candidate_audio)
                )

                tolerance = 1e-6 if dtype == torch.float32 else 1e-12
                torch.testing.assert_close(
                    candidate,
                    reference,
                    rtol=tolerance,
                    atol=tolerance,
                )
                for actual, expected in zip(
                    candidate_gradients, reference_gradients
                ):
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is unavailable",
    )
    def test_cuda_deterministic_flatten_matches_original_and_replays_exactly(
        self,
    ) -> None:
        self.assertTrue(
            torch.cuda.is_bf16_supported(),
            "the formal H200 check requires native bfloat16 support",
        )
        deterministic_before = torch.are_deterministic_algorithms_enabled()
        warn_only_before = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        cases = (
            # The tighter fp32 bounds allow only kernel-level round-off.
            (torch.float32, 2e-6, 2e-7, 2e-5, 2e-7),
            # BF16 has a 7-bit mantissa, so its explicit bounds reflect one
            # low-precision reduction while remaining far below loss scale.
            (torch.bfloat16, 2e-2, 2e-2, 3e-2, 3e-3),
        )
        try:
            generator = torch.Generator(device="cpu").manual_seed(228256)
            facial_seed = torch.randn(
                2, 22, 256, dtype=torch.float32, generator=generator
            )
            audio_seed = torch.randn(
                2, 88, 256, dtype=torch.float32, generator=generator
            )
            for (
                dtype,
                loss_rtol,
                loss_atol,
                gradient_rtol,
                gradient_atol,
            ) in cases:
                with self.subTest(dtype=str(dtype)):
                    facial = facial_seed.to(device="cuda", dtype=dtype)
                    audio = audio_seed.to(device="cuda", dtype=dtype)

                    torch.use_deterministic_algorithms(False)
                    reference_facial = facial.clone().requires_grad_(True)
                    reference_audio = audio.clone().requires_grad_(True)
                    reference_loss = _original_rhythmic_loss(
                        reference_facial,
                        reference_audio,
                        temperature=0.1,
                    )
                    reference_gradients = torch.autograd.grad(
                        reference_loss,
                        (reference_facial, reference_audio),
                    )

                    torch.use_deterministic_algorithms(True)
                    candidate_results = []
                    for _ in range(2):
                        candidate_facial = facial.clone().requires_grad_(True)
                        candidate_audio = audio.clone().requires_grad_(True)
                        candidate_loss = self.loss_type(temperature=0.1)(
                            candidate_facial,
                            candidate_audio,
                        )
                        candidate_gradients = torch.autograd.grad(
                            candidate_loss,
                            (candidate_facial, candidate_audio),
                        )
                        candidate_results.append(
                            (
                                candidate_loss.detach().clone(),
                                tuple(
                                    gradient.detach().clone()
                                    for gradient in candidate_gradients
                                ),
                            )
                        )

                    first_loss, first_gradients = candidate_results[0]
                    second_loss, second_gradients = candidate_results[1]
                    self.assertTrue(torch.equal(first_loss, second_loss))
                    for first, second in zip(
                        first_gradients, second_gradients
                    ):
                        self.assertTrue(torch.equal(first, second))

                    torch.testing.assert_close(
                        first_loss,
                        reference_loss.detach(),
                        rtol=loss_rtol,
                        atol=loss_atol,
                    )
                    for actual, expected in zip(
                        first_gradients, reference_gradients
                    ):
                        torch.testing.assert_close(
                            actual,
                            expected,
                            rtol=gradient_rtol,
                            atol=gradient_atol,
                        )
        finally:
            torch.use_deterministic_algorithms(
                deterministic_before,
                warn_only=warn_only_before,
            )


if __name__ == "__main__":
    unittest.main()
