from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - minimal local environments
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]


class ShowGlobalAsyncFiniteStaticContractTests(unittest.TestCase):
    def test_global_fastpath_has_no_host_boolean_finite_check(self) -> None:
        trainer_source = (REPO_ROOT / "aelowerfoot_trainer.py").read_text(
            encoding="utf-8"
        )
        trainer_tree = ast.parse(trainer_source)
        fastpath = next(
            node
            for node in trainer_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "global_foot_fastpath_losses"
        )

        async_calls = [
            node
            for node in ast.walk(fastpath)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_all_finite_async"
        ]
        self.assertEqual(len(async_calls), 1)
        self.assertEqual(
            ast.dump(async_calls[0].args[0]),
            ast.dump(ast.Name(id="lower_foot_local", ctx=ast.Load())),
        )
        self.assertNotIn(
            "if not torch.isfinite(lower_foot_local).all()",
            trainer_source,
        )
        finite_branches = [
            node
            for branch in ast.walk(fastpath)
            if isinstance(branch, ast.If)
            for node in ast.walk(branch.test)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "isfinite"
        ]
        self.assertEqual(finite_branches, [])

    def test_helper_uses_torch_async_assert_without_scalar_materialization(
        self,
    ) -> None:
        utility_source = (
            REPO_ROOT / "utils" / "show_base_tensor_ops.py"
        ).read_text(encoding="utf-8")
        utility_tree = ast.parse(utility_source)
        helper = next(
            node
            for node in utility_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "assert_all_finite_async"
        )

        async_calls = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "_assert_async"
        ]
        self.assertEqual(len(async_calls), 1)
        predicate = async_calls[0].args[0]
        self.assertIsInstance(predicate, ast.Call)
        self.assertIsInstance(predicate.func, ast.Attribute)
        self.assertEqual(predicate.func.attr, "all")
        finite_call = predicate.func.value
        self.assertIsInstance(finite_call, ast.Call)
        self.assertIsInstance(finite_call.func, ast.Attribute)
        self.assertIsInstance(finite_call.func.value, ast.Name)
        self.assertEqual(finite_call.func.value.id, "torch")
        self.assertEqual(finite_call.func.attr, "isfinite")

        forbidden_scalar_calls = {
            (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
            )
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"item", "tolist", "numpy"}
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "bool"
                )
            )
        }
        self.assertEqual(forbidden_scalar_calls, set())


@unittest.skipIf(torch is None, "torch is unavailable")
class ShowGlobalAsyncFiniteCpuContractTests(unittest.TestCase):
    def test_finite_tensor_passes_without_mutation(self) -> None:
        from utils.show_base_tensor_ops import assert_all_finite_async

        value = torch.tensor(
            [[0.0, -1.5, 2.25], [4.0, 5.5, -6.75]],
            dtype=torch.float32,
        )
        before = value.clone()
        result = assert_all_finite_async(
            value,
            field_name="lower_foot_local",
        )
        self.assertIsNone(result)
        self.assertTrue(torch.equal(value, before))

    def test_non_finite_tensor_raises_the_existing_contract_message(
        self,
    ) -> None:
        from utils.show_base_tensor_ops import assert_all_finite_async

        message = re.escape(
            "lower_foot_local contains non-finite values"
        )
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                value = torch.tensor(
                    [0.0, invalid, 1.0],
                    dtype=torch.float32,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    assert_all_finite_async(
                        value,
                        field_name="lower_foot_local",
                    )


if __name__ == "__main__":
    unittest.main()
