from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SYNC_CALL_SOURCES = (
    REPOSITORY / "scripts" / "show_base" / "build_representation_lmdb.py",
    REPOSITORY
    / "scripts"
    / "show_base"
    / "build_lower_target_joints_cache.py",
    REPOSITORY / "scripts" / "show_base" / "build_base_features.py",
)


class PositionalOnlyEnvironment:
    def __init__(self) -> None:
        self.forces: list[bool] = []

    def sync(self, force: bool, /) -> None:
        self.forces.append(force)


class LmdbSyncCompatibilityTest(unittest.TestCase):
    def test_all_formal_builders_use_positional_force_true(self) -> None:
        for path in SYNC_CALL_SOURCES:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("sync(force=True)", source)
                calls = [
                    node
                    for node in ast.walk(ast.parse(source, filename=str(path)))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "sync"
                ]
                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertEqual(call.keywords, [])
                self.assertEqual(len(call.args), 1)
                self.assertIsInstance(call.args[0], ast.Constant)
                self.assertIs(call.args[0].value, True)
                self.assertIsInstance(call.func.value, ast.Name)

                environment = PositionalOnlyEnvironment()
                expression = ast.Expression(body=call)
                ast.fix_missing_locations(expression)
                eval(
                    compile(expression, str(path), "eval"),
                    {call.func.value.id: environment},
                )
                self.assertEqual(environment.forces, [True])


if __name__ == "__main__":
    unittest.main()
