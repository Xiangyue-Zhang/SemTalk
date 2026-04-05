#!/usr/bin/env python3

import ast
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relpath: str):
    return yaml.safe_load((REPO_ROOT / relpath).read_text(encoding="utf-8"))


def parse_add_argument_defaults(script_path: Path) -> dict[str, object]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    defaults = {}

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                if not node.args:
                    return
                first = node.args[0]
                if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                    return
                option = first.value
                if not option.startswith("--"):
                    return
                default = None
                has_default = False
                for kw in node.keywords:
                    if kw.arg == "default":
                        has_default = True
                        try:
                            default = ast.literal_eval(kw.value)
                        except Exception:
                            default = None
                if has_default:
                    defaults[option.lstrip("-")] = default
            self.generic_visit(node)

    Visitor().visit(tree)
    return defaults


def check_equal(label: str, left, right, failures: list[str]):
    if left == right:
        print(f"OK {label}: {left}")
    else:
        failures.append(f"{label}: expected `{right}`, got `{left}`")


def main() -> int:
    failures: list[str] = []

    base_cfg = load_yaml("configs/semtalk_base.yaml")
    sparse_cfg = load_yaml("configs/semtalk_sparse.yaml")
    train_defaults = parse_add_argument_defaults(REPO_ROOT / "dataloaders" / "save_train_dataset.py")
    test_defaults = parse_add_argument_defaults(REPO_ROOT / "dataloaders" / "save_test_dataset.py")

    check_equal("train dataset output default", train_defaults.get("dst_lmdb"), "./datasets/beat2_semtalk_train/", failures)
    check_equal("train dataset cache default", train_defaults.get("cache_path"), "./datasets/beat2_cache_2/", failures)
    check_equal("test dataset output default", test_defaults.get("dst_pkl"), "./datasets/beat2_semtalk_test.pkl", failures)
    check_equal("test dataset cache default", test_defaults.get("cache_path"), "./datasets/beat2_cache_2/", failures)

    check_equal("base config train_path", base_cfg.get("train_path"), "datasets/beat2_semtalk_train", failures)
    check_equal("base config test_path", base_cfg.get("test_path"), "datasets/beat2_semtalk_test.pkl", failures)
    check_equal("base config cache_path", base_cfg.get("cache_path"), "datasets/beat2_cache_2", failures)

    check_equal("sparse config train_path", sparse_cfg.get("train_path"), "datasets/beat2_semtalk_train", failures)
    check_equal("sparse config test_path", sparse_cfg.get("test_path"), "datasets/beat2_semtalk_test.pkl", failures)
    check_equal("sparse config cache_path", sparse_cfg.get("cache_path"), "datasets/beat2_cache_2", failures)
    check_equal("sparse config base_ckpt", sparse_cfg.get("base_ckpt"), "./weights/best_semtalk_base.bin", failures)
    check_equal("sparse config data_path_1", sparse_cfg.get("data_path_1"), "./weights/", failures)

    if failures:
        print("\nCommand contract check failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nCommand contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
