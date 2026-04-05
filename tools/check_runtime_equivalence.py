#!/usr/bin/env python3
import argparse
import ast
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "45180aa"


TARGETS = {
    "semtalk_base_trainer.py": {
        "CustomTrainer": ["_g_training", "_g_test", "train", "test"],
    },
    "semtalk_sparse_trainer.py": {
        "CustomTrainer": ["_g_training", "_g_test", "train", "test", "inference"],
    },
}


def parse_source(source: str, file_label: str) -> ast.Module:
    try:
        return ast.parse(source, filename=file_label)
    except SyntaxError as exc:
        raise SystemExit(f"Failed to parse {file_label}: {exc}") from exc


def load_baseline_source(repo_root: Path, commit: str, relpath: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relpath}"],
        text=True,
    )


def class_methods(tree: ast.Module, class_name: str) -> dict:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name: child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise KeyError(class_name)


def normalized_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def compare_method(repo_root: Path, relpath: str, class_name: str, method_name: str, commit: str) -> tuple[bool, str]:
    current_source = (repo_root / relpath).read_text()
    baseline_source = load_baseline_source(repo_root, commit, relpath)

    current_tree = parse_source(current_source, f"current:{relpath}")
    baseline_tree = parse_source(baseline_source, f"baseline:{relpath}")

    current_methods = class_methods(current_tree, class_name)
    baseline_methods = class_methods(baseline_tree, class_name)

    if method_name not in current_methods:
        return False, f"missing current method {class_name}.{method_name}"
    if method_name not in baseline_methods:
        return False, f"missing baseline method {class_name}.{method_name}"

    same = normalized_dump(current_methods[method_name]) == normalized_dump(baseline_methods[method_name])
    if same:
        return True, f"OK {relpath}::{class_name}.{method_name}"
    return False, f"CHANGED {relpath}::{class_name}.{method_name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statically compare runtime-critical code paths against the original SemTalk baseline."
    )
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    failures = []

    for relpath, classes in TARGETS.items():
        for class_name, methods in classes.items():
            for method_name in methods:
                same, message = compare_method(repo_root, relpath, class_name, method_name, args.baseline_commit)
                print(message)
                if not same:
                    failures.append(message)

    if failures:
        print(f"\nFound {len(failures)} runtime-critical method changes relative to {args.baseline_commit}.")
        return 1

    print(f"\nAll tracked runtime-critical methods match baseline {args.baseline_commit}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
