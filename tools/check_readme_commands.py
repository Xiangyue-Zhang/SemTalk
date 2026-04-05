#!/usr/bin/env python3

import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


EXPECTED_COMMANDS = [
    "python tools/check_env.py --config configs/semtalk_sparse.yaml",
    "python tools/run.py infer --config configs/semtalk_sparse.yaml --audio demo/2_scott_0_1_1_test.wav",
    "python tools/run.py build-dataset --split train",
    "python tools/run.py build-dataset --split test",
    "python tools/run.py train-rvq configs/cnn_vqvae_face_30.yaml",
    "python tools/run.py train-rvq configs/cnn_vqvae_hands_30.yaml",
    "python tools/run.py train-rvq configs/cnn_vqvae_upper_30.yaml",
    "python tools/run.py train-rvq configs/cnn_vqvae_lower_foot_30.yaml",
    "python tools/run.py train-rvq configs/cnn_vqvae_lower_30.yaml",
    "python tools/run.py train-base --config configs/semtalk_base.yaml",
    "python tools/run.py train-sparse --config configs/semtalk_sparse.yaml",
    "python tools/run.py test --config configs/semtalk_sparse.yaml",
    "python tools/run.py infer --config configs/semtalk_sparse.yaml --audio ./demo/2_scott_0_1_1.wav",
]


def normalize_command(command: str) -> str:
    return " ".join(command.replace("\\\n", " ").replace("\\", " ").split())


def normalized_readme_text() -> str:
    text = README_PATH.read_text(encoding="utf-8")
    return normalize_command(text)


def assert_readme_contains_expected_commands(readme_text: str) -> list[str]:
    failures = []
    for command in EXPECTED_COMMANDS:
        normalized = normalize_command(command)
        if normalized not in readme_text:
            failures.append(f"README missing command: {command}")
        else:
            print(f"OK README command: {command}")
    return failures


def assert_paths_exist() -> list[str]:
    required_paths = [
        REPO_ROOT / "tools" / "check_env.py",
        REPO_ROOT / "tools" / "run.py",
        REPO_ROOT / "configs" / "semtalk_base.yaml",
        REPO_ROOT / "configs" / "semtalk_sparse.yaml",
        REPO_ROOT / "configs" / "cnn_vqvae_face_30.yaml",
        REPO_ROOT / "configs" / "cnn_vqvae_hands_30.yaml",
        REPO_ROOT / "configs" / "cnn_vqvae_upper_30.yaml",
        REPO_ROOT / "configs" / "cnn_vqvae_lower_foot_30.yaml",
        REPO_ROOT / "configs" / "cnn_vqvae_lower_30.yaml",
        REPO_ROOT / "demo" / "2_scott_0_1_1_test.wav",
    ]
    failures = []
    for path in required_paths:
        if path.exists():
            print(f"OK path exists: {path.relative_to(REPO_ROOT)}")
        else:
            failures.append(f"Missing required path: {path.relative_to(REPO_ROOT)}")
    return failures


def run_help_checks() -> list[str]:
    checks = [
        [sys.executable, "tools/check_env.py", "--help"],
        [sys.executable, "tools/run.py", "check-env", "--help"],
        [sys.executable, "tools/run.py", "build-dataset", "--help"],
        [sys.executable, "tools/run.py", "train-rvq", "--help"],
        [sys.executable, "tools/run.py", "train-base", "--help"],
        [sys.executable, "tools/run.py", "train-sparse", "--help"],
        [sys.executable, "tools/run.py", "test", "--help"],
        [sys.executable, "tools/run.py", "infer", "--help"],
    ]
    failures = []
    for cmd in checks:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        printable = " ".join(shlex.quote(part) for part in cmd[1:])
        if result.returncode == 0:
            print(f"OK help command: {printable}")
        else:
            failures.append(f"Help command failed: {printable}")
    return failures


def main() -> int:
    failures = []
    failures.extend(assert_readme_contains_expected_commands(normalized_readme_text()))
    failures.extend(assert_paths_exist())
    failures.extend(run_help_checks())

    if failures:
        print("\nREADME command check failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nREADME command check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
