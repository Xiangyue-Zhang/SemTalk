#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd: Path):
    print(f"[RUN] ({cwd}) {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def collect_manifest(target: Path):
    if target.is_file():
        return {
            "type": "file",
            "path": str(target),
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
        }

    if target.is_dir():
        files = []
        for path in sorted(target.rglob("*")):
            if path.is_file():
                files.append({
                    "relative_path": str(path.relative_to(target)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        return {
            "type": "dir",
            "path": str(target),
            "files": files,
        }

    raise FileNotFoundError(target)


def compare_manifests(left, right):
    return left == right


def main():
    parser = argparse.ArgumentParser(description="Validate current SemTalk against an original baseline checkout.")
    parser.add_argument("--baseline-repo", required=True, help="Path to the original baseline checkout, e.g. SemTalk at commit 45180aa")
    parser.add_argument("--current-repo", default=".", help="Path to the current SemTalk checkout")
    parser.add_argument("--mode", choices=["check-env", "infer", "test"], default="check-env")
    parser.add_argument("--config", default="configs/semtalk_sparse.yaml")
    parser.add_argument("--audio", default="demo/2_scott_0_1_1_test.wav")
    parser.add_argument("--baseline-output", required=True, help="Output path produced by the baseline run")
    parser.add_argument("--current-output", required=True, help="Output path produced by the current run")
    parser.add_argument("--skip-run", action="store_true", help="Only compare existing outputs; do not execute commands")
    args = parser.parse_args()

    baseline_repo = Path(args.baseline_repo).resolve()
    current_repo = Path(args.current_repo).resolve()
    baseline_output = Path(args.baseline_output).resolve()
    current_output = Path(args.current_output).resolve()

    if not args.skip_run:
        if args.mode == "check-env":
            run([sys.executable, "tools/check_env.py", "--config", args.config], baseline_repo)
            run([sys.executable, "tools/check_env.py", "--config", args.config], current_repo)
        elif args.mode == "infer":
            run([sys.executable, "tools/run.py", "infer", "--config", args.config, "--audio", args.audio], baseline_repo)
            run([sys.executable, "tools/run.py", "infer", "--config", args.config, "--audio", args.audio], current_repo)
        elif args.mode == "test":
            run([sys.executable, "tools/run.py", "test", "--config", args.config], baseline_repo)
            run([sys.executable, "tools/run.py", "test", "--config", args.config], current_repo)

    baseline_manifest = collect_manifest(baseline_output)
    current_manifest = collect_manifest(current_output)
    same = compare_manifests(baseline_manifest, current_manifest)

    print(json.dumps({
        "same": same,
        "baseline": baseline_manifest,
        "current": current_manifest,
    }, indent=2, ensure_ascii=False))

    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
