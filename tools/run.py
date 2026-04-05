#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd, cwd):
    printable = " ".join(str(x) for x in cmd)
    print(f"[RUN] ({cwd}) {printable}")
    subprocess.run(cmd, cwd=cwd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Unified entrypoint for SemTalk.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_env = subparsers.add_parser("check-env", help="Run the environment checker.")
    check_env.add_argument("extra", nargs=argparse.REMAINDER)

    build_data = subparsers.add_parser("build-dataset", help="Generate train or test dataset cache.")
    build_data.add_argument("--split", choices=["train", "test"], default="train")

    train_rvq = subparsers.add_parser("train-rvq", help="Train one RVQ-VAE stage.")
    train_rvq.add_argument("config", help="Config path, e.g. configs/cnn_vqvae_face_30.yaml")
    train_rvq.add_argument("extra", nargs=argparse.REMAINDER)

    train_base = subparsers.add_parser("train-base", help="Train SemTalk base stage.")
    train_base.add_argument("--config", default="configs/semtalk_base.yaml")
    train_base.add_argument("extra", nargs=argparse.REMAINDER)

    train_sparse = subparsers.add_parser("train-sparse", help="Train SemTalk sparse stage.")
    train_sparse.add_argument("--config", default="configs/semtalk_sparse.yaml")
    train_sparse.add_argument("extra", nargs=argparse.REMAINDER)

    test = subparsers.add_parser("test", help="Run evaluation with --test_state.")
    test.add_argument("--config", default="configs/semtalk_sparse.yaml")
    test.add_argument("--load-ckpt", default=None)
    test.add_argument("extra", nargs=argparse.REMAINDER)

    infer = subparsers.add_parser("infer", help="Run inference from an audio file.")
    infer.add_argument("--config", default="configs/semtalk_sparse.yaml")
    infer.add_argument("--audio", default="demo/2_scott_0_1_1_test.wav")
    infer.add_argument("--load-ckpt", default=None)
    infer.add_argument("extra", nargs=argparse.REMAINDER)

    return parser


def _train_cmd(config, extra, *flags):
    return [sys.executable, "train.py", "--config", config, *flags, *extra]


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check-env":
        cmd = [sys.executable, str(REPO_ROOT / "tools" / "check_env.py"), *args.extra]
        run_command(cmd, REPO_ROOT)
        return 0

    if args.command == "build-dataset":
        script = "save_train_dataset.py" if args.split == "train" else "save_test_dataset.py"
        cmd = [sys.executable, script]
        run_command(cmd, REPO_ROOT / "dataloaders")
        return 0

    if args.command == "train-rvq":
        cmd = _train_cmd(args.config, args.extra, "--train_rvq")
        run_command(cmd, REPO_ROOT)
        return 0

    if args.command == "train-base":
        cmd = _train_cmd(args.config, args.extra)
        run_command(cmd, REPO_ROOT)
        return 0

    if args.command == "train-sparse":
        cmd = _train_cmd(args.config, args.extra)
        run_command(cmd, REPO_ROOT)
        return 0

    if args.command == "test":
        flags = ["--test_state"]
        if args.load_ckpt:
            flags.extend(["--load_ckpt", args.load_ckpt])
        cmd = _train_cmd(args.config, args.extra, *flags)
        run_command(cmd, REPO_ROOT)
        return 0

    if args.command == "infer":
        flags = ["--inference", "--audio_infer_path", args.audio]
        if args.load_ckpt:
            flags.extend(["--load_ckpt", args.load_ckpt])
        cmd = _train_cmd(args.config, args.extra, *flags)
        run_command(cmd, REPO_ROOT)
        return 0

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
