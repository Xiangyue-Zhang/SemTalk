#!/usr/bin/env python3

import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.project_paths import hubert_dir, whisper_dir


REQUIRED_PACKAGES = [
    "torch",
    "torchaudio",
    "numpy",
    "librosa",
    "lmdb",
    "smplx",
    "configargparse",
    "loguru",
    "tensorboard",
    "transformers",
    "soundfile",
]


def _status(ok: bool, label: str, detail: str = ""):
    prefix = "[OK]" if ok else "[MISSING]"
    print(f"{prefix} {label}" + (f": {detail}" if detail else ""))


def _repo_root() -> Path:
    return REPO_ROOT


def _load_config(repo_root: Path, config_path: str):
    config_file = (repo_root / config_path).resolve()
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_file, data


def _check_python():
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 8
    _status(ok, "Python", f"{version.major}.{version.minor}.{version.micro}")
    return ok


def _check_packages():
    ok = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            _status(True, f"package `{package}`")
        except Exception:
            _status(False, f"package `{package}`")
            ok = False
    return ok


def _check_torch():
    try:
        import torch
    except Exception:
        _status(False, "PyTorch")
        return False
    cuda_available = torch.cuda.is_available()
    devices = torch.cuda.device_count() if cuda_available else 0
    _status(True, "PyTorch", f"version={torch.__version__}, cuda_available={cuda_available}, devices={devices}")
    return True


def _exists(path: Path, label: str):
    ok = path.exists()
    _status(ok, label, str(path))
    return ok


def _join(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (repo_root / path)


def main():
    parser = argparse.ArgumentParser(description="Check whether SemTalk is ready to run.")
    parser.add_argument("--config", default="configs/semtalk_sparse.yaml", help="Config file to inspect.")
    parser.add_argument(
        "--mode",
        choices=["inference", "train-base", "train-sparse"],
        default="inference",
        help="Which workflow to validate.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    print(f"Checking SemTalk in: {repo_root}")
    config_file, cfg = _load_config(repo_root, args.config)
    _status(True, "config", str(config_file))

    ok = True
    ok &= _check_python()
    ok &= _check_torch()
    ok &= _check_packages()

    data_path = _join(repo_root, cfg.get("data_path", "./BEAT2/beat_english_v2.0.0/"))
    data_path_1 = _join(repo_root, cfg.get("data_path_1", "./BEAT2/pretrained/"))
    weights_dir = repo_root / "weights"
    demo_dir = repo_root / "demo"

    ok &= _exists(data_path, "BEAT2 data root")
    ok &= _exists(data_path / "wave16k", "BEAT2 wave16k")
    ok &= _exists(data_path / "smplxflame_30", "BEAT2 smplxflame_30")
    ok &= _exists(data_path / "sem", "BEAT2 sem")
    ok &= _exists(data_path_1 / "smplx_models", "SMPL-X model folder")
    ok &= _exists(weights_dir, "weights directory")
    ok &= _exists(demo_dir, "demo directory")

    dataset_cache = _join(repo_root, cfg.get("cache_path", "datasets/beat2_cache_2"))
    train_path = _join(repo_root, cfg.get("train_path", "datasets/beat2_semtalk_train"))
    test_path = _join(repo_root, cfg.get("test_path", "datasets/beat2_semtalk_test.pkl"))

    _exists(dataset_cache, "dataset cache path")
    _exists(train_path, "train dataset path")
    _exists(test_path, "test dataset path")

    base_ckpt = _join(repo_root, cfg.get("base_ckpt", "weights/best_semtalk_base.bin"))
    load_ckpt = cfg.get("load_ckpt")
    if load_ckpt:
        _exists(_join(repo_root, load_ckpt), "load_ckpt")
    if args.mode in {"inference", "train-sparse"}:
        ok &= _exists(base_ckpt, "base_ckpt")

    pretrained_vq = [
        "rvq_face_600.bin",
        "rvq_upper_500.bin",
        "rvq_hands_500.bin",
        "rvq_lower_600.bin",
        "last_1700_foot.bin",
    ]
    for name in pretrained_vq:
        _exists(weights_dir / "pretrained_vq" / name, f"pretrained VQ `{name}`")

    _exists(hubert_dir(), "HuBERT directory")
    _exists(whisper_dir(), "faster-whisper directory")

    if ok:
        print("Environment check passed.")
        return 0

    print("Environment check finished with missing required items.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
