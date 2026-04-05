from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[1]

PRETRAINED_VQ_FILES = {
    "face": "rvq_face_600.bin",
    "upper": "rvq_upper_500.bin",
    "hands": "rvq_hands_500.bin",
    "lower": "rvq_lower_600.bin",
    "global_motion": "last_1700_foot.bin",
}


def repo_path(*parts) -> Path:
    return REPO_ROOT.joinpath(*parts)


def resolve_path(path_like, base: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if base is not None:
        return (base / path).resolve()
    return (REPO_ROOT / path).resolve()


def smplx_model_dir(args) -> Path:
    base = Path(getattr(args, "data_path_1", "./BEAT2/pretrained/"))
    return resolve_path(base / "smplx_models")


def pretrained_vq_dir() -> Path:
    return repo_path("weights", "pretrained_vq")


def pretrained_vq_path(name: str) -> Path:
    return pretrained_vq_dir() / PRETRAINED_VQ_FILES[name]


def configure_runtime_env():
    default_cache_root = Path(os.environ.get("SEMTALK_CACHE_ROOT", repo_path(".cache")))
    default_tmp_root = Path(os.environ.get("SEMTALK_TMPDIR", repo_path(".tmp")))

    os.environ.setdefault("HF_HOME", str(default_cache_root / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(default_cache_root / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(default_cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(default_cache_root))
    os.environ.setdefault("TMPDIR", str(default_tmp_root))
