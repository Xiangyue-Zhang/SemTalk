#!/usr/bin/env python3
"""Compute only DiffSHEG validation FGD from frozen SemTalk SHOW outputs.

This adapter deliberately does not implement metric formulas.  It verifies and
imports the pinned PASPA DiffSHEG evaluator, then calls that evaluator's input
validation, normalization, gesture-AE feature extraction, and Fréchet-distance
functions.  Only the 129-D gesture autoencoder is loaded.  The emitted
``metrics`` object is exactly ``{"fgd": <finite nonnegative float>}``.

The command accepts only an explicit 1,715-clip validation manifest and
validation-labelled input paths.  Test-labelled, Speaker2-labelled, withdrawn
e30-labelled, symlinked, unpinned, or non-exact inputs are rejected before
PyTorch is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_VAL_CLIPS = 1_715
WINDOW_LENGTH = 88
WINDOW_STRIDE = 88
GESTURE_DIM = 129
LATENT_DIM = 300
PROTOCOL_NAME = "diffsheg_show_reconstructed"
PROTOCOL_VERSION = 1
PRECISION = "float32 AE inference; no autocast"

PASPA_ORIGIN = "git@github.com:Ly403/PASPA.git"
PASPA_COMMIT = "0df27e6cab4b5ced19cc923afe352f77d547924b"
PASPA_TREE = "574662eaf3122beb5631c02c847456e752d77f0b"
PASPA_EVALUATOR_RELATIVE = Path("scripts/diffsheg_show_eval.py")
PASPA_EVALUATOR_SHA256 = (
    "21fa84fdb9c3f64eb2920714e1d27a685210a225bcffa78503452c4018a53f8c"
)

DIFFSHEG_REFERENCE_COMMIT = "3ebf3058f48cba3da9146afb7623e9ec1ab9e9a5"
DIFFSHEG_STATS_RELATIVE = Path("data/SHOW/talkshow_mean_std.npy")
DIFFSHEG_STATS_SHA256 = (
    "b90320eba94d0777e7160fd31d0fe6f04a7c86822ac875fb7db5cf58d298cef0"
)
DIFFSHEG_GESTURE_AE_RELATIVE = Path(
    "data/SHOW/ae_weights/gesture.pth.tar"
)
DIFFSHEG_GESTURE_AE_SHA256 = (
    "5eaf9b882a5ccd5f6eb4385aaadf3d28f3ee4382360ecb13c12f4904b3c3216e"
)

SHOW_SPEAKERS = frozenset({"oliver", "chemistry", "seth", "conan"})
FORBIDDEN_E30 = re.compile(
    r"(^|[^a-z0-9])e[-_]?30([^a-z0-9]|$)", re.IGNORECASE
)
FORBIDDEN_SPEAKER2 = re.compile(
    r"(^|[^a-z0-9])speaker[-_]?2([^a-z0-9]|$)", re.IGNORECASE
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ValidationFGDError(RuntimeError):
    """Raised when the strict validation-only FGD contract is not met."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValidationFGDError(
            f"{label} must be a lowercase 64-character SHA-256"
        )
    return value


def _reject_forbidden_label(value: object, label: str) -> None:
    text = str(value).replace("\\", "/")
    for component in text.split("/"):
        normalized = component.casefold()
        if "test" in normalized:
            raise ValidationFGDError(
                f"{label} exposes a test-labelled component: {value}"
            )
        if FORBIDDEN_E30.search(normalized):
            raise ValidationFGDError(
                f"{label} exposes withdrawn e30: {value}"
            )
        if FORBIDDEN_SPEAKER2.search(normalized):
            raise ValidationFGDError(
                f"{label} exposes forbidden Speaker2: {value}"
            )


def _absolute_path(value: object, label: str, *, val_only: bool) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValidationFGDError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValidationFGDError(f"{label} must be absolute: {path}")
    if val_only:
        _reject_forbidden_label(path, label)
    return path


def _require_non_symlink_directory(
    value: object,
    label: str,
    *,
    val_only: bool,
) -> Path:
    path = _absolute_path(value, label, val_only=val_only)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise ValidationFGDError(f"{label} does not exist: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValidationFGDError(
            f"{label} must be a non-symlink directory: {path}"
        )
    resolved = path.resolve(strict=True)
    if val_only:
        _reject_forbidden_label(resolved, label)
    return resolved


def _require_non_symlink_file(
    value: object,
    label: str,
    *,
    val_only: bool,
) -> Path:
    path = _absolute_path(value, label, val_only=val_only)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise ValidationFGDError(f"{label} does not exist: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValidationFGDError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    if val_only:
        _reject_forbidden_label(resolved, label)
    return resolved


def _verify_file_sha256(
    value: object,
    expected_sha256: str,
    label: str,
    *,
    val_only: bool,
) -> tuple[Path, str]:
    expected = _require_sha256(expected_sha256, f"{label} expected SHA-256")
    path = _require_non_symlink_file(value, label, val_only=val_only)
    observed = _sha256_file(path)
    if observed != expected:
        raise ValidationFGDError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return path, observed


def _git_value(root: Path, arguments: Sequence[str], label: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationFGDError(
            f"cannot inspect {label} Git checkout {root}: {error}"
        ) from error
    value = result.stdout.strip()
    if not value:
        raise ValidationFGDError(f"{label} Git query returned an empty value")
    return value


def _verify_git_checkout(
    value: object,
    *,
    label: str,
    expected_head: str,
    expected_tree: str | None = None,
    expected_origin: str | None = None,
) -> tuple[Path, dict[str, str]]:
    root = _require_non_symlink_directory(value, label, val_only=False)
    head = _git_value(root, ["rev-parse", "HEAD^{commit}"], label)
    if head != expected_head:
        raise ValidationFGDError(
            f"{label} HEAD mismatch: {head} != {expected_head}"
        )
    receipt = {"path": str(root), "git_head": head}
    if expected_tree is not None:
        tree = _git_value(root, ["rev-parse", "HEAD^{tree}"], label)
        if tree != expected_tree:
            raise ValidationFGDError(
                f"{label} tree mismatch: {tree} != {expected_tree}"
            )
        receipt["git_tree"] = tree
    if expected_origin is not None:
        origin = _git_value(root, ["remote", "get-url", "origin"], label)
        if origin != expected_origin:
            raise ValidationFGDError(
                f"{label} origin mismatch: {origin} != {expected_origin}"
            )
        receipt["origin"] = origin
    return root, receipt


def _load_pinned_paspa_evaluator(
    paspa_root_value: object,
) -> tuple[ModuleType, dict[str, Any]]:
    root, git_receipt = _verify_git_checkout(
        paspa_root_value,
        label="pinned PASPA evaluator repository",
        expected_head=PASPA_COMMIT,
        expected_tree=PASPA_TREE,
        expected_origin=PASPA_ORIGIN,
    )
    evaluator_path, evaluator_sha = _verify_file_sha256(
        root / PASPA_EVALUATOR_RELATIVE,
        PASPA_EVALUATOR_SHA256,
        "pinned PASPA DiffSHEG evaluator",
        val_only=False,
    )
    module_name = "_semtalk_pinned_paspa_diffsheg_show_eval"
    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
    if spec is None or spec.loader is None:
        raise ValidationFGDError(
            f"cannot import pinned evaluator: {evaluator_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    expected_constants = {
        "PROTOCOL_NAME": PROTOCOL_NAME,
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "DIFFSHEG_REFERENCE_COMMIT": DIFFSHEG_REFERENCE_COMMIT,
        "WINDOW_LENGTH": WINDOW_LENGTH,
        "DEFAULT_WINDOW_STRIDE": WINDOW_STRIDE,
        "GESTURE_DIM": GESTURE_DIM,
        "LATENT_DIM": LATENT_DIM,
    }
    for name, expected in expected_constants.items():
        if getattr(module, name, None) != expected:
            raise ValidationFGDError(
                f"pinned evaluator constant {name} is incompatible"
            )
    for name in (
        "validate_inputs",
        "load_normalization_stats",
        "load_embedding_model",
        "extract_features",
        "frechet_distance",
    ):
        if not callable(getattr(module, name, None)):
            raise ValidationFGDError(
                f"pinned evaluator function {name} is unavailable"
            )
    if (
        ("fgd", "gesture", GESTURE_DIM, "gesture.pth.tar")
        not in getattr(module, "FEATURE_SPECS", ())
    ):
        raise ValidationFGDError(
            "pinned evaluator does not expose the audited FGD feature spec"
        )
    return module, {
        "path": str(evaluator_path),
        "sha256": evaluator_sha,
        "repository_root": str(root),
        "repository_git_head": git_receipt["git_head"],
        "repository_git_tree": git_receipt["git_tree"],
        "repository_origin": git_receipt["origin"],
    }


def _verify_diffsheg_assets(
    diffsheg_root_value: object,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    root, git_receipt = _verify_git_checkout(
        diffsheg_root_value,
        label="pinned DiffSHEG repository",
        expected_head=DIFFSHEG_REFERENCE_COMMIT,
    )
    stats_path, stats_sha = _verify_file_sha256(
        root / DIFFSHEG_STATS_RELATIVE,
        DIFFSHEG_STATS_SHA256,
        "DiffSHEG SHOW normalization statistics",
        val_only=False,
    )
    gesture_path, gesture_sha = _verify_file_sha256(
        root / DIFFSHEG_GESTURE_AE_RELATIVE,
        DIFFSHEG_GESTURE_AE_SHA256,
        "DiffSHEG SHOW gesture autoencoder",
        val_only=False,
    )
    weights_dir = _require_non_symlink_directory(
        gesture_path.parent,
        "DiffSHEG SHOW weights directory",
        val_only=False,
    )
    return root, stats_path, gesture_path, {
        "diffsheg_root": git_receipt,
        "stats": {
            "path": str(stats_path),
            "sha256": stats_sha,
        },
        "weights_dir": str(weights_dir),
        "gesture_autoencoder": {
            "path": str(gesture_path),
            "sha256": gesture_sha,
            "input_dim": GESTURE_DIM,
        },
    }


def _read_exact_val_manifest(
    value: object,
    expected_sha256: object,
) -> tuple[Path, tuple[str, ...], str]:
    expected = _require_sha256(
        expected_sha256, "validation clip manifest SHA-256"
    )
    path, observed = _verify_file_sha256(
        value,
        expected,
        "validation clip manifest",
        val_only=True,
    )
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise ValidationFGDError(
            f"cannot read validation clip manifest {path}: {error}"
        ) from error
    if not text.endswith("\n"):
        raise ValidationFGDError(
            "validation clip manifest must end with one newline"
        )
    lines = text.splitlines()
    if len(lines) != EXPECTED_VAL_CLIPS:
        raise ValidationFGDError(
            "validation clip manifest must contain exactly "
            f"{EXPECTED_VAL_CLIPS} clips, got {len(lines)}"
        )
    if len(set(lines)) != len(lines):
        raise ValidationFGDError(
            "validation clip manifest contains duplicate clip IDs"
        )
    for index, clip_id in enumerate(lines, 1):
        if not clip_id or clip_id != clip_id.strip():
            raise ValidationFGDError(
                f"validation clip manifest line {index} is not canonical"
            )
        _reject_forbidden_label(
            clip_id, f"validation clip manifest line {index}"
        )
        if "/" in clip_id or "\\" in clip_id or "\x00" in clip_id:
            raise ValidationFGDError(
                f"validation clip manifest line {index} is unsafe"
            )
        if "__" not in clip_id:
            raise ValidationFGDError(
                f"validation clip ID {clip_id!r} lacks speaker separator"
            )
        speaker, sequence = clip_id.split("__", 1)
        if speaker not in SHOW_SPEAKERS or not sequence:
            raise ValidationFGDError(
                f"validation clip ID {clip_id!r} is not canonical SHOW"
            )
    return path, tuple(lines), observed


def _validate_exact_directory_cover(
    directory_value: object,
    clip_ids: Sequence[str],
    *,
    role: str,
    prefix: str,
) -> Path:
    directory = _require_non_symlink_directory(
        directory_value,
        f"validation {role} directory",
        val_only=True,
    )
    expected_names = {f"{prefix}{clip_id}.npz" for clip_id in clip_ids}
    observed_names: set[str] = set()
    for entry in directory.iterdir():
        try:
            mode = os.lstat(entry).st_mode
        except FileNotFoundError:
            raise ValidationFGDError(
                f"validation {role} entry disappeared: {entry}"
            ) from None
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValidationFGDError(
                f"validation {role} directory contains a non-regular "
                f"or symlink entry: {entry}"
            )
        _reject_forbidden_label(entry.name, f"validation {role} filename")
        observed_names.add(entry.name)
    if observed_names != expected_names:
        missing = sorted(expected_names.difference(observed_names))[:20]
        extra = sorted(observed_names.difference(expected_names))[:20]
        raise ValidationFGDError(
            f"validation {role} directory is not the exact "
            f"{len(clip_ids)}-file cover: missing={missing}, extra={extra}"
        )
    return directory


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _current_git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _atomic_write_new_json(path_value: object, report: Mapping[str, Any]) -> Path:
    path = _absolute_path(path_value, "output report", val_only=True)
    parent = _require_non_symlink_directory(
        path.parent,
        "output report parent",
        val_only=True,
    )
    output = parent / path.name
    if output.exists() or output.is_symlink():
        raise ValidationFGDError(f"refusing to overwrite output: {output}")
    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise ValidationFGDError(
                f"refusing to overwrite output: {output}"
            ) from None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Validation-only directory containing exactly 1,715 res_*.npz.",
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Validation-only directory containing exactly 1,715 gt_*.npz.",
    )
    parser.add_argument(
        "--clip-manifest",
        required=True,
        help="Explicit ordered 1,715-line validation clip-ID manifest.",
    )
    parser.add_argument(
        "--clip-manifest-sha256",
        required=True,
        help="Expected byte SHA-256 of --clip-manifest.",
    )
    parser.add_argument(
        "--paspa-root",
        required=True,
        help="Pinned PASPA checkout at the audited evaluator commit.",
    )
    parser.add_argument(
        "--diffsheg-root",
        required=True,
        help="Pinned DiffSHEG checkout containing SHOW stats and gesture AE.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for gesture AE inference (default: cuda:0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Prediction windows per PASPA gesture-AE call (default: 64).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New validation-only JSON report path; never overwritten.",
    )
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValidationFGDError("--batch-size must be positive")
    clip_manifest, clip_ids, clip_manifest_file_sha = (
        _read_exact_val_manifest(
            args.clip_manifest,
            args.clip_manifest_sha256,
        )
    )
    prediction_dir = _validate_exact_directory_cover(
        args.pred_dir,
        clip_ids,
        role="prediction",
        prefix="res_",
    )
    ground_truth_dir = _validate_exact_directory_cover(
        args.gt_dir,
        clip_ids,
        role="ground-truth",
        prefix="gt_",
    )
    if prediction_dir == ground_truth_dir:
        raise ValidationFGDError(
            "validation prediction and ground-truth directories must differ"
        )

    evaluator, evaluator_receipt = _load_pinned_paspa_evaluator(
        args.paspa_root
    )
    (
        diffsheg_root,
        stats_path,
        gesture_checkpoint,
        asset_receipt,
    ) = _verify_diffsheg_assets(args.diffsheg_root)
    try:
        stats = evaluator.load_normalization_stats(stats_path)
        validation = evaluator.validate_inputs(
            prediction_dir=prediction_dir,
            ground_truth_dir=ground_truth_dir,
            clip_manifest=clip_manifest,
            window_stride=WINDOW_STRIDE,
            require_betas=False,
        )
    except evaluator.ProtocolError as error:
        raise ValidationFGDError(str(error)) from error
    if len(validation.clips) != EXPECTED_VAL_CLIPS:
        raise ValidationFGDError(
            "pinned evaluator did not validate exactly "
            f"{EXPECTED_VAL_CLIPS} clips"
        )
    if validation.clip_order != f"explicit_manifest:{clip_manifest}":
        raise ValidationFGDError(
            "pinned evaluator did not preserve the explicit clip order"
        )

    try:
        import torch
    except ImportError as error:
        raise ValidationFGDError(
            "PyTorch is required for validation FGD"
        ) from error
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValidationFGDError(
                f"requested {device}, but CUDA is unavailable"
            )
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")

    try:
        model, checkpoint_info = evaluator.load_embedding_model(
            torch=torch,
            checkpoint_path=gesture_checkpoint,
            input_dim=GESTURE_DIM,
            device=device,
        )
        predicted, ground_truth = evaluator.extract_features(
            torch=torch,
            model=model,
            device=device,
            validation=validation,
            stats=stats,
            component="gesture",
            batch_size=args.batch_size,
        )
        if (
            predicted.shape
            != (validation.total_windows, LATENT_DIM)
            or ground_truth.shape != predicted.shape
        ):
            raise ValidationFGDError(
                "pinned gesture AE returned incompatible feature shapes: "
                f"{predicted.shape}, {ground_truth.shape}"
            )
        if predicted.dtype != np.float32 or ground_truth.dtype != np.float32:
            raise ValidationFGDError(
                "gesture AE features must be exactly float32"
            )
        if (
            not np.isfinite(predicted).all()
            or not np.isfinite(ground_truth).all()
        ):
            raise ValidationFGDError(
                "gesture AE features contain NaN or Inf"
            )
        fgd = float(evaluator.frechet_distance(predicted, ground_truth))
    except evaluator.ProtocolError as error:
        raise ValidationFGDError(str(error)) from error
    finally:
        if "model" in locals():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not math.isfinite(fgd) or fgd < 0.0:
        raise ValidationFGDError(
            f"validation FGD must be finite and nonnegative, got {fgd}"
        )

    checkpoint_provenance = {
        "path": str(gesture_checkpoint),
        "sha256": DIFFSHEG_GESTURE_AE_SHA256,
        "input_dim": GESTURE_DIM,
        "latent_dim": LATENT_DIM,
        "state_container": checkpoint_info.get("state_container"),
        "load_mode": checkpoint_info.get("load_mode"),
        "feature_count": int(predicted.shape[0]),
    }
    if (
        checkpoint_info.get("path") != str(gesture_checkpoint)
        or checkpoint_info.get("sha256") != DIFFSHEG_GESTURE_AE_SHA256
        or checkpoint_info.get("input_dim") != GESTURE_DIM
    ):
        raise ValidationFGDError(
            "pinned evaluator returned mismatched gesture-AE provenance"
        )

    adapter_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "status": "ok",
        "protocol": {
            "name": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "status": "reconstructed_from_public_components",
            "diffsheg_reference_commit": DIFFSHEG_REFERENCE_COMMIT,
            "window_length": WINDOW_LENGTH,
            "window_stride": WINDOW_STRIDE,
            "precision": PRECISION,
            "ba": None,
            "clip_order": f"explicit_manifest:{clip_manifest}",
            "selection_split": "val",
            "test_visible": False,
            "metric_scope": "fgd_only",
            "parameter_order": (
                "PASPA/BEAT2 poses[165] -> DiffSHEG "
                "ShowDataset.extract_pose gesture[129]"
            ),
            "normalization": "DiffSHEG talkshow_mean_std.npy",
            "tail_policy": "drop_incomplete_tail",
        },
        "inputs": {
            "prediction_dir": str(prediction_dir),
            "ground_truth_dir": str(ground_truth_dir),
            "clip_manifest": str(clip_manifest),
            "clip_count": len(validation.clips),
            "frame_count": int(validation.total_frames),
            "window_count": int(validation.total_windows),
            "uncovered_tail_frames": int(
                validation.uncovered_tail_frames
            ),
            "clip_manifest_sha256": validation.clip_manifest_sha256,
            "clip_manifest_file_sha256": clip_manifest_file_sha,
            "stats": asset_receipt["stats"],
            "weights_dir": asset_receipt["weights_dir"],
            "checkpoint_paths": {
                "fgd": str(gesture_checkpoint),
            },
        },
        "metrics": {"fgd": fgd},
        "diagnostics": {
            "gesture_feature_count": int(predicted.shape[0]),
            "gesture_feature_dim": int(predicted.shape[1]),
        },
        "provenance": {
            "evaluator": evaluator_receipt,
            "diffsheg_root": {
                "path": str(diffsheg_root),
                "git_head": asset_receipt["diffsheg_root"]["git_head"],
            },
            # Truthful provenance: no holistic or expression AE was loaded.
            "autoencoders": {"fgd": checkpoint_provenance},
            "adapter": {
                "path": str(adapter_path),
                "sha256": _sha256_file(adapter_path),
                "repository_root": str(adapter_path.parents[2]),
                "repository_git_head": _current_git_head(
                    adapter_path.parents[2]
                ),
            },
            "device": str(device),
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": getattr(torch, "__version__", None),
                "scipy": _distribution_version("scipy"),
            },
        },
    }
    if set(report["metrics"]) != {"fgd"}:
        raise AssertionError("validation report metric scope widened")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
        output = _atomic_write_new_json(args.output, report)
    except ValidationFGDError as error:
        parser.exit(2, f"[abort] {error}\n")
    print(f"[FGD] {report['metrics']['fgd']}")
    print(f"[done] validation-only DiffSHEG FGD report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
