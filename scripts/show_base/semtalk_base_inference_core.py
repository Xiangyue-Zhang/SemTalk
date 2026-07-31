#!/usr/bin/env python3
"""Neutral SemTalk Base model loading and deterministic inference core.

This module contains only the callable closure needed by formal Base
validation and test inference.  Dataset selection, metric evaluation,
publication, topology, and transaction policy deliberately live elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import MethodType, SimpleNamespace
from typing import Any
import zipfile

import numpy as np


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


POSE_FPS = 30


POSE_DIM = 165


EXPRESSION_DIM = 100


BETA_DIM = 300


WINDOW = 64


PRE_FRAMES = 4


STRIDE = WINDOW - PRE_FRAMES


CODEBOOK_SIZE = 256


RVQ_LEVELS = 6


RVQ_TOKEN_FRAMES = 4


RVQ_TOKENS_PER_WINDOW = WINDOW // RVQ_TOKEN_FRAMES


CANONICAL_FIELDS = (
    "pose",
    "contact",
    "facial",
    "beta",
    "trans",
    "speaker_id",
)


AUDIO_FIELDS = ("beat", "hubert")


OUTPUT_FIELDS = (
    "betas",
    "poses",
    "expressions",
    "trans",
    "model",
    "gender",
    "mocap_frame_rate",
)


RVQ_DIMS = {
    "face": 106,
    "upper": 78,
    "hands": 180,
    "lower": 61,
}


OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT = (
    "semtalk_show_base_official_adapt_checkpoint_v1"
)


INFERENCE_AUXILIARY_LOSS_BYPASS_FORMAT = (
    "semtalk_inference_auxiliary_loss_bypass_v1"
)


INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS = (
    "hubert_face_cons_loss",
    "beat_cons_loss",
)


PINNED_SEMTALK_MODEL_SOURCE = {
    "relative_path": "models/semtalk.py",
    "sha256": (
        "ddcc622c9778413b73c2b51b90354cd82dcedce62fea25006f8219de17f7ba2e"
    ),
    "git_blob_sha1": "6e786a5f5d29c070145c49f2a277cc7667c78ace",
}


PINNED_SEMTALK_BASE_FORWARD = {
    "source_sha256": (
        "032ee956dde6297cb9e3211c713f9eb1183b4937663f42a76586f1a42c0dbbcb"
    ),
    "signature": (
        "(self, in_audio=None, in_word=None, mask=None, is_train=False, "
        "in_motion=None, use_attentions=True, use_word=True, in_id=None, "
        "hubert=None)"
    ),
}


PINNED_RHYTHMIC_LOSS_FORWARD = {
    "source_sha256": (
        "eae08dca9b9f7605dc4ff35576c9630380a365e4ab2b61429755aa490fef659e"
    ),
    "signature": "(self, facial_features, audio_features)",
}


PINNED_RVQ_INDICES_SOURCE_SHA256 = (
    "467e0115387738b2ccc9aa46a0174d4a7b93aa1199d8e8bae9070e90f1150ffc"
)


PINNED_INFER_CLIP = {
    "source_sha256": (
        "40e4b6982c85ded6f6fef2dafb2f9f5f4bc876150ecf40fac59de9c7c6e8242c"
    ),
    "signature": (
        "(*, pose: 'np.ndarray', trans: 'np.ndarray', beat: 'np.ndarray', "
        "hubert: 'np.ndarray', speaker_id: 'int', "
        "models: 'Mapping[str, Any]', masks: 'Mapping[str, Any]', "
        "device: 'str') -> 'dict[str, np.ndarray]'"
    ),
}


RELEASED_ALL_SPEAKERS_MODELS = {
    "base": {
        "filename": "best_semtalk_base.bin",
        "sha256": (
            "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc"
            "1d89603"
        ),
        "model_class": "semtalk_base",
    },
    "face": {
        "filename": "rvq_face_600.bin",
        "sha256": (
            "31b04c88456a25f4d57841c0cb507b4c856daccb3875878d06545110"
            "a6152127"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 106,
        "vae_layer": 2,
    },
    "hands": {
        "filename": "rvq_hands_500.bin",
        "sha256": (
            "08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a5"
            "39b03e436"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 180,
        "vae_layer": 2,
    },
    "upper": {
        "filename": "rvq_upper_500.bin",
        "sha256": (
            "05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d8"
            "8d08ac17"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 78,
        "vae_layer": 2,
    },
    "lower": {
        "filename": "rvq_lower_600.bin",
        "sha256": (
            "2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c17"
            "1ae4efce8"
        ),
        "model_class": "RVQVAE",
        "vae_test_dim": 61,
        "vae_layer": 4,
    },
    "global": {
        "filename": "last_1700_foot.bin",
        "sha256": (
            "6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e127e19e"
            "0144ee12"
        ),
        "model_class": "VAEConvZero",
        "vae_test_dim": 61,
        "vae_layer": 4,
    },
}


SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}


class InferenceContractError(RuntimeError):
    """Raised when frozen inputs or generated outputs violate the contract."""


_STABLE_FILE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(metadata, field)) for field in _STABLE_FILE_FIELDS
    )


def _safe_file_snapshot(
    path: str | Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, bytes, str, os.stat_result]:
    """Read and hash one stable regular file through one descriptor."""
    expected = (
        _require_sha256(expected_sha256, f"{label} expected SHA")
        if expected_sha256 is not None
        else None
    )
    resolved = _resolved_regular_file(path, label)
    try:
        path_before = os.lstat(resolved)
    except OSError as exc:
        raise InferenceContractError(
            f"cannot safely inspect {label}: {resolved}"
        ) from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
        path_before.st_mode
    ):
        raise InferenceContractError(
            f"{label} must be a regular non-symlink file: {resolved}"
        )
    parts = resolved.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise InferenceContractError(
            f"{label} must be below the filesystem root: {resolved}"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    file_flags = os.O_RDONLY | nofollow
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(os.sep, directory_flags)
        for component in parts[1:-1]:
            next_fd = os.open(
                component,
                directory_flags,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or _file_identity(path_before) != _file_identity(before)
        ):
            raise InferenceContractError(
                f"{label} changed before it was opened: {resolved}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if _file_identity(before) != _file_identity(after):
            raise InferenceContractError(
                f"{label} changed while it was read: {resolved}"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise InferenceContractError(
                f"{label} size changed while it was read: {resolved}"
            )
        leaf_after = os.stat(
            parts[-1],
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        path_after = os.lstat(resolved)
        if (
            _file_identity(after) != _file_identity(leaf_after)
            or _file_identity(after) != _file_identity(path_after)
        ):
            raise InferenceContractError(
                f"{label} path changed while it was read: {resolved}"
            )
        observed = hashlib.sha256(payload).hexdigest()
        if expected is not None and observed != expected:
            raise InferenceContractError(
                f"{label} SHA mismatch: {observed} != {expected} "
                f"({resolved})"
            )
        return resolved, payload, observed, after
    except InferenceContractError:
        raise
    except OSError as exc:
        raise InferenceContractError(
            f"cannot safely read {label}: {resolved}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def sha256_file(path: str | Path) -> str:
    _resolved, _payload, observed, _metadata = _safe_file_snapshot(
        path,
        "file",
    )
    return observed


def _require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise InferenceContractError(f"{label} must be an exact integer, got {value!r}")
    return value


def compact_json_sha256(value: Any) -> str:
    """Match show_base_train._payload_sha256 for checkpoint source receipts."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    canonical_array = np.asarray(array)
    if canonical_array.ndim:
        canonical_array = np.ascontiguousarray(canonical_array)
    np.lib.format.write_array(
        buffer,
        canonical_array,
        allow_pickle=False,
    )
    return buffer.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != OUTPUT_FIELDS:
        raise ValueError(
            f"output NPZ fields must be exactly {OUTPUT_FIELDS}, "
            f"observed {tuple(arrays)}"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in OUTPUT_FIELDS:
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    return buffer.getvalue()


def _resolved_regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise InferenceContractError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _verify_file_sha(path: Path, expected: str, label: str) -> str:
    expected = _require_sha256(expected, f"{label} expected SHA")
    observed = sha256_file(path)
    if observed != expected:
        raise InferenceContractError(
            f"{label} SHA mismatch: {observed} != {expected} ({path})"
        )
    return observed


def _read_verified_checkpoint_snapshot(
    path: str | Path,
    expected: str,
    label: str,
) -> tuple[Path, bytes, str]:
    """Read, hash, and later deserialize one immutable byte snapshot."""
    resolved, payload, observed, _metadata = _safe_file_snapshot(
        path,
        label,
        expected_sha256=expected,
    )
    return resolved, payload, observed


def _load_canonical_clip(
    row: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], int]:
    path = _resolved_regular_file(row["canonical_npz"], "canonical NPZ")
    _verify_file_sha(
        path,
        str(row["canonical_npz_sha256"]),
        "canonical NPZ",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != CANONICAL_FIELDS:
                raise InferenceContractError(
                    f"{path}: fields {tuple(archive.files)} != {CANONICAL_FIELDS}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in CANONICAL_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise InferenceContractError(f"cannot read canonical NPZ {path}: {exc}") from exc
    frames = _require_exact_int(row.get("frames"), "canonical row frames")
    expected = {
        "pose": ((frames, POSE_DIM), np.float32),
        "contact": ((frames, 4), np.float32),
        "facial": ((frames, EXPRESSION_DIM), np.float32),
        "beta": ((frames, BETA_DIM), np.float32),
        "trans": ((frames, 3), np.float32),
        "speaker_id": ((frames, 1), np.int64),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise InferenceContractError(
                f"{path}: {name} {array.shape}/{array.dtype} != {shape}/{dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise InferenceContractError(f"{path}: non-finite {name}")
    contact = arrays["contact"]
    if not np.logical_or(contact == 0.0, contact == 1.0).all():
        raise InferenceContractError(f"{path}: contact is not binary")
    speaker_id = SHOW_SPEAKER_IDS[str(row["speaker"])]
    if not np.array_equal(
        arrays["speaker_id"],
        np.full((frames, 1), speaker_id, dtype=np.int64),
    ):
        raise InferenceContractError(f"{path}: speaker_id changes or is remapped")
    if not np.array_equal(
        arrays["beta"],
        np.broadcast_to(arrays["beta"][0], arrays["beta"].shape),
    ):
        raise InferenceContractError(f"{path}: betas vary within one SHOW clip")
    return arrays, frames


def _load_audio_features(
    row: Mapping[str, Any],
    *,
    expected_frames: int,
) -> dict[str, np.ndarray]:
    path = _resolved_regular_file(row["audio_feature_npz"], "audio feature NPZ")
    _verify_file_sha(
        path,
        str(row["audio_feature_npz_sha256"]),
        "audio feature NPZ",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != AUDIO_FIELDS:
                raise InferenceContractError(
                    f"{path}: fields {tuple(archive.files)} != {AUDIO_FIELDS}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in AUDIO_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise InferenceContractError(
            f"cannot read audio feature NPZ {path}: {exc}"
        ) from exc
    expected = {
        "beat": (expected_frames, 3),
        "hubert": (expected_frames, 1024),
    }
    for name, shape in expected.items():
        array = arrays[name]
        if (
            array.shape != shape
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise InferenceContractError(
                f"{path}: invalid {name} {array.shape}/{array.dtype}"
            )
    if (
        _require_exact_int(row.get("frames"), "audio feature row frames")
        != expected_frames
    ):
        raise InferenceContractError(f"{path}: feature frame count mismatch")
    return arrays


def _torch_load_checkpoint(
    payload_bytes: bytes,
    path: Path,
) -> dict[str, Any]:
    import torch

    buffer = io.BytesIO(payload_bytes)
    try:
        payload = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        payload = torch.load(buffer, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(
        payload.get("model_state"), dict
    ):
        raise InferenceContractError(f"{path}: checkpoint lacks model_state")
    return payload


def _finite_state_dict(state: Mapping[str, Any], path: Path) -> None:
    import torch

    for name, value in state.items():
        if torch.is_tensor(value) and (
            value.is_floating_point() or value.is_complex()
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise InferenceContractError(
                    f"{path}: non-finite checkpoint tensor {name}"
                )


def _normalize_data_parallel_state(
    state: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    import torch

    keys = list(state)
    if not keys:
        raise InferenceContractError(f"{path}: empty model_state")
    if any(not isinstance(key, str) for key in keys):
        raise InferenceContractError(f"{path}: model_state keys must be strings")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise InferenceContractError(
            f"{path}: every model_state value must be a tensor"
        )
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise InferenceContractError(
            f"{path}: mixed DataParallel and unprefixed state keys"
        )
    normalized = {
        (key[7:] if prefixed[0] else key): value
        for key, value in state.items()
    }
    if len(normalized) != len(state):
        raise InferenceContractError(f"{path}: state-key collision after normalization")
    forbidden_tokens = ("sparse", "semgate", "speaker2", "clip", "emotion")
    for key in normalized:
        lowered = key.lower()
        if any(token in lowered for token in forbidden_tokens):
            raise InferenceContractError(
                f"{path}: forbidden checkpoint component in key {key!r}"
            )
    return normalized


def _strict_load_freeze_eval(
    model: Any,
    state: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise InferenceContractError(
            f"{path}: strict released state load was not exact"
        )
    model.eval()
    model.requires_grad_(False)
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise InferenceContractError(
            f"{path}: released model did not freeze in eval mode"
        )


_BASE_MODEL_STATE_SCHEMA: dict[str, tuple[Any, tuple[int, ...]]] | None = None


def _expected_base_model_state_schema(
) -> dict[str, tuple[Any, tuple[int, ...]]]:
    global _BASE_MODEL_STATE_SCHEMA
    if _BASE_MODEL_STATE_SCHEMA is None:
        import torch
        from models.semtalk import semtalk_base

        try:
            with torch.device("meta"):
                reference = semtalk_base(_model_args())
        except Exception as exc:
            raise InferenceContractError(
                "cannot construct the strict SemTalk Base schema on the "
                "PyTorch meta device"
            ) from exc
        _BASE_MODEL_STATE_SCHEMA = {
            key: (value.dtype, tuple(value.shape))
            for key, value in reference.state_dict().items()
        }
        del reference
    return _BASE_MODEL_STATE_SCHEMA


def _validate_base_model_state_schema(
    state: Mapping[str, Any],
    path: Path,
) -> None:
    normalized = _normalize_data_parallel_state(state, path)
    expected = _expected_base_model_state_schema()
    if set(normalized) != set(expected):
        raise InferenceContractError(
            f"{path}: model_state keys are not the exact SemTalk Base schema"
        )
    for key, value in normalized.items():
        expected_dtype, expected_shape = expected[key]
        if value.dtype != expected_dtype or tuple(value.shape) != expected_shape:
            raise InferenceContractError(
                f"{path}: model_state tensor schema mismatch for {key!r}"
            )


def _model_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=768,
        audio_f=256,
        motion_f=256,
        pose_dims=330,
        pose_length=WINDOW,
        pre_frames=PRE_FRAMES,
        pose_fps=POSE_FPS,
        vae_codebook_size=CODEBOOK_SIZE,
        vae_test_dim=330,
        vae_layer=4,
        vae_length=240,
    )


def _callable_source_sha256(function: Any) -> str:
    try:
        lines, _ = inspect.getsourcelines(function)
    except (OSError, TypeError) as error:
        raise InferenceContractError(
            f"cannot inspect pinned callable {function!r}"
        ) from error
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _inference_auxiliary_loss_bypass_receipt() -> dict[str, Any]:
    model_path = (PROJECT_ROOT / PINNED_SEMTALK_MODEL_SOURCE["relative_path"])
    try:
        model_mode = os.lstat(model_path).st_mode
    except FileNotFoundError:
        raise InferenceContractError(
            f"missing pinned SemTalk model source: {model_path}"
        ) from None
    if stat.S_ISLNK(model_mode) or not stat.S_ISREG(model_mode):
        raise InferenceContractError(
            f"pinned SemTalk model source is not a regular file: {model_path}"
        )
    model_payload = model_path.read_bytes()
    model_blob = hashlib.sha1(
        f"blob {len(model_payload)}\0".encode("ascii") + model_payload
    ).hexdigest()
    if (
        hashlib.sha256(model_payload).hexdigest()
        != PINNED_SEMTALK_MODEL_SOURCE["sha256"]
        or model_blob != PINNED_SEMTALK_MODEL_SOURCE["git_blob_sha1"]
    ):
        raise InferenceContractError("pinned SemTalk model source changed")

    from models.semtalk import RhythmicIdentificationLoss, semtalk_base

    model_resolved = model_path.resolve(strict=True)
    semtalk_source = inspect.getsourcefile(semtalk_base)
    loss_source = inspect.getsourcefile(RhythmicIdentificationLoss)
    if (
        semtalk_base.__module__ != "models.semtalk"
        or RhythmicIdentificationLoss.__module__ != "models.semtalk"
        or not isinstance(semtalk_source, str)
        or not isinstance(loss_source, str)
        or Path(semtalk_source).resolve(strict=True) != model_resolved
        or Path(loss_source).resolve(strict=True) != model_resolved
        or str(inspect.signature(semtalk_base.forward))
        != PINNED_SEMTALK_BASE_FORWARD["signature"]
        or _callable_source_sha256(semtalk_base.forward)
        != PINNED_SEMTALK_BASE_FORWARD["source_sha256"]
        or str(inspect.signature(RhythmicIdentificationLoss.forward))
        != PINNED_RHYTHMIC_LOSS_FORWARD["signature"]
        or _callable_source_sha256(RhythmicIdentificationLoss.forward)
        != PINNED_RHYTHMIC_LOSS_FORWARD["source_sha256"]
        or _callable_source_sha256(_rvq_indices)
        != PINNED_RVQ_INDICES_SOURCE_SHA256
        or str(inspect.signature(_infer_clip))
        != PINNED_INFER_CLIP["signature"]
        or _callable_source_sha256(_infer_clip)
        != PINNED_INFER_CLIP["source_sha256"]
    ):
        raise InferenceContractError(
            "inference-only auxiliary-loss bypass source contract changed"
        )
    receipt = {
        "format": INFERENCE_AUXILIARY_LOSS_BYPASS_FORMAT,
        "scope": "validation_and_test_inference_only",
        "model_source": dict(PINNED_SEMTALK_MODEL_SOURCE),
        "base_forward": dict(PINNED_SEMTALK_BASE_FORWARD),
        "loss_forward": dict(PINNED_RHYTHMIC_LOSS_FORWARD),
        "rvq_indices_source_sha256": PINNED_RVQ_INDICES_SOURCE_SHA256,
        "infer_clip": dict(PINNED_INFER_CLIP),
        "patched_attributes": list(INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS),
        "patch_mechanism": "instance_forward_MethodType_with_finally_restore",
        "replacement": "first_input_new_zeros_scalar",
        "module_identity_preserved": True,
        "state_dict_preserved": True,
        "deterministic_algorithms_required": True,
        "deterministic_warn_only_required": False,
        "model_eval_required": True,
        "all_parameters_frozen_required": True,
        "torch_inference_mode_required": True,
    }
    receipt["receipt_sha256"] = compact_json_sha256(receipt)
    return receipt


@contextmanager
def _inference_only_auxiliary_loss_bypass(
    base: Any,
    *,
    expected_calls: int,
) -> Iterable[dict[str, Any]]:
    import torch
    from models.semtalk import RhythmicIdentificationLoss, semtalk_base

    receipt = _inference_auxiliary_loss_bypass_receipt()
    if (
        type(expected_calls) is not int
        or expected_calls < 1
        or type(base) is not semtalk_base
        or base.training
        or any(parameter.requires_grad for parameter in base.parameters())
        or not torch.is_inference_mode_enabled()
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise InferenceContractError(
            "auxiliary-loss bypass requires frozen eval inference under "
            "strict deterministic algorithms"
        )

    modules: dict[str, Any] = {}
    original_forwards: dict[str, Any] = {}
    patched_forwards: dict[str, Any] = {}
    calls = {name: 0 for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS}
    for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS:
        module = getattr(base, name, None)
        if (
            type(module) is not RhythmicIdentificationLoss
            or module.temperature != 0.1
            or module.training
            or module in modules.values()
            or module is not base._modules.get(name)
            or "forward" in module.__dict__
            or module.state_dict()
            or tuple(module.parameters())
            or tuple(module.buffers())
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
        ):
            raise InferenceContractError(
                f"{name}: auxiliary-loss module contract changed"
            )
        modules[name] = module
        original_forwards[name] = module.forward

    def make_zero_loss(name: str, module: Any) -> Any:
        def zero_loss(
            self: Any,
            facial_features: Any,
            audio_features: Any,
        ) -> Any:
            if (
                self is not module
                or not torch.is_inference_mode_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not isinstance(facial_features, torch.Tensor)
                or not isinstance(audio_features, torch.Tensor)
                or tuple(facial_features.shape) != (1, 16, 256)
                or tuple(audio_features.shape) != (1, 64, 256)
                or facial_features.device != audio_features.device
                or facial_features.dtype != audio_features.dtype
                or facial_features.requires_grad
                or audio_features.requires_grad
            ):
                raise InferenceContractError(
                    f"{name}: invalid inference-only auxiliary-loss call"
                )
            calls[name] += 1
            return facial_features.new_zeros(())

        return MethodType(zero_loss, module)

    try:
        for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS:
            module = modules[name]
            patched = make_zero_loss(name, module)
            patched_forwards[name] = patched
            module.forward = patched
            if module.__dict__.get("forward") is not patched:
                raise InferenceContractError(
                    f"{name}: failed to install auxiliary-loss bypass"
                )
    except BaseException:
        for name in reversed(tuple(patched_forwards)):
            module = modules[name]
            if module.__dict__.get("forward") is patched_forwards[name]:
                delattr(module, "forward")
        raise

    activation = {
        **receipt,
        "expected_calls_per_loss": expected_calls,
        "calls": calls,
    }
    completed = False
    try:
        yield activation
        completed = True
    finally:
        restoration_error: str | None = None
        for name in reversed(INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS):
            module = modules[name]
            installed = module.__dict__.get("forward")
            if installed is not patched_forwards[name]:
                restoration_error = (
                    f"{name}: auxiliary-loss bypass was changed while active"
                )
            if "forward" in module.__dict__:
                delattr(module, "forward")
            restored = module.forward
            original = original_forwards[name]
            if (
                getattr(restored, "__self__", None) is not module
                or getattr(restored, "__func__", None)
                is not getattr(original, "__func__", None)
                or getattr(base, name, None) is not module
                or module is not base._modules.get(name)
                or module.state_dict()
            ):
                restoration_error = (
                    f"{name}: auxiliary-loss module restoration failed"
                )
        if restoration_error is not None:
            raise InferenceContractError(restoration_error)
    if completed and any(
        calls[name] != expected_calls
        for name in INFERENCE_AUXILIARY_LOSS_BYPASS_ATTRS
    ):
        raise InferenceContractError(
            "auxiliary-loss bypass call count does not match inference rounds"
        )
    if completed:
        activation["activation_sha256"] = compact_json_sha256(activation)


def _joint_masks(device: Any) -> dict[str, Any]:
    import torch
    from dataloaders.data_tools import joints_list

    source = joints_list["beat_smplx_joints"]
    targets = {
        "upper": joints_list["beat_smplx_upper"],
        "hands": joints_list["beat_smplx_hands"],
        "lower": joints_list["beat_smplx_lower"],
    }
    result: dict[str, Any] = {}
    observed = torch.zeros(POSE_DIM, dtype=torch.int64, device=device)
    expected_counts = {"upper": 13 * 3, "hands": 30 * 3, "lower": 9 * 3}
    for name, target in targets.items():
        mask = np.zeros(POSE_DIM, dtype=bool)
        for joint_name in target:
            width, end = source[joint_name]
            mask[end - width : end] = True
        if int(mask.sum()) != expected_counts[name]:
            raise InferenceContractError(f"{name}: unexpected joint mask width")
        tensor = torch.from_numpy(mask).to(device=device)
        observed += tensor.to(torch.int64)
        result[name] = tensor
    if bool((observed > 1).any().item()):
        raise InferenceContractError("upper/hands/lower joint masks overlap")
    return result


def _aa_to_rotation_6d(axis_angle: Any) -> Any:
    from utils import rotation_conversions as rc

    batch, frames, dimensions = axis_angle.shape
    if dimensions != POSE_DIM:
        raise InferenceContractError(f"axis-angle pose has shape {axis_angle.shape}")
    return rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(axis_angle.reshape(batch, frames, 55, 3))
    ).reshape(batch, frames, 330)


def _scatter_axis_angle(
    values: Any,
    mask: Any,
    *,
    batch: int,
    frames: int,
) -> Any:
    import torch

    result = torch.zeros(
        batch,
        frames,
        POSE_DIM,
        dtype=values.dtype,
        device=values.device,
    )
    result[..., mask] = values.reshape(batch, frames, -1)
    return result


def _decode_body_axis_angle(
    upper: Any,
    lower: Any,
    hands: Any,
    masks: Mapping[str, Any],
) -> tuple[Any, Any]:
    from utils import rotation_conversions as rc

    batch, frames = int(lower.shape[0]), int(lower.shape[1])
    expected = {
        "upper": (batch, frames, 78),
        "lower": (batch, frames, 61),
        "hands": (batch, frames, 180),
    }
    observed = {
        "upper": tuple(upper.shape),
        "lower": tuple(lower.shape),
        "hands": tuple(hands.shape),
    }
    if observed != expected:
        raise InferenceContractError(
            f"decoded RVQ body shapes {observed} != {expected}"
        )
    upper_aa = rc.matrix_to_axis_angle(
        rc.rotation_6d_to_matrix(upper.reshape(batch, frames, 13, 6))
    ).reshape(batch, frames, 39)
    lower_matrix = rc.rotation_6d_to_matrix(
        lower[..., :54].reshape(batch, frames, 9, 6)
    )
    lower_aa = rc.matrix_to_axis_angle(lower_matrix).reshape(batch, frames, 27)
    hands_aa = rc.matrix_to_axis_angle(
        rc.rotation_6d_to_matrix(hands.reshape(batch, frames, 30, 6))
    ).reshape(batch, frames, 90)
    pose = (
        _scatter_axis_angle(
            upper_aa,
            masks["upper"],
            batch=batch,
            frames=frames,
        )
        + _scatter_axis_angle(
            lower_aa,
            masks["lower"],
            batch=batch,
            frames=frames,
        )
        + _scatter_axis_angle(
            hands_aa,
            masks["hands"],
            batch=batch,
            frames=frames,
        )
    )
    lower_projected_6d = rc.matrix_to_rotation_6d(lower_matrix).reshape(
        batch,
        frames,
        54,
    )
    return pose, lower_projected_6d


def _velocity_to_position(velocity: Any, anchor: Any) -> Any:
    import torch

    if velocity.ndim != 3 or velocity.shape[-1] != 1:
        raise InferenceContractError(
            f"velocity must be [B,T,1], got {tuple(velocity.shape)}"
        )
    positions = [anchor.unsqueeze(1)]
    for frame in range(1, int(velocity.shape[1])):
        positions.append(positions[-1] + velocity[:, frame - 1 : frame] / POSE_FPS)
    return torch.cat(positions, dim=1)


def _translation_from_channels(channels: Any, anchor: Any) -> Any:
    import torch

    if channels.ndim != 3 or channels.shape[-1] != 3:
        raise InferenceContractError(
            f"translation channels must be [B,T,3], got {tuple(channels.shape)}"
        )
    x = _velocity_to_position(channels[..., 0:1], anchor[:, 0:1])
    z = _velocity_to_position(channels[..., 2:3], anchor[:, 2:3])
    return torch.cat([x, channels[..., 1:2], z], dim=-1)


def _edge_pad_frames(array: Any, required_frames: int) -> Any:
    import torch

    current = int(array.shape[1])
    if current >= required_frames:
        return array[:, :required_frames]
    if current < 1:
        raise InferenceContractError("cannot edge-pad an empty feature sequence")
    extension = array[:, -1:].expand(
        array.shape[0],
        required_frames - current,
        *array.shape[2:],
    )
    return torch.cat([array, extension], dim=1)


def _rvq_indices(output: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for name in RVQ_DIMS:
        key = f"cls_{name}"
        logits = output.get(key)
        expected = (1, RVQ_TOKENS_PER_WINDOW, CODEBOOK_SIZE, RVQ_LEVELS)
        if logits is None or tuple(logits.shape) != expected:
            shape = None if logits is None else tuple(logits.shape)
            raise InferenceContractError(f"{key} {shape} != {expected}")
        if not bool(logits.isfinite().all().item()):
            raise InferenceContractError(f"{key} contains NaN or Inf")
        indices = logits.argmax(dim=2)
        if (
            tuple(indices.shape)
            != (1, RVQ_TOKENS_PER_WINDOW, RVQ_LEVELS)
            or indices.dtype != torch.int64
        ):
            raise InferenceContractError(
                f"{key} indices have invalid shape/dtype: "
                f"{tuple(indices.shape)}/{indices.dtype}"
            )
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= CODEBOOK_SIZE:
            raise InferenceContractError(f"{key} index outside codebook")
        result[name] = indices
    return result


def _decode_checked(
    model: Any,
    indices: Any,
    *,
    name: str,
    dimension: int,
) -> Any:
    decoded = model.decode(indices)
    expected_frames = int(indices.shape[1]) * RVQ_TOKEN_FRAMES
    expected = (1, expected_frames, dimension)
    if tuple(decoded.shape) != expected:
        raise InferenceContractError(
            f"{name} RVQ decode {tuple(decoded.shape)} != {expected}"
        )
    if not bool(decoded.isfinite().all().item()):
        raise InferenceContractError(f"{name} RVQ decode contains NaN or Inf")
    return decoded


def _infer_clip(
    *,
    pose: np.ndarray,
    trans: np.ndarray,
    beat: np.ndarray,
    hubert: np.ndarray,
    speaker_id: int,
    models: Mapping[str, Any],
    masks: Mapping[str, Any],
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    from utils import rotation_conversions as rc

    frames = int(pose.shape[0])
    if frames < PRE_FRAMES:
        raise InferenceContractError("clip is shorter than the four-frame seed")
    rounds = max(1, math.ceil((frames - PRE_FRAMES) / STRIDE))
    generated_frames = PRE_FRAMES + STRIDE * rounds
    expected_tokens = RVQ_TOKENS_PER_WINDOW + (
        rounds - 1
    ) * (RVQ_TOKENS_PER_WINDOW - 1)
    if expected_tokens * RVQ_TOKEN_FRAMES != generated_frames:
        raise AssertionError("RVQ stitching arithmetic is inconsistent")

    pose_tensor = torch.from_numpy(pose[:PRE_FRAMES]).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    trans_tensor = torch.from_numpy(trans[:PRE_FRAMES]).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    anchor = trans_tensor[:, 0]
    # Contact is intentionally zero: the only GT conditioning contract is the
    # first four pose/translation frames.
    seed = torch.cat(
        [
            _aa_to_rotation_6d(pose_tensor),
            trans_tensor,
            torch.zeros(1, PRE_FRAMES, 4, device=device),
        ],
        dim=-1,
    )
    if tuple(seed.shape) != (1, PRE_FRAMES, 337):
        raise AssertionError("seed shape is not [1,4,337]")

    beat_tensor = _edge_pad_frames(
        torch.from_numpy(beat).to(device=device, dtype=torch.float32).unsqueeze(0),
        generated_frames,
    )
    hubert_tensor = _edge_pad_frames(
        torch.from_numpy(hubert).to(
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0),
        generated_frames,
    )
    in_word = torch.zeros(
        1,
        generated_frames,
        dtype=torch.long,
        device=device,
    )
    if bool(in_word.any().item()):
        raise AssertionError("in_word must be identically zero")

    stitched: dict[str, list[Any]] = {name: [] for name in RVQ_DIMS}
    latent_last = None
    base = models["base"]
    with torch.inference_mode():
        for round_index in range(rounds):
            start = round_index * STRIDE
            motion = torch.zeros(1, WINDOW, 337, device=device)
            motion[:, :PRE_FRAMES] = (
                seed
                if round_index == 0
                else latent_last[:, -PRE_FRAMES:]
            )
            mask = torch.ones_like(motion)
            mask[:, :PRE_FRAMES] = 0.0
            if bool(motion[:, PRE_FRAMES:].any().item()):
                raise AssertionError("future motion payload is not identically zero")
            identifier = torch.full(
                (1, WINDOW, 1),
                speaker_id,
                dtype=torch.long,
                device=device,
            )
            output = base(
                in_audio=beat_tensor[:, start : start + WINDOW],
                in_word=in_word[:, start : start + WINDOW],
                mask=mask,
                in_motion=motion,
                in_id=identifier,
                hubert=hubert_tensor[:, start : start + WINDOW],
                use_attentions=True,
                is_train=False,
            )
            indices = _rvq_indices(output)
            for name, value in indices.items():
                stitched[name].append(
                    value if round_index == 0 else value[:, 1:]
                )

            upper = _decode_checked(
                models["upper"],
                indices["upper"],
                name="upper",
                dimension=RVQ_DIMS["upper"],
            )
            lower = _decode_checked(
                models["lower"],
                indices["lower"],
                name="lower",
                dimension=RVQ_DIMS["lower"],
            )
            hands = _decode_checked(
                models["hands"],
                indices["hands"],
                name="hands",
                dimension=RVQ_DIMS["hands"],
            )
            body_axis_angle, _ = _decode_body_axis_angle(
                upper,
                lower,
                hands,
                masks,
            )
            preliminary_trans = _translation_from_channels(
                lower[..., 54:57],
                anchor,
            )
            latent_last = torch.cat(
                [
                    _aa_to_rotation_6d(body_axis_angle),
                    preliminary_trans,
                    lower[..., 57:61],
                ],
                dim=-1,
            )
            if tuple(latent_last.shape) != (1, WINDOW, 337):
                raise AssertionError("generated recurrent latent has wrong shape")
            if not bool(latent_last.isfinite().all().item()):
                raise InferenceContractError(
                    "generated recurrent latent contains NaN or Inf"
                )

        stitched_indices = {
            name: torch.cat(parts, dim=1)
            for name, parts in stitched.items()
        }
        for name, indices in stitched_indices.items():
            if tuple(indices.shape) != (1, expected_tokens, RVQ_LEVELS):
                raise InferenceContractError(
                    f"{name} stitched indices have shape {tuple(indices.shape)}"
                )
        decoded = {
            name: _decode_checked(
                models[name],
                stitched_indices[name],
                name=name,
                dimension=dimension,
            )
            for name, dimension in RVQ_DIMS.items()
        }
        body_axis_angle, lower_projected_6d = _decode_body_axis_angle(
            decoded["upper"],
            decoded["lower"],
            decoded["hands"],
            masks,
        )
        face = decoded["face"]
        jaw = rc.matrix_to_axis_angle(
            rc.rotation_6d_to_matrix(face[..., :6].reshape(1, generated_frames, 1, 6))
        ).reshape(1, generated_frames, 3)
        body_axis_angle[..., 66:69] = jaw
        body_axis_angle[..., 69:75] = 0.0

        to_global = decoded["lower"].clone()
        to_global[..., :54] = lower_projected_6d
        to_global[..., 54:57] = 0.0
        global_output = models["global"](to_global)
        global_pose = global_output.get("rec_pose")
        if (
            global_pose is None
            or tuple(global_pose.shape) != (1, generated_frames, 61)
            or not bool(global_pose.isfinite().all().item())
        ):
            shape = None if global_pose is None else tuple(global_pose.shape)
            raise InferenceContractError(
                f"global/root VAE output is invalid: {shape}"
            )
        prediction_trans = _translation_from_channels(
            global_pose[..., 54:57],
            anchor,
        )
        prediction_expression = face[..., 6:]

    result = {
        "poses": body_axis_angle[0, :frames].float().cpu().numpy(),
        "expressions": prediction_expression[0, :frames].float().cpu().numpy(),
        "trans": prediction_trans[0, :frames].float().cpu().numpy(),
    }
    expected_shapes = {
        "poses": (frames, POSE_DIM),
        "expressions": (frames, EXPRESSION_DIM),
        "trans": (frames, 3),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(result[name], dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise InferenceContractError(
                f"invalid generated {name}: {array.shape}/{array.dtype}"
            )
        result[name] = array
    if not np.array_equal(
        result["poses"][:, 69:75],
        np.zeros((frames, 6), dtype=np.float32),
    ):
        raise InferenceContractError("unpredicted eye pose dimensions are not zero")
    return result


def _output_arrays(
    *,
    betas: np.ndarray,
    poses: np.ndarray,
    expressions: np.ndarray,
    trans: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "betas": np.asarray(betas, dtype=np.float32).reshape(BETA_DIM),
        "poses": np.asarray(poses, dtype=np.float32),
        "expressions": np.asarray(expressions, dtype=np.float32),
        "trans": np.asarray(trans, dtype=np.float32),
        "model": np.asarray("smplx2020"),
        "gender": np.asarray("neutral"),
        "mocap_frame_rate": np.asarray(POSE_FPS, dtype=np.int64),
    }
    frames = int(arrays["poses"].shape[0])
    expected = {
        "betas": (BETA_DIM,),
        "poses": (frames, POSE_DIM),
        "expressions": (frames, EXPRESSION_DIM),
        "trans": (frames, 3),
        "model": (),
        "gender": (),
        "mocap_frame_rate": (),
    }
    if tuple(arrays) != OUTPUT_FIELDS:
        raise AssertionError("output field order is not canonical")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise InferenceContractError(
                f"output field {name} shape {arrays[name].shape} != {shape}"
            )
        if arrays[name].dtype.kind in "fc" and not np.isfinite(arrays[name]).all():
            raise InferenceContractError(f"output field {name} is non-finite")
    return arrays
