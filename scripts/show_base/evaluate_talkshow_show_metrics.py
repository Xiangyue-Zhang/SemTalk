#!/usr/bin/env python3
"""Offline SemTalk NPZ adapter for the released TalkSHOW SHOW metrics.

This module deliberately has no dependency on a SemTalk or GlobalDiff
generator.  It consumes the immutable canonical SHOW manifest, the finalized
SemTalk prediction manifest, and pinned metric-only assets.  A single
deterministic SemTalk prediction is interpreted as a delta distribution:

* released2 uses logical slots 0 and 1;
* paper16 uses logical slots 0 through 15;
* released face and DiffSHEG use logical slot 0.

Every logical slot is bound to the same physical prediction artifact.  The
single prediction is repeated before the pinned metric models, producing the
same extractor/SMPL-X batch shapes as the released two- and sixteen-sample
paths while preserving one generator inference.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
import wave
import zipfile

import numpy as np


SCHEMA_VERSION = 1
REPORT_FORMAT = "semtalk_show_talkshow_metrics_v1"
REPORT_PAYLOAD_HASH_ALGORITHM = (
    "canonical_json_utf8_sorted_compact_newline_v1"
)
DISTRIBUTION_PROTOCOL = (
    "deterministic_replication_of_single_prediction_v1"
)
REPLICATION_ALGORITHM = "logical_reference_v1"
NUM_LOGICAL_SLOTS = 16
RELEASED2_SLOTS = (0, 1)
PAPER16_SLOTS = tuple(range(NUM_LOGICAL_SLOTS))
FACE_SLOT = 0
DIFFSHEG_SLOT = 0
METRIC_INPUT_MATERIALIZATION = {
    "generator_inferences_per_clip": 1,
    "feature_extractor_batches": [2, 16],
    "smplx_body_batch": 16,
    "face_smplx_batch": 2,
    "materialization_operation": (
        "repeat_prediction_before_metric_model_v1"
    ),
}
POSE_FPS = 30
POSE_DIM = 165
EXPRESSION_DIM = 100
BETA_DIM = 300
SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
SPEAKER_NAMES = tuple(SHOW_SPEAKER_IDS)

OUTPUT_FIELDS = (
    "betas",
    "poses",
    "expressions",
    "trans",
    "model",
    "gender",
    "mocap_frame_rate",
)
CANONICAL_FIELDS = (
    "pose",
    "contact",
    "facial",
    "beta",
    "trans",
    "speaker_id",
)
CANONICAL_ROW_KEYS = {
    "global_index",
    "clip_id",
    "split",
    "speaker",
    "speaker_id",
    "video",
    "sequence",
    "source_pkl",
    "source_wav",
    "canonical_npz",
    "canonical_npz_relative",
    "global_foot_fastpath_contract",
    "lower_foot_local",
    "lower_foot_local_relative",
    "frames",
    "pose_fps",
    "wav_channels",
    "wav_sample_width",
    "wav_sample_rate",
    "wav_frames",
    "wav_mono_policy",
    "source_pkl_sha256",
    "source_wav_sha256",
    "canonical_npz_sha256",
    "lower_foot_local_sha256",
    "lineage_contract_sha256",
}
VAL_PREDICTION_ROW_KEYS = {
    "global_index",
    "split",
    "source_clip_id",
    "canonical_clip_id",
    "frames",
    "epoch",
    "candidate_checkpoint_sha256",
    "prediction",
    "ground_truth",
}
TEST_PREDICTION_ROW_KEYS = {
    "global_index",
    "source_clip_id",
    "canonical_clip_id",
    "speaker",
    "speaker_id",
    "frames",
    "canonical_npz",
    "canonical_npz_sha256",
    "audio_feature_npz",
    "audio_feature_npz_sha256",
    "prediction",
    "ground_truth",
    "evaluation_index",
}
ARTIFACT_KEYS = {"path", "bytes", "sha256"}
SHA256_LENGTH = 64
PAYLOAD_HASH_ALGORITHM = "canonical_json_utf8_sorted_compact_v1"
REPLICATION_GATE_MODULE_ENV = "SEMTALK_SHOW_REPLICATION_GATE_MODULE"

TALKSHOW_METRIC_COMMIT = "9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
FEATURE_EXTRACTOR_SHA256 = (
    "154259bfe8ae1e0fb477ba5afdd5674659ef9eb4d44d4ad49cd2ef4d20ccecfb"
)
SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
TALKSHOW_PATCH_MARKERS = (
    {
        "upstream_commit": TALKSHOW_METRIC_COMMIT,
        "patch_version": 6,
    },
    {
        "upstream_commit": TALKSHOW_METRIC_COMMIT,
        "patch_version": "globaldiff-show-metric-v1",
        "scope": "released-show-body-feature-extractor-only",
        "source_file": "nets/__init__.py",
        "source_sha256": (
            "d12f3ebd1b8f4b251085061404d72530df58bc972272995a3810236aefdd4d21"
        ),
        "patched_sha256": (
            "e0e277d46475c203d78d355affd01a9e6f55462fbaa1911d140f91c438a0bef2"
        ),
    },
)
TALKSHOW_FGD_ENTRY_MODULES = ("nets.body_ae",)
TALKSHOW_FGD_SOURCE_FILES = (
    "data_utils/__init__.py",
    "data_utils/consts.py",
    "data_utils/dataloader_torch.py",
    "data_utils/lower_body.py",
    "data_utils/mesh_dataset.py",
    "data_utils/rotation_conversion.py",
    "data_utils/utils.py",
    "nets/__init__.py",
    "nets/base.py",
    "nets/body_ae.py",
    "nets/layers.py",
    "nets/spg/s2glayers.py",
    "nets/spg/vqvae_1d.py",
    "nets/spg/vqvae_modules.py",
    "nets/spg/wav2vec.py",
)
TALKSHOW_FGD_IMPORT_ALGORITHM = "python-ast-local-import-closure-v1"

PROTOCOLS = {
    "released2": {
        "logical_slots": list(RELEASED2_SLOTS),
        "num_samples": len(RELEASED2_SLOTS),
        "bc_sample_policy": "first",
    },
    "paper16": {
        "logical_slots": list(PAPER16_SLOTS),
        "num_samples": len(PAPER16_SLOTS),
        "bc_sample_policy": "all",
    },
}

BC_CHANGE_ANGLE = np.asarray(
    [6.0181e-05, 5.1597e-05, 2.1344e-04, 2.1899e-04],
    dtype=np.float64,
)
TALKSHOW_RELEASED_LOWER_POSE = np.asarray(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0747,
        -0.0158,
        -0.0152,
        -1.1826512813568115,
        0.23866955935955048,
        0.15146760642528534,
        -1.2604516744613647,
        -0.3160211145877838,
        -0.1603458970785141,
        1.1654603481292725,
        0.0,
        0.0,
        1.2521806955337524,
        0.041598282754421234,
        -0.06312154978513718,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


class MetricAdapterContractError(RuntimeError):
    """Raised when the offline metric evidence is not self-consistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compact_canonical_json_bytes(value: Any) -> bytes:
    """Canonical receipt bytes shared with deterministic_replication_gate."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compact_canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(compact_canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_REPLICATION_GATE_MODULE: Any | None = None


def _replication_gate_module() -> Any:
    """Load the sole deterministic-replication receipt implementation."""

    global _REPLICATION_GATE_MODULE
    if _REPLICATION_GATE_MODULE is not None:
        return _REPLICATION_GATE_MODULE
    try:
        module = importlib.import_module(
            "scripts.show_base.deterministic_replication_gate"
        )
    except ModuleNotFoundError as exc:
        override = os.environ.get(REPLICATION_GATE_MODULE_ENV)
        if not override:
            raise MetricAdapterContractError(
                "deterministic_replication_gate is required; the metric "
                "adapter does not maintain a second receipt schema"
            ) from exc
        source = Path(override).expanduser().resolve()
        if not source.is_file():
            raise MetricAdapterContractError(
                f"{REPLICATION_GATE_MODULE_ENV} is not a file: {source}"
            ) from exc
        spec = importlib.util.spec_from_file_location(
            "_semtalk_show_deterministic_replication_gate",
            source,
        )
        if spec is None or spec.loader is None:
            raise MetricAdapterContractError(
                f"cannot load replication gate module {source}"
            ) from exc
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if (
        getattr(module, "PAYLOAD_HASH_ALGORITHM", None)
        != PAYLOAD_HASH_ALGORITHM
        or getattr(module, "DISTRIBUTION_FORMAT", None)
        != "semtalk_show_deterministic_distribution_receipt_v1"
        or getattr(module, "PROTOCOL", None) != DISTRIBUTION_PROTOCOL
        or not callable(
            getattr(module, "validate_distribution_receipt", None)
        )
        or not callable(
            getattr(
                module,
                "build_distribution_receipt_from_validated_artifacts",
                None,
            )
        )
    ):
        raise MetricAdapterContractError(
            "deterministic replication gate API/schema version mismatch"
        )
    _REPLICATION_GATE_MODULE = module
    return module


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MetricAdapterContractError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _require_exact_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise MetricAdapterContractError(
            f"{label} must be an exact integer, got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise MetricAdapterContractError(
            f"{label} must be >= {minimum}, got {value}"
        )
    return value


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise MetricAdapterContractError(
            f"{name} contains NaN or infinity"
        )
    return array


def _resolved_regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise MetricAdapterContractError(
            f"{label} must not be a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MetricAdapterContractError(
            f"{label} is not a regular file: {resolved}"
        )
    return resolved


def _verified_file_snapshot(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    expected = _require_sha256(expected_sha256, f"{label} expected SHA")
    resolved = _resolved_regular_file(path, label)
    payload = resolved.read_bytes()
    observed = sha256_bytes(payload)
    if observed != expected:
        raise MetricAdapterContractError(
            f"{label} SHA-256 {observed} != {expected}"
        )
    return resolved, payload


def _strict_json_snapshot(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetricAdapterContractError(
            f"{label} is not valid JSON"
        ) from exc
    if type(value) is not dict:
        raise MetricAdapterContractError(f"{label} must be a JSON object")
    if canonical_json_bytes(value) != payload:
        raise MetricAdapterContractError(
            f"{label} is not canonical JSON bytes"
        )
    return value


def _strict_jsonl_snapshot(
    payload: bytes,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise MetricAdapterContractError(
                f"{label} line {line_number} lacks a newline"
            )
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetricAdapterContractError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if type(value) is not dict or canonical_json_bytes(value) != line:
            raise MetricAdapterContractError(
                f"{label} line {line_number} is not a canonical JSON object"
            )
        rows.append(value)
    if not rows:
        raise MetricAdapterContractError(f"{label} is empty")
    return rows


def canonical_clip_id(source_clip_id: str) -> str:
    pieces = source_clip_id.split("/")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        raise MetricAdapterContractError(
            f"invalid canonical SHOW clip ID: {source_clip_id!r}"
        )
    speaker, _video, sequence = pieces
    if speaker not in SHOW_SPEAKER_IDS:
        raise MetricAdapterContractError(
            f"unknown SHOW speaker {speaker!r}"
        )
    if (
        "__" in speaker
        or sequence in {"", ".", ".."}
        or "/" in sequence
        or "\x00" in sequence
    ):
        raise MetricAdapterContractError(
            f"unsafe canonical SHOW clip ID: {source_clip_id!r}"
        )
    return f"{speaker}__{sequence}"


@dataclass
class FeatureMoments:
    """Mergeable sufficient statistics using the unbiased covariance."""

    count: int = 0
    dimension: int = 0
    feature_sum: np.ndarray | None = None
    feature_outer_sum: np.ndarray | None = None

    def update(self, features: Any, *, repeat: int = 1) -> None:
        array = _finite_array(features, name="features")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or 0 in array.shape:
            raise MetricAdapterContractError(
                f"features must be a non-empty matrix, got {array.shape}"
            )
        repeats = _require_exact_int(repeat, "feature repeat", minimum=1)
        if repeats != 1:
            # Materialize the metric rows, not the prediction artifact.  This
            # exactly matches np.repeat followed by the released sum and
            # matrix-product reduction, including floating-point reduction
            # order and the unbiased (N-1) denominator.
            array = np.repeat(array, repeats, axis=0)
        if self.dimension not in (0, int(array.shape[1])):
            raise MetricAdapterContractError(
                "feature dimension changed during evaluation"
            )
        if self.count == 0:
            self.dimension = int(array.shape[1])
            self.feature_sum = np.zeros(self.dimension, dtype=np.float64)
            self.feature_outer_sum = np.zeros(
                (self.dimension, self.dimension),
                dtype=np.float64,
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        self.count += int(array.shape[0])
        self.feature_sum += array.sum(axis=0, dtype=np.float64)
        self.feature_outer_sum += array.T @ array

    def merge(self, other: "FeatureMoments") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.dimension = other.dimension
            assert other.feature_sum is not None
            assert other.feature_outer_sum is not None
            self.feature_sum = other.feature_sum.copy()
            self.feature_outer_sum = other.feature_outer_sum.copy()
            return
        if self.dimension != other.dimension:
            raise MetricAdapterContractError(
                "cannot merge moments with different feature dimensions"
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        assert other.feature_sum is not None
        assert other.feature_outer_sum is not None
        self.count += other.count
        self.feature_sum += other.feature_sum
        self.feature_outer_sum += other.feature_outer_sum

    def mean_and_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise MetricAdapterContractError(
                "at least two feature rows are required"
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        mean = self.feature_sum / self.count
        covariance = (
            self.feature_outer_sum
            - np.outer(self.feature_sum, self.feature_sum) / self.count
        ) / (self.count - 1)
        covariance = (covariance + covariance.T) * 0.5
        if not np.isfinite(covariance).all():
            raise MetricAdapterContractError(
                "unbiased feature covariance is non-finite"
            )
        return mean, covariance

    def to_json(self) -> dict[str, Any]:
        if self.count < 1:
            raise MetricAdapterContractError(
                "cannot serialize empty feature moments"
            )
        assert self.feature_sum is not None
        assert self.feature_outer_sum is not None
        return {
            "count": self.count,
            "dimension": self.dimension,
            "sum": self.feature_sum.tolist(),
            "outer_sum": self.feature_outer_sum.tolist(),
            "covariance_denominator": self.count - 1,
            "covariance_estimator": "unbiased_ddof_1",
        }


def frechet_distance(first: FeatureMoments, second: FeatureMoments) -> float:
    mean_1, covariance_1 = first.mean_and_covariance()
    mean_2, covariance_2 = second.mean_and_covariance()
    if mean_1.shape != mean_2.shape:
        raise MetricAdapterContractError("feature dimensions differ")
    eigenvalues_1, eigenvectors_1 = np.linalg.eigh(covariance_1)
    scale = max(
        1.0,
        float(np.linalg.norm(covariance_1, ord=2)),
        float(np.linalg.norm(covariance_2, ord=2)),
    )
    tolerance = 1e-8 * scale
    if float(eigenvalues_1.min(initial=0.0)) < -tolerance:
        raise MetricAdapterContractError(
            "first covariance is not positive semidefinite"
        )
    root_1 = (
        eigenvectors_1
        * np.sqrt(np.clip(eigenvalues_1, 0.0, None))
    ) @ eigenvectors_1.T
    middle = root_1 @ covariance_2 @ root_1
    middle = (middle + middle.T) * 0.5
    middle_eigenvalues = np.linalg.eigvalsh(middle)
    if float(middle_eigenvalues.min(initial=0.0)) < -tolerance:
        raise MetricAdapterContractError(
            "covariance product is not positive semidefinite"
        )
    difference = mean_1 - mean_2
    distance = (
        float(difference @ difference)
        + float(np.trace(covariance_1))
        + float(np.trace(covariance_2))
        - 2.0
        * float(np.sqrt(np.clip(middle_eigenvalues, 0.0, None)).sum())
    )
    if distance < 0.0 and abs(distance) <= 1e-7 * scale:
        distance = 0.0
    if not math.isfinite(distance) or distance < 0.0:
        raise MetricAdapterContractError(
            f"invalid Fréchet distance: {distance}"
        )
    return distance


def variation_from_joints(generated_joints: Any) -> float:
    joints = _finite_array(generated_joints, name="generated joints")
    if joints.ndim != 4 or joints.shape[0] < 2 or joints.shape[-1] != 3:
        raise MetricAdapterContractError(
            "generated joints must be [B,T,J,3] with B >= 2"
        )
    coordinate_variance = np.var(joints, axis=0, ddof=1)
    value = np.linalg.norm(
        coordinate_variance,
        axis=-1,
    ).sum(axis=-1).mean()
    if not math.isfinite(float(value)):
        raise MetricAdapterContractError("Variation is non-finite")
    return float(value)


def bc_components_for_sequence(
    joints: Any,
    audio_beat_times: Any,
    *,
    threshold: float = 0.01,
    sigma: float = 0.1,
) -> tuple[float, int]:
    sequence = _finite_array(joints, name="BC joints").copy()
    beats = _finite_array(
        audio_beat_times,
        name="audio beat times",
    ).reshape(-1)
    if (
        sequence.ndim != 3
        or sequence.shape[1] < 22
        or sequence.shape[2] != 3
    ):
        raise MetricAdapterContractError(
            "BC joints must be [T,J,3] with J >= 22"
        )
    if sequence.shape[0] < 3 or beats.size == 0:
        return 0.0, 0
    sequence[:, 15:21] = sequence[:, 16:22]
    vectors = sequence[:, 15:21] - sequence[:, 13:19]
    inner = np.einsum(
        "kij,kij->ki",
        vectors[:, 2:],
        vectors[:, :-2],
    )
    angles = np.arccos(np.clip(inner, -1.0, 1.0)) / math.pi
    angular_velocity = (
        np.abs(angles[1:] - angles[:-1])
        / BC_CHANGE_ANGLE.reshape(1, -1)
        / len(BC_CHANGE_ANGLE)
    )
    differences = np.concatenate(
        [np.zeros((1, 4), dtype=np.float64), angular_velocity],
        axis=0,
    )
    numerator = 0.0
    denominator = 0
    for channel in range(differences.shape[1]):
        motion_beats = []
        for frame in range(1, sequence.shape[0] - 1):
            current = differences[frame, channel]
            previous = differences[frame - 1, channel]
            following = differences[frame + 1, channel]
            if (
                current < previous
                and current < following
                and (
                    previous - current >= threshold
                    or following - current >= threshold
                )
            ):
                motion_beats.append(frame / POSE_FPS)
        if not motion_beats:
            continue
        motion = np.asarray(motion_beats, dtype=np.float64)
        squared_distance = np.min(
            np.square(beats.reshape(-1, 1) - motion.reshape(1, -1)),
            axis=1,
        )
        numerator += float(
            np.exp(-squared_distance / (2.0 * sigma * sigma)).sum()
        )
        denominator += int(beats.size)
    return numerator, denominator


def bc_components_for_batch(
    generated_joints: Any,
    audio_beat_times: Any,
    sample_policy: str,
) -> tuple[float, int]:
    """Return released BC sufficient statistics for an actual metric batch."""

    joints = _finite_array(generated_joints, name="generated joints")
    if joints.ndim != 4:
        raise MetricAdapterContractError(
            "generated joints must be [B,T,J,3]"
        )
    if sample_policy == "first":
        indices: Iterable[int] = (0,)
    elif sample_policy == "all":
        indices = range(joints.shape[0])
    else:
        raise MetricAdapterContractError(
            f"unknown BC sample policy: {sample_policy}"
        )
    numerator = 0.0
    denominator = 0
    for index in indices:
        partial_numerator, partial_denominator = (
            bc_components_for_sequence(
                joints[index],
                audio_beat_times,
            )
        )
        numerator += partial_numerator
        denominator += partial_denominator
    return numerator, denominator


def released_face_metrics(
    ground_truth: Any,
    prediction: Any,
) -> dict[str, float]:
    gt = _finite_array(ground_truth, name="face ground truth")
    predicted = _finite_array(prediction, name="face prediction")
    if (
        gt.shape != predicted.shape
        or gt.ndim != 3
        or gt.shape[-1] != 3
    ):
        raise MetricAdapterContractError(
            "face joints must have the same [T,J,3] shape"
        )
    if gt.shape[0] < 2 or gt.shape[1] < 75:
        raise MetricAdapterContractError(
            "face metric input is too short or lacks landmarks"
        )
    jaw_gt = gt[:, 22:25]
    jaw_prediction = predicted[:, 22:25]
    landmark_gt = gt[:, 74:]
    landmark_prediction = predicted[:, 74:]
    jaw_l1 = float(
        np.linalg.norm(jaw_gt - jaw_prediction, axis=-1).sum(-1).mean()
    )
    landmark_l1 = float(
        np.linalg.norm(
            landmark_gt - landmark_prediction,
            axis=-1,
        ).sum(-1).mean()
    )
    face_gt = np.concatenate((jaw_gt, landmark_gt), axis=1)
    face_prediction = np.concatenate(
        (jaw_prediction, landmark_prediction),
        axis=1,
    )
    gt_velocity = np.linalg.norm(
        face_gt[1:] - face_gt[:-1],
        axis=-1,
    )
    prediction_velocity = np.linalg.norm(
        face_prediction[1:] - face_prediction[:-1],
        axis=-1,
    )
    lvd = float(
        np.abs(prediction_velocity - gt_velocity).sum(-1).mean()
    )
    result = {
        "jaw_l1": jaw_l1,
        "landmark_l1": landmark_l1,
        "LVD": lvd,
        "face_l2_combined": jaw_l1 + landmark_l1,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise MetricAdapterContractError("face metrics are non-finite")
    return result


def reorder_to_talkshow(
    poses_165: Any,
    expressions_100: Any,
) -> np.ndarray:
    poses = np.asarray(poses_165, dtype=np.float32)
    expressions = np.asarray(expressions_100, dtype=np.float32)
    if poses.shape[:-1] != expressions.shape[:-1]:
        raise MetricAdapterContractError(
            "pose/expression leading shapes differ"
        )
    if poses.shape[-1] != 165 or expressions.shape[-1] != 100:
        raise MetricAdapterContractError(
            "expected pose width 165 and expression width 100"
        )
    output = np.empty((*poses.shape[:-1], 265), dtype=np.float32)
    output[..., 0:3] = poses[..., 66:69]
    output[..., 3:6] = poses[..., 69:72]
    output[..., 6:9] = poses[..., 72:75]
    output[..., 9:12] = poses[..., 0:3]
    output[..., 12:75] = poses[..., 3:66]
    output[..., 75:120] = poses[..., 75:120]
    output[..., 120:165] = poses[..., 120:165]
    output[..., 165:265] = expressions
    return output


def released_body_parameters(parameters_265: Any) -> np.ndarray:
    parameters = np.asarray(parameters_265)
    if parameters.ndim < 2 or parameters.shape[-1] != 265:
        raise MetricAdapterContractError(
            "TalkSHOW body parameters must end in 265 channels"
        )
    if not np.issubdtype(parameters.dtype, np.floating):
        raise MetricAdapterContractError(
            "TalkSHOW body parameters must be floating point"
        )
    if not np.isfinite(parameters).all():
        raise MetricAdapterContractError(
            "TalkSHOW body parameters contain NaN or infinity"
        )
    lower = TALKSHOW_RELEASED_LOWER_POSE.astype(
        parameters.dtype,
        copy=False,
    )
    prefix_shape = parameters.shape[:-1]
    fixed = np.concatenate(
        (
            parameters[..., 0:3],
            np.broadcast_to(lower[0:15], (*prefix_shape, 15)),
            parameters[..., 18:21],
            np.broadcast_to(lower[15:21], (*prefix_shape, 6)),
            parameters[..., 27:30],
            np.broadcast_to(lower[21:27], (*prefix_shape, 6)),
            parameters[..., 36:39],
            np.broadcast_to(lower[27:33], (*prefix_shape, 6)),
            parameters[..., 45:265],
        ),
        axis=-1,
    ).copy()
    if fixed.shape != parameters.shape:
        raise MetricAdapterContractError(
            "TalkSHOW fixed-lower transform changed shape"
        )
    fixed[..., 0:3] = 0.0
    fixed[..., 165:265] = 0.0
    return fixed


def released_face_parameters(
    ground_truth_265: Any,
    prediction_265: Any,
) -> tuple[np.ndarray, np.ndarray]:
    ground_truth = np.asarray(ground_truth_265)
    prediction = np.asarray(prediction_265)
    if (
        ground_truth.shape != prediction.shape
        or ground_truth.ndim < 2
        or ground_truth.shape[-1] != 265
    ):
        raise MetricAdapterContractError(
            "face parameters must have matching [...,T,265] shapes"
        )
    if (
        not np.issubdtype(ground_truth.dtype, np.floating)
        or not np.issubdtype(prediction.dtype, np.floating)
        or not np.isfinite(ground_truth).all()
        or not np.isfinite(prediction).all()
    ):
        raise MetricAdapterContractError(
            "face parameters must be finite floating point"
        )
    isolated_ground_truth = ground_truth.copy()
    isolated_prediction = prediction.copy()
    isolated_ground_truth[..., 3:165] = 0.0
    isolated_prediction[..., 3:165] = 0.0
    return isolated_ground_truth, isolated_prediction


class MetricBackend(Protocol):
    """Pinned metric backend interface used by the numpy adapter."""

    @property
    def asset_receipt(self) -> Mapping[str, Any]:
        ...

    @property
    def runtime_receipt(self) -> Mapping[str, Any]:
        ...

    def extract_body_features(self, parameters_265: np.ndarray) -> np.ndarray:
        ...

    def joints(
        self,
        parameters_265: np.ndarray,
        betas_300: np.ndarray,
    ) -> np.ndarray:
        ...


def _validate_backend_receipts(
    backend: MetricBackend,
    *,
    require_cuda: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = dict(backend.asset_receipt)
    runtime = dict(backend.runtime_receipt)
    expected_asset_keys = {
        "format",
        "status",
        "execution_device",
        "talkshow",
        "feature_extractor",
        "smplx",
    }
    if (
        set(assets) != expected_asset_keys
        or assets.get("format")
        != "semtalk_show_talkshow_metric_assets_v1"
        or assets.get("status") != "pass"
        or not isinstance(assets.get("talkshow"), dict)
        or assets["talkshow"].get("commit") != TALKSHOW_METRIC_COMMIT
        or not isinstance(assets.get("feature_extractor"), dict)
        or assets["feature_extractor"].get("sha256")
        != FEATURE_EXTRACTOR_SHA256
        or not isinstance(assets.get("smplx"), dict)
        or assets["smplx"].get("sha256") != SMPLX_SHA256
    ):
        raise MetricAdapterContractError(
            "metric backend does not attest pinned TalkSHOW assets"
        )
    common_runtime = {"python", "numpy", "torch", "smplx", "librosa"}
    if require_cuda:
        expected_runtime_keys = common_runtime | {
            "scipy",
            "cuda",
            "cudnn",
            "device",
            "device_type",
            "device_index",
            "device_name",
        }
        if (
            set(runtime) != expected_runtime_keys
            or runtime.get("device_type") != "cuda"
            or type(runtime.get("device_index")) is not int
            or runtime["device_index"] < 0
            or runtime.get("device")
            != f"cuda:{runtime['device_index']}"
            or assets.get("execution_device") != runtime.get("device")
            or type(runtime.get("device_name")) is not str
            or not runtime["device_name"]
            or type(runtime.get("cuda")) is not str
            or not runtime["cuda"]
            or type(runtime.get("cudnn")) is not str
            or not runtime["cudnn"]
            or assets["feature_extractor"].get("runtime_dtype")
            != "float32"
            or assets["smplx"].get("runtime_dtype") != "float64"
            or any(
                type(runtime.get(key)) is not str or not runtime[key]
                for key in common_runtime | {"scipy"}
            )
        ):
            raise MetricAdapterContractError(
                "formal metric backend lacks exact CUDA runtime attestation"
            )
    else:
        expected_runtime_keys = common_runtime | {"device"}
        if (
            set(runtime) != expected_runtime_keys
            or runtime.get("device") != "cpu"
            or assets.get("execution_device") != "cpu"
            or any(
                type(runtime.get(key)) is not str or not runtime[key]
                for key in common_runtime
            )
        ):
            raise MetricAdapterContractError(
                "invalid CPU unit-test metric runtime receipt"
            )
    return assets, runtime


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MetricAdapterContractError(
            f"cannot inspect TalkSHOW metric root {root}: {exc}"
        ) from exc


def _local_module_sources(
    root: Path,
    module: str,
) -> list[tuple[str, Path]]:
    if not module:
        return []
    pieces = module.split(".")
    package = root.joinpath(*pieces, "__init__.py")
    source = root.joinpath(*pieces).with_suffix(".py")
    result: list[tuple[str, Path]] = []
    if package.is_file():
        result.append((module, package))
    if source.is_file():
        result.append((module, source))
    return result


def _relative_import_module(
    current_module: str,
    current_path: Path,
    *,
    level: int,
    module: str | None,
) -> str:
    package = (
        current_module
        if current_path.name == "__init__.py"
        else current_module.rpartition(".")[0]
    )
    pieces = package.split(".") if package else []
    if level > len(pieces) + 1:
        return ""
    prefix = pieces[: len(pieces) - level + 1]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


def _talkshow_fgd_import_closure(root: Path) -> tuple[str, ...]:
    root = root.resolve()
    queue: list[tuple[str, Path]] = []
    queued: set[Path] = set()
    for entry in TALKSHOW_FGD_ENTRY_MODULES:
        for module, source in _local_module_sources(root, entry):
            resolved = source.resolve()
            if resolved not in queued:
                queued.add(resolved)
                queue.append((module, resolved))
    visited: set[Path] = set()
    cursor = 0
    while cursor < len(queue):
        current_module, current_path = queue[cursor]
        cursor += 1
        if current_path in visited:
            continue
        visited.add(current_path)
        try:
            tree = ast.parse(
                current_path.read_text(encoding="utf-8"),
                filename=str(current_path),
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetricAdapterContractError(
                f"cannot inspect TalkSHOW metric source {current_path}"
            ) from exc
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    value = _relative_import_module(
                        current_module,
                        current_path,
                        level=node.level,
                        module=node.module,
                    )
                else:
                    value = node.module or ""
                if value:
                    imported.append(value)
                if node.module is None and value:
                    imported.extend(
                        f"{value}.{alias.name}"
                        for alias in node.names
                        if alias.name != "*"
                    )
        for imported_module in imported:
            for module, source in _local_module_sources(
                root,
                imported_module,
            ):
                resolved = source.resolve()
                if resolved not in queued:
                    queued.add(resolved)
                    queue.append((module, resolved))
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in visited)
    )


def validate_talkshow_metric_root(root: str | Path) -> dict[str, Any]:
    candidate = Path(root).expanduser()
    if candidate.is_symlink():
        raise MetricAdapterContractError(
            "TalkSHOW metric root must not be a symlink"
        )
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise MetricAdapterContractError(
            f"TalkSHOW metric root is not a directory: {resolved}"
        )
    commit = _git_output(resolved, "rev-parse", "HEAD^{commit}")
    if commit != TALKSHOW_METRIC_COMMIT:
        raise MetricAdapterContractError(
            f"TalkSHOW metric commit {commit} != {TALKSHOW_METRIC_COMMIT}"
        )
    marker_path = _resolved_regular_file(
        resolved / ".paspa_talkshow_patch.json",
        "TalkSHOW metric patch marker",
    )
    marker_payload = marker_path.read_bytes()
    try:
        marker = json.loads(marker_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetricAdapterContractError(
            "TalkSHOW patch marker is invalid JSON"
        ) from exc
    if marker not in TALKSHOW_PATCH_MARKERS:
        raise MetricAdapterContractError(
            f"unexpected TalkSHOW patch marker: {marker}"
        )
    closure = _talkshow_fgd_import_closure(resolved)
    if closure != TALKSHOW_FGD_SOURCE_FILES:
        raise MetricAdapterContractError(
            "TalkSHOW FGD import closure differs from the pinned contract"
        )
    files: dict[str, dict[str, Any]] = {}
    for relative in closure:
        path = _resolved_regular_file(
            resolved / relative,
            f"TalkSHOW metric source {relative}",
        )
        payload = path.read_bytes()
        files[relative] = {
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        }
    metric_only = TALKSHOW_PATCH_MARKERS[1]
    if (
        marker == metric_only
        and files["nets/__init__.py"]["sha256"]
        != metric_only["patched_sha256"]
    ):
        raise MetricAdapterContractError(
            "TalkSHOW metric-only import-isolation SHA mismatch"
        )
    return {
        "path": str(resolved),
        "commit": commit,
        "entry_modules": list(TALKSHOW_FGD_ENTRY_MODULES),
        "import_closure_algorithm": TALKSHOW_FGD_IMPORT_ALGORITHM,
        "patch_marker": marker,
        "marker_sha256": sha256_bytes(marker_payload),
        "files": files,
    }


def _package_version(module: Any, distribution: str) -> str:
    from importlib import metadata

    value = getattr(module, "__version__", None)
    if type(value) is not str or not value.strip():
        value = metadata.version(distribution)
    if type(value) is not str or not value.strip():
        raise MetricAdapterContractError(
            f"cannot attest runtime version for {distribution}"
        )
    return value


class TalkShowCudaMetricBackend:
    """Pinned TalkSHOW feature/SMPL-X backend on one explicit CUDA device."""

    def __init__(
        self,
        *,
        talkshow_root: str | Path,
        feature_extractor: str | Path,
        smplx_asset: str | Path,
        device: str,
        torch_threads: int = 1,
    ) -> None:
        torch_threads = _require_exact_int(
            torch_threads,
            "torch_threads",
            minimum=1,
        )
        talkshow = validate_talkshow_metric_root(talkshow_root)
        feature_path = _resolved_regular_file(
            feature_extractor,
            "TalkSHOW released feature extractor",
        )
        feature_payload = feature_path.read_bytes()
        feature_sha = sha256_bytes(feature_payload)
        if feature_sha != FEATURE_EXTRACTOR_SHA256:
            raise MetricAdapterContractError(
                "TalkSHOW feature extractor SHA differs from the pinned release"
            )
        smplx_path = _resolved_regular_file(
            smplx_asset,
            "SMPL-X neutral asset",
        )
        smplx_payload = smplx_path.read_bytes()
        smplx_sha = sha256_bytes(smplx_payload)
        if smplx_sha != SMPLX_SHA256:
            raise MetricAdapterContractError(
                "SMPL-X asset SHA differs from the pinned release"
            )
        try:
            import torch
            import smplx
            import librosa
            import scipy
        except ImportError as exc:
            raise MetricAdapterContractError(
                "torch, smplx, librosa, and scipy are required by the "
                "formal CUDA metric backend"
            ) from exc
        if (
            type(device) is not str
            or not device.startswith("cuda:")
            or not device[5:].isdigit()
        ):
            raise MetricAdapterContractError(
                "formal metric device must be one explicit cuda:N ordinal"
            )
        torch_device = torch.device(device)
        device_index = int(device[5:])
        if (
            torch_device.type != "cuda"
            or torch_device.index != device_index
            or not torch.cuda.is_available()
            or device_index >= torch.cuda.device_count()
        ):
            raise MetricAdapterContractError(
                f"formal CUDA metric device is unavailable: {device}"
            )
        torch.set_num_threads(torch_threads)
        original_cwd = Path.cwd()
        original_sys_path = list(sys.path)
        try:
            os.chdir(Path(talkshow["path"]))
            sys.path.insert(0, str(talkshow["path"]))
            from nets.body_ae import TrainWrapper as BodyFeatureExtractor
        finally:
            os.chdir(original_cwd)
            sys.path[:] = original_sys_path
        config = SimpleNamespace(
            Data=SimpleNamespace(
                pose=SimpleNamespace(
                    convert_to_6d=False,
                    pre_pose_length=0,
                    expression=False,
                )
            ),
            Train=SimpleNamespace(
                learning_rate=SimpleNamespace(
                    generator_learning_rate=1e-4,
                    discriminator_learning_rate=1e-4,
                )
            ),
        )
        extractor = BodyFeatureExtractor(
            SimpleNamespace(gpu=str(torch_device)),
            config,
        )
        try:
            checkpoint = torch.load(
                io.BytesIO(feature_payload),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise MetricAdapterContractError(
                "cannot deserialize pinned TalkSHOW feature extractor"
            ) from exc
        if isinstance(checkpoint, Mapping):
            checkpoint = checkpoint.get("generator", checkpoint)
        if not isinstance(checkpoint, Mapping):
            raise MetricAdapterContractError(
                "TalkSHOW feature-extractor checkpoint is not a mapping"
            )
        state = checkpoint.get("g", checkpoint)
        if not isinstance(state, Mapping):
            raise MetricAdapterContractError(
                "TalkSHOW feature-extractor checkpoint lacks g state"
            )
        normalized = {
            (name[7:] if name.startswith("module.") else name): value
            for name, value in state.items()
        }
        extractor.g.load_state_dict(normalized, strict=True)
        extractor.g.to(torch_device).eval()
        extractor.g.requires_grad_(False)
        for name, tensor in extractor.g.state_dict().items():
            if not bool(torch.isfinite(tensor).all().item()):
                raise MetricAdapterContractError(
                    f"non-finite TalkSHOW feature tensor: {name}"
                )
        smplx_model = smplx.create(
            str(smplx_path),
            model_type="smplx",
            gender="neutral",
            ext="npz",
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            num_betas=300,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            use_pca=False,
            flat_hand_mean=False,
            create_expression=True,
            num_expression_coeffs=100,
            num_pca_comps=12,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=False,
            dtype=torch.float64,
        ).to(torch_device).eval()
        smplx_model.requires_grad_(False)
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        if (
            type(cuda_version) is not str
            or not cuda_version.strip()
            or type(cudnn_version) is not int
            or cudnn_version < 1
        ):
            raise MetricAdapterContractError(
                "formal CUDA metric backend cannot attest CUDA/cuDNN"
            )
        self._torch = torch
        self._device = torch_device
        self._extractor = extractor
        self._smplx = smplx_model
        self._librosa = librosa
        self._asset_receipt = {
            "format": "semtalk_show_talkshow_metric_assets_v1",
            "status": "pass",
            "execution_device": str(torch_device),
            "talkshow": talkshow,
            "feature_extractor": {
                "path": str(feature_path),
                "sha256": feature_sha,
                "bytes": len(feature_payload),
                "runtime_dtype": "float32",
            },
            "smplx": {
                "path": str(smplx_path),
                "sha256": smplx_sha,
                "bytes": len(smplx_payload),
                "runtime_dtype": "float64",
            },
        }
        self._runtime_receipt = {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "smplx": _package_version(smplx, "smplx"),
            "librosa": _package_version(librosa, "librosa"),
            "scipy": _package_version(scipy, "scipy"),
            "cuda": cuda_version,
            "cudnn": str(cudnn_version),
            "device": str(torch_device),
            "device_type": "cuda",
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(torch_device),
        }

    @property
    def asset_receipt(self) -> Mapping[str, Any]:
        return self._asset_receipt

    @property
    def runtime_receipt(self) -> Mapping[str, Any]:
        return self._runtime_receipt

    def extract_body_features(self, parameters_265: np.ndarray) -> np.ndarray:
        parameters = np.asarray(parameters_265, dtype=np.float32)
        if parameters.ndim != 3 or parameters.shape[-1] != 265:
            raise MetricAdapterContractError(
                "feature parameters must be [B,T,265]"
            )
        with self._torch.no_grad():
            features, _ = self._extractor.extract(
                self._torch.from_numpy(parameters).to(self._device)
            )
        array = features.detach().cpu().numpy()
        array = array.reshape(-1, array.shape[-1])
        if array.size == 0 or not np.isfinite(array).all():
            raise MetricAdapterContractError(
                "TalkSHOW feature extractor returned invalid features"
            )
        return array

    def joints(
        self,
        parameters_265: np.ndarray,
        betas_300: np.ndarray,
    ) -> np.ndarray:
        parameters = np.asarray(parameters_265)
        betas = np.asarray(betas_300)
        if parameters.ndim != 3 or parameters.shape[-1] != 265:
            raise MetricAdapterContractError(
                "SMPL-X parameters must be [B,T,265]"
            )
        if betas.shape != (BETA_DIM,):
            raise MetricAdapterContractError(
                "SMPL-X betas must have 300 coefficients"
            )
        flat = self._torch.from_numpy(parameters.reshape(-1, 265)).to(
            device=self._device,
            dtype=self._torch.float64,
        )
        beta = self._torch.from_numpy(betas.reshape(1, -1)).to(
            device=self._device,
            dtype=self._torch.float64,
        )
        chunks = []
        with self._torch.no_grad():
            for start in range(0, flat.shape[0], 2048):
                current = flat[start : start + 2048]
                output = self._smplx(
                    betas=beta.expand(current.shape[0], -1),
                    expression=current[:, 165:265],
                    jaw_pose=current[:, 0:3],
                    leye_pose=current[:, 3:6],
                    reye_pose=current[:, 6:9],
                    global_orient=current[:, 9:12],
                    body_pose=current[:, 12:75],
                    left_hand_pose=current[:, 75:120],
                    right_hand_pose=current[:, 120:165],
                    return_verts=False,
                )
                chunks.append(output.joints.detach().cpu())
        array = self._torch.cat(chunks).reshape(
            parameters.shape[0],
            parameters.shape[1],
            -1,
            3,
        ).numpy()
        if not np.isfinite(array).all():
            raise MetricAdapterContractError(
                "SMPL-X backend returned non-finite joints"
            )
        return array

    def audio_beats(self, waveform_16k: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        if not np.isfinite(waveform).all() or waveform.size == 0:
            raise MetricAdapterContractError(
                "audio waveform is empty or non-finite"
            )
        return self._librosa.onset.onset_detect(
            y=waveform,
            sr=16000,
            units="time",
        ).reshape(-1)


def _strict_npz_members(path: Path, fields: tuple[str, ...]) -> None:
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = tuple(info.filename for info in archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise MetricAdapterContractError(
            f"cannot inspect NPZ {path}"
        ) from exc
    expected = tuple(f"{field}.npy" for field in fields)
    if members != expected or len(members) != len(set(members)):
        raise MetricAdapterContractError(
            f"{path}: NPZ members {members} != {expected}"
        )


def _load_canonical_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    frames: int,
    speaker_id: int,
) -> dict[str, np.ndarray]:
    resolved, _payload = _verified_file_snapshot(
        path,
        expected_sha256,
        "canonical SHOW NPZ",
    )
    _strict_npz_members(resolved, CANONICAL_FIELDS)
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if tuple(archive.files) != CANONICAL_FIELDS:
                raise MetricAdapterContractError(
                    f"{resolved}: canonical NPZ field order mismatch"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in CANONICAL_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise MetricAdapterContractError(
            f"cannot load canonical SHOW NPZ {resolved}"
        ) from exc
    expected = {
        "pose": ((frames, POSE_DIM), np.dtype(np.float32)),
        "contact": ((frames, 4), np.dtype(np.float32)),
        "facial": ((frames, EXPRESSION_DIM), np.dtype(np.float32)),
        "beta": ((frames, BETA_DIM), np.dtype(np.float32)),
        "trans": ((frames, 3), np.dtype(np.float32)),
        "speaker_id": ((frames, 1), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise MetricAdapterContractError(
                f"{resolved}: {name} {array.shape}/{array.dtype} "
                f"!= {shape}/{dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise MetricAdapterContractError(
                f"{resolved}: non-finite canonical {name}"
            )
    if not np.array_equal(
        arrays["speaker_id"],
        np.full((frames, 1), speaker_id, dtype=np.int64),
    ):
        raise MetricAdapterContractError(
            f"{resolved}: canonical speaker_id mismatch"
        )
    if not np.array_equal(
        arrays["beta"],
        np.broadcast_to(arrays["beta"][0], arrays["beta"].shape),
    ):
        raise MetricAdapterContractError(
            f"{resolved}: canonical betas change across frames"
        )
    return arrays


def _load_output_npz(
    artifact: Mapping[str, Any],
    *,
    frames: int,
    prediction: bool,
    expected_name: str,
) -> dict[str, np.ndarray]:
    if type(artifact) is not dict or set(artifact) != ARTIFACT_KEYS:
        raise MetricAdapterContractError(
            f"invalid {'prediction' if prediction else 'ground-truth'} artifact"
        )
    path, payload = _verified_file_snapshot(
        artifact["path"],
        _require_sha256(
            artifact["sha256"],
            "output NPZ SHA",
        ),
        "prediction NPZ" if prediction else "ground-truth NPZ",
    )
    if path.name != expected_name:
        raise MetricAdapterContractError(
            f"output NPZ basename {path.name!r} != {expected_name!r}"
        )
    if _require_exact_int(
        artifact["bytes"],
        "output NPZ bytes",
        minimum=1,
    ) != len(payload):
        raise MetricAdapterContractError(
            f"{path}: output NPZ byte count mismatch"
        )
    _strict_npz_members(path, OUTPUT_FIELDS)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != OUTPUT_FIELDS:
                raise MetricAdapterContractError(
                    f"{path}: output NPZ field order mismatch"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in OUTPUT_FIELDS
            }
    except (OSError, ValueError) as exc:
        raise MetricAdapterContractError(
            f"cannot load output NPZ {path}"
        ) from exc
    expected = {
        "betas": ((BETA_DIM,), np.dtype(np.float32)),
        "poses": ((frames, POSE_DIM), np.dtype(np.float32)),
        "expressions": (
            (frames, EXPRESSION_DIM),
            np.dtype(np.float32),
        ),
        "trans": ((frames, 3), np.dtype(np.float32)),
        "model": ((), np.asarray("smplx2020").dtype),
        "gender": ((), np.asarray("neutral").dtype),
        "mocap_frame_rate": ((), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise MetricAdapterContractError(
                f"{path}: {name} {array.shape}/{array.dtype} "
                f"!= {shape}/{dtype}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise MetricAdapterContractError(
                f"{path}: non-finite {name}"
            )
    if (
        arrays["model"].item() != "smplx2020"
        or arrays["gender"].item() != "neutral"
        or int(arrays["mocap_frame_rate"].item()) != POSE_FPS
    ):
        raise MetricAdapterContractError(
            f"{path}: invalid model/gender/frame-rate metadata"
        )
    if prediction and not np.array_equal(
        arrays["poses"][:, 69:75],
        np.zeros((frames, 6), dtype=np.float32),
    ):
        raise MetricAdapterContractError(
            f"{path}: prediction eye pose is not zero"
        )
    return arrays


def _validate_wav_row_and_decode(
    row: Mapping[str, Any],
    *,
    motion_frames: int,
) -> np.ndarray:
    path, payload = _verified_file_snapshot(
        row["source_wav"],
        _require_sha256(
            row["source_wav_sha256"],
            "canonical source WAV SHA",
        ),
        "canonical source WAV",
    )
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            rate = int(handle.getframerate())
            wav_frames = int(handle.getnframes())
            compression = handle.getcomptype()
            raw = handle.readframes(wav_frames)
    except (EOFError, wave.Error) as exc:
        raise MetricAdapterContractError(
            f"cannot decode canonical WAV {path}"
        ) from exc
    if (
        channels not in (1, 2)
        or width != 2
        or rate != 16000
        or wav_frames < 1
        or compression != "NONE"
        or row.get("wav_channels") != channels
        or row.get("wav_sample_width") != width
        or row.get("wav_sample_rate") != rate
        or row.get("wav_frames") != wav_frames
    ):
        raise MetricAdapterContractError(
            f"{path}: canonical WAV metadata mismatch"
        )
    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size != wav_frames * channels:
        raise MetricAdapterContractError(
            f"{path}: decoded WAV sample count mismatch"
        )
    waveform = samples.reshape(wav_frames, channels).astype(np.float32)
    waveform = waveform.mean(axis=1) / np.float32(32768.0)
    expected_audio_frames = 16000 * motion_frames // POSE_FPS
    if waveform.shape[0] < expected_audio_frames:
        raise MetricAdapterContractError(
            f"{path}: canonical WAV is shorter than the motion-aligned "
            f"prefix {waveform.shape[0]} < {expected_audio_frames}"
        )
    waveform = waveform[:expected_audio_frames]
    if not np.isfinite(waveform).all():
        raise MetricAdapterContractError(
            f"{path}: decoded WAV is non-finite"
        )
    return waveform


def _validate_canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_clip_count: int,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    global_indices: set[int] = set()
    for raw in rows:
        if type(raw) is not dict or set(raw) != CANONICAL_ROW_KEYS:
            raise MetricAdapterContractError(
                "canonical manifest row schema mismatch"
            )
        row = dict(raw)
        if row.get("split") != split:
            continue
        clip_id = row.get("clip_id")
        if type(clip_id) is not str:
            raise MetricAdapterContractError(
                "canonical clip_id must be a string"
            )
        output_id = canonical_clip_id(clip_id)
        if output_id in selected:
            raise MetricAdapterContractError(
                f"duplicate canonical clip {output_id}"
            )
        index = _require_exact_int(
            row.get("global_index"),
            f"{clip_id} global_index",
            minimum=0,
        )
        if index in global_indices:
            raise MetricAdapterContractError(
                f"duplicate canonical global_index {index}"
            )
        speaker = row.get("speaker")
        if (
            speaker not in SHOW_SPEAKER_IDS
            or row.get("speaker_id") != SHOW_SPEAKER_IDS[speaker]
            or clip_id.split("/", 1)[0] != speaker
            or row.get("pose_fps") != POSE_FPS
        ):
            raise MetricAdapterContractError(
                f"{clip_id}: canonical speaker/frame-rate mismatch"
            )
        _require_exact_int(
            row.get("frames"),
            f"{clip_id} frames",
            minimum=2,
        )
        _require_sha256(
            row.get("canonical_npz_sha256"),
            f"{clip_id} canonical NPZ SHA",
        )
        _require_sha256(
            row.get("source_wav_sha256"),
            f"{clip_id} source WAV SHA",
        )
        selected[output_id] = row
        global_indices.add(index)
    if len(selected) != expected_clip_count:
        raise MetricAdapterContractError(
            f"canonical {split} clips {len(selected)} != {expected_clip_count}"
        )
    if set(row["speaker"] for row in selected.values()) != set(SPEAKER_NAMES):
        raise MetricAdapterContractError(
            "canonical metric split must contain all four SHOW speakers"
        )
    return selected


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    canonical: Mapping[str, Mapping[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    expected_keys = (
        VAL_PREDICTION_ROW_KEYS
        if split == "val"
        else TEST_PREDICTION_ROW_KEYS
    )
    by_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if type(raw) is not dict or set(raw) != expected_keys:
            raise MetricAdapterContractError(
                f"{split} prediction manifest row schema mismatch"
            )
        row = dict(raw)
        output_id = row.get("canonical_clip_id")
        if type(output_id) is not str or output_id in by_id:
            raise MetricAdapterContractError(
                f"invalid/duplicate prediction clip {output_id!r}"
            )
        canonical_row = canonical.get(output_id)
        if canonical_row is None:
            raise MetricAdapterContractError(
                f"prediction clip {output_id} is not canonical {split}"
            )
        clip_id = canonical_row["clip_id"]
        if (
            row.get("source_clip_id") != clip_id
            or row.get("global_index") != canonical_row["global_index"]
            or row.get("frames") != canonical_row["frames"]
            or output_id != canonical_clip_id(clip_id)
        ):
            raise MetricAdapterContractError(
                f"{output_id}: prediction/canonical identity mismatch"
            )
        if split == "val":
            if (
                row.get("split") != "val"
                or type(row.get("epoch")) is not int
            ):
                raise MetricAdapterContractError(
                    f"{output_id}: invalid validation candidate binding"
                )
            _require_sha256(
                row.get("candidate_checkpoint_sha256"),
                f"{output_id} candidate checkpoint SHA",
            )
        else:
            if (
                row.get("speaker") != canonical_row["speaker"]
                or row.get("speaker_id") != canonical_row["speaker_id"]
                or Path(str(row.get("canonical_npz", ""))).resolve()
                != Path(canonical_row["canonical_npz"]).resolve()
                or row.get("canonical_npz_sha256")
                != canonical_row["canonical_npz_sha256"]
            ):
                raise MetricAdapterContractError(
                    f"{output_id}: test row canonical binding mismatch"
                )
            _require_exact_int(
                row.get("evaluation_index"),
                f"{output_id} evaluation_index",
                minimum=0,
            )
        for role in ("prediction", "ground_truth"):
            artifact = row.get(role)
            if type(artifact) is not dict or set(artifact) != ARTIFACT_KEYS:
                raise MetricAdapterContractError(
                    f"{output_id}: invalid {role} artifact receipt"
                )
            _require_sha256(
                artifact.get("sha256"),
                f"{output_id} {role} SHA",
            )
            _require_exact_int(
                artifact.get("bytes"),
                f"{output_id} {role} bytes",
                minimum=1,
            )
        by_id[output_id] = row
    if set(by_id) != set(canonical):
        missing = sorted(set(canonical) - set(by_id))
        extra = sorted(set(by_id) - set(canonical))
        raise MetricAdapterContractError(
            "prediction manifest does not cover canonical split exactly once: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    if split == "test":
        ordered = sorted(
            by_id.values(),
            key=lambda row: row["evaluation_index"],
        )
        indices = [row["evaluation_index"] for row in ordered]
        if sorted(indices) != list(range(len(ordered))):
            raise MetricAdapterContractError(
                "test evaluation_index does not cover an exact range"
            )
    else:
        ordered = sorted(
            by_id.values(),
            key=lambda row: row["global_index"],
        )
        epochs = {row["epoch"] for row in ordered}
        checkpoints = {
            row["candidate_checkpoint_sha256"] for row in ordered
        }
        if len(epochs) != 1 or len(checkpoints) != 1:
            raise MetricAdapterContractError(
                "validation predictions mix candidates"
            )
    return ordered


def _validate_prediction_lineage(
    lineage: Mapping[str, Any],
    *,
    split: str,
    prediction_manifest_path: Path,
    prediction_manifest_sha256: str,
    expected_clip_count: int,
) -> str | None:
    def reject_randomness_conflicts(value: Any, context: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                current = f"{context}.{key}"
                if (
                    key in {"independent_samples", "stochastic"}
                    and nested is not False
                ):
                    raise MetricAdapterContractError(
                        f"{current} conflicts with deterministic delta metrics"
                    )
                if key == "deterministic" and nested is not True:
                    raise MetricAdapterContractError(
                        f"{current} conflicts with deterministic delta metrics"
                    )
                if (
                    key in {
                        "physical_samples_per_clip",
                        "prediction_samples_per_clip",
                    }
                    and nested != 1
                ):
                    raise MetricAdapterContractError(
                        f"{current} conflicts with one physical prediction"
                    )
                reject_randomness_conflicts(nested, current)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                reject_randomness_conflicts(
                    nested,
                    f"{context}[{index}]",
                )

    reject_randomness_conflicts(lineage, "prediction_lineage")
    if lineage.get("status") != "complete":
        raise MetricAdapterContractError(
            "prediction lineage is not complete"
        )
    payload_sha: str | None = None
    if split == "val":
        if (
            lineage.get("format")
            != "semtalk_show_base_official_adapt_val_inference_lineage_v1"
            or lineage.get("split") != "val"
            or lineage.get("test_visible") is not False
            or lineage.get("exact_once") is not True
            or lineage.get("finite") is not True
            or lineage.get("clip_count") != expected_clip_count
            or not isinstance(lineage.get("final_manifest"), dict)
            or Path(lineage["final_manifest"].get("path", "")).resolve()
            != prediction_manifest_path
            or lineage["final_manifest"].get("sha256")
            != prediction_manifest_sha256
        ):
            raise MetricAdapterContractError(
                "validation prediction lineage/manifest mismatch"
            )
        payload_sha = _require_sha256(
            lineage.get("receipt_payload_sha256"),
            "validation lineage payload SHA",
        )
        unsigned = dict(lineage)
        del unsigned["receipt_payload_sha256"]
        if canonical_json_sha256(unsigned) != payload_sha:
            raise MetricAdapterContractError(
                "validation lineage payload SHA mismatch"
            )
    else:
        if (
            lineage.get("format")
            != "semtalk_show_base_inference_final_lineage_v1"
            or lineage.get("final_manifest_sha256")
            != prediction_manifest_sha256
            or not isinstance(lineage.get("contract"), dict)
            or lineage["contract"].get("test_clips")
            != expected_clip_count
        ):
            raise MetricAdapterContractError(
                "test prediction lineage/manifest mismatch"
            )
    return payload_sha


def _validate_external_validation_gate(
    receipt: Mapping[str, Any],
    *,
    expected_scope: str,
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    if type(receipt) is not dict or set(receipt) != {
        "path",
        "sha256",
        "bytes",
        "receipt_payload_sha256",
    }:
        raise MetricAdapterContractError(
            "validation_gate artifact schema mismatch"
        )
    module = _replication_gate_module()
    try:
        artifact, gate = module.load_gate(
            Path(str(receipt["path"])),
            _require_sha256(
                receipt["sha256"],
                "validation gate file SHA",
            ),
            expected_scope=expected_scope,
            test_only_allow_four_clip_subset=(
                test_only_allow_four_clip_subset
            ),
        )
    except Exception as exc:
        if isinstance(exc, MetricAdapterContractError):
            raise
        raise MetricAdapterContractError(
            f"validation gate fresh replay failed: {exc}"
        ) from exc
    expected_bytes = _require_exact_int(
        receipt["bytes"],
        "validation gate bytes",
        minimum=1,
    )
    expected_payload_sha = _require_sha256(
        receipt["receipt_payload_sha256"],
        "validation gate receipt payload SHA",
    )
    if (
        artifact != dict(receipt)
        or artifact.get("bytes") != expected_bytes
        or artifact.get("receipt_payload_sha256")
        != expected_payload_sha
        or gate.get("payload_hash_algorithm")
        != PAYLOAD_HASH_ALGORITHM
    ):
        raise MetricAdapterContractError(
            "validation gate artifact/payload contract mismatch"
        )
    return dict(artifact)


def build_distribution_receipt(
    *,
    prediction_manifest_artifact: Mapping[str, Any],
    prediction_artifacts: Sequence[Mapping[str, Any]],
    validation_gate: Mapping[str, Any],
    expected_scope: str = "validation_candidate_family",
    test_only_allow_four_clip_subset: bool = False,
) -> dict[str, Any]:
    """Delegate to the Base gate's sole canonical receipt constructor."""

    gate_artifact = _validate_external_validation_gate(
        validation_gate,
        expected_scope=expected_scope,
        test_only_allow_four_clip_subset=(
            test_only_allow_four_clip_subset
        ),
    )
    module = _replication_gate_module()
    try:
        return module.build_distribution_receipt_from_validated_artifacts(
            gate_artifact=gate_artifact,
            prediction_manifest_artifact=prediction_manifest_artifact,
            prediction_records=prediction_artifacts,
        )
    except Exception as exc:
        raise MetricAdapterContractError(
            f"cannot build canonical Base distribution receipt: {exc}"
        ) from exc


def validate_distribution_declaration(
    declaration: Mapping[str, Any],
    *,
    expected_gate_artifact: Mapping[str, Any],
    expected_prediction_manifest: Mapping[str, Any],
    expected_prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate through deterministic_replication_gate, never a local copy."""

    module = _replication_gate_module()
    try:
        observed = module.validate_distribution_receipt(
            declaration,
            expected_gate_artifact=expected_gate_artifact,
            expected_prediction_manifest=expected_prediction_manifest,
            expected_prediction_records=expected_prediction_records,
        )
    except Exception as exc:
        raise MetricAdapterContractError(
            f"distribution declaration rejected by Base gate: {exc}"
        ) from exc
    if (
        observed.get("format")
        != "semtalk_show_deterministic_distribution_receipt_v1"
        or observed.get("payload_hash_algorithm")
        != PAYLOAD_HASH_ALGORITHM
        or observed.get("protocol") != DISTRIBUTION_PROTOCOL
        or observed.get("physical_samples_per_clip") != 1
        or observed.get("independent_samples") is not False
        or observed.get("deterministic_delta_distribution") is not True
        or observed.get("replication_algorithm") != REPLICATION_ALGORITHM
        or observed.get("seed_consumed") is not False
        or observed.get("released2_slots") != list(RELEASED2_SLOTS)
        or observed.get("paper16_slots") != list(PAPER16_SLOTS)
        or observed.get("face_slot") != FACE_SLOT
        or observed.get("diffsheg_slot") != DIFFSHEG_SLOT
        or observed.get("metric_input_materialization")
        != METRIC_INPUT_MATERIALIZATION
    ):
        raise MetricAdapterContractError(
            "distribution randomness/slot declaration is invalid"
        )
    return dict(observed)


def _ground_truth_matches_canonical(
    ground_truth: Mapping[str, np.ndarray],
    canonical: Mapping[str, np.ndarray],
) -> None:
    expected = {
        "betas": canonical["beta"][0],
        "poses": canonical["pose"],
        "expressions": canonical["facial"],
        "trans": canonical["trans"],
    }
    for name, value in expected.items():
        if not np.array_equal(ground_truth[name], value):
            raise MetricAdapterContractError(
                f"ground-truth NPZ {name} differs from canonical SHOW"
            )


def _validated_backend_features(
    backend: MetricBackend,
    parameters_265: np.ndarray,
) -> np.ndarray:
    value = _finite_array(
        backend.extract_body_features(parameters_265),
        name="body features",
    )
    if value.ndim != 2 or 0 in value.shape:
        raise MetricAdapterContractError(
            f"body features must be a non-empty matrix, got {value.shape}"
        )
    return value


def _validated_backend_joints(
    backend: MetricBackend,
    parameters_265: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    value = _finite_array(
        backend.joints(parameters_265, betas),
        name="SMPL-X joints",
    )
    expected_prefix = parameters_265.shape[:2]
    if (
        value.ndim != 4
        or value.shape[:2] != expected_prefix
        or value.shape[2] < 75
        or value.shape[3] != 3
    ):
        raise MetricAdapterContractError(
            f"SMPL-X joints have invalid shape {value.shape}"
        )
    return value


def _audio_beats(
    backend: MetricBackend,
    waveform: np.ndarray,
    override: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    function = override
    if function is None:
        function = getattr(backend, "audio_beats", None)
    if not callable(function):
        raise MetricAdapterContractError(
            "metric backend does not provide audio-beat extraction"
        )
    beats = _finite_array(
        function(waveform),
        name="audio beat times",
    ).reshape(-1)
    return beats


def evaluate_canonical_bundle(
    *,
    canonical_manifest: str | Path,
    expected_canonical_manifest_sha256: str,
    prediction_manifest: str | Path,
    expected_prediction_manifest_sha256: str,
    prediction_lineage: str | Path,
    expected_prediction_lineage_sha256: str,
    validation_gate: Mapping[str, Any],
    distribution_declaration: Mapping[str, Any],
    backend: MetricBackend,
    split: str,
    expected_clip_count: int,
    audio_beat_extractor: Callable[[np.ndarray], np.ndarray] | None = None,
    formal_mode: bool = True,
    test_only_allow_four_clip_gate: bool = False,
) -> dict[str, Any]:
    """Evaluate one immutable SemTalk bundle without invoking a generator."""

    if split not in {"val", "test"}:
        raise MetricAdapterContractError("split must be val or test")
    expected_clip_count = _require_exact_int(
        expected_clip_count,
        "expected_clip_count",
        minimum=1,
    )
    if (
        type(formal_mode) is not bool
        or type(test_only_allow_four_clip_gate) is not bool
    ):
        raise MetricAdapterContractError(
            "formal/test-only mode flags must be boolean"
        )
    assets, runtime = _validate_backend_receipts(
        backend,
        require_cuda=formal_mode,
    )
    canonical_path, canonical_payload = _verified_file_snapshot(
        canonical_manifest,
        expected_canonical_manifest_sha256,
        "canonical SHOW manifest",
    )
    canonical_rows = _strict_jsonl_snapshot(
        canonical_payload,
        "canonical SHOW manifest",
    )
    canonical_by_id = _validate_canonical_rows(
        canonical_rows,
        split=split,
        expected_clip_count=expected_clip_count,
    )
    prediction_path, prediction_payload = _verified_file_snapshot(
        prediction_manifest,
        expected_prediction_manifest_sha256,
        "SemTalk prediction manifest",
    )
    prediction_rows = _strict_jsonl_snapshot(
        prediction_payload,
        "SemTalk prediction manifest",
    )
    ordered_predictions = _validate_prediction_rows(
        prediction_rows,
        canonical=canonical_by_id,
        split=split,
    )
    lineage_path, lineage_payload = _verified_file_snapshot(
        prediction_lineage,
        expected_prediction_lineage_sha256,
        "SemTalk prediction lineage",
    )
    lineage = _strict_json_snapshot(
        lineage_payload,
        "SemTalk prediction lineage",
    )
    lineage_payload_sha = _validate_prediction_lineage(
        lineage,
        split=split,
        prediction_manifest_path=prediction_path,
        prediction_manifest_sha256=sha256_bytes(prediction_payload),
        expected_clip_count=expected_clip_count,
    )
    artifact_rows = [
        {
            "canonical_clip_id": row["canonical_clip_id"],
            "prediction_sha256": row["prediction"]["sha256"],
            "prediction_bytes": row["prediction"]["bytes"],
        }
        for row in ordered_predictions
    ]
    gate_artifact = _validate_external_validation_gate(
        validation_gate,
        expected_scope=(
            "validation_candidate_family"
            if split == "val"
            else "final_winner"
        ),
        test_only_allow_four_clip_subset=(
            test_only_allow_four_clip_gate
        ),
    )
    prediction_manifest_artifact = {
        "path": str(prediction_path),
        "sha256": sha256_bytes(prediction_payload),
        "bytes": len(prediction_payload),
    }
    distribution = validate_distribution_declaration(
        distribution_declaration,
        expected_gate_artifact=gate_artifact,
        expected_prediction_manifest=prediction_manifest_artifact,
        expected_prediction_records=artifact_rows,
    )
    body = {
        protocol: {
            "real": FeatureMoments(),
            "generated": FeatureMoments(),
            "variation_sum": 0.0,
            "bc_numerator": 0.0,
            "bc_denominator": 0,
            "bc_sample_evaluations": 0,
        }
        for protocol in PROTOCOLS
    }
    face_sums = {
        "jaw_l1": 0.0,
        "landmark_l1": 0.0,
        "LVD": 0.0,
        "face_l2_combined": 0.0,
    }
    per_speaker = {name: 0 for name in SPEAKER_NAMES}
    frame_count = 0
    for row in ordered_predictions:
        output_id = row["canonical_clip_id"]
        canonical_row = canonical_by_id[output_id]
        frames = canonical_row["frames"]
        canonical_arrays = _load_canonical_npz(
            canonical_row["canonical_npz"],
            expected_sha256=canonical_row["canonical_npz_sha256"],
            frames=frames,
            speaker_id=canonical_row["speaker_id"],
        )
        prediction = _load_output_npz(
            row["prediction"],
            frames=frames,
            prediction=True,
            expected_name=f"res_{output_id}.npz",
        )
        ground_truth = _load_output_npz(
            row["ground_truth"],
            frames=frames,
            prediction=False,
            expected_name=f"gt_{output_id}.npz",
        )
        _ground_truth_matches_canonical(ground_truth, canonical_arrays)
        if not np.array_equal(
            prediction["betas"],
            canonical_arrays["beta"][0],
        ):
            raise MetricAdapterContractError(
                f"{output_id}: prediction betas differ from canonical SHOW"
            )
        ground_truth_265 = reorder_to_talkshow(
            canonical_arrays["pose"],
            canonical_arrays["facial"],
        )
        prediction_265 = reorder_to_talkshow(
            prediction["poses"],
            prediction["expressions"],
        )
        real_features = _validated_backend_features(
            backend,
            ground_truth_265[None],
        )
        body_parameters = released_body_parameters(
            prediction_265[None],
        )
        generated_joints = _validated_backend_joints(
            backend,
            np.repeat(
                body_parameters,
                NUM_LOGICAL_SLOTS,
                axis=0,
            ),
            prediction["betas"],
        )
        waveform = _validate_wav_row_and_decode(
            canonical_row,
            motion_frames=frames,
        )
        beats = _audio_beats(
            backend,
            waveform,
            audio_beat_extractor,
        )
        for protocol, definition in PROTOCOLS.items():
            samples = int(definition["num_samples"])
            state = body[protocol]
            generated_features = _validated_backend_features(
                backend,
                np.repeat(
                    prediction_265[None],
                    samples,
                    axis=0,
                ),
            )
            state["real"].update(real_features)
            state["generated"].update(generated_features)
            physically_repeated = generated_joints[:samples]
            observed_variation = variation_from_joints(
                physically_repeated
            )
            scale = max(
                1.0,
                float(np.max(np.abs(generated_joints))),
            )
            roundoff_tolerance = (
                np.finfo(np.float64).eps
                * scale
                * scale
                * generated_joints.shape[1]
                * generated_joints.shape[2]
                * 64.0
            )
            if (
                not all(
                    np.array_equal(
                        physically_repeated[index],
                        physically_repeated[0],
                    )
                    for index in range(samples)
                )
                or observed_variation > roundoff_tolerance
            ):
                raise MetricAdapterContractError(
                    f"{output_id}: delta-distribution Variation is not zero"
                )
            # The distribution is a mathematically exact point mass.  The
            # released np.var implementation can leave a few floating-point
            # ulps after subtracting its recomputed mean, so canonicalize that
            # proven roundoff to the protocol's exact value.
            variation = 0.0
            state["variation_sum"] += variation
            bc_numerator, bc_denominator = bc_components_for_batch(
                physically_repeated,
                beats,
                str(definition["bc_sample_policy"]),
            )
            state["bc_numerator"] += bc_numerator
            state["bc_denominator"] += bc_denominator
            state["bc_sample_evaluations"] += (
                1
                if definition["bc_sample_policy"] == "first"
                else samples
            )
        isolated_gt, isolated_prediction = released_face_parameters(
            ground_truth_265,
            prediction_265,
        )
        face_joints = _validated_backend_joints(
            backend,
            np.stack((isolated_gt, isolated_prediction)),
            prediction["betas"],
        )
        clip_face = released_face_metrics(
            face_joints[0],
            face_joints[1],
        )
        for name, value in clip_face.items():
            face_sums[name] += value
        per_speaker[canonical_row["speaker"]] += 1
        frame_count += frames
    if sum(per_speaker.values()) != expected_clip_count:
        raise MetricAdapterContractError(
            "per-speaker counts do not cover the metric split"
        )
    body_report: dict[str, Any] = {}
    for protocol, definition in PROTOCOLS.items():
        state = body[protocol]
        if state["bc_denominator"] < 1:
            raise MetricAdapterContractError(
                f"{protocol} produced zero BC denominator"
            )
        metrics = {
            "FGD": frechet_distance(
                state["real"],
                state["generated"],
            ),
            "Variation": state["variation_sum"] / expected_clip_count,
            "BC": state["bc_numerator"] / state["bc_denominator"],
        }
        if (
            metrics["Variation"] != 0.0
            or not all(math.isfinite(value) for value in metrics.values())
        ):
            raise MetricAdapterContractError(
                f"{protocol} metrics violate the finite delta contract"
            )
        body_report[protocol] = {
            "protocol": protocol,
            "logical_slots": list(definition["logical_slots"]),
            "num_samples": definition["num_samples"],
            "bc_sample_policy": definition["bc_sample_policy"],
            "counts": {
                "clips": expected_clip_count,
                "per_speaker": per_speaker,
                "real_features": state["real"].count,
                "generated_features": state["generated"].count,
                "bc_sample_evaluations": state[
                    "bc_sample_evaluations"
                ],
                "bc_denominator": state["bc_denominator"],
            },
            "metrics": metrics,
            "feature_statistics": {
                "real": state["real"].to_json(),
                "generated": state["generated"].to_json(),
            },
        }
    face_means = {
        name: value / expected_clip_count
        for name, value in face_sums.items()
    }
    if not all(math.isfinite(value) for value in face_means.values()):
        raise MetricAdapterContractError("face metric means are non-finite")
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "report_payload_hash_algorithm": (
            REPORT_PAYLOAD_HASH_ALGORITHM
        ),
        "status": "complete",
        "generator": "SemTalk Base-only",
        "dataset": {
            "name": "SHOW",
            "release": "ReleaseV1.0",
            "selection": ">2s",
            "speakers": dict(SHOW_SPEAKER_IDS),
        },
        "split": split,
        "selection_protocol": {
            "primary_metric": "body.released2.metrics.FGD",
            "mode": "min",
            "validation_only_for_selection": True,
            "test_evaluations": 0 if split == "val" else 1,
        },
        "distribution_receipt": distribution,
        "inputs": {
            "canonical_manifest": {
                "path": str(canonical_path),
                "sha256": sha256_bytes(canonical_payload),
                "bytes": len(canonical_payload),
                "rows": len(canonical_rows),
                "selected_rows": expected_clip_count,
            },
            "prediction_manifest": {
                "path": str(prediction_path),
                "sha256": sha256_bytes(prediction_payload),
                "bytes": len(prediction_payload),
                "rows": len(ordered_predictions),
            },
            "prediction_lineage": {
                "path": str(lineage_path),
                "sha256": sha256_bytes(lineage_payload),
                "bytes": len(lineage_payload),
                "payload_sha256": lineage_payload_sha,
            },
        },
        "metric_assets": assets,
        "counts": {
            "clips": expected_clip_count,
            "frames": frame_count,
            "physical_prediction_artifacts": expected_clip_count,
            "logical_samples_per_clip": NUM_LOGICAL_SLOTS,
            "per_speaker": per_speaker,
            "all_tensors_finite": True,
            "exact_once": True,
        },
        "body": body_report,
        "face": {
            "protocol": "released_face_sample0",
            "logical_slot": FACE_SLOT,
            "aggregation": "equal_clip_mean",
            "counts": {
                "clips": expected_clip_count,
                "per_speaker": per_speaker,
            },
            "metrics": {
                "released": {
                    "jaw_l1": face_means["jaw_l1"],
                    "landmark_l1": face_means["landmark_l1"],
                    "LVD": face_means["LVD"],
                },
                "derived": {
                    "face_l2_combined": face_means[
                        "face_l2_combined"
                    ],
                },
            },
        },
        "rs": {"status": "N/A/unreleased", "value": None},
        "runtime": runtime,
        "formal_mode": formal_mode,
        "test_only_mode": test_only_allow_four_clip_gate,
    }
    report["report_payload_sha256"] = canonical_json_sha256(report)
    return report


def _report_number(value: Any, label: str, *, nonnegative: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        raise MetricAdapterContractError(
            f"{label} must be a finite"
            f"{' nonnegative' if nonnegative else ''} number"
        )
    return float(value)


def _validate_report_feature_statistics(
    value: Any,
    *,
    expected_count: int,
    label: str,
) -> None:
    if type(value) is not dict or set(value) != {
        "count",
        "dimension",
        "sum",
        "outer_sum",
        "covariance_denominator",
        "covariance_estimator",
    }:
        raise MetricAdapterContractError(
            f"{label} feature-statistics schema mismatch"
        )
    count = _require_exact_int(value["count"], f"{label} count", minimum=2)
    dimension = _require_exact_int(
        value["dimension"],
        f"{label} dimension",
        minimum=1,
    )
    if (
        count != expected_count
        or value["covariance_denominator"] != count - 1
        or value["covariance_estimator"] != "unbiased_ddof_1"
    ):
        raise MetricAdapterContractError(
            f"{label} unbiased-covariance receipt mismatch"
        )
    feature_sum = _finite_array(value["sum"], name=f"{label} sum")
    outer_sum = _finite_array(
        value["outer_sum"],
        name=f"{label} outer sum",
    )
    if (
        feature_sum.shape != (dimension,)
        or outer_sum.shape != (dimension, dimension)
    ):
        raise MetricAdapterContractError(
            f"{label} sufficient-statistics shape mismatch"
        )


def validate_report(
    report: Mapping[str, Any],
    *,
    expected_split: str,
    expected_clip_count: int,
    expected_prediction_manifest: Mapping[str, Any],
    expected_distribution_receipt: Mapping[str, Any],
    expected_selection_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Fresh, fail-closed validator for selector/merge consumers.

    The Base selector must separately fresh-validate the shared deterministic
    receipt against its prediction records before calling this function.
    This validator then binds that exact receipt, the formal CUDA runtime,
    pinned metric source/assets, and every report field used for selection.
    """

    if expected_split not in {"val", "test"}:
        raise MetricAdapterContractError("invalid expected report split")
    expected_clip_count = _require_exact_int(
        expected_clip_count,
        "expected report clip count",
        minimum=1,
    )
    expected_manifest = dict(expected_prediction_manifest)
    if set(expected_manifest) != {"path", "sha256", "bytes"}:
        raise MetricAdapterContractError(
            "expected prediction-manifest artifact schema mismatch"
        )
    if (
        type(expected_manifest["path"]) is not str
        or not Path(expected_manifest["path"]).is_absolute()
    ):
        raise MetricAdapterContractError(
            "expected prediction-manifest path must be absolute"
        )
    _require_sha256(
        expected_manifest["sha256"],
        "expected prediction-manifest SHA",
    )
    _require_exact_int(
        expected_manifest["bytes"],
        "expected prediction-manifest bytes",
        minimum=1,
    )
    frozen_selection_protocol = {
        "primary_metric": "body.released2.metrics.FGD",
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 0 if expected_split == "val" else 1,
    }
    if dict(expected_selection_protocol) != frozen_selection_protocol:
        raise MetricAdapterContractError(
            "caller selection protocol differs from frozen SHOW policy"
        )
    if type(report) is not dict or set(report) != {
        "format",
        "report_payload_hash_algorithm",
        "status",
        "generator",
        "dataset",
        "split",
        "selection_protocol",
        "distribution_receipt",
        "inputs",
        "metric_assets",
        "counts",
        "body",
        "face",
        "rs",
        "runtime",
        "formal_mode",
        "test_only_mode",
        "report_payload_sha256",
    }:
        raise MetricAdapterContractError("metric report schema mismatch")
    unsigned = dict(report)
    claimed_report_sha = _require_sha256(
        unsigned.pop("report_payload_sha256"),
        "metric report payload SHA",
    )
    if (
        canonical_json_sha256(unsigned) != claimed_report_sha
        or report["report_payload_hash_algorithm"]
        != REPORT_PAYLOAD_HASH_ALGORITHM
    ):
        raise MetricAdapterContractError(
            "metric report payload hash mismatch"
        )
    if (
        report["format"] != REPORT_FORMAT
        or report["status"] != "complete"
        or report["generator"] != "SemTalk Base-only"
        or report["split"] != expected_split
        or report["formal_mode"] is not True
        or report["test_only_mode"] is not False
        or report["selection_protocol"] != frozen_selection_protocol
    ):
        raise MetricAdapterContractError(
            "metric report identity/selection protocol mismatch"
        )
    if report["dataset"] != {
        "name": "SHOW",
        "release": "ReleaseV1.0",
        "selection": ">2s",
        "speakers": dict(SHOW_SPEAKER_IDS),
    }:
        raise MetricAdapterContractError(
            "metric report SHOW dataset contract mismatch"
        )
    distribution = report["distribution_receipt"]
    if (
        distribution != dict(expected_distribution_receipt)
        or distribution.get("format")
        != "semtalk_show_deterministic_distribution_receipt_v1"
        or distribution.get("payload_hash_algorithm")
        != PAYLOAD_HASH_ALGORITHM
        or distribution.get("protocol") != DISTRIBUTION_PROTOCOL
        or distribution.get("physical_samples_per_clip") != 1
        or distribution.get("independent_samples") is not False
        or distribution.get("deterministic_delta_distribution") is not True
        or distribution.get("replication_algorithm")
        != REPLICATION_ALGORITHM
        or distribution.get("seed_consumed") is not False
        or distribution.get("released2_slots")
        != list(RELEASED2_SLOTS)
        or distribution.get("paper16_slots") != list(PAPER16_SLOTS)
        or distribution.get("face_slot") != FACE_SLOT
        or distribution.get("diffsheg_slot") != DIFFSHEG_SLOT
        or distribution.get("metric_input_materialization")
        != METRIC_INPUT_MATERIALIZATION
    ):
        raise MetricAdapterContractError(
            "metric report deterministic distribution receipt mismatch"
        )
    distribution_unsigned = dict(distribution)
    distribution_sha = _require_sha256(
        distribution_unsigned.pop("receipt_payload_sha256"),
        "distribution receipt payload SHA",
    )
    if (
        compact_canonical_json_sha256(distribution_unsigned)
        != distribution_sha
    ):
        raise MetricAdapterContractError(
            "distribution receipt payload hash mismatch"
        )
    inputs = report["inputs"]
    if type(inputs) is not dict or set(inputs) != {
        "canonical_manifest",
        "prediction_manifest",
        "prediction_lineage",
    }:
        raise MetricAdapterContractError("metric report inputs schema mismatch")
    prediction_input = inputs["prediction_manifest"]
    if (
        type(prediction_input) is not dict
        or set(prediction_input) != {
            "path",
            "sha256",
            "bytes",
            "rows",
        }
        or {
            key: prediction_input[key]
            for key in ("path", "sha256", "bytes")
        }
        != expected_manifest
        or distribution.get("prediction_manifest") != expected_manifest
        or prediction_input["rows"] != expected_clip_count
    ):
        raise MetricAdapterContractError(
            "metric report prediction-manifest binding mismatch"
        )
    for role, expected_keys in (
        (
            "canonical_manifest",
            {"path", "sha256", "bytes", "rows", "selected_rows"},
        ),
        (
            "prediction_manifest",
            {"path", "sha256", "bytes", "rows"},
        ),
        (
            "prediction_lineage",
            {"path", "sha256", "bytes", "payload_sha256"},
        ),
    ):
        artifact = inputs[role]
        if type(artifact) is not dict or set(artifact) != expected_keys:
            raise MetricAdapterContractError(
                f"metric report {role} schema mismatch"
            )
        _path, payload = _verified_file_snapshot(
            artifact["path"],
            _require_sha256(
                artifact["sha256"],
                f"metric report {role} SHA",
            ),
            f"metric report {role}",
        )
        if artifact["bytes"] != len(payload):
            raise MetricAdapterContractError(
                f"metric report {role} byte count mismatch"
            )
    canonical_input = inputs["canonical_manifest"]
    if (
        canonical_input["selected_rows"] != expected_clip_count
        or canonical_input["rows"] < expected_clip_count
    ):
        raise MetricAdapterContractError(
            "metric report canonical-manifest counts mismatch"
        )
    assets = report["metric_assets"]
    runtime = report["runtime"]
    _validate_backend_receipts(
        SimpleNamespace(
            asset_receipt=assets,
            runtime_receipt=runtime,
        ),
        require_cuda=True,
    )
    talkshow = assets["talkshow"]
    if (
        not isinstance(talkshow, dict)
        or "path" not in talkshow
        or validate_talkshow_metric_root(talkshow["path"]) != talkshow
    ):
        raise MetricAdapterContractError(
            "metric report TalkSHOW source closure changed"
        )
    for role, expected_sha, expected_dtype in (
        ("feature_extractor", FEATURE_EXTRACTOR_SHA256, "float32"),
        ("smplx", SMPLX_SHA256, "float64"),
    ):
        artifact = assets[role]
        if (
            type(artifact) is not dict
            or set(artifact)
            != {"path", "sha256", "bytes", "runtime_dtype"}
            or artifact["runtime_dtype"] != expected_dtype
        ):
            raise MetricAdapterContractError(
                f"metric report {role} schema/dtype mismatch"
            )
        _path, payload = _verified_file_snapshot(
            artifact["path"],
            expected_sha,
            f"metric report {role}",
        )
        if (
            artifact["sha256"] != expected_sha
            or artifact["bytes"] != len(payload)
        ):
            raise MetricAdapterContractError(
                f"metric report {role} artifact changed"
            )
    counts = report["counts"]
    if type(counts) is not dict or set(counts) != {
        "clips",
        "frames",
        "physical_prediction_artifacts",
        "logical_samples_per_clip",
        "per_speaker",
        "all_tensors_finite",
        "exact_once",
    }:
        raise MetricAdapterContractError("metric report counts schema mismatch")
    per_speaker = counts["per_speaker"]
    if (
        counts["clips"] != expected_clip_count
        or counts["physical_prediction_artifacts"] != expected_clip_count
        or counts["logical_samples_per_clip"] != NUM_LOGICAL_SLOTS
        or counts["all_tensors_finite"] is not True
        or counts["exact_once"] is not True
        or type(per_speaker) is not dict
        or set(per_speaker) != set(SPEAKER_NAMES)
        or any(type(value) is not int or value < 1 for value in per_speaker.values())
        or sum(per_speaker.values()) != expected_clip_count
        or type(counts["frames"]) is not int
        or counts["frames"] < expected_clip_count * 2
    ):
        raise MetricAdapterContractError(
            "metric report exact-once counts mismatch"
        )
    body = report["body"]
    if type(body) is not dict or set(body) != set(PROTOCOLS):
        raise MetricAdapterContractError("metric report body schema mismatch")
    for protocol, definition in PROTOCOLS.items():
        value = body[protocol]
        if type(value) is not dict or set(value) != {
            "protocol",
            "logical_slots",
            "num_samples",
            "bc_sample_policy",
            "counts",
            "metrics",
            "feature_statistics",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} schema mismatch"
            )
        samples = int(definition["num_samples"])
        if (
            value["protocol"] != protocol
            or value["logical_slots"] != definition["logical_slots"]
            or value["num_samples"] != samples
            or value["bc_sample_policy"] != definition["bc_sample_policy"]
        ):
            raise MetricAdapterContractError(
                f"metric report {protocol} protocol mismatch"
            )
        protocol_counts = value["counts"]
        if type(protocol_counts) is not dict or set(protocol_counts) != {
            "clips",
            "per_speaker",
            "real_features",
            "generated_features",
            "bc_sample_evaluations",
            "bc_denominator",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} counts schema mismatch"
            )
        if (
            protocol_counts["clips"] != expected_clip_count
            or protocol_counts["per_speaker"] != per_speaker
            or type(protocol_counts["real_features"]) is not int
            or protocol_counts["real_features"] < 2
            or protocol_counts["generated_features"]
            != protocol_counts["real_features"] * samples
            or protocol_counts["bc_sample_evaluations"]
            != expected_clip_count * (1 if protocol == "released2" else 16)
            or type(protocol_counts["bc_denominator"]) is not int
            or protocol_counts["bc_denominator"] < 1
        ):
            raise MetricAdapterContractError(
                f"metric report {protocol} counts mismatch"
            )
        metrics = value["metrics"]
        if type(metrics) is not dict or set(metrics) != {
            "FGD",
            "Variation",
            "BC",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} metrics schema mismatch"
            )
        fgd = _report_number(
            metrics["FGD"],
            f"{protocol} FGD",
            nonnegative=True,
        )
        variation = _report_number(
            metrics["Variation"],
            f"{protocol} Variation",
            nonnegative=True,
        )
        bc = _report_number(
            metrics["BC"],
            f"{protocol} BC",
            nonnegative=True,
        )
        if variation != 0.0 or bc > 1.0 or not math.isfinite(fgd):
            raise MetricAdapterContractError(
                f"metric report {protocol} metric contract mismatch"
            )
        statistics = value["feature_statistics"]
        if type(statistics) is not dict or set(statistics) != {
            "real",
            "generated",
        }:
            raise MetricAdapterContractError(
                f"metric report {protocol} statistics schema mismatch"
            )
        _validate_report_feature_statistics(
            statistics["real"],
            expected_count=protocol_counts["real_features"],
            label=f"{protocol} real",
        )
        _validate_report_feature_statistics(
            statistics["generated"],
            expected_count=protocol_counts["generated_features"],
            label=f"{protocol} generated",
        )
    face = report["face"]
    if (
        type(face) is not dict
        or set(face)
        != {
            "protocol",
            "logical_slot",
            "aggregation",
            "counts",
            "metrics",
        }
        or face["protocol"] != "released_face_sample0"
        or face["logical_slot"] != FACE_SLOT
        or face["aggregation"] != "equal_clip_mean"
        or face["counts"]
        != {"clips": expected_clip_count, "per_speaker": per_speaker}
        or type(face["metrics"]) is not dict
        or set(face["metrics"]) != {"released", "derived"}
        or type(face["metrics"]["released"]) is not dict
        or set(face["metrics"]["released"])
        != {"jaw_l1", "landmark_l1", "LVD"}
        or type(face["metrics"]["derived"]) is not dict
        or set(face["metrics"]["derived"]) != {"face_l2_combined"}
    ):
        raise MetricAdapterContractError("metric report face schema mismatch")
    released_face = face["metrics"]["released"]
    jaw = _report_number(released_face["jaw_l1"], "face jaw_l1", nonnegative=True)
    landmark = _report_number(
        released_face["landmark_l1"],
        "face landmark_l1",
        nonnegative=True,
    )
    _report_number(released_face["LVD"], "face LVD", nonnegative=True)
    combined = _report_number(
        face["metrics"]["derived"]["face_l2_combined"],
        "face combined",
        nonnegative=True,
    )
    if not math.isclose(
        combined,
        jaw + landmark,
        rel_tol=0.0,
        abs_tol=1e-12 * max(1.0, combined, jaw + landmark),
    ):
        raise MetricAdapterContractError(
            "metric report derived face combined mismatch"
        )
    if report["rs"] != {"status": "N/A/unreleased", "value": None}:
        raise MetricAdapterContractError("metric report RS contract mismatch")
    return {
        "status": "pass",
        "split": expected_split,
        "clips": expected_clip_count,
        "primary_metric_path": "body.released2.metrics.FGD",
        "primary_metric": float(
            body["released2"]["metrics"]["FGD"]
        ),
        "report_payload_sha256": claimed_report_sha,
    }


def _atomic_write_new(path: Path, payload: bytes) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-canonical-manifest-sha256",
        required=True,
    )
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-prediction-manifest-sha256",
        required=True,
    )
    parser.add_argument("--prediction-lineage", type=Path, required=True)
    parser.add_argument(
        "--expected-prediction-lineage-sha256",
        required=True,
    )
    parser.add_argument("--validation-gate-json", type=Path, required=True)
    parser.add_argument(
        "--expected-validation-gate-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-validation-gate-receipt-payload-sha256",
        required=True,
    )
    parser.add_argument(
        "--distribution-declaration-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-distribution-declaration-sha256",
        required=True,
    )
    parser.add_argument("--talkshow-metric-root", type=Path, required=True)
    parser.add_argument("--feature-extractor", type=Path, required=True)
    parser.add_argument("--smplx-asset", type=Path, required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="Explicit formal CUDA ordinal, for example cuda:0",
    )
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument(
        "--expected-clip-count",
        type=int,
        required=True,
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _declaration_path, declaration_payload = _verified_file_snapshot(
        args.distribution_declaration_json,
        args.expected_distribution_declaration_sha256,
        "distribution declaration",
    )
    declaration = _strict_json_snapshot(
        declaration_payload,
        "distribution declaration",
    )
    validation_gate = {
        "path": str(args.validation_gate_json.resolve()),
        "sha256": _require_sha256(
            args.expected_validation_gate_sha256,
            "validation gate file SHA",
        ),
        "bytes": args.validation_gate_json.resolve().stat().st_size,
        "receipt_payload_sha256": _require_sha256(
            args.expected_validation_gate_receipt_payload_sha256,
            "validation gate payload SHA",
        ),
    }
    backend = TalkShowCudaMetricBackend(
        talkshow_root=args.talkshow_metric_root,
        feature_extractor=args.feature_extractor,
        smplx_asset=args.smplx_asset,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    report = evaluate_canonical_bundle(
        canonical_manifest=args.canonical_manifest,
        expected_canonical_manifest_sha256=(
            args.expected_canonical_manifest_sha256
        ),
        prediction_manifest=args.prediction_manifest,
        expected_prediction_manifest_sha256=(
            args.expected_prediction_manifest_sha256
        ),
        prediction_lineage=args.prediction_lineage,
        expected_prediction_lineage_sha256=(
            args.expected_prediction_lineage_sha256
        ),
        validation_gate=validation_gate,
        distribution_declaration=declaration,
        backend=backend,
        split=args.split,
        expected_clip_count=args.expected_clip_count,
    )
    _atomic_write_new(
        args.output_json,
        canonical_json_bytes(report),
    )
    print(
        f"SHOW TalkSHOW metrics complete: split={args.split} "
        f"clips={args.expected_clip_count} "
        f"released2_FGD={report['body']['released2']['metrics']['FGD']:.8f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
