"""Minimal LMDB readers for the SHOW SemTalk Base-only reproduction.

The representation cache stores canonical SHOW motion windows.  The Base cache
stores only tensors consumed by ``semtalk_base_trainer._g_training``.  Neither
reader imports text, semantic, SemGate, Sparse, or test-time dependencies.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch

GLOBAL_FOOT_FIELD = "lower_foot_local"
GLOBAL_FOOT_FASTPATH_ENV = "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH"

# Keep the formal SHOW training adapter independent of the legacy data loader's
# fastText/TextGrid import closure.  These are the exact 55 SMPL-X joints and
# component selections used by the public SemTalk configs.
_SMPLX_55 = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)
_COMPONENT_JOINTS = {
    "beat_smplx_full": _SMPLX_55,
    "beat_smplx_upper": (
        "spine1",
        "spine2",
        "spine3",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ),
    "beat_smplx_hands": _SMPLX_55[25:],
    "beat_smplx_lower": (
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
    ),
    "beat_smplx_face": ("jaw",),
}


def _global_foot_fastpath_enabled(args: Any) -> bool:
    raw = os.environ.get(GLOBAL_FOOT_FASTPATH_ENV, "0")
    if raw not in {"0", "1"}:
        raise RuntimeError(
            f"{GLOBAL_FOOT_FASTPATH_ENV} must be exactly 0 or 1, got {raw!r}"
        )
    enabled = raw == "1"
    if enabled and getattr(args, "formal_stage", None) != "global":
        raise RuntimeError(
            f"{GLOBAL_FOOT_FASTPATH_ENV}=1 is restricted to formal_stage=global"
        )
    if enabled and getattr(args, "tar_joints", None) != "beat_smplx_lower":
        raise RuntimeError(
            "Global foot fastpath requires tar_joints=beat_smplx_lower"
        )
    return enabled


def _joint_context(args: Any) -> tuple[np.ndarray, int]:
    if args.ori_joints != "beat_smplx_joints":
        raise RuntimeError(
            "formal SHOW training requires ori_joints=beat_smplx_joints"
        )
    try:
        target = _COMPONENT_JOINTS[args.tar_joints]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported formal SHOW target joints: {args.tar_joints!r}"
        ) from error
    source_indices = {name: index for index, name in enumerate(_SMPLX_55)}
    mask = np.zeros(len(_SMPLX_55) * 3, dtype=bool)
    for joint_name in target:
        start = source_indices[joint_name] * 3
        mask[start:start + 3] = True
    return mask, len(target)


class _NPZLMDB(torch.utils.data.Dataset):
    def __init__(self, path: str, required_fields: tuple[str, ...]):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"LMDB directory not found: {self.path}")
        self.required_fields = required_fields
        self._env: lmdb.Environment | None = None
        env = self._open()
        with env.begin(buffers=True) as txn:
            self.length = int(txn.stat()["entries"])
        env.close()
        if self.length <= 0:
            raise RuntimeError(f"empty LMDB: {self.path}")
        self._validate_sample(self._read(0))
        assert self._env is not None
        self._env.close()
        self._env = None

    def _open(self) -> lmdb.Environment:
        return lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=512,
            subdir=True,
        )

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = self._open()
        return self._env

    def _read(
        self,
        index: int,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        key = f"{index:010d}".encode("ascii")
        env = self._ensure_env()
        with env.begin(buffers=True) as txn:
            value = txn.get(key)
            if value is None:
                raise IndexError(f"missing contiguous LMDB key {key!r}")
            with io.BytesIO(bytes(value)) as handle:
                archive = np.load(handle, allow_pickle=False)
                selected = tuple(archive.files) if fields is None else fields
                missing = sorted(set(selected) - set(archive.files))
                if missing:
                    raise RuntimeError(
                        f"LMDB sample {index} missing selected fields: {missing}"
                    )
                return {name: archive[name].copy() for name in selected}

    def _validate_sample(self, sample: dict[str, np.ndarray]) -> None:
        missing = sorted(set(self.required_fields) - set(sample))
        if missing:
            raise RuntimeError(f"LMDB sample missing fields: {missing}")
        for name in self.required_fields:
            array = np.asarray(sample[name])
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise RuntimeError(f"non-finite LMDB field {name}")

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()


class CustomDataset(_NPZLMDB):
    """Representation-training windows derived from canonical SHOW clips."""

    _FIELDS = ("pose", "contact", "facial", "beta", "trans", "speaker_id")

    def __init__(self, args: Any, loader_type: str, **_: Any):
        if loader_type != "train":
            raise RuntimeError("formal SHOW representation training exposes train only")
        self.args = args
        self.global_foot_fastpath = _global_foot_fastpath_enabled(args)
        self.joint_mask, self.joints = _joint_context(args)
        stage_fields = {
            "face": ("pose", "facial", "beta", "trans"),
            "hands": ("pose", "beta", "trans"),
            "upper": ("pose", "beta", "trans"),
            "lower": ("pose", "contact", "beta", "trans"),
            "global": (
                "pose",
                "contact",
                "trans",
                GLOBAL_FOOT_FIELD,
            ),
        }
        try:
            self.consumed_fields = stage_fields[args.formal_stage]
        except KeyError as error:
            raise RuntimeError(
                f"unsupported representation formal_stage "
                f"{args.formal_stage!r}"
            ) from error
        required_fields = self._FIELDS + (
            (GLOBAL_FOOT_FIELD,) if self.global_foot_fastpath else ()
        )
        super().__init__(args.train_path, required_fields)

    def _validate_sample(self, sample: dict[str, np.ndarray]) -> None:
        super()._validate_sample(sample)
        pose = sample["pose"]
        length = int(self.args.pose_length)
        expected = {
            "pose": (length, 165),
            "contact": (length, 4),
            "facial": (length, 100),
            "beta": (length, 300),
            "trans": (length, 3),
            "speaker_id": (length, 1),
        }
        if self.global_foot_fastpath:
            expected[GLOBAL_FOOT_FIELD] = (length, 4, 3)
        for name, shape in expected.items():
            if sample[name].shape != shape:
                raise RuntimeError(
                    f"invalid representation field {name}: "
                    f"{sample[name].shape} != {shape}"
                )

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample = self._read(index, self.consumed_fields)
        pose = sample["pose"][:, self.joint_mask]
        if self.args.tar_joints == "beat_smplx_lower":
            pose = np.concatenate([pose, sample["contact"]], axis=-1)
        result = {
            # Immutable LMDB key index requested by DataLoader.  Cache lookup
            # must never infer identity from mutable sample payload fields.
            "sample_index": np.int64(index),
            "pose": pose.astype(np.float32, copy=False),
            "trans": sample["trans"].astype(np.float32, copy=False),
        }
        if "facial" in sample:
            result["facial"] = sample["facial"].astype(
                np.float32,
                copy=False,
            )
        if "beta" in sample:
            result["beta"] = sample["beta"].astype(
                np.float32,
                copy=False,
            )
        if self.global_foot_fastpath:
            result[GLOBAL_FOOT_FIELD] = sample[GLOBAL_FOOT_FIELD].astype(
                np.float32, copy=False
            )
        return result


class LMDBNPZDataset(_NPZLMDB):
    """Train-only Base windows after frozen VQ/HuBERT feature extraction."""

    _FIELDS = (
        "tar_pose",
        "beat",
        "in_word",
        "tar_id",
        "latent_all",
        "hubert",
        "zq_face",
        "zq_upper",
        "zq_hands",
        "zq_lower",
        "tar_index_value_face_top",
        "tar_index_value_upper_top",
        "tar_index_value_hands_top",
        "tar_index_value_lower_top",
    )

    def __init__(self, args: Any, loader_type: str):
        if loader_type != "train":
            raise RuntimeError("formal SHOW Base training exposes train only")
        self.args = args
        super().__init__(args.train_path, self._FIELDS)

    def _validate_sample(self, sample: dict[str, np.ndarray]) -> None:
        super()._validate_sample(sample)
        length = int(self.args.pose_length)
        frame_fields = (
            "tar_pose",
            "beat",
            "in_word",
            "tar_id",
            "latent_all",
            "hubert",
        )
        for name in frame_fields:
            if sample[name].ndim == 0 or sample[name].shape[0] != length:
                raise RuntimeError(f"Base field {name} has invalid shape {sample[name].shape}")
        if sample["tar_pose"].shape != (length, 165):
            raise RuntimeError("Base tar_pose must be [T,165]")
        if sample["beat"].shape != (length, 3):
            raise RuntimeError("Base beat must be [T,3]")
        if sample["hubert"].shape != (length, 1024):
            raise RuntimeError("Base hubert must be [T,1024]")
        if sample["latent_all"].shape != (length, 337):
            raise RuntimeError("Base latent_all must be [T,337]")
        if sample["in_word"].shape not in {(length,), (length, 1)}:
            raise RuntimeError("Base in_word must be a zero [T] placeholder")
        if np.any(sample["in_word"]):
            raise RuntimeError("Base in_word placeholder must be identically zero")
        if sample["tar_id"].shape != (length, 1):
            raise RuntimeError("Base tar_id must be [T,1]")
        if sample["tar_id"].min() < 0 or sample["tar_id"].max() > 3:
            raise RuntimeError("SHOW speaker IDs must be in [0,3]")
        latent_length = length // 4
        for name in self._FIELDS:
            if name.startswith("tar_index_value_"):
                if sample[name].shape != (latent_length, 6):
                    raise RuntimeError(f"{name} must be [T/4,6]")
                if sample[name].dtype.kind not in "iu":
                    raise RuntimeError(f"{name} must have integer dtype")
            if name.startswith("zq_") and sample[name].shape != (
                6,
                1,
                latent_length,
                256,
            ):
                raise RuntimeError(f"{name} must be [6,1,T/4,256]")

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return self._read(index)


class PickleDataset(torch.utils.data.Dataset):
    """Deliberately unavailable: final inference uses a separate audited path."""

    def __init__(self, *_: Any, **__: Any):
        raise RuntimeError("SHOW train-only adapter does not expose test data")
