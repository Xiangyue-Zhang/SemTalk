#!/usr/bin/env python3
"""Build the canonical, lossless SHOW motion cache used by SemTalk Base.

The input is the immutable symlink-only split view produced from the official
TalkSHOW split::

    <split-root>/<speaker>/<video>/<split>/<clip>/
        <clip>.pkl
        <clip>.wav

Each build invocation owns one deterministic modulo shard.  It writes one
compressed NPZ per source clip plus a shard-local JSONL manifest, summary, and
lineage receipt.  After every shard succeeds, invoke the same command with
``--finalize`` to verify all source/output hashes and NPZ payloads, prove
cross-split exact-once coverage, and write the root manifest and receipts.

No audio features, VQ latents, language features, or training windows are
created here.  GPU isolation is deliberately external: callers should set
``CUDA_VISIBLE_DEVICES`` (through the guarded runner) and pass ``--device
cuda:0``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import platform
import subprocess
import sys
import uuid
import wave
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.dont_write_bytecode = True


SCHEMA_NAME = "semtalk-show-canonical-motion"
SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}
EXPECTED_SPLIT_COUNTS = {"train": 13_687, "val": 1_715, "test": 1_708}
EXPECTED_MISSING_COUNT = 55
OFFICIAL_SPLIT_SHA256 = (
    "2df6e745cdf7473f13ce3ae2ed759c3cceb60c9197e7f3fd65110e7bc20b6f2d"
)
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SHOW_SPEAKER_ID = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
SPEAKERS = tuple(SHOW_SPEAKER_ID)
POSE_FPS = 30
SOURCE_AUDIO_SAMPLE_RATE = 22_000
SOURCE_AUDIO_SAMPLE_WIDTH = 2
HUBERT_AUDIO_SAMPLE_RATE = 16_000
ACCEPTED_AUDIO_CHANNELS = (1, 2)
AUDIO_MONO_POLICY = "librosa.load(sr=None,mono=True):arithmetic_channel_mean"
FOOT_JOINTS = (7, 8, 10, 11)
CONTACT_THRESHOLD = 0.01
NPZ_FIELDS = ("pose", "contact", "facial", "beta", "trans", "speaker_id")
GLOBAL_FOOT_FASTPATH_CONTRACT = "semtalk_show_global_foot_fastpath_v1"
GLOBAL_FOOT_FIELD = "lower_foot_local"
# Indices in the canonical 55-joint SMPL-X axis-angle pose.  This is the
# insertion order of dataloaders.data_tools.joints_list["beat_smplx_lower"].
LOWER_JOINT_INDICES = (0, 1, 2, 4, 5, 7, 8, 10, 11)
REQUIRED_SOURCE_FIELDS = (
    "betas",
    "global_orient",
    "body_pose_axis",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "left_hand_pose",
    "right_hand_pose",
    "expression",
    "transl",
)


class ShowCacheError(RuntimeError):
    """Raised when the frozen SHOW input or cache output violates the contract."""


@dataclass(frozen=True)
class Clip:
    """One source clip in deterministic global manifest order."""

    global_index: int
    split: str
    speaker: str
    video: str
    sequence: str
    clip_root: Path
    pkl_path: Path
    wav_path: Path

    @property
    def clip_id(self) -> str:
        return f"{self.speaker}/{self.video}/{self.sequence}"

    @property
    def output_relative_path(self) -> Path:
        return (
            Path("clips")
            / self.split
            / self.speaker
            / self.video
            / f"{self.sequence}.npz"
        )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict, stable JSON."""

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


def _require_sha256(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        ch not in "0123456789abcdef" for ch in normalized
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _require_git_oid(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) not in {40, 64} or any(
        ch not in "0123456789abcdef" for ch in normalized
    ):
        raise ValueError(f"{label} must be a lowercase Git object ID")
    return normalized


def source_receipt(
    expected_commit: str,
    expected_tree: str,
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ShowCacheError(
            "canonical builder requires a clean source checkout; first change: "
            f"{status.splitlines()[0]}"
        )
    receipt = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
    }
    if receipt["origin"] != EXPECTED_ORIGIN:
        raise ShowCacheError(
            f"canonical source origin {receipt['origin']!r} "
            f"!= {EXPECTED_ORIGIN!r}"
        )
    if receipt["commit"] != expected_commit:
        raise ShowCacheError(
            f"canonical source commit {receipt['commit']} != {expected_commit}"
        )
    if receipt["tree"] != expected_tree:
        raise ShowCacheError(
            f"canonical source tree {receipt['tree']} != {expected_tree}"
        )
    return receipt


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_new(path: str | Path, payload: bytes) -> None:
    """Atomically create *path* while refusing to replace an existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing file: {destination}"
            ) from exc
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json_new(path: str | Path, value: Any) -> None:
    atomic_write_new(path, canonical_json_bytes(value))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Encode a compressed NPZ with fixed ZIP metadata for byte reproducibility."""

    if tuple(arrays) != NPZ_FIELDS:
        raise ValueError(
            f"NPZ fields must be exactly {NPZ_FIELDS}, observed {tuple(arrays)}"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in NPZ_FIELDS:
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


def load_split_receipt(
    split_root: str | Path,
    expected_split_sha256: str = OFFICIAL_SPLIT_SHA256,
) -> tuple[dict[str, Any], Path, str]:
    root = Path(split_root).expanduser().resolve()
    receipt_path = root / "split_view_summary.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"frozen split receipt not found: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShowCacheError(f"cannot parse split receipt {receipt_path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ShowCacheError(f"split receipt must be an object: {receipt_path}")
    if receipt.get("split_sha256") != expected_split_sha256:
        raise ShowCacheError(
            "split receipt SHA does not match the required frozen manifest: "
            f"{receipt.get('split_sha256')!r} != {expected_split_sha256!r}"
        )
    counts = receipt.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(SPLITS)
        or any(
            isinstance(counts[name], bool)
            or not isinstance(counts[name], int)
            or counts[name] < 1
            for name in SPLITS
        )
    ):
        raise ShowCacheError(f"invalid split counts in {receipt_path}: {counts!r}")
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ShowCacheError(
            f"frozen SHOW split counts changed: {counts!r} "
            f"!= {EXPECTED_SPLIT_COUNTS!r}"
        )
    missing_count = receipt.get("missing_count")
    if (
        isinstance(missing_count, bool)
        or not isinstance(missing_count, int)
        or missing_count < 0
    ):
        raise ShowCacheError(
            f"invalid missing_count in {receipt_path}: {missing_count!r}"
        )
    if missing_count != EXPECTED_MISSING_COUNT:
        raise ShowCacheError(
            f"frozen SHOW missing count changed: {missing_count} "
            f"!= {EXPECTED_MISSING_COUNT}"
        )
    return receipt, receipt_path, sha256_file(receipt_path)


def scan_split_view(
    split_root: str | Path,
    receipt: Mapping[str, Any],
) -> list[Clip]:
    """Scan every split and prove deterministic, disjoint exact-once coverage."""

    root = Path(split_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"split root is not a directory: {root}")
    source_root_value = receipt.get("source_root")
    if not isinstance(source_root_value, str) or not source_root_value:
        raise ShowCacheError("split receipt has no immutable source_root")
    source_root = Path(source_root_value).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"immutable SHOW source root is unavailable: {source_root}"
        )
    observed_speakers = {path.name for path in root.iterdir() if path.is_dir()}
    if observed_speakers != set(SPEAKERS):
        raise ShowCacheError(
            "split-view top-level speaker directories differ from the "
            f"four-speaker contract: {sorted(observed_speakers)}"
        )

    unordered: list[tuple[str, str, str, str, Path, Path, Path]] = []
    seen_clip_ids: dict[str, str] = {}
    seen_source_roots: dict[Path, str] = {}
    observed_counts: Counter[str] = Counter()
    problems: list[str] = []
    for split in SPLITS:
        for speaker in SPEAKERS:
            speaker_root = root / speaker
            if not speaker_root.is_dir():
                problems.append(f"missing speaker directory: {speaker_root}")
                continue
            for video_root in sorted(
                (path for path in speaker_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            ):
                split_dir = video_root / split
                if not split_dir.is_dir():
                    continue
                for clip_root in sorted(
                    (path for path in split_dir.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                ):
                    if not clip_root.is_symlink():
                        problems.append(
                            f"clip entry is not a frozen split-view symlink: {clip_root}"
                        )
                        continue
                    resolved_clip_root = clip_root.resolve()
                    try:
                        resolved_clip_root.relative_to(source_root)
                    except ValueError:
                        problems.append(
                            "split-view symlink escapes immutable source root: "
                            f"{clip_root} -> {resolved_clip_root}"
                        )
                        continue
                    sequence = clip_root.name
                    pkl_path = clip_root / f"{sequence}.pkl"
                    wav_path = clip_root / f"{sequence}.wav"
                    missing = [
                        str(path)
                        for path in (pkl_path, wav_path)
                        if not path.is_file()
                    ]
                    if missing:
                        problems.append(
                            f"incomplete source clip {clip_root}; missing {missing}"
                        )
                        continue
                    clip_id = f"{speaker}/{video_root.name}/{sequence}"
                    previous_clip_id = seen_source_roots.get(resolved_clip_root)
                    if previous_clip_id is not None:
                        problems.append(
                            "source directory is linked more than once: "
                            f"{resolved_clip_root} -> {previous_clip_id}, {clip_id}"
                        )
                        continue
                    previous_split = seen_clip_ids.get(clip_id)
                    if previous_split is not None:
                        problems.append(
                            f"cross-split duplicate {clip_id}: "
                            f"{previous_split!r} and {split!r}"
                        )
                        continue
                    seen_clip_ids[clip_id] = split
                    seen_source_roots[resolved_clip_root] = clip_id
                    observed_counts[split] += 1
                    unordered.append(
                        (
                            split,
                            speaker,
                            video_root.name,
                            sequence,
                            clip_root,
                            pkl_path,
                            wav_path,
                        )
                    )

    expected_counts = receipt["counts"]
    for split in SPLITS:
        if observed_counts[split] != expected_counts[split]:
            problems.append(
                f"{split} count mismatch: scanned={observed_counts[split]}, "
                f"receipt={expected_counts[split]}"
            )
    if problems:
        preview = "\n".join(f"  - {problem}" for problem in problems[:30])
        raise ShowCacheError(
            f"frozen split scan found {len(problems)} problem(s):\n{preview}"
        )

    unordered.sort(
        key=lambda row: (
            SPLIT_ORDER[row[0]],
            row[1],
            row[2],
            row[3],
        )
    )
    return [
        Clip(
            global_index=index,
            split=row[0],
            speaker=row[1],
            video=row[2],
            sequence=row[3],
            clip_root=row[4],
            pkl_path=row[5],
            wav_path=row[6],
        )
        for index, row in enumerate(unordered)
    ]


def select_shard(
    clips: Sequence[Clip],
    shard_id: int,
    num_shards: int,
) -> list[Clip]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    return [clip for clip in clips if clip.global_index % num_shards == shard_id]


def load_hand_components(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    component_path = Path(path).expanduser().resolve()
    if not component_path.is_file():
        raise FileNotFoundError(f"hand PCA component file not found: {component_path}")
    try:
        payload = json.loads(component_path.read_text(encoding="utf-8"))
        left = np.asarray(payload["left"], dtype=np.float32)[:12]
        right = np.asarray(payload["right"], dtype=np.float32)[:12]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ShowCacheError(
            f"cannot load canonical hand components {component_path}: {exc}"
        ) from exc
    if left.shape != (12, 45) or right.shape != (12, 45):
        raise ShowCacheError(
            f"hand PCA shapes must be (12,45), got {left.shape} and {right.shape}"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ShowCacheError(f"hand components contain NaN/Inf: {component_path}")
    return left, right


def normalize_betas(raw_betas: Any, frames: int, context: str) -> np.ndarray:
    betas = np.asarray(raw_betas, dtype=np.float32)
    if betas.ndim == 1:
        vector = betas
    elif betas.ndim == 2 and betas.shape[0] == 1:
        vector = betas[0]
    elif betas.ndim == 2 and betas.shape[0] == frames:
        vector = betas[0]
        if not np.allclose(
            betas,
            np.broadcast_to(vector, betas.shape),
            rtol=1e-5,
            atol=1e-6,
            equal_nan=False,
        ):
            raise ShowCacheError(f"{context}: per-frame betas are not constant")
    else:
        raise ShowCacheError(f"{context}: invalid betas shape {betas.shape}")
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 10:
        vector = np.pad(vector, (0, 290))
    if vector.shape != (300,) or not np.isfinite(vector).all():
        raise ShowCacheError(
            f"{context}: betas must resolve to finite (300,), got {vector.shape}"
        )
    return np.broadcast_to(vector, (frames, 300)).copy()


def load_canonical_motion(
    pkl_path: str | Path,
    left_components: np.ndarray,
    right_components: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load and validate one SHOW pickle, excluding SMPL-X contact."""

    source = Path(pkl_path)
    try:
        with source.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ShowCacheError(f"cannot load SHOW pickle {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShowCacheError(f"{source}: top-level pickle object is not a dict")
    missing = [name for name in REQUIRED_SOURCE_FIELDS if name not in payload]
    if missing:
        raise ShowCacheError(f"{source}: missing source fields {missing}")

    body = np.asarray(payload["body_pose_axis"], dtype=np.float32)
    if body.ndim < 1 or body.shape[0] < 1:
        raise ShowCacheError(f"{source}: invalid body_pose_axis shape {body.shape}")
    frames = int(body.shape[0])
    try:
        global_orient = np.asarray(
            payload["global_orient"], dtype=np.float32
        ).reshape(frames, 3)
        body = body.reshape(frames, 63)
        jaw = np.asarray(payload["jaw_pose"], dtype=np.float32).reshape(frames, 3)
        left_eye = np.asarray(
            payload["leye_pose"], dtype=np.float32
        ).reshape(frames, 3)
        right_eye = np.asarray(
            payload["reye_pose"], dtype=np.float32
        ).reshape(frames, 3)
        left_pca = np.asarray(
            payload["left_hand_pose"], dtype=np.float32
        ).reshape(frames, 12)
        right_pca = np.asarray(
            payload["right_hand_pose"], dtype=np.float32
        ).reshape(frames, 12)
        facial = np.asarray(
            payload["expression"], dtype=np.float32
        ).reshape(frames, 100)
        trans = np.asarray(payload["transl"], dtype=np.float32).reshape(frames, 3)
    except (TypeError, ValueError) as exc:
        raise ShowCacheError(f"{source}: invalid source array shape: {exc}") from exc

    left_hand = (left_pca @ left_components).astype(np.float32)
    right_hand = (right_pca @ right_components).astype(np.float32)
    pose = np.concatenate(
        (
            global_orient,
            body,
            jaw,
            left_eye,
            right_eye,
            left_hand,
            right_hand,
        ),
        axis=1,
    ).astype(np.float32)
    beta = normalize_betas(payload["betas"], frames, str(source))
    arrays = {
        "pose": pose,
        "facial": facial.astype(np.float32),
        "beta": beta.astype(np.float32),
        "trans": trans.astype(np.float32),
    }
    expected = {
        "pose": (frames, 165),
        "facial": (frames, 100),
        "beta": (frames, 300),
        "trans": (frames, 3),
    }
    for name, array in arrays.items():
        if array.shape != expected[name] or not np.isfinite(array).all():
            raise ShowCacheError(
                f"{source}: invalid {name}: shape={array.shape}, "
                f"expected={expected[name]}, finite={np.isfinite(array).all()}"
            )
    return arrays


def inspect_wav(
    path: str | Path,
    expected_rate: int,
) -> dict[str, int | str]:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_width = int(handle.getsampwidth())
            rate = int(handle.getframerate())
            frames = int(handle.getnframes())
            compression = handle.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise ShowCacheError(f"cannot parse SHOW WAV {source}: {exc}") from exc
    if channels not in ACCEPTED_AUDIO_CHANNELS:
        raise ShowCacheError(
            f"{source}: expected one of {ACCEPTED_AUDIO_CHANNELS} source "
            f"channels, got {channels}"
        )
    if rate != expected_rate:
        raise ShowCacheError(
            f"{source}: sample rate {rate} does not match required {expected_rate}"
        )
    if (
        sample_width != SOURCE_AUDIO_SAMPLE_WIDTH
        or frames < 1
        or compression != "NONE"
    ):
        raise ShowCacheError(
            f"{source}: unsupported WAV metadata: width={sample_width}, "
            f"frames={frames}, compression={compression!r}"
        )
    return {
        "wav_channels": channels,
        "wav_sample_width": sample_width,
        "wav_sample_rate": rate,
        "wav_frames": frames,
        "wav_mono_policy": AUDIO_MONO_POLICY,
    }


def validate_wav_manifest_metadata(
    row: Mapping[str, Any],
    observed: Mapping[str, int | str],
    context: str,
) -> None:
    """Require the manifest to preserve every inspected source WAV property."""

    for key, value in observed.items():
        if row.get(key) != value:
            raise ShowCacheError(
                f"{context} {key} mismatch: {row.get(key)!r} != {value!r}"
            )


def resolve_smplx_asset(path: str | Path) -> tuple[Path, Path, str]:
    """Resolve an asset and the root accepted by ``smplx.create``."""

    supplied_input = Path(path).expanduser()
    if supplied_input.is_symlink():
        raise ShowCacheError(
            f"SMPL-X input must not be a symlink: {supplied_input}"
        )
    supplied = supplied_input.resolve()
    candidates: list[Path]
    if supplied.is_file():
        candidates = [supplied]
    else:
        candidates = [
            supplied / "smplx" / "SMPLX_NEUTRAL.npz",
            supplied / "smplx" / "SMPLX_NEUTRAL_2020.npz",
            supplied / "SMPLX_NEUTRAL.npz",
            supplied / "SMPLX_NEUTRAL_2020.npz",
        ]
    assets = [
        candidate
        for candidate in candidates
        if candidate.is_file() and not candidate.is_symlink()
    ]
    if len(assets) != 1:
        raise FileNotFoundError(
            f"expected exactly one neutral SMPL-X asset under {supplied}, "
            f"found {assets}"
        )
    asset = assets[0]
    if asset.parent.name == "smplx":
        model_root = asset.parent.parent
    else:
        model_root = asset.parent
    gender = "NEUTRAL_2020" if "_2020" in asset.stem else "NEUTRAL"
    return asset, model_root, gender


class ContactComputer:
    """Compute contact and the trainer-exact Global-VAE local foot cache."""

    def __init__(
        self,
        model_root: Path,
        gender: str,
        device: str,
        chunk_frames: int,
    ):
        if chunk_frames < 1:
            raise ValueError("--smplx-chunk-frames must be positive")
        try:
            import torch
            import smplx
        except ImportError as exc:
            raise RuntimeError(
                "torch and smplx are required to compute SHOW foot contact"
            ) from exc
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from utils import rotation_conversions as rc

        self.torch = torch
        self.rc = rc
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is unavailable for requested {device}")
            index = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(index)
            self.device = torch.device(f"cuda:{index}")
        self.chunk_frames = int(chunk_frames)
        cuda_runtime = getattr(torch.version, "cuda", None)
        cudnn_runtime = (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        )
        device_properties = None
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            device_properties = {
                "name": str(properties.name),
                "total_memory": int(properties.total_memory),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "multi_processor_count": int(properties.multi_processor_count),
            }
        self.runtime = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "torch_cuda": cuda_runtime,
            "torch_cudnn": cudnn_runtime,
            "smplx": str(getattr(smplx, "__version__", "unknown")),
            "device": str(self.device),
            "device_properties": device_properties,
            "smplx_chunk_frames": self.chunk_frames,
            "global_foot_fastpath_contract": GLOBAL_FOOT_FASTPATH_CONTRACT,
            "rotation_conversions_path": str(Path(rc.__file__).resolve()),
            "rotation_conversions_sha256": sha256_file(Path(rc.__file__).resolve()),
        }
        self.model = smplx.create(
            model_path=str(model_root),
            model_type="smplx",
            gender=gender,
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext="npz",
            use_pca=False,
            flat_hand_mean=False,
            dtype=torch.float32,
        ).to(self.device).eval()

    def __call__(
        self,
        pose: np.ndarray,
        facial: np.ndarray,
        beta: np.ndarray,
        trans: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = int(pose.shape[0])
        joint_chunks = []
        lower_local_chunks = []
        torch = self.torch
        for start in range(0, frames, self.chunk_frames):
            end = min(frames, start + self.chunk_frames)
            pose_chunk = torch.from_numpy(pose[start:end]).to(self.device)
            beta_chunk = torch.from_numpy(beta[start:end]).to(self.device)
            facial_chunk = torch.from_numpy(facial[start:end]).to(self.device)
            trans_chunk = torch.from_numpy(trans[start:end]).to(self.device)
            with torch.inference_mode():
                output = self.model(
                    betas=beta_chunk,
                    transl=trans_chunk,
                    expression=facial_chunk,
                    global_orient=pose_chunk[:, 0:3],
                    body_pose=pose_chunk[:, 3:66],
                    jaw_pose=pose_chunk[:, 66:69],
                    leye_pose=pose_chunk[:, 69:72],
                    reye_pose=pose_chunk[:, 72:75],
                    left_hand_pose=pose_chunk[:, 75:120],
                    right_hand_pose=pose_chunk[:, 120:165],
                    return_verts=False,
                    return_joints=True,
                )
                # Reproduce aelowerfoot_trainer exactly:
                # AA -> matrix -> 6D -> matrix -> AA for the nine selected
                # lower joints, insert them into an otherwise-zero 165-D
                # pose, and use zero translation/expression.
                selected = pose_chunk.reshape(-1, 55, 3)[
                    :, LOWER_JOINT_INDICES, :
                ]
                selected = self.rc.axis_angle_to_matrix(selected)
                selected = self.rc.matrix_to_rotation_6d(selected)
                selected = self.rc.rotation_6d_to_matrix(selected)
                selected = self.rc.matrix_to_axis_angle(selected)
                lower_only = torch.zeros(
                    (end - start, 55, 3),
                    dtype=pose_chunk.dtype,
                    device=self.device,
                )
                lower_only[:, LOWER_JOINT_INDICES, :] = selected
                lower_only = lower_only.reshape(end - start, 165)
                local_output = self.model(
                    betas=beta_chunk,
                    transl=torch.zeros_like(trans_chunk),
                    expression=torch.zeros_like(facial_chunk),
                    global_orient=lower_only[:, 0:3],
                    body_pose=lower_only[:, 3:66],
                    jaw_pose=lower_only[:, 66:69],
                    leye_pose=lower_only[:, 69:72],
                    reye_pose=lower_only[:, 72:75],
                    left_hand_pose=lower_only[:, 75:120],
                    right_hand_pose=lower_only[:, 120:165],
                    return_verts=False,
                    return_joints=True,
                )
            joint_chunks.append(
                output.joints[:, FOOT_JOINTS, :]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
            lower_local_chunks.append(
                local_output.joints[:, FOOT_JOINTS, :]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
        joints = np.concatenate(joint_chunks, axis=0)
        lower_foot_local = np.concatenate(lower_local_chunks, axis=0)
        if joints.shape != (frames, 4, 3) or not np.isfinite(joints).all():
            raise ShowCacheError(
                f"SMPL-X returned invalid foot joints: {joints.shape}"
            )
        if (
            lower_foot_local.shape != (frames, 4, 3)
            or lower_foot_local.dtype != np.float32
            or not np.isfinite(lower_foot_local).all()
        ):
            raise ShowCacheError(
                "SMPL-X returned invalid trainer-exact lower foot joints: "
                f"{lower_foot_local.shape}/{lower_foot_local.dtype}"
            )
        speed = np.zeros((frames, 4), dtype=np.float32)
        if frames > 1:
            speed[:-1] = np.linalg.norm(joints[1:] - joints[:-1], axis=-1)
        contact = (speed < CONTACT_THRESHOLD).astype(np.float32)
        if not np.array_equal(contact[-1], np.ones(4, dtype=np.float32)):
            raise AssertionError("last-frame contact must be exactly one")
        return contact, lower_foot_local


def validate_canonical_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    frames: int,
    speaker_id: int,
    context: str,
) -> None:
    if tuple(arrays) != NPZ_FIELDS:
        raise ShowCacheError(
            f"{context}: NPZ fields {tuple(arrays)} do not equal {NPZ_FIELDS}"
        )
    expected = {
        "pose": (frames, 165),
        "contact": (frames, 4),
        "facial": (frames, 100),
        "beta": (frames, 300),
        "trans": (frames, 3),
        "speaker_id": (frames, 1),
    }
    for name, array in arrays.items():
        if array.shape != expected[name] or not np.isfinite(array).all():
            raise ShowCacheError(
                f"{context}: invalid {name}: shape={array.shape}, "
                f"expected={expected[name]}, finite={np.isfinite(array).all()}"
            )
    if arrays["pose"].dtype != np.float32:
        raise ShowCacheError(f"{context}: pose dtype must be float32")
    for name in ("contact", "facial", "beta", "trans"):
        if arrays[name].dtype != np.float32:
            raise ShowCacheError(f"{context}: {name} dtype must be float32")
    if arrays["speaker_id"].dtype != np.int64:
        raise ShowCacheError(f"{context}: speaker_id dtype must be int64")
    if not np.all(arrays["speaker_id"] == speaker_id):
        raise ShowCacheError(f"{context}: inconsistent speaker_id values")
    if not np.all((arrays["contact"] == 0.0) | (arrays["contact"] == 1.0)):
        raise ShowCacheError(f"{context}: contact must be binary")
    if not np.array_equal(
        arrays["contact"][-1], np.ones(4, dtype=np.float32)
    ):
        raise ShowCacheError(f"{context}: last-frame contact is not all ones")


def lineage_contract(
    *,
    split_root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    receipt_sha256: str,
    hand_component_path: Path,
    hand_component_sha256: str,
    smplx_asset_path: Path,
    smplx_sha256: str,
    expected_source_audio_rate: int,
    source: Mapping[str, str],
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "source_receipt": dict(source),
        "split_root": str(split_root),
        "split_manifest_sha256": receipt["split_sha256"],
        "split_receipt_path": str(receipt_path),
        "split_receipt_sha256": receipt_sha256,
        "split_counts": {name: int(receipt["counts"][name]) for name in SPLITS},
        "split_missing_count": int(receipt["missing_count"]),
        "hand_component_path": str(hand_component_path),
        "hand_component_sha256": hand_component_sha256,
        "smplx_asset_path": str(smplx_asset_path),
        "smplx_asset_sha256": smplx_sha256,
        "speaker_mapping": dict(SHOW_SPEAKER_ID),
        "pose_fps": POSE_FPS,
        "source_audio_sample_rate": int(expected_source_audio_rate),
        "hubert_target_sample_rate": HUBERT_AUDIO_SAMPLE_RATE,
        "audio_channel_protocol": {
            "accepted_source_channels": list(ACCEPTED_AUDIO_CHANNELS),
            "source_sample_width_bytes": SOURCE_AUDIO_SAMPLE_WIDTH,
            "source_compression": "NONE",
            "decode": "librosa.load(BytesIO(wav_payload),sr=None,mono=True)",
            "multichannel_mix": "arithmetic_mean_across_channels",
            "manifest_policy": AUDIO_MONO_POLICY,
        },
        "pose_protocol": (
            "root3+body63+jaw3+leye3+reye3+left_hand45+right_hand45"
        ),
        "hand_protocol": "TalkSHOW PCA12 @ canonical first-12 components -> AA45",
        "contact_protocol": {
            "smplx_joints": list(FOOT_JOINTS),
            "distance": "adjacent-frame Euclidean displacement",
            "threshold": CONTACT_THRESHOLD,
            "comparison": "strictly_less_than",
            "last_frame_speed": 0.0,
            "last_frame_contact": 1.0,
        },
        "global_foot_fastpath": {
            "contract": GLOBAL_FOOT_FASTPATH_CONTRACT,
            "sidecar_field": GLOBAL_FOOT_FIELD,
            "shape": ["frames", 4, 3],
            "dtype": "float32",
            "smplx_joints": list(FOOT_JOINTS),
            "selected_pose_joint_indices": list(LOWER_JOINT_INDICES),
            "pose_roundtrip": "AA->matrix->6D->matrix->AA",
            "non_lower_pose": 0.0,
            "translation": 0.0,
            "expression": 0.0,
        },
        "npz_fields": {
            "pose": ["frames", 165],
            "contact": ["frames", 4],
            "facial": ["frames", 100],
            "beta": ["frames", 300],
            "trans": ["frames", 3],
            "speaker_id": ["frames", 1],
        },
        "npz_encoding": (
            "deterministic ZIP_DEFLATED NumPy .npy members, fixed 1980 timestamp"
        ),
    }


def _manifest_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def build_shard(args: argparse.Namespace) -> dict[str, Any]:
    split_root = args.split_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    receipt, receipt_path, receipt_sha = load_split_receipt(
        split_root, args.expected_split_sha256
    )
    clips = scan_split_view(split_root, receipt)
    source = source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    selected = select_shard(clips, args.shard_id, args.num_shards)
    shard_name = f"shard-{args.shard_id:05d}-of-{args.num_shards:05d}"
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"output root is not a directory: {output_root}")
    for name in ("manifest.jsonl", "summary.json", "lineage.json"):
        if (output_root / name).exists():
            raise FileExistsError(
                f"refusing to append to finalized cache: {output_root / name}"
            )
    shard_root = output_root / "shards" / shard_name
    if shard_root.exists():
        raise FileExistsError(f"refusing to reuse existing shard output: {shard_root}")
    shard_root.mkdir(parents=True, exist_ok=False)

    hand_path = args.hand_component.expanduser().resolve()
    hand_sha = sha256_file(hand_path)
    if args.expected_hand_component_sha256 is not None:
        if hand_sha != args.expected_hand_component_sha256:
            raise ShowCacheError(
                f"hand component SHA mismatch: {hand_sha} != "
                f"{args.expected_hand_component_sha256}"
            )
    left_components, right_components = load_hand_components(hand_path)

    smplx_asset, smplx_root, gender = resolve_smplx_asset(args.smplx_model)
    smplx_sha = sha256_file(smplx_asset)
    if args.expected_smplx_sha256 is not None:
        if smplx_sha != args.expected_smplx_sha256:
            raise ShowCacheError(
                f"SMPL-X SHA mismatch: {smplx_sha} != "
                f"{args.expected_smplx_sha256}"
            )
    lineage = lineage_contract(
        split_root=split_root,
        receipt=receipt,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        hand_component_path=hand_path,
        hand_component_sha256=hand_sha,
        smplx_asset_path=smplx_asset,
        smplx_sha256=smplx_sha,
        expected_source_audio_rate=args.expected_source_audio_rate,
        source=source,
    )
    lineage_sha = canonical_json_sha256(lineage)
    contact_computer = ContactComputer(
        model_root=smplx_root,
        gender=gender,
        device=args.device,
        chunk_frames=args.smplx_chunk_frames,
    )
    runtime = contact_computer.runtime
    runtime_sha = canonical_json_sha256(runtime)

    rows: list[dict[str, Any]] = []
    frame_count = 0
    selected_counts: Counter[str] = Counter()
    for ordinal, clip in enumerate(selected, start=1):
        motion = load_canonical_motion(
            clip.pkl_path, left_components, right_components
        )
        contact, lower_foot_local = contact_computer(
            motion["pose"],
            motion["facial"],
            motion["beta"],
            motion["trans"],
        )
        frames = int(motion["pose"].shape[0])
        speaker_id = SHOW_SPEAKER_ID[clip.speaker]
        arrays = {
            "pose": motion["pose"],
            "contact": contact,
            "facial": motion["facial"],
            "beta": motion["beta"],
            "trans": motion["trans"],
            "speaker_id": np.full((frames, 1), speaker_id, dtype=np.int64),
        }
        validate_canonical_arrays(
            arrays,
            frames=frames,
            speaker_id=speaker_id,
            context=clip.clip_id,
        )
        wav_info = inspect_wav(
            clip.wav_path,
            args.expected_source_audio_rate,
        )
        pkl_sha = sha256_file(clip.pkl_path)
        wav_sha = sha256_file(clip.wav_path)
        npz_payload = deterministic_npz_bytes(arrays)
        npz_sha = hashlib.sha256(npz_payload).hexdigest()
        destination = output_root / clip.output_relative_path
        foot_destination = destination.with_name(
            f"{destination.stem}.{GLOBAL_FOOT_FIELD}.npy"
        )
        foot_payload = _npy_bytes(lower_foot_local)
        foot_sha = hashlib.sha256(foot_payload).hexdigest()
        atomic_write_new(destination, npz_payload)
        atomic_write_new(foot_destination, foot_payload)
        if sha256_file(destination) != npz_sha:
            raise ShowCacheError(f"post-write SHA mismatch: {destination}")
        if sha256_file(foot_destination) != foot_sha:
            raise ShowCacheError(f"post-write SHA mismatch: {foot_destination}")

        row = {
            "global_index": clip.global_index,
            "clip_id": clip.clip_id,
            "split": clip.split,
            "speaker": clip.speaker,
            "speaker_id": speaker_id,
            "video": clip.video,
            "sequence": clip.sequence,
            "source_pkl": str(clip.pkl_path.resolve()),
            "source_wav": str(clip.wav_path.resolve()),
            "canonical_npz": str(destination.resolve()),
            "canonical_npz_relative": clip.output_relative_path.as_posix(),
            "global_foot_fastpath_contract": GLOBAL_FOOT_FASTPATH_CONTRACT,
            GLOBAL_FOOT_FIELD: str(foot_destination.resolve()),
            f"{GLOBAL_FOOT_FIELD}_relative": (
                foot_destination.relative_to(output_root).as_posix()
            ),
            "frames": frames,
            "pose_fps": POSE_FPS,
            **wav_info,
            "source_pkl_sha256": pkl_sha,
            "source_wav_sha256": wav_sha,
            "canonical_npz_sha256": npz_sha,
            f"{GLOBAL_FOOT_FIELD}_sha256": foot_sha,
            "lineage_contract_sha256": lineage_sha,
        }
        rows.append(row)
        frame_count += frames
        selected_counts[clip.split] += 1
        if ordinal == 1 or ordinal % args.progress_every == 0 or ordinal == len(selected):
            print(
                f"[{shard_name}] {ordinal}/{len(selected)} "
                f"{clip.split}:{clip.clip_id} frames={frames}",
                flush=True,
            )

    rows.sort(key=lambda row: int(row["global_index"]))
    manifest_payload = _manifest_bytes(rows)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    summary = {
        "status": "complete",
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "shard_name": shard_name,
        "expected_global_clip_count": len(clips),
        "selected_clip_count": len(selected),
        "selected_frame_count": frame_count,
        "selected_split_counts": {
            name: int(selected_counts[name]) for name in SPLITS
        },
        "first_global_index": (
            int(rows[0]["global_index"]) if rows else None
        ),
        "last_global_index": (
            int(rows[-1]["global_index"]) if rows else None
        ),
        "manifest_sha256": manifest_sha,
        "lineage_contract_sha256": lineage_sha,
        "runtime_sha256": runtime_sha,
    }
    shard_lineage = {
        "lineage_contract": lineage,
        "lineage_contract_sha256": lineage_sha,
        "runtime": runtime,
        "runtime_sha256": runtime_sha,
        "shard": {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "selected_clip_count": len(selected),
            "manifest_sha256": manifest_sha,
        },
    }
    atomic_write_new(shard_root / "manifest.jsonl", manifest_payload)
    atomic_json_new(shard_root / "summary.json", summary)
    atomic_json_new(shard_root / "lineage.json", shard_lineage)
    return summary


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShowCacheError(f"cannot parse JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ShowCacheError(f"{path}:{line_number}: blank JSONL line")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ShowCacheError(
                        f"{path}:{line_number}: row is not an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ShowCacheError(f"cannot parse JSONL {path}: {exc}") from exc
    return rows


def _verify_npz(path: Path, row: Mapping[str, Any]) -> None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if tuple(payload.files) != NPZ_FIELDS:
                raise ShowCacheError(
                    f"{path}: fields {tuple(payload.files)} != {NPZ_FIELDS}"
                )
            arrays = {name: payload[name] for name in NPZ_FIELDS}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ShowCacheError(f"cannot load canonical NPZ {path}: {exc}") from exc
    validate_canonical_arrays(
        arrays,
        frames=int(row["frames"]),
        speaker_id=int(row["speaker_id"]),
        context=str(path),
    )


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    split_root = args.split_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for name in ("manifest.jsonl", "summary.json", "lineage.json"):
        if (output_root / name).exists():
            raise FileExistsError(
                f"refusing to overwrite finalized cache artifact: {output_root / name}"
            )
    receipt, receipt_path, receipt_sha = load_split_receipt(
        split_root, args.expected_split_sha256
    )
    clips = scan_split_view(split_root, receipt)
    source = source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    expected_by_index = {clip.global_index: clip for clip in clips}

    hand_path = args.hand_component.expanduser().resolve()
    hand_sha = sha256_file(hand_path)
    if (
        args.expected_hand_component_sha256 is not None
        and hand_sha != args.expected_hand_component_sha256
    ):
        raise ShowCacheError("hand component SHA mismatch during finalize")
    smplx_asset, _, _ = resolve_smplx_asset(args.smplx_model)
    smplx_sha = sha256_file(smplx_asset)
    if (
        args.expected_smplx_sha256 is not None
        and smplx_sha != args.expected_smplx_sha256
    ):
        raise ShowCacheError("SMPL-X SHA mismatch during finalize")
    lineage = lineage_contract(
        split_root=split_root,
        receipt=receipt,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        hand_component_path=hand_path,
        hand_component_sha256=hand_sha,
        smplx_asset_path=smplx_asset,
        smplx_sha256=smplx_sha,
        expected_source_audio_rate=args.expected_source_audio_rate,
        source=source,
    )
    lineage_sha = canonical_json_sha256(lineage)

    rows_by_index: dict[int, dict[str, Any]] = {}
    shard_receipts = []
    build_runtime: dict[str, Any] | None = None
    build_runtime_sha: str | None = None
    for shard_id in range(args.num_shards):
        shard_name = f"shard-{shard_id:05d}-of-{args.num_shards:05d}"
        shard_root = output_root / "shards" / shard_name
        manifest_path = shard_root / "manifest.jsonl"
        summary_path = shard_root / "summary.json"
        lineage_path = shard_root / "lineage.json"
        if not all(path.is_file() for path in (manifest_path, summary_path, lineage_path)):
            raise ShowCacheError(f"incomplete shard receipt directory: {shard_root}")
        summary = _load_json(summary_path)
        shard_lineage = _load_json(lineage_path)
        rows = _load_jsonl(manifest_path)
        manifest_sha = sha256_file(manifest_path)
        if summary.get("status") != "complete":
            raise ShowCacheError(f"{summary_path}: shard is not complete")
        if summary.get("shard_id") != shard_id:
            raise ShowCacheError(f"{summary_path}: shard_id mismatch")
        if summary.get("num_shards") != args.num_shards:
            raise ShowCacheError(f"{summary_path}: num_shards mismatch")
        if summary.get("manifest_sha256") != manifest_sha:
            raise ShowCacheError(f"{summary_path}: manifest SHA mismatch")
        if summary.get("selected_clip_count") != len(rows):
            raise ShowCacheError(f"{summary_path}: row count mismatch")
        if summary.get("lineage_contract_sha256") != lineage_sha:
            raise ShowCacheError(f"{summary_path}: lineage contract mismatch")
        if shard_lineage.get("lineage_contract_sha256") != lineage_sha:
            raise ShowCacheError(f"{lineage_path}: lineage contract mismatch")
        if shard_lineage.get("lineage_contract") != lineage:
            raise ShowCacheError(f"{lineage_path}: lineage payload mismatch")
        shard_runtime = shard_lineage.get("runtime")
        shard_runtime_sha = shard_lineage.get("runtime_sha256")
        if not isinstance(shard_runtime, dict):
            raise ShowCacheError(f"{lineage_path}: missing build runtime")
        if shard_runtime_sha != canonical_json_sha256(shard_runtime):
            raise ShowCacheError(f"{lineage_path}: build runtime SHA mismatch")
        if summary.get("runtime_sha256") != shard_runtime_sha:
            raise ShowCacheError(f"{summary_path}: build runtime SHA mismatch")
        if build_runtime is None:
            build_runtime = shard_runtime
            build_runtime_sha = shard_runtime_sha
        elif shard_runtime != build_runtime:
            raise ShowCacheError(
                f"{lineage_path}: runtime differs across cache shards"
            )
        shard_receipts.append(
            {
                "shard_id": shard_id,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "lineage_path": str(lineage_path),
                "lineage_sha256": sha256_file(lineage_path),
                "clip_count": len(rows),
            }
        )
        for row in rows:
            index = row.get("global_index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ShowCacheError(f"{manifest_path}: invalid global_index {index!r}")
            if index % args.num_shards != shard_id:
                raise ShowCacheError(
                    f"{manifest_path}: global_index {index} belongs to another shard"
                )
            if index in rows_by_index:
                raise ShowCacheError(f"duplicate global_index across shards: {index}")
            rows_by_index[index] = row

    if set(rows_by_index) != set(expected_by_index):
        missing = sorted(set(expected_by_index) - set(rows_by_index))
        extra = sorted(set(rows_by_index) - set(expected_by_index))
        raise ShowCacheError(
            f"shards do not cover the frozen manifest exactly once: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    final_rows = []
    split_counts: Counter[str] = Counter()
    frame_count = 0
    required_row_fields = {
        "clip_id",
        "split",
        "speaker",
        "speaker_id",
        "source_pkl",
        "source_wav",
        "canonical_npz",
        "frames",
        "source_pkl_sha256",
        "source_wav_sha256",
        "canonical_npz_sha256",
        "wav_channels",
        "wav_sample_width",
        "wav_sample_rate",
        "wav_frames",
        "wav_mono_policy",
        "global_foot_fastpath_contract",
        GLOBAL_FOOT_FIELD,
        f"{GLOBAL_FOOT_FIELD}_sha256",
    }
    for ordinal, index in enumerate(sorted(rows_by_index), start=1):
        row = rows_by_index[index]
        clip = expected_by_index[index]
        missing_fields = required_row_fields - set(row)
        if missing_fields:
            raise ShowCacheError(f"manifest index {index} misses {missing_fields}")
        expected_values = {
            "clip_id": clip.clip_id,
            "split": clip.split,
            "speaker": clip.speaker,
            "speaker_id": SHOW_SPEAKER_ID[clip.speaker],
            "source_pkl": str(clip.pkl_path.resolve()),
            "source_wav": str(clip.wav_path.resolve()),
            "canonical_npz": str(
                (output_root / clip.output_relative_path).resolve()
            ),
            "canonical_npz_relative": clip.output_relative_path.as_posix(),
            "global_foot_fastpath_contract": GLOBAL_FOOT_FASTPATH_CONTRACT,
            GLOBAL_FOOT_FIELD: str(
                (
                    output_root
                    / clip.output_relative_path
                ).with_name(
                    f"{clip.output_relative_path.stem}.{GLOBAL_FOOT_FIELD}.npy"
                ).resolve()
            ),
            f"{GLOBAL_FOOT_FIELD}_relative": (
                clip.output_relative_path.with_name(
                    f"{clip.output_relative_path.stem}.{GLOBAL_FOOT_FIELD}.npy"
                ).as_posix()
            ),
            "lineage_contract_sha256": lineage_sha,
        }
        for key, expected in expected_values.items():
            if row.get(key) != expected:
                raise ShowCacheError(
                    f"manifest index {index} {key} mismatch: "
                    f"{row.get(key)!r} != {expected!r}"
                )
        source_pkl = Path(row["source_pkl"])
        source_wav = Path(row["source_wav"])
        canonical_npz = Path(row["canonical_npz"])
        lower_foot_path = Path(row[GLOBAL_FOOT_FIELD])
        observed_hashes = {
            "source_pkl_sha256": sha256_file(source_pkl),
            "source_wav_sha256": sha256_file(source_wav),
            "canonical_npz_sha256": sha256_file(canonical_npz),
            f"{GLOBAL_FOOT_FIELD}_sha256": sha256_file(lower_foot_path),
        }
        for key, observed in observed_hashes.items():
            if row.get(key) != observed:
                raise ShowCacheError(
                    f"manifest index {index} {key} mismatch: "
                    f"{row.get(key)!r} != {observed!r}"
                )
        observed_wav_info = inspect_wav(
            source_wav,
            args.expected_source_audio_rate,
        )
        validate_wav_manifest_metadata(
            row,
            observed_wav_info,
            f"manifest index {index}",
        )
        _verify_npz(canonical_npz, row)
        try:
            lower_foot_local = np.load(lower_foot_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ShowCacheError(
                f"cannot load Global foot sidecar {lower_foot_path}: {exc}"
            ) from exc
        expected_foot_shape = (int(row["frames"]), 4, 3)
        if (
            lower_foot_local.shape != expected_foot_shape
            or lower_foot_local.dtype != np.float32
            or not np.isfinite(lower_foot_local).all()
        ):
            raise ShowCacheError(
                f"{lower_foot_path}: invalid Global foot cache "
                f"{lower_foot_local.shape}/{lower_foot_local.dtype}"
            )
        final_rows.append(row)
        split_counts[clip.split] += 1
        frame_count += int(row["frames"])
        if ordinal == 1 or ordinal % args.progress_every == 0 or ordinal == len(clips):
            print(
                f"[finalize] {ordinal}/{len(clips)} "
                f"{clip.split}:{clip.clip_id}",
                flush=True,
            )

    for split in SPLITS:
        if split_counts[split] != receipt["counts"][split]:
            raise ShowCacheError(
                f"final {split} count mismatch: {split_counts[split]} != "
                f"{receipt['counts'][split]}"
            )
    manifest_payload = _manifest_bytes(final_rows)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    final_lineage = {
        "lineage_contract": lineage,
        "lineage_contract_sha256": lineage_sha,
        "build_runtime": build_runtime,
        "build_runtime_sha256": build_runtime_sha,
        "shards": shard_receipts,
        "final_manifest_sha256": manifest_sha,
    }
    final_summary = {
        "status": "complete",
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "clip_count": len(final_rows),
        "frame_count": frame_count,
        "split_counts": {name: int(split_counts[name]) for name in SPLITS},
        "num_shards": args.num_shards,
        "manifest_sha256": manifest_sha,
        "lineage_contract_sha256": lineage_sha,
        "source_receipt_sha256": canonical_json_sha256(source),
        "build_runtime_sha256": build_runtime_sha,
        "lineage_sha256": canonical_json_sha256(final_lineage),
        "finite": True,
        "exact_once": True,
        "split_disjoint": True,
    }
    atomic_write_new(output_root / "manifest.jsonl", manifest_payload)
    atomic_json_new(output_root / "lineage.json", final_lineage)
    atomic_json_new(output_root / "summary.json", final_summary)
    return final_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hand-component", type=Path, required=True)
    parser.add_argument(
        "--smplx-model",
        type=Path,
        required=True,
        help=(
            "SMPL-X asset file or model root containing "
            "smplx/SMPLX_NEUTRAL.npz or SMPLX_NEUTRAL_2020.npz"
        ),
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smplx-chunk-frames", type=int, default=128)
    parser.add_argument(
        "--expected-source-audio-rate",
        type=int,
        default=SOURCE_AUDIO_SAMPLE_RATE,
    )
    parser.add_argument(
        "--expected-split-sha256",
        default=OFFICIAL_SPLIT_SHA256,
    )
    parser.add_argument("--expected-hand-component-sha256", required=True)
    parser.add_argument("--expected-smplx-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="verify and merge already-complete shard receipts",
    )
    args = parser.parse_args(argv)
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, --num-shards)")
    if args.smplx_chunk_frames < 1:
        parser.error("--smplx-chunk-frames must be positive")
    if args.expected_source_audio_rate < 1:
        parser.error("--expected-source-audio-rate must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    args.expected_split_sha256 = _require_sha256(
        args.expected_split_sha256, "--expected-split-sha256"
    )
    args.expected_hand_component_sha256 = _require_sha256(
        args.expected_hand_component_sha256,
        "--expected-hand-component-sha256",
    )
    args.expected_smplx_sha256 = _require_sha256(
        args.expected_smplx_sha256, "--expected-smplx-sha256"
    )
    args.expected_source_commit = _require_git_oid(
        args.expected_source_commit,
        "--expected-source-commit",
    )
    args.expected_source_tree = _require_git_oid(
        args.expected_source_tree,
        "--expected-source-tree",
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = finalize(args) if args.finalize else build_shard(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
