#!/usr/bin/env python3
"""Official All-Speakers initialized transfer training for canonical SHOW.

This is deliberately separate from ``show_base_train.py``.  It implements two
and only two adaptation stages:

``face``
    Cache frozen official RVQ code IDs/zq, then train the Face decoder only
    with jaw SO(3), expression, velocity, and acceleration losses.

``global``
    Cache the actual frozen-official-Lower decoded interface, forcing Lower
    channels 54:57 to exact zero and using GT contact only in the four observed
    prefix frames.  Train the official Global model on root-channel and
    shared-anchor integrated-root objectives, without SMPL-X.

Cache construction is clip-shardable.  Training supports torchrun DDP.  Test
clips are rejected by construction and must only be used by the separately
locked final inference/evaluation path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.show_official_transfer import (  # noqa: E402
    CACHE_FORMAT,
    CODEBOOK_SIZE,
    EXPECTED_ORIGIN,
    FPS,
    OFFICIAL_CHECKPOINTS,
    PRE_FRAMES,
    RVQ_LEVELS,
    TRANSFER_FORMAT,
    WINDOW_LENGTH,
    WINDOW_STRIDE,
    canonical_json_bytes,
    canonical_payload_sha256,
    compose_global_input,
    configure_global_transfer_policy,
    face_task_losses,
    freeze_face_for_decoder_transfer,
    frozen_face_state,
    frozen_global_state,
    global_root_losses,
    load_official_model_state,
    reject_e30_reference,
    require_sha256,
    root_channels_from_translation,
    sha256_file,
    state_dict_sha256,
    validate_loss_weights,
    verify_official_checkpoint,
)


TRAINING_SPLITS = ("train", "val")
EXPECTED_SPLIT_COUNTS = {"train": 13_687, "val": 1_715, "test": 1_708}
FACE_ARRAYS = {
    "code_ids": ((WINDOW_LENGTH // 4, RVQ_LEVELS), "uint16"),
    "zq": ((256, WINDOW_LENGTH // 4), "float32"),
    "target": ((WINDOW_LENGTH, 106), "float32"),
}
GLOBAL_ARRAYS = {
    "global_input": ((WINDOW_LENGTH, 61), "float32"),
    "target_root": ((WINDOW_LENGTH, 3), "float32"),
    "target_translation": ((WINDOW_LENGTH, 3), "float32"),
    "anchor": ((3,), "float32"),
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_source_receipt(expected_commit: str, expected_tree: str) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    branch_result = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if branch_result.returncode not in {0, 1}:
        raise RuntimeError("cannot prove official transfer source branch state")
    actual = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": (
            branch_result.stdout.strip()
            if branch_result.returncode == 0
            else None
        ),
        "clean": not bool(
            git("status", "--porcelain=v1", "--untracked-files=all")
        ),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
    }
    if actual["origin"] != EXPECTED_ORIGIN:
        raise RuntimeError(
            f"transfer source origin {actual['origin']!r} != {EXPECTED_ORIGIN!r}"
        )
    if actual["commit"] != expected_commit or actual["tree"] != expected_tree:
        raise RuntimeError(
            "transfer source commit/tree mismatch: "
            f"{actual['commit']}/{actual['tree']} != "
            f"{expected_commit}/{expected_tree}"
        )
    if not actual["clean"]:
        raise RuntimeError("official transfer requires a clean frozen source checkout")
    if actual["branch"] is not None:
        raise RuntimeError(
            "official transfer requires a detached frozen source checkout"
        )
    return actual


def _load_canonical_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Reuse the already-audited canonical receipt contract, not its measurement
    # or decision logic.
    from scripts.show_base.gate_released_all_speakers_on_show import (
        load_canonical_receipt,
    )

    rows, receipt = load_canonical_receipt(
        manifest_path=args.canonical_manifest,
        summary_path=args.canonical_summary,
        lineage_path=args.canonical_lineage,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_summary_sha256=args.expected_summary_sha256,
        expected_lineage_sha256=args.expected_lineage_sha256,
        expected_canonical_commit=args.expected_canonical_commit,
        expected_canonical_tree=args.expected_canonical_tree,
    )
    held_in = [row for row in rows if row["split"] in TRAINING_SPLITS]
    counts = Counter(row["split"] for row in held_in)
    expected = Counter(
        {split: EXPECTED_SPLIT_COUNTS[split] for split in TRAINING_SPLITS}
    )
    if counts != expected:
        raise RuntimeError(f"train/val canonical coverage {counts} != {expected}")
    if any(row["split"] == "test" for row in held_in):
        raise RuntimeError("test leakage into transfer cache is forbidden")
    return held_in, receipt


def _load_canonical_npz(np: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(row["canonical_npz"])
    if path.is_symlink():
        raise RuntimeError(f"canonical NPZ must not be a symlink: {path}")
    resolved = path.resolve()
    payload = resolved.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != row["canonical_npz_sha256"]:
        raise RuntimeError(
            f"canonical NPZ SHA mismatch for {resolved}: "
            f"{actual} != {row['canonical_npz_sha256']}"
        )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        expected_fields = {
            "pose",
            "contact",
            "facial",
            "beta",
            "trans",
            "speaker_id",
        }
        if set(archive.files) != expected_fields:
            raise RuntimeError(f"canonical NPZ field mismatch: {resolved}")
        arrays = {name: archive[name] for name in archive.files}
    frames = int(row["frames"])
    expected = {
        "pose": ((frames, 165), np.float32),
        "contact": ((frames, 4), np.float32),
        "facial": ((frames, 100), np.float32),
        "beta": ((frames, 300), np.float32),
        "trans": ((frames, 3), np.float32),
        "speaker_id": ((frames, 1), np.int64),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != dtype:
            raise RuntimeError(
                f"canonical {name} schema mismatch: "
                f"{array.shape}/{array.dtype} != {shape}/{dtype}"
            )
        if name != "speaker_id" and not bool(np.isfinite(array).all()):
            raise RuntimeError(f"canonical {name} contains NaN/Inf: {resolved}")
    return arrays


def _window_count(frames: int) -> int:
    usable = (int(frames) // FPS) * FPS
    return max(0, (usable - WINDOW_LENGTH) // WINDOW_STRIDE + 1)


def _window_batches(
    np: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    fields: dict[str, list[Any]] = {
        "pose": [],
        "contact": [],
        "facial": [],
        "trans": [],
    }
    for row in sorted(rows, key=lambda item: int(item["global_index"])):
        arrays = _load_canonical_npz(np, row)
        for window_index in range(_window_count(int(row["frames"]))):
            start = window_index * WINDOW_STRIDE
            end = start + WINDOW_LENGTH
            records.append(
                {
                    "split": row["split"],
                    "global_index": int(row["global_index"]),
                    "clip_id": row["clip_id"],
                    "start": start,
                    "end": end,
                    "canonical_npz_sha256": row["canonical_npz_sha256"],
                }
            )
            for name in fields:
                fields[name].append(arrays[name][start:end])
            if len(records) == batch_size:
                yield records, {
                    name: np.stack(values)
                    for name, values in fields.items()
                }
                records = []
                fields = {name: [] for name in fields}
    if records:
        yield records, {
            name: np.stack(values)
            for name, values in fields.items()
        }


def _build_features(torch: Any, batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    from utils import rotation_conversions as rc

    pose = torch.from_numpy(batch["pose"]).to(device)
    contact = torch.from_numpy(batch["contact"]).to(device)
    facial = torch.from_numpy(batch["facial"]).to(device)
    translation = torch.from_numpy(batch["trans"]).to(device)
    batch_size, frames, _ = pose.shape
    joints = pose.reshape(batch_size, frames, 55, 3)
    jaw = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(joints[:, :, 22])
    )
    lower_joint_indices = (0, 1, 2, 4, 5, 7, 8, 10, 11)
    lower_rotation = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(joints[:, :, lower_joint_indices])
    ).reshape(batch_size, frames, 54)
    return {
        "face": torch.cat([jaw, facial], dim=-1),
        "lower": torch.cat(
            [lower_rotation, torch.zeros_like(translation), contact],
            dim=-1,
        ),
        "translation": translation,
        "contact": contact,
    }


def _encode_npz(np: Any, arrays: Mapping[str, Any]) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _validate_array_payload(np: Any, stage: str, payload: bytes) -> dict[str, Any]:
    spec = FACE_ARRAYS if stage == "face" else GLOBAL_ARRAYS
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(spec):
            raise RuntimeError(f"{stage} cache array fields mismatch")
        arrays = {name: archive[name] for name in archive.files}
    for name, (shape, dtype_name) in spec.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype(dtype_name):
            raise RuntimeError(
                f"{stage} cache {name} schema "
                f"{value.shape}/{value.dtype} != {shape}/{dtype_name}"
            )
        if np.issubdtype(value.dtype, np.floating) and not bool(
            np.isfinite(value).all()
        ):
            raise RuntimeError(f"{stage} cache {name} contains NaN/Inf")
    if stage == "face":
        code_ids = arrays["code_ids"]
        if int(code_ids.max()) >= CODEBOOK_SIZE:
            raise RuntimeError("face cache code ID out of range")
    else:
        if not bool(np.array_equal(
            arrays["global_input"][..., 54:57],
            np.zeros_like(arrays["global_input"][..., 54:57]),
        )):
            raise RuntimeError("Global cache lower translation is not exact zero")
    return arrays


def _torch_model_args(stage: str) -> SimpleNamespace:
    if stage == "face":
        return SimpleNamespace(vae_test_dim=106, vae_layer=2, vae_length=256)
    if stage in {"lower", "global"}:
        return SimpleNamespace(vae_test_dim=61, vae_layer=4, vae_length=256)
    raise ValueError(stage)


def _load_model(torch: Any, stage: str, checkpoint: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    if stage in {"face", "lower"}:
        from models.rvq import RVQVAE

        model = RVQVAE(_torch_model_args(stage))
    elif stage == "global":
        from models.motion_representation import VAEConvZero

        model = VAEConvZero(_torch_model_args(stage))
    else:
        raise ValueError(stage)
    receipt = load_official_model_state(
        torch,
        stage=stage,
        checkpoint=checkpoint,
        model=model,
    )
    return model.to(device), receipt


def _configure_determinism(torch: Any, np: Any, *, seed: int, device: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _build_cache(args: argparse.Namespace) -> int:
    import lmdb
    import numpy as np
    import torch

    if args.stage not in {"face", "global"}:
        raise ValueError("cache stage must be face or global")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite cache root {args.output_root}")
    source_receipt = _git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    rows, canonical_receipt = _load_canonical_rows(args)
    rows = [
        row
        for row in rows
        if int(row["global_index"]) % args.shard_count == args.shard_index
    ]
    if not rows:
        raise RuntimeError("cache shard has no canonical clips")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    _configure_determinism(torch, np, seed=args.seed, device=device)

    if args.stage == "face":
        model, checkpoint_receipt = _load_model(
            torch,
            "face",
            args.official_face_checkpoint,
            device,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        lower_model = None
        lower_receipt = None
    else:
        lower_model, lower_receipt = _load_model(
            torch,
            "lower",
            args.official_lower_checkpoint,
            device,
        )
        lower_model.eval()
        for parameter in lower_model.parameters():
            parameter.requires_grad_(False)
        model = None
        checkpoint_receipt = lower_receipt

    args.output_root.mkdir(parents=True, exist_ok=False)
    lmdb_path = args.output_root / "data.lmdb"
    environment = lmdb.open(
        str(lmdb_path),
        map_size=int(args.map_size_gb * (1024 ** 3)),
        subdir=False,
        lock=True,
        sync=True,
        metasync=True,
        map_async=False,
    )
    records_path = args.output_root / "records.jsonl"
    key_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    code_digest = hashlib.sha256()
    zq_digest = hashlib.sha256()
    split_counts: Counter[str] = Counter()
    clip_counts: Counter[str] = Counter()
    record_count = 0
    try:
        with records_path.open("w", encoding="utf-8") as records_file:
            with torch.inference_mode():
                for records, raw_batch in _window_batches(
                    np,
                    rows,
                    batch_size=args.batch_size,
                ):
                    features = _build_features(torch, raw_batch, device)
                    arrays_per_record: list[dict[str, Any]] = []
                    if args.stage == "face":
                        target = features["face"]
                        code_ids = model.map2index(target)
                        expected_shape = (
                            target.shape[0],
                            WINDOW_LENGTH // 4,
                            RVQ_LEVELS,
                        )
                        if (
                            tuple(code_ids.shape) != expected_shape
                            or code_ids.dtype != torch.long
                            or int(code_ids.min().item()) < 0
                            or int(code_ids.max().item()) >= CODEBOOK_SIZE
                        ):
                            raise RuntimeError("official Face code ID contract failed")
                        zq = model.quantizer.get_codebook_entry(code_ids)
                        replay_zq = model.quantizer.get_codebook_entry(code_ids)
                        if not torch.equal(zq, replay_zq):
                            raise RuntimeError("Face code ID -> zq replay is not exact")
                        for index in range(target.shape[0]):
                            arrays_per_record.append(
                                {
                                    "code_ids": code_ids[index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.uint16, copy=False),
                                    "zq": zq[index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "target": target[index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                }
                            )
                    else:
                        lower = features["lower"]
                        code_ids = lower_model.map2index(lower)
                        decoded = lower_model.decode(code_ids)
                        global_input = compose_global_input(
                            torch,
                            decoded,
                            features["contact"],
                            pre_frames=PRE_FRAMES,
                        )
                        target_root = root_channels_from_translation(
                            torch,
                            features["translation"],
                        )
                        for index in range(lower.shape[0]):
                            arrays_per_record.append(
                                {
                                    "global_input": global_input[index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "target_root": target_root[index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "target_translation": features["translation"][index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                    "anchor": features["translation"][index, 0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False),
                                }
                            )

                    with environment.begin(write=True) as transaction:
                        for record, arrays in zip(records, arrays_per_record):
                            key = (
                                f"{record['split']}:{record['global_index']:05d}:"
                                f"{record['start']:07d}"
                            ).encode("ascii")
                            payload = _encode_npz(np, arrays)
                            _validate_array_payload(np, args.stage, payload)
                            if not transaction.put(key, payload, overwrite=False):
                                raise RuntimeError(f"duplicate cache key {key!r}")
                            payload_sha = hashlib.sha256(payload).hexdigest()
                            row = {
                                **record,
                                "key": key.decode("ascii"),
                                "payload_sha256": payload_sha,
                            }
                            records_file.write(
                                json.dumps(
                                    row,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                            key_digest.update(key)
                            key_digest.update(b"\n")
                            payload_digest.update(bytes.fromhex(payload_sha))
                            if args.stage == "face":
                                code_digest.update(
                                    arrays["code_ids"].tobytes(order="C")
                                )
                                zq_digest.update(arrays["zq"].tobytes(order="C"))
                            split_counts[record["split"]] += 1
                            clip_counts[record["split"]] += int(
                                record["start"] == 0
                            )
                            record_count += 1
            records_file.flush()
            os.fsync(records_file.fileno())
        environment.sync(True)
    finally:
        environment.close()

    summary = {
        "format": CACHE_FORMAT,
        "status": "complete",
        "stage": args.stage,
        "test_visible": False,
        "splits": list(TRAINING_SPLITS),
        "shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "assignment": "canonical_global_index_modulo_shard_count",
        },
        "coverage": {
            "record_count": record_count,
            "split_window_counts": dict(sorted(split_counts.items())),
            "split_clip_counts": dict(sorted(clip_counts.items())),
            "key_order_sha256": key_digest.hexdigest(),
            "payload_order_sha256": payload_digest.hexdigest(),
            "records_jsonl_sha256": sha256_file(records_path),
            "exact_once_within_shard": True,
            "finite": True,
        },
        "source_receipt": source_receipt,
        "canonical_receipt": canonical_receipt,
        "official_initialization": {
            "face": checkpoint_receipt if args.stage == "face" else None,
            "lower": lower_receipt if args.stage == "global" else None,
            "withdrawn_e30_allowed": False,
        },
        "protocol": {
            "window_length": WINDOW_LENGTH,
            "window_stride": WINDOW_STRIDE,
            "fps": FPS,
            "face": (
                {
                    "encoder": "frozen official",
                    "quantizer": "frozen official eval/no EMA update",
                    "stored": ["exact code_ids", "exact float32 zq", "target"],
                    "code_id_order_sha256": code_digest.hexdigest(),
                    "zq_order_sha256": zq_digest.hexdigest(),
                }
                if args.stage == "face"
                else None
            ),
            "global": (
                {
                    "cache_role": (
                        "teacher_forced_training_interface_only: canonical GT "
                        "lower motion -> frozen official Lower code IDs -> decode"
                    ),
                    "locked_inference_role": (
                        "Base-predicted official Lower code IDs -> frozen "
                        "official Lower decode; this cache is never an "
                        "inference substitute"
                    ),
                    "lower": "frozen official map2index -> decode",
                    "rotation": "decoded then projected through SO(3)",
                    "lower_translation_54_57": "exact zero for every frame",
                    "contact_0_4": "canonical observed contact",
                    "contact_4_64": "decoded lower contact",
                    "world_root_in_lower_seed": False,
                    "targets": "root channels plus shared-anchor integrated root",
                }
                if args.stage == "global"
                else None
            ),
        },
    }
    summary["receipt_sha256"] = canonical_payload_sha256(summary)
    _atomic_json(args.output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "records": record_count,
                "summary": str(args.output_root / "summary.json"),
                "receipt_sha256": summary["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_receipt_hash(payload: Mapping[str, Any], label: str) -> None:
    expected = require_sha256(payload.get("receipt_sha256"), f"{label} receipt SHA")
    unhashed = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if canonical_payload_sha256(unhashed) != expected:
        raise RuntimeError(f"{label} receipt payload hash mismatch")


def _read_cache_roots(
    roots: Sequence[Path],
    *,
    stage: str,
    official_receipts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not roots:
        raise ValueError("at least one --cache-root is required")
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    shard_count: int | None = None
    seen_shards: set[int] = set()
    seen_keys: set[str] = set()
    canonical_binding: dict[str, Any] | None = None
    for raw_root in roots:
        reject_e30_reference(raw_root, "transfer cache root")
        root = raw_root.resolve()
        summary_path = root / "summary.json"
        records_path = root / "records.jsonl"
        lmdb_path = root / "data.lmdb"
        for path in (summary_path, records_path, lmdb_path):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"invalid transfer cache artifact {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise TypeError(f"cache summary must be an object: {summary_path}")
        _verify_receipt_hash(summary, f"cache {root}")
        if (
            summary.get("format") != CACHE_FORMAT
            or summary.get("status") != "complete"
            or summary.get("stage") != stage
            or summary.get("test_visible") is not False
            or summary.get("splits") != list(TRAINING_SPLITS)
        ):
            raise RuntimeError(f"cache summary contract mismatch: {summary_path}")
        if (
            summary.get("coverage", {}).get("records_jsonl_sha256")
            != sha256_file(records_path)
        ):
            raise RuntimeError(f"cache records SHA mismatch: {records_path}")
        official = summary.get("official_initialization")
        if not isinstance(official, dict) or official.get(
            "withdrawn_e30_allowed"
        ) is not False:
            raise RuntimeError("cache does not explicitly forbid e30")
        if stage == "face":
            if official.get("face", {}).get("sha256") != official_receipts[
                "face"
            ]["sha256"]:
                raise RuntimeError("Face cache official checkpoint binding mismatch")
        else:
            if official.get("lower", {}).get("sha256") != official_receipts[
                "lower"
            ]["sha256"]:
                raise RuntimeError("Global cache official Lower binding mismatch")
        canonical = summary.get("canonical_receipt")
        if canonical_binding is None:
            canonical_binding = canonical
        elif canonical_binding != canonical:
            raise RuntimeError("cache shards disagree on canonical receipt")

        shard = summary.get("shard")
        if not isinstance(shard, dict):
            raise RuntimeError("cache shard receipt is missing")
        index = shard.get("index")
        count = shard.get("count")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not 0 <= index < count
        ):
            raise RuntimeError("invalid cache shard index/count")
        if shard_count is None:
            shard_count = count
        elif shard_count != count:
            raise RuntimeError("cache shards disagree on shard_count")
        if index in seen_shards:
            raise RuntimeError(f"duplicate cache shard {index}")
        seen_shards.add(index)

        observed = 0
        with records_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    not isinstance(row, dict)
                    or row.get("split") not in TRAINING_SPLITS
                    or not isinstance(row.get("key"), str)
                ):
                    raise RuntimeError(
                        f"invalid cache record {records_path}:{line_number}"
                    )
                # LMDB keys are the stage-wide sample identity.  Including the
                # shard root here would let the same sample appear once in
                # every shard while still passing the exact-once audit.
                identifier = row["key"]
                if identifier in seen_keys:
                    raise RuntimeError(f"duplicate cache record {identifier!r}")
                seen_keys.add(identifier)
                require_sha256(
                    row.get("payload_sha256"),
                    f"{records_path}:{line_number} payload SHA",
                )
                records.append(
                    {
                        **row,
                        "root": str(root),
                        "lmdb": str(lmdb_path),
                    }
                )
                observed += 1
        if observed != summary.get("coverage", {}).get("record_count"):
            raise RuntimeError(f"cache record count mismatch: {root}")
        summaries.append(summary)
    assert shard_count is not None
    if seen_shards != set(range(shard_count)):
        raise RuntimeError(
            f"cache shard coverage {sorted(seen_shards)} "
            f"!= {list(range(shard_count))}"
        )
    if not records or canonical_binding is None:
        raise RuntimeError("transfer cache is empty")
    return records, {
        "format": CACHE_FORMAT,
        "stage": stage,
        "roots": [str(path.resolve()) for path in roots],
        "summary_receipts": [summary["receipt_sha256"] for summary in summaries],
        "shard_count": shard_count,
        "shards": sorted(seen_shards),
        "record_count": len(records),
        "canonical_receipt": canonical_binding,
    }


class TransferCacheDataset:
    """LMDB-backed dataset which opens readers lazily per DataLoader process."""

    def __init__(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        stage: str,
        split: str,
        verify_payloads: bool,
    ) -> None:
        if split not in TRAINING_SPLITS:
            raise ValueError(f"unsupported split {split!r}")
        self.stage = stage
        self.verify_payloads = verify_payloads
        self.records = [
            dict(record) for record in records if record["split"] == split
        ]
        if not self.records:
            raise RuntimeError(f"cache has no records for split {split}")
        self._environments: dict[tuple[int, str], Any] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _environment(self, path: str) -> Any:
        import lmdb

        key = (os.getpid(), path)
        environment = self._environments.get(key)
        if environment is None:
            environment = lmdb.open(
                path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048,
            )
            self._environments[key] = environment
        return environment

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        record = self.records[index]
        environment = self._environment(record["lmdb"])
        with environment.begin(write=False, buffers=False) as transaction:
            payload = transaction.get(record["key"].encode("ascii"))
        if payload is None:
            raise RuntimeError(f"missing transfer cache key {record['key']!r}")
        if self.verify_payloads:
            actual = hashlib.sha256(payload).hexdigest()
            if actual != record["payload_sha256"]:
                raise RuntimeError(
                    f"cache payload SHA mismatch for {record['key']!r}"
                )
        arrays = _validate_array_payload(np, self.stage, payload)
        result = {
            name: torch.from_numpy(value.copy())
            for name, value in arrays.items()
        }
        result["cache_key"] = record["key"]
        return result


def _distributed_context(torch: Any, requested_device: str) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise RuntimeError("invalid torchrun WORLD_SIZE/RANK")
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP official transfer requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(device)
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
    }


def _barrier(torch: Any, context: Mapping[str, Any]) -> None:
    if context["world_size"] > 1:
        torch.distributed.barrier()


def _rank_indices(
    length: int,
    *,
    rank: int,
    world_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    equal: bool,
) -> list[int]:
    import numpy as np

    indices = np.arange(length, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed + epoch).shuffle(indices)
    if equal:
        usable = (length // world_size) * world_size
        if usable <= 0:
            raise RuntimeError("dataset is smaller than DDP world size")
        indices = indices[:usable]
    return indices[rank::world_size].tolist()


def _loader(
    torch: Any,
    dataset: Any,
    *,
    context: Mapping[str, Any],
    seed: int,
    epoch: int,
    batch_size: int,
    workers: int,
    training: bool,
) -> Any:
    indices = _rank_indices(
        len(dataset),
        rank=context["rank"],
        world_size=context["world_size"],
        seed=seed,
        epoch=epoch,
        shuffle=training,
        equal=training,
    )
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=context["device"].type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _reduce_metrics(
    torch: Any,
    metrics: Mapping[str, float],
    *,
    samples: int,
    context: Mapping[str, Any],
) -> dict[str, float]:
    keys = sorted(metrics)
    values = [float(metrics[key]) for key in keys] + [float(samples)]
    tensor = torch.tensor(
        values,
        dtype=torch.float64,
        device=context["device"],
    )
    if context["world_size"] > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    total_samples = float(tensor[-1].item())
    if total_samples <= 0:
        raise RuntimeError("metric reduction has no samples")
    return {
        key: float(tensor[index].item()) / total_samples
        for index, key in enumerate(keys)
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(torch: Any, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_state_cpu(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }


def _checkpoint_payload(
    torch: Any,
    *,
    stage: str,
    epoch: int,
    model: Any,
    optimizer: Any,
    official_receipts: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    cache_receipt: Mapping[str, Any],
    trainable_receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    validation: Mapping[str, float],
    frozen_state_digest: str | None,
) -> dict[str, Any]:
    state = _model_state_cpu(model)
    receipt = {
        "format": TRANSFER_FORMAT,
        "stage": stage,
        "epoch": epoch,
        "official_initialization": dict(official_receipts),
        "source_receipt": dict(source_receipt),
        "cache_receipt": dict(cache_receipt),
        "trainable_policy": dict(trainable_receipt),
        "protocol": dict(protocol),
        "validation": dict(validation),
        "model_state_sha256": state_dict_sha256(state),
        "frozen_encoder_quantizer_sha256": frozen_state_digest,
        "withdrawn_e30_allowed": False,
        "test_visible": False,
    }
    receipt["receipt_sha256"] = canonical_payload_sha256(receipt)
    return {
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "transfer_receipt": receipt,
    }


def _unwrap(module: Any) -> Any:
    return getattr(module, "module", module)


def _run_face_epoch(
    torch: Any,
    *,
    model: Any,
    decoder: Any,
    loader: Any,
    optimizer: Any | None,
    device: Any,
    weights: Mapping[str, float],
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    if training:
        _unwrap(decoder).train()
    else:
        _unwrap(decoder).eval()
    metrics = {
        "total": 0.0,
        "jaw_geodesic": 0.0,
        "expression_l1": 0.0,
        "velocity": 0.0,
        "acceleration": 0.0,
    }
    samples = 0
    gradient_context = torch.enable_grad() if training else torch.inference_mode()
    with gradient_context:
        for batch in loader:
            code_ids = batch["code_ids"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            cached_zq = batch["zq"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            target = batch["target"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with torch.inference_mode():
                exact_zq = model.quantizer.get_codebook_entry(code_ids)
                if not torch.equal(exact_zq, cached_zq):
                    raise RuntimeError(
                        "cached Face zq is not exact for stored official code IDs"
                    )
            if training:
                optimizer.zero_grad(set_to_none=True)
            # ``exact_zq`` is deliberately recomputed under inference mode so
            # the cache remains fail-closed against the frozen official
            # quantizer.  An inference tensor cannot be saved by Conv1d for
            # the trainable decoder's backward pass.  The byte-exact cached
            # tensor was materialized normally by the LMDB dataset, so use it
            # after the equality proof instead of weakening that proof or
            # cloning inference state into the autograd path.
            prediction = decoder(cached_zq)
            losses = face_task_losses(
                torch,
                prediction,
                target,
                jaw_weight=weights["jaw"],
                expression_weight=weights["expression"],
                velocity_weight=weights["velocity"],
                acceleration_weight=weights["acceleration"],
            )
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    _unwrap(decoder).parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
            batch_size = int(target.shape[0])
            for key in metrics:
                metrics[key] += float(losses[key].detach().item()) * batch_size
            samples += batch_size
    return metrics, samples


def _run_global_epoch(
    torch: Any,
    *,
    global_model: Any,
    loader: Any,
    optimizer: Any | None,
    device: Any,
    weights: Mapping[str, float],
    policy: str,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    raw_model = _unwrap(global_model)
    if policy == "decoder_only":
        raw_model.eval()
        raw_model.encoder.eval()
        raw_model.decoder.train(training)
    elif policy == "full_model_fallback":
        raw_model.train(training)
    else:
        raise ValueError(policy)
    metrics = {
        "total": 0.0,
        "root_channel_l1": 0.0,
        "integrated_root_l1": 0.0,
        "root_velocity_l1": 0.0,
        "root_acceleration_l1": 0.0,
    }
    samples = 0
    gradient_context = torch.enable_grad() if training else torch.inference_mode()
    with gradient_context:
        for batch in loader:
            global_input = batch["global_input"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if not torch.equal(
                global_input[..., 54:57],
                torch.zeros_like(global_input[..., 54:57]),
            ):
                raise RuntimeError(
                    "Global transfer input lower translation is not exact zero"
                )
            target_root = batch["target_root"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            target_translation = batch["target_translation"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            anchor = batch["anchor"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = global_model(global_input)["rec_pose"]
            losses = global_root_losses(
                torch,
                prediction,
                target_root,
                target_translation,
                anchor,
                channel_weight=weights["channel"],
                integrated_weight=weights["integrated"],
                velocity_weight=weights["velocity"],
                acceleration_weight=weights["acceleration"],
            )
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    _unwrap(global_model).parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
            batch_size = int(global_input.shape[0])
            for key in metrics:
                metrics[key] += float(losses[key].detach().item()) * batch_size
            samples += batch_size
    return metrics, samples


def _load_resume(
    torch: Any,
    *,
    path: Path,
    stage: str,
    model: Any,
    optimizer: Any,
    official_receipts: Mapping[str, Any],
    cache_receipt: Mapping[str, Any],
    frozen_digest: str | None,
) -> int:
    reject_e30_reference(path, "transfer resume checkpoint")
    if path.is_symlink() or not path.resolve().is_file():
        raise RuntimeError(f"invalid transfer resume checkpoint {path}")
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("transfer resume checkpoint must be an object")
    receipt = payload.get("transfer_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("transfer resume checkpoint lacks transfer_receipt")
    _verify_receipt_hash(receipt, "transfer resume")
    if (
        receipt.get("format") != TRANSFER_FORMAT
        or receipt.get("stage") != stage
        or receipt.get("withdrawn_e30_allowed") is not False
        or receipt.get("test_visible") is not False
        or receipt.get("official_initialization") != dict(official_receipts)
        or receipt.get("cache_receipt") != dict(cache_receipt)
        or receipt.get("frozen_encoder_quantizer_sha256") != frozen_digest
    ):
        raise RuntimeError("transfer resume receipt contract mismatch")
    state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    epoch = payload.get("epoch")
    if (
        not isinstance(state, dict)
        or not isinstance(optimizer_state, dict)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise RuntimeError("transfer resume payload schema mismatch")
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    return epoch


def _train_transfer(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    if args.stage not in {"face", "global"}:
        raise ValueError("training stage must be face or global")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size/workers are invalid")
    if args.lr <= 0.0 or not np.isfinite(args.lr):
        raise ValueError("--lr must be finite and positive")
    reject_e30_reference(args.output_dir, "transfer output directory")
    if args.resume is not None:
        reject_e30_reference(args.resume, "transfer resume checkpoint")
    for root in args.cache_root:
        reject_e30_reference(root, "transfer cache root")

    context = _distributed_context(torch, args.device)
    device = context["device"]
    _configure_determinism(
        torch,
        np,
        seed=args.seed + context["rank"],
        device=device,
    )
    source_receipt = _git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )

    if args.stage == "face":
        model, face_receipt = _load_model(
            torch,
            "face",
            args.official_face_checkpoint,
            device,
        )
        official_receipts: dict[str, Any] = {"face": face_receipt}
        trainable_receipt = freeze_face_for_decoder_transfer(model)
        frozen_digest = state_dict_sha256(frozen_face_state(model))
        if frozen_digest != state_dict_sha256(frozen_face_state(model)):
            raise RuntimeError("Face frozen state hash is not stable")
        wrapped: Any = model.decoder
        if context["world_size"] > 1:
            wrapped = torch.nn.parallel.DistributedDataParallel(
                model.decoder,
                device_ids=[context["local_rank"]],
                output_device=context["local_rank"],
                broadcast_buffers=False,
            )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.decoder.parameters() if parameter.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        weights = {
            "jaw": args.jaw_weight,
            "expression": args.expression_weight,
            "velocity": args.velocity_weight,
            "acceleration": args.acceleration_weight,
        }
        validate_loss_weights(weights, "Face")
        protocol = {
            "stage": "face",
            "selection_split": "val",
            "test_visible": False,
            "initialization": "exact official All-Speakers Face checkpoint",
            "trainable": "decoder only",
            "encoder": "frozen eval",
            "quantizer": "frozen eval, no EMA update",
            "input": "precomputed exact official code IDs/zq",
            "per_batch_code_audit": "code IDs -> official zq must equal cached zq",
            "losses": {
                "jaw": "SO(3) geodesic radians",
                "expression": "L1",
                "velocity": (
                    "jaw relative-rotation geodesic plus expression delta L1"
                ),
                "acceleration": (
                    "jaw rotation-matrix second difference plus expression "
                    "second difference L1"
                ),
                "weights": weights,
            },
            "smplx_hot_loop": False,
        }
    else:
        model, global_receipt = _load_model(
            torch,
            "global",
            args.official_global_checkpoint,
            device,
        )
        lower_receipt = verify_official_checkpoint(
            "lower",
            args.official_lower_checkpoint,
        )
        official_receipts = {
            "global": global_receipt,
            "lower": lower_receipt,
        }
        trainable_receipt = configure_global_transfer_policy(
            model,
            policy=args.global_policy,
            authorize_full_model_fallback=args.authorize_full_global_fallback,
        )
        frozen = frozen_global_state(model, policy=args.global_policy)
        frozen_digest = state_dict_sha256(frozen) if frozen else None
        wrapped = model
        if context["world_size"] > 1:
            wrapped = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context["local_rank"]],
                output_device=context["local_rank"],
                broadcast_buffers=False,
            )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        weights = {
            "channel": args.channel_weight,
            "integrated": args.integrated_weight,
            "velocity": args.velocity_weight,
            "acceleration": args.acceleration_weight,
        }
        validate_loss_weights(weights, "Global")
        protocol = {
            "stage": "global",
            "selection_split": "val",
            "test_visible": False,
            "initialization": "exact official All-Speakers Global checkpoint",
            "trainable": args.global_policy,
            "full_model_fallback_authorized": (
                args.authorize_full_global_fallback
            ),
            "teacher_forced_train_interface": (
                "canonical GT lower -> frozen official Lower code IDs -> "
                "frozen official Lower decode"
            ),
            "locked_inference_interface": (
                "Base-predicted Lower code IDs -> frozen official Lower decode"
            ),
            "lower_rotation": "decoded and SO(3)-projected",
            "lower_translation_54_57": "exact zero every frame",
            "observed_contact_0_4": "canonical GT",
            "generated_contact_4_64": "decoded Lower",
            "world_root_in_lower_recurrent_seed": False,
            "losses": {
                "root_channels": "L1 on x/z velocity and absolute y",
                "integrated_root": "L1 with target frame-0 x/z anchor",
                "root_velocity": "first-difference L1",
                "root_acceleration": "second-difference L1",
                "weights": weights,
            },
            "smplx_hot_loop": False,
        }

    records, cache_receipt = _read_cache_roots(
        args.cache_root,
        stage=args.stage,
        official_receipts=official_receipts,
    )
    train_dataset = TransferCacheDataset(
        records=records,
        stage=args.stage,
        split="train",
        verify_payloads=args.verify_cache_payloads,
    )
    validation_dataset = TransferCacheDataset(
        records=records,
        stage=args.stage,
        split="val",
        verify_payloads=args.verify_cache_payloads,
    )

    if context["rank"] == 0:
        if args.resume is None:
            if args.output_dir.exists():
                raise FileExistsError(
                    f"refusing to overwrite transfer output {args.output_dir}"
                )
            args.output_dir.mkdir(parents=True, exist_ok=False)
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
    _barrier(torch, context)

    start_epoch = 0
    if args.resume is not None:
        start_epoch = _load_resume(
            torch,
            path=args.resume,
            stage=args.stage,
            model=model,
            optimizer=optimizer,
            official_receipts=official_receipts,
            cache_receipt=cache_receipt,
            frozen_digest=frozen_digest,
        )
    if start_epoch >= args.epochs:
        raise RuntimeError(
            f"resume epoch {start_epoch} is not below target {args.epochs}"
        )

    best_validation = float("inf")
    best_epoch: int | None = None
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loader = _loader(
            torch,
            train_dataset,
            context=context,
            seed=args.seed,
            epoch=epoch,
            batch_size=args.batch_size,
            workers=args.workers,
            training=True,
        )
        validation_loader = _loader(
            torch,
            validation_dataset,
            context=context,
            seed=args.seed,
            epoch=epoch,
            batch_size=args.batch_size,
            workers=args.workers,
            training=False,
        )
        if args.stage == "face":
            train_sums, train_samples = _run_face_epoch(
                torch,
                model=model,
                decoder=wrapped,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                weights=weights,
            )
            validation_sums, validation_samples = _run_face_epoch(
                torch,
                model=model,
                decoder=wrapped,
                loader=validation_loader,
                optimizer=None,
                device=device,
                weights=weights,
            )
        else:
            train_sums, train_samples = _run_global_epoch(
                torch,
                global_model=wrapped,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                weights=weights,
                policy=args.global_policy,
            )
            validation_sums, validation_samples = _run_global_epoch(
                torch,
                global_model=wrapped,
                loader=validation_loader,
                optimizer=None,
                device=device,
                weights=weights,
                policy=args.global_policy,
            )
        train_metrics = _reduce_metrics(
            torch,
            train_sums,
            samples=train_samples,
            context=context,
        )
        validation_metrics = _reduce_metrics(
            torch,
            validation_sums,
            samples=validation_samples,
            context=context,
        )
        current_frozen = (
            state_dict_sha256(frozen_face_state(model))
            if args.stage == "face"
            else (
                state_dict_sha256(
                    frozen_global_state(model, policy=args.global_policy)
                )
                if frozen_digest is not None
                else None
            )
        )
        if current_frozen != frozen_digest:
            raise RuntimeError(
                f"{args.stage} frozen encoder/quantizer state changed"
            )
        elapsed = time.time() - started
        event = {
            "epoch": epoch,
            "train": train_metrics,
            "val": validation_metrics,
            "elapsed_seconds": elapsed,
            "rank_count": context["world_size"],
            "finite": all(
                np.isfinite(value)
                for mapping in (train_metrics, validation_metrics)
                for value in mapping.values()
            ),
        }
        if not event["finite"]:
            raise RuntimeError("transfer metrics contain NaN/Inf")
        is_best = validation_metrics["total"] < best_validation
        if is_best:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
        if context["rank"] == 0:
            _append_jsonl(metrics_path, event)
            checkpoint = _checkpoint_payload(
                torch,
                stage=args.stage,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                official_receipts=official_receipts,
                source_receipt=source_receipt,
                cache_receipt=cache_receipt,
                trainable_receipt=trainable_receipt,
                protocol=protocol,
                validation=validation_metrics,
                frozen_state_digest=frozen_digest,
            )
            if is_best:
                _atomic_torch_save(
                    torch,
                    args.output_dir / f"best_{args.stage}_transfer.bin",
                    checkpoint,
                )
            if epoch % args.save_every == 0 or epoch == args.epochs:
                _atomic_torch_save(
                    torch,
                    args.output_dir
                    / f"{args.stage}_transfer_epoch_{epoch:04d}.bin",
                    checkpoint,
                )
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "epoch": epoch,
                        "train_total": train_metrics["total"],
                        "val_total": validation_metrics["total"],
                        "best_epoch": best_epoch,
                        "best_val_total": best_validation,
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _barrier(torch, context)

    if context["rank"] == 0:
        final = {
            "format": TRANSFER_FORMAT,
            "status": "complete",
            "stage": args.stage,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_total": best_validation,
            "official_initialization": official_receipts,
            "source_receipt": source_receipt,
            "cache_receipt": cache_receipt,
            "trainable_policy": trainable_receipt,
            "protocol": protocol,
            "frozen_encoder_quantizer_sha256": frozen_digest,
            "withdrawn_e30_allowed": False,
            "test_visible": False,
            "metrics_jsonl": str(metrics_path.resolve()),
            "metrics_jsonl_sha256": sha256_file(metrics_path),
            "elapsed_seconds": time.time() - started,
        }
        final["receipt_sha256"] = canonical_payload_sha256(final)
        _atomic_json(args.output_dir / "summary.json", final)
    _barrier(torch, context)
    if context["world_size"] > 1:
        torch.distributed.destroy_process_group()
    return 0


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)


def _add_canonical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--expected-canonical-commit", required=True)
    parser.add_argument("--expected-canonical-tree", required=True)


def _add_cache_build_arguments(parser: argparse.ArgumentParser) -> None:
    _add_source_arguments(parser)
    _add_canonical_arguments(parser)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--map-size-gb", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=20260730)


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    _add_source_arguments(parser)
    parser.add_argument(
        "--cache-root",
        type=Path,
        action="append",
        required=True,
        help="repeat once for each exact cache shard",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--verify-cache-payloads",
        action="store_true",
        help="rehash every LMDB payload while loading (strict but slower)",
    )
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official All-Speakers initialized SHOW transfer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    face_cache = subparsers.add_parser("build-face-cache")
    _add_cache_build_arguments(face_cache)
    face_cache.add_argument(
        "--official-face-checkpoint",
        type=Path,
        required=True,
    )
    face_cache.set_defaults(handler=_build_cache, stage="face")

    global_cache = subparsers.add_parser("build-global-cache")
    _add_cache_build_arguments(global_cache)
    global_cache.add_argument(
        "--official-lower-checkpoint",
        type=Path,
        required=True,
    )
    global_cache.set_defaults(handler=_build_cache, stage="global")

    face_train = subparsers.add_parser("train-face")
    _add_train_arguments(face_train)
    face_train.add_argument(
        "--official-face-checkpoint",
        type=Path,
        required=True,
    )
    face_train.add_argument("--jaw-weight", type=float, default=1.0)
    face_train.add_argument("--expression-weight", type=float, default=1.0)
    face_train.set_defaults(handler=_train_transfer, stage="face")

    global_train = subparsers.add_parser("train-global")
    _add_train_arguments(global_train)
    global_train.add_argument(
        "--official-global-checkpoint",
        type=Path,
        required=True,
    )
    global_train.add_argument(
        "--official-lower-checkpoint",
        type=Path,
        required=True,
    )
    global_train.add_argument("--channel-weight", type=float, default=1.0)
    global_train.add_argument("--integrated-weight", type=float, default=1.0)
    global_train.add_argument(
        "--global-policy",
        choices=("decoder_only", "full_model_fallback"),
        default="decoder_only",
    )
    global_train.add_argument(
        "--authorize-full-global-fallback",
        action="store_true",
        help=(
            "required only for predeclared full_model_fallback after the "
            "frozen decoder-only gate has failed"
        ),
    )
    global_train.set_defaults(handler=_train_transfer, stage="global")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "save_every", 1) <= 0:
        raise ValueError("--save-every must be positive")
    if getattr(args, "map_size_gb", 1.0) <= 0.0:
        raise ValueError("--map-size-gb must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
