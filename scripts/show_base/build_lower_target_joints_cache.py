#!/usr/bin/env python3
"""Build the frozen SHOW-All lower target-joints raw-value LMDB.

This producer runs the exact target-side SMPL-X call used by
``aelower_trainer.py``.  It stores one little-endian float32 C-order
``[64,127,3]`` payload per representation sample and never stores the 51
compute-only padding windows in the final tail batch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lower_target_cache import (  # noqa: E402
    CACHE_FORMAT,
    CACHE_VERSION,
    COMPUTE_BATCH_WINDOWS,
    COMPUTE_ROWS,
    COORDINATE_COUNT,
    ENTRY_BYTES,
    ENTRY_SHAPE,
    EXPECTED_ENTRIES,
    EXPECTED_SPEAKER_IDS,
    EXPECTED_SPEAKER_MAP,
    FULL_COMPUTE_BATCHES,
    JOINT_COUNT,
    LOWER_JOINT_INDICES,
    LOWER_POSE_COLUMNS,
    TAIL_PADDING_POLICY,
    TAIL_PADDING_WINDOWS,
    TAIL_REAL_WINDOWS,
    TOTAL_COMPUTE_BATCHES,
    TARGET_POSE_PREPROCESS,
    WINDOW_LENGTH,
    cache_key,
    canonical_json_sha256,
    current_runtime_receipt,
    encode_raw_entry,
    read_regular_file_snapshot,
    require_exact_int,
    require_exact_json_value,
    require_git_oid,
    sha256_file,
    update_entry_aggregate,
    validate_canonical_files,
    validate_manifest_payload,
    validate_representation_canonical_coverage,
    validate_representation_summary_payload,
)


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
REPRESENTATION_FORMATS = {
    "semtalk_show_representation_lmdb_v1",
    "semtalk_show_representation_lmdb_v2_global_foot",
}


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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def require_hex(value: str, length: int, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase {length}-hex digest")
    return normalized


def verified_json(
    path_value: str,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    path, raw = read_regular_file_snapshot(path_value, label)
    expected = require_hex(expected_sha256, 64, f"{label} SHA-256")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label} JSON: {path}") from error
    if type(value) is not dict:
        raise RuntimeError(f"{label} must contain a JSON object")
    return path, value, actual


def source_receipt(expected_commit: str, expected_tree: str) -> dict[str, str]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "lower target cache production requires a clean checkout; "
            f"first change: {status.splitlines()[0]}"
        )
    receipt = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
    }
    if receipt["origin"] != EXPECTED_ORIGIN:
        raise RuntimeError(f"unexpected source origin: {receipt['origin']!r}")
    if receipt["commit"] != expected_commit:
        raise RuntimeError(
            f"source commit {receipt['commit']} != {expected_commit}"
        )
    if receipt["tree"] != expected_tree:
        raise RuntimeError(f"source tree {receipt['tree']} != {expected_tree}")
    return receipt


def representation_receipt(args: argparse.Namespace) -> dict[str, Any]:
    lmdb_input = Path(args.representation_lmdb)
    if lmdb_input.is_symlink():
        raise RuntimeError(
            f"representation LMDB must not be a symlink: {lmdb_input}"
        )
    lmdb_path = lmdb_input.resolve()
    if not lmdb_path.is_dir():
        raise FileNotFoundError(lmdb_path)
    data_path = lmdb_path / "data.mdb"
    lock_path = lmdb_path / "lock.mdb"
    for path in (data_path, lock_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid representation LMDB artifact: {path}")
    expected_data_sha = require_hex(
        args.expected_representation_data_sha256,
        64,
        "representation data.mdb SHA-256",
    )
    data_sha = sha256_file(data_path)
    if data_sha != expected_data_sha:
        raise RuntimeError(
            f"representation data.mdb SHA mismatch: {data_sha} != {expected_data_sha}"
        )
    lock_sha = sha256_file(lock_path)
    summary_path, summary, summary_sha = verified_json(
        args.representation_summary,
        args.expected_representation_summary_sha256,
        "representation summary",
    )
    lineage_path, lineage, lineage_sha = verified_json(
        args.representation_lineage,
        args.expected_representation_lineage_sha256,
        "representation lineage",
    )
    protocol = summary.get("protocol")
    expected_aggregate = require_hex(
        args.expected_representation_entry_aggregate_sha256,
        64,
        "representation entry aggregate SHA-256",
    )
    require_exact_int(
        summary.get("entries"),
        "representation summary entries",
        expected=EXPECTED_ENTRIES,
    )
    if type(protocol) is not dict:
        raise RuntimeError("representation summary protocol must be an object")
    require_exact_int(
        protocol.get("window_length"),
        "representation summary window_length",
        expected=WINDOW_LENGTH,
    )
    require_exact_json_value(
        protocol.get("speaker_map"),
        EXPECTED_SPEAKER_MAP,
        "representation summary speaker_map",
    )
    if (
        summary.get("format") not in REPRESENTATION_FORMATS
        or summary.get("status") != "complete"
        or summary.get("data_mdb_sha256") != data_sha
        or summary.get("lock_mdb_sha256") != lock_sha
        or summary.get("entry_aggregate_sha256") != expected_aggregate
        or protocol.get("split") != "train"
    ):
        raise RuntimeError("representation summary is not the frozen SHOW-All cache")
    if (
        summary_path != lineage_path
        or summary_sha != lineage_sha
        or summary != lineage
    ):
        raise RuntimeError(
            "formal representation summary must also be its exact lineage manifest"
        )
    source = summary.get("source_receipt")
    if type(source) is not dict or source.get("origin") != EXPECTED_ORIGIN:
        raise RuntimeError("representation summary has an invalid source receipt")
    require_git_oid(source.get("commit"), "representation source commit")
    require_git_oid(source.get("tree"), "representation source tree")
    if (
        source.get("commit") != args.expected_source_commit
        or source.get("tree") != args.expected_source_tree
    ):
        raise RuntimeError(
            "representation source commit/tree do not match cache production"
        )
    receipt = {
        "lmdb_path": str(lmdb_path),
        "data_mdb_sha256": data_sha,
        "lock_mdb_sha256": lock_sha,
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha,
        "lineage_path": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "lineage_payload_sha256": canonical_json_sha256(lineage),
        "format": summary["format"],
        "entry_aggregate_sha256": expected_aggregate,
        "entries": EXPECTED_ENTRIES,
        "window_length": WINDOW_LENGTH,
        "speaker_scope": "All",
        "speaker_ids": list(EXPECTED_SPEAKER_IDS),
        "speaker_map": dict(EXPECTED_SPEAKER_MAP),
        "source_receipt": source,
        "canonical_manifest_sha256": summary.get(
            "canonical_manifest_sha256"
        ),
        "canonical_receipt": summary.get("canonical_receipt"),
    }
    validate_representation_summary_payload(
        summary,
        representation=receipt,
    )
    canonical_files = validate_canonical_files(receipt)
    receipt["canonical_smplx_asset_sha256"] = canonical_files[
        "smplx_asset_sha256"
    ]
    validate_representation_canonical_coverage(
        summary,
        canonical_files,
    )
    return receipt


def smplx_receipt(args: argparse.Namespace) -> dict[str, Any]:
    model_input = Path(args.smplx_model_dir)
    asset_input = Path(args.smplx_asset)
    if model_input.is_symlink() or asset_input.is_symlink():
        raise RuntimeError("SMPL-X model directory and asset must not be symlinks")
    model_dir = model_input.resolve()
    asset = asset_input.resolve()
    if not model_dir.is_dir() or not asset.is_file():
        raise FileNotFoundError(f"invalid SMPL-X model/asset: {model_dir}, {asset}")
    expected_asset = (
        model_dir / "smplx" / "SMPLX_NEUTRAL_2020.npz"
    ).resolve()
    if asset != expected_asset:
        raise RuntimeError(
            f"SMPL-X asset {asset} is not the trainer-selected {expected_asset}"
        )
    expected = require_hex(
        args.expected_smplx_asset_sha256,
        64,
        "SMPL-X asset SHA-256",
    )
    actual = sha256_file(asset)
    if actual != expected:
        raise RuntimeError(f"SMPL-X asset SHA mismatch: {actual} != {expected}")
    return {
        "model_dir": str(model_dir),
        "asset_path": str(asset),
        "asset_sha256": actual,
        "model_type": "smplx",
        "gender": "NEUTRAL_2020",
        "use_face_contour": False,
        "num_betas": 300,
        "num_expression_coeffs": 100,
        "ext": "npz",
        "use_pca": False,
        "output_joints": JOINT_COUNT,
        "return_verts": False,
        "return_joints": True,
        "return_shaped": False,
        "target_forward": "torch.no_grad+return_shaped_false",
        "translation": "tar_trans_minus_itself",
        "expression": "zeros_float32",
        "lower_joint_indices": list(LOWER_JOINT_INDICES),
        "lower_pose_columns": list(LOWER_POSE_COLUMNS),
        "target_pose_preprocess": TARGET_POSE_PREPROCESS,
    }


def read_representation_entry(
    transaction: Any,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    key = cache_key(index)
    value = transaction.get(key)
    if value is None:
        raise RuntimeError(f"representation LMDB misses key {key!r}")
    with io.BytesIO(bytes(value)) as handle:
        archive = np.load(handle, allow_pickle=False)
        required = {"pose", "beta", "trans", "speaker_id"}
        if not required.issubset(archive.files):
            raise RuntimeError(
                f"representation entry {index} misses {sorted(required - set(archive.files))}"
            )
        pose = archive["pose"].copy()
        beta = archive["beta"].copy()
        trans = archive["trans"].copy()
        speaker_id = archive["speaker_id"].copy()
    expected_shapes = {
        "pose": (WINDOW_LENGTH, 165),
        "beta": (WINDOW_LENGTH, 300),
        "trans": (WINDOW_LENGTH, 3),
        "speaker_id": (WINDOW_LENGTH, 1),
    }
    for name, array in (
        ("pose", pose),
        ("beta", beta),
        ("trans", trans),
        ("speaker_id", speaker_id),
    ):
        if array.shape != expected_shapes[name]:
            raise RuntimeError(
                f"representation entry {index} {name} shape "
                f"{array.shape} != {expected_shapes[name]}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise RuntimeError(
                f"representation entry {index} {name} is non-finite"
            )
    for name, array in (("pose", pose), ("beta", beta), ("trans", trans)):
        if array.dtype != np.float32:
            raise RuntimeError(
                f"representation entry {index} {name} dtype "
                f"{array.dtype} != float32"
            )
    if speaker_id.dtype.kind not in "iu":
        raise RuntimeError(
            f"representation entry {index} speaker_id must be integer"
        )
    unique_speakers = np.unique(speaker_id)
    if unique_speakers.size != 1:
        raise RuntimeError(
            f"representation entry {index} changes speaker within a window"
        )
    speaker = int(unique_speakers[0])
    if speaker not in EXPECTED_SPEAKER_IDS:
        raise RuntimeError(
            f"representation entry {index} has invalid speaker {speaker}"
        )
    return pose, beta, trans, speaker


def compute_target_joints(
    *,
    torch_module: Any,
    model: Any,
    pose: np.ndarray,
    beta: np.ndarray,
    trans: np.ndarray,
    device: Any,
    rotation_conversions: Any,
) -> Any:
    torch = torch_module
    source_pose = torch.from_numpy(pose).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    ).reshape(COMPUTE_ROWS, 165)
    lower_columns = torch.tensor(
        LOWER_POSE_COLUMNS,
        dtype=torch.int64,
        device=device,
    )
    lower_axis_angle = torch.index_select(
        source_pose,
        1,
        lower_columns,
    ).reshape(COMPUTE_ROWS, len(LOWER_JOINT_INDICES), 3)
    lower_matrix = rotation_conversions.axis_angle_to_matrix(lower_axis_angle)
    lower_rotation_6d = rotation_conversions.matrix_to_rotation_6d(
        lower_matrix
    ).reshape(COMPUTE_ROWS, len(LOWER_JOINT_INDICES), 6)
    lower_matrix_roundtrip = rotation_conversions.rotation_6d_to_matrix(
        lower_rotation_6d
    )
    lower_axis_angle_roundtrip = rotation_conversions.matrix_to_axis_angle(
        lower_matrix_roundtrip
    ).reshape(COMPUTE_ROWS, len(LOWER_POSE_COLUMNS))
    flat_pose = torch.zeros(
        (COMPUTE_ROWS, 165),
        dtype=torch.float32,
        device=device,
    )
    flat_pose[:, lower_columns] = lower_axis_angle_roundtrip
    flat_beta = torch.from_numpy(beta).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    ).reshape(COMPUTE_ROWS, 300)
    flat_trans = torch.from_numpy(trans).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    ).reshape(COMPUTE_ROWS, 3)
    expression = torch.zeros(
        (COMPUTE_ROWS, 100),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        output = model(
            betas=flat_beta,
            transl=flat_trans - flat_trans,
            expression=expression,
            jaw_pose=flat_pose[:, 66:69],
            global_orient=flat_pose[:, 0:3],
            body_pose=flat_pose[:, 3:66],
            left_hand_pose=flat_pose[:, 75:120],
            right_hand_pose=flat_pose[:, 120:165],
            return_verts=False,
            return_joints=True,
            return_shaped=False,
            leye_pose=flat_pose[:, 69:72],
            reye_pose=flat_pose[:, 72:75],
        )
        joints = output["joints"]
    if (
        tuple(joints.shape) != (COMPUTE_ROWS, JOINT_COUNT, COORDINATE_COUNT)
        or joints.dtype != torch.float32
        or not joints.is_contiguous()
        or joints.requires_grad
        or joints.grad_fn is not None
        or not bool(torch.isfinite(joints).all().item())
    ):
        raise RuntimeError(
            "SMPL-X target output violates [4096,127,3] contiguous finite "
            "float32 no-grad contract"
        )
    return joints.reshape(
        COMPUTE_BATCH_WINDOWS,
        WINDOW_LENGTH,
        JOINT_COUNT,
        COORDINATE_COUNT,
    ).cpu().numpy()


def protocol_receipt() -> dict[str, Any]:
    return {
        "dataset": "show_base",
        "formal_stage": "lower",
        "speaker_scope": "All",
        "speaker_ids": list(EXPECTED_SPEAKER_IDS),
        "speaker_map": dict(EXPECTED_SPEAKER_MAP),
        "split": "train",
        "entries": EXPECTED_ENTRIES,
        "window_length": WINDOW_LENGTH,
        "joints": JOINT_COUNT,
        "coordinates": COORDINATE_COUNT,
        "dtype": "<f4",
        "byte_order": "little",
        "storage": "raw_c_order",
        "key_format": "%010d",
        "exact_once": True,
        "compute_batch_windows": COMPUTE_BATCH_WINDOWS,
        "compute_rows": COMPUTE_ROWS,
        "full_compute_batches": FULL_COMPUTE_BATCHES,
        "total_compute_batches": TOTAL_COMPUTE_BATCHES,
        "tail_real_windows": TAIL_REAL_WINDOWS,
        "tail_padding_windows": TAIL_PADDING_WINDOWS,
        "tail_padding_policy": TAIL_PADDING_POLICY,
        "lower_joint_indices": list(LOWER_JOINT_INDICES),
        "lower_pose_columns": list(LOWER_POSE_COLUMNS),
        "target_pose_preprocess": TARGET_POSE_PREPROCESS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-lmdb", required=True)
    parser.add_argument("--representation-summary", required=True)
    parser.add_argument("--representation-lineage", required=True)
    parser.add_argument("--expected-representation-data-sha256", required=True)
    parser.add_argument("--expected-representation-summary-sha256", required=True)
    parser.add_argument("--expected-representation-lineage-sha256", required=True)
    parser.add_argument(
        "--expected-representation-entry-aggregate-sha256",
        required=True,
    )
    parser.add_argument("--smplx-model-dir", required=True)
    parser.add_argument("--smplx-asset", required=True)
    parser.add_argument("--expected-smplx-asset-sha256", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument("--expected-entries", type=int, default=EXPECTED_ENTRIES)
    parser.add_argument(
        "--compute-batch-windows",
        type=int,
        default=COMPUTE_BATCH_WINDOWS,
    )
    parser.add_argument("--map-size-gib", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.expected_source_commit = require_hex(
        args.expected_source_commit, 40, "--expected-source-commit"
    )
    args.expected_source_tree = require_hex(
        args.expected_source_tree, 40, "--expected-source-tree"
    )
    if (
        args.expected_entries != EXPECTED_ENTRIES
        or args.compute_batch_windows != COMPUTE_BATCH_WINDOWS
        or args.map_size_gib != 16
    ):
        raise RuntimeError(
            "formal lower target cache is fixed at 127286 entries, "
            "64 compute windows, and a 16-GiB map"
        )
    source = source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    representation = representation_receipt(args)
    smplx_info = smplx_receipt(args)

    output_input = Path(args.output_lmdb)
    manifest_input = Path(args.output_manifest)
    if output_input.is_symlink() or manifest_input.is_symlink():
        raise RuntimeError("output paths must not be symlinks")
    output = output_input.resolve()
    manifest_path = manifest_input.resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite lower target cache outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)

    try:
        import lmdb
        import smplx
        import torch
        from utils import rotation_conversions
    except ImportError as error:
        raise RuntimeError("lmdb, smplx, and torch are required") from error

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal lower target builder requires CUDA")
    device_name = torch.cuda.get_device_name(device)
    if "H200" not in device_name or device_name != args.expected_device_name:
        raise RuntimeError(
            f"builder requires the explicitly bound H200, got {device_name!r}"
        )
    model = smplx.create(
        smplx_info["model_dir"],
        model_type="smplx",
        gender="NEUTRAL_2020",
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext="npz",
        use_pca=False,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    runtime = current_runtime_receipt(device)
    representation_env = lmdb.open(
        representation["lmdb_path"],
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
        subdir=True,
    )
    with representation_env.begin(buffers=True) as transaction:
        representation_entries = int(transaction.stat()["entries"])
    if representation_entries != EXPECTED_ENTRIES:
        representation_env.close()
        raise RuntimeError("representation LMDB entry count changed")

    temporary.mkdir()
    output_env = lmdb.open(
        str(temporary),
        subdir=True,
        map_size=16 * 1024**3,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=False,
    )
    aggregate = hashlib.sha256()
    coverage = np.zeros(EXPECTED_ENTRIES, dtype=np.bool_)
    observed_speakers: set[int] = set()
    started = time.time()
    written = 0
    try:
        for batch_index, start in enumerate(
            range(0, EXPECTED_ENTRIES, COMPUTE_BATCH_WINDOWS)
        ):
            stop = min(start + COMPUTE_BATCH_WINDOWS, EXPECTED_ENTRIES)
            real_windows = stop - start
            pose = np.zeros(
                (COMPUTE_BATCH_WINDOWS, WINDOW_LENGTH, 165),
                dtype=np.float32,
            )
            beta = np.zeros(
                (COMPUTE_BATCH_WINDOWS, WINDOW_LENGTH, 300),
                dtype=np.float32,
            )
            trans = np.zeros(
                (COMPUTE_BATCH_WINDOWS, WINDOW_LENGTH, 3),
                dtype=np.float32,
            )
            with representation_env.begin(buffers=True) as transaction:
                for offset, index in enumerate(range(start, stop)):
                    if coverage[index]:
                        raise RuntimeError(f"duplicate source index {index}")
                    (
                        pose[offset],
                        beta[offset],
                        trans[offset],
                        speaker,
                    ) = read_representation_entry(transaction, index)
                    observed_speakers.add(speaker)
                    coverage[index] = True
            joints = compute_target_joints(
                torch_module=torch,
                model=model,
                pose=pose,
                beta=beta,
                trans=trans,
                device=device,
                rotation_conversions=rotation_conversions,
            )
            with output_env.begin(write=True) as transaction:
                for offset, index in enumerate(range(start, stop)):
                    payload = encode_raw_entry(joints[offset])
                    key = cache_key(index)
                    if not transaction.put(key, payload, overwrite=False):
                        raise RuntimeError(f"duplicate output key {key!r}")
                    update_entry_aggregate(aggregate, index, payload)
                    written += 1
            if batch_index == TOTAL_COMPUTE_BATCHES - 1:
                if (
                    real_windows != TAIL_REAL_WINDOWS
                    or COMPUTE_BATCH_WINDOWS - real_windows
                    != TAIL_PADDING_WINDOWS
                ):
                    raise RuntimeError("tail 54+10 padding contract changed")
        output_env.sync(True)
    except BaseException:
        output_env.close()
        representation_env.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    output_env.close()
    representation_env.close()

    if (
        written != EXPECTED_ENTRIES
        or int(coverage.sum()) != EXPECTED_ENTRIES
        or not bool(coverage.all())
        or observed_speakers != set(EXPECTED_SPEAKER_IDS)
    ):
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(
            "producer did not cover 127286 SHOW-All records exactly once"
        )
    final_source = source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    final_representation = representation_receipt(args)
    final_smplx = smplx_receipt(args)
    if (
        final_source != source
        or final_representation != representation
        or final_smplx != smplx_info
    ):
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError("formal input receipts changed during cache production")

    try:
        verification_env = lmdb.open(
            str(temporary),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=8,
            subdir=True,
        )
        try:
            with verification_env.begin(buffers=True) as transaction:
                if int(transaction.stat()["entries"]) != EXPECTED_ENTRIES:
                    raise RuntimeError("output LMDB entry count is not exact")
                cursor_entries = 0
                for expected_index, (key, value) in enumerate(
                    transaction.cursor()
                ):
                    if bytes(key) != cache_key(expected_index):
                        raise RuntimeError("output LMDB keys are not contiguous")
                    if len(value) != ENTRY_BYTES:
                        raise RuntimeError("output LMDB raw payload size changed")
                    cursor_entries += 1
                if cursor_entries != EXPECTED_ENTRIES:
                    raise RuntimeError(
                        "output LMDB cursor did not cover every entry"
                    )
        finally:
            verification_env.close()
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    os.replace(temporary, output)
    manifest = {
        "format": CACHE_FORMAT,
        "cache_version": CACHE_VERSION,
        "status": "complete",
        "protocol": protocol_receipt(),
        "source_receipt": source,
        "representation_receipt": representation,
        "smplx_receipt": smplx_info,
        "runtime": runtime,
        "lmdb": {
            "path": str(output),
            "map_size_bytes": 16 * 1024**3,
            "entries": EXPECTED_ENTRIES,
            "data_mdb_sha256": sha256_file(output / "data.mdb"),
            "lock_mdb_sha256": sha256_file(output / "lock.mdb"),
        },
        "entries": EXPECTED_ENTRIES,
        "entry_shape": list(ENTRY_SHAPE),
        "entry_bytes": ENTRY_BYTES,
        "entry_aggregate_sha256": aggregate.hexdigest(),
        "observed_speaker_ids": sorted(observed_speakers),
        "exact_once": True,
        "finite": True,
        "target_requires_grad": False,
        "target_optimizer_member": False,
        "started_unix": started,
        "completed_unix": time.time(),
        "argv": sys.argv,
    }
    validate_manifest_payload(manifest)
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "entries": manifest["entries"],
                "entry_aggregate_sha256": manifest[
                    "entry_aggregate_sha256"
                ],
                "manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
