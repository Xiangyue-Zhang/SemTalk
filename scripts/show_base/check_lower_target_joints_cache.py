#!/usr/bin/env python3
"""Independently recompute every frozen SHOW lower target-joints entry.

The checker intentionally does not import the producer.  It traverses a seeded
permutation of all 127,309 representation indices exactly once, independently
reconstructs the trainer's SMPL-X arguments, and requires bitwise
``torch.equal`` equality for every cached float.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lower_target_cache import (  # noqa: E402
    CACHE_VERSION,
    CHECKER_PERMUTATION_SEED,
    CHECKER_FORMAT,
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
    canonical_json_sha256,
    current_runtime_receipt,
    read_regular_file_snapshot,
    require_exact_int,
    require_exact_json_value,
    require_git_oid,
    sha256_file,
    validate_canonical_files,
    validate_manifest_payload,
    validate_representation_canonical_coverage,
    validate_representation_summary_payload,
)


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
PERMUTATION_SEED = CHECKER_PERMUTATION_SEED
REPRESENTATION_FORMATS = {
    "semtalk_show_representation_lmdb_v1",
    "semtalk_show_representation_lmdb_v2_global_foot",
}


def _canonical_json_bytes(value: Any) -> bytes:
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_bytes(_canonical_json_bytes(value))
    os.replace(temporary, path)


def _require_hex(value: str, length: int, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase {length}-hex digest")
    return normalized


def _verified_json(
    path_value: str,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    path, raw = read_regular_file_snapshot(path_value, label)
    expected = _require_hex(expected_sha256, 64, f"{label} SHA-256")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label} JSON: {path}") from error
    if type(value) is not dict:
        raise RuntimeError(f"{label} must contain one JSON object")
    return path, value, actual


def _source_receipt(expected_commit: str, expected_tree: str) -> dict[str, str]:
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
            "lower target checker requires a clean checkout; "
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
    if receipt["commit"] != expected_commit or receipt["tree"] != expected_tree:
        raise RuntimeError("checker source commit/tree do not match explicit inputs")
    return receipt


def _representation_receipt(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.representation_lmdb)
    if input_path.is_symlink():
        raise RuntimeError("representation LMDB must not be a symlink")
    lmdb_path = input_path.resolve()
    if not lmdb_path.is_dir():
        raise FileNotFoundError(lmdb_path)
    data_path = lmdb_path / "data.mdb"
    lock_path = lmdb_path / "lock.mdb"
    for path in (data_path, lock_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid representation LMDB artifact: {path}")
    expected_data = _require_hex(
        args.expected_representation_data_sha256,
        64,
        "representation data.mdb SHA-256",
    )
    data_sha = sha256_file(data_path)
    if data_sha != expected_data:
        raise RuntimeError("representation data.mdb SHA mismatch")
    lock_sha = sha256_file(lock_path)
    summary_path, summary, summary_sha = _verified_json(
        args.representation_summary,
        args.expected_representation_summary_sha256,
        "representation summary",
    )
    lineage_path, lineage, lineage_sha = _verified_json(
        args.representation_lineage,
        args.expected_representation_lineage_sha256,
        "representation lineage",
    )
    protocol = summary.get("protocol")
    expected_aggregate = _require_hex(
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
        raise RuntimeError("representation inputs are not the frozen SHOW-All cache")
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
        raise RuntimeError("representation source origin is invalid")
    require_git_oid(source.get("commit"), "representation source commit")
    require_git_oid(source.get("tree"), "representation source tree")
    if (
        source.get("commit") != args.expected_source_commit
        or source.get("tree") != args.expected_source_tree
    ):
        raise RuntimeError(
            "representation source commit/tree do not match cache checking"
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


def _smplx_receipt(args: argparse.Namespace) -> dict[str, Any]:
    model_input = Path(args.smplx_model_dir)
    asset_input = Path(args.smplx_asset)
    if model_input.is_symlink() or asset_input.is_symlink():
        raise RuntimeError("SMPL-X paths must not be symlinks")
    model_dir = model_input.resolve()
    asset = asset_input.resolve()
    if not model_dir.is_dir() or not asset.is_file():
        raise FileNotFoundError("invalid SMPL-X model directory or asset")
    expected_asset = (
        model_dir / "smplx" / "SMPLX_NEUTRAL_2020.npz"
    ).resolve()
    if asset != expected_asset:
        raise RuntimeError(
            f"SMPL-X asset {asset} is not the trainer-selected {expected_asset}"
        )
    expected = _require_hex(
        args.expected_smplx_asset_sha256,
        64,
        "SMPL-X asset SHA-256",
    )
    actual = sha256_file(asset)
    if actual != expected:
        raise RuntimeError("SMPL-X asset SHA mismatch")
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


def _key(index: int) -> bytes:
    if index < 0 or index >= EXPECTED_ENTRIES:
        raise IndexError(index)
    return ("%010d" % index).encode("ascii")


def _decode_raw(payload: Any) -> np.ndarray:
    raw = bytes(payload)
    if len(raw) != ENTRY_BYTES:
        raise RuntimeError(
            f"cache payload has {len(raw)} bytes instead of {ENTRY_BYTES}"
        )
    array = np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(ENTRY_SHAPE).copy()
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise RuntimeError("cache payload is not finite little-endian float32")
    return array


def _read_representation(
    transaction: Any,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    value = transaction.get(_key(index))
    if value is None:
        raise RuntimeError(f"representation LMDB misses index {index}")
    with io.BytesIO(bytes(value)) as handle:
        archive = np.load(handle, allow_pickle=False)
        required = {"pose", "beta", "trans", "speaker_id"}
        if not required.issubset(archive.files):
            raise RuntimeError(f"representation index {index} lacks required fields")
        pose = archive["pose"].copy()
        beta = archive["beta"].copy()
        trans = archive["trans"].copy()
        speaker_id = archive["speaker_id"].copy()
    expectations = (
        ("pose", pose, (WINDOW_LENGTH, 165)),
        ("beta", beta, (WINDOW_LENGTH, 300)),
        ("trans", trans, (WINDOW_LENGTH, 3)),
        ("speaker_id", speaker_id, (WINDOW_LENGTH, 1)),
    )
    for name, array, shape in expectations:
        if array.shape != shape:
            raise RuntimeError(
                f"representation index {index} {name} shape {array.shape} != {shape}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise RuntimeError(f"representation index {index} {name} is non-finite")
    if pose.dtype != np.float32 or beta.dtype != np.float32 or trans.dtype != np.float32:
        raise RuntimeError("checker requires original float32 pose/beta/trans")
    if speaker_id.dtype.kind not in "iu":
        raise RuntimeError("checker requires integer speaker_id")
    unique = np.unique(speaker_id)
    if unique.size != 1 or int(unique[0]) not in EXPECTED_SPEAKER_IDS:
        raise RuntimeError(f"invalid speaker window at representation index {index}")
    return pose, beta, trans, int(unique[0])


def _live_target_joints(
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
        device=device, dtype=torch.float32
    ).reshape(COMPUTE_ROWS, 165)
    lower_columns = torch.tensor(
        LOWER_POSE_COLUMNS,
        dtype=torch.int64,
        device=device,
    )
    selected_axis_angle = torch.index_select(
        source_pose,
        1,
        lower_columns,
    ).reshape(COMPUTE_ROWS, len(LOWER_JOINT_INDICES), 3)
    selected_matrix = rotation_conversions.axis_angle_to_matrix(
        selected_axis_angle
    )
    selected_six = rotation_conversions.matrix_to_rotation_6d(
        selected_matrix
    ).reshape(COMPUTE_ROWS, len(LOWER_JOINT_INDICES), 6)
    selected_matrix_again = rotation_conversions.rotation_6d_to_matrix(
        selected_six
    )
    selected_axis_angle_again = rotation_conversions.matrix_to_axis_angle(
        selected_matrix_again
    ).reshape(COMPUTE_ROWS, len(LOWER_POSE_COLUMNS))
    pose_tensor = torch.zeros(
        (COMPUTE_ROWS, 165),
        dtype=torch.float32,
        device=device,
    )
    pose_tensor[:, lower_columns] = selected_axis_angle_again
    beta_tensor = torch.from_numpy(beta).to(
        device=device, dtype=torch.float32
    ).reshape(COMPUTE_ROWS, 300)
    trans_tensor = torch.from_numpy(trans).to(
        device=device, dtype=torch.float32
    ).reshape(COMPUTE_ROWS, 3)
    zero_expression = torch.zeros(
        (COMPUTE_ROWS, 100),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        result = model(
            betas=beta_tensor,
            transl=trans_tensor - trans_tensor,
            expression=zero_expression,
            jaw_pose=pose_tensor[:, 66:69],
            global_orient=pose_tensor[:, :3],
            body_pose=pose_tensor[:, 3:66],
            left_hand_pose=pose_tensor[:, 75:120],
            right_hand_pose=pose_tensor[:, 120:165],
            return_verts=False,
            return_joints=True,
            return_shaped=False,
            leye_pose=pose_tensor[:, 69:72],
            reye_pose=pose_tensor[:, 72:75],
        )
        joints = result["joints"]
    if (
        tuple(joints.shape) != (COMPUTE_ROWS, JOINT_COUNT, COORDINATE_COUNT)
        or joints.dtype != torch.float32
        or not joints.is_contiguous()
        or joints.requires_grad
        or joints.grad_fn is not None
        or not bool(torch.isfinite(joints).all().item())
    ):
        raise RuntimeError("independent live SMPL-X output violates the contract")
    return joints.reshape(
        COMPUTE_BATCH_WINDOWS,
        WINDOW_LENGTH,
        JOINT_COUNT,
        COORDINATE_COUNT,
    ).cpu()


def _protocol() -> dict[str, Any]:
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
    parser.add_argument("--cache-lmdb", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--output-checker-receipt", required=True)
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
    parser.add_argument("--permutation-seed", type=int, default=PERMUTATION_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.expected_source_commit = _require_hex(
        args.expected_source_commit, 40, "--expected-source-commit"
    )
    args.expected_source_tree = _require_hex(
        args.expected_source_tree, 40, "--expected-source-tree"
    )
    if (
        args.expected_entries != EXPECTED_ENTRIES
        or args.compute_batch_windows != COMPUTE_BATCH_WINDOWS
        or args.permutation_seed != PERMUTATION_SEED
    ):
        raise RuntimeError(
            "formal checker is fixed at 127309 entries, 64-window batches, "
            "and permutation seed 20260729"
        )
    output_input = Path(args.output_checker_receipt)
    if output_input.is_symlink():
        raise RuntimeError("checker output must not be a symlink")
    output = output_input.resolve()
    if output.exists():
        raise FileExistsError(output)

    source = _source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    representation = _representation_receipt(args)
    smplx_info = _smplx_receipt(args)
    manifest_path, manifest, manifest_sha = _verified_json(
        args.cache_manifest,
        args.expected_cache_manifest_sha256,
        "cache manifest",
    )
    validate_manifest_payload(manifest)
    if manifest.get("representation_receipt") != representation:
        raise RuntimeError("producer/checker representation receipts disagree")
    if manifest.get("smplx_receipt") != smplx_info:
        raise RuntimeError("producer/checker SMPL-X receipts disagree")
    producer_source = manifest.get("source_receipt")
    if type(producer_source) is not dict or any(
        source[key] != producer_source.get(key)
        for key in ("origin", "commit", "tree")
    ):
        raise RuntimeError("producer/checker source commit receipts disagree")

    cache_input = Path(args.cache_lmdb)
    if cache_input.is_symlink():
        raise RuntimeError("cache LMDB must not be a symlink")
    cache_path = cache_input.resolve()
    if not cache_path.is_dir():
        raise FileNotFoundError(cache_path)
    if cache_path != Path(manifest["lmdb"]["path"]).resolve():
        raise RuntimeError("cache LMDB path does not match the producer manifest")
    cache_data = cache_path / "data.mdb"
    cache_lock = cache_path / "lock.mdb"
    if (
        cache_data.is_symlink()
        or cache_lock.is_symlink()
        or not cache_data.is_file()
        or not cache_lock.is_file()
    ):
        raise RuntimeError("cache LMDB artifacts are incomplete")
    cache_data_sha = sha256_file(cache_data)
    if cache_data_sha != manifest["lmdb"]["data_mdb_sha256"]:
        raise RuntimeError("cache data.mdb changed after producer finalization")
    if sha256_file(cache_lock) != manifest["lmdb"]["lock_mdb_sha256"]:
        raise RuntimeError("cache lock.mdb changed after producer finalization")

    try:
        import lmdb
        import smplx
        import torch
        from utils import rotation_conversions
    except ImportError as error:
        raise RuntimeError("lmdb, smplx, and torch are required") from error
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal lower target checker requires CUDA")
    device_name = torch.cuda.get_device_name(device)
    if "H200" not in device_name or device_name != args.expected_device_name:
        raise RuntimeError(
            f"checker requires the explicitly bound H200, got {device_name!r}"
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
    if runtime != manifest.get("runtime"):
        raise RuntimeError(
            "checker runtime does not exactly match the producer runtime"
        )

    representation_env = lmdb.open(
        representation["lmdb_path"],
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
        subdir=True,
    )
    cache_env = lmdb.open(
        str(cache_path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
        subdir=True,
    )
    try:
        with representation_env.begin(buffers=True) as transaction:
            representation_entries = int(transaction.stat()["entries"])
        with cache_env.begin(buffers=True) as transaction:
            cache_entries = int(transaction.stat()["entries"])
        if representation_entries != EXPECTED_ENTRIES:
            raise RuntimeError("representation LMDB entry count changed")
        if cache_entries != EXPECTED_ENTRIES:
            raise RuntimeError("cache LMDB entry count changed")
    except BaseException:
        representation_env.close()
        cache_env.close()
        raise

    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(
        EXPECTED_ENTRIES
    )
    if (
        permutation.shape != (EXPECTED_ENTRIES,)
        or np.unique(permutation).size != EXPECTED_ENTRIES
        or int(permutation.min()) != 0
        or int(permutation.max()) != EXPECTED_ENTRIES - 1
    ):
        raise RuntimeError("checker permutation is not an exact cover")
    visited = np.zeros(EXPECTED_ENTRIES, dtype=np.bool_)
    per_index_digest = np.empty((EXPECTED_ENTRIES, 32), dtype=np.uint8)
    observed_speakers: set[int] = set()
    mismatch_count = 0
    started = time.time()
    try:
        for start in range(0, EXPECTED_ENTRIES, COMPUTE_BATCH_WINDOWS):
            batch_indices = permutation[start : start + COMPUTE_BATCH_WINDOWS]
            real_windows = int(batch_indices.size)
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
            cached_entries: list[np.ndarray] = []
            with representation_env.begin(buffers=True) as source_transaction:
                with cache_env.begin(buffers=True) as cache_transaction:
                    for offset, raw_index in enumerate(batch_indices):
                        index = int(raw_index)
                        if visited[index]:
                            raise RuntimeError(f"checker revisited index {index}")
                        (
                            pose[offset],
                            beta[offset],
                            trans[offset],
                            speaker,
                        ) = _read_representation(source_transaction, index)
                        observed_speakers.add(speaker)
                        value = cache_transaction.get(_key(index))
                        if value is None:
                            raise RuntimeError(f"cache LMDB misses index {index}")
                        payload = bytes(value)
                        cached_entries.append(_decode_raw(payload))
                        per_index_digest[index] = np.frombuffer(
                            hashlib.sha256(payload).digest(),
                            dtype=np.uint8,
                        )
                        visited[index] = True
            live = _live_target_joints(
                torch_module=torch,
                model=model,
                pose=pose,
                beta=beta,
                trans=trans,
                device=device,
                rotation_conversions=rotation_conversions,
            )
            for offset, cached in enumerate(cached_entries):
                if not torch.equal(live[offset], torch.from_numpy(cached)):
                    mismatch_count += 1
                    raise RuntimeError(
                        "bitwise target mismatch at representation index "
                        f"{int(batch_indices[offset])}"
                    )
            if start + COMPUTE_BATCH_WINDOWS >= EXPECTED_ENTRIES:
                if (
                    real_windows != TAIL_REAL_WINDOWS
                    or COMPUTE_BATCH_WINDOWS - real_windows
                    != TAIL_PADDING_WINDOWS
                ):
                    raise RuntimeError("checker tail is not 13 real + 51 padding")
    finally:
        representation_env.close()
        cache_env.close()

    if (
        mismatch_count != 0
        or int(visited.sum()) != EXPECTED_ENTRIES
        or not bool(visited.all())
        or observed_speakers != set(EXPECTED_SPEAKER_IDS)
    ):
        raise RuntimeError("checker failed exact-once SHOW-All verification")
    aggregate = hashlib.sha256()
    for index in range(EXPECTED_ENTRIES):
        aggregate.update(_key(index))
        aggregate.update(per_index_digest[index].tobytes())
    aggregate_sha = aggregate.hexdigest()
    if aggregate_sha != manifest["entry_aggregate_sha256"]:
        raise RuntimeError("randomized checker observed the wrong entry aggregate")

    final_source = _source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    final_representation = _representation_receipt(args)
    final_smplx = _smplx_receipt(args)
    final_manifest_sha = sha256_file(manifest_path)
    final_cache_data_sha = sha256_file(cache_data)
    final_cache_lock_sha = sha256_file(cache_lock)
    if (
        final_source != source
        or final_representation != representation
        or final_smplx != smplx_info
        or final_manifest_sha != manifest_sha
        or final_cache_data_sha != cache_data_sha
        or final_cache_lock_sha != manifest["lmdb"]["lock_mdb_sha256"]
    ):
        raise RuntimeError("formal checker inputs changed during full traversal")

    receipt = {
        "format": CHECKER_FORMAT,
        "cache_version": CACHE_VERSION,
        "status": "complete",
        "protocol": _protocol(),
        "source_receipt": source,
        "representation_receipt": representation,
        "smplx_receipt": smplx_info,
        "runtime": runtime,
        "cache_path": str(cache_path),
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_sha256": manifest_sha,
        "cache_data_mdb_sha256": cache_data_sha,
        "entry_aggregate_sha256": aggregate_sha,
        "traversal": {
            "method": "numpy_default_rng_permutation",
            "seed": PERMUTATION_SEED,
            "entries": EXPECTED_ENTRIES,
            "covers_all_entries": True,
            "duplicates": 0,
            "missing": 0,
            "batch_windows": COMPUTE_BATCH_WINDOWS,
            "tail_real_windows": TAIL_REAL_WINDOWS,
            "tail_padding_windows": TAIL_PADDING_WINDOWS,
            "tail_padding_policy": TAIL_PADDING_POLICY,
        },
        "checked_entries": EXPECTED_ENTRIES,
        "mismatch_count": 0,
        "torch_equal_all": True,
        "exact_once": True,
        "finite": True,
        "observed_speaker_ids": sorted(observed_speakers),
        "started_unix": started,
        "completed_unix": time.time(),
        "argv": sys.argv,
    }
    _atomic_json(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checked_entries": receipt["checked_entries"],
                "torch_equal_all": receipt["torch_equal_all"],
                "checker_receipt_sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
