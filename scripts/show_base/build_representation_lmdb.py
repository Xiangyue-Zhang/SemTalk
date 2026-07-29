#!/usr/bin/env python3
"""Build the shared 64-frame SHOW representation-training LMDB."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import lmdb
import numpy as np

sys.dont_write_bytecode = True


FIELDS = ("pose", "contact", "facial", "beta", "trans", "speaker_id")
GLOBAL_FOOT_FIELD = "lower_foot_local"
GLOBAL_FOOT_FASTPATH_CONTRACT = "semtalk_show_global_foot_fastpath_v1"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_SPEAKERS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified_bytes(
    input_path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    if input_path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {input_path}")
    path = input_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA mismatch for {path}: "
            f"{actual} != {expected_sha256}"
        )
    return path, payload


def canonical_payload_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_git_oid(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) not in {40, 64} or any(
        ch not in "0123456789abcdef" for ch in normalized
    ):
        raise ValueError(f"{label} must be a lowercase Git object ID")
    return normalized


def require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{label} must be an exact integer")
    return value


def require_exact_int_mapping(
    value: Any,
    expected: dict[str, int],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimeError(f"{label} must have exactly {sorted(expected)}")
    for key, expected_value in expected.items():
        if (
            require_exact_int(value[key], f"{label}.{key}")
            != expected_value
        ):
            raise RuntimeError(
                f"{label}.{key} must equal {expected_value}"
            )
    return value


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
        raise RuntimeError(
            "representation builder requires a clean source checkout; "
            f"first change: {status.splitlines()[0]}"
        )
    receipt = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
    }
    if receipt["origin"] != EXPECTED_ORIGIN:
        raise RuntimeError(
            f"representation source origin {receipt['origin']!r} "
            f"!= {EXPECTED_ORIGIN!r}"
        )
    if receipt["commit"] != expected_commit:
        raise RuntimeError(
            f"representation source commit {receipt['commit']} "
            f"!= {expected_commit}"
        )
    if receipt["tree"] != expected_tree:
        raise RuntimeError(
            f"representation source tree {receipt['tree']} != {expected_tree}"
        )
    return receipt


def load_canonical_receipts(
    *,
    manifest: Path,
    manifest_sha256: str,
    summary_path: Path,
    lineage_path: Path,
    expected_canonical_source_commit: str,
    expected_canonical_source_tree: str,
    expected_train_clips: int,
) -> dict[str, Any]:
    for path in (manifest, summary_path, lineage_path):
        if path.is_symlink():
            raise RuntimeError(f"canonical receipt must not be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = manifest.resolve()
    summary_path = summary_path.resolve()
    lineage_path = lineage_path.resolve()
    summary = json.loads(summary_path.read_text())
    lineage = json.loads(lineage_path.read_text())
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "complete"
        or summary.get("schema_name") != "semtalk-show-canonical-motion"
        or require_exact_int(
            summary.get("schema_version"),
            "canonical summary schema_version",
        )
        != 1
        or summary.get("manifest_sha256") != manifest_sha256
        or require_exact_int(
            summary.get("split_counts", {}).get("train"),
            "canonical summary train clips",
        )
        != expected_train_clips
        or require_exact_int_mapping(
            summary.get("split_counts"),
            {"train": 13_687, "val": 1_715, "test": 1_708},
            "canonical summary split_counts",
        )
        != {"train": 13_687, "val": 1_715, "test": 1_708}
        or require_exact_int(
            summary.get("clip_count"),
            "canonical summary clip_count",
        )
        != 17_110
        or summary.get("exact_once") is not True
        or summary.get("finite") is not True
        or summary.get("split_disjoint") is not True
    ):
        raise RuntimeError("canonical summary is not a complete formal receipt")
    if (
        not isinstance(lineage, dict)
        or lineage.get("final_manifest_sha256") != manifest_sha256
        or canonical_payload_sha256(lineage) != summary.get("lineage_sha256")
        or lineage.get("lineage_contract_sha256")
        != summary.get("lineage_contract_sha256")
    ):
        raise RuntimeError("canonical lineage/summary binding mismatch")
    source = lineage.get("lineage_contract", {}).get("source_receipt")
    if (
        not isinstance(source, dict)
        or source.get("origin") != EXPECTED_ORIGIN
        or source.get("commit") != expected_canonical_source_commit
        or source.get("tree") != expected_canonical_source_tree
        or summary.get("source_receipt_sha256")
        != canonical_payload_sha256(source)
    ):
        raise RuntimeError("canonical source receipt mismatch")
    return {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "lineage": str(lineage_path),
        "lineage_sha256": sha256(lineage_path),
        "lineage_contract_sha256": summary["lineage_contract_sha256"],
        "source_receipt": source,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(
                f"canonical manifest must not be a symlink: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        hashes[str(resolved)] = sha256(resolved)
        with resolved.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_number}: expected JSON object")
                rows.append(row)
    train_rows = [row for row in rows if row.get("split") == "train"]
    train_rows.sort(key=lambda row: (row["clip_id"], row["canonical_npz"]))
    ids = [row["clip_id"] for row in train_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate train clip_id across canonical manifests")
    return train_rows, hashes


def serialize_window(arrays: dict[str, np.ndarray]) -> bytes:
    with io.BytesIO() as handle:
        np.savez(handle, **arrays)
        return handle.getvalue()


def validate_clip(
    path: Path,
    row: dict[str, Any],
    *,
    enable_global_foot_fastpath: bool,
) -> tuple[dict[str, np.ndarray], int]:
    expected_sha = row.get("canonical_npz_sha256")
    if not isinstance(expected_sha, str):
        raise RuntimeError(f"{path}: missing canonical NPZ SHA")
    path, payload = read_verified_bytes(
        path,
        expected_sha,
        "canonical NPZ",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        missing = sorted(set(FIELDS) - set(archive.files))
        if missing:
            raise RuntimeError(f"{path}: missing fields {missing}")
        arrays = {name: archive[name].copy() for name in FIELDS}
    frames = arrays["pose"].shape[0]
    if enable_global_foot_fastpath:
        if (
            row.get("global_foot_fastpath_contract")
            != GLOBAL_FOOT_FASTPATH_CONTRACT
        ):
            raise RuntimeError(
                f"{path}: missing {GLOBAL_FOOT_FASTPATH_CONTRACT} manifest contract"
            )
        foot_input = Path(str(row.get(GLOBAL_FOOT_FIELD, "")))
        expected_foot_sha = row.get(f"{GLOBAL_FOOT_FIELD}_sha256")
        if not isinstance(expected_foot_sha, str):
            raise RuntimeError(f"{path}: missing Global foot sidecar receipt")
        foot_path, foot_payload = read_verified_bytes(
            foot_input,
            expected_foot_sha,
            "Global foot sidecar",
        )
        arrays[GLOBAL_FOOT_FIELD] = np.load(
            io.BytesIO(foot_payload),
            allow_pickle=False,
        ).copy()
    expected = {
        "pose": (frames, 165),
        "contact": (frames, 4),
        "facial": (frames, 100),
        "beta": (frames, 300),
        "trans": (frames, 3),
        "speaker_id": (frames, 1),
    }
    if enable_global_foot_fastpath:
        expected[GLOBAL_FOOT_FIELD] = (frames, 4, 3)
    for name, shape in expected.items():
        array = arrays[name]
        if array.shape != shape:
            raise RuntimeError(f"{path}: {name} {array.shape} != {shape}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise RuntimeError(f"{path}: non-finite {name}")
    speaker_name = row.get("speaker")
    if not isinstance(speaker_name, str) or speaker_name not in EXPECTED_SPEAKERS:
        raise RuntimeError(f"{path}: invalid manifest SHOW speaker")
    manifest_speaker_id = require_exact_int(
        row.get("speaker_id"),
        f"{path}: manifest speaker_id",
    )
    if manifest_speaker_id != EXPECTED_SPEAKERS[speaker_name]:
        raise RuntimeError(f"{path}: manifest speaker/name mismatch")
    speaker_ids = arrays["speaker_id"]
    if speaker_ids.dtype.kind not in "iu":
        raise RuntimeError(f"{path}: speaker_id is not integer")
    if speaker_ids.min() < 0 or speaker_ids.max() > 3:
        raise RuntimeError(f"{path}: invalid SHOW speaker ID")
    if np.unique(speaker_ids).size != 1:
        raise RuntimeError(f"{path}: speaker ID changes within clip")
    if not np.all(speaker_ids == manifest_speaker_id):
        raise RuntimeError(f"{path}: NPZ/manifest speaker ID mismatch")
    if (
        row.get("frames") is not None
        and require_exact_int(row["frames"], f"{path}: manifest frames")
        != frames
    ):
        raise RuntimeError(f"{path}: frame count disagrees with manifest")
    return arrays, frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--canonical-manifest", action="append", required=True)
    parser.add_argument("--canonical-summary", required=True)
    parser.add_argument("--canonical-lineage", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--map-size-gib", type=int, default=128)
    parser.add_argument("--expected-train-clips", type=int, required=True)
    parser.add_argument("--expected-entries", type=int, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-canonical-source-commit", required=True)
    parser.add_argument("--expected-canonical-source-tree", required=True)
    parser.add_argument(
        "--enable-global-foot-fastpath",
        action="store_true",
        help=(
            "include the trainer-exact lower-foot sidecar required by the "
            "explicit Global-VAE SMPL-X-free formal contract"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_length != 64 or args.stride != 20:
        raise RuntimeError("formal SemTalk representation windows are fixed at 64/20")
    manifests = [Path(path) for path in args.canonical_manifest]
    if len(manifests) != 1:
        raise RuntimeError("formal representation build requires one final manifest")
    canonical_summary_path = Path(args.canonical_summary)
    canonical_lineage_path = Path(args.canonical_lineage)
    args.expected_source_commit = require_git_oid(
        args.expected_source_commit,
        "--expected-source-commit",
    )
    args.expected_source_tree = require_git_oid(
        args.expected_source_tree,
        "--expected-source-tree",
    )
    args.expected_canonical_source_commit = require_git_oid(
        args.expected_canonical_source_commit,
        "--expected-canonical-source-commit",
    )
    args.expected_canonical_source_tree = require_git_oid(
        args.expected_canonical_source_tree,
        "--expected-canonical-source-tree",
    )
    if args.expected_train_clips != 13_687:
        raise RuntimeError("formal SHOW train clip count must be exactly 13687")
    if args.expected_entries != 127_309:
        raise RuntimeError("formal SHOW representation entries must be exactly 127309")
    source = source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    output = Path(args.output_lmdb).resolve()
    summary_path = Path(args.summary_json).resolve()
    if output.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite representation cache outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temp.exists():
        raise FileExistsError(temp)

    rows, manifest_hashes = load_rows(manifests)
    manifest_sha = manifest_hashes[str(manifests[0].resolve())]
    canonical_receipt = load_canonical_receipts(
        manifest=manifests[0],
        manifest_sha256=manifest_sha,
        summary_path=canonical_summary_path,
        lineage_path=canonical_lineage_path,
        expected_canonical_source_commit=(
            args.expected_canonical_source_commit
        ),
        expected_canonical_source_tree=args.expected_canonical_source_tree,
        expected_train_clips=args.expected_train_clips,
    )
    if len(rows) != args.expected_train_clips:
        raise RuntimeError(
            f"representation train clips {len(rows)} "
            f"!= {args.expected_train_clips}"
        )
    observed_speakers = {
        (
            str(row.get("speaker")),
            require_exact_int(
                row.get("speaker_id"),
                f"{row.get('clip_id')}: speaker_id",
            ),
        )
        for row in rows
    }
    if observed_speakers != set(EXPECTED_SPEAKERS.items()):
        raise RuntimeError(
            f"representation speaker coverage {sorted(observed_speakers)} "
            f"!= {sorted(EXPECTED_SPEAKERS.items())}"
        )
    observed_contracts = {
        str(row.get("lineage_contract_sha256")) for row in rows
    }
    if observed_contracts != {
        canonical_receipt["lineage_contract_sha256"]
    }:
        raise RuntimeError("representation rows/canonical lineage mismatch")
    temp.mkdir()
    env = lmdb.open(
        str(temp),
        subdir=True,
        map_size=args.map_size_gib * 1024**3,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=False,
    )
    entries = 0
    skipped_short: list[str] = []
    dropped_tail_frames = 0
    per_clip: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    speaker_clip_counts: Counter[str] = Counter()
    speaker_window_counts: Counter[str] = Counter()
    started = time.time()
    try:
        with env.begin(write=True) as txn:
            for row in rows:
                speaker = str(row["speaker"])
                speaker_clip_counts[speaker] += 1
                path = Path(row["canonical_npz"])
                arrays, frames = validate_clip(
                    path,
                    row,
                    enable_global_foot_fastpath=args.enable_global_foot_fastpath,
                )
                # The public SemTalk loader first converts a clip length to
                # whole seconds (`frames // pose_fps`) before subdivision.
                # Preserve that exact 30-fps boundary rule for training while
                # leaving the canonical clip itself untouched for evaluation.
                usable_frames = (frames // 30) * 30
                dropped_tail = frames - usable_frames
                dropped_tail_frames += dropped_tail
                windows = max(
                    0,
                    (usable_frames - args.window_length) // args.stride + 1,
                )
                if windows == 0:
                    skipped_short.append(row["clip_id"])
                for window_index in range(windows):
                    start = window_index * args.stride
                    end = start + args.window_length
                    item = {
                        "pose": arrays["pose"][start:end].astype(np.float32),
                        "contact": arrays["contact"][start:end].astype(np.float32),
                        "facial": arrays["facial"][start:end].astype(np.float32),
                        "beta": arrays["beta"][start:end].astype(np.float32),
                        "trans": arrays["trans"][start:end].astype(np.float32),
                        "speaker_id": arrays["speaker_id"][start:end].astype(np.int64),
                    }
                    if args.enable_global_foot_fastpath:
                        item[GLOBAL_FOOT_FIELD] = arrays[GLOBAL_FOOT_FIELD][
                            start:end
                        ].astype(np.float32)
                    value = serialize_window(item)
                    key = f"{entries:010d}".encode("ascii")
                    if not txn.put(key, value, overwrite=False):
                        raise RuntimeError(f"duplicate LMDB key {key!r}")
                    aggregate.update(key)
                    aggregate.update(hashlib.sha256(value).digest())
                    entries += 1
                speaker_window_counts[speaker] += windows
                per_clip.append(
                    {
                        "clip_id": row["clip_id"],
                        "canonical_npz": str(path.resolve()),
                        "frames": frames,
                        "usable_frames": usable_frames,
                        "dropped_tail_frames": dropped_tail,
                        "windows": windows,
                    }
                )
        env.sync(True)
    except BaseException:
        env.close()
        shutil.rmtree(temp, ignore_errors=True)
        raise
    env.close()
    if entries <= 0:
        shutil.rmtree(temp)
        raise RuntimeError("representation cache contains no windows")
    if entries != args.expected_entries:
        shutil.rmtree(temp)
        raise RuntimeError(
            f"representation entries {entries} != {args.expected_entries}"
        )
    try:
        final_source = source_receipt(
            args.expected_source_commit,
            args.expected_source_tree,
        )
        final_rows, final_manifest_hashes = load_rows(manifests)
        final_manifest_sha = final_manifest_hashes[
            str(manifests[0].resolve())
        ]
        final_canonical_receipt = load_canonical_receipts(
            manifest=manifests[0],
            manifest_sha256=final_manifest_sha,
            summary_path=canonical_summary_path,
            lineage_path=canonical_lineage_path,
            expected_canonical_source_commit=(
                args.expected_canonical_source_commit
            ),
            expected_canonical_source_tree=(
                args.expected_canonical_source_tree
            ),
            expected_train_clips=args.expected_train_clips,
        )
        if (
            final_source != source
            or final_rows != rows
            or final_manifest_hashes != manifest_hashes
            or final_manifest_sha != manifest_sha
            or final_canonical_receipt != canonical_receipt
        ):
            raise RuntimeError(
                "formal source/canonical receipts changed during "
                "representation cache construction"
            )
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    os.replace(temp, output)

    data_sha = sha256(output / "data.mdb")
    lock_sha = sha256(output / "lock.mdb")
    summary = {
        "format": (
            "semtalk_show_representation_lmdb_v2_global_foot"
            if args.enable_global_foot_fastpath
            else "semtalk_show_representation_lmdb_v1"
        ),
        "status": "complete",
        "protocol": {
            "split": "train",
            "window_length": args.window_length,
            "stride": args.stride,
            "tail_policy": "drop_incomplete",
            "clip_boundary_policy": "floor_to_whole_seconds_at_30fps",
            "filtering": "none",
            "global_foot_fastpath": (
                {
                    "enabled": True,
                    "contract": GLOBAL_FOOT_FASTPATH_CONTRACT,
                    "field": GLOBAL_FOOT_FIELD,
                    "shape": [args.window_length, 4, 3],
                    "dtype": "float32",
                    "activation_env": "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH=1",
                }
                if args.enable_global_foot_fastpath
                else {
                    "enabled": False,
                    "contract": None,
                }
            ),
            "speaker_map": {
                "oliver": 0,
                "chemistry": 1,
                "seth": 2,
                "conan": 3,
            },
        },
        "canonical_manifest_sha256": manifest_hashes,
        "canonical_receipt": canonical_receipt,
        "source_receipt": source,
        "train_clips": len(rows),
        "speaker_clip_counts": {
            speaker: int(speaker_clip_counts[speaker])
            for speaker in EXPECTED_SPEAKERS
        },
        "speaker_window_counts": {
            speaker: int(speaker_window_counts[speaker])
            for speaker in EXPECTED_SPEAKERS
        },
        "entries": entries,
        "dropped_tail_frames": dropped_tail_frames,
        "skipped_short_clip_ids": skipped_short,
        "entry_aggregate_sha256": aggregate.hexdigest(),
        "lmdb": str(output),
        "data_mdb_sha256": data_sha,
        "lock_mdb_sha256": lock_sha,
        "per_clip": per_clip,
        "started_unix": started,
        "completed_unix": time.time(),
        "argv": sys.argv,
    }
    atomic_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in ("status", "entries", "train_clips")}))


if __name__ == "__main__":
    main()
