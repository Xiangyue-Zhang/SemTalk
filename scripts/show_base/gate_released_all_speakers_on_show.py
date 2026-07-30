#!/usr/bin/env python3
"""Audit the official released all-speakers prerequisite models on SHOW.

This entrypoint intentionally has two disjoint phases:

``measure``
    Runs the four official released RVQ models through real encode -> decode
    inference on every deterministic window from the canonical SHOW held-out
    validation and test clips.  It also feeds the decoded lower-body result to
    the released global/root model.  The resulting receipt contains
    measurements only and always has ``authorization=false``.

``decide``
    Is CPU-only.  It verifies a frozen measurement receipt and an independently
    supplied threshold file, then produces a separate accept/reject receipt.
    There are deliberately no built-in quality thresholds.

The implementation is independent of the training/import/inference adapters so
that this cross-domain gate cannot silently change their behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
OFFICIAL_RELEASE_COMMIT = "806b008c97bf51fce203e54109e4c22325253618"
OFFICIAL_RELEASE_TREE = "029deb438330fcaa36377195bad79ffe0f06335c"
OFFICIAL_RELEASE_TRUST_ROOT = {
    "origin": EXPECTED_ORIGIN,
    "commit": OFFICIAL_RELEASE_COMMIT,
    "tree": OFFICIAL_RELEASE_TREE,
    "archive_sha256": (
        "3cbe7a3a923075ad39bcdd41bb88e828fd6161be4c5299c4"
        "e62eedcf20ea5664"
    ),
    "readme_sha256": (
        "27f846e150e8101c1124c3a8bfd026617508554f59ee359c5"
        "5eae825b2edeb1e"
    ),
    "sha256s_sha256": (
        "f7c08cb621f884c7deb0c0cba757fcd08c8b1ad471bab939e"
        "b0d1bd02d4abc1b"
    ),
    "best_run_sha256": (
        "6edaae9f21b7a7164f7457ca93240989fcfa3cb8ea02aa94f"
        "7602c17f9491c9a"
    ),
    "all_speaker_metrics_sha256": (
        "3ecd9586b6eb34eb4a05cdd57f29ed51de50399e02ff476d"
        "f8c4d4c7fb4cc317"
    ),
}

GATE_FORMAT = "semtalk_released_all_speakers_show_cross_domain_gate_v1"
THRESHOLD_FORMAT = "semtalk_released_all_speakers_show_thresholds_v1"

EXPECTED_SPLIT_COUNTS = {"train": 13_687, "val": 1_715, "test": 1_708}
HELD_OUT_SPLITS = ("val", "test")
SHOW_SPEAKERS = {"oliver": 0, "chemistry": 1, "seth": 2, "conan": 3}
WINDOW_LENGTH = 64
WINDOW_STRIDE = 20
FPS = 30
RVQ_LEVELS = 6
CODEBOOK_SIZE = 256

UPPER_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
LOWER_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)

WEIGHT_SPECS: dict[str, dict[str, Any]] = {
    "face": {
        "filename": "rvq_face_600.bin",
        "sha256": "31b04c88456a25f4d57841c0cb507b4c856daccb3875878d06545110a6152127",
        "model": "RVQVAE",
        "dimension": 106,
        "vae_layer": 2,
    },
    "hands": {
        "filename": "rvq_hands_500.bin",
        "sha256": "08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a539b03e436",
        "model": "RVQVAE",
        "dimension": 180,
        "vae_layer": 2,
    },
    "upper": {
        "filename": "rvq_upper_500.bin",
        "sha256": "05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d88d08ac17",
        "model": "RVQVAE",
        "dimension": 78,
        "vae_layer": 2,
    },
    "lower": {
        "filename": "rvq_lower_600.bin",
        "sha256": "2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c171ae4efce8",
        "model": "RVQVAE",
        "dimension": 61,
        "vae_layer": 4,
    },
    "global": {
        "filename": "last_1700_foot.bin",
        "sha256": "6e6f88abd98ccbe2c52102b937067f4ade0aa307d6e1dac8e127e19e0144ee12",
        "model": "VAEConvZero",
        "dimension": 61,
        "vae_layer": 4,
    },
}
OFFICIAL_WEIGHT_RECEIPTS = {
    "base": {
        "filename": "best_semtalk_base.bin",
        "sha256": "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603",
    },
    **{
        name: {
            "filename": spec["filename"],
            "sha256": spec["sha256"],
        }
        for name, spec in WEIGHT_SPECS.items()
    },
}
FORMAL_GATE_KEYS = {
    "format",
    "status",
    "authorization",
    "release_trust_root",
    "official_weights",
    "canonical_receipt",
    "source_roles",
    "protocol",
    "gate_script",
    "measurement_receipt",
    "threshold_receipt",
    "measurements",
    "thresholds",
    "decisions",
    "receipt_sha256",
}
SOURCE_RECEIPT_KEYS = {
    "source_root",
    "origin",
    "commit",
    "tree",
    "clean",
    "script",
    "script_relative",
    "script_sha256",
}
SOURCE_RECEIPT_COMMON_KEYS = {
    "source_root",
    "origin",
    "commit",
    "tree",
    "clean",
}
STAGE_BASELINE_PROTOCOLS = {
    "face": {
        "full_error": "identity_rotation6d_plus_zero_expression",
        "rotation_geodesic_radians": "identity_rotation6d",
        "expression_error": "zero_expression",
    },
    "hands": {
        "full_error": "identity_rotation6d",
        "rotation_geodesic_radians": "identity_rotation6d",
    },
    "upper": {
        "full_error": "identity_rotation6d",
        "rotation_geodesic_radians": "identity_rotation6d",
    },
    "lower": {
        "full_error": (
            "identity_rotation6d_plus_zero_translation_plus_zero_contact"
        ),
        "rotation_geodesic_radians": "identity_rotation6d",
        "translation_error": "zero_translation",
        "contact_error": "zero_contact",
    },
}
GLOBAL_BASELINE_PROTOCOLS = {
    "root_channels_error": "zero_root_channels",
    "integrated_translation_error": "zero_translation",
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


def canonical_payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git object ID")
    return value


def require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def read_verified_bytes(path: Path, expected_sha256: str, label: str) -> tuple[Path, bytes]:
    require_sha256(expected_sha256, f"{label} expected SHA-256")
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = resolved.read_bytes()
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch for {resolved}: {actual} != {expected_sha256}"
        )
    return resolved, payload


def atomic_json_new(path: Path, value: Any) -> None:
    """Write canonical JSON without following or overwriting an existing path."""
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    parent = path.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def add_receipt_payload_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_payload_sha256" in payload:
        raise ValueError("payload already has receipt_payload_sha256")
    result = dict(payload)
    result["receipt_payload_sha256"] = canonical_payload_sha256(result)
    return result


def verify_receipt_payload_hash(payload: Mapping[str, Any], label: str) -> None:
    recorded = require_sha256(
        payload.get("receipt_payload_sha256"),
        f"{label}.receipt_payload_sha256",
    )
    unsigned = dict(payload)
    del unsigned["receipt_payload_sha256"]
    actual = canonical_payload_sha256(unsigned)
    if recorded != actual:
        raise RuntimeError(f"{label} self-hash mismatch: {recorded} != {actual}")


def git_source_receipt(
    expected_commit: str,
    expected_tree: str,
    *,
    script: Path | None = None,
) -> dict[str, Any]:
    """Bind one tracked script to a clean, committed implementation."""
    require_git_oid(expected_commit, "expected source commit")
    require_git_oid(expected_tree, "expected source tree")
    root = Path(__file__).resolve().parents[2]
    script = Path(__file__).resolve() if script is None else script.resolve()
    try:
        script_relative = str(script.relative_to(root))
    except ValueError as error:
        raise RuntimeError(f"source script escapes repository: {script}") from error
    if not script.is_file() or script.is_symlink():
        raise RuntimeError(f"source script is not a regular file: {script}")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "measurement requires a clean source checkout; first change: "
            f"{dirty.splitlines()[0]}"
        )
    tracked = git("ls-files", "--error-unmatch", script_relative)
    if tracked != script_relative:
        raise RuntimeError(f"source script is not tracked by Git: {script_relative}")
    receipt = {
        "source_root": str(root),
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": True,
        "script": str(script),
        "script_relative": script_relative,
        "script_sha256": sha256_file(script),
    }
    if receipt["origin"] != EXPECTED_ORIGIN:
        raise RuntimeError(f"unexpected SemTalk origin: {receipt['origin']!r}")
    if receipt["commit"] != expected_commit:
        raise RuntimeError("measurement source commit mismatch")
    if receipt["tree"] != expected_tree:
        raise RuntimeError("measurement source tree mismatch")
    return receipt


def validate_entrypoint_source_receipt(
    receipt: Any,
    *,
    label: str,
    expected_commit: str,
    expected_tree: str,
    expected_script_relative: str,
) -> None:
    """Revalidate one immutable, tracked entrypoint receipt fail-closed."""
    if not isinstance(receipt, dict) or set(receipt) != SOURCE_RECEIPT_KEYS:
        raise RuntimeError(f"{label} source receipt schema mismatch")
    if (
        receipt.get("origin") != EXPECTED_ORIGIN
        or receipt.get("commit") != expected_commit
        or receipt.get("tree") != expected_tree
        or receipt.get("clean") is not True
        or receipt.get("script_relative") != expected_script_relative
    ):
        raise RuntimeError(f"{label} source receipt binding mismatch")
    require_git_oid(receipt["commit"], f"{label}.commit")
    require_git_oid(receipt["tree"], f"{label}.tree")
    recorded_sha256 = require_sha256(
        receipt.get("script_sha256"),
        f"{label}.script_sha256",
    )
    source_root = Path(receipt.get("source_root", ""))
    script = Path(receipt.get("script", ""))
    if (
        not source_root.is_absolute()
        or source_root != source_root.resolve()
        or not script.is_absolute()
        or script != script.resolve()
        or script != (source_root / expected_script_relative).resolve()
        or not script.is_file()
        or script.is_symlink()
    ):
        raise RuntimeError(f"{label} source path binding mismatch")
    actual_sha256 = sha256_file(script)
    if recorded_sha256 != actual_sha256:
        raise RuntimeError(
            f"{label} source SHA-256 mismatch: "
            f"{recorded_sha256} != {actual_sha256}"
        )


@dataclass
class ErrorAccumulator:
    count: int = 0
    sum_abs: float = 0.0
    sum_squared: float = 0.0
    max_abs: float = 0.0

    def update_differences(self, differences: Iterable[float]) -> None:
        for raw in differences:
            value = require_finite_number(raw, "difference")
            absolute = abs(value)
            self.count += 1
            self.sum_abs += absolute
            self.sum_squared += value * value
            self.max_abs = max(self.max_abs, absolute)

    def update_summary(
        self,
        *,
        count: int,
        sum_abs: float,
        sum_squared: float,
        max_abs: float,
    ) -> None:
        count = require_exact_int(count, "error count")
        if count < 0:
            raise ValueError("error count must not be negative")
        values = [
            require_finite_number(sum_abs, "error sum_abs"),
            require_finite_number(sum_squared, "error sum_squared"),
            require_finite_number(max_abs, "error max_abs"),
        ]
        if any(value < 0 for value in values):
            raise ValueError("error summary values must not be negative")
        self.count += count
        self.sum_abs += values[0]
        self.sum_squared += values[1]
        self.max_abs = max(self.max_abs, values[2])

    def finalize(self) -> dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError("cannot finalize an empty error accumulator")
        result = {
            "count": self.count,
            "mae": self.sum_abs / self.count,
            "rmse": math.sqrt(self.sum_squared / self.count),
            "max_abs": self.max_abs,
        }
        for key in ("mae", "rmse", "max_abs"):
            if not math.isfinite(result[key]):
                raise RuntimeError(f"non-finite aggregate {key}")
        return result


def baseline_comparison(
    model_error: Mapping[str, Any],
    baseline_error: Mapping[str, Any],
    *,
    protocol: str,
) -> dict[str, Any]:
    """Build a self-checking, lower-is-better comparison against a trivial baseline."""
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("baseline protocol must be nonempty")
    expected_error_keys = {"count", "mae", "rmse", "max_abs"}
    if (
        not isinstance(model_error, Mapping)
        or set(model_error) != expected_error_keys
        or not isinstance(baseline_error, Mapping)
        or set(baseline_error) != expected_error_keys
    ):
        raise RuntimeError("model/baseline error summary schema mismatch")
    model_count = require_exact_int(model_error["count"], "model error count")
    baseline_count = require_exact_int(
        baseline_error["count"],
        "baseline error count",
    )
    if model_count <= 0 or model_count != baseline_count:
        raise RuntimeError("model/baseline error counts do not match")
    ratios = {}
    for statistic in ("mae", "rmse"):
        model_value = require_finite_number(
            model_error[statistic],
            f"model {statistic}",
        )
        baseline_value = require_finite_number(
            baseline_error[statistic],
            f"baseline {statistic}",
        )
        if model_value < 0.0 or baseline_value <= 0.0:
            raise RuntimeError(
                f"invalid model/baseline {statistic}: "
                f"{model_value}/{baseline_value}"
            )
        ratio = model_value / baseline_value
        if not math.isfinite(ratio):
            raise RuntimeError(f"non-finite model/baseline {statistic} ratio")
        ratios[statistic] = ratio
    return {
        "protocol": protocol,
        "baseline_error": dict(baseline_error),
        "model_to_baseline_ratio": ratios,
    }


def validate_baseline_comparison(
    comparison: Any,
    model_error: Any,
    *,
    protocol: str,
    label: str,
) -> None:
    if (
        not isinstance(comparison, dict)
        or set(comparison)
        != {"protocol", "baseline_error", "model_to_baseline_ratio"}
        or comparison.get("protocol") != protocol
    ):
        raise RuntimeError(f"{label} baseline comparison schema mismatch")
    expected = baseline_comparison(
        model_error,
        comparison.get("baseline_error"),
        protocol=protocol,
    )
    if comparison != expected:
        raise RuntimeError(f"{label} baseline comparison is not self-consistent")


class CodebookAccumulator:
    def __init__(self, levels: int = RVQ_LEVELS, codebook_size: int = CODEBOOK_SIZE):
        if levels <= 0 or codebook_size <= 1:
            raise ValueError("invalid codebook dimensions")
        self.levels = levels
        self.codebook_size = codebook_size
        self.histograms = [[0] * codebook_size for _ in range(levels)]

    def update_rows(self, rows: Iterable[Sequence[int]]) -> None:
        for row in rows:
            if len(row) != self.levels:
                raise ValueError("RVQ index row has wrong number of levels")
            for level, raw in enumerate(row):
                index = require_exact_int(raw, "RVQ index")
                if index < 0 or index >= self.codebook_size:
                    raise ValueError(f"RVQ index out of range: {index}")
                self.histograms[level][index] += 1

    def finalize(self) -> list[dict[str, Any]]:
        result = []
        for level, histogram in enumerate(self.histograms):
            tokens = sum(histogram)
            if tokens <= 0:
                raise RuntimeError(f"RVQ level {level} has no tokens")
            occupied = sum(count > 0 for count in histogram)
            entropy = 0.0
            for count in histogram:
                if count:
                    probability = count / tokens
                    entropy -= probability * math.log(probability)
            result.append(
                {
                    "level": level,
                    "tokens": tokens,
                    "occupied_codes": occupied,
                    "occupancy_fraction": occupied / self.codebook_size,
                    "dead_fraction": 1.0 - occupied / self.codebook_size,
                    "entropy_nats": entropy,
                    "normalized_entropy": entropy / math.log(self.codebook_size),
                    "histogram": histogram,
                }
            )
        return result


def window_count(frames: int) -> int:
    frames = require_exact_int(frames, "frames")
    if frames <= 0:
        raise ValueError("frames must be positive")
    usable = (frames // FPS) * FPS
    return max(0, (usable - WINDOW_LENGTH) // WINDOW_STRIDE + 1)


def window_records(row: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    global_index = require_exact_int(row.get("global_index"), "global_index")
    frames = require_exact_int(row.get("frames"), "frames")
    clip_id = row.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id:
        raise ValueError("clip_id must be nonempty")
    canonical_sha = require_sha256(
        row.get("canonical_npz_sha256"),
        "canonical_npz_sha256",
    )
    for index in range(window_count(frames)):
        start = index * WINDOW_STRIDE
        yield {
            "global_index": global_index,
            "clip_id": clip_id,
            "start": start,
            "end": start + WINDOW_LENGTH,
            "canonical_npz_sha256": canonical_sha,
        }


def validate_manifest_rows(
    rows: list[dict[str, Any]],
    expected_split_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
) -> None:
    expected_total = sum(expected_split_counts.values())
    if len(rows) != expected_total:
        raise RuntimeError(
            f"canonical manifest has {len(rows)} rows, expected {expected_total}"
        )
    split_counts: Counter[str] = Counter()
    clip_ids: set[str] = set()
    canonical_paths: set[str] = set()
    global_indices: set[int] = set()
    for line_number, row in enumerate(rows, 1):
        split = row.get("split")
        if split not in expected_split_counts:
            raise RuntimeError(f"manifest row {line_number} has invalid split")
        split_counts[split] += 1
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise RuntimeError(f"manifest row {line_number} has invalid clip_id")
        if clip_id in clip_ids:
            raise RuntimeError(f"duplicate clip_id {clip_id!r}")
        clip_ids.add(clip_id)
        global_index = require_exact_int(
            row.get("global_index"),
            f"manifest row {line_number} global_index",
        )
        if global_index < 0 or global_index >= expected_total:
            raise RuntimeError(f"manifest row {line_number} global_index out of range")
        if global_index in global_indices:
            raise RuntimeError(f"duplicate global_index {global_index}")
        global_indices.add(global_index)
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or speaker not in SHOW_SPEAKERS:
            raise RuntimeError(f"manifest row {line_number} has invalid speaker")
        speaker_id = require_exact_int(
            row.get("speaker_id"),
            f"manifest row {line_number} speaker_id",
        )
        if speaker_id != SHOW_SPEAKERS[speaker]:
            raise RuntimeError(f"manifest row {line_number} speaker/name mismatch")
        frames = require_exact_int(
            row.get("frames"),
            f"manifest row {line_number} frames",
        )
        if frames <= 0:
            raise RuntimeError(f"manifest row {line_number} has invalid frames")
        if not isinstance(row.get("canonical_npz"), str) or not row["canonical_npz"]:
            raise RuntimeError(
                f"manifest row {line_number} has invalid canonical_npz"
            )
        if row["canonical_npz"] in canonical_paths:
            raise RuntimeError(
                f"duplicate canonical_npz path {row['canonical_npz']!r}"
            )
        canonical_paths.add(row["canonical_npz"])
        require_sha256(
            row.get("canonical_npz_sha256"),
            f"manifest row {line_number} canonical_npz_sha256",
        )
    if dict(split_counts) != dict(expected_split_counts):
        raise RuntimeError(
            f"manifest split counts {dict(split_counts)} != "
            f"{dict(expected_split_counts)}"
        )
    if global_indices != set(range(expected_total)):
        raise RuntimeError("manifest global_index is not exact-once")


def load_canonical_receipt(
    *,
    manifest_path: Path,
    summary_path: Path,
    lineage_path: Path,
    expected_manifest_sha256: str,
    expected_summary_sha256: str,
    expected_lineage_sha256: str,
    expected_canonical_commit: str,
    expected_canonical_tree: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_git_oid(expected_canonical_commit, "expected canonical commit")
    require_git_oid(expected_canonical_tree, "expected canonical tree")
    paths = {
        "manifest": (manifest_path, expected_manifest_sha256),
        "summary": (summary_path, expected_summary_sha256),
        "lineage": (lineage_path, expected_lineage_sha256),
    }
    verified: dict[str, tuple[Path, bytes]] = {}
    for label, (path, expected) in paths.items():
        verified[label] = read_verified_bytes(path, expected, f"canonical {label}")

    manifest, manifest_payload = verified["manifest"]
    summary_file, summary_payload = verified["summary"]
    lineage_file, lineage_payload = verified["lineage"]
    summary = json.loads(summary_payload)
    lineage = json.loads(lineage_payload)
    if not isinstance(summary, dict) or not isinstance(lineage, dict):
        raise TypeError("canonical summary/lineage must be JSON objects")
    split_counts = summary.get("split_counts")
    valid_split_counts = (
        isinstance(split_counts, dict)
        and set(split_counts) == set(EXPECTED_SPLIT_COUNTS)
        and all(
            require_exact_int(split_counts[split], f"summary split_counts.{split}")
            == count
            for split, count in EXPECTED_SPLIT_COUNTS.items()
        )
    )
    if (
        summary.get("status") != "complete"
        or summary.get("schema_name") != "semtalk-show-canonical-motion"
        or require_exact_int(summary.get("schema_version"), "summary schema_version")
        != 1
        or require_exact_int(summary.get("clip_count"), "summary clip_count")
        != sum(EXPECTED_SPLIT_COUNTS.values())
        or not valid_split_counts
        or summary.get("split_disjoint") is not True
        or summary.get("exact_once") is not True
        or summary.get("finite") is not True
        or summary.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise RuntimeError("invalid canonical SHOW summary")
    if (
        lineage.get("final_manifest_sha256") != expected_manifest_sha256
        or lineage.get("lineage_contract_sha256")
        != summary.get("lineage_contract_sha256")
        or canonical_payload_sha256(lineage) != summary.get("lineage_sha256")
    ):
        raise RuntimeError("canonical lineage binding mismatch")
    canonical_source = lineage.get("lineage_contract", {}).get("source_receipt")
    if (
        not isinstance(canonical_source, dict)
        or canonical_source.get("origin") != EXPECTED_ORIGIN
        or canonical_source.get("commit") != expected_canonical_commit
        or canonical_source.get("tree") != expected_canonical_tree
        or summary.get("source_receipt_sha256")
        != canonical_payload_sha256(canonical_source)
    ):
        raise RuntimeError("canonical source receipt mismatch")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{manifest}:{line_number}: non-object row")
        rows.append(row)
    validate_manifest_rows(rows)
    lineage_contract_sha = summary.get("lineage_contract_sha256")
    for line_number, row in enumerate(rows, 1):
        if row.get("lineage_contract_sha256") != lineage_contract_sha:
            raise RuntimeError(
                f"manifest row {line_number} lineage contract mismatch"
            )

    return rows, {
        "manifest": str(manifest),
        "manifest_sha256": expected_manifest_sha256,
        "summary": str(summary_file),
        "summary_sha256": expected_summary_sha256,
        "lineage": str(lineage_file),
        "lineage_sha256": expected_lineage_sha256,
        "lineage_contract_sha256": lineage_contract_sha,
        "source_receipt": canonical_source,
    }


def _extract_numeric_path(payload: Mapping[str, Any], dotted_path: str) -> float:
    if (
        not isinstance(dotted_path, str)
        or not dotted_path
        or dotted_path.startswith(".")
        or dotted_path.endswith(".")
    ):
        raise ValueError("metric path must be a nonempty dotted path")
    current: Any = payload
    for component in dotted_path.split("."):
        if not component:
            raise ValueError("metric path contains an empty component")
        if isinstance(current, Mapping) and component in current:
            current = current[component]
            continue
        if isinstance(current, list) and component.isdecimal():
            index = int(component)
            if index < len(current):
                current = current[index]
                continue
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(f"unknown metric path {dotted_path!r}")
        current = current[component]
    return require_finite_number(current, f"metric {dotted_path}")


def validate_measurement_contract(
    measurement: Mapping[str, Any],
    *,
    expected_source_commit: str,
    expected_source_tree: str,
) -> None:
    require_git_oid(expected_source_commit, "expected source commit")
    require_git_oid(expected_source_tree, "expected source tree")
    verify_receipt_payload_hash(measurement, "measurement")
    if (
        measurement.get("format") != GATE_FORMAT
        or measurement.get("status") != "measured"
        or measurement.get("authorization") is not False
        or measurement.get("finite") is not True
        or measurement.get("exact_once") is not True
        or measurement.get("deterministic") is not True
    ):
        raise RuntimeError("measurement is not a valid measurement-only gate receipt")
    source = measurement.get("source_receipt")
    validate_entrypoint_source_receipt(
        source,
        label="measurement gate entrypoint",
        expected_commit=expected_source_commit,
        expected_tree=expected_source_tree,
        expected_script_relative=(
            "scripts/show_base/gate_released_all_speakers_on_show.py"
        ),
    )
    if measurement.get("release_trust_root") != OFFICIAL_RELEASE_TRUST_ROOT:
        raise RuntimeError("measurement official release trust root mismatch")
    if measurement.get("official_weights") != OFFICIAL_WEIGHT_RECEIPTS:
        raise RuntimeError("measurement official weight binding mismatch")
    canonical_show = measurement.get("canonical_receipt")
    if (
        not isinstance(canonical_show, dict)
        or set(canonical_show)
        != {
            "manifest",
            "manifest_sha256",
            "summary",
            "summary_sha256",
            "lineage",
            "lineage_sha256",
            "lineage_contract_sha256",
        }
    ):
        raise RuntimeError("measurement canonical SHOW receipt is incomplete")
    for field in ("manifest", "summary", "lineage"):
        if not isinstance(canonical_show.get(field), str) or not canonical_show[field]:
            raise RuntimeError(f"measurement canonical SHOW {field} path is invalid")
        require_sha256(
            canonical_show.get(f"{field}_sha256"),
            f"canonical SHOW {field} SHA-256",
        )
    source_roles = measurement.get("source_roles")
    if (
        not isinstance(source_roles, dict)
        or set(source_roles) != {"current", "gate", "canonical", "input_artifact"}
        or source_roles.get("canonical")
        != measurement.get("canonical_source_receipt")
        or source_roles.get("gate") != source
    ):
        raise RuntimeError("measurement source role binding mismatch")
    current_source = source_roles["current"]
    gate_source = source_roles["gate"]
    validate_entrypoint_source_receipt(
        current_source,
        label="measurement inference entrypoint",
        expected_commit=expected_source_commit,
        expected_tree=expected_source_tree,
        expected_script_relative="scripts/show_base/run_base_inference.py",
    )
    validate_entrypoint_source_receipt(
        gate_source,
        label="measurement gate role entrypoint",
        expected_commit=expected_source_commit,
        expected_tree=expected_source_tree,
        expected_script_relative=(
            "scripts/show_base/gate_released_all_speakers_on_show.py"
        ),
    )
    if any(
        current_source[key] != gate_source[key]
        for key in SOURCE_RECEIPT_COMMON_KEYS
    ):
        raise RuntimeError("measurement entrypoints do not share one source tree")
    if any(
        current_source[key] == gate_source[key]
        for key in ("script", "script_relative", "script_sha256")
    ):
        raise RuntimeError("measurement entrypoints are not distinct")
    input_artifact_source = source_roles.get("input_artifact")
    if (
        not isinstance(input_artifact_source, dict)
        or set(input_artifact_source)
        != {"format", "origin", "commit", "tree"}
        or input_artifact_source.get("format")
        != "semtalk_show_input_artifact_source_v1"
        or input_artifact_source.get("origin") != EXPECTED_ORIGIN
    ):
        raise RuntimeError("measurement input artifact source role mismatch")
    require_git_oid(
        input_artifact_source.get("commit"),
        "measurement input artifact commit",
    )
    require_git_oid(
        input_artifact_source.get("tree"),
        "measurement input artifact tree",
    )
    weights = measurement.get("verified_prerequisite_weights")
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_SPECS):
        raise RuntimeError("measurement weight coverage mismatch")
    for name, spec in WEIGHT_SPECS.items():
        weight = weights[name]
        if (
            not isinstance(weight, dict)
            or weight.get("filename") != spec["filename"]
            or weight.get("sha256") != spec["sha256"]
            or weight.get("model") != spec["model"]
            or require_exact_int(weight.get("dimension"), f"weights.{name}.dimension")
            != spec["dimension"]
            or require_exact_int(weight.get("vae_layer"), f"weights.{name}.vae_layer")
            != spec["vae_layer"]
            or require_exact_int(weight.get("vae_length"), f"weights.{name}.vae_length")
            != 256
        ):
            raise RuntimeError(f"measurement {name} weight contract mismatch")
    protocol = measurement.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("splits") != list(HELD_OUT_SPLITS)
        or protocol.get("window_length") != WINDOW_LENGTH
        or protocol.get("window_stride") != WINDOW_STRIDE
        or protocol.get("fps") != FPS
        or protocol.get("rvq_operation") != "real map2index -> decode"
        or protocol.get("global_input") != "decoded lower"
        or protocol.get("thresholds") is not None
        or protocol.get("trivial_baselines")
        != {
            "stage": STAGE_BASELINE_PROTOCOLS,
            "global": GLOBAL_BASELINE_PROTOCOLS,
            "extra_model_forwards": 0,
        }
    ):
        raise RuntimeError("measurement protocol mismatch")
    coverage = measurement.get("coverage")
    if (
        not isinstance(coverage, dict)
        or require_exact_int(coverage.get("clip_count"), "coverage.clip_count")
        != sum(EXPECTED_SPLIT_COUNTS[split] for split in HELD_OUT_SPLITS)
        or coverage.get("split_clip_counts")
        != {split: EXPECTED_SPLIT_COUNTS[split] for split in HELD_OUT_SPLITS}
        or require_exact_int(
            coverage.get("expected_windows"), "coverage.expected_windows"
        )
        != require_exact_int(
            coverage.get("observed_windows"), "coverage.observed_windows"
        )
        or coverage.get("window_exact_once") is not True
    ):
        raise RuntimeError("measurement coverage mismatch")
    require_sha256(coverage.get("window_records_sha256"), "window records SHA-256")
    determinism = measurement.get("determinism")
    if (
        not isinstance(determinism, dict)
        or determinism.get("torch_deterministic_algorithms") is not True
        or determinism.get("cudnn_benchmark") is not False
        or determinism.get("tf32") is not False
        or determinism.get("cublas_workspace_config") not in {":4096:8", ":16:8"}
        or require_exact_int(determinism.get("batches"), "determinism.batches")
        <= 0
        or determinism.get("batches") != determinism.get("replayed_batches")
        or determinism.get("full_batch_exact_replay") is not True
    ):
        raise RuntimeError("measurement determinism contract mismatch")
    require_sha256(determinism.get("rvq_indices_sha256"), "RVQ indices SHA-256")
    stages = measurement.get("stages")
    if not isinstance(stages, dict) or set(stages) != {"face", "hands", "upper", "lower"}:
        raise RuntimeError("measurement stage coverage mismatch")
    for name, stage in stages.items():
        if (
            not isinstance(stage, dict)
            or stage.get("input_finite") is not True
            or stage.get("reconstruction_finite") is not True
            or not isinstance(stage.get("codebook"), list)
            or len(stage["codebook"]) != RVQ_LEVELS
        ):
            raise RuntimeError(f"measurement stage {name} is incomplete")
        baselines = stage.get("baselines")
        protocols = STAGE_BASELINE_PROTOCOLS[name]
        if not isinstance(baselines, dict) or set(baselines) != set(protocols):
            raise RuntimeError(f"measurement stage {name} baselines are incomplete")
        for metric, baseline_protocol in protocols.items():
            validate_baseline_comparison(
                baselines[metric],
                stage.get(metric),
                protocol=baseline_protocol,
                label=f"measurement stage {name}.{metric}",
            )
    global_result = measurement.get("global")
    if (
        not isinstance(global_result, dict)
        or global_result.get("input_finite") is not True
        or global_result.get("output_finite") is not True
        or not isinstance(global_result.get("input_semantics"), str)
        or not global_result["input_semantics"].startswith("decoded lower;")
    ):
        raise RuntimeError("measurement global result is incomplete")
    global_baselines = global_result.get("baselines")
    if (
        not isinstance(global_baselines, dict)
        or set(global_baselines) != set(GLOBAL_BASELINE_PROTOCOLS)
    ):
        raise RuntimeError("measurement global baselines are incomplete")
    for metric, baseline_protocol in GLOBAL_BASELINE_PROTOCOLS.items():
        validate_baseline_comparison(
            global_baselines[metric],
            global_result.get(metric),
            protocol=baseline_protocol,
            label=f"measurement global.{metric}",
        )


def evaluate_thresholds(
    measurement: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    measurement_file_sha256: str,
) -> list[dict[str, Any]]:
    if set(thresholds) != {
        "format",
        "measurement_format",
        "measurement_sha256",
        "provenance",
        "rules",
    }:
        raise RuntimeError("threshold file has unexpected/missing top-level fields")
    if thresholds.get("format") != THRESHOLD_FORMAT:
        raise RuntimeError("unexpected threshold format")
    if thresholds.get("measurement_format") != GATE_FORMAT:
        raise RuntimeError("threshold measurement format mismatch")
    if thresholds.get("measurement_sha256") != measurement_file_sha256:
        raise RuntimeError("threshold file is not bound to this measurement")
    provenance = thresholds.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "basis",
        "authorized_by",
    }:
        raise RuntimeError("threshold provenance must be an object")
    for field in ("basis", "authorized_by"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise RuntimeError(f"threshold provenance.{field} must be nonempty")
    rules = thresholds.get("rules")
    if not isinstance(rules, list) or not rules:
        raise RuntimeError("threshold rules must be a nonempty list")
    operators = {
        "<": lambda actual, expected: actual < expected,
        "<=": lambda actual, expected: actual <= expected,
        ">": lambda actual, expected: actual > expected,
        ">=": lambda actual, expected: actual >= expected,
        "==": lambda actual, expected: actual == expected,
    }
    seen: set[str] = set()
    results = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or set(rule) != {"metric", "operator", "value"}:
            raise RuntimeError(f"threshold rule {index} has invalid fields")
        metric = rule["metric"]
        if not isinstance(metric, str) or metric in seen:
            raise RuntimeError(f"threshold rule {index} has duplicate/invalid metric")
        seen.add(metric)
        operator = rule["operator"]
        if operator not in operators:
            raise RuntimeError(f"threshold rule {index} has invalid operator")
        expected = require_finite_number(rule["value"], f"threshold rule {index} value")
        actual = _extract_numeric_path(measurement, metric)
        results.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": expected,
                "actual": actual,
                "passed": bool(operators[operator](actual, expected)),
            }
        )
    return results


def _torch_error_update(accumulator: ErrorAccumulator, difference: Any) -> None:
    absolute = difference.abs()
    accumulator.update_summary(
        count=int(difference.numel()),
        sum_abs=float(absolute.double().sum().item()),
        sum_squared=float(difference.double().square().sum().item()),
        max_abs=float(absolute.max().item()),
    )


def _identity_rotation6d_like(torch: Any, rotations: Any) -> Any:
    if rotations.shape[-1] != 6:
        raise ValueError("rotation-6D tensor must end in dimension 6")
    identity = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=rotations.dtype,
        device=rotations.device,
    )
    return identity.view(*([1] * (rotations.ndim - 1)), 6).expand_as(rotations)


def _rotation_geodesic_error(torch: Any, rc: Any, target: Any, reconstruction: Any) -> Any:
    if target.shape != reconstruction.shape or target.shape[-1] != 6:
        raise ValueError("rotation tensors must have matching (..., 6) shape")
    target_matrix = rc.rotation_6d_to_matrix(target)
    reconstruction_matrix = rc.rotation_6d_to_matrix(reconstruction)
    relative = reconstruction_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    return torch.acos(cosine)


def _central_velocity(torch: Any, translation: Any) -> Any:
    """Match utils.other_tools.estimate_linear_velocity at dt=1/30."""
    dt = 1.0 / FPS
    velocity = torch.zeros_like(translation)
    velocity[:, 1:-1] = (translation[:, 2:] - translation[:, :-2]) / (2.0 * dt)
    velocity[:, 0] = (translation[:, 1] - translation[:, 0]) / dt
    velocity[:, -1] = (translation[:, -1] - translation[:, -2]) / dt
    return velocity


def _integrate_xz(torch: Any, channels: Any, anchor: Any) -> Any:
    """Match the public velocity2position recurrence for x/z and raw y."""
    if channels.shape[-1] != 3 or anchor.shape[-1] != 3:
        raise ValueError("translation channels/anchor must end in dimension 3")
    result = torch.zeros_like(channels)
    result[..., 1] = channels[..., 1]
    result[:, 0, 0] = anchor[:, 0]
    result[:, 0, 2] = anchor[:, 2]
    dt = 1.0 / FPS
    for frame in range(1, channels.shape[1]):
        result[:, frame, 0] = result[:, frame - 1, 0] + channels[:, frame - 1, 0] * dt
        result[:, frame, 2] = result[:, frame - 1, 2] + channels[:, frame - 1, 2] * dt
    return result


def _state_dict_from_payload(torch: Any, payload: bytes, label: str) -> dict[str, Any]:
    import io

    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state"), dict
    ):
        raise RuntimeError(f"{label} checkpoint lacks model_state")
    state: dict[str, Any] = {}
    for raw_key, value in checkpoint["model_state"].items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            raise RuntimeError(f"{label} has an invalid state entry")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in state:
            raise RuntimeError(f"{label} has a duplicate normalized state key {key!r}")
        if (
            (value.is_floating_point() or value.is_complex())
            and not bool(value.isfinite().all().item())
        ):
            raise RuntimeError(f"{label} state tensor {key!r} contains NaN/Inf")
        state[key] = value
    return state


def _load_models(torch: Any, weights_root: Path, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from models.motion_representation import VAEConvZero
    from models.rvq import RVQVAE

    models: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    for name, spec in WEIGHT_SPECS.items():
        path, payload = read_verified_bytes(
            weights_root / spec["filename"],
            spec["sha256"],
            f"official released {name} weight",
        )
        namespace = SimpleNamespace(
            vae_test_dim=spec["dimension"],
            vae_layer=spec["vae_layer"],
            vae_length=256,
        )
        model = RVQVAE(namespace) if spec["model"] == "RVQVAE" else VAEConvZero(namespace)
        state = _state_dict_from_payload(torch, payload, f"official released {name}")
        expected_keys = set(model.state_dict())
        actual_keys = set(state)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise RuntimeError(
                f"{name} state schema mismatch; missing={missing[:3]}, extra={extra[:3]}"
            )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
        receipts[name] = {
            "path": str(path),
            "filename": spec["filename"],
            "sha256": spec["sha256"],
            "model": spec["model"],
            "dimension": spec["dimension"],
            "vae_layer": spec["vae_layer"],
            "vae_length": 256,
            "rvq_levels": RVQ_LEVELS if spec["model"] == "RVQVAE" else None,
            "codebook_size": CODEBOOK_SIZE if spec["model"] == "RVQVAE" else None,
        }
    return models, receipts


def _load_canonical_npz(np: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    path, payload = read_verified_bytes(
        Path(row["canonical_npz"]),
        row["canonical_npz_sha256"],
        f"canonical clip {row['clip_id']}",
    )
    import io

    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {
            "pose",
            "contact",
            "facial",
            "beta",
            "trans",
            "speaker_id",
        }:
            raise RuntimeError(f"canonical NPZ field mismatch: {path}")
        arrays = {key: archive[key] for key in archive.files}
    frames = require_exact_int(row["frames"], "canonical row frames")
    expected = {
        "pose": ((frames, 165), np.float32),
        "contact": ((frames, 4), np.float32),
        "facial": ((frames, 100), np.float32),
        "beta": ((frames, 300), np.float32),
        "trans": ((frames, 3), np.float32),
        "speaker_id": ((frames, 1), np.int64),
    }
    for key, (shape, dtype) in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                f"canonical {key} schema mismatch for {path}: "
                f"{value.shape}/{value.dtype} != {shape}/{dtype}"
            )
        if key != "speaker_id" and not bool(np.isfinite(value).all()):
            raise RuntimeError(f"canonical {key} contains NaN/Inf: {path}")
    return arrays


def _batch_iterator(
    np: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    arrays: dict[str, list[Any]] = {
        "pose": [],
        "contact": [],
        "facial": [],
        "trans": [],
    }
    for row in rows:
        canonical = _load_canonical_npz(np, row)
        for record in window_records(row):
            start, end = record["start"], record["end"]
            records.append(record)
            for key in arrays:
                arrays[key].append(canonical[key][start:end])
            if len(records) == batch_size:
                yield records, {key: np.stack(value) for key, value in arrays.items()}
                records = []
                arrays = {key: [] for key in arrays}
    if records:
        yield records, {key: np.stack(value) for key, value in arrays.items()}


def _build_features(torch: Any, rc: Any, batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    pose = torch.from_numpy(batch["pose"]).to(device=device)
    contact = torch.from_numpy(batch["contact"]).to(device=device)
    facial = torch.from_numpy(batch["facial"]).to(device=device)
    translation = torch.from_numpy(batch["trans"]).to(device=device)
    batch_size, frames, _ = pose.shape
    pose_joints = pose.reshape(batch_size, frames, 55, 3)

    jaw = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(pose_joints[:, :, 22])
    )
    hands = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(pose_joints[:, :, 25:55])
    ).reshape(batch_size, frames, 180)
    upper = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(pose_joints[:, :, UPPER_JOINTS])
    ).reshape(batch_size, frames, 78)
    lower_rotation = rc.matrix_to_rotation_6d(
        rc.axis_angle_to_matrix(pose_joints[:, :, LOWER_JOINTS])
    ).reshape(batch_size, frames, 54)
    return {
        "face": torch.cat([jaw, facial], dim=-1),
        "hands": hands,
        "upper": upper,
        "lower": torch.cat([lower_rotation, translation, contact], dim=-1),
        "translation": translation,
        "contact": contact,
    }


def _decode_rvq_checked(
    torch: Any,
    model: Any,
    value: Any,
    name: str,
) -> tuple[Any, Any]:
    indices = model.map2index(value)
    expected_tokens = WINDOW_LENGTH // 4
    expected_shape = (value.shape[0], expected_tokens, RVQ_LEVELS)
    if tuple(indices.shape) != expected_shape or indices.dtype != torch.long:
        raise RuntimeError(
            f"{name} RVQ indices {tuple(indices.shape)}/{indices.dtype} "
            f"!= {expected_shape}/torch.long"
        )
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= CODEBOOK_SIZE:
        raise RuntimeError(f"{name} RVQ index out of range")
    reconstruction = model.decode(indices)
    if tuple(reconstruction.shape) != tuple(value.shape):
        raise RuntimeError(f"{name} RVQ reconstruction shape mismatch")
    if not bool(reconstruction.isfinite().all().item()):
        raise RuntimeError(f"{name} RVQ reconstruction contains NaN/Inf")
    return indices, reconstruction


def _fresh_stage_metrics() -> dict[str, Any]:
    return {
        name: {
            "input_elements": 0,
            "reconstruction_elements": 0,
            "full_error": ErrorAccumulator(),
            "rotation_geodesic_radians": ErrorAccumulator(),
            "baseline_full_error": ErrorAccumulator(),
            "baseline_rotation_geodesic_radians": ErrorAccumulator(),
            "codebook": CodebookAccumulator(),
        }
        for name in ("face", "hands", "upper", "lower")
    }


def _finish_stage_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, stage in metrics.items():
        entry = {
            "input_finite": True,
            "reconstruction_finite": True,
            "input_elements": stage["input_elements"],
            "reconstruction_elements": stage["reconstruction_elements"],
            "full_error": stage["full_error"].finalize(),
            "rotation_geodesic_radians": stage[
                "rotation_geodesic_radians"
            ].finalize(),
            "codebook": stage["codebook"].finalize(),
        }
        for optional in (
            "jaw_geodesic_radians",
            "expression_error",
            "translation_error",
            "contact_error",
        ):
            if optional in stage:
                entry[optional] = stage[optional].finalize()
        if "contact_correct" in stage:
            entry["contact_binary_accuracy"] = (
                stage["contact_correct"] / stage["contact_total"]
            )
            entry["contact_binary_count"] = stage["contact_total"]
        entry["baselines"] = {
            metric: baseline_comparison(
                entry[metric],
                stage[f"baseline_{metric}"].finalize(),
                protocol=protocol,
            )
            for metric, protocol in STAGE_BASELINE_PROTOCOLS[name].items()
        }
        result[name] = entry
    return result


def _measure(args: argparse.Namespace) -> int:
    # Heavy dependencies are deliberately lazy so ``decide`` stays CPU/stdlib-only.
    import numpy as np
    import torch
    from utils import rotation_conversions as rc

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    source = git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
    )
    inference_source = git_source_receipt(
        args.expected_source_commit,
        args.expected_source_tree,
        script=Path(__file__).resolve().with_name("run_base_inference.py"),
    )
    input_artifact_source = {
        "format": "semtalk_show_input_artifact_source_v1",
        "origin": EXPECTED_ORIGIN,
        "commit": require_git_oid(
            args.expected_input_source_commit,
            "expected input source commit",
        ),
        "tree": require_git_oid(
            args.expected_input_source_tree,
            "expected input source tree",
        ),
    }
    rows, canonical_receipt = load_canonical_receipt(
        manifest_path=args.canonical_manifest,
        summary_path=args.canonical_summary,
        lineage_path=args.canonical_lineage,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_summary_sha256=args.expected_summary_sha256,
        expected_lineage_sha256=args.expected_lineage_sha256,
        expected_canonical_commit=args.expected_canonical_commit,
        expected_canonical_tree=args.expected_canonical_tree,
    )
    held_out = sorted(
        (row for row in rows if row["split"] in HELD_OUT_SPLITS),
        key=lambda row: row["global_index"],
    )
    if Counter(row["split"] for row in held_out) != Counter(
        {split: EXPECTED_SPLIT_COUNTS[split] for split in HELD_OUT_SPLITS}
    ):
        raise RuntimeError("held-out split coverage mismatch")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "official RVQVAE construction uses CUDA-resident codebook buffers; "
            "run measurement on CUDA (the decide phase remains stdlib/CPU-only)"
        )
    workspace_config = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )
    if workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError("unsupported CUBLAS_WORKSPACE_CONFIG for deterministic gate")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.index is not None:
        torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    models, weight_receipts = _load_models(torch, args.weights_root, device)
    stage_metrics = _fresh_stage_metrics()
    stage_metrics["face"]["jaw_geodesic_radians"] = ErrorAccumulator()
    stage_metrics["face"]["expression_error"] = ErrorAccumulator()
    stage_metrics["face"]["baseline_expression_error"] = ErrorAccumulator()
    stage_metrics["lower"]["translation_error"] = ErrorAccumulator()
    stage_metrics["lower"]["baseline_translation_error"] = ErrorAccumulator()
    stage_metrics["lower"]["contact_error"] = ErrorAccumulator()
    stage_metrics["lower"]["baseline_contact_error"] = ErrorAccumulator()
    stage_metrics["lower"]["contact_correct"] = 0
    stage_metrics["lower"]["contact_total"] = 0
    global_metrics = {
        "input_elements": 0,
        "output_elements": 0,
        "rotation_geodesic_radians": ErrorAccumulator(),
        "velocity_xz_error": ErrorAccumulator(),
        "height_y_error": ErrorAccumulator(),
        "root_channels_error": ErrorAccumulator(),
        "integrated_translation_error": ErrorAccumulator(),
        "contact_error": ErrorAccumulator(),
        **{
            f"baseline_{metric}": ErrorAccumulator()
            for metric in GLOBAL_BASELINE_PROTOCOLS
        },
    }

    expected_windows = sum(window_count(row["frames"]) for row in held_out)
    seen_windows: set[tuple[int, int]] = set()
    window_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    batches = 0
    replayed_batches = 0
    with torch.inference_mode():
        for records, batch in _batch_iterator(np, held_out, args.batch_size):
            batches += 1
            for record in records:
                identifier = (record["global_index"], record["start"])
                if identifier in seen_windows:
                    raise RuntimeError(f"duplicate held-out window {identifier}")
                seen_windows.add(identifier)
                window_digest.update(canonical_json_bytes(record))
            features = _build_features(torch, rc, batch, device)
            for name in ("face", "hands", "upper", "lower"):
                if not bool(features[name].isfinite().all().item()):
                    raise RuntimeError(f"{name} input contains NaN/Inf")

            first: dict[str, tuple[Any, Any]] = {}
            second: dict[str, tuple[Any, Any]] = {}
            for name in ("face", "hands", "upper", "lower"):
                first[name] = _decode_rvq_checked(
                    torch, models[name], features[name], name
                )
                second[name] = _decode_rvq_checked(
                    torch, models[name], features[name], name
                )
                if not torch.equal(first[name][0], second[name][0]):
                    raise RuntimeError(f"{name} RVQ indices are non-deterministic")
                if not torch.equal(first[name][1], second[name][1]):
                    raise RuntimeError(f"{name} RVQ reconstruction is non-deterministic")
                indices, reconstruction = first[name]
                index_digest.update(indices.detach().cpu().contiguous().numpy().tobytes())
                stage = stage_metrics[name]
                stage["input_elements"] += int(features[name].numel())
                stage["reconstruction_elements"] += int(reconstruction.numel())
                _torch_error_update(stage["full_error"], reconstruction - features[name])
                rotation_width = {"face": 6, "hands": 180, "upper": 78, "lower": 54}[name]
                target_rotations = features[name][..., :rotation_width].reshape(
                    *features[name].shape[:2], -1, 6
                )
                identity_rotations = _identity_rotation6d_like(
                    torch,
                    target_rotations,
                )
                trivial_full = torch.zeros_like(features[name])
                trivial_full[..., :rotation_width] = identity_rotations.reshape(
                    *features[name].shape[:2],
                    rotation_width,
                )
                _torch_error_update(
                    stage["baseline_full_error"],
                    trivial_full - features[name],
                )
                geodesic = _rotation_geodesic_error(
                    torch,
                    rc,
                    target_rotations,
                    reconstruction[..., :rotation_width].reshape(
                        *reconstruction.shape[:2], -1, 6
                    ),
                )
                _torch_error_update(stage["rotation_geodesic_radians"], geodesic)
                _torch_error_update(
                    stage["baseline_rotation_geodesic_radians"],
                    _rotation_geodesic_error(
                        torch,
                        rc,
                        target_rotations,
                        identity_rotations,
                    ),
                )
                stage["codebook"].update_rows(
                    indices.detach().cpu().reshape(-1, RVQ_LEVELS).tolist()
                )

            face_reconstruction = first["face"][1]
            _torch_error_update(
                stage_metrics["face"]["jaw_geodesic_radians"],
                _rotation_geodesic_error(
                    torch,
                    rc,
                    features["face"][..., :6],
                    face_reconstruction[..., :6],
                ),
            )
            _torch_error_update(
                stage_metrics["face"]["expression_error"],
                face_reconstruction[..., 6:] - features["face"][..., 6:],
            )
            _torch_error_update(
                stage_metrics["face"]["baseline_expression_error"],
                -features["face"][..., 6:],
            )
            lower_reconstruction = first["lower"][1]
            _torch_error_update(
                stage_metrics["lower"]["translation_error"],
                lower_reconstruction[..., 54:57] - features["lower"][..., 54:57],
            )
            _torch_error_update(
                stage_metrics["lower"]["baseline_translation_error"],
                -features["lower"][..., 54:57],
            )
            _torch_error_update(
                stage_metrics["lower"]["contact_error"],
                lower_reconstruction[..., 57:61] - features["lower"][..., 57:61],
            )
            _torch_error_update(
                stage_metrics["lower"]["baseline_contact_error"],
                -features["lower"][..., 57:61],
            )
            contact_prediction = lower_reconstruction[..., 57:61] >= 0.5
            contact_target = features["lower"][..., 57:61] >= 0.5
            stage_metrics["lower"]["contact_correct"] += int(
                (contact_prediction == contact_target).sum().item()
            )
            stage_metrics["lower"]["contact_total"] += int(contact_target.numel())

            projected_lower = lower_reconstruction.clone()
            projected_lower[..., :54] = rc.matrix_to_rotation_6d(
                rc.rotation_6d_to_matrix(
                    lower_reconstruction[..., :54].reshape(
                        *lower_reconstruction.shape[:2], 9, 6
                    )
                )
            ).reshape(*lower_reconstruction.shape[:2], 54)
            projected_lower[..., 54:57] = 0.0
            if not bool(projected_lower.isfinite().all().item()):
                raise RuntimeError("global/root decoded-lower input contains NaN/Inf")
            global_first = models["global"](projected_lower).get("rec_pose")
            global_second = models["global"](projected_lower).get("rec_pose")
            if (
                global_first is None
                or tuple(global_first.shape) != tuple(projected_lower.shape)
                or not bool(global_first.isfinite().all().item())
            ):
                raise RuntimeError("global/root output is invalid")
            if not torch.equal(global_first, global_second):
                raise RuntimeError("global/root reconstruction is non-deterministic")
            replayed_batches += 1
            global_metrics["input_elements"] += int(projected_lower.numel())
            global_metrics["output_elements"] += int(global_first.numel())
            global_target_rotations = features["lower"][..., :54].reshape(
                *features["lower"].shape[:2], 9, 6
            )
            _torch_error_update(
                global_metrics["rotation_geodesic_radians"],
                _rotation_geodesic_error(
                    torch,
                    rc,
                    global_target_rotations,
                    global_first[..., :54].reshape(
                        *global_first.shape[:2], 9, 6
                    ),
                ),
            )
            target_velocity = _central_velocity(torch, features["translation"])
            target_root_channels = torch.stack(
                [
                    target_velocity[..., 0],
                    features["translation"][..., 1],
                    target_velocity[..., 2],
                ],
                dim=-1,
            )
            trivial_root_channels = torch.zeros_like(target_root_channels)
            _torch_error_update(
                global_metrics["velocity_xz_error"],
                global_first[..., (54, 56)] - target_root_channels[..., (0, 2)],
            )
            _torch_error_update(
                global_metrics["height_y_error"],
                global_first[..., 55] - target_root_channels[..., 1],
            )
            _torch_error_update(
                global_metrics["root_channels_error"],
                global_first[..., 54:57] - target_root_channels,
            )
            _torch_error_update(
                global_metrics["baseline_root_channels_error"],
                trivial_root_channels - target_root_channels,
            )
            integrated = _integrate_xz(
                torch,
                global_first[..., 54:57],
                features["translation"][:, 0],
            )
            _torch_error_update(
                global_metrics["integrated_translation_error"],
                integrated - features["translation"],
            )
            _torch_error_update(
                global_metrics["baseline_integrated_translation_error"],
                -features["translation"],
            )
            _torch_error_update(
                global_metrics["contact_error"],
                global_first[..., 57:61] - features["contact"],
            )

    if len(seen_windows) != expected_windows:
        raise RuntimeError(
            f"held-out window coverage {len(seen_windows)} != {expected_windows}"
        )
    if batches <= 0 or replayed_batches != batches:
        raise RuntimeError("determinism replay coverage mismatch")

    finished_global = {
        "input_semantics": (
            "decoded lower; first 54 rotation-6D channels projected through "
            "matrix; channels 54:57 zeroed; decoded contacts 57:61 preserved"
        ),
        "input_finite": True,
        "output_finite": True,
        "input_elements": global_metrics["input_elements"],
        "output_elements": global_metrics["output_elements"],
        **{
            key: value.finalize()
            for key, value in global_metrics.items()
            if isinstance(value, ErrorAccumulator)
            and not key.startswith("baseline_")
        },
    }
    finished_global["baselines"] = {
        metric: baseline_comparison(
            finished_global[metric],
            global_metrics[f"baseline_{metric}"].finalize(),
            protocol=protocol,
        )
        for metric, protocol in GLOBAL_BASELINE_PROTOCOLS.items()
    }
    measurement = add_receipt_payload_hash(
        {
            "format": GATE_FORMAT,
            "status": "measured",
            "authorization": False,
            "finite": True,
            "exact_once": True,
            "deterministic": True,
            "authorization_reason": (
                "measurement-only receipt; no quality thresholds were supplied"
            ),
            "source_receipt": source,
            "release_trust_root": OFFICIAL_RELEASE_TRUST_ROOT,
            "official_weights": OFFICIAL_WEIGHT_RECEIPTS,
            "verified_prerequisite_weights": weight_receipts,
            "canonical_receipt": {
                key: canonical_receipt[key]
                for key in (
                    "manifest",
                    "manifest_sha256",
                    "summary",
                    "summary_sha256",
                    "lineage",
                    "lineage_sha256",
                    "lineage_contract_sha256",
                )
            },
            "canonical_source_receipt": canonical_receipt["source_receipt"],
            "source_roles": {
                "current": inference_source,
                "gate": source,
                "canonical": canonical_receipt["source_receipt"],
                "input_artifact": input_artifact_source,
            },
            "protocol": {
                "splits": list(HELD_OUT_SPLITS),
                "split_policy": "canonical held-out val+test only",
                "fps": FPS,
                "usable_frames": "leading floor(frames/30)*30 frames",
                "window_length": WINDOW_LENGTH,
                "window_stride": WINDOW_STRIDE,
                "rvq_operation": "real map2index -> decode",
                "global_input": "decoded lower",
                "trivial_baselines": {
                    "stage": STAGE_BASELINE_PROTOCOLS,
                    "global": GLOBAL_BASELINE_PROTOCOLS,
                    "extra_model_forwards": 0,
                },
                "thresholds": None,
            },
            "coverage": {
                "clip_count": len(held_out),
                "split_clip_counts": dict(Counter(row["split"] for row in held_out)),
                "expected_windows": expected_windows,
                "observed_windows": len(seen_windows),
                "window_exact_once": True,
                "window_records_sha256": window_digest.hexdigest(),
            },
            "determinism": {
                "seed": args.seed,
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "tf32": False,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "batches": batches,
                "replayed_batches": replayed_batches,
                "full_batch_exact_replay": True,
                "rvq_indices_sha256": index_digest.hexdigest(),
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "batch_size": args.batch_size,
            },
            "stages": _finish_stage_metrics(stage_metrics),
            "global": finished_global,
        }
    )
    atomic_json_new(args.output_json, measurement)
    print(
        json.dumps(
            {
                "status": "measured",
                "authorization": False,
                "output": str(args.output_json.resolve()),
                "windows": expected_windows,
                "receipt_payload_sha256": measurement["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _decide(args: argparse.Namespace) -> int:
    measurement_path, measurement_bytes = read_verified_bytes(
        args.measurement_json,
        args.expected_measurement_sha256,
        "measurement receipt",
    )
    threshold_path, threshold_bytes = read_verified_bytes(
        args.thresholds_json,
        args.expected_thresholds_sha256,
        "threshold file",
    )
    measurement = json.loads(measurement_bytes)
    thresholds = json.loads(threshold_bytes)
    if not isinstance(measurement, dict) or not isinstance(thresholds, dict):
        raise TypeError("measurement and thresholds must be JSON objects")
    validate_measurement_contract(
        measurement,
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
    )
    results = evaluate_thresholds(
        measurement,
        thresholds,
        args.expected_measurement_sha256,
    )
    accepted = all(result["passed"] for result in results)
    gate_source = measurement["source_roles"]["gate"]
    gate_script_receipt = {
        "path": gate_source["script"],
        "sha256": gate_source["script_sha256"],
    }
    measurement_receipt = {
        "path": str(measurement_path),
        "sha256": args.expected_measurement_sha256,
    }
    threshold_receipt = {
        "path": str(threshold_path),
        "sha256": args.expected_thresholds_sha256,
    }
    if len(
        {
            gate_script_receipt["path"],
            measurement_receipt["path"],
            threshold_receipt["path"],
        }
    ) != 3:
        raise RuntimeError("formal gate artifacts must be three distinct files")

    def numeric_tree(value: Any) -> Any | None:
        if type(value) in (int, float):
            return require_finite_number(value, "measurement numeric leaf")
        if isinstance(value, dict):
            result = {
                key: converted
                for key, item in value.items()
                if (converted := numeric_tree(item)) is not None
            }
            return result or None
        if isinstance(value, list):
            result = [
                converted
                for item in value
                if (converted := numeric_tree(item)) is not None
            ]
            return result or None
        return None

    numeric_measurements = numeric_tree(
        {
            "stages": measurement["stages"],
            "global": measurement["global"],
            "coverage": measurement["coverage"],
            "determinism": measurement["determinism"],
        }
    )
    if not isinstance(numeric_measurements, dict) or not numeric_measurements:
        raise RuntimeError("formal numeric measurement tree is empty")
    numeric_thresholds = {
        f"rule_{index:03d}": {
            "actual": result["actual"],
            "threshold": result["threshold"],
        }
        for index, result in enumerate(results)
    }
    decisions = {
        result["metric"]: bool(result["passed"])
        for result in results
    }
    formal_protocol = {
        "mode": "fully_released_zero_shot_v1",
        "split": "test",
        "show_speakers": [0, 1, 2, 3],
        "exact_once": True,
        "all_tensors_finite": True,
        "deterministic": True,
        "evaluated_components": [
            "face",
            "upper",
            "hands",
            "lower",
            "global_sanity",
        ],
        "bound_not_evaluated": ["base"],
        "forbidden_components": [
            "ASR",
            "CLIP",
            "SemGate",
            "Sparse",
            "Speaker2",
            "TextGrid",
            "emotion",
            "semantic",
            "vocabulary",
        ],
    }
    decision_without_sha = {
        "format": GATE_FORMAT,
        "status": "pass" if accepted else "reject",
        "authorization": accepted,
        "release_trust_root": measurement["release_trust_root"],
        "official_weights": measurement["official_weights"],
        "canonical_receipt": measurement["canonical_receipt"],
        "source_roles": measurement["source_roles"],
        "protocol": formal_protocol,
        "gate_script": gate_script_receipt,
        "measurement_receipt": measurement_receipt,
        "threshold_receipt": threshold_receipt,
        "measurements": numeric_measurements,
        "thresholds": numeric_thresholds,
        "decisions": decisions,
    }
    if set(decision_without_sha) | {"receipt_sha256"} != FORMAL_GATE_KEYS:
        raise AssertionError("internal formal gate schema mismatch")
    decision = {
        **decision_without_sha,
        "receipt_sha256": canonical_payload_sha256(decision_without_sha),
    }
    atomic_json_new(args.output_json, decision)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "authorization": accepted,
                "output": str(args.output_json.resolve()),
                "receipt_sha256": decision["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official released all-speakers -> canonical SHOW cross-domain gate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure = subparsers.add_parser(
        "measure",
        help="produce measurements only; authorization is always false",
    )
    measure.add_argument("--canonical-manifest", type=Path, required=True)
    measure.add_argument("--canonical-summary", type=Path, required=True)
    measure.add_argument("--canonical-lineage", type=Path, required=True)
    measure.add_argument("--expected-manifest-sha256", required=True)
    measure.add_argument("--expected-summary-sha256", required=True)
    measure.add_argument("--expected-lineage-sha256", required=True)
    measure.add_argument("--expected-canonical-commit", required=True)
    measure.add_argument("--expected-canonical-tree", required=True)
    measure.add_argument("--weights-root", type=Path, required=True)
    measure.add_argument("--device", required=True)
    measure.add_argument("--batch-size", type=int, default=8)
    measure.add_argument("--seed", type=int, default=20260730)
    measure.add_argument("--expected-source-commit", required=True)
    measure.add_argument("--expected-source-tree", required=True)
    measure.add_argument("--expected-input-source-commit", required=True)
    measure.add_argument("--expected-input-source-tree", required=True)
    measure.add_argument("--output-json", type=Path, required=True)
    measure.set_defaults(handler=_measure)

    decide = subparsers.add_parser(
        "decide",
        help="apply an explicit frozen threshold file to a frozen measurement",
    )
    decide.add_argument("--measurement-json", type=Path, required=True)
    decide.add_argument("--expected-measurement-sha256", required=True)
    decide.add_argument("--thresholds-json", type=Path, required=True)
    decide.add_argument("--expected-thresholds-sha256", required=True)
    decide.add_argument("--expected-source-commit", required=True)
    decide.add_argument("--expected-source-tree", required=True)
    decide.add_argument("--output-json", type=Path, required=True)
    decide.set_defaults(handler=_decide)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
